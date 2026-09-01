"""Sample trajectories from the recalibrated MDN predictive density."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from src.calibration.LocalPITRecalibrator import LocalPITRecalibrator
from src.calibration.metrics import probability_integral_transform
from src.calibration.recalibration import ALPHAS, DEFAULT_NUM_BASIS
from src.conditional_theoretical_moments.theoretical_quantiles import (
    compute_theoretical_quantiles,
)
from src.forecast_methods.MDN import MixtureDensityNetwork
from src.forecast_methods.utils import check_cde, make_grid, prepare_tensors
from src.recursive_forecasting.blocks import fit_mar
from src.stable_mar.stable_mar import (
    MARMA_TRIM,
    _marma_dynamics_bounds,
    _marma_nll,
    _marma_split,
    check_marma_existence,
    fit_marma,
    marma_recover_eps,
    simulate_MARMA,
)
from src.stable_mar.stable_mar import (
    stablemar as sm,
)
from src.utils.setup_config_device import (
    get_allowed_cpu_count,
    set_seed,
    setup_config_device,
    setup_device,
)
from src.utils.setup_logger import setup_logger

ALPHA = 1.4
HORIZON = 1
GRID_MARGIN = 0.10
GCOV_H, GCOV_K = 2, 2
METHODS = (("mle", "Estimates (ML)"),)

MAIN_ANCHOR_ORDER = (0, 1)
APPENDIX_ANCHOR_ORDER = (0, 2)


def _order_seed(cfg, key, order_mar):
    """Seed configured under ``sampled_trajectories[key]`` for ``order_mar``."""
    table = cfg.get("sampled_trajectories", {}).get(key, {})
    for k, v in table.items():
        if tuple(map(int, str(k).strip("()").split(","))) == tuple(order_mar):
            return v
    return None


def sample_from_density(density, grid, rng):
    """Inverse-CDF draw from a density tabulated on ``grid``."""
    density = np.clip(density, 0.0, None)
    cdf = np.cumsum(density) * (grid[1] - grid[0])
    if cdf[-1] <= 0:
        raise ValueError("Degenerate density: total mass is zero")
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, rng.uniform())
    return float(grid[min(idx, len(grid) - 1)])


def _marma_param_names(order):
    """Parameter labels matching the MARMA layout [phi_c, phi_nc, theta_c, theta_nc]."""
    r, s, p, q = order
    return (
        [f"phi_{i + 1}" for i in range(r)]
        + [f"psi_{i + 1}" for i in range(s)]
        + [f"eta_{i + 1}" for i in range(p)]
        + [f"theta_{i + 1}" for i in range(q)]
        + ["alpha", "beta", "sigma"]
    )


def _marma_mle_vcv(y, order, params):
    """Observed information for a MARMA MLE fit.

    ``fit_marma`` returns point estimates only, so the standard errors come from
    the inverse numerical Hessian of the profile NLL -- the same observed-information
    estimator ``stablemar._calculate_mle_vcv`` uses for the MAR case.
    """
    d = len(params)
    bounds = _marma_dynamics_bounds(order) + [(1.05, 1.98), (-0.95, 0.95), (1e-3, 1e3)]
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    h = 1e-5

    def nll(v):
        return _marma_nll(np.clip(v, lower, upper), y, order)

    def unit(i):
        e = np.zeros(d)
        e[i] = h
        return e

    fx = nll(params)
    grdp = np.array([nll(params + unit(i)) for i in range(d)])
    grdm = np.array([nll(params - unit(i)) for i in range(d)])
    hess = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(i, d):
            fpp = nll(params + unit(i) + unit(j))
            fmm = nll(params - unit(i) - unit(j))
            hess[i, j] = hess[j, i] = (
                fpp - grdp[i] - grdp[j] + fx + fx - grdm[i] - grdm[j] + fmm
            ) / (2.0 * h**2)

    try:
        return np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hess)


def _marma_resid_acov(y, order, params, lag):
    """Autocovariance at ``lag`` of the innovations filtered at ``params``."""
    e = marma_recover_eps(y, *_marma_split(params, order))
    e = e[MARMA_TRIM : len(y) - MARMA_TRIM]
    if e.size <= lag or not np.all(np.isfinite(e)):
        return 0.0
    e = e - e.mean()
    return float(np.dot(e[lag:], e[: e.size - lag]) / (e.size - lag))


def _marma_gcov_vcv(y, order, params, resid, max_lag=10):
    """Variance-covariance for a MARMA GCov fit."""
    n_dyn = sum(order)
    params = np.asarray(params, dtype=float)
    n = len(resid)
    n_lags = max(1, min(max_lag, n // 4))
    gamma_0 = max(float(np.var(resid)), 1e-6)

    deltas = np.maximum(1e-4, np.abs(params[:n_dyn]) * 0.01)
    grads = np.empty((n_dyn, n_lags))
    for i in range(n_dyn):
        plus, minus = params.copy(), params.copy()
        plus[i] += deltas[i]
        minus[i] -= deltas[i]
        for h in range(1, n_lags + 1):
            grads[i, h - 1] = (
                _marma_resid_acov(y, order, plus, h)
                - _marma_resid_acov(y, order, minus, h)
            ) / (2.0 * deltas[i])

    theta = grads @ grads.T / gamma_0**2
    theta += np.eye(n_dyn) * max(1e-6, 0.001 * np.trace(theta) / n_dyn)
    try:
        vcv_dyn = np.linalg.inv(theta) / n
    except np.linalg.LinAlgError:
        vcv_dyn = np.linalg.pinv(theta) / n

    vcv = np.zeros((n_dyn + 3, n_dyn + 3))
    vcv[:n_dyn, :n_dyn] = vcv_dyn
    vcv[n_dyn:, n_dyn:] = sm(order=order[:2])._calculate_stable_params_vcv(
        resid, list(params[n_dyn:])
    )
    return vcv


def marma_inference(y, order, res, alpha_level=0.05):
    """Inference table for a MARMA fit, MLE or GCov.

    ``res`` is a ``fit_marma`` result; for GCov it must already carry the stable
    parameters recovered from ``res["E"]`` (GCov fits the dynamics only).
    """
    params = np.concatenate(
        [
            np.asarray(res["params"], dtype=float),
            [res["alpha"], res["beta"], res["sigma"]],
        ]
    )
    names = _marma_param_names(order)
    assert len(names) == len(params) == sum(order) + 3

    if res["method"] == "mle":
        vcv = _marma_mle_vcv(y, order, params)
    else:
        vcv = _marma_gcov_vcv(y, order, params, np.asarray(res["E"], dtype=float))

    with np.errstate(invalid="ignore"):
        std_errors = np.sqrt(np.diag(vcv))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stats = params / std_errors
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    z = stats.norm.ppf(1 - alpha_level / 2)

    return pd.DataFrame(
        {
            "Parameter": names,
            "Estimate": params,
            "Std. Error": std_errors,
            "t-statistic": t_stats,
            "p-value": p_values,
            f"CI {100 * (1 - alpha_level)}% Lower": params - z * std_errors,
            f"CI {100 * (1 - alpha_level)}% Upper": params + z * std_errors,
        }
    )


def build_true_params(order_mar, proc):
    """Return the generating values keyed like ``inference_df['Parameter']``."""

    def as_list(v):
        return [v] if isinstance(v, (int, float)) else list(v)

    counts = (
        list(zip(("phi", "psi", "eta", "theta"), order_mar))
        if len(order_mar) == 4
        else list(zip(("phi", "psi"), order_mar))
    )
    source = {
        "phi": proc["PHI"],
        "psi": proc["PSI"],
        "eta": proc["ETA"],
        "theta": proc["THETA"],
    }

    true = {}
    for prefix, count in counts:
        if count == 0:
            continue
        values = as_list(source[prefix])
        for i in range(count):
            true[f"{prefix}_{i + 1}"] = values[i]
    true["alpha"] = ALPHA
    true["beta"] = proc["BETA"]
    true["sigma"] = proc["SIGMA"]
    return true


def generate_params_latex(
    inference, order_mar, true_params=None, caption=None, label=None
):
    """LaTeX table of the estimates with standard errors and significance stars."""
    is_marma = len(order_mar) == 4
    model_type = "MARMA" if is_marma else "MAR"
    order_str = ",".join(str(o) for o in order_mar)

    if is_marma:
        r, s, p, q = order_mar
        param_order, param_display = [], {}
        for prefix, sym, count in (
            ("phi", r"\phi", r),
            ("psi", r"\psi", s),
            ("eta", r"\eta", p),
            ("theta", r"\theta", q),
        ):
            for i in range(count):
                key = f"{prefix}_{i + 1}"
                param_order.append(key)
                param_display[key] = f"${sym}" + (f"_{i + 1}$" if count > 1 else "$")
    else:
        r, s = order_mar
        param_order, param_display = [], {}
        for prefix, sym, count in (("phi", r"\phi", r), ("psi", r"\psi", s)):
            for i in range(count):
                key = f"{prefix}_{i + 1}"
                param_order.append(key)
                param_display[key] = f"${sym}" + (f"_{i + 1}$" if count > 1 else "$")

    for key, sym in (("alpha", r"\alpha"), ("beta", r"\beta"), ("sigma", r"\sigma")):
        param_order.append(key)
        param_display[key] = f"${sym}$"

    methods = [(key, header) for key, header in METHODS if key in inference]
    caption = caption or (
        f"Estimated {model_type}({order_str}) parameters from the MDN-sampled "
        f"trajectory ($\\alpha = {ALPHA}$)"
    )
    label = (
        label or f"tab:{model_type.lower()}{''.join(str(o) for o in order_mar)}_sampled"
    )
    anchor_order = (
        MAIN_ANCHOR_ORDER
        if tuple(order_mar) == MAIN_ANCHOR_ORDER
        else APPENDIX_ANCHOR_ORDER
    )
    anchor = f"tab:mar{''.join(str(o) for o in anchor_order)}_sampled"

    def get_stars(p_value):
        if p_value is None or not np.isfinite(p_value):
            return ""
        if p_value < 0.01:
            return "$^{***}$"
        if p_value < 0.05:
            return "$^{**}$"
        if p_value < 0.10:
            return "$^{*}$"
        return ""

    def format_number(value):
        if value is None or not np.isfinite(value):
            return "--"
        val_str = f"{value:.3f}"
        return f"${val_str}$" if value < 0 else val_str

    def format_estimate(value, std, p_value):
        if value is None or not np.isfinite(value):
            return "--"
        std_str = f"({std:.3f})" if std is not None and np.isfinite(std) else ""
        cell = f"{format_number(value)}{get_stars(p_value)}"
        return f"\\makecell{{{cell} \\\\ {std_str}}}"

    show_true = true_params is not None
    headers = (["True"] if show_true else []) + [header for _, header in methods]
    note = (
        rf"\item \textit{{Note:}} See note to \autoref{{{anchor}}}."
        if label != anchor
        else (
            r"\item \textit{Note:} Standard errors in parentheses. "
            r"$^{***}$ $p<0.01$, $^{**}$ $p<0.05$, $^{*}$ $p<0.10$."
        )
    )
    latex = [
        r"\begin{table}[!htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "c" * len(headers) + "}",
        r"\toprule",
        " & ".join(["Parameter"] + headers) + r" \\",
        r"\midrule",
    ]
    for param in param_order:
        cells = [param_display[param]]
        if show_true:
            true_value = true_params.get(param)
            cells.append(
                "--" if true_value is None else format_number(float(true_value))
            )
        found = False
        for key, _ in methods:
            df = inference[key]
            row = df[df["Parameter"] == param]
            if len(row) == 0:
                cells.append("--")
                continue
            found = True
            row = row.iloc[0]
            cells.append(
                format_estimate(row["Estimate"], row["Std. Error"], row["p-value"])
            )
        if not found:
            continue
        latex.append(" & ".join(cells) + r" \\")
    latex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[para,flushleft]",
        r"\footnotesize",
        note,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(latex)


def plot_side_by_side(trajectory, true_trajectory, ylim, title, path):
    """Plot the sampled path next to the true path on a shared y scale."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 4), sharey=True)
    axes[0].plot(trajectory, color="black")
    axes[0].set_ylim(*ylim)
    axes[0].set_title(f"MDN-sampled {title} simulation", fontsize=16, color="black")
    axes[0].set_xlabel("Time index", fontsize=14)
    axes[0].set_ylabel("Value", fontsize=14)
    axes[1].plot(true_trajectory, color="black")
    axes[1].set_ylim(*ylim)
    axes[1].set_title(f"True {title} simulation", fontsize=16, color="black")
    axes[1].set_xlabel("Time index", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _as_list(value):
    """Coerce a config parameter (None, scalar or sequence) to a list."""
    if value is None:
        return []
    return [value] if isinstance(value, (int, float)) else list(value)


def build_generator(order_mar, params):
    """Return a ``generate_series(n)`` closure for a parameter set."""
    PSI, PHI, THETA, ETA = (params.get(k) for k in ("PSI", "PHI", "THETA", "ETA"))
    alpha, beta, sigma = params["ALPHA"], params["BETA"], params["SIGMA"]

    if len(order_mar) == 4:

        def generate_series(n):
            traj = simulate_MARMA(
                N=float(n),
                psi=PSI,
                phi=PHI,
                theta=THETA,
                eta=ETA,
                alpha=alpha,
                beta=beta,
                sigma=sigma,
            )
            return traj.dropna().reset_index(drop=True)

        return generate_series

    r, s = order_mar
    mar = sm(order_mar)
    mar.model = "MAR"
    mar.par = _as_list(PHI)[:r] + _as_list(PSI)[:s] + [alpha, beta, sigma]

    def generate_series(n):
        return mar.generate(n).trajectory.dropna().reset_index(drop=True)

    return generate_series


def estimate_params(y, order_mar, method, seed=0, verbose=True):
    """Fit ``order_mar`` on ``y``; return (inference DataFrame, parameter dict).

    The parameter dict is keyed like config.yaml's ``mar_params`` so it can be
    handed straight to ``build_generator``. GCov fits the dynamics only, so its
    stable parameters come from the filtered innovations (as in
    simulations_realized_outcomes.py).
    """
    y = np.asarray(y, dtype=float)
    r, s = order_mar[0], order_mar[1]

    if len(order_mar) == 4:
        if order_mar != (1, 1, 1, 1):
            raise NotImplementedError(
                f"MARMA estimation is implemented for order (1,1,1,1); got {order_mar}."
            )
        res = fit_marma(y, order=order_mar, method=method, seed=seed, verbose=verbose)
        if method == "gcov":
            alpha_h, beta_h, sigma_h = sm(order=(1, 1)).fit_stable_noise(res["E"])
            res = {
                **res,
                "alpha": float(alpha_h),
                "beta": float(beta_h),
                "sigma": float(sigma_h),
            }
        params = {
            "PHI": float(res["phi_c"][0]),
            "PSI": float(res["phi_nc"][0]),
            "ETA": float(res["theta_c"][0]),
            "THETA": float(res["theta_nc"][0]),
            "ALPHA": float(res["alpha"]),
            "BETA": float(res["beta"]),
            "SIGMA": float(res["sigma"]),
        }
        return marma_inference(y, order_mar, res), params

    par, est = fit_mar(y, order_mar, method, GCOV_H, GCOV_K, rng_seed=seed)
    phi_hat = [float(v) for v in par[:r]]
    psi_hat = [float(v) for v in par[r : r + s]]
    params = {
        "PHI": phi_hat[0] if r == 1 else (phi_hat or None),
        "PSI": psi_hat[0] if s == 1 else psi_hat,
        "THETA": None,
        "ETA": None,
        "ALPHA": float(par[r + s]),
        "BETA": float(par[r + s + 1]),
        "SIGMA": float(par[r + s + 2]),
    }
    return est.inference(y, par), params


def resolve_process(cfg, order_mar):
    """Build the generating parameters and a simulator for ``order_mar``."""
    mar_params = {
        tuple(map(int, k.strip("()").split(","))): v
        for k, v in cfg["mar_params"].items()
    }
    if order_mar not in mar_params:
        raise ValueError(
            f"No parameters for order {order_mar} in config.yaml mar_params "
            f"(available: {sorted(mar_params)})"
        )
    params = {**mar_params[order_mar], **cfg["alpha_stable_params"]}
    params["ALPHA"] = ALPHA
    PSI, PHI, BETA, SIGMA, THETA, ETA = (
        params.get(k) for k in ["PSI", "PHI", "BETA", "SIGMA", "THETA", "ETA"]
    )

    is_marma = len(order_mar) == 4
    model_type = "MARMA" if is_marma else "MAR"
    generate_series = build_generator(order_mar, params)

    return {
        "params": params,
        "is_marma": is_marma,
        "model_type": model_type,
        "title": f"{model_type}{order_mar}",
        "generate_series": generate_series,
        "PSI": PSI,
        "PHI": PHI,
        "BETA": BETA,
        "SIGMA": SIGMA,
        "THETA": THETA,
        "ETA": ETA,
    }


def build_sampler(cfg, order_mar, device, n_process, estimation="mle", logger=None):
    """Estimate the DGP, then fit the tuned MDN and its local PIT recalibrator.

    ``estimation`` selects the parameters the training and calibration series are
    simulated from: "mle" or "gcov" estimates them on a ``length_estimation``
    draw from the true process (parameter uncertainty), "true" uses the
    generating values.
    """
    logger = logger or setup_logger()
    proc = resolve_process(cfg, order_mar)
    title = proc["title"]
    tr = cfg["mdn_params"]["training_params"]

    mdn_config_path = (
        f"outputs/simulations_realized_outcomes/{title}/horizon_{HORIZON}/"
        f"alpha_{ALPHA}/mdn/config.yaml"
    )
    if not os.path.exists(mdn_config_path):
        raise FileNotFoundError(
            f"No tuned MDN config at {mdn_config_path}; run "
            f"simulations_realized_outcomes.py --model mdn --order "
            f"{' '.join(map(str, order_mar))} --alpha {ALPHA} first."
        )
    with open(mdn_config_path) as f:
        best_params = yaml.safe_load(f)["params"]
    LAGS = best_params["lags"]
    logger.info(f"Tuned MDN config from {mdn_config_path}: {best_params}")

    # Estimation stage: the DGP the MDN is trained on
    n_est = cfg["length_estimation"]
    est_seed = _order_seed(cfg, "estimation_seed", order_mar) or cfg["seed"]
    set_seed(est_seed)
    estimation_series = np.asarray(proc["generate_series"](n_est), dtype=float)[:n_est]
    logger.info(
        f"Estimation trajectory (true DGP): n={len(estimation_series)} seed={est_seed}"
    )

    if estimation == "true":
        est_inference, est_params = None, dict(proc["params"])
        logger.info("Training data simulated from the generating parameters.")
    else:
        est_inference, est_params = estimate_params(
            estimation_series, order_mar, estimation, seed=cfg["seed"]
        )
        logger.info(
            "%s estimates on the true %d-observation trajectory:\n%s",
            estimation.upper(),
            len(estimation_series),
            est_inference.to_string(),
        )
        valid = check_marma_existence(
            psi_vec=_as_list(est_params["PSI"]),
            phi_vec=_as_list(est_params["PHI"]) or [0.0],
            theta_vec=est_params["THETA"],
            eta_vec=est_params["ETA"],
        )
        if not valid:
            logger.warning(
                "Estimated parameters violate the MARMA existence conditions; "
                "the simulated training data may be explosive."
            )

    generate_series = build_generator(order_mar, est_params)
    y_sim = generate_series(cfg["length_simulation"])
    set_seed(cfg["calibration_seed"])
    y_cali = generate_series(cfg["length_calibration"])
    logger.info(f"train n={len(y_sim)} | calibration n={len(y_cali)}")

    quantiles = compute_theoretical_quantiles(
        phi_vec=[proc["PHI"]],
        psi_vec=[proc["PSI"]] if isinstance(proc["PSI"], float) else proc["PSI"],
        alpha=ALPHA,
        beta=proc["BETA"],
        sigma=proc["SIGMA"],
        theta=proc["THETA"],
        eta=proc["ETA"],
    )
    _, grid_y = make_grid(quantiles, cfg["n_points_grid"])
    observed = [y_sim, y_cali, estimation_series]
    grid_y_min = min([float(grid_y[0])] + [float(np.min(v)) for v in observed])
    grid_y_max = max([float(grid_y[-1])] + [float(np.max(v)) for v in observed])
    margin = GRID_MARGIN * (grid_y_max - grid_y_min)
    grid_y = np.linspace(grid_y_min - margin, grid_y_max + margin, cfg["n_points_grid"])
    logger.info(f"grid_y = [{grid_y[0]:.3f}, {grid_y[-1]:.3f}]")

    X_train, y_train, _, _, _, _, weights_train, _ = prepare_tensors(
        pd.DataFrame(y_sim),
        X=None,
        y=None,
        lags=LAGS,
        horizon=HORIZON,
        proportions=(1.0, 0.0, 0.0),
        device=device,
    )
    X_cali, y_cali_t, _, _, _, _, weights_cali, _ = prepare_tensors(
        pd.DataFrame(y_cali),
        X=None,
        y=None,
        lags=LAGS,
        horizon=HORIZON,
        proportions=(1.0, 0.0, 0.0),
        device=device,
    )

    set_seed(cfg["seed"])
    mdn = MixtureDensityNetwork(
        input_dim=LAGS,
        hidden_layers=[best_params["mlp_width"]] * best_params["mlp_depth"],
        n_mixtures=best_params["n_mixtures"],
        dropout=best_params["dropout"],
        n_jobs=n_process,
        device=device,
    ).to(device)
    mdn.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_cali,
        y_val=y_cali_t,
        weights_train=weights_train,
        weights_val=weights_cali,
        max_epochs=tr["MAX_EPOCHS"],
        learning_rate=best_params["learning_rate"],
        batch_size=tr["BATCH_SIZE"],
        max_norm=tr["MAX_GRAD_NORM"],
        patience=tr["EARLY_STOPPING_PATIENCE"],
        scheduler_patience=tr["SCHEDULER_PATIENCE"],
        scheduler_factor=tr["SCHEDULER_FACTOR"],
    )
    mdn.eval()

    cde_cali = mdn.pred(X_cali, grid_y)
    check_cde(cde_cali, grid_y)
    pit_cali = probability_integral_transform(cde_cali, grid_y, y_cali_t)
    calibrator = LocalPITRecalibrator(n_jobs=n_process, num_basis=DEFAULT_NUM_BASIS)
    calibrator.fit(X_cali, pit_cali, alphas=ALPHAS)

    mdn.n_jobs = 1  # one row per call: joblib fan-out would only add overhead

    return {
        **proc,
        "mdn": mdn,
        "calibrator": calibrator,
        "grid_y": grid_y,
        "lags": LAGS,
        "device": device,
        "mdn_config_path": mdn_config_path,
        "mdn_params": best_params,
        "estimation": estimation,
        "estimation_series": estimation_series,
        "estimation_inference": est_inference,
        "estimated_params": est_params,
    }


def draw_trajectory(sampler, draw_seed, init_window, length):
    """Sample ``length`` values, conditioning on ``init_window``.

    ``init_window`` holds the ``lags`` observations the path starts from -- the
    head of the true estimation trajectory -- so no burn-in is needed and the
    draw is comparable to the true path observation by observation.
    """
    mdn = sampler["mdn"]
    calibrator = sampler["calibrator"]
    grid_y = sampler["grid_y"]
    lags = sampler["lags"]
    device = sampler["device"]

    window = [float(v) for v in init_window]
    if len(window) != lags:
        raise ValueError(f"init_window has {len(window)} values, expected {lags}")

    set_seed(draw_seed)
    rng = np.random.default_rng(draw_seed)
    sampled = []
    for _ in range(length):
        x_current = torch.tensor([window[-lags:]], dtype=torch.float32, device=device)
        pred_density = mdn.pred(x_current, grid_y).cpu().numpy()
        recalibrated = calibrator.transform(
            x_current, pred_density, grid_y, verbose=False
        ).flatten()
        x_next = sample_from_density(recalibrated, grid_y, rng)
        window.append(x_next)
        sampled.append(x_next)
    return np.array(sampled)


def main():
    """Sample a trajectory from the recalibrated MDN density and report it."""
    logger = setup_logger()
    device = setup_device()
    n_process = setup_config_device(get_allowed_cpu_count())

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        required=True,
        help="Model order: 2 values for MAR (r, s) or 4 values for MARMA (r, s, p, q)",
    )
    parser.add_argument(
        "--draw_seed",
        type=int,
        default=None,
        help="Seed for the sampling innovations. Defaults to the seed "
        "configured for this order, otherwise config.yaml's seed.",
    )
    parser.add_argument(
        "--estimation",
        choices=["gcov", "mle", "true"],
        default="mle",
        help=(
            "Parameters the MDN's training and calibration series are simulated "
            "from: 'mle' (default, alpha-stable MLE) or 'gcov' estimated on a "
            "length_estimation draw from the true process, or 'true' (the "
            "generating parameters, i.e. no parameter uncertainty)."
        ),
    )
    args = parser.parse_args()

    horizon = HORIZON
    order_mar = tuple(int(x) for x in args.order)
    if len(order_mar) not in (2, 4):
        raise ValueError(f"--order takes 2 (MAR) or 4 (MARMA) values, got {order_mar}")
    if len(order_mar) == 4 and order_mar != (1, 1, 1, 1):
        raise NotImplementedError(
            f"MARMA estimation is implemented for order (1,1,1,1); got {order_mar}."
        )

    LENGTH_SIMULATION = cfg["length_simulation"]
    LENGTH_CALIBRATION = cfg["length_calibration"]
    LENGTH_ESTIMATION = cfg["length_estimation"]
    N_POINTS_GRID = cfg["n_points_grid"]

    proc = resolve_process(cfg, order_mar)
    params_alpha_mar = proc["params"]
    PSI, PHI, THETA, ETA = proc["PSI"], proc["PHI"], proc["THETA"], proc["ETA"]
    title = proc["title"]

    draw_seed = args.draw_seed
    if draw_seed is None:
        draw_seed = _order_seed(cfg, "draw_seed", order_mar)
        if draw_seed is None:
            draw_seed = cfg["seed"]
            logger.warning(
                "No draw seed configured for %s -- falling back to "
                "cfg['seed']=%d. Pass --draw_seed, or add an entry under "
                "sampled_trajectories.draw_seed in config.yaml.",
                title,
                draw_seed,
            )
    logger.info("=" * 60)
    logger.info(
        f"{title} | alpha={ALPHA} | horizon={horizon} | draw_seed={draw_seed} "
        f"| estimation={args.estimation}"
    )
    logger.info(f"Simulation parameters: {params_alpha_mar}")
    logger.info(
        f"length_simulation={LENGTH_SIMULATION}, "
        f"length_calibration={LENGTH_CALIBRATION}, "
        f"length_estimation={LENGTH_ESTIMATION}, n_points_grid={N_POINTS_GRID}"
    )
    logger.info("=" * 60)

    model_valid = check_marma_existence(
        psi_vec=[PSI] if isinstance(PSI, float) else PSI,
        phi_vec=[PHI],
        theta_vec=THETA,
        eta_vec=ETA,
    )
    logger.info(f"MARMA existence conditions satisfied: {model_valid}")

    file_name = f"outputs/sampled_trajectories/{title}/horizon_{horizon}/alpha_{ALPHA}"
    os.makedirs(file_name, exist_ok=True)
    logger.info(f"Artifacts will be saved in: {file_name}")

    sampler = build_sampler(
        cfg, order_mar, device, n_process, estimation=args.estimation, logger=logger
    )
    lags = sampler["lags"]

    estimation_series = sampler["estimation_series"]
    init_window = estimation_series[:lags]
    true_trajectory = estimation_series[lags:]
    trajectory = draw_trajectory(sampler, draw_seed, init_window, len(true_trajectory))

    logger.info(
        f"Conditioning window (first {lags} obs of the true trajectory): "
        f"{np.round(init_window, 3).tolist()}"
    )
    logger.info(
        f"MDN-sampled n={len(trajectory)} "
        f"range=[{trajectory.min():.2f}, {trajectory.max():.2f}] | "
        f"True n={len(true_trajectory)} "
        f"range=[{true_trajectory.min():.2f}, {true_trajectory.max():.2f}]"
    )

    # plots (shared y scale)
    ymin = min(trajectory.min(), true_trajectory.min())
    ymax = max(trajectory.max(), true_trajectory.max())
    pad = 0.05 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    plot_path = f"{file_name}/sampled_trajectory.png"
    plot_side_by_side(trajectory, true_trajectory, ylim, title, plot_path)
    logger.info(f"Saved plot: {plot_path}")

    # ML estimation on the sampled trajectory
    inference, estimates = {}, {}
    for method, _ in METHODS:
        inference[method], estimates[method] = estimate_params(
            trajectory, order_mar, method, seed=cfg["seed"]
        )
        logger.info(
            "%s estimates on the sampled trajectory:\n%s",
            method.upper(),
            inference[method].to_string(),
        )

    tex_path = f"{file_name}/estimated_params.tex"
    with open(tex_path, "w") as f:
        f.write(
            generate_params_latex(
                inference, order_mar, true_params=build_true_params(order_mar, proc)
            )
            + "\n"
        )
    logger.info(f"Saved LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
