"""Recursive real-time density-forecast application for the real gas price.

Follows the Baumeister et al. (2025, JAE) protocol: at each forecast origin the
model is re-estimated on the real-time vintage window (expanding from 1976M1),
and forecasts are evaluated against the final (May-2024) vintage.
"""

import argparse
import json
import multiprocessing as mp
import os

import flexcode
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from flexcode.regression_models import XGBoost
from joblib import Parallel, delayed
from scipy.io import loadmat
from tqdm import tqdm

from src.calibration.recalibration import (
    DEFAULT_NUM_BASIS,
    fit_and_apply_recalibration,
    lagged_pairs,
    recalibrate_density,
)
from src.evaluate.cdetools import cde_loss
from src.evaluate.evaluate_applications import (
    evaluate_and_plot_densities,
    mspe_by_region,
)
from src.forecast_methods.gaussian_linear import (
    ar_ma_weights,
    gaussian_predictive_density,
    gaussian_sum_params,
    ima11_ma_weights,
    ses_filter,
)
from src.forecast_methods.gj2026 import gj2026_predictive_density_given_yt
from src.forecast_methods.kcde import kcde_predictive_density
from src.forecast_methods.lls2012 import lls2012_predictive_density_given_yt
from src.forecast_methods.MDN import MixtureDensityNetwork
from src.forecast_methods.utils import check_cde, prepare_tensors
from src.recursive_forecasting.blocks import (
    ar_forecast,
    fit_mar,
    get_block,
    get_exact_block,
    make_blocks,
)
from src.recursive_forecasting.levels import (
    build_level_model_grid,
    build_log_grid,
    log_density_to_level,
    restrict_to_level_grid,
)
from src.recursive_forecasting.vintage import (
    _col,
    _row,
    enumerate_origins,
    final_series,
    origin_features,
    origin_targets,
    vintage_logprice,
)
from src.utils.setup_config_device import (
    get_allowed_cpu_count,
    set_seed,
    setup_config_device,
    setup_device,
)
from src.utils.setup_logger import setup_logger

N_LAGS = 10
UPPER_TAIL_ONLY = True
ESTIMATIONS = ("mle", "gcov")
REESTIMATION_FREQ = 1
TAIL_Q = 0.90
GRID_CAP = 5.0
APPLICATION_QS_ALPHAS = (0.90, 0.95)


def _causal_pole_start(order, y):
    """Causal-dominant start both estimators are initialised at.

    Left to itself ``fit_mar`` scans its mirror grid and keeps the best fit by
    criterion, which on this series picks the noncausal mirror on a criterion
    gap of a few percent -- a tie in practice. Passing this start as the warm
    start short-circuits the grid, so the orientation is pinned rather than
    selected; each estimator then chains from its own previous estimate. The
    stable tail matches the one ``_mirror_starts`` uses.
    """
    r, s = order
    scale = max(
        float(np.subtract(*np.percentile(np.asarray(y, dtype=float), [75, 25]))) / 2.0,
        1e-3,
    )
    return [*([0.9] * r), *([0.1] * s), 1.4, 0.0, scale]


def origin_price_quantile(ng, cpi, origins, sample_start, level):
    """Per-origin ``level`` quantile of the real price seen up to that origin.

    Each origin reads its own vintage, so the weighted region of the twCRPS and
    the CSL is known at the time the forecast is made.
    """
    quantiles = np.empty(len(origins), dtype=float)
    current = np.empty(len(origins), dtype=float)
    row0 = _row(sample_start)
    for i, origin in enumerate(origins):
        col = _col(origin.origin_month)
        rows = slice(row0, _row(origin.origin_month) + 1)
        real = 100.0 * ng[rows, col] / cpi[rows, col]
        real = real[np.isfinite(real) & (real > 0)]
        quantiles[i] = np.quantile(real, level)
        current[i] = real[-1]
    return quantiles, current


def _density_median(density, grid):
    """Per-row 0.5-quantile of densities on a (uniform) ``grid`` via the CDF."""
    dx = np.diff(grid)
    cdf = np.concatenate(
        [
            np.zeros((density.shape[0], 1)),
            np.cumsum((density[:, 1:] + density[:, :-1]) / 2 * dx, axis=1),
        ],
        axis=1,
    )
    cdf /= cdf[:, -1:]
    return np.array([np.interp(0.5, cdf[i], grid) for i in range(density.shape[0])])


def kcde_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """KCDE for one block. Lag search on block 0, frozen afterwards."""
    nw = cfg["kcde_params"]
    yt, cal = art.y_train, art.calibration
    logger = ctx["logger"]
    tag = ctx.get("model", "kcde")

    if frozen is None:
        best = {"score": -np.inf, "lags": None}
        lag_scores = {}
        for lags in range(1, N_LAGS + 1):
            Xtr, ytr = lagged_pairs(yt, horizon, lags)
            Xc, yc = lagged_pairs(cal, horizon, lags)
            dc = kcde_predictive_density(
                X_train=Xtr,
                y_train=ytr,
                X_query=Xc,
                grid_y=model_grid,
                kernel=nw["KERNEL"],
                bandwidth=nw["BANDWIDTH"],
            )
            ll = float(
                np.mean(
                    np.log(
                        np.asarray(
                            [np.interp(y, model_grid, d) for y, d in zip(yc, dc)]
                        )
                        + 1e-12
                    )
                )
            )
            lag_scores[lags] = ll
            logger.info(f"[{tag}] lag {lags:>2}: calibration log likelihood {ll:.6f}")
            if ll > best["score"]:
                best = {"score": ll, "lags": lags}
        frozen = {"lags": best["lags"]}
        with open(os.path.join(ctx["model_dir"], "kcde_lags.json"), "w") as f:
            json.dump(
                {
                    "lags": best["lags"],
                    "criterion": "log_likelihood",
                    "direction": "max",
                    "scores": {str(k): v for k, v in lag_scores.items()},
                },
                f,
            )
        logger.info(f"[{tag}] block 0 lags={frozen['lags']} (ll={best['score']:.4f})")

    lags = frozen["lags"]
    Xtr, ytr = lagged_pairs(yt, horizon, lags)
    feats = block_feats[lags]
    dens = kcde_predictive_density(
        X_train=Xtr.astype(np.float64),
        y_train=ytr.astype(np.float64),
        X_query=feats.astype(np.float64),
        grid_y=model_grid,
        kernel=nw["KERNEL"],
        bandwidth=nw["BANDWIDTH"],
    )
    if not ctx.get("recalibrate", True):
        return dens, frozen

    Xc, yc = lagged_pairs(cal, horizon, lags)
    dens_cali = kcde_predictive_density(
        X_train=Xtr.astype(np.float64),
        y_train=ytr.astype(np.float64),
        X_query=Xc.astype(np.float64),
        grid_y=model_grid,
        kernel=nw["KERNEL"],
        bandwidth=nw["BANDWIDTH"],
    )
    dens = recalibrate_density(
        Xc,
        yc,
        dens_cali,
        feats,
        dens,
        model_grid,
        ctx["n_process"],
        artifacts=ctx["artifacts"],
    )
    return dens, frozen


def exact_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """Closed-form predictive density of a causal linear benchmark."""
    spec = art.spec
    logger = ctx["logger"]

    if spec == "exp":
        theta = art.par[0]
        psi = ima11_ma_weights(theta, horizon)
    else:
        p = art.ar_order
        const, coefs = art.par[0], np.asarray(art.par[1 : 1 + p], dtype=float)
        psi = ar_ma_weights(coefs, horizon)

    sigma = art.par[-1]  # par ends with sigma alone
    sigma_h = gaussian_sum_params(psi, sigma)

    locations = np.empty(len(art.origins), dtype=float)
    for i, o in enumerate(art.origins):
        y = vintage_logprice(
            ctx["ng"], ctx["cpi"], o, sample_start=ctx["sample_start"], log=ctx["log"]
        )
        if spec == "exp":
            locations[i] = ses_filter(y, theta)[0]
        else:
            locations[i] = ar_forecast(const, coefs, y, horizon)

    logger.info(
        f"[{ctx.get('model', spec)}] block {art.block_index} exact Gaussian "
        f"density | sigma_h={sigma_h:.5f} (sigma={sigma:.5f}, h={horizon})"
    )
    return gaussian_predictive_density(model_grid, locations, sigma_h), frozen


def _make_flexcode(bc, model_grid, n_process):
    return flexcode.FlexCodeModel(
        XGBoost,
        max_basis=bc["MAX_BASIS"],
        basis_system=bc["BASIS_SYSTEM"],
        z_min=model_grid[0],
        z_max=model_grid[-1],
        regression_params={"verbosity": 0, "n_jobs": n_process},
    )


def flexzboost_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """FlexZBoost for one block. Lag search on block 0, frozen afterwards."""
    bc = cfg["flexzboost_params"]
    yt, cal = art.y_train, art.calibration
    n_grid = len(model_grid)
    logger = ctx["logger"]

    if frozen is None:
        best = {"score": np.inf, "lags": None}
        lag_scores = {}
        for lags in range(1, N_LAGS + 1):
            Xtr, ytr = lagged_pairs(yt, horizon, lags)
            Xc, yc = lagged_pairs(cal, horizon, lags)
            cand = _make_flexcode(bc, model_grid, ctx["n_process"])
            cand.fit(x_train=Xtr, z_train=ytr)
            dc, _ = cand.predict(Xc, n_grid=n_grid)
            score = float(cde_loss(dc, model_grid, yc)[0])
            lag_scores[lags] = score
            logger.info(f"[flexzboost] lag {lags:>2}: calibration CDE loss {score:.6e}")
            if score < best["score"]:
                best = {"score": score, "lags": lags}
        frozen = {"lags": best["lags"]}
        with open(os.path.join(ctx["model_dir"], "flexzboost_lags.json"), "w") as f:
            json.dump(
                {
                    "lags": best["lags"],
                    "criterion": "cde_loss",
                    "direction": "min",
                    "scores": {str(k): v for k, v in lag_scores.items()},
                },
                f,
            )
        logger.info(
            f"[flexzboost] block 0 lags={frozen['lags']} (cde={best['score']:.4e})"
        )

    lags = frozen["lags"]
    Xtr, ytr = lagged_pairs(yt, horizon, lags)
    Xc, yc = lagged_pairs(cal, horizon, lags)
    model = _make_flexcode(bc, model_grid, ctx["n_process"])
    model.fit(x_train=Xtr, z_train=ytr)
    dens_cali, _ = model.predict(Xc, n_grid=n_grid)
    feats = block_feats[lags]
    dens, _ = model.predict(feats, n_grid=n_grid)
    dens = recalibrate_density(
        Xc,
        yc,
        dens_cali,
        feats,
        dens,
        model_grid,
        ctx["n_process"],
        artifacts=ctx["artifacts"],
    )
    return dens, frozen


def _mdn_tune(yt, cal, horizon, cfg, ctx):
    """Optuna search on block 0; returns best hyperparameter dict."""
    op, tr = cfg["optuna_params"], cfg["mdn_params"]["training_params"]
    device, logger = ctx["device"], ctx["logger"]
    model_dir = ctx["model_dir"]
    tag = ctx.get("model", "mdn")
    config_path = os.path.join(model_dir, "config.yaml")

    if os.path.exists(config_path) and (ctx["evaluate"] or ctx["n_trials"] is None):
        with open(config_path) as f:
            bp = yaml.safe_load(f)["params"]
        ctx["mdn_config_source"] = config_path
        logger.info(f"[{tag}] loaded frozen config {config_path}: {bp}")
        return bp

    cache = {}

    def data(lags):
        if lags not in cache:
            Xtr, ytr, *_, wtr, _ = prepare_tensors(
                pd.DataFrame(yt),
                None,
                None,
                lags,
                horizon,
                (1.0, 0.0, 0.0),
                device,
                upper_tail_only=UPPER_TAIL_ONLY,
            )
            Xc, yc, *_, wc, _ = prepare_tensors(
                pd.DataFrame(cal),
                None,
                None,
                lags,
                horizon,
                (1.0, 0.0, 0.0),
                device,
                upper_tail_only=UPPER_TAIL_ONLY,
            )
            cache[lags] = (Xtr, ytr, wtr, Xc, yc, wc)
        return cache[lags]

    def objective(trial):
        set_seed(cfg["seed"])
        lags = trial.suggest_int("lags", *op["LAGS_RANGE"])
        n_mix = trial.suggest_int("n_mixtures", *op["N_MIXTURES_RANGE"])
        lr = trial.suggest_categorical("learning_rate", op["LR_GRID"])
        dropout = trial.suggest_float(
            "dropout", 0.0, op["DROPOUT_MAX"], step=op["DROPOUT_STEP"]
        )
        depth = trial.suggest_int("mlp_depth", *op["DEPTH_RANGE"])
        width = trial.suggest_categorical("mlp_width", op["WIDTH_GRID"])
        Xtr, ytr, wtr, Xc, yc, wc = data(lags)
        cand = MixtureDensityNetwork(
            input_dim=lags,
            hidden_layers=[width] * depth,
            n_mixtures=n_mix,
            dropout=dropout,
            n_jobs=1,
            device=device,
        ).to(device)
        try:
            val = cand.fit(
                X_train=Xtr,
                y_train=ytr,
                X_val=Xc,
                y_val=yc,
                weights_train=wtr,
                weights_val=wc,
                max_epochs=tr["MAX_EPOCHS"],
                learning_rate=lr,
                batch_size=tr["BATCH_SIZE"],
                max_norm=tr["MAX_GRAD_NORM"],
                patience=tr["EARLY_STOPPING_PATIENCE"],
                scheduler_patience=tr["SCHEDULER_PATIENCE"],
                scheduler_factor=tr["SCHEDULER_FACTOR"],
            )
        except Exception as e:
            logger.warning(f"[{tag}] trial {trial.number} failed: {e}")
            return float("inf")
        return val if (val is not None and np.isfinite(val)) else float("inf")

    os.makedirs(model_dir, exist_ok=True)
    storage = f"sqlite:///{model_dir}/study.db"
    study = optuna.create_study(
        study_name="mdn",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        storage=storage,
        load_if_exists=True,
    )
    n_trials = ctx["n_trials"] if ctx["n_trials"] is not None else op["N_TRIALS"]
    remaining = max(0, n_trials - len(study.trials))
    logger.info(f"[{tag}] optuna: {len(study.trials)} done, running {remaining} more")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    bp = dict(study.best_trial.params)
    logger.info(f"[{tag}] best val NLL {study.best_trial.value:.5f} | {bp}")
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "model": tag,
                "params": bp,
                "val": {"nll": float(study.best_trial.value)},
            },
            f,
        )
    ctx["mdn_config_source"] = config_path
    return bp


def mdn_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """MDN for one block. Optuna on block 0 (frozen), weights refit per block."""
    tr = cfg["mdn_params"]["training_params"]
    device = ctx["device"]
    yt, cal = art.y_train, art.calibration

    if frozen is None:
        frozen = {"params": _mdn_tune(yt, cal, horizon, cfg, ctx)}
        if "mdn_config_source" in ctx:
            frozen["config_source"] = ctx["mdn_config_source"]
    bp = frozen["params"]
    lags = bp["lags"]

    Xtr, ytr, *_, wtr, _ = prepare_tensors(
        pd.DataFrame(yt),
        None,
        None,
        lags,
        horizon,
        (1.0, 0.0, 0.0),
        device,
        upper_tail_only=UPPER_TAIL_ONLY,
    )
    Xc, yc, *_, wc, _ = prepare_tensors(
        pd.DataFrame(cal),
        None,
        None,
        lags,
        horizon,
        (1.0, 0.0, 0.0),
        device,
        upper_tail_only=UPPER_TAIL_ONLY,
    )

    set_seed(cfg["seed"])
    mdn = MixtureDensityNetwork(
        input_dim=lags,
        hidden_layers=[bp["mlp_width"]] * bp["mlp_depth"],
        n_mixtures=bp["n_mixtures"],
        dropout=bp["dropout"],
        n_jobs=ctx["n_process"],
        device=device,
    ).to(device)
    mdn.fit(
        X_train=Xtr,
        y_train=ytr,
        X_val=Xc,
        y_val=yc,
        weights_train=wtr,
        weights_val=wc,
        max_epochs=tr["MAX_EPOCHS"],
        learning_rate=bp["learning_rate"],
        batch_size=tr["BATCH_SIZE"],
        max_norm=tr["MAX_GRAD_NORM"],
        patience=tr["EARLY_STOPPING_PATIENCE"],
        scheduler_patience=tr["SCHEDULER_PATIENCE"],
        scheduler_factor=tr["SCHEDULER_FACTOR"],
    )
    mdn.eval()

    feats = block_feats[lags]
    Xtest = torch.tensor(feats.astype(np.float32), device=device)
    dens_cali = np.asarray(mdn.pred(Xc, model_grid).detach().cpu().numpy())
    dens = np.asarray(mdn.pred(Xtest, model_grid).detach().cpu().numpy())
    dens = recalibrate_density(
        Xc.detach().cpu().numpy(),
        yc.detach().cpu().numpy(),
        dens_cali,
        feats,
        dens,
        model_grid,
        ctx["n_process"],
        artifacts=ctx["artifacts"],
    )
    return dens, frozen


def _mar_params(art, cfg):
    """Unpack the block's MAR(r,s) parameters into (r, s, phi, psi, a, b, sig)."""
    r, s = cfg["application"]["order"]
    par = art.par
    phi = list(par[:r]) if r > 0 else [0.0]
    psi = list(par[r : r + s])
    alpha, beta, sigma = par[r + s], par[r + s + 1], par[r + s + 2]
    return r, s, phi, psi, alpha, beta, sigma


def gj2026_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """Gourieroux-Jasiak (2026) density for one block."""
    r, s, phi, psi, alpha, beta, sigma = _mar_params(art, cfg)
    p = r + s
    bw = cfg["kcde_params"]["BANDWIDTH"]
    n_process = ctx["n_process"]
    feats = block_feats[p]  # (n_block, p) oldest-first; gj wants most-recent-first

    y_train = art.y_train

    def _gj(hist):  # hist oldest-first -> reverse for gj y_history
        return gj2026_predictive_density_given_yt(
            psi_hat=psi,
            phi_hat=phi,
            alpha_hat=alpha,
            beta_hat=beta,
            sigma_hat=sigma,
            y_train=y_train,
            y_history=list(hist[::-1]),
            grid_y=model_grid,
            bandwidth=bw,
            horizon=horizon,
        )

    dens = np.array(
        Parallel(n_jobs=n_process)(delayed(_gj)(f) for f in tqdm(feats, desc="gj2026"))
    )
    X_cali, y_cali = lagged_pairs(art.calibration, horizon, p)
    dens_cali = np.array(
        Parallel(n_jobs=n_process)(
            delayed(_gj)(x) for x in tqdm(X_cali, desc="gj2026 calibration")
        )
    )
    dens = recalibrate_density(
        X_cali,
        y_cali,
        dens_cali,
        feats,
        dens,
        model_grid,
        n_process,
        artifacts=ctx["artifacts"],
    )
    return dens, frozen


def lls2012_block(art, horizon, model_grid, block_feats, frozen, cfg, ctx):
    """Lanne-Luoto-Saikkonen (2012)."""
    r, s, phi, psi, alpha, beta, sigma = _mar_params(art, cfg)
    p = r + s
    lc = cfg["lls2012_params"]
    n_process = ctx["n_process"]
    feats = block_feats[p]

    def _lls(hist):
        pdf, _ = lls2012_predictive_density_given_yt(
            psi_hat=psi,
            phi_hat=phi,
            alpha_hat=alpha,
            beta_hat=beta,
            sigma_hat=sigma,
            y=list(hist),
            horizon=horizon,
            M=lc["M"],
            N_draws=lc["N_DRAWS"],
            grid_y=model_grid,
        )
        return pdf

    dens = np.array(Parallel(n_jobs=n_process)(delayed(_lls)(f) for f in feats))
    X_cali, y_cali = lagged_pairs(art.calibration, horizon, p)
    dens_cali = np.array(Parallel(n_jobs=n_process)(delayed(_lls)(x) for x in X_cali))
    dens = recalibrate_density(
        X_cali,
        y_cali,
        dens_cali,
        feats,
        dens,
        model_grid,
        n_process,
        artifacts=ctx["artifacts"],
    )
    return dens, frozen


MODEL_FNS = {
    "kcde": kcde_block,
    "flexzboost": flexzboost_block,
    "mdn": mdn_block,
    "gj2026": gj2026_block,
    "lls2012": lls2012_block,
    "ar1": exact_block,
    "araic": exact_block,
    "exp_smoothing": exact_block,
}

EXACT_BLOCK_SPECS = {"ar1": "ar1", "araic": "araic", "exp_smoothing": "exp"}


def _normalise_on_grid(density, grid, model, logger):
    """Renormalise each row to unit mass on ``grid`` and report what fell outside."""
    mass = np.trapz(density, grid, axis=1)
    outside = 1.0 - mass
    if np.nanmax(np.abs(outside)) > 1e-6:
        logger.info(
            f"[{model}] renormalised on the capped support: mass outside "
            f"median {np.nanmedian(outside):.2%}, worst {np.nanmax(outside):.2%}"
        )
    safe = np.where(mass > 0, mass, 1.0)[:, None]
    return density / safe


def _pool_prices(data, origins, sample_start, log=True):
    """Pool observed prices (final vintage + real-time) to size the grid."""
    months = [origins[0].origin_month] + [o.target_month for o in origins]
    final = final_series(data["NG_May24"], data["CPI_May24"], months, log=log)
    rt = origin_features(
        data["NG_HENRY"], data["CPI_AC"], origins, 1, sample_start=sample_start, log=log
    )[:, 0]
    return np.concatenate([np.asarray(final).ravel(), rt])


def _param_sort_key(name):
    """Order phi_1..phi_r, then psi_1..psi_s, then alpha, beta, sigma."""
    if name.startswith("phi_"):
        return (0, int(name.split("_")[1]))
    if name.startswith("psi_"):
        return (1, int(name.split("_")[1]))
    return {"alpha": (2, 0), "beta": (3, 0), "sigma": (4, 0)}.get(name, (5, 0))


def _param_display_name(name):
    """MAR internal parameter name -> LaTeX symbol."""
    if name.startswith("phi_"):
        return f"$\\phi_{{{name.split('_')[1]}}}$"
    if name.startswith("psi_"):
        return f"$\\psi_{{{name.split('_')[1]}}}$"
    return {"alpha": r"$\alpha$", "beta": r"$\beta$", "sigma": r"$\sigma$"}.get(
        name, name
    )


def generate_params_table(summary):
    """Build the LaTeX table of MAR(r, s) parameter estimates.

    Covers ML and GCov, real-time in-sample and full-period post-revised, with
    standard errors in parentheses and significance stars. Input is a
    ``mar_inference_summary``-shaped dict (``applications.run_estimation_only``).

    Significance levels:
    - *** p < 0.01
    - **  p < 0.05
    - *   p < 0.10
    """
    rt = summary.get("real_time_in_sample", {})
    full = summary.get("full_period_post_revised", {})
    rt_mle = rt.get("mle", {})
    rt_gcov = rt.get("gcov", {})
    full_mle = full.get("mle", {})
    full_gcov = full.get("gcov", {})

    all_params = set(rt_mle) | set(rt_gcov) | set(full_mle) | set(full_gcov)
    all_params.discard("error")
    param_order = sorted(all_params, key=_param_sort_key)

    if not param_order:
        print("Warning: No parameters found in inference summary")
        return None

    r = sum(1 for p in param_order if p.startswith("phi_"))
    s = sum(1 for p in param_order if p.startswith("psi_"))
    caption = f"Estimated MAR({r},{s}) parameters for the real Henry Hub spot price"
    label = "tab:gas_params"

    def get_stars(p_value):
        """Return significance stars based on p-value."""
        if p_value is None:
            return ""
        if p_value < 0.01:
            return "$^{***}$"
        elif p_value < 0.05:
            return "$^{**}$"
        elif p_value < 0.10:
            return "$^{*}$"
        return ""

    def format_estimate(entry):
        """Format estimate with std in parentheses and significance stars."""
        if entry is None:
            return "--"
        value = entry.get("estimate")
        if value is None:
            return "--"

        if abs(value) < 0.001:
            val_str = f"{value:.4f}"
        else:
            val_str = f"{value:.3f}"

        if value < 0:
            val_str = f"${val_str}$"

        stars = get_stars(entry.get("p_value"))

        std = entry.get("std_error")
        std_str = f"({std:.3f})" if std is not None else ""

        return f"\\makecell{{{val_str}{stars} \\\\ {std_str}}}"

    latex = []
    latex.append(r"\begin{table}[!htbp]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\begin{threeparttable}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(
        r" & \multicolumn{2}{c}{Real-Time In-Sample} & \multicolumn{2}{c}{Full Period "
        r"Post-Revised} \\"
    )
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    latex.append(r"Parameter & ML & GCov & ML & GCov \\")
    latex.append(r"\midrule")

    for i, param in enumerate(param_order):
        display_name = _param_display_name(param)
        row = [display_name]
        for method_dict in (rt_mle, rt_gcov, full_mle, full_gcov):
            row.append(format_estimate(method_dict.get(param)))
        latex.append(" & ".join(row) + r" \\")
        if i < len(param_order) - 1:
            latex.append(r" & & & & \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{tablenotes}[para,flushleft]")
    latex.append(r"\footnotesize")
    latex.append(
        r"\item \textit{Note:} Standard errors in parentheses. $^{***}$ $p<0.01$, "
        r"$^{**}$ $p<0.05$, $^{*}$ $p<0.10$."
    )
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{threeparttable}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def run_estimation_only(cfg, app, data, logger, log=False):
    """Fit the MAR on the first origin and on the full post-revised sample.

    Runs when no --model is given, and writes the parameter table of the paper.
    """
    order = tuple(app["order"])
    ss = app["sample_start"]
    H, K = app["gcov_params"]["H"], app["gcov_params"]["K"]
    ng, cpi = data["NG_HENRY"], data["CPI_AC"]

    origins = enumerate_origins(1, ng.shape[1], first_origin=app["first_origin"])
    logger.info(
        f"ESTIMATION-ONLY: MLE and GCOV MAR{order} on "
        f"{'log ' if log else ''}real prices"
    )

    # Console summary: first estimation vs full post-revised estimation
    first = origins[0]
    y_first = vintage_logprice(ng, cpi, first, sample_start=ss, log=log)

    last_col = ng.shape[1] - 1
    r0 = _row(ss)
    real_full = 100.0 * ng[r0:, last_col] / cpi[r0:, last_col]
    y_full = np.log(real_full) if log else real_full

    def _show(title, y, seed_col):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        result = {}
        for method in ESTIMATIONS:
            try:
                _, mar = fit_mar(
                    y,
                    order,
                    method,
                    H,
                    K,
                    rng_seed=cfg["seed"] + seed_col,
                    warm_start=_causal_pole_start(order, y),
                )
                inf = mar.inference(y, mar.par)
                print(f"\n[{method.upper()}]  n={len(y)}")
                print(inf.to_string(index=False))
                result[method] = {
                    r["Parameter"]: {
                        "estimate": float(r["Estimate"]),
                        "std_error": float(r["Std. Error"]),
                        "t_statistic": float(r["t-statistic"]),
                        "p_value": float(r["p-value"]),
                    }
                    for _, r in inf.iterrows()
                }
            except Exception as e:
                print(f"\n[{method.upper()}]  inference failed: {e}")
                result[method] = {"error": str(e)}
        return result

    print(f"\nMAR{order} on the {'log ' if log else ''}real gas price")
    summary = {
        "real_time_in_sample": _show(
            f"FIRST ESTIMATION  |  real-time initial window "
            f"({first.origin_month}, from {ss})",
            y_first,
            first.vintage_col,
        ),
        "full_period_post_revised": _show(
            f"FULL POST-REVISED ESTIMATION  |  latest vintage, full sample (from {ss})",
            y_full,
            last_col,
        ),
    }
    print()

    params_table = generate_params_table(summary)
    if params_table is not None:
        tex_path = "outputs/applications/mar_inference_summary.tex"
        with open(tex_path, "w") as f:
            f.write(params_table + "\n")
        logger.info(f"Saved MLE+GCOV parameter table -> {tex_path}")
    else:
        logger.warning("No parameters found; mar_inference_summary.tex not written")


def main():
    """Run one recursive real-time forecasting job from the command line."""
    logger = setup_logger()
    device = setup_device()
    n_process = setup_config_device(get_allowed_cpu_count())

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    app = cfg["application"]

    parser = argparse.ArgumentParser()
    for arg in cfg["arguments"]:
        name = arg.pop("name")
        if name == "--order":  # the application order is fixed by config.yaml
            continue
        if isinstance(arg.get("type"), str):
            arg.pop("type")
        parser.add_argument(name, **arg)

    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--n_trials", type=int, default=None, help="Optuna trials (mdn)."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Load cached level densities if present; else fit and cache.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Score the cached final densities directly, without refitting the "
        "recalibrator. Implies --evaluate.",
    )
    args = parser.parse_args()
    args.evaluate = args.evaluate or args.rescore

    model, horizon = args.model, args.horizon
    estimation = "mle"
    order = tuple(app["order"])
    LOG = True
    sample_start = app["sample_start"]
    simulation_seed = app["simulation_seed"]
    exact_spec = EXACT_BLOCK_SPECS.get(model)  # closed-form predictive density
    if exact_spec is not None and not LOG:
        logger.info(f"[{model}] exact benchmark: forcing log space")
        LOG = True
    length_simulation = cfg["length_simulation"]
    length_calibration = cfg["length_calibration"]
    if exact_spec is not None:
        training_dgp = f"{exact_spec}-exact-gaussian"
    else:
        training_dgp = f"MAR{order}-{estimation}"

    # Data
    NG_HENRY = np.loadtxt("data/NG_HENRY.txt")
    CPI_AC = np.loadtxt("data/CPI_AC.txt")
    mat = loadmat("data/HH_CPI_May2024vintage.mat")
    data = {
        "NG_HENRY": NG_HENRY,
        "CPI_AC": CPI_AC,
        "NG_May24": mat["NG_May24"],
        "CPI_May24": mat["CPI_May24"],
    }

    if model is None:
        run_estimation_only(cfg, app, data, logger, LOG)
        return

    if exact_spec is None:
        model_key = model
    else:
        model_key = model if estimation == "mle" else f"{model}_{estimation}"
    model_path = (
        model_key.rsplit("_", 1)[0] + "/" + model_key.rsplit("_", 1)[1]
        if exact_spec is not None
        else model_key
    )
    out_root = "outputs/applications"
    file_name = f"{out_root}/horizon_{horizon}"
    model_dir = f"{file_name}/{model_path}"
    os.makedirs(model_dir, exist_ok=True)
    density_path = f"{model_dir}/{model_key}_densities_level.parquet"
    log_density_path = f"{model_dir}/{model_key}_densities_log.parquet"
    raw_density_path = f"{model_dir}/{model_key}_densities_raw.parquet"
    calibration_path = f"{model_dir}/{model_key}_calibration.parquet"
    target_feats_path = f"{model_dir}/{model_key}_target_features.parquet"

    logger.info("=" * 60)
    logger.info(
        f"RECURSIVE APPLICATION | model={model} horizon={horizon} "
        f"estimation={estimation} training_dgp={training_dgp} "
        f"space={'logs' if LOG else 'levels'} "
        f"train={length_simulation} cali={length_calibration}"
    )
    logger.info("=" * 60)

    # Origins, blocks, fixed grids
    origins = enumerate_origins(
        horizon, NG_HENRY.shape[1], first_origin=app["first_origin"]
    )
    blocks = make_blocks(origins, REESTIMATION_FREQ)
    n_grid = cfg["n_points_grid"]
    prices = _pool_prices(data, origins, sample_start, log=False)
    prices = prices[prices > 0]
    LEVEL_LO = float(np.min(prices)) / GRID_CAP
    LEVEL_HI = float(np.max(prices)) * GRID_CAP
    LOG_LO, LOG_HI = float(np.log(LEVEL_LO)), float(np.log(LEVEL_HI))
    level_grid = np.linspace(LEVEL_LO, LEVEL_HI, n_grid)
    log_grid = build_log_grid(
        _pool_prices(data, origins, sample_start, log=True),
        n_grid,
        lo=LOG_LO,
        hi=LOG_HI,
    )

    model_grid = log_grid if LOG else build_level_model_grid(n_grid, LEVEL_HI)
    logger.info(
        f"{len(origins)} origins in {len(blocks)} block(s); "
        f"{model} fitted in {'logs' if LOG else 'levels'}; "
        f"model grid [{model_grid[0]:.2f},{model_grid[-1]:.2f}] "
        f"level grid [{level_grid[0]:.2f},{level_grid[-1]:.2f}]"
    )

    # Targets & no-change benchmark
    targets_level = origin_targets(
        data["NG_May24"], data["CPI_May24"], origins, log=False
    )
    naive_level = origin_features(
        NG_HENRY, CPI_AC, origins, 1, sample_start=sample_start, log=False
    )[:, 0]
    se_naive = (targets_level - naive_level) ** 2
    mspe_naive = float(np.mean(se_naive))
    logger.info(f"No-change MSPE (levels, h={horizon}): {mspe_naive:.6f}")

    # Fit or load
    frozen = None  # per-model choices frozen on block 0; recorded in run_meta
    recalibrated_from_cache = bool(
        args.evaluate
        and not args.rescore
        and os.path.exists(raw_density_path)
        and os.path.exists(calibration_path)
        and os.path.exists(target_feats_path)
    )
    loaded_from_cache = recalibrated_from_cache or bool(
        args.evaluate and os.path.exists(density_path)
    )
    if recalibrated_from_cache:
        logger.info(
            f"[evaluate] refitting the recalibrator on cached PITs from "
            f"{calibration_path} at num_basis={DEFAULT_NUM_BASIS}"
        )
        cal = pd.read_parquet(calibration_path)
        target_feats = pd.read_parquet(target_feats_path).values
        density_raw_cached = pd.read_parquet(raw_density_path).values

        tasks, slices, row = [], [], 0
        for b, block in enumerate(blocks):
            g = cal[cal["block"] == b]
            sl = slice(row, row + len(block))
            tasks.append(
                (
                    g.drop(columns=["block", "pit"]).to_numpy(),
                    g["pit"].to_numpy(),
                    target_feats[sl],
                    density_raw_cached[sl],
                )
            )
            slices.append(sl)
            row += len(block)

        parts = Parallel(n_jobs=n_process)(
            delayed(fit_and_apply_recalibration)(
                X_cal,
                pit_cal,
                X_tgt,
                dens_raw,
                model_grid,
                1,
                num_basis=DEFAULT_NUM_BASIS,
            )
            for X_cal, pit_cal, X_tgt, dens_raw in tqdm(
                tasks, desc=f"[{model}] recalibrating blocks"
            )
        )
        density = np.empty_like(density_raw_cached)
        for sl, part in zip(slices, parts):
            density[sl] = part
        density = _normalise_on_grid(density, model_grid, model, logger)
        check_cde(density, model_grid)
        if LOG:
            density_log = density
            density_level = log_density_to_level(density, log_grid, level_grid)
        else:
            density_log = None
            density_level = restrict_to_level_grid(density, model_grid, level_grid)
        pd.DataFrame(
            density_level, columns=[f"y_{i}" for i in range(density_level.shape[1])]
        ).to_parquet(density_path, index=False)
        if density_log is not None:
            pd.DataFrame(
                density_log, columns=[f"y_{i}" for i in range(density_log.shape[1])]
            ).to_parquet(log_density_path, index=False)
    elif loaded_from_cache:
        logger.info(f"Loading cached level densities from {density_path}")
        density_level = pd.read_parquet(density_path).values
        density_log = (
            pd.read_parquet(log_density_path).values
            if os.path.exists(log_density_path)
            else None
        )
    else:
        feats_by_lag = {
            lags: origin_features(
                NG_HENRY,
                CPI_AC,
                origins,
                lags,
                sample_start=sample_start,
                log=LOG,
            )
            for lags in range(1, N_LAGS + 1)
        }
        block_kwargs = dict(sample_start=sample_start, log=LOG)
        if exact_spec is not None:
            fetch_block = get_exact_block
            block_kwargs.update(
                spec=exact_spec,
                estimation=estimation,
            )
        else:
            fetch_block = get_block
            block_kwargs.update(
                order=order,
                estimation=estimation,
                H=app["gcov_params"]["H"],
                K=app["gcov_params"]["K"],
                length_simulation=length_simulation,
                length_calibration=length_calibration,
                seed=simulation_seed,
            )
        ctx = dict(
            logger=logger,
            device=device,
            n_process=n_process,
            model_dir=model_dir,
            evaluate=args.evaluate,
            n_trials=args.n_trials,
            model=model,
            ng=NG_HENRY,
            cpi=CPI_AC,
            sample_start=sample_start,
            log=LOG,
            artifacts={},  # refilled by recalibrate_density on every block
        )

        density = np.empty((len(origins), n_grid), dtype=float)
        density_raw = np.empty((len(origins), n_grid), dtype=float)
        cal_rows, target_feats = [], []
        frozen, row = None, 0
        warm_start = (
            None
            if exact_spec is not None
            else _causal_pole_start(
                order,
                vintage_logprice(
                    NG_HENRY, CPI_AC, blocks[0][0], sample_start=sample_start, log=LOG
                ),
            )
        )
        for b, block in enumerate(blocks):
            if exact_spec is None:
                block_kwargs["warm_start"] = warm_start
            art = fetch_block(b, block, NG_HENRY, CPI_AC, **block_kwargs)
            warm_start = art.par or None
            block_feats = {
                lags: feats_by_lag[lags][row : row + len(block)]
                for lags in feats_by_lag
            }
            ctx["artifacts"] = {}
            dens_block, frozen = MODEL_FNS[model](
                art, horizon, model_grid, block_feats, frozen, cfg, ctx
            )
            density[row : row + len(block)] = dens_block
            art_block = ctx["artifacts"]
            density_raw[row : row + len(block)] = art_block.get("raw", dens_block)
            if "pit" in art_block:
                cal = pd.DataFrame(art_block["X_calibration"])
                cal.columns = [f"x_{i}" for i in range(cal.shape[1])]
                cal.insert(0, "pit", art_block["pit"])
                cal.insert(0, "block", b)
                cal_rows.append(cal)
                target_feats.append(np.asarray(art_block["X_target"], dtype=float))
            row += len(block)
            logger.info(
                f"[{model}] block {b + 1}/{len(blocks)} done ({len(block)} origins)"
            )

        density = _normalise_on_grid(density, model_grid, model, logger)
        check_cde(density, model_grid)
        if LOG:
            density_log = density
            density_level = log_density_to_level(density, log_grid, level_grid)
        else:
            density_log = None
            density_level = restrict_to_level_grid(density, model_grid, level_grid)
        pd.DataFrame(
            density_level, columns=[f"y_{i}" for i in range(density_level.shape[1])]
        ).to_parquet(density_path, index=False)
        if density_log is not None:
            pd.DataFrame(
                density_log, columns=[f"y_{i}" for i in range(density_log.shape[1])]
            ).to_parquet(log_density_path, index=False)

        pd.DataFrame(
            density_raw, columns=[f"y_{i}" for i in range(density_raw.shape[1])]
        ).to_parquet(raw_density_path, index=False)
        if cal_rows:
            pd.concat(cal_rows, ignore_index=True).to_parquet(
                calibration_path, index=False
            )
            feats = np.concatenate(target_feats, axis=0)
            pd.DataFrame(
                feats, columns=[f"x_{i}" for i in range(feats.shape[1])]
            ).to_parquet(target_feats_path, index=False)
            logger.info(
                f"Saved uncorrected densities and calibration PITs "
                f"(num_basis={DEFAULT_NUM_BASIS}); rerun with --evaluate to "
                f"refit the recalibrator without refitting the model"
            )

    # At each origin, the 90% quantile of the real price observed so far on that
    # origin's own vintage, and the price the forecast conditions on.
    r90, origin_price = origin_price_quantile(
        NG_HENRY, CPI_AC, origins, sample_start, TAIL_Q
    )
    weight_thresholds = {TAIL_Q: r90}
    elevated_idx = origin_price > r90
    logger.info(
        f"weighted-score threshold q{TAIL_Q:.0%}: {r90.min():.4f} -> "
        f"{r90.max():.4f}; {int(elevated_idx.sum())} of {len(r90)} origins "
        f"elevated (expanding window, own vintage)"
    )

    # Point forecasts, evaluated in levels against the final vintage.
    cond_median = _density_median(density_level, level_grid)

    se_median = (cond_median - targets_level) ** 2
    mspe_region = mspe_by_region(
        {"median": se_median}, se_naive, targets_level, TAIL_Q, tails_idx=elevated_idx
    )
    total = mspe_region["total"]
    mspe_median = total["mspe_median"]
    mspe_ratio_median = total["mspe_ratio_median"]

    tails_idx = elevated_idx
    pd.DataFrame(
        {
            "origin_month": [str(o.origin_month) for o in origins],
            "target_month": [str(o.target_month) for o in origins],
            "target": targets_level,
            "cond_median": cond_median,
            "naive": naive_level,
            "se_median": se_median,
            "se_naive": se_naive,
            "region": np.where(tails_idx, "tails", "center"),
        }
    ).to_parquet(f"{model_dir}/{model_key}_point_forecasts.parquet", index=False)

    # Density-forecast evaluation in levels (CRPS/CDE/coverage)
    cde_metrics = evaluate_and_plot_densities(
        predictive_density=density_level,
        grid_y=level_grid,
        ts=pd.Series(targets_level),
        horizon=horizon,
        file_name=file_name,
        model=model_key,
        output_dir=model_dir,
        save_densities=False,
        density_log=density_log,
        log_grid=log_grid if LOG else None,
        weight_thresholds=weight_thresholds,
        qs_alphas=APPLICATION_QS_ALPHAS,
    )
    # Save point-forecast MSPE metrics (mean & median relative to no-change),
    # for the full sample and for the center/tails regions of the density scores.
    point_metrics = {
        "model": model_key,
        "horizon": horizon,
        "n_origins": len(origins),
        "tail_q": TAIL_Q,
        "mspe_median": mspe_median,
        "mspe_ratio_median": mspe_ratio_median,
    }
    for region in ("center", "tails"):
        for key, value in mspe_region[region].items():
            point_metrics[f"{region}_{key}"] = value
    with open(f"{model_dir}/{model_key}_point_metrics.json", "w") as f:
        json.dump(point_metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"EVALUATION (levels) | {model} h={horizon}")
    logger.info(f"MSPE ratio (cond. median / naive): {mspe_ratio_median:.6f}")
    logger.info(
        f"CDE Loss: {cde_metrics['cde_loss']:.6e} | CRPS: {cde_metrics['crps']:.6f} "
        f"| Log score: {cde_metrics['log_prob']:.4f}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
