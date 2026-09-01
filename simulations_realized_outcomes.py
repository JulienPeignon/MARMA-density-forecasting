"""Monte Carlo study of density forecasts scored against realised outcomes."""

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
from tqdm import tqdm

from src.calibration.recalibration import (
    DEFAULT_NUM_BASIS,
    fit_and_apply_recalibration,
    lagged_pairs,
    recalibrate_density,
)
from src.conditional_theoretical_moments.theoretical_quantiles import (
    compute_theoretical_quantiles,
)
from src.evaluate.cdetools import cde_loss
from src.evaluate.evaluate_applications import (
    evaluate_and_plot_densities,
)
from src.forecast_methods.gj2026 import gj2026_predictive_density_given_yt
from src.forecast_methods.kcde import kcde_predictive_density
from src.forecast_methods.lls2012 import lls2012_predictive_density_given_yt
from src.forecast_methods.MDN import MixtureDensityNetwork
from src.forecast_methods.utils import (
    check_cde,
    make_grid,
    prepare_tensors,
)
from src.stable_mar.stable_mar import (
    check_marma_existence,
    fit_marma,
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

GRID_MARGIN = 0.10
N_EVAL = 1000
N_LAGS = 10


def main():
    """Run one simulation job scored against realised outcomes."""
    # Logger and device
    logger = setup_logger()
    device = setup_device()
    cpu_count = get_allowed_cpu_count()
    n_process = setup_config_device(cpu_count)

    # Arguments and config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    arg_config = cfg["arguments"]

    parser = argparse.ArgumentParser()

    for arg in arg_config:
        name = arg.pop("name")
        if isinstance(arg.get("type"), str):
            arg.pop("type")
        parser.add_argument(name, **arg)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides optuna_params.N_TRIALS).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Skip tuning; reload the saved best config (+ checkpoint) and evaluate.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Score the cached final densities directly, without refitting the "
        "recalibrator. Implies --evaluate.",
    )
    parser.add_argument(
        "--estimation",
        choices=["gcov", "mle", "true"],
        default="mle",
        help=(
            "MAR parameter source: 'gcov', 'mle' (default, alpha-stable MLE), "
            "or 'true' (the simulation's generating parameters)."
        ),
    )
    args = parser.parse_args()
    args.evaluate = args.evaluate or args.rescore

    nw_cfg = cfg["kcde_params"]
    lls2012_cfg = cfg["lls2012_params"]
    boost_cfg = cfg["flexzboost_params"]
    mdn_cfg = cfg["mdn_params"]
    dl = mdn_cfg["dataloaders_params"]
    m = mdn_cfg["model_params"]
    tr = mdn_cfg["training_params"]

    logger.info("=" * 60)
    logger.info("CONFIGURATION PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"length_simulation: {cfg['length_simulation']}")
    logger.info(f"length_calibration: {cfg['length_calibration']}")
    logger.info(f"n_points_grid: {cfg['n_points_grid']}")
    logger.info(f"CONDITIONAL_LAGS: {dl['CONDITIONAL_LAGS']}")
    logger.info("Nadaraya-Watson params:")
    logger.info(f"  KERNEL: {nw_cfg['KERNEL']}")
    logger.info(f"  BANDWIDTH: {nw_cfg['BANDWIDTH']}")
    logger.info("FlexZBoost params:")
    logger.info(f"  MAX_BASIS: {boost_cfg['MAX_BASIS']}")
    logger.info(f"  BASIS_SYSTEM: {boost_cfg['BASIS_SYSTEM']}")
    logger.info("MDN params:")
    logger.info(f"  Model: DENSITY={m['DENSITY']}")
    logger.info(f"  Dataloaders: CONDITIONAL_LAGS={dl['CONDITIONAL_LAGS']}")
    logger.info(
        f"  Training: BATCH_SIZE={tr['BATCH_SIZE']}, MAX_EPOCHS={tr['MAX_EPOCHS']}"
    )
    logger.info(
        f"  Training: MAX_GRAD_NORM={tr['MAX_GRAD_NORM']}, "
        f"EARLY_STOPPING={tr['EARLY_STOPPING_PATIENCE']}"
    )
    logger.info(
        f"  Training: SCHEDULER_PATIENCE={tr['SCHEDULER_PATIENCE']}, "
        f"SCHEDULER_FACTOR={tr['SCHEDULER_FACTOR']}"
    )
    logger.info("=" * 60)

    horizon = args.horizon
    LAGS = dl["CONDITIONAL_LAGS"]
    prediction_offset = LAGS + horizon - 1

    # Process / order parameters (see simulations.py for the generation logic)
    order_mar = tuple(int(x) for x in args.order)
    r, s = order_mar[0], order_mar[1]
    mar_cfg = cfg["mar_params"]
    mar_params = {
        tuple(map(int, k.strip("()").split(","))): v for k, v in mar_cfg.items()
    }
    params_alpha_mar = {**mar_params[order_mar], **cfg["alpha_stable_params"]}

    if args.alpha is not None:
        params_alpha_mar["ALPHA"] = float(args.alpha)

    PSI, PHI, ALPHA, BETA, SIGMA, THETA, ETA = (
        params_alpha_mar.get(k)
        for k in ["PSI", "PHI", "ALPHA", "BETA", "SIGMA", "THETA", "ETA"]
    )

    is_marma = len(order_mar) == 4
    model_type = "MARMA" if is_marma else "MAR"

    logger.info(f"Selected model: {model_type}{order_mar}")
    logger.info(f"Simulation parameters: {params_alpha_mar}")

    model_valid = check_marma_existence(
        psi_vec=[PSI] if isinstance(PSI, float) else PSI,
        phi_vec=[PHI],
        theta_vec=THETA,
        eta_vec=ETA,
    )
    logger.info(
        f"Model validation - MARMA existence conditions satisfied: {model_valid}"
    )

    file_name = (
        f"outputs/simulations_realized_outcomes/{model_type}{order_mar}"
        f"/horizon_{horizon}/alpha_{ALPHA}"
    )
    logger.info(f"Predictive densities & graphics will be saved in: {file_name}")

    # Setup model (MAR only; MARMA is simulated directly)
    if not is_marma:
        mar = sm(order_mar)
        mar.model = model_type
        mar.par = (
            [PHI, PSI, ALPHA, BETA, SIGMA]
            if order_mar == (1, 1)
            else [PSI[0], PSI[1], ALPHA, BETA, SIGMA]
            if order_mar == (0, 2)
            else [PSI, ALPHA, BETA, SIGMA]
        )

    estimated_params = None
    estimated_marma = None

    def generate_series(n, use_estimated=False):
        """Generate a length-n trajectory for the selected process."""
        if is_marma and use_estimated and estimated_marma is not None:
            ps, ph, th, et, al, be, si = estimated_marma
            traj = simulate_MARMA(
                N=float(n),
                psi=ps,
                phi=ph,
                theta=th,
                eta=et,
                alpha=al,
                beta=be,
                sigma=si,
            )
            return traj.dropna().reset_index(drop=True)
        if use_estimated and estimated_params is not None:
            phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = estimated_params
            est_mar = sm(order_mar)
            est_mar.model = "MAR"
            est_mar.par = (
                (list(phi_hat) if r > 0 else [])
                + list(psi_hat)
                + [alpha_hat, beta_hat, sigma_hat]
            )
            return est_mar.generate(n).trajectory.dropna().reset_index(drop=True)
        if is_marma:
            traj = simulate_MARMA(
                N=float(n),
                psi=PSI,
                phi=PHI,
                theta=THETA,
                eta=ETA,
                alpha=ALPHA,
                beta=BETA,
                sigma=SIGMA,
            )
            return traj.dropna().reset_index(drop=True)
        return mar.generate(n).trajectory.dropna().reset_index(drop=True)

    def forecast_parameters(data):
        """Return fitted or generating MAR parameters for forecast methods."""
        if is_marma:
            raise ValueError("LLS2012 and GJ2026 do not support MARMA processes.")

        if args.estimation == "true":
            phi_hat = [0.0] if r == 0 else [PHI]
            psi_hat = [PSI] if s == 1 else list(PSI)
            return phi_hat, psi_hat, ALPHA, BETA, SIGMA

        mar_model = sm(order=order_mar)
        if args.estimation == "mle":
            mar_model.fit(data, None, method="mle", verbose=True)
        else:
            random_guess = mar_model.generate_initial_guess(random=True)
            coefficients = mar_model.fit(
                data, random_guess, H=2, K=2, method="gcov", verbose=True
            ).results["Parameters"]
            resid, _ = mar_model._pseudo_residuals(data, coefficients)
            resid = mar_model.results.get("PseudoResiduals")
            mar_model.par.extend(mar_model.fit_stable_noise(resid))

        return (
            [0.0] if r == 0 else mar_model.par[:r],
            mar_model.par[r : r + s],
            mar_model.par[r + s],
            mar_model.par[r + s + 1],
            mar_model.par[r + s + 2],
        )

    def estimate_marma_params(series, method):
        """Fit a MARMA via fit_marma; return (psi, phi, theta, eta, alpha, beta, sigma).

        Mapping (validated for order (1,1,1,1), the only MARMA in config.yaml):
        simulate_MARMA(psi, phi, theta, eta) <-> fit_marma phi_nc=[psi], phi_c=[phi],
        theta_nc=[theta], theta_c=[eta]. For gcov (dynamics only) the stable
        parameters are recovered from the filtered innovations, mirroring the MAR path.
        """
        if tuple(order_mar) != (1, 1, 1, 1):
            raise NotImplementedError(
                f"MARMA estimation is implemented for order (1,1,1,1); got {order_mar}."
            )
        res = fit_marma(
            np.asarray(series, dtype=float),
            order=(1, 1, 1, 1),
            method=method,
            seed=cfg["seed"],
        )
        psi_h = float(res["phi_nc"][0])
        phi_h = float(res["phi_c"][0])
        theta_h = float(res["theta_nc"][0])
        eta_h = float(res["theta_c"][0])
        if method == "mle":
            alpha_h, beta_h, sigma_h = res["alpha"], res["beta"], res["sigma"]
        else:  # gcov returns dynamics only
            alpha_h, beta_h, sigma_h = sm(order=(1, 1)).fit_stable_noise(res["E"])
        return (psi_h, phi_h, theta_h, eta_h, alpha_h, beta_h, sigma_h)

    if args.model == "kcde":
        model = "kcde"
    elif args.model == "lls2012":
        model = "lls2012"
    elif args.model == "gj2026":
        model = "gj2026"
    elif args.model == "flexzboost":
        model = "FlexZBoost"
    elif args.model == "mdn":
        model = "mdn"
    else:
        raise ValueError("No model selected.")

    estimation_series = None
    if is_marma:
        if args.estimation == "true":
            estimated_marma = (PSI, PHI, THETA, ETA, ALPHA, BETA, SIGMA)
            logger.info("MARMA: using generating parameters (--estimation true).")
        else:
            logger.info(
                "Simulating %d observations for MARMA %s estimation...",
                cfg["length_estimation"],
                args.estimation,
            )
            estimation_series = generate_series(cfg["length_estimation"])
            estimated_marma = estimate_marma_params(estimation_series, args.estimation)
        ps, ph, th, et, al, be, si = estimated_marma
        logger.info(
            f"MARMA forecast parameters ({args.estimation}):"
            f"\n  psi_hat   = {ps:.4f}"
            f"\n  phi_hat   = {ph:.4f}"
            f"\n  theta_hat = {th:.4f}"
            f"\n  eta_hat   = {et:.4f}"
            f"\n  alpha_hat = {al:.4f}"
            f"\n  beta_hat  = {be:.4f}"
            f"\n  sigma_hat = {si:.4f}"
        )
    else:
        logger.info(
            "Simulating %d observations for MAR parameter estimation...",
            cfg["length_estimation"],
        )
        estimation_series = generate_series(cfg["length_estimation"])
        estimated_params = forecast_parameters(estimation_series)

        phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = estimated_params
        logger.info(
            f"Forecast parameters ({args.estimation}, n={len(estimation_series)}):"
            f"\n  phi_hat  = {phi_hat}"
            f"\n  psi_hat  = {psi_hat}"
            f"\n  alpha_hat = {alpha_hat:.4f}"
            f"\n  beta_hat  = {beta_hat:.4f}"
            f"\n  sigma_hat = {sigma_hat:.4f}"
        )

    logger.info("Simulating training data...")
    y_train_sim = generate_series(cfg["length_simulation"], use_estimated=True)

    logger.info("Simulating test data...")
    set_seed(cfg["test_seed"])
    y_test_sim = generate_series(N_EVAL + N_LAGS + horizon - 1, use_estimated=False)

    data_for_training = None if model == "lls2012" else y_train_sim
    data_to_predict = y_test_sim
    n_predictions = len(data_to_predict) - prediction_offset

    if data_for_training is not None:
        logger.info(f"Training data: {len(data_for_training)} samples")
    else:
        logger.info("LLS2012 uses no simulated training series.")
    logger.info(f"Test data: {len(data_to_predict)} samples")
    logger.info(f"Number of predictions: {n_predictions}")

    # Define grid using theoretical quantiles (same as simulations.py)
    quantiles = compute_theoretical_quantiles(
        phi_vec=[PHI],
        psi_vec=[PSI] if isinstance(PSI, float) else PSI,
        alpha=ALPHA,
        beta=BETA,
        sigma=SIGMA,
        theta=THETA,
        eta=ETA,
    )

    _, grid_y = make_grid(quantiles, cfg["n_points_grid"])

    data_min = float(min(np.min(y_train_sim), np.min(y_test_sim)))
    data_max = float(max(np.max(y_train_sim), np.max(y_test_sim)))
    data_range_message = (
        f"train range: [{float(np.min(y_train_sim)):.3f}, "
        f"{float(np.max(y_train_sim)):.3f}], "
        f"test range: [{float(np.min(y_test_sim)):.3f}, "
        f"{float(np.max(y_test_sim)):.3f}]"
    )
    grid_y_min = min(float(grid_y[0]), data_min)
    grid_y_max = max(float(grid_y[-1]), data_max)
    margin = GRID_MARGIN * (grid_y_max - grid_y_min)
    grid_y_min -= margin
    grid_y_max += margin
    grid_y = np.linspace(grid_y_min, grid_y_max, cfg["n_points_grid"])
    logger.info(
        f"grid_y range extended to cover data (+{GRID_MARGIN:.0%} margin): "
        f"[{grid_y_min:.3f}, {grid_y_max:.3f}] "
        f"({data_range_message})"
    )

    if data_for_training is not None:
        (
            X_train,
            y_train,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = prepare_tensors(
            pd.DataFrame(data_for_training),
            X=None,
            y=None,
            lags=LAGS,
            horizon=horizon,
            proportions=(1.0, 0.0, 0.0),
            device=device,
        )
    else:
        X_train, y_train = None, None

    model_dir = f"{file_name}/{model}"
    density_path = f"{model_dir}/{model}_densities.parquet"
    lag_path = f"{model_dir}/{model}_lags.json"

    def evaluation_inputs(lags):
        """Features and targets for ``lags``, aligned on a common test window."""
        X_eval, y_eval = lagged_pairs(data_to_predict, horizon, lags)
        if len(y_eval) < N_EVAL:
            raise ValueError(
                f"test path too short: {len(y_eval)} usable targets at lags={lags}, "
                f"horizon={horizon}; need {N_EVAL}."
            )
        X_eval, y_eval = X_eval[-N_EVAL:], y_eval[-N_EVAL:]
        offset = len(data_to_predict) - N_EVAL
        y_naive = data_to_predict.iloc[lags - 1 : -horizon].values[-N_EVAL:]
        mspe = float(np.mean((y_eval - y_naive) ** 2))
        return X_eval.astype(np.float32), y_eval, offset, mspe

    if model in {"lls2012", "gj2026"}:
        # Both need the full p = r + s history: lls2012 pre-filters the causal
        # part (v_t = y_t - phi_1 y_{t-1} - ...) before recovering the noncausal
        # state, and gj2026 builds a p-dimensional companion state. Handing them
        # fewer values silently skips that filtering instead of failing, so this
        # matches applications.py, which has always used r + s.
        selected_lags = r + s
    else:
        selected_lags = LAGS

    if args.evaluate and os.path.exists(lag_path):
        with open(lag_path) as f:
            lag_info = json.load(f)
        scores = lag_info.get("scores")
        if scores:
            direction = lag_info.get("direction", "max")
            pick = max if direction == "max" else min
            selected_lags = int(pick(scores, key=lambda k: scores[k]))
        else:
            selected_lags = int(lag_info["lags"])
    elif model == "mdn" and args.evaluate:
        mdn_config_path = f"{model_dir}/config.yaml"
        if os.path.exists(mdn_config_path):
            with open(mdn_config_path) as f:
                selected_lags = int(yaml.safe_load(f)["params"]["lags"])

    X_test, y_true, prediction_offset, mspe_naive = evaluation_inputs(selected_lags)
    n_predictions = len(X_test)
    logger.info(
        f"Using {selected_lags} lag(s) | No-change MSPE (horizon {horizon}): "
        f"{mspe_naive:.6f}"
    )

    artifacts = {}
    raw_density_path = f"{model_dir}/{model}_densities_raw.parquet"
    calibration_path = f"{model_dir}/{model}_calibration.parquet"
    target_feats_path = f"{model_dir}/{model}_target_features.parquet"
    recalibrated_from_cache = bool(
        args.evaluate
        and not args.rescore
        and os.path.exists(raw_density_path)
        and os.path.exists(calibration_path)
        and os.path.exists(target_feats_path)
    )

    if recalibrated_from_cache:
        logger.info(
            f"[evaluate] refitting the recalibrator on cached PITs from "
            f"{calibration_path} at num_basis={DEFAULT_NUM_BASIS}"
        )
        cal = pd.read_parquet(calibration_path)
        predictive_density = fit_and_apply_recalibration(
            cal.drop(columns="pit").to_numpy(),
            cal["pit"].to_numpy(),
            pd.read_parquet(target_feats_path).values,
            pd.read_parquet(raw_density_path).values,
            grid_y,
            n_process,
            num_basis=DEFAULT_NUM_BASIS,
        )
        check_cde(predictive_density, grid_y)
    elif args.evaluate and os.path.exists(density_path):
        logger.info(f"Loading predictive density from {density_path}")
        df_density = pd.read_parquet(density_path)
        predictive_density = df_density.values
    else:
        if args.evaluate:
            logger.info(
                "[evaluate] fitting (MDN uses the saved config, no Optuna); "
                "predictive densities are not cached to parquet"
            )

        logger.info(
            f"Computing {model} predictive densities | {model_type}{order_mar} | "
            f"Alpha: {ALPHA} | Horizon: {horizon}"
        )

        if model == "kcde":
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            best_score, best_lags, density_cali = -np.inf, None, None
            lag_scores = {}
            for lags in range(1, N_LAGS + 1):
                Xtr, ytr = lagged_pairs(data_for_training, horizon, lags)
                X_cali, y_cali = lagged_pairs(calibration, horizon, lags)
                candidate_density = kcde_predictive_density(
                    X_train=Xtr,
                    y_train=ytr,
                    X_query=X_cali,
                    grid_y=grid_y,
                    kernel=nw_cfg["KERNEL"],
                    bandwidth=nw_cfg["BANDWIDTH"],
                )
                score = float(
                    np.mean(
                        np.log(
                            np.asarray(
                                [
                                    np.interp(y, grid_y, density)
                                    for y, density in zip(y_cali, candidate_density)
                                ]
                            )
                            + 1e-12
                        )
                    )
                )
                lag_scores[lags] = score
                logger.info(
                    f"KCDE lag {lags:>2}: calibration log likelihood {score:.6f}"
                )
                if score > best_score:
                    best_score, best_lags, density_cali = (
                        score,
                        lags,
                        candidate_density,
                    )

            selected_lags = best_lags
            X_test, y_true, prediction_offset, mspe_naive = evaluation_inputs(
                selected_lags
            )
            n_predictions = len(X_test)
            X_train, y_train = lagged_pairs(data_for_training, horizon, selected_lags)
            X_cali, y_cali = lagged_pairs(calibration, horizon, selected_lags)
            predictive_density = kcde_predictive_density(
                X_train=X_train,
                y_train=y_train,
                X_query=X_test,
                grid_y=grid_y,
                kernel=nw_cfg["KERNEL"],
                bandwidth=nw_cfg["BANDWIDTH"],
            )
            os.makedirs(model_dir, exist_ok=True)
            with open(lag_path, "w") as f:
                json.dump(
                    {
                        "lags": selected_lags,
                        "criterion": "log_likelihood",
                        "direction": "max",
                        "scores": {str(k): v for k, v in lag_scores.items()},
                    },
                    f,
                )
            logger.info(
                f"KCDE selected {selected_lags} lag(s) with calibration log "
                f"likelihood {best_score:.6f}"
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                X_test,
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )

        elif model == "lls2012":
            phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = (
                estimated_params
                if estimated_params is not None
                else forecast_parameters(None)
            )

            logger.info(
                f"Estimated parameters:"
                f"\n  phi_hat  = {phi_hat}"
                f"\n  psi_hat  = {psi_hat}"
                f"\n  alpha_hat = {alpha_hat:.4f}"
                f"\n  beta_hat  = {beta_hat:.4f}"
                f"\n  sigma_hat = {sigma_hat:.4f}"
            )

            predictive_density = np.zeros((n_predictions, len(grid_y)))

            for i in tqdm(range(n_predictions), desc="lls2012"):
                pdf_hat, _ = lls2012_predictive_density_given_yt(
                    psi_hat=psi_hat,
                    phi_hat=phi_hat,
                    alpha_hat=alpha_hat,
                    beta_hat=beta_hat,
                    sigma_hat=sigma_hat,
                    y=X_test[i].tolist(),
                    horizon=horizon,
                    M=lls2012_cfg["M"],
                    N_draws=lls2012_cfg["N_DRAWS"],
                    grid_y=grid_y,
                )
                predictive_density[i, :] = pdf_hat

            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, selected_lags)
            density_cali = np.array(
                [
                    lls2012_predictive_density_given_yt(
                        psi_hat=psi_hat,
                        phi_hat=phi_hat,
                        alpha_hat=alpha_hat,
                        beta_hat=beta_hat,
                        sigma_hat=sigma_hat,
                        y=x.tolist(),
                        horizon=horizon,
                        M=lls2012_cfg["M"],
                        N_draws=lls2012_cfg["N_DRAWS"],
                        grid_y=grid_y,
                    )[0]
                    for x in tqdm(X_cali, desc="lls2012 calibration")
                ]
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                X_test,
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )

        elif model == "gj2026":
            phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = (
                estimated_params
                if estimated_params is not None
                else forecast_parameters(None)
            )

            logger.info(
                f"Estimated parameters:"
                f"\n  phi_hat  = {phi_hat}"
                f"\n  psi_hat  = {psi_hat}"
                f"\n  alpha_hat = {alpha_hat:.4f}"
                f"\n  beta_hat  = {beta_hat:.4f}"
                f"\n  sigma_hat = {sigma_hat:.4f}"
            )

            y_train_np = data_for_training.values

            def compute_gj2026_density(i):
                return gj2026_predictive_density_given_yt(
                    psi_hat=psi_hat,
                    phi_hat=phi_hat,
                    alpha_hat=alpha_hat,
                    beta_hat=beta_hat,
                    sigma_hat=sigma_hat,
                    y_train=y_train_np,
                    y_history=X_test[i][::-1].tolist(),
                    grid_y=grid_y,
                    bandwidth=nw_cfg["BANDWIDTH"],
                    horizon=horizon,
                )

            predictive_density = np.array(
                Parallel(n_jobs=n_process)(
                    delayed(compute_gj2026_density)(i)
                    for i in tqdm(range(n_predictions), desc="gj2026")
                )
            )
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, selected_lags)
            density_cali = np.array(
                Parallel(n_jobs=n_process)(
                    delayed(gj2026_predictive_density_given_yt)(
                        psi_hat=psi_hat,
                        phi_hat=phi_hat,
                        alpha_hat=alpha_hat,
                        beta_hat=beta_hat,
                        sigma_hat=sigma_hat,
                        y_train=y_train_np,
                        y_history=x[::-1].tolist(),
                        grid_y=grid_y,
                        bandwidth=nw_cfg["BANDWIDTH"],
                        horizon=horizon,
                    )
                    for x in tqdm(X_cali, desc="gj2026 calibration")
                )
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                X_test,
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )

        elif model == "FlexZBoost":
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            best_score, best_lags, best_model, density_cali = np.inf, None, None, None
            lag_scores = {}
            for lags in range(1, N_LAGS + 1):
                Xtr, ytr = lagged_pairs(data_for_training, horizon, lags)
                X_cali, y_cali = lagged_pairs(calibration, horizon, lags)
                candidate = flexcode.FlexCodeModel(
                    XGBoost,
                    max_basis=boost_cfg["MAX_BASIS"],
                    basis_system=boost_cfg["BASIS_SYSTEM"],
                    z_min=grid_y[0],
                    z_max=grid_y[-1],
                    regression_params={"verbosity": 0, "n_jobs": n_process},
                )
                candidate.fit(x_train=Xtr, z_train=ytr)
                candidate_density, _ = candidate.predict(
                    X_cali, n_grid=cfg["n_points_grid"]
                )
                score = float(cde_loss(candidate_density, grid_y, y_cali)[0])
                lag_scores[lags] = score
                logger.info(
                    f"FlexZBoost lag {lags:>2}: calibration CDE loss {score:.6e}"
                )
                if score < best_score:
                    best_score, best_lags, best_model, density_cali = (
                        score,
                        lags,
                        candidate,
                        candidate_density,
                    )

            selected_lags = best_lags
            X_test, y_true, prediction_offset, mspe_naive = evaluation_inputs(
                selected_lags
            )
            n_predictions = len(X_test)
            X_cali, y_cali = lagged_pairs(calibration, horizon, selected_lags)
            predictive_density, _ = best_model.predict(
                X_test, n_grid=cfg["n_points_grid"]
            )
            os.makedirs(model_dir, exist_ok=True)
            with open(lag_path, "w") as f:
                json.dump(
                    {
                        "lags": selected_lags,
                        "criterion": "cde_loss",
                        "direction": "min",
                        "scores": {str(k): v for k, v in lag_scores.items()},
                    },
                    f,
                )
            logger.info(
                f"FlexZBoost selected {selected_lags} lag(s) with calibration "
                f"CDE loss {best_score:.6e}"
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                X_test,
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )

        elif model == "mdn":
            op = cfg["optuna_params"]

            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            y_cali_sim = generate_series(cfg["length_calibration"], use_estimated=True)

            # Train/cali tensors depend on the number of lags, which is searched,
            # so build (and cache) them per lags value.
            _mdn_data_cache = {}

            def get_mdn_data(lags):
                if lags not in _mdn_data_cache:
                    Xtr, ytr, _, _, _, _, wtr, _ = prepare_tensors(
                        pd.DataFrame(data_for_training),
                        X=None,
                        y=None,
                        lags=lags,
                        horizon=horizon,
                        proportions=(1.0, 0.0, 0.0),
                        device=device,
                    )
                    Xc, yc, _, _, _, _, wc, _ = prepare_tensors(
                        pd.DataFrame(y_cali_sim),
                        X=None,
                        y=None,
                        lags=lags,
                        horizon=horizon,
                        proportions=(1.0, 0.0, 0.0),
                        device=device,
                    )
                    _mdn_data_cache[lags] = (Xtr, ytr, wtr, Xc, yc, wc)
                return _mdn_data_cache[lags]

            # Optuna hyperparameter search (minimize validation NLL)
            # lags are searched jointly with the density hyperparameters; the
            # calibration set doubles as the validation set for model selection.
            def objective(trial):
                set_seed(cfg["seed"])
                lags = trial.suggest_int(
                    "lags", op["LAGS_RANGE"][0], op["LAGS_RANGE"][1]
                )
                n_mixtures = trial.suggest_int(
                    "n_mixtures",
                    op["N_MIXTURES_RANGE"][0],
                    op["N_MIXTURES_RANGE"][1],
                )
                learning_rate = trial.suggest_categorical(
                    "learning_rate", op["LR_GRID"]
                )
                dropout = trial.suggest_float(
                    "dropout", 0.0, op["DROPOUT_MAX"], step=op["DROPOUT_STEP"]
                )
                mlp_depth = trial.suggest_int(
                    "mlp_depth", op["DEPTH_RANGE"][0], op["DEPTH_RANGE"][1]
                )
                mlp_width = trial.suggest_categorical("mlp_width", op["WIDTH_GRID"])

                Xtr, ytr, wtr, Xc, yc, wc = get_mdn_data(lags)
                candidate = MixtureDensityNetwork(
                    input_dim=lags,
                    hidden_layers=[mlp_width] * mlp_depth,
                    n_mixtures=n_mixtures,
                    dropout=dropout,
                    n_jobs=1,
                    device=device,
                ).to(device)

                try:
                    val_nll = candidate.fit(
                        X_train=Xtr,
                        y_train=ytr,
                        X_val=Xc,
                        y_val=yc,
                        weights_train=wtr,
                        weights_val=wc,
                        max_epochs=tr["MAX_EPOCHS"],
                        learning_rate=learning_rate,
                        batch_size=tr["BATCH_SIZE"],
                        max_norm=tr["MAX_GRAD_NORM"],
                        patience=tr["EARLY_STOPPING_PATIENCE"],
                        scheduler_patience=tr["SCHEDULER_PATIENCE"],
                        scheduler_factor=tr["SCHEDULER_FACTOR"],
                    )
                except Exception as e:  # numerical blow-ups, invalid configs
                    logger.warning(f"Trial {trial.number} failed: {e}")
                    return float("inf")

                if val_nll is None or not np.isfinite(val_nll):
                    return float("inf")

                pct_skipped = float(
                    getattr(candidate, "last_fit_diagnostics", {}).get(
                        "pct_skipped", 0.0
                    )
                )
                trial.set_user_attr("pct_skipped", pct_skipped)
                if pct_skipped > 5.0:
                    logger.warning(
                        f"Trial {trial.number}: {pct_skipped:.1f}% batches skipped "
                        f"(> 5%) -- marked infeasible."
                    )
                    return float("inf")

                return val_nll

            # Best-config artifact
            config_path = f"{model_dir}/config.yaml"

            def _build_best(bp):
                return MixtureDensityNetwork(
                    input_dim=bp["lags"],
                    hidden_layers=[bp["mlp_width"]] * bp["mlp_depth"],
                    n_mixtures=bp["n_mixtures"],
                    dropout=bp["dropout"],
                    n_jobs=n_process,
                    device=device,
                ).to(device)

            def _fit_best(mdn, bp, Xtr, ytr, wtr, Xc, yc, wc):
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

            if args.evaluate:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        f"--evaluate set but no config at {config_path}; "
                        f"run tuning first."
                    )
                with open(config_path) as f:
                    best_params = yaml.safe_load(f)["params"]
                logger.info(
                    f"[evaluate] loaded best config from {config_path}: {best_params}"
                )
            else:
                n_trials = (
                    args.n_trials if args.n_trials is not None else op["N_TRIALS"]
                )

                os.makedirs(model_dir, exist_ok=True)
                storage = f"sqlite:///{model_dir}/study.db"
                study = optuna.create_study(
                    study_name="mdn",
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                    storage=storage,
                    load_if_exists=True,
                )

                n_done = len(study.trials)
                remaining = max(0, n_trials - n_done)
                logger.info(
                    f"Optuna study at {storage} | {n_done} trial(s) already done, "
                    f"running {remaining} more (target {n_trials}) (MDN, lags searched)"
                )
                if remaining > 0:
                    study.optimize(objective, n_trials=remaining)

                best = study.best_trial
                best_params = dict(best.params)
                logger.info(
                    f"Best validation NLL: {best.value:.5f} | params: {best_params}"
                )

                # Persist the best config.
                os.makedirs(model_dir, exist_ok=True)
                with open(config_path, "w") as f:
                    yaml.safe_dump(
                        {
                            "model": "mdn",
                            "params": best_params,
                            "fixed": {
                                "horizon": horizon,
                                "density": m["DENSITY"],
                                "batch_size": tr["BATCH_SIZE"],
                                "max_epochs": tr["MAX_EPOCHS"],
                                "max_norm": tr["MAX_GRAD_NORM"],
                                "patience": tr["EARLY_STOPPING_PATIENCE"],
                                "scheduler_patience": tr["SCHEDULER_PATIENCE"],
                                "scheduler_factor": tr["SCHEDULER_FACTOR"],
                            },
                            "val": {"nll": float(best.value)},
                        },
                        f,
                    )

            best_lags = best_params["lags"]

            # Final fit: train the selected config once on the shared
            # simulated series and evaluate on the shared test path, exactly
            # like the other forecast methods.
            Xtr, ytr, wtr, Xc, yc, wc = get_mdn_data(best_lags)

            set_seed(cfg["seed"])  # deterministic init + training RNG
            mdn = _build_best(best_params)
            _fit_best(mdn, best_params, Xtr, ytr, wtr, Xc, yc, wc)
            mdn.eval()

            # Test features + no-change baseline at the selected lags on the
            # shared test path.
            selected_lags = best_lags
            X_test, y_true, prediction_offset, mspe_naive = evaluation_inputs(
                selected_lags
            )
            n_predictions = len(X_test)

            # Recalibrate on the calibration set and predict on the test path.
            cde_cali = mdn.pred(Xc, grid_y)
            X_test_tensor = torch.tensor(X_test, device=device)
            predictive_density = recalibrate_density(
                Xc,
                yc,
                cde_cali,
                X_test_tensor,
                mdn.pred(X_test_tensor, grid_y).cpu().numpy(),
                grid_y,
                n_process,
                artifacts=artifacts,
            )

        check_cde(predictive_density, grid_y)
        logger.info(
            f"{model} | {model_type}{order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
        )

        if artifacts:
            os.makedirs(model_dir, exist_ok=True)
            pd.DataFrame(
                artifacts["raw"],
                columns=[f"y_{i}" for i in range(artifacts["raw"].shape[1])],
            ).to_parquet(raw_density_path, index=False)
            cal = pd.DataFrame(artifacts["X_calibration"])
            cal.columns = [f"x_{i}" for i in range(cal.shape[1])]
            cal.insert(0, "pit", artifacts["pit"])
            cal.to_parquet(calibration_path, index=False)
            pd.DataFrame(
                artifacts["X_target"],
                columns=[f"x_{i}" for i in range(artifacts["X_target"].shape[1])],
            ).to_parquet(target_feats_path, index=False)
            logger.info(
                f"Saved uncorrected densities and calibration PITs "
                f"(num_basis={DEFAULT_NUM_BASIS}); rerun with --evaluate to "
                f"refit the recalibrator without refitting the model"
            )

    # Thresholds for the weighted scores
    flat_quantiles = {
        prob: value
        for cat in ("Lower Quantiles", "Upper Quantiles")
        for prob, value in quantiles[cat].items()
    }
    weight_thresholds = {
        level: flat_quantiles[level]
        for level in (0.05, 0.10, 0.90, 0.95)
        if level in flat_quantiles
    }

    # Evaluate the predictive density on the shared test path.
    cde_metrics = evaluate_and_plot_densities(
        predictive_density=predictive_density,
        grid_y=grid_y,
        ts=data_to_predict.iloc[prediction_offset:],
        horizon=horizon,
        file_name=file_name,
        model=model,
        theoretical_quantiles=quantiles,
        weight_thresholds=weight_thresholds,
        save_densities=True,
    )

    os.makedirs(model_dir, exist_ok=True)
    pd.DataFrame({"y_true": np.asarray(y_true, dtype=float)}).to_parquet(
        f"{model_dir}/{model}_realized_targets.parquet", index=False
    )

    # Log all evaluation metrics
    logger.info("=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"Estimation mode: {args.estimation} (test path: generating params)")
    logger.info(f"Model: {cde_metrics['model']}")
    logger.info(f"CDE Loss: {cde_metrics['cde_loss']:.6e}")
    logger.info(f"CRPS: {cde_metrics['crps']:.6f}")
    logger.info(f"Log Probability Score: {cde_metrics['log_prob']:.4f}")
    for lvl in ("qs_05", "qs_10", "qs_90", "qs_95"):
        if cde_metrics[lvl] is not None:
            pct = int(float(lvl.split("_")[1]))
            logger.info(f"Quantile Score ({pct}%): {cde_metrics[lvl]:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
