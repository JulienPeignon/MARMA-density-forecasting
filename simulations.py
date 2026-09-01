"""Monte Carlo study of density forecasts on simulated MAR/MARMA paths."""

import argparse
import multiprocessing as mp
import os
import time

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
from src.conditional_theoretical_moments.theoretical_moments import (
    compute_theoretical_moments,
)
from src.conditional_theoretical_moments.theoretical_quantiles import (
    compute_theoretical_quantiles,
)
from src.evaluate.cauchy_closed_form import cauchy_ar1_predictive_density
from src.evaluate.evaluate_predictive_density import (
    run_postprocessing,
)
from src.forecast_methods.gj2026 import gj2026_predictive_density_given_yt
from src.forecast_methods.kcde import kcde_predictive_density
from src.forecast_methods.lls2012 import lls2012_predictive_density
from src.forecast_methods.MDN import (
    MixtureDensityNetwork,
)
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
from src.stable_mar.stable_mar import stablemar as sm
from src.utils.save_predictive_densities import (
    load_recalibration_cache,
    save_recalibration_cache,
)
from src.utils.setup_config_device import (
    get_allowed_cpu_count,
    set_seed,
    setup_config_device,
    setup_device,
)
from src.utils.setup_logger import setup_logger


def main():
    """Run one Monte Carlo density-forecasting job from the command line."""
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
        "--estimation",
        choices=["gcov", "mle", "true"],
        default="mle",
        help=(
            "MAR parameter source: 'gcov', 'mle' (default, alpha-stable MLE), "
            "or 'true' (the simulation's generating parameters)."
        ),
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Generate a moving-density GIF during postprocessing.",
    )
    args = parser.parse_args()

    if args.order is None:
        parser.error("the following arguments are required: --order")

    # Simulation
    alpha_stable_cfg = cfg["alpha_stable_params"]
    mar_cfg = cfg["mar_params"]
    nw_cfg = cfg["kcde_params"]
    lls2012_cfg = cfg["lls2012_params"]
    mdn_cfg = cfg["mdn_params"]
    m = mdn_cfg["model_params"]
    boost_cfg = cfg["flexzboost_params"]

    horizon = args.horizon
    order_mar = tuple(int(x) for x in args.order)
    mar_params = {
        tuple(map(int, k.strip("()").split(","))): v for k, v in mar_cfg.items()
    }

    params_alpha_mar = {**mar_params[order_mar], **alpha_stable_cfg}

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
        f"outputs/simulations/{model_type}{order_mar}/horizon_{horizon}/alpha_{ALPHA}"
    )
    logger.info(f"Predictive densities & graphics will be saved in: {file_name}")

    _model_out = {
        "kcde": "kcde",
        "lls2012": "lls2012",
        "gj2026": "gj2026",
        "flexzboost": "FlexZBoost",
        "mdn": "mdn",
    }[args.model]
    density_path = f"{file_name}/{_model_out}/{_model_out}.parquet"

    effective_load = args.evaluate and os.path.exists(density_path)
    artifacts = {}  # filled by recalibrate_density: uncorrected densities + PITs
    if args.evaluate and not effective_load:
        logger.info(
            f"[evaluate] no cached densities at {density_path}; fitting "
            f"(MDN skips Optuna and uses the saved config) and saving for future runs"
        )

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
            r = order_mar[0]
            est_mar = sm(order_mar)
            est_mar.model = "MAR"
            est_mar.par = (
                (list(phi_hat) if r > 0 else [])
                + list(psi_hat)
                + [
                    alpha_hat,
                    beta_hat,
                    sigma_hat,
                ]
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

    quantiles = compute_theoretical_quantiles(
        phi_vec=[PHI],
        psi_vec=[PSI] if isinstance(PSI, float) else PSI,
        alpha=ALPHA,
        beta=BETA,
        sigma=SIGMA,
        theta=THETA,
        eta=ETA,
    )

    grid_X, grid_y = make_grid(quantiles, cfg["n_points_grid"])

    logger.info("Computing theoretical moments...")
    theoretical_moments = compute_theoretical_moments(
        grid=grid_X,
        h=horizon,
        phi_vec=[PHI],
        psi_vec=[PSI] if isinstance(PSI, float) else PSI,
        alpha=ALPHA,
        beta=BETA,
        sigma=SIGMA,
        theta=THETA,
        eta=ETA,
        ma_trunc=100,
    )

    # Compute true predictive density if available (MAR(0,1) with alpha=1.0)
    if order_mar == (0, 1) and ALPHA == 1.0:
        logger.info("Computing true predictive density (Cauchy AR(1))...")
        true_predictive_density = cauchy_ar1_predictive_density(
            grid_X, grid_y, PSI, SIGMA, horizon
        )
    else:
        true_predictive_density = None

    def forecast_parameters(data):
        """Return fitted or generating MAR parameters for forecast methods."""
        r, s = order_mar[:2]
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
            stable_params = mar_model.fit_stable_noise(resid)
            mar_model.par.extend(stable_params)

        phi_hat = [0.0] if r == 0 else mar_model.par[0:r]
        psi_hat = mar_model.par[r : r + s]
        alpha_hat = mar_model.par[r + s]
        beta_hat = mar_model.par[r + s + 1]
        sigma_hat = mar_model.par[r + s + 2]
        return phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat

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
        else:
            alpha_h, beta_h, sigma_h = sm(order=(1, 1)).fit_stable_noise(res["E"])
        return (psi_h, phi_h, theta_h, eta_h, alpha_h, beta_h, sigma_h)

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

    if args.model in {"kcde", "flexzboost", "mdn"}:
        logger.info("Simulating training data...")
        y_sim = generate_series(cfg["length_simulation"], use_estimated=True)
    elif args.model == "gj2026":
        logger.info("Simulating data for the GJ2026 kernel...")
        y_sim = generate_series(cfg["length_simulation"], use_estimated=True)

    if args.model == "kcde":
        model = "kcde"

        if effective_load:
            logger.info(
                f"Loading predictive density from {file_name}/{model}/{model}.parquet"
            )
            predictive_density, start, end = None, None, None
        else:
            logger.info(
                f"Computing {model} predictive densities | "
                f"Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
            )

            start = time.time()
            predictive_density = kcde_predictive_density(
                X_train=np.asarray(y_sim)[:-horizon].reshape(-1, 1),
                y_train=np.asarray(y_sim)[horizon:],
                X_query=np.asarray(grid_X).reshape(-1, 1),
                grid_y=grid_y,
                kernel=nw_cfg["KERNEL"],
                bandwidth=nw_cfg["BANDWIDTH"],
            )
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, lags=1)
            density_cali = kcde_predictive_density(
                X_train=np.asarray(y_sim)[:-horizon].reshape(-1, 1),
                y_train=np.asarray(y_sim)[horizon:],
                X_query=X_cali,
                grid_y=grid_y,
                kernel=nw_cfg["KERNEL"],
                bandwidth=nw_cfg["BANDWIDTH"],
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                np.asarray(grid_X).reshape(-1, 1),
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )
            end = time.time()
            check_cde(predictive_density, grid_y)

            logger.info(
                f"{model} | Order: {order_mar} | Alpha: {ALPHA} | "
                f"Horizon: {horizon} | "
                f"Running time: {(end - start) / 60:.1f} minutes"
            )

    if args.model == "lls2012":
        model = "lls2012"

        if effective_load:
            logger.info(
                f"Loading predictive density from {file_name}/{model}/{model}.parquet"
            )
            predictive_density, start, end = None, None, None
        else:
            logger.info(
                f"Computing {model} predictive densities | "
                f"Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
            )

            start = time.time()
            phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = (
                estimated_params
                if estimated_params is not None
                else forecast_parameters(None)
            )

            predictive_density = lls2012_predictive_density(
                psi_hat=psi_hat,
                alpha_hat=alpha_hat,
                beta_hat=beta_hat,
                sigma_hat=sigma_hat,
                horizon=horizon,
                M=lls2012_cfg["M"],
                N_draws=lls2012_cfg["N_DRAWS"],
                grid_x=grid_X,
                grid_y=grid_y,
                n_process=n_process,
                phi_hat=phi_hat,
                seed=cfg["seed"],
            )
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, lags=1)
            density_cali = lls2012_predictive_density(
                psi_hat=psi_hat,
                alpha_hat=alpha_hat,
                beta_hat=beta_hat,
                sigma_hat=sigma_hat,
                horizon=horizon,
                M=lls2012_cfg["M"],
                N_draws=lls2012_cfg["N_DRAWS"],
                grid_x=X_cali.ravel(),
                grid_y=grid_y,
                n_process=n_process,
                phi_hat=phi_hat,
                seed=cfg["seed"],
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                np.asarray(grid_X).reshape(-1, 1),
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )
            end = time.time()
            check_cde(predictive_density, grid_y)

            logger.info(
                f"{model} | Order: {order_mar} | Alpha: {ALPHA} | "
                f"Horizon: {horizon} | "
                f"Running time: {(end - start) / 60:.1f} minutes"
            )

    if args.model == "gj2026":
        model = "gj2026"
        if effective_load:
            logger.info(
                f"Loading predictive density from {file_name}/{model}/{model}.parquet"
            )
            predictive_density, start, end = None, None, None
        else:
            logger.info(
                f"Computing {model} predictive densities | "
                f"Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
            )

            start = time.time()

            phi_hat, psi_hat, alpha_hat, beta_hat, sigma_hat = (
                estimated_params
                if estimated_params is not None
                else forecast_parameters(None)
            )

            predictive_density = np.array(
                Parallel(n_jobs=n_process)(
                    delayed(gj2026_predictive_density_given_yt)(
                        psi_hat=psi_hat,
                        phi_hat=phi_hat,
                        alpha_hat=alpha_hat,
                        beta_hat=beta_hat,
                        sigma_hat=sigma_hat,
                        y_train=y_sim,
                        y_history=[x],
                        grid_y=grid_y,
                        bandwidth=nw_cfg["BANDWIDTH"],
                        horizon=horizon,
                    )
                    for x in tqdm(grid_X, desc="gj2026")
                )
            )
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, lags=1)
            density_cali = np.array(
                Parallel(n_jobs=n_process)(
                    delayed(gj2026_predictive_density_given_yt)(
                        psi_hat=psi_hat,
                        phi_hat=phi_hat,
                        alpha_hat=alpha_hat,
                        beta_hat=beta_hat,
                        sigma_hat=sigma_hat,
                        y_train=y_sim,
                        y_history=[x],
                        grid_y=grid_y,
                        bandwidth=nw_cfg["BANDWIDTH"],
                        horizon=horizon,
                    )
                    for x in tqdm(X_cali.ravel(), desc="gj2026 calibration")
                )
            )
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                np.asarray(grid_X).reshape(-1, 1),
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )
            end = time.time()
            check_cde(predictive_density, grid_y)

            logger.info(
                f"{model} | Order: {order_mar} | Alpha: {ALPHA} | "
                f"Horizon: {horizon} | "
                f"Running time: {(end - start) / 60:.1f} minutes"
            )

    if args.model == "flexzboost":
        model = "FlexZBoost"

        if effective_load:
            logger.info(
                f"Loading predictive density from {file_name}/{model}/{model}.parquet"
            )
            predictive_density, start, end = None, None, None
        else:
            dl = mdn_cfg["dataloaders_params"]

            logger.info(
                f"Computing {model} predictive densities | "
                f"Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
            )

            start = time.time()
            (
                X_train,
                y_train,
                X_val,
                y_val,
                _,
                _,
                _,
                _,
            ) = prepare_tensors(
                pd.DataFrame(y_sim),
                X=None,
                y=None,
                lags=dl["CONDITIONAL_LAGS"],
                horizon=horizon,
                proportions=(1.0, 0.0, 0.0),
                device=device,
            )

            flexzboost = flexcode.FlexCodeModel(
                XGBoost,
                max_basis=boost_cfg["MAX_BASIS"],
                basis_system=boost_cfg["BASIS_SYSTEM"],
                z_min=grid_y[0],
                z_max=grid_y[-1],
                regression_params={
                    "verbosity": 0,
                    "n_jobs": n_process,
                    "random_state": cfg["seed"],
                },
            )

            flexzboost.fit(
                x_train=X_train.detach().cpu().numpy(),
                z_train=y_train.detach().cpu().numpy(),
            )

            predictive_density, _ = flexzboost.predict(
                np.asarray(grid_X).reshape(-1, 1), n_grid=cfg["n_points_grid"]
            )
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)
            X_cali, y_cali = lagged_pairs(calibration, horizon, lags=1)
            density_cali, _ = flexzboost.predict(X_cali, n_grid=cfg["n_points_grid"])
            predictive_density = recalibrate_density(
                X_cali,
                y_cali,
                density_cali,
                np.asarray(grid_X).reshape(-1, 1),
                predictive_density,
                grid_y,
                n_process,
                artifacts=artifacts,
            )
            end = time.time()
            check_cde(predictive_density, grid_y)

            logger.info(
                f"{model} | Order: {order_mar} | Alpha: {ALPHA} | "
                f"Horizon: {horizon} | "
                f"Running time: {(end - start) / 60:.1f} minutes"
            )

    if args.model == "mdn":
        model = "mdn"

        model_dir = f"{file_name}/{model}"

        if effective_load:
            logger.info(
                f"Loading predictive density from {file_name}/{model}/{model}.parquet"
            )
            predictive_density_calibrated, start, end = None, None, None
        else:
            dl = mdn_cfg["dataloaders_params"]
            tr = mdn_cfg["training_params"]
            op = cfg["optuna_params"]

            logger.info(
                f"Computing {model} predictive densities | "
                f"Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon}"
            )

            # Training data: full simulated series (train)
            (
                X_train,
                y_train,
                _,
                _,
                _,
                _,
                weights_train,
                _,
            ) = prepare_tensors(
                pd.DataFrame(y_sim),
                X=None,
                y=None,
                lags=dl["CONDITIONAL_LAGS"],
                horizon=horizon,
                proportions=(1.0, 0.0, 0.0),
                device=device,
            )

            # Validation + calibration set: separate simulated series
            set_seed(cfg["calibration_seed"])  # shared calibration set (see config)
            calibration = generate_series(cfg["length_calibration"], use_estimated=True)

            (
                X_cali,
                y_cali,
                _,
                _,
                _,
                _,
                weights_cali,
                _,
            ) = prepare_tensors(
                pd.DataFrame(calibration),
                X=None,
                y=None,
                lags=dl["CONDITIONAL_LAGS"],
                horizon=horizon,
                proportions=(1.0, 0.0, 0.0),
                device=device,
            )

            logger.info(
                f"Train samples: {len(X_train)} | Cali/val samples: {len(X_cali)}"
            )

            # Optuna hyperparameter search (minimize validation NLL)
            # lags are fixed to CONDITIONAL_LAGS (= 1) for simulations.py; the
            # calibration set doubles as the validation set for model selection.
            def objective(trial):
                set_seed(cfg["seed"])
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

                candidate = MixtureDensityNetwork(
                    input_dim=dl["CONDITIONAL_LAGS"],
                    hidden_layers=[mlp_width] * mlp_depth,
                    n_mixtures=n_mixtures,
                    dropout=dropout,
                    n_jobs=1,
                    device=device,
                ).to(device)

                try:
                    val_nll = candidate.fit(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_cali,
                        y_val=y_cali,
                        weights_train=weights_train,
                        weights_val=weights_cali,
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

            def _build_best(best_params, n_jobs):
                return MixtureDensityNetwork(
                    input_dim=dl["CONDITIONAL_LAGS"],
                    hidden_layers=[best_params["mlp_width"]] * best_params["mlp_depth"],
                    n_mixtures=best_params["n_mixtures"],
                    dropout=best_params["dropout"],
                    n_jobs=n_jobs,
                    device=device,
                ).to(device)

            def _fit_best(mdn, best_params):
                mdn.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_cali,
                    y_val=y_cali,
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
                    f"running {remaining} more (target {n_trials}) "
                    f"(MDN, lags={dl['CONDITIONAL_LAGS']} fixed)"
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

            # Final fit: train the selected config once on the shared
            # simulated series and recalibrate on the shared calibration set.
            grid_X_torch = torch.as_tensor(
                grid_X, dtype=torch.float32, device=device
            ).unsqueeze(1)

            start = time.time()
            set_seed(cfg["seed"])  # deterministic init + training RNG
            mdn = _build_best(best_params, n_jobs=n_process)
            _fit_best(mdn, best_params)
            mdn.eval()

            # Recalibrate on the calibration set.
            cde_cali = mdn.pred(X_cali, grid_y)
            predictive_density_calibrated = recalibrate_density(
                X_cali,
                y_cali,
                cde_cali,
                grid_X_torch,
                mdn.pred(grid_X_torch, grid_y).cpu().numpy(),
                grid_y,
                n_process,
                artifacts=artifacts,
            )
            check_cde(predictive_density_calibrated, grid_y)
            end = time.time()

            logger.info(
                f"{model} | Order: {order_mar} | Alpha: {ALPHA} | Horizon: {horizon} "
                f"| Running time: {(end - start) / 60:.1f} minutes"
            )

    if args.model:
        if args.model == "mdn":
            model = "mdn"
            density = predictive_density_calibrated if not effective_load else None
        elif args.model == "flexzboost":
            model = "FlexZBoost"
            density = predictive_density if not effective_load else None
        elif args.model == "gj2026":
            model = "gj2026"
            density = predictive_density if not effective_load else None
        elif args.model == "lls2012":
            model = "lls2012"
            density = predictive_density if not effective_load else None
        elif args.model == "kcde":
            model = "kcde"
            density = predictive_density if not effective_load else None

        # Recalibration cache
        recalibrated_from_cache = False
        if effective_load:
            cached = load_recalibration_cache(file_name, model)
            if cached is not None:
                logger.info(
                    f"[evaluate] refitting the recalibrator on cached PITs at "
                    f"num_basis={DEFAULT_NUM_BASIS}"
                )
                raw, x_target, x_cal, pit = cached
                density = fit_and_apply_recalibration(
                    x_cal,
                    pit,
                    x_target,
                    raw,
                    grid_y,
                    n_process,
                    num_basis=DEFAULT_NUM_BASIS,
                )
                check_cde(density, grid_y)
                recalibrated_from_cache = True
        elif save_recalibration_cache(artifacts, file_name, model):
            logger.info(
                f"Saved uncorrected densities and calibration PITs "
                f"(num_basis={DEFAULT_NUM_BASIS}); rerun with --evaluate to "
                f"refit the recalibrator without refitting the model"
            )

        run_postprocessing(
            grid_X,
            grid_y,
            theoretical_moments,
            quantiles,
            file_name,
            model,
            density=m["DENSITY"],
            predictive_density=density,
            true_predictive_density=true_predictive_density,
            load=effective_load and not recalibrated_from_cache,
            gif=args.gif,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
