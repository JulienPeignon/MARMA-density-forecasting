"""Scoring of application predictive densities: CDE loss, CRPS and MSPE."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.cdetools import cde_loss
from src.evaluate.CRPS import crps_from_cdf
from src.evaluate.weighted_scores import (
    csl_per_obs,
    side_for_level,
    twcrps_per_obs,
)

QS_ALPHAS = (0.05, 0.10, 0.90, 0.95)


def level_metrics_from_log_density(
    density_log: np.ndarray,
    log_grid: np.ndarray,
    y_true: np.ndarray,
    alphas=(0.05, 0.10, 0.90, 0.95),
    price_floor: float = 0.05,
):
    """Level-space CDE loss and quantiles obtained exactly from the log density.

    Returns per-observation CDE losses and a dict of exact level quantiles.
    """
    density_log = np.atleast_2d(np.asarray(density_log, dtype=float))
    log_grid = np.asarray(log_grid, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    dx = np.diff(log_grid)
    mass = np.concatenate(
        [
            np.zeros((density_log.shape[0], 1)),
            np.cumsum(dx * (density_log[:, 1:] + density_log[:, :-1]) / 2.0, axis=1),
        ],
        axis=1,
    )
    total = mass[:, -1][:, None]
    density_log = density_log / np.where(total <= 0, 1.0, total)
    cdf = mass / np.where(total <= 0, 1.0, total)

    ln_y = np.log(y_true)
    f_at_true = np.array(
        [np.interp(ln_y[i], log_grid, density_log[i]) for i in range(len(y_true))]
    )
    keep = log_grid >= np.log(price_floor)
    integral = np.trapz(
        (density_log**2 * np.exp(-log_grid))[:, keep], log_grid[keep], axis=1
    )

    cde_per_obs = integral - 2.0 * f_at_true / y_true
    quantiles = {
        a: np.exp([np.interp(a, cdf[i], log_grid) for i in range(len(y_true))])
        for a in alphas
    }
    return cde_per_obs, quantiles


def calculate_quantile_score(y_true, quantile_pred, alpha):
    """Pinball loss at level ``alpha``.

    QS = mean (y - Q) (alpha - 1{y <= Q}); None when every value is NaN.
    """
    valid_mask = ~(np.isnan(y_true) | np.isnan(quantile_pred))

    if not np.any(valid_mask):
        return None

    y_true_valid = y_true[valid_mask]
    quantile_pred_valid = quantile_pred[valid_mask]

    errors = y_true_valid - quantile_pred_valid
    indicator = (y_true_valid <= quantile_pred_valid).astype(float)
    scores = errors * (alpha - indicator)

    return np.mean(scores)


def tail_center_masks(
    y_eval: np.ndarray, tail_q: float = 0.90, bounds: tuple | None = None
):
    """Split the test set by realised target value into center vs tails."""
    y_eval = np.asarray(y_eval)
    if bounds is not None:
        lo, hi = bounds
        tails_idx = (y_eval < float(lo)) | (y_eval > float(hi))
    else:
        tails_idx = y_eval >= float(np.quantile(y_eval, tail_q))
    return ~tails_idx, tails_idx


def mspe_by_region(
    squared_errors, se_naive, targets, tail_q: float = 0.90, tails_idx=None
):
    """Per-region MSPE relative to the no-change forecast.

    Regions are those of :func:`tail_center_masks` plus ``total``, each
    benchmarked on its own observations, so they line up with the regional CRPS,
    CDE loss and log score. Each name in ``squared_errors`` yields ``mspe_<k>``
    and ``mspe_ratio_<k>``; the no-change benchmark forms the ratios and is not
    reported.
    """
    targets = np.asarray(targets)
    se_naive = np.asarray(se_naive)
    squared_errors = {k: np.asarray(v) for k, v in squared_errors.items()}

    if tails_idx is None:
        _, tails_idx = tail_center_masks(targets, tail_q)
    tails_idx = np.asarray(tails_idx, dtype=bool)
    center_idx = ~tails_idx
    regions = {
        "center": center_idx,
        "tails": tails_idx,
        "total": np.ones(len(targets), dtype=bool),
    }
    keys = []
    for name in squared_errors:
        keys += [f"mspe_{name}", f"mspe_ratio_{name}"]

    out = {}
    for region, idx in regions.items():
        if not np.any(idx):
            out[region] = dict.fromkeys(keys)
            continue
        naive = float(np.mean(se_naive[idx]))
        scores = {}
        for name, se in squared_errors.items():
            mspe = float(np.mean(se[idx]))
            scores[f"mspe_{name}"] = mspe
            scores[f"mspe_ratio_{name}"] = mspe / naive if naive > 0 else None
        out[region] = scores
    return out


def evaluate_and_plot_densities(
    predictive_density: np.ndarray,
    grid_y: np.ndarray,
    ts: np.ndarray,
    horizon: int,
    file_name: str,
    model: str,
    theoretical_quantiles: dict = None,
    mae_naive: float = None,
    mad_naive: float = None,
    save_densities: bool = True,
    output_dir: str = None,
    density_log: np.ndarray = None,
    log_grid: np.ndarray = None,
    price_floor: float = 0.05,
    weight_thresholds: dict = None,
    qs_alphas: tuple = QS_ALPHAS,
):
    """Score predictive densities and write the CDE metrics JSON.

    ``predictive_density`` is (n_obs, n_grid) over ``grid_y``. ``weight_thresholds``
    maps a level to the threshold of the proper weighted scores; levels below 0.5
    weight the lower tail, the others the upper tail, and the thresholds must be
    fixed independently of the outcomes. ``theoretical_quantiles`` defines the
    center/tails split, ``qs_alphas`` the quantile-score levels.
    """
    base_dir = Path(file_name)
    base_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = base_dir / model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predictions are already aligned 1-to-1 with ts
    y_eval = np.asarray(ts[: len(predictive_density)])

    exact_cde, exact_q = None, None
    if density_log is not None and log_grid is not None:
        exact_cde, exact_q = level_metrics_from_log_density(
            density_log, log_grid, y_eval, alphas=qs_alphas, price_floor=price_floor
        )

    if exact_cde is not None:
        loss_value = float(np.mean(exact_cde))
    else:
        loss_value, _ = cde_loss(predictive_density, grid_y, y_eval)

    cdf = np.cumsum(predictive_density, axis=1)
    cdf_norm = cdf[:, -1][:, None]
    cdf_norm = np.where(cdf_norm == 0, 1e-10, cdf_norm)  # Avoid division by zero
    cdf /= cdf_norm
    cdf = np.nan_to_num(cdf, nan=0.0, posinf=1.0, neginf=0.0)
    crps = crps_from_cdf(cdf, grid_y, y_eval)
    crps_mean = float(np.nanmean(crps))

    density_at_true = np.array(
        [
            np.interp(y_eval[i], grid_y, predictive_density[i])
            for i in range(len(y_eval))
        ]
    )
    log_prob_score = float(np.mean(np.log(density_at_true + 1e-10)))

    weighted_scores = {}
    if weight_thresholds:
        for level, threshold in sorted(weight_thresholds.items()):
            if threshold is None or not np.all(np.isfinite(threshold)):
                continue
            side = side_for_level(level)
            key = f"{round(level * 100):02d}"
            tw = twcrps_per_obs(cdf, grid_y, y_eval, threshold, side)
            csl, _ = csl_per_obs(predictive_density, grid_y, y_eval, threshold, side)
            weighted_scores[f"twcrps_{key}"] = float(np.nanmean(tw))
            weighted_scores[f"csl_{key}"] = float(np.nanmean(csl))

    n_obs = predictive_density.shape[0]
    if exact_q is not None:
        quantile_preds = exact_q
    else:
        quantile_preds = {}
        dy = grid_y[1] - grid_y[0]
        for alpha in qs_alphas:
            quantile_values = np.zeros(n_obs)
            for i in range(n_obs):
                try:
                    density = predictive_density[i, :]
                    if np.all(density <= 0) or not np.isfinite(density).all():
                        quantile_values[i] = np.nan
                        continue

                    cdf = np.cumsum(density) * dy

                    if not np.isfinite(cdf).all() or cdf[-1] <= 0:
                        quantile_values[i] = np.nan
                        continue

                    quantile_values[i] = np.interp(alpha, cdf, grid_y)
                except Exception as e:
                    print(f"Warning: quantile {alpha} failed for observation {i}: {e}")
                    quantile_values[i] = np.nan

            quantile_preds[alpha] = quantile_values

    quantile_scores = {
        alpha: calculate_quantile_score(y_eval, quantile_preds[alpha], alpha)
        for alpha in qs_alphas
    }

    cde_metrics = {
        "model": model,
        "cde_loss": float(loss_value),
        "crps": crps_mean,
        "log_prob": log_prob_score,
        **{
            f"qs_{round(alpha * 100):02d}": (
                float(score) if score is not None else None
            )
            for alpha, score in quantile_scores.items()
        },
        "n_observations": int(len(y_eval)),
        **weighted_scores,
    }

    metrics_path = output_dir / f"{model}_cde_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(cde_metrics, f, indent=2)

    if save_densities:
        df_densities = pd.DataFrame(
            predictive_density,
            columns=[f"y_{i}" for i in range(predictive_density.shape[1])],
        )
        densities_path = output_dir / f"{model}_densities.parquet"
        df_densities.to_parquet(densities_path, index=False)

    return cde_metrics
