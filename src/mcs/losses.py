"""Per-observation losses feeding the Model Confidence Set."""

import numpy as np

from src.evaluate.CRPS import crps_from_cdf
from src.evaluate.evaluate_applications import (
    QS_ALPHAS,
    level_metrics_from_log_density,
)
from src.evaluate.weighted_scores import csl_per_obs, side_for_level, twcrps_per_obs

QUANTILE_LEVELS = QS_ALPHAS


def density_at_true(density, grid_y, y_true):
    """Interpolate each row of ``density`` at its realised outcome."""
    density = np.asarray(density, dtype=float)
    return np.array(
        [np.interp(y_true[i], grid_y, density[i]) for i in range(len(y_true))]
    )


def cde_loss_per_obs(density, grid_y, y_true):
    """CDE loss ``int p^2 - 2 p(y)`` per observation (cdetools convention)."""
    density = np.asarray(density, dtype=float)
    integrals = np.trapz(density * density, grid_y, axis=1)
    return integrals - 2.0 * density_at_true(density, grid_y, y_true)


def normalised_cdf(density):
    """Row-normalised cumulative sum, the CDF convention used for the CRPS."""
    cdf = np.cumsum(np.asarray(density, dtype=float), axis=1)
    norm = cdf[:, -1][:, None]
    norm = np.where(norm == 0, 1e-10, norm)
    return np.nan_to_num(cdf / norm, nan=0.0, posinf=1.0, neginf=0.0)


def crps_per_obs(density, grid_y, y_true):
    """CRPS per observation, from the row-normalised cumulative sum."""
    return np.asarray(
        crps_from_cdf(normalised_cdf(density), grid_y, y_true), dtype=float
    )


def neg_log_score_per_obs(density, grid_y, y_true):
    """Negative log predictive score, so that lower is better."""
    return -np.log(density_at_true(density, grid_y, y_true) + 1e-10)


def predictive_quantiles(density, grid_y, level):
    """Row-wise ``level`` quantile of the densities; NaN where the row is unusable."""
    density = np.asarray(density, dtype=float)
    dy = grid_y[1] - grid_y[0]
    out = np.full(density.shape[0], np.nan)
    for i in range(density.shape[0]):
        row = density[i]
        if np.all(row <= 0) or not np.isfinite(row).all():
            continue
        cdf = np.cumsum(row) * dy
        if not np.isfinite(cdf).all() or cdf[-1] <= 0:
            continue
        out[i] = np.interp(level, cdf, grid_y)
    return out


def quantile_score_per_obs(density, grid_y, y_true, level):
    """Pinball loss ``(y - q)(level - 1{y <= q})`` at ``level``, per observation."""
    q = predictive_quantiles(density, grid_y, level)
    errors = np.asarray(y_true, dtype=float) - q
    indicator = (np.asarray(y_true, dtype=float) <= q).astype(float)
    return errors * (level - indicator)


def estimated_moments(density, grid_y, n_moments):
    """Conditional moments 1..``n_moments`` implied by each predictive density.

    Same quadrature as ``evaluate_predictive_density``. The BLAS ``matmul``
    raises spurious floating-point flags on some platforms (macOS/Accelerate
    reports them even for well-scaled inputs), so they are silenced here; the
    arithmetic and the result are untouched.
    """
    dy = float(np.diff(grid_y).mean())
    powers_y = np.stack([grid_y**k for k in np.arange(1, n_moments + 1)], axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.asarray(density, dtype=float) @ powers_y * dy


def moment_squared_errors(density, grid_y, theoretical_moments):
    """Squared moment errors, shape ``(n_x, n_moments)``; RMSE is sqrt of the mean."""
    n_moments = theoretical_moments.shape[1]
    return (estimated_moments(density, grid_y, n_moments) - theoretical_moments) ** 2


def kl_per_obs(true_density, estimated_density, grid_y, eps=1e-12):
    """Kullback-Leibler divergence from the true density, per conditioning point."""
    true_density = np.clip(np.asarray(true_density, dtype=float), eps, None)
    estimated_density = np.clip(np.asarray(estimated_density, dtype=float), eps, None)
    return np.trapz(
        true_density * np.log(true_density / estimated_density), grid_y, axis=1
    )


def ise_per_obs(true_density, estimated_density, grid_y, eps=1e-12):
    """Integrated squared error against the true density, per conditioning point."""
    true_density = np.clip(np.asarray(true_density, dtype=float), eps, None)
    estimated_density = np.clip(np.asarray(estimated_density, dtype=float), eps, None)
    return np.trapz((true_density - estimated_density) ** 2, grid_y, axis=1)


def density_losses(
    density,
    grid_y,
    y_true,
    density_log=None,
    log_grid=None,
    weight_thresholds=None,
    quantile_levels=QUANTILE_LEVELS,
):
    """All realised-outcome density losses, keyed as in the metrics JSON."""
    exact_cde, exact_q = None, None
    if density_log is not None and log_grid is not None:
        exact_cde, exact_q = level_metrics_from_log_density(
            density_log, log_grid, y_true, alphas=quantile_levels
        )

    losses = {
        "cde_loss": (
            exact_cde
            if exact_cde is not None
            else cde_loss_per_obs(density, grid_y, y_true)
        ),
        "crps": crps_per_obs(density, grid_y, y_true),
        "log_prob": neg_log_score_per_obs(density, grid_y, y_true),
    }
    for level in quantile_levels:
        if exact_q is not None:
            q = exact_q[level]
            y = np.asarray(y_true, dtype=float)
            loss = (y - q) * (level - (y <= q))
        else:
            loss = quantile_score_per_obs(density, grid_y, y_true, level)
        losses[f"qs_{round(level * 100):02d}"] = loss

    if weight_thresholds:
        cdf = normalised_cdf(density)
        for level, threshold in sorted(weight_thresholds.items()):
            if threshold is None or not np.all(np.isfinite(threshold)):
                continue
            side = side_for_level(level)
            key = f"{round(level * 100):02d}"
            losses[f"twcrps_{key}"] = twcrps_per_obs(
                cdf, grid_y, y_true, threshold, side
            )
            losses[f"csl_{key}"] = csl_per_obs(
                density, grid_y, y_true, threshold, side
            )[0]
    return losses


def moment_region_indices(grid_x, quantiles):
    """Conditioning-grid index sets for the three simulation regions.

    Mirrors :func:`src.evaluate.evaluate_predictive_density.evaluate_predictive_density`
    so the MCS is run on exactly the points the reported RMSEs average over.
    """
    n_x = len(grid_x)
    flat_q = {
        str(prob): value
        for category in ("Lower Quantiles", "Upper Quantiles")
        for prob, value in quantiles[category].items()
    }

    def cut(prob, default_frac):
        return np.searchsorted(
            grid_x, flat_q.get(str(prob), grid_x[int(default_frac * n_x)])
        )

    i_001 = cut(0.01, 0.01)
    i_010 = cut(0.1, 0.1)
    i_090 = cut(0.9, 0.9)
    i_099 = cut(0.99, 0.99)
    return {
        "Total": np.arange(i_001, i_099 + 1),
        "Between 0.1-0.9": np.arange(i_010, i_090 + 1),
        "Between 0.01-0.1 and 0.9-0.99": np.concatenate(
            [np.arange(i_001, i_010), np.arange(i_090, i_099 + 1)]
        ),
    }
