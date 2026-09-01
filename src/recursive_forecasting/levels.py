"""Grids and log->level density transform for the recursive application."""

from __future__ import annotations

import warnings

import numpy as np


def build_log_grid(
    log_prices: np.ndarray,
    n_points: int,
    margin: float = 1.0,
    margin_low: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Uniform log-price grid spanning the observed range with additive margin.

    ``log_prices`` should pool everything the densities must cover (real-time and
    final-vintage log prices, train and test). Data leakage is accepted here by
    design -- the grid is fixed once and shared across all blocks so the density
    matrix is coherent.

    ``lo`` and ``hi`` pin the bounds explicitly; when ``None`` they are the
    observed extremes plus the corresponding margin. Pinning them wide enough
    matters with alpha-stable predictive laws, whose polynomial tails otherwise
    leave a percent of mass outside the grid at long horizons.
    """
    lp = np.asarray(log_prices, dtype=float)
    lp = lp[np.isfinite(lp)]
    m_low = margin if margin_low is None else margin_low
    lo = lp.min() - m_low if lo is None else float(lo)
    hi = lp.max() + margin if hi is None else float(hi)
    return np.linspace(lo, hi, n_points)


def log_density_to_level(
    density_log: np.ndarray,
    log_grid: np.ndarray,
    level_grid: np.ndarray,
    warn_above: float | None = 0.01,
) -> np.ndarray:
    """Map predictive densities from the log grid to a uniform level grid.

    ``f_P(p) = f_Y(ln p) / p`` evaluated by interpolating each row of
    ``density_log`` at ``ln(level_grid)``. Rows are renormalised to unit mass on
    ``level_grid`` (trapezoidal) so the level densities integrate to 1. The level
    grid must stay strictly positive (``ln`` undefined otherwise).
    """
    density_log = np.atleast_2d(np.asarray(density_log, dtype=float))
    ln_p = np.log(level_grid)
    out = np.empty((density_log.shape[0], level_grid.size), dtype=float)
    lost = np.empty(density_log.shape[0], dtype=float)
    dx = np.diff(log_grid)
    for i in range(density_log.shape[0]):
        d = density_log[i]
        cdf = np.concatenate([[0.0], np.cumsum(dx * (d[1:] + d[:-1]) / 2.0)])
        total = cdf[-1]
        if total > 0:
            inside = np.interp(ln_p[-1], log_grid, cdf) - np.interp(
                ln_p[0], log_grid, cdf
            )
            lost[i] = 1.0 - inside / total
        else:
            lost[i] = 0.0
        fy = np.interp(ln_p, log_grid, d, left=0.0, right=0.0)
        fp = fy / level_grid
        mass = np.trapz(fp, level_grid)
        out[i] = fp / mass if mass > 0 else fp
    if warn_above is not None and lost.max() > warn_above:
        n_bad = int((lost > warn_above).sum())
        warnings.warn(
            f"level grid [{level_grid[0]:.3g}, {level_grid[-1]:.3g}] drops "
            f"more than {warn_above:.1%} of the predictive mass on {n_bad} of "
            f"{len(lost)} rows (mean {lost.mean():.2%}, worst {lost.max():.2%}); "
            f"renormalisation redistributes it inward.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def build_level_model_grid(n_points: int, hi: float) -> np.ndarray:
    """Symmetric price grid for models fitted directly in levels."""
    return np.linspace(-float(hi), float(hi), n_points)


def restrict_to_level_grid(
    density_model: np.ndarray,
    model_grid: np.ndarray,
    level_grid: np.ndarray,
    warn_above: float | None = 0.01,
) -> np.ndarray:
    """Restrict level-space densities to the positive price grid and renormalise.

    Counterpart of ``log_density_to_level`` for models fitted in levels: no
    Jacobian, only interpolation onto ``level_grid`` and renormalisation of the
    mass that fell on non-positive prices.
    """
    density_model = np.atleast_2d(np.asarray(density_model, dtype=float))
    out = np.empty((density_model.shape[0], level_grid.size), dtype=float)
    lost = np.empty(density_model.shape[0], dtype=float)
    for i in range(density_model.shape[0]):
        row = density_model[i]
        total = np.trapz(row, model_grid)
        fp = np.interp(level_grid, model_grid, row, left=0.0, right=0.0)
        mass = np.trapz(fp, level_grid)
        lost[i] = 1.0 - mass / total if total > 0 else 0.0
        out[i] = fp / mass if mass > 0 else fp
    if warn_above is not None and lost.max() > warn_above:
        n_bad = int((lost > warn_above).sum())
        warnings.warn(
            f"level fit: {n_bad} of {len(lost)} predictive densities put more than "
            f"{warn_above:.1%} of their mass on non-positive prices "
            f"(mean {lost.mean():.2%}, worst {lost.max():.2%}); "
            f"renormalisation redistributes it onto the positive grid.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def model_rel_path(model: str) -> str:
    """Directory of a model under ``horizon_*``, relative to it."""
    return model
