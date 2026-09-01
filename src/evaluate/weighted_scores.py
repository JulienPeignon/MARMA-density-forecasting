"""Proper weighted scoring rules for tail-focused density evaluation."""

import numpy as np

DENSITY_FLOOR = 1e-12


def side_for_level(level: float) -> str:
    """Levels below one half weight the lower tail, the others the upper tail."""
    return "lower" if level < 0.5 else "upper"


def weight_at(values, threshold, side: str) -> np.ndarray:
    """Return the indicator weight ``1{z >= r}`` upper, ``1{z <= r}`` lower."""
    values = np.asarray(values, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    return values >= threshold if side == "upper" else values <= threshold


def broadcast_thresholds(threshold, n_obs: int) -> np.ndarray:
    """``threshold`` as one value per observation.

    A scalar fixes the weighted region once for the whole sample; an array of
    length ``n_obs`` lets it move with the forecast origin, which is what the
    real-time application does.
    """
    threshold = np.atleast_1d(np.asarray(threshold, dtype=float))
    if threshold.size == 1:
        return np.full(n_obs, float(threshold[0]))
    if threshold.size != n_obs:
        raise ValueError(f"expected 1 or {n_obs} thresholds, got {threshold.size}")
    return threshold


def twcrps_per_obs(cdf, grid_y, y_true, threshold, side: str = "upper"):
    """Threshold-weighted CRPS (Gneiting and Ranjan, 2011) with indicator weight."""
    cdf = np.atleast_2d(np.asarray(cdf, dtype=float))
    grid_y = np.asarray(grid_y, dtype=float)
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    thresholds = broadcast_thresholds(threshold, len(cdf))

    out = np.empty(len(cdf), dtype=float)
    for i, (row, y, r) in enumerate(zip(cdf, y_true, thresholds)):
        keep = weight_at(grid_y, r, side)
        if keep.sum() < 2:
            out[i] = np.nan
            continue
        indicator = (grid_y >= y).astype(float)
        out[i] = np.trapz(((row - indicator) ** 2)[keep], grid_y[keep])
    return out


def csl_per_obs(
    density,
    grid_y,
    y_true,
    threshold,
    side: str = "upper",
    floor: float = DENSITY_FLOOR,
):
    """Censored likelihood score (Diks, Panchenko and van Dijk, 2011).."""
    density = np.atleast_2d(np.asarray(density, dtype=float))
    grid_y = np.asarray(grid_y, dtype=float)
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    thresholds = broadcast_thresholds(threshold, len(density))
    total = np.trapz(density, grid_y, axis=1)[:, None]
    f = density / np.where(total <= 0, 1.0, total)

    f_at_y = np.array([np.interp(y_true[i], grid_y, f[i]) for i in range(len(f))])
    n_floored = int((f_at_y < floor).sum())

    mass = np.empty(len(f), dtype=float)
    for i, r in enumerate(thresholds):
        keep = weight_at(grid_y, r, side)
        mass[i] = np.trapz(f[i, keep], grid_y[keep]) if keep.sum() >= 2 else 0.0
    mass = np.clip(mass, 0.0, 1.0)

    w_y = weight_at(y_true, thresholds, side).astype(float)
    scores = -(
        w_y * np.log(np.maximum(f_at_y, floor))
        + (1.0 - w_y) * np.log(np.maximum(1.0 - mass, floor))
    )
    return scores, n_floored
