"""Shared local-PIT recalibration helpers."""

import numpy as np

from src.calibration.LocalPITRecalibrator import LocalPITRecalibrator
from src.calibration.metrics import probability_integral_transform

DEFAULT_NUM_BASIS = 5
ALPHAS = np.linspace(0.0, 1.0, 41)


def _to_numpy(x):
    """Float array from a numpy array or a torch tensor on any device."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=float)


def lagged_pairs(series, horizon, lags=1):
    """Return lagged features and h-step outcomes from a time series."""
    values = np.asarray(series)
    offset = lags + horizon - 1
    X = np.array([values[i : i + lags] for i in range(len(values) - offset)])
    return X, values[offset:]


def fit_and_apply_recalibration(
    X_calibration,
    pit,
    X_target,
    density_target,
    grid_y,
    n_jobs,
    num_basis=DEFAULT_NUM_BASIS,
):
    """Fit the local PIT recalibrator on ``(X_calibration, pit)`` and apply it."""
    calibrator = LocalPITRecalibrator(n_jobs=n_jobs, num_basis=num_basis)
    calibrator.fit(X_calibration, pit, alphas=ALPHAS)
    return calibrator.transform(X_target, density_target, grid_y)


def recalibrate_density(
    X_calibration,
    y_calibration,
    density_calibration,
    X_target,
    density_target,
    grid_y,
    n_jobs,
    num_basis=DEFAULT_NUM_BASIS,
    artifacts=None,
):
    """Fit local PIT recalibration data and apply it to target densities."""
    pit = probability_integral_transform(density_calibration, grid_y, y_calibration)
    recalibrated = fit_and_apply_recalibration(
        X_calibration, pit, X_target, density_target, grid_y, n_jobs, num_basis
    )

    if artifacts is not None:
        artifacts["raw"] = _to_numpy(density_target)
        artifacts["pit"] = _to_numpy(pit).ravel()
        artifacts["X_calibration"] = _to_numpy(X_calibration)
        artifacts["X_target"] = _to_numpy(X_target)
        artifacts["num_basis"] = int(num_basis)

    return recalibrated
