"""Exact predictive densities for causal linear models with Gaussian innovations.

For any causal linear model driven by i.i.d. Gaussian errors, the h-step ahead
law is itself Gaussian and its parameters are available in closed form. Writing
the model in MA(infinity) form, ``y_t = mu_t + sum_j psi_j eps_{t-j}``,

    y_{t+h} = (h-step point forecast) + sum_{j=0}^{h-1} psi_j eps_{t+h-j}

because every eps dated t or earlier is known at the forecast origin.
"""

from __future__ import annotations

import numpy as np

SES_ALPHA = 0.8


def ses_filter(y: np.ndarray, alpha: float = SES_ALPHA):
    """Exponential smoothing recursion -- exact port of ``Table2/smooth.m``."""
    y = np.asarray(y, dtype=float).ravel()
    level = float(y[0])
    resid = np.empty(len(y) - 1, dtype=float)
    for i in range(1, len(y)):
        resid[i - 1] = y[i] - level
        level = alpha * level + (1.0 - alpha) * y[i]
    return level, resid


def ar_ma_weights(coefs, horizon: int) -> np.ndarray:
    """First ``horizon`` MA(infinity) weights of a causal AR(p)."""
    coefs = np.asarray(coefs, dtype=float)
    p = len(coefs)
    psi = np.zeros(horizon, dtype=float)
    psi[0] = 1.0
    for k in range(1, horizon):
        psi[k] = sum(coefs[i - 1] * psi[k - i] for i in range(1, min(k, p) + 1))
    return psi


def ima11_ma_weights(theta: float, horizon: int) -> np.ndarray:
    """MA weights of the IMA(1,1) behind exponential smoothing."""
    psi = np.full(horizon, 1.0 - float(theta), dtype=float)
    psi[0] = 1.0
    return psi


def gaussian_sum_params(psi, sigma: float) -> float:
    """Scale of ``sum_j psi_j eps_j`` for i.i.d. Gaussian ``eps`` with sd ``sigma``."""
    psi = np.asarray(psi, dtype=float)
    return float(sigma) * float(np.sqrt(np.sum(psi**2)))


def gaussian_predictive_density(grid_y, locations, sigma_h: float):
    """Build normal densities on ``grid_y``, one row per ``locations`` entry."""
    grid_y = np.asarray(grid_y, dtype=float).ravel()
    locations = np.asarray(locations, dtype=float).ravel()
    z = (grid_y[None, :] - locations[:, None]) / float(sigma_h)
    return np.exp(-0.5 * z**2) / (float(sigma_h) * np.sqrt(2.0 * np.pi))
