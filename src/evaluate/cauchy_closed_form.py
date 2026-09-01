"""Closed-form predictive density of a noncausal Cauchy AR(1)."""

import numpy as np


def cauchy_ar1_predictive_density(grid_x, grid_y, psi, sigma, h):
    """Predictive density of a noncausal Cauchy AR(1), in closed form.

    Evaluated on ``grid_y`` for each conditioning value in ``grid_x``.
    """
    # Convert inputs to numpy arrays
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    # Check if grid_x is scalar
    is_scalar_x = grid_x.ndim == 0
    if is_scalar_x:
        grid_x = grid_x.reshape(1)

    # Compute sigma_h
    sigma_h = sigma * (1 - np.abs(psi) ** h) / (1 - np.abs(psi))

    # Create meshgrid for vectorized computation
    Y_t_minus_h = grid_x[:, np.newaxis]  # Shape: (len(grid_x), 1)
    Y_t = grid_y[np.newaxis, :]  # Shape: (1, len(grid_y))

    # Compute the density components
    # Term 1: 1 / (pi * sigma_h)
    term1 = 1.0 / (np.pi * sigma_h)

    # Term 2: 1 / (1 + (Y_{t-h} - psi^h * Y_t)^2 / sigma_h^2)
    residual_sq = (Y_t_minus_h - psi**h * Y_t) ** 2
    term2 = 1.0 / (1.0 + residual_sq / sigma_h**2)

    # Term 3: (sigma^2 + (1-|psi|)^2 * Y_{t-h}^2) / (sigma^2 + (1-|psi|)^2 * Y_t^2)
    factor = (1 - np.abs(psi)) ** 2
    numerator3 = sigma**2 + factor * Y_t_minus_h**2
    denominator3 = sigma**2 + factor * Y_t**2
    term3 = numerator3 / denominator3

    # Combine all terms
    density = term1 * term2 * term3

    # Return scalar output if input was scalar
    if is_scalar_x:
        return density.flatten()

    return density
