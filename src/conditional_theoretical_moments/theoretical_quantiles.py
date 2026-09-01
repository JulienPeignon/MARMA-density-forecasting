"""Conditional quantiles of a stable MAR/MARMA predictive density."""

import numpy as np

from src.stable_mar.stable import stable_ppf
from src.stable_mar.stable_mar import madelta


def compute_theoretical_quantiles(
    phi_vec,
    psi_vec,
    alpha,
    beta,
    sigma,
    theta=None,
    eta=None,
    ma_trunc=100,
):
    """Lower and upper quantiles of a stable MAR process.

    ``ma_trunc`` truncates the infinite MA expansion.
    """
    coeff_ma = np.array(
        [
            madelta(cvec=phi_vec, ncvec=psi_vec, k=k, theta=theta, eta=eta)
            for k in range(-ma_trunc, ma_trunc)
        ],
        dtype=float,
    )
    coeff_ma = np.real(coeff_ma)

    # Compute scale and skewness-adjusted parameters
    abs_ma_alpha = np.abs(coeff_ma) ** alpha
    sig1 = (sigma**alpha) * np.sum(abs_ma_alpha)
    bet1 = beta * np.sum(np.sign(coeff_ma) * abs_ma_alpha) / np.sum(abs_ma_alpha)
    scale = sig1 ** (1 / alpha)

    # Quantile probabilities
    lower_probs = [0.1, 0.05, 0.01, 0.005, 0.001]
    upper_probs = [0.9, 0.95, 0.99, 0.995, 0.999]

    # Compute quantiles
    lower_quantiles = stable_ppf(lower_probs, alpha, bet1, loc=0, scale=scale)
    upper_quantiles = stable_ppf(upper_probs, alpha, bet1, loc=0, scale=scale)

    return {
        "Lower Quantiles": dict(zip(lower_probs, lower_quantiles)),
        "Upper Quantiles": dict(zip(upper_probs, upper_quantiles)),
    }
