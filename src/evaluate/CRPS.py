"""Continuous ranked probability score from a discretised CDF."""

import numpy as np


def crps_from_cdf(F, x_grid, y):
    """Compute the CRPS of a discretised CDF ``F`` at observation ``y``.

    ``F`` is (n,) or (batch, n) over ``x_grid`` (n,); ``y`` is scalar or (batch,).
    """
    F = np.asarray(F)
    x_grid = np.asarray(x_grid)
    if F.ndim == 1:
        indicator = (x_grid >= y).astype(float)
        return np.trapz((F - indicator) ** 2, x_grid)
    else:
        # vectorized batch version
        crps = []
        for Fi, yi in zip(F, np.atleast_1d(y)):
            indicator = (x_grid >= yi).astype(float)
            crps.append(np.trapz((Fi - indicator) ** 2, x_grid))
        return np.array(crps)
