"""CDE loss and coverage diagnostics, ported from the cdetools package."""

import numpy as np


def cde_loss(cde_estimates, z_grid, z_test):
    """Compute the CDE loss of densities ``cde_estimates`` at ``z_test``.

    Each row of ``cde_estimates`` (n_samples, n_grid) is a PDF over ``z_grid``.
    """
    cde_estimates = np.asarray(cde_estimates, float)
    z_grid = np.asarray(z_grid, float).ravel()
    z_test = np.asarray(z_test, float).ravel()

    n_samples, _ = cde_estimates.shape

    # integral term  ∫ p^2 dy
    integrals = np.trapz(cde_estimates * cde_estimates, z_grid, axis=1)

    # pointwise term  p(y_i | x_i) using interpolation
    likeli = np.array(
        [np.interp(z_test[i], z_grid, cde_estimates[i]) for i in range(n_samples)]
    )

    losses = integrals - 2.0 * likeli
    loss = losses.mean()
    se = losses.std(ddof=1) / np.sqrt(n_samples)
    return loss, se
