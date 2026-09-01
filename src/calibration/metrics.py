"""Calibration diagnostics: PIT values and the Anderson-Darling distance."""

import numpy as np


def probability_integral_transform(cde, y_grid, y_test):
    """Probability integral transform of ``y_test`` under the densities ``cde``."""

    def _to_numpy(x):
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    # ensure numpy
    cde = _to_numpy(cde)
    y_grid = _to_numpy(y_grid).ravel()
    y_test = _to_numpy(y_test).ravel()

    # sanity checks
    nrow_cde, ncol_cde = cde.shape
    n_samples = y_test.shape[0]
    n_grid_points = y_grid.shape[0]

    if nrow_cde != n_samples:
        raise ValueError(
            f"Number of samples in CDEs should match y_test. "
            f"Got {nrow_cde} vs {n_samples}."
        )
    if ncol_cde != n_grid_points:
        raise ValueError(
            f"Number of grid points in CDEs should match y_grid. "
            f"Got {ncol_cde} vs {n_grid_points}."
        )

    # integrate density up to each y_test via masked trapezoid rule
    pit_masked = np.ma.masked_array(cde, (y_grid > y_test[:, None]))
    pit = np.trapezoid(pit_masked, y_grid, axis=-1)

    return np.asarray(pit)
