"""Nadaraya-Watson conditional density estimator."""

import numpy as np
from sklearn.preprocessing import RobustScaler


def _silverman_bandwidth(n, d):
    """Silverman's normal-reference bandwidth for robustly scaled d-dim data."""
    return (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * n ** (-1.0 / (d + 4.0))


def _gaussian(z):
    """Evaluate the standard normal density elementwise."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def kcde_predictive_density(
    X_train,
    y_train,
    X_query,
    grid_y,
    kernel="gaussian",
    bandwidth="silverman",
    chunk_size=512,
):
    """Nadaraya-Watson conditional density p(y | x) on ``grid_y``.

    ``X_train`` is (n_train, d), one column per lag; ``X_query`` the points to
    evaluate at. ``bandwidth`` accepts "silverman" or a float.
    """
    if kernel != "gaussian":
        raise ValueError(f"kcde is restricted to the gaussian kernel, got {kernel!r}")
    if bandwidth != "silverman":
        raise ValueError(
            f"kcde is restricted to the silverman bandwidth, got {bandwidth!r}"
        )

    X_train = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=np.float64))
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    grid_y = np.asarray(grid_y, dtype=np.float64).ravel()

    n_train, d = X_train.shape
    n_query = X_query.shape[0]
    n_grid = grid_y.shape[0]

    # Robust (median/IQR) scaling, fitted on the training split only.
    x_scaler = RobustScaler().fit(X_train)
    Xtr = x_scaler.transform(X_train)
    Xq = x_scaler.transform(X_query)

    y_scaler = RobustScaler().fit(y_train.reshape(-1, 1))
    ytr = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    gy = y_scaler.transform(grid_y.reshape(-1, 1)).ravel()
    y_scale = float(y_scaler.scale_[0]) or 1.0

    hx = _silverman_bandwidth(n_train, d)
    hy = _silverman_bandwidth(n_train, 1)

    # Response kernel, shared across all queries: (n_train, n_grid).
    Ky = _gaussian((gy[None, :] - ytr[:, None]) / hy) / hy

    out = np.empty((n_query, n_grid), dtype=np.float64)
    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)

        u = (Xq[start:stop, None, :] - Xtr[None, :, :]) / hx
        Kx = _gaussian(u).prod(axis=2)  # (chunk, n_train)
        denom = Kx.sum(axis=1, keepdims=True)

        bad = denom[:, 0] <= 1e-300
        if np.any(bad):
            dist2 = np.sum(u[bad] ** 2, axis=2)
            nn = np.argmin(dist2, axis=1)
            bad_rows = np.where(bad)[0]
            Kx[bad_rows, :] = 0.0
            Kx[bad_rows, nn] = 1.0
            denom = Kx.sum(axis=1, keepdims=True)

        # Self-normalizing mixture, mapped back to the original y scale.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out[start:stop] = (Kx @ Ky) / denom / y_scale

    area = np.trapezoid(out, grid_y, axis=1)[:, None]
    out /= np.clip(area, 1e-300, None)

    return out
