"""Shared helpers for the forecast methods: grids, splits and CDE checks."""

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from src.utils.setup_logger import setup_logger

logger = setup_logger()


def generalized_boxplot(data, alpha=0.05, p=0.9, side="both"):
    """Compute generalized-boxplot whiskers for skewed, heavy-tailed data.

    `data` is (n,) univariate or (n, p) multivariate; returns
    (lower_whisker, upper_whisker) as scalars (univariate) or (p,) arrays.
    """
    if side not in ("both", "upper"):
        raise ValueError(f"side must be 'both' or 'upper', got {side!r}")
    data = np.array(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
        is_univariate = True
    else:
        is_univariate = False

    _, n_dims = data.shape

    lower_whiskers = np.zeros(n_dims)
    upper_whiskers = np.zeros(n_dims)

    for dim in range(n_dims):
        data_dim = data[:, dim]

        l0 = np.median(data_dim)
        s0 = np.percentile(data_dim, 75) - np.percentile(data_dim, 25)  # IQR

        if s0 < 1e-10:  # Handle constant data
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: constant data detected, using median as whiskers"
            )
            continue

        x_star = (data_dim - l0) / s0

        zeta = 0.1
        r = x_star - np.min(x_star) + zeta

        r_range = np.max(r) - np.min(r)
        if r_range < 1e-10:
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: insufficient range, using median as whiskers"
            )
            continue

        r_tilde = (r - np.min(r)) / r_range

        r_tilde = np.clip(r_tilde, 1e-10, 1 - 1e-10)
        w = stats.norm.ppf(r_tilde)

        w_median = np.median(w)
        w_iqr = np.percentile(w, 75) - np.percentile(w, 25)

        if w_iqr < 1e-10:
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: insufficient IQR in transformed space, "
                f"using median as whiskers"
            )
            continue

        w_star = (w - w_median) / (w_iqr / 1.3426)

        zp = stats.norm.ppf(p)
        Qp = np.percentile(w_star, p * 100)
        Q1mp = np.percentile(w_star, (1 - p) * 100)

        if np.abs(Qp + Q1mp) > 1e-10:
            g = (1 / zp) * np.log(-Qp / Q1mp)
        else:
            g = 0

        if g != 0 and np.abs(Qp + Q1mp) > 1e-10:
            denom = Qp + Q1mp
            if np.abs(denom) > 1e-10:
                h = (2 * np.log(-g * Qp * Q1mp / denom)) / (zp**2)
            else:
                h = 0
        else:
            h = 0

        # Standard-normal quantiles at which to place the whiskers.
        if side == "upper":
            z_lower = None
            z_upper = stats.norm.ppf(1 - alpha)
        else:
            z_lower = stats.norm.ppf(alpha / 2)
            z_upper = stats.norm.ppf(1 - alpha / 2)

        def _whisker(z):
            """Map a normal quantile through the fitted g-and-h to data scale."""
            if np.abs(g) > 1e-10:
                xi = (1 / g) * (np.exp(g * z) - 1) * np.exp(h * z**2 / 2)
            else:
                xi = z * np.exp(h * z**2 / 2)
            transform = stats.norm.cdf(w_median + (w_iqr / 1.3426) * xi)
            return (transform * r_range + np.min(x_star)) * s0 + l0

        lower_whisker = -np.inf if z_lower is None else _whisker(z_lower)
        upper_whisker = _whisker(z_upper)

        lower_whiskers[dim] = lower_whisker
        upper_whiskers[dim] = upper_whisker

        logger.info(
            f"Dimension {dim}: generalized boxplot boundaries: "
            f"{lower_whisker:.3f}, {upper_whisker:.3f}"
        )

    if is_univariate:
        return lower_whiskers[0], upper_whiskers[0]
    else:
        return lower_whiskers, upper_whiskers


def prepare_tensors(
    df,
    X,
    y,
    lags,
    horizon,
    proportions,
    device,
    upper_tail_only=False,
):
    """Prepare lagged train/val/test tensors with tail weighting."""
    logger = setup_logger()

    if df is None and (X is None or y is None):
        raise ValueError("Either df OR both (X, y) must be provided")

    if df is not None and X is not None:
        logger.warning("Both df and (X, y) provided. Using X and y, ignoring df.")

    if len(proportions) != 3:
        raise ValueError("proportions must be a 3-tuple (train, val, test)")
    if any(p < 0 for p in proportions):
        raise ValueError("proportions must be non-negative")
    if abs(sum(proportions) - 1.0) > 1e-8:
        raise ValueError(f"proportions must sum to 1, got {sum(proportions)}")

    p_train, p_val, p_test = proportions

    if X is not None and y is not None:
        X_data = X.flatten() if isinstance(X, np.ndarray) else np.array(X).flatten()
        y_data = y.flatten() if isinstance(y, np.ndarray) else np.array(y).flatten()
        logger.info(f"Using provided X and y arrays: X{X_data.shape}, y{y_data.shape}")
    else:
        if df.shape[1] > 1:
            logger.warning(
                f"DataFrame has {df.shape[1]} columns, using only the first "
                f"column for univariate mode"
            )
        X_data = df.iloc[:, 0].values if hasattr(df, "iloc") else df.values.flatten()
        y_data = X_data

    X_list, y_list = [], []
    max_t = min(len(X_data), len(y_data) - horizon + 1)
    for t in range(lags, max_t):
        X_list.append(X_data[t - lags : t])
        y_list.append(y_data[t + horizon - 1])

    X_array = np.array(X_list, dtype=np.float32)  # (n_samples, lags)
    y_array = np.array(y_list, dtype=np.float32)  # (n_samples,)

    X_tensor = torch.tensor(X_array, device=device)
    y_tensor = torch.tensor(y_array, device=device)
    n_samples = len(X_tensor)

    # Minimum gap to make windows non-overlapping
    gap = lags + horizon - 1

    n_train = int(p_train * n_samples)
    n_val = int(p_val * n_samples)

    if n_train <= 0:
        raise ValueError(
            f"Empty training set (n_samples={n_samples}, proportions={proportions}, "
            f"gap={gap}). Provide more data or adjust proportions."
        )

    train_end = n_train
    X_train, y_train = X_tensor[:train_end], y_tensor[:train_end]

    if p_val > 0:
        val_start = train_end + gap
        if p_test > 0:
            val_end = val_start + n_val
        else:
            val_end = n_samples
        if val_start >= n_samples or val_end <= val_start:
            raise ValueError(
                f"Empty/invalid validation set: val=[{val_start}:{val_end}], "
                f"n_samples={n_samples}, gap={gap}. Adjust proportions/lags/horizon."
            )
        X_val, y_val = X_tensor[val_start:val_end], y_tensor[val_start:val_end]
    else:
        val_end = train_end
        X_val, y_val = None, None

    if p_test > 0:
        test_start = val_end + gap
        if test_start >= n_samples:
            raise ValueError(
                f"Empty test set: test starts at {test_start} >= "
                f"n_samples={n_samples}, "
                f"gap={gap}. Adjust proportions/lags/horizon."
            )
        X_test, y_test = X_tensor[test_start:], y_tensor[test_start:]
    else:
        X_test, y_test = None, None

    logger.info(
        f"Split (samples): train={len(X_train)}, "
        f"val={0 if X_val is None else len(X_val)}, "
        f"test={0 if X_test is None else len(X_test)} | "
        f"gap={gap} (lags={lags}, horizon={horizon})"
    )

    # Tail boundaries.
    side = "upper" if upper_tail_only else "both"
    train_values_np = X_train[:, -1].detach().cpu().numpy()
    lower_bound, upper_bound = generalized_boxplot(train_values_np, side=side)
    logger.info(
        f"Tail bounds ({side}) estimated on TRAIN only: "
        f"lower={lower_bound:.3f}, upper={upper_bound:.3f}"
    )

    # Weights. A sample is a tail sample when its most recent observation
    # (last column of the lag window) is outside the whiskers
    weights_train = torch.ones(len(X_train), device=device)
    last_train = X_train[:, -1]
    is_tail = (last_train < lower_bound) | (last_train > upper_bound)
    tail_count = is_tail.sum().item()
    logger.info(f"Tail samples (train) = {tail_count}, {tail_count / len(X_train):.2%}")
    if tail_count > 0:
        inverse_tail_proportion = np.sqrt(len(X_train) / tail_count)
        weights_train[is_tail] = inverse_tail_proportion
        # Pre-normalize to mean 1
        weight_norm = weights_train.mean().item()
        weights_train /= weight_norm
        logger.info(
            f"Tail weight = {inverse_tail_proportion:.3f} "
            f"(mean-1 normalized: tail={inverse_tail_proportion / weight_norm:.3f}, "
            f"bulk={1 / weight_norm:.3f})"
        )
    else:
        inverse_tail_proportion = None
        weight_norm = 1.0
    weights_train = weights_train.cpu()

    if X_val is not None:
        weights_val = torch.ones(len(X_val), device=device)
        last_val = X_val[:, -1]
        is_tail_val = (last_val < lower_bound) | (last_val > upper_bound)
        if inverse_tail_proportion is not None:
            weights_val[is_tail_val] = inverse_tail_proportion
            weights_val /= weight_norm
        weights_val = weights_val.cpu()
    else:
        weights_val = None

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        weights_train,
        weights_val,
    )


def check_cde(cde, grid, tol=5e-2, verbose=True):
    """Check that the densities integrate to one within ``tol``.

    Raises:
        ValueError: if any row is outside the tolerance.
    """
    dx = float(grid[1] - grid[0])
    cde_np = cde.detach().cpu().numpy() if hasattr(cde, "detach") else np.asarray(cde)

    if not np.isfinite(cde_np).all():
        n_nan = np.isnan(cde_np).sum()
        n_inf = np.isinf(cde_np).sum()
        raise ValueError(f"CDE contains {n_nan} NaN and {n_inf} Inf values")

    integrals = np.sum(cde_np * dx, axis=1)

    if verbose:
        print(
            f"CDE integrals: mean = {integrals.mean():.6f}, std = {integrals.std():.6f}"
        )

    if not np.allclose(integrals, 1.0, atol=tol):
        max_error = np.abs(integrals - 1.0).max()
        n_bad = np.sum(np.abs(integrals - 1.0) > tol)
        raise ValueError(
            f"CDEs do not integrate to 1: {n_bad}/{len(integrals)} rows "
            f"exceed tolerance. "
            f"Max error = {max_error:.6f} (tolerance = {tol})"
        )

    return integrals


def compute_mode(density, grid):
    """Mode of each density row, by peak detection."""
    density = np.atleast_2d(density)  # Ensure shape (n_samples, len(grid))
    modes = []

    for i in range(density.shape[0]):
        dens = density[i]
        peaks, _ = find_peaks(dens)

        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(dens[peaks])]
            mode = grid[peak_idx]
        else:
            # Fallback: use optimization
            def density_interp(x):
                return -np.interp(x, grid, dens)

            result = minimize_scalar(
                density_interp, bounds=(grid[0], grid[-1]), method="bounded"
            )
            mode = result.x

        modes.append(mode)

    return np.array(modes)


def make_grid(quantiles, n_points):
    """Build the conditioning and evaluation grids from the process quantiles."""
    grid_X = np.linspace(
        quantiles["Lower Quantiles"][0.01], quantiles["Upper Quantiles"][0.99], n_points
    )

    grid_y = np.linspace(
        quantiles["Lower Quantiles"][0.001],
        quantiles["Upper Quantiles"][0.999],
        n_points,
    )

    return grid_X, grid_y
