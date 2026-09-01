"""Simulation plots: trajectories, conditional moments and density panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.setup_logger import setup_logger

logger = setup_logger()


def plot_predictive_density_given_yt(
    predictive_density_at_yt, y_t, grid, quantiles=None
):
    """Plot the predictive density at ``y_t``, marking the quantiles."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        grid, predictive_density_at_yt, lw=2, label=r"$\hat{p}_h(X_{t+h} \mid X_t)$"
    )

    if y_t is not None:
        ax.axvline(y_t, color="red", ls="--", lw=1.8, label="$X_t$")

    tick_positions = []
    tick_labels = []
    if quantiles is not None:
        for cat in ["Lower Quantiles", "Upper Quantiles"]:
            if cat in quantiles:
                for p, q in quantiles[cat].items():
                    ax.axvline(q, color="gray", ls=":", lw=1.2, alpha=0.7)
                    tick_positions.append(q)
                    tick_labels.append(rf"$q_{{{100 * p:g}\%}}$")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        fontsize=12,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()

    return fig


def plot_predictive_moments_compare(
    predictive_density,
    theoretical_moments,
    grid_x,
    grid_y,
    quantiles=None,
    max_quantile=None,
    file_name=None,
):
    """Plot estimated against theoretical conditional moments (orders 1-4)."""
    orders = theoretical_moments.shape[1]

    dy = float(np.diff(grid_y).mean())
    powers = np.stack([grid_y**k for k in range(1, orders + 1)], axis=1)
    estimated_moments = predictive_density @ powers * dy

    def get_quantile_value(quant_dict, target_q):
        target_str = f"{target_q:.3f}"
        keys_map = {f"{float(k):.3f}": v for k, v in quant_dict.items()}
        if target_str in keys_map:
            return keys_map[target_str]
        else:
            raise ValueError(
                f"Quantile {target_q} not found. "
                f"Available quantiles: {list(keys_map.keys())}"
            )

    if quantiles is not None and (
        quantiles.get("Lower Quantiles") or quantiles.get("Upper Quantiles")
    ):
        all_q = []
        if "Lower Quantiles" in quantiles:
            all_q.extend(quantiles["Lower Quantiles"].values())
        if "Upper Quantiles" in quantiles:
            all_q.extend(quantiles["Upper Quantiles"].values())
        q_min_all, q_max_all = min(all_q), max(all_q)
    else:
        q_min_all, q_max_all = grid_x.min(), grid_x.max()

    if max_quantile is not None:
        if quantiles is None or (
            "Lower Quantiles" not in quantiles or "Upper Quantiles" not in quantiles
        ):
            raise ValueError("Quantiles dict must be provided when using max_quantile.")
        lower_q = get_quantile_value(quantiles["Lower Quantiles"], max_quantile)
        upper_q = get_quantile_value(quantiles["Upper Quantiles"], 1 - max_quantile)
        q_min, q_max = lower_q, upper_q
    else:
        q_min, q_max = q_min_all, q_max_all

    x_range = q_max - q_min
    x_margin = 0.03 * x_range
    q_min_plot = q_min - x_margin
    q_max_plot = q_max + x_margin

    mask = (grid_x >= q_min) & (grid_x <= q_max)
    grid_trimmed = grid_x[mask]
    est_trimmed = estimated_moments[mask]
    theo_trimmed = theoretical_moments[mask]

    est_style = dict(color="tab:blue", lw=2)
    theo_style = dict(color="tab:orange", lw=2, ls="--")

    if orders == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.5))
        axes = [axes]
    elif orders == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    else:
        ncols = 2
        nrows = int(np.ceil(orders / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[orders:]:
            fig.delaxes(ax)
        axes = axes[:orders]

    for idx, k in enumerate(range(1, orders + 1)):
        ax = axes[idx]
        ax.set_title(rf"$k={k}$ moment", visible=True)
        ax.set_ylabel(rf"$E\,[Y_{{t+1}}^{k}\mid Y_t=y_t]$", visible=True)
        ax.plot(grid_trimmed, est_trimmed[:, idx], label="Estimated", **est_style)
        ax.plot(grid_trimmed, theo_trimmed[:, idx], label="Theoretical", **theo_style)
        if quantiles is not None:
            for label, qs in quantiles.items():
                for p, q in qs.items():
                    if q_min <= q <= q_max:
                        ax.axvline(q, color="gray", ls=":", lw=1.2, alpha=0.7)
                        ax.text(
                            q,
                            -0.02,
                            f"${{{p}}}$",
                            ha="center",
                            va="top",
                            transform=ax.get_xaxis_transform(),
                            fontsize=8,
                            rotation=45,
                        )
        for ax in axes:
            ax.set_xticks([])
            ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
        ax.set_xlim(q_min_plot, q_max_plot)
        ax.set_title(rf"$k={k}$ moment")
        ax.set_ylabel(rf"$E\,[Y_{{t+1}}^{k}\mid Y_t=y_t]$")
        ax.grid(axis="x", alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    # Set x-labels only on bottom row
    bottom_start = (int(np.ceil(orders / 2)) - 1) * 2
    for i, ax in enumerate(axes):
        if i >= bottom_start:
            ax.set_xlabel(r"$y_t$")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.98)

    if file_name:
        fig.savefig(file_name, dpi=300)
        logger.info(f"Saved plot to {file_name}")
    else:
        plt.show()


def plot_three_density_slices(
    densities_path,
    grid_x,
    grid_y,
    low_high_quantiles=(0.10, 0.90),  # (low_prob, high_prob) e.g., (0.10, 0.90)
    quantiles=None,  # dict from compute_theoretical_quantiles(...)
    abs_thresh=1e-3,  # absolute threshold ONLY
    pad_frac=0.02,  # padding fraction of full grid_y span
    allow_nearest_prob=True,  # if requested prob missing, use nearest available
):
    """Plot predictive densities at three conditioning values of X_t.

    The panels are X_t at the low process quantile, X_t = 0, and X_t at the high
    process quantile.
    """
    if quantiles is None:
        raise ValueError(
            "quantiles is required (output of compute_theoretical_quantiles)."
        )

    # Merge lower & upper quantiles into a single probability->value map
    qmap = {}
    for key in ("Lower Quantiles", "Upper Quantiles"):
        if key in quantiles and isinstance(quantiles[key], dict):
            for p, v in quantiles[key].items():
                qmap[float(p)] = float(v)
    if not qmap:
        raise ValueError(
            "quantiles must contain 'Lower Quantiles' and/or 'Upper Quantiles' dicts."
        )

    probs_available = np.array(sorted(qmap.keys()), dtype=float)

    def prob_to_value(p_req: float) -> float:
        """Map quantile probability to its process value using provided quantiles."""
        if p_req in qmap:
            return qmap[p_req]
        if not allow_nearest_prob:
            raise KeyError(
                f"Requested quantile level {p_req} not found in provided quantiles."
            )
        idx = int(np.argmin(np.abs(probs_available - p_req)))
        return qmap[float(probs_available[idx])]

    # Determine the three conditioning X_t values: q_low, 0, q_high
    p_low, p_high = map(float, low_high_quantiles)

    x_low = prob_to_value(p_low)
    x_zero = 0.0
    x_high = prob_to_value(p_high)

    def snap_to_index(x_val: float) -> tuple[int, float]:
        i = int(np.argmin(np.abs(grid_x - x_val)))
        return i, float(grid_x[i])

    i_low, x_low_snap = snap_to_index(x_low)
    i_zero, x_zero_snap = snap_to_index(x_zero)
    i_high, x_high_snap = snap_to_index(x_high)

    df = pd.read_parquet(densities_path)
    if "density" in df.columns:
        densities = df["density"].values
    else:
        densities = df.values.flatten()
    densities_2d = densities.reshape(len(grid_x), len(grid_y))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    grid_width = float(grid_y.max() - grid_y.min())
    pad = pad_frac * grid_width

    panels = [
        (axes[0], i_low, f"$X_t = q_{{{int(round(p_low * 100))}}}$", x_low_snap),
        (axes[1], i_zero, r"$X_t = 0$", x_zero_snap),
        (axes[2], i_high, f"$X_t = q_{{{int(round(p_high * 100))}}}$", x_high_snap),
    ]

    for ax, idx, title, x_vline in panels:
        density_slice = densities_2d[idx, :]

        ax.plot(grid_y, density_slice, lw=2)
        ax.axvline(x_vline, color="red", ls="--", lw=1.8)
        ax.set_ylim(bottom=0)

        # Trim x-axis using ONLY absolute threshold (allow internal zeros/gaps)
        mask = np.asarray(density_slice) >= abs_thresh
        if np.any(mask):
            y_min = grid_y[np.argmax(mask)]
            y_max = grid_y[len(mask) - 1 - np.argmax(mask[::-1])]
            xlo = max(grid_y.min(), y_min - pad)
            xhi = min(grid_y.max(), y_max + pad)
            if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
                xlo, xhi = grid_y.min(), grid_y.max()
            ax.set_xlim(xlo, xhi)
        else:
            ax.set_xlim(grid_y.min(), grid_y.max())

        ax.set_title(f"{title}", fontsize=14)

    axes[0].set_ylabel("Predictive conditional density", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=0.96)

    out_dir = Path(densities_path).parent
    out_path = out_dir / f"{Path(densities_path).stem}_densities.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
