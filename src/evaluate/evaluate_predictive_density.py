"""Scoring of simulated predictive densities: moment RMSE, KL and ISE."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.plots.gif import generate_moving_density_gif
from src.plots.simulate_and_plot import (
    plot_predictive_density_given_yt,
    plot_predictive_moments_compare,
    plot_three_density_slices,
)
from src.utils.save_predictive_densities import (
    load_predictive_density,
    save_predictive_density,
)
from src.utils.setup_logger import setup_logger

logger = setup_logger()


def compute_density_metrics(
    true_density, estimated_density, grid_y, idx=None, eps=1e-12
):
    """Mean KL divergence and ISE between true and estimated densities.

    Both are computed per conditioning value, then averaged; ``idx`` restricts the
    comparison to a slice of ``grid_y``.
    """
    true_density = np.asarray(true_density)
    estimated_density = np.asarray(estimated_density)
    grid_y = np.asarray(grid_y)

    if idx is not None:
        true_density = true_density[idx]
        estimated_density = estimated_density[idx]

    true_density = np.clip(true_density, eps, None)
    estimated_density = np.clip(estimated_density, eps, None)

    # KL divergence: int f_true log(f_true / f_est)
    kl = np.trapezoid(
        true_density * np.log(true_density / estimated_density), grid_y, axis=1
    )

    # ISE: int (f_true - f_est)^2
    ise = np.trapezoid((true_density - estimated_density) ** 2, grid_y, axis=1)

    mean_KL = np.mean(kl)
    mean_ISE = np.mean(ise)

    return mean_KL, mean_ISE


def evaluate_predictive_density(
    predictive_density: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    theoretical_moments: np.ndarray,
    quantiles: dict,
    file_name: str | None = None,
):
    """RMSE of the predictive moments, by region of ``grid_x``.

    ``quantiles`` sets the region bounds; with ``file_name`` the per-region results
    are also written to CSV/JSON.
    """
    n_x = len(grid_x)
    n_moments = theoretical_moments.shape[1]

    # integrate over y_{t+1} (columns): use grid_y
    dy = float(np.diff(grid_y).mean())
    orders = np.arange(1, n_moments + 1)
    powers_y = np.stack([grid_y**k for k in orders], axis=1)  # (n_y, n_moments)

    # predictive_density: (n_x, n_y)
    # powers_y:           (n_y, n_moments)
    estimated_moments = predictive_density @ powers_y * dy  # (n_x, n_moments)

    flat_q = {
        str(p): v
        for cat in ["Lower Quantiles", "Upper Quantiles"]
        for p, v in quantiles[cat].items()
    }

    def q(prob, default_idx):
        """Return the quantile location or a reasonable default on grid_x."""
        return flat_q.get(str(prob), grid_x[int(default_idx * n_x)])

    # Build index ranges over grid_x
    grid_ranges = [
        {
            "name": "Total",
            "idx": np.arange(
                np.searchsorted(grid_x, q(0.01, 0.01)),
                np.searchsorted(grid_x, q(0.99, 0.99)) + 1,
            ),
        },
        {
            "name": "Between 0.1-0.9",
            "idx": np.arange(
                np.searchsorted(grid_x, q(0.1, 0.1)),
                np.searchsorted(grid_x, q(0.9, 0.9)) + 1,
            ),
        },
        {
            "name": "Between 0.01-0.1 and 0.9-0.99",
            "idx": np.concatenate(
                [
                    np.arange(
                        np.searchsorted(grid_x, q(0.01, 0.01)),
                        np.searchsorted(grid_x, q(0.1, 0.1)),
                    ),
                    np.arange(
                        np.searchsorted(grid_x, q(0.9, 0.9)),
                        np.searchsorted(grid_x, q(0.99, 0.99)) + 1,
                    ),
                ]
            ),
        },
    ]

    results = {}
    for r in grid_ranges:
        rng_name, idx = r["name"], r["idx"]
        results[rng_name] = {}
        if idx.size == 0:
            for k in range(1, n_moments + 1):
                results[rng_name][f"Moment {k}"] = np.nan
            continue
        for k in range(n_moments):
            est, theo = estimated_moments[idx, k], theoretical_moments[idx, k]
            results[rng_name][f"Moment {k + 1}"] = float(
                np.sqrt(np.mean((est - theo) ** 2))
            )

    logger.info("RMSE Evaluation of Predictive Density\n" + "=" * 37)
    for rng_name, vals in results.items():
        logger.info(f"{rng_name}:")
        for m, rmse in vals.items():
            logger.info(f"  {m}: {rmse:.6f}")

    if file_name:
        if file_name.endswith(".csv"):
            df = pd.DataFrame(results).T
            df.to_csv(file_name)
            logger.info(f"Results saved to {file_name} (CSV format)")
        elif file_name.endswith(".json"):
            with open(file_name, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {file_name} (JSON format)")
        else:
            logger.warning(
                f"Unsupported file extension for {file_name}. No file saved."
            )

    mean_rmse = float(
        np.nanmean([v for k, v in results["Total"].items() if k.startswith("Moment")])
    )

    return mean_rmse


def run_postprocessing(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    theoretical_moments: np.ndarray,
    quantiles: dict,
    file_name: str,
    model: str,
    density: str,
    predictive_density: np.ndarray | None = None,
    true_predictive_density: np.ndarray | None = None,
    load: bool = False,
    gif: bool = False,
):
    """Save or reload the densities, plot them, and score the moments.

    ``predictive_density`` is saved when given and reloaded when ``load`` is set;
    ``true_predictive_density`` enables the KL/ISE metrics and ``gif`` the
    moving-density animation.
    """
    if model == "MDN":
        model = f"MDN_{density}"
    if load:
        predictive_density = load_predictive_density(file_name, model)
    elif predictive_density is not None:
        save_predictive_density(predictive_density, file_name, model)
    else:
        raise ValueError("Either set load=True or provide predictive_density.")

    plot_predictive_moments_compare(
        predictive_density,
        theoretical_moments,
        grid_x,
        grid_y,
        quantiles,
        max_quantile=0.01,
        file_name=f"{file_name}/{model}/{model}.png",
    )

    plot_three_density_slices(
        densities_path=f"{file_name}/{model}/{model}.parquet",
        grid_x=grid_x,
        grid_y=grid_y,
        low_high_quantiles=(0.01, 0.99),
        quantiles=quantiles,
        abs_thresh=1e-5,
    )

    if gif:
        generate_moving_density_gif(
            predictive_density,
            grid_x,
            grid_y,
            quantiles,
            plot_predictive_density_given_yt,
            y_start=0,
            y_end=len(grid_x) - 1,
            gif_path=f"{file_name}/{model}/{model}_moving_density.gif",
            total_duration=5.0,
        )

    mean_rmse = evaluate_predictive_density(
        predictive_density,
        grid_x,
        grid_y,
        theoretical_moments,
        quantiles,
        file_name=f"{file_name}/{model}/{model}.json",
    )

    if true_predictive_density is not None:
        logger.info("Computing density metrics (KL divergence and ISE)...")

        n_x = len(grid_x)
        flat_q = {
            str(p): v
            for cat in ["Lower Quantiles", "Upper Quantiles"]
            for p, v in quantiles[cat].items()
        }

        def q(prob, default_idx):
            """Return the quantile location or a reasonable default on grid_x."""
            return flat_q.get(str(prob), grid_x[int(default_idx * n_x)])

        grid_ranges = [
            {
                "name": "center",
                "idx": np.arange(
                    np.searchsorted(grid_x, q(0.1, 0.1)),
                    np.searchsorted(grid_x, q(0.9, 0.9)) + 1,
                ),
            },
            {
                "name": "tails",
                "idx": np.concatenate(
                    [
                        np.arange(
                            np.searchsorted(grid_x, q(0.01, 0.01)),
                            np.searchsorted(grid_x, q(0.1, 0.1)),
                        ),
                        np.arange(
                            np.searchsorted(grid_x, q(0.9, 0.9)),
                            np.searchsorted(grid_x, q(0.99, 0.99)) + 1,
                        ),
                    ]
                ),
            },
        ]

        density_metrics = {}
        for r in grid_ranges:
            rng_name, idx = r["name"], r["idx"]
            if idx.size == 0:
                density_metrics[f"{rng_name}_KL_divergence"] = None
                density_metrics[f"{rng_name}_ISE"] = None
                continue

            mean_KL, mean_ISE = compute_density_metrics(
                true_predictive_density, predictive_density, grid_y, idx=idx
            )
            density_metrics[f"{rng_name}_KL_divergence"] = float(mean_KL)
            density_metrics[f"{rng_name}_ISE"] = float(mean_ISE)

        # Whole-grid metrics; these are the "Total" column of the paper table
        mean_KL, mean_ISE = compute_density_metrics(
            true_predictive_density, predictive_density, grid_y
        )
        density_metrics["mean_KL_divergence"] = float(mean_KL)
        density_metrics["mean_ISE"] = float(mean_ISE)

        metrics_path = Path(f"{file_name}/{model}/{model}_density_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(density_metrics, f, indent=2)
        logger.info(f"Density metrics saved to {metrics_path}")
        for region in ("center", "tails"):
            logger.info(
                f"  {region.capitalize()} - "
                f"KL: {density_metrics[f'{region}_KL_divergence']:.6f}, "
                f"ISE: {density_metrics[f'{region}_ISE']:.6f}"
            )

    return mean_rmse
