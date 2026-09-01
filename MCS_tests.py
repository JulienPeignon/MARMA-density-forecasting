"""Model Confidence Set p-values for every cached forecast run.

python MCS_tests.py applications
python MCS_tests.py simulations                      # paper process/horizon set
python MCS_tests.py simulations --order 0 1 --horizon 1
python MCS_tests.py realized-outcomes
python MCS_tests.py all
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.conditional_theoretical_moments.theoretical_moments import (
    compute_theoretical_moments,
)
from src.conditional_theoretical_moments.theoretical_quantiles import (
    compute_theoretical_quantiles,
)
from src.evaluate.baumeister_et_al_benchmarks import (
    FIRST_ORIGIN,
    HORIZON_COLUMNS,
    PUBLISHED_TOTAL_RATIOS,
    UNIVARIATE_BENCHMARKS,
    benchmark_files_available,
    benchmark_forecasts,
)
from src.evaluate.cauchy_closed_form import cauchy_ar1_predictive_density
from src.forecast_methods.utils import make_grid
from src.mcs.losses import (
    density_losses,
    ise_per_obs,
    kl_per_obs,
    moment_region_indices,
    moment_squared_errors,
)
from src.mcs.runner import DEFAULT_ALPHA, DEFAULT_N_BOOT, DEFAULT_SEED, run_mcs
from src.recursive_forecasting.levels import model_rel_path

# Model families
APPLICATION_FAMILIES = {
    "density_models": ["kcde", "lls2012", "gj2026", "flexzboost", "mdn"],
    "training_dgp_models": ["ar1", "araic", "exp_smoothing", "mdn"],
}
# Metrics displayed in the two application density-comparison tables.
APPLICATION_QS_ALPHAS = (0.90, 0.95)
APPLICATION_TABLE_METRICS = {
    "cde_loss",
    "crps",
    "log_prob",
    "twcrps_90",
    "csl_90",
    "qs_90",
    "qs_95",
}
PAPER_PROCESS_ORDERS = (
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 1, 1, 1),
)
PAPER_SIMULATION_HORIZONS = (1, 2, 5)
PAPER_APPLICATION_HORIZONS = (1, 3, 6, 9, 12, 15, 18, 21, 24)
BENCHMARK_FAMILY = "mspe_benchmarks"

BENCHMARK_MDN_KEY = "MDN"
BENCHMARK_POINT_STAT = "median"
BENCHMARK_METRIC = f"mspe_{BENCHMARK_POINT_STAT}"

SIMULATION_MODELS = ["kcde", "lls2012", "gj2026", "FlexZBoost", "mdn"]

REALISED_REGIONS = ("total", "center", "tails")

DENSITY_METRIC_REGIONS = {
    "Total": "total",
    "Between 0.1-0.9": "center",
    "Between 0.01-0.1 and 0.9-0.99": "tails",
}
APPLICATION_TAIL_Q = 0.90
SAMPLE_START = "1976-01"
APPLICATION_WEIGHT_LEVEL = 0.90

METRIC_NOTES = {
    "cde_loss": "mean == cde_loss",
    "crps": "mean == crps",
    "log_prob": "loss is -log p(y); mean == -log_prob",
    "qs_05": "mean == qs_05",
    "qs_10": "mean == qs_10",
    "qs_90": "mean == qs_90",
    "qs_95": "mean == qs_95",
    "mspe_median": "squared error of the conditional median; mean == mspe_median",
    "Moment k": "squared moment error; sqrt of the mean == the reported RMSE",
    "KL divergence": "mean == *_KL_divergence",
    "ISE": "mean == *_ISE",
}


# Shared helpers
def load_config(path="config.yaml"):
    """Load the YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_process_dir(name):
    """``"MAR(0, 1)"`` -> ``("MAR", (0, 1))``; ``None`` when the name is not one."""
    match = re.fullmatch(r"(MARMA|MAR)\((.*)\)", name)
    if match is None:
        return None
    try:
        order = tuple(int(part) for part in match.group(2).split(","))
    except ValueError:
        return None
    return match.group(1), order


def discover(path, prefix, cast=int):
    """Sorted suffixes of the ``prefix_*`` sub-directories of ``path``."""
    path = Path(path)
    if not path.exists():
        return []
    found = []
    for item in path.iterdir():
        if item.is_dir() and item.name.startswith(f"{prefix}_"):
            try:
                found.append(cast(item.name[len(prefix) + 1 :]))
            except ValueError:
                continue
    return sorted(found)


def read_json(path):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def write_json(payload, path):
    """Write ``payload`` as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {path}")


def base_meta(source, schema, settings):
    """Build the provenance header of an ``mcs_tests.json``."""
    return {
        "source": source,
        "schema": schema,
        "produced_by": "MCS_tests.py",
        "refitted": False,
        "mcs_alpha": settings["alpha"],
        "n_boot": settings["n_boot"],
        "seed": settings["seed"],
        "bootstrap": "stationary (numba RNG seeded per test for reproducibility)",
        "membership_rule": f"model is in the MCS when pvalue >= {settings['alpha']}",
        "metrics": METRIC_NOTES,
        "verification": {
            "note": (
                "mean of each rebuilt per-observation loss compared with the "
                "aggregate the producer saved; only mismatches are listed. "
                "'published' checks compare against the three-decimal ratios "
                "printed in Baumeister et al. (2025), so their residual is "
                "rounding, not reproduction error"
            ),
            "n_checks": 0,
            "max_abs_diff": 0.0,
            "n_published_checks": 0,
            "max_abs_diff_published": 0.0,
            "mismatches": [],
        },
    }


def record_check(meta, label, rebuilt, reported, tol=1e-8):
    """Compare a rebuilt loss mean with the producer's aggregate.

    ``tol`` above the default marks a comparison against externally rounded
    figures; those are tallied separately so they cannot be mistaken for our
    own reproduction accuracy.
    """
    if reported is None or rebuilt is None or not np.isfinite(rebuilt):
        return
    diff = abs(float(rebuilt) - float(reported))
    check = meta["verification"]
    if tol > 1e-8:
        check["n_published_checks"] += 1
        check["max_abs_diff_published"] = max(check["max_abs_diff_published"], diff)
    else:
        check["n_checks"] += 1
        check["max_abs_diff"] = max(check["max_abs_diff"], diff)
    if diff / max(1.0, abs(float(reported))) > tol:
        check["mismatches"].append(
            {
                "item": label,
                "rebuilt": float(rebuilt),
                "reported": float(reported),
                "abs_diff": diff,
            }
        )
        print(
            f"  ! {label}: rebuilt {rebuilt:.10g} vs reported {reported:.10g} "
            f"(|diff| {diff:.3g})"
        )


def summarise_checks(meta):
    """Print how many rebuilt losses matched the values the producer saved."""
    check = meta["verification"]
    n_bad = len(check["mismatches"])
    status = "all match" if not n_bad else f"{n_bad} MISMATCH"
    extra = ""
    if check["n_published_checks"]:
        extra = (
            f" + {check['n_published_checks']} vs published "
            f"(max |diff| {check['max_abs_diff_published']:.3g})"
        )
    print(
        f"  verification: {check['n_checks']} checks, {status} "
        f"(max |diff| {check['max_abs_diff']:.3g}){extra}"
    )


def mcs_over_models(loss_by_model, settings):
    """Run one MCS over ``{model: loss vector}``; ``{}`` when nothing is available."""
    keys = [key for key, loss in loss_by_model.items() if loss is not None]
    if not keys:
        return {}
    losses = np.column_stack(
        [np.asarray(loss_by_model[key], dtype=float) for key in keys]
    )
    return run_mcs(
        losses,
        keys,
        n_boot=settings["n_boot"],
        alpha=settings["alpha"],
        seed=settings["seed"],
        block_len=settings["block_len"],
    )


# Applications
def application_model_dir(base_path, horizon, model):
    """Return the directory holding one application model's cached run."""
    return Path(base_path) / f"horizon_{horizon}" / model_rel_path(model)


def load_application_model(base_path, horizon, model, weight_thresholds=None):
    """Per-observation losses for one applications model, or ``None`` if absent."""
    model_dir = application_model_dir(base_path, horizon, model)
    density_file = model_dir / f"{model}_densities_level.parquet"
    point_file = model_dir / f"{model}_point_forecasts.parquet"
    meta_file = model_dir / f"{model}_run_meta.json"
    if not (density_file.exists() and point_file.exists() and meta_file.exists()):
        return None

    run_meta = read_json(meta_file)
    density = pd.read_parquet(density_file).to_numpy()
    points = pd.read_parquet(point_file)
    lo, hi = run_meta["level_grid"]
    level_grid = np.linspace(lo, hi, run_meta["n_points_grid"])
    y_true = points["target"].to_numpy()

    log_density_file = model_dir / f"{model}_densities_log.parquet"
    density_log, log_grid = None, None
    if log_density_file.exists() and "log_grid" in run_meta:
        density_log = pd.read_parquet(log_density_file).to_numpy()
        lo_x, hi_x = run_meta["log_grid"]
        log_grid = np.linspace(lo_x, hi_x, run_meta["n_points_grid"])

    losses = density_losses(
        density,
        level_grid,
        y_true,
        density_log,
        log_grid,
        weight_thresholds=weight_thresholds,
        quantile_levels=APPLICATION_QS_ALPHAS,
    )
    losses["mspe_median"] = points["se_median"].to_numpy()

    tails = (points["region"] == "tails").to_numpy()
    regions = {
        "total": np.ones(len(y_true), dtype=bool),
        "center": ~tails,
        "tails": tails,
    }

    reported = {}
    metrics_file = model_dir / f"{model}_cde_metrics.json"
    if metrics_file.exists():
        reported.update(read_json(metrics_file))
    point_metrics_file = model_dir / f"{model}_point_metrics.json"
    if point_metrics_file.exists():
        reported.update(read_json(point_metrics_file))

    return {
        "losses": losses,
        "regions": regions,
        "reported": reported,
        "y_true": y_true,
        "se_naive": points["se_naive"].to_numpy(),
    }


def benchmark_targets(horizon, data_dir="data"):
    """Targets the benchmark forecasts are aligned on, rebuilt from the vintages."""
    from scipy.io import loadmat

    from src.recursive_forecasting.vintage import enumerate_origins, origin_targets

    ng = np.loadtxt(os.path.join(data_dir, "NG_HENRY.txt"))
    final = loadmat(os.path.join(data_dir, "HH_CPI_May2024vintage.mat"))
    origins = enumerate_origins(horizon, ng.shape[1], first_origin=FIRST_ORIGIN)
    return origin_targets(final["NG_May24"], final["CPI_May24"], origins, log=False)


def origin_thresholds(horizon, level=0.90, data_dir="data", sample_start=SAMPLE_START):
    """Per-origin ``level`` quantile of the real price, in $/MMBtu.

    Mirrors ``applications.origin_price_quantile``: each origin reads its own
    vintage, so the weighted region is known when the forecast is made.
    """
    from src.recursive_forecasting.vintage import _col, _row, enumerate_origins

    ng = np.loadtxt(os.path.join(data_dir, "NG_HENRY.txt"))
    cpi = np.loadtxt(os.path.join(data_dir, "CPI_AC.txt"))
    origins = enumerate_origins(horizon, ng.shape[1], first_origin=FIRST_ORIGIN)
    row0 = _row(sample_start)
    out = np.empty(len(origins), dtype=float)
    for i, origin in enumerate(origins):
        col = _col(origin.origin_month)
        rows = slice(row0, _row(origin.origin_month) + 1)
        real = 100.0 * ng[rows, col] / cpi[rows, col]
        real = real[np.isfinite(real) & (real > 0)]
        out[i] = np.quantile(real, level)
    return out


def load_benchmark_losses(horizon, y_true, data_dir="data"):
    """Squared errors of the univariate BHLR benchmarks, or ``None`` if unavailable.

    The benchmarks are point forecasts in levels, so their loss is directly
    comparable with our models' ``se_median``. Alignment is not assumed: the
    targets the ``.mat`` forecasts belong to are rebuilt and required to match
    ours bit for bit, since a silent off-by-one would invalidate the comparison.
    """
    if horizon not in HORIZON_COLUMNS or not benchmark_files_available(data_dir):
        return None

    targets = benchmark_targets(horizon, data_dir)
    if targets.shape != y_true.shape or not np.array_equal(targets, y_true):
        raise ValueError(
            f"h={horizon}: the BHLR benchmarks are aligned on different targets "
            f"than our models; refusing to compare them"
        )

    forecasts = benchmark_forecasts(horizon, data_dir)
    losses = {}
    for name in UNIVARIATE_BENCHMARKS:
        forecast = np.asarray(forecasts[name], dtype=float)
        if len(forecast) != len(y_true):
            raise ValueError(
                f"{name} h={horizon}: {len(forecast)} forecasts for "
                f"{len(y_true)} targets"
            )
        losses[name] = (forecast - y_true) ** 2
    return losses


def check_aligned(loaded, models, label):
    """Every model must score the same targets, or the losses are not comparable."""
    reference = loaded[models[0]]["y_true"]
    for model in models[1:]:
        target = loaded[model]["y_true"]
        if target.shape != reference.shape or not np.array_equal(target, reference):
            raise ValueError(
                f"{label}: {model} is evaluated on different targets than "
                f"{models[0]}; the loss series cannot be compared"
            )


def reported_application_value(reported, metric, region="total"):
    """Return the producer's aggregate for ``metric`` in ``region``, if saved."""
    if metric == "mspe_median":
        key = metric if region == "total" else f"{region}_{metric}"
        return reported.get(key)
    if metric == "log_prob":
        key = "log_prob" if region == "total" else f"{region}_log_prob"
        value = reported.get(key)
        return None if value is None else -value  # rebuilt loss is the negation
    if region == "total":
        return reported.get(metric)
    if metric in ("cde_loss", "crps"):
        return reported.get(f"{region}_{metric}")
    return None  # quantile scores are only reported for the full sample


def run_benchmark_family(mdn, horizon, meta, settings):
    """MCS of the univariate BHLR benchmarks against our MDN, on point-forecast MSPE.

    ``mdn`` is the loaded MDN entry the MSPE table displays.
    """
    losses = load_benchmark_losses(horizon, mdn["y_true"])
    if not losses:
        print(f"  {BENCHMARK_FAMILY}: no benchmark forecasts for horizon {horizon}")
        return None

    losses[BENCHMARK_MDN_KEY] = mdn["losses"][BENCHMARK_METRIC]
    published = PUBLISHED_TOTAL_RATIOS.get(horizon, {})
    result = {BENCHMARK_METRIC: {}}
    for region in REALISED_REGIONS:
        mask = mdn["regions"][region]
        if not mask.any():
            continue
        if region == "total":
            naive = float(np.mean(mdn["se_naive"][mask]))
            for name in UNIVARIATE_BENCHMARKS:
                record_check(
                    meta,
                    f"h{horizon}/{name}/mspe_ratio/total (published)",
                    float(np.mean(losses[name][mask])) / naive,
                    published.get(name),
                    tol=1e-3,  # the paper reports three decimals
                )
        result[BENCHMARK_METRIC][region] = mcs_over_models(
            {name: loss[mask] for name, loss in losses.items()}, settings
        )
    print(f"  {BENCHMARK_FAMILY}: {len(losses)} models x 1 metric")
    return result


def run_mcs_applications(base_path, horizons, settings):
    """MCS p-values for the three applications model families, all horizons."""
    base_path = Path(base_path)
    horizons = horizons or discover(base_path, "horizon")
    if not horizons:
        print(f"No horizons found under {base_path}")
        return

    meta = base_meta(
        "applications.py",
        "family -> horizon -> metric -> model -> pvalue "
        f"({BENCHMARK_FAMILY} keeps a region level)",
        settings,
    )
    meta["weight_thresholds"] = (
        f"per-origin {APPLICATION_WEIGHT_LEVEL:.0%} quantile of the real price on "
        f"each origin's own vintage"
    )
    meta["tail_q"] = APPLICATION_TAIL_Q
    meta["regions"] = {
        "total": "all origins",
        "center": f"target below its {APPLICATION_TAIL_Q:.0%} empirical quantile",
        "tails": f"target at or above its {APPLICATION_TAIL_Q:.0%} empirical quantile",
    }
    meta["families"] = {
        **{
            name: {"models": models, "metrics": "all"}
            for name, models in APPLICATION_FAMILIES.items()
        },
        BENCHMARK_FAMILY: {
            "models": UNIVARIATE_BENCHMARKS + [BENCHMARK_MDN_KEY],
            "metrics": [BENCHMARK_METRIC],
            "note": (
                "univariate Baumeister et al. (2025) point forecasts read from "
                "the published .mat files against our MDN, on the same origins "
                "and targets; p-values are unchanged by the no-change rescaling "
                "the MSPE table displays"
            ),
        },
    }
    payload = {family: {} for family in (*APPLICATION_FAMILIES, BENCHMARK_FAMILY)}

    for horizon in horizons:
        print(f"\napplications | horizon {horizon}")
        weight_thresholds = {
            APPLICATION_WEIGHT_LEVEL: origin_thresholds(
                horizon, APPLICATION_WEIGHT_LEVEL
            )
        }
        cache = {}

        def fetch(model, horizon=horizon, cache=cache):
            if model not in cache:
                cache[model] = load_application_model(
                    base_path, horizon, model, weight_thresholds
                )
            return cache[model]

        for family, models in APPLICATION_FAMILIES.items():
            loaded = {}
            for model in models:
                entry = fetch(model)
                if entry is None:
                    print(f"  missing: {model}")
                    continue
                loaded[model] = entry

            available = [model for model in models if model in loaded]
            if len(available) < 2:
                print(f"  {family}: {len(available)} model(s) available, skipped")
                continue
            check_aligned(loaded, available, f"applications h={horizon} {family}")

            metrics = sorted(
                APPLICATION_TABLE_METRICS.intersection(
                    set.intersection(*(set(loaded[m]["losses"]) for m in available))
                )
            )
            family_result = {}
            for metric in metrics:
                loss_by_model = {}
                for model in available:
                    loss = loaded[model]["losses"][metric]
                    loss_by_model[model] = loss
                    record_check(
                        meta,
                        f"h{horizon}/{model}/{metric}",
                        float(np.mean(loss)),
                        reported_application_value(loaded[model]["reported"], metric),
                    )
                family_result[metric] = mcs_over_models(loss_by_model, settings)
            payload[family][str(horizon)] = family_result
            print(f"  {family}: {len(available)} models x {len(metrics)} metrics")

        mdn = fetch("mdn")
        if not mdn:
            print(f"  {BENCHMARK_FAMILY}: no MDN losses for horizon {horizon}")
        benchmark_result = (
            run_benchmark_family(mdn, horizon, meta, settings) if mdn else None
        )
        if benchmark_result is not None:
            payload[BENCHMARK_FAMILY][str(horizon)] = benchmark_result

    summarise_checks(meta)
    write_json({"_meta": meta, **payload}, base_path / "mcs_tests.json")


# Simulations
def process_parameters(cfg, order):
    """Return the generating parameters for ``order``, as the scripts read them."""
    mar_params = {
        tuple(int(v) for v in key.strip("()").split(",")): value
        for key, value in cfg["mar_params"].items()
    }
    params = {**mar_params[order], **cfg["alpha_stable_params"]}
    return {
        key: params.get(key) for key in ("PSI", "PHI", "BETA", "SIGMA", "THETA", "ETA")
    }


def theoretical_grids(params, alpha, n_points_grid):
    """Conditioning/evaluation grids and quantiles, as built by simulations.py."""
    psi = params["PSI"]
    quantiles = compute_theoretical_quantiles(
        phi_vec=[params["PHI"]],
        psi_vec=[psi] if isinstance(psi, float) else psi,
        alpha=alpha,
        beta=params["BETA"],
        sigma=params["SIGMA"],
        theta=params["THETA"],
        eta=params["ETA"],
    )
    grid_x, grid_y = make_grid(quantiles, n_points_grid)
    return quantiles, grid_x, grid_y


def run_mcs_simulations_case(base_path, model_type, order, horizon, cfg, settings):
    """MCS p-values for one process/horizon, every alpha on disk."""
    process_dir = Path(base_path) / f"{model_type}{order}"
    horizon_dir = process_dir / f"horizon_{horizon}"
    alphas = discover(horizon_dir, "alpha", cast=float)
    if not alphas:
        return

    params = process_parameters(cfg, order)
    n_points_grid = cfg["n_points_grid"]

    meta = base_meta(
        "simulations.py",
        "alpha -> region -> metric -> model -> pvalue "
        "(region/metric order kept for the published tables)",
        settings,
    )
    meta["process"] = f"{model_type}{order}"
    meta["horizon"] = horizon
    meta["regions"] = "conditioning-grid slices of the theoretical quantiles of y_t"
    payload = {}

    for alpha in alphas:
        alpha_dir = horizon_dir / f"alpha_{alpha}"
        quantiles, grid_x, grid_y = theoretical_grids(params, alpha, n_points_grid)
        theoretical = compute_theoretical_moments(
            grid=grid_x,
            h=horizon,
            phi_vec=[params["PHI"]],
            psi_vec=[params["PSI"]]
            if isinstance(params["PSI"], float)
            else params["PSI"],
            alpha=alpha,
            beta=params["BETA"],
            sigma=params["SIGMA"],
            theta=params["THETA"],
            eta=params["ETA"],
            ma_trunc=100,
        )
        n_moments = theoretical.shape[1]

        true_density = None
        if order == (0, 1) and alpha == 1.0:
            true_density = cauchy_ar1_predictive_density(
                grid_x, grid_y, params["PSI"], params["SIGMA"], horizon
            )

        squared, kl, ise, reported, reported_density = {}, {}, {}, {}, {}
        for model in SIMULATION_MODELS:
            density_file = alpha_dir / model / f"{model}.parquet"
            if not density_file.exists():
                continue
            density = pd.read_parquet(density_file).to_numpy()
            squared[model] = moment_squared_errors(density, grid_y, theoretical)
            if true_density is not None:
                kl[model] = kl_per_obs(true_density, density, grid_y)
                ise[model] = ise_per_obs(true_density, density, grid_y)
            rmse_file = alpha_dir / model / f"{model}.json"
            if rmse_file.exists():
                reported[model] = read_json(rmse_file)
            density_file = alpha_dir / model / f"{model}_density_metrics.json"
            if density_file.exists():
                reported_density[model] = read_json(density_file)

        if len(squared) < 2:
            print(f"  alpha {alpha}: {len(squared)} model(s) available, skipped")
            continue

        regions = moment_region_indices(grid_x, quantiles)
        alpha_result = {}
        for region_name, idx in regions.items():
            alpha_result[region_name] = {}
            for k in range(n_moments):
                loss_by_model = {}
                for model, errors in squared.items():
                    loss = errors[idx, k]
                    loss_by_model[model] = loss
                    record_check(
                        meta,
                        f"alpha{alpha}/{model}/{region_name}/Moment {k + 1}",
                        float(np.sqrt(np.mean(loss))),
                        reported.get(model, {})
                        .get(region_name, {})
                        .get(f"Moment {k + 1}"),
                    )
                alpha_result[region_name][f"Moment {k + 1}"] = mcs_over_models(
                    loss_by_model, settings
                )
            for label, per_model, suffix in (
                ("KL divergence", kl, "KL_divergence"),
                ("ISE", ise, "ISE"),
            ):
                if not per_model:
                    continue
                for model, values in per_model.items():
                    record_check(
                        meta,
                        f"alpha{alpha}/{model}/{region_name}/{label}",
                        float(np.mean(values[idx])),
                        reported_density.get(model, {}).get(
                            f"{DENSITY_METRIC_REGIONS[region_name]}_{suffix}"
                        ),
                    )
                alpha_result[region_name][label] = mcs_over_models(
                    {model: values[idx] for model, values in per_model.items()},
                    settings,
                )

        payload[str(alpha)] = alpha_result
        extra = " + KL/ISE" if kl else ""
        print(
            f"  alpha {alpha}: {len(squared)} models x {n_moments} moments{extra} "
            f"x 3 regions"
        )

    if not payload:
        return
    summarise_checks(meta)
    write_json({"_meta": meta, **payload}, horizon_dir / "mcs_tests.json")


def run_mcs_simulations(base_path, orders, horizons, cfg, settings):
    """Run the MCS over the simulated moment and density metrics."""
    for model_type, order, horizon in iter_simulation_cases(
        base_path, orders, horizons
    ):
        print(f"\nsimulations | {model_type}{order} horizon {horizon}")
        run_mcs_simulations_case(base_path, model_type, order, horizon, cfg, settings)


def iter_simulation_cases(base_path, orders, horizons):
    """Every ``(model_type, order, horizon)`` present under ``base_path``."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"No such directory: {base_path}")
        return
    for process_dir in sorted(base_path.iterdir()):
        parsed = parse_process_dir(process_dir.name)
        if parsed is None:
            continue
        model_type, order = parsed
        if orders and order not in orders:
            continue
        for horizon in discover(process_dir, "horizon"):
            if horizons and horizon not in horizons:
                continue
            yield model_type, order, horizon


# Simulated realized outcomes
def load_realised_model(alpha_dir, model, weight_thresholds=None):
    """Per-observation losses for one realized-outcome run, or ``None``."""
    model_dir = alpha_dir / model
    density_file = model_dir / f"{model}_densities.parquet"
    target_file = model_dir / f"{model}_realized_targets.parquet"
    meta_file = model_dir / f"{model}_run_meta.json"
    if not (density_file.exists() and target_file.exists() and meta_file.exists()):
        return None

    run_meta = read_json(meta_file)
    density = pd.read_parquet(density_file).to_numpy()
    y_true = pd.read_parquet(target_file)["y_true"].to_numpy()
    lo, hi = run_meta["grid_y"]
    grid_y = np.linspace(lo, hi, run_meta["n_points_grid"])

    metrics_file = model_dir / f"{model}_cde_metrics.json"
    return {
        "losses": density_losses(
            density, grid_y, y_true, weight_thresholds=weight_thresholds
        ),
        "y_true": y_true,
        "reported": read_json(metrics_file) if metrics_file.exists() else {},
    }


def reported_realised_value(reported, metric):
    """Return the producer's aggregate for ``metric`` on the whole sample."""
    if metric == "log_prob":
        value = reported.get("log_prob")
        return None if value is None else -value
    return reported.get(metric)


def run_mcs_realised_case(base_path, model_type, order, horizon, cfg, settings):
    """MCS p-values for one realized-outcome process/horizon, every alpha on disk."""
    horizon_dir = Path(base_path) / f"{model_type}{order}" / f"horizon_{horizon}"
    alphas = discover(horizon_dir, "alpha", cast=float)
    if not alphas:
        return

    params = process_parameters(cfg, order)
    meta = base_meta(
        "simulations_realized_outcomes.py",
        "alpha -> metric -> model -> pvalue",
        settings,
    )
    meta["process"] = f"{model_type}{order}"
    meta["horizon"] = horizon
    meta["weight_thresholds"] = (
        "theoretical marginal quantiles at 5/10/90/95%, the thresholds of the "
        "twCRPS and CSL weight functions"
    )
    payload = {}

    for alpha in alphas:
        alpha_dir = horizon_dir / f"alpha_{alpha}"

        quantiles = compute_theoretical_quantiles(
            phi_vec=[params["PHI"]],
            psi_vec=[params["PSI"]]
            if isinstance(params["PSI"], float)
            else params["PSI"],
            alpha=alpha,
            beta=params["BETA"],
            sigma=params["SIGMA"],
            theta=params["THETA"],
            eta=params["ETA"],
        )
        flat_q = {
            prob: value
            for category in ("Lower Quantiles", "Upper Quantiles")
            for prob, value in quantiles[category].items()
        }
        weight_thresholds = {
            level: flat_q[level]
            for level in (0.05, 0.10, 0.90, 0.95)
            if level in flat_q
        }

        loaded = {}
        for model in SIMULATION_MODELS:
            entry = load_realised_model(alpha_dir, model, weight_thresholds)
            if entry is not None:
                loaded[model] = entry
        if len(loaded) < 2:
            print(
                f"  alpha {alpha}: {len(loaded)} model(s) with cached densities, "
                f"skipped"
            )
            continue
        check_aligned(loaded, list(loaded), f"realized outcomes alpha={alpha}")

        metrics = sorted(set.intersection(*(set(e["losses"]) for e in loaded.values())))

        alpha_result = {}
        for metric in metrics:
            loss_by_model = {}
            for model, entry in loaded.items():
                loss = entry["losses"][metric]
                loss_by_model[model] = loss
                record_check(
                    meta,
                    f"alpha{alpha}/{model}/{metric}",
                    float(np.mean(loss)),
                    reported_realised_value(entry["reported"], metric),
                )
            alpha_result[metric] = mcs_over_models(loss_by_model, settings)
        payload[str(alpha)] = alpha_result
        print(f"  alpha {alpha}: {len(loaded)} models x {len(metrics)} metrics")

    if not payload:
        return
    summarise_checks(meta)
    write_json({"_meta": meta, **payload}, horizon_dir / "mcs_tests.json")


def run_mcs_realised(base_path, orders, horizons, cfg, settings):
    """Run the MCS over the realised-outcome scores."""
    for model_type, order, horizon in iter_simulation_cases(
        base_path, orders, horizons
    ):
        print(f"\nrealized outcomes | {model_type}{order} horizon {horizon}")
        run_mcs_realised_case(base_path, model_type, order, horizon, cfg, settings)


# CLI
def build_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "scope",
        choices=["applications", "simulations", "realized-outcomes", "all"],
        help="Which cached results to test.",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        action="append",
        default=None,
        help="Restrict to a process order, repeatable (e.g. --order 0 1 --order 1 1).",
    )
    parser.add_argument(
        "--horizon",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to these horizons (default: every horizon on disk).",
    )
    parser.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="MCS significance level; membership is pvalue >= alpha.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed."
    )
    parser.add_argument(
        "--block_len",
        type=int,
        default=None,
        help="Stationary-bootstrap block length (default: sqrt of the sample size).",
    )
    return parser


def main():
    """Run the requested Model Confidence Set families."""
    args = build_parser().parse_args()
    cfg = load_config()
    settings = {
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "seed": args.seed,
        "block_len": args.block_len,
    }
    orders = (
        [tuple(order) for order in args.order]
        if args.order
        else list(PAPER_PROCESS_ORDERS)
    )
    simulation_horizons = args.horizon or list(PAPER_SIMULATION_HORIZONS)
    application_horizons = args.horizon or list(PAPER_APPLICATION_HORIZONS)

    if args.scope in ("applications", "all"):
        run_mcs_applications("outputs/applications", application_horizons, settings)
    if args.scope in ("simulations", "all"):
        run_mcs_simulations(
            "outputs/simulations", orders, simulation_horizons, cfg, settings
        )
    if args.scope in ("realized-outcomes", "all"):
        run_mcs_realised(
            "outputs/simulations_realized_outcomes",
            orders,
            simulation_horizons,
            cfg,
            settings,
        )


if __name__ == "__main__":
    main()
