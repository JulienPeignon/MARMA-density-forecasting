"""Center/tails/total MSPE for the Baumeister et al. (2025, JAE) benchmarks."""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat

from src.evaluate.evaluate_applications import mspe_by_region
from src.recursive_forecasting.blocks import ar_forecast, ar_lag_order, fit_ar_ols
from src.recursive_forecasting.vintage import (
    enumerate_origins,
    origin_features,
    origin_targets,
    vintage_logprice,
)

# Column index of each horizon in the fcast_*.mat matrices.
HORIZON_COLUMNS = {h: i for i, h in enumerate([1, 3, 6, 9, 12, 15, 18, 21, 24])}

UNIVARIATE_BENCHMARKS = ["AR(1)", "AR(AIC)", "Exp. Smoothing"]
BENCHMARK_FILES = {
    "AR(1)": ("fcast_ar1.mat", "fuar1"),
    "AR(AIC)": ("fcast_araic.mat", "fuaraic"),
    "Exp. Smoothing": ("fcast_expsmooth.mat", "fsmooth"),
    "BVAR(AIC)": ("fcast_econ_var6log_aic.mat", "feconbvar6log_aic"),
    "BVAR(1)": ("fcast_econ_var6log_p1.mat", "feconbvar6log_p1"),
}

FIRST_ORIGIN = "1997-01"
SAMPLE_START = "1976-01"

PUBLISHED_TOTAL_RATIOS = {
    1: {
        "AR(1)": 0.987,
        "AR(AIC)": 1.043,
        "Exp. Smoothing": 2.120,
        "BVAR(AIC)": 1.006,
        "BVAR(1)": 0.969,
    },
    3: {
        "AR(1)": 0.956,
        "AR(AIC)": 0.978,
        "Exp. Smoothing": 1.257,
        "BVAR(AIC)": 0.972,
        "BVAR(1)": 0.926,
    },
    6: {
        "AR(1)": 0.900,
        "AR(AIC)": 0.891,
        "Exp. Smoothing": 0.941,
        "BVAR(AIC)": 0.898,
        "BVAR(1)": 0.852,
    },
    9: {
        "AR(1)": 0.855,
        "AR(AIC)": 0.849,
        "Exp. Smoothing": 0.819,
        "BVAR(AIC)": 0.827,
        "BVAR(1)": 0.804,
    },
    12: {
        "AR(1)": 0.837,
        "AR(AIC)": 0.827,
        "Exp. Smoothing": 0.798,
        "BVAR(AIC)": 0.810,
        "BVAR(1)": 0.766,
    },
    15: {
        "AR(1)": 0.836,
        "AR(AIC)": 0.825,
        "Exp. Smoothing": 0.802,
        "BVAR(AIC)": 0.827,
        "BVAR(1)": 0.767,
    },
    18: {
        "AR(1)": 0.843,
        "AR(AIC)": 0.830,
        "Exp. Smoothing": 0.808,
        "BVAR(AIC)": 0.852,
        "BVAR(1)": 0.785,
    },
    21: {
        "AR(1)": 0.853,
        "AR(AIC)": 0.836,
        "Exp. Smoothing": 0.792,
        "BVAR(AIC)": 0.856,
        "BVAR(1)": 0.819,
    },
    24: {
        "AR(1)": 0.871,
        "AR(AIC)": 0.844,
        "Exp. Smoothing": 0.782,
        "BVAR(AIC)": 0.883,
        "BVAR(1)": 0.850,
    },
}


def benchmark_files_available(data_dir: str = "data") -> bool:
    """Check that every ``fcast_*.mat`` used by :func:`benchmark_mspe` exists."""
    return all(
        os.path.exists(os.path.join(data_dir, f)) for f, _ in BENCHMARK_FILES.values()
    )


def benchmark_forecasts(horizon: int, data_dir: str = "data"):
    """Level point forecasts of each benchmark at ``horizon``, NaN padding dropped."""
    if horizon not in HORIZON_COLUMNS:
        raise KeyError(
            f"horizon {horizon} is not stored in the BHLR forecasts "
            f"(available: {sorted(HORIZON_COLUMNS)})"
        )
    col = HORIZON_COLUMNS[horizon]
    out = {}
    for name, (fname, var) in BENCHMARK_FILES.items():
        series = np.asarray(loadmat(os.path.join(data_dir, fname))[var], dtype=float)
        # Trailing NaN padding is exactly the h-1 rows MATLAB appended.
        column = series[:, col]
        n_valid = series.shape[0] - (horizon - 1)
        if np.isnan(column[:n_valid]).any():
            raise ValueError(f"{name} h={horizon}: NaN inside the evaluation sample")
        out[name] = column[:n_valid]
    return out


def benchmark_mspe(
    horizons,
    data_dir: str = "data",
    tail_q: float = 0.90,
    first_origin: str = FIRST_ORIGIN,
    sample_start: str = SAMPLE_START,
):
    """Center/tails/total MSPE for every benchmark, by horizon.

    Returns ``{horizon: {benchmark: {region: scores}}}`` where ``scores`` is the
    :func:`mspe_by_region` payload (``mspe_point``,
    ``mspe_ratio_point``). Nothing here consults the published numbers -- use
    :func:`verify_published_totals` to confirm the reconstruction.
    """
    ng = np.loadtxt(os.path.join(data_dir, "NG_HENRY.txt"))
    cpi = np.loadtxt(os.path.join(data_dir, "CPI_AC.txt"))
    final = loadmat(os.path.join(data_dir, "HH_CPI_May2024vintage.mat"))

    out = {}
    for horizon in sorted(horizons):
        if horizon not in HORIZON_COLUMNS:
            continue
        origins = enumerate_origins(horizon, ng.shape[1], first_origin=first_origin)
        targets = origin_targets(
            final["NG_May24"], final["CPI_May24"], origins, log=False
        )
        naive = np.exp(
            origin_features(ng, cpi, origins, 1, sample_start=sample_start)[:, 0]
        )
        se_naive = (targets - naive) ** 2

        fcasts = benchmark_forecasts(horizon, data_dir)
        out[horizon] = {}
        for name, fcast in fcasts.items():
            if len(fcast) != len(origins):
                raise ValueError(
                    f"{name} h={horizon}: {len(fcast)} forecasts for "
                    f"{len(origins)} origins"
                )
            out[horizon][name] = mspe_by_region(
                {"point": (fcast - targets) ** 2}, se_naive, targets, tail_q
            )
    return out


PUBLISHED_TOLERANCE = 1e-3


def verify_published_totals(
    regions=None,
    data_dir: str = "data",
    tol: float = PUBLISHED_TOLERANCE,
    **kwargs,
):
    """Check reconstructed total MSPE ratios against :data:`PUBLISHED_TOTAL_RATIOS`.

    ``regions`` is the output of :func:`benchmark_mspe`; it is computed here when
    omitted. Returns ``(rows, all_ok)`` with one row per published cell, each
    holding ``horizon``, ``benchmark``, ``published``, ``reconstructed``,
    ``diff`` and ``ok``. Cells with no reconstruction (a horizon absent from the
    stored forecasts) get ``reconstructed=None`` and ``ok=False``.
    """
    if regions is None:
        regions = benchmark_mspe(
            sorted(PUBLISHED_TOTAL_RATIOS), data_dir=data_dir, **kwargs
        )

    rows = []
    for horizon in sorted(PUBLISHED_TOTAL_RATIOS):
        for name, published in PUBLISHED_TOTAL_RATIOS[horizon].items():
            scores = regions.get(horizon, {}).get(name)
            total = scores["total"]["mspe_ratio_point"] if scores else None
            diff = None if total is None else abs(total - published)
            rows.append(
                {
                    "horizon": horizon,
                    "benchmark": name,
                    "published": published,
                    "reconstructed": total,
                    "diff": diff,
                    "ok": diff is not None and diff <= tol,
                }
            )
    return rows, all(r["ok"] for r in rows)


# Replication check for the AR refits used as training DGPs.

AR_SPEC_FILES = {
    "ar1": ("fcast_ar1.mat", "fuar1"),
    "araic": ("fcast_araic.mat", "fuaraic"),
}
AR_REPLICATION_TOLERANCE = 1e-8


def refit_ar_forecasts(
    spec: str,
    horizon: int,
    data_dir: str = "data",
    first_origin: str = FIRST_ORIGIN,
    sample_start: str = SAMPLE_START,
):
    """Level forecasts of our AR refit at each origin, plus the lag order used.

    Reproduces one pass of ``main_ar1.m`` / ``main_araic.m``: fit on the origin's
    real-time vintage window, iterate ``horizon`` steps, exponentiate.
    """
    ng = np.loadtxt(os.path.join(data_dir, "NG_HENRY.txt"))
    cpi = np.loadtxt(os.path.join(data_dir, "CPI_AC.txt"))
    origins = enumerate_origins(horizon, ng.shape[1], first_origin=first_origin)

    fcasts, orders = np.empty(len(origins)), np.empty(len(origins), dtype=int)
    for i, o in enumerate(origins):
        y = vintage_logprice(ng, cpi, o, sample_start=sample_start, log=True)
        p = ar_lag_order(y, spec)
        const, coefs, _ = fit_ar_ols(y, p)
        fcasts[i] = np.exp(ar_forecast(const, coefs, y, horizon))
        orders[i] = p
    return fcasts, orders


def verify_ar_replication(
    horizons=None,
    data_dir: str = "data",
    tol: float = AR_REPLICATION_TOLERANCE,
    **kwargs,
):
    """Check our AR refits against the stored ``fcast_ar1`` / ``fcast_araic``.

    Returns ``(rows, all_ok)`` with one row per (spec, horizon), each holding the
    largest absolute and relative forecast deviation and the range of lag orders
    the AIC rule selected.
    """
    if horizons is None:
        horizons = sorted(HORIZON_COLUMNS)

    rows = []
    for spec in AR_SPEC_FILES:
        fname, var = AR_SPEC_FILES[spec]
        stored = np.asarray(loadmat(os.path.join(data_dir, fname))[var], dtype=float)
        for horizon in sorted(horizons):
            if horizon not in HORIZON_COLUMNS:
                continue
            column = stored[:, HORIZON_COLUMNS[horizon]]
            published = column[: stored.shape[0] - (horizon - 1)]
            mine, orders = refit_ar_forecasts(spec, horizon, data_dir, **kwargs)
            abs_diff = np.abs(mine - published)
            rel_diff = abs_diff / np.abs(published)
            rows.append(
                {
                    "spec": spec,
                    "horizon": horizon,
                    "n": len(mine),
                    "max_abs_diff": float(abs_diff.max()),
                    "max_rel_diff": float(rel_diff.max()),
                    "lag_orders": (int(orders.min()), int(orders.max())),
                    "ok": bool(rel_diff.max() <= tol),
                }
            )
    return rows, all(r["ok"] for r in rows)


def format_ar_replication_table(rows, tol: float = AR_REPLICATION_TOLERANCE) -> str:
    """Render :func:`verify_ar_replication` rows as a printable table."""
    lines = [
        f"{'spec':>6} {'h':>3} {'n':>5} {'max|diff|':>11} {'max rel':>10} {'p':>7}  ok",
        "-" * 58,
    ]
    for r in rows:
        lo, hi = r["lag_orders"]
        p = f"{lo}" if lo == hi else f"{lo}-{hi}"
        lines.append(
            f"{r['spec']:>6} {r['horizon']:>3} {r['n']:>5} {r['max_abs_diff']:>11.3e} "
            f"{r['max_rel_diff']:>10.3e} {p:>7}  {'ok' if r['ok'] else 'MISMATCH'}"
        )
    n_ok = sum(r["ok"] for r in rows)
    lines.append("-" * 58)
    lines.append(
        f"{n_ok}/{len(rows)} AR refits reproduce the stored BHLR forecasts "
        f"(relative tolerance {tol:g})"
    )
    return "\n".join(lines)


def format_verification_table(rows, tol: float = PUBLISHED_TOLERANCE) -> str:
    """Render :func:`verify_published_totals` rows as a printable table."""
    lines = [
        f"{'h':>3} {'benchmark':>15} {'published':>10} {'rebuilt':>10} {'diff':>9}  ok",
        "-" * 56,
    ]
    for r in rows:
        rebuilt = "--" if r["reconstructed"] is None else f"{r['reconstructed']:.4f}"
        diff = "--" if r["diff"] is None else f"{r['diff']:.2e}"
        lines.append(
            f"{r['horizon']:>3} {r['benchmark']:>15} {r['published']:>10.3f} "
            f"{rebuilt:>10} {diff:>9}  {'ok' if r['ok'] else 'MISMATCH'}"
        )
    n_ok = sum(r["ok"] for r in rows)
    worst = max((r["diff"] for r in rows if r["diff"] is not None), default=None)
    lines.append("-" * 56)
    lines.append(
        f"{n_ok}/{len(rows)} totals match the published table "
        f"(tolerance {tol:g}"
        + (f", largest deviation {worst:.2e}" if worst is not None else "")
        + ")"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if not benchmark_files_available():
        sys.exit("Benchmark forecasts missing from data/; nothing to verify.")
    verification_rows, ok = verify_published_totals()
    print(format_verification_table(verification_rows))
    print()
    ar_rows, ar_ok = verify_ar_replication()
    print(format_ar_replication_table(ar_rows))
    sys.exit(0 if (ok and ar_ok) else 1)
