"""Parquet I/O for predictive densities and recalibration artifacts."""

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def density_path(file_name, model, suffix=""):
    """Path of a model's density-shaped parquet."""
    return f"{file_name}/{model}/{model}{suffix}.parquet"


def save_predictive_density(array, file_name, model, suffix=""):
    """Write ``array`` to Parquet under ``file_name``/``model``."""
    df = pd.DataFrame(array)
    os.makedirs(file_name + "/" + model, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        density_path(file_name, model, suffix),
        compression="zstd",
        compression_level=22,
    )


def save_recalibration_cache(artifacts, file_name, model):
    """Persist what a later ``--evaluate`` run needs to refit the recalibrator."""
    if not artifacts:
        return False
    save_predictive_density(artifacts["raw"], file_name, model, "_raw")
    save_predictive_density(artifacts["X_target"], file_name, model, "_target_features")
    cal = pd.DataFrame(artifacts["X_calibration"])
    cal.columns = [f"x_{i}" for i in range(cal.shape[1])]
    cal.insert(0, "pit", artifacts["pit"])
    save_predictive_density(cal, file_name, model, "_calibration")
    return True


def load_recalibration_cache(file_name, model):
    """``(raw, X_target, X_calibration, pit)`` when all three files are on disk."""
    paths = [
        density_path(file_name, model, s)
        for s in ("_raw", "_target_features", "_calibration")
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    raw = load_predictive_density(file_name, model, "_raw")
    x_target = load_predictive_density(file_name, model, "_target_features")
    cal = pd.read_parquet(paths[2])
    return raw, x_target, cal.drop(columns="pit").to_numpy(), cal["pit"].to_numpy()


def load_predictive_density(file_name, model, suffix=""):
    """Read back a density written by :func:`save_predictive_density`."""
    df = pd.read_parquet(density_path(file_name, model, suffix))
    return df.to_numpy()
