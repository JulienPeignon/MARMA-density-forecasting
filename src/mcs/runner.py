"""Seeded wrapper around :class:`model_confidence_set.ModelConfidenceSet`."""

import numpy as np
import pandas as pd
from model_confidence_set import ModelConfidenceSet
from numba import njit

DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.25
DEFAULT_SEED = 1


@njit
def _seed_numba_rng(seed):
    """Seed the RNG the jitted bootstrap actually draws from."""
    np.random.seed(seed)


def run_mcs(
    losses,
    model_keys,
    n_boot=DEFAULT_N_BOOT,
    alpha=DEFAULT_ALPHA,
    seed=DEFAULT_SEED,
    block_len=None,
):
    """MCS p-values for an ``(n_obs, n_models)`` loss matrix.

    Returns ``{model_key: pvalue}``. A model belongs to the MCS at confidence
    ``1 - alpha`` when its p-value is at least ``alpha``. A single surviving
    model is trivially the whole set, so it is given a p-value of 1.
    """
    losses = np.asarray(losses, dtype=float)
    model_keys = list(model_keys)
    if losses.ndim != 2 or losses.shape[1] != len(model_keys):
        raise ValueError(
            f"losses of shape {losses.shape} do not match {len(model_keys)} models"
        )
    if not np.isfinite(losses).all():
        bad = [
            key
            for key, column in zip(model_keys, losses.T)
            if not np.isfinite(column).all()
        ]
        raise ValueError(f"non-finite losses for {bad}; refusing to bootstrap")
    if len(model_keys) == 1:
        return {model_keys[0]: 1.0}

    _seed_numba_rng(int(seed))
    mcs = ModelConfidenceSet(
        pd.DataFrame(losses, columns=model_keys),
        n_boot=int(n_boot),
        alpha=float(alpha),
        block_len=block_len,
        show_progress=False,
    )
    mcs.compute()
    pvalues = mcs.results()["pvalues"]
    return {str(key): float(pvalues[key]) for key in model_keys}
