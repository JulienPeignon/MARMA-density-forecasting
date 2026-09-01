"""Alpha-stable distribution helpers backed by the libstable C library (pystable)."""

import os

import numpy as np
import pystable

_LIBSTABLE_PATH = os.environ.get("PYSTABLE_LIBSTABLE_PATH") or None


def _dist(alpha, beta, scale, loc, parametrization):
    return pystable.create(
        float(alpha),
        float(beta),
        float(scale),
        float(loc),
        int(parametrization),
        path=_LIBSTABLE_PATH,
    )


def stable_pdf(x, alpha, beta, loc=0.0, scale=1.0, parametrization=1):
    """Alpha-stable pdf evaluated elementwise on ``x`` (any shape preserved)."""
    xa = np.asarray(x, dtype=float)
    dist = _dist(alpha, beta, scale, loc, parametrization)
    flat = np.atleast_1d(xa).ravel()
    out = np.asarray(
        pystable.pdf(dist, flat.tolist(), int(flat.size), path=_LIBSTABLE_PATH),
        dtype=float,
    )
    if xa.ndim == 0:
        return float(out[0])
    return out.reshape(xa.shape)


def stable_ppf(q, alpha, beta, loc=0.0, scale=1.0, parametrization=1):
    """Alpha-stable quantile function (inverse cdf) at probabilities ``q``."""
    qa = np.asarray(q, dtype=float)
    dist = _dist(alpha, beta, scale, loc, parametrization)
    flat = np.atleast_1d(qa).ravel()
    out = np.asarray(
        pystable.q(dist, flat.tolist(), int(flat.size), path=_LIBSTABLE_PATH),
        dtype=float,
    )
    if qa.ndim == 0:
        return float(out[0])
    return out.reshape(qa.shape)


def stable_rvs(alpha, beta, loc=0.0, scale=1.0, size=1, parametrization=1, seed=None):
    """Draw alpha-stable variates. ``size`` may be an int or a shape tuple.

    When ``seed`` is None it is drawn from numpy's global RNG, so seeding the
    numpy RNG (e.g. via the project's set_seed) makes the draws reproducible.
    """
    if np.isscalar(size):
        shape = (int(size),)
    else:
        shape = tuple(int(v) for v in size)
    n = int(np.prod(shape)) if shape else 0

    dist = _dist(alpha, beta, scale, loc, parametrization)
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    draws = np.asarray(
        pystable.rnd(dist, n, int(seed), path=_LIBSTABLE_PATH), dtype=float
    )
    return draws.reshape(shape)
