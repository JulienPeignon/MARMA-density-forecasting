"""Shared per-block artifacts for the recursive application."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.forecast_methods.gaussian_linear import SES_ALPHA, ses_filter
from src.recursive_forecasting.vintage import Origin, vintage_logprice
from src.stable_mar.stable_mar import stablemar as sm


@dataclass
class BlockArtifact:
    """One MAR fit and the forecast origins it covers."""

    block_index: int
    boundary: Origin  # first origin of the block -> estimation window
    origins: List[Origin]  # all origins covered by the block
    par: List[float]  # MAR params [phi(r), psi(s), alpha, beta, sigma]
    level: float  # level the simulated paths are placed at (see _implied_level)
    y_train: np.ndarray  # simulated training series (log space, recentred)
    calibration: np.ndarray  # simulated calibration series (log space, recentred)
    ar_order: Optional[int] = None  # AR blocks only: p; then par is [c, coefs, sigma]
    spec: Optional[str] = None  # "ar1"/"araic"/"exp"; None -> MAR block


def make_blocks(origins: List[Origin], reestimation_freq: int) -> List[List[Origin]]:
    """Group forecast origins into consecutive blocks of ``reestimation_freq``."""
    X = reestimation_freq
    return [origins[i : i + X] for i in range(0, len(origins), X)]


_ROOT_TOL = 0.02


def _min_char_root(coefs) -> float:
    """Smallest characteristic-root modulus of a MAR AR/MA polynomial."""
    c = np.asarray([float(v) for v in coefs], dtype=float)
    if c.size == 0 or np.allclose(c, 0.0):
        return np.inf
    roots = np.roots(np.concatenate([-c[::-1], [1.0]]))
    return float(np.min(np.abs(roots)))


def _is_unstable(par, r, s) -> bool:
    """Report whether a GCOV fit sits on or inside the unit circle.

    The only guard applied: a characteristic root at or below 1 makes the
    process non-stationary, so the fit cannot be simulated from.
    """
    if _min_char_root(par[:r]) <= 1.0 + _ROOT_TOL:  # causal phi
        return True
    return _min_char_root(par[r : r + s]) <= 1.0 + _ROOT_TOL  # noncausal psi


def _mirror_starts(order, y):
    """Deterministic grid of starts covering every causal/noncausal orientation."""
    r, s = order
    scale = max(
        float(np.subtract(*np.percentile(np.asarray(y, dtype=float), [75, 25]))) / 2.0,
        1e-3,
    )
    tail = [1.4, 0.0, scale]
    levels = (0.2, 0.6, 0.9)
    poles = [
        [*([0.9] * r), *([0.1] * s)],
        [*([0.1] * r), *([0.9] * s)],
    ]
    combos = poles + [
        list(c) for c in itertools.product(levels, repeat=r + s) if list(c) not in poles
    ]
    return [[*combo, *tail] for combo in combos]


def _mle_candidate(y, order, rng_seed, warm_start=None):
    """Best stationary MLE fit by log-likelihood, across the deterministic grid.

    Returns ``(par, mar, least_explosive)``; ``par`` is None when no start is
    stationary, in which case ``least_explosive`` holds the best of them.
    """
    r, s = order
    best_ll = None  # (log-likelihood, par, mar)
    best_root = None  # (min char root, par, mar) -- fallback when none is stable

    if warm_start is not None:
        try:
            mar = sm(order=order)
            mar.fit(y, list(warm_start), method="mle")
            par = [float(v) for v in mar.par]
            if not _is_unstable(par, r, s):
                return par, mar, None
        except Exception:
            pass  # fall through to the grid

    for start in [None, *_mirror_starts(order, y)]:
        try:
            mar = sm(order=order)
            mar.fit(y, start, method="mle")
            par = [float(v) for v in mar.par]
        except Exception:
            continue

        if _is_unstable(par, r, s):
            stability = min(_min_char_root(par[:r]), _min_char_root(par[r : r + s]))
            if best_root is None or stability > best_root[0]:
                best_root = (stability, par, mar)
            continue

        # results["Loglik"] holds the NLL, so the smaller it is the better.
        loglik = -float(mar.results["Loglik"])
        if best_ll is None or loglik > best_ll[0]:
            best_ll = (loglik, par, mar)

    if best_ll is not None:
        return best_ll[1], best_ll[2], None
    return None, None, best_root


def _gcov_candidate(y, order, H, K, rng_seed, warm_start=None):
    """Best stationary GCOV fit by criterion value, across the deterministic grid."""
    r, s = order
    best = None  # (criterion, par, mar)

    if warm_start is not None:
        try:
            mar = sm(order=order)
            mar.fit(y, [float(v) for v in warm_start[: r + s]], H=H, K=K, method="gcov")
            mar.par.extend(mar.fit_stable_noise(mar.results.get("PseudoResiduals")))
            par = [float(v) for v in mar.par]
            if not _is_unstable(par, r, s):
                return par, mar
        except Exception:
            pass  # fall through to the grid

    for start in _mirror_starts(order, y):
        try:
            mar = sm(order=order)
            mar.fit(y, start[: r + s], H=H, K=K, method="gcov")
            mar.par.extend(mar.fit_stable_noise(mar.results.get("PseudoResiduals")))
            par = [float(v) for v in mar.par]
        except Exception:
            continue
        if _is_unstable(par, r, s):
            continue
        criterion = float(mar.results.get("GStatistic", np.inf))
        if best is None or criterion < best[0]:
            best = (criterion, par, mar)
    return (best[1], best[2]) if best else None


def fit_mar(
    y: np.ndarray,
    order,
    estimation: str,
    H: int,
    K: int,
    rng_seed=None,
    warm_start=None,
):
    """Fit MAR(order) on ``y``; return (full parameter vector, fitted model)."""
    r, s = order
    if estimation == "mle":
        par, mar, best = _mle_candidate(y, order, rng_seed, warm_start=warm_start)
        if par is not None:
            return par, mar
        warnings.warn(
            f"MLE MAR{order}: no stationary fit from the default start nor from the "
            f"grid; keeping least-explosive (min root {best[0]:.3f}).",
            stacklevel=2,
        )
        return best[1], best[2]
    if estimation != "gcov":
        raise ValueError(f"Unknown estimation method: {estimation}")

    # 1) configured (H, K)
    got = _gcov_candidate(y, order, H, K, rng_seed, warm_start=warm_start)
    if got is not None:
        return got

    # 2) last resort
    best = None
    for start in _mirror_starts(order, y):
        try:
            mar = sm(order=order)
            mar.fit(y, start[: r + s], H=H, K=K, method="gcov")
            mar.par.extend(mar.fit_stable_noise(mar.results.get("PseudoResiduals")))
            par = [float(v) for v in mar.par]
        except Exception:
            continue
        stability = min(_min_char_root(par[:r]), _min_char_root(par[r : r + s]))
        if best is None or stability > best[0]:
            best = (stability, par, mar)
    if best is None:
        raise RuntimeError(
            f"GCOV MAR{order}: every start failed at H={H}, K={K} on a window of "
            f"{len(y)} points; no fit to fall back on."
        )
    warnings.warn(
        f"GCOV MAR{order}: no stationary fit at H={H}, K={K}; keeping "
        f"least-explosive (min root {best[0]:.3f}).",
        stacklevel=2,
    )
    return best[1], best[2]


def _simulate(order, par, length: int, level: float, seed: int) -> np.ndarray:
    """Draw a length-``length`` MAR path and recentre to the (log) price level."""
    gen = sm(order=order)
    gen.par = list(par)
    traj = gen.generate(length, seed=seed).trajectory.dropna().reset_index(drop=True)
    vals = traj.values
    return vals - np.median(vals) + level


def _implied_level(par, order, mar, y):
    """Level the fitted MAR implies for the series."""
    r, s = order
    results = getattr(mar, "results", None) or {}
    mu = results.get("mu_hat")
    if mu is None:
        pseudo = results.get("PseudoResiduals")
        if pseudo is None:
            return float(np.mean(y))
        mu = float(np.mean(np.asarray(pseudo, dtype=float).ravel()))
    denom = (1.0 - float(np.sum(par[:r]))) * (1.0 - float(np.sum(par[r : r + s])))
    if not np.isfinite(denom) or abs(denom) < 1e-6:
        warnings.warn(
            f"MAR{order}: (1-sum phi)(1-sum psi) = {denom:.2e} is degenerate; "
            f"placing the simulated paths at the sample mean instead.",
            stacklevel=2,
        )
        return float(np.mean(y))
    level = float(mu) / denom
    if not np.isfinite(level):
        return float(np.mean(y))
    return level


def compute_block(
    block_index: int,
    origins_block: List[Origin],
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    *,
    order,
    estimation: str,
    H: int,
    K: int,
    length_simulation: int,
    length_calibration: int,
    seed: int,
    sample_start: str = "1976-01",
    log: bool = True,
    warm_start: Optional[List[float]] = None,
) -> BlockArtifact:
    """Estimate the MAR and draw the shared simulations for one block."""
    boundary = origins_block[0]
    y = vintage_logprice(ng_henry, cpi_ac, boundary, sample_start=sample_start, log=log)

    par, mar = fit_mar(y, order, estimation, H, K, rng_seed=seed, warm_start=warm_start)
    level = _implied_level(par, order, mar, y)
    path = _simulate(
        order,
        par,
        length_simulation + length_calibration + 1,
        level,
        seed,
    )
    train_n = length_simulation + 1
    y_train = path[:train_n]
    calibration = path[train_n : train_n + length_calibration + 1]

    return BlockArtifact(
        block_index=block_index,
        boundary=boundary,
        origins=origins_block,
        par=par,
        level=level,
        y_train=y_train,
        calibration=calibration,
    )


def get_block(
    block_index: int,
    origins_block: List[Origin],
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    **kwargs,
) -> BlockArtifact:
    """Fit the MAR block covering ``origins_block``."""
    return compute_block(block_index, origins_block, ng_henry, cpi_ac, **kwargs)


# Causal univariate benchmarks of Baumeister et al. (2025), ported from
# Table2/OLSvar.m, Table2/aicfind.m and Table2/smooth.m. They feed the exact
# stable predictive densities and the replication check in
# src.evaluate.baumeister_et_al_benchmarks; nothing here is simulated.

AIC_MAX_LAG = 6  # parsimony cap of main_araic.m
AR_SPECS = ("ar1", "araic")


def _ar_design(y: np.ndarray, p: int, n_drop: int):
    """Regressors ``[1, y_{t-1}, ..., y_{t-p}]`` and response ``y_t``."""
    y = np.asarray(y, dtype=float).ravel()
    t = len(y)
    cols = [np.ones(t - n_drop)]
    for i in range(1, p + 1):
        cols.append(y[n_drop - i : t - i])
    return np.column_stack(cols), y[n_drop:]


def fit_ar_ols(y: np.ndarray, p: int):
    """OLS AR(p) with intercept -- scalar port of ``Table2/OLSvar.m``."""
    X, Y = _ar_design(y, p, p)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(beta[0]), np.asarray(beta[1:], dtype=float), Y - X @ beta


def ar_aic_ranking(y: np.ndarray, pmax: int = AIC_MAX_LAG):
    """Lag orders ``0..pmax`` ranked by AIC -- port of ``Table2/aicfind.m``."""
    n = len(np.asarray(y).ravel()) - pmax
    crit = np.empty(pmax + 1)
    for p in range(pmax + 1):
        X, Y = _ar_design(y, p, pmax)
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        crit[p] = np.log(float(np.sum((Y - X @ beta) ** 2) / n)) + 2.0 * (p + 1) / n
    return [int(p) for p in np.argsort(crit)], crit


def ar_lag_order(y: np.ndarray, spec: str, pmax: int = AIC_MAX_LAG) -> int:
    """Lag order used by ``spec``: 1 for ``"ar1"``, ``aicfind`` for ``"araic"``."""
    if spec == "ar1":
        return 1
    if spec == "araic":
        return ar_aic_ranking(y, pmax)[0][0]
    raise ValueError(f"Unknown AR spec: {spec!r} (expected one of {AR_SPECS})")


def ar_forecast(const: float, coefs: np.ndarray, y: np.ndarray, horizon: int) -> float:
    """Iterate the AR forecast ``horizon`` steps; port of the ``OLSvar.m`` loop."""
    coefs = np.asarray(coefs, dtype=float)
    p = len(coefs)
    path = list(np.asarray(y, dtype=float).ravel())
    for _ in range(horizon):
        history = path[len(path) - p :][::-1] if p else []
        path.append(const + float(np.dot(coefs, history)))
    return path[-1]


def fit_ar_dgp(
    y: np.ndarray,
    spec: str,
    pmax: int = AIC_MAX_LAG,
    estimation: str = "mle",
):
    """Fit the AR training DGP for ``spec`` on the estimation window ``y``.

    The lag order always comes from the OLS AIC ranking of ``aicfind.m``; the
    coefficients are OLS and the Gaussian scale is fitted on the residuals.
    """
    y = np.asarray(y, dtype=float)
    p = ar_lag_order(y, spec, pmax)

    const, coefs, resid = fit_ar_ols(y, p)
    resid = np.asarray(resid, dtype=float).ravel()
    if estimation == "mle":
        sigma = float(np.sqrt(np.mean(resid**2)))
    elif estimation == "gcov":
        sigma = float(np.sqrt(np.sum(resid**2) / max(len(resid) - p - 1, 1)))
    else:
        raise ValueError(f"Unknown estimation method: {estimation!r}")
    return p, const, coefs, [sigma]


EXACT_SPECS = ("ar1", "araic", "exp")


def fit_exp_dgp(y: np.ndarray):
    """Fit the exponential-smoothing benchmark as its IMA(1,1) representation.

    The smoothing constant is fixed by the protocol, so only the Gaussian scale
    is estimated, by MLE on the filtered residuals.
    """
    level, resid = ses_filter(np.asarray(y, dtype=float), SES_ALPHA)
    return SES_ALPHA, level, [float(np.sqrt(np.mean(np.asarray(resid) ** 2)))]


def compute_exact_block(
    block_index: int,
    origins_block: List[Origin],
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    *,
    spec: str,
    sample_start: str = "1976-01",
    log: bool = True,
    pmax: int = AIC_MAX_LAG,
    estimation: str = "mle",
) -> BlockArtifact:
    """Fit a causal linear benchmark whose predictive law is closed-form.

    Innovations are Gaussian, so ``par`` ends with sigma alone.
    """
    if spec not in EXACT_SPECS:
        raise ValueError(
            f"Unknown exact spec: {spec!r} (expected one of {EXACT_SPECS})"
        )

    boundary = origins_block[0]
    y = vintage_logprice(ng_henry, cpi_ac, boundary, sample_start=sample_start, log=log)

    if spec == "exp":
        theta, _level, noise_par = fit_exp_dgp(y)
        par, ar_order = [float(theta), *noise_par], None
    else:
        p, const, coefs, noise_par = fit_ar_dgp(y, spec, pmax, estimation=estimation)
        par, ar_order = [float(const), *(float(v) for v in coefs), *noise_par], p

    return BlockArtifact(
        block_index=block_index,
        boundary=boundary,
        origins=origins_block,
        par=par,
        level=float(np.mean(y)),
        y_train=np.empty(0, dtype=float),
        calibration=np.empty(0, dtype=float),
        ar_order=ar_order,
        spec=spec,
    )


def get_exact_block(
    block_index: int,
    origins_block: List[Origin],
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    **kwargs,
) -> BlockArtifact:
    """Signature-compatible with :func:`get_block`, for the closed-form models."""
    return compute_exact_block(block_index, origins_block, ng_henry, cpi_ac, **kwargs)
