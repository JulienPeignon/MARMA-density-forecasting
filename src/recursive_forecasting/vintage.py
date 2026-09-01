"""Real-time vintage indexing for the recursive gas-price application.

Faithful port of the indexing in Baumeister et al. (2025), ``Table2/main_ar1.m``.

The real-time matrices ``NG_HENRY`` / ``CPI_AC`` are laid out as::

    rows    = calendar month, row 1 (1-based) == 1973M1
    columns = real-time vintage, column 1 (1-based) == 1991M1

At each forecast *origin* the estimation sample is a single vintage **column**
(the data as known at that date, with its revised history), rows from
``sample_start`` up to the origin month, expanding over time. This mirrors the
MATLAB line

    rpg = log(100*NG_HENRY(37:ind, jx) ./ CPI_AC(37:ind, jx))

where ``jx`` advances the vintage column and ``ind`` advances the last row, in
lockstep (``main_ar1.m`` lines 31, 58). The model is fit on the **log** real
price; forecasts are ``exp()``-ed back for level metrics and evaluated against
the final (May-2024) vintage, ``x = 100*NG_May24 ./ CPI_May24`` (line 36).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# Calendar anchors of the BHLR real-time matrices (1-based row/col 1).
ROW0 = pd.Period("1973-01", freq="M")  # matrix row 1
COL0 = pd.Period("1991-01", freq="M")  # matrix column 1


def _period(m) -> pd.Period:
    """Coerce a month specifier to a monthly ``pd.Period``."""
    if isinstance(m, pd.Period):
        return m.asfreq("M")
    return pd.Period(pd.Timestamp(m), freq="M")


def _row(month) -> int:
    """0-based row index of ``month`` in the real-time / final matrices."""
    return (_period(month) - ROW0).n


def _col(month) -> int:
    """0-based vintage-column index for a forecast origin at ``month``."""
    return (_period(month) - COL0).n


@dataclass(frozen=True)
class Origin:
    """One forecast origin in the recursive scheme."""

    origin_month: pd.Period
    target_month: pd.Period
    vintage_col: int  # column of NG_HENRY / CPI_AC used at this origin
    origin_row: int  # last (0-based) estimation row available at the origin


def enumerate_origins(
    horizon: int,
    n_cols: int,
    first_origin="1997-01",
) -> List[Origin]:
    """List forecast origins for a given horizon.

    Follows ``main_ar1.m``: origins run over vintage columns from
    ``first_origin`` (jx=73, i.e. 1997M1) to ``n_cols - horizon`` (1-based
    ``size(CPI_AC,2)-h``); the target is the origin plus ``horizon`` months.
    For h=1 this yields 325 origins (1997M1 -> target 1997M2, ..., last target
    2024M2), matching the paper's 1997M2-2024M2 evaluation window.
    """
    first_col = _col(first_origin)
    last_col = n_cols - horizon - 1  # 0-based of 1-based (n_cols - horizon)
    origins: List[Origin] = []
    for c in range(first_col, last_col + 1):
        om = COL0 + c
        origins.append(
            Origin(
                origin_month=om,
                target_month=om + horizon,
                vintage_col=c,
                origin_row=_row(om),
            )
        )
    return origins


def vintage_logprice(
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    origin: Origin,
    sample_start="1976-01",
    log: bool = True,
) -> np.ndarray:
    """Real-time (log) real gas price available at ``origin``.

    Returns ``log(100 * NG / CPI)`` for vintage column ``origin.vintage_col``,
    rows from ``sample_start`` through ``origin.origin_row`` (inclusive) --
    Baumeister's ``rpg``. With ``log=False`` returns the level series.
    """
    r0 = _row(sample_start)
    r1 = origin.origin_row
    col = origin.vintage_col
    nom = np.asarray(ng_henry[r0 : r1 + 1, col], dtype=float)
    defl = np.asarray(cpi_ac[r0 : r1 + 1, col], dtype=float)
    real = 100.0 * nom / defl
    return np.log(real) if log else real


def final_series(
    ng_may24: np.ndarray,
    cpi_may24: np.ndarray,
    months,
    log: bool = False,
) -> np.ndarray:
    """Return the final (May-2024) vintage real price at the given month(s).

    This is the ex-post evaluation target ``x = 100*NG_May24 ./ CPI_May24``.
    Returns levels by default (Baumeister evaluates level MSPE); ``log=True``
    returns the log real price for log-space density scoring.
    """
    ng = np.asarray(ng_may24, dtype=float).ravel()
    cpi = np.asarray(cpi_may24, dtype=float).ravel()
    rows = [_row(m) for m in months] if _is_iterable(months) else _row(months)
    real = 100.0 * ng[rows] / cpi[rows]
    return np.log(real) if log else real


def _is_iterable(x) -> bool:
    return hasattr(x, "__iter__") and not isinstance(x, (str, pd.Period))


def origin_features(
    ng_henry: np.ndarray,
    cpi_ac: np.ndarray,
    origins: List[Origin],
    lags: int,
    sample_start="1976-01",
    log: bool = True,
) -> np.ndarray:
    """Feature matrix: the last ``lags`` (log) prices available at each origin.

    Each origin uses its **own** vintage column (the freshest data available at
    that date), so the conditioning history is real-time correct even when the
    MAR parameters come from a block re-estimated at an earlier origin.
    Row order is oldest-first (``[y_{t-lags+1}, ..., y_t]``), matching the
    layout produced by ``src.calibration.recalibration.lagged_pairs``.
    """
    feats = np.empty((len(origins), lags), dtype=np.float64)
    for i, o in enumerate(origins):
        series = vintage_logprice(
            ng_henry, cpi_ac, o, sample_start=sample_start, log=log
        )
        feats[i] = series[-lags:]
    return feats


def origin_targets(
    ng_may24: np.ndarray,
    cpi_may24: np.ndarray,
    origins: List[Origin],
    log: bool = False,
) -> np.ndarray:
    """Final-vintage target values at each origin's target month."""
    return final_series(ng_may24, cpi_may24, [o.target_month for o in origins], log=log)
