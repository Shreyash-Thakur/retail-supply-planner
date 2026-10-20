"""Forecast and service metrics. Positions where actual is NaN are ignored.

Aggregations across series are volume-weighted: pass the pooled rows of every
series to wape or bias, never average per-series values.
"""

import numpy as np


def _clean(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    keep = ~np.isnan(a)
    return a[keep], f[keep]


def wape(actual, forecast) -> float:
    """sum(|a - f|) / sum(a); NaN if sum(a) is 0."""
    a, f = _clean(actual, forecast)
    total = a.sum()
    return float(np.abs(a - f).sum() / total) if total != 0 else np.nan


def bias(actual, forecast) -> float:
    """(sum(f) - sum(a)) / sum(a); positive means over-forecasting. NaN if sum(a) is 0."""
    a, f = _clean(actual, forecast)
    total = a.sum()
    return float((f.sum() - total) / total) if total != 0 else np.nan


def mase_scale(insample, season: int = 52) -> float:
    """Mean absolute seasonal difference of the in-sample history.

    Falls back to season 1 when there are fewer than season + 1 points or the
    seasonal scale is 0. NaN if the lag-1 scale is also 0 or undefined.
    """
    y = np.asarray(insample, dtype=float)
    y = y[~np.isnan(y)]
    for lag in (season, 1) if season != 1 else (1,):
        if len(y) >= lag + 1:
            scale = np.abs(y[lag:] - y[:-lag]).mean()
            if scale > 0:
                return float(scale)
    return np.nan


def mase(actual, forecast, insample, season: int = 52) -> float:
    """mean(|a - f|) divided by mase_scale(insample, season)."""
    a, f = _clean(actual, forecast)
    scale = mase_scale(insample, season)
    if len(a) == 0 or np.isnan(scale):
        return np.nan
    return float(np.abs(a - f).mean() / scale)


def fva(wape_model: float, wape_benchmark: float) -> float:
    """Forecast value added in percentage points: benchmark WAPE minus model WAPE. Positive adds value."""
    return float((wape_benchmark - wape_model) * 100)


def fill_rate(sold, demand) -> float:
    """sum(sold) / sum(demand); 1.0 if there was no demand."""
    total = float(np.nansum(np.asarray(demand, dtype=float)))
    return float(np.nansum(np.asarray(sold, dtype=float)) / total) if total != 0 else 1.0
