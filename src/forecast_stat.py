"""Statistical forecasters: Holt-Winters (with fallbacks), Croston and TSB."""

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecast_baselines import SERIES_KEYS, finalize, seasonal_naive

logger = logging.getLogger(__name__)

BLAS_THREAD_VARS = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]


def croston_forecast(y: np.ndarray, alpha: float) -> float:
    """Croston: smooth non-zero size z and inter-demand interval p separately; forecast z / p.

    z starts at the first non-zero demand and p at its 1-based position. Each later
    non-zero demand d, q weeks after the previous one, updates z += alpha (d - z)
    and p += alpha (q - p). All-zero history forecasts 0.
    """
    nonzero = np.flatnonzero(np.asarray(y) > 0)
    if len(nonzero) == 0:
        return 0.0
    z, p = float(y[nonzero[0]]), float(nonzero[0] + 1)
    for previous, current in zip(nonzero[:-1], nonzero[1:]):
        z += alpha * (y[current] - z)
        p += alpha * ((current - previous) - p)
    return z / p


def tsb_forecast(y: np.ndarray, alpha: float, beta: float) -> float:
    """Teunter-Syntetos-Babai: smooth demand probability b every week and size z on non-zero weeks.

    b starts at the share of non-zero weeks, z at the mean non-zero demand.
    Forecast b x z, which decays toward 0 when an item stops selling.
    """
    y = np.asarray(y, dtype=float)
    occurs = y > 0
    if not occurs.any():
        return 0.0
    b, z = occurs.mean(), y[occurs].mean()
    for d, happened in zip(y, occurs):
        b += beta * (float(happened) - b)
        if happened:
            z += alpha * (d - z)
    return b * z


def _per_series(history: pd.DataFrame, func) -> pd.Series:
    ordered = history.sort_values(SERIES_KEYS + ["t"])
    return ordered.groupby(SERIES_KEYS)["units"].agg(lambda s: func(s.to_numpy(dtype=float)))


def croston(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    level = _per_series(history, lambda y: croston_forecast(y, cfg["forecast"]["croston_alpha"]))
    values = future.set_index(SERIES_KEYS).index.map(level)
    return finalize(future, pd.Series(values, index=future.index), "croston", history)


def tsb(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    f = cfg["forecast"]
    level = _per_series(history, lambda y: tsb_forecast(y, f["tsb_alpha"], f["tsb_beta"]))
    values = future.set_index(SERIES_KEYS).index.map(level)
    return finalize(future, pd.Series(values, index=future.index), "tsb", history)


def _fit_one(args: tuple) -> tuple[np.ndarray | None, str]:
    """Fit one series. Returns (forecast or None on failure, method used)."""
    y, horizon, seasonal = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if seasonal:
                model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=52)
                method = "holt_winters"
            else:
                model = ExponentialSmoothing(y, trend=None, seasonal=None)
                method = "ses"
            values = np.asarray(model.fit().forecast(horizon), dtype=float)
        except Exception:  # noqa: BLE001  any statsmodels failure falls back to seasonal naive
            return None, "error"
    if not np.all(np.isfinite(values)):
        return None, "non_finite"
    return values, method


def _run_pool(tasks: list[tuple], workers: int) -> list[tuple[np.ndarray | None, str]]:
    """Fit series in a bounded process pool with one BLAS thread per worker.

    Each worker otherwise starts one OpenBLAS thread per CPU with its own buffer;
    with one worker per CPU that exhausted memory on a 20-CPU laptop. The fits are
    tiny, so extra BLAS threads only cost memory. Children read the variables at
    their own numpy import, so setting them here, before the pool starts, is enough.
    """
    saved = {k: os.environ.get(k) for k in BLAS_THREAD_VARS}
    os.environ.update({k: "1" for k in BLAS_THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_fit_one, tasks, chunksize=8))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def holt_winters(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Damped additive Holt-Winters (period 52) for series with enough history and not inactive;
    simple exponential smoothing otherwise; seasonal naive (fallback) on any error or non-finite output."""
    min_weeks = cfg["forecast"]["min_history_weeks_for_seasonal"]
    ordered = history.sort_values(SERIES_KEYS + ["t"])
    groups = list(ordered.groupby(SERIES_KEYS))
    inactive = ordered.groupby(SERIES_KEYS)["pattern"].first() == "inactive"
    tasks = [
        (g["units"].to_numpy(dtype=float), horizon, len(g) >= min_weeks and not inactive.loc[key])
        for key, g in groups
    ]
    results = _run_pool(tasks, cfg["forecast"]["stat_workers"])

    origin = int(history["t"].max())
    rows, methods = [], []
    for (key, _), (values, method) in zip(groups, results):
        methods.append(method)
        if values is not None:
            rows.append(pd.DataFrame({"item_id": key[0], "dc_id": key[1], "t": origin + 1 + np.arange(horizon), "hw": values}))
    counts = pd.Series(methods).value_counts().to_dict()
    logger.info("holt_winters methods: %s", counts)

    fitted = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SERIES_KEYS + ["t", "hw"])
    merged = future[SERIES_KEYS + ["t"]].merge(fitted, on=SERIES_KEYS + ["t"], how="left")
    failed = merged["hw"].isna()
    if failed.any():
        fallback_values = seasonal_naive(history, future, horizon, cfg)["forecast"].to_numpy()
        merged.loc[failed, "hw"] = fallback_values[failed.to_numpy()]
    out = finalize(future, pd.Series(merged["hw"].to_numpy(), index=future.index), "holt_winters", history,
                   fallback=pd.Series(failed.to_numpy(), index=future.index))
    out.attrs["methods"] = counts
    return out
