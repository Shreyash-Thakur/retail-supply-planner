"""Baseline forecasters and the shared output contract for every model.

Every model has the signature forecast(history, future, horizon, cfg) -> DataFrame
with columns item_id, dc_id, t, h, model, forecast, fallback. history holds the
launched rows up to the origin; future holds rows t = origin + 1 .. origin + horizon
with known features and no units.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
OUTPUT_COLUMNS = ["item_id", "dc_id", "t", "h", "model", "forecast", "fallback"]


def last_values(history: pd.DataFrame) -> pd.Series:
    """Last observed units per series, indexed by (item_id, dc_id)."""
    return history.sort_values("t").groupby(SERIES_KEYS)["units"].last()


def finalize(future: pd.DataFrame, values: pd.Series, model: str, history: pd.DataFrame,
             fallback: pd.Series | None = None) -> pd.DataFrame:
    """Apply the output contract: naive fallback for missing or non-finite values, clip at 0.

    values is aligned to future's index. fallback marks rows a model already
    replaced with its own fallback; rows filled with naive here are added to it.
    """
    out = future[SERIES_KEYS + ["t"]].copy()
    origin = int(history["t"].max())
    out["h"] = out["t"] - origin
    out["model"] = model
    out["forecast"] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out["fallback"] = False if fallback is None else fallback.to_numpy(dtype=bool)

    missing = ~np.isfinite(out["forecast"])
    if missing.any():
        naive_value = out.set_index(SERIES_KEYS).index.map(last_values(history)).to_numpy(dtype=float)
        out.loc[missing, "forecast"] = naive_value[missing]
        out.loc[missing, "fallback"] = True
    out["forecast"] = out["forecast"].fillna(0.0).clip(lower=0.0)

    n_fallback = int(out.drop_duplicates(SERIES_KEYS + ["fallback"]).query("fallback").shape[0])
    if n_fallback:
        logger.info("%s: %d series used a fallback", model, n_fallback)
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def naive(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Last observed value, repeated."""
    values = future.set_index(SERIES_KEYS).index.map(last_values(history))
    return finalize(future, pd.Series(values, index=future.index), "naive", history)


def seasonal_naive(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Value from 52 weeks before each target week; naive (fallback) when that week is not in history."""
    lookup = history.set_index(SERIES_KEYS + ["t"])["units"]
    keys = pd.MultiIndex.from_arrays([future["item_id"], future["dc_id"], future["t"] - 52])
    values = pd.Series(keys.map(lookup), index=future.index, dtype=float)
    return finalize(future, values, "seasonal_naive", history)


def moving_average_4(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Mean of the last 4 weeks (fewer if the series is shorter), repeated."""
    mean4 = history.sort_values("t").groupby(SERIES_KEYS).tail(4).groupby(SERIES_KEYS)["units"].mean()
    values = future.set_index(SERIES_KEYS).index.map(mean4)
    return finalize(future, pd.Series(values, index=future.index), "moving_average_4", history)
