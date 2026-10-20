"""History and future frames at an origin, and leakage-safe features for direct horizon-grouped models.

For a horizon group whose largest horizon is L, every lag and rolling feature for
target week t uses units from week t - L or earlier. Since every target in the
group has h <= L, t - L <= origin: all features are known at the origin.
"""

import numpy as np
import pandas as pd

from src.segment import PATTERNS

SERIES_KEYS = ["item_id", "dc_id"]
ROLL_WINDOWS = [4, 8, 13, 26]
PRICE_FEATURES = ["price", "price_rel_median", "price_change"]
CALENDAR_FEATURES = [
    "event_days", "event_sporting", "event_cultural", "event_national", "event_religious",
    "snap_days", "week_of_year", "year",
]
CATEGORICAL_FEATURES = ["item_id", "dc_id", "abc", "pattern"]


def split_at_origin(
    weekly: pd.DataFrame, segments: pd.DataFrame, origin: int, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (history, future, actuals) for one origin.

    history: launched rows with t <= origin, with the origin's abc and pattern.
    future: rows t = origin + 1 .. origin + horizon for the same series, without units.
    actuals: item_id, dc_id, t, units for the future rows (for scoring only).
    segments must be the segment set computed at this origin.
    """
    seg = segments[SERIES_KEYS + ["abc", "pattern"]]
    history = weekly[weekly["is_launched"] & (weekly["t"] <= origin)].merge(seg, on=SERIES_KEYS, validate="many_to_one")
    series = history[SERIES_KEYS].drop_duplicates()
    ahead = weekly[weekly["t"].between(origin + 1, origin + horizon)].merge(series, on=SERIES_KEYS)
    ahead = ahead.merge(seg, on=SERIES_KEYS, validate="many_to_one")
    actuals = ahead[SERIES_KEYS + ["t", "units"]].reset_index(drop=True)
    future = ahead.drop(columns=["units"]).reset_index(drop=True)
    return history.reset_index(drop=True), future, actuals


def lags_for(min_lag: int) -> list[int]:
    lags = [min_lag, min_lag + 1, min_lag + 2, min_lag + 4]
    if 52 >= min_lag and 52 not in lags:
        lags.append(52)
    return lags


def history_feature_columns(min_lag: int) -> list[str]:
    """Features computed from past units (must be non-missing in training rows)."""
    cols = [f"lag_{k}" for k in lags_for(min_lag)]
    for w in ROLL_WINDOWS:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    return cols


def feature_columns(min_lag: int) -> list[str]:
    return history_feature_columns(min_lag) + PRICE_FEATURES + CALENDAR_FEATURES + CATEGORICAL_FEATURES


def build_features(history: pd.DataFrame, future: pd.DataFrame, min_lag: int) -> pd.DataFrame:
    """Panel of history and future rows with all features for one horizon group.

    Future rows carry units = NaN, so no feature can read a future value. Rows
    must be contiguous in t within each series (guaranteed by the pipeline checks).
    """
    panel = pd.concat(
        [history.assign(is_future=False), future.assign(units=np.nan, is_future=True)], ignore_index=True
    ).sort_values(SERIES_KEYS + ["t"], ignore_index=True)
    steps = panel.groupby(SERIES_KEYS)["t"].diff().dropna()
    if not (steps == 1).all():
        raise ValueError("Rows are not contiguous in t within every series; lags would be wrong.")

    by_series = [panel["item_id"], panel["dc_id"]]
    units = panel.groupby(SERIES_KEYS, sort=False)["units"]
    for k in lags_for(min_lag):
        panel[f"lag_{k}"] = units.shift(k)
    shifted = units.shift(min_lag)
    for w in ROLL_WINDOWS:
        rolling = shifted.groupby(by_series).rolling(w, min_periods=w)
        panel[f"roll_mean_{w}"] = rolling.mean().reset_index(level=[0, 1], drop=True)
        panel[f"roll_std_{w}"] = rolling.std().reset_index(level=[0, 1], drop=True)

    median_price = history.groupby(SERIES_KEYS)["price"].median()
    panel["price_rel_median"] = panel["price"] / panel.set_index(SERIES_KEYS).index.map(median_price).to_numpy()
    panel["price_change"] = panel["price"] / panel.groupby(SERIES_KEYS, sort=False)["price"].shift(1) - 1

    categories = {
        "item_id": sorted(panel["item_id"].unique()),
        "dc_id": sorted(panel["dc_id"].unique()),
        "abc": ["A", "B", "C"],
        "pattern": PATTERNS,
    }
    for col, cats in categories.items():
        panel[col] = pd.Categorical(panel[col], categories=cats)
    return panel
