"""Hierarchical consistency on the last backtest fold: bottom-up vs top-down at item-DC, item, DC and total level."""

import logging

import pandas as pd

from src.backtest import MODELS
from src.metrics import bias, wape
from src.segment import fold_origins

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
LEVELS = {"item_dc": ["item_id", "dc_id"], "item": ["item_id"], "dc": ["dc_id"], "total": []}
# LightGBM learns across many series and has no meaning on one aggregate series, so the total uses univariate models.
TOTAL_CANDIDATES = ["naive", "seasonal_naive", "moving_average_4", "holt_winters", "croston", "tsb"]


def total_series(weekly: pd.DataFrame) -> pd.DataFrame:
    """Weekly total units over all launched series, as one series in the forecasting interface's shape."""
    launched = weekly[weekly["is_launched"]]
    total = launched.groupby("t", as_index=False)["units"].sum()
    return total.assign(item_id="TOTAL", dc_id="ALL", pattern="smooth", abc="A")


def forecast_total(total: pd.DataFrame, model: str, origin: int, horizon: int, cfg: dict) -> pd.DataFrame:
    history = total[total["t"] <= origin]
    future = total[total["t"].between(origin + 1, origin + horizon)].drop(columns="units")
    return MODELS[model](history, future, horizon, cfg)


def choose_total_model(total: pd.DataFrame, origins: list[int], horizon: int, cfg: dict) -> pd.DataFrame:
    """WAPE of each candidate on the total series, pooled over the given (earlier) origins."""
    rows = []
    for model in TOTAL_CANDIDATES:
        preds = pd.concat([forecast_total(total, model, o, horizon, cfg) for o in origins], ignore_index=True)
        scored = preds.merge(total[["t", "units"]], on="t")
        rows.append({"model": model, "wape": wape(scored["units"], scored["forecast"]), "bias": bias(scored["units"], scored["forecast"])})
    return pd.DataFrame(rows).sort_values("wape", ignore_index=True)


def history_shares(weekly: pd.DataFrame, origin: int, weeks: int = 52) -> pd.Series:
    """Each series' share of total units over the `weeks` weeks up to and including the origin."""
    window = weekly[weekly["is_launched"] & weekly["t"].between(origin - weeks + 1, origin)]
    units = window.groupby(SERIES_KEYS)["units"].sum()
    return units / units.sum()


def top_down(total_forecast: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Split each week's total forecast by the series' historical shares."""
    split = total_forecast[["t", "forecast"]].merge(shares.rename("share").reset_index(), how="cross")
    split["forecast"] = split["forecast"] * split["share"]
    return split[SERIES_KEYS + ["t", "forecast"]]


def score_levels(bottom: pd.DataFrame, top: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """WAPE and bias per level for both methods. Forecasts are summed to each level before scoring."""
    rows = []
    for method, fc in (("bottom_up", bottom), ("top_down", top)):
        data = actual.merge(fc, on=SERIES_KEYS + ["t"], how="left").fillna({"forecast": 0.0})
        for level, keys in LEVELS.items():
            agg = data.groupby(keys + ["t"], as_index=False)[["actual", "forecast"]].sum() if keys else \
                data.groupby("t", as_index=False)[["actual", "forecast"]].sum()
            rows.append({"level": level, "method": method, "wape": wape(agg["actual"], agg["forecast"]),
                         "bias": bias(agg["actual"], agg["forecast"]), "groups": agg[keys].drop_duplicates().shape[0] if keys else 1})
    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    """Stage `hierarchy`: write data/outputs/hierarchy_scores.csv and hierarchy_total_models.csv."""
    paths = cfg["data"]
    out = paths["outputs_dir"]
    gold = pd.read_parquet(paths["gold_dir"] / "weekly.parquet")
    segments = pd.read_parquet(paths["gold_dir"] / "segments.parquet")
    forecasts = pd.read_parquet(out / "backtest_forecasts.parquet")
    selection = pd.read_csv(out / "model_selection.csv").set_index("pattern")["model"]
    horizon = cfg["forecast"]["horizon_weeks"]
    origins = fold_origins(int(gold["t"].max()), horizon, cfg["forecast"]["backtest_folds"])
    last = max(origins)
    origin = origins[last]

    seg = segments[segments["segment_set"] == last][SERIES_KEYS + ["pattern"]]
    chosen = seg.assign(model=seg["pattern"].map(selection))
    fold = forecasts[forecasts["fold"] == last].merge(chosen[SERIES_KEYS + ["model"]], on=SERIES_KEYS + ["model"])
    bottom = fold[SERIES_KEYS + ["t", "forecast"]]
    actual = fold[SERIES_KEYS + ["t", "actual"]]

    total = total_series(gold)
    earlier = [o for name, o in origins.items() if name != last]
    candidates = choose_total_model(total, earlier, horizon, cfg)
    total_model = candidates.iloc[0]["model"]
    top = top_down(forecast_total(total, total_model, origin, horizon, cfg), history_shares(gold, origin))

    scores = score_levels(bottom, top, actual)
    scores.to_csv(out / "hierarchy_scores.csv", index=False)
    candidates.to_csv(out / "hierarchy_total_models.csv", index=False)

    print(f"\nHierarchy ({last}, origin t {origin}; shares from the 52 weeks up to the origin)")
    print(f"  total-series model chosen on folds before {last}: {total_model}")
    print("  " + candidates.round(3).to_string(index=False).replace("\n", "\n  "))
    table = scores.pivot(index="level", columns="method", values="wape").reindex(list(LEVELS))
    print("  WAPE by level:\n  " + table.round(3).to_string().replace("\n", "\n  "))
