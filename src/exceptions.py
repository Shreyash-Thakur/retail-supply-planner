"""Planner exceptions ranked by value at risk: stockout risk, overstock, forecast bias, demand anomaly."""

import logging

import numpy as np
import pandas as pd

from src.simulate import Scenario, build_inputs, simulate

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
COLUMNS = ["item_id", "dc_id", "abc", "exception_type", "value_at_risk", "detail"]


def stockout_risk(projection: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Projected on hand below safety stock in any of the first `weeks` weeks.

    projection: item_id, dc_id, week, on_hand (end of week), lost, safety_stock, unit_price, abc.
    Value at risk = projected lost units in those weeks x unit price.
    """
    near = projection[projection["week"] <= weeks]
    below = near[near["on_hand"] < near["safety_stock"]]
    if below.empty:
        return pd.DataFrame(columns=COLUMNS)
    first = below.groupby(SERIES_KEYS)["week"].min()
    lost = near.groupby(SERIES_KEYS).agg(lost=("lost", "sum"), unit_price=("unit_price", "first"), abc=("abc", "first"),
                                         safety_stock=("safety_stock", "first"), min_on_hand=("on_hand", "min"))
    out = lost.loc[first.index].assign(first_week=first).reset_index()
    out["value_at_risk"] = out["lost"] * out["unit_price"]
    out["detail"] = [f"on hand falls to {m:.0f} (safety stock {s:.0f}) by week {w}; {l:.0f} units projected lost in weeks 1 to {weeks}"
                     for m, s, w, l in zip(out["min_on_hand"], out["safety_stock"], out["first_week"], out["lost"])]
    return out.assign(exception_type="stockout_risk")[COLUMNS]


def overstock(policy: pd.DataFrame, unit_cost: pd.Series, max_days: float = 60) -> pd.DataFrame:
    """Days of cover above max_days. Excess = opening stock minus max_days of forecast demand; value = excess x unit cost.

    A series with stock but a zero forecast has unlimited cover and counts as overstock.
    """
    p = policy.assign(unit_cost=policy["item_id"].map(unit_cost))
    over = p[(p["days_of_cover"] > max_days) | (p["days_of_cover"].isna() & (p["opening_stock"] > 0))].copy()
    over["excess"] = over["opening_stock"] - max_days * over["mean_weekly_forecast"] / 7
    over["value_at_risk"] = over["excess"] * over["unit_cost"]
    cover = over["days_of_cover"].map(lambda d: "unlimited (zero forecast)" if pd.isna(d) else f"{d:.0f} days")
    over["detail"] = [f"cover {c}; {e:.0f} units above {max_days:.0f} days" for c, e in zip(cover, over["excess"])]
    return over.assign(exception_type="overstock")[COLUMNS]


def forecast_bias(last_fold: pd.DataFrame, unit_price: pd.Series, threshold: float = 0.20) -> pd.DataFrame:
    """Chosen model's bias beyond +/- threshold over the last backtest fold.

    last_fold: item_id, dc_id, abc, forecast, actual (rows of one fold, chosen model).
    Value at risk = |sum forecast - sum actual| x unit price.
    """
    g = last_fold.groupby(SERIES_KEYS).agg(forecast=("forecast", "sum"), actual=("actual", "sum"), abc=("abc", "first")).reset_index()
    g = g[g["actual"] > 0]
    g["bias"] = (g["forecast"] - g["actual"]) / g["actual"]
    flagged = g[g["bias"].abs() > threshold].copy()
    flagged["value_at_risk"] = (flagged["forecast"] - flagged["actual"]).abs() * flagged["item_id"].map(unit_price)
    flagged["detail"] = [f"bias {b:+.0%} over the last fold ({f:.0f} forecast vs {a:.0f} actual)"
                         for b, f, a in zip(flagged["bias"], flagged["forecast"], flagged["actual"])]
    return flagged.assign(exception_type="forecast_bias")[COLUMNS]


def demand_anomaly(last_week: pd.DataFrame, residuals: pd.DataFrame, unit_price: pd.Series) -> pd.DataFrame:
    """Last actual week outside forecast + [5th, 95th] percentile of the series' residuals.

    last_week: item_id, dc_id, abc, forecast, actual for the final week of data (chosen model, last fold).
    Value at risk = distance outside the range x unit price.
    """
    bands = residuals.groupby(SERIES_KEYS)["residual"].quantile([0.05, 0.95]).unstack().rename(columns={0.05: "r05", 0.95: "r95"})
    d = last_week.merge(bands.reset_index(), on=SERIES_KEYS)
    d["low"], d["high"] = d["forecast"] + d["r05"], d["forecast"] + d["r95"]
    outside = d[(d["actual"] < d["low"]) | (d["actual"] > d["high"])].copy()
    outside["distance"] = np.where(outside["actual"] > outside["high"], outside["actual"] - outside["high"], outside["low"] - outside["actual"])
    outside["value_at_risk"] = outside["distance"] * outside["item_id"].map(unit_price)
    outside["detail"] = [f"last week {a:.0f} units, expected range {lo:.0f} to {hi:.0f}"
                         for a, lo, hi in zip(outside["actual"], outside["low"], outside["high"])]
    return outside.assign(exception_type="demand_anomaly")[COLUMNS]


def projected_baseline(policy: pd.DataFrame, final_forecast: pd.DataFrame, supply: pd.DataFrame, items: pd.DataFrame,
                       cfg: dict) -> pd.DataFrame:
    """Deterministic projection: expected demand (no noise), baseline supplier capacity and LP allocation."""
    inputs = build_inputs(policy, final_forecast, supply, items)
    noise = np.zeros((1, len(inputs.item_ids), len(inputs.dc_ids), inputs.forecast.shape[-1]))
    res = simulate(inputs, noise, Scenario("projection"), cfg["allocation"]["abc_weight"], cfg["allocation"]["cover_basis"])
    weeks = res["on_hand"].shape[1]
    frame = pd.DataFrame({
        "item_id": np.tile(np.repeat(inputs.item_ids, len(inputs.dc_ids)), weeks),
        "dc_id": np.tile(inputs.dc_ids, len(inputs.item_ids) * weeks),
        "week": np.repeat(np.arange(1, weeks + 1), len(inputs.item_ids) * len(inputs.dc_ids)),
        "on_hand": res["on_hand"][0].reshape(weeks, -1).ravel(),
        "lost": res["lost"][0].reshape(weeks, -1).ravel(),
    })
    return frame.merge(policy[SERIES_KEYS + ["safety_stock", "abc"]], on=SERIES_KEYS).merge(
        items[["item_id", "unit_price"]], on="item_id")


def run(cfg: dict) -> None:
    """Stage `exceptions`: write data/outputs/exceptions.csv ranked by value at risk."""
    paths = cfg["data"]
    out = paths["outputs_dir"]
    policy = pd.read_parquet(out / "inventory_policy.parquet")
    final_forecast = pd.read_parquet(out / "final_forecast.parquet")
    residuals = pd.read_parquet(out / "residuals.parquet")
    supply = pd.read_parquet(paths["gold_dir"] / "supply_params.parquet")
    items = pd.read_parquet(paths["gold_dir"] / "item_params.parquet")
    exc_cfg = cfg["exceptions"]
    price = items.set_index("item_id")["unit_price"]

    last_fold_name = residuals["fold"].max()
    last_fold = residuals[residuals["fold"] == last_fold_name].merge(policy[SERIES_KEYS + ["abc"]], on=SERIES_KEYS)
    last_week = last_fold[last_fold["t"] == last_fold["t"].max()]

    found = pd.concat([
        stockout_risk(projected_baseline(policy, final_forecast, supply, items, cfg), exc_cfg["stockout_weeks"]),
        overstock(policy, items.set_index("item_id")["unit_cost"], exc_cfg["overstock_days"]),
        forecast_bias(last_fold, price, exc_cfg["bias_threshold"]),
        demand_anomaly(last_week, residuals, price),
    ], ignore_index=True)
    found = found.sort_values("value_at_risk", ascending=False, ignore_index=True)
    found.insert(0, "rank", np.arange(1, len(found) + 1))
    found.to_csv(out / "exceptions.csv", index=False)

    print("\nPlanner exceptions")
    summary = found.groupby("exception_type").agg(count=("rank", "size"), value_at_risk=("value_at_risk", "sum"))
    print("  " + summary.round(0).to_string().replace("\n", "\n  "))
    print(f"  series with at least one exception: {found[SERIES_KEYS].drop_duplicates().shape[0]} of {len(policy)}")
    print("  top 5:")
    print("  " + found.head(5)[["rank", "item_id", "dc_id", "exception_type", "value_at_risk", "detail"]].round(0)
          .to_string(index=False).replace("\n", "\n  "))
