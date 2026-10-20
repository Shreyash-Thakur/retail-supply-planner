"""Inventory policy per item and DC: classic and quantile safety stock, reorder point, order-up-to level.

The service level here is the cycle service level: the probability of no stockout
in a replenishment cycle. Fill rate (share of demand met) is a different measure;
the simulation reports both.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.allocation import demo_constrained_week
from src.errors import DataQualityError
from src.supply_params import recent_launched_weeks

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]


def safety_stock_classic(z, sigma, lead_time):
    """z x sigma_e x sqrt(L): forecast error over the lead time, assuming independent weekly errors."""
    return np.asarray(z) * np.asarray(sigma) * np.sqrt(np.asarray(lead_time, dtype=float))


def safety_stock_quantile(q_high: np.ndarray, q50: np.ndarray) -> float:
    """sqrt(sum over lead-time weeks of (q_high - q50)^2).

    Weekly quantiles cannot simply be added (the sum of weekly 95th percentiles is
    far above the 95th percentile of the sum), so this treats each week's
    (q_high - q50) as an independent error band and adds them in quadrature.
    """
    gap = np.asarray(q_high, dtype=float) - np.asarray(q50, dtype=float)
    return float(np.sqrt(np.sum(gap ** 2)))


def quantile_column(service_level: float, quantiles: list[float]) -> str:
    matches = [q for q in quantiles if np.isclose(q, service_level)]
    if not matches:
        raise DataQualityError(f"No quantile forecast for service level {service_level}; quantiles are {quantiles}.")
    return f"q{round(matches[0] * 100)}"


def forecast_error_sigma(residuals: pd.DataFrame, gold: pd.DataFrame, min_points: int) -> pd.DataFrame:
    """sigma_e per series: std of the chosen model's backtest residuals at h 1 to 4.

    With fewer than min_points residuals, the std of weekly demand over the last
    52 launched weeks is used instead (sigma_source = demand_std).
    """
    near = residuals[residuals["h"].between(1, 4)]
    err = near.groupby(SERIES_KEYS)["residual"].agg(sigma_e="std", n_errors="size").reset_index()
    demand_std = recent_launched_weeks(gold, 52).groupby(SERIES_KEYS)["units"].std().rename("demand_std").reset_index()
    out = demand_std.merge(err, on=SERIES_KEYS, how="left")
    out["n_errors"] = out["n_errors"].fillna(0).astype(int)
    use_demand = (out["n_errors"] < min_points) | out["sigma_e"].isna()
    out["sigma_source"] = np.where(use_demand, "demand_std", "residuals")
    out["sigma_e"] = np.where(use_demand, out["demand_std"], out["sigma_e"])
    return out[SERIES_KEYS + ["sigma_e", "sigma_source", "n_errors"]]


def compute_policy(
    final_forecast: pd.DataFrame,
    sigma: pd.DataFrame,
    supply: pd.DataFrame,
    items: pd.DataFrame,
    cfg: dict,
    service_level: float | None = None,
    extra_lead_weeks: int = 0,
) -> pd.DataFrame:
    """Policy per series. service_level overrides the ABC levels; extra_lead_weeks lengthens every lead time."""
    inv = cfg["inventory"]
    quantiles = cfg["forecast"]["quantiles"]
    series = (
        final_forecast.drop_duplicates(SERIES_KEYS)[SERIES_KEYS + ["abc", "pattern", "model"]]
        .merge(sigma, on=SERIES_KEYS, validate="one_to_one")
        .merge(supply[SERIES_KEYS + ["opening_stock", "mean_weekly_demand"]], on=SERIES_KEYS, validate="one_to_one")
        .merge(items[["item_id", "lead_time_weeks"]], on="item_id", validate="many_to_one")
    )
    series["service_level"] = service_level if service_level is not None else series["abc"].map(inv["service_level_by_abc"])
    series["lead_time_weeks"] = series["lead_time_weeks"] + extra_lead_weeks
    series["z"] = norm.ppf(series["service_level"])
    series["ss_classic"] = safety_stock_classic(series["z"], series["sigma_e"], series["lead_time_weeks"])

    fc = final_forecast.merge(series[SERIES_KEYS + ["lead_time_weeks", "service_level"]], on=SERIES_KEYS)
    needed_h = series.set_index(SERIES_KEYS)["lead_time_weeks"] + 1
    if (fc.groupby(SERIES_KEYS)["h"].max().reindex(needed_h.index) < needed_h).any():
        raise DataQualityError("Forecast horizon is shorter than lead time + 1 week for some series.")
    column_for = {sl: quantile_column(sl, quantiles) for sl in fc["service_level"].unique()}
    fc["q_high"] = np.select([fc["service_level"] == sl for sl in column_for], [fc[c] for c in column_for.values()])
    in_lead = fc[fc["h"] <= fc["lead_time_weeks"]]
    agg = in_lead.groupby(SERIES_KEYS).agg(lead_time_demand=("forecast", "sum"))
    agg["ss_quantile"] = np.sqrt(((in_lead["q_high"] - in_lead["q50"]) ** 2).groupby([in_lead["item_id"], in_lead["dc_id"]]).sum())
    review = fc[fc["h"] == fc["lead_time_weeks"] + 1].set_index(SERIES_KEYS)["forecast"].rename("review_week_forecast")
    mean_fc = fc.groupby(SERIES_KEYS)["forecast"].mean().rename("mean_weekly_forecast")
    series = series.merge(agg.join(review).join(mean_fc).reset_index(), on=SERIES_KEYS, validate="one_to_one")

    method = inv["safety_stock_method"]
    series["safety_stock"] = series[f"ss_{method}"]
    series["reorder_point"] = series["lead_time_demand"] + series["safety_stock"]
    series["order_up_to"] = series["reorder_point"] + series["review_week_forecast"]
    daily = series["mean_weekly_forecast"] / 7
    series["days_of_cover"] = (series["opening_stock"] / daily).where(daily > 0)
    return series


def run(cfg: dict) -> None:
    """Stage `inventory`: write data/outputs/inventory_policy.parquet."""
    paths = cfg["data"]
    final_forecast = pd.read_parquet(paths["outputs_dir"] / "final_forecast.parquet")
    residuals = pd.read_parquet(paths["outputs_dir"] / "residuals.parquet")
    gold = pd.read_parquet(paths["gold_dir"] / "weekly.parquet")
    supply = pd.read_parquet(paths["gold_dir"] / "supply_params.parquet")
    items = pd.read_parquet(paths["gold_dir"] / "item_params.parquet")

    sigma = forecast_error_sigma(residuals, gold, cfg["inventory"]["min_error_points"])
    policy = compute_policy(final_forecast, sigma, supply, items, cfg)
    policy.to_parquet(paths["outputs_dir"] / "inventory_policy.parquet", index=False)

    print("\nInventory policy (safety stock method: %s)" % cfg["inventory"]["safety_stock_method"])
    print(f"  series: {len(policy)}; sigma source: {policy['sigma_source'].value_counts().to_dict()}")
    summary = policy.groupby("abc").agg(
        series=("item_id", "size"),
        service_level=("service_level", "first"),
        mean_ss_classic=("ss_classic", "mean"),
        mean_ss_quantile=("ss_quantile", "mean"),
        mean_reorder_point=("reorder_point", "mean"),
        mean_order_up_to=("order_up_to", "mean"),
        mean_days_of_cover=("days_of_cover", "mean"),
        mean_weekly_forecast=("mean_weekly_forecast", "mean"),
    )
    print(summary.round(2).to_string())
    print(f"  total safety stock: classic {policy['ss_classic'].sum():.0f}, quantile {policy['ss_quantile'].sum():.0f} units")
    print(f"  series with no positive forecast (days of cover undefined): {int(policy['days_of_cover'].isna().sum())}")

    demo = demo_constrained_week(policy, supply, items, cfg)
    demo.to_csv(paths["outputs_dir"] / "allocation_demo.csv", index=False)
    s = demo.attrs["summary"]
    print("\nAllocation demo: one constrained week (opening stock on hand, empty pipeline, "
          f"supplier capacity x {cfg['scenarios']['supplier_disruption']['capacity_factor']})")
    print(f"  requested {s['requested']} units, supply {s['supply']} units, {s['constrained_items']} items constrained")
    by_abc = demo.groupby("abc")[["request", "lp", "fair_share"]].sum()
    by_abc["lp_fill"] = by_abc["lp"] / by_abc["request"]
    by_abc["fair_fill"] = by_abc["fair_share"] / by_abc["request"]
    print("  " + by_abc.round(3).to_string().replace("\n", "\n  "))
    print(f"  weighted shortfall: LP {s['lp_weighted_shortfall']:.0f}, fair share {s['fair_weighted_shortfall']:.0f} "
          f"({1 - s['lp_weighted_shortfall'] / s['fair_weighted_shortfall']:.1%} lower with LP)")
