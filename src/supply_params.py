"""Simulated supply parameters. M5 has no supply data, so every value here is simulated
from the real demand history using config.yaml settings and the seeded generator."""

import logging

import numpy as np
import pandas as pd

from src.errors import DataQualityError

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]


def _ceil_units(values: pd.Series | np.ndarray) -> np.ndarray:
    """Round up to whole units, ignoring floating point noise (1.1 x 50 is 55, not 56)."""
    return np.ceil(np.round(np.asarray(values, dtype=float), 9)).astype("int64")


def recent_launched_weeks(gold: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """The last n_weeks launched rows of each item and DC (fewer if the series is shorter)."""
    launched = gold[gold["is_launched"]].sort_values(SERIES_KEYS + ["t"])
    return launched.groupby(SERIES_KEYS).tail(n_weeks)


def compute_supply_params(
    gold: pd.DataFrame, supply_cfg: dict, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (supply_params per item and DC, item_params per item).

    supply_params: mean_weekly_demand (at least 1 unit), history_weeks_used,
    opening_stock, dc_capacity (units per DC, shared by all items).
    item_params: lead_time_weeks (same for every DC), supplier_capacity per week,
    unit_price, unit_cost, weekly_holding_cost.
    """
    recent = recent_launched_weeks(gold, supply_cfg["history_weeks"])
    if recent.empty:
        raise DataQualityError("No launched weeks found; cannot compute supply parameters.")

    supply = (
        recent.groupby(SERIES_KEYS)
        .agg(mean_weekly_demand=("units", "mean"), history_weeks_used=("units", "size"))
        .reset_index()
    )
    supply["mean_weekly_demand"] = supply["mean_weekly_demand"].clip(lower=1.0)
    supply["opening_stock"] = np.round(supply_cfg["opening_stock_weeks"] * supply["mean_weekly_demand"]).astype("int64")
    dc_capacity = supply.groupby("dc_id")["mean_weekly_demand"].sum() * supply_cfg["dc_capacity_weeks"]
    supply["dc_capacity"] = _ceil_units(supply["dc_id"].map(dc_capacity))

    items = pd.DataFrame({"item_id": sorted(supply["item_id"].unique())})
    items["lead_time_weeks"] = rng.integers(
        supply_cfg["lead_time_weeks_min"], supply_cfg["lead_time_weeks_max"], size=len(items), endpoint=True
    )
    demand_all_dcs = supply.groupby("item_id")["mean_weekly_demand"].sum()
    items["supplier_capacity"] = _ceil_units(items["item_id"].map(demand_all_dcs) * supply_cfg["supplier_capacity_factor"])
    items["unit_price"] = items["item_id"].map(recent.groupby("item_id")["price"].median())
    items["unit_cost"] = supply_cfg["unit_cost_share_of_price"] * items["unit_price"]
    items["weekly_holding_cost"] = items["unit_cost"] * supply_cfg["annual_holding_cost_rate"] / 52

    bad_price = items[~(items["unit_price"] > 0)]
    if len(bad_price):
        raise DataQualityError(f"{len(bad_price)} items have no positive price:\n{bad_price.head(5).to_string()}")
    return supply, items


def run(cfg: dict) -> None:
    """Stage `supply`: write data/gold/supply_params.parquet and item_params.parquet."""
    gold_dir = cfg["data"]["gold_dir"]
    gold = pd.read_parquet(gold_dir / "weekly.parquet")
    supply, items = compute_supply_params(gold, cfg["supply"], np.random.default_rng(cfg["seed"]))
    supply.to_parquet(gold_dir / "supply_params.parquet", index=False)
    items.to_parquet(gold_dir / "item_params.parquet", index=False)
    logger.info("Wrote %d item-DC rows and %d item rows of simulated supply parameters", len(supply), len(items))

    short = int((supply["history_weeks_used"] < cfg["supply"]["history_weeks"]).sum())
    floored = int((supply["mean_weekly_demand"] == 1.0).sum())
    print("\nSupply parameters (simulated)")
    print(f"  series: {len(supply)}, items: {len(items)}")
    print(f"  series with fewer than {cfg['supply']['history_weeks']} launched weeks: {short}; mean demand floored at 1 unit: {floored}")
    print(f"  lead time distribution (weeks: items): {items['lead_time_weeks'].value_counts().sort_index().to_dict()}")
    print(f"  total supplier capacity: {int(items['supplier_capacity'].sum())} units per week")
    print(f"  total mean weekly demand: {supply['mean_weekly_demand'].sum():.1f} units")
    print(f"  DC capacities (units): {supply.drop_duplicates('dc_id').set_index('dc_id')['dc_capacity'].to_dict()}")
    print(f"  total opening stock: {int(supply['opening_stock'].sum())} units")
    print(f"  unit price: median {items['unit_price'].median():.2f}, range {items['unit_price'].min():.2f} to {items['unit_price'].max():.2f}")
