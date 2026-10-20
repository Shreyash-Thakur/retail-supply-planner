import numpy as np
import pandas as pd
import pytest

from src.supply_params import compute_supply_params

SUPPLY_CFG = {
    "history_weeks": 52,
    "lead_time_weeks_min": 1,
    "lead_time_weeks_max": 4,
    "supplier_capacity_factor": 1.10,
    "opening_stock_weeks": 3,
    "dc_capacity_weeks": 8,
    "unit_cost_share_of_price": 0.60,
    "annual_holding_cost_rate": 0.25,
}


def make_gold(series: dict[tuple[str, str], list], price: float = 2.0) -> pd.DataFrame:
    rows = []
    for (item, dc), units in series.items():
        rows += [
            {"item_id": item, "dc_id": dc, "t": t, "units": u, "price": price, "is_launched": True}
            for t, u in enumerate(units)
        ]
    return pd.DataFrame(rows)


def test_capacity_rounding_and_derived_costs():
    gold = make_gold({("I1", "CA"): [10] * 60, ("I1", "TX"): [0.5] * 60})
    supply, items = compute_supply_params(gold, SUPPLY_CFG, np.random.default_rng(0))

    by_dc = supply.set_index("dc_id")
    assert by_dc.loc["CA", "mean_weekly_demand"] == 10.0
    assert by_dc.loc["TX", "mean_weekly_demand"] == 1.0  # floored at 1 unit
    assert by_dc.loc["CA", "opening_stock"] == 30
    assert by_dc.loc["CA", "history_weeks_used"] == 52
    assert items.loc[0, "supplier_capacity"] == 13  # ceil(1.1 x 11) = ceil(12.1)
    assert by_dc.loc["CA", "dc_capacity"] == 80  # 8 weeks x 10 units
    assert items.loc[0, "unit_cost"] == pytest.approx(1.2)
    assert items.loc[0, "weekly_holding_cost"] == pytest.approx(1.2 * 0.25 / 52)


def test_supplier_capacity_exact_multiple_not_rounded_up():
    gold = make_gold({("I1", "CA"): [50] * 52})
    _, items = compute_supply_params(gold, SUPPLY_CFG, np.random.default_rng(0))
    assert items.loc[0, "supplier_capacity"] == 55  # 1.1 x 50 is 55.00000000000001 in floating point


def test_uses_only_last_launched_weeks():
    gold = make_gold({("I1", "CA"): [100] * 10 + [4] * 52})
    gold.loc[gold["t"] < 3, "is_launched"] = False
    supply, _ = compute_supply_params(gold, SUPPLY_CFG, np.random.default_rng(0))
    assert supply.loc[0, "mean_weekly_demand"] == 4.0


def test_lead_times_in_range_same_per_item_and_reproducible():
    series = {(f"I{i}", dc): [5] * 10 for i in range(30) for dc in ("CA", "TX", "WI")}
    gold = make_gold(series)
    supply_a, items_a = compute_supply_params(gold, SUPPLY_CFG, np.random.default_rng(42))
    supply_b, items_b = compute_supply_params(gold, SUPPLY_CFG, np.random.default_rng(42))

    pd.testing.assert_frame_equal(items_a, items_b)
    pd.testing.assert_frame_equal(supply_a, supply_b)
    assert items_a["lead_time_weeks"].between(1, 4).all()
    assert items_a["item_id"].is_unique
    assert set(items_a["lead_time_weeks"]) == {1, 2, 3, 4}  # endpoint included
