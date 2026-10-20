import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.inventory import compute_policy, forecast_error_sigma, safety_stock_classic, safety_stock_quantile

CFG = {
    "forecast": {"quantiles": [0.5, 0.9, 0.95, 0.98]},
    "inventory": {"service_level_by_abc": {"A": 0.98, "B": 0.95, "C": 0.90}, "min_error_points": 6,
                  "safety_stock_method": "classic"},
}


def test_classic_safety_stock_reference_value():
    z = norm.ppf(0.95)
    assert z == pytest.approx(1.6449, abs=1e-4)
    assert safety_stock_classic(z, 10, 4) == pytest.approx(32.90, abs=0.01)


def test_higher_service_level_gives_higher_safety_stock():
    levels = [0.90, 0.95, 0.98]
    ss = [safety_stock_classic(norm.ppf(sl), 10, 3) for sl in levels]
    assert ss == sorted(ss) and len(set(ss)) == 3


def test_quantile_safety_stock_zero_when_quantiles_equal_median():
    q50 = np.array([5.0, 6.0, 7.0])
    assert safety_stock_quantile(q50, q50) == 0.0
    assert safety_stock_quantile(q50 + np.array([3.0, 4.0, 0.0]), q50) == pytest.approx(5.0)


def make_inputs(lead_time: int = 2):
    rows = []
    for item, dc, abc in [("I1", "CA", "A"), ("I2", "CA", "C")]:
        for h in range(1, 13):
            rows.append({"item_id": item, "dc_id": dc, "abc": abc, "pattern": "smooth", "model": "naive",
                         "h": h, "t": 100 + h, "forecast": 10.0, "q50": 10.0, "q90": 13.0, "q95": 14.0, "q98": 16.0})
    final = pd.DataFrame(rows)
    sigma = pd.DataFrame({"item_id": ["I1", "I2"], "dc_id": ["CA", "CA"], "sigma_e": [4.0, 4.0],
                          "sigma_source": "residuals", "n_errors": 12})
    supply = pd.DataFrame({"item_id": ["I1", "I2"], "dc_id": ["CA", "CA"], "opening_stock": [30, 30], "mean_weekly_demand": [10.0, 10.0]})
    items = pd.DataFrame({"item_id": ["I1", "I2"], "lead_time_weeks": [lead_time, lead_time]})
    return final, sigma, supply, items


def test_policy_formulas():
    policy = compute_policy(*make_inputs(lead_time=2), CFG).set_index("item_id")
    a = policy.loc["I1"]
    assert a["service_level"] == 0.98
    assert a["ss_classic"] == pytest.approx(norm.ppf(0.98) * 4 * np.sqrt(2))
    assert a["ss_quantile"] == pytest.approx(np.sqrt(2 * 6.0 ** 2))  # q98 - q50 = 6 for 2 weeks
    assert a["lead_time_demand"] == pytest.approx(20.0)
    assert a["reorder_point"] == pytest.approx(20.0 + a["ss_classic"])
    assert a["order_up_to"] == pytest.approx(a["reorder_point"] + 10.0)
    assert a["days_of_cover"] == pytest.approx(21.0)
    c = policy.loc["I2"]
    assert c["ss_quantile"] == pytest.approx(np.sqrt(2 * 3.0 ** 2))  # C uses q90
    assert c["ss_classic"] < a["ss_classic"]


def test_policy_independent_of_row_order():
    final, sigma, supply, items = make_inputs(lead_time=2)
    shuffled = compute_policy(final.iloc[::-1], sigma.iloc[::-1], supply, items, CFG)
    ordered = compute_policy(final, sigma, supply, items, CFG)
    cols = ["item_id", "ss_classic", "ss_quantile", "reorder_point", "order_up_to"]
    pd.testing.assert_frame_equal(
        shuffled[cols].sort_values("item_id", ignore_index=True), ordered[cols].sort_values("item_id", ignore_index=True)
    )


def test_overrides_for_scenarios():
    inputs = make_inputs(lead_time=2)
    flat = compute_policy(*inputs, CFG, service_level=0.98).set_index("item_id")
    assert (flat["service_level"] == 0.98).all()
    longer = compute_policy(*inputs, CFG, extra_lead_weeks=2).set_index("item_id")
    assert (longer["lead_time_weeks"] == 4).all()
    assert longer.loc["I1", "lead_time_demand"] == pytest.approx(40.0)


def test_sigma_falls_back_to_demand_std_when_few_residuals():
    gold = pd.DataFrame({"item_id": "I1", "dc_id": "CA", "t": range(60), "units": [10, 20] * 30, "is_launched": True})
    residuals = pd.DataFrame({"item_id": "I1", "dc_id": "CA", "h": [1, 2, 3], "residual": [1.0, -1.0, 2.0]})
    out = forecast_error_sigma(residuals, gold, min_points=6).iloc[0]
    assert out["sigma_source"] == "demand_std"
    assert out["sigma_e"] == pytest.approx(pd.Series([10, 20] * 26).std())
