import pandas as pd

from src.exceptions import demand_anomaly, forecast_bias, overstock, stockout_risk

PRICE = pd.Series({"OK": 2.0, "BAD": 2.0})


def test_stockout_risk_flags_only_series_below_safety_stock_within_4_weeks():
    rows = []
    for week in range(1, 7):
        rows.append({"item_id": "OK", "dc_id": "CA", "week": week, "on_hand": 50, "lost": 0, "safety_stock": 20, "unit_price": 2.0, "abc": "A"})
        rows.append({"item_id": "BAD", "dc_id": "CA", "week": week, "on_hand": 10 if week >= 3 else 40, "lost": 5 if week == 4 else 0,
                     "safety_stock": 20, "unit_price": 2.0, "abc": "B"})
        rows.append({"item_id": "LATE", "dc_id": "CA", "week": week, "on_hand": 5 if week == 6 else 40, "lost": 0,
                     "safety_stock": 20, "unit_price": 2.0, "abc": "C"})
    out = stockout_risk(pd.DataFrame(rows), weeks=4)
    assert out["item_id"].tolist() == ["BAD"]
    assert out["value_at_risk"].iloc[0] == 10.0  # 5 lost units x 2.0


def test_overstock_flags_long_cover_and_zero_forecast_stock():
    policy = pd.DataFrame({
        "item_id": ["OK", "LONG", "ZERO"], "dc_id": "CA", "abc": "C",
        "days_of_cover": [20.0, 90.0, None], "opening_stock": [30, 90, 12], "mean_weekly_forecast": [10.5, 7.0, 0.0],
    })
    out = overstock(policy, pd.Series({"OK": 1.0, "LONG": 1.0, "ZERO": 1.0}), max_days=60).set_index("item_id")
    assert sorted(out.index) == ["LONG", "ZERO"]
    assert out.loc["LONG", "value_at_risk"] == 30.0  # 90 - 60 days x 1 unit per day
    assert out.loc["ZERO", "value_at_risk"] == 12.0


def test_forecast_bias_threshold():
    fold = pd.DataFrame({"item_id": ["OK"] * 2 + ["BAD"] * 2, "dc_id": "CA", "abc": "A",
                         "forecast": [11, 10, 30, 30], "actual": [10, 10, 20, 20]})
    out = forecast_bias(fold, PRICE, threshold=0.20)
    assert out["item_id"].tolist() == ["BAD"]
    assert out["value_at_risk"].iloc[0] == 40.0  # |60 - 40| x 2.0


def test_demand_anomaly_outside_residual_range():
    residuals = pd.DataFrame({"item_id": ["OK"] * 21 + ["BAD"] * 21, "dc_id": "CA", "residual": list(range(-10, 11)) * 2})
    last = pd.DataFrame({"item_id": ["OK", "BAD"], "dc_id": "CA", "abc": "A", "forecast": [100, 100], "actual": [105, 150]})
    out = demand_anomaly(last, residuals, PRICE)
    assert out["item_id"].tolist() == ["BAD"]
    assert out["value_at_risk"].iloc[0] == (150 - 109) * 2.0  # 95th percentile of -10..10 is 9


def test_each_case_triggers_exactly_its_own_type():
    stock = stockout_risk(pd.DataFrame([{"item_id": "S", "dc_id": "CA", "week": 1, "on_hand": 0, "lost": 3, "safety_stock": 5,
                                         "unit_price": 1.0, "abc": "A"}]))
    over = overstock(pd.DataFrame({"item_id": ["S"], "dc_id": "CA", "abc": "A", "days_of_cover": [10.0], "opening_stock": [5],
                                   "mean_weekly_forecast": [3.5]}), pd.Series({"S": 1.0}))
    assert stock["exception_type"].tolist() == ["stockout_risk"]
    assert over.empty
