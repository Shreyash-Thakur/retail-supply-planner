import numpy as np
import pandas as pd
import pytest

from src.simulate import Scenario, SimInputs, draw_noise, residual_pools, run_kpis, simulate, summarise

ABC_WEIGHT = {"A": 3.0, "B": 2.0, "C": 1.0}
H = 12


def make_inputs(supplier_capacity: int = 10_000, opening: int = 100, lead_time: int = 1, forecast: float = 10.0) -> SimInputs:
    n_items, n_dcs = 2, 3
    return SimInputs(
        item_ids=["I1", "I2"], dc_ids=["CA", "TX", "WI"],
        abc=np.array([["A", "B", "C"], ["C", "A", "B"]]),
        forecast=np.full((n_items, n_dcs, H), forecast),
        order_up_to=np.full((n_items, n_dcs), 60.0),
        lead_time=np.full(n_items, lead_time),
        supplier_capacity=np.full(n_items, supplier_capacity),
        dc_capacity=np.full(n_dcs, 1_000_000),
        opening_stock=np.full((n_items, n_dcs), opening),
        pipeline_per_week=np.full((n_items, n_dcs), int(forecast)),
        mean_weekly_forecast=np.full((n_items, n_dcs), forecast),
        unit_cost=np.array([1.0, 2.0]), unit_price=np.array([2.0, 3.0]), weekly_holding_cost=np.array([0.01, 0.02]),
    )


def test_deterministic_demand_with_ample_supply_has_full_fill_rate():
    inputs = make_inputs()
    noise = np.zeros((5, 2, 3, H))
    results = simulate(inputs, noise, Scenario("t"), ABC_WEIGHT)  # raises if stock balance breaks
    kpis = run_kpis(results, inputs)
    assert (kpis["fill_rate"] == 1.0).all()
    assert (results["demand"] == 10).all()
    assert (results["on_hand"] >= 0).all()


def test_zero_supplier_capacity_fill_rate_falls_and_stock_never_negative():
    inputs = make_inputs(supplier_capacity=0, opening=30)
    results = simulate(inputs, np.zeros((3, 2, 3, H)), Scenario("t"), ABC_WEIGHT)
    weekly_fill = results["sold"].sum(axis=(0, 2, 3)) / results["demand"].sum(axis=(0, 2, 3))
    assert weekly_fill[0] == 1.0
    assert weekly_fill[-1] == 0.0
    assert (np.diff(weekly_fill) <= 0).all()
    assert (results["on_hand"] >= 0).all()
    assert (results["shipped"] == 0).all()


def test_same_seed_gives_identical_kpis():
    inputs = make_inputs(supplier_capacity=25, opening=20)
    pools = [[np.array([-8.0, -3.0, 0.0, 4.0, 9.0])] * 3 for _ in range(6)]
    kpi = []
    for _ in range(2):
        noise = draw_noise(pools, 2, 3, [4, 8, 12], runs=20, rng=np.random.default_rng(42))
        kpi.append(summarise(run_kpis(simulate(inputs, noise, Scenario("t"), ABC_WEIGHT), inputs)))
    pd.testing.assert_frame_equal(kpi[0], kpi[1])


def test_lost_sales_not_backordered():
    inputs = make_inputs(supplier_capacity=0, opening=5)
    results = simulate(inputs, np.zeros((1, 2, 3, H)), Scenario("t"), ABC_WEIGHT)
    assert (results["demand"] == 10).all()  # demand does not grow with past lost sales
    # Week 1: 5 opening + 10 pipeline = 15, sell 10. Week 2: 5 left, lose 5. Weeks 3 to 12: lose 10 each.
    lost_per_series = results["lost"][0].sum(axis=0)
    assert (lost_per_series == 5 + 10 * 10).all()
    assert (results["lost"][0, 0] == 0).all()


def test_demand_multiplier_and_capacity_factor_apply_only_in_their_weeks():
    inputs = make_inputs()
    multiplier = np.ones((H, 2, 3))
    multiplier[1:5, 0, 0] = 1.5
    results = simulate(inputs, np.zeros((1, 2, 3, H)), Scenario("surge", demand_multiplier=multiplier), ABC_WEIGHT)
    assert (results["demand"][0, 1:5, 0, 0] == 15).all()
    assert results["demand"][0, 0, 0, 0] == 10 and (results["demand"][0, :, 1, :] == 10).all()

    cut = simulate(make_inputs(supplier_capacity=12), np.zeros((1, 2, 3, H)),
                   Scenario("cut", capacity_factor=np.where(np.arange(H) == 3, 0.0, 1.0)), ABC_WEIGHT)
    assert cut["shipped"][0, 3].sum() == 0


def test_longer_lead_time_delays_new_arrivals():
    base = simulate(make_inputs(opening=20), np.zeros((1, 2, 3, H)), Scenario("b"), ABC_WEIGHT)
    shock = simulate(make_inputs(opening=20), np.zeros((1, 2, 3, H)), Scenario("s", extra_lead_weeks=2), ABC_WEIGHT)
    first_order_arrival = lambda r: np.argmax(r["arrivals"][0, 1:, 0, 0] > 10) + 2  # noqa: E731
    assert first_order_arrival(shock) == first_order_arrival(base) + 2


def test_residual_pools_fall_back_to_scaled_pattern_pool():
    residuals = pd.DataFrame({
        "item_id": ["I1"] * 12 + ["I2"] * 2, "dc_id": "CA",
        "h": list(range(1, 13)) + [1, 2], "residual": [2.0] * 12 + [99.0, 99.0],
    })
    series = pd.DataFrame({"item_id": ["I1", "I2"], "dc_id": ["CA", "CA"], "pattern": ["smooth", "smooth"],
                           "mean_weekly_demand": [10.0, 30.0]})
    pools = residual_pools(residuals, series, [4, 8, 12], min_n=3)
    assert np.array_equal(pools[0][0], np.full(4, 2.0))
    # I2 has only 2 residuals in h 1 to 4: pooled smooth residuals scaled by 30 / mean(10, 30) = 1.5
    assert pools[1][0] == pytest.approx(np.array([2.0] * 4 + [99.0, 99.0]) * 1.5)
