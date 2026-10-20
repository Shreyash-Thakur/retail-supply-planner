import numpy as np
import pytest

from src.allocation import (
    _greedy_by_item,
    _solve_lp,
    allocate_fair_share,
    allocate_lp,
    allocate_two_tier,
    allocation_weights,
    weighted_shortfall,
)

METHODS = [
    lambda r, s, c, w: allocate_lp(r, s, c, w),
    lambda r, s, c, w: allocate_fair_share(r, s, c, w),
]


@pytest.mark.parametrize("allocate", METHODS, ids=["lp", "fair_share"])
def test_supply_limited_single_item(allocate):
    requests = np.array([[80, 60, 0]])
    weights = np.array([[2.0, 1.0, 3.0]])
    alloc = allocate(requests, np.array([100]), np.array([1000, 1000, 1000]), weights)
    assert alloc.sum() <= 100
    assert (alloc <= requests).all() and (alloc >= 0).all()
    assert alloc[0, 2] == 0
    assert alloc.sum() == 100


@pytest.mark.parametrize("allocate", METHODS, ids=["lp", "fair_share"])
def test_ample_supply_meets_every_request(allocate):
    requests = np.array([[10, 20, 5], [0, 7, 3]])
    alloc = allocate(requests, np.array([100, 100]), np.array([500, 500, 500]), np.ones((2, 3)))
    np.testing.assert_array_equal(alloc, requests)


@pytest.mark.parametrize("allocate", METHODS, ids=["lp", "fair_share"])
def test_dc_capacity_binding(allocate):
    requests = np.array([[50, 10, 0], [40, 5, 0]])
    dc_free = np.array([30, 100, 100])
    alloc = allocate(requests, np.array([100, 100]), dc_free, np.ones((2, 3)))
    assert alloc[:, 0].sum() <= 30
    assert (alloc <= requests).all()


def test_lp_beats_fair_share_when_a_and_c_items_compete():
    requests = np.array([[60, 0, 0], [60, 0, 0]])  # an A item and a C item both want DC 1
    dc_free = np.array([60, 100, 100])  # DC 1 only has room for 60 units
    weights = allocation_weights(np.array([["A"] * 3, ["C"] * 3]), np.zeros((2, 3)), {"A": 3.0, "B": 2.0, "C": 1.0})
    supply = np.array([100, 100])
    lp = allocate_lp(requests, supply, dc_free, weights)
    fair = allocate_fair_share(requests, supply, dc_free, weights)
    assert lp[0, 0] == 60 and lp[1, 0] == 0
    assert weighted_shortfall(requests, lp, weights) < weighted_shortfall(requests, fair, weights)


def test_weights_favour_a_and_low_cover():
    w = allocation_weights(np.array([["A", "A"], ["C", "C"]]), np.array([[0.0, 4.0], [0.0, 4.0]]), {"A": 3.0, "C": 1.0})
    assert w[0, 0] > w[0, 1] > w[1, 0] > w[1, 1]
    assert w[0, 0] == pytest.approx(6.0)


@pytest.mark.parametrize("allocate", METHODS, ids=["lp", "fair_share"])
def test_random_cases_respect_every_constraint_and_return_integers(allocate):
    rng = np.random.default_rng(7)
    for _ in range(50):
        requests = rng.integers(0, 40, size=(8, 3))
        supply = rng.integers(0, 80, size=8)
        dc_free = rng.integers(0, 120, size=3)
        weights = rng.uniform(1, 6, size=(8, 3))
        alloc = allocate(requests, supply, dc_free, weights)
        assert alloc.dtype.kind == "i"
        assert (alloc >= 0).all() and (alloc <= requests).all()
        assert (alloc.sum(axis=1) <= supply).all()
        assert (alloc.sum(axis=0) <= dc_free).all()


def test_lp_never_worse_than_fair_share_on_random_cases():
    rng = np.random.default_rng(11)
    for _ in range(50):
        requests = rng.integers(0, 40, size=(6, 3))
        supply = rng.integers(0, 60, size=6)
        dc_free = rng.integers(0, 90, size=3)
        weights = rng.uniform(1, 6, size=(6, 3))
        lp = allocate_lp(requests, supply, dc_free, weights)
        fair = allocate_fair_share(requests, supply, dc_free, weights)
        assert weighted_shortfall(requests, lp, weights) <= weighted_shortfall(requests, fair, weights) + 1e-6


def test_greedy_shortcut_matches_lp_when_dc_capacity_does_not_bind():
    rng = np.random.default_rng(3)
    for _ in range(50):
        requests = rng.integers(0, 40, size=(10, 3))
        supply = rng.integers(0, 90, size=10)
        dc_free = requests.sum(axis=0) + 5  # DC capacity slack
        weights = rng.uniform(1, 6, size=(10, 3))
        greedy = _greedy_by_item(requests, supply, weights)
        lp = _solve_lp(requests, supply, dc_free, weights)
        assert np.sum(weights * greedy) == pytest.approx(np.sum(weights * lp), rel=1e-9)


def test_two_tier_covers_expected_demand_everywhere_before_safety_stock():
    requests = np.array([[100, 50, 0]])  # DC 1: 20 units of expected demand + 80 safety stock top-up; DC 2: 50 expected demand
    cycle_need = np.array([[20, 50, 0]])
    weights = np.array([[5.0, 4.0, 3.0]])  # DC 1 has the higher weight
    dc_free = np.array([1000, 1000, 1000])
    plain = allocate_lp(requests, np.array([60]), dc_free, weights)
    tiered = allocate_two_tier(requests, cycle_need, np.array([60]), dc_free, weights, tier1_factor=10.0)
    assert plain.tolist() == [[60, 0, 0]]
    assert tiered.tolist() == [[20, 40, 0]]


@pytest.mark.parametrize("slack", [True, False], ids=["dc_slack", "dc_binding"])
def test_two_tier_respects_every_constraint(slack):
    rng = np.random.default_rng(5)
    for _ in range(50):
        requests = rng.integers(0, 40, size=(8, 3))
        cycle_need = rng.integers(-10, 40, size=(8, 3)).astype(float)
        supply = rng.integers(0, 80, size=8)
        dc_free = requests.sum(axis=0) + 5 if slack else rng.integers(0, 60, size=3)
        alloc = allocate_two_tier(requests, cycle_need, supply, dc_free, rng.uniform(1, 6, size=(8, 3)), 10.0)
        assert alloc.dtype.kind == "i"
        assert (alloc >= 0).all() and (alloc <= requests).all()
        assert (alloc.sum(axis=1) <= supply).all()
        assert (alloc.sum(axis=0) <= dc_free).all()


def test_two_tier_greedy_matches_solver_when_dc_capacity_does_not_bind():
    rng = np.random.default_rng(9)
    for _ in range(30):
        requests = rng.integers(0, 40, size=(6, 3))
        tier1 = np.minimum(requests, rng.integers(0, 40, size=(6, 3)))
        split = np.concatenate([tier1, requests - tier1], axis=1)
        w = rng.uniform(1, 6, size=(6, 3))
        split_w = np.concatenate([w * 10, w], axis=1)
        supply = rng.integers(0, 90, size=6)
        greedy = _greedy_by_item(split, supply, split_w)
        lp = _solve_lp(split, supply, requests.sum(axis=0) + 5, split_w, np.tile(np.arange(3), 2))
        assert np.sum(split_w * greedy) == pytest.approx(np.sum(split_w * lp), rel=1e-9)
