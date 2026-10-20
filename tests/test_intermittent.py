import numpy as np
import pytest

from src.forecast_stat import croston_forecast, tsb_forecast


def test_croston_regular_intermittent_gives_one():
    assert croston_forecast(np.array([0, 0, 3, 0, 0, 3, 0, 0, 3]), alpha=0.1) == pytest.approx(1.0)


def test_croston_all_zeros_gives_zero():
    assert croston_forecast(np.zeros(20), alpha=0.1) == 0.0


def test_tsb_decays_after_demand_stops():
    history = np.array([0, 4, 0, 0, 5, 0, 3, 0, 0, 6] * 3, dtype=float)
    with_zeros = np.concatenate([history, np.zeros(20)])
    assert tsb_forecast(with_zeros, alpha=0.1, beta=0.1) < tsb_forecast(history, alpha=0.1, beta=0.1)


def test_tsb_all_zeros_gives_zero():
    assert tsb_forecast(np.zeros(10), alpha=0.1, beta=0.1) == 0.0


def test_croston_does_not_decay_after_demand_stops():
    # The contrast with TSB: Croston only updates on non-zero weeks.
    history = np.array([0, 4, 0, 0, 5, 0, 3, 0, 0, 6] * 3, dtype=float)
    assert croston_forecast(np.concatenate([history, np.zeros(20)]), 0.1) == croston_forecast(history, 0.1)
