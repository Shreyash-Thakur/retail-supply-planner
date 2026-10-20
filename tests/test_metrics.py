import numpy as np
import pytest

from src.metrics import bias, fill_rate, fva, mase, mase_scale, wape


def test_wape_and_bias():
    actual, forecast = [10, 20, 0], [12, 18, 1]
    assert wape(actual, forecast) == pytest.approx(5 / 30)
    assert bias(actual, forecast) == pytest.approx(1 / 30)


def test_wape_bias_nan_when_no_demand():
    assert np.isnan(wape([0, 0], [1, 1]))
    assert np.isnan(bias([0, 0], [1, 1]))


def test_nan_actuals_ignored():
    assert wape([10, np.nan], [12, 500]) == pytest.approx(0.2)


def test_volume_weighting_differs_from_average_of_wapes():
    big = wape([100], [90])
    small = wape([1], [3])
    pooled = wape([100, 1], [90, 3])
    assert pooled == pytest.approx(12 / 101)
    assert pooled != pytest.approx((big + small) / 2)


def test_mase_seasonal_and_fallback():
    insample = np.arange(60, dtype=float)  # every seasonal difference is 52
    assert mase_scale(insample, 52) == pytest.approx(52.0)
    assert mase([10, 10], [0, 20], insample) == pytest.approx(10 / 52)
    short = np.array([1.0, 3.0, 2.0])  # fewer than 53 points: lag 1 scale = 1.5
    assert mase_scale(short, 52) == pytest.approx(1.5)
    flat_seasonal = np.tile([1.0, 5.0], 40)  # seasonal differences all 0, lag 1 differences 4
    assert mase_scale(flat_seasonal, 52) == pytest.approx(4.0)
    assert np.isnan(mase([1], [2], np.ones(80)))


def test_fva_sign():
    assert fva(wape_model=0.30, wape_benchmark=0.45) == pytest.approx(15.0)
    assert fva(wape_model=0.50, wape_benchmark=0.45) < 0


def test_fill_rate():
    assert fill_rate([5, 5], [10, 5]) == pytest.approx(10 / 15)
    assert fill_rate([0], [0]) == 1.0
