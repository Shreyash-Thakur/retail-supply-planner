import numpy as np
import pandas as pd
import pytest

from src.segment import abc_classes, fold_origins, segment

CFG = {"abc_cutoffs": [0.80, 0.95], "xyz_cutoffs": [0.5, 1.0], "adi_cutoff": 1.32, "cv2_cutoff": 0.49, "window_weeks": 52}


def make_weekly(series: dict[tuple[str, str], list], price: float = 1.0) -> pd.DataFrame:
    rows = []
    for (item, dc), units in series.items():
        rows += [
            {"item_id": item, "dc_id": dc, "t": t, "units": u, "price": price, "is_launched": True}
            for t, u in enumerate(units)
        ]
    return pd.DataFrame(rows)


def one(units: list) -> pd.Series:
    seg = segment(make_weekly({("I1", "CA"): units}), origin=len(units) - 1, cfg=CFG)
    return seg.iloc[0]


def test_constant_series_is_smooth_x():
    row = one([5] * 52)
    assert row["pattern"] == "smooth"
    assert row["xyz"] == "X"
    assert row["cv"] == 0


def test_regular_intermittent():
    row = one([0, 0, 0, 10] * 13)
    assert row["adi"] == pytest.approx(4.0)
    assert row["cv2"] == pytest.approx(0.0)
    assert row["pattern"] == "intermittent"


def test_lumpy():
    row = one([0, 0, 0, 1] * 6 + [0, 0, 0, 40] * 7)
    assert row["adi"] == pytest.approx(4.0)
    assert row["cv2"] > 0.49
    assert row["pattern"] == "lumpy"


def test_inactive_and_single_nonzero_week():
    assert one([0] * 51 + [0])["pattern"] == "inactive"
    single = one([0] * 51 + [7])
    assert single["cv2"] == 0
    assert single["pattern"] == "intermittent"


def test_zero_mean_is_z():
    row = one([0] * 52)
    assert row["xyz"] == "Z"


def test_abc_rule_on_constructed_revenue():
    revenue = pd.Series({"a": 50.0, "b": 30.0, "c": 10.0, "d": 6.0, "e": 4.0})
    # share before: 0, 0.50, 0.80, 0.90, 0.96
    assert abc_classes(revenue, [0.80, 0.95]).to_dict() == {"a": "A", "b": "A", "c": "B", "d": "B", "e": "C"}


def test_abc_series_crossing_cutoff_is_a():
    revenue = pd.Series({"a": 50.0, "b": 28.0, "c": 10.0, "d": 8.0, "e": 4.0})
    # share before: 0, 0.50, 0.78 (c crosses 0.80, so stays A), 0.88, 0.96
    assert abc_classes(revenue, [0.80, 0.95]).to_dict() == {"a": "A", "b": "A", "c": "A", "d": "B", "e": "C"}


def test_window_uses_last_52_launched_weeks_only():
    row = one([100, 0, 300] + [5] * 52)
    assert row["pattern"] == "smooth" and row["cv"] == 0


def test_fold_origins():
    assert fold_origins(276, 12, 3) == {"fold_1": 240, "fold_2": 252, "fold_3": 264}


def test_fold_segments_never_use_weeks_after_origin():
    rng = np.random.default_rng(0)
    series = {(f"I{i}", dc): rng.poisson(rng.uniform(0.2, 20), size=120).tolist() for i in range(12) for dc in ("CA", "TX")}
    weekly = make_weekly(series)
    weekly["price"] = rng.uniform(1, 5, size=len(weekly))
    origin = 80
    before = segment(weekly, origin, CFG)

    changed = weekly.copy()
    future = changed["t"] > origin
    changed.loc[future, "units"] = rng.integers(0, 1000, size=future.sum())
    changed.loc[future, "price"] = 999.0
    after = segment(changed, origin, CFG)

    pd.testing.assert_frame_equal(before, after)
