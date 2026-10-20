import numpy as np
import pandas as pd
import pytest

from src.features import build_features, history_feature_columns, split_at_origin


def make_weekly(n_weeks: int, launch_t: int = 0) -> pd.DataFrame:
    """Two series where units equal t, so any feature value reveals which week it came from."""
    rows = []
    for item in ("I1", "I2"):
        for t in range(n_weeks):
            rows.append({
                "item_id": item, "dc_id": "CA", "t": t, "week_id": 11101 + t, "units": float(t),
                "price": 1.0 + 0.01 * t, "is_launched": t >= launch_t,
                "event_days": 0, "event_sporting": 0, "event_cultural": 0, "event_national": 0,
                "event_religious": 0, "snap_days": 0, "week_of_year": 1 + t % 52, "year": 2011 + t // 52,
            })
    return pd.DataFrame(rows)


SEGMENTS = pd.DataFrame({"item_id": ["I1", "I2"], "dc_id": ["CA", "CA"], "abc": ["A", "C"], "pattern": ["smooth", "smooth"]})


@pytest.mark.parametrize("min_lag, h_range", [(4, range(1, 5)), (8, range(5, 9)), (12, range(9, 13))])
def test_no_feature_uses_units_after_origin(min_lag, h_range):
    origin = 150
    history, future, actuals = split_at_origin(make_weekly(200), SEGMENTS, origin, horizon=12)
    assert "units" not in future.columns
    assert actuals["t"].between(origin + 1, origin + 12).all()

    panel = build_features(history, future, min_lag)
    targets = panel[panel["is_future"] & (panel["t"] - origin).isin(list(h_range))]
    assert len(targets) == 2 * len(h_range)

    for col in history_feature_columns(min_lag):
        values = targets[col]
        assert values.notna().all(), f"{col} missing for a target week"
        if col.startswith("lag_") or col.startswith("roll_mean_"):
            # units equal t, so a value above the origin would mean a future week leaked in.
            assert (values <= origin).all(), f"{col} uses a week after the origin"
    lag_cols = [c for c in history_feature_columns(min_lag) if c.startswith("lag_")]
    expected_min_lag = targets["t"] - min_lag
    assert (targets[f"lag_{min_lag}"] == expected_min_lag).all()
    assert all((targets[c] <= expected_min_lag).all() for c in lag_cols)


def test_future_units_never_reach_features_even_if_present():
    origin = 150
    weekly = make_weekly(200)
    history, future, _ = split_at_origin(weekly, SEGMENTS, origin, horizon=12)
    panel_clean = build_features(history, future, 4)
    # If someone passed real future units, the features for the targets must still be identical.
    panel_leaky = build_features(history, future.assign(units=999.0), 4)
    cols = history_feature_columns(4)
    pd.testing.assert_frame_equal(panel_clean[cols], panel_leaky[cols])


def test_pre_launch_weeks_are_not_history():
    history, _, _ = split_at_origin(make_weekly(200, launch_t=100), SEGMENTS, origin=150, horizon=12)
    assert history["t"].min() == 100
    panel = build_features(history, history.iloc[0:0].drop(columns="units"), 4)
    first = panel[panel["t"] == 100]
    assert first["lag_4"].isna().all()
