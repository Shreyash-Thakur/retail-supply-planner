import numpy as np
import pandas as pd
import pytest

from src.errors import DataQualityError
from src.pipeline import (
    aggregate_weekly,
    check_gold,
    find_complete_weeks,
    finalize_gold,
    select_slice,
    verify_raw_columns,
    weekly_calendar_features,
)


def make_calendar(n_days: int) -> pd.DataFrame:
    """Daily calendar starting on a Saturday, 7 days per week, no events or SNAP."""
    day = np.arange(n_days)
    return pd.DataFrame({
        "d": [f"d_{i + 1}" for i in day],
        "date": pd.date_range("2011-01-29", periods=n_days),
        "wm_yr_wk": 11101 + day // 7,
        "event_name_1": None, "event_type_1": None, "event_name_2": None, "event_type_2": None,
        "snap_CA": 0,
    })


def make_weekly(series: dict[tuple[str, str], list], prices: dict[tuple[str, str], list] | None = None) -> pd.DataFrame:
    """Weekly rows (item_id, dc_id, week_id, units, price) for t = 0..n-1; price 1.0 unless given."""
    rows = []
    for (item, dc), units in series.items():
        price = (prices or {}).get((item, dc), [1.0] * len(units))
        rows += [
            {"item_id": item, "dc_id": dc, "week_id": 11101 + t, "units": u, "price": p}
            for t, (u, p) in enumerate(zip(units, price))
        ]
    return pd.DataFrame(rows)


def cal_feats_for(n_weeks: int) -> pd.DataFrame:
    cal = make_calendar(7 * n_weeks)
    weeks, _ = find_complete_weeks(cal, n_days=7 * n_weeks)
    return weekly_calendar_features(cal, weeks, states=["CA"])


def test_incomplete_week_is_dropped():
    cal = make_calendar(21)
    weeks, dropped = find_complete_weeks(cal, n_days=17)

    assert weeks["week_id"].tolist() == [11101, 11102]
    assert weeks["t"].tolist() == [0, 1]
    assert dropped.to_dict("records") == [{"week_id": 11103, "days": 3}]


def test_pre_launch_rows_flagged_and_post_launch_zeros_kept():
    gold, _ = finalize_gold(make_weekly({("I1", "CA"): [0, 0, 5, 0, 3]}), cal_feats_for(5))

    assert gold["is_launched"].tolist() == [False, False, True, True, True]
    assert gold["units"].tolist() == [0, 0, 5, 0, 3]


def test_price_forward_fill_after_launch():
    weekly = make_weekly({("I1", "CA"): [1, 2, 0, 4]}, prices={("I1", "CA"): [2.0, np.nan, np.nan, 2.5]})
    gold, stats = finalize_gold(weekly, cal_feats_for(4))

    assert gold["price"].tolist() == [2.0, 2.0, 2.0, 2.5]
    assert stats["price_ffill"] == 2
    assert stats["price_still_missing"] == 0


def test_price_back_fill_then_item_median():
    weekly = make_weekly(
        {("I1", "CA"): [1, 1, 1], ("I1", "TX"): [1, 1, 1]},
        prices={("I1", "CA"): [np.nan, 3.0, 3.0], ("I1", "TX"): [np.nan, np.nan, np.nan]},
    )
    gold, stats = finalize_gold(weekly, cal_feats_for(3))

    assert gold.loc[gold["dc_id"] == "CA", "price"].tolist() == [3.0, 3.0, 3.0]
    assert gold.loc[gold["dc_id"] == "TX", "price"].tolist() == [3.0, 3.0, 3.0]
    assert (stats["price_bfill"], stats["price_item_median"]) == (1, 3)


def test_negative_unit_raises():
    cal = make_calendar(7)
    weeks, _ = find_complete_weeks(cal, n_days=7)
    daily = pd.DataFrame({
        "item_id": "I1", "store_id": "CA_1", "dc_id": "CA", "wm_yr_wk": 11101,
        "units": [1, 2, -1, 0, 0, 0, 0], "sell_price": 1.0,
    })
    with pytest.raises(DataQualityError, match="negative units"):
        aggregate_weekly(daily, weeks)


def test_missing_week_after_launch_raises():
    weekly = make_weekly({("I1", "CA"): [3, 0, 2, 1, 4]})
    weekly = weekly[weekly["week_id"] != 11103]
    gold, _ = finalize_gold(weekly, cal_feats_for(5))

    with pytest.raises(DataQualityError, match="gap"):
        check_gold(gold, ["I1"], expected_dcs=1)


def test_never_launched_series_dropped_and_counted():
    weekly = make_weekly({("I1", "CA"): [0, 1, 1], ("I1", "TX"): [0, 0, 0], ("I2", "CA"): [0, 0, 0]})
    gold, stats = finalize_gold(weekly, cal_feats_for(3))

    assert stats["never_launched_series"] == 2
    assert stats["dropped_items"] == ["I2"]
    assert sorted(gold[["item_id", "dc_id"]].drop_duplicates().itertuples(index=False)) == [("I1", "CA")]
    check_gold(gold, ["I1", "I2"], expected_dcs=1, dropped_items=stats["dropped_items"])


def test_clean_gold_passes_checks():
    weekly = make_weekly({("I1", "CA"): [0, 2, 0, 1], ("I2", "CA"): [1, 1, 1, 1]})
    gold, _ = finalize_gold(weekly, cal_feats_for(4))
    check_gold(gold, ["I1", "I2"], expected_dcs=1)


def test_select_slice_top_n_with_ties_and_reproducible():
    volume = pd.Series({"A": 5, "B": 10, "C": 5, "D": 1, "E": 0, "F": 3})
    first = select_slice(volume, top_n=2, n_random=2, rng=np.random.default_rng(42))
    second = select_slice(volume, top_n=2, n_random=2, rng=np.random.default_rng(42))

    pd.testing.assert_frame_equal(first, second)
    top = first.loc[first["selection_reason"] == "top_volume", "item_id"].tolist()
    extra = first.loc[first["selection_reason"] == "random_extra", "item_id"].tolist()
    assert top == ["A", "B"]
    assert len(extra) == 2 and set(extra) <= {"C", "D", "E", "F"}
    assert first["item_id"].tolist() == sorted(first["item_id"])


def test_verify_raw_columns_raises_on_renamed_column(tmp_path):
    (tmp_path / "sales_train_evaluation.csv").write_text(
        "id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2\n", encoding="utf-8"
    )
    (tmp_path / "calendar.csv").write_text(
        "date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,"
        "event_name_2,event_type_2,snap_CA,snap_TX,snap_WI\n", encoding="utf-8"
    )
    (tmp_path / "sell_prices.csv").write_text("store_id,item_id,week,sell_price\n", encoding="utf-8")

    with pytest.raises(DataQualityError, match="sell_prices.csv"):
        verify_raw_columns(tmp_path)
