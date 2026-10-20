import pandas as pd
import pytest

from src.errors import DataQualityError
from src.final_forecast import future_calendar, snap_schedule


def make_calendar(start: str, end: str, snap_days: list[int]) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    cal = pd.DataFrame({
        "date": dates,
        "d": [f"d_{i + 1}" for i in range(len(dates))],
        "wm_yr_wk": 11101 + (dates - dates[0]).days // 7,
        "event_name_1": None, "event_type_1": None, "event_name_2": None, "event_type_2": None,
        "snap_CA": dates.day.isin(snap_days).astype(int),
    })
    july4 = cal["date"] == pd.Timestamp("2015-07-04")
    cal.loc[july4, ["event_name_1", "event_type_1"]] = ["IndependenceDay", "National"]
    return cal


def test_snap_schedule_inferred_from_complete_months():
    cal = make_calendar("2015-01-15", "2016-06-19", snap_days=[1, 2, 3])
    assert snap_schedule(cal, ["CA"]) == {"CA": [1, 2, 3]}


def test_inconsistent_snap_schedule_raises():
    cal = make_calendar("2015-01-01", "2016-06-19", snap_days=[1, 2, 3])
    cal.loc[cal["date"] == pd.Timestamp("2015-03-05"), "snap_CA"] = 1
    with pytest.raises(DataQualityError, match="SNAP schedule"):
        snap_schedule(cal, ["CA"])


def test_future_calendar_fills_by_date_and_flags():
    cal = make_calendar("2015-01-01", "2016-06-19", snap_days=[1, 2, 3])
    daily = future_calendar(cal, pd.Timestamp("2016-06-18"), 30, ["CA"]).set_index("date")

    assert not daily.loc["2016-06-19", "calendar_estimated"]
    assert daily.loc["2016-06-20", "calendar_estimated"]
    assert daily.loc["2016-07-04", "event_name_1"] == "IndependenceDay"
    assert daily.loc["2016-07-04", "event_type_1"] == "National"
    assert daily.loc["2016-07-02", "snap_CA"] == 1
    assert daily.loc["2016-07-05", "snap_CA"] == 0
