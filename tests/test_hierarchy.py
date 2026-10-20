import pandas as pd
import pytest

from src.hierarchy import history_shares, score_levels, top_down


def make_actual() -> pd.DataFrame:
    rows = []
    for item, dc, level in [("I1", "CA", 30.0), ("I1", "TX", 10.0), ("I2", "CA", 20.0), ("I2", "TX", 40.0)]:
        rows += [{"item_id": item, "dc_id": dc, "t": t, "actual": level} for t in (10, 11)]
    return pd.DataFrame(rows)


def test_perfect_bottom_level_forecast_is_perfect_at_every_level():
    actual = make_actual()
    perfect = actual.rename(columns={"actual": "forecast"})
    scores = score_levels(perfect, perfect, actual)
    assert (scores["wape"] == 0).all()
    assert scores.set_index(["level", "method"]).loc[("item", "bottom_up"), "groups"] == 2


def test_errors_cancel_when_aggregated():
    actual = make_actual()
    fc = actual.rename(columns={"actual": "forecast"}).copy()
    fc.loc[(fc["item_id"] == "I1") & (fc["dc_id"] == "CA"), "forecast"] += 5
    fc.loc[(fc["item_id"] == "I1") & (fc["dc_id"] == "TX"), "forecast"] -= 5
    scores = score_levels(fc, fc, actual).set_index(["level", "method"])["wape"]
    assert scores[("item_dc", "bottom_up")] == pytest.approx(20 / 200)
    assert scores[("item", "bottom_up")] == 0
    assert scores[("total", "bottom_up")] == 0


def test_top_down_split_preserves_total_and_uses_shares():
    weekly = pd.DataFrame({"item_id": ["I1", "I2"] * 3, "dc_id": "CA", "t": [1, 1, 2, 2, 3, 3],
                           "units": [3, 1, 3, 1, 3, 1], "is_launched": True})
    shares = history_shares(weekly, origin=3, weeks=52)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[("I1", "CA")] == pytest.approx(0.75)
    split = top_down(pd.DataFrame({"t": [4, 5], "forecast": [100.0, 40.0]}), shares)
    assert split.groupby("t")["forecast"].sum().tolist() == pytest.approx([100.0, 40.0])
    assert split[(split["item_id"] == "I1") & (split["t"] == 4)]["forecast"].iloc[0] == pytest.approx(75.0)


def test_shares_use_only_weeks_up_to_origin():
    weekly = pd.DataFrame({"item_id": ["I1", "I2"] * 2, "dc_id": "CA", "t": [1, 1, 2, 2], "units": [1, 1, 100, 0], "is_launched": True})
    assert history_shares(weekly, origin=1)[("I1", "CA")] == pytest.approx(0.5)
