import numpy as np
import pandas as pd
import pytest

from src.elasticity import estimate_all

CFG = {"elasticity": {"min_distinct_prices": 3, "min_price_cv": 0.02, "shrinkage_weeks": 20, "clip": [-4.0, 0.0]}}


def make_item(item: str, elasticity: float, rng: np.random.Generator, n: int = 200, vary: bool = True) -> pd.DataFrame:
    weeks = pd.date_range("2013-01-05", periods=n, freq="7D")
    rows = []
    for dc, level in (("CA", 100.0), ("TX", 60.0)):
        price = rng.choice([2.0, 2.2, 2.5, 2.8], size=n) if vary else np.full(n, 2.5)
        units = level * price ** elasticity * np.exp(rng.normal(0, 0.05, n))
        rows.append(pd.DataFrame({"item_id": item, "dc_id": dc, "week_start": weeks, "units": units, "price": price,
                                  "event_days": 0, "snap_days": rng.integers(0, 4, n), "is_launched": True}))
    return pd.concat(rows, ignore_index=True)


def test_recovers_known_elasticity_and_shrinks_toward_department_mean():
    rng = np.random.default_rng(0)
    weekly = pd.concat([make_item("I1", -2.0, rng), make_item("I2", -1.0, rng), make_item("FLAT", -1.5, rng, vary=False)])
    table = estimate_all(weekly, CFG).set_index("item_id")
    assert table.loc["I1", "e_raw"] == pytest.approx(-2.0, abs=0.05)
    assert table.loc["I2", "e_raw"] == pytest.approx(-1.0, abs=0.05)
    dept = (table.loc["I1", "e_raw"] + table.loc["I2", "e_raw"]) / 2
    assert table.loc["I1", "e_dept"] == pytest.approx(dept)
    n = table.loc["I1", "n_weeks"]
    assert table.loc["I1", "e_shrunk"] == pytest.approx(n / (n + 20) * table.loc["I1", "e_raw"] + 20 / (n + 20) * dept)
    assert table.loc["FLAT", "status"] == "insufficient_variation"
    assert table.loc["FLAT", "elasticity"] == pytest.approx(dept)


def test_positive_estimate_is_flagged_and_clipped_to_zero():
    rng = np.random.default_rng(1)
    weekly = pd.concat([make_item("UP", 1.5, rng, n=30), make_item("I1", -2.0, rng), make_item("I2", -2.5, rng)])
    table = estimate_all(weekly, CFG).set_index("item_id")
    assert table.loc["UP", "raw_positive"]
    assert table.loc["UP", "elasticity"] <= 0.0
    assert table["elasticity"].between(-4.0, 0.0).all()
