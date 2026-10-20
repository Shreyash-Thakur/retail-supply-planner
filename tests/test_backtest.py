import pandas as pd

from src.backtest import select_models


def scores(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "model", "wape", "bias", "fva_vs_naive_pp", "n_series_folds"])


PATTERN_SCORES = scores([
    ("intermittent", "naive", 0.744, -0.173, 0.0, 164),
    ("intermittent", "lightgbm", 0.746, -0.119, -0.2, 164),
    ("intermittent", "moving_average_4", 0.760, 0.010, -1.6, 164),
    ("smooth", "lightgbm", 0.242, -0.053, 8.7, 1129),
    ("smooth", "tsb", 0.247, 0.001, 8.2, 1129),
    ("inactive", "croston", 0.636, -0.4, 36.0, 2),
    ("inactive", "tsb", 0.999, -0.99, 0.1, 2),
])
OVERALL = scores([("all", "lightgbm", 0.28, -0.05, 8.9, 1350), ("all", "naive", 0.37, 0.0, 0.0, 1350)])


def chosen(margin: float) -> dict:
    return select_models(PATTERN_SCORES, OVERALL, margin_pp=margin).set_index("pattern")["model"].to_dict()


def test_zero_margin_is_plain_lowest_wape():
    assert chosen(0.0)["intermittent"] == "naive"
    assert chosen(0.0)["smooth"] == "lightgbm"


def test_margin_prefers_lower_bias_only_inside_the_margin():
    picks = chosen(1.0)
    assert picks["intermittent"] == "lightgbm"  # 0.2 points worse, much less biased; moving average is 1.6 points away
    assert picks["smooth"] == "tsb"  # 0.5 points worse, bias near zero
    assert chosen(0.4)["smooth"] == "lightgbm"


def test_inactive_always_tsb_and_missing_pattern_gets_overall_best():
    picks = chosen(1.0)
    assert picks["inactive"] == "tsb"
    assert picks["lumpy"] == "lightgbm"  # no lumpy rows here: overall best
