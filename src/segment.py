"""ABC, XYZ and Syntetos-Boylan demand pattern segmentation, computed as of a given origin."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
PATTERNS = ["smooth", "erratic", "intermittent", "lumpy", "inactive"]
SEGMENT_COLUMNS = ["item_id", "dc_id", "revenue", "abc", "cv", "xyz", "adi", "cv2", "pattern"]


def fold_origins(t_last: int, horizon: int, n_folds: int) -> dict[str, int]:
    """Backtest origins: non-overlapping horizon-week windows ending at t_last.

    With 3 folds and horizon 12: fold_1 = t_last - 36, fold_2 = t_last - 24, fold_3 = t_last - 12.
    """
    return {f"fold_{k}": t_last - horizon * (n_folds - k + 1) for k in range(1, n_folds + 1)}


def abc_classes(revenue: pd.Series, cutoffs: list[float]) -> pd.Series:
    """A if the cumulative revenue share before adding the series is below cutoffs[0], B if below cutoffs[1], else C.

    So the series that pushes the share past 0.80 is still A, and the top series is always A.
    Ties keep the input order, so callers should pass revenue sorted by key.
    """
    ordered = revenue.sort_values(ascending=False, kind="mergesort")
    total = ordered.sum()
    if total <= 0:
        return pd.Series("C", index=revenue.index)
    share_before = (ordered.cumsum() - ordered) / total
    classes = np.select([share_before < cutoffs[0], share_before < cutoffs[1]], ["A", "B"], "C")
    return pd.Series(classes, index=ordered.index).reindex(revenue.index)


def demand_classes(window: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """XYZ and demand pattern per series from its window of launched weeks.

    cv uses all weeks including zeros; cv2 uses non-zero weeks only. Standard
    deviations are population (ddof = 0), so one non-zero week gives cv2 = 0.
    """
    by_series = window.groupby(SERIES_KEYS)["units"]
    stats = pd.DataFrame({
        "n_weeks": by_series.size(),
        "mean": by_series.mean(),
        "std": by_series.std(ddof=0),
    })
    nonzero = window[window["units"] > 0].groupby(SERIES_KEYS)["units"]
    stats["n_nonzero"] = nonzero.size().reindex(stats.index, fill_value=0)
    stats["nz_mean"] = nonzero.mean()
    stats["nz_std"] = nonzero.std(ddof=0)

    stats["cv"] = (stats["std"] / stats["mean"]).where(stats["mean"] > 0)
    x_cut, y_cut = cfg["xyz_cutoffs"]
    stats["xyz"] = np.select([stats["cv"] <= x_cut, stats["cv"] <= y_cut], ["X", "Y"], "Z")

    active = stats["n_nonzero"] > 0
    stats["adi"] = (stats["n_weeks"] / stats["n_nonzero"]).where(active)
    stats["cv2"] = ((stats["nz_std"] / stats["nz_mean"]) ** 2).where(active)
    frequent = stats["adi"] < cfg["adi_cutoff"]
    stable = stats["cv2"] < cfg["cv2_cutoff"]
    stats["pattern"] = np.select(
        [~active, frequent & stable, frequent & ~stable, ~frequent & stable],
        ["inactive", "smooth", "erratic", "intermittent"],
        "lumpy",
    )
    return stats[["cv", "xyz", "adi", "cv2", "pattern"]].reset_index()


def segment(weekly: pd.DataFrame, origin: int, cfg: dict) -> pd.DataFrame:
    """Segments as of origin, using only the last window_weeks launched weeks with t <= origin.

    Series not launched by origin get no segment. cfg is the `segment` config section.
    """
    history = weekly[weekly["is_launched"] & (weekly["t"] <= origin)]
    window = history.sort_values(SERIES_KEYS + ["t"]).groupby(SERIES_KEYS).tail(cfg["window_weeks"])
    window = window.assign(revenue=window["units"] * window["price"])

    revenue = window.groupby(SERIES_KEYS)["revenue"].sum().sort_index()
    segments = revenue.reset_index()
    segments["abc"] = abc_classes(revenue, cfg["abc_cutoffs"]).to_numpy()
    segments = segments.merge(demand_classes(window, cfg), on=SERIES_KEYS, validate="one_to_one")
    return segments[SEGMENT_COLUMNS]


def run(cfg: dict) -> None:
    """Stage `segment`: one segment set per backtest fold origin plus a final set at T_last."""
    gold_dir = cfg["data"]["gold_dir"]
    gold = pd.read_parquet(gold_dir / "weekly.parquet")
    t_last = int(gold["t"].max())
    origins = fold_origins(t_last, cfg["forecast"]["horizon_weeks"], cfg["forecast"]["backtest_folds"])
    origins["final"] = t_last

    sets = []
    for name, origin in origins.items():
        seg = segment(gold, origin, cfg["segment"])
        sets.append(seg.assign(segment_set=name, origin=origin))
        logger.info("Segment set %s (origin t = %d): %d series", name, origin, len(seg))
    segments = pd.concat(sets, ignore_index=True)
    segments = segments[["segment_set", "origin"] + SEGMENT_COLUMNS]
    segments.to_parquet(gold_dir / "segments.parquet", index=False)

    final = segments[segments["segment_set"] == "final"]
    print("\nSegmentation (final set, origin t = %d, last %d launched weeks)" % (t_last, cfg["segment"]["window_weeks"]))
    print(f"  ABC: {final['abc'].value_counts().sort_index().to_dict()}")
    print(f"  XYZ: {final['xyz'].value_counts().sort_index().to_dict()}")
    print(f"  pattern: {final['pattern'].value_counts().reindex(PATTERNS, fill_value=0).to_dict()}")
    print("  ABC by XYZ (series):")
    print(pd.crosstab(final["abc"], final["xyz"]).to_string().replace("\n", "\n    ").join(["    ", ""]))
    revenue_share = final.groupby("abc")["revenue"].sum() / final["revenue"].sum()
    print(f"  revenue share by ABC: {revenue_share.round(3).to_dict()}")
    print("  pattern counts per segment set:")
    per_set = pd.crosstab(segments["segment_set"], segments["pattern"]).reindex(columns=PATTERNS, fill_value=0)
    print(per_set.to_string().replace("\n", "\n    ").join(["    ", ""]))
    changed = (
        segments[segments["segment_set"] != "final"]
        .merge(final[SERIES_KEYS + ["pattern"]], on=SERIES_KEYS, suffixes=("", "_final"))
        .assign(changed=lambda d: d["pattern"] != d["pattern_final"])
        .groupby("segment_set")["changed"].mean()
    )
    print(f"  share of series whose pattern differs from the final set: {changed.round(3).to_dict()}")
