"""Rolling-origin backtest, scoring and model selection per demand pattern."""

import logging
import time

import numpy as np
import pandas as pd

from src import forecast_baselines as fb
from src import forecast_ml as fm
from src import forecast_stat as fs
from src.features import split_at_origin
from src.metrics import bias, fva, mase_scale, wape
from src.segment import PATTERNS, fold_origins

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
MODELS = {
    "naive": fb.naive,
    "seasonal_naive": fb.seasonal_naive,
    "moving_average_4": fb.moving_average_4,
    "holt_winters": fs.holt_winters,
    "croston": fs.croston,
    "tsb": fs.tsb,
    "lightgbm": fm.lightgbm,
}
BENCHMARK = "naive"
INACTIVE_MODEL = "tsb"


def run_fold(gold: pd.DataFrame, segments: pd.DataFrame, fold: str, origin: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every model trained on t <= origin, forecasting origin + 1 .. origin + horizon.

    Returns (forecasts with actual, abc and pattern of this fold's segment set; MASE scales; feature importance).
    """
    horizon = cfg["forecast"]["horizon_weeks"]
    seg = segments[segments["segment_set"] == fold]
    history, future, actuals = split_at_origin(gold, seg, origin, horizon)

    outputs, importance = [], None
    for name, model in MODELS.items():
        start = time.perf_counter()
        out = model(history, future, horizon, cfg)
        if "importance" in out.attrs:
            importance = out.attrs["importance"].assign(fold=fold)
        outputs.append(out)
        logger.info("%s %s: %.1f s", fold, name, time.perf_counter() - start)

    forecasts = pd.concat(outputs, ignore_index=True)
    forecasts = forecasts.merge(actuals.rename(columns={"units": "actual"}), on=SERIES_KEYS + ["t"], validate="many_to_one")
    forecasts = forecasts.merge(seg[SERIES_KEYS + ["abc", "pattern"]], on=SERIES_KEYS, validate="many_to_one")
    forecasts.insert(0, "fold", fold)

    scales = (
        history.sort_values("t").groupby(SERIES_KEYS)["units"]
        .agg(lambda s: mase_scale(s.to_numpy(), season=52)).rename("mase_scale").reset_index()
        .assign(fold=fold)
    )
    return forecasts, scales, importance


def score(forecasts: pd.DataFrame, scales: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Volume-weighted WAPE and bias, mean MASE and FVA vs naive per model (and per group if by is given).

    Rows are pooled across folds; a group is the series' class in that fold's segment set.
    """
    data = forecasts.assign(abs_error=(forecasts["actual"] - forecasts["forecast"]).abs())
    group_cols = ["model"] + ([by] if by else [])
    series_cols = ["fold"] + SERIES_KEYS

    per_series = (
        data.groupby(group_cols + series_cols)["abs_error"].mean().rename("mae").reset_index()
        .merge(scales, on=series_cols, how="left")
    )
    per_series["mase"] = per_series["mae"] / per_series["mase_scale"]

    rows = []
    for key, g in data.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(group_cols, key)) | {
            "wape": wape(g["actual"], g["forecast"]),
            "bias": bias(g["actual"], g["forecast"]),
            "n_series": g[SERIES_KEYS].drop_duplicates().shape[0],
            "n_series_folds": g[series_cols].drop_duplicates().shape[0],
            "actual_units": float(g["actual"].sum()),
        })
    table = pd.DataFrame(rows)
    mean_mase = per_series.groupby(group_cols)["mase"].mean().rename("mean_mase").reset_index()
    table = table.merge(mean_mase, on=group_cols, how="left")

    bench = table[table["model"] == BENCHMARK].set_index(by)["wape"] if by else table.loc[table["model"] == BENCHMARK, "wape"].iloc[0]
    if by:
        table["fva_vs_naive_pp"] = [fva(w, bench.get(g, np.nan)) for w, g in zip(table["wape"], table[by])]
    else:
        table["fva_vs_naive_pp"] = [fva(w, bench) for w in table["wape"]]
    table.insert(0, "level", by or "overall")
    if by:
        table = table.rename(columns={by: "group"})
    else:
        table.insert(2, "group", "all")
    return table


def select_models(pattern_scores: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Per pattern: lowest WAPE, tie-break by absolute bias. inactive always uses TSB.

    A pattern with no scored series in any fold gets the overall best model.
    """
    best_overall = overall.assign(abs_bias=overall["bias"].abs()).sort_values(["wape", "abs_bias"]).iloc[0]["model"]
    rows = []
    for pattern in PATTERNS:
        g = pattern_scores[(pattern_scores["group"] == pattern) & pattern_scores["wape"].notna()]
        g = g.assign(abs_bias=g["bias"].abs()).sort_values(["wape", "abs_bias"])
        row = {"pattern": pattern, "n_series_folds": int(g["n_series_folds"].max()) if len(g) else 0}
        if pattern == "inactive":
            chosen, rule = INACTIVE_MODEL, "inactive always uses tsb"
        elif len(g) == 0:
            chosen, rule = best_overall, "no backtest series with this pattern: overall best model"
        else:
            chosen, rule = g.iloc[0]["model"], "lowest WAPE across all folds"
        row |= {"model": chosen, "rule": rule}
        best = g[g["model"] == chosen]
        if len(best):
            row |= {"wape": best["wape"].iloc[0], "bias": best["bias"].iloc[0], "fva_vs_naive_pp": best["fva_vs_naive_pp"].iloc[0]}
        others = g[g["model"] != chosen]
        if len(others):
            row |= {
                "second_model": others.iloc[0]["model"],
                "second_wape": others.iloc[0]["wape"],
                "gap_pp": (others.iloc[0]["wape"] - row.get("wape", np.nan)) * 100,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def chosen_residuals(forecasts: pd.DataFrame, selection: pd.DataFrame, final_segments: pd.DataFrame) -> pd.DataFrame:
    """actual - forecast of the model each series will use in the final forecast (chosen for its final pattern)."""
    model_for = final_segments[SERIES_KEYS + ["pattern"]].rename(columns={"pattern": "final_pattern"})
    model_for["model"] = model_for["final_pattern"].map(selection.set_index("pattern")["model"])
    res = forecasts.merge(model_for, on=SERIES_KEYS + ["model"], validate="many_to_one")
    res["residual"] = res["actual"] - res["forecast"]
    return res[["fold", "item_id", "dc_id", "t", "h", "model", "final_pattern", "actual", "forecast", "residual"]]


def run(cfg: dict) -> None:
    """Stage `backtest`."""
    start = time.perf_counter()
    paths = cfg["data"]
    gold = pd.read_parquet(paths["gold_dir"] / "weekly.parquet")
    segments = pd.read_parquet(paths["gold_dir"] / "segments.parquet")
    t_last = int(gold["t"].max())
    origins = fold_origins(t_last, cfg["forecast"]["horizon_weeks"], cfg["forecast"]["backtest_folds"])

    results = [run_fold(gold, segments, fold, origin, cfg) for fold, origin in origins.items()]
    forecasts = pd.concat([r[0] for r in results], ignore_index=True)
    scales = pd.concat([r[1] for r in results], ignore_index=True)
    importance = pd.concat([r[2] for r in results if r[2] is not None], ignore_index=True)

    out = paths["outputs_dir"]
    forecasts.to_parquet(out / "backtest_forecasts.parquet", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    overall = score(forecasts, scales, None)
    by_pattern = score(forecasts, scales, "pattern")
    by_abc = score(forecasts, scales, "abc")
    scores = pd.concat([overall, by_pattern, by_abc], ignore_index=True)
    scores.to_csv(out / "backtest_scores.csv", index=False)

    selection = select_models(by_pattern, overall)
    selection.to_csv(out / "model_selection.csv", index=False)
    final_segments = segments[segments["segment_set"] == "final"]
    residuals = chosen_residuals(forecasts, selection, final_segments)
    residuals.to_parquet(out / "residuals.parquet", index=False)

    pd.set_option("display.width", 200)
    leaderboard = scores[scores["level"] != "abc"].pivot_table(index="model", columns="group", values="wape")
    leaderboard = leaderboard[["all"] + [p for p in PATTERNS if p in leaderboard.columns]]
    print("\nBacktest leaderboard: volume-weighted WAPE, pooled over 3 folds (pattern = fold segment set)")
    print(leaderboard.sort_values("all").round(3).to_string())
    counts = by_pattern.drop_duplicates("group").set_index("group")["n_series_folds"].to_dict()
    print(f"  series-folds per pattern: {counts}")
    print("\nOverall")
    print(overall[["model", "wape", "bias", "mean_mase", "fva_vs_naive_pp"]].sort_values("wape").round(3).to_string(index=False))
    print("\nBy ABC (WAPE)")
    print(by_abc.pivot_table(index="model", columns="group", values="wape").round(3).to_string())
    print("\nSelected model per pattern")
    cols = ["pattern", "model", "wape", "bias", "fva_vs_naive_pp", "second_model", "gap_pp", "n_series_folds", "rule"]
    print(selection.reindex(columns=cols).round(3).to_string(index=False))
    print(f"\nResiduals saved: {len(residuals)} rows for {residuals[SERIES_KEYS].drop_duplicates().shape[0]} series")
    print(f"Runtime: {time.perf_counter() - start:.1f} s")
