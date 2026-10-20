"""Stage `app_data`: small result files for the dashboard, and RESULTS.md generated from the run outputs.

RESULTS.md is written from data/outputs by this stage, so its numbers always
match the files they come from; each block names the stage command that produced them.
"""

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.segment import PATTERNS

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
SCENARIO_ORDER = [
    "baseline", "baseline_fair_share", "demand_surge", "supplier_disruption",
    "lead_time_shock_policy_not_updated", "lead_time_shock_policy_updated", "flat_service_level",
    "diagnostic_unlimited_supply", "price_cut_policy_not_updated", "price_cut_policy_updated",
]


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    """Plain markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_app_data(cfg: dict) -> dict:
    paths = cfg["data"]
    out, gold_dir, app = paths["outputs_dir"], paths["gold_dir"], paths["app_data_dir"]
    gold = pd.read_parquet(gold_dir / "weekly.parquet")
    segments = pd.read_parquet(gold_dir / "segments.parquet")
    t_last = int(gold["t"].max())

    final_seg = segments[segments["segment_set"] == "final"].drop(columns=["segment_set", "origin"])
    final_seg.to_csv(app / "segments.csv", index=False)
    pd.crosstab(segments["segment_set"], segments["pattern"]).reindex(columns=PATTERNS, fill_value=0).reset_index().to_csv(
        app / "segment_sets.csv", index=False)
    for name in ("backtest_scores.csv", "model_selection.csv", "allocation_demo.csv", "sim_kpis.csv", "feature_importance.csv"):
        pd.read_csv(out / name).to_csv(app / name, index=False)
    for name in ("exceptions.csv", "elasticity.csv"):  # stretch outputs: copied when their stage has run
        if (out / name).exists():
            pd.read_csv(out / name).to_csv(app / name, index=False)

    final_forecast = pd.read_parquet(out / "final_forecast.parquet")
    final_forecast.to_csv(app / "final_forecast.csv", index=False)
    history = gold[gold["is_launched"] & (gold["t"] > t_last - 52)][SERIES_KEYS + ["t", "week_id", "week_start", "units"]]
    history.to_csv(app / "history.csv", index=False)
    backtest = pd.read_parquet(out / "backtest_forecasts.parquet")[["fold", "item_id", "dc_id", "t", "h", "model", "forecast", "actual"]]
    backtest.to_parquet(app / "backtest_forecasts.parquet", index=False)
    policy_cols = ["item_id", "dc_id", "abc", "pattern", "model", "service_level", "lead_time_weeks", "sigma_e", "sigma_source",
                   "ss_classic", "ss_quantile", "lead_time_demand", "reorder_point", "order_up_to", "opening_stock",
                   "mean_weekly_forecast", "days_of_cover"]
    pd.read_parquet(out / "inventory_policy.parquet")[policy_cols].to_csv(app / "inventory_policy.csv", index=False)
    pd.read_parquet(out / "sim_weekly.parquet").to_csv(app / "sim_weekly.csv", index=False)

    week_starts = gold.drop_duplicates("t").set_index("t")["week_start"]
    meta = {
        "items": int(gold["item_id"].nunique()), "dcs": sorted(gold["dc_id"].unique()), "series": int(final_seg.shape[0]),
        "weeks": int(gold["t"].nunique()), "first_week": str(week_starts.iloc[0].date()), "last_week": str(week_starts.iloc[-1].date()),
        "t_last": t_last, "horizon_weeks": cfg["forecast"]["horizon_weeks"], "simulation_runs": cfg["simulation"]["runs"],
        "generated": str(date.today()),
    }
    (app / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in app.iterdir() if f.is_file()) / 1e6
    logger.info("app_data written: %.1f MB in %d files", size_mb, sum(1 for f in app.iterdir() if f.is_file()))
    meta["size_mb"] = size_mb
    return meta


def write_results(cfg: dict, meta: dict) -> str:
    """Build RESULTS.md from the run outputs."""
    paths = cfg["data"]
    out, gold_dir = paths["outputs_dir"], paths["gold_dir"]
    gold = pd.read_parquet(gold_dir / "weekly.parquet")
    segments = pd.read_parquet(gold_dir / "segments.parquet")
    final = segments[segments["segment_set"] == "final"]
    scores = pd.read_csv(out / "backtest_scores.csv")
    selection = pd.read_csv(out / "model_selection.csv")
    policy = pd.read_parquet(out / "inventory_policy.parquet")
    demo = pd.read_csv(out / "allocation_demo.csv")
    kpis = pd.read_csv(out / "sim_kpis.csv")
    slice_ = pd.read_csv(paths["silver_dir"] / "item_slice.csv")

    parts = [
        "# Results",
        "",
        f"Every number below is generated from the files in `data/outputs/` and `data/gold/` by `python run_all.py --stage app_data` "
        f"(last generated {meta['generated']}). Each block names the stage command that produced its numbers. Demand is real M5 data; "
        "supply parameters are simulated (see README).",
        "",
        "## Data slice",
        "",
        "Command: `python run_all.py --stage pipeline`",
        "",
        _md_table(pd.DataFrame([
            ["Items", f"{gold['item_id'].nunique()} ({(slice_['selection_reason'] == 'top_volume').sum()} top volume, "
                      f"{(slice_['selection_reason'] == 'random_extra').sum()} random extra) from FOODS_3"],
            ["DCs", ", ".join(sorted(gold["dc_id"].unique()))],
            ["Weeks", f"{gold['t'].nunique()} complete weeks (week starting {meta['first_week']} to week starting {meta['last_week']})"],
            ["Gold rows", f"{len(gold):,} ({int(gold['is_launched'].sum()):,} launched, {int((~gold['is_launched']).sum()):,} pre-launch)"],
            ["Units sold", f"{int(gold['units'].sum()):,}"],
        ], columns=["Measure", "Value"])),
        "",
        "## Segments (final set, last 52 launched weeks up to the last week)",
        "",
        "Command: `python run_all.py --stage segment`",
        "",
        _md_table(pd.crosstab(final["abc"], final["xyz"], margins=True, margins_name="Total").reset_index().rename(columns={"abc": "ABC \\ XYZ"})),
        "",
        _md_table(final["pattern"].value_counts().reindex(PATTERNS, fill_value=0).rename_axis("pattern").reset_index(name="series")),
        "",
        "## Backtest (3 folds of 12 weeks, volume-weighted, pattern as of each fold's origin)",
        "",
        "Command: `python run_all.py --stage backtest`",
        "",
    ]
    board = scores[scores["level"] != "abc"].pivot_table(index="model", columns="group", values="wape")
    board = board[["all"] + [p for p in PATTERNS if p in board.columns]].sort_values("all").reset_index()
    parts += ["WAPE by model and pattern:", "", _md_table(board), ""]
    overall = scores[scores["level"] == "overall"][["model", "wape", "bias", "mean_mase", "fva_vs_naive_pp"]].sort_values("wape")
    parts += ["Overall:", "", _md_table(overall), ""]
    sel = selection.reindex(columns=["pattern", "model", "wape", "bias", "fva_vs_naive_pp", "second_model", "gap_pp", "n_series_folds", "rule"])
    parts += ["Selected model per pattern (gap_pp = second-best WAPE minus selected WAPE, in points):", "", _md_table(sel), ""]

    ss = policy.groupby("abc").agg(series=("item_id", "size"), service_level=("service_level", "first"),
                                   mean_ss_classic=("ss_classic", "mean"), mean_ss_quantile=("ss_quantile", "mean"),
                                   mean_reorder_point=("reorder_point", "mean"), median_days_of_cover=("days_of_cover", "median")).reset_index()
    parts += ["## Safety stock by ABC class", "", "Command: `python run_all.py --stage inventory`", "", _md_table(ss, "{:,.2f}"), "",
              f"Total safety stock: classic {policy['ss_classic'].sum():,.0f} units, quantile {policy['ss_quantile'].sum():,.0f} units.", ""]

    demo_abc = demo.groupby("abc")[["request", "lp", "fair_share"]].sum().reset_index()
    w_short = lambda col: float((demo["weight"] * (demo["request"] - demo[col])).sum())  # noqa: E731
    parts += ["## LP vs fair share", "", "Command: `python run_all.py --stage inventory` (one constrained week) and "
              "`python run_all.py --stage simulate` (12 weeks, 200 runs)", "",
              "One constrained week (opening stock on hand, empty pipeline, supplier capacity at 50%):", "", _md_table(demo_abc), "",
              f"Weighted shortfall: LP {w_short('lp'):,.0f}, fair share {w_short('fair_share'):,.0f}.", ""]
    sim_cmp = kpis[(kpis["scenario"].isin(["baseline", "baseline_fair_share"])) & (kpis["kpi"] == "fill_rate")]
    sim_cmp = sim_cmp.pivot(index="group", columns="scenario", values="mean").reset_index().rename(
        columns={"baseline": "fill_rate_lp", "baseline_fair_share": "fill_rate_fair_share"})
    parts += ["Simulated fill rate over 12 weeks (mean of 200 runs):", "", _md_table(sim_cmp), ""]

    overall_k = kpis[kpis["group"] == "All"]
    table = overall_k.pivot(index="scenario", columns="kpi", values="mean").reindex(SCENARIO_ORDER)
    table = table.dropna(how="all")
    table = table[["fill_rate", "cycle_service", "stockout_weeks", "avg_inventory_value", "holding_cost", "lost_sales_value",
                   "days_of_cover", "revenue"]]
    band = overall_k[overall_k["kpi"] == "fill_rate"].set_index("scenario").reindex(table.index)
    money = ["avg_inventory_value", "holding_cost", "lost_sales_value", "revenue"]
    table[money] = table[money].round(0).astype("int64")
    table.insert(1, "fill_rate_p10", band["p10"])
    table.insert(2, "fill_rate_p90", band["p90"])
    parts += ["## Scenario KPIs (all series, mean of 200 runs)", "", "Command: `python run_all.py --stage simulate`", "",
              _md_table(table.reset_index()), "",
              "Values are in US dollars at M5 shelf prices; unit cost is 60% of price and holding cost 25% of cost per year (simulated).", "",
              "`diagnostic_unlimited_supply` is not one of the specified scenarios; it removes the supplier capacity limit to separate "
              "the policy's own performance from the capacity effect.", ""]
    fill_groups = kpis[kpis["kpi"] == "fill_rate"].pivot(index="scenario", columns="group", values="mean").reindex(table.index).reset_index()
    parts += ["Fill rate by group (model_naive and model_lightgbm group series by their chosen forecasting model):", "", _md_table(fill_groups), ""]
    binding = pd.read_csv(out / "sim_supply_binding.csv").set_index("scenario").reindex(table.index).reset_index()
    parts += ["Share of item-weeks in which DC requests exceeded the item's supplier capacity:", "", _md_table(binding), ""]
    if (out / "exceptions.csv").exists():
        exc = pd.read_csv(out / "exceptions.csv")
        summary = exc.groupby("exception_type").agg(count=("rank", "size"), value_at_risk=("value_at_risk", "sum")).reset_index()
        summary["value_at_risk"] = summary["value_at_risk"].round(0).astype("int64")
        top = exc.head(10)[["rank", "item_id", "dc_id", "abc", "exception_type", "value_at_risk", "detail"]].copy()
        top["value_at_risk"] = top["value_at_risk"].round(0).astype("int64")
        parts += ["## Planner exceptions (stretch)", "", "Command: `python run_all.py --stage exceptions`", "", _md_table(summary), "",
                  f"Series with at least one exception: {exc[SERIES_KEYS].drop_duplicates().shape[0]}. Top 10 by value at risk (USD):", "",
                  _md_table(top), ""]
    if (out / "elasticity.csv").exists():
        el = pd.read_csv(out / "elasticity.csv")
        ok = el[el["status"] == "ok"]
        stats = pd.DataFrame([
            ["Items", len(el)], ["Status ok / insufficient variation", f"{len(ok)} / {int((el['status'] == 'insufficient_variation').sum())}"],
            ["Department mean (raw)", f"{el['e_dept'].iloc[0]:.3f}"], ["Raw estimate median (ok items)", f"{ok['e_raw'].median():.3f}"],
            ["Raw estimates with a positive sign (flagged)", int(el["raw_positive"].sum())],
            ["Positive with a time trend added (sensitivity only)", int((ok["e_raw_with_trend"] > 0).sum())],
            ["Final elasticity median / mean", f"{el['elasticity'].median():.3f} / {el['elasticity'].mean():.3f}"],
        ], columns=["Measure", "Value"])
        parts += ["## Price elasticity and price cut (stretch)", "", "Command: `python run_all.py --stage elasticity` and "
                  "`python run_all.py --stage simulate` (scenarios price_cut_policy_not_updated and price_cut_policy_updated above)", "",
                  _md_table(stats), ""]
    text = "\n".join(parts)
    (PROJECT_ROOT / "RESULTS.md").write_text(text, encoding="utf-8")
    return text


def run(cfg: dict) -> None:
    meta = write_app_data(cfg)
    write_results(cfg, meta)
    print("\napp_data")
    print(f"  {meta['size_mb']:.1f} MB written to {cfg['data']['app_data_dir']}")
    print("  RESULTS.md regenerated from data/outputs")
