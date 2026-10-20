"""Planner dashboard. Reads only app_data/ (written by `python run_all.py --stage app_data`)."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DATA = Path(__file__).resolve().parents[1] / "app_data"

# Validated categorical slots 1 to 3 (blue, orange, aqua); colour follows the entity, never its rank.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
CLASS_COLOR = {"A": SLOT[0], "B": SLOT[1], "C": SLOT[2]}
GRID = "rgba(137,135,129,0.25)"  # muted ink at low alpha: a hairline in light and dark themes
SCENARIO_LABELS = {
    "baseline": "Baseline (LP allocation)",
    "baseline_fair_share": "Baseline, fair-share allocation",
    "demand_surge": "Demand surge: A items in CA +25%, weeks 2 to 5",
    "supplier_disruption": "Supplier disruption: capacity 50%, weeks 3 to 5",
    "lead_time_shock_policy_not_updated": "Lead time +2 weeks, policy not updated",
    "lead_time_shock_policy_updated": "Lead time +2 weeks, policy updated",
    "flat_service_level": "Flat 98% service level for every series",
    "diagnostic_unlimited_supply": "Diagnostic: unlimited supplier capacity",
    "price_cut_policy_not_updated": "Price cut 10% on top 10 items, policy not updated",
    "price_cut_policy_updated": "Price cut 10% on top 10 items, policy updated",
}

st.set_page_config(page_title="Retail Supply Planner", layout="wide")


@st.cache_data
def load() -> dict:
    if not (APP_DATA / "meta.json").exists():
        return {}
    csv = {name: pd.read_csv(APP_DATA / f"{name}.csv") for name in (
        "segments", "segment_sets", "backtest_scores", "model_selection", "final_forecast", "history",
        "inventory_policy", "allocation_demo", "sim_kpis", "sim_weekly", "feature_importance")}
    for optional in ("exceptions",):
        path = APP_DATA / f"{optional}.csv"
        csv[optional] = pd.read_csv(path) if path.exists() else None
    csv["backtest_forecasts"] = pd.read_parquet(APP_DATA / "backtest_forecasts.parquet")
    csv["meta"] = json.loads((APP_DATA / "meta.json").read_text(encoding="utf-8"))
    for name in ("final_forecast", "history"):
        csv[name]["week_start"] = pd.to_datetime(csv[name]["week_start"])
    return csv


def style(fig: go.Figure, height: int = 380, y_title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08, x=0),
                      hovermode="x unified" if any(t.type == "scatter" for t in fig.data) else "closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title)
    bars = [t for t in fig.data if t.type == "bar" and t.orientation != "h"]
    if bars:  # headroom so outside value labels clear the legend
        fig.update_yaxes(range=[0, 1.18 * max(max(t.y) for t in bars)])
    return fig


def kpi_value(kpis: pd.DataFrame, scenario: str, kpi: str, group: str = "All") -> pd.Series:
    row = kpis[(kpis["scenario"] == scenario) & (kpis["kpi"] == kpi) & (kpis["group"] == group)]
    return row.iloc[0]


data = load()
st.sidebar.title("Retail Supply Planner")
if not data:
    st.title("Retail Demand and Supply Planner")
    st.warning("No app data found. Run `python run_all.py --stage all` to build `app_data/`.")
    st.stop()

meta = data["meta"]
PAGES = ["Overview", "Segmentation", "Forecast accuracy", "Item explorer", "Inventory policy", "Allocation", "Scenarios", "Exceptions"]
requested = st.query_params.get("page", "Overview")  # deep link, for example ?page=Scenarios
page = st.sidebar.radio("Page", PAGES, index=PAGES.index(requested) if requested in PAGES else 0)
st.sidebar.caption(f"{meta['items']} items x {len(meta['dcs'])} DCs, {meta['weeks']} weeks of real M5 demand "
                   f"(weeks starting {meta['first_week']} to {meta['last_week']}). Supply parameters are simulated.")

if page == "Overview":
    st.title("Retail Demand and Supply Planner")
    st.write(
        f"Weekly demand is real Walmart sales (M5, FOODS_3, {meta['items']} items in {', '.join(meta['dcs'])}, "
        f"each state treated as one distribution centre). Lead times, supplier capacity, DC capacity, costs and opening stock "
        f"are simulated from that demand. Forecasts are chosen per demand pattern from a 3-fold rolling backtest; safety stock "
        f"comes from measured forecast error; limited supplier capacity is allocated across DCs each week; and "
        f"{meta['simulation_runs']} Monte Carlo runs test the policy over the {meta['horizon_weeks']} weeks after the data ends."
    )
    kpis = data["sim_kpis"]
    available = [s for s in SCENARIO_LABELS if s in set(data["sim_kpis"]["scenario"])]
    scenario = st.selectbox("Scenario", available, format_func=SCENARIO_LABELS.get)
    cols = st.columns(4)
    for col, (kpi, label, fmt) in zip(cols, [
        ("fill_rate", "Fill rate", "{:.1%}"), ("cycle_service", "Cycle service", "{:.1%}"),
        ("avg_inventory_value", "Average inventory value (USD)", "{:,.0f}"), ("lost_sales_value", "Lost sales value (USD)", "{:,.0f}"),
    ]):
        value, base = kpi_value(kpis, scenario, kpi), kpi_value(kpis, "baseline", kpi)
        delta = None if scenario == "baseline" else (
            f"{(value['mean'] - base['mean']) * 100:+.1f} pts vs baseline" if "{:.1%}" == fmt else f"{value['mean'] - base['mean']:+,.0f} vs baseline")
        col.metric(label, fmt.format(value["mean"]), delta,
                   help={"fill_rate": "Share of demand units served from stock.",
                         "cycle_service": "Share of series-weeks with no lost sales."}.get(kpi),
                   delta_color={"lost_sales_value": "inverse", "avg_inventory_value": "off"}.get(kpi, "normal"))
        col.caption(f"p10 to p90: {fmt.format(value['p10'])} to {fmt.format(value['p90'])}")

elif page == "Segmentation":
    st.title("Segmentation")
    st.write("Final segment set: the last 52 launched weeks up to the last week of data. Backtest folds use their own sets.")
    seg = data["segments"]
    left, right = st.columns(2)
    with left:
        st.subheader("ABC by XYZ (series)")
        matrix = pd.crosstab(seg["abc"], seg["xyz"])
        fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
                                   text=matrix.values, texttemplate="%{text}", hovertemplate="ABC %{y}, XYZ %{x}: %{z} series<extra></extra>",
                                   showscale=False))
        st.plotly_chart(style(fig, 300).update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)"), width="stretch")
    with right:
        st.subheader("Demand patterns (series)")
        counts = seg["pattern"].value_counts().reindex(["smooth", "erratic", "intermittent", "lumpy", "inactive"], fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=SLOT[0], text=counts.values, textposition="outside",
                               hovertemplate="%{x}: %{y} series<extra></extra>"))
        st.plotly_chart(style(fig, 300, "series"), width="stretch")
    st.subheader("Pattern counts per segment set")
    st.dataframe(data["segment_sets"], hide_index=True)
    st.subheader("Series")
    f1, f2, f3 = st.columns(3)
    abc = f1.multiselect("ABC", ["A", "B", "C"], default=["A", "B", "C"])
    xyz = f2.multiselect("XYZ", ["X", "Y", "Z"], default=["X", "Y", "Z"])
    patterns = f3.multiselect("Pattern", sorted(seg["pattern"].unique()), default=sorted(seg["pattern"].unique()))
    shown = seg[seg["abc"].isin(abc) & seg["xyz"].isin(xyz) & seg["pattern"].isin(patterns)]
    st.dataframe(shown.sort_values("revenue", ascending=False).round(3), hide_index=True)

elif page == "Forecast accuracy":
    st.title("Forecast accuracy")
    st.write("Rolling-origin backtest: 3 non-overlapping 12-week folds; every model trains only on weeks up to its origin. "
             "WAPE and bias are volume-weighted; FVA is the naive WAPE minus the model WAPE, in points.")
    scores, selection = data["backtest_scores"], data["model_selection"]
    level = st.radio("Group by", ["pattern", "abc"], horizontal=True)
    board = scores[scores["level"].isin(["overall", level])].pivot_table(index="model", columns="group", values="wape")
    order = ["all"] + ([c for c in ["smooth", "erratic", "intermittent", "lumpy", "inactive"] if c in board.columns]
                       if level == "pattern" else ["A", "B", "C"])
    board = board[order].sort_values("all")
    fig = go.Figure(go.Heatmap(z=board.values, x=board.columns, y=board.index, colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
                               zmin=0, zmax=1, text=board.round(3).values, texttemplate="%{text}", showscale=False,
                               hovertemplate="%{y}, %{x}: WAPE %{text}<extra></extra>"))
    st.plotly_chart(style(fig, 330).update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)"), width="stretch")
    st.caption("Lower WAPE is better. Darker cells are worse; colour is capped at WAPE 1.0, the printed values are exact.")
    st.subheader("Selected model per pattern")
    st.dataframe(selection.round(3), hide_index=True)
    st.subheader("Leaderboard")
    table = scores[scores["level"].isin(["overall", level])][["group", "model", "wape", "bias", "mean_mase", "fva_vs_naive_pp", "n_series_folds"]]
    chosen = set(zip(selection["pattern"], selection["model"]))
    st.dataframe(table.sort_values(["group", "wape"]).style.apply(
        lambda r: ["font-weight: 700" if (r["group"], r["model"]) in chosen else "" for _ in r], axis=1).format(precision=3),
        hide_index=True)
    st.caption("Bold rows are the model selected for that pattern.")
    st.subheader("LightGBM feature importance (final models, share of gain)")
    imp = data["feature_importance"]
    imp = imp[imp["fold"] == "final"]
    group = st.selectbox("Horizon group", sorted(imp["horizon_group"].unique()))
    top = imp[imp["horizon_group"] == group].nlargest(15, "gain_share").sort_values("gain_share")
    fig = go.Figure(go.Bar(x=top["gain_share"], y=top["feature"], orientation="h", marker_color=SLOT[0],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    st.plotly_chart(style(fig, 420).update_xaxes(tickformat=".0%"), width="stretch")

elif page == "Item explorer":
    st.title("Item explorer")
    ff, hist, bt = data["final_forecast"], data["history"], data["backtest_forecasts"]
    c1, c2 = st.columns(2)
    item = c1.selectbox("Item", sorted(ff["item_id"].unique()))
    dc = c2.selectbox("DC", sorted(ff[ff["item_id"] == item]["dc_id"].unique()))
    f = ff[(ff["item_id"] == item) & (ff["dc_id"] == dc)]
    h = hist[(hist["item_id"] == item) & (hist["dc_id"] == dc)]
    model = f["model"].iloc[0]
    b = bt[(bt["item_id"] == item) & (bt["dc_id"] == dc) & (bt["model"] == model)].merge(h[["t", "week_start"]], on="t")
    st.write(f"Class **{f['abc'].iloc[0]}**, pattern **{f['pattern'].iloc[0]}**, forecast model **{model}**.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([f["week_start"], f["week_start"][::-1]]), y=pd.concat([f["q95"], f["q50"][::-1]]),
                             fill="toself", fillcolor="rgba(235,104,52,0.15)", line=dict(width=0), name="q50 to q95 band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h["week_start"], y=h["units"], name="Actual", line=dict(color=SLOT[0], width=2)))
    for fold, g in b.groupby("fold"):
        fig.add_trace(go.Scatter(x=g["week_start"], y=g["forecast"], name=f"Backtest forecasts ({model}, 3 folds)",
                                 line=dict(color=SLOT[2], width=2, dash="dot"), legendgroup="bt", showlegend=fold == b["fold"].min(),
                                 hovertemplate=f"{fold}: %{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=f["week_start"], y=f["forecast"], name=f"Final forecast ({model})", line=dict(color=SLOT[1], width=2)))
    st.plotly_chart(style(fig, 420, "units per week"), width="stretch")
    policy = data["inventory_policy"]
    p = policy[(policy["item_id"] == item) & (policy["dc_id"] == dc)].iloc[0]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, [("Lead time (weeks)", f"{p['lead_time_weeks']:.0f}"), ("Safety stock", f"{p['ss_classic']:,.0f}"),
                                          ("Reorder point", f"{p['reorder_point']:,.0f}"), ("Order-up-to", f"{p['order_up_to']:,.0f}"),
                                          ("Service level", f"{p['service_level']:.0%}")]):
        col.metric(label, value)
    with st.expander("Forecast table"):
        st.dataframe(f[["week_id", "week_start", "h", "forecast", "q50", "q90", "q95", "q98", "calendar_estimated"]].round(
            {c: 1 for c in ("forecast", "q50", "q90", "q95", "q98")}), hide_index=True)

elif page == "Inventory policy":
    st.title("Inventory policy")
    st.write("Safety stock from the chosen model's backtest errors (classic: z x sigma x sqrt(lead time)) and from LightGBM "
             "quantiles (quantile: weekly bands added in quadrature). Service levels are cycle service levels by ABC class.")
    policy = data["inventory_policy"]
    summary = policy.groupby("abc").agg(series=("item_id", "size"), service_level=("service_level", "first"),
                                        ss_classic=("ss_classic", "mean"), ss_quantile=("ss_quantile", "mean"),
                                        reorder_point=("reorder_point", "mean"), days_of_cover=("days_of_cover", "median"))
    fig = go.Figure()
    for i, (col, name) in enumerate([("ss_classic", "Classic"), ("ss_quantile", "Quantile")]):
        fig.add_trace(go.Bar(x=summary.index, y=summary[col], name=name, marker_color=SLOT[i], text=summary[col].round(0),
                             textposition="outside", hovertemplate=f"{name}, class %{{x}}: %{{y:,.1f}} units<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05)
    st.subheader("Mean safety stock per series by ABC class")
    st.plotly_chart(style(fig, 360, "units"), width="stretch")
    st.dataframe(summary.round(2).rename(columns={"days_of_cover": "median_days_of_cover"}))
    st.subheader("Series")
    abc = st.multiselect("ABC", ["A", "B", "C"], default=["A", "B", "C"], key="inv_abc")
    st.dataframe(policy[policy["abc"].isin(abc)].round(2), hide_index=True)

elif page == "Allocation":
    st.title("Allocation of constrained supply")
    demo = data["allocation_demo"]
    st.write("One constrained week: each DC holds its opening stock, nothing is in transit, and supplier capacity is at 50%. "
             "The LP maximises weighted shipments (weight = ABC weight x (1 + 1 / (1 + weeks of cover))); fair share gives every "
             "DC the same fraction of its request.")
    by = demo.groupby("abc")[["request", "lp", "fair_share"]].sum()
    fig = go.Figure()
    for i, (col, name) in enumerate([("request", "Requested"), ("lp", "LP"), ("fair_share", "Fair share")]):
        fig.add_trace(go.Bar(x=by.index, y=by[col], name=name, marker_color=SLOT[i], text=by[col].map("{:,}".format),
                             textposition="outside", hovertemplate=f"{name}, class %{{x}}: %{{y:,}} units<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05)
    st.plotly_chart(style(fig, 380, "units"), width="stretch")
    shortfall = {m: float((demo["weight"] * (demo["request"] - demo[m])).sum()) for m in ("lp", "fair_share")}
    c1, c2 = st.columns(2)
    c1.metric("Weighted shortfall, LP", f"{shortfall['lp']:,.0f}")
    c2.metric("Weighted shortfall, fair share", f"{shortfall['fair_share']:,.0f}")
    st.subheader("Over 12 simulated weeks")
    kpis = data["sim_kpis"]
    fill = kpis[(kpis["kpi"] == "fill_rate") & kpis["scenario"].isin(["baseline", "baseline_fair_share"])].pivot(
        index="group", columns="scenario", values="mean").rename(columns={"baseline": "LP", "baseline_fair_share": "Fair share"})
    st.dataframe(fill.round(3))
    st.write("The LP wins on its own objective in a single week, but fair share gives a higher fill rate over 12 weeks: a linear "
             "objective fills the top-weight DC's whole request (safety stock included) before the next DC gets anything, while "
             "fair share hedges across DCs. See DECISIONS.md, Phase 8.")

elif page == "Scenarios":
    st.title("Scenarios")
    kpis, weekly = data["sim_kpis"], data["sim_weekly"]
    scenario = st.selectbox("Scenario", [s for s in SCENARIO_LABELS if s != "baseline" and s in set(kpis["scenario"])],
                            format_func=SCENARIO_LABELS.get)
    group = st.radio("Series", ["All", "A", "B", "C"], horizontal=True)
    formats = {"fill_rate": "{:.1%}", "cycle_service": "{:.1%}", "stockout_weeks": "{:.2f}", "avg_inventory_value": "{:,.0f}",
               "holding_cost": "{:,.0f}", "lost_sales_value": "{:,.0f}", "days_of_cover": "{:.1f}", "revenue": "{:,.0f}"}
    rows = []
    for kpi, fmt in formats.items():
        sc, b = kpi_value(kpis, scenario, kpi, group), kpi_value(kpis, "baseline", kpi, group)
        change = sc["mean"] - b["mean"]
        rows.append({"KPI": kpi, "scenario mean": fmt.format(sc["mean"]),
                     "scenario p10 to p90": f"{fmt.format(sc['p10'])} to {fmt.format(sc['p90'])}",
                     "baseline mean": fmt.format(b["mean"]),
                     "change": f"{change * 100:+.1f} pts" if "%" in fmt else ("+" if change >= 0 else "") + fmt.format(change)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Values in USD for inventory, holding cost and lost sales.")
    st.subheader("Weekly fill rate")
    fig = go.Figure()
    for i, name in enumerate(["baseline", scenario]):
        w = weekly[(weekly["scenario"] == name) & (weekly["group"] == group) & (weekly["metric"] == "fill_rate")]
        color = SLOT[i]
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)"
        fig.add_trace(go.Scatter(x=pd.concat([w["week"], w["week"][::-1]]), y=pd.concat([w["p90"], w["p10"][::-1]]), fill="toself",
                                 fillcolor=rgba, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=w["week"], y=w["mean"], name=SCENARIO_LABELS[name], line=dict(color=color, width=2),
                                 mode="lines+markers", marker=dict(size=8)))
    st.plotly_chart(style(fig, 400, "fill rate").update_yaxes(tickformat=".0%").update_xaxes(title="week", dtick=1), width="stretch")
    st.caption("Lines are means across runs; shaded bands are the 10th to 90th percentile.")

elif page == "Exceptions":
    st.title("Planner exceptions")
    exc = data["exceptions"]
    if exc is None:
        st.info("Run `python run_all.py --stage exceptions` and then `--stage app_data` to build the exception list.")
        st.stop()
    st.write("Series that need a planner's attention, ranked by value at risk (USD): stockout risk (projected on hand below "
             "safety stock within 4 weeks, expected demand and baseline supply), overstock (more than 60 days of cover), forecast "
             "bias beyond 20% over the last backtest fold, and a last week outside the range implied by past forecast errors.")
    summary = exc.groupby("exception_type").agg(count=("rank", "size"), value_at_risk=("value_at_risk", "sum")).sort_values("value_at_risk")
    fig = go.Figure(go.Bar(x=summary["value_at_risk"], y=summary.index, orientation="h", marker_color=SLOT[0],
                           text=[f"{v:,.0f} ({c})" for v, c in zip(summary["value_at_risk"], summary["count"])], textposition="outside",
                           hovertemplate="%{y}: %{x:,.0f} USD<extra></extra>"))
    fig = style(fig, 260, None).update_xaxes(range=[0, 1.3 * summary["value_at_risk"].max()], title="value at risk (USD)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Labels show value at risk and, in brackets, the number of exceptions.")
    c1, c2 = st.columns(2)
    types = c1.multiselect("Type", sorted(exc["exception_type"].unique()), default=sorted(exc["exception_type"].unique()))
    abc = c2.multiselect("ABC", ["A", "B", "C"], default=["A", "B", "C"], key="exc_abc")
    shown = exc[exc["exception_type"].isin(types) & exc["abc"].isin(abc)]
    st.dataframe(shown.style.format({"value_at_risk": "{:,.0f}"}), hide_index=True)
