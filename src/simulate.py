"""Monte Carlo simulation of 12 weeks of replenishment under forecast uncertainty and scenarios.

State arrays are shaped (runs, items, dcs). Each week: receive arrivals, meet
demand from stock (unmet demand is lost, no backorders), order up to the policy
level, allocate limited supplier capacity across DCs, and ship with the item's
lead time. Every scenario uses the same demand noise (common random numbers), so
differences between scenarios come from the scenario, not from sampling luck.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.allocation import allocate_fair_share, allocate_lp
from src.errors import DataQualityError
from src.inventory import compute_policy, forecast_error_sigma

logger = logging.getLogger(__name__)

SERIES_KEYS = ["item_id", "dc_id"]
KPIS = ["fill_rate", "cycle_service", "stockout_weeks", "avg_inventory_units", "avg_inventory_value",
        "holding_cost", "lost_sales_value", "days_of_cover", "revenue"]


@dataclass(frozen=True)
class SimInputs:
    """Static inputs. Shapes: (I, D) per series, (I,) per item, (D,) per DC, (I, D, H) per week."""
    item_ids: list
    dc_ids: list
    abc: np.ndarray
    forecast: np.ndarray
    order_up_to: np.ndarray
    lead_time: np.ndarray
    supplier_capacity: np.ndarray
    dc_capacity: np.ndarray
    opening_stock: np.ndarray
    pipeline_per_week: np.ndarray
    mean_weekly_forecast: np.ndarray
    unit_cost: np.ndarray
    unit_price: np.ndarray
    weekly_holding_cost: np.ndarray
    groups: dict = field(default_factory=dict)  # extra KPI groups: name -> (I, D) bool mask


@dataclass(frozen=True)
class Scenario:
    name: str
    demand_multiplier: np.ndarray | None = None  # (H, I, D)
    capacity_factor: np.ndarray | None = None  # (H,)
    extra_lead_weeks: int = 0
    order_up_to: np.ndarray | None = None  # replaces the baseline policy
    method: str = "lp"
    price_multiplier: np.ndarray | None = None  # (I,) selling price relative to the baseline price


def residual_pools(residuals: pd.DataFrame, series: pd.DataFrame, horizon_groups: list[int], min_n: int) -> list[list[np.ndarray]]:
    """For each series (rows of `series`, with item_id, dc_id, pattern, mean_weekly_demand) and horizon group,
    the residuals to sample from: its own if it has at least min_n, otherwise the pooled residuals of its
    pattern, scaled by its mean demand relative to the pool's mean demand."""
    lower = [1] + [g + 1 for g in horizon_groups[:-1]]
    res = residuals.merge(series[SERIES_KEYS + ["pattern", "mean_weekly_demand"]], on=SERIES_KEYS)
    pools, n_pooled = [], 0
    for row in series.itertuples(index=False):
        own_all = res[(res["item_id"] == row.item_id) & (res["dc_id"] == row.dc_id)]
        per_group = []
        for lo, hi in zip(lower, horizon_groups):
            own = own_all.loc[own_all["h"].between(lo, hi), "residual"].to_numpy(dtype=float)
            if len(own) >= min_n:
                per_group.append(own)
                continue
            pool = res[(res["pattern"] == row.pattern) & res["h"].between(lo, hi)]
            pool_mean = pool.drop_duplicates(SERIES_KEYS)["mean_weekly_demand"].mean()
            if len(pool) == 0 or not pool_mean > 0:
                raise DataQualityError(f"No residuals to sample for {row.item_id} {row.dc_id}, h {lo} to {hi}.")
            per_group.append(pool["residual"].to_numpy(dtype=float) * row.mean_weekly_demand / pool_mean)
            n_pooled += 1
        pools.append(per_group)
    logger.info("Residual pools: %d of %d series-groups use pooled pattern residuals", n_pooled, len(series) * len(horizon_groups))
    return pools


def draw_noise(pools: list[list[np.ndarray]], n_items: int, n_dcs: int, horizon_groups: list[int],
               runs: int, rng: np.random.Generator) -> np.ndarray:
    """Residual draws shaped (runs, I, D, H); series are ordered item-major as in pools."""
    lower = [1] + [g + 1 for g in horizon_groups[:-1]]
    noise = np.zeros((runs, n_items * n_dcs, horizon_groups[-1]))
    for s, per_group in enumerate(pools):
        for (lo, hi), pool in zip(zip(lower, horizon_groups), per_group):
            noise[:, s, lo - 1:hi] = rng.choice(pool, size=(runs, hi - lo + 1))
    return noise.reshape(runs, n_items, n_dcs, -1)


def simulate(inputs: SimInputs, noise: np.ndarray, scenario: Scenario, abc_weight: dict,
             cover_basis: str = "position") -> dict[str, np.ndarray]:
    """Run the weekly loop. Returns arrays shaped (runs, H, I, D): demand, sold, lost, on_hand, in_transit, arrivals, shipped.

    cover_basis sets the weeks of cover in the allocation weights: "position" (on hand
    plus in transit, the stock a shipment will join when it arrives) or "on_hand".
    """
    runs, n_items, n_dcs, horizon = noise.shape
    lead_new = inputs.lead_time + scenario.extra_lead_weeks
    arrivals = np.zeros((runs, n_items, n_dcs, horizon + int(lead_new.max()) + 2), dtype=np.int64)
    weeks = np.arange(arrivals.shape[-1])
    initial = (weeks[None, :] >= 1) & (weeks[None, :] <= inputs.lead_time[:, None])  # (I, weeks)
    arrivals += (initial[:, None, :] * inputs.pipeline_per_week[:, :, None]).astype(np.int64)[None]

    on_hand = np.broadcast_to(inputs.opening_stock, (runs, n_items, n_dcs)).astype(np.int64).copy()
    order_up_to = inputs.order_up_to if scenario.order_up_to is None else scenario.order_up_to
    capacity_factor = np.ones(horizon) if scenario.capacity_factor is None else scenario.capacity_factor
    class_weight = np.vectorize(abc_weight.get)(inputs.abc).astype(float)
    allocate = allocate_lp if scenario.method == "lp" else allocate_fair_share
    out = {k: np.zeros((runs, horizon, n_items, n_dcs), dtype=np.int64)
           for k in ("demand", "sold", "lost", "on_hand", "in_transit", "arrivals", "shipped")}
    out["supply_binding"] = np.zeros((runs, horizon, n_items), dtype=bool)  # item requests exceeded its supply

    for w in range(1, horizon + 1):
        start = on_hand.copy()
        arrived = arrivals[..., w]
        on_hand += arrived

        expected = np.maximum(0.0, inputs.forecast[None, :, :, w - 1] + noise[..., w - 1])
        if scenario.demand_multiplier is not None:
            expected = expected * scenario.demand_multiplier[w - 1][None]
        demand = np.rint(expected).astype(np.int64)
        sold = np.minimum(demand, on_hand)
        on_hand -= sold

        in_transit = arrivals[..., w + 1:].sum(axis=-1)
        position = on_hand + in_transit
        requests = np.maximum(0, np.ceil(order_up_to[None] - position)).astype(np.int64)
        supply = np.floor(inputs.supplier_capacity * capacity_factor[w - 1]).astype(np.int64)
        dc_free = np.maximum(0, inputs.dc_capacity[None] - position.sum(axis=1)).astype(np.int64)
        stock = position if cover_basis == "position" else on_hand
        cover = np.divide(stock, inputs.mean_weekly_forecast[None], out=np.full(on_hand.shape, 52.0),
                          where=inputs.mean_weekly_forecast[None] > 0)
        weights = class_weight[None] * (1 + 1 / (1 + cover))
        shipped = np.stack([allocate(requests[r], supply, dc_free[r], weights[r]) for r in range(runs)])
        for lt in np.unique(lead_new):
            rows = lead_new == lt
            arrivals[..., w + lt][:, rows, :] += shipped[:, rows, :]  # basic index first keeps axis order

        if not (np.array_equal(start + arrived - sold, on_hand) and (on_hand >= 0).all()):
            raise DataQualityError(f"Stock balance violated in week {w} of scenario {scenario.name}.")
        for key, value in (("demand", demand), ("sold", sold), ("lost", demand - sold), ("on_hand", on_hand),
                           ("in_transit", arrivals[..., w + 1:].sum(axis=-1)), ("arrivals", arrived), ("shipped", shipped)):
            out[key][:, w - 1] = value
        out["supply_binding"][:, w - 1] = requests.sum(axis=2) > supply[None]
    return out


def _group_masks(inputs: SimInputs) -> dict[str, np.ndarray]:
    masks = {"All": np.ones(inputs.abc.shape, dtype=bool)}
    masks |= {c: inputs.abc == c for c in ("A", "B", "C")}
    return masks | inputs.groups


def run_kpis(results: dict[str, np.ndarray], inputs: SimInputs, price_multiplier: np.ndarray | None = None) -> pd.DataFrame:
    """Per run and group (All, A, B, C and any extra groups): the KPIs in KPIS.

    Revenue and lost sales are valued at the selling price, which a price scenario changes via price_multiplier.
    """
    horizon = results["demand"].shape[1]
    cost = np.broadcast_to(inputs.unit_cost[:, None], inputs.abc.shape)
    selling = inputs.unit_price * (1.0 if price_multiplier is None else price_multiplier)
    price = np.broadcast_to(selling[:, None], inputs.abc.shape)
    holding = np.broadcast_to(inputs.weekly_holding_cost[:, None], inputs.abc.shape)
    frames = []
    for group, mask in _group_masks(inputs).items():
        if not mask.any():
            continue
        d, s, lost, oh = (results[k][:, :, mask] for k in ("demand", "sold", "lost", "on_hand"))
        demand_total = d.sum(axis=(1, 2))
        inv_units = oh.sum(axis=2).mean(axis=1)
        frames.append(pd.DataFrame({
            "group": group,
            "run": np.arange(d.shape[0]),
            "fill_rate": np.divide(s.sum(axis=(1, 2)), demand_total, out=np.ones(len(demand_total)), where=demand_total > 0),
            "cycle_service": (lost == 0).mean(axis=(1, 2)),
            "stockout_weeks": (lost > 0).sum(axis=1).mean(axis=1),
            "avg_inventory_units": inv_units,
            "avg_inventory_value": (oh * cost[mask]).sum(axis=2).mean(axis=1),
            "holding_cost": (oh * holding[mask]).sum(axis=(1, 2)),
            "lost_sales_value": (lost * price[mask]).sum(axis=(1, 2)),
            "days_of_cover": np.divide(inv_units * 7, demand_total / horizon, out=np.full(len(inv_units), np.nan), where=demand_total > 0),
            "revenue": (s * price[mask]).sum(axis=(1, 2)),
        }))
    return pd.concat(frames, ignore_index=True)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean, 10th and 90th percentile across runs, per group and KPI."""
    long = per_run.melt(id_vars=["group", "run"], value_vars=KPIS, var_name="kpi")
    return long.groupby(["group", "kpi"])["value"].agg(
        mean="mean", p10=lambda v: v.quantile(0.10), p90=lambda v: v.quantile(0.90)).reset_index()


def weekly_summary(results: dict[str, np.ndarray], inputs: SimInputs) -> pd.DataFrame:
    """Per week and group: fill rate, demand, lost units and on-hand units (mean, p10, p90 across runs)."""
    frames = []
    for group, mask in _group_masks(inputs).items():
        if not mask.any():
            continue
        d, s, lost, oh = (results[k][:, :, mask].sum(axis=2) for k in ("demand", "sold", "lost", "on_hand"))
        metrics = {"fill_rate": np.divide(s, d, out=np.ones(d.shape), where=d > 0), "demand": d, "lost": lost, "on_hand": oh}
        for name, values in metrics.items():
            frames.append(pd.DataFrame({
                "group": group, "week": np.arange(1, d.shape[1] + 1), "metric": name,
                "mean": values.mean(axis=0), "p10": np.quantile(values, 0.10, axis=0), "p90": np.quantile(values, 0.90, axis=0),
            }))
    return pd.concat(frames, ignore_index=True)


def _grid(frame: pd.DataFrame, col: str, item_ids: list, dc_ids: list) -> np.ndarray:
    return frame.pivot(index="item_id", columns="dc_id", values=col).reindex(index=item_ids, columns=dc_ids).to_numpy()


def build_inputs(policy: pd.DataFrame, final_forecast: pd.DataFrame, supply: pd.DataFrame, items: pd.DataFrame) -> SimInputs:
    item_ids = sorted(policy["item_id"].unique())
    dc_ids = sorted(policy["dc_id"].unique())
    p = policy.merge(supply[SERIES_KEYS + ["dc_capacity"]], on=SERIES_KEYS)
    horizon = int(final_forecast["h"].max())
    fc = np.stack([_grid(final_forecast[final_forecast["h"] == h], "forecast", item_ids, dc_ids) for h in range(1, horizon + 1)], axis=-1)
    it = items.set_index("item_id").reindex(item_ids)
    model = _grid(p, "model", item_ids, dc_ids)
    return SimInputs(
        item_ids=item_ids, dc_ids=dc_ids,
        abc=_grid(p, "abc", item_ids, dc_ids),
        forecast=fc,
        order_up_to=_grid(p, "order_up_to", item_ids, dc_ids).astype(float),
        lead_time=it["lead_time_weeks"].to_numpy(dtype=np.int64),
        supplier_capacity=it["supplier_capacity"].to_numpy(dtype=np.int64),
        dc_capacity=p.drop_duplicates("dc_id").set_index("dc_id").reindex(dc_ids)["dc_capacity"].to_numpy(dtype=np.int64),
        opening_stock=_grid(p, "opening_stock", item_ids, dc_ids).astype(np.int64),
        pipeline_per_week=np.rint(_grid(p, "mean_weekly_demand", item_ids, dc_ids)).astype(np.int64),
        mean_weekly_forecast=_grid(p, "mean_weekly_forecast", item_ids, dc_ids).astype(float),
        unit_cost=it["unit_cost"].to_numpy(dtype=float),
        unit_price=it["unit_price"].to_numpy(dtype=float),
        weekly_holding_cost=it["weekly_holding_cost"].to_numpy(dtype=float),
        groups={f"model_{m}": model == m for m in sorted(set(model.ravel()))},
    )


def price_cut_inputs(final_forecast: pd.DataFrame, final_segments: pd.DataFrame, elasticity: pd.DataFrame,
                     cfg: dict) -> tuple[list[str], pd.Series, pd.DataFrame]:
    """Items to cut (top N by revenue), their demand multipliers price_factor ** elasticity, and the forecast scaled by them."""
    pc = cfg["scenarios"]["price_cut"]
    revenue = final_segments.groupby("item_id")["revenue"].sum().sort_values(ascending=False)
    cut = sorted(revenue.index[: pc["top_n_items_by_revenue"]])
    multiplier = (pc["price_factor"] ** elasticity.set_index("item_id")["elasticity"]).reindex(cut)
    scaled = final_forecast.copy()
    factor = scaled["item_id"].map(multiplier).fillna(1.0)
    for col in ["forecast"] + [c for c in scaled.columns if c.startswith("q") and c[1:].isdigit()]:
        scaled[col] = scaled[col] * factor
    return cut, multiplier, scaled


def build_scenarios(inputs: SimInputs, cfg: dict, policies: dict[str, pd.DataFrame],
                    price_cut: tuple[list[str], pd.Series] | None = None) -> list[Scenario]:
    horizon = inputs.forecast.shape[-1]
    sc = cfg["scenarios"]

    def window(start: int, duration: int) -> np.ndarray:
        weeks = np.arange(1, horizon + 1)
        return (weeks >= start) & (weeks < start + duration)

    surge = sc["demand_surge"]
    multiplier = np.ones((horizon, *inputs.abc.shape))
    cells = (inputs.abc == surge["abc_class"]) & (np.array(inputs.dc_ids)[None, :] == surge["dc_id"])
    multiplier[window(surge["start_week"], surge["duration_weeks"])] += surge["uplift"] * cells

    disruption = sc["supplier_disruption"]
    capacity = np.where(window(disruption["start_week"], disruption["duration_weeks"]), disruption["capacity_factor"], 1.0)
    extra = sc["lead_time_shock"]["extra_weeks"]

    def oup(name: str) -> np.ndarray:
        return _grid(policies[name], "order_up_to", inputs.item_ids, inputs.dc_ids).astype(float)

    return [
        Scenario("baseline"),
        Scenario("baseline_fair_share", method="fair_share"),
        Scenario("demand_surge", demand_multiplier=multiplier),
        Scenario("supplier_disruption", capacity_factor=capacity),
        Scenario("lead_time_shock_policy_not_updated", extra_lead_weeks=extra),
        Scenario("lead_time_shock_policy_updated", extra_lead_weeks=extra, order_up_to=oup("lead_time_shock")),
        Scenario("flat_service_level", order_up_to=oup("flat_service_level")),
        # Diagnostic, not a spec scenario: separates the policy's effect from the supplier capacity limit.
        Scenario("diagnostic_unlimited_supply", capacity_factor=np.full(horizon, 1000.0)),
    ] + _price_cut_scenarios(inputs, cfg, policies, price_cut)


def _price_cut_scenarios(inputs: SimInputs, cfg: dict, policies: dict, price_cut) -> list[Scenario]:
    """Stretch: a price cut on chosen items, with the policy left as is and with the policy updated for the uplift."""
    if price_cut is None:
        return []
    cut, multiplier = price_cut
    horizon = inputs.forecast.shape[-1]
    rows = np.isin(inputs.item_ids, cut)
    item_mult = np.ones(len(inputs.item_ids))
    item_mult[rows] = multiplier.reindex(np.array(inputs.item_ids)[rows]).to_numpy()
    demand = np.broadcast_to(item_mult[None, :, None], (horizon, *inputs.abc.shape)).copy()
    prices = np.where(rows, cfg["scenarios"]["price_cut"]["price_factor"], 1.0)
    updated = _grid(policies["price_cut"], "order_up_to", inputs.item_ids, inputs.dc_ids).astype(float)
    return [
        Scenario("price_cut_policy_not_updated", demand_multiplier=demand, price_multiplier=prices),
        Scenario("price_cut_policy_updated", demand_multiplier=demand, price_multiplier=prices, order_up_to=updated),
    ]


def run(cfg: dict) -> None:
    """Stage `simulate`: write data/outputs/sim_kpis.csv and sim_weekly.parquet."""
    start_all = time.perf_counter()
    paths = cfg["data"]
    out_dir = paths["outputs_dir"]
    final_forecast = pd.read_parquet(out_dir / "final_forecast.parquet")
    residuals = pd.read_parquet(out_dir / "residuals.parquet")
    policy = pd.read_parquet(out_dir / "inventory_policy.parquet")
    gold = pd.read_parquet(paths["gold_dir"] / "weekly.parquet")
    supply = pd.read_parquet(paths["gold_dir"] / "supply_params.parquet")
    items = pd.read_parquet(paths["gold_dir"] / "item_params.parquet")
    sim_cfg = cfg["simulation"]
    if sim_cfg["horizon_weeks"] != final_forecast["h"].max():
        raise DataQualityError("simulation.horizon_weeks must equal the final forecast horizon.")

    sigma = forecast_error_sigma(residuals, gold, cfg["inventory"]["min_error_points"])
    policies = {
        "lead_time_shock": compute_policy(final_forecast, sigma, supply, items, cfg,
                                          extra_lead_weeks=cfg["scenarios"]["lead_time_shock"]["extra_weeks"]),
        "flat_service_level": compute_policy(final_forecast, sigma, supply, items, cfg,
                                             service_level=cfg["scenarios"]["flat_service_level"]["service_level"]),
    }
    price_cut = None
    if (out_dir / "elasticity.csv").exists():
        segments = pd.read_parquet(paths["gold_dir"] / "segments.parquet")
        cut, multiplier, scaled = price_cut_inputs(final_forecast, segments[segments["segment_set"] == "final"],
                                                   pd.read_csv(out_dir / "elasticity.csv"), cfg)
        policies["price_cut"] = compute_policy(scaled, sigma, supply, items, cfg)
        price_cut = (cut, multiplier)
        logger.info("Price cut on %d items; demand multipliers %s", len(cut), multiplier.round(3).to_dict())
    else:
        logger.info("No elasticity.csv: price-cut scenarios skipped (run the elasticity stage first)")
    inputs = build_inputs(policy, final_forecast, supply, items)
    series = pd.DataFrame({"item_id": np.repeat(inputs.item_ids, len(inputs.dc_ids)), "dc_id": np.tile(inputs.dc_ids, len(inputs.item_ids))})
    series = series.merge(policy[SERIES_KEYS + ["pattern", "mean_weekly_demand"]], on=SERIES_KEYS, how="left", validate="one_to_one")
    pools = residual_pools(residuals, series, cfg["forecast"]["horizon_groups"], sim_cfg["min_residuals_per_series"])
    rng = np.random.default_rng(cfg["seed"])
    noise = draw_noise(pools, len(inputs.item_ids), len(inputs.dc_ids), cfg["forecast"]["horizon_groups"], sim_cfg["runs"], rng)

    kpi_frames, weekly_frames, timings, supply_binding = [], [], {}, {}
    for scenario in build_scenarios(inputs, cfg, policies, price_cut):
        start = time.perf_counter()
        results = simulate(inputs, noise, scenario, cfg["allocation"]["abc_weight"], cfg["allocation"]["cover_basis"])
        supply_binding[scenario.name] = round(float(results["supply_binding"].mean()), 3)
        kpi_frames.append(summarise(run_kpis(results, inputs, scenario.price_multiplier)).assign(scenario=scenario.name))
        weekly_frames.append(weekly_summary(results, inputs).assign(scenario=scenario.name))
        timings[scenario.name] = round(time.perf_counter() - start, 1)
        logger.info("Scenario %s: %.1f s", scenario.name, timings[scenario.name])

    kpis = pd.concat(kpi_frames, ignore_index=True)[["scenario", "group", "kpi", "mean", "p10", "p90"]]
    kpis.to_csv(out_dir / "sim_kpis.csv", index=False)
    pd.concat(weekly_frames, ignore_index=True).to_parquet(out_dir / "sim_weekly.parquet", index=False)
    pd.DataFrame({"scenario": list(supply_binding), "share_item_weeks_supply_binding": list(supply_binding.values())}).to_csv(
        out_dir / "sim_supply_binding.csv", index=False)

    pd.set_option("display.width", 220)
    overall = kpis[kpis["group"] == "All"].pivot(index="scenario", columns="kpi", values="mean")[KPIS]
    order = [s.name for s in build_scenarios(inputs, cfg, policies, price_cut)]
    print(f"\nSimulation: {sim_cfg['runs']} runs x {sim_cfg['horizon_weeks']} weeks, {len(series)} series (means across runs, all series)")
    print(overall.reindex(order).round(3).to_string())
    fill = kpis[kpis["kpi"] == "fill_rate"].pivot(index="scenario", columns="group", values="mean").reindex(order)
    print("\nFill rate by group (mean across runs)")
    print(fill.round(3).to_string())
    band = kpis[(kpis["group"] == "All") & (kpis["kpi"] == "fill_rate")].set_index("scenario").reindex(order)
    print("\nFill rate p10 to p90 (all series): " + "; ".join(f"{s} {r.p10:.3f} to {r.p90:.3f}" for s, r in band.iterrows()))
    print(f"\nShare of item-weeks where requests exceeded supplier capacity: {supply_binding}")
    print(f"Runtime per scenario (s): {timings}; total {time.perf_counter() - start_all:.1f} s")
