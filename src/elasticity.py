"""Own-price elasticity per item, pooled across DCs, with shrinkage toward the department mean."""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def estimate_item(rows: pd.DataFrame, trend: bool = False) -> dict:
    """OLS of log(units) on log(price) plus DC and month indicators, event_days and snap_days.

    rows: one item's launched weeks with units > 0 (item_id, dc_id, week_start, units, price, event_days, snap_days).
    trend adds a linear time trend in years (sensitivity check only; the spec's model has none).
    Returns e_raw = NaN when the price effect is not identified (constant price, or price collinear with the controls).
    """
    prices = rows["price"]
    out = {"n_weeks": len(rows), "distinct_prices": int(prices.round(2).nunique()),
           "price_cv": float(prices.std() / prices.mean()) if prices.mean() > 0 else 0.0}
    if prices.nunique() < 2:
        return out | {"e_raw": np.nan, "se": np.nan, "r2": np.nan}
    month = pd.to_datetime(rows["week_start"]).dt.month.astype(str)
    X = pd.concat([
        np.log(prices).rename("log_price"),
        pd.get_dummies(rows["dc_id"], prefix="dc", drop_first=True, dtype=float),
        pd.get_dummies(month, prefix="m", drop_first=True, dtype=float),
        rows[["event_days", "snap_days"]].astype(float),
    ], axis=1)
    if trend:
        X["trend_years"] = (pd.to_datetime(rows["week_start"]) - pd.Timestamp("2011-01-01")).dt.days / 365.25
    X = X.loc[:, (X.std() > 0) | (X.columns == "log_price")]  # constant controls duplicate the intercept
    design = sm.add_constant(X, has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return out | {"e_raw": np.nan, "se": np.nan, "r2": np.nan}
    fit = sm.OLS(np.log(rows["units"].astype(float)), design).fit()
    out |= {"e_raw": float(fit.params["log_price"]), "se": float(fit.bse["log_price"]), "r2": float(fit.rsquared)}
    return out


def estimate_all(weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per item: raw estimate, status, shrunk and clipped elasticity.

    Items with fewer than min_distinct_prices prices or a price CV below min_price_cv
    are `insufficient_variation` and take the department mean. Others are shrunk:
    e = n / (n + k) x e_item + k / (n + k) x e_dept, with k = shrinkage_weeks. Then clipped.
    """
    el = cfg["elasticity"]
    usable = weekly[weekly["is_launched"] & (weekly["units"] > 0)]
    rows = []
    for item, g in usable.groupby("item_id"):
        est = estimate_item(g)
        varies = est["distinct_prices"] >= el["min_distinct_prices"] and est["price_cv"] > el["min_price_cv"]
        status = "ok" if varies and np.isfinite(est["e_raw"]) else ("not_identified" if varies else "insufficient_variation")
        rows.append({"item_id": item, **est, "status": status, "e_raw_with_trend": estimate_item(g, trend=True)["e_raw"]})
    table = pd.DataFrame(rows)
    ok = table["status"] == "ok"
    e_dept = float(table.loc[ok, "e_raw"].mean())
    table["e_dept"] = e_dept
    table["raw_positive"] = ok & (table["e_raw"] > 0)
    item_e = table["e_raw"].where(ok, e_dept)
    k = el["shrinkage_weeks"]
    table["e_shrunk"] = table["n_weeks"] / (table["n_weeks"] + k) * item_e + k / (table["n_weeks"] + k) * e_dept
    low, high = el["clip"]
    table["elasticity"] = table["e_shrunk"].clip(low, high)
    table["clipped"] = table["elasticity"] != table["e_shrunk"]
    return table


def run(cfg: dict) -> None:
    """Stage `elasticity`: write data/outputs/elasticity.csv."""
    gold = pd.read_parquet(cfg["data"]["gold_dir"] / "weekly.parquet")
    table = estimate_all(gold, cfg)
    table.to_csv(cfg["data"]["outputs_dir"] / "elasticity.csv", index=False)
    ok = table[table["status"] == "ok"]
    print("\nPrice elasticity (per item, pooled across DCs)")
    print(f"  items: {len(table)}; status counts {table['status'].value_counts().to_dict()}")
    print(f"  department mean of raw estimates: {table['e_dept'].iloc[0]:.3f}")
    print(f"  raw estimates: median {ok['e_raw'].median():.3f}, 10th to 90th percentile {ok['e_raw'].quantile(0.1):.3f} to {ok['e_raw'].quantile(0.9):.3f}")
    print(f"  raw positive (flagged, wrong sign): {int(table['raw_positive'].sum())}; clipped to [{cfg['elasticity']['clip'][0]}, {cfg['elasticity']['clip'][1]}]: {int(table['clipped'].sum())}")
    print(f"  final elasticity: median {table['elasticity'].median():.3f}, mean {table['elasticity'].mean():.3f}")
    print(f"  share of ok items with |t| > 2: {(ok['e_raw'].abs() / ok['se'] > 2).mean():.1%}")
    trend = ok["e_raw_with_trend"].dropna()
    print(f"  sensitivity, with a time trend (not used): median {trend.median():.3f}, mean {trend.mean():.3f}, "
          f"positive {int((trend > 0).sum())} of {len(trend)} (spec model: {int((ok['e_raw'] > 0).sum())} of {len(ok)})")
