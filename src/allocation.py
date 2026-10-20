"""Allocation of limited supplier capacity across DCs: LP (weighted) and fair share.

All arrays are shaped (items, dcs). Requests, supply and DC free capacity are
whole units; allocations are returned as whole units.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

EPS = 1e-9


def allocation_weights(abc: np.ndarray, cover_weeks: np.ndarray, abc_weight: dict[str, float]) -> np.ndarray:
    """w = abc_weight[abc] x (1 + 1 / (1 + cover)): favours A items, then the DC closest to stocking out."""
    class_weight = np.vectorize(abc_weight.get)(abc).astype(float)
    return class_weight * (1 + 1 / (1 + np.maximum(cover_weeks, 0)))


def weighted_shortfall(requests: np.ndarray, allocation: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (requests - allocation)))


def _round_allocation(x: np.ndarray, requests: np.ndarray, supply: np.ndarray, dc_free: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    """Round down, then give each item's units lost to rounding, one at a time, to its
    highest-weight unmet request that still has DC room. Never breaks a constraint."""
    alloc = np.floor(x + EPS).astype(np.int64)
    leftover = np.floor(x.sum(axis=1) + EPS).astype(np.int64) - alloc.sum(axis=1)
    room_dc = dc_free - alloc.sum(axis=0)
    for i in np.flatnonzero(leftover > 0):  # rare: only items whose LP split was fractional
        for _ in range(int(min(leftover[i], supply[i] - alloc[i].sum()))):
            open_ = (alloc[i] < requests[i]) & (room_dc > 0)
            if not open_.any():
                break
            d = np.flatnonzero(open_)[np.argmax(weights[i, open_])]
            alloc[i, d] += 1
            room_dc[d] -= 1
    return alloc


def allocate_fair_share(requests: np.ndarray, supply: np.ndarray, dc_free_capacity: np.ndarray,
                        weights: np.ndarray | None = None) -> np.ndarray:
    """Every DC gets the same share of its request; then scale down any DC over its free capacity."""
    requests = np.asarray(requests, dtype=float)
    weights = np.ones_like(requests) if weights is None else weights
    total = requests.sum(axis=1)
    ratio = np.where(total > 0, np.minimum(1.0, np.divide(supply, total, out=np.ones_like(total), where=total > 0)), 1.0)
    x = requests * ratio[:, None]
    dc_sum = x.sum(axis=0)
    scale = np.where(dc_sum > dc_free_capacity, np.divide(dc_free_capacity, dc_sum, out=np.zeros_like(dc_sum), where=dc_sum > 0), 1.0)
    x = x * scale[None, :]
    return _round_allocation(x, requests.astype(np.int64), np.asarray(supply), np.asarray(dc_free_capacity), weights)


def allocate_lp(requests: np.ndarray, supply: np.ndarray, dc_free_capacity: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Maximise sum w x subject to x <= request, item supply and DC free capacity (scipy HiGHS).

    When nothing binds (every item's supply and every DC's room cover the requests),
    filling every request is optimal because all weights are positive, so the LP is skipped.
    On solver failure, falls back to fair share for that week.
    """
    requests = np.asarray(requests, dtype=np.int64)
    supply = np.asarray(supply, dtype=np.int64)
    dc_free = np.maximum(np.asarray(dc_free_capacity, dtype=np.int64), 0)
    if (requests.sum(axis=1) <= supply).all() and (requests.sum(axis=0) <= dc_free).all():
        return requests.copy()

    n_items, n_dcs = requests.shape
    item_rows = np.kron(np.eye(n_items), np.ones(n_dcs))
    dc_rows = np.tile(np.eye(n_dcs), n_items)
    result = linprog(
        c=-weights.ravel(),
        A_ub=np.vstack([item_rows, dc_rows]),
        b_ub=np.concatenate([supply, dc_free]).astype(float),
        bounds=list(zip(np.zeros(requests.size), requests.ravel().astype(float))),
        method="highs",
    )
    if not result.success:
        logger.warning("LP allocation failed (%s); using fair share for this week", result.message)
        return allocate_fair_share(requests, supply, dc_free, weights)
    return _round_allocation(result.x.reshape(n_items, n_dcs), requests, supply, dc_free, weights)


def demo_constrained_week(policy: pd.DataFrame, supply: pd.DataFrame, items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One constrained week, LP vs fair share.

    Setting: each DC holds its opening stock, nothing is in transit (the pipeline was
    emptied by a supply interruption), and supplier capacity is at the
    supplier_disruption level. Each DC requests up to its order-up-to level.
    """
    factor = cfg["scenarios"]["supplier_disruption"]["capacity_factor"]
    p = policy.merge(supply[["item_id", "dc_id", "dc_capacity"]], on=["item_id", "dc_id"])
    item_ids = sorted(p["item_id"].unique())
    dc_ids = sorted(p["dc_id"].unique())

    def grid(col: str) -> np.ndarray:
        return p.pivot(index="item_id", columns="dc_id", values=col).reindex(index=item_ids, columns=dc_ids).to_numpy()

    on_hand = grid("opening_stock").astype(float)
    requests = np.maximum(0, np.ceil(grid("order_up_to") - on_hand)).astype(np.int64)
    cover = np.divide(on_hand, grid("mean_weekly_forecast"), out=np.full_like(on_hand, 52.0), where=grid("mean_weekly_forecast") > 0)
    weights = allocation_weights(grid("abc"), cover, cfg["allocation"]["abc_weight"])
    cap = items.set_index("item_id").reindex(item_ids)["supplier_capacity"].to_numpy()
    week_supply = np.floor(cap * factor).astype(np.int64)
    dc_capacity = p.drop_duplicates("dc_id").set_index("dc_id").reindex(dc_ids)["dc_capacity"].to_numpy()
    dc_free = np.maximum(0, np.floor(dc_capacity - on_hand.sum(axis=0))).astype(np.int64)

    lp = allocate_lp(requests, week_supply, dc_free, weights)
    fair = allocate_fair_share(requests, week_supply, dc_free, weights)
    rows = pd.DataFrame({
        "item_id": np.repeat(item_ids, len(dc_ids)), "dc_id": np.tile(dc_ids, len(item_ids)),
        "abc": grid("abc").ravel(), "request": requests.ravel(), "weight": weights.ravel(),
        "cover_weeks": cover.ravel(), "lp": lp.ravel(), "fair_share": fair.ravel(),
    })
    rows.attrs["summary"] = {
        "requested": int(requests.sum()), "supply": int(week_supply.sum()),
        "constrained_items": int((requests.sum(axis=1) > week_supply).sum()),
        "lp_weighted_shortfall": weighted_shortfall(requests, lp, weights),
        "fair_weighted_shortfall": weighted_shortfall(requests, fair, weights),
    }
    return rows
