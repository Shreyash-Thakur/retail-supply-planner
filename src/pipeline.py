"""Medallion pipeline: raw M5 CSVs to bronze parquet, silver daily sales, gold weekly table."""

import logging
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.errors import DataQualityError

logger = logging.getLogger(__name__)

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
PRICES_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]
SERIES_KEYS = ["item_id", "dc_id"]
GOLD_KEYS = ["item_id", "dc_id", "week_id"]
GOLD_COLUMNS = [
    "item_id", "dc_id", "week_id", "week_start", "t", "week_of_year", "year", "units", "price",
    "event_days", "event_sporting", "event_cultural", "event_national", "event_religious",
    "snap_days", "is_launched",
]


def _raise(message: str, examples: pd.DataFrame | None = None) -> None:
    """Raise DataQualityError with up to 5 example rows appended."""
    if examples is not None and len(examples):
        message = f"{message}\nExamples:\n{examples.head(5).to_string()}"
    raise DataQualityError(message)


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return f.readline().strip().split(",")


def _compare_columns(name: str, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        _raise(f"{name}: columns differ from the spec. Missing: {missing}. Extra: {extra}.")


def verify_raw_columns(raw_dir: Path) -> int:
    """Check the three raw files have the spec's columns. Returns the number of sales days."""
    for file in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        if not (raw_dir / file).exists():
            _raise(f"{file} not found in {raw_dir}. See README 'Data download'.")

    sales_header = _header(raw_dir / SALES_FILE)
    _compare_columns(SALES_FILE, sales_header[: len(SALES_ID_COLUMNS)], SALES_ID_COLUMNS)
    day_columns = sales_header[len(SALES_ID_COLUMNS):]
    n_days = len(day_columns)
    _compare_columns(SALES_FILE, day_columns, [f"d_{i}" for i in range(1, n_days + 1)])
    _compare_columns(CALENDAR_FILE, _header(raw_dir / CALENDAR_FILE), CALENDAR_COLUMNS)
    _compare_columns(PRICES_FILE, _header(raw_dir / PRICES_FILE), PRICES_COLUMNS)

    calendar_days = set(pd.read_csv(raw_dir / CALENDAR_FILE, usecols=["d"])["d"])
    uncovered = [c for c in day_columns if c not in calendar_days]
    if uncovered:
        _raise(f"{len(uncovered)} sales days missing from {CALENDAR_FILE}, for example {uncovered[:5]}.")
    logger.info("Raw columns verified: %d sales days (d_1 to d_%d)", n_days, n_days)
    return n_days


def ingest_bronze(con: duckdb.DuckDBPyConnection, raw_dir: Path, bronze_dir: Path) -> dict[str, Path]:
    """Convert each raw CSV to parquet unchanged, adding _ingested_at and _source_file."""
    outputs = {}
    for name, file in (("sales", SALES_FILE), ("calendar", CALENDAR_FILE), ("prices", PRICES_FILE)):
        out = bronze_dir / f"{name}.parquet"
        con.execute(
            f"COPY (SELECT *, current_timestamp AS _ingested_at, '{file}' AS _source_file "
            f"FROM read_csv('{(raw_dir / file).as_posix()}', header = true)) "
            f"TO '{out.as_posix()}' (FORMAT parquet)"
        )
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
        logger.info("Bronze %s: %d rows", out.name, rows)
        outputs[name] = out
    return outputs


def find_complete_weeks(calendar_days: pd.DataFrame, n_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weeks with all 7 days inside d_1..d_n_days.

    Returns (weeks, dropped). weeks has week_id, week_start, t, where t is the
    position in sorted week_id order (week ids jump at year boundaries, so t is
    not week_id arithmetic). dropped has week_id and the number of days present.
    """
    day_number = calendar_days["d"].str.removeprefix("d_").astype(int)
    in_range = calendar_days[day_number.between(1, n_days)]
    counts = (
        in_range.groupby("wm_yr_wk")
        .agg(days=("d", "size"), week_start=("date", "min"))
        .reset_index()
        .rename(columns={"wm_yr_wk": "week_id"})
        .sort_values("week_id", ignore_index=True)
    )
    complete = counts["days"] == 7
    dropped = counts.loc[~complete, ["week_id", "days"]].reset_index(drop=True)

    first, last = counts.index[complete].min(), counts.index[complete].max()
    interior = dropped["week_id"].between(counts.loc[first, "week_id"], counts.loc[last, "week_id"])
    if interior.any():
        _raise("Incomplete weeks found between complete weeks.", dropped[interior])

    weeks = counts.loc[complete, ["week_id", "week_start"]].reset_index(drop=True)
    weeks["week_start"] = pd.to_datetime(weeks["week_start"])
    weeks["t"] = np.arange(len(weeks))
    return weeks, dropped


def item_volume(con: duckdb.DuckDBPyConnection, sales_path: Path, dept_id: str, day_columns: list[str]) -> pd.Series:
    """Total units per item over the given day columns, summed across stores, for one department."""
    columns = ", ".join(f'"{c}"' for c in day_columns)
    wide = con.execute(
        f"SELECT item_id, {columns} FROM read_parquet('{sales_path.as_posix()}') WHERE dept_id = ?",
        [dept_id],
    ).df()
    if wide.empty:
        _raise(f"No sales rows found for dept_id {dept_id}.")
    volume = wide[day_columns].sum(axis=1).groupby(wide["item_id"]).sum()
    volume.index.name = "item_id"
    return volume


def select_slice(volume: pd.Series, top_n: int, n_random: int, rng: np.random.Generator) -> pd.DataFrame:
    """Top top_n items by volume plus n_random items drawn from the rest.

    Ties in volume are broken by item_id so the selection is deterministic.
    Returns item_id, selection_reason (top_volume or random_extra), sorted by item_id.
    """
    ranked = volume.rename("volume").rename_axis("item_id").reset_index().sort_values(["volume", "item_id"], ascending=[False, True])
    top = ranked["item_id"].iloc[:top_n].tolist()
    remaining = sorted(ranked["item_id"].iloc[top_n:].tolist())
    if n_random > len(remaining):
        raise ValueError(f"Cannot draw {n_random} extra items from {len(remaining)} remaining.")
    extra = [remaining[i] for i in rng.choice(len(remaining), size=n_random, replace=False)]
    selection = pd.DataFrame({
        "item_id": top + extra,
        "selection_reason": ["top_volume"] * len(top) + ["random_extra"] * len(extra),
    })
    return selection.sort_values("item_id", ignore_index=True)


def build_silver(
    con: duckdb.DuckDBPyConnection,
    bronze: dict[str, Path],
    slice_items: list[str],
    dc_map: dict[str, str],
    out: Path,
    n_days: int,
) -> int:
    """Filter sales to the slice, unpivot to daily rows, add calendar, DC and price. Returns row count."""
    con.register("slice_items", pd.DataFrame({"item_id": slice_items}))
    con.register("dc_map", pd.DataFrame({"store_id": list(dc_map), "dc_id": list(dc_map.values())}))
    sales, calendar, prices = (bronze[k].as_posix() for k in ("sales", "calendar", "prices"))

    stores = con.execute(
        f"SELECT DISTINCT store_id FROM read_parquet('{sales}') "
        f"WHERE item_id IN (SELECT item_id FROM slice_items)"
    ).df()["store_id"].tolist()
    unmapped = sorted(set(stores) - set(dc_map))
    if unmapped:
        _raise(f"Stores missing from slice.dc_map in config.yaml: {unmapped}.")

    con.execute(f"""
        COPY (
            WITH sales AS (
                SELECT * EXCLUDE (_ingested_at, _source_file) FROM read_parquet('{sales}')
                WHERE item_id IN (SELECT item_id FROM slice_items)
            ),
            daily AS (
                UNPIVOT sales ON COLUMNS('^d_[0-9]+$') INTO NAME d VALUE units
            )
            SELECT s.item_id, s.store_id, m.dc_id, s.d, c.date, c.wm_yr_wk, s.units, p.sell_price
            FROM daily s
            JOIN read_parquet('{calendar}') c ON c.d = s.d
            JOIN dc_map m ON m.store_id = s.store_id
            LEFT JOIN read_parquet('{prices}') p
                ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
        ) TO '{out.as_posix()}' (FORMAT parquet)
    """)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    expected = len(slice_items) * len(stores) * n_days
    if rows != expected:
        _raise(f"Silver has {rows} rows, expected {expected} (items x stores x days).")
    logger.info("Silver %s: %d rows", out.name, rows)
    return rows


def weekly_calendar_features(calendar_days: pd.DataFrame, weeks: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    """One row per complete week: event counts, SNAP days per state, week_start, t, week_of_year, year."""
    cal = calendar_days[calendar_days["wm_yr_wk"].isin(weeks["week_id"])]
    unknown = set(cal["event_type_1"].dropna()) | set(cal["event_type_2"].dropna())
    unknown -= set(EVENT_TYPES)
    if unknown:
        _raise(f"Unknown event types in calendar: {sorted(unknown)}.")

    daily = pd.DataFrame({
        "week_id": cal["wm_yr_wk"],
        "event_days": (cal["event_name_1"].notna() | cal["event_name_2"].notna()).astype(int),
    })
    for event_type in EVENT_TYPES:
        is_type = (cal["event_type_1"] == event_type) | (cal["event_type_2"] == event_type)
        daily[f"event_{event_type.lower()}"] = is_type.astype(int)
    for state in states:
        daily[f"snap_{state}"] = cal[f"snap_{state}"].astype(int)

    feats = daily.groupby("week_id").sum().reset_index()
    feats = weeks.merge(feats, on="week_id", how="left", validate="one_to_one")
    feats["week_of_year"] = feats["week_id"] % 100
    feats["year"] = feats["week_start"].dt.year
    return feats


def aggregate_weekly(daily: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Daily store rows to weekly DC rows for complete weeks only.

    units: sum across the DC's stores and days. price: mean of the distinct
    store-week sell prices in the DC, ignoring missing (each store counts once,
    not once per day).
    """
    negative = daily[daily["units"] < 0]
    if len(negative):
        _raise(f"{len(negative)} daily rows have negative units.", negative)

    daily = daily[daily["wm_yr_wk"].isin(weeks["week_id"])].rename(columns={"wm_yr_wk": "week_id"})
    units = daily.groupby(GOLD_KEYS, as_index=False)["units"].sum()
    store_week = daily.drop_duplicates(["item_id", "store_id", "week_id"])
    price = store_week.groupby(GOLD_KEYS, as_index=False)["sell_price"].mean().rename(columns={"sell_price": "price"})
    return units.merge(price, on=GOLD_KEYS, how="left", validate="one_to_one")


def finalize_gold(weekly: pd.DataFrame, cal_feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add calendar features, launch flags and price fills; drop never-launched series.

    Returns (gold, stats). Prices are filled on launched rows only: forward-fill
    within item and DC, then back-fill, then the item's median launched price
    across DCs. Pre-launch rows keep their observed price, unfilled.
    """
    gold = weekly.merge(cal_feats, on="week_id", how="left", validate="many_to_one")
    no_calendar = gold[gold["t"].isna()]
    if len(no_calendar):
        _raise(f"{len(no_calendar)} weekly rows have no matching complete calendar week.", no_calendar)
    gold = gold.sort_values(["item_id", "dc_id", "t"], ignore_index=True)
    by_series = [gold["item_id"], gold["dc_id"]]

    gold["is_launched"] = (gold["units"] > 0).groupby(by_series).cummax()
    ever_launched = gold.groupby(SERIES_KEYS)["is_launched"].transform("any")
    never = gold.loc[~ever_launched, SERIES_KEYS].drop_duplicates()
    if len(never):
        logger.warning("Dropping %d never-launched series: %s", len(never), never.to_dict("records"))
    dropped_items = sorted(set(gold["item_id"]) - set(gold.loc[ever_launched, "item_id"]))
    gold = gold[ever_launched].reset_index(drop=True)
    by_series = [gold["item_id"], gold["dc_id"]]
    launched = gold["is_launched"]

    observed = gold["price"].where(launched)
    missing_before = int((launched & observed.isna()).sum())
    filled = observed.groupby(by_series).ffill().where(launched)
    after_ffill = int((launched & filled.isna()).sum())
    filled = filled.fillna(filled.groupby(by_series).bfill()).where(launched)
    after_bfill = int((launched & filled.isna()).sum())
    item_median = observed.groupby(gold["item_id"]).transform("median")
    filled = filled.fillna(item_median.where(launched))
    after_median = int((launched & filled.isna()).sum())
    gold["price"] = gold["price"].where(~launched, filled)

    gold["snap_days"] = 0
    for column in [c for c in gold.columns if c.startswith("snap_") and c != "snap_days"]:
        state = column.removeprefix("snap_")
        gold.loc[gold["dc_id"] == state, "snap_days"] = gold[column]

    gold["t"] = gold["t"].astype(int)
    gold["units"] = gold["units"].astype("int64")
    stats = {
        "never_launched_series": len(never),
        "never_launched": never.to_dict("records"),
        "dropped_items": dropped_items,
        "price_missing_after_launch": missing_before,
        "price_ffill": missing_before - after_ffill,
        "price_bfill": after_ffill - after_bfill,
        "price_item_median": after_bfill - after_median,
        "price_still_missing": after_median,
    }
    logger.info(
        "Price fills on launched rows: %d missing, %d forward-filled, %d back-filled, %d item median, %d still missing",
        missing_before, stats["price_ffill"], stats["price_bfill"], stats["price_item_median"], after_median,
    )
    return gold[GOLD_COLUMNS], stats


def check_gold(
    gold: pd.DataFrame, slice_items: list[str], expected_dcs: int, dropped_items: list[str] = ()
) -> None:
    """Raise DataQualityError if any gold table check fails.

    dropped_items are slice items whose every series never launched; they are
    the only items allowed to be missing from gold.
    """
    nulls = gold[gold[GOLD_KEYS + ["units"]].isna().any(axis=1)]
    if len(nulls):
        _raise(f"{len(nulls)} gold rows have nulls in {GOLD_KEYS + ['units']}.", nulls)

    negative = gold[gold["units"] < 0]
    if len(negative):
        _raise(f"{len(negative)} gold rows have negative units.", negative)

    duplicated = gold[gold.duplicated(GOLD_KEYS, keep=False)]
    if len(duplicated):
        _raise(f"{len(duplicated)} gold rows share an (item_id, dc_id, week_id) key.", duplicated)

    last_t = gold["t"].max()
    span = gold[gold["is_launched"]].groupby(SERIES_KEYS)["t"].agg(["min", "max", "size"])
    gaps = span[(span["max"] != last_t) | (span["size"] != span["max"] - span["min"] + 1)]
    if len(gaps):
        _raise(
            f"{len(gaps)} series have a gap between their launch week and the last week (t = {last_t}).",
            gaps.reset_index(),
        )

    bad_price = gold[gold["is_launched"] & ~(gold["price"] > 0)]
    if len(bad_price):
        _raise(f"{len(bad_price)} launched rows have a missing or non-positive price.", bad_price)

    n_dcs = gold["dc_id"].nunique()
    if n_dcs != expected_dcs:
        _raise(f"Gold has {n_dcs} DCs, expected {expected_dcs}.")

    expected_items = set(slice_items) - set(dropped_items)
    actual_items = set(gold["item_id"])
    if actual_items != expected_items:
        _raise(
            f"Gold has {len(actual_items)} items, expected {len(expected_items)} "
            f"(slice minus never-launched items). Unexpected: {sorted(actual_items - expected_items)[:5]}. "
            f"Missing: {sorted(expected_items - actual_items)[:5]}."
        )


def run(cfg: dict) -> None:
    """Stage `pipeline`: raw to bronze, silver and gold, with quality checks and a summary."""
    start = time.perf_counter()
    paths = cfg["data"]
    slice_cfg, forecast_cfg = cfg["slice"], cfg["forecast"]
    rng = np.random.default_rng(cfg["seed"])
    con = duckdb.connect()

    n_days = verify_raw_columns(paths["raw_dir"])
    bronze = ingest_bronze(con, paths["raw_dir"], paths["bronze_dir"])

    calendar_days = con.execute(f"SELECT * FROM read_parquet('{bronze['calendar'].as_posix()}')").df()
    calendar_days["date"] = pd.to_datetime(calendar_days["date"])
    weeks, dropped = find_complete_weeks(calendar_days, n_days)
    logger.info("Complete weeks: %d; dropped incomplete weeks: %s", len(weeks), dropped.to_dict("records"))

    t_last = int(weeks["t"].max())
    fold1_origin = t_last - forecast_cfg["horizon_weeks"] * forecast_cfg["backtest_folds"]
    window = weeks[weeks["t"].between(fold1_origin - slice_cfg["volume_window_weeks"] + 1, fold1_origin)]
    window_days = calendar_days.loc[calendar_days["wm_yr_wk"].isin(window["week_id"]), "d"].tolist()
    logger.info(
        "Slice volume window: t %d to %d (weeks %d to %d, %d days), ending at fold 1 origin",
        window["t"].min(), window["t"].max(), window["week_id"].min(), window["week_id"].max(), len(window_days),
    )
    volume = item_volume(con, bronze["sales"], slice_cfg["dept_id"], window_days)
    selection = select_slice(volume, slice_cfg["top_n_by_volume"], slice_cfg["random_extra_items"], rng)
    selection.to_csv(paths["silver_dir"] / "item_slice.csv", index=False)
    slice_items = selection["item_id"].tolist()
    logger.info("Item slice: %d items from %d in %s", len(slice_items), len(volume), slice_cfg["dept_id"])

    silver_path = paths["silver_dir"] / "sales_daily.parquet"
    silver_rows = build_silver(con, bronze, slice_items, slice_cfg["dc_map"], silver_path, n_days)
    daily = con.execute(
        f"SELECT item_id, store_id, dc_id, wm_yr_wk, units, sell_price FROM read_parquet('{silver_path.as_posix()}')"
    ).df()

    states = sorted(set(slice_cfg["dc_map"].values()))
    weekly = aggregate_weekly(daily, weeks)
    cal_feats = weekly_calendar_features(calendar_days, weeks, states)
    gold, stats = finalize_gold(weekly, cal_feats)
    check_gold(gold, slice_items, expected_dcs=len(states), dropped_items=stats["dropped_items"])
    gold.to_parquet(paths["gold_dir"] / "weekly.parquet", index=False)

    launched = gold["is_launched"]
    print("\nPipeline summary")
    print(f"  items:                 {gold['item_id'].nunique()} (slice {len(slice_items)})")
    print(f"  DCs:                   {gold['dc_id'].nunique()} {sorted(gold['dc_id'].unique())}")
    print(f"  weeks:                 {len(weeks)} (t 0 to {t_last}, weeks {weeks['week_id'].min()} to {weeks['week_id'].max()})")
    print(f"  silver daily rows:     {silver_rows}")
    print(f"  gold rows:             {len(gold)} ({int(launched.sum())} launched, {int((~launched).sum())} pre-launch)")
    print(f"  dropped weeks:         {dropped.to_dict('records')}")
    print(f"  never-launched series: {stats['never_launched_series']} {stats['never_launched']}")
    print(
        f"  price fills:           {stats['price_missing_after_launch']} missing after launch: "
        f"{stats['price_ffill']} forward-fill, {stats['price_bfill']} back-fill, "
        f"{stats['price_item_median']} item median, {stats['price_still_missing']} still missing"
    )
    print(f"  runtime:               {time.perf_counter() - start:.1f} s")
