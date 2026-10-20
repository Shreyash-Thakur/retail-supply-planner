"""Final 12-week forecast from T_last: chosen model per pattern plus LightGBM quantiles.

calendar.csv ends at d_1969, about 4 weeks after the sales data, so future dates it
does not cover are filled by date (amendment 5): SNAP from each state's fixed
day-of-month schedule, events from the same month and day of the previous year.
Those dates and weeks are flagged calendar_estimated = True.
"""

import logging

import numpy as np
import pandas as pd

from src import forecast_ml as fm
from src.backtest import MODELS
from src.errors import DataQualityError
from src.features import SERIES_KEYS
from src.forecast_baselines import finalize
from src.pipeline import weekly_calendar_features

logger = logging.getLogger(__name__)

EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def snap_schedule(calendar: pd.DataFrame, states: list[str]) -> dict[str, list[int]]:
    """Each state's SNAP days of the month, asserted identical in every complete month."""
    dates = pd.to_datetime(calendar["date"])
    month = dates.dt.to_period("M")
    complete = calendar[month.map(month.value_counts()) == dates.dt.days_in_month]
    if complete.empty:
        raise DataQualityError("No complete calendar month to infer the SNAP schedule from.")
    complete_dates = pd.to_datetime(complete["date"])
    schedule = {}
    for state in states:
        on = complete[complete[f"snap_{state}"] == 1]
        per_month = pd.to_datetime(on["date"]).dt.day.groupby(pd.to_datetime(on["date"]).dt.to_period("M")).agg(lambda d: tuple(sorted(d)))
        per_month = per_month.reindex(complete_dates.dt.to_period("M").unique(), fill_value=())
        patterns = per_month.value_counts()
        if len(patterns) != 1:
            raise DataQualityError(f"SNAP schedule for {state} differs between complete months: {patterns.to_dict()}")
        schedule[state] = list(patterns.index[0])
        logger.info("SNAP %s: days %s in all %d complete months", state, schedule[state], len(per_month))
    return schedule


def future_calendar(calendar: pd.DataFrame, start: pd.Timestamp, n_days: int, states: list[str]) -> pd.DataFrame:
    """Daily calendar for start .. start + n_days - 1, using real rows where they exist."""
    real = calendar.assign(date=pd.to_datetime(calendar["date"])).set_index("date")
    dates = pd.date_range(start, periods=n_days)
    daily = real.reindex(dates)
    estimated = ~dates.isin(real.index)
    if estimated.any():
        schedule = snap_schedule(calendar, states)
        previous_year = real.reindex(dates[estimated] - pd.DateOffset(years=1))
        for col in EVENT_COLUMNS:
            daily.loc[estimated, col] = previous_year[col].to_numpy()
        for state in states:
            daily.loc[estimated, f"snap_{state}"] = dates[estimated].day.isin(schedule[state]).astype(int)
    daily = daily.rename_axis("date").reset_index()
    daily["calendar_estimated"] = estimated
    return daily


def future_weeks(calendar: pd.DataFrame, last_week: pd.Series, n_weeks: int, states: list[str]) -> pd.DataFrame:
    """Weekly calendar features for the n_weeks after last_week (a row with week_id, week_start, t)."""
    start = pd.Timestamp(last_week["week_start"]) + pd.Timedelta(days=7)
    daily = future_calendar(calendar, start, 7 * n_weeks, states)
    offset = (daily["date"] - start).dt.days // 7
    daily["wm_yr_wk"] = int(last_week["week_id"]) + 1 + offset
    known = daily["d"].notna()
    real_ids = calendar.set_index(pd.to_datetime(calendar["date"]))["wm_yr_wk"]
    if not (daily.loc[known, "wm_yr_wk"].to_numpy() == daily.loc[known, "date"].map(real_ids).to_numpy()).all():
        raise DataQualityError("Future week ids do not match calendar.csv; the horizon crosses a Walmart year boundary.")
    if (int(last_week["week_id"]) % 100) + n_weeks > 52:
        raise DataQualityError("Final horizon crosses a Walmart year boundary; extend future_weeks to handle it.")

    weeks = pd.DataFrame({
        "week_id": int(last_week["week_id"]) + 1 + np.arange(n_weeks),
        "week_start": start + pd.to_timedelta(7 * np.arange(n_weeks), unit="D"),
        "t": int(last_week["t"]) + 1 + np.arange(n_weeks),
    })
    feats = weekly_calendar_features(daily, weeks, states)
    flags = daily.groupby("wm_yr_wk")["calendar_estimated"].agg(["any", "sum"]).rename(
        columns={"any": "calendar_estimated", "sum": "estimated_days"})
    return feats.merge(flags, left_on="week_id", right_index=True)


def build_future(gold: pd.DataFrame, final_segments: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """One row per final-set series and future week; price held flat at the last launched price."""
    launched = gold[gold["is_launched"]].sort_values("t")
    last_price = launched.groupby(SERIES_KEYS)["price"].last().rename("price").reset_index()
    series = final_segments[SERIES_KEYS + ["abc", "pattern"]].merge(last_price, on=SERIES_KEYS, validate="one_to_one")
    future = series.merge(weeks, how="cross")
    future["snap_days"] = 0
    for state in future["dc_id"].unique():
        rows = future["dc_id"] == state
        future.loc[rows, "snap_days"] = future.loc[rows, f"snap_{state}"]
    return future.drop(columns=[c for c in future.columns if c.startswith("snap_") and c != "snap_days"])


def run(cfg: dict) -> None:
    """Stage `forecast`: write data/outputs/final_forecast.parquet."""
    paths = cfg["data"]
    horizon = cfg["forecast"]["horizon_weeks"]
    gold = pd.read_parquet(paths["gold_dir"] / "weekly.parquet")
    segments = pd.read_parquet(paths["gold_dir"] / "segments.parquet")
    final_segments = segments[segments["segment_set"] == "final"]
    selection = pd.read_csv(paths["outputs_dir"] / "model_selection.csv")
    calendar = pd.read_parquet(paths["bronze_dir"] / "calendar.parquet")
    states = sorted(set(cfg["slice"]["dc_map"].values()))

    t_last = int(gold["t"].max())
    last_week = gold.loc[gold["t"] == t_last, ["week_id", "week_start", "t"]].iloc[0]
    weeks = future_weeks(calendar, last_week, horizon, states)
    history = gold[gold["is_launched"]].merge(final_segments[SERIES_KEYS + ["abc", "pattern"]], on=SERIES_KEYS)
    future = build_future(gold, final_segments, weeks)
    logger.info("Future weeks %d to %d; estimated calendar days per week: %s",
                weeks["week_id"].min(), weeks["week_id"].max(), weeks.set_index("week_id")["estimated_days"].to_dict())

    needed = sorted(set(selection["model"]) - {"lightgbm"})
    forecasts = [MODELS[name](history, future, horizon, cfg) for name in needed]
    point, quantiles, importance = fm.fit_predict(history, future, horizon, cfg, quantiles=cfg["forecast"]["quantiles"])
    lgbm_values = future.merge(point, on=SERIES_KEYS + ["t"], how="left")["prediction"]
    forecasts.append(finalize(future, pd.Series(lgbm_values.to_numpy(), index=future.index), "lightgbm", history))
    all_models = pd.concat(forecasts, ignore_index=True)

    chosen = final_segments[SERIES_KEYS + ["pattern"]].assign(model=lambda d: d["pattern"].map(selection.set_index("pattern")["model"]))
    final = all_models.merge(chosen[SERIES_KEYS + ["model"]], on=SERIES_KEYS + ["model"], validate="many_to_one")
    q_cols = [f"q{round(q * 100)}" for q in cfg["forecast"]["quantiles"]]
    final = final.merge(quantiles[SERIES_KEYS + ["t"] + q_cols], on=SERIES_KEYS + ["t"], validate="one_to_one")
    final = final.merge(future[SERIES_KEYS + ["t", "week_id", "week_start", "abc", "pattern", "price", "calendar_estimated"]],
                        on=SERIES_KEYS + ["t"], validate="one_to_one")
    if final[["forecast"] + q_cols].isna().any().any() or len(final) != len(final_segments) * horizon:
        raise DataQualityError("Final forecast is incomplete or has missing values.")
    final.to_parquet(paths["outputs_dir"] / "final_forecast.parquet", index=False)
    all_models.to_parquet(paths["outputs_dir"] / "final_forecast_all_models.parquet", index=False)

    imp_path = paths["outputs_dir"] / "feature_importance.csv"
    previous = pd.read_csv(imp_path)
    previous = previous[previous["fold"] != "final"]
    pd.concat([previous, importance.assign(fold="final")], ignore_index=True).to_csv(imp_path, index=False)

    print("\nFinal forecast")
    print(f"  origin t {t_last} (week {int(last_week['week_id'])}); weeks {weeks['week_id'].min()} to {weeks['week_id'].max()}")
    print(f"  calendar-estimated weeks: {int(weeks['calendar_estimated'].sum())} of {horizon} "
          f"({int(weeks['estimated_days'].sum())} of {7 * horizon} days)")
    print(f"  series per chosen model: {final.drop_duplicates(SERIES_KEYS)['model'].value_counts().to_dict()}")
    print(f"  fallback rows: {int(final['fallback'].sum())}")
    print(f"  total forecast units over 12 weeks: {final['forecast'].sum():.0f} (q50 {final['q50'].sum():.0f}, q95 {final['q95'].sum():.0f})")
    weekly_total = final.groupby("week_id")[["forecast", "q50", "q90"]].sum().round(0)
    print("  weekly totals:\n    " + weekly_total.to_string().replace("\n", "\n    "))
    recent = gold[gold["t"] > t_last - 12].groupby("t")["units"].sum()
    print(f"  last 12 actual weeks: mean {recent.mean():.0f} units per week")
