# BUILD_PLAN.md: Retail Demand and Supply Planner

Detailed specification. Build phases in order. Every phase has: goal, files, exact specification, edge cases, tests, acceptance checks, and a commit message. Do not start a phase until the previous one passes its acceptance checks.

Repository name: `retail-supply-planner`

**Amendments agreed with the author (2026-10-20).** These override the original text and are already applied in the sections below. Record each one in `docs/DECISIONS.md` in the phase where it is built.
1. Python 3.12 (3.11 not installed; all dependencies have stable 3.12 wheels).
2. Item slice volume window: the 52 weeks ending at fold 1's origin (`T_last - 36`), not the last 52 weeks, to avoid selecting items for how they sold in the test weeks.
3. ABC boundary rule: a series is A if the cumulative revenue share before adding it is below 0.80, B if it is below 0.95, else C.
4. Segments per fold: each backtest fold computes ABC, XYZ and pattern from data up to its own origin, and uses them for features, model selection and evaluation in that fold. The final forecast uses segments computed up to `T_last`.
5. Final forecast calendar gap: `calendar.csv` ends at `d_1969`, so future dates it does not cover are filled by date (SNAP from each state's fixed day-of-month schedule, events from the same month and day of the previous year) and flagged `calendar_estimated = True`. Final forecast only.
6. `Errors and their fixes/` is gitignored (local notes only).
7. Multi-agent work is allowed only within a phase for independent tasks. Phases still run in order, with acceptance checks and pytest before each commit.
8. Nothing is pushed to GitHub until the whole project is done.

---

## Part A: Overview

### A1. One-line description (use in README)

> A demand and supply planning system on real Walmart sales data (M5): demand segmentation, statistical, intermittent and machine learning forecasting with rolling backtests, probabilistic safety stock, LP-based allocation of constrained supply across distribution centres, Monte Carlo scenario simulation and a planner dashboard. Supply parameters are simulated.

### A2. Tiers and time

| Tier | Phases | Estimate |
|---|---|---|
| Core | 0 to 9 | 10 to 12 hours |
| Stretch | 10 to 12 | 3 to 5 hours |

A finished Core tier is a complete project. Start Stretch only when Core is fully done, tested, documented and committed.

**If time is short, cut in this order:** Stretch phases; then quantile safety stock (keep classic); then Holt-Winters (LightGBM plus Croston and TSB still cover all patterns); then reduce simulation runs from 200 to 50.

### A3. Environment

- Python 3.12 (amendment 1). Works on Windows, macOS, Linux or WSL. Use `pathlib` everywhere.
- If LightGBM fails to install on native Windows, use WSL.
- Data is about 450 MB raw. Never load all of `sales_train_evaluation.csv` into a long format; filter first (see Phase 1).

### A4. requirements.txt

```
pandas>=2.1
numpy>=1.26
pyarrow>=14
duckdb>=1.0
statsmodels>=0.14
lightgbm>=4.1
scipy>=1.11
pyyaml>=6.0
streamlit>=1.32
plotly>=5.18
pytest>=8.0
kaggle>=1.6
```

### A5. Repository layout

```
retail-supply-planner/
  CLAUDE.md
  README.md
  RESULTS.md
  requirements.txt
  config.yaml
  run_all.py
  .gitignore
  docs/
    BUILD_PLAN.md
    DECISIONS.md
  data/                     # gitignored
    raw/
    bronze/
    silver/
    gold/
    outputs/
  app_data/                 # small committed results for the deployed app (Phase 9)
  src/
    __init__.py
    config.py
    errors.py
    pipeline.py
    supply_params.py
    segment.py
    metrics.py
    forecast_baselines.py
    forecast_stat.py
    features.py
    forecast_ml.py
    backtest.py
    inventory.py
    allocation.py
    simulate.py
    exceptions.py           # stretch
    elasticity.py           # stretch
    hierarchy.py            # stretch
  app/
    streamlit_app.py
  tests/
    test_metrics.py
    test_pipeline.py
    test_segment.py
    test_intermittent.py
    test_features.py
    test_inventory.py
    test_allocation.py
    test_simulate.py
```

### A6. .gitignore

```
data/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
kaggle.json
.env
.streamlit/secrets.toml
Errors and their fixes/
```

### A7. Stages for run_all.py

`run_all.py --stage <name>` with stages, in order:

`pipeline`, `supply`, `segment`, `backtest`, `forecast`, `inventory`, `simulate`, `app_data`, and `all` (runs every stage in order). Each stage reads its inputs from `data/` and writes outputs to `data/`, so stages can be rerun independently. Print a short summary at the end of each stage.

---

## Part B: Shared conventions

### B1. Keys and naming

- `item_id`: M5 item, for example `FOODS_3_090`
- `dc_id`: distribution centre = US state: `CA`, `TX`, `WI`
- `week_id`: Walmart week `wm_yr_wk` (integer, for example `11101`)
- `week_start`: date of the first day of that Walmart week (a Saturday)
- `t`: integer week index 0..N-1 in chronological order, used for modelling
- `units`: weekly units sold

All weekly tables are long format with one row per `(item_id, dc_id, week_id)`.

### B2. config.yaml (create exactly this, then adjust only through config)

```yaml
seed: 42

data:
  raw_dir: data/raw
  bronze_dir: data/bronze
  silver_dir: data/silver
  gold_dir: data/gold
  outputs_dir: data/outputs
  app_data_dir: app_data

slice:
  dept_id: FOODS_3
  top_n_by_volume: 100
  random_extra_items: 50
  volume_window_weeks: 52
  dc_map: {CA_1: CA, CA_2: CA, CA_3: CA, CA_4: CA, TX_1: TX, TX_2: TX, TX_3: TX, WI_1: WI, WI_2: WI, WI_3: WI}

forecast:
  horizon_weeks: 12
  backtest_folds: 3
  min_history_weeks_for_seasonal: 104
  croston_alpha: 0.1
  tsb_alpha: 0.1
  tsb_beta: 0.1
  quantiles: [0.5, 0.9, 0.95, 0.98]
  lgbm:
    objective: tweedie
    tweedie_variance_power: 1.2
    learning_rate: 0.05
    num_leaves: 63
    min_data_in_leaf: 50
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    n_estimators: 800
    early_stopping_rounds: 50

segment:
  abc_cutoffs: [0.80, 0.95]
  xyz_cutoffs: [0.5, 1.0]
  adi_cutoff: 1.32
  cv2_cutoff: 0.49
  window_weeks: 52

supply:
  lead_time_weeks_min: 1
  lead_time_weeks_max: 4
  supplier_capacity_factor: 1.10
  opening_stock_weeks: 3
  dc_capacity_weeks: 8
  unit_cost_share_of_price: 0.60
  annual_holding_cost_rate: 0.25

inventory:
  service_level_by_abc: {A: 0.98, B: 0.95, C: 0.90}
  min_error_points: 6

allocation:
  abc_weight: {A: 3.0, B: 2.0, C: 1.0}

simulation:
  runs: 200
  horizon_weeks: 12
  min_residuals_per_series: 10

scenarios:
  baseline: {}
  demand_surge: {abc_class: A, dc_id: CA, uplift: 0.25, start_week: 2, duration_weeks: 4}
  supplier_disruption: {capacity_factor: 0.50, start_week: 3, duration_weeks: 3}
  lead_time_shock: {extra_weeks: 2}
  flat_service_level: {service_level: 0.98}
```

### B3. src/config.py and src/errors.py

- `load_config(path="config.yaml") -> dict` returning the parsed YAML, with `Path` objects for every directory under `data`. Create missing directories.
- `errors.py`: `class DataQualityError(Exception)`.

### B4. Metrics definitions (src/metrics.py), used everywhere

All functions take numpy arrays or pandas Series of equal length, ignore positions where actual is NaN, and return float.

- `wape(actual, forecast) = sum(|a - f|) / sum(a)`. If `sum(a) == 0`, return `np.nan`.
- `bias(actual, forecast) = (sum(f) - sum(a)) / sum(a)`. Positive means over-forecasting. If `sum(a) == 0`, return `np.nan`.
- `mase(actual, forecast, insample, season=52)`: numerator `mean(|a - f|)`; denominator `mean(|y_t - y_{t-season}|)` over the in-sample history. If fewer than `season + 1` in-sample points or the denominator is 0, fall back to `season=1`. If still 0, return `np.nan`.
- `fva(wape_model, wape_benchmark) = wape_benchmark - wape_model`, in percentage points. Positive means the model adds value. Benchmark is the **naive** forecast.
- `fill_rate(sold, demand) = sum(sold) / sum(demand)`; if `sum(demand) == 0` return 1.0.

Aggregations across series are always **volume-weighted** (compute WAPE on summed absolute errors and summed actuals), never a simple average of per-series WAPE.

---

## Part C: Core tier

### Phase 0: Setup (30 min)

**Goal:** working skeleton.

**Tasks**
1. Create the layout in A5, `requirements.txt`, `.gitignore`, `config.yaml`, `src/config.py`, `src/errors.py`.
2. `run_all.py` with `argparse`, the stages in A7 wired to placeholder functions that log "not implemented".
3. `app/streamlit_app.py` showing a title and "Coming soon".
4. `docs/DECISIONS.md` with a heading.
5. `README.md` with the one-line description and a "Data download" section:
   - Create a Kaggle account, accept the rules on the "M5 Forecasting - Accuracy" competition page, place `kaggle.json` in the user's home `.kaggle` folder (never in the repo).
   - `kaggle competitions download -c m5-forecasting-accuracy -p data/raw`
   - Unzip into `data/raw/`.

**Acceptance**
- `pip install -r requirements.txt` succeeds.
- `pytest -q` runs.
- `python run_all.py --stage all` runs and logs each stage as not implemented.
- `streamlit run app/streamlit_app.py` opens.

**Commit:** `Set up project skeleton, config and stage runner`

---

### Phase 1: Medallion pipeline (1.5 h), stage `pipeline`

**Goal:** turn raw M5 files into a clean weekly table per item and DC.

**Input files (in data/raw)**
- `sales_train_evaluation.csv`: columns `id, item_id, dept_id, cat_id, store_id, state_id, d_1 ... d_1941`
- `calendar.csv`: columns `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`
- `sell_prices.csv`: columns `store_id, item_id, wm_yr_wk, sell_price`

Before coding, verify these column names against the real files and stop if they differ.

**Bronze** (`data/bronze/*.parquet`)
- Use DuckDB to convert each CSV to parquet unchanged, adding `_ingested_at` (timestamp) and `_source_file`.

**Item slice selection** (do this BEFORE any reshaping, to avoid creating tens of millions of rows)
1. With DuckDB or pandas, read only rows where `dept_id = FOODS_3` from bronze sales (823 items x 10 stores).
2. Total units per `item_id` over the `volume_window_weeks` complete weeks ending at fold 1's origin (`T_last - 12 x backtest_folds`, amendment 2), summed across stores. Derive the complete weeks, `T_last` and the matching `d_` columns from `calendar.csv`; do not hardcode them. Rationale: ranking on the last 52 weeks would pick items that sold well in the very weeks being tested (selection bias).
3. Take the top `top_n_by_volume` items. From the remaining items, sample `random_extra_items` using `rng` with the configured seed. Sort the final list.
4. Save the list to `data/silver/item_slice.csv` with columns `item_id, selection_reason` (`top_volume` or `random_extra`).

**Silver** (`data/silver/sales_daily.parquet`)
1. Filter bronze sales to the slice items.
2. Unpivot from wide to long with DuckDB `UNPIVOT` (columns `d_1..d_1941` into `d`, `units`). Expect about 150 x 10 x 1941, roughly 2.9 million rows.
3. Join calendar on `d` to add `date, wm_yr_wk`.
4. Map `store_id` to `dc_id` using `dc_map`.
5. Join `sell_prices` on `store_id, item_id, wm_yr_wk`.

**Gold** (`data/gold/weekly.parquet`) one row per `item_id, dc_id, week_id`:
- `units`: sum of daily units across the DC's stores.
- `price`: mean `sell_price` across the DC's stores for that week, ignoring missing.
- `event_days`: number of days in the week with a non-null `event_name_1` or `event_name_2`.
- `event_sporting, event_cultural, event_national, event_religious`: count of days with that `event_type_1` or `event_type_2`.
- `snap_days`: number of SNAP days in the week **for that DC's state** (use `snap_CA` for CA and so on).
- `week_start`: min date in the week. `week_of_year`: `week_id % 100`. `year`: from `week_start`.
- `t`: week index across all weeks, starting at 0.
- `is_launched`: boolean, see edge cases.

**Edge cases (handle and log counts)**
1. **Incomplete weeks:** keep only `week_id`s that have all 7 days present within `d_1..d_1941`. The last week is partial and must be dropped. Log which weeks were dropped.
2. **Not-yet-launched items:** for each `item_id, dc_id`, the launch week is the first week with `units > 0`. Rows before it get `is_launched = False` and are excluded from modelling, metrics and segmentation. Weeks after launch with zero sales are real zeros and are kept.
3. **Missing prices after launch:** forward-fill price within `item_id, dc_id`; if still missing (rare), back-fill; if still missing, use the item's median price across DCs. Log counts per fill step.
4. **Series that never launch** in a DC (all zeros): drop that `item_id, dc_id` entirely and log it.
5. Negative units must not exist in M5; if any appear, raise.

**Data quality checks** (raise `DataQualityError` with counts and examples)
- No nulls in `item_id, dc_id, week_id, units`.
- `units >= 0`.
- Every `(item_id, dc_id)` that is kept has every week from its launch week to the last week (no gaps).
- `price > 0` for every launched row.
- Number of DCs is exactly 3. Number of items equals the slice size (minus any logged never-launched series).

**Tests** (`tests/test_pipeline.py`) using tiny in-memory fixtures, not the real data:
- Incomplete week is dropped.
- Pre-launch rows are flagged `is_launched = False`.
- Price forward-fill works.
- A fixture with a negative unit raises `DataQualityError`.
- A fixture with a missing week after launch raises `DataQualityError`.

**Acceptance**
- `python run_all.py --stage pipeline` completes in under about 3 minutes.
- Prints: number of items, DCs, weeks, gold rows, dropped weeks, never-launched series, price fills.
- Expect roughly 150 items x 3 DCs x about 277 weeks.

**Commit:** `Build DuckDB medallion pipeline with weekly gold table and quality checks`

---

### Phase 2: Simulated supply parameters (30 min), stage `supply`

**Goal:** realistic, reproducible supply-side parameters. Document clearly that they are simulated.

**Output:** `data/gold/supply_params.parquet`, one row per `item_id, dc_id`, plus `data/gold/item_params.parquet` per `item_id`.

Compute using the last 52 launched weeks of history:
- `mean_weekly_demand` per `item_id, dc_id` (if fewer than 52 launched weeks, use what exists, minimum 1 unit).
- `lead_time_weeks` per **item** (same for all DCs): integer drawn uniformly from `[lead_time_weeks_min, lead_time_weeks_max]` with `rng`.
- `supplier_capacity` per **item** per week: `supplier_capacity_factor` x sum over DCs of `mean_weekly_demand`, rounded up.
- `unit_price` per item: median launched price across DCs. `unit_cost` = `unit_cost_share_of_price` x `unit_price`.
- `weekly_holding_cost` per unit = `unit_cost` x `annual_holding_cost_rate` / 52.
- `opening_stock` per `item_id, dc_id` = `opening_stock_weeks` x `mean_weekly_demand`, rounded.
- `dc_capacity` per DC (units, shared across items) = `dc_capacity_weeks` x sum over items of that DC's `mean_weekly_demand`.

**Acceptance**
- Running twice produces identical files.
- Summary prints lead time distribution, total supplier capacity, DC capacities.

**Commit:** `Add simulated supply parameters`

---

### Phase 3: Segmentation (1 h), stage `segment`

**Goal:** classify each `item_id, dc_id`. Classes drive model selection, service levels and allocation weights.

**Output:** `data/gold/segments.parquet` with `segment_set, origin, item_id, dc_id, revenue, abc, cv, xyz, adi, cv2, pattern`, where `segment_set` is `fold_1`, `fold_2`, `fold_3` or `final`.

**Segments per origin (amendment 4):** a function `segment(weekly, origin, cfg)` uses only the last `window_weeks` launched weeks with `t <= origin`. Compute one set per backtest fold at that fold's origin, and a `final` set at `T_last`. Each fold uses its own set for LightGBM features, model selection grouping and evaluation; the final forecast, inventory policy, allocation and simulation use the `final` set.

- **ABC** (per `item_id, dc_id`): revenue = sum(units x price). Sort descending. A series is A if the cumulative revenue share **before** adding it is below 0.80, B if below 0.95, else C (amendment 3). So the series that pushes the share past 0.80 is still A, and the top series is always A.
- **XYZ:** `cv = std(units) / mean(units)` over the window, including zero weeks. `cv <= 0.5` X, `<= 1.0` Y, else Z. If mean is 0, class `Z`.
- **Demand pattern (Syntetos-Boylan):**
  - `adi = number_of_weeks / number_of_nonzero_weeks`
  - `cv2 = (std / mean)^2` of **non-zero** demand values only
  - `adi < 1.32` and `cv2 < 0.49`: `smooth`
  - `adi < 1.32` and `cv2 >= 0.49`: `erratic`
  - `adi >= 1.32` and `cv2 < 0.49`: `intermittent`
  - `adi >= 1.32` and `cv2 >= 0.49`: `lumpy`
  - Zero non-zero weeks in window: `inactive`. One non-zero week: `cv2 = 0`.

**Tests** (`tests/test_segment.py`)
- Constant series `[5]*52`: `smooth`, `X`.
- `[0,0,0,10]*13`: adi 4.0, cv2 0, `intermittent`.
- Series with long zero runs and very different non-zero sizes, for example `([0,0,0,1]*6 + [0,0,0,40]*7)`: `lumpy`.
- ABC on a constructed revenue list gives the expected classes, including a series that crosses 0.80 being A.
- Fold segments never use weeks after the origin: changing any `units` or `price` value with `t > origin` leaves `segment(weekly, origin, cfg)` unchanged.

**Acceptance:** prints counts per ABC, XYZ, pattern, and the ABC by XYZ matrix.

**Commit:** `Add ABC, XYZ and Syntetos-Boylan segmentation`

---

### Phase 4: Forecasting models (2.5 h)

**Goal:** a common interface so backtesting and final forecasting call every model the same way.

**Interface** (every model):

```python
def forecast(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """history: launched rows up to origin (item_id, dc_id, t, units, features...).
    future: rows for t = origin+1 .. origin+horizon with known features (price, events, snap).
    Returns: item_id, dc_id, t, h, model, forecast  (forecast >= 0)."""
```

Clip all forecasts at 0. Never return NaN: if a model cannot produce a forecast for a series, fall back to naive and record the fallback in a `fallback` column (`True/False`). Log fallback counts per model.

**4a. Baselines** (`forecast_baselines.py`)
- `naive`: last observed value, repeated.
- `seasonal_naive`: value from 52 weeks before each target week; if not available, naive.
- `moving_average_4`: mean of last 4 weeks, repeated.

**4b. Statistical** (`forecast_stat.py`)
- `holt_winters`: `statsmodels ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add", seasonal_periods=52)` only when the series has at least `min_history_weeks_for_seasonal` launched weeks and is not `inactive`. Otherwise use simple exponential smoothing without trend or seasonality. Wrap fitting in try/except and fall back to seasonal naive on any error or non-finite output.
- `croston` (hand-written):
  - Initialise `z` = first non-zero demand, `p` = position of first non-zero (1-based interval).
  - On each later non-zero demand with interval `q` since the previous non-zero: `z = z + alpha*(d - z)`, `p = p + alpha*(q - p)`.
  - Forecast for every future week = `z / p`.
  - All-zero history: forecast 0.
- `tsb` (Teunter-Syntetos-Babai, hand-written):
  - Track probability of demand `b` and size `z`. Initialise `b` = share of non-zero weeks in history, `z` = mean non-zero demand.
  - Every week: `b = b + beta*(I - b)` where `I` is 1 if demand > 0 else 0. Only on non-zero weeks: `z = z + alpha*(d - z)`.
  - Forecast = `b * z`. This decays toward 0 when an item stops selling, which Croston cannot do.

**4c. Machine learning** (`features.py`, `forecast_ml.py`)

Strategy: **direct, horizon-grouped models** to avoid recursive error build-up and leakage. Three horizon groups: `h 1 to 4`, `h 5 to 8`, `h 9 to 12`. For a group with maximum horizon `H`, every lag feature uses lag `>= H`, so the features are fully known at the origin.

Features for target week `t` in group with minimum lag `L` (`L` = 4, 8 or 12):
- Lags of units: `L, L+1, L+2, L+4, 52` (lag 52 only if `>= L`).
- Rolling mean and std of units over windows 4, 8, 13 and 26, computed on the series shifted by `L`.
- Price features (known in advance, treated as the planned price): `price`, `price / item_dc_median_price`, `price change vs previous week`.
- Calendar: `event_days`, the four event type counts, `snap_days`, `week_of_year`, `year`.
- Static categorical: `item_id`, `dc_id`, `abc`, `pattern`.
- `h` (horizon within the group) as a numeric feature.

Implementation notes:
- Build the training matrix only from rows with `t <= origin` and all required lags available.
- Use LightGBM categorical support for `item_id`, `dc_id`, `abc`, `pattern` (cast to pandas `category` with fixed categories from the full item list so train and predict match).
- Point model: objective from config (`tweedie`, power 1.2).
- Quantile models: objective `quantile` with `alpha` in `quantiles` from config (0.5, 0.9, 0.95, 0.98), one model per quantile per horizon group (12 small models). Enforce monotonic quantiles after prediction by sorting quantile predictions per row.
- Early stopping: hold out the last 12 weeks before the origin as validation inside training. Record the best iteration.
- Save feature importance per group to `data/outputs/feature_importance.csv`.

**Tests**
- `tests/test_intermittent.py`:
  - Croston on `[0,0,3,0,0,3,0,0,3]` gives forecast 1.0.
  - Croston on all zeros gives 0.
  - TSB on a history that ends with 20 zero weeks gives a forecast lower than on the same history without those zeros.
- `tests/test_features.py`:
  - For origin T and group `L`, no feature value for target week `T+h` uses units from any week later than T (construct a series where units equal `t` and assert all lag features are `<= T`).

**Acceptance**
- Each model returns forecasts for every launched series for a 12-week horizon from one sample origin, with no NaN and no negatives.
- Fallback counts printed.

**Commit:** `Add baseline, Holt-Winters, Croston, TSB and LightGBM forecasters with leakage tests`

---

### Phase 5: Backtesting and model selection (1.5 h), stage `backtest`

**Goal:** honest out-of-sample accuracy and a model choice per demand pattern.

**Origins:** let `T_last` be the last week index. Folds are non-overlapping 12-week windows ending at `T_last`:
- Fold 3 origin `T_last - 12`, fold 2 origin `T_last - 24`, fold 1 origin `T_last - 36`.
- For each fold, every model trains only on `t <= origin` and forecasts `origin+1 .. origin+12`.

**Outputs**
- `data/outputs/backtest_forecasts.parquet`: `fold, item_id, dc_id, t, h, model, forecast, actual, fallback`.
- `data/outputs/backtest_scores.csv`: per `model` x `pattern`, and per `model` x `abc`, and overall: volume-weighted WAPE, bias, mean MASE, FVA vs naive, number of series.
- `data/outputs/model_selection.csv`: chosen model per pattern.

**Selection rule:** each forecast row is grouped by the series' pattern **in that fold's segment set** (amendment 4). For each `pattern`, choose the model with the lowest volume-weighted WAPE across all three folds. Tie-break by absolute bias. `inactive` series always use `tsb`. Record also the second-best model and the gap.

**Residuals for later phases:** save `actual - forecast` for the chosen model per series and horizon to `data/outputs/residuals.parquet` (used for safety stock and simulation).

**Acceptance**
- Leaderboard table printed: models x patterns WAPE.
- Selected model per pattern printed with its WAPE, bias and FVA.
- Total runtime under about 10 minutes on a laptop. If slower, reduce LightGBM `n_estimators` or parallelise statistical models with `joblib`, and note it in `DECISIONS.md`.

**Commit:** `Add rolling-origin backtest, metrics and model selection per demand pattern`

---

### Phase 6: Final forecast and inventory policy (1.5 h), stages `forecast` and `inventory`

**6a. Final forecast** (`forecast` stage)
- Origin `T_last`. For each series, run the selected model for its pattern for the next 12 weeks. Also produce LightGBM quantile forecasts for all series.
- Future features: price = last known price held flat (state this assumption: planned prices unknown beyond the data); events and SNAP from `calendar.csv`. It only runs to `d_1969`, about 4 weeks past `d_1941`, so for future dates it does not cover (amendment 5):
  - SNAP: infer each state's SNAP days of the month from the historical calendar, assert the schedule is identical in every complete month (raise `DataQualityError` if not), then apply it to the future dates. Verified on 2026-10-20: CA days 1 to 10; TX 1, 3, 5, 6, 7, 9, 11, 12, 13, 15; WI 2, 3, 5, 6, 8, 9, 11, 12, 14, 15; identical in all 65 complete months.
  - Events: copy `event_name_*` and `event_type_*` from the same month and day of the previous year. Correct for fixed-date holidays; moving holidays may shift by a few days, which is acceptable.
  - Flag every filled date and week with `calendar_estimated = True` and log the counts. Final forecast only; backtests always use the real calendar.
- Output `data/outputs/final_forecast.parquet`: `item_id, dc_id, t, h, model, forecast, q50, q90, q95, q98`.

**6b. Inventory policy** (`inventory` stage), per `item_id, dc_id`. Output `data/outputs/inventory_policy.parquet`.

- `service_level` from `service_level_by_abc`.
- `z = scipy.stats.norm.ppf(service_level)`.
- `L = lead_time_weeks` for the item.
- `sigma_e`: standard deviation of the chosen model's backtest residuals for that series at horizons 1 to 4. If fewer than `min_error_points` residuals, use the standard deviation of weekly demand over the last 52 launched weeks, and set `sigma_source = "demand_std"`.
- **Classic safety stock:** `ss_classic = z * sigma_e * sqrt(L)`.
- **Quantile safety stock:** pick the quantile column matching the service level (0.98, 0.95 or 0.90 maps to q98, q95 or q90). `ss_quantile = sqrt( sum over the first L weeks of (q_high_w - q50_w)^2 )`. Explain in `DECISIONS.md`: weekly quantiles cannot simply be added, so this assumes weekly errors are independent.
- `lead_time_demand` = sum of the selected model's forecast over the first `L` weeks.
- `reorder_point = lead_time_demand + safety_stock`, using `ss_classic` as the policy default (configurable).
- `order_up_to = reorder_point + forecast for week L+1` (one week review period).
- `days_of_cover = opening_stock / (mean forecast per week / 7)`.

**Important concept to document:** the service level here is the cycle service level (probability of no stockout in a replenishment cycle), which is different from fill rate (share of demand met). The simulation measures both.

**Tests** (`tests/test_inventory.py`)
- `z` for 0.95 is about 1.6449; with `sigma_e = 10`, `L = 4`: `ss_classic` is about 32.90.
- Higher service level gives higher safety stock.
- Quantile safety stock is 0 when all quantiles equal q50.

**Acceptance:** table summary by ABC class: mean safety stock (both methods), mean reorder point, mean days of cover.

**Commit:** `Add final forecasts and inventory policy with classic and quantile safety stock`

---

### Phase 7: Allocation of constrained supply (1 h)

**Goal:** each week, each DC requests replenishment; the supplier's capacity per item is limited; decide how much each DC gets.

**Request** per `item_id, dc_id` in week w: `request = max(0, order_up_to - inventory_position)` where `inventory_position = on_hand + in_transit`.

**LP allocation** (`allocation.py`, function `allocate_lp(requests, supply, dc_free_capacity, weights) -> allocation`), solved once per week across all items:
- Variables: `x[i, d] >= 0` shipment of item i to DC d.
- `x[i, d] <= request[i, d]`
- For each item: `sum_d x[i, d] <= supply[i]`
- For each DC: `sum_i x[i, d] <= dc_free_capacity[d]` where free capacity = `dc_capacity - projected on_hand - in_transit`, floored at 0.
- Objective: maximise `sum w[i, d] * x[i, d]`, where `w[i, d] = abc_weight[abc] * (1 + 1 / (1 + cover[i, d]))` and `cover` is weeks of cover at that DC. This favours A items, and within an item favours the DC closest to stocking out.
- Solve with `scipy.optimize.linprog(method="highs")` by minimising the negative objective. If the solver fails, log it and fall back to fair share for that week.
- Round down to integers, then hand out leftover units one at a time to the highest-weight unmet requests.

**Fair share** (`allocate_fair_share`): `x[i, d] = request[i, d] * min(1, supply[i] / sum_d request[i, d])`, then scale down per DC if DC capacity is exceeded, then the same integer rounding.

**Tests** (`tests/test_allocation.py`)
- Supply 100, requests `[80, 60, 0]`: total allocated `<= 100`, each `<= request`, DC 3 gets 0.
- Supply exceeds total requests: every request fully met by both methods.
- DC capacity binding: allocation to that DC never exceeds its free capacity.
- In a constructed constrained case where an A item and a C item compete for one DC's capacity, LP weighted shortfall is lower than fair share.

**Acceptance:** demo printed for one constrained week: LP vs fair share, weighted shortfall for each.

**Commit:** `Add LP and fair-share allocation of constrained supply across DCs`

---

### Phase 8: Monte Carlo simulation and scenarios (2 h), stage `simulate`

**Goal:** test the inventory policy and allocation under forecast uncertainty and disruptions.

**State arrays** shaped `(runs, items, dcs)` with numpy, vectorised across runs where possible. Allocation is solved per run per week (runs x 12 small LPs; acceptable). If runtime exceeds about 5 minutes, reduce runs and note it.

**Initial state** at week 0:
- `on_hand = opening_stock`
- Pipeline: for each of the next `L` weeks, an in-transit arrival equal to `mean_weekly_demand`, so the system does not start empty.

**Demand draws** for week w, horizon h: `demand = max(0, round(forecast + residual))`, where `residual` is sampled with `rng` from that series' saved backtest residuals at the matching horizon group. If fewer than `min_residuals_per_series`, sample from the pooled residuals of the series' demand pattern, scaled by the series' mean demand relative to the pool mean.

**Weekly loop**, order of events, for w in 1..12:
1. Receive arrivals due this week into `on_hand`.
2. Draw demand. `sold = min(demand, on_hand)`. `lost = demand - sold`. `on_hand -= sold`. Unmet demand is a lost sale (no backorders), a reasonable assumption for retail and FMCG where shoppers switch brands.
3. Compute requests from the order-up-to policy using inventory position.
4. Allocate supplier capacity (LP by default; fair share as a comparison run in baseline only).
5. Shipments arrive after `L` weeks.
6. Record per run and week: demand, sold, lost, on_hand, in_transit.
7. **Assert stock balance:** `on_hand_start + arrivals - sold == on_hand_end` and `on_hand_end >= 0`. Raise on violation.

**KPIs** per scenario (mean, 10th and 90th percentile across runs), overall and by ABC class:
- `fill_rate` (units)
- `cycle_service` = share of item-DC-weeks with no lost sales
- `stockout_weeks` per series
- `avg_inventory_units`, `avg_inventory_value` (units x unit_cost)
- `holding_cost` over the horizon
- `lost_sales_value` (lost units x unit_price)
- `days_of_cover`

**Scenarios** (from config):
1. `baseline`
2. `demand_surge`: demand x (1 + uplift) for A items in the chosen DC, for the configured weeks
3. `supplier_disruption`: supplier capacity x capacity_factor for the configured weeks
4. `lead_time_shock`: every lead time + extra_weeks for new orders (recompute reorder points with the longer lead time to show how a planner would respond, and report both "policy not updated" and "policy updated")
5. `flat_service_level`: every series at 0.98, recompute safety stock, compare inventory value and fill rate against the ABC-differentiated policy

Also run baseline with fair-share allocation for the LP vs fair-share comparison.

**Outputs:** `data/outputs/sim_kpis.csv`, `data/outputs/sim_weekly.parquet` (aggregated per scenario, week, abc: means and percentiles only).

**Tests** (`tests/test_simulate.py`)
- With zero residuals (deterministic demand), ample supply and ample capacity: fill rate is 1.0 and stock balance holds.
- Supplier capacity set to 0: fill rate falls over time and never goes negative inventory.
- Same seed gives identical KPIs.

**Acceptance:** KPI comparison table across all scenarios printed; runtime reported.

**Commit:** `Add Monte Carlo simulation with scenarios and KPIs`

---

### Phase 9: Dashboard, tests, docs and results (2 h), stage `app_data`

**9a. App data.** `app_data` stage writes small files to `app_data/` (target under 25 MB total): segment table, backtest score tables, model selection, final forecast for all series, a 52-week actual history only for the series shown in the app, inventory policy, allocation demo, simulation KPIs and weekly aggregates. Before committing `app_data/`, add a note to the README: it contains derived results and weekly aggregates from M5; check the competition's data rules and, if in doubt, deploy without it and use screenshots instead. Ask the author before committing it.

**9b. Streamlit app** (`app/streamlit_app.py`, reading only `app_data/`), with a sidebar page selector:
1. **Overview:** KPI cards for the selected scenario (fill rate, cycle service, inventory value, lost sales value), plus a one-paragraph description stating real demand and simulated supply.
2. **Segmentation:** ABC by XYZ matrix (counts), demand pattern counts, table filterable by class.
3. **Forecast accuracy:** leaderboard of WAPE, bias, MASE and FVA by model and pattern; the selected model per pattern highlighted.
4. **Item explorer:** pick item and DC; chart of last 52 weeks actual, backtest forecasts, and the 12-week final forecast with the q50 to q95 band.
5. **Inventory policy:** safety stock classic vs quantile by ABC, reorder points, days of cover.
6. **Allocation:** constrained-week demo, LP vs fair share, weighted shortfall.
7. **Scenarios:** pick a scenario, KPIs vs baseline with 10th to 90th percentile ranges, and a weekly fill rate chart.

Use `st.cache_data` for loading. The app must load in under about 10 seconds.

**9c. Deploy** to Streamlit Community Cloud from the GitHub repo (only after the author confirms 9a).

**9d. README.md** sections: what it is (real demand, simulated supply); a Mermaid diagram of the stages; data download; how to run (`run_all.py --stage all`, then the app); design decisions (link to `DECISIONS.md`); results summary (link to `RESULTS.md`); limitations:
- Supply parameters simulated; single echelon (supplier to DCs); lost sales assumption; prices held flat in the final horizon; US retail data rather than Indian FMCG; three DCs represent states, not real warehouses.

**9e. RESULTS.md**, filled only from the final run, each block preceded by the command used:
- Data slice summary (items, DCs, weeks).
- Segment counts.
- Backtest leaderboard and selected model per pattern with WAPE, bias, FVA.
- Safety stock comparison by ABC.
- LP vs fair-share comparison.
- Scenario KPI table.

**Acceptance**
- `pytest -q` passes.
- Fresh clone, download data, `python run_all.py --stage all`, `streamlit run app/streamlit_app.py` works end to end.
- `RESULTS.md` numbers match `data/outputs/` files.

**Commit:** `Add dashboard, README, results and full test suite`

---

## Part D: Stretch tier (only after Core is complete)

### Phase 10: Planner exceptions (1 h)

`exceptions.py` produces `data/outputs/exceptions.csv`, ranked by value at risk, using the final forecast and inventory policy (baseline, expected demand):
- **Stockout risk:** projected on-hand falls below safety stock within 4 weeks. Value at risk = projected lost units x unit_price.
- **Overstock:** days of cover above 60. Value at risk = excess units x unit_cost.
- **Forecast bias alert:** chosen model bias beyond plus or minus 20% over the last backtest fold.
- **Demand anomaly:** last actual week outside the backtest q5 to q95 range implied by residuals.

Add an "Exceptions" page to the app. Test: constructed cases trigger exactly the expected exception types.

**Commit:** `Add planner exception list`

### Phase 11: Price elasticity and a price scenario (1.5 h)

`elasticity.py`, per item, pooled across DCs:
- Use launched weeks with `units > 0`.
- Regress `log(units)` on `log(price)` plus DC indicator, month indicator, `event_days`, `snap_days` (statsmodels OLS).
- Require at least 3 distinct prices and price coefficient of variation above 2%; otherwise mark `insufficient_variation` and use the department mean.
- Shrink toward the department mean: `e_shrunk = (n / (n + 20)) * e_item + (20 / (n + 20)) * e_dept`, where n is the number of weeks.
- Clip to `[-4, 0]`; flag any raw positive elasticity.
- Scenario `price_cut`: 10% cut on chosen items; demand multiplier `(0.9) ** elasticity`; report revenue, fill rate and inventory changes in the simulation.

**Commit:** `Add price elasticity estimation and price-cut scenario`

### Phase 12: Hierarchical consistency (1 to 2 h)

`hierarchy.py`, on the last backtest fold:
- Levels: item-DC, item, DC, total.
- **Bottom-up:** sum item-DC forecasts to each level.
- **Top-down:** forecast the total with the selected approach, split down by each series' historical share over the last 52 weeks.
- Report WAPE at each level for both methods and explain the result in `DECISIONS.md`.

**Commit:** `Add hierarchical bottom-up vs top-down comparison`

---

## Part E: DECISIONS.md template

Append one entry per phase:

```
## Phase N: <name>
What was built: <two sentences>
Key decision: <the choice>
Why: <plain-language reason the author can say in an interview>
Numbers: <any result from this phase, copied from the run>
```

Decisions that must be explained somewhere: excluding pre-launch weeks; dropping partial weeks; segmentation on training window only; direct horizon-grouped LightGBM and why no leakage; Tweedie objective for zero-heavy sales; WAPE over MAPE; FVA vs naive; model selection per pattern instead of per item (per-item selection overfits); safety stock from forecast error; quantile safety stock aggregation assumption; cycle service vs fill rate; ABC-differentiated service levels; LP weights; lost sales assumption; prices held flat in the forecast horizon.

---

## Part F: Troubleshooting

- **Kaggle 403 on download:** the competition rules were not accepted on the competition page.
- **Kaggle 401:** `kaggle.json` missing or in the wrong folder; on Linux or WSL run `chmod 600 ~/.kaggle/kaggle.json`.
- **Memory errors in Phase 1:** reshaping was done before filtering to the slice. Filter first.
- **LightGBM install fails on Windows:** use WSL, or install a prebuilt wheel with `pip install lightgbm --prefer-binary`.
- **Holt-Winters warnings or failures:** expected on short or flat series; the fallback must handle them silently with logged counts.
- **Simulation too slow:** reduce `simulation.runs`, or vectorise the request and arrival steps across runs; keep the LP per run-week.
- **Streamlit Cloud memory limits:** keep `app_data/` small; never load raw data in the app.

---

## Part G: Final checklist

- [ ] All Core phases committed separately with real timestamps
- [ ] `pytest -q` passes
- [ ] `python run_all.py --stage all` reproduces outputs from a fresh clone (after data download)
- [ ] `README.md`, `RESULTS.md`, `DECISIONS.md` complete, every number from a real run
- [ ] No raw data, no credentials in git history
- [ ] No em dashes or en dashes anywhere (`grep` for them before the final commit)
- [ ] App deployed, or screenshots in the README if deployment was skipped