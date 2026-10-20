# Design decisions

One entry per phase. Spec amendments agreed with the author are listed at the top of `docs/BUILD_PLAN.md`; each is explained here in the phase that builds it.

## Phase 0: Setup
What was built: the project skeleton (config, error type, stage runner with placeholder stages, placeholder dashboard, README with data download steps) and a Python 3.12 virtual environment with every dependency installed.
Key decision: Python 3.12 instead of the spec's 3.11 (amendment 1); config paths resolve against the project root; a `pytest.ini` sets `pythonpath = .`.
Why: 3.11 was not installed and every dependency ships stable 3.12 wheels, including LightGBM on native Windows, so no WSL is needed. Resolving paths against the project root means `run_all.py`, the tests and the app behave the same from any working directory. Without `pythonpath = .`, pytest cannot import `src` from the `tests` folder.
Numbers: installed versions: pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0, duckdb 1.5.6, statsmodels 0.15.0, lightgbm 4.7.0, scipy 1.18.1, pyyaml 6.0.3, streamlit 1.66.0, plotly 7.1.0, pytest 9.1.1. LightGBM native `lgb.train` with a Tweedie objective runs on native Windows. `pytest -q`: 2 passed.

## Phase 1: Medallion pipeline
What was built: a DuckDB pipeline that converts the three M5 CSVs to bronze parquet, selects a 150-item FOODS_3 slice, unpivots it to 2.9 million daily rows (silver), and aggregates to one weekly row per item and DC (gold) with calendar, SNAP and price features. Quality checks raise `DataQualityError` on nulls, negative units, duplicate keys, gaps after launch, missing prices, or wrong DC and item counts.
Key decisions:
- Slice before reshaping: rank FOODS_3 items on the wide table, then unpivot only the 150 chosen items, so the long table has 2.9 million rows instead of 59.2 million (30,490 series x 1,941 days) for the whole dataset.
- Slice volume window ends at fold 1's origin (amendment 2): the top 100 items by units over t 189 to 240 (weeks 11433 to 11532), plus 50 random extras from the other 723 items with the seeded generator. Ties are broken by item_id.
- Drop the partial last week: week 11617 has only 2 of its 7 days in the sales data.
- Pre-launch weeks are kept in gold but flagged `is_launched = False`; zero weeks after launch are real zeros and stay.
- DC weekly price is the mean of the distinct store-week prices, so each store counts once regardless of how many days it has. Pre-launch rows keep their observed price (M5 has none there) and are never filled.
- Price fill order for launched rows: forward-fill, then back-fill, then the item's median price across DCs, with counts logged for each step.
Why: reshaping all 30,490 series first is what causes memory errors. Ranking items on the last 52 weeks would pick items that happened to sell well in the very weeks we test on, which inflates backtest scores (selection bias); ranking on the 52 weeks before the first test window avoids that. A partial week looks like a sales drop and would distort models, metrics and segments. Weeks before an item is stocked are not zero demand: counting them as zeros would make items look intermittent and drag forecasts down. Averaging daily rows would weight stores by days; averaging store-weeks gives each store equal weight.
Numbers (from `python run_all.py --stage pipeline`): 150 items (100 top_volume, 50 random_extra), 3 DCs, 277 weeks (t 0 to 276, weeks 11101 to 11616), 2,911,500 silver rows, 124,650 gold rows (109,368 launched, 15,282 pre-launch), dropped week 11617 (2 days), 0 never-launched series, 0 launched rows missing a price (M5 prices every store-week in which an item sells), 8,100 launched zero-sales weeks kept. Gold units per DC match the raw CSV exactly (CA 7,617,313; TX 5,878,140; WI 5,089,205). Runtime 10.3 s. A second run gives identical slice and gold files.

## Phase 2: Simulated supply parameters
What was built: per item and DC, mean weekly demand, opening stock and DC capacity; per item, lead time, weekly supplier capacity, unit price, unit cost and weekly holding cost. All supply values are simulated from the real demand history, because M5 contains sales only.
Key decisions:
- Every parameter uses the last 52 launched weeks up to `T_last` (`supply.history_weeks`, added to config so the number is not hidden in code). These parameters feed only the forward simulation after `T_last`, never a backtest, so using the latest weeks is not leakage.
- Lead time is drawn once per item (uniform 1 to 4 weeks, seeded) and shared by all three DCs, because one supplier ships the item to every DC.
- Supplier capacity is 1.10 x the item's mean weekly demand summed over DCs: about 10% headroom on average, so capacity can bind when demand spikes or several DCs reorder in the same week, which is the situation the LP allocation is for (Phase 7 measures how often it does).
- Capacities are rounded up to whole units after rounding to 9 decimals, because floating point turns 1.1 x 50 into 55.00000000000001, which a plain ceiling would push to 56. A test covers this exact case.
- DC capacity is stored as a column on `supply_params.parquet` (one value per DC, repeated per item), since the spec names only two output files.
- Opening stock uses numpy rounding (round half to even); the difference from round half up is at most one unit on a tie.
Why: a planner needs supply constraints to make allocation and safety stock meaningful, and M5 has none. Deriving them from each series' own demand keeps them realistic in scale, and the fixed seed makes every run reproducible.
Numbers (from `python run_all.py --stage supply`): 450 series, 150 items, 0 series with fewer than 52 launched weeks, 0 floored at 1 unit. Lead times: 1 week 35 items, 2 weeks 36, 3 weeks 38, 4 weeks 41. Total supplier capacity 75,013 units per week against 68,129.6 units mean weekly demand. DC capacities: CA 221,380, TX 163,794, WI 159,864 units. Total opening stock 204,395 units. Unit price median 1.97, range 0.20 to 10.84. Two runs give byte-identical files.
