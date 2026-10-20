# CLAUDE.md: Retail Demand and Supply Planner

Read this file first in every session. The full step-by-step specification is in `docs/BUILD_PLAN.md`. Follow it in order.

## What we are building

A demand and supply planning system on **real Walmart sales data (the M5 dataset)** with **simulated supply parameters**. It mirrors what enterprise planning platforms like o9 do:

1. Medallion data pipeline (DuckDB, parquet)
2. Segmentation (ABC, XYZ, Syntetos-Boylan demand patterns)
3. Forecasting: baselines, Holt-Winters, Croston, TSB, LightGBM (point and quantile)
4. Rolling-origin backtesting and model selection per demand pattern
5. Safety stock (classic and quantile-based) with service levels by ABC class
6. Weekly replenishment with LP-based allocation of limited supplier capacity across 3 distribution centres
7. Monte Carlo simulation of 12 weeks across 5 scenarios
8. Streamlit dashboard
9. Stretch only: planner exceptions, price elasticity, hierarchical consistency

The author will defend this project in a supply chain interview. Correctness and explainability matter more than breadth.

## Non-negotiable rules

1. **Never invent or estimate results.** Numbers in `README.md` and `RESULTS.md` are copied from real runs, with the command that produced them.
2. **Never commit raw or derived M5 data** except the small app results file described in Phase 9. `data/` is in `.gitignore`.
3. **Never commit `kaggle.json`** or any credential.
4. **Commit after every phase** with real timestamps. Never rewrite, squash-to-hide or backdate history.
5. **No em dashes or en dashes** anywhere: code, comments, docs, app text, commit messages. Use commas, colons, or "to".
6. **Private RNG only:** `rng = np.random.default_rng(seed)` passed explicitly. Never `np.random.*` global functions or `random`.
7. **No data leakage:** any feature or statistic used for a forecast made at origin T uses only data up to and including week T.
8. **All logic in `src/`.** The app and `run_all.py` only call `src/` functions. Paths via `pathlib`, settings via `config.yaml`.
9. **Fail loudly.** Data quality checks raise `DataQualityError` with a readable message. Never silently drop or fill data without logging counts.
10. **After every phase:** run the acceptance checks in `BUILD_PLAN.md`, run `pytest`, append an entry to `docs/DECISIONS.md`, then commit. Stop and report if any check fails.

## Working style

- Start each phase by restating its goal and listing the files you will create or change, then build.
- Prefer simple, readable code over clever code. Type hints and short docstrings on public functions.
- Vectorise with pandas or numpy; avoid Python loops over rows.
- Log progress with the standard `logging` module, not print, except for final summaries.
- When a choice is not specified in the plan, pick the simplest reasonable option, and record it in `docs/DECISIONS.md`.
- If something in the plan is wrong or impossible (for example a column name differs), stop, explain, propose a fix, and continue only after it is resolved.

## Commands

- Full pipeline: `python run_all.py --stage all`
- Single stage: `python run_all.py --stage forecast` (stages listed in `BUILD_PLAN.md`)
- Tests: `pytest -q`
- App: `streamlit run app/streamlit_app.py`