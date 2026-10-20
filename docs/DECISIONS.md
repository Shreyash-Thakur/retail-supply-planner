# Design decisions

One entry per phase. Spec amendments agreed with the author are listed at the top of `docs/BUILD_PLAN.md`; each is explained here in the phase that builds it.

## Phase 0: Setup
What was built: the project skeleton (config, error type, stage runner with placeholder stages, placeholder dashboard, README with data download steps) and a Python 3.12 virtual environment with every dependency installed.
Key decision: Python 3.12 instead of the spec's 3.11 (amendment 1); config paths resolve against the project root; a `pytest.ini` sets `pythonpath = .`.
Why: 3.11 was not installed and every dependency ships stable 3.12 wheels, including LightGBM on native Windows, so no WSL is needed. Resolving paths against the project root means `run_all.py`, the tests and the app behave the same from any working directory. Without `pythonpath = .`, pytest cannot import `src` from the `tests` folder.
Numbers: installed versions: pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0, duckdb 1.5.6, statsmodels 0.15.0, lightgbm 4.7.0, scipy 1.18.1, pyyaml 6.0.3, streamlit 1.66.0, plotly 7.1.0, pytest 9.1.1. LightGBM native `lgb.train` with a Tweedie objective runs on native Windows. `pytest -q`: 2 passed.
