"""LightGBM point (Tweedie) and quantile forecasts with direct, horizon-grouped models."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import CATEGORICAL_FEATURES, build_features, feature_columns, history_feature_columns
from src.forecast_baselines import SERIES_KEYS, finalize

logger = logging.getLogger(__name__)

VALIDATION_WEEKS = 12


def _params(cfg: dict, objective: str, alpha: float | None = None) -> dict:
    lg = cfg["forecast"]["lgbm"]
    params = {
        "objective": objective,
        "learning_rate": lg["learning_rate"],
        "num_leaves": lg["num_leaves"],
        "min_data_in_leaf": lg["min_data_in_leaf"],
        "feature_fraction": lg["feature_fraction"],
        "bagging_fraction": lg["bagging_fraction"],
        "bagging_freq": lg["bagging_freq"],
        "seed": cfg["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = lg["tweedie_variance_power"]
    if objective == "quantile":
        params["alpha"] = alpha
    return params


def _train(train: pd.DataFrame, features: list[str], params: dict, cfg: dict, origin: int) -> tuple[lgb.Booster, int]:
    """Early-stop on the last 12 weeks before the origin, then refit on all rows with the best iteration."""
    lg = cfg["forecast"]["lgbm"]
    fit = train["t"] <= origin - VALIDATION_WEEKS
    dtrain = lgb.Dataset(train.loc[fit, features], train.loc[fit, "units"], categorical_feature=CATEGORICAL_FEATURES)
    dvalid = lgb.Dataset(train.loc[~fit, features], train.loc[~fit, "units"], reference=dtrain)
    early = lgb.train(
        params, dtrain, num_boost_round=lg["n_estimators"], valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(lg["early_stopping_rounds"], verbose=False)],
    )
    best = early.best_iteration or lg["n_estimators"]
    full = lgb.Dataset(train[features], train["units"], categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(params, full, num_boost_round=best), best


def fit_predict(
    history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict, quantiles: list[float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Train one model per horizon group (and per quantile) and predict the future rows.

    Returns (point, quantile_frame or None, importance). point has item_id, dc_id,
    t, h, prediction; quantile_frame has q50, q90, ... sorted to be monotonic per row.
    """
    origin = int(history["t"].max())
    points, quants, importance = [], [], []
    lower = 1
    for min_lag in cfg["forecast"]["horizon_groups"]:
        panel = build_features(history, future, min_lag)
        features = feature_columns(min_lag)
        train = panel[~panel["is_future"] & panel[history_feature_columns(min_lag)].notna().all(axis=1)]
        target = panel[panel["is_future"] & (panel["t"] - origin).between(lower, min_lag)]
        keys = target[SERIES_KEYS + ["t"]].astype({"item_id": str, "dc_id": str}).assign(h=target["t"] - origin)

        booster, best = _train(train, features, _params(cfg, cfg["forecast"]["lgbm"]["objective"]), cfg, origin)
        points.append(keys.assign(prediction=booster.predict(target[features])))
        gain = booster.feature_importance("gain")
        importance.append(pd.DataFrame({
            "origin": origin, "horizon_group": f"h{lower}_{min_lag}", "feature": features,
            "gain": gain, "gain_share": gain / gain.sum(), "best_iteration": best, "train_rows": len(train),
        }))
        logger.info("LightGBM h %d to %d: %d training rows, best iteration %d", lower, min_lag, len(train), best)

        if quantiles:
            q_values = np.column_stack([
                _train(train, features, _params(cfg, "quantile", q), cfg, origin)[0].predict(target[features])
                for q in quantiles
            ])
            q_values = np.clip(np.sort(q_values, axis=1), 0, None)
            quants.append(keys.assign(**{f"q{round(q * 100)}": q_values[:, i] for i, q in enumerate(quantiles)}))
        lower = min_lag + 1

    point = pd.concat(points, ignore_index=True)
    quantile_frame = pd.concat(quants, ignore_index=True) if quantiles else None
    return point, quantile_frame, pd.concat(importance, ignore_index=True)


def lightgbm(history: pd.DataFrame, future: pd.DataFrame, horizon: int, cfg: dict) -> pd.DataFrame:
    """Point forecasts from the Tweedie models. Feature importance is attached as out.attrs['importance']."""
    point, _, importance = fit_predict(history, future, horizon, cfg)
    values = future.merge(point, on=SERIES_KEYS + ["t"], how="left")["prediction"]
    out = finalize(future, pd.Series(values.to_numpy(), index=future.index), "lightgbm", history)
    out.attrs["importance"] = importance
    return out
