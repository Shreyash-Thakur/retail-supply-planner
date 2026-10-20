# Results

Every number below is generated from the files in `data/outputs/` and `data/gold/` by `python run_all.py --stage app_data` (last generated 2026-10-20). Each block names the stage command that produced its numbers. Demand is real M5 data; supply parameters are simulated (see README).

## Data slice

Command: `python run_all.py --stage pipeline`

| Measure | Value |
|---|---|
| Items | 150 (100 top volume, 50 random extra) from FOODS_3 |
| DCs | CA, TX, WI |
| Weeks | 277 complete weeks (week starting 2011-01-29 to week starting 2016-05-14) |
| Gold rows | 124,650 (109,368 launched, 15,282 pre-launch) |
| Units sold | 18,584,658 |

## Segments (final set, last 52 launched weeks up to the last week)

Command: `python run_all.py --stage segment`

| ABC \ XYZ | X | Y | Z | Total |
|---|---|---|---|---|
| A | 129 | 34 | 3 | 166 |
| B | 83 | 39 | 10 | 132 |
| C | 85 | 52 | 15 | 152 |
| Total | 297 | 125 | 28 | 450 |

| pattern | series |
|---|---|
| smooth | 378 |
| erratic | 7 |
| intermittent | 61 |
| lumpy | 4 |
| inactive | 0 |

## Backtest (3 folds of 12 weeks, volume-weighted, pattern as of each fold's origin)

Command: `python run_all.py --stage backtest`

WAPE by model and pattern:

| model | all | smooth | erratic | intermittent | lumpy | inactive |
|---|---|---|---|---|---|---|
| lightgbm | 0.280 | 0.242 | 0.547 | 0.746 | 0.913 | 0.969 |
| tsb | 0.333 | 0.285 | 0.795 | 0.849 | 1.527 | 0.999 |
| croston | 0.345 | 0.285 | 0.806 | 1.016 | 1.913 | 0.636 |
| moving_average_4 | 0.348 | 0.306 | 0.837 | 0.760 | 1.623 | 1.000 |
| holt_winters | 0.365 | 0.322 | 0.795 | 0.838 | 1.053 | 1.000 |
| naive | 0.369 | 0.329 | 0.863 | 0.744 | 1.624 | 1.000 |
| seasonal_naive | 0.382 | 0.331 | 0.761 | 0.976 | 1.538 | 1.000 |

Overall:

| model | wape | bias | mean_mase | fva_vs_naive_pp |
|---|---|---|---|---|
| lightgbm | 0.280 | -0.058 | 0.619 | 8.920 |
| tsb | 0.333 | 0.018 | 0.695 | 3.588 |
| croston | 0.345 | 0.049 | 0.713 | 2.430 |
| moving_average_4 | 0.348 | 0.015 | 0.728 | 2.083 |
| holt_winters | 0.365 | -0.004 | 0.734 | 0.458 |
| naive | 0.369 | -0.002 | 0.785 | 0.000 |
| seasonal_naive | 0.382 | -0.040 | 0.821 | -1.296 |

Selected model per pattern (gap_pp = second-best WAPE minus selected WAPE, in points):

| pattern | model | wape | bias | fva_vs_naive_pp | second_model | gap_pp | n_series_folds | rule |
|---|---|---|---|---|---|---|---|---|
| smooth | lightgbm | 0.242 | -0.053 | 8.736 | tsb | 4.339 | 1,129 | lowest WAPE across all folds |
| erratic | lightgbm | 0.547 | -0.105 | 31.578 | seasonal_naive | 21.421 | 31 | lowest WAPE across all folds |
| intermittent | lightgbm | 0.746 | -0.119 | -0.201 | naive | -0.201 | 164 | lowest absolute bias within 1 WAPE points of the best |
| lumpy | lightgbm | 0.913 | 0.010 | 71.078 | holt_winters | 14.018 | 24 | lowest WAPE across all folds |
| inactive | tsb | 0.999 | -0.998 | 0.092 | croston | -36.344 | 2 | inactive always uses tsb |

## Safety stock by ABC class

Command: `python run_all.py --stage inventory`

| abc | series | service_level | mean_ss_classic | mean_ss_quantile | mean_reorder_point | median_days_of_cover |
|---|---|---|---|---|---|---|
| A | 166 | 0.98 | 246.67 | 245.84 | 1,044.27 | 19.65 |
| B | 132 | 0.95 | 86.52 | 115.66 | 317.05 | 18.82 |
| C | 152 | 0.90 | 45.15 | 47.90 | 164.80 | 18.66 |

Total safety stock: classic 59,231 units, quantile 63,358 units.

## LP vs fair share

Command: `python run_all.py --stage inventory` (one constrained week) and `python run_all.py --stage simulate` (12 weeks, 200 runs)

One constrained week (opening stock on hand, empty pipeline, supplier capacity at 50%):

| abc | request | lp | fair_share |
|---|---|---|---|
| A | 87,357 | 23,464 | 21,440 |
| B | 21,091 | 3,742 | 4,954 |
| C | 10,991 | 1,733 | 2,545 |

Weighted shortfall: LP 300,071, fair share 304,215.

Simulated fill rate over 12 weeks (mean of 200 runs; `two_tier_lp` is amendment 10):

| group | fill_rate_lp | fill_rate_two_tier_lp | fill_rate_fair_share |
|---|---|---|---|
| A | 0.952 | 0.957 | 0.958 |
| All | 0.937 | 0.945 | 0.952 |
| B | 0.908 | 0.923 | 0.933 |
| C | 0.895 | 0.911 | 0.944 |
| model_lightgbm | 0.937 | 0.945 | 0.952 |

All series, by allocation method:

| kpi | lp | two_tier_lp | fair_share |
|---|---|---|---|
| fill_rate | 0.937 | 0.945 | 0.952 |
| cycle_service | 0.889 | 0.898 | 0.889 |
| stockout_weeks | 1.330 | 1.226 | 1.333 |
| avg_inventory_value | 140,441.005 | 137,478.469 | 134,914.558 |
| lost_sales_value | 124,908.861 | 112,162.325 | 101,536.391 |

## Scenario KPIs (all series, mean of 200 runs)

Command: `python run_all.py --stage simulate`

| scenario | fill_rate | fill_rate_p10 | fill_rate_p90 | cycle_service | stockout_weeks | avg_inventory_value | holding_cost | lost_sales_value | days_of_cover | revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.937 | 0.934 | 0.941 | 0.889 | 1.330 | 140,441 | 8,102 | 124,909 | 10.383 | 1,702,135 |
| baseline_fair_share | 0.952 | 0.948 | 0.955 | 0.889 | 1.333 | 134,915 | 7,784 | 101,536 | 9.918 | 1,725,508 |
| baseline_two_tier_lp | 0.945 | 0.941 | 0.949 | 0.898 | 1.226 | 137,478 | 7,931 | 112,162 | 10.126 | 1,714,882 |
| demand_surge | 0.928 | 0.924 | 0.932 | 0.875 | 1.498 | 130,738 | 7,543 | 146,566 | 9.503 | 1,731,662 |
| supplier_disruption | 0.860 | 0.856 | 0.864 | 0.788 | 2.542 | 119,316 | 6,884 | 262,810 | 8.887 | 1,564,234 |
| lead_time_shock_policy_not_updated | 0.804 | 0.799 | 0.808 | 0.652 | 4.172 | 87,116 | 5,026 | 346,260 | 6.510 | 1,480,784 |
| lead_time_shock_policy_updated | 0.883 | 0.878 | 0.887 | 0.812 | 2.261 | 128,071 | 7,389 | 228,034 | 9.716 | 1,599,011 |
| flat_service_level | 0.939 | 0.935 | 0.943 | 0.896 | 1.253 | 142,310 | 8,210 | 123,449 | 10.668 | 1,703,595 |
| diagnostic_unlimited_supply | 0.994 | 0.993 | 0.995 | 0.980 | 0.236 | 160,674 | 9,270 | 10,481 | 11.698 | 1,816,563 |
| price_cut_policy_not_updated | 0.923 | 0.919 | 0.926 | 0.883 | 1.405 | 133,262 | 7,688 | 148,279 | 9.559 | 1,671,495 |
| price_cut_policy_updated | 0.928 | 0.924 | 0.932 | 0.885 | 1.375 | 137,078 | 7,908 | 140,320 | 9.824 | 1,679,455 |

Values are in US dollars at M5 shelf prices; unit cost is 60% of price and holding cost 25% of cost per year (simulated).

`diagnostic_unlimited_supply` is not one of the specified scenarios; it removes the supplier capacity limit to separate the policy's own performance from the capacity effect.

Fill rate by group (model_naive and model_lightgbm group series by their chosen forecasting model):

| scenario | A | All | B | C | model_lightgbm |
|---|---|---|---|---|---|
| baseline | 0.952 | 0.937 | 0.908 | 0.895 | 0.937 |
| baseline_fair_share | 0.958 | 0.952 | 0.933 | 0.944 | 0.952 |
| baseline_two_tier_lp | 0.957 | 0.945 | 0.923 | 0.911 | 0.945 |
| demand_surge | 0.942 | 0.928 | 0.896 | 0.892 | 0.928 |
| supplier_disruption | 0.880 | 0.860 | 0.807 | 0.824 | 0.860 |
| lead_time_shock_policy_not_updated | 0.818 | 0.804 | 0.775 | 0.765 | 0.804 |
| lead_time_shock_policy_updated | 0.904 | 0.883 | 0.828 | 0.837 | 0.883 |
| flat_service_level | 0.952 | 0.939 | 0.912 | 0.902 | 0.939 |
| diagnostic_unlimited_supply | 0.996 | 0.994 | 0.987 | 0.990 | 0.994 |
| price_cut_policy_not_updated | 0.931 | 0.923 | 0.908 | 0.895 | 0.923 |
| price_cut_policy_updated | 0.938 | 0.928 | 0.908 | 0.895 | 0.928 |

Share of item-weeks in which DC requests exceeded the item's supplier capacity:

| scenario | share_item_weeks_supply_binding |
|---|---|
| baseline | 0.589 |
| baseline_fair_share | 0.592 |
| baseline_two_tier_lp | 0.591 |
| demand_surge | 0.628 |
| supplier_disruption | 0.778 |
| lead_time_shock_policy_not_updated | 0.327 |
| lead_time_shock_policy_updated | 0.801 |
| flat_service_level | 0.604 |
| diagnostic_unlimited_supply | 0.000 |
| price_cut_policy_not_updated | 0.600 |
| price_cut_policy_updated | 0.602 |

## Planner exceptions (stretch)

Command: `python run_all.py --stage exceptions`

| exception_type | count | value_at_risk |
|---|---|---|
| demand_anomaly | 46 | 2,161 |
| forecast_bias | 185 | 204,431 |
| stockout_risk | 57 | 9,152 |

Series with at least one exception: 230. Top 10 by value at risk (USD):

| rank | item_id | dc_id | abc | exception_type | value_at_risk | detail |
|---|---|---|---|---|---|---|
| 1 | FOODS_3_120 | CA | A | forecast_bias | 15,843 | bias +35% over the last fold (12289 forecast vs 9108 actual) |
| 2 | FOODS_3_120 | WI | A | forecast_bias | 7,834 | bias +38% over the last fold (5740 forecast vs 4167 actual) |
| 3 | FOODS_3_090 | CA | A | forecast_bias | 6,348 | bias -21% over the last fold (16296 forecast vs 20585 actual) |
| 4 | FOODS_3_444 | WI | A | forecast_bias | 5,726 | bias -69% over the last fold (375 forecast vs 1195 actual) |
| 5 | FOODS_3_318 | WI | A | forecast_bias | 5,373 | bias -61% over the last fold (2340 forecast vs 5971 actual) |
| 6 | FOODS_3_499 | CA | A | forecast_bias | 4,705 | bias +44% over the last fold (6146 forecast vs 4264 actual) |
| 7 | FOODS_3_120 | TX | A | forecast_bias | 4,668 | bias +22% over the last fold (5231 forecast vs 4294 actual) |
| 8 | FOODS_3_444 | TX | A | forecast_bias | 4,580 | bias -57% over the last fold (486 forecast vs 1142 actual) |
| 9 | FOODS_3_444 | CA | A | forecast_bias | 4,058 | bias -67% over the last fold (292 forecast vs 873 actual) |
| 10 | FOODS_3_077 | CA | A | forecast_bias | 3,928 | bias +230% over the last fold (1958 forecast vs 594 actual) |

## Price elasticity and price cut (stretch)

Command: `python run_all.py --stage elasticity` and `python run_all.py --stage simulate` (scenarios price_cut_policy_not_updated and price_cut_policy_updated above)

| Measure | Value |
|---|---|
| Items | 150 |
| Status ok / insufficient variation | 108 / 42 |
| Department mean (raw) | -0.111 |
| Raw estimate median (ok items) | -0.629 |
| Raw estimates with a positive sign (flagged) | 40 |
| Positive with a time trend added (sensitivity only) | 43 |
| Final elasticity median / mean | -0.111 / -0.829 |

## Hierarchical consistency (stretch, last backtest fold)

Command: `python run_all.py --stage hierarchy`

WAPE by level (forecasts summed to each level before scoring):

| level | bottom_up | top_down |
|---|---|---|
| item_dc | 0.261 | 0.316 |
| item | 0.213 | 0.274 |
| dc | 0.059 | 0.105 |
| total | 0.049 | 0.102 |

Model for the total series in top-down, chosen on the earlier folds only:

| model | wape | bias |
|---|---|---|
| seasonal_naive | 0.057 | -0.006 |
| moving_average_4 | 0.089 | 0.014 |
| holt_winters | 0.096 | -0.056 |
| croston | 0.099 | 0.062 |
| tsb | 0.099 | 0.062 |
| naive | 0.154 | 0.009 |
