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
| intermittent | naive | 0.744 | -0.173 | 0.000 | lightgbm | 0.201 | 164 | lowest WAPE across all folds |
| lumpy | lightgbm | 0.913 | 0.010 | 71.078 | holt_winters | 14.018 | 24 | lowest WAPE across all folds |
| inactive | tsb | 0.999 | -0.998 | 0.092 | croston | -36.344 | 2 | inactive always uses tsb |

## Safety stock by ABC class

Command: `python run_all.py --stage inventory`

| abc | series | service_level | mean_ss_classic | mean_ss_quantile | mean_reorder_point | median_days_of_cover |
|---|---|---|---|---|---|---|
| A | 166 | 0.98 | 247.16 | 245.84 | 1,063.35 | 19.55 |
| B | 132 | 0.95 | 82.11 | 115.66 | 297.60 | 18.82 |
| C | 152 | 0.90 | 44.35 | 47.90 | 161.55 | 18.68 |

Total safety stock: classic 58,608 units, quantile 63,358 units.

## LP vs fair share

Command: `python run_all.py --stage inventory` (one constrained week) and `python run_all.py --stage simulate` (12 weeks, 200 runs)

One constrained week (opening stock on hand, empty pipeline, supplier capacity at 50%):

| abc | request | lp | fair_share |
|---|---|---|---|
| A | 91,453 | 23,464 | 21,676 |
| B | 19,770 | 3,303 | 4,371 |
| C | 10,842 | 1,668 | 2,388 |

Weighted shortfall: LP 317,784, fair share 321,704.

Simulated fill rate over 12 weeks (mean of 200 runs):

| group | fill_rate_lp | fill_rate_fair_share |
|---|---|---|
| A | 0.937 | 0.942 |
| All | 0.924 | 0.938 |
| B | 0.900 | 0.930 |
| C | 0.877 | 0.928 |
| model_lightgbm | 0.950 | 0.965 |
| model_naive | 0.651 | 0.668 |

## Scenario KPIs (all series, mean of 200 runs)

Command: `python run_all.py --stage simulate`

| scenario | fill_rate | fill_rate_p10 | fill_rate_p90 | cycle_service | stockout_weeks | avg_inventory_value | holding_cost | lost_sales_value | days_of_cover |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.924 | 0.919 | 0.927 | 0.885 | 1.382 | 140,206 | 8,089 | 201,771 | 10.652 |
| baseline_fair_share | 0.938 | 0.935 | 0.942 | 0.882 | 1.413 | 134,146 | 7,739 | 176,876 | 10.158 |
| demand_surge | 0.914 | 0.911 | 0.918 | 0.871 | 1.551 | 130,463 | 7,527 | 224,473 | 9.765 |
| supplier_disruption | 0.845 | 0.841 | 0.849 | 0.784 | 2.593 | 119,023 | 6,867 | 343,640 | 9.146 |
| lead_time_shock_policy_not_updated | 0.794 | 0.789 | 0.798 | 0.657 | 4.116 | 87,823 | 5,067 | 422,418 | 6.859 |
| lead_time_shock_policy_updated | 0.865 | 0.861 | 0.870 | 0.805 | 2.337 | 126,698 | 7,310 | 312,423 | 9.875 |
| flat_service_level | 0.925 | 0.920 | 0.928 | 0.890 | 1.325 | 141,889 | 8,186 | 200,838 | 10.906 |
| diagnostic_unlimited_supply | 0.994 | 0.993 | 0.995 | 0.982 | 0.216 | 165,701 | 9,560 | 15,121 | 12.180 |

Values are in US dollars at M5 shelf prices; unit cost is 60% of price and holding cost 25% of cost per year (simulated).

`diagnostic_unlimited_supply` is not one of the specified scenarios; it removes the supplier capacity limit to separate the policy's own performance from the capacity effect.

Fill rate by group (model_naive and model_lightgbm group series by their chosen forecasting model):

| scenario | A | All | B | C | model_lightgbm | model_naive |
|---|---|---|---|---|---|---|
| baseline | 0.937 | 0.924 | 0.900 | 0.877 | 0.950 | 0.651 |
| baseline_fair_share | 0.942 | 0.938 | 0.930 | 0.928 | 0.965 | 0.668 |
| demand_surge | 0.927 | 0.914 | 0.887 | 0.873 | 0.941 | 0.639 |
| supplier_disruption | 0.864 | 0.845 | 0.793 | 0.805 | 0.870 | 0.592 |
| lead_time_shock_policy_not_updated | 0.804 | 0.794 | 0.783 | 0.753 | 0.816 | 0.572 |
| lead_time_shock_policy_updated | 0.886 | 0.865 | 0.813 | 0.816 | 0.895 | 0.566 |
| flat_service_level | 0.937 | 0.925 | 0.902 | 0.883 | 0.951 | 0.651 |
| diagnostic_unlimited_supply | 0.995 | 0.994 | 0.992 | 0.987 | 0.997 | 0.966 |

Share of item-weeks in which DC requests exceeded the item's supplier capacity:

| scenario | share_item_weeks_supply_binding |
|---|---|
| baseline | 0.594 |
| baseline_fair_share | 0.597 |
| demand_surge | 0.631 |
| supplier_disruption | 0.771 |
| lead_time_shock_policy_not_updated | 0.339 |
| lead_time_shock_policy_updated | 0.789 |
| flat_service_level | 0.607 |
| diagnostic_unlimited_supply | 0.000 |

## Planner exceptions (stretch)

Command: `python run_all.py --stage exceptions`

| exception_type | count | value_at_risk |
|---|---|---|
| demand_anomaly | 49 | 2,207 |
| forecast_bias | 180 | 216,611 |
| overstock | 23 | 4,412 |
| stockout_risk | 58 | 27,322 |

Series with at least one exception: 226. Top 10 by value at risk (USD):

| rank | item_id | dc_id | abc | exception_type | value_at_risk | detail |
|---|---|---|---|---|---|---|
| 1 | FOODS_3_120 | CA | A | forecast_bias | 15,843 | bias +35% over the last fold (12289 forecast vs 9108 actual) |
| 2 | FOODS_3_444 | WI | A | stockout_risk | 12,201 | on hand falls to 0 (safety stock 324) by week 1; 1748 units projected lost in weeks 1 to 4 |
| 3 | FOODS_3_444 | TX | A | stockout_risk | 8,962 | on hand falls to 0 (safety stock 413) by week 1; 1284 units projected lost in weeks 1 to 4 |
| 4 | FOODS_3_444 | WI | A | forecast_bias | 8,341 | bias -100% over the last fold (0 forecast vs 1195 actual) |
| 5 | FOODS_3_444 | TX | A | forecast_bias | 7,971 | bias -100% over the last fold (0 forecast vs 1142 actual) |
| 6 | FOODS_3_120 | WI | A | forecast_bias | 7,834 | bias +38% over the last fold (5740 forecast vs 4167 actual) |
| 7 | FOODS_3_090 | CA | A | forecast_bias | 6,348 | bias -21% over the last fold (16296 forecast vs 20585 actual) |
| 8 | FOODS_3_362 | WI | A | forecast_bias | 6,215 | bias -100% over the last fold (0 forecast vs 2486 actual) |
| 9 | FOODS_3_362 | TX | A | forecast_bias | 5,352 | bias -100% over the last fold (0 forecast vs 2141 actual) |
| 10 | FOODS_3_499 | CA | A | forecast_bias | 4,705 | bias +44% over the last fold (6146 forecast vs 4264 actual) |
