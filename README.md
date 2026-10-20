# Retail Demand and Supply Planner

A demand and supply planning system on real Walmart sales data (M5): demand segmentation, statistical, intermittent and machine learning forecasting with rolling backtests, probabilistic safety stock, LP-based allocation of constrained supply across distribution centres, Monte Carlo scenario simulation and a planner dashboard. Supply parameters are simulated.

## Data download

The data is not in this repository. To get it:

1. Create a Kaggle account and accept the rules on the "M5 Forecasting - Accuracy" competition page. Without this the download fails with a 403 error.
2. Create an API key on your Kaggle account page and place `kaggle.json` in your home `.kaggle` folder (`C:\Users\<you>\.kaggle\kaggle.json` on Windows, `~/.kaggle/kaggle.json` on Linux or macOS, then `chmod 600 ~/.kaggle/kaggle.json`). Never put it inside this repository.
3. Download and unzip into `data/raw/`:

```
kaggle competitions download -c m5-forecasting-accuracy -p data/raw
cd data/raw
unzip m5-forecasting-accuracy.zip
```

The pipeline uses `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv`.
