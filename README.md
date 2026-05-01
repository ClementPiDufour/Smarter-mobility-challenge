# Smarter Mobility Challenge

A hierarchical time series forecasting pipeline for electric vehicle charging station occupancy prediction. The system combines gradient-boosted tree models (CatBoost) with deep learning temporal convolutional networks (TCN) via the Darts library, operating at multiple spatial aggregation levels: station, area, and global.

![Results Overview](Tableau.png)

---

## Pipeline Overview

The forecasting pipeline executes the following steps in sequence:

1. **Baseline forecast** — A CatBoost benchmark model is trained and generates initial predictions, stored in `Station.pkl`.
2. **Data ingestion** — Raw station-level data is loaded and structured into hierarchical time series via `load_data` and `hierarchy`.
3. **Covariate construction** — Traffic flow data is aggregated from raw CSV files, resampled to 15-minute intervals, scaled, and serialised as `covariates.pkl` for reuse across runs.
4. **TCN model training** — A Temporal Convolutional Network (TCN) is fitted independently for each target variable at station level, using cyclic time encodings (hour, day of week, month) as past covariates.
5. **Area and global prediction** — Forecasts are generated at the area and global aggregation levels.


Output prediction files: `Station.pkl`, `Available.pkl`, `Charging.pkl`, `Passive.pkl`, `Other.pkl`.

---

## Model Architecture

### Baseline: CatBoost (`utils/catboost_model.py`)

A gradient-boosted tree model trained independently per target variable. Supports both regression (MAE loss) and classification modes. Key hyperparameters:

| Parameter | Default |
|---|---|
| `iterations` | 150 |
| `learning_rate` | 0.1 |
| `depth` | 6 |
| `loss_function` | MAE |

### Primary Model: TCN (`main.py`)

A Temporal Convolutional Network trained on hierarchical time series using the Darts framework. The model is fitted sequentially on two time windows (from `L2` then `L`) to account for temporal distribution shift.

| Parameter | Value |
|---|---|
| Input chunk length | 2000 |
| Output chunk length | 1921 |
| Kernel size | 10 |
| Number of layers | 5 |
| Number of filters | 10 |
| Dilation base | 6 |
| Loss function | L1 (MAE) |
| Optimiser | Adam (lr = 1e-3) |
| LR scheduler | StepLR (step=2, γ=0.4) |

Cyclic encodings of `hour`, `dayofweek`, and `month` are used as past covariates, scaled via a Darts `Scaler`.


---

## Covariates (`utils/covariates.py`)

Traffic flow covariates are constructed from hourly CSV files (2020–2021) containing a flow quantity column `q`. The `create_covariates` function:

1. Loads and concatenates all raw CSV files from `../0. Input Data/2020/` and `../0. Input Data/2021/`.
2. Groups by timestamp and aggregates flow by sum.
3. Resamples to 15-minute frequency with forward-fill for missing values.
4. Serialises the result to `covariates.pkl` on first run; subsequent calls load from cache.
5. Returns a scaled Darts `TimeSeries` aligned to the requested start date.

