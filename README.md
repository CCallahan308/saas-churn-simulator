# Churn ROI Simulator

An end-to-end churn-prediction and retention-ROI pipeline for event-log products, built for **correctness and reproducibility** and demonstrated honestly on the RetailRocket e-commerce dataset (2.7M events, 1.4M visitors).

**[Live demo](https://saas-churn-simulator-ccallahan308.streamlit.app/)** · leakage-safe labeling · Optuna-tuned + calibrated · reproducible via `make data && make train`

## TL;DR

- **What:** predict which active buyers won't come back, then turn the risk scores into a retention-budget decision (the ROI simulator).
- **How:** time-windowed leakage-safe labeling → behavioral features → LightGBM (Optuna-tuned, isotonic-calibrated) benchmarked against a LogisticRegression baseline.
- **Result (committed, reproducible):** the tuned LightGBM does **not** beat the simple baseline on the holdout (test ROC-AUC 0.83 vs 0.91; 5-fold CV 0.88 ± 0.06 — single holdouts swing wildly at this base rate). The durable deliverable is the leakage-safe pipeline and the **ROI simulator** that turns risk scores into a contact-budget decision (see [Limitations](#limitations)).

## Problem

Subscription and e-commerce products lose revenue to silent churn: users disengage before they cancel. The task is two-fold — (1) rank who is about to lapse from behavioral signals, and (2) decide *who is worth contacting*, since blanket discounting destroys margin.

## Data

- **Real:** [RetailRocket e-commerce](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.76M events (view / addtocart / transaction), 1.41M visitors, ~137 days. Download with `make data` (needs Kaggle API credentials). Raw data is **git-ignored**, not committed.
- **Synthetic:** the live Streamlit demo renders a **synthetic prediction distribution** for visualization; its metric cards are read from the committed `models/metrics.json`, not hand-entered.

## Method

- **Leakage-safe labeling.** A sliding `observation (60d) → gap (7d) → check (30d)` window: features come only from the observation window; churn = no purchase in the check window. The gap stops features from peeking at the label.
- **Split.** The dataset is too short (137d) for a clean temporal holdout, so one cohort is labeled and split **by visitor** (stratified 70/10/20) — no customer appears in two splits.
- **Features.** Six families from the observation window: recency, frequency, monetary proxy (no prices in the data), engagement (view→cart→purchase), trend, item diversity.
- **Model.** LightGBM tuned with **Optuna** (TPE, regularization-focused) and **isotonic-calibrated**, vs a LogisticRegression baseline. `class_weight="balanced"`; all tuning/calibration uses CV on the training split only.

## Results

From the committed run (`models/metrics.json`; reproduce with `make data && make train`):

| Metric (test set) | LightGBM (Optuna + calibrated) | LogisticRegression (baseline) |
|---|---|---|
| ROC-AUC | 0.83 | **0.91** |
| Avg precision | 0.998 | 0.999 |
| Lift @ 10% | 1.01× | 1.01× |

- **5-fold CV ROC-AUC (LightGBM):** 0.883 ± 0.058 — the stable estimate; single holdouts swing widely (val 0.60 vs test 0.83) because only ~1% of the cohort is retained.
- **Calibration** (isotonic) cut the test Brier score **0.065 → 0.009**.
- **Top predictors** (`models/feature_importance.csv`): days-since-last-view leads by a wide margin, then browsing-intensity features (view count, active days, sessions) — *viewing* behavior carries the signal, not purchase history.

Full data exploration: **[docs/EDA.md](docs/EDA.md)**.

## Limitations

- **~99% churn base rate** under a 30-day repurchase definition (only ~9% of buyers ever order twice), so lift over random is ~1.0 — there is little targeting headroom on this dataset. Earlier inflated "3× lift / 0.85 AUC" style claims were unreproducible and have been removed.
- **Tiny retained class** (~1%, ~12 customers in the test split) → high-variance holdout metrics; trust the CV number.
- **Monetary value is a proxy** (item/transaction counts; the dataset has no prices).
- This is a **portfolio/analysis project**, not a production service — there is no serving API or live scoring.

## How to run

```bash
make install   # pinned environment from requirements.lock (Python 3.11)
make data      # download RetailRocket via Kaggle CLI (needs Kaggle credentials)
make train     # reproduce models/metrics.json + feature_importance.csv
make test      # pytest
make lint      # ruff

streamlit run app.py    # local demo at http://localhost:8501 (synthetic distribution)
docker build -t saas-churn . && docker run -p 8501:8501 --rm saas-churn
```

## Repository layout

```
app.py                 Streamlit demo (metric cards read from models/metrics.json)
src/
  config.py            seed, default windows, paths
  data_loader.py       Kaggle download, cleaning, parquet cache
  churn_definition.py  leakage-safe time-window labeling
  features.py          six behavioral feature families
  models.py            RetentionModel: Optuna tuning, calibration, metrics, SHAP
  segmentation.py      RFM + k-means segmentation
  simulator.py         retention ROI simulator
  train.py             end-to-end reproducible pipeline (entry point)
models/                committed run artifacts: metrics.json, feature_importance.csv, lightgbm.joblib
docs/EDA.md            data exploration (narrative + figures)
tests/                 pytest (labeling + feature engineering)
requirements.txt       loose floors (CI + Dependabot)   ·   requirements.lock  pinned env
```

## Related work

Three retention projects, three different questions:

- [SignalForge](https://github.com/CCallahan308/signalforge) — *which model, and is the difference real?* (bootstrap CIs, paired tests, calibration)
- **This repo** — *what is a churn score worth in dollars when the base rate caps lift?* (retention-budget ROI simulator)
- [Ecommerce Retention & Growth](https://github.com/CCallahan308/ecommerce-retention-growth) — *which customers to win back, at what LTV?* (KKBox segmentation)

## License

MIT
