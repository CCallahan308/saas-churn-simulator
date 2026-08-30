"""End-to-end training pipeline for the SaaS churn simulator.

Wires the existing modules together into a single reproducible run:

    DataLoader  ->  CustomerStateLabeler (time-based train/val/test split)
                ->  FeatureEngineer (per-split, observation-window only)
                ->  RetentionModel: baseline LogisticRegression + tuned, calibrated main model
                ->  metrics on train / val / test  ->  saved model + metrics.json

The headline metrics quoted in the README are produced here; run it on the
RetailRocket data (``make data`` first) to reproduce them.

Usage:
    python -m src.train                       # full data, lightgbm, tuned + calibrated
    python -m src.train --sample 0.1          # 10% sample for a fast smoke run
    python -m src.train --no-tune             # skip hyperparameter search
    python -m src.train --model random_forest --mlflow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from src.churn_definition import CustomerStateLabeler, StateWindows
from src.config import (
    DEFAULT_CHK_DAYS,
    DEFAULT_GAP_DAYS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_OBS_DAYS,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models import ModelMetrics, RetentionModel

BASELINE_MODEL = "logistic"


def build_xy(
    engineer: FeatureEngineer,
    events: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix aligned row-for-row with the labels.

    Features are computed only from the labels' observation window (enforced
    inside ``FeatureEngineer``); merging on visitorid guarantees X and y stay
    aligned even if row order ever changes.
    """
    features = engineer.build_features(events, labels)
    merged = labels[["visitorid", "churned"]].merge(features, on="visitorid", how="left")
    y = merged["churned"].astype(int)
    X = merged.drop(columns=["visitorid", "churned"])
    return X, y


def metrics_row(name: str, m: ModelMetrics) -> dict[str, object]:
    """Flatten a ModelMetrics into a single table row."""
    return {
        "split": name,
        "auc_roc": round(m.auc_roc, 4),
        "avg_precision": round(m.avg_precision, 4),
        "precision": round(m.precision, 4),
        "recall": round(m.recall, 4),
        "f1": round(m.f1, 4),
        "prec@10": round(m.precision_at_k.get(10, 0.0), 4),
        "lift@10": round(m.lift_at_k.get(10, 0.0), 4),
    }


def run(
    model_type: str = DEFAULT_MODEL_TYPE,
    sample: float | None = None,
    windows: StateWindows | None = None,
    track_mlflow: bool = False,
    tune: bool = True,
    calibrate: bool = True,
    n_iter: int = 25,
    tuner: str = "auto",
    n_trials: int = 60,
    output_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Train and evaluate, returning the train/val/test metrics table."""
    windows = windows or StateWindows()
    logger.info(
        f"Config: model={model_type}, sample={sample}, windows={windows}, "
        f"tune={tune}, calibrate={calibrate}"
    )

    # 1. Load events ---------------------------------------------------------
    loader = DataLoader()
    try:
        events = loader.load_events(sample=sample)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nNo dataset found. Run `make data` to download RetailRocket first."
        ) from exc

    # 2. Label one cohort at a single snapshot, then build features ----------
    # The dataset spans only ~137 days, less than two non-overlapping 97-day
    # (obs+gap+check) windows, so a multi-snapshot chronological split would put
    # the SAME visitor in several splits (overlapping observation windows) and
    # leak train rows into test. Instead we label one cohort and split it by
    # visitor. Target leakage is still prevented by the obs/gap/check windowing;
    # the visitor-disjoint split prevents train/test contamination.
    labeler = CustomerStateLabeler(windows=windows)
    labels = labeler.label(events)
    logger.info(
        f"Cohort: {len(labels):,} active customers, churn rate {labels['churned'].mean():.1%}"
    )

    engineer = FeatureEngineer()
    X_all, y_all = build_xy(engineer, events, labels)
    feature_cols = list(X_all.columns)

    # 3. Stratified, visitor-disjoint split (70 / 10 / 20) -------------------
    # Each row is one visitor, so splitting rows guarantees disjoint visitors.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=2 / 3, stratify=y_holdout, random_state=RANDOM_STATE
    )
    for name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        logger.info(f"{name}: {len(split_y):,} customers, churn rate {split_y.mean():.1%}")

    # 4. Baseline for comparison --------------------------------------------
    baseline = RetentionModel(model_type=BASELINE_MODEL, random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train, feature_names=feature_cols)
    baseline_test = baseline.evaluate(X_test, y_test)
    logger.info(f"Baseline ({BASELINE_MODEL}) test AUC: {baseline_test.auc_roc:.4f}")

    # 5. Main model ----------------------------------------------------------
    model = RetentionModel(
        model_type=model_type,
        random_state=RANDOM_STATE,
        track_mlflow=track_mlflow,
        calibrate=calibrate,
    )

    tune_result = None
    if tune:
        tune_result = model.tune(
            X_train, y_train, n_iter=n_iter, engine=tuner, n_trials=n_trials
        )

    model.fit(X_train, y_train, feature_names=feature_cols)

    cv = model.cross_validate(X_train, y_train)
    logger.info(f"5-fold CV AUC on train: {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")

    # Calibration quality: Brier score before vs after (lower is better). This
    # is the evidence that calibration helped (or, honestly, if it did not).
    calibration_info = None
    if calibrate:
        brier_uncal = float(brier_score_loss(y_test, model.predict_proba_uncalibrated(X_test)))
        brier_cal = float(brier_score_loss(y_test, model.predict_proba(X_test)))
        calibration_info = {
            "method": model.calibration_method,
            "brier_uncalibrated_test": round(brier_uncal, 5),
            "brier_calibrated_test": round(brier_cal, 5),
            "improved": brier_cal <= brier_uncal,
        }
        logger.info(
            f"Brier (test): uncalibrated={brier_uncal:.5f} -> calibrated={brier_cal:.5f}"
        )

    rows = [
        metrics_row("train", model.evaluate(X_train, y_train)),
        metrics_row("val", model.evaluate(X_val, y_val)),
        metrics_row("test", model.evaluate(X_test, y_test)),
    ]
    rows.append(
        {**metrics_row("baseline(test)", baseline_test), "split": f"baseline:{BASELINE_MODEL} (test)"}
    )
    table = pd.DataFrame(rows)

    # 6. Persist model + metrics --------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_type}.joblib"
    model.save(str(model_path))
    importance_path = output_dir / "feature_importance.csv"
    model.get_feature_importance(top_n=len(feature_cols)).to_csv(importance_path, index=False)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "random_state": RANDOM_STATE,
                "sample": sample,
                "cohort_size": len(labels),
                "churn_rate": round(float(labels["churned"].mean()), 4),
                "windows": {"obs": windows.obs, "gap": windows.gap, "chk": windows.chk},
                "tuned": tune,
                "tuner": tuner if tune else None,
                "best_params": tune_result["best_params"] if tune_result else None,
                "best_cv_score": tune_result["best_score"] if tune_result else None,
                "cv_auc_mean": cv["mean_auc"],
                "cv_auc_std": cv["std_auc"],
                "calibration": calibration_info,
                "metrics": rows,
            },
            indent=2,
        )
    )
    logger.info(f"Saved model -> {model_path}")
    logger.info(f"Saved metrics -> {metrics_path}")

    logger.info("Results:\n" + table.to_string(index=False))
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the churn model end-to-end.")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_TYPE,
        choices=RetentionModel.SUPPORTED_MODELS,
        help="model type to train",
    )
    p.add_argument(
        "--sample",
        type=float,
        default=None,
        help="fraction of events to sample for a fast run (e.g. 0.1)",
    )
    p.add_argument("--obs", type=int, default=DEFAULT_OBS_DAYS, help="observation window (days)")
    p.add_argument("--gap", type=int, default=DEFAULT_GAP_DAYS, help="gap buffer (days)")
    p.add_argument("--chk", type=int, default=DEFAULT_CHK_DAYS, help="check window (days)")
    p.add_argument("--mlflow", action="store_true", help="log run to MLflow")
    p.add_argument("--no-tune", dest="tune", action="store_false", help="skip hyperparameter search")
    p.add_argument(
        "--no-calibrate", dest="calibrate", action="store_false", help="skip probability calibration"
    )
    p.add_argument(
        "--tuner",
        choices=["auto", "optuna", "random"],
        default="auto",
        help="hyperparameter search engine (auto picks optuna if installed)",
    )
    p.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    p.add_argument("--n-trials", type=int, default=60, help="Optuna trials")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        model_type=args.model,
        sample=args.sample,
        windows=StateWindows(obs=args.obs, gap=args.gap, chk=args.chk),
        track_mlflow=args.mlflow,
        tune=args.tune,
        calibrate=args.calibrate,
        n_iter=args.n_iter,
        tuner=args.tuner,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
