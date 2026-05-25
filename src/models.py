# ML models for churn prediction.
# logistic regression, lightgbm, random forest, gradient boosting

import warnings
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _has_optuna() -> bool:
    """True if optuna is importable (checked lazily so it stays an optional dep)."""
    import importlib.util

    return importlib.util.find_spec("optuna") is not None


@dataclass
class ModelMetrics:
    """Holds evaluation results."""

    auc_roc: float
    avg_precision: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    precision_at_k: dict[int, float]
    lift_at_k: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc_roc": self.auc_roc,
            "avg_precision": self.avg_precision,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "prec@10": self.precision_at_k.get(10, 0),
            "prec@20": self.precision_at_k.get(20, 0),
            "lift@10": self.lift_at_k.get(10, 0),
            "lift@20": self.lift_at_k.get(20, 0),
        }


# Default constructor params per model. Merged with (and overridden by) any
# user/tuned params, so tuning can set e.g. n_estimators without colliding with
# a hardcoded default.
DEFAULT_PARAMS = {
    "logistic": {"max_iter": 1000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "class_weight": "balanced",
        "verbose": -1,
    },
}

_ESTIMATORS = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Search spaces for RandomizedSearchCV. Meaningful ranges for each model, not
# noise: capacity (estimators/leaves/depth), step size, and regularization.
PARAM_DISTRIBUTIONS = {
    "logistic": {"C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "random_forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 4, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
}


class RetentionModel:
    """Wrapper around sklearn/lgb models for churn prediction.

    Supports: logistic, lightgbm, random_forest, gradient_boosting
    """

    SUPPORTED_MODELS = list(DEFAULT_PARAMS.keys())

    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = RANDOM_STATE,
        track_mlflow: bool = False,
        calibrate: bool = False,
        calibration_method: str = "isotonic",
        **params,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"pick from {self.SUPPORTED_MODELS}")

        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            warnings.warn("no lightgbm, using gradient_boosting", stacklevel=2)
            model_type = "gradient_boosting"

        self.model_type = model_type
        self.random_state = random_state
        self.params = params
        self.track_mlflow = track_mlflow
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model = None
        self.calibrated_model = None
        self.scaler = None
        self.feature_names = None
        self._fitted = False
        self._mlflow_run = None

    def _create_model(self):
        """Build the underlying estimator using self.params."""
        return self._build_estimator(self.params)

    def _build_estimator(self, params: dict):
        """Build an estimator, merging model defaults with the given params."""
        kwargs = {
            "random_state": self.random_state,
            **DEFAULT_PARAMS[self.model_type],
            **params,
        }
        if self.model_type == "lightgbm":
            return lgb.LGBMClassifier(**kwargs)
        return _ESTIMATORS[self.model_type](**kwargs)

    def fit(self, X, y, scale_features: bool = True, feature_names: list[str] | None = None):
        """Train the model. X can be dataframe or array."""
        self.feature_names = feature_names or (list(X.columns) if hasattr(X, "columns") else None)

        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y

        # scale for logistic only
        if scale_features and self.model_type == "logistic":
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        if self.track_mlflow:
            try:
                if mlflow.active_run() is None:
                    self._mlflow_run = mlflow.start_run(run_name=f"{self.model_type}_training")
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_params(self.params)
                mlflow.log_param("scale_features", scale_features)
            except Exception:
                if self._mlflow_run:
                    mlflow.end_run()
                raise

        self.model = self._create_model()
        self.model.fit(X_arr, y_arr)
        self._fitted = True

        # Optional probability calibration. CalibratedClassifierCV uses internal
        # cross-validation (fit folds vs. calibration folds) so the calibration
        # map never sees the data it was fit on - leakage-safe by construction.
        # The base estimator stays available as self.model for feature importance
        # and SHAP, which calibration would otherwise obscure.
        self.calibrated_model = None
        if self.calibrate:
            calibration_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            self.calibrated_model = CalibratedClassifierCV(
                self._create_model(), method=self.calibration_method, cv=calibration_cv
            )
            self.calibrated_model.fit(X_arr, y_arr)

        # We leave the MLflow run open here so evaluate() can log metrics to the same run.
        # It's up to the user to mlflow.end_run() if they started it themselves.

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Get churn probabilities.

        Returns calibrated probabilities when the model was fit with
        calibrate=True; this is what downstream consumers (the ROI simulator)
        should use, since calibrated probabilities make the expected-value math
        meaningful.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")

        X_arr = X.values if hasattr(X, "values") else X
        if self.scaler:
            X_arr = self.scaler.transform(X_arr)
        estimator = self.calibrated_model if self.calibrated_model is not None else self.model
        return estimator.predict_proba(X_arr)[:, 1]

    def predict_proba_uncalibrated(self, X) -> np.ndarray:
        """Raw (pre-calibration) probabilities from the base estimator.

        Used to quantify the effect of calibration (e.g. Brier score before vs
        after); equals predict_proba when the model was not calibrated.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")
        X_arr = X.values if hasattr(X, "values") else X
        if self.scaler:
            X_arr = self.scaler.transform(X_arr)
        return self.model.predict_proba(X_arr)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions at given threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X, y, threshold: float = 0.5) -> ModelMetrics:
        """Get all the metrics."""
        y_true = y.values if hasattr(y, "values") else y
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_proba)
        avg_prec = average_precision_score(y_true, y_proba)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)

        # business metrics at top K%
        prec_at_k = {}
        lift_at_k = {}
        base_rate = float(np.mean(y_true))

        for k in [5, 10, 20, 30]:
            n_top = int(len(y_proba) * k / 100)
            top_idx = np.argsort(y_proba)[::-1][:n_top]
            precision_in_top_k = float(np.mean(y_true[top_idx]))
            prec_at_k[k] = precision_in_top_k
            lift_at_k[k] = precision_in_top_k / base_rate if base_rate > 0 else 0

        metrics = ModelMetrics(
            auc_roc=float(auc),
            avg_precision=float(avg_prec),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            confusion_matrix=confusion,
            precision_at_k=prec_at_k,
            lift_at_k=lift_at_k,
        )

        if self.track_mlflow:
            try:
                mlflow.log_metrics({
                    "auc_roc": float(auc),
                    "avg_precision": float(avg_prec),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "prec_at_10": prec_at_k.get(10, 0),
                    "lift_at_10": lift_at_k.get(10, 0),
                })
            finally:
                if self._mlflow_run:
                    mlflow.end_run()
                    self._mlflow_run = None

        return metrics

    def cross_validate(self, X, y, n_folds: int = 5) -> dict[str, float]:
        """5-fold CV.

        For the scaled (logistic) path, scaling is wrapped in a Pipeline so the
        StandardScaler is fit only on each fold's training partition - fitting it
        on the full X first would leak validation-fold statistics into training.
        """
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y

        estimator = self._create_model()
        if self.model_type == "logistic":
            estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(estimator, X_arr, y_arr, cv=cv, scoring="roc_auc")

        return {"mean_auc": scores.mean(), "std_auc": scores.std(), "all": scores.tolist()}

    def tune(
        self,
        X,
        y,
        n_iter: int = 25,
        n_folds: int = 5,
        scoring: str = "roc_auc",
        engine: str = "auto",
        n_trials: int = 60,
    ) -> dict[str, object]:
        """Hyperparameter search on the TRAINING data only.

        engine="optuna" (default when optuna is installed) runs a TPE search over
        a regularization-focused space - more sample-efficient than random search
        and better at curbing overfitting on this small, imbalanced cohort.
        engine="random" uses RandomizedSearchCV. Either way the search only ever
        sees the training split via CV, and for logistic the scaler lives inside
        the CV pipeline (re-fit per fold), so there is no leakage.

        Updates self.params with the best configuration so a subsequent fit() uses it.
        """
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y

        if engine == "auto":
            engine = "optuna" if _has_optuna() else "random"

        if engine == "optuna":
            result = self._tune_optuna(X_arr, y_arr, n_folds, scoring, n_trials)
        else:
            result = self._tune_random(X_arr, y_arr, n_iter, n_folds, scoring)

        self.params.update(result["best_params"])
        logger.info(
            f"Tuned {self.model_type} ({engine}): best {scoring}="
            f"{result['best_score']:.4f}, params={result['best_params']}"
        )
        return result

    def _tune_random(self, X_arr, y_arr, n_iter, n_folds, scoring) -> dict[str, object]:
        from math import prod

        distribution = PARAM_DISTRIBUTIONS[self.model_type]
        base = self._create_model()
        if self.model_type == "logistic":
            estimator = Pipeline([("scaler", StandardScaler()), ("model", base)])
            param_distributions = {f"model__{n}": v for n, v in distribution.items()}
        else:
            estimator = base
            param_distributions = distribution

        grid_size = prod(len(values) for values in param_distributions.values())
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator,
            param_distributions,
            n_iter=min(n_iter, grid_size),
            scoring=scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X_arr, y_arr)
        best = {n.replace("model__", ""): v for n, v in search.best_params_.items()}
        return {"best_params": best, "best_score": float(search.best_score_)}

    def _optuna_space(self, trial) -> dict[str, object]:
        """Search space per model. For tree models it leans on regularization
        (min_child_samples, feature fraction, L1/L2) to fight overfitting."""
        if self.model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 120),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        if self.model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 25),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        if self.model_type == "gradient_boosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}  # logistic

    def _tune_optuna(self, X_arr, y_arr, n_folds, scoring, n_trials) -> dict[str, object]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        def objective(trial):
            params = self._optuna_space(trial)
            estimator = self._build_estimator(params)
            if self.model_type == "logistic":
                estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            scores = cross_val_score(estimator, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return {"best_params": dict(study.best_params), "best_score": float(study.best_value)}

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance (if available)."""
        if not self._fitted:
            raise RuntimeError("model not fitted")

        if self.model_type == "logistic":
            importances = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            raise ValueError(f"no importance for {self.model_type}")

        names = self.feature_names or [f"f{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame({"feature": names, "importance": importances})
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["pct"] = importance_df["importance"] / importance_df["importance"].sum() * 100

        return importance_df.head(top_n).reset_index(drop=True)

    def get_shap_values(self, X, max_samples: int = 1000):
        """Compute SHAP for interpretability."""
        if not HAS_SHAP:
            raise ImportError("pip install shap")
        if not self._fitted:
            raise RuntimeError("fit first")

        X_arr = X.values if hasattr(X, "values") else X
        if self.scaler:
            X_arr = self.scaler.transform(X_arr)

        if len(X_arr) > max_samples:
            idx = np.random.choice(len(X_arr), max_samples, replace=False)
            X_arr = X_arr[idx]

        if self.model_type in ["lightgbm", "random_forest", "gradient_boosting"]:
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.LinearExplainer(self.model, X_arr)

        shap_vals = explainer.shap_values(X_arr)
        if isinstance(shap_vals, list):  # binary classification
            shap_vals = shap_vals[1]

        return shap_vals, explainer

    def get_calibration_curve(self, X, y, n_bins: int = 10):
        """Calibration data."""
        y_true = y.values if hasattr(y, "values") else y
        y_proba = self.predict_proba(X)
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
        return frac_pos, mean_pred

    def save(self, path: str):
        """Pickle the model with metadata (type, features, seed, params, timestamp)."""
        import datetime

        import joblib

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "type": self.model_type,
                "features": self.feature_names,
                "rs": self.random_state,
                "params": self.params,
                "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "RetentionModel":
        """Load pickled model."""
        import joblib

        saved = joblib.load(path)
        model = cls(model_type=saved["type"], random_state=saved["rs"], **saved["params"])
        model.model = saved["model"]
        model.scaler = saved["scaler"]
        model.feature_names = saved["features"]
        model._fitted = True
        return model


def compare_models(X_train, y_train, X_test, y_test, types=None) -> pd.DataFrame:
    """Quick comparison of different model types."""
    types = types or ["logistic", "lightgbm", "random_forest"]
    rows = []

    for model_type in types:
        try:
            model = RetentionModel(model_type=model_type)
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            rows.append({"model": model_type, **metrics.to_dict()})
        except Exception as e:
            warnings.warn(f"{model_type} failed: {e}", stacklevel=2)

    return pd.DataFrame(rows).sort_values("auc_roc", ascending=False)


def print_model_report(model: RetentionModel, X_test, y_test, threshold: float = 0.5) -> str:
    """Detailed evaluation report.
    """
    metrics = model.evaluate(X_test, y_test, threshold)
    y_pred = model.predict(X_test, threshold)

    return f"""
================================================================================
CHURN MODEL REPORT
================================================================================

Model: {model.model_type}

METRICS (threshold={threshold})
--------------------------------------------------------------------------------
{classification_report(y_test, y_pred, target_names=["Retained", "Churned"])}

AUC:          {metrics.auc_roc:.4f}
Avg Precision:{metrics.avg_precision:.4f}

BUSINESS METRICS
--------------------------------------------------------------------------------
Prec @ 10%:   {metrics.precision_at_k.get(10, 0):.1%}
Prec @ 20%:   {metrics.precision_at_k.get(20, 0):.1%}
Lift @ 10%:   {metrics.lift_at_k.get(10, 0):.2f}x
Lift @ 20%:   {metrics.lift_at_k.get(20, 0):.2f}x

CONFUSION MATRIX
--------------------------------------------------------------------------------
              Pred Retained  Pred Churned
Actual Retained  {metrics.confusion_matrix[0, 0]:>10,}    {metrics.confusion_matrix[0, 1]:>10,}
Actual Churned   {metrics.confusion_matrix[1, 0]:>10,}    {metrics.confusion_matrix[1, 1]:>10,}

================================================================================
"""
