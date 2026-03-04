# ML models for churn prediction.
# logistic regression, lightgbm, random forest, gradient boosting

import warnings
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

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


# model configs in dict form instead of if/else chain
MODEL_CONFIGS = {
    "logistic": lambda rs, p: LogisticRegression(
        random_state=rs, max_iter=1000, class_weight="balanced", **p
    ),
    "random_forest": lambda rs, p: RandomForestClassifier(
        random_state=rs, n_estimators=100, max_depth=10, class_weight="balanced", **p
    ),
    "gradient_boosting": lambda rs, p: GradientBoostingClassifier(
        random_state=rs, n_estimators=100, learning_rate=0.1, max_depth=6, **p
    ),
}


class RetentionModel:
    """Wrapper around sklearn/lgb models for churn prediction.

    Supports: logistic, lightgbm, random_forest, gradient_boosting
    """

    SUPPORTED_MODELS = list(MODEL_CONFIGS.keys()) + ["lightgbm"]

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42, track_mlflow: bool = False, **params):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"pick from {self.SUPPORTED_MODELS}")

        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            warnings.warn("no lightgbm, using gradient_boosting")
            model_type = "gradient_boosting"

        self.model_type = model_type
        self.random_state = random_state
        self.params = params
        self.track_mlflow = track_mlflow
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._fitted = False
        self._mlflow_run = None

    def _create_model(self):
        """Make the actual model object."""
        if self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                class_weight="balanced",
                verbose=-1,
                **self.params,
            )
        # use dict lookup for the rest
        return MODEL_CONFIGS[self.model_type](self.random_state, self.params)

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
            if mlflow.active_run() is None:
                self._mlflow_run = mlflow.start_run(run_name=f"{self.model_type}_training")
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_params(self.params)
            mlflow.log_param("scale_features", scale_features)

        self.model = self._create_model()
        self.model.fit(X_arr, y_arr)
        self._fitted = True

        # We leave the MLflow run open here so evaluate() can log metrics to the same run.
        # It's up to the user to mlflow.end_run() if they started it themselves.

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Get churn probabilities."""
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
        ap = average_precision_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # business metrics at top K%
        prec_at_k = {}
        lift_at_k = {}
        base_rate = float(np.mean(y_true))

        for k in [5, 10, 20, 30]:
            n_top = int(len(y_proba) * k / 100)
            top_idx = np.argsort(y_proba)[::-1][:n_top]
            pk = float(np.mean(y_true[top_idx]))
            prec_at_k[k] = pk
            lift_at_k[k] = pk / base_rate if base_rate > 0 else 0

        metrics = ModelMetrics(
            auc_roc=float(auc),
            avg_precision=float(ap),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            confusion_matrix=cm,
            precision_at_k=prec_at_k,
            lift_at_k=lift_at_k,
        )

        if self.track_mlflow:
            mlflow.log_metrics({
                "auc_roc": float(auc),
                "avg_precision": float(ap),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "prec_at_10": prec_at_k.get(10, 0),
                "lift_at_10": lift_at_k.get(10, 0),
            })
            if self._mlflow_run:
                mlflow.end_run()

        return metrics

    def cross_validate(self, X, y, n_folds: int = 5) -> dict[str, float]:
        """5-fold CV."""
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y

        if self.scaler:
            X_arr = self.scaler.fit_transform(X_arr)

        m = self._create_model()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(m, X_arr, y_arr, cv=cv, scoring="roc_auc")

        return {"mean_auc": scores.mean(), "std_auc": scores.std(), "all": scores.tolist()}

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance (if available)."""
        if not self._fitted:
            raise RuntimeError("model not fitted")

        if self.model_type == "logistic":
            imp = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        else:
            raise ValueError(f"no importance for {self.model_type}")

        names = self.feature_names or [f"f{i}" for i in range(len(imp))]

        df = pd.DataFrame({"feature": names, "importance": imp})
        df = df.sort_values("importance", ascending=False)
        df["pct"] = df["importance"] / df["importance"].sum() * 100

        return df.head(top_n).reset_index(drop=True)

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

        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list):  # binary classification
            sv = sv[1]

        return sv, explainer

    def get_calibration_curve(self, X, y, n_bins: int = 10):
        """Calibration data."""
        y_true = y.values if hasattr(y, "values") else y
        y_proba = self.predict_proba(X)
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
        return frac_pos, mean_pred

    def save(self, path: str):
        """Pickle the model."""
        import joblib

        # TODO: add version info
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "type": self.model_type,
                "features": self.feature_names,
                "rs": self.random_state,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "RetentionModel":
        """Load pickled model."""
        import joblib

        d = joblib.load(path)
        inst = cls(model_type=d["type"], random_state=d["rs"], **d["params"])
        inst.model = d["model"]
        inst.scaler = d["scaler"]
        inst.feature_names = d["features"]
        inst._fitted = True
        return inst


def compare_models(X_train, y_train, X_test, y_test, types=None) -> pd.DataFrame:
    """Quick comparison of different model types."""
    types = types or ["logistic", "lightgbm", "random_forest"]
    out = []

    for t in types:
        try:
            m = RetentionModel(model_type=t)
            m.fit(X_train, y_train)
            met = m.evaluate(X_test, y_test)
            out.append({"model": t, **met.to_dict()})
        except Exception as e:
            print(f"{t} failed: {e}")

    return pd.DataFrame(out).sort_values("auc_roc", ascending=False)


def print_model_report(model: RetentionModel, X_test, y_test, threshold: float = 0.5) -> str:
    """Detailed evaluation report.
    """
    met = model.evaluate(X_test, y_test, threshold)
    y_pred = model.predict(X_test, threshold)

    return f"""
================================================================================
CHURN MODEL REPORT
================================================================================

Model: {model.model_type}

METRICS (threshold={threshold})
--------------------------------------------------------------------------------
{classification_report(y_test, y_pred, target_names=["Retained", "Churned"])}

AUC:          {met.auc_roc:.4f}
Avg Precision:{met.avg_precision:.4f}

BUSINESS METRICS
--------------------------------------------------------------------------------
Prec @ 10%:   {met.precision_at_k.get(10, 0):.1%}
Prec @ 20%:   {met.precision_at_k.get(20, 0):.1%}
Lift @ 10%:   {met.lift_at_k.get(10, 0):.2f}x
Lift @ 20%:   {met.lift_at_k.get(20, 0):.2f}x

CONFUSION MATRIX
--------------------------------------------------------------------------------
              Pred Retained  Pred Churned
Actual Retained  {met.confusion_matrix[0, 0]:>10,}    {met.confusion_matrix[0, 1]:>10,}
Actual Churned   {met.confusion_matrix[1, 0]:>10,}    {met.confusion_matrix[1, 1]:>10,}

================================================================================
"""
