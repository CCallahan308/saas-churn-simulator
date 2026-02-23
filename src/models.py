"""
Machine learning models for churn prediction.

Implements baseline models with proper evaluation, interpretability,
and business metrics for targeting at-risk customers.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

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
    """Container for model evaluation metrics."""
    
    auc_roc: float
    avg_precision: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    precision_at_k: Dict[int, float]  # Precision at top K%
    lift_at_k: Dict[int, float]  # Lift at top K%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "auc_roc": self.auc_roc,
            "avg_precision": self.avg_precision,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "precision_at_10": self.precision_at_k.get(10, 0),
            "precision_at_20": self.precision_at_k.get(20, 0),
            "lift_at_10": self.lift_at_k.get(10, 0),
            "lift_at_20": self.lift_at_k.get(20, 0),
        }


class ChurnModel:
    """
    Churn prediction model with multiple algorithm support.
    
    Supports:
    - Logistic Regression (interpretable baseline)
    - LightGBM (high performance)
    - Random Forest (robust baseline)
    - Gradient Boosting (sklearn fallback)
    
    Example:
        >>> model = ChurnModel(model_type="lightgbm")
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict_proba(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    SUPPORTED_MODELS = ["logistic", "lightgbm", "random_forest", "gradient_boosting"]
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        **model_params
    ):
        """
        Initialize churn model.
        
        Args:
            model_type: Type of model to use
            random_state: Random seed for reproducibility
            **model_params: Additional parameters for the model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")
        
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            warnings.warn("LightGBM not available, falling back to gradient_boosting")
            model_type = "gradient_boosting"
        
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                **self.model_params
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                class_weight="balanced",
                verbose=-1,
                **self.model_params
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                **self.model_params
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                **self.model_params
            )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True,
        feature_names: Optional[List[str]] = None
    ) -> "ChurnModel":
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable (0=retained, 1=churned)
            scale_features: Whether to standardize features (recommended for logistic)
            feature_names: Names of features (if X is numpy array)
            
        Returns:
            self
        """
        self.feature_names = feature_names or list(X.columns) if hasattr(X, "columns") else None
        
        X_train = X.values if hasattr(X, "values") else X
        y_train = y.values if hasattr(y, "values") else y
        
        # Scale features if requested (especially for logistic regression)
        if scale_features and self.model_type == "logistic":
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of churn probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_pred = X.values if hasattr(X, "values") else X
        
        if self.scaler is not None:
            X_pred = self.scaler.transform(X_pred)
        
        return self.model.predict_proba(X_pred)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict churn labels.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Array of predicted labels
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Classification threshold
            
        Returns:
            ModelMetrics object with all metrics
        """
        y_true = y.values if hasattr(y, "values") else y
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Standard metrics
        auc_roc = roc_auc_score(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Business metrics: Precision and Lift at top K%
        precision_at_k = {}
        lift_at_k = {}
        base_rate = y_true.mean()
        
        for k in [5, 10, 20, 30]:
            top_k_idx = np.argsort(y_proba)[::-1][:int(len(y_proba) * k / 100)]
            precision_k = y_true[top_k_idx].mean()
            precision_at_k[k] = precision_k
            lift_at_k[k] = precision_k / base_rate if base_rate > 0 else 0
        
        return ModelMetrics(
            auc_roc=auc_roc,
            avg_precision=avg_precision,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            precision_at_k=precision_at_k,
            lift_at_k=lift_at_k,
        )
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        X_cv = X.values if hasattr(X, "values") else X
        y_cv = y.values if hasattr(y, "values") else y
        
        if self.scaler is not None:
            X_cv = self.scaler.fit_transform(X_cv)
        
        model = self._create_model()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="roc_auc")
        
        return {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "scores": scores.tolist(),
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if self.model_type == "logistic":
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not available for {self.model_type}")
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        importance_df["importance_pct"] = (
            importance_df["importance"] / importance_df["importance"].sum() * 100
        )
        
        return importance_df.head(top_n).reset_index(drop=True)
    
    def get_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000
    ) -> Tuple[np.ndarray, Any]:
        """
        Compute SHAP values for interpretability.
        
        Args:
            X: Feature matrix
            max_samples: Maximum samples to compute SHAP for
            
        Returns:
            Tuple of (shap_values, explainer)
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_shap = X.values if hasattr(X, "values") else X
        
        if self.scaler is not None:
            X_shap = self.scaler.transform(X_shap)
        
        # Subsample if needed
        if len(X_shap) > max_samples:
            idx = np.random.choice(len(X_shap), max_samples, replace=False)
            X_shap = X_shap[idx]
        
        # Create appropriate explainer
        if self.model_type in ["lightgbm", "random_forest", "gradient_boosting"]:
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.LinearExplainer(self.model, X_shap)
        
        shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values, explainer
    
    def get_calibration_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve data.
        
        Args:
            X: Feature matrix
            y: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        """
        y_true = y.values if hasattr(y, "values") else y
        y_proba = self.predict_proba(X)
        
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )
        
        return fraction_pos, mean_pred
    
    def save(self, path: str):
        """Save model to disk."""
        import joblib
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "random_state": self.random_state,
            "model_params": self.model_params,
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> "ChurnModel":
        """Load model from disk."""
        import joblib
        
        model_data = joblib.load(path)
        
        instance = cls(
            model_type=model_data["model_type"],
            random_state=model_data["random_state"],
            **model_data["model_params"]
        )
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = True
        
        return instance


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple model types.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_types: List of model types to compare
        
    Returns:
        DataFrame with comparison metrics
    """
    model_types = model_types or ["logistic", "lightgbm", "random_forest"]
    
    results = []
    
    for model_type in model_types:
        try:
            model = ChurnModel(model_type=model_type)
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            result = {"model": model_type, **metrics.to_dict()}
            results.append(result)
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
    
    return pd.DataFrame(results).sort_values("auc_roc", ascending=False)


def print_model_report(
    model: ChurnModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> str:
    """
    Print comprehensive model evaluation report.
    
    Args:
        model: Fitted ChurnModel
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Report string
    """
    metrics = model.evaluate(X_test, y_test, threshold)
    y_pred = model.predict(X_test, threshold)
    
    report = f"""
================================================================================
CHURN MODEL EVALUATION REPORT
================================================================================

Model Type: {model.model_type}

CLASSIFICATION METRICS (threshold={threshold})
--------------------------------------------------------------------------------
{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}

RANKING METRICS
--------------------------------------------------------------------------------
AUC-ROC:           {metrics.auc_roc:.4f}
Average Precision: {metrics.avg_precision:.4f}

BUSINESS METRICS
--------------------------------------------------------------------------------
Precision @ Top 10%: {metrics.precision_at_k.get(10, 0):.1%}
Precision @ Top 20%: {metrics.precision_at_k.get(20, 0):.1%}
Lift @ Top 10%:      {metrics.lift_at_k.get(10, 0):.2f}x
Lift @ Top 20%:      {metrics.lift_at_k.get(20, 0):.2f}x

CONFUSION MATRIX
--------------------------------------------------------------------------------
              Predicted Retained    Predicted Churned
Actual Retained    {metrics.confusion_matrix[0, 0]:>10,}    {metrics.confusion_matrix[0, 1]:>10,}
Actual Churned     {metrics.confusion_matrix[1, 0]:>10,}    {metrics.confusion_matrix[1, 1]:>10,}

================================================================================
"""
    return report
