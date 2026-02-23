"""
SaaS Churn Simulator - End-to-end subscription churn prediction and intervention toolkit.

This package provides:
- Data loading and preprocessing for RetailRocket ecommerce dataset
- Churn definition and labeling with configurable time windows
- Feature engineering (RFM, behavioral, engagement metrics)
- ML modeling with interpretability (SHAP)
- Customer segmentation (RFM segments, behavioral clustering)
- Intervention ROI simulation
"""

from src.data_loader import DataLoader
from src.churn_definition import ChurnLabeler
from src.features import FeatureEngineer
from src.models import ChurnModel
from src.segmentation import CustomerSegmenter
from src.simulator import InterventionSimulator

__version__ = "1.0.0"

__all__ = [
    "DataLoader",
    "ChurnLabeler",
    "FeatureEngineer",
    "ChurnModel",
    "CustomerSegmenter",
    "InterventionSimulator",
]
