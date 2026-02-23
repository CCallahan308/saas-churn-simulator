"""
Customer segmentation module.

Implements:
- RFM (Recency, Frequency, Monetary) segmentation with named segments
- Behavioral clustering using K-Means
- Segment profiling and recommendations
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentProfile:
    """Profile of a customer segment."""
    
    name: str
    size: int
    size_pct: float
    avg_churn_prob: Optional[float]
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    description: str
    recommended_action: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "size": self.size,
            "size_pct": f"{self.size_pct:.1%}",
            "avg_churn_prob": f"{self.avg_churn_prob:.1%}" if self.avg_churn_prob else "N/A",
            "avg_recency": f"{self.avg_recency:.1f} days",
            "avg_frequency": f"{self.avg_frequency:.1f}",
            "avg_monetary": f"{self.avg_monetary:.1f}",
            "description": self.description,
            "recommended_action": self.recommended_action,
        }


class CustomerSegmenter:
    """
    Customer segmentation using RFM analysis and behavioral clustering.
    
    RFM Segmentation:
    - Assigns quintile scores (1-5) for Recency, Frequency, Monetary
    - Maps score combinations to named business segments
    
    Behavioral Clustering:
    - K-Means on scaled features
    - Automatic cluster profiling
    
    Example:
        >>> segmenter = CustomerSegmenter()
        >>> segments = segmenter.rfm_segment(features_df)
        >>> clusters = segmenter.behavioral_cluster(features_df, n_clusters=5)
    """
    
    # RFM segment definitions: (R_range, F_range, M_range) -> segment_name
    RFM_SEGMENT_RULES = {
        "Champions": {"R": (4, 5), "F": (4, 5), "M": (4, 5)},
        "Loyal Customers": {"R": (3, 5), "F": (3, 5), "M": (3, 5)},
        "Potential Loyalists": {"R": (4, 5), "F": (2, 4), "M": (2, 4)},
        "Recent Customers": {"R": (4, 5), "F": (1, 2), "M": (1, 2)},
        "Promising": {"R": (3, 4), "F": (1, 2), "M": (1, 2)},
        "Needs Attention": {"R": (2, 3), "F": (2, 3), "M": (2, 3)},
        "About to Sleep": {"R": (2, 3), "F": (1, 2), "M": (1, 2)},
        "At Risk": {"R": (1, 2), "F": (3, 5), "M": (3, 5)},
        "Cannot Lose Them": {"R": (1, 2), "F": (4, 5), "M": (4, 5)},
        "Hibernating": {"R": (1, 2), "F": (1, 2), "M": (1, 2)},
        "Lost": {"R": (1, 1), "F": (1, 2), "M": (1, 2)},
    }
    
    SEGMENT_DESCRIPTIONS = {
        "Champions": "Best customers who bought recently, buy often, and spend the most",
        "Loyal Customers": "Spend good money with us often. Responsive to promotions",
        "Potential Loyalists": "Recent customers but spent a good amount and bought more than once",
        "Recent Customers": "Bought most recently, but not often",
        "Promising": "Recent shoppers, but haven't spent much",
        "Needs Attention": "Above average recency, frequency and monetary values. May not have bought very recently",
        "About to Sleep": "Below average recency, frequency and monetary values. Will lose them if not reactivated",
        "At Risk": "Spent big money and purchased often but long time ago. Need to bring them back",
        "Cannot Lose Them": "Made biggest purchases and often, but haven't returned for a long time",
        "Hibernating": "Last purchase was long back, low spenders and low number of orders",
        "Lost": "Lowest recency, frequency and monetary scores",
    }
    
    SEGMENT_ACTIONS = {
        "Champions": "Reward them. Can be early adopters for new products. Will promote your brand",
        "Loyal Customers": "Upsell higher value products. Ask for reviews. Engage them",
        "Potential Loyalists": "Offer membership/loyalty program, recommend other products",
        "Recent Customers": "Provide on-boarding support, give them early success, start building relationship",
        "Promising": "Create brand awareness, offer free trials",
        "Needs Attention": "Make limited time offers, recommend based on past purchases. Reactivate them",
        "About to Sleep": "Share valuable resources, recommend popular products/renewals at discount, reconnect with them",
        "At Risk": "Send personalized emails to reconnect, offer renewals, provide helpful resources",
        "Cannot Lose Them": "Win them back via renewals or newer products, don't lose them to competition, talk to them",
        "Hibernating": "Offer other relevant products and special discounts. Recreate brand value",
        "Lost": "Revive interest with reach out campaign, ignore otherwise",
    }
    
    def __init__(self):
        """Initialize segmenter."""
        self.rfm_scores = None
        self.cluster_model = None
        self.cluster_scaler = None
        
    def compute_rfm_scores(
        self,
        features: pd.DataFrame,
        recency_col: str = "days_since_last_purchase",
        frequency_col: str = "transaction_count",
        monetary_col: str = "total_items_purchased",
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Compute RFM scores using quintiles.
        
        Args:
            features: Feature DataFrame with RFM columns
            recency_col: Column name for recency (lower is better)
            frequency_col: Column name for frequency (higher is better)
            monetary_col: Column name for monetary (higher is better)
            n_quantiles: Number of quantile bins
            
        Returns:
            DataFrame with RFM scores
        """
        rfm = features[["visitorid"]].copy()
        
        # Recency score (inverted - lower days = higher score)
        if recency_col in features.columns:
            rfm["R_score"] = pd.qcut(
                features[recency_col].rank(method="first"),
                q=n_quantiles,
                labels=range(n_quantiles, 0, -1)
            ).astype(int)
        else:
            rfm["R_score"] = 3  # Default middle score
        
        # Frequency score (higher = better)
        if frequency_col in features.columns:
            rfm["F_score"] = pd.qcut(
                features[frequency_col].rank(method="first"),
                q=n_quantiles,
                labels=range(1, n_quantiles + 1),
                duplicates="drop"
            ).astype(int)
        else:
            rfm["F_score"] = 3
        
        # Monetary score (higher = better)
        if monetary_col in features.columns:
            rfm["M_score"] = pd.qcut(
                features[monetary_col].rank(method="first"),
                q=n_quantiles,
                labels=range(1, n_quantiles + 1),
                duplicates="drop"
            ).astype(int)
        else:
            rfm["M_score"] = 3
        
        # Combined RFM score
        rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
        rfm["RFM_string"] = (
            rfm["R_score"].astype(str) + 
            rfm["F_score"].astype(str) + 
            rfm["M_score"].astype(str)
        )
        
        self.rfm_scores = rfm
        return rfm
    
    def assign_rfm_segments(
        self,
        rfm_scores: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Assign named segments based on RFM scores.
        
        Args:
            rfm_scores: DataFrame with RFM scores (uses stored if None)
            
        Returns:
            DataFrame with segment assignments
        """
        rfm = rfm_scores if rfm_scores is not None else self.rfm_scores
        
        if rfm is None:
            raise ValueError("No RFM scores available. Run compute_rfm_scores first.")
        
        def get_segment(row):
            """Assign segment based on RFM scores."""
            r, f, m = row["R_score"], row["F_score"], row["M_score"]
            
            for segment, rules in self.RFM_SEGMENT_RULES.items():
                r_range = rules["R"]
                f_range = rules["F"]
                m_range = rules["M"]
                
                if (r_range[0] <= r <= r_range[1] and
                    f_range[0] <= f <= f_range[1] and
                    m_range[0] <= m <= m_range[1]):
                    return segment
            
            # Default segment for unmatched combinations
            if r >= 3:
                return "Needs Attention"
            else:
                return "Hibernating"
        
        rfm["segment"] = rfm.apply(get_segment, axis=1)
        
        return rfm
    
    def rfm_segment(
        self,
        features: pd.DataFrame,
        recency_col: str = "days_since_last_purchase",
        frequency_col: str = "transaction_count",
        monetary_col: str = "total_items_purchased",
    ) -> pd.DataFrame:
        """
        Full RFM segmentation pipeline.
        
        Args:
            features: Feature DataFrame
            recency_col, frequency_col, monetary_col: Column names
            
        Returns:
            DataFrame with visitorid and segment assignment
        """
        rfm = self.compute_rfm_scores(
            features, recency_col, frequency_col, monetary_col
        )
        rfm = self.assign_rfm_segments(rfm)
        
        return rfm
    
    def behavioral_cluster(
        self,
        features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Cluster customers by behavioral features.
        
        Args:
            features: Feature DataFrame
            feature_cols: Columns to use for clustering (auto-selects if None)
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            DataFrame with visitorid and cluster assignment
        """
        # Select numeric features if not specified
        if feature_cols is None:
            exclude_cols = ["visitorid", "churned", "segment"]
            feature_cols = [
                c for c in features.columns
                if c not in exclude_cols and features[c].dtype in ["int64", "float64"]
            ]
        
        X = features[feature_cols].values
        
        # Scale features
        self.cluster_scaler = StandardScaler()
        X_scaled = self.cluster_scaler.fit_transform(X)
        
        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, 0)
        
        # Cluster
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        clusters = self.cluster_model.fit_predict(X_scaled)
        
        result = pd.DataFrame({
            "visitorid": features["visitorid"],
            "cluster": clusters,
        })
        
        return result
    
    def get_segment_profiles(
        self,
        features: pd.DataFrame,
        segments: pd.DataFrame,
        churn_proba: Optional[pd.Series] = None,
        segment_col: str = "segment",
    ) -> List[SegmentProfile]:
        """
        Generate profiles for each segment.
        
        Args:
            features: Feature DataFrame
            segments: Segment assignments
            churn_proba: Optional churn probabilities per customer
            segment_col: Column name with segment labels
            
        Returns:
            List of SegmentProfile objects
        """
        # Merge features and segments
        data = features.merge(segments[["visitorid", segment_col]], on="visitorid")
        
        if churn_proba is not None:
            data["churn_proba"] = churn_proba.values
        
        profiles = []
        total_customers = len(data)
        
        for segment_name in data[segment_col].unique():
            segment_data = data[data[segment_col] == segment_name]
            
            # Calculate metrics
            size = len(segment_data)
            size_pct = size / total_customers
            
            avg_churn = None
            if "churn_proba" in segment_data.columns:
                avg_churn = segment_data["churn_proba"].mean()
            
            # Get RFM metrics (use fallbacks if columns don't exist)
            avg_recency = segment_data.get(
                "days_since_last_purchase",
                segment_data.get("days_since_last_any", pd.Series([0]))
            ).mean()
            
            avg_frequency = segment_data.get(
                "transaction_count",
                segment_data.get("total_events", pd.Series([0]))
            ).mean()
            
            avg_monetary = segment_data.get(
                "total_items_purchased",
                pd.Series([0])
            ).mean()
            
            # Get descriptions (for RFM segments)
            description = self.SEGMENT_DESCRIPTIONS.get(
                segment_name, f"Behavioral cluster {segment_name}"
            )
            action = self.SEGMENT_ACTIONS.get(
                segment_name, "Analyze behavior and create targeted campaign"
            )
            
            profile = SegmentProfile(
                name=segment_name,
                size=size,
                size_pct=size_pct,
                avg_churn_prob=avg_churn,
                avg_recency=avg_recency,
                avg_frequency=avg_frequency,
                avg_monetary=avg_monetary,
                description=description,
                recommended_action=action,
            )
            profiles.append(profile)
        
        # Sort by size
        profiles.sort(key=lambda x: x.size, reverse=True)
        
        return profiles
    
    def find_optimal_clusters(
        self,
        features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        max_clusters: int = 10,
    ) -> Dict[int, float]:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            features: Feature DataFrame
            feature_cols: Columns to use
            max_clusters: Maximum clusters to try
            
        Returns:
            Dictionary mapping n_clusters to inertia
        """
        if feature_cols is None:
            exclude_cols = ["visitorid", "churned", "segment"]
            feature_cols = [
                c for c in features.columns
                if c not in exclude_cols and features[c].dtype in ["int64", "float64"]
            ]
        
        X = features[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, 0)
        
        inertias = {}
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias[k] = kmeans.inertia_
        
        return inertias
    
    def get_segment_summary_table(
        self,
        profiles: List[SegmentProfile]
    ) -> pd.DataFrame:
        """
        Convert segment profiles to summary DataFrame.
        
        Args:
            profiles: List of SegmentProfile objects
            
        Returns:
            DataFrame with segment summary
        """
        rows = []
        for p in profiles:
            row = {
                "Segment": p.name,
                "Customers": p.size,
                "% of Base": f"{p.size_pct:.1%}",
                "Avg Churn Risk": f"{p.avg_churn_prob:.1%}" if p.avg_churn_prob else "-",
                "Avg Recency": f"{p.avg_recency:.0f}d",
                "Avg Frequency": f"{p.avg_frequency:.1f}",
                "Recommendation": p.recommended_action[:50] + "..." if len(p.recommended_action) > 50 else p.recommended_action,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def target_high_value_at_risk(
        self,
        features: pd.DataFrame,
        segments: pd.DataFrame,
        churn_proba: pd.Series,
        risk_threshold: float = 0.5,
        value_percentile: float = 0.75,
    ) -> pd.DataFrame:
        """
        Identify high-value customers at risk of churning.
        
        Args:
            features: Feature DataFrame
            segments: Segment assignments
            churn_proba: Churn probabilities
            risk_threshold: Minimum churn probability to be "at risk"
            value_percentile: Percentile threshold for "high value"
            
        Returns:
            DataFrame of targeted customers
        """
        data = features.merge(segments, on="visitorid")
        data["churn_proba"] = churn_proba.values
        
        # Define value metric
        value_col = "total_items_purchased" if "total_items_purchased" in data.columns else "transaction_count"
        value_threshold = data[value_col].quantile(value_percentile)
        
        # Filter to high-value at-risk
        targeted = data[
            (data["churn_proba"] >= risk_threshold) &
            (data[value_col] >= value_threshold)
        ].copy()
        
        targeted = targeted.sort_values("churn_proba", ascending=False)
        
        return targeted
