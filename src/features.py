"""
Feature engineering module for churn prediction.

Builds comprehensive features from customer behavioral data:
- RFM (Recency, Frequency, Monetary) metrics
- Engagement ratios
- Behavioral trends
- Category preferences
"""

from typing import Optional, List, Dict
from datetime import timedelta

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Build features from customer event data for churn prediction.
    
    Feature Categories:
    1. Recency: Days since last activity (view/cart/purchase)
    2. Frequency: Event counts, session counts
    3. Monetary: Transaction value proxies (count-based for this dataset)
    4. Engagement: Conversion ratios (view→cart→purchase)
    5. Trend: Activity changes over time (first half vs second half)
    6. Category: Purchase diversity and preferences
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.build_features(events_df, labels_df)
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize feature engineer.
        
        Args:
            session_timeout_minutes: Gap in minutes to define new session
        """
        self.session_timeout = session_timeout_minutes
        
    def build_features(
        self,
        events: pd.DataFrame,
        labels: pd.DataFrame,
        include_categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build all features for labeled customers.
        
        Args:
            events: Events within observation window
            labels: Customer labels with observation window metadata
            include_categories: Feature categories to include. 
                              Options: recency, frequency, monetary, engagement, trend, category
                              If None, includes all.
        
        Returns:
            DataFrame with visitorid and all computed features
        """
        categories = include_categories or [
            "recency", "frequency", "monetary", "engagement", "trend", "category"
        ]
        
        obs_end = labels["observation_end"].iloc[0]
        obs_start = labels["observation_start"].iloc[0]
        customer_ids = labels["visitorid"].values
        
        # Filter events to observation window
        mask = (
            (events["timestamp"] >= obs_start) &
            (events["timestamp"] < obs_end) &
            (events["visitorid"].isin(customer_ids))
        )
        obs_events = events[mask].copy()
        
        print(f"Building features for {len(customer_ids):,} customers from {len(obs_events):,} events")
        
        # Initialize features DataFrame
        features = pd.DataFrame({"visitorid": customer_ids})
        
        # Build each feature category
        if "recency" in categories:
            recency_feats = self._build_recency_features(obs_events, obs_end)
            features = features.merge(recency_feats, on="visitorid", how="left")
            
        if "frequency" in categories:
            frequency_feats = self._build_frequency_features(obs_events)
            features = features.merge(frequency_feats, on="visitorid", how="left")
            
        if "monetary" in categories:
            monetary_feats = self._build_monetary_features(obs_events)
            features = features.merge(monetary_feats, on="visitorid", how="left")
            
        if "engagement" in categories:
            engagement_feats = self._build_engagement_features(obs_events)
            features = features.merge(engagement_feats, on="visitorid", how="left")
            
        if "trend" in categories:
            trend_feats = self._build_trend_features(obs_events, obs_start, obs_end)
            features = features.merge(trend_feats, on="visitorid", how="left")
            
        if "category" in categories:
            category_feats = self._build_category_features(obs_events)
            features = features.merge(category_feats, on="visitorid", how="left")
        
        # Fill NaN with appropriate defaults
        features = self._fill_missing_values(features)
        
        print(f"Created {len(features.columns) - 1} features")
        
        return features
    
    def _build_recency_features(
        self,
        events: pd.DataFrame,
        reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Build recency features - days since last activity.
        
        Features:
        - days_since_last_view: Days since last view event
        - days_since_last_cart: Days since last addtocart event
        - days_since_last_purchase: Days since last transaction
        - days_since_last_any: Days since any event
        """
        features = []
        
        for event_type, col_name in [
            ("view", "days_since_last_view"),
            ("addtocart", "days_since_last_cart"),
            ("transaction", "days_since_last_purchase"),
        ]:
            event_subset = events[events["event"] == event_type]
            if len(event_subset) > 0:
                last_event = event_subset.groupby("visitorid")["timestamp"].max().reset_index()
                last_event[col_name] = (reference_date - last_event["timestamp"]).dt.days
                features.append(last_event[["visitorid", col_name]])
        
        # Days since any event
        last_any = events.groupby("visitorid")["timestamp"].max().reset_index()
        last_any["days_since_last_any"] = (reference_date - last_any["timestamp"]).dt.days
        features.append(last_any[["visitorid", "days_since_last_any"]])
        
        # Merge all recency features
        result = features[0]
        for df in features[1:]:
            result = result.merge(df, on="visitorid", how="outer")
        
        return result
    
    def _build_frequency_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Build frequency features - event counts and patterns.
        
        Features:
        - total_events: Total number of events
        - view_count: Number of view events
        - cart_count: Number of addtocart events
        - purchase_count: Number of transactions
        - unique_items_viewed: Distinct items viewed
        - unique_items_carted: Distinct items added to cart
        - unique_items_purchased: Distinct items purchased
        - session_count: Number of distinct sessions
        - avg_events_per_session: Average events per session
        - active_days: Number of days with any activity
        """
        # Event type counts
        event_counts = events.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
        event_counts.columns = [f"{col}_count" for col in event_counts.columns]
        event_counts["total_events"] = event_counts.sum(axis=1)
        event_counts = event_counts.reset_index()
        
        # Unique items per event type
        unique_items = events.groupby(["visitorid", "event"])["itemid"].nunique().unstack(fill_value=0)
        unique_items.columns = [f"unique_items_{col}" for col in unique_items.columns]
        unique_items = unique_items.reset_index()
        
        # Session features
        session_feats = self._compute_session_features(events)
        
        # Active days
        active_days = events.groupby("visitorid").apply(
            lambda x: x["timestamp"].dt.date.nunique()
        ).reset_index(name="active_days")
        
        # Merge all
        result = event_counts.merge(unique_items, on="visitorid", how="outer")
        result = result.merge(session_feats, on="visitorid", how="outer")
        result = result.merge(active_days, on="visitorid", how="outer")
        
        return result
    
    def _compute_session_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """Compute session-based features."""
        events_sorted = events.sort_values(["visitorid", "timestamp"])
        
        # Identify session boundaries (gap > threshold = new session)
        events_sorted["time_diff"] = events_sorted.groupby("visitorid")["timestamp"].diff()
        events_sorted["new_session"] = (
            events_sorted["time_diff"] > timedelta(minutes=self.session_timeout)
        ) | events_sorted["time_diff"].isna()
        events_sorted["session_id"] = events_sorted.groupby("visitorid")["new_session"].cumsum()
        
        # Session count per customer
        session_count = events_sorted.groupby("visitorid")["session_id"].max().reset_index()
        session_count.columns = ["visitorid", "session_count"]
        
        # Average events per session
        events_per_session = events_sorted.groupby(["visitorid", "session_id"]).size()
        avg_events = events_per_session.groupby("visitorid").mean().reset_index(name="avg_events_per_session")
        
        return session_count.merge(avg_events, on="visitorid", how="outer")
    
    def _build_monetary_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Build monetary features.
        
        Note: RetailRocket doesn't have explicit prices, so we use proxies.
        
        Features:
        - transaction_count: Number of purchase events
        - avg_items_per_transaction: Items per transaction
        - total_items_purchased: Total items bought
        """
        transactions = events[events["event"] == "transaction"]
        
        if len(transactions) == 0:
            return pd.DataFrame(columns=["visitorid", "transaction_count", 
                                        "avg_items_per_transaction", "total_items_purchased"])
        
        # Transaction-level aggregations
        txn_stats = transactions.groupby("visitorid").agg(
            transaction_count=("transactionid", "nunique"),
            total_items_purchased=("itemid", "count"),
        ).reset_index()
        
        txn_stats["avg_items_per_transaction"] = (
            txn_stats["total_items_purchased"] / txn_stats["transaction_count"]
        )
        
        return txn_stats
    
    def _build_engagement_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Build engagement features - conversion ratios.
        
        Features:
        - view_to_cart_rate: % of views that became cart additions
        - cart_to_purchase_rate: % of cart items that were purchased
        - view_to_purchase_rate: % of views that led to purchase
        - cart_abandonment_rate: % of carted items not purchased
        """
        # Get counts per user and event type
        counts = events.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
        
        result = pd.DataFrame({"visitorid": counts.index})
        
        # View to cart rate
        if "view" in counts.columns and "addtocart" in counts.columns:
            result["view_to_cart_rate"] = np.where(
                counts["view"] > 0,
                counts["addtocart"] / counts["view"],
                0
            )
        else:
            result["view_to_cart_rate"] = 0
        
        # Cart to purchase rate
        if "addtocart" in counts.columns and "transaction" in counts.columns:
            result["cart_to_purchase_rate"] = np.where(
                counts["addtocart"] > 0,
                counts["transaction"] / counts["addtocart"],
                0
            )
        else:
            result["cart_to_purchase_rate"] = 0
        
        # View to purchase rate
        if "view" in counts.columns and "transaction" in counts.columns:
            result["view_to_purchase_rate"] = np.where(
                counts["view"] > 0,
                counts["transaction"] / counts["view"],
                0
            )
        else:
            result["view_to_purchase_rate"] = 0
        
        # Cart abandonment rate
        if "addtocart" in counts.columns and "transaction" in counts.columns:
            result["cart_abandonment_rate"] = np.where(
                counts["addtocart"] > 0,
                1 - (counts["transaction"] / counts["addtocart"]),
                1
            )
        else:
            result["cart_abandonment_rate"] = 1
        
        # Clip rates to [0, 1]
        rate_cols = ["view_to_cart_rate", "cart_to_purchase_rate", 
                     "view_to_purchase_rate", "cart_abandonment_rate"]
        for col in rate_cols:
            if col in result.columns:
                result[col] = result[col].clip(0, 1)
        
        return result.reset_index(drop=True)
    
    def _build_trend_features(
        self,
        events: pd.DataFrame,
        obs_start: pd.Timestamp,
        obs_end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Build trend features - activity changes over time.
        
        Compare first half vs second half of observation window.
        
        Features:
        - activity_trend: (second_half - first_half) / first_half
        - purchase_trend: Same for transactions
        - engagement_trend: Same for cart additions
        - is_declining: Boolean flag for declining activity
        """
        midpoint = obs_start + (obs_end - obs_start) / 2
        
        first_half = events[events["timestamp"] < midpoint]
        second_half = events[events["timestamp"] >= midpoint]
        
        # Activity counts per half
        first_counts = first_half.groupby("visitorid").size().reset_index(name="first_half_events")
        second_counts = second_half.groupby("visitorid").size().reset_index(name="second_half_events")
        
        # Transaction counts per half
        first_txn = first_half[first_half["event"] == "transaction"].groupby("visitorid").size()
        first_txn = first_txn.reset_index(name="first_half_purchases")
        second_txn = second_half[second_half["event"] == "transaction"].groupby("visitorid").size()
        second_txn = second_txn.reset_index(name="second_half_purchases")
        
        # Merge all
        result = first_counts.merge(second_counts, on="visitorid", how="outer")
        result = result.merge(first_txn, on="visitorid", how="outer")
        result = result.merge(second_txn, on="visitorid", how="outer")
        result = result.fillna(0)
        
        # Calculate trends (avoid division by zero)
        epsilon = 1e-6
        result["activity_trend"] = (
            (result["second_half_events"] - result["first_half_events"]) / 
            (result["first_half_events"] + epsilon)
        )
        result["purchase_trend"] = (
            (result["second_half_purchases"] - result["first_half_purchases"]) / 
            (result["first_half_purchases"] + epsilon)
        )
        
        # Clip extreme values
        result["activity_trend"] = result["activity_trend"].clip(-10, 10)
        result["purchase_trend"] = result["purchase_trend"].clip(-10, 10)
        
        # Declining flag
        result["is_declining"] = (result["activity_trend"] < -0.2).astype(int)
        
        # Drop intermediate columns
        result = result[["visitorid", "activity_trend", "purchase_trend", "is_declining"]]
        
        return result
    
    def _build_category_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Build category/item diversity features.
        
        Features:
        - unique_items_interacted: Total distinct items
        - item_diversity_ratio: unique_items / total_events
        - repeat_item_rate: % of events on previously seen items
        - favorite_item_visits: Events on most visited item
        """
        # Basic item diversity
        item_stats = events.groupby("visitorid").agg(
            unique_items_interacted=("itemid", "nunique"),
            total_interactions=("itemid", "count"),
        ).reset_index()
        
        item_stats["item_diversity_ratio"] = (
            item_stats["unique_items_interacted"] / item_stats["total_interactions"]
        )
        
        # Favorite item visits
        item_visit_counts = events.groupby(["visitorid", "itemid"]).size().reset_index(name="visits")
        max_visits = item_visit_counts.groupby("visitorid")["visits"].max().reset_index()
        max_visits.columns = ["visitorid", "favorite_item_visits"]
        
        result = item_stats.merge(max_visits, on="visitorid", how="left")
        
        # Repeat item rate
        result["repeat_item_rate"] = 1 - result["item_diversity_ratio"]
        
        # Drop intermediate column
        result = result.drop(columns=["total_interactions"])
        
        return result
    
    def _fill_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Recency features: fill with max (worst case)
        recency_cols = [c for c in features.columns if c.startswith("days_since")]
        for col in recency_cols:
            if col in features.columns:
                max_val = features[col].max()
                features[col] = features[col].fillna(max_val if pd.notna(max_val) else 999)
        
        # Count features: fill with 0
        count_cols = [c for c in features.columns if "_count" in c or c.startswith("total_")]
        for col in count_cols:
            features[col] = features[col].fillna(0)
        
        # Rate features: fill with 0
        rate_cols = [c for c in features.columns if "_rate" in c or "_ratio" in c]
        for col in rate_cols:
            features[col] = features[col].fillna(0)
        
        # Trend features: fill with 0 (no change)
        trend_cols = [c for c in features.columns if "_trend" in c]
        for col in trend_cols:
            features[col] = features[col].fillna(0)
        
        # Any remaining NaN
        features = features.fillna(0)
        
        return features
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get human-readable descriptions of all features.
        
        Returns:
            Dictionary mapping feature name to description
        """
        return {
            # Recency
            "days_since_last_view": "Days since customer last viewed a product",
            "days_since_last_cart": "Days since customer last added item to cart",
            "days_since_last_purchase": "Days since customer last made a purchase",
            "days_since_last_any": "Days since any customer activity",
            
            # Frequency
            "total_events": "Total number of customer interactions",
            "view_count": "Number of product views",
            "addtocart_count": "Number of add-to-cart actions",
            "transaction_count": "Number of completed purchases",
            "unique_items_view": "Distinct products viewed",
            "unique_items_addtocart": "Distinct products added to cart",
            "unique_items_transaction": "Distinct products purchased",
            "session_count": "Number of browsing sessions",
            "avg_events_per_session": "Average actions per session",
            "active_days": "Days with at least one activity",
            
            # Monetary
            "avg_items_per_transaction": "Average items per order",
            "total_items_purchased": "Total items bought",
            
            # Engagement
            "view_to_cart_rate": "Conversion rate from view to cart",
            "cart_to_purchase_rate": "Conversion rate from cart to purchase",
            "view_to_purchase_rate": "Direct conversion from view to purchase",
            "cart_abandonment_rate": "Rate of items carted but not purchased",
            
            # Trend
            "activity_trend": "Change in activity (positive = increasing)",
            "purchase_trend": "Change in purchases (positive = increasing)",
            "is_declining": "Flag for significantly declining activity",
            
            # Category/Item
            "unique_items_interacted": "Total distinct items engaged with",
            "item_diversity_ratio": "Variety of items relative to total actions",
            "repeat_item_rate": "Rate of returning to same items",
            "favorite_item_visits": "Visits to most popular item for customer",
        }
