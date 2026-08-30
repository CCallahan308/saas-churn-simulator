# Feature engineering for churn prediction.
# Takes raw events and builds usable features for modeling.

from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """Build features from customer event data.

    Main categories:
    - Recency: days since last activity
    - Frequency: counts, sessions
    - Monetary: transaction stuff (proxy since no prices)
    - Engagement: conversion ratios
    - Trend: first half vs second half changes
    - Category: item diversity
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self.session_timeout = session_timeout_minutes

    def build_features(
        self,
        events: pd.DataFrame,
        labels: pd.DataFrame,
        include_categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build features for the labeled customers.

        include_categories lets you pick which feature types to compute.
        If None, does everything.
        """
        categories = include_categories or [
            "recency",
            "frequency",
            "monetary",
            "engagement",
            "trend",
            "category",
        ]

        obs_end = labels["observation_end"].iloc[0]
        obs_start = labels["observation_start"].iloc[0]
        customer_ids = labels["visitorid"].values

        mask = (
            (events["timestamp"] >= obs_start)
            & (events["timestamp"] < obs_end)
            & (events["visitorid"].isin(customer_ids))
        )
        obs_events = events[mask].copy()

        logger.info(
            f"Building features for {len(customer_ids):,} customers from {len(obs_events):,} events"
        )

        features = pd.DataFrame({"visitorid": customer_ids})

        # recency features - days since stuff
        if "recency" in categories:
            recency_feats = self._build_recency(obs_events, obs_end)
            features = features.merge(recency_feats, on="visitorid", how="left")

        # frequency - counts of things
        if "frequency" in categories:
            frequency_feats = self._build_frequency(obs_events)
            features = features.merge(frequency_feats, on="visitorid", how="left")

        # monetary - txn related (no actual prices in dataset)
        if "monetary" in categories:
            monetary_feats = self._build_monetary(obs_events)
            features = features.merge(monetary_feats, on="visitorid", how="left")

        # engagement ratios - view->cart->purchase
        if "engagement" in categories:
            engagement_feats = self._build_engagement(obs_events)
            features = features.merge(engagement_feats, on="visitorid", how="left")

        # trends - comparing first/second half of observation
        if "trend" in categories:
            trend_feats = self._build_trend(obs_events, obs_start, obs_end)
            features = features.merge(trend_feats, on="visitorid", how="left")

        # category/item diversity
        if "category" in categories:
            category_feats = self._build_category(obs_events)
            features = features.merge(category_feats, on="visitorid", how="left")

        # fill missing
        features = self._fill_missing(features)

        logger.info(f"Created {len(features.columns) - 1} features")

        return features

    def _build_recency(self, events: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
        """Recency = days since last activity of each type."""
        feature_frames = []

        for event_type, column in [
            ("view", "days_view"),
            ("addtocart", "days_since_cart"),
            ("transaction", "days_since_purchase"),
        ]:
            subset = events[events["event"] == event_type]
            if len(subset) > 0:
                last_event = subset.groupby("visitorid")["timestamp"].max().reset_index()
                last_event[column] = (ref_date - last_event["timestamp"]).dt.days
                feature_frames.append(last_event[["visitorid", column]])

        # any activity
        last_any_activity = events.groupby("visitorid")["timestamp"].max().reset_index()
        last_any_activity["days_since_any"] = (
            ref_date - last_any_activity["timestamp"]
        ).dt.days
        feature_frames.append(last_any_activity[["visitorid", "days_since_any"]])

        result = feature_frames[0]
        for frame in feature_frames[1:]:
            result = result.merge(frame, on="visitorid", how="outer")

        return result

    def _build_frequency(self, events: pd.DataFrame) -> pd.DataFrame:
        """Count-based features."""
        if events.empty:
            return pd.DataFrame(columns=["visitorid"])

        # event counts by type. observed=False keeps a column per known event
        # category even when absent in this window (downstream code expects the
        # full set); passing it explicitly silences a pandas FutureWarning.
        event_counts = (
            events.groupby(["visitorid", "event"], observed=False).size().unstack(fill_value=0)
        )
        event_counts.columns = [f"{name}_count" for name in event_counts.columns]
        event_counts["total_events"] = event_counts.sum(axis=1)
        event_counts = event_counts.reset_index()

        # unique items per event type
        unique_items = (
            events.groupby(["visitorid", "event"], observed=False)["itemid"]
            .nunique()
            .unstack(fill_value=0)
        )
        unique_items.columns = [f"uniq_{name}" for name in unique_items.columns]
        unique_items = unique_items.reset_index()

        # sessions
        sessions = self._compute_sessions(events)

        # active days
        active_days = (
            events.groupby("visitorid")
            .apply(lambda group: group["timestamp"].dt.date.nunique(), include_groups=False)
            .reset_index(name="active_days")
        )

        result = event_counts.merge(unique_items, on="visitorid", how="outer")
        result = result.merge(sessions, on="visitorid", how="outer")
        result = result.merge(active_days, on="visitorid", how="outer")

        return result

    def _compute_sessions(self, events: pd.DataFrame) -> pd.DataFrame:
        """Figure out session boundaries."""
        sorted_events = events.sort_values(["visitorid", "timestamp"])

        sorted_events["time_gap"] = sorted_events.groupby("visitorid")["timestamp"].diff()
        sorted_events["is_new_session"] = (
            sorted_events["time_gap"] > timedelta(minutes=self.session_timeout)
        ) | sorted_events["time_gap"].isna()
        sorted_events["session_id"] = sorted_events.groupby("visitorid")["is_new_session"].cumsum()

        session_counts = sorted_events.groupby("visitorid")["session_id"].max().reset_index()
        session_counts.columns = ["visitorid", "session_count"]

        events_per_session = sorted_events.groupby(["visitorid", "session_id"]).size()
        avg_events = (
            events_per_session.groupby("visitorid").mean().reset_index(name="avg_evts_per_sess")
        )

        return session_counts.merge(avg_events, on="visitorid", how="outer")

    def _build_monetary(self, events: pd.DataFrame) -> pd.DataFrame:
        """Monetary features - using counts since no prices."""
        transactions = events[events["event"] == "transaction"]

        if len(transactions) == 0:
            return pd.DataFrame(
                columns=["visitorid", "txn_count", "avg_items_per_txn", "total_items"]
            )

        # aggregated at visitor level
        stats = (
            transactions.groupby("visitorid")
            .agg(
                txn_count=("transactionid", "nunique"),
                total_items=("itemid", "count"),
            )
            .reset_index()
        )

        stats["avg_items_per_txn"] = stats["total_items"] / stats["txn_count"]

        return stats

    def _build_engagement(self, events: pd.DataFrame) -> pd.DataFrame:
        """Conversion ratios - how engaged is this person."""
        counts = events.groupby(["visitorid", "event"], observed=False).size().unstack(fill_value=0)

        result = pd.DataFrame({"visitorid": counts.index})

        # view -> cart
        if "view" in counts.columns and "addtocart" in counts.columns:
            result["v2c_rate"] = np.where(
                counts["view"] > 0, counts["addtocart"] / counts["view"], 0
            )
        else:
            result["v2c_rate"] = 0

        # cart -> purchase
        if "addtocart" in counts.columns and "transaction" in counts.columns:
            result["c2p_rate"] = np.where(
                counts["addtocart"] > 0, counts["transaction"] / counts["addtocart"], 0
            )
        else:
            result["c2p_rate"] = 0

        # view -> purchase (direct)
        if "view" in counts.columns and "transaction" in counts.columns:
            result["v2p_rate"] = np.where(
                counts["view"] > 0, counts["transaction"] / counts["view"], 0
            )
        else:
            result["v2p_rate"] = 0

        # cart abandon
        if "addtocart" in counts.columns and "transaction" in counts.columns:
            result["cart_abandon"] = np.where(
                counts["addtocart"] > 0, 1 - (counts["transaction"] / counts["addtocart"]), 1
            )
        else:
            result["cart_abandon"] = 1

        for rate_col in ["v2c_rate", "c2p_rate", "v2p_rate", "cart_abandon"]:
            if rate_col in result.columns:
                result[rate_col] = result[rate_col].clip(0, 1)

        return result.reset_index(drop=True)

    def _build_trend(
        self, events: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp
    ) -> pd.DataFrame:
        """Compare first half to second half - are they ramping up or down?"""
        midpoint = obs_start + (obs_end - obs_start) / 2

        first_half = events[events["timestamp"] < midpoint]
        second_half = events[events["timestamp"] >= midpoint]

        first_half_events = first_half.groupby("visitorid").size().reset_index(name="first_events")
        second_half_events = (
            second_half.groupby("visitorid").size().reset_index(name="second_events")
        )

        first_half_txns = (
            first_half[first_half["event"] == "transaction"]
            .groupby("visitorid")
            .size()
            .reset_index(name="first_txns")
        )
        second_half_txns = (
            second_half[second_half["event"] == "transaction"]
            .groupby("visitorid")
            .size()
            .reset_index(name="second_txns")
        )

        result = first_half_events.merge(second_half_events, on="visitorid", how="outer")
        result = result.merge(first_half_txns, on="visitorid", how="outer")
        result = result.merge(second_half_txns, on="visitorid", how="outer")
        result = result.fillna(0)

        epsilon = 1e-6  # avoid div by zero
        result["activity_trend"] = (result["second_events"] - result["first_events"]) / (
            result["first_events"] + epsilon
        )
        result["purchase_trend"] = (result["second_txns"] - result["first_txns"]) / (
            result["first_txns"] + epsilon
        )

        result["activity_trend"] = result["activity_trend"].clip(-10, 10)
        result["purchase_trend"] = result["purchase_trend"].clip(-10, 10)

        result["is_declining"] = (result["activity_trend"] < -0.2).astype(int)

        return result[["visitorid", "activity_trend", "purchase_trend", "is_declining"]]

    def _build_category(self, events: pd.DataFrame) -> pd.DataFrame:
        """Item diversity and repeat behavior."""
        stats = (
            events.groupby("visitorid")
            .agg(
                uniq_items=("itemid", "nunique"),
                total_interactions=("itemid", "count"),
            )
            .reset_index()
        )

        stats["diversity_ratio"] = stats["uniq_items"] / stats["total_interactions"]

        # most-visited item
        item_visits = events.groupby(["visitorid", "itemid"]).size().reset_index(name="visits")
        max_visits = item_visits.groupby("visitorid")["visits"].max().reset_index()
        max_visits.columns = ["visitorid", "fav_item_visits"]

        result = stats.merge(max_visits, on="visitorid", how="left")
        result["repeat_rate"] = 1 - result["diversity_ratio"]

        return result.drop(columns=["total_interactions"])

    def _fill_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs with reasonable defaults.

        Wrapped in ``future.no_silent_downcasting`` so the implicit object->numeric
        downcast on fillna is opt-in - this is the pandas>=2.2 forward-compatible
        behavior and silences a FutureWarning that becomes an error in pandas 3.0.
        """
        with pd.option_context("future.no_silent_downcasting", True):
            # recency - worst case (longest observed gap, or 999 if none)
            recency_cols = [col for col in features.columns if "days" in col]
            for col in recency_cols:
                max_recency = features[col].max()
                features[col] = features[col].fillna(
                    max_recency if pd.notna(max_recency) else 999
                )

            # counts get 0
            count_cols = [
                col
                for col in features.columns
                if "count" in col or col.startswith(("total", "txn"))
            ]
            features[count_cols] = features[count_cols].fillna(0)

            # rates and ratios
            rate_cols = [
                col
                for col in features.columns
                if "_rate" in col or "_ratio" in col or "abandon" in col
            ]
            features[rate_cols] = features[rate_cols].fillna(0)

            # trends - neutral
            trend_cols = [col for col in features.columns if "trend" in col]
            features[trend_cols] = features[trend_cols].fillna(0)

            # anything left
            features = features.fillna(0).infer_objects(copy=False)

        return features

    def get_feature_descriptions(self) -> dict[str, str]:
        """Short descriptions of what each feature means."""
        return {
            "days_view": "days since last view",
            "days_since_cart": "days since added to cart",
            "days_since_purchase": "days since bought something",
            "days_since_any": "days since any activity",
            "total_events": "how many total interactions",
            "view_count": "product views",
            "addtocart_count": "add to cart actions",
            "transaction_count": "purchases made",
            "uniq_view": "different products viewed",
            "uniq_addtocart": "different products in cart",
            "uniq_transaction": "different products bought",
            "session_count": "browsing sessions",
            "avg_evts_per_sess": "actions per session",
            "active_days": "days with activity",
            "txn_count": "purchase count",
            "avg_items_per_txn": "items per order",
            "total_items": "total items bought",
            "v2c_rate": "view to cart rate",
            "c2p_rate": "cart to purchase rate",
            "v2p_rate": "view to purchase rate",
            "cart_abandon": "cart abandonment rate",
            "activity_trend": "activity change (pos = up)",
            "purchase_trend": "purchase change (pos = up)",
            "is_declining": "activity dropping off",
            "uniq_items": "different items seen",
            "diversity_ratio": "variety of items / total actions",
            "repeat_rate": "how often returning to same items",
            "fav_item_visits": "visits to favorite item",
        }
