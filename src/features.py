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
            rec = self._build_recency(obs_events, obs_end)
            features = features.merge(rec, on="visitorid", how="left")

        # frequency - counts of things
        if "frequency" in categories:
            freq = self._build_frequency(obs_events)
            features = features.merge(freq, on="visitorid", how="left")

        # monetary - txn related (no actual prices in dataset)
        if "monetary" in categories:
            mon = self._build_monetary(obs_events)
            features = features.merge(mon, on="visitorid", how="left")

        # engagement ratios - view->cart->purchase
        if "engagement" in categories:
            eng = self._build_engagement(obs_events)
            features = features.merge(eng, on="visitorid", how="left")

        # trends - comparing first/second half of observation
        if "trend" in categories:
            tr = self._build_trend(obs_events, obs_start, obs_end)
            features = features.merge(tr, on="visitorid", how="left")

        # category/item diversity
        if "category" in categories:
            cat = self._build_category(obs_events)
            features = features.merge(cat, on="visitorid", how="left")

        # fill missing
        features = self._fill_missing(features)

        logger.info(f"Created {len(features.columns) - 1} features")

        return features

    def _build_recency(self, events: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
        """Recency = days since last activity of each type."""
        feats = []

        # using abbreviated names for some
        for evt, col in [
            ("view", "days_view"),
            ("addtocart", "days_since_cart"),
            ("transaction", "days_since_purchase"),
        ]:
            subset = events[events["event"] == evt]
            if len(subset) > 0:
                last = subset.groupby("visitorid")["timestamp"].max().reset_index()
                last[col] = (ref_date - last["timestamp"]).dt.days
                feats.append(last[["visitorid", col]])

        # any activity
        last_any = events.groupby("visitorid")["timestamp"].max().reset_index()
        last_any["days_since_any"] = (ref_date - last_any["timestamp"]).dt.days
        feats.append(last_any[["visitorid", "days_since_any"]])

        result = feats[0]
        for f in feats[1:]:
            result = result.merge(f, on="visitorid", how="outer")

        return result

    def _build_frequency(self, events: pd.DataFrame) -> pd.DataFrame:
        """Count-based features."""
        # event counts by type
        evt_counts = events.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
        evt_counts.columns = [f"{c}_count" for c in evt_counts.columns]
        evt_counts["total_events"] = evt_counts.sum(axis=1)
        evt_counts = evt_counts.reset_index()

        # unique items per event type
        uniq = events.groupby(["visitorid", "event"])["itemid"].nunique().unstack(fill_value=0)
        uniq.columns = [f"uniq_{c}" for c in uniq.columns]
        uniq = uniq.reset_index()

        # sessions
        sess = self._compute_sessions(events)

        # active days
        active = (
            events.groupby("visitorid")
            .apply(lambda x: x["timestamp"].dt.date.nunique())
            .reset_index(name="active_days")
        )

        result = evt_counts.merge(uniq, on="visitorid", how="outer")
        result = result.merge(sess, on="visitorid", how="outer")
        result = result.merge(active, on="visitorid", how="outer")

        return result

    def _compute_sessions(self, events: pd.DataFrame) -> pd.DataFrame:
        """Figure out session boundaries."""
        sorted_evts = events.sort_values(["visitorid", "timestamp"])

        sorted_evts["t_diff"] = sorted_evts.groupby("visitorid")["timestamp"].diff()
        sorted_evts["new_sess"] = (
            sorted_evts["t_diff"] > timedelta(minutes=self.session_timeout)
        ) | sorted_evts["t_diff"].isna()
        sorted_evts["sess_id"] = sorted_evts.groupby("visitorid")["new_sess"].cumsum()

        sess_cnt = sorted_evts.groupby("visitorid")["sess_id"].max().reset_index()
        sess_cnt.columns = ["visitorid", "session_count"]

        evts_per = sorted_evts.groupby(["visitorid", "sess_id"]).size()
        avg_evts = evts_per.groupby("visitorid").mean().reset_index(name="avg_evts_per_sess")

        return sess_cnt.merge(avg_evts, on="visitorid", how="outer")

    def _build_monetary(self, events: pd.DataFrame) -> pd.DataFrame:
        """Monetary features - using counts since no prices."""
        txns = events[events["event"] == "transaction"]

        if len(txns) == 0:
            return pd.DataFrame(
                columns=["visitorid", "txn_count", "avg_items_per_txn", "total_items"]
            )

        # aggregated at visitor level
        stats = (
            txns.groupby("visitorid")
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
        cnts = events.groupby(["visitorid", "event"]).size().unstack(fill_value=0)

        res = pd.DataFrame({"visitorid": cnts.index})

        # view -> cart
        if "view" in cnts.columns and "addtocart" in cnts.columns:
            res["v2c_rate"] = np.where(cnts["view"] > 0, cnts["addtocart"] / cnts["view"], 0)
        else:
            res["v2c_rate"] = 0

        # cart -> purchase
        if "addtocart" in cnts.columns and "transaction" in cnts.columns:
            res["c2p_rate"] = np.where(
                cnts["addtocart"] > 0, cnts["transaction"] / cnts["addtocart"], 0
            )
        else:
            res["c2p_rate"] = 0

        # view -> purchase (direct)
        if "view" in cnts.columns and "transaction" in cnts.columns:
            res["v2p_rate"] = np.where(cnts["view"] > 0, cnts["transaction"] / cnts["view"], 0)
        else:
            res["v2p_rate"] = 0

        # cart abandon
        if "addtocart" in cnts.columns and "transaction" in cnts.columns:
            res["cart_abandon"] = np.where(
                cnts["addtocart"] > 0, 1 - (cnts["transaction"] / cnts["addtocart"]), 1
            )
        else:
            res["cart_abandon"] = 1

        for c in ["v2c_rate", "c2p_rate", "v2p_rate", "cart_abandon"]:
            if c in res.columns:
                res[c] = res[c].clip(0, 1)

        return res.reset_index(drop=True)

    def _build_trend(
        self, events: pd.DataFrame, obs_start: pd.Timestamp, obs_end: pd.Timestamp
    ) -> pd.DataFrame:
        """Compare first half to second half - are they ramping up or down?"""
        mid = obs_start + (obs_end - obs_start) / 2

        first = events[events["timestamp"] < mid]
        second = events[events["timestamp"] >= mid]

        f_cnt = first.groupby("visitorid").size().reset_index(name="first_evts")
        s_cnt = second.groupby("visitorid").size().reset_index(name="second_evts")

        f_txn = (
            first[first["event"] == "transaction"]
            .groupby("visitorid")
            .size()
            .reset_index(name="first_txns")
        )
        s_txn = (
            second[second["event"] == "transaction"]
            .groupby("visitorid")
            .size()
            .reset_index(name="second_txns")
        )

        result = f_cnt.merge(s_cnt, on="visitorid", how="outer")
        result = result.merge(f_txn, on="visitorid", how="outer")
        result = result.merge(s_txn, on="visitorid", how="outer")
        result = result.fillna(0)

        eps = 1e-6  # avoid div by zero
        result["activity_trend"] = (result["second_evts"] - result["first_evts"]) / (
            result["first_evts"] + eps
        )
        result["purchase_trend"] = (result["second_txns"] - result["first_txns"]) / (
            result["first_txns"] + eps
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
                tot_interactions=("itemid", "count"),
            )
            .reset_index()
        )

        stats["diversity_ratio"] = stats["uniq_items"] / stats["tot_interactions"]

        # most-visited item
        item_visits = events.groupby(["visitorid", "itemid"]).size().reset_index(name="visits")
        max_visits = item_visits.groupby("visitorid")["visits"].max().reset_index()
        max_visits.columns = ["visitorid", "fav_item_visits"]

        result = stats.merge(max_visits, on="visitorid", how="left")
        result["repeat_rate"] = 1 - result["diversity_ratio"]

        return result.drop(columns=["tot_interactions"])

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs with reasonable defaults."""
        # recency - worst case
        rec_cols = [c for c in df.columns if "days" in c]
        for c in rec_cols:
            if c in df.columns:
                mx = df[c].max()
                df[c] = df[c].fillna(mx if pd.notna(mx) else 999)

        # counts get 0
        cnt_cols = [
            c for c in df.columns if "count" in c or c.startswith("total") or c.startswith("txn")
        ]
        df[cnt_cols] = df[cnt_cols].fillna(0)

        # rates and ratios
        rate_cols = [c for c in df.columns if "_rate" in c or "_ratio" in c or "abandon" in c]
        df[rate_cols] = df[rate_cols].fillna(0)

        # trends - neutral
        trend_cols = [c for c in df.columns if "trend" in c]
        df[trend_cols] = df[trend_cols].fillna(0)

        # anything left
        df = df.fillna(0)

        return df

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
