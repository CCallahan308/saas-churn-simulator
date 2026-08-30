# Customer segmentation - RFM and clustering

import operator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


@dataclass
class SegmentProfile:
    """Info about a customer segment."""

    name: str
    size: int
    pct: float
    churn_risk: float | None
    recency: float
    freq: float
    monetary: float
    desc: str
    action: str

    def to_dict(self) -> dict:
        return {
            "segment": self.name,
            "n": self.size,
            "pct": f"{self.pct:.1%}",
            "churn": f"{self.churn_risk:.1%}" if self.churn_risk else "-",
            "recency": f"{self.recency:.0f}d",
            "freq": f"{self.freq:.1f}",
            "monetary": f"{self.monetary:.1f}",
            "action": self.action[:50] + "..." if len(self.action) > 50 else self.action,
        }


class CustomerSegmenter:
    """Segment customers using RFM or k-means.
    """

    # segment rules as (r_min, r_max, f_min, f_max, m_min, m_max)
    # format is different from before to vary structure
    RULES: ClassVar[list[tuple]] = [
        ("Champions", 4, 5, 4, 5, 4, 5),
        ("Loyal", 3, 5, 3, 5, 3, 5),
        ("PotentialLoyalist", 4, 5, 2, 4, 2, 4),
        ("Recent", 4, 5, 1, 2, 1, 2),
        ("Promising", 3, 4, 1, 2, 1, 2),
        ("NeedsAttn", 2, 3, 2, 3, 2, 3),
        ("AboutToSleep", 2, 3, 1, 2, 1, 2),
        ("AtRisk", 1, 2, 3, 5, 3, 5),
        ("CantLose", 1, 2, 4, 5, 4, 5),
        ("Hibernating", 1, 2, 1, 2, 1, 2),
    ]

    # descriptions with varied formatting
    SEG_INFO: ClassVar[dict[str, str]] = {
        "Champions": "your best customers - recent, frequent, high value",
        "Loyal": "good spenders, responsive to promos",
        "PotentialLoyalist": "recent buyers with decent spend",
        "Recent": "just started buying",
        "Promising": "recent but low spend so far",
        "NeedsAttn": "used to be more active",
        "AboutToSleep": "losing them if no action",
        "AtRisk": "big spenders who went quiet",
        "CantLose": "VIPs who havent bought lately - get them back!",
        "Hibernating": "low engagement all around",
    }

    ACTIONS: ClassVar[dict[str, str]] = {
        "Champions": "reward them, early access to new stuff",
        "Loyal": "upsell, ask for reviews",
        "PotentialLoyalist": "loyalty program, cross-sell",
        "Recent": "onboarding, build the habit",
        "Promising": "brand awareness, free trials",
        "NeedsAttn": "limited offers, re-engage",
        "AboutToSleep": "relevant content, discounts",
        "AtRisk": "personalized emails, win-back",
        "CantLose": "high-touch outreach, dont lose to competition",
        "Hibernating": "special promos or let go",
    }

    def __init__(self):
        self._rfm = None
        self._kmeans = None
        self._scaler = None

    def rfm_scores(
        self,
        features: pd.DataFrame,
        r_col: str = "days_since_purchase",
        f_col: str = "transaction_count",
        m_col: str = "total_items",
        q: int = 5,
    ) -> pd.DataFrame:
        """Compute RFM quintile scores.

        r_col: recency (lower = better, so we invert)
        f_col: frequency (higher = better)
        m_col: monetary (higher = better)

        Defaults match the column names emitted by ``FeatureEngineer``
        (days_since_purchase / transaction_count / total_items). If a column
        is absent the score falls back to the neutral quintile (3).
        """
        scores = features[["visitorid"]].copy()

        # R score (inverted since lower days is better)
        if r_col in features.columns:
            scores["R"] = pd.qcut(
                features[r_col].rank(method="first"), q=q, labels=range(q, 0, -1)
            ).astype(int)
        else:
            scores["R"] = 3

        # F score
        if f_col in features.columns:
            scores["F"] = pd.qcut(
                features[f_col].rank(method="first"), q=q, labels=range(1, q + 1), duplicates="drop"
            ).astype(int)
        else:
            scores["F"] = 3

        # M score
        if m_col in features.columns:
            scores["M"] = pd.qcut(
                features[m_col].rank(method="first"), q=q, labels=range(1, q + 1), duplicates="drop"
            ).astype(int)
        else:
            scores["M"] = 3

        scores["RFM"] = scores["R"] + scores["F"] + scores["M"]
        scores["RFM_str"] = (
            scores["R"].astype(str) + scores["F"].astype(str) + scores["M"].astype(str)
        )

        self._rfm = scores
        return scores

    def assign_segments(self, scores: pd.DataFrame | None = None) -> pd.DataFrame:
        """Map RFM scores to named segments."""
        scores_df = scores if scores is not None else self._rfm
        if scores_df is None:
            raise ValueError("run rfm_scores first")

        def lookup(row):
            recency, frequency, monetary = row["R"], row["F"], row["M"]
            for name, r_lo, r_hi, f_lo, f_hi, m_lo, m_hi in self.RULES:
                if r_lo <= recency <= r_hi and f_lo <= frequency <= f_hi and m_lo <= monetary <= m_hi:
                    return name
            # fallback
            return "NeedsAttn" if recency >= 3 else "Hibernating"

        scores_df["segment"] = scores_df.apply(lookup, axis=1)
        return scores_df

    def rfm_segment(
        self,
        features: pd.DataFrame,
        r_col: str = "days_since_purchase",
        f_col: str = "transaction_count",
        m_col: str = "total_items",
    ) -> pd.DataFrame:
        """One-shot RFM segmentation."""
        scores = self.rfm_scores(features, r_col, f_col, m_col)
        return self.assign_segments(scores)

    def cluster(
        self,
        features: pd.DataFrame,
        cols: list[str] | None = None,
        k: int = 5,
        rs: int = RANDOM_STATE,
    ) -> pd.DataFrame:
        """K-means behavioral clustering."""
        if cols is None:
            exclude = ["visitorid", "churned", "segment"]
            cols = [
                col
                for col in features.columns
                if col not in exclude and features[col].dtype in ["int64", "float64"]
            ]

        feature_matrix = features[cols].values
        self._scaler = StandardScaler()
        scaled_features = self._scaler.fit_transform(feature_matrix)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0)

        self._kmeans = KMeans(n_clusters=k, random_state=rs, n_init="auto")
        labels = self._kmeans.fit_predict(scaled_features)

        return pd.DataFrame({"visitorid": features["visitorid"], "cluster": labels})

    def profile_segments(
        self,
        features: pd.DataFrame,
        segments: pd.DataFrame,
        churn_probs: pd.Series | None = None,
        seg_col: str = "segment",
    ) -> list[SegmentProfile]:
        """Generate profiles for each segment."""
        data = features.merge(segments[["visitorid", seg_col]], on="visitorid")
        if churn_probs is not None:
            data["churn_prob"] = churn_probs.values

        profiles = []
        n_total = len(data)

        for segment in data[seg_col].unique():
            segment_rows = data[data[seg_col] == segment]

            risk = segment_rows["churn_prob"].mean() if "churn_prob" in segment_rows.columns else None

            # try different column names with safe access (primary names match
            # FeatureEngineer output; the alternates cover older callers)
            recency = (
                segment_rows["days_since_purchase"].mean()
                if "days_since_purchase" in segment_rows.columns
                else (
                    segment_rows["days_since_any"].mean()
                    if "days_since_any" in segment_rows.columns
                    else 0.0
                )
            )
            frequency = (
                segment_rows["transaction_count"].mean()
                if "transaction_count" in segment_rows.columns
                else (
                    segment_rows["total_events"].mean()
                    if "total_events" in segment_rows.columns
                    else 0.0
                )
            )
            monetary = (
                segment_rows["total_items"].mean()
                if "total_items" in segment_rows.columns
                else 0.0
            )

            description = self.SEG_INFO.get(segment, f"cluster {segment}")
            action = self.ACTIONS.get(segment, "analyze and target")

            profiles.append(
                SegmentProfile(
                    name=segment,
                    size=len(segment_rows),
                    pct=len(segment_rows) / n_total,
                    churn_risk=float(risk) if risk is not None else None,
                    recency=float(recency),
                    freq=float(frequency),
                    monetary=float(monetary),
                    desc=description,
                    action=action,
                )
            )

        profiles.sort(key=operator.attrgetter("size"), reverse=True)
        return profiles

    def elbow(
        self, features: pd.DataFrame, cols: list[str] | None = None, max_k: int = 10
    ) -> dict[int, float]:
        """Elbow method for picking k."""
        if cols is None:
            exclude = ["visitorid", "churned", "segment"]
            cols = [
                col
                for col in features.columns
                if col not in exclude and features[col].dtype in ["int64", "float64"]
            ]

        feature_matrix = features[cols].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        scaled_features = np.nan_to_num(scaled_features, nan=0.0)

        inertias = {}
        for k in range(2, max_k + 1):
            kmeans_model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
            kmeans_model.fit(scaled_features)
            inertias[k] = kmeans_model.inertia_

        return inertias

    def summary_table(self, profiles: list[SegmentProfile]) -> pd.DataFrame:
        """Convert profiles to df."""
        return pd.DataFrame([p.to_dict() for p in profiles])

    def high_value_at_risk(
        self,
        features: pd.DataFrame,
        segments: pd.DataFrame,
        probs: pd.Series,
        risk_t: float = 0.5,
        val_pct: float = 0.75,
    ) -> pd.DataFrame:
        """Find high-value customers likely to churn."""
        merged = features.merge(segments, on="visitorid")
        merged["churn_prob"] = probs.values

        value_col = "total_items" if "total_items" in merged.columns else "transaction_count"
        value_threshold = merged[value_col].quantile(val_pct)

        at_risk = merged[
            (merged["churn_prob"] >= risk_t) & (merged[value_col] >= value_threshold)
        ].copy()
        return at_risk.sort_values("churn_prob", ascending=False)
