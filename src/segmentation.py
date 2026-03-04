# Customer segmentation - RFM and clustering

import operator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    RULES = [
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
    SEG_INFO = {
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

    ACTIONS = {
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
        features,
        r_col="days_since_last_purchase",
        f_col="transaction_count",
        m_col="total_items_purchased",
        q=5,
    ) -> pd.DataFrame:
        """Compute RFM quintile scores.

        r_col: recency (lower = better, so we invert)
        f_col: frequency (higher = better)
        m_col: monetary (higher = better)
        """
        out = features[["visitorid"]].copy()

        # R score (inverted since lower days is better)
        if r_col in features.columns:
            out["R"] = pd.qcut(
                features[r_col].rank(method="first"), q=q, labels=range(q, 0, -1)
            ).astype(int)
        else:
            out["R"] = 3

        # F score
        if f_col in features.columns:
            out["F"] = pd.qcut(
                features[f_col].rank(method="first"), q=q, labels=range(1, q + 1), duplicates="drop"
            ).astype(int)
        else:
            out["F"] = 3

        # M score
        if m_col in features.columns:
            out["M"] = pd.qcut(
                features[m_col].rank(method="first"), q=q, labels=range(1, q + 1), duplicates="drop"
            ).astype(int)
        else:
            out["M"] = 3

        out["RFM"] = out["R"] + out["F"] + out["M"]
        out["RFM_str"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)

        self._rfm = out
        return out

    def assign_segments(self, scores=None) -> pd.DataFrame:
        """Map RFM scores to named segments."""
        df = scores if scores is not None else self._rfm
        if df is None:
            raise ValueError("run rfm_scores first")

        def lookup(row):
            r, f, m = row["R"], row["F"], row["M"]
            for name, rlo, rhi, flo, fhi, mlo, mhi in self.RULES:
                if rlo <= r <= rhi and flo <= f <= fhi and mlo <= m <= mhi:
                    return name
            # fallback
            return "NeedsAttn" if r >= 3 else "Hibernating"

        df["segment"] = df.apply(lookup, axis=1)
        return df

    def rfm_segment(
        self,
        features,
        r_col="days_since_last_purchase",
        f_col="transaction_count",
        m_col="total_items_purchased",
    ) -> pd.DataFrame:
        """One-shot RFM segmentation."""
        scores = self.rfm_scores(features, r_col, f_col, m_col)
        return self.assign_segments(scores)

    def cluster(self, features, cols=None, k=5, rs=42) -> pd.DataFrame:
        """K-means behavioral clustering."""
        if cols is None:
            exclude = ["visitorid", "churned", "segment"]
            cols = [
                c
                for c in features.columns
                if c not in exclude and features[c].dtype in ["int64", "float64"]
            ]

        X = features[cols].values
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0)

        self._kmeans = KMeans(n_clusters=k, random_state=rs, n_init="auto")
        labels = self._kmeans.fit_predict(Xs)

        return pd.DataFrame({"visitorid": features["visitorid"], "cluster": labels})

    def profile_segments(self, features, segments, churn_probs=None, seg_col="segment"):
        """Generate profiles for each segment."""
        data = features.merge(segments[["visitorid", seg_col]], on="visitorid")
        if churn_probs is not None:
            data["cp"] = churn_probs.values

        profiles = []
        n_total = len(data)

        for seg in data[seg_col].unique():
            sub = data[data[seg_col] == seg]

            risk = sub["cp"].mean() if "cp" in sub.columns else None

            # try different column names
            rec = sub.get(
                "days_since_last_purchase", sub.get("days_since_any", pd.Series([0]))
            ).mean()
            freq = sub.get("transaction_count", sub.get("total_events", pd.Series([0]))).mean()
            mon = sub.get("total_items_purchased", pd.Series([0])).mean()

            desc = self.SEG_INFO.get(seg, f"cluster {seg}")
            act = self.ACTIONS.get(seg, "analyze and target")

            profiles.append(
                SegmentProfile(
                    name=seg,
                    size=len(sub),
                    pct=len(sub) / n_total,
                    churn_risk=float(risk) if risk is not None else None,
                    recency=float(rec),
                    freq=float(freq),
                    monetary=float(mon),
                    desc=desc,
                    action=act,
                )
            )

        profiles.sort(key=operator.attrgetter("size"), reverse=True)
        return profiles

    def elbow(self, features, cols=None, max_k=10):
        """Elbow method for picking k."""
        if cols is None:
            exclude = ["visitorid", "churned", "segment"]
            cols = [
                c
                for c in features.columns
                if c not in exclude and features[c].dtype in ["int64", "float64"]
            ]

        X = features[cols].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0)

        inertias = {}
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(Xs)
            inertias[k] = km.inertia_

        return inertias

    def summary_table(self, profiles):
        """Convert profiles to df."""
        return pd.DataFrame([p.to_dict() for p in profiles])

    def high_value_at_risk(self, features, segments, probs, risk_t=0.5, val_pct=0.75):
        """Find high-value customers likely to churn."""
        df = features.merge(segments, on="visitorid")
        df["p"] = probs.values

        val_col = (
            "total_items_purchased"
            if "total_items_purchased" in df.columns
            else "transaction_count"
        )
        val_thresh = df[val_col].quantile(val_pct)

        out = df[(df["p"] >= risk_t) & (df[val_col] >= val_thresh)].copy()
        return out.sort_values("p", ascending=False)
