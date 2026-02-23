# Churn labeling - deciding who "churned"
# based on time windows: observation -> gap -> churn check

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class Windows:
    """Time window config for churn labeling."""

    obs: int = 60  # observation days
    gap: int = 7  # buffer
    chk: int = 30  # churn check days

    @property
    def total(self):
        return self.obs + self.gap + self.chk

    def __repr__(self):
        return f"Windows(obs={self.obs}, gap={self.gap}, check={self.chk})"


class ChurnLabeler:
    """
    Label customers as churned or not.

    Simple rule: no purchase in churn window = churned.
    Only customers with at least one purchase in observation period are labeled.
    """

    def __init__(self, windows=None):
        self.w = windows or Windows()

    def label(self, events, snapshot=None, min_txns=1):
        """
        Main labeling function.

        events: df with timestamp, visitorid, event, itemid, transactionid
        snapshot: reference date (if None, auto-calculated)
        min_txns: min transactions in obs period to be included

        Returns df with visitorid, churned, and window info
        """
        w = self.w

        # figure out snapshot
        if snapshot:
            snap = pd.to_datetime(snapshot)
        else:
            max_dt = events["timestamp"].max()
            snap = max_dt - timedelta(days=w.chk)

        # boundaries
        obs_end = snap - timedelta(days=w.gap)
        obs_start = obs_end - timedelta(days=w.obs)
        chk_start = snap
        chk_end = snap + timedelta(days=w.chk)

        print(f"obs: {obs_start.date()} to {obs_end.date()}")
        print(f"gap: {obs_end.date()} to {chk_start.date()}")
        print(f"check: {chk_start.date()} to {chk_end.date()}")

        # just transactions
        txns = events[events["event"] == "transaction"].copy()

        # obs period txns
        obs_txns = txns[(txns["timestamp"] >= obs_start) & (txns["timestamp"] < obs_end)]
        obs_cnts = obs_txns.groupby("visitorid").size().reset_index(name="n_obs")

        # filter to active customers
        active = obs_cnts[obs_cnts["n_obs"] >= min_txns]["visitorid"].values
        print(f"active customers: {len(active):,}")

        # check period txns
        chk_txns = txns[(txns["timestamp"] >= chk_start) & (txns["timestamp"] < chk_end)]
        chk_cnts = chk_txns.groupby("visitorid").size().reset_index(name="n_chk")

        # build output
        out = pd.DataFrame({"visitorid": active})
        out = out.merge(obs_cnts, on="visitorid", how="left")
        out = out.merge(chk_cnts, on="visitorid", how="left")
        out["n_chk"] = out["n_chk"].fillna(0).astype(int)

        # churn = no txns in check period
        out["churned"] = (out["n_chk"] == 0).astype(int)

        # metadata
        out["obs_start"] = obs_start
        out["obs_end"] = obs_end
        out["chk_start"] = chk_start
        out["chk_end"] = chk_end

        rate = out["churned"].mean()
        print(f"churn rate: {rate:.1%}")

        return out

    def train_val_test_split(self, events, test_size=0.2, val_size=0.1):
        """Make time-based splits to avoid leakage."""
        mn = events["timestamp"].min()
        mx = events["timestamp"].max()
        total_days = (mx - mn).days

        buf = self.w.total
        usable = total_days - buf
        test_days = int(usable * test_size)
        val_days = int(usable * val_size)

        test_snap = mx - timedelta(days=self.w.chk)
        val_snap = test_snap - timedelta(days=test_days)
        train_snap = val_snap - timedelta(days=val_days)

        print(f"data: {mn.date()} to {mx.date()} ({total_days}d)")

        tr = self.label(events, snapshot=str(train_snap.date()))
        va = self.label(events, snapshot=str(val_snap.date()))
        te = self.label(events, snapshot=str(test_snap.date()))

        return tr, va, te

    def obs_events(self, events, labels):
        """Get events from observation period for the labeled customers."""
        start = labels["obs_start"].iloc[0]
        end = labels["obs_end"].iloc[0]
        vids = labels["visitorid"].values

        m = (
            (events["timestamp"] >= start)
            & (events["timestamp"] < end)
            & (events["visitorid"].isin(vids))
        )

        return events[m].copy()

    def explain(self):
        """Human-readable explanation."""
        w = self.w
        return f"""
## How we define churn

A customer is "churned" if they don't buy anything in the {w.chk} day churn window.

Setup:
- Observation: {w.obs} days (build features here)
- Gap: {w.gap} days (buffer to avoid peeking at future)
- Check: {w.chk} days (if no purchase = churned)

So a churned customer hasn't bought anything in {w.obs + w.gap + w.chk}+ days.
That's a pretty clear signal they've disengaged.
"""
