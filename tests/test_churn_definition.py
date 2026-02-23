# tests for churn labeling

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys

sys.path.insert(0, "..")

from src.churn_definition import ChurnLabeler, Windows as ChurnWindows


def make_events():
    """build test events - just returns a df directly, no fixture"""
    base = datetime(2024, 1, 1)
    rows = []

    # cust 1: keeps buying, won't churn
    for d in [10, 30, 50, 70, 90, 100]:
        rows.append(
            {
                "timestamp": base + timedelta(days=d),
                "visitorid": 1,
                "event": "transaction",
                "itemid": 100 + d,
                "transactionid": 1000 + d,
            }
        )

    # cust 2: buys early then stops (churn)
    for d in [10, 20, 30]:
        rows.append(
            {
                "timestamp": base + timedelta(days=d),
                "visitorid": 2,
                "event": "transaction",
                "itemid": 200 + d,
                "transactionid": 2000 + d,
            }
        )

    # cust 3: views only, no purchases (filtered out)
    for d in [10, 30, 50, 70, 90]:
        rows.append(
            {
                "timestamp": base + timedelta(days=d),
                "visitorid": 3,
                "event": "view",
                "itemid": 300 + d,
                "transactionid": np.nan,
            }
        )

    # cust 4: one in obs, one in check (retained)
    rows.append(
        {
            "timestamp": base + timedelta(days=20),
            "visitorid": 4,
            "event": "transaction",
            "itemid": 401,
            "transactionid": 4001,
        }
    )
    rows.append(
        {
            "timestamp": base + timedelta(days=85),
            "visitorid": 4,
            "event": "transaction",
            "itemid": 402,
            "transactionid": 4002,
        }
    )

    # cust 5: one in obs, none in check (churn)
    rows.append(
        {
            "timestamp": base + timedelta(days=20),
            "visitorid": 5,
            "event": "transaction",
            "itemid": 501,
            "transactionid": 5001,
        }
    )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


class TestWindows:
    def test_defaults(self):
        w = ChurnWindows()
        assert w.obs == 60
        assert w.gap == 7
        assert w.chk == 30

    def test_total(self):
        w = ChurnWindows(obs=60, gap=7, chk=30)
        assert w.total == 97

    def test_custom(self):
        w = ChurnWindows(obs=30, gap=3, chk=14)
        assert w.obs == 30
        assert w.total == 47


class TestLabeler:
    def test_basic_labeling(self):
        evts = make_events()
        lbl = ChurnLabeler(windows=ChurnWindows(obs=60, gap=7, chk=30))
        out = lbl.label(evts, snapshot="2024-03-01", min_txns=1)

        assert len(out) > 0
        assert "churned" in out.columns
        assert "visitorid" in out.columns

    def test_non_buyers_excluded(self):
        evts = make_events()
        lbl = ChurnLabeler()
        out = lbl.label(evts, snapshot="2024-03-01")
        # cust 3 only viewed, shouldn't be in output
        assert 3 not in out["visitorid"].values

    def test_labels_are_binary(self):
        evts = make_events()
        lbl = ChurnLabeler()
        out = lbl.label(evts, snapshot="2024-03-15")
        assert out["churned"].isin([0, 1]).all()

    def test_window_cols_present(self):
        evts = make_events()
        lbl = ChurnLabeler()
        out = lbl.label(evts, snapshot="2024-03-01")
        for c in ["obs_start", "obs_end", "chk_start", "chk_end"]:
            assert c in out.columns

    def test_obs_events_filtering(self):
        evts = make_events()
        lbl = ChurnLabeler()
        labels = lbl.label(evts, snapshot="2024-03-01")
        obs = lbl.obs_events(evts, labels)

        start = labels["obs_start"].iloc[0]
        end = labels["obs_end"].iloc[0]
        assert (obs["timestamp"] >= start).all()
        assert (obs["timestamp"] < end).all()


class TestEdgeCases:
    def test_empty_df(self):
        lbl = ChurnLabeler()
        empty = pd.DataFrame(columns=["timestamp", "visitorid", "event", "itemid", "transactionid"])
        empty["timestamp"] = pd.to_datetime(empty["timestamp"])
        out = lbl.label(empty, snapshot="2024-03-01")
        assert len(out) == 0

    def test_no_transactions(self):
        lbl = ChurnLabeler()
        evts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "visitorid": 1,
                "event": "view",
                "itemid": range(10),
                "transactionid": np.nan,
            }
        )
        out = lbl.label(evts, snapshot="2024-02-01")
        assert len(out) == 0

    def test_min_txns_threshold(self):
        evts = make_events()
        lbl = ChurnLabeler()

        out1 = lbl.label(evts, snapshot="2024-03-01", min_txns=1)
        out5 = lbl.label(evts, snapshot="2024-03-01", min_txns=5)

        # higher threshold = fewer or equal customers
        assert len(out5) <= len(out1)
