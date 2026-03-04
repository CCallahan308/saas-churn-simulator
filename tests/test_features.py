# feature engineering tests

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "..")

from src.features import FeatureEngineer


# using different structure than other test file
@pytest.fixture
def events():
    """Sample events for feature tests"""
    base = datetime(2024, 1, 1)
    rows = []

    # customer 1: mixed activity
    for d in range(0, 30, 2):
        rows.append(
            dict(
                timestamp=base + timedelta(days=d),
                visitorid=1,
                event="view",
                itemid=100 + (d % 10),
                transactionid=np.nan,
            )
        )
    for d in [5, 15, 25]:
        rows.append(
            dict(
                timestamp=base + timedelta(days=d),
                visitorid=1,
                event="addtocart",
                itemid=100 + d,
                transactionid=np.nan,
            )
        )
    for d in [10, 20]:
        rows.append(
            dict(
                timestamp=base + timedelta(days=d),
                visitorid=1,
                event="transaction",
                itemid=100 + d,
                transactionid=1000 + d,
            )
        )

    # customer 2: only views
    for d in range(0, 20, 5):
        rows.append(
            dict(
                timestamp=base + timedelta(days=d),
                visitorid=2,
                event="view",
                itemid=200 + d,
                transactionid=np.nan,
            )
        )

    # customer 3: one purchase
    rows.append(
        dict(
            timestamp=base + timedelta(days=15),
            visitorid=3,
            event="transaction",
            itemid=301,
            transactionid=3001,
        )
    )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def labels():
    return pd.DataFrame(
        {
            "visitorid": [1, 2, 3],
            "churned": [0, 1, 0],
            "observation_start": pd.to_datetime("2024-01-01"),
            "observation_end": pd.to_datetime("2024-02-01"),
        }
    )


eng = FeatureEngineer(session_timeout_minutes=30)


def test_basic_build(events, labels):
    feats = eng.build_features(events, labels)
    assert len(feats) == 3
    assert "visitorid" in feats.columns
    assert len(feats.columns) > 5


def test_all_customers_included(events, labels):
    feats = eng.build_features(events, labels)
    for vid in [1, 2, 3]:
        assert vid in feats["visitorid"].values


def test_recency_values(events, labels):
    feats = eng.build_features(events, labels, include_categories=["recency"])
    rec_cols = [c for c in feats.columns if "days" in c]
    assert len(rec_cols) > 0
    for c in rec_cols:
        assert (feats[c] >= 0).all()


def test_counts_nonneg(events, labels):
    feats = eng.build_features(events, labels, include_categories=["frequency"])
    cnt_cols = [c for c in feats.columns if "count" in c or c == "total_events"]
    for c in cnt_cols:
        assert (feats[c] >= 0).all()


def test_rates_in_range(events, labels):
    feats = eng.build_features(events, labels, include_categories=["engagement"])
    for c in feats.columns:
        if "rate" in c or "abandon" in c:
            assert (feats[c] >= 0).all()
            assert (feats[c] <= 1).all()


def test_trend_exists(events, labels):
    feats = eng.build_features(events, labels, include_categories=["trend"])
    assert "activity_trend" in feats.columns or "purchase_trend" in feats.columns


def test_diversity_features(events, labels):
    feats = eng.build_features(events, labels, include_categories=["category"])
    div_cols = [c for c in feats.columns if "uniq" in c or "diversity" in c]
    assert len(div_cols) > 0


def test_no_nan(events, labels):
    feats = eng.build_features(events, labels)
    nans = feats.isna().sum().sum()
    assert nans == 0, f"found {nans} nans"


def test_descriptions():
    d = eng.get_feature_descriptions()
    assert isinstance(d, dict)
    assert len(d) > 0


# edge cases
def test_single_evt():
    e = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime("2024-01-15")],
            "visitorid": [1],
            "event": ["transaction"],
            "itemid": [100],
            "transactionid": [1000],
        }
    )
    l = pd.DataFrame(
        {
            "visitorid": [1],
            "churned": [0],
            "observation_start": pd.to_datetime("2024-01-01"),
            "observation_end": pd.to_datetime("2024-02-01"),
        }
    )
    f = FeatureEngineer().build_features(e, l)
    assert len(f) == 1
    assert f.isna().sum().sum() == 0


def test_no_evts_in_window():
    e = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime("2024-03-15")],  # outside
            "visitorid": [1],
            "event": ["transaction"],
            "itemid": [100],
            "transactionid": [1000],
        }
    )
    l = pd.DataFrame(
        {
            "visitorid": [1],
            "churned": [0],
            "observation_start": pd.to_datetime("2024-01-01"),
            "observation_end": pd.to_datetime("2024-02-01"),
        }
    )
    f = FeatureEngineer().build_features(e, l)
    assert len(f) == 1


def test_lots_of_customers():
    n = 100
    base = datetime(2024, 1, 1)
    rows = []
    for cid in range(n):
        for d in range(0, 30, 5):
            rows.append(
                dict(
                    timestamp=base + timedelta(days=d),
                    visitorid=cid,
                    event=np.random.choice(["view", "addtocart", "transaction"]),
                    itemid=cid * 100 + d,
                    transactionid=cid * 1000 + d if np.random.random() > 0.5 else np.nan,
                )
            )
    evts = pd.DataFrame(rows)
    evts["timestamp"] = pd.to_datetime(evts["timestamp"])
    labs = pd.DataFrame(
        {
            "visitorid": range(n),
            "churned": np.random.randint(0, 2, n),
            "observation_start": pd.to_datetime("2024-01-01"),
            "observation_end": pd.to_datetime("2024-02-01"),
        }
    )
    feats = FeatureEngineer().build_features(evts, labs)
    assert len(feats) == n
