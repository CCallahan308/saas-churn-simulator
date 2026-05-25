from datetime import timedelta

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.config import DEFAULT_CHK_DAYS, DEFAULT_GAP_DAYS, DEFAULT_OBS_DAYS


class StateWindows(BaseModel):
    """Time window config for state transitions (Active vs Inactive)."""

    obs: int = DEFAULT_OBS_DAYS  # observation days
    gap: int = DEFAULT_GAP_DAYS  # buffer
    chk: int = DEFAULT_CHK_DAYS  # churn check days

    model_config = ConfigDict(frozen=True)

    @property
    def total(self) -> int:
        return self.obs + self.gap + self.chk

    def __repr__(self) -> str:
        return f"StateWindows(obs={self.obs}, gap={self.gap}, check={self.chk})"


class CustomerStateLabeler:
    """Label customers as churned or not.

    Simple rule: no purchase in churn window = churned.
    Only customers with at least one purchase in observation period are labeled.
    """

    def __init__(self, windows: StateWindows | None = None):
        self.windows = windows or StateWindows()

    def label(
        self,
        events: pd.DataFrame,
        snapshot: str | pd.Timestamp | None = None,
        min_txns: int = 1,
    ) -> pd.DataFrame:
        """Main labeling function.

        events: df with timestamp, visitorid, event, itemid, transactionid
        snapshot: reference date (if None, auto-calculated)
        min_txns: min transactions in obs period to be included

        Returns df with visitorid, churned, and window info
        """
        windows = self.windows

        # figure out the snapshot (reference) date
        if snapshot is not None:
            snapshot_date = pd.to_datetime(snapshot)
        else:
            max_timestamp = events["timestamp"].max()
            snapshot_date = max_timestamp - timedelta(days=windows.chk)

        # window boundaries
        obs_end = snapshot_date - timedelta(days=windows.gap)
        obs_start = obs_end - timedelta(days=windows.obs)
        check_start = snapshot_date
        check_end = snapshot_date + timedelta(days=windows.chk)

        logger.info(f"Observation window: {obs_start.date()} to {obs_end.date()}")
        logger.info(f"Gap buffer: {obs_end.date()} to {check_start.date()}")
        logger.info(f"Check window: {check_start.date()} to {check_end.date()}")

        # just transactions
        transactions = events[events["event"] == "transaction"].copy()

        # transactions in the observation window
        obs_transactions = transactions[
            (transactions["timestamp"] >= obs_start) & (transactions["timestamp"] < obs_end)
        ]
        obs_counts = obs_transactions.groupby("visitorid").size().reset_index(name="obs_txn_count")

        # keep only customers active (>= min_txns purchases) in the observation window
        active_customers = obs_counts[obs_counts["obs_txn_count"] >= min_txns]["visitorid"].values
        logger.info(f"Active customers identified: {len(active_customers):,}")

        # transactions in the check window
        check_transactions = transactions[
            (transactions["timestamp"] >= check_start) & (transactions["timestamp"] < check_end)
        ]
        check_counts = (
            check_transactions.groupby("visitorid").size().reset_index(name="check_txn_count")
        )

        # build output
        labels = pd.DataFrame({"visitorid": active_customers})
        labels = labels.merge(obs_counts, on="visitorid", how="left")
        labels = labels.merge(check_counts, on="visitorid", how="left")
        labels["check_txn_count"] = labels["check_txn_count"].fillna(0).astype(int)

        # churn = no transactions in the check window
        labels["churned"] = (labels["check_txn_count"] == 0).astype(int)

        # metadata: carry window boundaries so feature building can reuse them
        labels["observation_start"] = obs_start
        labels["observation_end"] = obs_end
        labels["checkpoint_start"] = check_start
        labels["checkpoint_end"] = check_end

        churn_rate = labels["churned"].mean()
        logger.info(f"Calculated transition (churn) rate: {churn_rate:.1%}")

        return labels

    def train_val_test_split(
        self, events: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Make time-based splits to avoid leakage."""
        min_timestamp = events["timestamp"].min()
        max_timestamp = events["timestamp"].max()
        total_days = (max_timestamp - min_timestamp).days

        buffer_days = self.windows.total
        usable_days = total_days - buffer_days
        test_days = int(usable_days * test_size)
        val_days = int(usable_days * val_size)

        test_snapshot = max_timestamp - timedelta(days=self.windows.chk)
        val_snapshot = test_snapshot - timedelta(days=test_days)
        train_snapshot = val_snapshot - timedelta(days=val_days)

        logger.info(
            f"Dataset span: {min_timestamp.date()} to {max_timestamp.date()} ({total_days}d)"
        )

        train = self.label(events, snapshot=str(train_snapshot.date()))
        val = self.label(events, snapshot=str(val_snapshot.date()))
        test = self.label(events, snapshot=str(test_snapshot.date()))

        return train, val, test

    def obs_events(self, events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """Get events from observation period for the labeled customers."""
        start = labels["observation_start"].iloc[0]
        end = labels["observation_end"].iloc[0]
        visitor_ids = labels["visitorid"].values

        mask = (
            (events["timestamp"] >= start)
            & (events["timestamp"] < end)
            & (events["visitorid"].isin(visitor_ids))
        )

        return events[mask].copy()

    def explain(self) -> str:
        """Human-readable explanation."""
        windows = self.windows
        return f"""
## How we define churn

A customer is "churned" if they don't buy anything in the {windows.chk} day churn window.

Setup:
- Observation: {windows.obs} days (build features here)
- Gap: {windows.gap} days (buffer to avoid peeking at future)
- Check: {windows.chk} days (if no purchase = churned)

So a churned customer hasn't bought anything in {windows.obs + windows.gap + windows.chk}+ days.
That's a pretty clear signal they've disengaged.
"""
