"""
Churn definition and labeling module.

Implements configurable churn labeling based on observation and churn windows.
Business logic for defining when a customer has "churned" in an ecommerce context.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class ChurnWindows:
    """
    Configuration for churn labeling time windows.
    
    Timeline visualization:
    
    |<-- observation_days -->|<-- gap_days -->|<-- churn_days -->|
    |   Build features here  |    Buffer      |  Check for churn |
    
    Attributes:
        observation_days: Days to look back for feature building
        gap_days: Buffer period to prevent data leakage
        churn_days: Window to check for churn (no transaction = churned)
    """
    observation_days: int = 60
    gap_days: int = 7
    churn_days: int = 30
    
    @property
    def total_days_needed(self) -> int:
        """Total days needed for one observation."""
        return self.observation_days + self.gap_days + self.churn_days
    
    def __repr__(self) -> str:
        return (
            f"ChurnWindows(observation={self.observation_days}d, "
            f"gap={self.gap_days}d, churn={self.churn_days}d)"
        )


class ChurnLabeler:
    """
    Label customers as churned or retained based on transaction behavior.
    
    Churn Definition:
    - A customer is "churned" if they have no transactions in the churn window
    - Only customers with at least one transaction in observation window are labeled
    - This ensures we're predicting churn for "active" customers
    
    Example:
        >>> labeler = ChurnLabeler(
        ...     windows=ChurnWindows(observation_days=60, gap_days=7, churn_days=30)
        ... )
        >>> labeled_df = labeler.label_churn(events_df, snapshot_date="2015-08-01")
    """
    
    def __init__(self, windows: Optional[ChurnWindows] = None):
        """
        Initialize churn labeler.
        
        Args:
            windows: ChurnWindows configuration. Uses defaults if None.
        """
        self.windows = windows or ChurnWindows()
        
    def label_churn(
        self,
        events: pd.DataFrame,
        snapshot_date: Optional[str] = None,
        min_transactions_observation: int = 1,
    ) -> pd.DataFrame:
        """
        Label customers with churn status.
        
        Args:
            events: DataFrame with columns [timestamp, visitorid, event, itemid]
            snapshot_date: Date to use as reference point. If None, uses max date
                           minus total_days_needed to ensure enough data.
            min_transactions_observation: Minimum transactions in observation window
                                         to be included (filters to active base)
        
        Returns:
            DataFrame with columns:
            - visitorid: Customer ID
            - churned: 1 if churned, 0 if retained
            - observation_start: Start of observation window
            - observation_end: End of observation window
            - churn_window_start: Start of churn evaluation window
            - churn_window_end: End of churn evaluation window
            - txn_count_observation: Transaction count in observation period
            - txn_count_churn: Transaction count in churn period
        """
        # Determine snapshot date
        if snapshot_date:
            snapshot = pd.to_datetime(snapshot_date)
        else:
            # Use the latest date that allows full window calculation
            max_date = events["timestamp"].max()
            snapshot = max_date - timedelta(days=self.windows.churn_days)
        
        # Calculate window boundaries
        observation_end = snapshot - timedelta(days=self.windows.gap_days)
        observation_start = observation_end - timedelta(days=self.windows.observation_days)
        churn_window_start = snapshot
        churn_window_end = snapshot + timedelta(days=self.windows.churn_days)
        
        print(f"Labeling churn with windows:")
        print(f"  Observation: {observation_start.date()} to {observation_end.date()}")
        print(f"  Gap: {observation_end.date()} to {churn_window_start.date()}")
        print(f"  Churn window: {churn_window_start.date()} to {churn_window_end.date()}")
        
        # Filter to transactions only
        transactions = events[events["event"] == "transaction"].copy()
        
        # Get transactions in observation window
        obs_mask = (
            (transactions["timestamp"] >= observation_start) &
            (transactions["timestamp"] < observation_end)
        )
        obs_transactions = transactions[obs_mask]
        
        # Count transactions per customer in observation window
        obs_counts = obs_transactions.groupby("visitorid").size().reset_index(name="txn_count_observation")
        
        # Filter to customers with minimum transactions
        active_customers = obs_counts[
            obs_counts["txn_count_observation"] >= min_transactions_observation
        ]["visitorid"].values
        
        print(f"  Active customers (>={min_transactions_observation} txns): {len(active_customers):,}")
        
        # Get transactions in churn window
        churn_mask = (
            (transactions["timestamp"] >= churn_window_start) &
            (transactions["timestamp"] < churn_window_end)
        )
        churn_transactions = transactions[churn_mask]
        
        # Count transactions per customer in churn window
        churn_counts = churn_transactions.groupby("visitorid").size().reset_index(name="txn_count_churn")
        
        # Build labeled dataset
        labeled = pd.DataFrame({"visitorid": active_customers})
        labeled = labeled.merge(obs_counts, on="visitorid", how="left")
        labeled = labeled.merge(churn_counts, on="visitorid", how="left")
        labeled["txn_count_churn"] = labeled["txn_count_churn"].fillna(0).astype(int)
        
        # Label churn: 1 if no transactions in churn window
        labeled["churned"] = (labeled["txn_count_churn"] == 0).astype(int)
        
        # Add window metadata
        labeled["observation_start"] = observation_start
        labeled["observation_end"] = observation_end
        labeled["churn_window_start"] = churn_window_start
        labeled["churn_window_end"] = churn_window_end
        
        churn_rate = labeled["churned"].mean()
        print(f"  Churn rate: {churn_rate:.1%}")
        print(f"  Churned: {labeled['churned'].sum():,} | Retained: {(~labeled['churned'].astype(bool)).sum():,}")
        
        return labeled
    
    def create_train_test_split(
        self,
        events: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test splits.
        
        Uses different snapshot dates to prevent data leakage:
        - Training: Earliest window
        - Validation: Middle window
        - Test: Latest window
        
        Args:
            events: Raw events DataFrame
            test_size: Fraction of timeline for test set
            validation_size: Fraction of timeline for validation set
            
        Returns:
            Tuple of (train_labels, val_labels, test_labels)
        """
        min_date = events["timestamp"].min()
        max_date = events["timestamp"].max()
        total_days = (max_date - min_date).days
        
        # Calculate snapshot dates for each split
        # Each split needs total_days_needed from its snapshot
        buffer = self.windows.total_days_needed
        
        usable_days = total_days - buffer
        test_days = int(usable_days * test_size)
        val_days = int(usable_days * validation_size)
        
        test_snapshot = max_date - timedelta(days=self.windows.churn_days)
        val_snapshot = test_snapshot - timedelta(days=test_days)
        train_snapshot = val_snapshot - timedelta(days=val_days)
        
        print(f"Creating time-based splits:")
        print(f"  Data range: {min_date.date()} to {max_date.date()} ({total_days} days)")
        
        train_labels = self.label_churn(events, snapshot_date=str(train_snapshot.date()))
        val_labels = self.label_churn(events, snapshot_date=str(val_snapshot.date()))
        test_labels = self.label_churn(events, snapshot_date=str(test_snapshot.date()))
        
        return train_labels, val_labels, test_labels
    
    def get_observation_events(
        self,
        events: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get events within observation window for labeled customers.
        
        Args:
            events: Full events DataFrame
            labels: Labeled customers DataFrame (output from label_churn)
            
        Returns:
            Events filtered to observation window and labeled customers
        """
        obs_start = labels["observation_start"].iloc[0]
        obs_end = labels["observation_end"].iloc[0]
        customer_ids = labels["visitorid"].values
        
        mask = (
            (events["timestamp"] >= obs_start) &
            (events["timestamp"] < obs_end) &
            (events["visitorid"].isin(customer_ids))
        )
        
        return events[mask].copy()
    
    def explain_churn_definition(self) -> str:
        """
        Return business-friendly explanation of churn definition.
        
        Returns:
            Markdown-formatted explanation string
        """
        return f"""
## Churn Definition

**What is churn?**
A customer is considered "churned" if they do not make any purchase within 
{self.windows.churn_days} days after the observation period ends.

**How we measure it:**

1. **Observation Window** ({self.windows.observation_days} days):
   - We look at customer behavior during this period
   - Build features like purchase frequency, recency, engagement
   - Only customers with at least 1 transaction are included (active base)

2. **Gap Period** ({self.windows.gap_days} days):
   - Buffer to prevent data leakage
   - Ensures features don't accidentally include future information

3. **Churn Window** ({self.windows.churn_days} days):
   - If customer makes a purchase here → Retained (churned=0)
   - If no purchase in this window → Churned (churned=1)

**Business interpretation:**
- A churned customer hasn't purchased in over {self.windows.observation_days + self.windows.gap_days + self.windows.churn_days} days
- This represents a meaningful drop-off in engagement for an ecommerce platform
- Targeting these at-risk customers with retention campaigns can save revenue
"""
