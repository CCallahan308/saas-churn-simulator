"""
Tests for churn definition module.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '..')

from src.churn_definition import ChurnLabeler, ChurnWindows


@pytest.fixture
def sample_events():
    """Create sample events data for testing."""
    base_date = datetime(2024, 1, 1)
    
    # Create events for 10 customers over 120 days
    events = []
    
    # Customer 1: Active, will not churn (purchases throughout)
    for day in [10, 30, 50, 70, 90, 100]:
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 1,
            'event': 'transaction',
            'itemid': 100 + day,
            'transactionid': 1000 + day
        })
    
    # Customer 2: Will churn (purchases only early, none in churn window)
    for day in [10, 20, 30]:
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 2,
            'event': 'transaction',
            'itemid': 200 + day,
            'transactionid': 2000 + day
        })
    
    # Customer 3: Active viewer but no purchases (should be filtered out)
    for day in [10, 30, 50, 70, 90]:
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 3,
            'event': 'view',
            'itemid': 300 + day,
            'transactionid': np.nan
        })
    
    # Customer 4: One purchase in observation, one in churn window (retained)
    events.append({
        'timestamp': base_date + timedelta(days=20),
        'visitorid': 4,
        'event': 'transaction',
        'itemid': 401,
        'transactionid': 4001
    })
    events.append({
        'timestamp': base_date + timedelta(days=85),
        'visitorid': 4,
        'event': 'transaction',
        'itemid': 402,
        'transactionid': 4002
    })
    
    # Customer 5: One purchase in observation, none in churn window (churned)
    events.append({
        'timestamp': base_date + timedelta(days=20),
        'visitorid': 5,
        'event': 'transaction',
        'itemid': 501,
        'transactionid': 5001
    })
    
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


@pytest.fixture
def windows():
    """Create standard churn windows."""
    return ChurnWindows(
        observation_days=60,
        gap_days=7,
        churn_days=30
    )


class TestChurnWindows:
    """Tests for ChurnWindows configuration."""
    
    def test_default_values(self):
        """Test default window values."""
        windows = ChurnWindows()
        assert windows.observation_days == 60
        assert windows.gap_days == 7
        assert windows.churn_days == 30
    
    def test_total_days_needed(self):
        """Test total days calculation."""
        windows = ChurnWindows(observation_days=60, gap_days=7, churn_days=30)
        assert windows.total_days_needed == 97
    
    def test_custom_values(self):
        """Test custom window values."""
        windows = ChurnWindows(observation_days=30, gap_days=3, churn_days=14)
        assert windows.observation_days == 30
        assert windows.gap_days == 3
        assert windows.churn_days == 14
        assert windows.total_days_needed == 47


class TestChurnLabeler:
    """Tests for ChurnLabeler."""
    
    def test_label_churn_basic(self, sample_events, windows):
        """Test basic churn labeling."""
        labeler = ChurnLabeler(windows=windows)
        
        # Use a snapshot date that gives us full windows
        labels = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',  # Day 60
            min_transactions_observation=1
        )
        
        # Should have labeled customers
        assert len(labels) > 0
        assert 'churned' in labels.columns
        assert 'visitorid' in labels.columns
    
    def test_label_churn_excludes_non_purchasers(self, sample_events, windows):
        """Test that customers without purchases are excluded."""
        labeler = ChurnLabeler(windows=windows)
        
        labels = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=1
        )
        
        # Customer 3 only has views, should not be in labels
        assert 3 not in labels['visitorid'].values
    
    def test_churn_label_values(self, sample_events, windows):
        """Test that churn labels are 0 or 1."""
        labeler = ChurnLabeler(windows=windows)
        
        labels = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-15',
            min_transactions_observation=1
        )
        
        assert labels['churned'].isin([0, 1]).all()
    
    def test_window_metadata(self, sample_events, windows):
        """Test that window metadata is included."""
        labeler = ChurnLabeler(windows=windows)
        
        labels = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=1
        )
        
        assert 'observation_start' in labels.columns
        assert 'observation_end' in labels.columns
        assert 'churn_window_start' in labels.columns
        assert 'churn_window_end' in labels.columns
    
    def test_get_observation_events(self, sample_events, windows):
        """Test filtering events to observation window."""
        labeler = ChurnLabeler(windows=windows)
        
        labels = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=1
        )
        
        obs_events = labeler.get_observation_events(sample_events, labels)
        
        # All events should be within observation window
        obs_start = labels['observation_start'].iloc[0]
        obs_end = labels['observation_end'].iloc[0]
        
        assert (obs_events['timestamp'] >= obs_start).all()
        assert (obs_events['timestamp'] < obs_end).all()
    
    def test_explain_churn_definition(self, windows):
        """Test explanation generation."""
        labeler = ChurnLabeler(windows=windows)
        explanation = labeler.explain_churn_definition()
        
        assert isinstance(explanation, str)
        assert '60' in explanation  # observation days
        assert '30' in explanation  # churn days
        assert '7' in explanation   # gap days


class TestChurnLabelerEdgeCases:
    """Edge case tests for ChurnLabeler."""
    
    def test_empty_events(self, windows):
        """Test with empty events dataframe."""
        labeler = ChurnLabeler(windows=windows)
        empty_events = pd.DataFrame(columns=['timestamp', 'visitorid', 'event', 'itemid', 'transactionid'])
        empty_events['timestamp'] = pd.to_datetime(empty_events['timestamp'])
        
        labels = labeler.label_churn(
            empty_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=1
        )
        
        assert len(labels) == 0
    
    def test_no_transactions(self, windows):
        """Test with events but no transactions."""
        labeler = ChurnLabeler(windows=windows)
        
        # Create events with only views
        events = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'visitorid': 1,
            'event': 'view',
            'itemid': range(10),
            'transactionid': np.nan
        })
        
        labels = labeler.label_churn(
            events,
            snapshot_date='2024-02-01',
            min_transactions_observation=1
        )
        
        assert len(labels) == 0
    
    def test_min_transactions_filter(self, sample_events, windows):
        """Test minimum transactions filter."""
        labeler = ChurnLabeler(windows=windows)
        
        # With min=1
        labels_1 = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=1
        )
        
        # With min=5 (higher threshold)
        labels_5 = labeler.label_churn(
            sample_events,
            snapshot_date='2024-03-01',
            min_transactions_observation=5
        )
        
        # Higher threshold should result in fewer or equal customers
        assert len(labels_5) <= len(labels_1)
