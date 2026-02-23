"""
Tests for feature engineering module.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '..')

from src.features import FeatureEngineer


@pytest.fixture
def sample_events():
    """Create sample events data for testing."""
    base_date = datetime(2024, 1, 1)
    
    events = []
    
    # Customer 1: Active with multiple event types
    for day in range(0, 30, 2):
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 1,
            'event': 'view',
            'itemid': 100 + (day % 10),
            'transactionid': np.nan
        })
    
    for day in [5, 15, 25]:
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 1,
            'event': 'addtocart',
            'itemid': 100 + day,
            'transactionid': np.nan
        })
    
    for day in [10, 20]:
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 1,
            'event': 'transaction',
            'itemid': 100 + day,
            'transactionid': 1000 + day
        })
    
    # Customer 2: Only views
    for day in range(0, 20, 5):
        events.append({
            'timestamp': base_date + timedelta(days=day),
            'visitorid': 2,
            'event': 'view',
            'itemid': 200 + day,
            'transactionid': np.nan
        })
    
    # Customer 3: Single transaction
    events.append({
        'timestamp': base_date + timedelta(days=15),
        'visitorid': 3,
        'event': 'transaction',
        'itemid': 301,
        'transactionid': 3001
    })
    
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


@pytest.fixture
def sample_labels():
    """Create sample labels data."""
    return pd.DataFrame({
        'visitorid': [1, 2, 3],
        'churned': [0, 1, 0],
        'observation_start': pd.to_datetime('2024-01-01'),
        'observation_end': pd.to_datetime('2024-02-01'),
    })


@pytest.fixture
def engineer():
    """Create feature engineer instance."""
    return FeatureEngineer(session_timeout_minutes=30)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_build_features_basic(self, sample_events, sample_labels, engineer):
        """Test basic feature building."""
        features = engineer.build_features(sample_events, sample_labels)
        
        assert len(features) == len(sample_labels)
        assert 'visitorid' in features.columns
        # Should have multiple feature columns
        assert len(features.columns) > 5
    
    def test_build_features_includes_all_customers(self, sample_events, sample_labels, engineer):
        """Test that all labeled customers get features."""
        features = engineer.build_features(sample_events, sample_labels)
        
        for vid in sample_labels['visitorid']:
            assert vid in features['visitorid'].values
    
    def test_recency_features(self, sample_events, sample_labels, engineer):
        """Test recency feature computation."""
        features = engineer.build_features(
            sample_events, sample_labels,
            include_categories=['recency']
        )
        
        recency_cols = [c for c in features.columns if 'days_since' in c]
        assert len(recency_cols) > 0
        
        # Recency should be non-negative
        for col in recency_cols:
            assert (features[col] >= 0).all()
    
    def test_frequency_features(self, sample_events, sample_labels, engineer):
        """Test frequency feature computation."""
        features = engineer.build_features(
            sample_events, sample_labels,
            include_categories=['frequency']
        )
        
        # Should have count features
        count_cols = [c for c in features.columns if '_count' in c or c == 'total_events']
        assert len(count_cols) > 0
        
        # Counts should be non-negative
        for col in count_cols:
            assert (features[col] >= 0).all()
    
    def test_engagement_features(self, sample_events, sample_labels, engineer):
        """Test engagement feature computation."""
        features = engineer.build_features(
            sample_events, sample_labels,
            include_categories=['engagement']
        )
        
        rate_cols = [c for c in features.columns if '_rate' in c]
        assert len(rate_cols) > 0
        
        # Rates should be between 0 and 1
        for col in rate_cols:
            assert (features[col] >= 0).all()
            assert (features[col] <= 1).all()
    
    def test_trend_features(self, sample_events, sample_labels, engineer):
        """Test trend feature computation."""
        features = engineer.build_features(
            sample_events, sample_labels,
            include_categories=['trend']
        )
        
        assert 'activity_trend' in features.columns or 'purchase_trend' in features.columns
    
    def test_category_features(self, sample_events, sample_labels, engineer):
        """Test category/item diversity features."""
        features = engineer.build_features(
            sample_events, sample_labels,
            include_categories=['category']
        )
        
        diversity_cols = [c for c in features.columns if 'unique' in c or 'diversity' in c]
        assert len(diversity_cols) > 0
    
    def test_no_nan_in_final_features(self, sample_events, sample_labels, engineer):
        """Test that missing values are properly filled."""
        features = engineer.build_features(sample_events, sample_labels)
        
        # Check for NaN
        nan_counts = features.isna().sum()
        assert nan_counts.sum() == 0, f"Found NaN values: {nan_counts[nan_counts > 0]}"
    
    def test_feature_descriptions(self, engineer):
        """Test feature descriptions are available."""
        descriptions = engineer.get_feature_descriptions()
        
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        
        # All values should be strings
        for desc in descriptions.values():
            assert isinstance(desc, str)


class TestFeatureEngineerEdgeCases:
    """Edge case tests for FeatureEngineer."""
    
    def test_single_event_customer(self, engineer):
        """Test customer with only one event."""
        events = pd.DataFrame({
            'timestamp': [pd.to_datetime('2024-01-15')],
            'visitorid': [1],
            'event': ['transaction'],
            'itemid': [100],
            'transactionid': [1000]
        })
        
        labels = pd.DataFrame({
            'visitorid': [1],
            'churned': [0],
            'observation_start': pd.to_datetime('2024-01-01'),
            'observation_end': pd.to_datetime('2024-02-01'),
        })
        
        features = engineer.build_features(events, labels)
        
        assert len(features) == 1
        assert features.isna().sum().sum() == 0
    
    def test_customer_no_events_in_window(self, engineer):
        """Test customer with no events in observation window."""
        events = pd.DataFrame({
            'timestamp': [pd.to_datetime('2024-03-15')],  # Outside window
            'visitorid': [1],
            'event': ['transaction'],
            'itemid': [100],
            'transactionid': [1000]
        })
        
        labels = pd.DataFrame({
            'visitorid': [1],
            'churned': [0],
            'observation_start': pd.to_datetime('2024-01-01'),
            'observation_end': pd.to_datetime('2024-02-01'),
        })
        
        features = engineer.build_features(events, labels)
        
        # Should still return a row with default values
        assert len(features) == 1
    
    def test_many_customers(self, engineer):
        """Test with many customers."""
        n_customers = 100
        base_date = datetime(2024, 1, 1)
        
        events = []
        for cid in range(n_customers):
            for day in range(0, 30, 5):
                events.append({
                    'timestamp': base_date + timedelta(days=day),
                    'visitorid': cid,
                    'event': np.random.choice(['view', 'addtocart', 'transaction']),
                    'itemid': cid * 100 + day,
                    'transactionid': cid * 1000 + day if np.random.random() > 0.5 else np.nan
                })
        
        events_df = pd.DataFrame(events)
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        
        labels = pd.DataFrame({
            'visitorid': range(n_customers),
            'churned': np.random.randint(0, 2, n_customers),
            'observation_start': pd.to_datetime('2024-01-01'),
            'observation_end': pd.to_datetime('2024-02-01'),
        })
        
        features = engineer.build_features(events_df, labels)
        
        assert len(features) == n_customers
        assert features.isna().sum().sum() == 0


class TestSessionFeatures:
    """Tests for session-based features."""
    
    def test_session_detection(self):
        """Test that sessions are properly detected."""
        engineer = FeatureEngineer(session_timeout_minutes=30)
        
        # Create events with clear session boundaries
        base_date = datetime(2024, 1, 1, 10, 0)  # 10:00 AM
        events = pd.DataFrame({
            'timestamp': [
                base_date,                         # Session 1 start
                base_date + timedelta(minutes=5),  # Still session 1
                base_date + timedelta(minutes=60), # Session 2 (gap > 30 min)
                base_date + timedelta(minutes=65), # Still session 2
            ],
            'visitorid': [1, 1, 1, 1],
            'event': ['view', 'view', 'view', 'transaction'],
            'itemid': [100, 101, 102, 103],
            'transactionid': [np.nan, np.nan, np.nan, 1000]
        })
        
        labels = pd.DataFrame({
            'visitorid': [1],
            'churned': [0],
            'observation_start': pd.to_datetime('2024-01-01'),
            'observation_end': pd.to_datetime('2024-01-02'),
        })
        
        features = engineer.build_features(
            events, labels,
            include_categories=['frequency']
        )
        
        # Should detect 2 sessions
        if 'session_count' in features.columns:
            assert features['session_count'].iloc[0] == 2
