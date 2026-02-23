"""
Data loader module for RetailRocket ecommerce dataset.

Handles downloading, caching, and loading the raw event data
with proper type handling and date parsing.
"""

import os
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


class DataLoader:
    """
    Load and cache RetailRocket ecommerce dataset.
    
    The dataset contains:
    - events.csv: User behavioral events (view, addtocart, transaction)
    - item_properties_part1.csv, item_properties_part2.csv: Item metadata
    - category_tree.csv: Category hierarchy
    
    Attributes:
        data_dir: Path to data directory
        raw_dir: Path to raw data subdirectory
        processed_dir: Path to processed data subdirectory
    """
    
    KAGGLE_DATASET = "retailrocket/ecommerce-dataset"
    
    EXPECTED_FILES = [
        "events.csv",
        "item_properties_part1.csv",
        "item_properties_part2.csv",
        "category_tree.csv",
    ]
    
    EVENT_DTYPES = {
        "visitorid": "int64",
        "itemid": "int64",
        "event": "category",
        "transactionid": "float64",  # Can be NaN for non-transaction events
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_kaggle(self, force: bool = False) -> bool:
        """
        Download dataset from Kaggle.
        
        Requires kaggle CLI to be configured with API credentials.
        See: https://github.com/Kaggle/kaggle-api#api-credentials
        
        Args:
            force: If True, re-download even if files exist
            
        Returns:
            True if download successful or files already exist
        """
        # Check if files already exist
        if not force and self._check_files_exist():
            print("Dataset files already exist. Use force=True to re-download.")
            return True
            
        print(f"Downloading dataset from Kaggle: {self.KAGGLE_DATASET}")
        
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", self.KAGGLE_DATASET,
                "-p", str(self.raw_dir),
                "--unzip"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Kaggle download failed: {result.stderr}")
                return False
                
            print("Download complete!")
            return True
            
        except FileNotFoundError:
            print("Kaggle CLI not found. Please install: pip install kaggle")
            print("And configure API credentials.")
            return False
    
    def _check_files_exist(self) -> bool:
        """Check if all expected raw files exist."""
        for filename in self.EXPECTED_FILES:
            if not (self.raw_dir / filename).exists():
                return False
        return True
    
    def load_events(
        self,
        sample_frac: Optional[float] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load events data with proper typing and date parsing.
        
        Args:
            sample_frac: If set, randomly sample this fraction of data
            cache: If True, cache processed data as parquet
            
        Returns:
            DataFrame with columns:
            - timestamp: datetime64[ms]
            - visitorid: int64
            - event: category (view, addtocart, transaction)
            - itemid: int64
            - transactionid: float64 (NaN for non-transactions)
        """
        cache_path = self.processed_dir / "events_processed.parquet"
        
        # Try loading from cache
        if cache and cache_path.exists():
            print("Loading events from cache...")
            df = pd.read_parquet(cache_path)
            if sample_frac:
                df = df.sample(frac=sample_frac, random_state=42)
            return df
        
        # Load raw CSV
        raw_path = self.raw_dir / "events.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Events file not found at {raw_path}. "
                "Run download_from_kaggle() first."
            )
        
        print("Loading events from CSV (this may take a moment)...")
        df = pd.read_csv(
            raw_path,
            dtype=self.EVENT_DTYPES,
            parse_dates=False,  # We'll handle timestamp manually
        )
        
        # Convert timestamp (milliseconds since epoch)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Validate and clean
        df = self._validate_events(df)
        
        # Cache processed data
        if cache:
            print(f"Caching processed events to {cache_path}")
            df.to_parquet(cache_path, index=False)
        
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
            
        return df
    
    def _validate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean events data."""
        initial_rows = len(df)
        
        # Remove rows with invalid visitor IDs
        df = df[df["visitorid"] > 0]
        
        # Remove rows with invalid item IDs
        df = df[df["itemid"] > 0]
        
        # Ensure timestamp is valid
        df = df[df["timestamp"].notna()]
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"Removed {removed:,} invalid rows ({removed/initial_rows*100:.2f}%)")
        
        return df
    
    def load_item_properties(self, cache: bool = True) -> pd.DataFrame:
        """
        Load and merge item properties from both files.
        
        Args:
            cache: If True, cache processed data
            
        Returns:
            DataFrame with item properties (pivoted to wide format)
        """
        cache_path = self.processed_dir / "item_properties.parquet"
        
        if cache and cache_path.exists():
            print("Loading item properties from cache...")
            return pd.read_parquet(cache_path)
        
        # Load both property files
        dfs = []
        for filename in ["item_properties_part1.csv", "item_properties_part2.csv"]:
            path = self.raw_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError("Item properties files not found")
        
        # Combine and get latest property value for each item
        props = pd.concat(dfs, ignore_index=True)
        props = props.sort_values("timestamp")
        props = props.drop_duplicates(subset=["itemid", "property"], keep="last")
        
        # Pivot to wide format (one row per item)
        props_wide = props.pivot(
            index="itemid",
            columns="property",
            values="value"
        ).reset_index()
        
        if cache:
            props_wide.to_parquet(cache_path, index=False)
        
        return props_wide
    
    def load_category_tree(self) -> pd.DataFrame:
        """
        Load category hierarchy.
        
        Returns:
            DataFrame with categoryid and parentid columns
        """
        path = self.raw_dir / "category_tree.csv"
        if not path.exists():
            raise FileNotFoundError(f"Category tree not found at {path}")
        
        return pd.read_csv(path)
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics about the loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        events = self.load_events()
        
        summary = {
            "total_events": len(events),
            "unique_visitors": events["visitorid"].nunique(),
            "unique_items": events["itemid"].nunique(),
            "date_range": {
                "start": events["timestamp"].min().isoformat(),
                "end": events["timestamp"].max().isoformat(),
                "days": (events["timestamp"].max() - events["timestamp"].min()).days,
            },
            "event_distribution": events["event"].value_counts().to_dict(),
            "transactions": {
                "total": events[events["event"] == "transaction"].shape[0],
                "unique_transaction_ids": events["transactionid"].dropna().nunique(),
            },
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation."""
        files_status = "ready" if self._check_files_exist() else "not downloaded"
        return f"DataLoader(data_dir='{self.data_dir}', status='{files_status}')"
