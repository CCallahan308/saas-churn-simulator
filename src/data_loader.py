# Data loader for RetailRocket ecommerce dataset
# handles download from kaggle and caching

import subprocess
from pathlib import Path

import pandas as pd


class DataLoader:
    """Download and load the RetailRocket data.

    Files:
    - events.csv: user events (view/addtocart/transaction)
    - item_properties_part1.csv, part2.csv: item metadata
    - category_tree.csv: category hierarchy
    """

    KAGGLE_DS = "retailrocket/ecommerce-dataset"
    FILES = [
        "events.csv",
        "item_properties_part1.csv",
        "item_properties_part2.csv",
        "category_tree.csv",
    ]
    DTYPES = {
        "visitorid": "int64",
        "itemid": "int64",
        "event": "category",
        "transactionid": "float64",
    }

    def __init__(self, data_dir="data"):
        self.root = Path(data_dir)
        self.raw = self.root / "raw"
        self.proc = self.root / "processed"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.proc.mkdir(parents=True, exist_ok=True)

    def download(self, force=False) -> bool:
        """Pull from kaggle. Needs kaggle CLI configured."""
        if not force and self._check_files():
            print("files exist, skipping download")
            return True

        print(f"downloading {self.KAGGLE_DS}...")
        try:
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                self.KAGGLE_DS,
                "-p",
                str(self.raw),
                "--unzip",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"failed: {res.stderr}")
                return False
            print("done")
            return True
        except FileNotFoundError:
            print("kaggle CLI not found - pip install kaggle and set up credentials")
            return False

    def _check_files(self) -> bool:
        for f in self.FILES:
            if not (self.raw / f).exists():
                return False
        return True

    def load_events(self, sample=None, cache=True) -> pd.DataFrame:
        """Load events with proper types."""
        cachepath = self.proc / "events_processed.parquet"

        if cache and cachepath.exists():
            print("loading from cache...")
            df = pd.read_parquet(cachepath)
            if sample:
                df = df.sample(frac=sample, random_state=42)
            return df

        path = self.raw / "events.csv"
        if not path.exists():
            raise FileNotFoundError(f"no events at {path} - run download() first")

        print("reading csv (may take a bit)...")
        df = pd.read_csv(path, dtype=self.DTYPES, parse_dates=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        df = self._clean_events(df)

        if cache:
            print(f"caching to {cachepath}")
            df.to_parquet(cachepath, index=False)

        if sample:
            df = df.sample(frac=sample, random_state=42)

        return df

    def _clean_events(self, df):
        """Drop invalid rows."""
        n0 = len(df)
        df = df[df["visitorid"] > 0]
        df = df[df["itemid"] > 0]
        df = df[df["timestamp"].notna()]
        df = df.sort_values("timestamp").reset_index(drop=True)
        dropped = n0 - len(df)
        if dropped > 0:
            print(f"removed {dropped:,} bad rows ({dropped / n0 * 100:.2f}%)")
        return df

    def load_item_props(self, cache=True):
        """Load and merge item property files."""
        cachepath = self.proc / "item_props.parquet"

        if cache and cachepath.exists():
            return pd.read_parquet(cachepath)

        dfs = []
        for fn in ["item_properties_part1.csv", "item_properties_part2.csv"]:
            p = self.raw / fn
            if p.exists():
                d = pd.read_csv(p)
                d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms")
                dfs.append(d)

        if not dfs:
            raise FileNotFoundError("no item property files")

        props = pd.concat(dfs, ignore_index=True)
        props = props.sort_values("timestamp")
        props = props.drop_duplicates(subset=["itemid", "property"], keep="last")

        wide = props.pivot(index="itemid", columns="property", values="value").reset_index()

        if cache:
            wide.to_parquet(cachepath, index=False)

        return wide

    def load_categories(self):
        """Load category tree."""
        p = self.raw / "category_tree.csv"
        if not p.exists():
            raise FileNotFoundError(f"no category tree at {p}")
        return pd.read_csv(p)

    def summary(self):
        """Quick stats on the data."""
        ev = self.load_events()
        return {
            "events": len(ev),
            "visitors": ev["visitorid"].nunique(),
            "items": ev["itemid"].nunique(),
            "range": {
                "start": ev["timestamp"].min().isoformat(),
                "end": ev["timestamp"].max().isoformat(),
                "days": (ev["timestamp"].max() - ev["timestamp"].min()).days,
            },
            "by_type": ev["event"].value_counts().to_dict(),
            "txns": {
                "total": ev[ev["event"] == "transaction"].shape[0],
                "unique_ids": ev["transactionid"].dropna().nunique(),
            },
        }

    def __repr__(self):
        status = "ready" if self._check_files() else "need download"
        return f"DataLoader(dir='{self.root}', {status})"
