"""Central configuration constants.

Keeps reproducibility knobs (the random seed), the default churn time-windows,
and on-disk paths in one place instead of scattered literals across modules.
"""

from pathlib import Path

# Reproducibility ------------------------------------------------------------
RANDOM_STATE: int = 42

# Default churn-labeling time windows (days). See src.churn_definition for the
# observation / gap / check semantics.
DEFAULT_OBS_DAYS: int = 60
DEFAULT_GAP_DAYS: int = 7
DEFAULT_CHK_DAYS: int = 30

# Modeling defaults ----------------------------------------------------------
DEFAULT_MODEL_TYPE: str = "lightgbm"
DEFAULT_LTV: float = 100.0

# Paths ----------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
