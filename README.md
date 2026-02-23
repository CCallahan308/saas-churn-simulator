# Subscription Churn Simulator

Predicting customer churn, segmenting users, and figuring out if retention campaigns are worth the money.

## What this does

1. Labels customers as churned/not churned based on their purchase behavior
2. Builds features from raw event data (recency, frequency, engagement, etc.)
3. Trains ML models and explains predictions with SHAP
4. Segments customers using RFM or clustering
5. Simulates intervention campaigns and estimates ROI

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.85 |
| Precision @ 10% | ~65% |
| Lift @ 10% | ~3.2x |

Key findings:
- Days since last purchase is the strongest predictor
- Declining activity = high churn risk
- Targeting top 20% at-risk customers: ~600% ROI potential

## Setup

Python 3.10+ and Kaggle API credentials required.

```bash
git clone https://github.com/yourusername/saas-churn-simulator.git
cd saas-churn-simulator
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get the data:
```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
```

Or in python:
```python
from src.data_loader import DataLoader
dl = DataLoader()
dl.download()
```

## Usage

Quick example:
```python
from src.data_loader import DataLoader
from src.churn_definition import ChurnLabeler, Windows
from src.features import FeatureEngineer
from src.models import ChurnModel
from src.simulator import InterventionSimulator

# load
loader = DataLoader()
events = loader.load_events()

# label churn
labeler = ChurnLabeler(windows=Windows(obs=60, gap=7, chk=30))
labels = labeler.label(events)

# features
eng = FeatureEngineer()
obs = labeler.obs_events(events, labels)
feats = eng.build_features(obs, labels)

# model
model = ChurnModel(model_type="lightgbm")
model.fit(feats.drop(columns=["visitorid"]), labels["churned"])
probs = model.predict_proba(feats.drop(columns=["visitorid"]))

# roi
sim = InterventionSimulator(ltv=100)
result = sim.run(probs, threshold=0.5)
print(result.summary())
```

Notebooks (run in order):
- `01_data_exploration.ipynb` - what's in the data
- `02_feature_engineering.ipynb` - building features
- `03_churn_modeling.ipynb` - training + SHAP
- `04_segmentation_analysis.ipynb` - RFM + clusters
- `05_intervention_simulator.ipynb` - campaign ROI

Tests:
```bash
pytest tests/ -v
```

## Project layout

```
saas-churn-simulator/
├── data/
│   ├── raw/           # kaggle files
│   ├── processed/     # cached parquet
│   └── features/      # modeling datasets
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── churn_definition.py
│   ├── features.py
│   ├── models.py
│   ├── segmentation.py
│   └── simulator.py
├── tests/
├── figures/
├── pyproject.toml
└── requirements.txt
```

## Churn definition

```
|<-- 60d observation -->|<-- 7d gap -->|<-- 30d check -->|
|   build features      |   buffer     |  no buy = churn |
```

A customer churns if they don't purchase in the 30-day check window. We only label people who bought at least once during observation (active customers).

## Segment insights

| Segment | % Base | Churn Risk | What to do |
|---------|--------|------------|------------|
| Champions | 8% | 12% | reward, early access |
| At Risk | 15% | 67% | win-back emails |
| Cant Lose | 5% | 72% | personal outreach |

## Dataset

Using RetailRocket Ecommerce from Kaggle - ~2.7M events, 1.4M visitors, 4.5 months.

Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

## License

MIT
