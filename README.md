# Subscription Churn, Value, and Intervention Simulator

An end-to-end churn prediction and retention analytics system demonstrating real-world SaaS/subscription retention workflows. This project goes beyond basic classification to show feature engineering, customer segmentation, and ROI-based intervention simulation.

## Business Context

**Problem**: Subscription and ecommerce businesses lose 5-7% of revenue annually to preventable customer churn. Identifying at-risk customers and intervening effectively can significantly improve retention.

**Solution**: This project provides a complete analytics pipeline that:
1. Defines churn using business-relevant time windows
2. Engineers features that marketing and product teams understand
3. Builds interpretable ML models with SHAP analysis
4. Segments customers for targeted campaigns
5. Simulates intervention ROI to optimize campaign spend

## Dataset

This project uses the **RetailRocket Ecommerce Dataset** from Kaggle:
- ~2.7 million behavioral events (view, add-to-cart, transaction)
- 1.4 million unique visitors
- 4.5 months of data

**Source**: [Kaggle - RetailRocket Ecommerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

### Alternative Datasets (Inspiration)

- [SaaS Subscription & Churn Analytics](https://www.kaggle.com/datasets/user/saas-churn) - Multi-table synthetic SaaS dataset
- [Customer Subscription Churn and Usage Patterns](https://www.kaggle.com/datasets/user/subscription-churn) - Digital subscription data
- [Online Retail Customer Churn](https://www.kaggle.com/datasets/user/retail-churn) - Retail churn with inactivity-based labels

## Project Structure

```
saas-churn-simulator/
├── data/
│   ├── raw/                    # Downloaded Kaggle files
│   ├── processed/              # Cleaned feature tables
│   └── features/               # Final modeling datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and data quality
│   ├── 02_feature_engineering.ipynb    # Feature building pipeline
│   ├── 03_churn_modeling.ipynb         # Model training with SHAP
│   ├── 04_segmentation_analysis.ipynb  # RFM and behavioral segments
│   └── 05_intervention_simulator.ipynb # Interactive ROI simulator
├── src/
│   ├── data_loader.py          # Raw data ingestion
│   ├── churn_definition.py     # Churn labeling logic
│   ├── features.py             # Feature engineering
│   ├── models.py               # ML models
│   ├── segmentation.py         # Customer segmentation
│   └── simulator.py            # Intervention ROI calculator
├── tests/
│   ├── test_churn_definition.py
│   └── test_features.py
├── figures/                    # Saved visualizations
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Key Features

### 1. Churn Definition Framework

Business-aligned churn labeling with configurable time windows:

```
|<-- Observation (60d) -->|<-- Gap (7d) -->|<-- Churn Window (30d) -->|
|   Build features here   |    Buffer      |  No purchase = churned  |
```

### 2. Feature Engineering Pipeline

| Category | Features | Business Meaning |
|----------|----------|------------------|
| **Recency** | Days since last view/cart/purchase | Customer engagement freshness |
| **Frequency** | Event counts, sessions, active days | Activity intensity |
| **Monetary** | Transaction count, items per order | Customer value |
| **Engagement** | View-to-cart, cart-to-purchase rates | Conversion funnel health |
| **Trend** | Activity slope (first vs. second half) | Engagement direction |
| **Category** | Item diversity, repeat rates | Shopping behavior |

### 3. ML Models with Interpretability

- **Logistic Regression**: Interpretable baseline
- **LightGBM**: High-performance gradient boosting
- **SHAP Analysis**: Feature contribution explanations
- **Business Metrics**: Lift charts, precision@K, calibration curves

### 4. Customer Segmentation

**RFM Segmentation** with named business segments:
- Champions, Loyal Customers, Potential Loyalists
- At Risk, Cannot Lose Them, Hibernating
- Each segment includes recommended actions

**Behavioral Clustering** for data-driven segments:
- K-Means on engagement features
- Cluster profiling with radar charts

### 5. Intervention Simulator

Interactive ROI calculator for campaign planning:

```python
# Example output
{
    "targeted_customers": 1,250,
    "expected_saves": 312,
    "campaign_cost": "$6,250",
    "incremental_revenue": "$44,304",
    "roi": "608%",
    "cost_per_save": "$20.03"
}
```

Features:
- Threshold optimization
- Sensitivity analysis
- Scenario comparison
- Targeting list export

## Installation

### Prerequisites

- Python 3.10+
- Kaggle API credentials ([setup guide](https://github.com/Kaggle/kaggle-api#api-credentials))

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/saas-churn-simulator.git
cd saas-churn-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Download Data

```bash
# Using Kaggle CLI
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip

# Or programmatically
from src.data_loader import DataLoader
loader = DataLoader(data_dir='data')
loader.download_from_kaggle()
```

## Usage

### Quick Start

```python
from src.data_loader import DataLoader
from src.churn_definition import ChurnLabeler, ChurnWindows
from src.features import FeatureEngineer
from src.models import ChurnModel
from src.segmentation import CustomerSegmenter
from src.simulator import InterventionSimulator

# 1. Load data
loader = DataLoader(data_dir='data')
events = loader.load_events()

# 2. Define and label churn
windows = ChurnWindows(observation_days=60, gap_days=7, churn_days=30)
labeler = ChurnLabeler(windows=windows)
labels = labeler.label_churn(events)

# 3. Build features
engineer = FeatureEngineer()
obs_events = labeler.get_observation_events(events, labels)
features = engineer.build_features(obs_events, labels)

# 4. Train model
model = ChurnModel(model_type='lightgbm')
X = features.drop(columns=['visitorid'])
y = labels['churned']
model.fit(X, y)
predictions = model.predict_proba(X)

# 5. Segment customers
segmenter = CustomerSegmenter()
segments = segmenter.rfm_segment(features)

# 6. Simulate intervention
simulator = InterventionSimulator(avg_ltv=100)
result = simulator.simulate(predictions, risk_threshold=0.5)
print(result.summary())
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

Execute notebooks in order:
1. `01_data_exploration.ipynb` - Understand the data
2. `02_feature_engineering.ipynb` - Build features
3. `03_churn_modeling.ipynb` - Train and evaluate models
4. `04_segmentation_analysis.ipynb` - Segment customers
5. `05_intervention_simulator.ipynb` - Plan campaigns

### Run Tests

```bash
pytest tests/ -v
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.85 |
| Precision @ 10% | ~65% |
| Lift @ 10% | ~3.2x |

### Key Predictors

1. **Days since last purchase** - Most predictive of churn
2. **Activity trend** - Declining engagement signals risk
3. **Engagement ratios** - Low conversion indicates disengagement
4. **Transaction frequency** - More purchases = lower churn

### Segment Insights

| Segment | % of Base | Avg Churn Risk | Action |
|---------|-----------|----------------|--------|
| Champions | 8% | 12% | Maintain with rewards |
| At Risk | 15% | 67% | Win-back campaigns |
| Cannot Lose | 5% | 72% | High-touch outreach |

## Business Impact

### Campaign ROI Simulation

Targeting top 20% at-risk customers:
- **Expected saves**: ~300 customers
- **Campaign cost**: ~$6,000
- **Incremental revenue**: ~$40,000
- **Projected ROI**: ~600%

### Sensitivity Analysis

- ROI most sensitive to campaign effectiveness (lift)
- Positive ROI achievable with lift > 10%
- Higher LTV customers justify higher contact costs

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- RetailRocket for the dataset
- scikit-learn, LightGBM, SHAP communities
- Inspiration from SaaS analytics and customer success practices
