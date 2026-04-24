# Quick Start Guide: Real Dunnhumby Data

## Step 1: Download Real Dataset

```bash
pip install kagglehub
python download_data.py
```

This downloads the Dunnhumby Complete Journey dataset to `data/` folder.

## Step 2: Explore the Data (Optional)

```bash
jupyter notebook notebooks/eda_dunnhumby.ipynb
```

Run the EDA notebook to understand the dataset before modeling.

## Step 3: Run Pipeline with Real Data

Update `uplift_pipeline.py` to use real data:

```python
# Replace this line:
df = simulate_dunnhumby_data(n_customers=5000)

# With this:
from data_loader import load_dunnhumby_data
df = load_dunnhumby_data(sample_size=5000)  # or None for all households
```

Then run:

```bash
python uplift_pipeline.py
```

## Step 4: View Results

Open `outputs/dashboard.html` in your browser.

Models are automatically saved to `models/` folder.

## File Structure

```
ProAnalytics_Uplift/
├── data/                          ← Real Dunnhumby CSVs
│   ├── transaction_data.csv
│   ├── hh_demographic.csv
│   ├── campaign_table.csv
│   └── ...
├── models/                        ← Saved trained models
│   ├── t_learner_*.pkl
│   ├── x_learner_*.pkl
│   └── model_manifest.json
├── outputs/                       ← Results
│   ├── dashboard.html
│   └── dashboard_data.json
├── notebooks/                     ← Analysis notebooks
│   └── eda_dunnhumby.ipynb
├── download_data.py              ← Download dataset
├── data_loader.py                ← Real data loader
├── model_utils.py                ← Model saving utilities
└── uplift_pipeline.py            ← Main pipeline
```

## Key Differences: Simulated vs Real Data

| Feature | Simulated | Real Dunnhumby |
|---------|-----------|----------------|
| Customers | 5,000 | 2,500 households |
| Transactions | Generated | 2.5M+ real purchases |
| Treatment | Random assignment | Actual campaigns |
| True Uplift | Known (ground truth) | Unknown (estimated) |
| Demographics | Synthetic | Real customer data |

## Notes

- Real data has ~2,500 households (vs 5,000 simulated)
- Treatment assignment based on actual TypeA campaigns
- No ground truth uplift (that's what we're estimating!)
- Models saved automatically to `models/` folder
