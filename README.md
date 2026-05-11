# Promotion Uplift Modeling Using Causal Machine Learning

**Team ProAnalytics** · Dunnhumby Complete Journey Dataset

---

## Overview

This project estimates the **true causal impact** of retail promotions using uplift modeling — identifying which customers buy *because* of a promotion versus those who would have bought anyway. By targeting only persuadable customers, marketers can maximize ROI while avoiding wasted discount spend.

The pipeline combines propensity score modeling, meta-learners (T-Learner and X-Learner), and behavioral clustering to produce actionable customer segments and an interactive ROI simulator.

### Key Results

| Metric | Value |
|--------|-------|
| Customers Analyzed | 2,500 households |
| Persuadable Customers | 183 (7.3%) |
| Treatment vs Control Lift | +24.23 pp |
| Qini Coefficient | 0.49 |
| Max Net ROI | $5,596 at 50% targeting |

---

## Architecture

```
Dunnhumby CSVs
      │
      ▼
  Data Loading          ← transaction_data, demographics, campaigns, coupon_redempt
      │
      ▼
Feature Engineering     ← RFM scores, promo sensitivity, engagement, value scores
      │
      ▼
Propensity Scoring      ← Logistic Regression → P(T=1|X)
      │
      ▼
Baseline Models         ← LR · Random Forest · XGBoost · Gradient Boosting
      │
      ▼
Causal Models           ← T-Learner (XGBoost) · X-Learner (XGBoost + propensity)
      │
      ▼
Evaluation              ← Uplift Curve · Qini Curve · SHAP importance
      │
      ▼
Segmentation            ← Persuadable · Moderate · Sure Thing · Do-Not-Disturb
      │
      ▼
ROI Simulation          ← Net ROI across targeting thresholds (5%–100%)
      │
      ▼
Dashboard JSON          ← outputs/dashboard_data.json
      │
      ▼
Streamlit Dashboard     ← outputs/dashboard.py
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
python download_data.py
```

This downloads the [Dunnhumby Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) dataset to `data/` and runs the EDA notebook automatically.

> **Manual download:** Place all CSV files into `data/` if you prefer to download directly from Kaggle.

### 3. Run the Pipeline

```bash
python uplift_pipeline.py
```

With options:

```bash
python uplift_pipeline.py \
  --data-dir data \
  --sample-size 2500 \
  --discount-cost 2.0 \
  --no-shap          # skip SHAP for faster runs
```

### 4. Launch the Dashboard

```bash
streamlit run outputs/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## File Structure

```
ProAnalytics_Uplift/
├── data/                          ← Dunnhumby CSVs (not included, ~100MB)
│   ├── transaction_data.csv
│   ├── hh_demographic.csv
│   ├── campaign_table.csv
│   ├── campaign_desc.csv
│   ├── coupon.csv
│   ├── coupon_redempt.csv
│   └── causal_data.csv
│
├── models/                        ← Saved trained models (auto-created)
│   ├── t_learner_treated_*.pkl
│   ├── t_learner_control_*.pkl
│   ├── propensity_model_*.pkl
│   └── model_manifest.json
│
├── notebooks/
│   ├── eda_dunnhumby.ipynb        ← Exploratory data analysis (source)
│   └── eda_dunnhumby_output.ipynb ← EDA results with charts
│
├── outputs/
│   ├── dashboard.py               ← Streamlit dashboard app
│   └── dashboard_data.json        ← Pipeline output consumed by dashboard
│
├── uplift_pipeline.py             ← Main modeling pipeline (10-step)
├── data_loader.py                 ← Real data loader (alternative entry point)
├── model_utils.py                 ← Model persistence utilities
├── download_data.py               ← Kaggle dataset downloader + EDA runner
├── requirements.txt
└── README.md
```

---

## Dataset

The **Dunnhumby Complete Journey** dataset contains two years of household-level grocery shopping data for 2,500 households. It is not included in this repository due to file size.

| Table | Description |
|-------|-------------|
| `transaction_data.csv` | 2.5M+ purchase records with basket IDs, spend, discounts |
| `hh_demographic.csv` | Age, income, household composition, homeowner status |
| `campaign_table.csv` | Which households were exposed to which campaigns |
| `campaign_desc.csv` | Campaign type metadata (TypeA, TypeB, TypeC) |
| `coupon.csv` | Coupon catalogue |
| `coupon_redempt.csv` | Coupon redemptions per household |
| `causal_data.csv` | Display and mailer flags at product level |

---

## Modeling Approach

### Step 1 — Feature Engineering

18 features are derived from raw transaction history:

- **Behavioral:** `purchase_frequency`, `avg_basket_value`, `recency_days`, `store_visits`, `num_categories`, `weekend_shopper`
- **Promotion affinity:** `coupon_redemption_rate`, `past_promo_response`, `discount_sensitivity`, `promo_sensitivity_score`
- **Composite scores:** `rfm_score`, `engagement_score`, `value_score`, `recency_score`
- **Demographics:** `age_encoded`, `income_encoded`, `married`, `household_size`

### Step 2 — Propensity Score Modeling

A logistic regression model estimates `P(T=1 | X)` — the probability a customer was targeted — used downstream in the X-Learner to correct for selection bias.

### Step 3 — Baseline Classifiers

Four classifiers predict `P(purchase)` as a non-causal benchmark:

| Model | AUC |
|-------|-----|
| Logistic Regression | 0.878 |
| Gradient Boosting | 0.874 |
| Random Forest | 0.870 |
| XGBoost | 0.836 |

### Step 4 — T-Learner

Two separate XGBoost models are fit on the treated and control groups:

```
uplift(x) = P(Y=1 | T=1, X=x)  −  P(Y=1 | T=0, X=x)
```

### Step 5 — X-Learner (Primary Model)

The X-Learner (Künzel et al., 2019) improves on the T-Learner for imbalanced treatment/control splits:

1. Fit outcome models `m1` (treated) and `m0` (control)
2. Compute imputed treatment effects: `d1 = Y − m0(X)` for treated; `d0 = m1(X) − Y` for control
3. Fit meta-learners `τ1`, `τ0` on each group
4. Combine with propensity weighting: `τ(x) = g(x)·τ0(x) + (1−g(x))·τ1(x)`

### Step 6 — Customer Segmentation

Each customer is assigned to one of four uplift segments:

| Segment | Uplift Threshold | Strategy |
|---------|-----------------|----------|
| **Persuadable** | > 0.15 | Primary targets — promotion causally drives purchase |
| **Moderate Responder** | 0.05 – 0.15 | Include in broader campaigns |
| **Sure Thing / Neutral** | −0.05 – 0.05 | Would buy anyway — skip to save budget |
| **Do-Not-Disturb** | < −0.05 | Promotion actively reduces purchase probability |

K-Means clustering (k=4) on behavioral features provides an additional behavioral lens, projected to 2D via PCA for visualization.

---

## Dashboard

The Streamlit dashboard at `outputs/dashboard.py` provides five views:

| Tab | Description |
|-----|-------------|
| **Overview** | KPI cards, uplift distribution, Qini curve, feature importance |
| **Segments** | Segment breakdown with average uplift and T vs C comparisons |
| **Model Performance** | Baseline AUC scores, uplift curve, feature ranking |
| **ROI Simulator** | Interactive slider to explore net ROI at different targeting thresholds |
| **Cluster Analysis** | PCA scatter colored by K-Means cluster and uplift segment |

---

## CLI Reference

```
usage: uplift_pipeline.py [-h] [--data-dir DATA_DIR]
                          [--sample-size SAMPLE_SIZE]
                          [--discount-cost DISCOUNT_COST]
                          [--no-shap]

optional arguments:
  --data-dir       Folder with Dunnhumby CSVs (default: data)
  --sample-size    Number of households to sample (default: all)
  --discount-cost  Promotional cost per customer in $ (default: 2.0)
  --no-shap        Skip SHAP explainability (faster runs)

---

## Requirements

```
numpy>=2.0.2
pandas>=2.2.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
seaborn>=0.13.0
xgboost>=2.1.0
shap>=0.45.0
joblib>=1.4.0
kagglehub>=0.2.0
ipykernel>=6.29.0
papermill>=2.6.0
plotly
streamlit
```

---

## References

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://arxiv.org/abs/1706.03461). *PNAS*.
- Radcliffe, N. J. (2007). Using control groups to target on predicted lift. *Analytics Magazine*.
- Dunnhumby. [The Complete Journey Dataset](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey). Kaggle.
