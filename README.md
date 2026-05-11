# Promotion Uplift Modeling Using Causal Machine Learning

**Team ProAnalytics**

> A production-ready causal ML pipeline that estimates the true incremental impact of retail promotions — identifying which customers buy *because* of a promotion versus those who would have purchased anyway.

---

## Overview

Traditional response models predict *who* will buy, but they cannot answer the causal question: *did the promotion cause the purchase?* This project implements **uplift modeling** using state-of-the-art causal machine learning techniques to estimate heterogeneous treatment effects at the individual customer level.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Causal Estimation** | T-Learner & X-Learner meta-algorithms for Individual Treatment Effect (ITE) estimation |
| **Propensity Scoring** | Logistic regression to correct for selection bias in observational data |
| **Customer Segmentation** | K-Means + PCA clustering with 4 actionable uplift segments |
| **ROI Optimization** | Interactive simulator to find optimal targeting thresholds |
| **Explainability** | SHAP values for model interpretability |
| **Production Dashboard** | Streamlit-based interactive analytics dashboard |

---

## Architecture

```
ProAnalytics_Uplift/
├── data/                          ← Dunnhumby CSV files (not in repo)
│   ├── transaction_data.csv
│   ├── hh_demographic.csv
│   ├── campaign_table.csv
│   └── coupon_redempt.csv
├── models/                        ← Serialized trained models (.pkl)
│   ├── t_learner_treated_*.pkl
│   ├── t_learner_control_*.pkl
│   ├── propensity_model_*.pkl
│   └── model_manifest.json
├── outputs/                       ← Generated artifacts
│   ├── dashboard_data.json        ← Pipeline results for dashboard
│   └── dashboard.py             ← Streamlit application
├── notebooks/                     ← Analysis & EDA
│   └── eda_dunnhumby.ipynb
├── download_data.py              ← Kaggle dataset downloader
├── uplift_pipeline.py            ← Main end-to-end pipeline
├── model_utils.py                ← Model persistence utilities
├── requirements.txt              ← Python dependencies
├── QUICKSTART.md                 ← Quick start guide
└── README.md                     ← This file
```

---

## Models Implemented

### Causal Models (Primary)

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| **X-Learner** | XGBoost (primary) | Estimates ITE using imputed treatment effects with propensity-weighted combination |
| **T-Learner** | XGBoost | Two separate outcome models (treated vs control) |
| **Propensity Score** | Logistic Regression | Estimates P(Treatment \| X) for bias correction |

### Baseline Predictive Models (Benchmark)

| Model | AUC (from real data run) |
|-------|--------------------------|
| Logistic Regression | 0.8784 |
| Random Forest | 0.8701 |
| Gradient Boosting | 0.8736 |
| XGBoost | 0.8361 |

---

## Dataset

**Dunnhumby Complete Journey** — Real retail transaction data from 2,500 households over 2 years.

| Feature | Details |
|---------|---------|
| Transactions | 2.5M+ real purchases |
| Households | ~2,500 |
| Time Period | 2 years |
| Treatment | Actual TypeA campaigns (not simulated) |
| Ground Truth | Unknown (estimated via causal models) |

**Key Tables:**
- `transaction_data.csv` — Purchase transactions with product, store, quantity, value
- `hh_demographic.csv` — Household demographics (age, income, size, homeowner)
- `campaign_table.csv` — Campaign assignments by household
- `coupon_redempt.csv` — Coupon redemption behavior

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- 4GB+ RAM recommended
- Kaggle account (for dataset download)

### 1. Clone Repository

```bash
git clone https://github.com/hrushh22/ProAnalytics-Uplift.git
cd ProAnalytics-Uplift
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
pip install kagglehub
python download_data.py
```

This downloads the Dunnhumby Complete Journey dataset from Kaggle and places CSV files in `data/`.

> **Note:** The dataset is >100MB and not included in this repository. You can also manually download from [Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey).

### 4. Run Pipeline

```bash
python uplift_pipeline.py
```

**CLI Options:**
```bash
python uplift_pipeline.py --help
# Options:
#   --data-dir PATH        Data folder (default: data)
#   --sample-size N        Sample N households (default: all)
#   --discount-cost FLOAT  Cost per customer in $ (default: 2.0)
#   --no-shap              Skip SHAP explainability (faster)
```

### 5. Launch Dashboard

```bash
cd outputs
streamlit run dashboard.py
```

---

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING                                           │
│  Load Dunnhumby tables, sample households, build features      │
├─────────────────────────────────────────────────────────────────┤
│  STEP 2: FEATURE ENGINEERING                                    │
│  Encode demographics, create composite scores (RFM, engagement)  │
├─────────────────────────────────────────────────────────────────┤
│  STEP 3: PROPENSITY SCORING                                     │
│  Logistic regression: P(Treatment=1 | X)                        │
├─────────────────────────────────────────────────────────────────┤
│  STEP 4: BASELINE MODELS                                        │
│  Train LR, RF, XGBoost, GBM for benchmark comparison            │
├─────────────────────────────────────────────────────────────────┤
│  STEP 5: T-LEARNER                                              │
│  Separate models for treated/control; uplift = P1 - P0          │
├─────────────────────────────────────────────────────────────────┤
│  STEP 6: X-LEARNER (Primary)                                    │
│  Imputed treatment effects with propensity weighting            │
├─────────────────────────────────────────────────────────────────┤
│  STEP 7: EVALUATION                                             │
│  Uplift curves, Qini curves, Qini coefficient                   │
├─────────────────────────────────────────────────────────────────┤
│  STEP 8: SEGMENTATION                                           │
│  K-Means clustering + PCA visualization                         │
├─────────────────────────────────────────────────────────────────┤
│  STEP 9: ROI SIMULATION                                         │
│  Net ROI at varying targeting thresholds                        │
├─────────────────────────────────────────────────────────────────┤
│  STEP 10: EXPORT & SAVE                                         │
│  JSON for dashboard + serialized models                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Results (Real Data Run)

| Metric | Value |
|--------|-------|
| **Customers Analyzed** | 2,500 |
| **Persuadable Customers** | 183 (7.3%) |
| **Campaign Lift** | +24.23 percentage points |
| **Treatment Purchase Rate** | 91.41% |
| **Control Purchase Rate** | 67.17% |
| **Average Uplift Score** | 0.0133 |
| **Qini Coefficient** | 0.4889 |
| **Optimal Targeting** | Top 50% of customers |
| **Maximum Net ROI** | $5,596 |
| **Total Revenue** | $415,504 |

### Customer Segments

| Segment | Count | % | Description |
|---------|-------|---|-------------|
| **Persuadable** | 183 | 7.3% | High positive uplift — promotion causally drives purchase |
| **Moderate Responder** | 365 | 14.6% | Some uplift — include in broader campaigns |
| **Sure Thing / Neutral** | 1,491 | 59.6% | Would buy anyway — avoid wasting discounts |
| **Do-Not-Disturb** | 461 | 18.4% | Negative uplift — promotion reduces purchase probability |

---

## Key Features

### Engineered Features

| Feature | Description |
|---------|-------------|
| `rfm_score` | Recency-Frequency-Monetary composite (top predictor) |
| `promo_sensitivity_score` | Weighted blend of coupon redemption, past response, discount sensitivity |
| `engagement_score` | Normalized purchase frequency, store visits, category diversity |
| `value_score` | Basket value + income-normalized spending potential |
| `recency_score` | Days since last purchase (inverse normalized) |
| `past_promo_response` | Historical coupon redemption rate |
| `discount_sensitivity` | Campaign diversity in redemption history |

### SHAP Feature Importance (Top 5)

1. **RFM Score** (1.297) — Customer lifecycle value
2. **Recency Days** (0.774) — Time since last purchase
3. **Purchase Frequency** (0.745) — Historical transaction count
4. **Weekend Shopper** (0.493) — Temporal shopping pattern
5. **Value Score** (0.473) — Spending potential composite

---

## Dashboard

The Streamlit dashboard provides 5 interactive tabs:

| Tab | Content |
|-----|---------|
| **◈ Overview** | KPIs, uplift distribution, segment pie chart, uplift/Qini curves, feature importance |
| **⬡ Segments** | 4-segment cards, average uplift by segment, treatment vs control comparison |
| **◎ Model Performance** | Baseline AUC scores, uplift curve, feature ranking |
| **◆ ROI Simulator** | Interactive slider (5-100%), net ROI curve, revenue vs cost breakdown |
| **● Cluster Analysis** | PCA scatter plots (K-Means clusters vs uplift segments) |

---

## Technical Details

### Causal Inference Methodology

**X-Learner** (Künzel et al., 2019):

```
Stage 1: Fit μ₁(x) = E[Y | T=1, X=x]  and  μ₀(x) = E[Y | T=0, X=x]
Stage 2: Compute imputed effects
         D¹ = Y¹ - μ₀(X¹)  (treated group)
         D⁰ = μ₁(X⁰) - Y⁰  (control group)
Stage 3: Fit τ₁(x) = E[D¹ | X=x]  and  τ₀(x) = E[D⁰ | X=x]
Stage 4: Combine: τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x)
         where g(x) = propensity score P(T=1 | X=x)
```

### Bug Fixes & Improvements

This pipeline includes extensive bug fixes over the original implementation:

| Issue | Fix |
|-------|-----|
| Duplicate column rename causing KeyError | Consolidated single rename block |
| `recency_days` computed twice | Removed duplicate computation |
| Treatment column filtered before existence check | Added safe fallback to first campaign type |
| Inline propensity model overwrote Step 3 | Removed duplicate fit, delegated to Step 3 |
| `household_size` float/integer mismatch | Added explicit `.astype(float)` cast |
| Division by zero in normalization | Guarded with `replace(0, 1)` |
| `np.corrcoef` crash on all-NaN `true_uplift` | Added NaN guard with informative message |
| X-Learner `tau0` sign inversion | Corrected to `Y_control - m1.predict` (proper ATE) |
| Qini curve division by zero | Added safe conditional logic |
| Random sampling index overflow | Fixed with local RNG seed |
| `top_customers` missing `customer_id` | Added fallback column detection |

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
ipykernel
papermill
xgboost
shap
joblib
kagglehub
plotly
streamlit
```

See `requirements.txt` for full dependency list.

## License

MIT License — see repository for details.

*Built with XGBoost, scikit-learn, SHAP, and Streamlit.*
