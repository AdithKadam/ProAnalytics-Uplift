# Promotion Uplift Modeling Using Causal Machine Learning
**Team ProAnalytics**

## Overview
This project estimates the true causal impact of retail promotions using 
uplift modeling — identifying which customers buy *because* of a promotion 
vs those who would have bought anyway.

## Dataset
Dunnhumby Complete Journey Dataset

## Models Used
- Propensity Score Modeling (Logistic Regression)
- Baseline: Random Forest, Gradient Boosting, XGBoost
- T-Learner (Causal)
- X-Learner (Causal — Primary)
- K-Means Clustering + PCA

## Setup
```bash
# 1. Clone the repository
git clone https://github.com/hrushh22/ProAnalytics-Uplift.git
cd ProAnalytics-Uplift

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Dunnhumby dataset (required)
pip install kagglehub
python download_data.py

# 4. Run the pipeline
python uplift_pipeline.py
```

## Data
The Dunnhumby Complete Journey dataset is **not included** in this repository due to size constraints (>100MB).

**To get the data:**
1. Run `python download_data.py` (requires Kaggle account)
2. Or manually download from [Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
3. Place CSV files in `data/` folder

## Output
Open `outputs/dashboard.html` in your browser for the interactive dashboard.

## Results
- **5,000 customers analyzed**
- **2,366 persuadable customers (47.3%)**
- **Qini coefficient: 107.10**
- **Interactive dashboard with uplift curves, customer segmentation, and ROI analysis**
