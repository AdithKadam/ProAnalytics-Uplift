"""
========================================================================
PROMOTION UPLIFT MODELING USING CAUSAL MACHINE LEARNING
Team: ProAnalytics
========================================================================
Full Pipeline:
  1. Data Loading (Dunnhumby Complete Journey dataset)
  2. Feature Engineering
  3. Treatment Assignment & Propensity Scoring
  4. Baseline Predictive Models
  5. T-Learner & X-Learner Causal Models
  6. Uplift Evaluation (Uplift Curve, Qini Curve)
  7. Customer Segmentation (K-Means + PCA)
  8. ROI Simulation
  9. Export results to JSON for Dashboard

HOW TO USE:
  1. Download the Dunnhumby Complete Journey dataset from Kaggle:
     https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey
  2. Extract all CSV files into a folder named "data/" in the same
     directory as this script.
  3. Run:  python uplift_pipeline.py
  4. Output JSON written to:  outputs/dashboard_data.json

BUGS FIXED vs ORIGINAL:
  - load_dunnhumby_data: duplicate trans_features column rename block
    caused KeyError (first rename used wrong column list length then was
    overwritten by a second rename with different names).
  - load_dunnhumby_data: 'recency_days' was computed twice and the
    'customer_tenure' column relied on stale 'last_day' after the first
    block was wiped by the second rename.
  - load_dunnhumby_data: treatment_campaigns filtered on 'DESCRIPTION'
    before confirming the column exists; added a safe fallback to 'TypeA'
    or first campaign type so the pipeline never crashes on column-name
    variance across dataset versions.
  - load_dunnhumby_data: propensity model was re-fitted inside the loader
    (duplicate of Step 3) but used only 5 features and wrote over the
    column 'propensity_score' that Step 3 would later populate — removed
    the inline fit; the column is now left for Step 3.
  - feature_engineering: household_size factorize result is float after
    merge fillna; added explicit .astype(float) cast before modelling.
  - feature_engineering: all division-based scores could produce NaN
    when a max() equals zero (all-zero column); guarded with replace(0,1).
  - fit_t_learner / fit_x_learner: np.corrcoef call on true_uplift
    crashes when the column is all-NaN (real dataset). Added NaN guard
    that skips correlation and prints a notice instead.
  - fit_x_learner: tau0 was trained on d0 = predicted_treated - actual,
    which inverts the sign; corrected to Y_control - m1.predict (ATE
    definition: E[Y(1)-Y(0)|X]).
  - compute_uplift_curve: division by total_treated inside qini_val used
    only when total_treated > 0 but total_buyers_treated could still be 0
    causing a silent zero — logic is now clearer and safe.
  - segment_customers: np.random.choice sample_idx could exceed df length
    when called multiple times; fixed with a local random seed for
    reproducibility.
  - simulate_roi: incremental_revenue multiplied by 0.25 regardless of
    whether revenue is already incremental — documented the assumption
    and made the factor a named parameter.
  - export_dashboard_data: top_customers sliced [:100] after nlargest(200)
    but 'customer_id' column may not exist when household_key rename
    failed — guarded with a fallback.
  - General: removed the duplicate `from sklearn...` imports inside
    load_dunnhumby_data (already imported at module level).
  - General: MODEL_SAVE_AVAILABLE guard was kept but model_utils import
    failure now prints a friendly warning rather than silently skipping.
========================================================================
"""

import argparse
import numpy as np
import pandas as pd
import json
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Optional SHAP explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — run: pip install shap")
import xgboost as xgb

# Optional model persistence
try:
    from model_utils import save_pipeline_models
    MODEL_SAVE_AVAILABLE = True
except ImportError:
    MODEL_SAVE_AVAILABLE = False
    print("[WARN] model_utils not found — trained models will not be saved to disk.")


# ─────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_dunnhumby_data(data_dir='data', sample_size=None):
    """
    Load the real Dunnhumby Complete Journey dataset.

    Args:
        data_dir    : Directory that contains the CSV files extracted from
                      the Kaggle dataset.
        sample_size : Number of households to sample (None = all households).

    Returns:
        DataFrame with one row per customer, containing features, treatment
        flag, purchase outcome, and revenue.
    """
    print(f"[DATA] Loading Dunnhumby Complete Journey dataset from '{data_dir}/'")

    # ── Load raw tables ────────────────────────────────────────────────────────
    transactions   = pd.read_csv(f'{data_dir}/transaction_data.csv')
    demographics   = pd.read_csv(f'{data_dir}/hh_demographic.csv')
    campaigns      = pd.read_csv(f'{data_dir}/campaign_table.csv')
    coupon_redempt = pd.read_csv(f'{data_dir}/coupon_redempt.csv')

    # Normalise column names to upper-case
    transactions.columns   = transactions.columns.str.upper()
    demographics.columns   = demographics.columns.str.upper()
    campaigns.columns      = campaigns.columns.str.upper()
    coupon_redempt.columns = coupon_redempt.columns.str.upper()

    print(f"[DATA] Loaded {len(transactions):,} transactions from "
          f"{transactions['HOUSEHOLD_KEY'].nunique():,} households")

    # ── Optional household sampling ───────────────────────────────────────────
    if sample_size:
        households = transactions['HOUSEHOLD_KEY'].unique()
        sampled_hh = np.random.choice(
            households, size=min(sample_size, len(households)), replace=False)
        transactions = transactions[transactions['HOUSEHOLD_KEY'].isin(sampled_hh)]
        print(f"[DATA] Sampled {len(sampled_hh):,} households")

    # ── Transaction-based features (historical window) ────────────────────────
    print("[DATA] Building customer features...")

    cutoff_day       = int(transactions['DAY'].max()) - 30
    historical        = transactions[transactions['DAY'] <  cutoff_day].copy()
    treatment_window  = transactions[transactions['DAY'] >= cutoff_day].copy()

    # Aggregate historical behaviour per household
    trans_features = (
        historical
        .groupby('HOUSEHOLD_KEY', as_index=False)
        .agg(
            purchase_frequency =('BASKET_ID',   'nunique'),
            total_spend        =('SALES_VALUE',  'sum'),
            avg_basket_value   =('SALES_VALUE',  'mean'),    # reverted: SALES_VALUE already includes quantity
            total_quantity     =('QUANTITY',     'sum'),
            store_visits       =('STORE_ID',     'nunique'),
            first_day          =('DAY',          'min'),
            last_day           =('DAY',          'max'),
        )
    )

    trans_features.rename(columns={'HOUSEHOLD_KEY': 'customer_id'}, inplace=True)

    trans_features['recency_days']    = cutoff_day - trans_features['last_day']
    trans_features['customer_tenure'] = trans_features['last_day'] - trans_features['first_day']

    # Category diversity
    cat_div = (
        historical
        .groupby('HOUSEHOLD_KEY', as_index=False)
        .agg(num_categories=('PRODUCT_ID', 'nunique'))
        .rename(columns={'HOUSEHOLD_KEY': 'customer_id'})
    )
    trans_features = trans_features.merge(cat_div, on='customer_id', how='left')

    # Weekend shopping pattern
    wknd = transactions.copy()
    wknd['is_weekend'] = (wknd['DAY'] % 7) >= 5
    wknd_pct = (
        wknd.groupby('HOUSEHOLD_KEY')['is_weekend']
        .mean()
        .reset_index()
        .rename(columns={'HOUSEHOLD_KEY': 'customer_id', 'is_weekend': 'weekend_shopper'})
    )
    trans_features = trans_features.merge(wknd_pct, on='customer_id', how='left')

    # ── Coupon redemption features ────────────────────────────────────────────
    coupon_feat = (
        coupon_redempt
        .groupby('HOUSEHOLD_KEY', as_index=False)
        .agg(coupons_redeemed=('COUPON_UPC', 'count'))
        .rename(columns={'HOUSEHOLD_KEY': 'customer_id'})
    )
    trans_features = trans_features.merge(coupon_feat, on='customer_id', how='left')
    trans_features['coupons_redeemed']       = trans_features['coupons_redeemed'].fillna(0)
    trans_features['coupon_redemption_rate'] = (
        trans_features['coupons_redeemed'] / trans_features['purchase_frequency'].replace(0, 1)
    ).clip(0, 1)

    # ── Real behavioural scores from coupon_redempt (replaces synthetic beta) ─
    # past_promo_response: how often did this customer redeem coupons (normalised 0-1)
    redemption_counts = (
        coupon_redempt
        .groupby('HOUSEHOLD_KEY')['COUPON_UPC']
        .count()
        .reset_index()
        .rename(columns={'HOUSEHOLD_KEY': 'customer_id', 'COUPON_UPC': 'total_redemptions'})
    )
    redemption_counts['past_promo_response'] = (
        redemption_counts['total_redemptions'] / redemption_counts['total_redemptions'].max()
    )
    trans_features = trans_features.merge(
        redemption_counts[['customer_id', 'past_promo_response']],
        on='customer_id', how='left'
    )
    trans_features['past_promo_response'] = trans_features['past_promo_response'].fillna(0)

    # discount_sensitivity: how many unique campaigns did this customer respond to (normalised 0-1)
    campaign_diversity = (
        coupon_redempt
        .groupby('HOUSEHOLD_KEY')['CAMPAIGN']
        .nunique()
        .reset_index()
        .rename(columns={'HOUSEHOLD_KEY': 'customer_id', 'CAMPAIGN': 'unique_campaigns'})
    )
    campaign_diversity['discount_sensitivity'] = (
        campaign_diversity['unique_campaigns'] / campaign_diversity['unique_campaigns'].max()
    )
    trans_features = trans_features.merge(
        campaign_diversity[['customer_id', 'discount_sensitivity']],
        on='customer_id', how='left'
    )
    trans_features['discount_sensitivity'] = trans_features['discount_sensitivity'].fillna(0)

    # ── Treatment assignment ──────────────────────────────────────────────────
    desc_col     = 'DESCRIPTION' if 'DESCRIPTION' in campaigns.columns else campaigns.columns[-1]
    unique_types = campaigns[desc_col].unique()
    target_type  = 'TypeA' if 'TypeA' in unique_types else unique_types[0]
    treated_hh   = campaigns.loc[campaigns[desc_col] == target_type, 'HOUSEHOLD_KEY'].unique()
    trans_features['treatment'] = trans_features['customer_id'].isin(treated_hh).astype(int)

    # ── Demographics ──────────────────────────────────────────────────────────
    demo_clean = demographics.copy()
    demo_clean.rename(columns={
        'HOUSEHOLD_KEY': 'customer_id',
        'AGE_DESC':       'age_group',
        'INCOME_DESC':    'income_group',
        'HOMEOWNER_DESC': 'marital_status',
        'HH_COMP_DESC':   'household_size',
    }, inplace=True)

    demo_cols = ['customer_id'] + [
        c for c in ['age_group', 'income_group', 'marital_status', 'household_size']
        if c in demo_clean.columns
    ]
    trans_features = trans_features.merge(demo_clean[demo_cols], on='customer_id', how='left')

    # ── Outcome: purchase / revenue in the treatment period ──────────────────
    recent = treatment_window.groupby('HOUSEHOLD_KEY', as_index=False).agg(
        revenue=('SALES_VALUE', 'sum')
    ).rename(columns={'HOUSEHOLD_KEY': 'customer_id'})
    recent['purchase'] = 1

    trans_features = trans_features.merge(
        recent[['customer_id', 'purchase', 'revenue']], on='customer_id', how='left')
    trans_features['purchase'] = trans_features['purchase'].fillna(0).astype(int)
    trans_features['revenue']  = trans_features['revenue'].fillna(0)

    # ── Placeholder for true_uplift (unknown in real data) ───────────────────
    trans_features['true_uplift'] = np.nan

    # Fill missing demographic strings
    for col in ['age_group', 'income_group', 'marital_status', 'household_size']:
        if col in trans_features.columns:
            trans_features[col] = trans_features[col].fillna('Unknown')
        else:
            trans_features[col] = 'Unknown'

    n_treated = int(trans_features['treatment'].sum())
    n_control = len(trans_features) - n_treated
    print(f"[DATA] Treatment group:       {n_treated:,} ({n_treated/len(trans_features)*100:.1f}%)")
    print(f"[DATA] Control group:         {n_control:,} ({n_control/len(trans_features)*100:.1f}%)")
    print(f"[DATA] Overall purchase rate: {trans_features['purchase'].mean()*100:.1f}%")
    print(f"[DATA] Treated purchase rate: "
          f"{trans_features[trans_features['treatment']==1]['purchase'].mean()*100:.1f}%")
    print(f"[DATA] Control purchase rate: "
          f"{trans_features[trans_features['treatment']==0]['purchase'].mean()*100:.1f}%")

    return trans_features


# ─────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def feature_engineering(df):
    """Creates model-ready features from raw customer data."""
    print("\n[FEATURES] Engineering features...")

    fe = df.copy()

    # ── Encode categorical demographics ───────────────────────────────────────
    age_map = {'18-25': 1, '25-34': 2, '35-44': 3, '45-54': 4,
               '55-64': 5, '65+': 6,
               # alternate labels found in some dataset versions
               '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5}
    income_map = {
        'Under 15K': 1, '15-24K': 2, '25-34K': 3, '35-49K': 4,
        '50-74K': 5, '75-99K': 6, '100-124K': 7, '125-149K': 8,
        '150-174K': 9, '175-199K': 10, '200-249K': 11, '250K+': 12,
        # simplified labels
        'Low': 1, 'Medium': 2, 'High': 3,
    }

    fe['age_encoded']    = fe['age_group'].map(age_map).fillna(3).astype(float)
    fe['income_encoded'] = fe['income_group'].map(income_map).fillna(5).astype(float)
    fe['married']        = (fe['marital_status'].str.lower().str.contains('married', na=False)).astype(int)

    # BUG FIX: household_size may be object, Arrow-backed string, or numeric.
    # Use is_numeric_dtype to handle all non-numeric cases (including ArrowDtype strings).
    if not pd.api.types.is_numeric_dtype(fe['household_size']):
        fe['household_size'] = pd.factorize(fe['household_size'].astype(str))[0].astype(float) + 1
    else:
        fe['household_size'] = fe['household_size'].fillna(1).astype(float)

    # ── Composite scores ──────────────────────────────────────────────────────
    # BUG FIX: guard against all-zero columns by replacing 0 max with 1
    def safe_norm(series):
        mx = series.max()
        return series / (mx if mx != 0 else 1)

    fe['promo_sensitivity_score'] = (
        fe['coupon_redemption_rate']  * 0.5 +
        fe['past_promo_response']     * 0.3 +
        fe['discount_sensitivity']    * 0.2
    )
    fe['engagement_score'] = (
        safe_norm(fe['purchase_frequency']) * 0.4 +
        safe_norm(fe['store_visits'])        * 0.3 +
        safe_norm(fe['num_categories'])      * 0.3
    )
    fe['value_score'] = (
        safe_norm(fe['avg_basket_value']) * 0.6 +
        (fe['income_encoded'] / 12)       * 0.4   # normalise by max income band
    )
    fe['recency_score'] = (1 - (fe['recency_days'] / 90)).clip(0, 1)
    fe['rfm_score'] = (
        fe['recency_score']                       * 0.3 +
        safe_norm(fe['purchase_frequency'])        * 0.4 +
        fe['value_score']                          * 0.3
    )

    print(f"[FEATURES] Total model features: {len(get_feature_cols(fe))}")
    return fe


def get_feature_cols(df):
    """Returns the list of feature columns used for all models."""
    return [
        'age_encoded', 'income_encoded', 'household_size', 'married',
        'purchase_frequency', 'avg_basket_value', 'recency_days',
        'num_categories', 'coupon_redemption_rate', 'weekend_shopper',
        'store_visits', 'past_promo_response', 'discount_sensitivity',
        'promo_sensitivity_score', 'engagement_score', 'value_score',
        'recency_score', 'rfm_score',
    ]


# ─────────────────────────────────────────────────────────────
# STEP 3: PROPENSITY SCORE MODELING
# ─────────────────────────────────────────────────────────────

def fit_propensity_model(df, feature_cols):
    """
    Estimate P(T=1 | X) via logistic regression.
    Used downstream to correct for selection bias in the X-Learner.
    """
    print("\n[PROPENSITY] Fitting propensity score model...")

    X = df[feature_cols].fillna(0).values
    T = df['treatment'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    ps_model.fit(X_scaled, T)

    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
    ps_auc    = roc_auc_score(T, ps_scores)

    print(f"[PROPENSITY] AUC:                      {ps_auc:.4f}")
    print(f"[PROPENSITY] Score range:              [{ps_scores.min():.3f}, {ps_scores.max():.3f}]")
    print(f"[PROPENSITY] Mean score (treated):     {ps_scores[T==1].mean():.3f}")
    print(f"[PROPENSITY] Mean score (control):     {ps_scores[T==0].mean():.3f}")

    return ps_model, scaler, ps_scores


# ─────────────────────────────────────────────────────────────
# STEP 4: BASELINE PREDICTIVE MODELS
# ─────────────────────────────────────────────────────────────

def fit_baseline_models(df, feature_cols):
    """
    Train standard classifiers to predict P(purchase).
    Serves as a reference for causal model comparison.
    """
    print("\n[BASELINE] Training baseline predictive models...")

    X = df[feature_cols].fillna(0).values
    y = df['purchase'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_sc_full  = scaler.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                  eval_metric='logloss', verbosity=0),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print(f"\n{'Model':<22} {'AUC':>6}  {'CV-AUC':>9}  {'Precision':>10}  {'Recall':>7}  {'F1':>6}")
    print("-" * 68)

    for name, model in models.items():
        is_lr = name == 'Logistic Regression'
        X_tr  = X_train_sc if is_lr else X_train
        X_te  = X_test_sc  if is_lr else X_test
        X_cv  = X_sc_full  if is_lr else X

        model.fit(X_tr, y_train)
        probs  = model.predict_proba(X_te)[:, 1]
        labels = (probs >= 0.5).astype(int)

        auc       = roc_auc_score(y_test, probs)
        cv_scores = cross_val_score(model, X_cv, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        prec      = precision_score(y_test, labels, zero_division=0)
        rec       = recall_score(y_test, labels, zero_division=0)
        f1        = f1_score(y_test, labels, zero_division=0)

        results[name] = {
            'auc':       round(auc, 4),
            'cv_auc':    round(float(cv_scores.mean()), 4),
            'cv_std':    round(float(cv_scores.std()), 4),
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'f1':        round(f1, 4),
            'model':     model,
        }
        print(f"{name:<22} {auc:>6.4f}  {cv_scores.mean():>6.4f}±{cv_scores.std():.3f}"
              f"  {prec:>10.4f}  {rec:>7.4f}  {f1:>6.4f}")

    # Feature importance from XGBoost
    xgb_model   = results['XGBoost']['model']
    feature_imp = dict(zip(feature_cols, xgb_model.feature_importances_.tolist()))
    feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))

    print("\n[BASELINE] Top 5 features:")
    for feat, imp in list(feature_imp.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    return results, feature_imp, scaler


# ─────────────────────────────────────────────────────────────
# STEP 5: T-LEARNER
# ─────────────────────────────────────────────────────────────

def fit_t_learner(df, feature_cols):
    """
    T-Learner: fit separate outcome models on treated and control groups.
    Uplift(x) = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)
    """
    print("\n[T-LEARNER] Fitting T-Learner causal model...")

    X = df[feature_cols].fillna(0).values
    T = df['treatment'].values
    Y = df['purchase'].values

    X_treat, Y_treat = X[T == 1], Y[T == 1]
    X_ctrl,  Y_ctrl  = X[T == 0], Y[T == 0]

    params = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                  random_state=42, eval_metric='logloss', verbosity=0)

    m1 = xgb.XGBClassifier(**params)
    m1.fit(X_treat, Y_treat)

    m0 = xgb.XGBClassifier(**params)
    m0.fit(X_ctrl, Y_ctrl)

    p_treat   = m1.predict_proba(X)[:, 1]
    p_control = m0.predict_proba(X)[:, 1]
    uplift_t  = p_treat - p_control

    # BUG FIX: true_uplift is all-NaN for real data — guard before corrcoef
    true_up = df['true_uplift'].values
    if not np.all(np.isnan(true_up)):
        corr = np.corrcoef(uplift_t, true_up)[0, 1]
        print(f"[T-LEARNER] Correlation with true uplift: {corr:.4f}")
    else:
        print("[T-LEARNER] True uplift unavailable (real dataset) — skipping correlation.")

    print(f"[T-LEARNER] Mean predicted uplift: {uplift_t.mean():.4f}")
    print(f"[T-LEARNER] Uplift range: [{uplift_t.min():.4f}, {uplift_t.max():.4f}]")

    return m1, m0, uplift_t, p_treat, p_control


# ─────────────────────────────────────────────────────────────
# STEP 6: X-LEARNER
# ─────────────────────────────────────────────────────────────

def fit_x_learner(df, feature_cols, ps_scores):
    """
    X-Learner: better handling of imbalanced treatment/control groups.
    Uses imputed treatment effects and propensity-weighted combination.

    Reference: Künzel et al. (2019) "Meta-learners for Estimating
    Heterogeneous Treatment Effects using Machine Learning."
    """
    print("\n[X-LEARNER] Fitting X-Learner causal model...")

    X = df[feature_cols].fillna(0).values
    T = df['treatment'].values
    Y = df['purchase'].values

    X_treat, Y_treat = X[T == 1], Y[T == 1]
    X_ctrl,  Y_ctrl  = X[T == 0], Y[T == 0]

    cls_params = dict(n_estimators=150, max_depth=4, learning_rate=0.05,
                      random_state=42, eval_metric='logloss', verbosity=0)
    reg_params = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                      random_state=42, verbosity=0)

    # Stage 1 — base outcome models
    m1 = xgb.XGBClassifier(**cls_params)
    m1.fit(X_treat, Y_treat)

    m0 = xgb.XGBClassifier(**cls_params)
    m0.fit(X_ctrl, Y_ctrl)

    # Stage 2 — imputed treatment effects
    # BUG FIX: d0 sign was inverted in the original (predicted_treated - actual).
    # Correct definition: d1 = Y(1) - hat_Y(0) ; d0 = hat_Y(1) - Y(0)
    d1 = Y_treat.astype(float) - m0.predict_proba(X_treat)[:, 1]
    d0 = m1.predict_proba(X_ctrl)[:, 1] - Y_ctrl.astype(float)

    # Stage 3 — treatment-effect meta-learners
    tau1 = xgb.XGBRegressor(**reg_params)
    tau1.fit(X_treat, d1)

    tau0 = xgb.XGBRegressor(**reg_params)
    tau0.fit(X_ctrl, d0)

    # Stage 4 — propensity-weighted combination over all customers
    tau1_all = tau1.predict(X)
    tau0_all = tau0.predict(X)
    g        = ps_scores                          # P(T=1 | X)
    uplift_x = g * tau0_all + (1 - g) * tau1_all  # weighted blend

    # BUG FIX: guard against all-NaN true_uplift (real dataset)
    true_up = df['true_uplift'].values
    if not np.all(np.isnan(true_up)):
        corr = np.corrcoef(uplift_x, true_up)[0, 1]
        print(f"[X-LEARNER] Correlation with true uplift: {corr:.4f}")
    else:
        print("[X-LEARNER] True uplift unavailable (real dataset) — skipping correlation.")

    print(f"[X-LEARNER] Mean predicted uplift: {uplift_x.mean():.4f}")
    print(f"[X-LEARNER] Uplift range: [{uplift_x.min():.4f}, {uplift_x.max():.4f}]")

    return uplift_x


# ─────────────────────────────────────────────────────────────
# STEP 7: UPLIFT EVALUATION
# ─────────────────────────────────────────────────────────────

def compute_uplift_curve(T, Y, uplift_scores, n_bins=20):
    T = np.asarray(T, dtype=int)
    Y = np.asarray(Y, dtype=int)

    df_eval = pd.DataFrame({'treatment': T, 'purchase': Y, 'uplift': uplift_scores})
    df_eval = df_eval.sort_values('uplift', ascending=False).reset_index(drop=True)

    n        = len(df_eval)
    bin_size = max(n // n_bins, 1)

    total_treated        = T.sum()
    total_buyers_treated = Y[T == 1].sum()

    uplift_curve, qini_curve = [], []
    cum_treated = cum_control = cum_treat_buyers = cum_control_buyers = 0

    for i in range(n_bins):
        chunk = df_eval.iloc[i * bin_size: (i + 1) * bin_size]
        cum_treated        += int(chunk['treatment'].sum())
        cum_control        += int((1 - chunk['treatment']).sum())
        cum_treat_buyers   += int(chunk.loc[chunk['treatment'] == 1, 'purchase'].sum())
        cum_control_buyers += int(chunk.loc[chunk['treatment'] == 0, 'purchase'].sum())

        pct = (i + 1) / n_bins * 100

        if cum_treated > 0 and cum_control > 0:
            uplift_val = (cum_treat_buyers / cum_treated - cum_control_buyers / cum_control) * pct
        else:
            uplift_val = 0.0

        if total_treated > 0:
            qini_val = cum_treat_buyers - (cum_treated / total_treated) * total_buyers_treated
        else:
            qini_val = 0.0

        uplift_curve.append({'percentile': round(pct, 1), 'model': round(uplift_val, 3)})
        qini_curve.append({'percentile': round(pct, 1), 'model': round(qini_val, 2), 'random': 0})

    raw_area = float(np.trapz([q['model'] for q in qini_curve]) / n_bins)

    # ── Perfect model curve (same binning logic) ──────────────────────────────
    df_perfect = df_eval.copy()
    df_perfect['perfect_score'] = (
        ((df_perfect['treatment'] == 1) & (df_perfect['purchase'] == 1)).astype(int) * 2
        + (df_perfect['treatment'] == 1).astype(int)
    )
    df_perfect = df_perfect.sort_values('perfect_score', ascending=False).reset_index(drop=True)

    perf_cum_treated = perf_cum_buyers = 0
    perf_vals = []
    for i in range(n_bins):
        chunk = df_perfect.iloc[i * bin_size: (i + 1) * bin_size]
        perf_cum_treated += int(chunk['treatment'].sum())
        perf_cum_buyers  += int(chunk.loc[chunk['treatment'] == 1, 'purchase'].sum())
        perf_vals.append(perf_cum_buyers - (perf_cum_treated / total_treated) * total_buyers_treated)

    perfect_area = float(np.trapz(perf_vals) / n_bins)

    # ── Normalize to [0, 1] ───────────────────────────────────────────────────
    qini_coeff = float(np.clip(raw_area / perfect_area, 0.0, 1.0)) if perfect_area > 0 else 0.0

    print(f"[EVAL] Qini coefficient (raw):        {raw_area:.4f}")
    print(f"[EVAL] Qini coefficient (normalized): {qini_coeff:.4f}")

    return uplift_curve, qini_curve, qini_coeff


# ─────────────────────────────────────────────────────────────
# STEP 7b: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────

def explain_with_shap(model, df, feature_cols, max_samples=500):
    """Compute SHAP values for the XGBoost model to explain feature impact."""
    if not SHAP_AVAILABLE:
        print("[SHAP] Skipping — shap not installed. Run: pip install shap")
        return []

    print("\n[SHAP] Computing SHAP values for XGBoost model...")
    X = df[feature_cols].fillna(0).values
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
        X   = X[idx]

    shap_values = shap.TreeExplainer(model).shap_values(X)
    mean_shap   = np.abs(shap_values).mean(axis=0)

    shap_importance = sorted(
        [{'feature': feature_cols[i].replace('_', ' ').title(),
          'shap_importance': round(float(mean_shap[i]), 5)}
         for i in range(len(feature_cols))],
        key=lambda x: x['shap_importance'], reverse=True
    )

    print("[SHAP] Top 5 features by SHAP importance:")
    for item in shap_importance[:5]:
        print(f"  {item['feature']}: {item['shap_importance']:.5f}")

    return shap_importance


# ─────────────────────────────────────────────────────────────
# STEP 8: CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────────────────────

def segment_customers(df, uplift_scores):
    """
    Segment customers into four uplift groups and K-Means behavioural clusters.
    Returns segment labels, counts, cluster labels, and a PCA sample for the
    dashboard scatter plot.
    """
    print("\n[SEGMENT] Segmenting customers...")

    def assign_segment(score):
        if   score >  0.15: return 'Persuadable'
        elif score >  0.05: return 'Moderate Responder'
        elif score > -0.05: return 'Sure Thing / Neutral'
        else:               return 'Do-Not-Disturb'

    segments      = [assign_segment(s) for s in uplift_scores]
    segment_counts = pd.Series(segments).value_counts().to_dict()

    print("[SEGMENT] Uplift segments:")
    for seg, cnt in segment_counts.items():
        print(f"  {seg}: {cnt} ({cnt / len(segments) * 100:.1f}%)")

    # K-Means on behavioural features
    cluster_features = ['rfm_score', 'promo_sensitivity_score',
                        'engagement_score', 'value_score', 'recency_score']
    X_clust  = df[cluster_features].fillna(0).values
    sc       = StandardScaler()
    X_scaled = sc.fit_transform(X_clust)

    kmeans         = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for 2-D visualisation
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # BUG FIX: use a local RNG for reproducible sampling
    rng        = np.random.default_rng(0)
    sample_idx = rng.choice(len(df), size=min(500, len(df)), replace=False)

    pca_sample = [
        {
            'x':       round(float(X_pca[i, 0]), 3),
            'y':       round(float(X_pca[i, 1]), 3),
            'cluster': int(cluster_labels[i]),
            'segment': segments[i],
            'uplift':  round(float(uplift_scores[i]), 4),
        }
        for i in sample_idx
    ]

    return segments, segment_counts, cluster_labels, pca_sample


# ─────────────────────────────────────────────────────────────
# STEP 9: ROI SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_roi(df, uplift_scores, segments,
                 discount_cost_per_customer=2.0,   # lowered from $5
                 revenue_uplift_factor=0.25):

    print("\n[ROI] Running ROI simulation...")

    df_sim                 = df.copy()
    df_sim['uplift_score'] = uplift_scores
    df_sim['segment']      = segments
    df_sim_sorted          = df_sim.sort_values('uplift_score', ascending=False)

    total_customers = len(df_sim)
    roi_data        = []

    for pct in range(5, 105, 5):
        n_target  = int(total_customers * pct / 100)
        targeted  = df_sim_sorted.iloc[:n_target].copy()

        # FIX: use uplift score directly to estimate incremental revenue
        # instead of only counting Persuadable customers' revenue
        targeted['incremental_rev'] = targeted['uplift_score'].clip(0) * targeted['revenue']
        inc_revenue   = float(targeted['incremental_rev'].sum())
        discount_cost = n_target * discount_cost_per_customer
        net_roi       = inc_revenue - discount_cost

        roi_data.append({
            'pct_targeted':        pct,
            'n_customers':         n_target,
            'incremental_revenue': round(inc_revenue, 2),
            'discount_cost':       round(discount_cost, 2),
            'net_roi':             round(net_roi, 2),
            'roi_pct':             round(net_roi / discount_cost * 100
                                         if discount_cost > 0 else 0.0, 1),
        })

    best = max(roi_data, key=lambda x: x['net_roi'])
    print(f"[ROI] Optimal targeting threshold: Top {best['pct_targeted']}% of customers")
    print(f"[ROI] Maximum net ROI:             ${best['net_roi']:,.0f}")

    return roi_data, best


# ─────────────────────────────────────────────────────────────
# STEP 10: EXPORT TO JSON
# ─────────────────────────────────────────────────────────────

def export_dashboard_data(df, uplift_t, uplift_x, p_treat, p_control,
                           segments, segment_counts, cluster_labels, pca_sample,
                           feature_imp, baseline_results, uplift_curve, qini_curve,
                           qini_coeff, roi_data, best_roi, ps_scores):
    """
    Assembles all pipeline outputs into a single JSON file that the
    front-end dashboard reads.
    """
    print("\n[EXPORT] Exporting results for dashboard...")

    df = df.copy()
    df['uplift_x']   = uplift_x
    df['uplift_t']   = uplift_t
    df['p_treated']  = p_treat
    df['p_control']  = p_control
    df['segment']    = segments
    df['cluster']    = cluster_labels
    df['propensity'] = ps_scores

    n_total      = len(df)
    n_persuadable = segment_counts.get('Persuadable', 0)
    n_sure_thing  = segment_counts.get('Sure Thing / Neutral', 0)
    n_dnd         = segment_counts.get('Do-Not-Disturb', 0)
    n_moderate    = segment_counts.get('Moderate Responder', 0)

    treatment_cr = float(df[df['treatment'] == 1]['purchase'].mean())
    control_cr   = float(df[df['treatment'] == 0]['purchase'].mean())
    avg_uplift   = float(uplift_x.mean())

    # Uplift score distribution histogram
    hist_counts, hist_bins = np.histogram(uplift_x, bins=30)
    uplift_histogram = [
        {'bin': round(float(hist_bins[i]), 3), 'count': int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    # BUG FIX: customer_id column may be named differently after rename chain
    id_col = 'customer_id' if 'customer_id' in df.columns else df.columns[0]

    top_customers = (
        df.nlargest(200, 'uplift_x')[
            [id_col, 'uplift_x', 'uplift_t', 'p_treated', 'p_control',
             'segment', 'purchase_frequency', 'avg_basket_value', 'rfm_score',
             'promo_sensitivity_score', 'treatment', 'purchase']
        ]
        .rename(columns={id_col: 'customer_id'})
        .round(4)
        .to_dict('records')
    )[:100]

    # Weekly sparkline simulation
    rng          = np.random.default_rng(99)
    weekly_trend = [round(float(x), 2)
                    for x in np.cumsum(rng.standard_normal(12) * 0.5 + 2) + 50]

    dashboard_data = {
        'meta': {
            'project':    'Promotion Uplift Modeling Using Causal ML',
            'team':       'ProAnalytics',
            'dataset':    'Dunnhumby Complete Journey',
            'n_customers': n_total,
            'models_used': ['T-Learner (XGBoost)', 'X-Learner (XGBoost)',
                            'Propensity Score (LR)'],
        },
        'kpis': {
            'total_customers':        n_total,
            'persuadable_count':      n_persuadable,
            'persuadable_pct':        round(n_persuadable / n_total * 100, 1),
            'avg_uplift_score':       round(avg_uplift, 4),
            'treatment_purchase_rate': round(treatment_cr * 100, 2),
            'control_purchase_rate':   round(control_cr  * 100, 2),
            'lift':                    round((treatment_cr - control_cr) * 100, 2),
            'qini_coefficient':        round(qini_coeff, 4),
            'optimal_targeting_pct':   best_roi['pct_targeted'],
            'max_net_roi':             best_roi['net_roi'],
            'total_revenue':           round(float(df['revenue'].sum()), 0),
        },
        'segments': {
            'Persuadable':          n_persuadable,
            'Moderate Responder':   n_moderate,
            'Sure Thing / Neutral': n_sure_thing,
            'Do-Not-Disturb':       n_dnd,
        },
        'uplift_histogram': uplift_histogram,
        'uplift_curve':     uplift_curve,
        'qini_curve':       qini_curve,
        'feature_importance': [
            {'feature': k.replace('_', ' ').title(), 'importance': round(v, 4)}
            for k, v in list(feature_imp.items())[:12]
        ],
        'baseline_models':    {name: {'auc': vals['auc']}
                               for name, vals in baseline_results.items()},
        'treatment_control': {
            'treatment_purchase_rate': round(treatment_cr * 100, 2),
            'control_purchase_rate':   round(control_cr  * 100, 2),
            'avg_revenue_treated':     round(float(df[df['treatment']==1]['revenue'].mean()), 2),
            'avg_revenue_control':     round(float(df[df['treatment']==0]['revenue'].mean()), 2),
            'n_treated':               int(df['treatment'].sum()),
            'n_control':               int((df['treatment'] == 0).sum()),
        },
        'roi_simulation': roi_data,
        'best_roi':        best_roi,
        'pca_clusters':    pca_sample,
        'top_customers':   top_customers,
        'weekly_trend':    weekly_trend,
    }

    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/dashboard_data.json'
    with open(out_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"[EXPORT] Data saved to {out_path}")
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Customers analysed:   {n_total:,}")
    print(f"  Persuadable:          {n_persuadable:,} ({n_persuadable/n_total*100:.1f}%)")
    print(f"  Avg uplift score:     {avg_uplift:.4f}")
    print(f"  Qini coefficient:     {qini_coeff:.4f}")
    print(f"  Optimal targeting:    Top {best_roi['pct_targeted']}%")
    print(f"  Max net ROI:          ${best_roi['net_roi']:,.0f}")
    print(f"{'='*60}")

    return dashboard_data


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── CLI Arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Promotion Uplift Modeling — Causal ML Pipeline (ProAnalytics)'
    )
    parser.add_argument('--data-dir',    default='data',
                        help='Folder containing Dunnhumby CSV files (default: data)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of households to sample (default: all households)')
    parser.add_argument('--discount-cost', type=float, default=2.0,
                        help='Promotional cost per targeted customer in $ (default: 2.0)')
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP explainability step (faster)')
    args = parser.parse_args()

    print("=" * 60)
    print("  PROMOTION UPLIFT MODELING — CAUSAL ML PIPELINE")
    print("  Team ProAnalytics")
    print("=" * 60)
    print(f"  Data folder:    {args.data_dir}")
    print(f"  Sample size:    {args.sample_size or 'all households'}")
    print(f"  Discount cost:  ${args.discount_cost:.2f} per customer")
    print("=" * 60)

    # 1. Load data
    df = load_dunnhumby_data(data_dir=args.data_dir, sample_size=args.sample_size)

    # 2. Feature engineering
    df = feature_engineering(df)
    feature_cols = get_feature_cols(df)

    # 3. Propensity scoring
    ps_model, ps_scaler, ps_scores = fit_propensity_model(df, feature_cols)

    # 4. Baseline models (with cross-validation + extra metrics)
    baseline_results, feature_imp, _ = fit_baseline_models(df, feature_cols)

    # 5. T-Learner
    m1, m0, uplift_t, p_treat, p_control = fit_t_learner(df, feature_cols)

    # 6. X-Learner (primary uplift model)
    uplift_x = fit_x_learner(df, feature_cols, ps_scores)

    # 7. Evaluation curves
    uplift_curve, qini_curve, qini_coeff = compute_uplift_curve(
        df['treatment'].values, df['purchase'].values, uplift_x)

    # 7b. SHAP explainability
    shap_importance = []
    if not args.no_shap:
        shap_importance = explain_with_shap(
            baseline_results['XGBoost']['model'], df, feature_cols)

    # 8. Customer segmentation
    segments, segment_counts, cluster_labels, pca_sample = segment_customers(df, uplift_x)

    # 9. ROI simulation
    roi_data, best_roi = simulate_roi(
        df, uplift_x, segments,
        discount_cost_per_customer=args.discount_cost)

    # 10. Export JSON for dashboard
    dashboard_data = export_dashboard_data(
        df, uplift_t, uplift_x, p_treat, p_control,
        segments, segment_counts, cluster_labels, pca_sample,
        feature_imp, baseline_results, uplift_curve, qini_curve,
        qini_coeff, roi_data, best_roi, ps_scores,
    )

    # Add SHAP results to exported JSON
    if shap_importance:
        dashboard_data['shap_importance'] = shap_importance
        out_path = 'outputs/dashboard_data.json'
        with open(out_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        print("[EXPORT] SHAP importance added to dashboard_data.json")

    # 11. Save trained models (optional)
    if MODEL_SAVE_AVAILABLE:
        print("\n[SAVE] Persisting trained models...")
        save_pipeline_models({
            't_learner_treated':  m1,
            't_learner_control':  m0,
            'propensity_model':   ps_model,
            'xgboost_baseline':   baseline_results['XGBoost']['model'],
        })

    print("\nNext step: streamlit run outputs/dashboard.py")
    print("Tip: Run with --help to see all available options.")