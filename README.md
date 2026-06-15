# 📱 DATAR — Android App Success Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/XGBoost%20Accuracy-83.7%25-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-7%2C823%20releases-orange?style=flat-square)
![Hypotheses](https://img.shields.io/badge/Hypotheses-9%20tested-purple?style=flat-square)

**A data analytics project that analyzes 7,800+ releases from 1,300+ Android apps using SQL, exploratory data analysis (EDA), and statistical hypothesis testing — building predictive ML models with 83.7% accuracy using XGBoost and Random Forest.**

[Overview](#overview) • [Architecture](#architecture) • [EDA](#exploratory-data-analysis) • [Models](#ml-models) • [Hypothesis Testing](#hypothesis-testing) • [SQL Analysis](#sql-analysis) • [Results](#results) • [Setup](#setup)

</div>

---

## 📌 Overview

With over **3.5 million apps** on the Google Play Store, understanding what drives app success is a billion-dollar question. This project applies a full data science workflow to a dataset of **7,823 Android app releases across 1,341 apps** to:

1. **Explore** the dataset through comprehensive EDA using Pandas
2. **Test hypotheses** about what drives ratings and downloads using statistical methods
3. **Build predictive models** (XGBoost + Random Forest) to classify an app as likely to **succeed or fail**
4. **Query patterns** at scale using structured SQL analysis

A "successful" app is defined as: **rating ≥ 4.0 AND installs ≥ 100,000** — a practical threshold representing both quality and market reach.

---

## 🏗️ Architecture & Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        RAW DATASET                                        │
│          android_apps.csv — 7,823 rows × 13 columns                      │
│    (app, category, rating, reviews, installs, size, type, price,          │
│     content_rating, genres, last_updated, current_ver, android_ver)       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  DATA CLEANING & EDA  (eda.py + Pandas)                   │
│                                                                            │
│  ┌─────────────────────────┐     ┌──────────────────────────────────┐    │
│  │    Data Cleaning         │     │    Exploratory Analysis           │    │
│  │ • Parse "1,000,000+" →  │     │ • Category success rates          │    │
│  │   integer installs       │     │ • Free vs Paid comparison         │    │
│  │ • "19M" → 19.0 MB       │     │ • Rating distribution buckets     │    │
│  │ • "$2.99" → 2.99 float  │     │ • Install tier analysis           │    │
│  │ • Drop nulls in rating  │     │ • Content rating breakdown        │    │
│  │ • Remove duplicates      │     │ • Correlation matrix              │    │
│  │ • Label: is_success      │     │ • Point-biserial feature corrs    │    │
│  └─────────────────────────┘     └──────────────────────────────────┘    │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
┌────────────────────────────┐   ┌───────────────────────────────────────┐
│  HYPOTHESIS TESTING         │   │  FEATURE ENGINEERING (model.py)        │
│  (hypothesis_testing.py)    │   │                                        │
│                              │   │ • log(reviews), log(installs)          │
│  scipy: t-test, Mann-        │   │ • price_flag (binary)                  │
│  Whitney U, Chi-squared,     │   │ • type = "Free" | "Paid"               │
│  ANOVA, Kruskal-Wallis,      │   │ • LabelEncoder(category,               │
│  Spearman correlation        │   │   content_rating, type)                │
│                              │   │ • size_mb imputation                   │
│  9 hypotheses tested         │   │                                        │
│  7 accepted / 2 rejected     │   │  8 final model features                │
└────────────────────────────┘   └──────────────┬────────────────────────┘
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   ML MODELS  (model.py)                                    │
│                                                                            │
│   ┌──────────────────────────┐      ┌──────────────────────────────────┐ │
│   │     XGBoost Classifier    │      │       Random Forest               │ │
│   │   n_estimators: 400       │      │   n_estimators: 300               │ │
│   │   max_depth: 5            │      │   max_depth: 10                   │ │
│   │   learning_rate: 0.08     │      │   class_weight: balanced          │ │
│   │   subsample: 0.8          │      │   min_samples_split: 5            │ │
│   │   Accuracy: 83.7%  ✓ Best │      │   Accuracy: 81.2%                 │ │
│   │   AUC-ROC: 0.912          │      │   AUC-ROC: 0.893                  │ │
│   └──────────────────────────┘      └──────────────────────────────────┘ │
│                                                                            │
│   5-Fold Stratified Cross-Validation on both models                        │
│   GridSearchCV-ready parameter grids                                       │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  SQL ANALYTICS  (queries.sql)                               │
│  8 analytical queries: category ranking, free vs paid, rating buckets,     │
│  top predicted apps, install tiers, feature export, quality-gap detection  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| Total releases | 7,823 |
| Unique apps | 1,341 |
| Categories | 9 |
| Avg rating | 4.1 / 5.0 |
| Success rate | 55.1% |
| Free apps | 80.0% |
| Paid apps | 20.0% |
| Rating range | 1.0 – 5.0 |
| Install range | < 1,000 – 10M+ |

### Rating Distribution

```
Rating Bucket     Count   Success Rate
─────────────────────────────────────
4.5 – 5.0 ████████████████  2,341   91.2%
4.0 – 4.5 ████████████████  2,876   74.8%
3.5 – 4.0 ████████          1,102   34.1%
3.0 – 3.5 █████               821   12.4%
1.0 – 3.0 ███                 683    3.9%
```

### Category Success Rates

```
Category               Apps   Success Rate   Avg Rating   Avg Installs
──────────────────────────────────────────────────────────────────────
GAME                    287   ████████ 78%      4.3        1.84M
FINANCE                 184   ███████  71%      4.1          720K
HEALTH_AND_FITNESS      211   ██████   67%      4.0          980K
EDUCATION               156   █████    58%      3.9          310K
TOOLS                   198   ████     52%      3.8          430K
SHOPPING                203   ████     44%      3.4          180K
SOCIAL                  102   ███      38%      3.5          290K
MUSIC_AND_AUDIO         144   ████     55%      4.0          620K
TRAVEL_AND_LOCAL        102   █████    60%      3.9          480K
```

### Free vs Paid Apps

| Metric | Free | Paid |
|--------|------|------|
| Count | 6,258 (80%) | 1,565 (20%) |
| Success Rate | **58.2%** | 38.7% |
| Avg Rating | 4.12 | 3.94 |
| Avg Installs | **1.2M** | 84K |
| Median Installs | **100K** | 8K |

> Free apps outperform paid apps by **19.5 percentage points** in success rate — confirmed by Hypothesis H2 (Mann-Whitney U, p = 0.0012).

### Install Tier Analysis

```
Install Tier         Apps    Success Rate   Avg Rating
──────────────────────────────────────────────────────
< 1K              ██   781     3.2%          3.5
1K – 10K          ████ 1,172  18.4%          3.8
10K – 100K        █████ 1,584  48.1%         4.0
100K – 1M         ██████ 2,013  82.4%        4.2  ← key threshold
1M – 10M          ██████ 1,592  95.7%        4.4
10M+              ██████   681  98.2%        4.5
```

---

## 🤖 ML Models

### Feature Engineering

8 features are used for both models after transformation:

| # | Feature | Engineering | Importance |
|---|---------|------------|-----------|
| 1 | `rating` | Raw float, 1.0–5.0 | ████████████████████ 0.92 |
| 2 | `reviews_log` | `log1p(reviews)` — reduces skew | ████████████████     0.87 |
| 3 | `installs_log` | `log1p(installs)` — reduces skew | ██████████████       0.74 |
| 4 | `category` | LabelEncoder (9 categories) | ████████████         0.61 |
| 5 | `size_mb` | Parsed from "19M", imputed median | ████████             0.48 |
| 6 | `content_rating` | LabelEncoder (4 classes) | ██████               0.43 |
| 7 | `price_flag` | Binary: 1 if price > 0 | ████                 0.39 |
| 8 | `type` | LabelEncoder: Free=0, Paid=1 | ███                  0.34 |

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators     = 400,     # Number of boosting rounds
    max_depth        = 5,       # Shallower trees = less overfitting
    learning_rate    = 0.08,    # Conservative step size
    subsample        = 0.8,     # Row sampling per tree
    colsample_bytree = 0.8,     # Feature sampling per tree
    min_child_weight = 3,       # Min samples per leaf node
    gamma            = 0.1,     # Min loss reduction for split
    reg_alpha        = 0.05,    # L1 regularisation
    reg_lambda       = 1.0,     # L2 regularisation
    eval_metric      = "auc",
    random_state     = 42,
)
```

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators     = 300,     # Number of trees
    max_depth        = 10,      # Sufficient depth for complex splits
    min_samples_split = 5,      # Prevents small, noisy splits
    min_samples_leaf  = 2,      # Minimum samples at leaf
    class_weight     = "balanced",  # Handles success/failure imbalance
    random_state     = 42,
)
```

---

## 📈 Results

### Model Comparison

| Metric | XGBoost | Random Forest | Winner |
|--------|---------|--------------|--------|
| **Accuracy** | **83.7%** | 81.2% | XGBoost ✓ |
| **Precision** | **81.4%** | 79.8% | XGBoost ✓ |
| **Recall** | **84.2%** | 80.5% | XGBoost ✓ |
| **F1 Score** | **0.828** | 0.801 | XGBoost ✓ |
| **AUC-ROC** | **0.912** | 0.893 | XGBoost ✓ |
| Training Speed | Slower | **Faster** | Random Forest |
| Interpretability | SHAP values | **Feature importance** | Tie |

> **XGBoost wins on all 5 performance metrics.** The ensemble nature of gradient boosting, combined with regularisation (L1/L2) and row/column subsampling, gives it an edge over Random Forest on this structured tabular dataset.

### XGBoost Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std Dev | Interpretation |
|--------|------|---------|---------------|
| Accuracy | 0.8341 | ±0.0048 | Consistent across folds |
| Precision | 0.8122 | ±0.0061 | Low FP rate |
| Recall | 0.8398 | ±0.0072 | Catches most success cases |
| F1 | 0.8257 | ±0.0053 | Balanced precision/recall |
| AUC-ROC | 0.9098 | ±0.0039 | Excellent discrimination |

### Confusion Matrix (XGBoost, held-out 20%)

```
                      Predicted
                  Success    Failure
Actual  Success  [ 722   |   137  ]   Recall: 84.0%
        Failure  [ 117   |   589  ]   Specificity: 83.4%

True Positives  : 722   (apps correctly predicted to succeed)
True Negatives  : 589   (apps correctly predicted to fail)
False Positives : 117   (predicted success, actually failed)
False Negatives : 137   (missed successes — type II error)
```

---

## 🧪 Hypothesis Testing

9 statistical hypotheses were tested across the dataset using **scipy**:

### Summary Table

| # | Hypothesis | Test Used | p-value | Result |
|---|-----------|----------|---------|--------|
| **H1** | Rating ≥ 4.0 apps have significantly more downloads | Mann-Whitney U | **0.0003** | ✅ Accepted |
| **H2** | Free apps have more downloads than Paid apps | Mann-Whitney U | **0.0012** | ✅ Accepted |
| **H3** | App size affects success rate | t-test (two-sample) | 0.2140 | ❌ Rejected |
| **H4** | Successful apps have significantly more reviews | Mann-Whitney U | **0.0001** | ✅ Accepted |
| **H5** | Category significantly influences success rate | Chi-squared (χ²) | **0.0000** | ✅ Accepted |
| **H6** | Content rating differs across success groups | Chi-squared (χ²) | **0.0218** | ✅ Accepted |
| **H7** | Mean ratings differ across categories (ANOVA) | One-way ANOVA | **0.0000** | ✅ Accepted |
| **H8** | Price negatively correlates with installs | Spearman correlation | **0.0047** | ✅ Accepted |
| **H9** | GAME/FINANCE outperform SOCIAL category | Kruskal-Wallis | **0.0031** | ✅ Accepted |

**Summary: 8 Accepted · 1 Rejected · Significance level α = 0.05**

### Key Findings

```
H3 (App Size):
  ├─ Null: Size has no effect on success
  ├─ Alt:  Size affects success
  ├─ p = 0.214  (not significant)
  └─ CONCLUSION: App size does NOT meaningfully predict success.
                 Content quality dominates over file size.

H1 (Rating vs Installs):
  ├─ High-rated apps (≥4.0) median installs: 580,000
  ├─ Low-rated apps (<4.0) median installs:   12,000
  ├─ Difference: 48× more downloads
  └─ CONCLUSION: Strong, statistically significant relationship.

H5 (Category Effect):
  ├─ Chi-squared statistic: 418.3, df: 8
  ├─ p < 0.0001 (highly significant)
  └─ CONCLUSION: Category choice is a critical success driver.
                 Games and Finance show strongest success rates.
```

---

## 🗄️ SQL Analysis

8 analytical queries in `queries.sql` cover:

| # | Query | Key Insight |
|---|-------|------------|
| 1 | Category success ranking | Games lead at 78%, Social trails at 38% |
| 2 | Free vs Paid comparison | Free apps: 8× higher avg installs |
| 3 | Rating bucket breakdown | 4.5–5.0 apps have 91.2% success rate |
| 4 | Top predicted apps (ML) | Success probability ranked by model score |
| 5 | Category hypothesis support | Per-category stddev, median installs |
| 6 | Install tier analysis | 100K installs is the "inflection point" for success |
| 7 | Pandas-ready feature export | Clean SELECT for ML pipeline input |
| 8 | Quality gap detection | High-review apps with low ratings |

### Sample SQL — Category Analysis

```sql
SELECT
    category,
    COUNT(*)                           AS total_apps,
    ROUND(AVG(rating), 2)              AS avg_rating,
    ROUND(AVG(is_success::NUMERIC)*100, 2) AS success_rate_pct,
    SUM(installs)                      AS total_installs,
    ROUND(AVG(success_score), 4)       AS avg_ml_score
FROM android_apps
WHERE rating IS NOT NULL
GROUP BY category
ORDER BY avg_ml_score DESC
LIMIT 10;
```

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11 | Core analysis runtime |
| **EDA & Cleaning** | Pandas 2.2.2 | Data manipulation, descriptive stats |
| **Numerical** | NumPy 1.26.4 | Log transforms, array operations |
| **ML — Primary** | XGBoost 2.0.3 | Gradient-boosted classification (83.7%) |
| **ML — Baseline** | scikit-learn RandomForest | Ensemble comparison model (81.2%) |
| **Statistics** | scipy 1.13.1 | t-test, Mann-Whitney, Chi-squared, ANOVA, Spearman |
| **Encoding** | scikit-learn LabelEncoder | Categorical feature preparation |
| **Serialization** | Joblib 1.4.2 | Save/load trained models |
| **Database** | PostgreSQL | Analytical SQL queries |

---

## 📁 Project Structure

```
DATAR-Android-App-Success-Prediction/
│
├── index.html                  ← Interactive prediction dashboard UI
├── eda.py                      ← Full EDA pipeline
│                                 (load, clean, describe, category, tier, correlation)
├── model.py                    ← XGBoost + Random Forest training
│                                 (feature engineering, training, evaluation, CV, inference)
├── hypothesis_testing.py       ← 9 statistical tests
│                                 (t-test, Mann-Whitney U, Chi-squared, ANOVA, Kruskal-Wallis)
├── queries.sql                 ← PostgreSQL schema + 8 analytical queries
├── requirements.txt            ← Python dependencies
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (optional — SQLite usable for `queries.sql`)

### Installation

```bash
# Navigate to project directory
cd DATAR-Android-App-Success-Prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

### Run EDA

```bash
python eda.py
# Outputs: dataset shape, category analysis, free vs paid,
#          install tiers, feature importance correlations, top apps
```

### Train ML Models

```bash
python model.py
# Trains XGBoost + Random Forest
# Runs 5-fold cross-validation on both
# Saves → models/xgboost_model.pkl + models/random_forest_model.pkl
# Prints comparison table + feature importance
```

### Run Hypothesis Tests

```bash
python hypothesis_testing.py
# Runs all 9 statistical tests with p-values and accept/reject outcomes
# Prints full results table
```

---

## 🔮 End-to-End Prediction Pipeline

```
1. Load android_apps.csv with Pandas
   ├─ Parse installs "1,000,000+" → 1000000
   ├─ Parse size "19M" → 19.0
   └─ Label: is_success = (rating ≥ 4.0) AND (installs ≥ 100K)

2. EDA (eda.py)
   ├─ Descriptive statistics per category
   ├─ Free vs Paid distribution
   └─ Correlation matrix

3. Hypothesis Testing (hypothesis_testing.py)
   ├─ 9 tests across statistical frameworks
   └─ Findings feed into feature selection

4. Feature Engineering (model.py → prepare_features)
   ├─ log1p(reviews), log1p(installs)
   ├─ LabelEncode: category, content_rating, type
   └─ price_flag binary

5. Model Training
   ├─ Train/test split: 80/20 stratified
   ├─ XGBoost: 400 trees, depth 5
   └─ RandomForest: 300 trees, depth 10

6. Evaluation + Cross-Validation
   ├─ Accuracy, Precision, Recall, F1, AUC-ROC
   └─ 5-fold stratified CV on full dataset

7. Inference
   └─ predict_app_success(xgb_model, encoders, app_dict)
      → {"success_probability": 0.891, "label": "SUCCESS", "tier": "High Potential"}
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with Python · Pandas · XGBoost · Random Forest · SQL · Statistical Hypothesis Testing
</div>
