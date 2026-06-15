"""
Predictive ML models for Android App Success Prediction.
XGBoost and Random Forest with full evaluation and comparison.
"""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

XGB_PATH = "models/xgboost_model.pkl"
RF_PATH  = "models/random_forest_model.pkl"
ENC_PATH = "models/label_encoders.pkl"

CATEGORICAL_FEATURES = ["category", "content_rating", "type"]
NUMERICAL_FEATURES   = ["rating", "reviews_log", "installs_log", "size_mb", "price", "price_flag"]
ALL_FEATURES         = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET               = "is_success"


# ---------------------------------------------------------------------------
# Feature Engineering for Modelling
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()

    # Log-transform skewed features
    df["reviews_log"]  = np.log1p(df["reviews"].fillna(0))
    df["installs_log"] = np.log1p(df["installs"].fillna(0))
    df["price_flag"]   = (df["price"] > 0).astype(int)
    df["type"]         = df["price"].apply(lambda p: "Free" if p == 0 else "Paid")
    df["size_mb"]      = df["size_mb"].fillna(df["size_mb"].median())

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    X = df[ALL_FEATURES].values
    y = df[TARGET].values if TARGET in df.columns else None
    return X, y, encoders


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.08,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.05,
    "reg_lambda":       1.0,
    "eval_metric":      "auc",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           -1,
}

def build_xgboost() -> XGBClassifier:
    return XGBClassifier(**XGBOOST_PARAMS)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "class_weight":  "balanced",
    "random_state":  42,
    "n_jobs":        -1,
}

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame):
    X, y, encoders = prepare_features(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost...")
    xgb_model = build_xgboost()
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("Training Random Forest...")
    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)

    return xgb_model, rf_model, encoders, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Failure", "Success"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def compare_models(xgb_metrics: dict, rf_metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([xgb_metrics, rf_metrics]).set_index("model")


def cross_validate_xgb(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    model = build_xgboost()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        metric: {"mean": results[f"test_{metric}"].mean(),
                 "std":  results[f"test_{metric}"].std()}
        for metric in scoring
    }


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(xgb_model, rf_model) -> pd.DataFrame:
    xgb_imp = xgb_model.feature_importances_
    rf_imp  = rf_model.feature_importances_
    return pd.DataFrame({
        "feature":    ALL_FEATURES,
        "xgboost":    xgb_imp.round(4),
        "random_forest": rf_imp.round(4),
        "avg":        ((xgb_imp + rf_imp) / 2).round(4),
    }).sort_values("avg", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_app_success(xgb_model, encoders: dict, app: dict) -> dict:
    df = pd.DataFrame([app])
    X, _, _ = prepare_features(df, encoders=encoders, fit=False)
    prob = xgb_model.predict_proba(X)[0][1]
    label = "SUCCESS" if prob >= 0.5 else "AT_RISK"
    tier = ("High Potential" if prob >= 0.75 else
            "Moderate Potential" if prob >= 0.5 else
            "Low Potential")
    return {"success_probability": round(float(prob), 4), "label": label, "tier": tier}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(xgb_model, rf_model, encoders) -> None:
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, XGB_PATH)
    joblib.dump(rf_model, RF_PATH)
    joblib.dump(encoders, ENC_PATH)
    print(f"XGBoost saved → {XGB_PATH}")
    print(f"RandomForest  → {RF_PATH}")


def load_models():
    return joblib.load(XGB_PATH), joblib.load(RF_PATH), joblib.load(ENC_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from eda import load_dataset, clean

    np.random.seed(42)
    n = 7823

    categories = ["GAME", "FINANCE", "HEALTH_AND_FITNESS", "EDUCATION",
                  "SHOPPING", "SOCIAL", "TOOLS", "MUSIC_AND_AUDIO"]
    content_ratings = ["Everyone", "Teen", "Mature 17+", "Everyone 10+"]

    df_raw = pd.DataFrame({
        "app":            [f"App_{i:05d}" for i in range(n)],
        "category":       np.random.choice(categories, n),
        "rating":         np.clip(np.random.normal(4.1, 0.6, n), 1.0, 5.0).round(1),
        "reviews":        np.random.lognormal(8, 2, n).astype(int),
        "installs":       np.random.choice(
                              [1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
                              n, p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.07, 0.03]),
        "price":          np.random.choice([0.0, 0.99, 1.99, 2.99, 4.99], n, p=[0.80, 0.08, 0.06, 0.04, 0.02]),
        "size":           [f"{np.random.randint(2, 120)}M" for _ in range(n)],
        "content_rating": np.random.choice(content_ratings, n),
    })
    df = clean(df_raw)

    xgb_model, rf_model, encoders, X_test, y_test = train_models(df)

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")

    print("\n=== Model Comparison ===")
    print(compare_models(xgb_metrics, rf_metrics))

    print("\n=== Feature Importance ===")
    print(get_feature_importance(xgb_model, rf_model).to_string(index=False))

    print("\n=== XGBoost Cross-Validation (5-fold) ===")
    X_all, y_all, _ = prepare_features(df, fit=True)
    cv_results = cross_validate_xgb(X_all, y_all)
    for metric, vals in cv_results.items():
        print(f"  {metric:<15}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    sample_app = {
        "app": "FitTrack Pro", "category": "HEALTH_AND_FITNESS",
        "rating": 4.5, "reviews": 12000, "installs": 500000,
        "price": 0.0, "size": "22M", "content_rating": "Everyone",
    }
    prediction = predict_app_success(xgb_model, encoders, sample_app)
    print(f"\nSample App Prediction: {prediction}")

    save_models(xgb_model, rf_model, encoders)
