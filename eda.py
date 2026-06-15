"""
Exploratory Data Analysis — Android App Success Prediction
Dataset: 7,823 releases across 1,341 apps from the Google Play Store
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data Loading & Cleaning
# ---------------------------------------------------------------------------

def load_dataset(filepath: str = "data/android_apps.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse installs: "1,000,000+" → 1000000
    if "installs" in df.columns:
        df["installs"] = (
            df["installs"]
            .astype(str)
            .str.replace(r"[+,]", "", regex=True)
            .str.strip()
            .replace("Free", "0")
            .pipe(pd.to_numeric, errors="coerce")
        )

    # Parse size: "19M" → 19.0, "512k" → 0.5
    if "size" in df.columns:
        df["size_mb"] = df["size"].apply(_parse_size)

    # Parse price: "$2.99" → 2.99
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str).str.replace("$", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

    # Rating → float, clamp to [1, 5]
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].between(1.0, 5.0) | df["rating"].isna()]

    # Reviews → int
    if "reviews" in df.columns:
        df["reviews"] = pd.to_numeric(
            df["reviews"].astype(str).str.replace(",", ""), errors="coerce"
        )

    # Drop duplicates and rows with no installs or rating
    df = df.drop_duplicates(subset=["app", "current_ver"] if "current_ver" in df.columns else ["app"])
    df = df.dropna(subset=["rating", "installs"])

    # Binary success label: rating ≥ 4.0 AND installs ≥ 100,000
    df["is_success"] = ((df["rating"] >= 4.0) & (df["installs"] >= 100_000)).astype(int)

    return df.reset_index(drop=True)


def _parse_size(val: str) -> float:
    val = str(val).strip()
    if val.endswith("M"):
        return float(val[:-1])
    if val.endswith("k"):
        return float(val[:-1]) / 1024
    return np.nan


# ---------------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------------

def describe_dataset(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Dataset Shape   : {df.shape}")
    print(f"Total Apps      : {df['app'].nunique() if 'app' in df.columns else 'N/A'}")
    print(f"Success Rate    : {df['is_success'].mean():.2%}")
    print(f"Missing Values  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print("\nNumerical Summary:")
    print(df[["rating", "reviews", "installs", "price"]].describe().round(2))
    print("=" * 60)


def category_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Success rate, avg rating, and install count per category."""
    return (
        df.groupby("category")
        .agg(
            total_apps=("app", "count"),
            success_rate=("is_success", "mean"),
            avg_rating=("rating", "mean"),
            avg_reviews=("reviews", "mean"),
            total_installs=("installs", "sum"),
            median_installs=("installs", "median"),
        )
        .round(3)
        .sort_values("success_rate", ascending=False)
        .reset_index()
    )


def free_vs_paid(df: pd.DataFrame) -> pd.DataFrame:
    """Compare free vs paid apps on key metrics."""
    df = df.copy()
    df["type"] = df["price"].apply(lambda p: "Free" if p == 0 else "Paid")
    return (
        df.groupby("type")
        .agg(
            count=("app", "count"),
            success_rate=("is_success", "mean"),
            avg_rating=("rating", "mean"),
            avg_installs=("installs", "mean"),
        )
        .round(3)
    )


def rating_distribution(df: pd.DataFrame) -> pd.Series:
    bins = [1, 2, 3, 3.5, 4.0, 4.5, 5.01]
    labels = ["1-2", "2-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
    return pd.cut(df["rating"], bins=bins, labels=labels, right=False).value_counts().sort_index()


def install_tier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["install_tier"] = pd.cut(
        df["installs"],
        bins=[0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, float("inf")],
        labels=["<1K", "1K-10K", "10K-100K", "100K-1M", "1M-10M", "10M+"],
        right=False,
    )
    return (
        df.groupby("install_tier", observed=True)
        .agg(count=("app", "count"), success_rate=("is_success", "mean"), avg_rating=("rating", "mean"))
        .round(3)
    )


def content_rating_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if "content_rating" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("content_rating")
        .agg(count=("app", "count"), success_rate=("is_success", "mean"), avg_rating=("rating", "mean"))
        .round(3)
        .sort_values("success_rate", ascending=False)
    )


def top_apps(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.sort_values(["rating", "installs"], ascending=False)
        [["app", "category", "rating", "reviews", "installs", "price", "is_success"]]
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Correlation & Feature Insights
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["rating", "reviews", "installs", "size_mb", "price", "is_success"]
    available = [c for c in numeric_cols if c in df.columns]
    return df[available].corr().round(3)


def feature_importance_proxy(df: pd.DataFrame) -> pd.Series:
    """Point-biserial correlation of each feature with is_success."""
    numeric_feats = ["rating", "reviews", "installs", "size_mb", "price"]
    available = [f for f in numeric_feats if f in df.columns]
    correlations = {}
    for feat in available:
        subset = df[[feat, "is_success"]].dropna()
        r, p = stats.pointbiserialr(subset["is_success"], subset[feat])
        correlations[feat] = round(abs(r), 4)
    return pd.Series(correlations).sort_values(ascending=False)


if __name__ == "__main__":
    # ── Generate synthetic dataset ──────────────────────────────────────────
    np.random.seed(42)
    n = 7823

    categories = ["GAME", "FINANCE", "HEALTH_AND_FITNESS", "EDUCATION",
                  "SHOPPING", "SOCIAL", "TOOLS", "MUSIC_AND_AUDIO", "TRAVEL_AND_LOCAL"]
    content_ratings = ["Everyone", "Teen", "Mature 17+", "Everyone 10+"]

    df_raw = pd.DataFrame({
        "app":            [f"App_{i:05d}" for i in range(n)],
        "category":       np.random.choice(categories, n,
                              p=[0.18, 0.12, 0.13, 0.10, 0.13, 0.10, 0.09, 0.08, 0.07]),
        "rating":         np.clip(np.random.normal(4.1, 0.6, n), 1.0, 5.0).round(1),
        "reviews":        np.random.lognormal(8, 2, n).astype(int),
        "installs":       np.random.choice(
                              [1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
                              n, p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.07, 0.03]),
        "price":          np.random.choice([0.0, 0.99, 1.99, 2.99, 4.99], n, p=[0.80, 0.08, 0.06, 0.04, 0.02]),
        "size":           [f"{np.random.randint(2, 120)}M" for _ in range(n)],
        "content_rating": np.random.choice(content_ratings, n),
        "current_ver":    [f"1.{np.random.randint(0,20)}.{np.random.randint(0,9)}" for _ in range(n)],
    })

    df = clean(df_raw)

    describe_dataset(df)

    print("\nCategory Analysis:")
    print(category_analysis(df).head(5).to_string(index=False))

    print("\nFree vs Paid:")
    print(free_vs_paid(df).to_string())

    print("\nInstall Tier Analysis:")
    print(install_tier_analysis(df).to_string())

    print("\nFeature Importance (Point-Biserial r):")
    print(feature_importance_proxy(df))

    print("\nTop 5 Apps:")
    print(top_apps(df, 5).to_string(index=False))
