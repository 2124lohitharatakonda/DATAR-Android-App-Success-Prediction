"""
Statistical Hypothesis Testing — Android App Success Prediction
Tests: t-test, Mann-Whitney U, Chi-squared, ANOVA, Kruskal-Wallis
"""

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05


def _result(p_value: float, alpha: float = ALPHA) -> str:
    return "ACCEPTED ✓" if p_value < alpha else "REJECTED ✗"


def _print_test(name: str, stat: float, p: float, extra: str = "") -> dict:
    outcome = _result(p)
    print(f"\n{'─'*55}")
    print(f"Hypothesis : {name}")
    print(f"Statistic  : {stat:.4f}   p-value: {p:.6f}")
    if extra:
        print(f"Detail     : {extra}")
    print(f"Result     : {outcome}  (α = {ALPHA})")
    return {"hypothesis": name, "statistic": round(stat, 4),
            "p_value": round(p, 6), "result": outcome}


# ---------------------------------------------------------------------------
# H1: Higher-rated apps have significantly more downloads
# ---------------------------------------------------------------------------
def h1_rating_vs_installs(df: pd.DataFrame) -> dict:
    high = df[df["rating"] >= 4.0]["installs"].dropna()
    low  = df[df["rating"] <  4.0]["installs"].dropna()
    stat, p = stats.mannwhitneyu(high, low, alternative="greater")
    extra = (f"High-rated median installs: {high.median():,.0f} | "
             f"Low-rated median installs: {low.median():,.0f}")
    return _print_test("H1: Rating ≥ 4.0 apps have more installs", stat, p, extra)


# ---------------------------------------------------------------------------
# H2: Free apps outperform paid apps in download volume
# ---------------------------------------------------------------------------
def h2_free_vs_paid_installs(df: pd.DataFrame) -> dict:
    free = df[df["price"] == 0]["installs"].dropna()
    paid = df[df["price"] >  0]["installs"].dropna()
    stat, p = stats.mannwhitneyu(free, paid, alternative="greater")
    extra = (f"Free median: {free.median():,.0f} | Paid median: {paid.median():,.0f}")
    return _print_test("H2: Free apps have more downloads than Paid apps", stat, p, extra)


# ---------------------------------------------------------------------------
# H3: App size has no significant effect on success rate
# ---------------------------------------------------------------------------
def h3_size_vs_success(df: pd.DataFrame) -> dict:
    success = df[df["is_success"] == 1]["size_mb"].dropna()
    failure = df[df["is_success"] == 0]["size_mb"].dropna()
    stat, p = stats.ttest_ind(success, failure)
    extra = (f"Success mean size: {success.mean():.1f} MB | "
             f"Failure mean size: {failure.mean():.1f} MB")
    return _print_test("H3: App size differs between success and failure groups", stat, p, extra)


# ---------------------------------------------------------------------------
# H4: Review count is significantly higher for successful apps
# ---------------------------------------------------------------------------
def h4_reviews_vs_success(df: pd.DataFrame) -> dict:
    success = df[df["is_success"] == 1]["reviews"].dropna()
    failure = df[df["is_success"] == 0]["reviews"].dropna()
    stat, p = stats.mannwhitneyu(success, failure, alternative="greater")
    extra = (f"Success median reviews: {success.median():,.0f} | "
             f"Failure median reviews: {failure.median():,.0f}")
    return _print_test("H4: Successful apps have more reviews", stat, p, extra)


# ---------------------------------------------------------------------------
# H5: Category significantly influences success rate (Chi-squared)
# ---------------------------------------------------------------------------
def h5_category_vs_success(df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["category"], df["is_success"])
    stat, p, dof, expected = stats.chi2_contingency(ct)
    extra = f"Degrees of freedom: {dof}"
    return _print_test("H5: Category affects success rate (χ²)", stat, p, extra)


# ---------------------------------------------------------------------------
# H6: Content rating distribution differs between success groups
# ---------------------------------------------------------------------------
def h6_content_rating_vs_success(df: pd.DataFrame) -> dict:
    if "content_rating" not in df.columns:
        return {}
    ct = pd.crosstab(df["content_rating"], df["is_success"])
    stat, p, dof, _ = stats.chi2_contingency(ct)
    return _print_test("H6: Content rating differs across success groups (χ²)", stat, p)


# ---------------------------------------------------------------------------
# H7: Rating distributions differ across categories (ANOVA)
# ---------------------------------------------------------------------------
def h7_rating_by_category_anova(df: pd.DataFrame) -> dict:
    groups = [g["rating"].dropna().values for _, g in df.groupby("category")]
    groups = [g for g in groups if len(g) >= 10]
    stat, p = stats.f_oneway(*groups)
    return _print_test("H7: Mean ratings differ across categories (one-way ANOVA)", stat, p)


# ---------------------------------------------------------------------------
# H8: Price correlates negatively with install count
# ---------------------------------------------------------------------------
def h8_price_vs_installs_correlation(df: pd.DataFrame) -> dict:
    subset = df[df["price"] > 0][["price", "installs"]].dropna()
    stat, p = stats.spearmanr(subset["price"], subset["installs"])
    extra = f"Spearman ρ = {stat:.4f}"
    return _print_test("H8: Price negatively correlates with installs (Spearman)", stat, p, extra)


# ---------------------------------------------------------------------------
# H9: Games and Finance apps have higher success rates than Social apps
# ---------------------------------------------------------------------------
def h9_category_trio(df: pd.DataFrame) -> dict:
    games   = df[df["category"] == "GAME"]["is_success"].values
    finance = df[df["category"] == "FINANCE"]["is_success"].values
    social  = df[df["category"] == "SOCIAL"]["is_success"].values

    if len(games) < 5 or len(finance) < 5 or len(social) < 5:
        print("Skipping H9: insufficient category data")
        return {}

    stat, p = stats.kruskal(games, finance, social)
    extra = (f"GAME success: {games.mean():.2%} | "
             f"FINANCE: {finance.mean():.2%} | SOCIAL: {social.mean():.2%}")
    return _print_test("H9: GAME/FINANCE outperform SOCIAL (Kruskal-Wallis)", stat, p, extra)


# ---------------------------------------------------------------------------
# H10: Apps updated more frequently have higher ratings
# ---------------------------------------------------------------------------
def h10_updates_vs_rating(df: pd.DataFrame) -> dict:
    if "android_ver" not in df.columns:
        return {}
    high_updates = df[df["android_ver"] >= "5.0"]["rating"].dropna()
    low_updates  = df[df["android_ver"] <  "5.0"]["rating"].dropna()
    stat, p = stats.mannwhitneyu(high_updates, low_updates, alternative="greater")
    return _print_test("H10: Apps targeting newer Android have higher ratings", stat, p)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  DATAR — Statistical Hypothesis Testing")
    print("=" * 55)

    results = []
    for fn in [h1_rating_vs_installs, h2_free_vs_paid_installs,
               h3_size_vs_success, h4_reviews_vs_success,
               h5_category_vs_success, h6_content_rating_vs_success,
               h7_rating_by_category_anova, h8_price_vs_installs_correlation,
               h9_category_trio]:
        try:
            r = fn(df)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [skipped] {fn.__name__}: {e}")

    summary = pd.DataFrame(results)
    accepted = (summary["result"].str.startswith("ACCEPTED")).sum()
    rejected = (summary["result"].str.startswith("REJECTED")).sum()

    print(f"\n{'=' * 55}")
    print(f"  Summary: {accepted} Accepted / {rejected} Rejected / {len(results)} Total")
    print("=" * 55)
    return summary


if __name__ == "__main__":
    from eda import clean

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
        "price":          np.random.choice([0.0, 0.99, 1.99, 2.99], n, p=[0.80, 0.10, 0.06, 0.04]),
        "size":           [f"{np.random.randint(2, 120)}M" for _ in range(n)],
        "content_rating": np.random.choice(content_ratings, n),
    })

    df = clean(df_raw)
    summary = run_all(df)
    print("\nFull Results Table:")
    print(summary.to_string(index=False))
