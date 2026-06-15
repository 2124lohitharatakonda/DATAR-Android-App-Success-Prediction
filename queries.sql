-- ============================================================
-- DATAR — Android App Success Prediction
-- SQL Analysis Queries
-- ============================================================

-- ------------------------------------------------------------
-- Schema
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS android_apps (
    id               SERIAL        PRIMARY KEY,
    app              VARCHAR(256)  NOT NULL,
    category         VARCHAR(64),
    rating           NUMERIC(3,1),
    reviews          BIGINT,
    size_mb          NUMERIC(8,2),
    installs         BIGINT,
    type             VARCHAR(10),         -- Free | Paid
    price            NUMERIC(8,2)  DEFAULT 0.00,
    content_rating   VARCHAR(32),
    genres           VARCHAR(128),
    last_updated     DATE,
    current_ver      VARCHAR(32),
    android_ver      VARCHAR(32),
    success_score    NUMERIC(5,4),        -- ML-predicted probability
    is_success       SMALLINT      DEFAULT 0,
    created_at       TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_category ON android_apps(category);
CREATE INDEX IF NOT EXISTS idx_app_rating   ON android_apps(rating DESC);
CREATE INDEX IF NOT EXISTS idx_app_installs ON android_apps(installs DESC);
CREATE INDEX IF NOT EXISTS idx_app_success  ON android_apps(is_success);
CREATE INDEX IF NOT EXISTS idx_app_type     ON android_apps(type);


-- ------------------------------------------------------------
-- Query 1: Top categories by average success score
-- ------------------------------------------------------------

SELECT
    category,
    COUNT(*)                             AS total_apps,
    ROUND(AVG(rating), 2)                AS avg_rating,
    ROUND(AVG(installs)::NUMERIC, 0)     AS avg_installs,
    SUM(installs)                        AS total_installs,
    ROUND(AVG(success_score), 4)         AS avg_success_score,
    ROUND(AVG(is_success::NUMERIC), 3)   AS success_rate
FROM android_apps
WHERE rating IS NOT NULL
GROUP BY category
ORDER BY avg_success_score DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Query 2: Free vs Paid app performance comparison
-- ------------------------------------------------------------

SELECT
    type,
    COUNT(*)                             AS total_apps,
    ROUND(AVG(rating), 2)                AS avg_rating,
    ROUND(AVG(reviews), 0)               AS avg_reviews,
    ROUND(AVG(installs)::NUMERIC, 0)     AS avg_installs,
    ROUND(AVG(success_score), 4)         AS avg_success_score,
    ROUND(AVG(is_success::NUMERIC), 3)   AS success_rate,
    ROUND(MEDIAN(installs)::NUMERIC, 0)  AS median_installs
FROM android_apps
GROUP BY type
ORDER BY avg_success_score DESC;


-- ------------------------------------------------------------
-- Query 3: Rating distribution bucket analysis
-- ------------------------------------------------------------

SELECT
    CASE
        WHEN rating >= 4.5 THEN '4.5 - 5.0 (Excellent)'
        WHEN rating >= 4.0 THEN '4.0 - 4.5 (Very Good)'
        WHEN rating >= 3.5 THEN '3.5 - 4.0 (Good)'
        WHEN rating >= 3.0 THEN '3.0 - 3.5 (Average)'
        ELSE                     'Below 3.0 (Poor)'
    END                                  AS rating_bucket,
    COUNT(*)                             AS app_count,
    ROUND(AVG(installs)::NUMERIC, 0)     AS avg_installs,
    ROUND(AVG(is_success::NUMERIC), 3)   AS success_rate
FROM android_apps
WHERE rating IS NOT NULL
GROUP BY rating_bucket
ORDER BY MIN(rating) DESC;


-- ------------------------------------------------------------
-- Query 4: Top 10 apps predicted to succeed (ML score)
-- ------------------------------------------------------------

SELECT
    app,
    category,
    rating,
    reviews,
    installs,
    type,
    ROUND(success_score * 100, 2)        AS success_probability_pct,
    is_success
FROM android_apps
ORDER BY success_score DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Query 5: Category success rate with hypothesis context
-- ------------------------------------------------------------

SELECT
    category,
    COUNT(*)                                          AS total,
    SUM(is_success)                                   AS successful,
    ROUND(AVG(is_success::NUMERIC) * 100, 2)          AS success_rate_pct,
    ROUND(AVG(rating), 2)                             AS avg_rating,
    ROUND(STDDEV(rating), 3)                          AS rating_stddev,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
          (ORDER BY installs)::NUMERIC, 0)            AS median_installs
FROM android_apps
WHERE rating IS NOT NULL
GROUP BY category
HAVING COUNT(*) >= 10
ORDER BY success_rate_pct DESC;


-- ------------------------------------------------------------
-- Query 6: Install tier breakdown and success correlation
-- ------------------------------------------------------------

SELECT
    CASE
        WHEN installs <       1000   THEN 'Tier 1: < 1K'
        WHEN installs <      10000   THEN 'Tier 2: 1K - 10K'
        WHEN installs <     100000   THEN 'Tier 3: 10K - 100K'
        WHEN installs <   1000000    THEN 'Tier 4: 100K - 1M'
        WHEN installs <  10000000    THEN 'Tier 5: 1M - 10M'
        ELSE                              'Tier 6: 10M+'
    END                                               AS install_tier,
    COUNT(*)                                          AS total_apps,
    ROUND(AVG(rating), 2)                             AS avg_rating,
    ROUND(AVG(is_success::NUMERIC) * 100, 2)          AS success_rate_pct
FROM android_apps
WHERE installs IS NOT NULL
GROUP BY install_tier
ORDER BY MIN(installs);


-- ------------------------------------------------------------
-- Query 7: Pandas-ready export — feature columns for modelling
-- ------------------------------------------------------------

SELECT
    app,
    category,
    COALESCE(rating, 0)                  AS rating,
    COALESCE(reviews, 0)                 AS reviews,
    COALESCE(installs, 0)                AS installs,
    COALESCE(size_mb, 0)                 AS size_mb,
    price,
    type,
    content_rating,
    is_success
FROM android_apps
WHERE rating   IS NOT NULL
  AND installs IS NOT NULL
ORDER BY installs DESC;


-- ------------------------------------------------------------
-- Query 8: Apps with high reviews but low rating (quality gap)
-- ------------------------------------------------------------

SELECT
    app,
    category,
    rating,
    reviews,
    installs,
    success_score,
    (reviews - AVG(reviews) OVER (PARTITION BY category)) AS review_deviation
FROM android_apps
WHERE rating < 3.5
  AND reviews > 10000
ORDER BY reviews DESC
LIMIT 20;
