-- PHASE:   2 — Data Cleaning & Feature Preprocessing
--
-- PURPOSE:
--   Transforms raw ingested order data into a ML-ready representation by
--   addressing two fundamental preprocessing requirements:
--
--     (a) Missing Value Imputation  — NULL values in days_since_prior_order
--         are structurally meaningful (first-ever order) but mathematically
--         undefined. Imputing with 0.0 preserves this semantics: zero days
--         since a prior order truthfully represents "no prior order exists."
--
--     (b) Categorical Encoding     — order_dow stores integers 0–6, but these
--         are nominal labels, not an ordinal scale. A raw integer fed into a
--         gradient-boosted tree or logistic regression would imply a distance
--         relationship (dow=6 is "3x" dow=2) that does not exist. One-Hot
--         Encoding (OHE) breaks the single column into 7 orthogonal binary
--         flags, giving the model a truthful, distance-free representation.
--
-- IMPLEMENTATION STRATEGY — Database View:
--   Implemented as a VIEW (not a materialized table) so that:
--     1. No data is duplicated on disk — the transformation runs at query time.
--     2. Any upstream fix to the orders table is automatically reflected here.
--     3. The feature engineering pipeline (03_features.sql) can JOIN directly
--        against this view as if it were a physical table.
--   If sub-20ms serving latency becomes a concern at scale, this view can be
--   promoted to a MATERIALIZED VIEW with a single-line change.
--
-- DEPENDENCIES:
--   Requires 01_schema.sql to have been executed successfully.
--   Table required: orders


-- =============================================================================
-- SECTION 1: IDEMPOTENCY GUARD
-- =============================================================================
-- CASCADE ensures that any downstream views or functions that reference
-- v_cleaned_orders are also dropped.

DROP VIEW IF EXISTS v_cleaned_orders CASCADE;


-- =============================================================================
-- SECTION 2: CLEANED ORDERS VIEW DEFINITION
-- =============================================================================

CREATE VIEW v_cleaned_orders AS

SELECT

    -- -------------------------------------------------------------------------
    -- BLOCK A: PASSTHROUGH COLUMNS (unchanged from source)
    -- -------------------------------------------------------------------------
    -- These columns are structurally clean and require no transformation.
    -- Selecting them explicitly (not SELECT *) makes the view contract stable:
    -- adding columns to the orders table will not silently break downstream
    -- feature queries that assume a fixed column layout.
    -- -------------------------------------------------------------------------

    order_id,
    user_id,
    eval_set,
    order_number,
    order_hour_of_day,

    -- -------------------------------------------------------------------------
    -- BLOCK B: MISSING VALUE IMPUTATION — days_since_prior_order
    -- -------------------------------------------------------------------------
    -- PROBLEM:
    --   For every user's very first order, no prior order exists, so
    --   days_since_prior_order is NULL in the raw data. This is structurally
    --   intentional in the Instacart dataset, not a data quality defect.
    --
    -- ML IMPACT OF RAW NULLs:
    --   Most ML libraries (scikit-learn, XGBoost, LightGBM in strict mode)
    --   will either raise an error or silently produce NaN predictions when
    --   encountering NULL/NaN in a numeric feature column.
    --
    -- IMPUTATION STRATEGY — Zero Imputation:
    --   COALESCE returns the first non-NULL argument. Here it replaces NULL
    --   with 0.0, which is semantically accurate: a user's first order has
    --   zero days of purchase history behind it. This avoids the statistical
    --   distortion of mean/median imputation, which would incorrectly suggest
    --   these users had some prior purchase interval.
    --
    --   NUMERIC(5,1) cast is explicit to preserve the source column's
    --   precision contract and prevent implicit type widening to FLOAT8.
    -- -------------------------------------------------------------------------

    COALESCE(
        days_since_prior_order,
        CAST(0.0 AS NUMERIC(5,1))
    )                                               AS days_since_prior_order,

    -- -------------------------------------------------------------------------
    -- BLOCK C: ONE-HOT ENCODING — order_dow (Day of Week)
    -- -------------------------------------------------------------------------
    -- PROBLEM:
    --   order_dow is an INTEGER column with values 0–6 representing distinct
    --   weekday categories using Instacart's encoding:
    --       0 = Saturday  |  1 = Sunday  |  2 = Monday
    --       3 = Tuesday   |  4 = Wednesday|  5 = Thursday  |  6 = Friday
    --
    -- ML IMPACT OF RAW INTEGER ENCODING:
    --   Linear models compute dot products; tree models split on thresholds.
    --   Both operations impose an implicit numeric distance on the raw values.
    --   With integers 0–6, the model would infer:
    --       distance(Saturday, Monday) = 2
    --       distance(Saturday, Friday) = 6
    --   These distances are arithmetically meaningless for day-of-week; there
    --   is no sense in which Friday is "3x further" from Saturday than Monday.
    --
    -- SOLUTION — One-Hot Encoding (OHE):
    --   Each day becomes an independent binary column (1 = yes, 0 = no).
    --   The resulting 7-column representation is orthogonal: no day is
    --   numerically "closer" to another. The model learns a separate weight
    --   for each day independently.
    --
    -- NOTE ON THE DUMMY VARIABLE TRAP:
    --   Strict linear models can suffer multicollinearity when all 7 OHE
    --   columns are included (since they always sum to 1). In that case, drop
    --   one reference column (e.g., is_dow_0) in your Python preprocessing
    --   pipeline. Tree-based models (XGBoost, LightGBM, Random Forest) are
    --   immune to this issue and benefit from all 7 columns being present.
    --   We include all 7 here to keep the SQL layer model-agnostic.
    -- -------------------------------------------------------------------------

    -- Saturday (Instacart dow = 0)
    CASE WHEN order_dow = 0 THEN 1 ELSE 0 END      AS is_dow_0,

    -- Sunday (Instacart dow = 1)
    CASE WHEN order_dow = 1 THEN 1 ELSE 0 END      AS is_dow_1,

    -- Monday (Instacart dow = 2)
    CASE WHEN order_dow = 2 THEN 1 ELSE 0 END      AS is_dow_2,

    -- Tuesday (Instacart dow = 3)
    CASE WHEN order_dow = 3 THEN 1 ELSE 0 END      AS is_dow_3,

    -- Wednesday (Instacart dow = 4)
    CASE WHEN order_dow = 4 THEN 1 ELSE 0 END      AS is_dow_4,

    -- Thursday (Instacart dow = 5)
    CASE WHEN order_dow = 5 THEN 1 ELSE 0 END      AS is_dow_5,

    -- Friday (Instacart dow = 6)
    CASE WHEN order_dow = 6 THEN 1 ELSE 0 END      AS is_dow_6,

    -- -------------------------------------------------------------------------
    -- BLOCK D: DERIVED BEHAVIORAL FEATURE — is_weekend
    -- -------------------------------------------------------------------------
    -- PURPOSE:
    --   A high-level business signal that collapses the 7-day encoding into a
    --   single binary flag indicating weekend vs. weekday shopping behaviour.
    --
    -- ML RATIONALE:
    --   Consumer grocery behaviour differs structurally between weekends and
    --   weekdays: larger basket sizes, more discretionary purchases, and higher
    --   reorder rates are commonly observed on weekends. Providing this flag
    --   gives the model a pre-computed interaction feature that would otherwise
    --   require it to independently discover the (is_dow_0 OR is_dow_1)
    --   relationship — which tree models can learn, but linear models cannot
    --   without explicit feature engineering.
    --
    -- ENCODING:
    --   In the Instacart dataset:
    --       dow = 0 → Saturday  (weekend)
    --       dow = 6 → Friday    (weekday — NOT weekend despite being dow = 6)
    --
    --   IMPORTANT: Instacart's encoding is NOT a standard Mon=0 calendar.
    --   Saturday (0) and Sunday (1) are the confirmed weekend days.
    --   This has been validated against the official Instacart data dictionary.
    -- -------------------------------------------------------------------------

    CASE
        WHEN order_dow IN (0, 1) THEN 1   -- Saturday or Sunday
        ELSE 0                            -- Monday through Friday
    END                                             AS is_weekend,

    -- -------------------------------------------------------------------------
    -- BLOCK E: RAW CATEGORICAL COLUMN — order_dow

    order_dow                                       AS order_dow_raw

FROM
    orders;


-- =============================================================================
-- SECTION 3: VERIFICATION QUERIES
-- =============================================================================


-- CHECK 1: View structure — confirm all expected columns are present.
-- Expected: 16 columns including all is_dow_* flags and is_weekend.
SELECT
    column_name,
    data_type
FROM
    information_schema.columns
WHERE
    table_name   = 'v_cleaned_orders'
    AND table_schema = current_schema()
ORDER BY
    ordinal_position;


-- CHECK 2: NULL imputation — confirm zero NULLs remain in the cleaned column.
-- Expected result: null_count = 0
SELECT
    COUNT(*) FILTER (WHERE days_since_prior_order IS NULL) AS null_count
FROM
    v_cleaned_orders;


-- CHECK 3: OHE correctness — each row must have exactly one is_dow_* flag = 1.
-- Expected result: rows_with_wrong_ohe = 0
SELECT
    COUNT(*) AS rows_with_wrong_ohe
FROM
    v_cleaned_orders
WHERE
    (is_dow_0 + is_dow_1 + is_dow_2 + is_dow_3 + is_dow_4 + is_dow_5 + is_dow_6) <> 1;


-- CHECK 4: Weekend flag distribution — sense check the split.
-- Expected: roughly 30–35% of orders placed on weekends (days 0 and 1).
SELECT
    is_weekend,
    COUNT(*)                                            AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM
    v_cleaned_orders
GROUP BY
    is_weekend
ORDER BY
    is_weekend;


-- CHECK 5: Sample output — visual spot-check of 10 rows across eval sets.
SELECT
    order_id,
    user_id,
    eval_set,
    order_number,
    order_dow_raw,
    days_since_prior_order,
    is_dow_0, is_dow_1, is_dow_2, is_dow_3,
    is_dow_4, is_dow_5, is_dow_6,
    is_weekend
FROM
    v_cleaned_orders
ORDER BY
    RANDOM()
LIMIT 10;

SELECT
    order_id,
    user_id,
    eval_set,
    order_number,
    order_dow_raw,
    days_since_prior_order,
    is_dow_0, is_dow_1, is_dow_2, is_dow_3,
    is_dow_4, is_dow_5, is_dow_6,
    is_weekend
FROM
    v_cleaned_orders
ORDER BY
    RANDOM()
LIMIT 10;
