-- PURPOSE:
--   Constructs the final, model-ready training dataset by joining the three
--   feature layers from Phase 3 against ground-truth reorder labels derived
--   from the 'train' eval_set. The output view v_ml_training_dataset is the
--   direct input to the Python/scikit-learn/XGBoost training pipeline.
--
-- ML FRAMING — Binary Classification:
--   Task:    Predict whether user U will reorder product P in their next order.
--   Input X: All columns except is_reordered_target.
--   Label y: is_reordered_target ∈ {0, 1}
--
--   Label construction logic:
--     Universe of candidates  = every (user, product) pair where the user
--                               has bought the product at least once historically
--                               (sourced from mv_user_product_features).
--     Positive label  (y = 1) = the product appears in the user's 'train' order.
--     Negative label  (y = 0) = the product does NOT appear in the 'train' order,
--                               but the user HAS bought it before. These are the
--                               "considered but skipped" items — the hardest and
--                               most information-rich negative examples.
--
-- WHY LEFT JOIN IS CRITICAL FOR LABEL GENERATION:
--   An INNER JOIN between candidates and train order items would silently
--   discard all (y = 0) rows, leaving only positives. The model would see
--   nothing but reorders during training, making it incapable of learning to
--   predict non-reorders. The LEFT JOIN preserves every candidate row, and
--   the CASE WHEN on the joined train column distinguishes 1 from 0.
--
-- DATA FLOW:
--   orders (eval_set='train')              → train_orders CTE (anchor)
--   order_products (train order_ids)       → train_items CTE  (positive labels)
--   mv_user_product_features               → candidate (user, product) universe
--   mv_user_features                       → user-level feature columns
--   mv_product_features                    → product-level feature columns
--   v_cleaned_orders (eval_set='train')    → train-order temporal features
--
-- DEPENDENCIES:
--   01_schema.sql              → orders, order_products, products
--   02_data_cleaning.sql       → v_cleaned_orders
--   03_feature_engineering.sql → mv_user_product_features,
--                                mv_user_features,
--                                mv_product_features


-- =============================================================================
-- SECTION 1: IDEMPOTENCY GUARD
-- =============================================================================

DROP VIEW IF EXISTS v_ml_training_dataset CASCADE;


-- =============================================================================
-- SECTION 2: TRAINING MATRIX VIEW DEFINITION
-- =============================================================================

CREATE VIEW v_ml_training_dataset AS

-- ─────────────────────────────────────────────────────────────────────────────
-- CTE 1: train_orders
-- ─────────────────────────────────────────────────────────────────────────────
-- Isolates the single 'train' eval_set order per user.
-- In the Instacart dataset, every user in the train split has exactly one
-- 'train' order — their most recent order, which is what we're predicting.
--
-- We source from v_cleaned_orders (not raw orders) to carry the imputed
-- days_since_prior_order and OHE columns as temporal context features.
-- These describe the CONDITIONS under which the prediction order was placed:
-- Was it placed on a weekend? After a long gap? This context is valid to
-- include because it describes the future order itself, not its contents.
-- ─────────────────────────────────────────────────────────────────────────────
WITH train_orders AS (
    SELECT
        vc.order_id                         AS train_order_id,
        vc.user_id,
        vc.order_number                     AS train_order_number,
        vc.order_hour_of_day                AS train_order_hour,
        vc.days_since_prior_order           AS train_days_since_prior,
        vc.order_dow_raw                    AS train_order_dow,
        vc.is_weekend                       AS train_is_weekend,
        -- OHE day-of-week flags for the prediction order.
        -- Included because day-of-week of the TARGET order influences what
        -- the user buys (weekend shop vs. weekday top-up).
        vc.is_dow_0, vc.is_dow_1, vc.is_dow_2, vc.is_dow_3,
        vc.is_dow_4, vc.is_dow_5, vc.is_dow_6
    FROM
        v_cleaned_orders vc
    WHERE
        vc.eval_set = 'train'
),

-- ─────────────────────────────────────────────────────────────────────────────
-- CTE 2: train_items
-- ─────────────────────────────────────────────────────────────────────────────
-- Retrieves every product that actually appeared in each user's train order.
-- This is the ground truth: the set of (user, product) pairs with label y = 1.
--
-- Joining through train_orders (rather than filtering order_products directly
-- on eval_set) keeps the logic self-documenting and avoids a redundant scan
-- of the orders table. The join is on train_order_id, which is indexed via
-- the PK on order_products(order_id, product_id).
-- ─────────────────────────────────────────────────────────────────────────────
train_items AS (
    SELECT
        to_.user_id,
        op.product_id,
        op.order_id                         AS train_order_id,
        -- Carry reordered from the fact table as a cross-check column.
        -- In a well-formed dataset this should always be 1 here (every item
        -- in the train order is by definition a reorder of a prior product),
        -- but retaining it surfaces any upstream data quality anomalies.
        op.reordered                        AS fact_table_reordered_flag
    FROM
        train_orders to_
        INNER JOIN order_products op ON op.order_id = to_.train_order_id
),

-- ─────────────────────────────────────────────────────────────────────────────
-- CTE 3: candidate_pairs
-- ─────────────────────────────────────────────────────────────────────────────
-- Defines the complete universe of (user, product) pairs for which we need
-- a prediction. This is every product a user has ever bought historically,
-- sourced from the interaction feature layer built in Phase 3.
--
-- DESIGN DECISION — Why mv_user_product_features as the candidate universe?
--   Option A: Cross join all users × all products (~206k × ~50k = 10B rows).
--             Computationally absurd; most pairs are meaningless.
--   Option B: Only products in the train order (y=1 rows only).
--             Creates a label-only dataset — no negative examples.
--   Option C (chosen): Products the user has bought before.
--             Realistic candidate set. A user cannot reorder something they
--             have never bought. This mirrors how real recommender systems
--             generate candidate sets before scoring.
-- ─────────────────────────────────────────────────────────────────────────────
candidate_pairs AS (
    SELECT
        upf.user_id,
        upf.product_id,
        -- Interaction features: the most predictive layer
        upf.user_product_total_buys,
        upf.user_product_reorder_rate,
        upf.user_product_order_streak,
        upf.user_product_first_buy_number,
        upf.user_product_last_buy_number
    FROM
        mv_user_product_features upf
    -- Restrict candidates to users who have a 'train' order.
    -- Users in the 'test' split are excluded from model training.
    WHERE EXISTS (
        SELECT 1
        FROM   train_orders to_
        WHERE  to_.user_id = upf.user_id
    )
)

-- =============================================================================
-- MAIN SELECT: Assemble the unified training matrix
-- =============================================================================
SELECT

    -- ─────────────────────────────────────────────────────────────────────────
    -- IDENTIFIER COLUMNS
    -- Not used as model features, but essential for:
    --   - Joining predictions back to source records during evaluation
    --   - Grouping errors by user/product during model diagnostics
    --   - Stratified train/validation splitting in Python (by user_id)
    -- ─────────────────────────────────────────────────────────────────────────
    cp.user_id,
    cp.product_id,
    to_.train_order_id,

    -- ─────────────────────────────────────────────────────────────────────────
    -- TARGET VARIABLE: is_reordered_target (y)
    -- ─────────────────────────────────────────────────────────────────────────
    -- CASE WHEN logic:
    --   ti.product_id IS NOT NULL → this product appeared in the train order
    --                               → the LEFT JOIN found a match → label = 1
    --   ti.product_id IS NULL     → this product was NOT in the train order
    --                               → the LEFT JOIN returned NULL → label = 0
    --
    -- This is the binary classification target. Every downstream model metric
    -- (AUC, F1, precision, recall) is computed against this column.
    -- ─────────────────────────────────────────────────────────────────────────
    CASE
        WHEN ti.product_id IS NOT NULL THEN 1
        ELSE                                0
    END                                                 AS is_reordered_target,

    -- ─────────────────────────────────────────────────────────────────────────
    -- FEATURE GROUP 1: User-Product Interaction Features (Layer 3)
    -- Source: mv_user_product_features (Phase 3)
    -- These are the most predictive features. They capture the specific
    -- relationship between this user and this product historically.
    -- ─────────────────────────────────────────────────────────────────────────
    cp.user_product_total_buys,
    cp.user_product_reorder_rate,
    cp.user_product_order_streak,
    cp.user_product_first_buy_number,
    cp.user_product_last_buy_number,

    -- Derived interaction feature: recency ratio.
    -- How recently (relative to total purchase count) did the user last buy
    -- this product? A ratio near 1.0 means the last buy was recent;
    -- near 0.0 means it was bought long ago and not since.
    -- NULLIF guards against division by zero for users with 1 total buy.
    ROUND(
        cp.user_product_last_buy_number::NUMERIC
        / NULLIF(cp.user_product_total_buys, 0),
        4
    )                                                   AS user_product_recency_ratio,

    -- ─────────────────────────────────────────────────────────────────────────
    -- FEATURE GROUP 2: User-Level Behavioural Features (Layer 1)
    -- Source: mv_user_features (Phase 3)
    -- Capture the overall shopping profile of this user.
    -- ─────────────────────────────────────────────────────────────────────────
    uf.user_total_orders,
    uf.user_avg_days_between_orders,
    uf.user_total_items_bought,

    -- Derived user feature: average basket size per order.
    -- Users with large baskets buy more diversely and may have lower per-
    -- product reorder rates simply due to basket breadth.
    ROUND(
        uf.user_total_items_bought::NUMERIC
        / NULLIF(uf.user_total_orders, 0),
        4
    )                                                   AS user_avg_basket_size,

    -- ─────────────────────────────────────────────────────────────────────────
    -- FEATURE GROUP 3: Product-Level Catalogue Features (Layer 2)
    -- Source: mv_product_features (Phase 3)
    -- Capture how popular and habit-forming this product is globally,
    -- independent of which user is buying it.
    -- ─────────────────────────────────────────────────────────────────────────
    pf.product_total_purchases,
    pf.product_reorder_rate,

    -- ─────────────────────────────────────────────────────────────────────────
    -- FEATURE GROUP 4: Train Order Temporal Context Features
    -- Source: train_orders CTE (derived from v_cleaned_orders)
    -- Describe the CONDITIONS of the prediction order: when was it placed,
    -- how long since the previous order, what day of the week?
    -- These are valid to include because they describe the future order's
    -- context (timing), NOT its contents (what was bought).
    -- ─────────────────────────────────────────────────────────────────────────
    to_.train_order_number,
    to_.train_order_hour,
    to_.train_days_since_prior,
    to_.train_order_dow,
    to_.train_is_weekend,
    -- One-hot encoded day-of-week for the train order
    to_.is_dow_0                                        AS train_is_dow_0,
    to_.is_dow_1                                        AS train_is_dow_1,
    to_.is_dow_2                                        AS train_is_dow_2,
    to_.is_dow_3                                        AS train_is_dow_3,
    to_.is_dow_4                                        AS train_is_dow_4,
    to_.is_dow_5                                        AS train_is_dow_5,
    to_.is_dow_6                                        AS train_is_dow_6

FROM
    candidate_pairs cp

    -- ─────────────────────────────────────────────────────────────────────────
    -- JOIN 1 (INNER): Attach train order context to each candidate.
    -- INNER JOIN is correct here: we only want candidates for users who have
    -- a train order. Users without a train order (test-split users) were
    -- already excluded in candidate_pairs via the WHERE EXISTS clause, but
    -- the INNER JOIN provides a second safety net and communicates intent.
    -- ─────────────────────────────────────────────────────────────────────────
    INNER JOIN train_orders to_
        ON  to_.user_id = cp.user_id

    -- ─────────────────────────────────────────────────────────────────────────
    -- JOIN 2 (LEFT): Label assignment — the most critical join in the script.
    -- LEFT JOIN preserves ALL candidate rows.
    --   Match found    → ti.product_id IS NOT NULL → label = 1
    --   No match found → ti.product_id IS NULL     → label = 0
    -- Using INNER JOIN here would be the single most common ML data pipeline
    -- bug: it would silently drop all negative examples, creating a 100%-
    -- positive training set that produces a useless model.
    -- ─────────────────────────────────────────────────────────────────────────
    LEFT JOIN train_items ti
        ON  ti.user_id    = cp.user_id
        AND ti.product_id = cp.product_id

    -- ─────────────────────────────────────────────────────────────────────────
    -- JOIN 3 (LEFT): User-level features.
    -- LEFT JOIN protects against any user_id present in interaction features
    -- but absent from mv_user_features due to an edge-case aggregation gap.
    -- In a clean run these should always match; LEFT JOIN surfaces NULLs
    -- as a data quality signal rather than silently dropping rows.
    -- ─────────────────────────────────────────────────────────────────────────
    LEFT JOIN mv_user_features uf
        ON  uf.user_id = cp.user_id

    -- ─────────────────────────────────────────────────────────────────────────
    -- JOIN 4 (LEFT): Product-level features.
    -- Same protective rationale as JOIN 3.
    -- ─────────────────────────────────────────────────────────────────────────
    LEFT JOIN mv_product_features pf
        ON  pf.product_id = cp.product_id;


-- =============================================================================
-- SECTION 3: QUALITY ASSURANCE VERIFICATION QUERIES
-- =============================================================================

-- QA-1: Visual spot-check of the training matrix.
-- Inspect that all feature columns are populated and is_reordered_target
-- contains both 0s and 1s (not just one class).
SELECT *
FROM   v_ml_training_dataset
LIMIT  10;


-- QA-2: Label distribution check.
-- CRITICAL: Both classes must be present. A 100% positive or 100% negative
-- result means the LEFT JOIN logic has failed. Expected: ~10–13% positive rate
-- (the Instacart dataset is naturally imbalanced toward non-reorders).
SELECT
    is_reordered_target,
    COUNT(*)                                                    AS row_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)         AS pct_of_total
FROM
    v_ml_training_dataset
GROUP BY
    is_reordered_target
ORDER BY
    is_reordered_target;


-- QA-3: NULL audit on all feature columns.
-- Expected: zero NULLs across all feature columns. Any NULLs indicate a
-- join gap between the feature layers that must be investigated before
-- passing this data to a model.
SELECT
    COUNT(*) FILTER (WHERE user_product_total_buys       IS NULL) AS null_total_buys,
    COUNT(*) FILTER (WHERE user_product_reorder_rate     IS NULL) AS null_up_reorder_rate,
    COUNT(*) FILTER (WHERE user_product_order_streak     IS NULL) AS null_streak,
    COUNT(*) FILTER (WHERE user_total_orders             IS NULL) AS null_user_orders,
    COUNT(*) FILTER (WHERE user_avg_days_between_orders  IS NULL) AS null_avg_days,
    COUNT(*) FILTER (WHERE product_total_purchases       IS NULL) AS null_prod_purchases,
    COUNT(*) FILTER (WHERE product_reorder_rate          IS NULL) AS null_prod_reorder_rate,
    COUNT(*) FILTER (WHERE train_days_since_prior        IS NULL) AS null_train_days
FROM
    v_ml_training_dataset;


-- QA-4: Row count and unique pair check.
-- Expected: total rows = total unique (user_id, product_id) pairs.
-- Any discrepancy means a user has duplicate train orders — a schema violation.
SELECT
    COUNT(*)                                        AS total_rows,
    COUNT(DISTINCT (user_id, product_id))           AS unique_user_product_pairs,
    COUNT(DISTINCT user_id)                         AS unique_users,
    COUNT(DISTINCT product_id)                      AS unique_products
FROM
    v_ml_training_dataset;


-- QA-5: Feature range sanity check.
-- All rates must be [0, 1]. Streaks and buy counts must be positive.
-- out_of_range_count = 0 across all columns confirms no corruption.
SELECT
    COUNT(*) FILTER (
        WHERE user_product_reorder_rate NOT BETWEEN 0 AND 1
    )                                               AS bad_up_reorder_rate,
    COUNT(*) FILTER (
        WHERE product_reorder_rate NOT BETWEEN 0 AND 1
    )                                               AS bad_prod_reorder_rate,
    COUNT(*) FILTER (
        WHERE user_product_order_streak < 1
    )                                               AS bad_streak,
    COUNT(*) FILTER (
        WHERE user_product_recency_ratio NOT BETWEEN 0 AND 1
    )                                               AS bad_recency_ratio
FROM
    v_ml_training_dataset;
