-- =============================================================================
-- FILE:    sql_scripts/03_feature_engineering.sql
-- PROJECT: E-Commerce Real-Time Feature Store for AI Models
-- PHASE:   3 — High-Level Feature Engineering
-- =============================================================================

-- =============================================================================
-- SECTION 1: IDEMPOTENCY GUARDS
-- =============================================================================
DROP MATERIALIZED VIEW IF EXISTS mv_user_product_features CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_product_features     CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_user_features         CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_user_product_prior_history CASCADE;

-- =============================================================================
-- SECTION 2: LAYER 1 — USER-LEVEL FEATURES (mv_user_features)
-- =============================================================================
CREATE MATERIALIZED VIEW mv_user_features AS
WITH user_core_aggregates AS (
    -- Compute total orders and averages directly from the clean orders view
    SELECT
        vc.user_id,
        MAX(vc.order_number) AS user_total_orders,
        ROUND(AVG(vc.days_since_prior_order), 4) AS user_avg_days_between_orders
    FROM
        v_cleaned_orders vc
    WHERE
        vc.eval_set = 'prior'
    GROUP BY
        vc.user_id
),
user_item_aggregates AS (
    -- Count total line items ever purchased per user from the fact table
    SELECT
        vc.user_id,
        COUNT(op.product_id) AS user_total_items_bought
    FROM
        order_products op
        INNER JOIN v_cleaned_orders vc ON vc.order_id = op.order_id
    WHERE
        vc.eval_set = 'prior'
    GROUP BY
        vc.user_id
)
SELECT
    uca.user_id,
    uca.user_total_orders,
    uca.user_avg_days_between_orders,
    uia.user_total_items_bought
FROM
    user_core_aggregates uca
    INNER JOIN user_item_aggregates uia ON uia.user_id = uca.user_id;

-- =============================================================================
-- SECTION 3: LAYER 2 — PRODUCT-LEVEL FEATURES (mv_product_features)
-- =============================================================================
CREATE MATERIALIZED VIEW mv_product_features AS
SELECT
    op.product_id,
    COUNT(*) AS product_total_purchases,
    ROUND(AVG(op.reordered::NUMERIC), 6) AS product_reorder_rate
FROM
    order_products op
    INNER JOIN v_cleaned_orders vc ON vc.order_id = op.order_id
WHERE
    vc.eval_set = 'prior'
GROUP BY
    op.product_id;

-- =============================================================================
-- SECTION 4: LAYER 3 — USER-PRODUCT INTERACTION FEATURES (STAGE A)
-- =============================================================================
CREATE MATERIALIZED VIEW mv_user_product_prior_history AS
SELECT
    vc.user_id,
    op.product_id,
    vc.order_id,
    op.reordered,
    ROW_NUMBER() OVER (
        PARTITION BY vc.user_id, op.product_id
        ORDER BY     vc.order_number ASC
    ) AS buy_sequence,
    vc.order_number,
    LAG(vc.order_number) OVER (
        PARTITION BY vc.user_id, op.product_id
        ORDER BY     vc.order_number ASC
    ) AS prev_order_number_for_product
FROM
    order_products op
    INNER JOIN v_cleaned_orders vc ON vc.order_id = op.order_id
WHERE
    vc.eval_set = 'prior';

-- Critical index to make Stage B execute quickly
CREATE UNIQUE INDEX idx_mv_uph_user_product ON mv_user_product_prior_history (user_id, product_id, order_number);

-- =============================================================================
-- STAGE B: Final User-Product Feature Aggregation (mv_user_product_features)
-- =============================================================================
CREATE MATERIALIZED VIEW mv_user_product_features AS
WITH streak_flags AS (
    SELECT
        user_id,
        product_id,
        order_id,
        order_number,
        buy_sequence,
        reordered,
        CASE
            WHEN prev_order_number_for_product IS NULL THEN 0
            WHEN (order_number - prev_order_number_for_product) > 1 THEN 1
            ELSE 0
        END AS is_streak_break
    FROM
        mv_user_product_prior_history
),
streak_anchor AS (
    SELECT
        user_id,
        product_id,
        COALESCE(MAX(buy_sequence) FILTER (WHERE is_streak_break = 1), 0) AS last_break_sequence
    FROM
        streak_flags
    GROUP BY
        user_id, product_id
)
SELECT
    sf.user_id,
    sf.product_id,
    COUNT(*) AS user_product_total_buys,
    ROUND(AVG(sf.reordered::NUMERIC), 6) AS user_product_reorder_rate,
    SUM(
        CASE
            WHEN sf.buy_sequence > sa.last_break_sequence THEN 1
            ELSE 0
        END
    ) AS user_product_order_streak,
    MIN(sf.buy_sequence) AS user_product_first_buy_number,
    MAX(sf.buy_sequence) AS user_product_last_buy_number
FROM
    streak_flags sf
    INNER JOIN streak_anchor sa
        ON  sa.user_id    = sf.user_id
        AND sa.product_id = sf.product_id
GROUP BY
    sf.user_id,
    sf.product_id;

-- =============================================================================
-- SECTION 5: SUPPORTING INDEXES ON MATERIALIZED VIEWS
-- =============================================================================
CREATE INDEX idx_mv_user_features_user_id ON mv_user_features (user_id);
CREATE INDEX idx_mv_product_features_product_id ON mv_product_features (product_id);
CREATE UNIQUE INDEX idx_mv_upf_user_product ON mv_user_product_features (user_id, product_id);

-- =============================================================================
-- SECTION 6: QUALITY ASSURANCE VERIFICATION QUERIES
-- =============================================================================
-- Run this block to confirm calculations match targets
SELECT
    upf.user_id,
    upf.product_id,
    uf.user_total_orders,
    uf.user_avg_days_between_orders,
    uf.user_total_items_bought,
    pf.product_total_purchases,
    pf.product_reorder_rate,
    upf.user_product_total_buys,
    upf.user_product_reorder_rate,
    upf.user_product_order_streak
FROM
    mv_user_product_features upf
    INNER JOIN mv_user_features    uf  ON uf.user_id    = upf.user_id
    INNER JOIN mv_product_features pf  ON pf.product_id = upf.product_id
ORDER BY
    upf.user_product_total_buys DESC
LIMIT 10;

