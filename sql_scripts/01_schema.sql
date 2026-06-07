-- =============================================================================
-- FILE: 01_schema.sql
-- PURPOSE:
--   Defines the foundational relational schema for raw ingestion of the
--   Instacart dataset. This script is the single source of truth for
--   table structure, constraints, and bulk data loading.
-- =============================================================================

-- =============================================================================
-- SECTION 1: CLEAN SLATE (Idempotency)
-- =============================================================================
-- DROP ORDER matters: child tables (those with FK references) must be dropped
-- before their parent tables. CASCADE handles any dependent views/indexes
-- automatically, making re-runs safe even if downstream objects exist.

DROP TABLE IF EXISTS order_products CASCADE;  -- child of orders + products
DROP TABLE IF EXISTS orders       CASCADE;  -- child of users (future)
DROP TABLE IF EXISTS products     CASCADE;  -- child of aisles + departments
DROP TABLE IF EXISTS aisles       CASCADE;  -- root lookup table
DROP TABLE IF EXISTS departments  CASCADE;  -- root lookup table

-- =============================================================================
-- SECTION 2: LOOKUP / DIMENSION TABLES
-- =============================================================================
-- These are small, rarely-updated tables. Loading them first satisfies FK
-- constraints when the larger fact tables are loaded below.

-- -----------------------------------------------------------------------------
-- TABLE: aisles
-- SOURCE FILE: data/aisles.csv
-- COLUMNS: aisle_id, aisle
-- SAMPLE ROW: 1,"prepared soups salads"
-- -----------------------------------------------------------------------------

CREATE TABLE aisles (
    -- SMALLINT (2 bytes) is sufficient — the dataset contains only 134 aisles.
    -- Using the smallest adequate type reduces index and cache footprint.
    aisle_id   SMALLINT     NOT NULL,
    aisle      VARCHAR(100) NOT NULL,   -- human-readable label; 100 chars is generous

    CONSTRAINT pk_aisles PRIMARY KEY (aisle_id)
);

COMMENT ON TABLE  aisles          IS 'Lookup table: store aisle identifiers and names.';
COMMENT ON COLUMN aisles.aisle_id IS 'Surrogate PK; maps to products.aisle_id.';
COMMENT ON COLUMN aisles.aisle    IS 'Human-readable aisle name (e.g. "fresh vegetables").';


-- -----------------------------------------------------------------------------
-- TABLE: departments
-- SOURCE FILE: data/departments.csv
-- COLUMNS: department_id, department
-- SAMPLE ROW: 1,"frozen"
-- -----------------------------------------------------------------------------

CREATE TABLE departments (
    -- Only 21 departments exist in the dataset; SMALLINT is ideal.
    department_id   SMALLINT     NOT NULL,
    department      VARCHAR(100) NOT NULL,

    CONSTRAINT pk_departments PRIMARY KEY (department_id)
);

COMMENT ON TABLE  departments               IS 'Lookup table: store department identifiers and names.';
COMMENT ON COLUMN departments.department_id IS 'Surrogate PK; maps to products.department_id.';
COMMENT ON COLUMN departments.department    IS 'Human-readable department name (e.g. "produce").';


-- =============================================================================
-- SECTION 3: CORE ENTITY TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- TABLE: products
-- SOURCE FILE: data/products.csv
-- COLUMNS: product_id, product_name, aisle_id, department_id
-- SAMPLE ROW: 1,"Chocolate Sandwich Cookies",61,19
-- -----------------------------------------------------------------------------

CREATE TABLE products (
    -- INTEGER (4 bytes) covers the ~49,688 products with room for future growth.
    product_id      INTEGER      NOT NULL,
    product_name    VARCHAR(500) NOT NULL,   -- some product names are long strings
    aisle_id        SMALLINT     NOT NULL,
    department_id   SMALLINT     NOT NULL,

    CONSTRAINT pk_products            PRIMARY KEY (product_id),

    -- FK constraints enforce referential integrity: you cannot load a product
    -- that references a non-existent aisle or department. This catches CSV
    -- join-key mismatches at ingest time rather than silently in feature queries.
    CONSTRAINT fk_products_aisles     FOREIGN KEY (aisle_id)
        REFERENCES aisles(aisle_id)
        ON DELETE RESTRICT,           -- prevent orphan products if aisle deleted

    CONSTRAINT fk_products_department FOREIGN KEY (department_id)
        REFERENCES departments(department_id)
        ON DELETE RESTRICT
);

COMMENT ON TABLE  products             IS 'Product catalog: master list of all SKUs with taxonomy.';
COMMENT ON COLUMN products.product_id  IS 'Surrogate PK; referenced by order_products.product_id.';
COMMENT ON COLUMN products.aisle_id    IS 'FK → aisles.aisle_id; used for aisle-level feature aggregation.';
COMMENT ON COLUMN products.department_id IS 'FK → departments.department_id; used for department-level features.';


-- -----------------------------------------------------------------------------
-- TABLE: orders
-- SOURCE FILE: data/orders.csv
-- COLUMNS: order_id, user_id, eval_set, order_number, order_dow,
--          order_hour_of_day, days_since_prior_order
--
-- NOTES ON eval_set:
--   The Instacart dataset splits orders into 3 sets:
--     "prior"  — historical orders used to build features
--     "train"  — the target order for supervised learning (has reordered labels)
--     "test"   — the target order for Kaggle submission (no labels)
--   Storing this as a 1-byte CHAR avoids creating a separate lookup table
--   while still being more type-safe than TEXT.
-- -----------------------------------------------------------------------------

CREATE TABLE orders (
    order_id               INTEGER      NOT NULL,
    user_id                INTEGER      NOT NULL,   -- ~206,209 unique users

    -- eval_set is a short categorical code; CHAR(5) is exact-fit and compact.
    eval_set               CHAR(5)      NOT NULL,   -- 'prior', 'train', 'test'

    -- order_number is the sequence of orders per user (1 = first order).
    -- SMALLINT supports up to 99 orders per user; the dataset max is ~100.
    order_number           SMALLINT     NOT NULL CHECK (order_number >= 1),

    -- Day of week: 0 (Saturday) – 6 (Friday) in Instacart encoding.
    order_dow              SMALLINT     NOT NULL CHECK (order_dow BETWEEN 0 AND 6),

    -- Hour of day: 0–23.
    order_hour_of_day      SMALLINT     NOT NULL CHECK (order_hour_of_day BETWEEN 0 AND 23),

    -- NULL for the very first order of each user (no prior order exists).
    -- NUMERIC(5,1) preserves the one decimal place present in the source data.
    days_since_prior_order NUMERIC(5,1) NULL,

    CONSTRAINT pk_orders PRIMARY KEY (order_id)
);

COMMENT ON TABLE  orders                        IS 'Transaction log: one row per order placed by a user.';
COMMENT ON COLUMN orders.order_id               IS 'Surrogate PK; referenced by order_products.order_id.';
COMMENT ON COLUMN orders.user_id                IS 'Anonymous user identifier; no FK until users table is created.';
COMMENT ON COLUMN orders.eval_set               IS 'Dataset split: prior | train | test.';
COMMENT ON COLUMN orders.order_number           IS 'Chronological order sequence per user (starts at 1).';
COMMENT ON COLUMN orders.order_dow              IS 'Day of week (0=Saturday, 6=Friday).';
COMMENT ON COLUMN orders.order_hour_of_day      IS 'Hour the order was placed (0–23, 24h clock).';
COMMENT ON COLUMN orders.days_since_prior_order IS 'Days elapsed since users previous order; NULL for first order.';


-- =============================================================================
-- SECTION 4: FACT TABLE (High-Volume)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- TABLE: order_products
-- SOURCE FILES: data/order_products__prior.csv  (~32M rows, 564 MB)
--               data/order_products__train.csv  (~1.4M rows,  24 MB)
-- COLUMNS: order_id, product_id, add_to_cart_order, reordered
-- -----------------------------------------------------------------------------

CREATE TABLE order_products (
    order_id          INTEGER NOT NULL,
    product_id        INTEGER NOT NULL,

    -- Position the item was placed into the cart (1 = first item added).
    -- SMALLINT is sufficient; no realistic cart exceeds 32,767 items.
    add_to_cart_order SMALLINT NOT NULL CHECK (add_to_cart_order >= 1),

    -- Binary ML target: 1 = reordered, 0 = new product for this user.
    -- SMALLINT used (not BOOLEAN) to keep it directly usable as a numeric
    -- feature in ML frameworks without casting. CHECK enforces the domain.
    reordered         SMALLINT NOT NULL CHECK (reordered IN (0, 1)),

    -- Composite PK prevents duplicate (order, product) pairs and creates a
    -- clustered index that accelerates the most common ML query pattern:
    --   WHERE order_id = ? AND product_id = ?
    CONSTRAINT pk_order_products PRIMARY KEY (order_id, product_id),

    CONSTRAINT fk_order_products_orders   FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
        ON DELETE CASCADE,    -- if an order is purged, remove its line items

    CONSTRAINT fk_order_products_products FOREIGN KEY (product_id)
        REFERENCES products(product_id)
        ON DELETE RESTRICT    -- do not allow deleting a product with order history
);

COMMENT ON TABLE  order_products                   IS 'Fact table: line items per order. Source of the ML target variable.';
COMMENT ON COLUMN order_products.order_id          IS 'FK → orders.order_id; part of composite PK.';
COMMENT ON COLUMN order_products.product_id        IS 'FK → products.product_id; part of composite PK.';
COMMENT ON COLUMN order_products.add_to_cart_order IS 'Sequence in which the product was added to the cart.';
COMMENT ON COLUMN order_products.reordered         IS 'ML TARGET: 1 if product was in a prior order, else 0.';


-- =============================================================================
-- SECTION 5: BULK DATA INGESTION (PostgreSQL COPY)
-- =============================================================================
-- COPY is the fastest ingestion path in PostgreSQL. It streams CSV directly
-- into heap pages, bypassing per-row trigger overhead and WAL amplification.
-- =============================================================================

-- ── Dimension tables (small, fast) ──────────────────────────────────────────

COPY aisles (aisle_id, aisle)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/aisles.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ','
);

COPY departments (department_id, department)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/departments.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ','
);

-- ── Core entity table ────────────────────────────────────────────────────────

COPY products (product_id, product_name, aisle_id, department_id)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/products.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ','
);

-- ── Transaction log ──────────────────────────────────────────────────────────

COPY orders (
    order_id,
    user_id,
    eval_set,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order
)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/orders.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ',',
    NULL ''
);

-- ── Fact table — prior split (~32M rows) ────────────────────────────────────

COPY order_products (order_id, product_id, add_to_cart_order, reordered)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/order_products__prior.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ','
);

-- ── Fact table — train split (~1.4M rows) ───────────────────────────────────

COPY order_products (order_id, product_id, add_to_cart_order, reordered)
FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/order_products__train.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    DELIMITER ','
);


-- =============================================================================
-- SECTION 6: ROW COUNT VERIFICATION
-- =============================================================================

SELECT 'aisles'          AS table_name, COUNT(*) AS row_count FROM aisles
UNION ALL
SELECT 'departments',                   COUNT(*)              FROM departments
UNION ALL
SELECT 'products',                      COUNT(*)              FROM products
UNION ALL
SELECT 'orders',                        COUNT(*)              FROM orders
UNION ALL
SELECT 'order_products',                COUNT(*)              FROM order_products
ORDER BY table_name;


-- =============================================================================
-- SECTION 7: POST-LOAD STATISTICS UPDATE
-- =============================================================================

ANALYZE aisles;
ANALYZE departments;
ANALYZE products;
ANALYZE orders;
ANALYZE order_products;