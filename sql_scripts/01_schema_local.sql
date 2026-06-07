-- Clean up any broken half-created tables first
DROP TABLE IF EXISTS order_products CASCADE;
DROP TABLE IF EXISTS orders       CASCADE;
DROP TABLE IF EXISTS products     CASCADE;
DROP TABLE IF EXISTS aisles       CASCADE;
DROP TABLE IF EXISTS departments  CASCADE;

-- Rebuild the clean schema structures
CREATE TABLE aisles (aisle_id SMALLINT NOT NULL PRIMARY KEY, aisle VARCHAR(100) NOT NULL);
CREATE TABLE departments (department_id SMALLINT NOT NULL PRIMARY KEY, department VARCHAR(100) NOT NULL);
CREATE TABLE products (product_id INTEGER NOT NULL PRIMARY KEY, product_name VARCHAR(500) NOT NULL, aisle_id SMALLINT NOT NULL, department_id SMALLINT NOT NULL);
CREATE TABLE orders (order_id INTEGER NOT NULL PRIMARY KEY, user_id INTEGER NOT NULL, eval_set CHAR(5) NOT NULL, order_number SMALLINT NOT NULL, order_dow SMALLINT NOT NULL, order_hour_of_day SMALLINT NOT NULL, days_since_prior_order NUMERIC(5,1) NULL);
CREATE TABLE order_products (order_id INTEGER NOT NULL, product_id INTEGER NOT NULL, add_to_cart_order SMALLINT NOT NULL, reordered SMALLINT NOT NULL, CONSTRAINT pk_order_products PRIMARY KEY (order_id, product_id));

-- Blazing fast client-side streaming (Bypasses Windows Permissions completely)
\copy aisles FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/aisles.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
\copy departments FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/departments.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
\copy products FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/products.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
\copy orders FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/orders.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', NULL '')

-- The massive fact tables
\copy order_products FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/order_products__prior.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
\copy order_products FROM 'C:/Users/dell/Desktop/Git Hub Projects/Final_Github_Projects/ecommerce_realtime_feature_store_sql/data/order_products__train.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')

-- Verification check
SELECT 'order_products' AS table_name, COUNT(*) FROM order_products;