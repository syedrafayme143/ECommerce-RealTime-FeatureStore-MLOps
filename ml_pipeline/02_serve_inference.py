from __future__ import annotations

import sys
import time
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from xgboost import XGBClassifier

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
DB_CONFIG: dict[str, str | int] = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "ecommerce",                      # <-- Perfect matching database
    "user":     "postgres",
    "password": urllib.parse.quote_plus("Syedrafay123/"), # <-- Protected against password slash crashes!
}

MODEL_PATH: str = "ml_pipeline/xgb_reorder_model.json"
TOP_N: int = 10
MIN_PROBABILITY_THRESHOLD: float = 0.05

MODEL_FEATURE_COLUMNS: list[str] = [
    "user_product_total_buys",
    "user_product_reorder_rate",
    "user_product_order_streak",
    "user_product_first_buy_number",
    "user_product_last_buy_number",
    "user_product_recency_ratio",
    "user_total_orders",
    "user_avg_days_between_orders",
    "user_total_items_bought",
    "user_avg_basket_size",
    "product_total_purchases",
    "product_reorder_rate",
    "train_order_number",
    "train_order_hour",
    "train_days_since_prior",
    "train_order_dow",
    "train_is_weekend",
    "train_is_dow_0",
    "train_is_dow_1",
    "train_is_dow_2",
    "train_is_dow_3",
    "train_is_dow_4",
    "train_is_dow_5",
    "train_is_dow_6",
]

# ==============================================================================
# SECTION 2: DATA CLASSES
# ==============================================================================
@dataclass
class ServingRequest:
    user_id:               int
    current_hour:          int
    current_dow:           int
    days_since_last_order: Optional[float] = None
    top_n:                 int = TOP_N

@dataclass
class ProductRecommendation:
    rank:          int
    product_id:    int
    product_name:  str
    reorder_prob:  float
    total_buys:    int
    order_streak:  int
    product_rate:  float

@dataclass
class ServingResponse:
    request:            ServingRequest
    recommendations:    list[ProductRecommendation] = field(default_factory=list)
    candidates_scored: int   = 0
    db_fetch_ms:        float = 0.0
    inference_ms:       float = 0.0
    total_ms:           float = 0.0

# ==============================================================================
# SECTION 3: DATABASE ENGINE
# ==============================================================================
def build_engine(config: dict[str, str | int]) -> Engine:
    url = (
        f"postgresql+psycopg2://"
        f"{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}"
        f"/{config['dbname']}"
    )
    try:
        engine = create_engine(
            url,
            pool_size=5,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[DB] Connection pool initialized securely for user serving framework.")
        return engine
    except Exception as exc:
        print(f"[ERROR] Database serving infrastructure configuration aborted: {exc}")
        sys.exit(1)

# ==============================================================================
# SECTION 4: MODEL LOADING
# ==============================================================================
def load_model(model_path: str) -> XGBClassifier:
    try:
        model = XGBClassifier()
        model.load_model(model_path)
        return model
    except Exception as exc:
        print(f"[ERROR] Serialization error: compiled booster artifact unreadable at source route '{model_path}': {exc}")
        sys.exit(1)

# ==============================================================================
# SECTION 5: FEATURE FETCHING (FIXED EXTRA FROM COMMA)
# ==============================================================================
def fetch_user_serving_features(user_id: int, engine: Engine) -> pd.DataFrame:
    query = text("""
        SELECT
            upf.product_id,
            upf.user_product_total_buys,
            upf.user_product_reorder_rate,
            upf.user_product_order_streak,
            upf.user_product_first_buy_number,
            upf.user_product_last_buy_number,
            ROUND(
                upf.user_product_last_buy_number::NUMERIC
                / NULLIF(upf.user_product_total_buys, 0),
                4
            )                                   AS user_product_recency_ratio,
            uf.user_total_orders,
            uf.user_avg_days_between_orders,
            uf.user_total_items_bought,
            ROUND(
                uf.user_total_items_bought::NUMERIC
                / NULLIF(uf.user_total_orders, 0),
                4
            )                                   AS user_avg_basket_size,
            pf.product_total_purchases,
            pf.product_reorder_rate             -- FIXED: Comma layout error neutralized!
        FROM  mv_user_product_features  upf
        JOIN  mv_user_features          uf   ON uf.user_id    = upf.user_id
        JOIN  mv_product_features       pf   ON pf.product_id = upf.product_id
        WHERE upf.user_id = :user_id
        ORDER BY upf.user_product_total_buys DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        rows   = result.fetchall()
        cols   = list(result.keys())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=cols)

def fetch_product_names(product_ids: list[int], engine: Engine) -> pd.DataFrame:
    query = text("""
        SELECT
            p.product_id,
            p.product_name,
            a.aisle        AS aisle_name,
            d.department   AS department_name
        FROM  products    p
        JOIN  aisles      a ON a.aisle_id      = p.aisle_id
        JOIN  departments d ON d.department_id = p.department_id
        WHERE p.product_id = ANY(:ids)
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"ids": product_ids})
        rows   = result.fetchall()
        cols   = list(result.keys())
    return pd.DataFrame(rows, columns=cols)

# ==============================================================================
# SECTION 6: TEMPORAL CONTEXT INJECTION
# ==============================================================================
def inject_temporal_context(
    features_df:           pd.DataFrame,
    request:               ServingRequest,
    user_avg_days_between: float,
) -> pd.DataFrame:
    df = features_df.copy()
    days_since = (
        request.days_since_last_order
        if request.days_since_last_order is not None
        else float(user_avg_days_between)
    )
    dow = request.current_dow

    df["train_order_number"]     = df["user_total_orders"] + 1
    df["train_order_hour"]       = request.current_hour
    df["train_days_since_prior"] = days_since
    df["train_order_dow"]        = dow
    df["train_is_weekend"]       = 1 if dow in (0, 1) else 0

    for d in range(7):
        df[f"train_is_dow_{d}"] = 1 if dow == d else 0

    return df

# ==============================================================================
# SECTION 7: INFERENCE ENGINE (FIXED COLUMN ARRAY ORDER ALIGNMENT)
# ==============================================================================
def run_inference(model: XGBClassifier, features_df: pd.DataFrame) -> np.ndarray:
    # FIXED: Enforce absolute feature layout sequencing sorting by passing array index map matching explicitly
    X_df = features_df[MODEL_FEATURE_COLUMNS].fillna(0)
    X = X_df.values
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities

# ==============================================================================
# SECTION 8: RESULT ASSEMBLY
# ==============================================================================
def build_recommendation_report(
    request:       ServingRequest,
    features_df:   pd.DataFrame,
    probabilities: np.ndarray,
    engine:        Engine,
) -> ServingResponse:
    response = ServingResponse(request=request)
    results_df = features_df[
        ["product_id",
         "user_product_total_buys",
         "user_product_order_streak",
         "product_reorder_rate"]
    ].copy()

    results_df["reorder_prob"] = probabilities
    response.candidates_scored = len(results_df)

    results_df = results_df[results_df["reorder_prob"] >= MIN_PROBABILITY_THRESHOLD]
    results_df = results_df.sort_values("reorder_prob", ascending=False).head(request.top_n).reset_index(drop=True)

    if results_df.empty:
        return response

    product_ids  = results_df["product_id"].tolist()
    products_df  = fetch_product_names(product_ids, engine)
    results_df   = results_df.merge(products_df, on="product_id", how="left")

    for rank, row in enumerate(results_df.itertuples(), start=1):
        response.recommendations.append(
            ProductRecommendation(
                rank         = rank,
                product_id   = int(row.product_id),
                product_name = str(getattr(row, "product_name", "Unknown")),
                reorder_prob = float(row.reorder_prob),
                total_buys   = int(row.user_product_total_buys),
                order_streak = int(row.user_product_order_streak),
                product_rate = float(row.product_reorder_rate),
            )
        )
    return response

# ==============================================================================
# SECTION 9: DISPLAY LAYER (FIXED FLOAT EVALUATION LABEL WARNING)
# ==============================================================================
def print_serving_report(response: ServingResponse) -> None:
    req  = response.request
    recs = response.recommendations
    dow_labels = {0: "Saturday", 1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}
    bar_width = 20

    print("\n" + "═" * 78)
    print(f"  REAL-TIME REORDER RECOMMENDATIONS")
    print("═" * 78)
    print(f"  User ID          : {req.user_id}")
    print(f"  Serving time     : {dow_labels.get(req.current_dow, '?')}, Hour {req.current_hour:02d}:00 ({'Weekend' if req.current_dow in (0,1) else 'Weekday'})")
    
    # FIXED: Clean string fallback injection prevention
    if req.days_since_last_order is not None:
        days_label = f"{req.days_since_last_order:.0f} days (provided)"
    else:
        days_label = "historical average (estimated)"
        
    print(f"  Days since last  : {days_label}")
    print(f"  Candidates scored: {response.candidates_scored:,} products")
    print(f"\n  ┌─ Latency Breakdown ──────────────────────────────────┐")
    print(f"  │  Feature fetch (PostgreSQL)  : {response.db_fetch_ms:>7.2f} ms             │")
    print(f"  │  Model inference (XGBoost)   : {response.inference_ms:>7.3f} ms             │")
    print(f"  │  Total end-to-end            : {response.total_ms:>7.2f} ms             │")
    print(f"  │  Sub-20ms target             : {'✓ TARGET MET' if response.total_ms < 20 else '✗ ABOVE 20ms':<27}│")
    print(f"  └──────────────────────────────────────────────────────┘")

    if not recs:
        print(f"\n  No products exceeded threshold ({MIN_PROBABILITY_THRESHOLD:.0%}).")
        return

    print(f"\n  TOP {len(recs)} PREDICTED REORDERS\n")
    print(f"  {'#':<3}  {'Probability':<26}  {'Buys':>5}  {'Streak':>6}  {'Global%':>7}  Product")
    print("  " + "─" * 74)

    for rec in recs:
        filled = int(rec.reorder_prob * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        streak_label = f"{rec.order_streak:>4}🔥" if rec.order_streak >= 3 else f"{rec.order_streak:>6}"
        name = (rec.product_name[:38] + "…") if len(rec.product_name) > 39 else rec.product_name
        print(f"  {rec.rank:<3}  [{bar}] {rec.reorder_prob:>5.1%}  {rec.total_buys:>5}  {streak_label}  {rec.product_rate:>6.1%}   {name}")
    print("═" * 78 + "\n")

# ==============================================================================
# SECTION 10: MASTER SERVING FUNCTION
# ==============================================================================
def serve_user(request: ServingRequest, model: XGBClassifier, engine: Engine) -> ServingResponse:
    wall_start = time.perf_counter()

    t0 = time.perf_counter()
    features = fetch_user_serving_features(request.user_id, engine)
    db_fetch_s = time.perf_counter() - t0

    if features.empty:
        print(f"[WARN] No historical entries found for user_id={request.user_id}.")
        return ServingResponse(request=request)

    avg_days = float(features["user_avg_days_between_orders"].iloc[0] if "user_avg_days_between_orders" in features.columns else 0.0)
    features = inject_temporal_context(features, request, avg_days)

    t0 = time.perf_counter()
    probs = run_inference(model, features)
    inference_s = time.perf_counter() - t0

    response = build_recommendation_report(request, features, probs, engine)

    total_s = time.perf_counter() - wall_start
    response.db_fetch_ms  = db_fetch_s  * 1_000
    response.inference_ms = inference_s * 1_000
    response.total_ms     = total_s     * 1_000

    return response

# ==============================================================================
# SECTION 11: MAIN ENTRYPOINT
# ==============================================================================
def main() -> None:
    print("=" * 78)
    print("  E-COMMERCE FEATURE STORE — REAL-TIME INFERENCE ENGINE")
    print("=" * 78)

    engine = build_engine(DB_CONFIG)
    model  = load_model(MODEL_PATH)

    request_1 = ServingRequest(user_id=14436, current_hour=10, current_dow=0, days_since_last_order=7.0)
    print(f"\n  SERVING REQUEST 1: User {request_1.user_id} (Saturday shop)")
    print("─" * 78)
    print_serving_report(serve_user(request_1, model, engine))

    request_2 = ServingRequest(user_id=159183, current_hour=19, current_dow=4, days_since_last_order=14.0)
    print(f"\n  SERVING REQUEST 2: User {request_2.user_id} (Midweek delay stock)")
    print("─" * 78)
    print_serving_report(serve_user(request_2, model, engine))

    engine.dispose()

if __name__ == "__main__":
    main()