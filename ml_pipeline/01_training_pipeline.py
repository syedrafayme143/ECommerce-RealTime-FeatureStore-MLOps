import sys
import time
import os
import warnings
import urllib.parse

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "ecommerce", 
    "user":     "postgres",
    "password": urllib.parse.quote_plus("Syedrafay123/"), # <-- This safely locks in your password slash!
}

SOURCE_VIEW = "v_ml_training_dataset"
IDENTIFIER_COLS = ["user_id", "product_id", "train_order_id"]
TARGET_COL = "is_reordered_target"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CHUNK_SIZE = 200_000

# Ensure the output folder structure exists safely
os.makedirs("ml_pipeline", exist_ok=True)
MODEL_OUTPUT_PATH = "ml_pipeline/xgb_reorder_model.json"


# ==============================================================================
# SECTION 2: DATABASE CONNECTION
# ==============================================================================
def build_engine(config: dict):
    connection_url = (
        f"postgresql+psycopg2://"
        f"{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}"
        f"/{config['dbname']}"
    )
    try:
        engine = create_engine(
            connection_url,
            pool_size=1,
            pool_recycle=1800,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[DB] Connected securely to '{config['dbname']}' on {config['host']}:{config['port']}")
        return engine
    except Exception as exc:
        print(f"\n[ERROR] Could not connect to PostgreSQL.\n  Detail: {exc}")
        sys.exit(1)


# ==============================================================================
# SECTION 3: MEMORY-EFFICIENT DATA INGESTION
# ==============================================================================
def ingest_training_data(engine, view_name: str, chunk_size: int) -> pd.DataFrame:
    query = f"SELECT * FROM {view_name} WHERE user_id % 4 = 0;"
    
    print(f"\n[INGEST] Downsampling active: Fetching ~25% of users to protect laptop RAM...")
    print(f"[INGEST] Reading from '{view_name}' in chunks of {chunk_size:,} rows...")

    chunks = []
    total_rows = 0
    t0 = time.time()

    try:
        for chunk in pd.read_sql(text(query), engine, chunksize=chunk_size):
            chunks.append(chunk)
            total_rows += len(chunk)
            elapsed = time.time() - t0
            print(f"  ...{total_rows:>10,} rows loaded  ({elapsed:.1f}s elapsed)", end="\r")

        df = pd.concat(chunks, ignore_index=True)
        elapsed = time.time() - t0
        print(f"\n[INGEST] Ingestion Complete: {len(df):,} rows × {len(df.columns)} columns in {elapsed:.1f}s")
        return df

    except Exception as exc:
        print(f"\n[ERROR] Data ingestion failed.\n  Detail: {exc}")
        sys.exit(1)


# ==============================================================================
# SECTION 4: FEATURE MATRIX PREPARATION
# ==============================================================================
def prepare_feature_matrix(df: pd.DataFrame):
    print(f"\n[PREP] Conditioning feature matrix data frames...")

    identifiers = df[IDENTIFIER_COLS].copy()
    groups      = df["user_id"].values

    y = df[TARGET_COL].astype(np.int8)

    drop_cols = IDENTIFIER_COLS + [TARGET_COL]
    X = df.drop(columns=drop_cols)

    null_counts = X.isnull().sum()
    null_cols   = null_counts[null_counts > 0]

    if not null_cols.empty:
        print(f"  [WARN] NULL values detected in {len(null_cols)} feature column(s). Filling with 0.")
        X = X.fillna(0)
    else:
        print("  [OK] Zero NULL values discovered in training schema matrix.")

    pos_rate = y.mean() * 100
    print(f"  Feature matrix template shape : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Target class layout distribution: {y.sum():,} labels (1) / {(y == 0).sum():,} labels (0) ({pos_rate:.2f}% true positives)")

    return identifiers, X, y, groups, list(X.columns)


# ==============================================================================
# SECTION 5: GROUP-AWARE TRAIN / TEST SPLIT
# ==============================================================================
def group_aware_split(X, y, groups):
    print(f"\n[SPLIT] Executing group-aware train/validation split (test_size={TEST_SIZE}, grouped by user_id)...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_users = set(groups[train_idx])
    test_users  = set(groups[test_idx])
    overlap     = train_users & test_users
    assert len(overlap) == 0, "[FATAL] User overlap split safety contract breached!"

    print(f"  Train Set Balance : {len(X_train):>10,} rows | {len(train_users):,} unique users")
    print(f"  Validation Set Balance : {len(X_test):>10,} rows | {len(test_users):,} unique users")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 6: CLASS IMBALANCE HANDLING
# ==============================================================================
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    weight = neg / pos
    print(f"\n[IMBALANCE] Computed scale_pos_weight ratios = {neg:,} / {pos:,} = {weight:.4f}")
    return float(weight)


# ==============================================================================
# SECTION 7: MODEL TRAINING (FIXED FOR MODERN XGBOOST)
# ==============================================================================
def train_model(X_train, y_train, X_test, y_test, scale_pos_weight: float) -> XGBClassifier:
    print("\n[TRAIN] Instantiating advanced XGBClassifier parameters...")

    model = XGBClassifier(
        objective          = "binary:logistic",
        eval_metric        = "auc",
        scale_pos_weight   = scale_pos_weight,
        n_estimators       = 500,
        max_depth          = 6,
        learning_rate      = 0.1,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 10,
        gamma              = 1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        tree_method        = "hist", 
        random_state       = RANDOM_STATE,
        n_jobs             = -1, 
        verbosity          = 1,
        early_stopping_rounds = 30, # <-- FIXED: Moved early stopping right here!
    )

    print("[TRAIN] Fitting engine booster (Early stopping tracking independent validation set, patience=30)...")
    t0 = time.time()

    # FIXED: Cleaned up fit parameters to match modern XGBoost syntax
    model.fit(
        X_train,
        y_train,
        eval_set = [(X_test, y_test)], 
        verbose  = 50, # prints evaluation metrics every 50 rounds
    )

    elapsed = time.time() - t0
    best_round = model.best_iteration
    print(f"\n[TRAIN] Optimization complete in {elapsed:.1f} seconds.")
    print(f"[TRAIN] Target plateau reached at iteration round: {best_round}")

    return model


# ==============================================================================
# SECTION 8: MODEL EVALUATION
# ==============================================================================
def evaluate_model(model: XGBClassifier, X_test, y_test, feature_names: list):
    print("\n" + "=" * 72)
    print("  MODEL EVALUATION REPORT")
    print("=" * 72)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n  CONFUSION MATRIX (threshold = 0.50)")
    print(f"  {'':20s}  Predicted 0   Predicted 1")
    print(f"  {'Actual 0 (No Reorder)':20s}  {tn:>10,}    {fp:>10,}  (FP)")
    print(f"  {'Actual 1 (Reorder)':20s}  {fn:>10,} (FN)  {tp:>10,}")

    print("\n  CLASSIFICATION REPORT")
    print("  " + "-" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=["Not Reordered (0)", "Reordered (1)"],
        digits=4
    )
    for line in report.split("\n"):
        print(f"  {line}")

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n  ROC-AUC SCORE : {auc:.6f}")
    
    if auc >= 0.85: tier = "EXCELLENT — production-grade reorder ranking quality"
    elif auc >= 0.80: tier = "GOOD — strong baseline, consider hyperparameter tuning"
    elif auc >= 0.75: tier = "FAIR — functional but investigate feature coverage"
    else: tier = "NEEDS REVIEW — check for target leakage or data issues"
    print(f"  Performance   : {tier}")

    print("\n" + "=" * 72)
    print("  FEATURE IMPORTANCE REPORT (metric: gain)")
    print("=" * 72)

    # FIXED: Robust dictionary translation fallback protection
    raw_scores = model.get_booster().get_score(importance_type="gain")
    importances = {feat: raw_scores.get(feat, 0.0) for feat in feature_names}

    imp_df = pd.DataFrame({
        "feature": list(importances.keys()),
        "gain":    list(importances.values())
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    imp_df["rank"]          = imp_df.index + 1
    total_gain = imp_df["gain"].sum()
    imp_df["gain_pct"]      = (imp_df["gain"] / total_gain * 100) if total_gain > 0 else 0
    imp_df["gain_pct_cum"]  = imp_df["gain_pct"].cumsum()

    print(f"\n  {'Rank':<5} {'Feature':<45} {'Gain':>12}  {'%':>7}  {'Cumul%':>8}")
    print("  " + "-" * 82)
    for _, row in imp_df.iterrows():
        marker = "  ◄ top 80%" if row["gain_pct_cum"] <= 80 else ""
        print(f"  {int(row['rank']):<5} {row['feature']:<45} {row['gain']:>12.2f}  {row['gain_pct']:>6.2f}%  {row['gain_pct_cum']:>7.2f}%{marker}")

    return auc, imp_df


# ==============================================================================
# SECTION 9: MODEL PERSISTENCE
# ==============================================================================
def save_model(model: XGBClassifier, path: str):
    try:
        model.save_model(path)
        print(f"\n[SAVE] Model successfully serialized to disk file format: {path}")
    except Exception as exc:
        print(f"\n[WARN] Failed serialization step at target route '{path}': {exc}")


# ==============================================================================
# SECTION 10: MAIN SYSTEM ENTRYPOINT
# ==============================================================================
def main():
    print("=" * 72)
    print("  E-COMMERCE FEATURE STORE — ML TRAINING PIPELINE")
    print("  Phase 5: XGBoost Reorder Prediction Model")
    print("=" * 72)

    pipeline_start = time.time()

    engine = build_engine(DB_CONFIG)
    df = ingest_training_data(engine, SOURCE_VIEW, CHUNK_SIZE)
    engine.dispose() 

    identifiers, X, y, groups, feature_names = prepare_feature_matrix(df)
    del df 

    X_train, X_test, y_train, y_test = group_aware_split(X, y, groups)
    del X, y, groups 

    spw = compute_scale_pos_weight(y_train)

    # FIXED: Added evaluation test parameters into train_model call sequence
    model = train_model(X_train, y_train, X_test, y_test, spw)
    del X_train, y_train 

    auc, importance_df = evaluate_model(model, X_test, y_test, feature_names)
    save_model(model, MODEL_OUTPUT_PATH)

    total_time = time.time() - pipeline_start
    print(f"\n[DONE] Full pipeline executed cleanly in {total_time:.1f}s")
    print(f"[DONE] Verified Final ROC-AUC Metric: {auc:.6f}")


if __name__ == "__main__":
    main()