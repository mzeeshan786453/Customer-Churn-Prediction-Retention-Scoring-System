"""
scoring_pipeline.py
-------------------
Weekly batch scoring pipeline.
- Loads model from disk
- Engineers features
- Scores all customers
- Writes results to outputs/scoring_results_<week>.csv
- Simulates a PostgreSQL insert (sqlite used locally; swap engine for Postgres)

PostgreSQL connection (production):
    from sqlalchemy import create_engine
    engine = create_engine("postgresql://user:pass@host:5432/churndb")

SQLite (local simulation):
    engine = create_engine("sqlite:///outputs/churn_scoring.db")
"""

import os
import sys
import json
import sqlite3
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from feature_engineering import load_and_clean, engineer_features, prepare_ml_features, FEATURE_COLS
from model_training import load_best_model, per_customer_explanation
from risk_segmentation import batch_score, summarise_risk_distribution, build_call_list

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH    = OUTPUT_DIR / "churn_scoring.db"


# ---------------------------------------------------------------------------
# SQLite schema (mirrors what you'd use in PostgreSQL)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS weekly_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    week_date           TEXT,
    customer_id         TEXT,
    tenure              INTEGER,
    monthly_charges     REAL,
    contract_type       TEXT,
    internet_service    TEXT,
    churn_probability   REAL,
    risk_segment        TEXT,
    top_churn_reasons   TEXT,
    recommended_actions TEXT,
    num_risk_factors    INTEGER,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weekly_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    week_date       TEXT,
    risk_segment    TEXT,
    count           INTEGER,
    count_pct       REAL,
    avg_churn_prob  REAL,
    avg_monthly_charges REAL,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"  DB initialised: {db_path}")


def store_scores(df_scored: pd.DataFrame, week_date: str, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    rows = []
    for _, row in df_scored.iterrows():
        rows.append((
            week_date,
            str(row.get("customerID", "")),
            int(row.get("tenure", 0)),
            float(row.get("MonthlyCharges", 0)),
            str(row.get("Contract", "")),
            str(row.get("InternetService", "")),
            float(row.get("churn_probability", 0)),
            str(row.get("risk_segment", "")),
            json.dumps(row.get("top_churn_reasons", [])),
            json.dumps(row.get("recommended_actions", [])),
            int(row.get("num_risk_factors", 0)),
        ))
    conn.executemany("""
        INSERT INTO weekly_scores (
            week_date, customer_id, tenure, monthly_charges,
            contract_type, internet_service,
            churn_probability, risk_segment,
            top_churn_reasons, recommended_actions, num_risk_factors
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"  Stored {len(rows)} records for week {week_date}")


def store_summary(summary_df: pd.DataFrame, week_date: str, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    for _, row in summary_df.iterrows():
        conn.execute("""
            INSERT INTO weekly_summary (
                week_date, risk_segment, count, count_pct,
                avg_churn_prob, avg_monthly_charges
            ) VALUES (?,?,?,?,?,?)
        """, (
            week_date,
            row["risk_segment"],
            int(row["count"]),
            float(row["count_pct"]),
            float(row["avg_churn_prob"]),
            float(row["avg_monthly_charges"]),
        ))
    conn.commit()
    conn.close()


def fetch_historical_summaries(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM weekly_summary ORDER BY week_date", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, week_date: str = None):
    if week_date is None:
        week_date = datetime.date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  CHURN SCORING PIPELINE — Week of {week_date}")
    print(f"{'='*60}")

    # 1. Load data
    print("\n[1/6] Loading and cleaning data ...")
    df_raw = load_and_clean(data_path)
    print(f"  Loaded {len(df_raw)} customers")

    # 2. Feature engineering
    print("\n[2/6] Engineering features ...")
    df_eng = engineer_features(df_raw)
    X, y, feature_cols = prepare_ml_features(df_eng)
    print(f"  Feature matrix: {X.shape}")

    # 3. Load model
    print("\n[3/6] Loading production model ...")
    model, model_name = load_best_model()
    print(f"  Model: {model_name}")

    # 4. Batch score
    print("\n[4/6] Scoring all customers ...")
    df_scored = batch_score(df_eng, model, feature_cols)
    avg_risk  = df_scored["churn_probability"].mean()
    print(f"  Mean churn probability: {avg_risk:.2%}")

    # 5. Risk summary
    print("\n[5/6] Risk segmentation summary:")
    summary = summarise_risk_distribution(df_scored)
    print(summary.to_string(index=False))

    # 6. Persist
    print("\n[6/6] Persisting results ...")
    init_db()
    store_scores(df_scored, week_date)
    store_summary(summary, week_date)

    # Export CSV
    csv_path = OUTPUT_DIR / f"scoring_results_{week_date}.csv"
    export_cols = [
        "customerID", "tenure", "MonthlyCharges", "Contract",
        "InternetService", "churn_probability", "risk_segment",
        "top_churn_reasons", "recommended_actions", "num_risk_factors",
    ]
    df_scored[export_cols].to_csv(csv_path, index=False)
    print(f"  CSV export: {csv_path}")

    # Call list
    call_list = build_call_list(df_scored)
    call_path = OUTPUT_DIR / f"call_list_{week_date}.csv"
    call_list.to_csv(call_path, index=False)
    print(f"  Call list: {call_path} ({len(call_list)} customers)")

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}\n")

    return df_scored, summary, call_list


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/telco_churn.csv"
    run_pipeline(data_path)
