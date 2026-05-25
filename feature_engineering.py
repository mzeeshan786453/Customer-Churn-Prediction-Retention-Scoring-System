"""
feature_engineering.py
-----------------------
Creates meaningful features from raw Telco customer data.
Covers usage patterns, billing history, lifecycle stage, and trend-based features.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Load & basic cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, coerce types, handle missing values."""
    df = pd.read_csv(path)

    # TotalCharges can be ' ' for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Binary target
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with all engineered features."""
    df = df.copy()

    # --- Lifecycle / tenure stage ---
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["New (0-12m)", "Growing (1-2yr)", "Mature (2-4yr)", "Loyal (4yr+)"],
    )
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_long_term"] = (df["tenure"] >= 36).astype(int)

    # --- Billing features ---
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["billing_spike"] = (
        (df["MonthlyCharges"] - df["avg_monthly_spend"])
        / df["avg_monthly_spend"].replace(0, 1)
    )
    df["high_monthly_charges"] = (df["MonthlyCharges"] > 80).astype(int)
    df["charge_per_service"] = df["MonthlyCharges"] / (
        df["tenure"].replace(0, 1)
    )  # proxy for value density

    # --- Service count ---
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["MultipleLines"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    # --- Support / risk signals ---
    df["no_online_security"] = (df["OnlineSecurity"] == "No").astype(int)
    df["no_tech_support"] = (df["TechSupport"] == "No").astype(int)
    df["no_device_protection"] = (df["DeviceProtection"] == "No").astype(int)

    # --- Contract risk ---
    df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_two_year"] = (df["Contract"] == "Two year").astype(int)

    # --- Payment method risk ---
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_auto_pay"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    # --- Demographics ---
    df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
    df["has_partner"] = (df["Partner"] == "Yes").astype(int)
    df["is_senior"] = df["SeniorCitizen"].astype(int)

    # --- Engagement score (higher = more engaged = lower churn risk) ---
    df["engagement_score"] = (
        df["num_services"] * 2
        + df["is_long_term"] * 3
        + df["is_auto_pay"] * 2
        + df["is_two_year"] * 3
        - df["is_monthly_contract"] * 2
        - df["is_electronic_check"] * 1
        - df["no_online_security"] * 1
    )

    # --- Paperless billing ---
    df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Encode for ML
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "is_new_customer",
    "is_long_term",
    "avg_monthly_spend",
    "billing_spike",
    "high_monthly_charges",
    "num_services",
    "has_internet",
    "is_fiber",
    "no_online_security",
    "no_tech_support",
    "no_device_protection",
    "is_monthly_contract",
    "is_two_year",
    "is_electronic_check",
    "is_auto_pay",
    "has_dependents",
    "has_partner",
    "is_senior",
    "engagement_score",
    "is_paperless",
]


def prepare_ml_features(df: pd.DataFrame):
    """Return X (features), y (target), and feature names."""
    X = df[FEATURE_COLS].fillna(0)
    y = df["Churn_Binary"]
    return X, y, FEATURE_COLS


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.generate_dataset import generate_telco_dataset

    raw = generate_telco_dataset()
    df = load_and_clean_from_df(raw)
    df = engineer_features(df)
    X, y, feat_names = prepare_ml_features(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Features: {feat_names}")
