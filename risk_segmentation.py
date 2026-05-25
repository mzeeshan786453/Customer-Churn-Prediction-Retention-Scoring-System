"""
risk_segmentation.py
--------------------
Classifies customers into HIGH / MEDIUM / LOW risk segments and
generates actionable, SHAP-driven intervention recommendations.
"""

import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Segment thresholds (business-driven)
# ---------------------------------------------------------------------------

HIGH_THRESHOLD   = 0.65   # ≥ 65% churn probability → call required
MEDIUM_THRESHOLD = 0.35   # 35-64% → email / promotion
# < 35% → LOW → monitor only


def assign_risk_segment(churn_prob: float) -> str:
    if churn_prob >= HIGH_THRESHOLD:
        return "HIGH"
    elif churn_prob >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# 2. Intervention rule engine
# ---------------------------------------------------------------------------

INTERVENTION_RULES = {
    # Contract risk
    "is_monthly_contract": {
        "condition": lambda v: v == 1,
        "reason":    "Month-to-month contract — no long-term commitment",
        "action":    "Offer 20% discount to upgrade to annual contract",
        "priority":  10,
    },
    # Payment method
    "is_electronic_check": {
        "condition": lambda v: v == 1,
        "reason":    "High-risk payment method (electronic check)",
        "action":    "Offer bill credit to switch to auto-pay",
        "priority":  8,
    },
    # New / short tenure
    "is_new_customer": {
        "condition": lambda v: v == 1,
        "reason":    "New customer (≤ 6 months) — high early-life churn risk",
        "action":    "Assign onboarding specialist; proactive welcome call",
        "priority":  9,
    },
    # No tech support
    "no_tech_support": {
        "condition": lambda v: v == 1,
        "reason":    "No active tech support subscription",
        "action":    "Offer 3-month free TechSupport trial",
        "priority":  7,
    },
    # No online security
    "no_online_security": {
        "condition": lambda v: v == 1,
        "reason":    "No online security service",
        "action":    "Offer bundled security package at reduced rate",
        "priority":  6,
    },
    # Fiber optic (high churn segment)
    "is_fiber": {
        "condition": lambda v: v == 1,
        "reason":    "Fiber optic customers show elevated churn",
        "action":    "Conduct satisfaction survey; offer loyalty reward",
        "priority":  7,
    },
    # High charges
    "high_monthly_charges": {
        "condition": lambda v: v == 1,
        "reason":    "Monthly charges above $80 — price sensitivity risk",
        "action":    "Offer a customized cost-saving bundle",
        "priority":  8,
    },
    # No device protection
    "no_device_protection": {
        "condition": lambda v: v == 1,
        "reason":    "No device protection subscription",
        "action":    "Provide free 1-month device protection trial",
        "priority":  5,
    },
    # Senior citizen
    "is_senior": {
        "condition": lambda v: v == 1,
        "reason":    "Senior citizen — may need assisted support",
        "action":    "Assign senior customer care line; simplified plan review",
        "priority":  6,
    },
    # No partner / dependents (lone customer, easier to churn)
    "has_partner": {
        "condition": lambda v: v == 0,
        "reason":    "Single-account customer — lower switching barrier",
        "action":    "Offer family plan discount to incentivize bundling",
        "priority":  4,
    },
    # Auto-pay gap
    "is_auto_pay": {
        "condition": lambda v: v == 0,
        "reason":    "Not on automatic payment — payment friction risk",
        "action":    "Encourage auto-pay enrollment with a $5 monthly credit",
        "priority":  5,
    },
    # Low engagement
    "engagement_score": {
        "condition": lambda v: v < 3,
        "reason":    "Low engagement score — under-utilizing services",
        "action":    "Personal outreach call to review service utilization",
        "priority":  6,
    },
}


def generate_intervention(customer_row: pd.Series, top_n: int = 3) -> dict:
    """
    Given a single customer row (with engineered features),
    return top-3 churn reasons and a recommended action plan.
    """
    triggered = []
    for feature, rule in INTERVENTION_RULES.items():
        val = customer_row.get(feature, np.nan)
        if pd.isna(val):
            continue
        try:
            if rule["condition"](val):
                triggered.append({
                    "feature":  feature,
                    "reason":   rule["reason"],
                    "action":   rule["action"],
                    "priority": rule["priority"],
                })
        except Exception:
            continue

    # Sort by priority descending, take top N
    triggered.sort(key=lambda x: -x["priority"])
    top = triggered[:top_n]

    reasons = [t["reason"] for t in top]
    actions = list(dict.fromkeys([t["action"] for t in top]))  # deduplicate

    return {
        "top_churn_reasons":    reasons,
        "recommended_actions":  actions,
        "num_risk_factors":     len(triggered),
    }


# ---------------------------------------------------------------------------
# 3. Batch scoring
# ---------------------------------------------------------------------------

def batch_score(df_engineered: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    """
    Score all customers, assign risk segment, generate interventions.
    Returns a scored DataFrame with all retention intelligence columns.
    """
    X = df_engineered[feature_cols].fillna(0)
    churn_proba = model.predict_proba(X)[:, 1]

    df_scored = df_engineered.copy()
    df_scored["churn_probability"] = churn_proba
    df_scored["risk_segment"] = df_scored["churn_probability"].apply(assign_risk_segment)

    interventions = df_scored.apply(generate_intervention, axis=1)
    df_scored["top_churn_reasons"]   = [i["top_churn_reasons"]   for i in interventions]
    df_scored["recommended_actions"] = [i["recommended_actions"]  for i in interventions]
    df_scored["num_risk_factors"]    = [i["num_risk_factors"]     for i in interventions]

    return df_scored


def summarise_risk_distribution(df_scored: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_scored.groupby("risk_segment")
        .agg(
            count=("customerID", "count"),
            avg_churn_prob=("churn_probability", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
    )
    summary["count_pct"] = (summary["count"] / summary["count"].sum() * 100).round(1)
    return summary


def build_call_list(df_scored: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Return top-N highest-risk customers for the weekly call list."""
    cols = [
        "customerID", "tenure", "MonthlyCharges", "Contract",
        "InternetService", "churn_probability", "risk_segment",
        "top_churn_reasons", "recommended_actions",
    ]
    call_list = (
        df_scored[df_scored["risk_segment"] == "HIGH"]
        [cols]
        .sort_values("churn_probability", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    call_list["rank"] = call_list.index + 1
    return call_list
