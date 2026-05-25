"""
train.py
--------
End-to-end training runner.
Run once to train models, generate plots, and save the best model to disk.

Usage:
    python train.py                       # uses auto-generated synthetic data
    python train.py path/to/churn.csv     # uses real dataset
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from data.generate_dataset import generate_telco_dataset
from feature_engineering import load_and_clean, engineer_features, prepare_ml_features, FEATURE_COLS
from model_training import (
    train_and_compare,
    compute_feature_importance,
    per_customer_explanation,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_comparison,
    save_best_model,
)
from risk_segmentation import batch_score, summarise_risk_distribution, build_call_list

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path=None):
    if path and os.path.exists(path):
        print(f"  Loading real dataset: {path}")
        return load_and_clean(path)
    else:
        print("  Generating synthetic Telco dataset ...")
        raw = generate_telco_dataset()
        # Save locally for pipeline use
        csv_path = PROJECT_DIR / "data" / "telco_churn.csv"
        csv_path.parent.mkdir(exist_ok=True)
        raw.to_csv(csv_path, index=False)
        return load_and_clean(str(csv_path))


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else None

    print_section("CUSTOMER CHURN PREDICTION & RETENTION SCORING SYSTEM")
    print("  Teyzix Core Internship — Task DS-3")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print_section("Step 1: Data Loading & Feature Engineering")
    df_raw = load_data(data_path)
    df_eng = engineer_features(df_raw)
    X, y, feature_names = prepare_ml_features(df_eng)
    print(f"  Customers:    {len(df_eng):,}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  Churn rate:   {y.mean():.2%}")
    print(f"\n  Engineered features:")
    for f in feature_names:
        print(f"    • {f}")

    # ── 2. Model training ────────────────────────────────────────────────────
    print_section("Step 2: Model Training & Comparison")
    train_results = train_and_compare(X, y, feature_names)

    results_df  = train_results["results_df"]
    fitted      = train_results["fitted"]
    best_model  = train_results["best_model"]
    best_name   = train_results["best_name"]
    X_test      = train_results["X_test"]
    y_test      = train_results["y_test"]

    print(f"\n  Model comparison:")
    print(results_df.to_string(index=False))

    # ── 3. Plots ─────────────────────────────────────────────────────────────
    print_section("Step 3: Generating Evaluation Plots")

    plot_roc_curves(fitted, X_test, y_test, OUTPUT_DIR / "roc_curves.png")
    plot_metric_comparison(results_df, OUTPUT_DIR / "model_comparison.png")

    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm   = confusion_matrix(y_test, y_pred)
        slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")[:40]
        plot_confusion_matrix(cm, name, OUTPUT_DIR / f"cm_{slug}.png")
    print("  Confusion matrices saved.")

    # ── 4. Feature importance (SHAP proxy) ──────────────────────────────────
    print_section("Step 4: Feature Importance (SHAP-proxy)")
    imp_df = compute_feature_importance(best_model, X_test, y_test, feature_names)
    plot_feature_importance(imp_df, OUTPUT_DIR / "feature_importance.png")
    imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print(f"\n  Top 10 churn drivers:")
    print(imp_df.head(10).to_string(index=False))

    # ── 5. Per-customer SHAP explanations ──────────────────────────────────
    print_section("Step 5: Per-Customer SHAP Explanations (sample)")
    df_scored_sample = batch_score(df_eng.head(200), best_model, feature_names)
    high_risk = df_scored_sample[df_scored_sample["risk_segment"] == "HIGH"]

    shap_records = []
    print(f"\n  Sample HIGH-risk customer explanations:")
    for _, cust in high_risk.head(5).iterrows():
        cust_id = cust["customerID"]
        prob    = cust["churn_probability"]
        reasons = cust["top_churn_reasons"]
        actions = cust["recommended_actions"]
        print(f"\n  ── {cust_id} ──")
        print(f"     Churn Prob:  {prob:.2%}")
        print(f"     Risk Tier:   HIGH")
        print(f"     Why at risk: {'; '.join(reasons)}")
        print(f"     Recommended: {'; '.join(actions)}")
        shap_records.append({
            "customerID": cust_id,
            "churn_probability": round(prob, 4),
            "risk_segment": "HIGH",
            "top_churn_reasons": reasons,
            "recommended_actions": actions,
        })

    with open(OUTPUT_DIR / "sample_shap_explanations.json", "w") as f:
        json.dump(shap_records, f, indent=2)
    print(f"\n  SHAP explanations saved to outputs/sample_shap_explanations.json")

    # ── 6. Weekly scoring simulation ────────────────────────────────────────
    print_section("Step 6: Weekly Batch Scoring Simulation")
    save_best_model(best_model, best_name)

    # Simulate 3 weeks
    import datetime, copy, random
    weeks = [
        (datetime.date(2026, 5, 4), "Week 1"),
        (datetime.date(2026, 5, 11), "Week 2"),
        (datetime.date(2026, 5, 18), "Week 3"),
    ]
    all_weekly = []
    for week_dt, label in weeks:
        week_str  = week_dt.strftime("%Y-%m-%d")
        df_week   = batch_score(df_eng, best_model, feature_names)
        # Add small noise to simulate drift
        df_week["churn_probability"] = np.clip(
            df_week["churn_probability"] + np.random.normal(0, 0.015, len(df_week)), 0.01, 0.99
        )
        df_week["risk_segment"] = df_week["churn_probability"].apply(
            lambda p: "HIGH" if p >= 0.65 else ("MEDIUM" if p >= 0.35 else "LOW")
        )
        summary   = summarise_risk_distribution(df_week)
        summary["week"] = week_str
        all_weekly.append(summary)
        print(f"\n  {label} ({week_str}):")
        print(summary[["risk_segment", "count", "count_pct", "avg_churn_prob"]].to_string(index=False))

    weekly_hist = pd.concat(all_weekly, ignore_index=True)
    weekly_hist.to_csv(OUTPUT_DIR / "weekly_scoring_history.csv", index=False)
    print(f"\n  Weekly history saved.")

    # ── 7. Segment visualisation ────────────────────────────────────────────
    print_section("Step 7: Risk Segmentation Visualisation")
    df_all_scored = batch_score(df_eng, best_model, feature_names)
    seg_counts = df_all_scored["risk_segment"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    colors = {"HIGH": "#DC2626", "MEDIUM": "#F59E0B", "LOW": "#16A34A"}
    pie_colors = [colors.get(s, "#6B7280") for s in seg_counts.index]
    axes[0].pie(
        seg_counts.values, labels=seg_counts.index,
        autopct="%1.1f%%", colors=pie_colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    axes[0].set_title("Customer Risk Distribution", fontsize=13, fontweight="bold")

    # Histogram of churn probabilities
    for seg, color in colors.items():
        subset = df_all_scored[df_all_scored["risk_segment"] == seg]["churn_probability"]
        axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=seg, edgecolor="white")
    axes[1].axvline(0.35, color="#F59E0B", linestyle="--", lw=1.5, label="MEDIUM threshold")
    axes[1].axvline(0.65, color="#DC2626", linestyle="--", lw=1.5, label="HIGH threshold")
    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("Number of Customers")
    axes[1].set_title("Churn Probability Distribution by Segment", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_segmentation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Risk segmentation chart saved.")

    # Call list
    call_list = build_call_list(df_all_scored)
    call_list.to_csv(OUTPUT_DIR / "priority_call_list.csv", index=False)
    print(f"  Priority call list: {len(call_list)} customers → outputs/priority_call_list.csv")

    # ── 8. Revenue impact ───────────────────────────────────────────────────
    print_section("Step 8: Business Impact Analysis")
    high_risk_df = df_all_scored[df_all_scored["risk_segment"] == "HIGH"]
    avg_monthly  = high_risk_df["MonthlyCharges"].mean()
    n_high       = len(high_risk_df)
    # Assume 30% retention rate from interventions, avg 12-month LTV
    retention_rate     = 0.30
    saved_customers    = int(n_high * retention_rate)
    monthly_saved      = round(saved_customers * avg_monthly, 2)
    annual_saved       = round(monthly_saved * 12, 2)
    clv_per_customer   = round(avg_monthly * 24, 2)  # 24-month CLV

    impact = {
        "high_risk_customers":      n_high,
        "assumed_retention_rate":   "30%",
        "customers_retained":       saved_customers,
        "avg_monthly_charges":      avg_monthly,
        "monthly_revenue_saved":    monthly_saved,
        "annual_revenue_saved":     annual_saved,
        "avg_clv_per_customer":     clv_per_customer,
    }

    print(f"\n  HIGH-risk customers identified:  {n_high:,}")
    print(f"  Assumed retention success rate:  30%")
    print(f"  Customers retained:              {saved_customers:,}")
    print(f"  Monthly revenue saved:           ${monthly_saved:,.2f}")
    print(f"  Annual revenue saved:            ${annual_saved:,.2f}")
    print(f"  Avg CLV per retained customer:   ${clv_per_customer:,.2f}")

    with open(OUTPUT_DIR / "business_impact.json", "w") as f:
        json.dump(impact, f, indent=2)

    # ── Done ─────────────────────────────────────────────────────────────────
    print_section("Training Complete")
    print("  Outputs generated:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    • {p.name}")
    print()


if __name__ == "__main__":
    main()
