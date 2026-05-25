"""
dashboard/app.py
----------------
Streamlit analytics dashboard for the Churn Prediction & Retention Scoring System.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import json
import pickle
import sqlite3
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR  = ROOT / "models"
DB_PATH    = OUTPUT_DIR / "churn_scoring.db"

sys.path.insert(0, str(ROOT))
from feature_engineering import load_and_clean, engineer_features, prepare_ml_features, FEATURE_COLS
from risk_segmentation import batch_score, summarise_risk_distribution, build_call_list, INTERVENTION_RULES


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {"HIGH": "#DC2626", "MEDIUM": "#F59E0B", "LOW": "#16A34A"}


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = MODEL_DIR / "best_model.pkl"
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["model"], obj["name"]
    return None, None


# ── Load & score data ─────────────────────────────────────────────────────────
@st.cache_data
def load_scored_data(data_path=None):
    if data_path is None:
        data_path = str(ROOT / "data" / "telco_churn.csv")
    model, model_name = load_model()
    if model is None:
        return None, None, None, None

    df_raw = load_and_clean(data_path)
    df_eng = engineer_features(df_raw)
    X, y, feature_cols = prepare_ml_features(df_eng)
    df_scored = batch_score(df_eng, model, feature_cols)
    return df_scored, model, model_name, feature_cols


# ── Load outputs ──────────────────────────────────────────────────────────────
@st.cache_data
def load_outputs():
    results = {}
    impact_path = OUTPUT_DIR / "business_impact.json"
    if impact_path.exists():
        with open(impact_path) as f:
            results["impact"] = json.load(f)
    feat_path = OUTPUT_DIR / "feature_importance.csv"
    if feat_path.exists():
        results["feature_importance"] = pd.read_csv(feat_path)
    weekly_path = OUTPUT_DIR / "weekly_scoring_history.csv"
    if weekly_path.exists():
        results["weekly_history"] = pd.read_csv(weekly_path)
    shap_path = OUTPUT_DIR / "sample_shap_explanations.json"
    if shap_path.exists():
        with open(shap_path) as f:
            results["shap_samples"] = json.load(f)
    call_path = OUTPUT_DIR / "priority_call_list.csv"
    if call_path.exists():
        results["call_list"] = pd.read_csv(call_path)
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=TEYZIX+CORE", width=180)
    st.title("Churn Intelligence")
    st.caption("DS-3 | Teyzix Core Internship")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🎯 Risk Segments",
            "📈 Model Performance",
            "🔍 Feature Importance",
            "👤 Customer Drill-Down",
            "📋 Call List",
            "💰 Business Impact",
        ],
    )
    st.divider()
    st.caption(f"Last updated: {datetime.date.today()}")


# ── Load data ─────────────────────────────────────────────────────────────────
df_scored, model, model_name, feature_cols = load_scored_data()
outputs = load_outputs()

if df_scored is None:
    st.error("⚠️ Model not found. Please run `python train.py` first.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Customer Churn Intelligence Dashboard")
    st.caption(f"Model: **{model_name}** | Customers scored: **{len(df_scored):,}**")

    seg_counts = df_scored["risk_segment"].value_counts()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers",    f"{len(df_scored):,}")
    col2.metric("Avg Churn Prob",      f"{df_scored['churn_probability'].mean():.1%}")
    col3.metric("🔴 HIGH Risk",        f"{seg_counts.get('HIGH', 0):,}",
                delta=f"{seg_counts.get('HIGH',0)/len(df_scored)*100:.1f}%")
    col4.metric("🟡 MEDIUM Risk",      f"{seg_counts.get('MEDIUM', 0):,}",
                delta=f"{seg_counts.get('MEDIUM',0)/len(df_scored)*100:.1f}%")
    col5.metric("🟢 LOW Risk",         f"{seg_counts.get('LOW', 0):,}",
                delta=f"{seg_counts.get('LOW',0)/len(df_scored)*100:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    # Risk pie
    with col_left:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        segs = ["HIGH", "MEDIUM", "LOW"]
        vals = [seg_counts.get(s, 0) for s in segs]
        pie_colors = [COLORS[s] for s in segs]
        ax.pie(vals, labels=segs, autopct="%1.1f%%", colors=pie_colors,
               startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
               textprops={"fontsize": 12})
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    # Churn prob histogram
    with col_right:
        st.subheader("Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        for seg, color in COLORS.items():
            subset = df_scored[df_scored["risk_segment"] == seg]["churn_probability"]
            ax.hist(subset, bins=30, alpha=0.65, color=color, label=seg, edgecolor="white")
        ax.axvline(0.35, color="#F59E0B", linestyle="--", lw=1.5, label="MEDIUM line")
        ax.axvline(0.65, color="#DC2626", linestyle="--", lw=1.5, label="HIGH line")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("# Customers")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()

    # Weekly trend
    if "weekly_history" in outputs:
        st.subheader("Weekly Risk Trend")
        wh = outputs["weekly_history"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for seg, color in COLORS.items():
            sub = wh[wh["risk_segment"] == seg]
            ax.plot(sub["week"], sub["count"], marker="o", color=color, label=seg, lw=2)
        ax.set_xlabel("Week")
        ax.set_ylabel("Customer Count")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Risk Segments
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Risk Segments":
    st.title("🎯 Risk Segmentation Engine")

    seg = st.selectbox("Filter by segment", ["ALL", "HIGH", "MEDIUM", "LOW"])
    display_df = df_scored if seg == "ALL" else df_scored[df_scored["risk_segment"] == seg]

    summary = summarise_risk_distribution(df_scored)
    st.dataframe(summary.style.format({
        "avg_churn_prob": "{:.2%}",
        "avg_monthly_charges": "${:.2f}",
        "count_pct": "{:.1f}%",
    }), use_container_width=True)

    st.subheader(f"Customers — {seg} ({len(display_df):,})")
    cols_show = ["customerID", "tenure", "MonthlyCharges", "Contract",
                 "InternetService", "churn_probability", "risk_segment"]
    st.dataframe(
        display_df[cols_show]
        .sort_values("churn_probability", ascending=False)
        .head(200)
        .style.format({"churn_probability": "{:.2%}", "MonthlyCharges": "${:.2f}"}),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    roc_path = OUTPUT_DIR / "roc_curves.png"
    cmp_path = OUTPUT_DIR / "model_comparison.png"

    col1, col2 = st.columns(2)
    if roc_path.exists():
        col1.subheader("ROC Curves")
        col1.image(str(roc_path), use_container_width=True)
    if cmp_path.exists():
        col2.subheader("Metric Comparison")
        col2.image(str(cmp_path), use_container_width=True)

    st.subheader("Confusion Matrices")
    cms = sorted(OUTPUT_DIR.glob("cm_*.png"))
    cols = st.columns(len(cms))
    for i, cm_path in enumerate(cms):
        cols[i].image(str(cm_path), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance (SHAP Proxy)")
    st.info("Permutation importance is used as a faithful SHAP proxy. "
            "Swap in `shap.TreeExplainer` when the SHAP library is available.")

    fi_path = OUTPUT_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_container_width=True)

    if "feature_importance" in outputs:
        st.subheader("Feature Importance Table")
        fi_df = outputs["feature_importance"]
        st.dataframe(
            fi_df.style.format({"importance": "{:.4f}", "std": "{:.4f}"}),
            use_container_width=True,
        )

    st.subheader("Sample SHAP Explanations")
    if "shap_samples" in outputs:
        for s in outputs["shap_samples"]:
            with st.expander(f"🔍 {s['customerID']} — Churn Prob: {s['churn_probability']:.2%}  |  {s['risk_segment']}"):
                st.markdown("**Why at risk:**")
                for r in s["top_churn_reasons"]:
                    st.markdown(f"  - ⚠️ {r}")
                st.markdown("**Recommended actions:**")
                for a in s["recommended_actions"]:
                    st.markdown(f"  - ✅ {a}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Customer Drill-Down
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👤 Customer Drill-Down":
    st.title("👤 Customer Drill-Down")

    customer_id = st.selectbox(
        "Select Customer ID",
        df_scored["customerID"].sort_values().tolist(),
    )

    cust = df_scored[df_scored["customerID"] == customer_id].iloc[0]
    prob = cust["churn_probability"]
    seg  = cust["risk_segment"]
    seg_color = COLORS.get(seg, "#6B7280")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Probability", f"{prob:.2%}")
    col2.metric("Risk Segment",      seg)
    col3.metric("Tenure (months)",   int(cust["tenure"]))
    col4.metric("Monthly Charges",   f"${cust['MonthlyCharges']:.2f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Profile")
        profile = {
            "Contract":         cust.get("Contract", ""),
            "Internet Service": cust.get("InternetService", ""),
            "Payment Method":   cust.get("PaymentMethod", ""),
            "Senior Citizen":   "Yes" if cust.get("is_senior") else "No",
            "Partner":          "Yes" if cust.get("has_partner") else "No",
            "Dependents":       "Yes" if cust.get("has_dependents") else "No",
            "Paperless Billing":"Yes" if cust.get("is_paperless") else "No",
            "Total Charges":    f"${cust.get('TotalCharges', 0):.2f}",
            "Num Services":     int(cust.get("num_services", 0)),
            "Engagement Score": int(cust.get("engagement_score", 0)),
        }
        st.table(pd.DataFrame.from_dict(profile, orient="index", columns=["Value"]))

    with col_b:
        st.subheader("Churn Intelligence")
        reasons = cust.get("top_churn_reasons", [])
        actions = cust.get("recommended_actions", [])
        if isinstance(reasons, str):
            import ast
            reasons = ast.literal_eval(reasons)
            actions = ast.literal_eval(actions)

        st.markdown("**Top 3 Churn Reasons:**")
        for i, r in enumerate(reasons, 1):
            st.markdown(f"  {i}. ⚠️ {r}")

        st.markdown("**Recommended Action Plan:**")
        for i, a in enumerate(actions, 1):
            st.markdown(f"  {i}. ✅ {a}")

        # Risk gauge
        st.subheader("Risk Gauge")
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh([0], [1], color="#e5e7eb", height=0.4, edgecolor="none")
        ax.barh([0], [prob], color=seg_color, height=0.4, edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.35, 0.65, 1])
        ax.set_xticklabels(["0%", "35%", "65%", "100%"])
        ax.axvline(0.35, color="#F59E0B", lw=1.5, linestyle="--")
        ax.axvline(0.65, color="#DC2626", lw=1.5, linestyle="--")
        ax.set_title(f"Churn Probability: {prob:.1%}", fontsize=11)
        ax.grid(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Call List
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Call List":
    st.title("📋 Weekly Priority Call List")
    st.caption("Top 50 HIGH-risk customers requiring immediate retention contact.")

    if "call_list" in outputs:
        cl = outputs["call_list"]
        st.metric("Total HIGH-risk customers on this list", len(cl))
        st.dataframe(
            cl[["rank", "customerID", "tenure", "MonthlyCharges",
                "Contract", "churn_probability", "top_churn_reasons", "recommended_actions"]]
            .style.format({
                "churn_probability": "{:.2%}",
                "MonthlyCharges": "${:.2f}",
            }),
            use_container_width=True,
        )
        csv = cl.to_csv(index=False).encode()
        st.download_button("📥 Download Call List CSV", csv,
                           "priority_call_list.csv", "text/csv")
    else:
        st.warning("Call list not found. Run `python train.py` first.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Business Impact
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰 Business Impact":
    st.title("💰 Business Impact Analysis")

    if "impact" in outputs:
        imp = outputs["impact"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("HIGH-Risk Customers",      f"{imp['high_risk_customers']:,}")
        col2.metric("Retention Rate (assumed)", imp["assumed_retention_rate"])
        col3.metric("Customers Retained",       f"{imp['customers_retained']:,}")
        col4.metric("Annual Revenue Saved",     f"${imp['annual_revenue_saved']:,.0f}")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Revenue Saved Breakdown")
            fig, ax = plt.subplots(figsize=(6, 4))
            months = list(range(1, 13))
            cumulative = [imp["monthly_revenue_saved"] * m for m in months]
            ax.fill_between(months, cumulative, alpha=0.3, color="#16A34A")
            ax.plot(months, cumulative, color="#16A34A", lw=2, marker="o")
            ax.set_xlabel("Month")
            ax.set_ylabel("Cumulative Revenue Saved ($)")
            ax.set_title("Projected Annual Retention Revenue")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.grid(alpha=0.3)
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Key Metrics")
            metrics_data = {
                "Metric": [
                    "Avg Monthly Charges (HIGH-risk)",
                    "Monthly Revenue Saved",
                    "Annual Revenue Saved",
                    "Avg CLV per Retained Customer",
                ],
                "Value": [
                    f"${imp['avg_monthly_charges']:.2f}",
                    f"${imp['monthly_revenue_saved']:,.2f}",
                    f"${imp['annual_revenue_saved']:,.2f}",
                    f"${imp['avg_clv_per_customer']:,.2f}",
                ],
            }
            st.table(pd.DataFrame(metrics_data))

        st.info(
            "💡 **Assumption**: 30% retention success rate from proactive interventions. "
            "CLV calculated over 24-month horizon. "
            "Actual figures will vary based on intervention execution quality."
        )
    else:
        st.warning("Business impact data not found. Run `python train.py` first.")
