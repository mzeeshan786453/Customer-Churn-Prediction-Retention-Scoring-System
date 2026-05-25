"""
model_training.py
-----------------
Train & compare Logistic Regression, Random Forest (proxy for XGBoost),
and Gradient Boosting (proxy for LightGBM) models.

When XGBoost / LightGBM / SHAP are available in your environment, swap them in:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import shap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_DIR  = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models():
    """Return dict of model name → sklearn pipeline."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    # RandomForest as drop-in proxy for XGBoost
    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

    # GradientBoosting as drop-in proxy for LightGBM
    gb = Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])

    return {
        "Logistic Regression (Baseline)": lr,
        "Random Forest (XGBoost proxy)": rf,
        "Gradient Boosting (LightGBM proxy)": gb,
    }


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_proba, model


def train_and_compare(X, y, feature_names, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    models   = build_models()
    results  = []
    cms      = {}
    probas   = {}
    fitted   = {}

    for name, model in models.items():
        print(f"  Training: {name} ...")
        metrics, cm, y_proba, trained = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        cms[name]    = cm
        probas[name] = y_proba
        fitted[name] = trained
        print(f"    AUC-ROC={metrics['auc_roc']}  F1={metrics['f1']}")

    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    best_name  = results_df.iloc[0]["model"]
    best_model = fitted[best_name]

    print(f"\n  Best model: {best_name}")

    return {
        "results_df":  results_df,
        "cms":         cms,
        "probas":      probas,
        "fitted":      fitted,
        "best_name":   best_name,
        "best_model":  best_model,
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,
        "y_test":      y_test,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Feature importance (SHAP-style permutation importance)
# ---------------------------------------------------------------------------

def compute_feature_importance(model, X_test, y_test, feature_names):
    """
    Use permutation importance as a faithful SHAP proxy.
    When shap is available:
        explainer = shap.TreeExplainer(model.named_steps['clf'])
        shap_values = explainer.shap_values(X_test)
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=15,
        random_state=42, n_jobs=-1, scoring="roc_auc"
    )
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False)
    return imp_df


def per_customer_explanation(model, X_row: pd.DataFrame, feature_names, top_n=3):
    """
    Produce a per-customer 'why at risk' explanation using
    permutation-importance weights (sign from logistic coefs or tree).

    When shap is available:
        explainer = shap.TreeExplainer(model.named_steps['clf'])
        sv = explainer.shap_values(X_row)[1]
        contributors = sorted(zip(feature_names, sv[0]), key=lambda x: abs(x[1]), reverse=True)
    """
    clf = model.named_steps.get("clf", model)

    # Get feature contributions: coef for LR, feature_importances for trees
    if hasattr(clf, "coef_"):
        scaler = model.named_steps.get("scaler", None)
        if scaler is not None:
            x_scaled = scaler.transform(X_row)
        else:
            x_scaled = X_row.values
        contributions = (x_scaled[0] * clf.coef_[0]).tolist()
    elif hasattr(clf, "feature_importances_"):
        contributions = (X_row.values[0] * clf.feature_importances_).tolist()
    else:
        contributions = X_row.values[0].tolist()

    pairs = sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True)
    top   = pairs[:top_n]
    return [{"feature": f, "contribution": round(c, 4)} for f, c in top]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(fitted, X_test, y_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#16A34A", "#DC2626"]
    for (name, model), color in zip(fitted.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, model_name, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix\n{model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(imp_df, save_path, top_n=15):
    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#DC2626" if i < 5 else "#2563EB" if i < 10 else "#6B7280"
              for i in range(len(top))]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Permutation Importance (AUC drop)")
    ax.set_title("Top Feature Importances (SHAP-proxy)")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metric_comparison(results_df, save_path):
    metrics = ["auc_roc", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=row["model"][:30], color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["AUC-ROC", "Precision", "Recall", "F1"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_best_model(model, name, model_dir=MODEL_DIR):
    path = model_dir / "best_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "name": name}, f)
    print(f"  Model saved: {path}")
    return path


def load_best_model(model_dir=MODEL_DIR):
    path = model_dir / "best_model.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["name"]
