"""
model_utils.py
--------------
Model training, evaluation, and comparison utilities for the
Mortgage Default Risk project.

Supports:
    - Logistic Regression (interpretable baseline)
    - Random Forest (non-linear ensemble)
    - Gradient Boosting / XGBoost (state-of-the-art tabular)
    - Calibration checking (Brier score)
    - Statistical comparison of models (DeLong AUC test approximation)
    - Feature importance extraction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Any, Tuple, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def get_models() -> Dict[str, Any]:
    """Return dict of model name → unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified k-fold CV for each model and return a summary DataFrame.

    Metrics: ROC-AUC, Average Precision (PR-AUC), Brier Score
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc":           "roc_auc",
        "avg_precision":     "average_precision",
        "brier_score":       "neg_brier_score",
    }

    records = []
    for name, model in models.items():
        results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        records.append({
            "Model":              name,
            "ROC-AUC (mean)":     results["test_roc_auc"].mean(),
            "ROC-AUC (std)":      results["test_roc_auc"].std(),
            "PR-AUC (mean)":      results["test_avg_precision"].mean(),
            "PR-AUC (std)":       results["test_avg_precision"].std(),
            "Brier Score (mean)": (-results["test_brier_score"]).mean(),
            "Brier Score (std)":  results["test_brier_score"].std(),
        })

    return pd.DataFrame(records).set_index("Model").round(4)


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximises F1 (default) or
    Youden's J statistic ('youden') on the provided data.

    Returns
    -------
    threshold : float
    best_metric_value : float
    """
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.01, 0.99, 200)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "youden":
            from sklearn.metrics import recall_score
            tpr = recall_score(y_true, y_pred, zero_division=0)
            tnr = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
            scores.append(tpr + tnr - 1)
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot ROC and Precision-Recall curves for all models side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        ap     = average_precision_score(y_test, y_prob)

        RocCurveDisplay.from_predictions(
            y_test, y_prob,
            name=f"{name} (AUC={auc:.3f})",
            ax=ax1, color=color, alpha=0.85
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob,
            name=f"{name} (AP={ap:.3f})",
            ax=ax2, color=color, alpha=0.85
        )

    ax1.set_title("ROC Curves", fontweight="bold")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.500)")
    ax1.legend(fontsize=8)

    ax2.set_title("Precision-Recall Curves", fontweight="bold")
    baseline = y_test.mean()
    ax2.axhline(baseline, color="k", linestyle="--", lw=1.2,
                label=f"Random (AP={baseline:.3f})")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_calibration(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Plot calibration curves (reliability diagrams) for each model."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfectly calibrated")
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
        brier = brier_score_loss(y_test, y_prob)
        ax.plot(mean_pred, frac_pos, "s-", color=color, alpha=0.85,
                label=f"{name} (Brier={brier:.4f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves (Reliability Diagrams)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
    figsize: Tuple[int, int] = (9, 7),
) -> plt.Figure:
    """Plot top-N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_.")
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        importances[idx[::-1]],
        color="#1f77b4", edgecolor="white"
    )
    ax.set_xlabel("Feature importance (MDI)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (5, 4),
) -> plt.Figure:
    """Plot confusion matrix at a given decision threshold."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Default", "Default"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"{model_name} — Confusion Matrix\n(threshold={threshold:.2f})",
        fontweight="bold"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Business metrics
# ---------------------------------------------------------------------------

def compute_business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    avg_loan_amount: float = 350_000,
    loss_given_default: float = 0.30,
    cost_per_review: float = 500,
) -> Dict[str, Any]:
    """
    Translate model performance into dollar-denominated business metrics.

    Parameters
    ----------
    loss_given_default : float
        Fraction of loan principal lost on default (LGD).
    cost_per_review : float
        Cost of manual underwriting review triggered per flagged loan.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)

    # Expected loss avoided by catching true positives
    loss_avoided = tp * avg_loan_amount * loss_given_default

    # Cost of false positives (manual reviews that turned out not to be defaults)
    review_cost  = fp * cost_per_review

    # Net business value
    net_value    = loss_avoided - review_cost

    return {
        "True Positives (defaults caught)":   int(tp),
        "False Positives (unnecessary flags)": int(fp),
        "False Negatives (missed defaults)":   int(fn),
        "True Negatives (correct approvals)":  int(tn),
        "Detection Rate":                      round(tp / max(tp + fn, 1), 3),
        "False Alarm Rate":                    round(fp / max(fp + tn, 1), 3),
        "Estimated Loss Avoided ($)":          f"${loss_avoided:,.0f}",
        "Cost of False Alerts ($)":            f"${review_cost:,.0f}",
        "Net Business Value ($)":              f"${net_value:,.0f}",
    }
