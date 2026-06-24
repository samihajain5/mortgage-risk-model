"""
feature_engineering.py
-----------------------
Transforms raw mortgage origination data into model-ready features.

Design philosophy
-----------------
* Keep every transformation traceable and business-interpretable.
* Avoid data leakage: all derived features use only information available
  at origination time.
* Expose a single `build_features()` function that returns X, y, and the
  fitted preprocessor so inference can reuse the same pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from typing import Tuple


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "credit_score",
    "annual_income",
    "employment_years",
    "dti_ratio",
    "num_open_accounts",
    "num_derogatory",
    "months_reserves",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "ltv_ratio",
    "property_value",
    # Engineered
    "log_income",
    "payment_to_income",
    "credit_utilization_proxy",
    "risk_score",
    "reserves_in_months_piti",
    "ltv_above_80",
    "dti_above_43",
]

CATEGORICAL_FEATURES = [
    "loan_purpose",
    "property_type",
    "occupancy_type",
    "state",
    "loan_term_bucket",
    "credit_tier",
    "ltv_bucket",
]

TARGET = "default"


# ---------------------------------------------------------------------------
# Domain-derived features
# ---------------------------------------------------------------------------

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable, domain-grounded features to a raw mortgage DataFrame.
    All computations are vectorised (no row-by-row iteration).
    """
    df = df.copy()

    # --- Continuous transformations ---

    # Log-income to compress right-skew
    df["log_income"] = np.log1p(df["annual_income"])

    # Estimated monthly mortgage payment (simplified amortisation)
    monthly_rate = df["interest_rate"] / 100 / 12
    n_payments   = df["loan_term_months"]
    # Avoid division by zero for zero-rate edge case
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            monthly_rate == 0,
            1 / n_payments,
            monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments)),
        )
    df["monthly_payment"] = df["loan_amount"] * factor

    # Payment-to-income ratio (monthly payment ÷ monthly gross income)
    df["payment_to_income"] = df["monthly_payment"] / (df["annual_income"] / 12)

    # Proxy for revolving credit utilisation using derogatory count
    df["credit_utilization_proxy"] = df["num_derogatory"] / (df["num_open_accounts"] + 1)

    # Composite risk score (higher = riskier): weighted sum of risk signals
    df["risk_score"] = (
        (850 - df["credit_score"]) * 0.04
        + np.maximum(df["ltv_ratio"] - 80, 0) * 0.5
        + np.maximum(df["dti_ratio"] - 43, 0) * 0.3
        + df["num_derogatory"] * 2.5
        - df["months_reserves"] * 0.2
        - df["employment_years"] * 0.1
    )

    # Months of reserves expressed in units of estimated PITI payment
    # (a standard underwriting concept)
    df["reserves_in_months_piti"] = df["months_reserves"]  # already in PITI months

    # Binary threshold indicators (important split points in underwriting)
    df["ltv_above_80"]  = (df["ltv_ratio"] > 80).astype(int)
    df["dti_above_43"]  = (df["dti_ratio"] > 43).astype(int)

    # --- Categorical bins ---

    # Loan term bucket
    df["loan_term_bucket"] = pd.cut(
        df["loan_term_months"],
        bins=[0, 180, 360],
        labels=["15yr", "30yr"],
    ).astype(str)

    # Credit tier (mirrors FICO tier used in mortgage pricing)
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[499, 579, 619, 659, 699, 739, 779, 851],
        labels=["<580", "580-619", "620-659", "660-699", "700-739", "740-779", "780+"],
    ).astype(str)

    # LTV bucket
    df["ltv_bucket"] = pd.cut(
        df["ltv_ratio"],
        bins=[0, 60, 70, 80, 90, 95, 105],
        labels=["≤60", "60-70", "70-80", "80-90", "90-95", ">95"],
    ).astype(str)

    return df


# ---------------------------------------------------------------------------
# Preprocessor pipeline
# ---------------------------------------------------------------------------

def _build_preprocessor() -> ColumnTransformer:
    """
    Scikit-learn ColumnTransformer that scales numeric features and
    one-hot encodes categorical ones.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    fit: bool = True,
    preprocessor=None,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Full feature-engineering pipeline: add domain features → preprocess.

    Parameters
    ----------
    df : pd.DataFrame
        Raw mortgage DataFrame (output of data_generator.generate_mortgage_dataset).
    fit : bool
        If True, fit the preprocessor on `df` (training).
        If False, transform only (inference / test set).
    preprocessor : fitted ColumnTransformer or None
        Required when fit=False.

    Returns
    -------
    X : np.ndarray
        Processed feature matrix.
    y : np.ndarray
        Binary target vector.
    preprocessor : ColumnTransformer
        Fitted preprocessor (reuse for test data).
    """
    df_eng = _add_engineered_features(df)

    y = df_eng[TARGET].values if TARGET in df_eng.columns else None

    if fit:
        preprocessor = _build_preprocessor()
        X = preprocessor.fit_transform(df_eng)
    else:
        if preprocessor is None:
            raise ValueError("Must provide a fitted preprocessor when fit=False.")
        X = preprocessor.transform(df_eng)

    return X, y, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return human-readable feature names after OHE expansion."""
    num_names = NUMERIC_FEATURES.copy()
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names
