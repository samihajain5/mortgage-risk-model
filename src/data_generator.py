"""
data_generator.py
-----------------
Generates a realistic synthetic mortgage dataset inspired by HMDA
(Home Mortgage Disclosure Act) and Fannie Mae loan performance data.

Features mirror what a mortgage lender like Rocket Mortgage would use
to assess default risk at origination.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROPERTY_TYPES   = ["Single Family", "Condo", "Multi-Family", "Townhouse"]
OCCUPANCY_TYPES  = ["Primary Residence", "Second Home", "Investment Property"]
LOAN_PURPOSES    = ["Purchase", "Refinance - Rate/Term", "Refinance - Cash-Out"]
STATES           = [
    "MI", "OH", "FL", "TX", "CA", "NY", "IL", "AZ", "GA", "NC",
    "WA", "CO", "PA", "NJ", "VA", "MN", "TN", "MO", "WI", "IN",
]

# Approximate default-rate lift by segment (multiplicative factors)
_DEFAULT_LIFT = {
    "occupancy": {"Primary Residence": 1.0, "Second Home": 1.4, "Investment Property": 2.2},
    "purpose":   {"Purchase": 1.0, "Refinance - Rate/Term": 0.85, "Refinance - Cash-Out": 1.3},
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_mortgage_dataset(
    n_samples: int = 20_000,
    default_rate: float = 0.07,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic mortgage origination dataset.

    Parameters
    ----------
    n_samples : int
        Number of loan records to generate.
    default_rate : float
        Approximate base default rate (before feature correlations).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per loan, with origination features and a binary
        `default` target (1 = defaulted within 24 months).
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Categorical features
    # ------------------------------------------------------------------
    property_type = rng.choice(PROPERTY_TYPES, n_samples,
                               p=[0.65, 0.15, 0.10, 0.10])
    occupancy     = rng.choice(OCCUPANCY_TYPES, n_samples,
                               p=[0.75, 0.10, 0.15])
    loan_purpose  = rng.choice(LOAN_PURPOSES, n_samples,
                               p=[0.55, 0.25, 0.20])
    state         = rng.choice(STATES, n_samples)
    loan_term     = rng.choice([180, 360], n_samples, p=[0.25, 0.75])  # 15 or 30 yr

    # ------------------------------------------------------------------
    # Borrower financials
    # ------------------------------------------------------------------
    # Annual income ($30k – $500k), log-normal
    income = np.clip(
        rng.lognormal(mean=11.2, sigma=0.6, size=n_samples), 30_000, 1_000_000
    )

    # Credit score (500–850), skewed toward prime
    credit_score = np.clip(
        rng.normal(loc=710, scale=65, size=n_samples).astype(int), 500, 850
    )

    # Employment years (0–35)
    employment_years = np.clip(
        rng.exponential(scale=8, size=n_samples), 0, 35
    ).round(1)

    # ------------------------------------------------------------------
    # Loan / property features
    # ------------------------------------------------------------------
    # Property value ($80k – $2M)
    property_value = np.clip(
        rng.lognormal(mean=12.3, sigma=0.55, size=n_samples), 80_000, 2_000_000
    )

    # LTV ratio: loan-to-value (50%–105%, allowing slight over-collateralisation)
    ltv_ratio = np.clip(rng.beta(a=5, b=2, size=n_samples) * 100, 50, 105)

    loan_amount = (property_value * ltv_ratio / 100).round(-2)  # round to $100

    # Interest rate (2.5% – 9%)
    # Higher rate for lower credit, higher LTV
    base_rate = 3.5 + (850 - credit_score) / 100 + (ltv_ratio - 80) / 40
    interest_rate = np.clip(
        base_rate + rng.normal(0, 0.3, n_samples), 2.5, 9.0
    ).round(3)

    # DTI: debt-to-income ratio (15%–65%)
    dti_ratio = np.clip(
        rng.normal(loc=36, scale=10, size=n_samples), 15, 65
    ).round(1)

    # Number of open accounts
    num_open_accounts = np.clip(
        rng.poisson(lam=6, size=n_samples), 0, 25
    )

    # Number of derogatory marks (late payments, collections, etc.)
    num_derogatory = np.clip(
        rng.poisson(lam=0.4, size=n_samples), 0, 8
    )

    # Months of reserves (months of PITI payment borrower has saved)
    months_reserves = np.clip(
        rng.exponential(scale=4, size=n_samples), 0, 24
    ).round(1)

    # ------------------------------------------------------------------
    # Default label — logistic model with realistic coefficients
    # ------------------------------------------------------------------
    # Log-odds base
    log_odds = (
        -5.5
        + 0.025  * (720 - credit_score) / 10       # lower score → higher risk
        + 0.08   * np.maximum(ltv_ratio - 80, 0)   # LTV > 80% adds risk
        + 0.05   * np.maximum(dti_ratio - 43, 0)   # DTI > 43% threshold
        - 0.03   * employment_years                 # tenure reduces risk
        - 0.008  * months_reserves                  # reserves reduce risk
        + 0.4    * num_derogatory                   # derogatory marks matter most
        - 0.002  * np.log(income + 1)               # higher income → lower risk
        + 0.01   * (interest_rate - 4.0)            # high rate correlates with risk
    )

    # Apply categorical lifts
    occ_lift  = np.array([_DEFAULT_LIFT["occupancy"][o] for o in occupancy])
    purp_lift = np.array([_DEFAULT_LIFT["purpose"][p] for p in loan_purpose])
    log_odds += np.log(occ_lift) + np.log(purp_lift)

    # Calibrate to target default rate
    prob_default = 1 / (1 + np.exp(-log_odds))
    calibration_shift = np.log(default_rate / (1 - default_rate)) - np.log(
        prob_default.mean() / (1 - prob_default.mean())
    )
    prob_default = 1 / (1 + np.exp(-(log_odds + calibration_shift)))

    default = (rng.uniform(size=n_samples) < prob_default).astype(int)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        # IDs
        "loan_id":            [f"LN{i:07d}" for i in range(1, n_samples + 1)],

        # Borrower
        "credit_score":       credit_score,
        "annual_income":      income.round(2),
        "employment_years":   employment_years,
        "dti_ratio":          dti_ratio,
        "num_open_accounts":  num_open_accounts,
        "num_derogatory":     num_derogatory,
        "months_reserves":    months_reserves,

        # Loan
        "loan_amount":        loan_amount,
        "loan_term_months":   loan_term,
        "interest_rate":      interest_rate,
        "ltv_ratio":          ltv_ratio.round(2),
        "loan_purpose":       loan_purpose,

        # Property
        "property_value":     property_value.round(2),
        "property_type":      property_type,
        "occupancy_type":     occupancy,
        "state":              state,

        # Target
        "default":            default,
    })

    return df


if __name__ == "__main__":
    df = generate_mortgage_dataset(n_samples=20_000)
    print(df.shape)
    print(df["default"].value_counts(normalize=True).round(3))
    print(df.head())
