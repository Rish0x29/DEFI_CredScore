"""
scoring.py — Label generation, credit score mapping, loan decision logic
"""

import numpy as np
import pandas as pd


# ── Credit Score Scale ─────────────────────────────────────────────────────────
SCORE_MIN = 300
SCORE_MAX = 850

SCORE_BANDS = {
    "Excellent": (750, 850, 0.30),   # (min, max, max_loan_pct)
    "Good":      (700, 749, 0.15),
    "Fair":      (650, 699, 0.08),
    "Poor":      (600, 649, 0.03),
    "Very Poor": (300, 599, 0.00),
}

APPROVAL_THRESHOLD = 650  # minimum score to approve


def generate_labels(
    df: pd.DataFrame, noise_std: float = 0.05, creditworthy_pct: float = 0.40
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive creditworthiness labels from engineered features.
    Uses a weighted scoring formula that mirrors real underwriting logic.

    Returns
    -------
    labels : np.ndarray of int (1 = creditworthy, 0 = high risk)
    raw_score : np.ndarray of float in [0, 1]
    """
    score = (
        + 2.5 * df["repayment_rate"]
        + 1.5 * np.log1p(df["wallet_age_days"]) / np.log1p(1500)
        + 1.2 * df["activity_density"]
        + 1.0 * np.log1p(df["avg_eth_balance"]) / np.log1p(500)
        - 3.0 * df["default_severity"]
        - 2.0 * df["liquidation_rate"]
        + 0.8 * df["defi_breadth"] / 30
        + 0.5 * df["governance_engagement"] / 4
        + 0.6 * df["recency_score"]
        - 1.0 * df["flash_loan_ratio"] * 5
        + 0.4 * df["balance_stability"] / 5
        - 0.3 * df["balance_volatility"]
        + 0.3 * df["collateral_discipline"]
    )

    score = (score - score.min()) / (score.max() - score.min())
    score += np.random.default_rng(42).normal(0, noise_std, len(score))
    score = score.clip(0, 1)

    threshold = np.percentile(score, (1 - creditworthy_pct) * 100)
    labels = (score > threshold).astype(int)
    return labels, score


def probability_to_credit_score(prob: np.ndarray | float) -> np.ndarray | int:
    """Map model probability [0,1] → FICO-like credit score [300,850]."""
    return np.round(SCORE_MIN + np.asarray(prob) * (SCORE_MAX - SCORE_MIN)).astype(int)


def get_score_band(score: int) -> str:
    for band, (lo, hi, _) in SCORE_BANDS.items():
        if lo <= score <= hi:
            return band
    return "Very Poor"


def get_loan_limit(score: int) -> float:
    """Return maximum loan as fraction of wallet balance."""
    band = get_score_band(score)
    return SCORE_BANDS[band][2]


def make_decision(score: int) -> str:
    return "APPROVE" if score >= APPROVAL_THRESHOLD else "DENY"


def build_score_report(wallet_features: dict, model, feature_cols: list) -> dict:
    """
    Full underwriting report for a single wallet.

    Parameters
    ----------
    wallet_features : raw feature dict (keys match RAW_FEATURE_SCHEMA)
    model           : fitted sklearn Pipeline
    feature_cols    : ordered list of feature column names

    Returns
    -------
    dict with: credit_score, band, decision, prob, max_loan_pct,
               top_risk_factors, top_positive_factors
    """
    from src.features import engineer_features
    import pandas as pd

    row  = engineer_features(pd.DataFrame([wallet_features]))
    X    = row[feature_cols]
    prob = float(model.predict_proba(X)[0, 1])
    cs   = int(probability_to_credit_score(prob))
    band = get_score_band(cs)

    # Simple factor attribution (rule-based, model-agnostic)
    risk_flags     = []
    positive_flags = []

    r = row.iloc[0]
    if r["default_severity"] > 0.5:   risk_flags.append("High default severity")
    if r["liquidation_rate"] > 0.3:   risk_flags.append("Elevated liquidation rate")
    if r["balance_volatility"] > 0.6: risk_flags.append("Volatile balance history")
    if r["flash_loan_ratio"] > 0.05:  risk_flags.append("Disproportionate flash loan use")
    if r["recency_score"] < 0.4:      risk_flags.append("Prolonged wallet inactivity")

    if r["repayment_rate"] > 0.85:        positive_flags.append("Strong repayment track record")
    if r["wallet_age_days"] > 365:         positive_flags.append("Established wallet age")
    if r["defi_breadth"] > 15:             positive_flags.append("Broad DeFi participation")
    if r["governance_engagement"] > 1.5:   positive_flags.append("Active governance participant")
    if r["balance_stability"] > 3:         positive_flags.append("Stable balance history")

    return {
        "credit_score":       cs,
        "band":               band,
        "decision":           make_decision(cs),
        "creditworthiness_prob": round(prob, 4),
        "max_loan_pct_balance":  get_loan_limit(cs),
        "risk_factors":          risk_flags,
        "positive_factors":      positive_flags,
    }
