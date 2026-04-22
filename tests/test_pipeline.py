"""
tests/test_pipeline.py — Unit tests for the credit scoring pipeline
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features, generate_synthetic_wallets, ALL_FEATURES, ENGINEERED_FEATURES
from src.scoring import (
    probability_to_credit_score,
    get_score_band,
    get_loan_limit,
    make_decision,
    generate_labels,
    SCORE_MIN,
    SCORE_MAX,
    APPROVAL_THRESHOLD,
)


# ── Feature generation ─────────────────────────────────────────────────────────

class TestFeatureGeneration:
    def test_shape(self):
        df = generate_synthetic_wallets(n=100)
        assert len(df) == 100

    def test_no_nulls(self):
        df = generate_synthetic_wallets(n=200)
        assert not df.isnull().any().any()

    def test_repayment_rate_bounded(self):
        df = generate_synthetic_wallets(n=500)
        assert df["repayment_rate"].between(0, 1).all()

    def test_balance_volatility_bounded(self):
        df = generate_synthetic_wallets(n=500)
        assert df["balance_volatility"].between(0, 1).all()

    def test_reproducible(self):
        df1 = generate_synthetic_wallets(n=50, seed=7)
        df2 = generate_synthetic_wallets(n=50, seed=7)
        pd.testing.assert_frame_equal(df1, df2)


# ── Feature engineering ────────────────────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture
    def engineered(self):
        raw = generate_synthetic_wallets(n=300)
        return engineer_features(raw)

    def test_engineered_columns_present(self, engineered):
        for col in ENGINEERED_FEATURES:
            assert col in engineered.columns, f"Missing column: {col}"

    def test_no_nulls_after_engineering(self, engineered):
        assert not engineered[ENGINEERED_FEATURES].isnull().any().any()

    def test_recency_score_in_01(self, engineered):
        assert engineered["recency_score"].between(0, 1).all()

    def test_liquidation_rate_nonneg(self, engineered):
        assert (engineered["liquidation_rate"] >= 0).all()

    def test_all_features_present(self, engineered):
        for col in ALL_FEATURES:
            assert col in engineered.columns


# ── Scoring functions ──────────────────────────────────────────────────────────

class TestScoringFunctions:
    def test_prob_to_score_bounds(self):
        scores = probability_to_credit_score(np.array([0.0, 0.5, 1.0]))
        assert scores[0] == SCORE_MIN
        assert scores[-1] == SCORE_MAX

    def test_prob_to_score_monotone(self):
        probs  = np.linspace(0, 1, 100)
        scores = probability_to_credit_score(probs)
        assert (np.diff(scores) >= 0).all()

    def test_score_band_excellent(self):
        assert get_score_band(800) == "Excellent"

    def test_score_band_very_poor(self):
        assert get_score_band(350) == "Very Poor"

    def test_score_band_boundaries(self):
        assert get_score_band(750) == "Excellent"
        assert get_score_band(749) == "Good"
        assert get_score_band(700) == "Good"
        assert get_score_band(699) == "Fair"
        assert get_score_band(650) == "Fair"
        assert get_score_band(649) == "Poor"
        assert get_score_band(600) == "Poor"
        assert get_score_band(599) == "Very Poor"

    def test_loan_limit_zero_for_very_poor(self):
        assert get_loan_limit(400) == 0.0

    def test_loan_limit_positive_for_excellent(self):
        assert get_loan_limit(800) > 0

    def test_decision_approve(self):
        assert make_decision(APPROVAL_THRESHOLD) == "APPROVE"
        assert make_decision(800) == "APPROVE"

    def test_decision_deny(self):
        assert make_decision(APPROVAL_THRESHOLD - 1) == "DENY"
        assert make_decision(300) == "DENY"


# ── Label generation ───────────────────────────────────────────────────────────

class TestLabelGeneration:
    @pytest.fixture
    def df_engineered(self):
        raw = generate_synthetic_wallets(n=500)
        return engineer_features(raw)

    def test_binary_labels(self, df_engineered):
        labels, _ = generate_labels(df_engineered)
        assert set(labels).issubset({0, 1})

    def test_raw_score_bounded(self, df_engineered):
        _, raw = generate_labels(df_engineered)
        assert ((raw >= 0) & (raw <= 1)).all()

    def test_class_balance(self, df_engineered):
        labels, _ = generate_labels(df_engineered, creditworthy_pct=0.4)
        ratio = labels.mean()
        assert 0.35 <= ratio <= 0.45, f"Unexpected class balance: {ratio:.2f}"
