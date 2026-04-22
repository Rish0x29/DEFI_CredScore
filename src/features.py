"""
features.py — On-chain feature generation & engineering
"""

import numpy as np
import pandas as pd


RAW_FEATURE_SCHEMA = {
    "wallet_age_days":       "int   — days since first transaction",
    "tx_count_total":        "int   — total transactions sent",
    "unique_active_days":    "int   — distinct days with tx activity",
    "loans_taken":           "int   — total DeFi loans initiated",
    "loans_repaid_on_time":  "int   — loans repaid before deadline",
    "loans_repaid_late":     "int   — loans repaid after deadline",
    "loans_defaulted":       "int   — loans unpaid / written off",
    "repayment_rate":        "float — (on_time + 0.5*late) / total_loans",
    "avg_eth_balance":       "float — avg ETH/USD balance over history",
    "balance_volatility":    "float — [0,1] normalized std dev of balance",
    "avg_collateral_ratio":  "float — average collateral factor held",
    "max_collateral_ratio":  "float — peak collateral factor observed",
    "liquidation_count":     "int   — number of liquidation events",
    "protocols_used":        "int   — distinct DeFi protocols interacted",
    "governance_votes":      "int   — on-chain governance votes cast",
    "flash_loan_count":      "int   — flash loan transactions",
    "lp_positions_count":    "int   — liquidity pool positions opened",
    "staking_months":        "float — total months of active staking",
    "nft_holdings":          "int   — NFT tokens held",
    "chains_used":           "int   — distinct chains active on",
    "days_since_last_tx":    "float — recency of last transaction",
    "contract_calls_ratio":  "float — contract calls / total txs",
}


def generate_synthetic_wallets(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic on-chain wallet feature data.
    Useful for model training, testing, and demo purposes.
    Replace with real indexer data (The Graph, Dune, Flipside) in production.
    """
    rng = np.random.default_rng(seed)

    wallet_age_days      = rng.gamma(3, 150, n).clip(1, 1500)
    tx_count_total       = (rng.pareto(1.5, n) * 50 + 5).astype(int).clip(1, 5000)
    unique_active_days   = (wallet_age_days * rng.uniform(0.05, 0.7, n)).astype(int).clip(1, 500)

    loans_taken          = rng.poisson(3, n).clip(0, 30)
    loans_repaid_on_time = np.floor(loans_taken * rng.beta(5, 1.5, n)).astype(int)
    loans_repaid_late    = np.floor(
        (loans_taken - loans_repaid_on_time) * rng.beta(2, 5, n)
    ).astype(int)
    loans_defaulted      = (loans_taken - loans_repaid_on_time - loans_repaid_late).clip(0)
    repayment_rate       = np.where(
        loans_taken > 0,
        (loans_repaid_on_time + 0.5 * loans_repaid_late) / loans_taken,
        0.5,
    )

    avg_eth_balance     = rng.lognormal(1.5, 1.5, n).clip(0.01, 500)
    balance_volatility  = rng.beta(2, 5, n)
    max_collateral_ratio = rng.uniform(1.0, 5.0, n)
    avg_collateral_ratio = max_collateral_ratio * rng.beta(4, 2, n)

    liquidation_count   = rng.poisson(0.3, n).clip(0, 10)
    liquidation_count  += (loans_defaulted * rng.poisson(0.5, loans_defaulted.shape)).clip(0, 5)

    protocols_used      = rng.poisson(3, n).clip(1, 15)
    governance_votes    = rng.poisson(2, n).clip(0, 50)
    flash_loan_count    = rng.poisson(0.5, n).clip(0, 20)
    lp_positions_count  = rng.poisson(1.5, n).clip(0, 10)
    staking_months      = rng.exponential(3, n).clip(0, 24)
    nft_holdings        = rng.poisson(2, n).clip(0, 100)
    chains_used         = rng.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    days_since_last_tx  = rng.exponential(30, n).clip(0, 365)
    contract_calls_ratio = rng.beta(2, 3, n)

    return pd.DataFrame({
        "wallet_age_days":        wallet_age_days,
        "tx_count_total":         tx_count_total,
        "unique_active_days":     unique_active_days,
        "loans_taken":            loans_taken,
        "loans_repaid_on_time":   loans_repaid_on_time,
        "loans_repaid_late":      loans_repaid_late,
        "loans_defaulted":        loans_defaulted,
        "repayment_rate":         repayment_rate,
        "avg_eth_balance":        avg_eth_balance,
        "balance_volatility":     balance_volatility,
        "avg_collateral_ratio":   avg_collateral_ratio,
        "max_collateral_ratio":   max_collateral_ratio,
        "liquidation_count":      liquidation_count,
        "protocols_used":         protocols_used,
        "governance_votes":       governance_votes,
        "flash_loan_count":       flash_loan_count,
        "lp_positions_count":     lp_positions_count,
        "staking_months":         staking_months,
        "nft_holdings":           nft_holdings,
        "chains_used":            chains_used,
        "days_since_last_tx":     days_since_last_tx,
        "contract_calls_ratio":   contract_calls_ratio,
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived creditworthiness features on top of raw on-chain signals.

    Returns a new DataFrame with both raw + engineered columns.
    Engineered columns are prefixed-free but documented below.
    """
    fe = df.copy()

    # --- Repayment quality: rate × log(volume) rewards consistent payers
    fe["repayment_quality"] = fe["repayment_rate"] * np.log1p(fe["loans_taken"])

    # --- Activity density: consistency of usage relative to wallet age
    fe["activity_density"] = fe["unique_active_days"] / (fe["wallet_age_days"] + 1)

    # --- Liquidation rate: liquidations per loan (normalized)
    fe["liquidation_rate"] = fe["liquidation_count"] / (fe["loans_taken"] + 1)

    # --- Default severity: defaults × liquidation exposure (top predictor)
    fe["default_severity"] = (
        fe["loans_defaulted"] / (fe["loans_taken"] + 1)
    ) * (1 + fe["liquidation_count"])

    # --- Balance stability: log-balance scaled by inverse volatility
    fe["balance_stability"] = (
        np.log1p(fe["avg_eth_balance"]) * (1 - fe["balance_volatility"])
    )

    # --- DeFi breadth: cross-protocol + cross-chain + yield participation
    fe["defi_breadth"] = (
        fe["protocols_used"] * fe["chains_used"]
        + fe["lp_positions_count"]
        + fe["staking_months"] / 6
    )

    # --- Governance engagement: log-scaled voting activity
    fe["governance_engagement"] = np.log1p(fe["governance_votes"])

    # --- Recency score: exponential decay for inactivity (90-day half-life)
    fe["recency_score"] = np.exp(-fe["days_since_last_tx"] / 90)

    # --- Flash loan ratio: suspicious signal if disproportionately high
    fe["flash_loan_ratio"] = fe["flash_loan_count"] / (fe["tx_count_total"] + 1)

    # --- Transaction sophistication: contract-call density × log(volume)
    fe["tx_sophistication"] = (
        fe["contract_calls_ratio"] * np.log1p(fe["tx_count_total"])
    )

    # --- Collateral discipline: avg vs max ratio (closer to 1 = conservative)
    fe["collateral_discipline"] = fe["avg_collateral_ratio"] / (
        fe["max_collateral_ratio"] + 1e-6
    )

    return fe


ENGINEERED_FEATURES = [
    "repayment_quality",
    "activity_density",
    "liquidation_rate",
    "default_severity",
    "balance_stability",
    "defi_breadth",
    "governance_engagement",
    "recency_score",
    "flash_loan_ratio",
    "tx_sophistication",
    "collateral_discipline",
]

RAW_FEATURES = list(RAW_FEATURE_SCHEMA.keys())

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
