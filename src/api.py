"""
api.py — FastAPI REST endpoint for wallet credit scoring

Run locally:
    uvicorn src.api:app --reload

Docker:
    docker build -t defi-credit-score .
    docker run -p 8000:8000 defi-credit-score
"""

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import engineer_features
from src.scoring import build_score_report, SCORE_BANDS
from src.train import load_model, ALL_FEATURES

import pandas as pd

app = FastAPI(
    title="DeFi Credit Score API",
    description="Score wallet creditworthiness for undercollateralized DeFi loans",
    version="1.0.0",
)

# Load model on startup
MODEL_PATH = Path("models/best_model.pkl")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run: python -m src.train",
            )
        _model = load_model(MODEL_PATH)
    return _model


# ── Request / Response schemas ─────────────────────────────────────────────────

class WalletFeatures(BaseModel):
    wallet_age_days:       float = Field(..., ge=0,   description="Days since first tx")
    tx_count_total:        int   = Field(..., ge=1,   description="Total transactions")
    unique_active_days:    int   = Field(..., ge=1)
    loans_taken:           int   = Field(..., ge=0)
    loans_repaid_on_time:  int   = Field(..., ge=0)
    loans_repaid_late:     int   = Field(..., ge=0)
    loans_defaulted:       int   = Field(..., ge=0)
    repayment_rate:        float = Field(..., ge=0, le=1)
    avg_eth_balance:       float = Field(..., ge=0)
    balance_volatility:    float = Field(..., ge=0, le=1)
    avg_collateral_ratio:  float = Field(..., ge=0)
    max_collateral_ratio:  float = Field(..., ge=0)
    liquidation_count:     int   = Field(..., ge=0)
    protocols_used:        int   = Field(..., ge=1)
    governance_votes:      int   = Field(..., ge=0)
    flash_loan_count:      int   = Field(..., ge=0)
    lp_positions_count:    int   = Field(..., ge=0)
    staking_months:        float = Field(..., ge=0)
    nft_holdings:          int   = Field(..., ge=0)
    chains_used:           int   = Field(..., ge=1)
    days_since_last_tx:    float = Field(..., ge=0)
    contract_calls_ratio:  float = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "wallet_age_days": 720,
                "tx_count_total": 350,
                "unique_active_days": 200,
                "loans_taken": 8,
                "loans_repaid_on_time": 7,
                "loans_repaid_late": 1,
                "loans_defaulted": 0,
                "repayment_rate": 0.94,
                "avg_eth_balance": 12.5,
                "balance_volatility": 0.15,
                "avg_collateral_ratio": 1.85,
                "max_collateral_ratio": 2.4,
                "liquidation_count": 0,
                "protocols_used": 6,
                "governance_votes": 12,
                "flash_loan_count": 0,
                "lp_positions_count": 3,
                "staking_months": 8.0,
                "nft_holdings": 5,
                "chains_used": 3,
                "days_since_last_tx": 5,
                "contract_calls_ratio": 0.72,
            }
        }


class ScoreResponse(BaseModel):
    credit_score:            int
    band:                    str
    decision:                str
    creditworthiness_prob:   float
    max_loan_pct_balance:    float
    risk_factors:            list[str]
    positive_factors:        list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/score", response_model=ScoreResponse)
def score_wallet(wallet: WalletFeatures):
    """
    Score a single wallet's creditworthiness.

    Returns a FICO-style credit score (300–850), approval decision,
    maximum loan size as % of balance, and key risk/positive factors.
    """
    model  = get_model()
    report = build_score_report(wallet.model_dump(), model, ALL_FEATURES)
    return ScoreResponse(**report)


@app.get("/bands")
def score_bands():
    """Return the credit score band thresholds and associated loan limits."""
    return {
        band: {"min": lo, "max": hi, "max_loan_pct": pct}
        for band, (lo, hi, pct) in SCORE_BANDS.items()
    }
