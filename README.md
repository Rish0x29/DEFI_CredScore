# DeFi Credit Scoring — Undercollateralized Loan Intelligence

[![CI](https://github.com/Rish0x29/defi-credit-score/actions/workflows/ci.yml/badge.svg)](https://github.com/Rish0x29/defi-credit-score/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)

> **Score any wallet's creditworthiness from on-chain behavioral history — enabling undercollateralized DeFi lending without traditional credit checks.**

---

## The Problem

Every DeFi lending protocol today requires overcollateralization (≥150%). This locks out billions in potential liquidity and excludes wallets with strong on-chain history but limited capital. TradFi solved this with credit scores — DeFi needs the same thing, built natively from on-chain data.

## The Solution

A machine learning pipeline that ingests 22 raw on-chain features per wallet, engineers 11 derived behavioral signals, and outputs a **FICO-style credit score (300–850)** with loan approval decision and maximum loan limit.

---

## Architecture

```
Raw On-Chain Data (The Graph / Dune / Flipside)
        ↓
  Feature Engineering (22 raw → 33 total features)
        ↓
  Ensemble ML (LogReg + RF + GBT, calibrated probabilities)
        ↓
  Credit Score (300–850) + Band + Loan Limit
        ↓
  FastAPI REST endpoint  /score
```

---

## Feature Categories

| Category | Features |
|---|---|
| Activity & History | `wallet_age_days`, `tx_count_total`, `unique_active_days` |
| Repayment Behavior | `repayment_rate`, `loans_taken`, `loans_defaulted` |
| Liquidity Signals | `avg_eth_balance`, `balance_volatility`, `balance_stability` |
| Protocol Interaction | `protocols_used`, `chains_used`, `governance_votes` |
| Risk Signals | `liquidation_count`, `flash_loan_count`, `default_severity` |
| DeFi Yield Behavior | `lp_positions_count`, `staking_months`, `defi_breadth` |

**Top predictors (RF feature importance):**
1. `default_severity` — 31.7%
2. `liquidation_rate` — 17.4%
3. `repayment_rate` — 10.0%
4. `liquidation_count` — 9.0%
5. `repayment_quality` — 6.5%

---

## Score Bands & Loan Limits

| Band | Score Range | Max Loan (% of Balance) | Decision |
|---|---|---|---|
| Excellent | 750 – 850 | 30% | APPROVE |
| Good | 700 – 749 | 15% | APPROVE |
| Fair | 650 – 699 | 8% | APPROVE |
| Poor | 600 – 649 | 3% | DENY |
| Very Poor | 300 – 599 | 0% | DENY |

---

## Model Performance

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression ⭐ | **0.9241** | **0.8910** |
| Random Forest | 0.9094 | 0.8689 |
| Gradient Boosting | 0.9074 | 0.8552 |

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Rish0x29/defi-credit-score.git
cd defi-credit-score
pip install -r requirements.txt

# 2. Train the model
python -m src.train

# 3. Start the API
uvicorn src.api:app --reload
```

### Score a wallet via API

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
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
    "contract_calls_ratio": 0.72
  }'
```

**Response:**
```json
{
  "credit_score": 849,
  "band": "Excellent",
  "decision": "APPROVE",
  "creditworthiness_prob": 0.9978,
  "max_loan_pct_balance": 0.3,
  "risk_factors": [],
  "positive_factors": [
    "Strong repayment track record",
    "Established wallet age",
    "Broad DeFi participation"
  ]
}
```

### Docker

```bash
docker build -t defi-credit-score .
docker run -p 8000:8000 defi-credit-score
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
defi-credit-score/
├── src/
│   ├── features.py       # Raw feature generation + engineering
│   ├── scoring.py        # Label generation, score mapping, decisions
│   ├── train.py          # Model training, evaluation, persistence
│   └── api.py            # FastAPI REST endpoint
├── notebooks/
│   └── exploration.py    # Interactive analysis (convert to .ipynb)
├── tests/
│   └── test_pipeline.py  # Unit tests (pytest)
├── docs/                 # Generated plots
├── models/               # Saved model artifacts (gitignored)
├── .github/workflows/    # CI (GitHub Actions)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Production Roadmap

- [ ] **Real data integration** — The Graph subgraph for Aave/Compound/Morpho
- [ ] **Temporal features** — 30/90/180-day rolling windows per wallet
- [ ] **Cross-protocol identity** — ENS + address clustering for multi-wallet detection
- [ ] **On-chain score attestation** — EAS (Ethereum Attestation Service) integration
- [ ] **Sybil resistance** — Wallet clustering to detect manufactured history
- [ ] **Model monitoring** — Score drift detection with Evidently AI
- [ ] **Privacy** — ZK proof of score without revealing wallet address

---

## Integration with NeuraMesh

This scoring system can plug directly into the NeuraMesh compute layer:
- Node operators query `/score` before issuing undercollateralized compute credits
- NMESH stakers earn yield on approved credit pools
- Score history stored on Walrus decentralized storage

---

## License

MIT — see [LICENSE](LICENSE)
