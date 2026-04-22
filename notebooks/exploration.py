# %% [markdown]
# # DeFi Credit Scoring — Exploration Notebook
# 
# This notebook walks through the full pipeline interactively:
# - Feature exploration & distributions
# - Label analysis
# - Model training & comparison
# - Feature importance
# - Individual wallet scoring

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

import sys; sys.path.insert(0, "..")
from src.features import generate_synthetic_wallets, engineer_features, ALL_FEATURES
from src.scoring import generate_labels, probability_to_credit_score, get_score_band
from src.train import train, evaluate, get_feature_importance, build_models

plt.style.use("dark_background")
plt.rcParams.update({"axes.facecolor": "#0d0d0d", "figure.facecolor": "#0d0d0d",
                     "axes.edgecolor": "#333", "grid.color": "#222"})

# %% [markdown]
# ## 1. Data Generation

# %%
raw = generate_synthetic_wallets(n=2000)
df  = engineer_features(raw)
labels, raw_scores = generate_labels(df)

print(f"Wallets: {len(df)}")
print(f"Creditworthy: {labels.sum()} ({labels.mean()*100:.1f}%)")
df.describe().T

# %% [markdown]
# ## 2. Feature Distributions

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

key_features = [
    "repayment_rate", "wallet_age_days", "avg_eth_balance", "liquidation_count",
    "protocols_used", "balance_volatility", "staking_months", "governance_votes",
    "repayment_quality", "default_severity", "defi_breadth", "recency_score"
]

for i, feat in enumerate(key_features):
    ax = axes[i]
    ax.hist(df[feat][labels==0], bins=40, alpha=0.6, color="#d85a30", label="High Risk", density=True)
    ax.hist(df[feat][labels==1], bins=40, alpha=0.6, color="#1d9e75", label="Creditworthy", density=True)
    ax.set_title(feat, fontsize=10, color="#aaa")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("docs/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Credit Score Distribution

# %%
X = df[ALL_FEATURES]
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

trained = train(X_train, y_train, verbose=True)
results = evaluate(trained, X_test, y_test, verbose=True)

# %%
best_name  = max(results, key=lambda k: results[k]["roc_auc"])
best_model = results[best_name]["model"]
all_probs  = best_model.predict_proba(X)[:, 1]
scores     = probability_to_credit_score(all_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(scores, bins=30, color="#4f8ef7", edgecolor="none", alpha=0.9)
axes[0].axvline(650, color="#d85a30", linestyle="--", label="Approval threshold (650)")
axes[0].set_xlabel("Credit Score", color="#aaa")
axes[0].set_ylabel("Wallets", color="#aaa")
axes[0].set_title("Credit Score Distribution", color="white")
axes[0].legend()

band_counts = pd.Series([get_score_band(s) for s in scores]).value_counts()
band_order  = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]
colors      = ["#888780", "#d85a30", "#ba7517", "#4f8ef7", "#1d9e75"]
axes[1].barh(
    [b for b in band_order if b in band_counts.index],
    [band_counts.get(b, 0) for b in band_order if b in band_counts.index],
    color=[c for b, c in zip(band_order, colors) if b in band_counts.index]
)
axes[1].set_xlabel("Wallets", color="#aaa")
axes[1].set_title("Score Band Breakdown", color="white")
plt.tight_layout()
plt.savefig("docs/score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. ROC & Precision-Recall Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_map = {"logistic_regression": "#1d9e75", "random_forest": "#4f8ef7", "gradient_boosting": "#d85a30"}

for name, info in results.items():
    RocCurveDisplay.from_predictions(
        y_test, info["y_prob"],
        name=f"{name} (AUC={info['roc_auc']:.3f})",
        ax=axes[0], color=colors_map.get(name, "#aaa")
    )
    PrecisionRecallDisplay.from_predictions(
        y_test, info["y_prob"],
        name=f"{name} (AP={info['avg_prec']:.3f})",
        ax=axes[1], color=colors_map.get(name, "#aaa")
    )

for ax, title in zip(axes, ["ROC Curve", "Precision-Recall Curve"]):
    ax.set_title(title, color="white")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("docs/model_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Feature Importance

# %%
fi = get_feature_importance(trained, ALL_FEATURES)

fig, ax = plt.subplots(figsize=(10, 8))
colors  = ["#d85a30" if i < 3 else "#4f8ef7" if i < 7 else "#7b5ea7" for i in range(15)]
fi.head(15).sort_values().plot.barh(ax=ax, color=colors[::-1])
ax.set_title("Feature Importance (Random Forest)", color="white")
ax.set_xlabel("Importance", color="#aaa")
plt.tight_layout()
plt.savefig("docs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Score a Single Wallet

# %%
from src.scoring import build_score_report

example_wallet = {
    "wallet_age_days": 720, "tx_count_total": 350, "unique_active_days": 200,
    "loans_taken": 8, "loans_repaid_on_time": 7, "loans_repaid_late": 1,
    "loans_defaulted": 0, "repayment_rate": 0.94,
    "avg_eth_balance": 12.5, "balance_volatility": 0.15,
    "avg_collateral_ratio": 1.85, "max_collateral_ratio": 2.4,
    "liquidation_count": 0, "protocols_used": 6, "governance_votes": 12,
    "flash_loan_count": 0, "lp_positions_count": 3, "staking_months": 8.0,
    "nft_holdings": 5, "chains_used": 3, "days_since_last_tx": 5,
    "contract_calls_ratio": 0.72,
}

report = build_score_report(example_wallet, best_model, ALL_FEATURES)
for k, v in report.items():
    print(f"  {k:28s}: {v}")
