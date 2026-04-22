"""
train.py — Model training, evaluation, and persistence
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import ALL_FEATURES, engineer_features, generate_synthetic_wallets
from src.scoring import generate_labels, probability_to_credit_score


# ── Model definitions ──────────────────────────────────────────────────────────

def build_models() -> dict:
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                ),
                method="isotonic",
                cv=3,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )),
        ]),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Train all models with stratified cross-validation.

    Returns dict: model_name → {"model": Pipeline, "cv_roc_auc": float, "cv_std": float}
    """
    models  = build_models()
    trained = {}
    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        trained[name] = {
            "model":      pipe,
            "cv_roc_auc": float(cv_scores.mean()),
            "cv_std":     float(cv_scores.std()),
        }
        if verbose:
            print(f"  {name:25s}  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return trained


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Evaluate all trained models on held-out test set.

    Returns dict: model_name → {"model", "y_prob", "roc_auc", "avg_prec"}
    """
    results = {}
    for name, info in trained_models.items():
        model  = info["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc    = roc_auc_score(y_test, y_prob)
        ap     = average_precision_score(y_test, y_prob)
        results[name] = {
            "model":    model,
            "y_prob":   y_prob,
            "roc_auc":  roc,
            "avg_prec": ap,
        }
        if verbose:
            print(f"\n  {name}  ROC-AUC: {roc:.4f}  Avg-Precision: {ap:.4f}")
            print(classification_report(y_test, y_pred, target_names=["High Risk", "Creditworthy"]))

    return results


# ── Feature importance ─────────────────────────────────────────────────────────

def get_feature_importance(trained_models: dict, feature_names: list) -> pd.Series:
    """Extract averaged feature importances from the Random Forest model."""
    calib = trained_models["random_forest"]["model"].named_steps["clf"]
    importances = np.mean([
        est.base_estimator.feature_importances_
        if hasattr(est, "base_estimator")
        else est.estimator.feature_importances_
        for est in calib.calibrated_classifiers_
    ], axis=0)
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def load_model(path: str | Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(results: dict, feature_importance: pd.Series, path: str | Path) -> None:
    metrics = {
        "model_metrics": {
            k: {"roc_auc": round(v["roc_auc"], 4), "avg_prec": round(v["avg_prec"], 4)}
            for k, v in results.items()
        },
        "feature_importance": {k: round(float(v), 6) for k, v in feature_importance.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {path}")


# ── Main entry point ───────────────────────────────────────────────────────────

def run_pipeline(
    n_wallets: int = 2000,
    test_size: float = 0.2,
    save_dir: str = "models",
    verbose: bool = True,
) -> dict:
    """
    End-to-end training pipeline.

    1. Generate synthetic data (replace with real indexer data for production)
    2. Engineer features
    3. Generate labels
    4. Train + evaluate all models
    5. Save best model + metrics

    Returns results dict.
    """
    print("=" * 60)
    print("  DeFi Credit Scoring — Training Pipeline")
    print("=" * 60)

    print(f"\n[1/4] Generating {n_wallets} synthetic wallets...")
    raw     = generate_synthetic_wallets(n_wallets)
    df      = engineer_features(raw)
    labels, _ = generate_labels(df)
    print(f"      Creditworthy: {labels.sum()} ({labels.mean()*100:.1f}%)")

    X = df[ALL_FEATURES]
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"      Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[2/4] Training models...")
    trained = train(X_train, y_train, verbose=verbose)

    print("\n[3/4] Evaluating on test set...")
    results = evaluate(trained, X_test, y_test, verbose=verbose)

    print("\n[4/4] Saving artifacts...")
    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    fi         = get_feature_importance(trained, ALL_FEATURES)

    save_model(best_model, Path(save_dir) / "best_model.pkl")
    save_metrics(results, fi, Path(save_dir) / "metrics.json")

    print(f"\n✓ Best model: {best_name}  AUC: {results[best_name]['roc_auc']:.4f}")
    return {"results": results, "best_model": best_model, "feature_importance": fi}


if __name__ == "__main__":
    run_pipeline()
