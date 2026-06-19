"""
Model Trainer -- Train, evaluate, and save ML models.

Supports RandomForest, GradientBoosting, and XGBoost.
Uses strict chronological train/test split (no shuffling).
Produces a comprehensive evaluation report.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    matthews_corrcoef, f1_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    TRAIN_TEST_SPLIT, RANDOM_STATE, MODELS_DIR,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
)
from ml.features import FEATURE_NAMES


def get_estimators() -> Dict[str, Any]:
    """
    Get available estimators with default hyperparameters.

    Returns dict of {name: estimator_instance}.
    """
    estimators = {
        "RandomForest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=6,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    # Try to add XGBoost if available
    try:
        from xgboost import XGBClassifier
        estimators["XGBoost"] = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
    except ImportError:
        pass

    return estimators


def chronological_split(dataset: pd.DataFrame,
                         train_ratio: float = TRAIN_TEST_SPLIT
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically (no shuffling -- critical for time series).

    Parameters
    ----------
    dataset : pd.DataFrame
        Labeled dataset with feature columns and 'label' column.
    train_ratio : float
        Fraction of data for training (e.g., 0.8).

    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    # Sort by index (timestamp) to ensure chronological order
    dataset = dataset.sort_index()

    split_idx = int(len(dataset) * train_ratio)

    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]

    X_train = train[FEATURE_NAMES]
    X_test = test[FEATURE_NAMES]
    y_train = train["label"]
    y_test = test["label"]

    return X_train, X_test, y_train, y_test


def train_and_evaluate(dataset: pd.DataFrame,
                        estimator_name: str = "RandomForest",
                        pair: str = "EURUSD",
                        timeframe: str = "1h") -> Dict:
    """
    Train a model and produce a full evaluation report.

    Parameters
    ----------
    dataset : pd.DataFrame
        Labeled dataset with feature columns and 'label' column.
    estimator_name : str
        Name of the estimator to use.
    pair : str
        Currency pair (for saving model files).
    timeframe : str
        Timeframe (for saving model files).

    Returns
    -------
    Dict with keys: model, scaler, metrics, feature_importance, model_path
    """
    estimators = get_estimators()
    if estimator_name not in estimators:
        available = ", ".join(estimators.keys())
        raise ValueError(f"Unknown estimator '{estimator_name}'. Available: {available}")

    model = estimators[estimator_name]

    # -- Split data chronologically --------------------------------------------
    X_train, X_test, y_train, y_test = chronological_split(dataset)

    print(f"\n  Train/Test Split:")
    print(f"    Train: {len(X_train):,} samples ({X_train.index[0]} -> {X_train.index[-1]})")
    print(f"    Test:  {len(X_test):,} samples ({X_test.index[0]} -> {X_test.index[-1]})")

    # Check class distribution
    print(f"\n  Train label distribution:")
    for label, count in y_train.value_counts().sort_index().items():
        pct = count / len(y_train) * 100
        label_name = {1: "+1 (profit)", 0: " 0 (timeout)", -1: "-1 (stoploss)"}
        print(f"    {label_name.get(label, str(label))}: {count:,} ({pct:.1f}%)")

    print(f"\n  Test label distribution:")
    for label, count in y_test.value_counts().sort_index().items():
        pct = count / len(y_test) * 100
        label_name = {1: "+1 (profit)", 0: " 0 (timeout)", -1: "-1 (stoploss)"}
        print(f"    {label_name.get(label, str(label))}: {count:,} ({pct:.1f}%)")

    # -- Scale features --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -- Train model -----------------------------------------------------------
    print(f"\n  Training {estimator_name}...")
    model.fit(X_train_scaled, y_train)

    # -- Evaluate --------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Confusion matrix
    labels_present = sorted(set(y_test.unique()) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)

    # Classification report
    class_report = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=[str(l) for l in labels_present],
        zero_division=0,
    )

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=FEATURE_NAMES
        ).sort_values(ascending=False)
    else:
        importance = pd.Series(dtype=float)

    # Probabilities on test set (for threshold analysis)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
    else:
        y_proba = None

    metrics = {
        "estimator": estimator_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "mcc": mcc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "labels_present": labels_present,
    }

    # -- Save model artifacts --------------------------------------------------
    model_dir = MODELS_DIR / f"{pair}_{timeframe}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    importance_path = model_dir / "feature_importance.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    if not importance.empty:
        joblib.dump(importance, importance_path)

    # -- Print Report ----------------------------------------------------------
    print(f"\n{'=' * 56}")
    print(f"  MODEL EVALUATION REPORT -- {estimator_name}")
    print(f"{'=' * 56}")
    print(f"  Train Accuracy : {train_accuracy:.4f}")
    print(f"  Test Accuracy  : {test_accuracy:.4f}")
    print(f"  MCC            : {mcc:.4f}")
    print(f"  F1 (macro)     : {f1_macro:.4f}")

    if train_accuracy - test_accuracy > 0.15:
        print(f"  WARNING OVERFITTING WARNING: Train-Test gap = {train_accuracy - test_accuracy:.4f}")

    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {labels_present}")
    for i, row in enumerate(cm):
        print(f"    {labels_present[i]:>3}: {row}")

    print(f"\n  Classification Report:")
    print(class_report)

    if not importance.empty:
        print(f"  Feature Importance (top 10):")
        for feat, imp in importance.head(10).items():
            bar = "#" * int(imp * 100)
            print(f"    {feat:<22} {imp:.4f}  {bar}")

    print(f"\n  Model saved to: {model_dir}")
    print(f"{'=' * 56}")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": importance,
        "model_dir": model_dir,
    }


def compare_models(dataset: pd.DataFrame,
                    pair: str = "EURUSD",
                    timeframe: str = "1h") -> pd.DataFrame:
    """
    Train and compare all available estimators.

    Returns a summary DataFrame sorted by MCC.
    """
    estimators = get_estimators()
    results = []

    for name in estimators:
        print(f"\n{'-' * 56}")
        print(f"  Training: {name}")
        print(f"{'-' * 56}")

        try:
            result = train_and_evaluate(dataset, name, pair, timeframe)
            m = result["metrics"]
            results.append({
                "Estimator": name,
                "Train Acc": f"{m['train_accuracy']:.4f}",
                "Test Acc": f"{m['test_accuracy']:.4f}",
                "MCC": f"{m['mcc']:.4f}",
                "F1 (macro)": f"{m['f1_macro']:.4f}",
                "Overfit Gap": f"{m['train_accuracy'] - m['test_accuracy']:.4f}",
            })
        except Exception as e:
            print(f"  X Error training {name}: {e}")
            results.append({
                "Estimator": name,
                "Train Acc": "ERROR",
                "Test Acc": "ERROR",
                "MCC": "ERROR",
                "F1 (macro)": "ERROR",
                "Overfit Gap": "ERROR",
            })

    summary = pd.DataFrame(results)
    print(f"\n{'=' * 72}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 72}")
    print(summary.to_string(index=False))
    print(f"{'=' * 72}")

    return summary
