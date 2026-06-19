"""
ML Predictor — Load a trained model and predict on new data.

Provides a cached inference wrapper that loads model + scaler once
and predicts class probabilities for each bar.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODELS_DIR
from ml.features import FEATURE_NAMES


class MLPredictor:
    """
    Cached ML predictor that loads model artifacts once and
    provides probability predictions for new feature vectors.

    Usage
    -----
    predictor = MLPredictor("EURUSD", "1h")
    probs = predictor.predict_proba(feature_dict)
    # probs = {1: 0.55, 0: 0.30, -1: 0.15}
    """

    def __init__(self, pair: str, timeframe: str, model_dir: Optional[Path] = None):
        self.pair = pair
        self.timeframe = timeframe

        if model_dir is None:
            model_dir = MODELS_DIR / f"{pair}_{timeframe}"

        model_path = model_dir / "model.pkl"
        scaler_path = model_dir / "scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train a model first using ml/trainer.py"
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.classes = self.model.classes_

        # Load feature importance if available
        importance_path = model_dir / "feature_importance.pkl"
        if importance_path.exists():
            self.feature_importance = joblib.load(importance_path)
        else:
            self.feature_importance = None

        print(f"  OK Model loaded: {pair} {timeframe} ({type(self.model).__name__})")
        print(f"    Classes: {list(self.classes)}")

    def predict_proba(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Predict class probabilities for a single feature vector.

        Parameters
        ----------
        features : dict
            Feature name -> value mapping. Must contain all FEATURE_NAMES.

        Returns
        -------
        Dict[int, float]
            Class -> probability mapping.
            e.g., {1: 0.55, 0: 0.30, -1: 0.15}
        """
        # Build feature vector in the correct order
        feature_vector = np.array([[features[name] for name in FEATURE_NAMES]])

        # Scale
        feature_scaled = self.scaler.transform(feature_vector)

        # Predict probabilities
        probas = self.model.predict_proba(feature_scaled)[0]

        # Map to class labels
        return {int(cls): float(prob) for cls, prob in zip(self.classes, probas)}

    def predict_proba_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for all rows in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with all feature columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with prob_+1, prob_0, prob_-1 columns.
        """
        X = df[FEATURE_NAMES].values
        X_scaled = self.scaler.transform(X)
        probas = self.model.predict_proba(X_scaled)

        result = pd.DataFrame(
            probas,
            columns=[f"prob_{int(c)}" for c in self.classes],
            index=df.index,
        )

        return result

    def predict_class(self, features: Dict[str, float]) -> int:
        """Predict the most likely class label."""
        probs = self.predict_proba(features)
        return max(probs, key=probs.get)
