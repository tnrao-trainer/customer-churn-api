import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any

MODEL_PATH = Path(__file__).parent.parent / "model" / "churn_model.pkl"


class ChurnModel:
    """Wrapper for loading and using the churn prediction pipeline."""

    def __init__(self):
        self.pipeline = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the trained sklearn Pipeline (preprocessor + classifier)."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run train_model.py first to generate the model."
            )
        self.pipeline = joblib.load(MODEL_PATH)

    def _prepare_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input dict to a DataFrame matching training feature order."""
        feature_order = [
            "age",
            "tenure",
            "monthly_charges",
            "total_charges",
            "contract",
            "internet_service",
            "support_calls",
            "payment_method",
        ]
        row = {k: data.get(k) for k in feature_order}
        return pd.DataFrame([row])

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction and return probability + label."""
        df = self._prepare_features(data)
        proba = self.pipeline.predict_proba(df)[0][1]
        pred = "Yes" if proba >= 0.5 else "No"
        confidence = float(max(proba, 1 - proba))
        return {
            "churn_probability": round(float(proba), 4),
            "churn_prediction": pred,
            "confidence": round(confidence, 4),
        }


# Singleton instance loaded at import time
model = ChurnModel()
