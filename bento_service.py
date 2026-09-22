"""BentoML service for the MLflow production career model."""

from __future__ import annotations

import os
from pathlib import Path

import bentoml
import joblib
import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)
MLFLOW_REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "career-recommendation-svm"
)
MLFLOW_PRODUCTION_MODEL_URI = (
    f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@prod"
)


@bentoml.service
class CareerRecommendationService:
    """Serve predictions from the model currently aliased as prod in MLflow."""

    def __init__(self) -> None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.sklearn.load_model(MLFLOW_PRODUCTION_MODEL_URI)
        self.scaler = joblib.load(PROJECT_ROOT / "1_scaler.pkl")

    @bentoml.api
    def predict(self, features: list[float]) -> dict[str, object]:
        """Return a career prediction for four unscaled model features."""
        if len(features) != 4:
            raise ValueError("features must contain exactly 4 values")

        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        return {
            "model": MLFLOW_REGISTERED_MODEL_NAME,
            "alias": "prod",
            "prediction": prediction,
        }
