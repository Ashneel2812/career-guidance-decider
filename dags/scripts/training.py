"""Train and evaluate the career recommendation SVM model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parent
configured_data_path = os.getenv("TRAINING_DATA_PATH")
DATA_PATH = Path(configured_data_path) if configured_data_path else next(
    (
        candidate
        for candidate in (
            PROJECT_ROOT / "synthetic_data.csv",
            SCRIPT_ROOT / "synthetic_data.csv",
        )
        if candidate.exists()
    ),
    PROJECT_ROOT / "synthetic_data.csv",
)
MODEL_PATH = PROJECT_ROOT / "1_svm_model.pkl"
SCALER_PATH = PROJECT_ROOT / "1_scaler.pkl"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "career-recommendation-s3")
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "career-recommendation-svm"
)
ACCURACY_THRESHOLD = 0.85


def train_model() -> dict[str, object]:
    """Train the SVM and register it as production when accuracy is sufficient."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. Mount synthetic_data.csv "
            "into /opt/airflow/dags or set TRAINING_DATA_PATH to its container path."
        )

    data = pd.read_csv(DATA_PATH)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, random_state=100, test_size=0.3
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    model = SVC(probability=True)
    model.fit(features_train_scaled, labels_train)

    predictions = model.predict(features_test_scaled)
    accuracy = accuracy_score(labels_test, predictions)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be configured for Airflow")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("model_type", "SVC")
        mlflow.log_param("accuracy_threshold", ACCURACY_THRESHOLD)

        registered_version = None
        if accuracy > ACCURACY_THRESHOLD:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            registered_version = model_info.registered_model_version
            if registered_version is not None:
                MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
                    REGISTERED_MODEL_NAME, "prod", registered_version
                )

    result = {
        "accuracy": accuracy,
        "threshold": ACCURACY_THRESHOLD,
        "registered_model": REGISTERED_MODEL_NAME if registered_version else None,
        "registered_version": registered_version,
        "classification_report": classification_report(
            labels_test, predictions, output_dict=True
        ),
        "run_id": run.info.run_id,
    }
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    train_model()