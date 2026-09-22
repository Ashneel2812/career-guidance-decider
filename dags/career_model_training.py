"""Airflow DAG for training and conditionally promoting the career model."""

import sys
from datetime import datetime
from pathlib import Path

# Airflow mounts this directory at /opt/airflow/dags. Keep task code inside
# that mounted tree so the scheduler and workers resolve the same module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airflow import DAG
from airflow.operators.python import PythonOperator

from scripts.training import train_model


with DAG(
    dag_id="career_model_training",
    start_date=datetime(2024, 1, 1),
    schedule="0 * * * *",
    catchup=False,
    tags=["ml", "training"],
) as dag:
    train_and_promote_model = PythonOperator(
        task_id="train_and_promote_model",
        python_callable=train_model,
    )