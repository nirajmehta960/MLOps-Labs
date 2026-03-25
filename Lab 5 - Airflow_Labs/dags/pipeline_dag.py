"""
Airflow DAG: breast cancer training pipeline.

Ingests data, preprocesses, trains logistic regression and random forest in parallel,
compares accuracies, branches on a quality threshold, then records a JSON manifest,
a completion marker, and a success email (or a failure email on the other branch).
"""
from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.providers.smtp.operators.smtp import EmailOperator

from src.pipeline_tasks import (
    compare_and_select_best,
    ingest_dataset,
    meets_quality_gate,
    preprocess_data,
    train_logistic_regression,
    train_random_forest,
    write_production_manifest,
)

# Minimum test accuracy for the "pass" branch (raise to e.g. 0.99 to force the email path).
QUALITY_THRESHOLD = 0.90

# Notification recipient; set PIPELINE_ALERT_EMAIL in docker-compose / .env.
ALERT_EMAIL = os.environ.get("PIPELINE_ALERT_EMAIL", "you@example.com")

default_args = {
    "owner": "ml-pipeline",
    "start_date": pendulum.datetime(2025, 1, 1, tz="UTC"),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _branch_on_quality(**context) -> str:
    """
    Read compare_models XCom and return the next task_id for BranchPythonOperator.
    """
    comparison = context["ti"].xcom_pull(task_ids="compare_models")
    if meets_quality_gate(comparison, QUALITY_THRESHOLD):
        return "record_production_manifest"
    return "quality_gate_failed_email"


def _write_manifest(**context) -> None:
    """Pull comparison dict from compare_models and write production_manifest.json."""
    comparison = context["ti"].xcom_pull(task_ids="compare_models")
    write_production_manifest(comparison)


with DAG(
    dag_id="breast_cancer_training_pipeline",
    default_args=default_args,
    description="Parallel models, quality gate, success/failure SMTP emails",
    schedule=None,
    catchup=False,
    tags=["breast_cancer", "ml", "training"],
    max_active_runs=1,
) as dag:
    # Step 1: Marker task
    start = BashOperator(
        task_id="start",
        bash_command='echo "Breast cancer ML pipeline run started"',
    )

    # Step 2: Load sklearn dataset to working_data/raw.pkl
    ingest = PythonOperator(
        task_id="ingest_breast_cancer",
        python_callable=ingest_dataset,
    )

    # Step 3: Scale, split, save preprocessed.pkl for downstream tasks
    preprocess = PythonOperator(
        task_id="preprocess_and_split",
        python_callable=preprocess_data,
        op_args=[ingest.output],
    )

    # Step 4a / 4b: Same input, two estimators (parallel in the graph)
    train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_logistic_regression,
        op_args=[preprocess.output],
    )

    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_random_forest,
        op_args=[preprocess.output],
    )

    # Step 5: Pick best model by accuracy; XCom holds the comparison dict
    compare_models = PythonOperator(
        task_id="compare_models",
        python_callable=compare_and_select_best,
        op_args=[train_lr.output, train_rf.output],
    )

    # Step 6: Branch to manifest + done + success email, or failure email
    branch = BranchPythonOperator(
        task_id="quality_gate_branch",
        python_callable=_branch_on_quality,
    )

    record_production_manifest = PythonOperator(
        task_id="record_production_manifest",
        python_callable=_write_manifest,
    )

    done = BashOperator(
        task_id="pipeline_complete",
        bash_command='echo "Quality gate passed — manifest written under working_data/"',
    )

    quality_gate_failed_email = EmailOperator(
        task_id="quality_gate_failed_email",
        to=ALERT_EMAIL,
        subject="[ML pipeline] Model quality gate failed",
        html_content=(
            f"<p>The best model did not reach the configured threshold "
            f"of <b>{QUALITY_THRESHOLD:.0%}</b>.</p>"
            "<p>Inspect task logs for <code>compare_models</code> and training tasks.</p>"
        ),
    )

    # Sent only on the success path after training + manifest + completion echo
    pipeline_success_email = EmailOperator(
        task_id="pipeline_success_email",
        to=ALERT_EMAIL,
        subject="[ML pipeline] Breast cancer training pipeline completed successfully",
        html_content=(
            "<p><strong>Your training pipeline finished successfully.</strong></p>"
            "<p>Ingest, preprocessing, parallel model training, and model comparison "
            "completed. The quality gate passed.</p>"
            "<p><strong>Best model:</strong> "
            "{{ ti.xcom_pull(task_ids='compare_models')['best_name'] }}<br/>"
            "<strong>Test accuracy:</strong> "
            "{{ ti.xcom_pull(task_ids='compare_models')['best_accuracy'] }}</p>"
            "<p>A production manifest was written to "
            "<code>working_data/production_manifest.json</code>.</p>"
        ),
    )

    # Task dependencies: fork after preprocess, join at compare, then branch
    start >> ingest >> preprocess >> [train_lr, train_rf]
    [train_lr, train_rf] >> compare_models >> branch
    branch >> record_production_manifest >> done >> pipeline_success_email
    branch >> quality_gate_failed_email
