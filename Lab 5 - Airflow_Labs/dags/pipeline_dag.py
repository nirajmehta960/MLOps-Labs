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

# Quality gate threshold.
QUALITY_THRESHOLD = 0.90

# Notification recipient from environment.
ALERT_EMAIL = (os.environ.get("PIPELINE_ALERT_EMAIL") or "").strip()

default_args = {
    "owner": "ml-pipeline",
    "start_date": pendulum.datetime(2025, 1, 1, tz="UTC"),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _branch_on_quality(**context) -> str:
    """
    Evaluate the best model's accuracy against our quality gate threshold using XCom data.
    If the threshold is met, the pipeline proceeds to record the production manifest.
    Otherwise, it aborts the deployment process and sends a failure alert email.
    
    Returns:
        str: The task_id of the next execution branch.
    """
    comparison = context["ti"].xcom_pull(task_ids="compare_models")
    if meets_quality_gate(comparison, QUALITY_THRESHOLD):
        return "record_production_manifest"
    return "quality_gate_failed_email"


def _write_manifest(**context) -> None:
    """
    Retrieve the evaluated model metrics from the `compare_models` task (via XCom)
    and delegate the artifact creation to the `write_production_manifest` utility.
    This manifest signals successful model training to downstream systems.
    """
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
    # Start task
    start = BashOperator(
        task_id="start",
        bash_command='echo "Breast cancer ML pipeline run started"',
    )

    # Ingest dataset
    ingest = PythonOperator(
        task_id="ingest_breast_cancer",
        python_callable=ingest_dataset,
    )

    # Preprocess data
    preprocess = PythonOperator(
        task_id="preprocess_and_split",
        python_callable=preprocess_data,
        op_args=[ingest.output],
    )

    # Train models in parallel
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

    # Compare model performance
    compare_models = PythonOperator(
        task_id="compare_models",
        python_callable=compare_and_select_best,
        op_args=[train_lr.output, train_rf.output],
    )

    # Branch on quality gate
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

    # Success notification
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

    # --- Define Task Dependencies ---
    
    # Linear flow up to preprocessing
    start >> ingest >> preprocess 
    
    # Parallel execution: branch out to train both models simultaneously
    preprocess >> [train_lr, train_rf]
    
    # Wait for both models to finish, then compare them
    [train_lr, train_rf] >> compare_models >> branch
    
    # Conditional branches depending on compare_models output
    # Pass path
    branch >> record_production_manifest >> done >> pipeline_success_email
    # Fail path
    branch >> quality_gate_failed_email
