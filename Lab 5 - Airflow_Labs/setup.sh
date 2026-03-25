#!/usr/bin/env bash
#
# Create local folders and a starter .env for Docker Compose.
# Run from this directory (where docker-compose.yaml lives) before `docker compose up airflow-init`.
#
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs plugins config working_data model

if [[ ! -f .env ]]; then
  echo "AIRFLOW_UID=$(id -u)" > .env
  echo "AIRFLOW_PROJ_DIR=$(pwd)" >> .env
  echo "Created .env with AIRFLOW_UID and AIRFLOW_PROJ_DIR. Add PIPELINE_ALERT_EMAIL and SMTP_* if needed."
else
  echo ".env already exists; skipping."
fi
