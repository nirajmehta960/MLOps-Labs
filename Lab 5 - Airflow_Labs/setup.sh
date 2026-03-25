#!/usr/bin/env bash
# Create folders and starter .env
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs plugins config working_data model

if [[ ! -f .env ]]; then
  echo "AIRFLOW_UID=$(id -u)" > .env
  echo "AIRFLOW_PROJ_DIR=$(pwd)" >> .env
  echo "" >> .env
  echo "Created .env with AIRFLOW_UID and AIRFLOW_PROJ_DIR."
else
  echo ".env already exists; skipping."
fi
