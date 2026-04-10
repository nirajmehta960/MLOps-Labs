# DADS 7305 - Machine Learning Operations (MLOps)

## Course Overview
This repository contains the lab assignments for the **DADS 7305 - Machine Learning Operations (MLOps)** course. The course focuses on the principles and practices of deploying, monitoring, and maintaining machine learning models in production environments.

## Lab Assignments

| Lab | Title | Description |
| :--- | :--- | :--- |
| **Lab 1** | **Docker Containers** | Introduction to building and running isolated ML applications with Docker. |
| **Lab 2** | **GitHub Actions & CI/CD** | Building automated pipelines for testing, training, and GCS/Artifact Registry deployment. |
| **Lab 3** | **FastAPI & XGBoost** | Training a Linear Booster XGBoost model and serving predictions via a FastAPI REST endpoint. |
| **Lab 4** | **GCP Cloud Run** | Deploying a Flask recommendation API (GREEN/YELLOW/RED) on Cloud Run with Cloud Storage and BigQuery. |
| **Lab 5** | **Apache Airflow** | Orchestrating an ML pipeline (ingest → preprocess → parallel training → quality gate → manifest or email). |
| **Lab 6** | **ELK Stack** | Docker Compose Elasticsearch, Logstash, and Kibana; JSON Lines from Lab 3 model training and simulated `POST /predict/finance` traffic; create Kibana index patterns (`ml-training-*`, `ml-serving-*`) and use Discover (optional Visualize/Dashboard). |

## Technologies Used
*   **Version Control:** Git & GitHub
*   **Containerization:** Docker
*   **CI/CD:** GitHub Actions
*   **Cloud Provider:** Google Cloud Platform (GCP)
    *   Google Cloud Storage (GCS)
    *   Artifact Registry
    *   Cloud Run
    *   BigQuery
    *   Cloud Build API
    *   IAM (Service Accounts)
*   **Programming Language:** Python 3.9 / 3.10
*   **Machine Learning:** Scikit-learn, XGBoost
*   **API Framework:** FastAPI, Uvicorn; Flask (Lab 4)
*   **Orchestration:** Apache Airflow (Lab 5)
*   **Logging & observability:** Elasticsearch, Logstash, Kibana; structured JSON Lines and file-based ingestion (Lab 6)
*   **Data Validation:** Pydantic
*   **Testing:** Pytest
*   **Data Handling:** Pandas, NumPy
*   **Environment Management:** Python venv, python-dotenv

## Setup & Configuration
The root directory contains configuration files shared across labs:
-   **`.env`**: Environment variables (e.g., GCP Bucket Name).
-   **`config/`**: Configuration files for various services.
-   **`.github/`**: Workflow definitions for CI/CD pipelines.

## How to Navigate
Each lab is organized in its own directory (e.g., `Lab 1 - Docker_Labs/`, `Lab 2 - GitHub_Labs/`, `Lab 3 - Fast_API/`, `Lab 4 - GCP_Labs/`, `Lab 5 - Airflow_Labs/`, `Lab 6 - ELK_Labs/`). Navigate to the specific lab folder to find detailed `README.md` instructions and source code for that assignment.
