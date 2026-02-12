# DADS 7305 - Machine Learning Operations (MLOps)

## Course Overview
This repository contains the lab assignments for the **DADS 7305 - Machine Learning Operations (MLOps)** course. The course focuses on the principles and practices of deploying, monitoring, and maintaining machine learning models in production environments.

## Lab Assignments

| Lab | Title | Description | Status |
| :--- | :--- | :--- | :--- |
| **Lab 1** | **Docker Containers** | Introduction to containerization using Docker. Building and running ML applications in isolated environments. | Completed |
| **Lab 2** | **GitHub Actions & CI/CD** | Building a Continuous Integration/Continuous Deployment pipeline using GitHub Actions. Automating testing, training, and deployment to Google Cloud Platform (GCS & Artifact Registry). | Completed |

## Technologies Used
*   **Version Control:** Git & GitHub
*   **Containerization:** Docker
*   **CI/CD:** GitHub Actions
*   **Cloud Provider:** Google Cloud Platform (GCP)
    *   Google Cloud Storage (GCS)
    *   Artifact Registry
    *   Cloud Build API
    *   IAM (Service Accounts)
*   **Programming Language:** Python 3.9
*   **Machine Learning:** Scikit-learn (Random Forest Classifier)
*   **Testing:** Pytest
*   **Data Handling:** Pandas, NumPy
*   **Environment Management:** Python venv, python-dotenv

## Setup & Configuration
The root directory contains configuration files shared across labs:
-   **`.env`**: Environment variables (e.g., GCP Bucket Name).
-   **`config/`**: Configuration files for various services.
-   **`.github/`**: Workflow definitions for CI/CD pipelines.

## How to Navigate
Each lab is organized in its own directory (e.g., `Lab 2 - GitHub Labs/`). Navigate to the specific lab folder to find detailed `README.md` instructions and source code for that assignment.
