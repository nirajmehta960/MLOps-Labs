# GitHub Actions Lab: Iris Classification MLOps Pipeline

This repository demonstrates a complete **Machine Learning Operations (MLOps)** pipeline involving Continuous Integration and Continuous Deployment (CI/CD). It automates the testing, training, versioning, and containerization of a Machine Learning model using **GitHub Actions** and **Google Cloud Platform (GCP)**.

## Key Features

*   **Automated Testing:** Runs unit tests (`pytest`) on every commit to ensure code integrity.
*   **Model Training & Versioning:**
    *   Trains a **Random Forest Classifier** on the Iris dataset.
    *   Automatically increments model versions (e.g., `v1`, `v2`) in Google Cloud Storage.
    *   Saves artifacts with timestamps for full traceability.
*   **Containerization:** Builds a Docker image containing the trained model and dependencies.
*   **Cloud Deployment:** Pushes the Docker image to **Google Artifact Registry**.

---

## Project Structure

```
.
├── .github/workflows/
│   └── ci_cd.yaml       # Defines the CI/CD pipeline steps
├── src/
│   └── train.py         # Main script: Data loading, training, versioning, and uploading
├── test/
│   └── test_train.py    # Unit tests for the training pipeline
├── config/              # Configuration files
├── Dockerfile           # Blueprint for the Docker image
├── requirements.txt     # Python dependencies
└── GCP_SETUP_GUIDE.md   # Step-by-step guide for GCP configuration
```

---

## Setup & Prerequisites

Before running the pipeline, ensure you have the following:

1.  **Google Cloud Platform Account:** With billing enabled.
2.  **GCP Resources:**
    *   A **Google Cloud Storage Bucket** (to store models).
    *   An **Artifact Registry Repository** named `iris-pipeline-repo` in `us-east1`.
    *   A **Service Account** with `Storage Admin` and `Artifact Registry Writer` roles.
    *   *(See [GCP_SETUP_GUIDE.md](GCP_SETUP_GUIDE.md) for detailed setup instructions)*.
3.  **GitHub Secrets:**
    *   `GCP_PROJECT_ID`
    *   `GCP_SA_KEY`
    *   `GCS_BUCKET_NAME`

---

## How to Run the Lab

### Option 1: Triggering the CI/CD Pipeline (Recommended)

The pipeline is designed to run automatically whenever you push changes to the `main` branch.

1.  **Make a Change:**
    Edit a file (e.g., add a comment to `src/train.py` or `README.md`).
2.  **Commit and Push:**
    ```bash
    git add .
    git commit -m "Trigger CI/CD pipeline"
    git push origin main
    ```
3.  **Monitor Execution:**
    *   Go to the **Actions** tab in your GitHub repository.
    *   Click on the running workflow (e.g., "AI/ML CI/CD Pipeline").
    *   Watch as it executes `Test`, `Train`, and `Build & Push` steps.

### Option 2: Running Locally

You can run the training script locally to verify it works before pushing.

1.  **Set Up Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```ini
    GCS_BUCKET_NAME=your-bucket-name
    # Optional: If you want to upload to GCS locally, you need authentication.
    # Otherwise, the script will just save locally.
    ```
3.  **Run the Training Script:**
    ```bash
    python src/train.py
    ```
    *Output:* This will train the model, save it as `model.joblib`, and (if credentials are set) upload it to your GCS bucket.

---

## Verification

After the pipeline runs successfully, verify the output:

1.  **Artifact Registry:**
    *   Go to **GCP Console > Artifact Registry > iris-pipeline-repo**.
    *   You should see a new Docker image tagged with the commit SHA and `latest`.
2.  **Google Cloud Storage:**
    *   Go to **GCP Console > Cloud Storage > Buckets > [Your Bucket]**.
    *   Check for a `trained_models/` folder containing the saved model (e.g., `model_v1_...joblib`).
    *   Check `model_version.txt` to see the current version code.

---

## Pipeline Workflow

The `.github/workflows/ci_cd.yaml` file defines the following steps:

1.  **Checkout Code:** Pulls the latest code from GitHub.
2.  **Install Dependencies:** Installs Python libraries from `requirements.txt`.
3.  **Run Tests:** Executes `pytest` to validate logic.
4.  **Authenticate to GCP:** A uses the `GCP_SA_KEY` secret.
5.  **Train & Version:** Runs `src/train.py` to train and upload the model.
6.  **Build Docker Image:** Creates a container with the model.
7.  **Push to Artifact Registry:** Uploads the container for deployment.
