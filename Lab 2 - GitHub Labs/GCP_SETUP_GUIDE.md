# Google Cloud Platform (GCP) Setup Guide
These instructions will guide you through setting up the necessary Google Cloud resources to automate your Machine Learning pipeline.

**Prerequisites:**
- A Google Cloud Platform Account (Sign up [here](https://cloud.google.com/)).
- Billing enabled for your project.

## 1. Create a Google Cloud Storage (GCS) Bucket
This bucket will store your trained model artifacts.

1.  Open the [Google Cloud Console](https://console.cloud.google.com/).
2.  In the navigation menu, go to **Cloud Storage > Buckets**.
3.  Click **+ CREATE**.
4.  **Name your bucket:** Choose a globally unique name (e.g., ` iris-pipeline-repo`).
    *   *Note: This name will be used for the `GCS_BUCKET_NAME` secret in GitHub.*
5.  **Location type:** Choose `Region` and select `us-east1` (or your preferred region).
6.  **Storage class:** Leave as `Standard`.
7.  **Access control:** Leave as `Uniform`.
8.  Click **CREATE**.
    *   *Note: If prompted about "Public access prevention", keep it enforced (recommended).*

## 2. Enable Required APIs
To allow GitHub Actions to push Docker images and interact with your project, you need to enable specific APIs.

1.  Go to **Loop APIs & Services > Library** in the GCP Console.
2.  Search for and **Enable** the following APIs:
    *   **Artifact Registry API** (Crucial for storing Docker images)
    *   **Cloud Operations API** (Optional, for logging)
    *   **IAM API** (Identity and Access Management)

## 3. Create Artifact Registry Repository
This is where your Docker images will be stored.

1.  Search for **Artifact Registry** in the GCP Console.
2.  Click **+ CREATE REPOSITORY**.
3.  **Name:** `iris-pipeline-repo`
    *   *Note: This MUST match the `REPO_NAME` variable in your `.github/workflows/ci_cd.yaml` file.*
4.  **Format:** `Docker`
5.  **Mode:** `Standard`
6.  **Location type:** `Region`
7.  **Region:** `us-east1`
    *   *Note: This MUST match the `REGION` variable in your `.github/workflows/ci_cd.yaml` file.*
8.  Click **CREATE**.

## 4. Create a Service Account (SA)
This account allows GitHub Actions to act on your behalf.

1.  Go to **IAM & Admin > Service Accounts**.
2.  Click **+ CREATE SERVICE ACCOUNT**.
3.  **Service account name:** `github-actions-sa` (or similar).
4.  **Service account ID:** Auto-filled.
5.  Click **CREATE AND CONTINUE**.
6.  **Grant this service account access to project:**
    *   Add Role: **Storage Admin** (Access to your Bucket)
    *   Add Role: **Artifact Registry Writer** (Access to push images)
    *   Add Role: **Service Account User** (Required for some deployment actions)
7.  Click **CONTINUE** and then **DONE**.

## 5. Generate & Download Key
1.  Click on the newly created Service Account (email address link).
2.  Go to the **KEYS** tab.
3.  Click **ADD KEY > Create new key**.
4.  Select **JSON**.
5.  Click **CREATE**.
6.  A `.json` file will automatically download to your computer. **Keep this safe!**

## 6. Configure GitHub Secrets
Now, link your GCP credentials to your GitHub Repository.

1.  Go to your GitHub Repository.
2.  Navigate to **Settings > Secrets and variables > Actions**.
3.  Click **New repository secret** for each of the following:

| Name | Value |
| :--- | :--- |
| `GCP_PROJECT_ID` | Your GCP Project ID (found on the dashboard home). |
| `GCS_BUCKET_NAME` | The name of the bucket you already created. |
| `GCP_SA_KEY` | Open the downloaded JSON key file, copy the **entire content** (curly braces and all), and paste it here. |

## 7. Verification
Once these steps are done:
1.  Commit and push your code to the `main` branch.
2.  Go to the **Actions** tab in GitHub.
3.  You should see the `AI/ML CI/CD Pipeline` workflow running.
4.  If successful, check your **Artifact Registry** to see the new Docker image!
