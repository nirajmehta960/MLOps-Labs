# Lab1 - Docker Labs

This lab demonstrates containerizing an ML pipeline with Docker: training a regression model on the California housing dataset and optionally serving predictions via a Flask API.

---

## Overview

- **Dataset:** California housing (8 features, regression). The model is evaluated with RMSE, MAE, and R².
- **Output:** Predictions are returned in full USD (e.g. $215,590).
- **Approach:** A single multi-stage Dockerfile is used so the final serving image contains only what is needed to run the API, keeping image size down. Train-only and train+serve flows are both supported.

---

## Project structure

```
Lab1 - Docker Labs/
├── README.md
├── Dockerfile             # Multi-stage: train (stage 1) + serve (stage 2)
├── Dockerfile.serving     # Serving image for Compose (deps pre-installed)
├── docker-compose.yml     # Train service + serving service (shared volume)
├── .gitignore
└── src/
    ├── main.py            # Standalone training script
    ├── model_training.py  # Training (Dockerfile + Compose)
    ├── app.py             # Flask API: / and /predict
    ├── requirements.txt
    ├── templates/
    │   └── predict.html
    └── statics/
```

---

## Prerequisites

- Docker (and Docker Compose, typically included with Docker Desktop).

---

## How to run

Run all commands from the **Lab1 - Docker Labs** directory:

```bash
cd "Lab1 - Docker Labs"
```

### Option 1: Train only

Build and run the training stage. The model is saved as `housing_model.pkl` inside the container.

```bash
docker build --target model_training -t lab1-train .
docker run lab1-train
```

To write the model to a folder on your host:

```bash
docker run -v "$(pwd)/output:/app/output" lab1-train
```

The model file will be at `./output/housing_model.pkl`.

---

### Option 2: Train + serve with Docker Compose

Two containers: one trains the model and writes it to a shared volume; the other loads the model and runs the Flask API on port 8080.

```bash
docker compose up --build
```

- **http://localhost:8080** — API welcome message  
- **http://localhost:8080/predict** — Form to enter the 8 features and get a predicted median house value

Stop the stack:

```bash
docker compose down
```

---

### Option 3: Train + serve with multi-stage image

Build a single image that trains the model during the build and then serves it. Only the serving stage is kept in the final image.

```bash
docker build -t lab1-serve .
docker run -p 8080:80 lab1-serve
```

- **http://localhost:8080** — API welcome message  
- **http://localhost:8080/predict** — Prediction form
