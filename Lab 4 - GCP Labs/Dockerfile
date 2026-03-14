# Stage 1: Train recommendation model from SavVio training_scenarios data
FROM python:3.10-slim AS train

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ ./data/
COPY src/model_training.py .
RUN python model_training.py

# Stage 2: Serve on Cloud Run (port 8080)
FROM python:3.10-slim AS serve

WORKDIR /app
COPY --from=train /app/recommendation_model.pkl /app/scaler.pkl /app/encoder.pkl /app/feature_config.joblib ./
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/app.py ./

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8080} --workers 1 app:app"]
