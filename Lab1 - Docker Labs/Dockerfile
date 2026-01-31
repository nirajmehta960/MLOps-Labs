# Stage 1: Build and Train the Model
FROM python:3.10-slim AS model_training

WORKDIR /app

COPY src/model_training.py /app/
COPY src/requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN python model_training.py

CMD ["python", "model_training.py"]


# Stage 2: Serve Predictions
FROM python:3.10-slim AS serving

WORKDIR /app

COPY --from=model_training /app/housing_model.pkl /app/
COPY src/app.py /app/
COPY src/requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY src/templates /app/templates
COPY src/statics /app/statics

EXPOSE 80
# Docker "exposes" port 80 because it is the standard, well-known port for the HTTP (web) protocol

CMD ["python", "app.py"]
