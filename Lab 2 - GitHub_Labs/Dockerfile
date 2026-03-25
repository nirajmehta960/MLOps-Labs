FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY test/ test/

# Set Python path to include /app so imports work
ENV PYTHONPATH=/app

CMD ["python", "src/train.py"]
