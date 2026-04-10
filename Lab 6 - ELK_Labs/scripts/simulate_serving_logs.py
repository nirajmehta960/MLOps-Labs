#!/usr/bin/env python3
"""
Append simulated FastAPI-style prediction requests as JSON Lines to logs/serving.jsonl.
Mirrors fields you would log from Lab 3 (POST /predict/finance): status, latency, route.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
LOG_FILE = BASE_DIR / "logs" / "serving.jsonl"
DATASET = "ml.serving"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic serving logs for Kibana demos.")
    parser.add_argument("--n", type=int, default=40, help="Number of request events to append.")
    args = parser.parse_args()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    weights = [("200", 0.88), ("400", 0.08), ("422", 0.03), ("500", 0.01)]
    codes = [c for c, _ in weights]
    probs = [p for _, p in weights]

    with LOG_FILE.open("a", encoding="utf-8") as f:
        for _ in range(args.n):
            status = random.choices(codes, weights=probs, k=1)[0]
            latency_ms = random.randint(8, 320)
            if status != "200":
                latency_ms = min(latency_ms, 120)

            row = {
                "timestamp": utc_now_iso(),
                "level": "INFO" if status == "200" else "WARN",
                "dataset": DATASET,
                "message": "predict_request",
                "http": {
                    "method": "POST",
                    "route": "/predict/finance",
                    "status_code": int(status),
                },
                "event": {"duration_ms": latency_ms},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
