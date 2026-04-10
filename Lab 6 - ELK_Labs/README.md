# Lab 6: ELK Stack for ML Observability

This lab runs **Elasticsearch**, **Logstash**, and **Kibana** in Docker and ships **structured JSON logs** from your **Lab 3** training pipeline plus **simulated FastAPI serving** events—so you can search and chart ML operations in Kibana the same way you would for a real service.

---

## What you will practice

- **Docker Compose** for a local ELK stack (single-node Elasticsearch, no security—dev only).
- **JSON Lines** (`.jsonl`) as a simple, grep-friendly log format for agents and Logstash.
- **Logstash** `file` inputs + **Elasticsearch** indices split by `dataset`: `ml-training-*` and `ml-serving-*`.
- **Kibana Discover** (and optional visualizations) on training lifecycle and HTTP/latency fields.

---

## Prerequisites

- **Docker** and **Docker Compose** (Compose V2: `docker compose`).
- **Python 3.10+** and a virtual environment (recommended).
- **Lab 3 data and modules** available at `Lab 3 - Fast_API/` (same repo root)—`train_and_log.py` imports `src.data` and `src.features` and expects `Lab 3 - Fast_API/data/financial_data.csv`.

---

## Project layout

```text
Lab 6 - ELK_Labs/
├── README.md
├── docker-compose.yml      # elasticsearch, logstash, kibana
├── requirements.txt
├── logstash/
│   └── logstash.conf       # tails *.jsonl → ES indices
├── logs/                   # training.jsonl & serving.jsonl (gitignored)
└── scripts/
    ├── train_and_log.py    # Lab 3 XGBoost train + JSON logs
    └── simulate_serving_logs.py   # synthetic /predict/finance events
```

---

## 1. Start the ELK stack

From the **MLOps Labs** repo root:

```bash
cd "Lab 6 - ELK_Labs"
docker compose up -d
```

Create the log files once so Logstash’s `file` inputs have a path to watch (they are populated when you run the scripts):

```bash
touch logs/training.jsonl logs/serving.jsonl
```

Wait until Elasticsearch answers (may take ~30–60s the first time):

```bash
curl -s http://localhost:9200 | head
```

Open **Kibana**: [http://localhost:5601](http://localhost:5601)

---

## 2. Python environment and log producers

```bash
cd "Lab 6 - ELK_Labs"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Training logs** (reuses Lab 3 pipeline; updates `Lab 3 - Fast_API/model/`):

```bash
python scripts/train_and_log.py
```

**Serving logs** (synthetic API traffic):

```bash
python scripts/simulate_serving_logs.py --n 50
```

Log files appear under `logs/training.jsonl` and `logs/serving.jsonl`. Logstash **tails** them and indexes into Elasticsearch (see `logstash/logstash.conf`).

---

## 3. Kibana: set up ML training and serving

Use **Stack Management** and **Discover**. You do **not** need **Integrations** or Elastic Agent—this lab uses Logstash only.

### 3.1 Confirm Elasticsearch has documents

After running the training and serving scripts, check that indices exist (names include today’s date):

```bash
curl -s "http://localhost:9200/_cat/indices/ml-*?v"
```

You should see `ml-training-YYYY.MM.DD` and `ml-serving-YYYY.MM.DD`. If the list is empty, ensure `docker compose` is running, run the Python scripts again, wait a few seconds, and retry.

### 3.2 Create two index patterns (training vs serving)

1. Open Kibana: [http://localhost:5601](http://localhost:5601).
2. Open the **main menu** (top left) → **Stack Management** → under **Kibana** choose **Index Patterns** (in newer UIs this may appear as **Data views**—same idea).
3. Click **Create index pattern**.
4. **ML training**
   - Index pattern name: **`ml-training-*`**
   - Click **Next step** (or continue).
   - **Time field:** choose **`@timestamp`**.
   - Click **Create index pattern**.
5. **ML serving**
   - Click **Create index pattern** again.
   - Index pattern name: **`ml-serving-*`**
   - **Time field:** **`@timestamp`**.
   - **Create index pattern**.

You now have separate views for training pipeline logs and API-style serving logs.

### 3.3 Discover: ML training logs

1. **Menu → Discover**.
2. In the index pattern dropdown (top left), select **`ml-training-*`**.
3. Set the **time range** (e.g. **Last 15 minutes** or **Last 7 days**) so it covers when you ran `train_and_log.py`.
4. Click **Refresh** if the histogram is empty.
5. In the **Available fields** list (left), add columns you care about, for example:
   - **`message`** — pipeline steps (`training_pipeline_started`, `data_loaded`, `model_training_completed`, …).
   - **`ml.rows`**, **`ml.train_rows`**, **`ml.test_rows`** — data sizes.
   - **`ml.booster`** — e.g. `gblinear`.
   - **`ml.test_accuracy`** — holdout accuracy after training.
   - **`ml.status`** — e.g. `success` on the final event.
6. **Optional filters (KQL)** in the search bar, e.g.:
   - `message: "model_training_completed"` — only the row with test accuracy.
   - `level: "ERROR"` — surface failures if you inject errors later.

Training runs produce **few documents** (one per pipeline milestone); that is expected.

### 3.4 Discover: ML serving logs

1. **Discover** → index pattern **`ml-serving-*`**.
2. Use a **time range** that includes when you ran `simulate_serving_logs.py`.
3. Add columns such as:
   - **`http.method`**, **`http.route`**, **`http.status_code`** — request shape and outcome (`200`, `400`, etc.).
   - **`event.duration_ms`** — simulated latency.
   - **`level`** — often `INFO` for `200`, `WARN` for client/server errors in the simulation.
4. **Optional KQL examples:**
   - `http.status_code: 400` — bad requests only.
   - `http.status_code: 200` — successes only.
   - `event.duration_ms > 200` — slow requests (adjust the threshold as you like).

Serving scripts produce **many documents** (one per simulated request).

### 3.5 Optional: saved searches, Visualize, and Dashboard

- **Saved search:** In **Discover**, set filters and columns as above → **Save** (top right) → name it e.g. `Training – completed` or `Serving – errors`.
- **Charts:** **Menu → Visualize** → **Create visualization** → pick a type (e.g. **Vertical bar**). Choose index **`ml-serving-*`**, add a **Terms** bucket on **`http.status_code`** (use the aggregatable field Kibana suggests—sometimes shown with a `.keyword` suffix), or use **Average** on **`event.duration_ms`** over time.
- **Dashboard:** **Menu → Dashboard** → **Create dashboard** → **Add** → add your saved searches or visualizations to compare training milestones with serving traffic on one screen.

If documents do not appear, widen the time range, click **Refresh**, and confirm indices with the `curl` command in §3.1.

---

## 4. Shut down

```bash
cd "Lab 6 - ELK_Labs"
docker compose down
```

To remove Elasticsearch data as well:

```bash
docker compose down -v
```
