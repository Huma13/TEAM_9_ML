# PathSnap — Distributed Map-Matching & Route Reconstruction (Laptop 1)

Local, end-to-end machine learning pipeline and FastAPI inference backend for the **PathSnap** map-matching hackathon challenge (PES University — UE24CS352A).

> **Local-only discipline:** All experiments, training, Bayesian optimization, HMM/Viterbi decoding, and API serving run locally on Laptop 1 (Apple M2, 8-core, 8 GB RAM). Original organiser files under `Guidelines/` are preserved unmodified. Reserved evaluator trips remain strictly outside all development decisions.

---

## Project Structure

```text
Pathsnap/
├── Guidelines/                 # Original organiser PDFs and Student folder (untouched)
├── src/
│   ├── data_io.py              # Dataset discovery, headerless parsers, trip-local AEQD meter projection
│   ├── eda.py                  # Milestone 1: inventory, quality checks, sampling gaps, Folium map (00000005)
│   ├── candidates.py           # Milestone 2 (Inference-only): STRtree spatial index & 21 leak-free features
│   ├── alignment.py            # Milestone 2 (Dev/Server-side ONLY): monotonic DP alignment to .route
│   ├── build_dataset.py        # Milestone 2 runner: caches parquet features/labels, baseline & worked example
│   ├── metrics.py              # Candidate-level, observation-level Top-1, continuity & sequence-level metrics
│   ├── models.py               # Milestone 3: 5-model 5-fold GroupKFold CV & student-test comparison
│   ├── tune.py                 # Milestone 4: Optuna TPE Bayesian optimization & artifact export
│   ├── hmm_viterbi.py          # Milestone 5: HMM/Viterbi route reconstruction & labelled Folium map
│   ├── inference.py            # Pure route-free predictor + separate server-side scorer
│   ├── test_api.py             # Milestone 6: end-to-end FastAPI test, restart & missing-eval-dir check
│   └── build_report.py         # Milestone 7: generates report/TECHNICAL_REPORT.md from measured outputs
├── backend/
│   └── main.py                 # FastAPI app (GET /health, GET /trips, POST /predict)
├── artifacts/                  # Exported final model bundle, feature spec, model config & HMM config
├── outputs/                    # Saved CSVs, JSONs, plots & interactive Folium HTML maps (milestones 1–6)
├── laptop2_handoff/            # API contract & live example responses for Laptop 2 (no raw data/weights)
├── report/                     # Consolidated technical report (draft pending Laptop 2/Laptop 1 screenshots)
├── PROGRESS.md                 # Short phase-wise evaluation checklist with reproduction commands
└── requirements.txt            # Pinned Python 3.12 dependencies
```

---

## 1. Environment Setup (Laptop 1)

All commands are runnable from the `Pathsnap/` project root using the project-local Python 3.12 virtual environment (`.venv`).

```bash
cd /Users/ud/Desktop/Pathsnap
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Data Locations

- **Development dataset (auto-discovered recursively under `Guidelines/`):**
  - Training (80 trips): `Guidelines/Student folder/train/`
  - Student test (10 trips): `Guidelines/Student folder/student_test/`
- **Reserved evaluation dataset (final demo only):**
  - Place the organiser-supplied 10 reserved trip folders in any local directory outside `Guidelines/Student folder` (for example, `/Users/ud/Desktop/Pathsnap_eval_trips/`) and pass it via `PATHSNAP_EVAL_DIR` when switching the backend to `evaluation` mode.

---

## 3. Reproducing the ML Pipeline (Milestones 1–5)

Activate `.venv` first (`source .venv/bin/activate`), then run from the `Pathsnap/` root:

```bash
# Milestone 1: Dataset inventory, quality checks, sampling-gap plot & Folium map for trip 00000005
python -m src.eda

# Milestone 2: Route-free spatial candidates, 21 inference features, monotonic alignment & nearest-road baseline
python -m src.build_dataset

# Milestone 3 (smoke test, ~30s): sanity-check all 5 models on 10 train / 3 test trips
python -m src.models --smoke

# Milestone 3 (full run, resumable): 5-fold GroupKFold CV & student-test comparison across all 5 models
python -m src.models --resume

# Milestone 4: Bayesian hyperparameter optimisation (Optuna TPE) + artifact export + subprocess reload check
python -m src.tune --trials 12

# Milestone 5: HMM/Viterbi route reconstruction, classifier vs HMM comparison & labelled Folium map
python -m src.hmm_viterbi

# Milestone 6: Automated API verification across all 10 student-test trips, restart & eval-dir error check
python -m src.test_api

# Milestone 7: Regenerate consolidated technical report from measured outputs
python -m src.build_report
```

---

## 4. Running the FastAPI Backend (Laptop 1)

### Development Mode (10 `student_test` trips)
```bash
cd /Users/ud/Desktop/Pathsnap
source .venv/bin/activate
export PATHSNAP_MODE=development
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Switching to Reserved Evaluation Mode (Final Demo)
```bash
cd /Users/ud/Desktop/Pathsnap
source .venv/bin/activate
export PATHSNAP_MODE=evaluation
export PATHSNAP_EVAL_DIR=/absolute/path/to/organiser_reserved_trips
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- If `PATHSNAP_EVAL_DIR` is unset, missing, or contains no trip folders, `GET /health` reports `ready: false` with the exact error and `GET /trips` / `POST /predict` return HTTP `503` rather than falling back to development trips.
- Reference `.route` files are **never** loaded during prediction and are **never** returned to Laptop 2. After predictions finish, Laptop 1's server-side scorer computes aggregate evaluation metrics locally and logs every request (`trip_id`, status, runtime, metrics) in Laptop 1's terminal.

### Verify Health Locally
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/trips
```

---

## 5. Laptop 2 UI launch

Give Laptop 2 the ui/ folder. It includes the standard-library Python launcher and bundled Leaflet. From the directory containing ui/, substitute Laptop 1's current Wi-Fi IPv4 address:

```bash
python ui/app.py --api http://192.168.1.23:8000
```

Open http://127.0.0.1:8501 on Laptop 2. The live client requests /health, /trips and /predict. Select two different development trips. Capture the UI result and receiving terminal log for the same trip. Update Section 7 of report/TECHNICAL_REPORT.md and export the final PDF after the physical test. See ui/README.md.
