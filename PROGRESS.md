# PathSnap — Laptop 1 Progress & Evaluation Evidence

Workspace root: `/Users/ud/Desktop/Pathsnap`  
Python environment: `.venv` (Python 3.12.6, Apple M2 8-core, 8 GB RAM, pinned in `requirements.txt`)

## Milestone 1 — Data Understanding & First Verified Map (COMPLETE)
- **Dataset discovered**: `Guidelines/Student folder/train` (80 trips, 208,417 GPS observations) and `Guidelines/Student folder/student_test` (10 trips, 19,160 GPS observations); zero trip overlap; zero reserved evaluator trips touched.
- **Data quality verified**: 0 malformed rows, 0 invalid coordinates, 0 out-of-bounds node/arc/route IDs, 0 non-monotonic timestamps; 96.68% exact 1 s sampling steps; max GPS gap 153,064 s (trip `00000090`); 89/90 trips have multiple weakly connected road components; 60/90 trips have parallel duplicate `(from_node, to_node)` arcs; 10/90 trips have repeated arcs in `.route`; mean `.route` directed continuity 99.94%.
- **Worked example (`00000005`)**: Verified exact counts — **1,045** GPS readings, **898** nodes, **1,579** directed arcs, **105** reference-route entries. Traced reference arc `176` (`node 143 -> node 140`, `[35.548172, 49.615574] -> [35.544044, 49.612933]`, length `418.661 m`, bearing `225.41°`).
- **Evidence**:
  - `outputs/milestone1/dataset_inventory.csv`
  - `outputs/milestone1/eda_summary.json`
  - `outputs/milestone1/worked_example_00000005.json`
  - `outputs/milestone1/sampling_gaps.png`
  - `outputs/milestone1/dataset_eda_overview.png`
  - `outputs/milestone1/trip_00000005_trajectory.png`
  - `outputs/milestone1/trip_00000005_map.html` (OpenStreetMap interactive Folium map with separate toggleable layers)
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.eda
  ```

## Milestone 2 — Candidate Generation, Alignment Labels & Features (COMPLETE)
- **Route-free verification**: Verified `generate_trip_candidates(trip)` succeeds with `trip.route is None` (`00000005`: 3,363 candidates, 21 inference-only features, all finite, zero `.route` access).
- **Candidate generation**: Shapely `STRtree` spatial index in trip-local Azimuthal Equidistant meters (`search_radius_m=60.0`, `max_candidates=10`, `fallback_k=4`).
  - Train (80 trips): 1,489,628 candidate rows (avg 7.15/obs), **candidate miss rate = 0.667%**.
  - Student Test (10 trips): 140,853 candidate rows (avg 7.35/obs), **candidate miss rate = 0.146%**.
- **Monotonic DP Reference Alignment** (`src/alignment.py`, development-only): Aligns `.track` to `.route` indices chronologically using exact projected route position `s(t,j) = cum_start[j] + proj[t,j] * len[j]`, heading disambiguation, zero stop penalty, relaxed gap transitions, and `50.0 m` outlier cutoff.
  - Label coverage: **99.859%** on student-test (19,133/19,160 scored; 27 unscored >50 m) and **88.836%** on train (185,149/208,417 scored; trips `00000054` and `00000033` alone account for ~20,700 GPS points >50 m from their reference routes — documenting the real-world reference-route limitations warned in `Dataset Exploration.pdf` §6).
  - Junction near-ties flagged transparently: 7.79% (train), 9.19% (student-test).
- **Nearest-Road Baseline** (candidate misses included in denominator):
  - Train: Top-1 = **54.99%** (55.36% given covered), Route Continuity = **77.87%**, Sequence LCS F1 = **44.07%**.
  - Student Test: Top-1 = **52.80%** (52.88% given covered), Route Continuity = **80.16%**, Sequence LCS F1 = **45.83%**.
- **Evidence**:
  - `outputs/milestone2/milestone2_summary.json`
  - `outputs/milestone2/per_trip_alignment_and_baseline.csv`
  - `outputs/milestone2/worked_candidate_example_00000005.csv` & `.json`
  - `outputs/milestone2/alignment_and_baseline_diagnostics.png`
  - `outputs/milestone2/cache/` (`train_candidates.parquet`, `student_test_candidates.parquet`, obs meta)
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.build_dataset
  ```

## Milestone 3 — Train & Evaluate Five Models Locally (COMPLETE)
- **Full local training executed** on Apple M2 across all 80 training trips (1,290,243 scored candidates, 5-fold `GroupKFold` by `trip_id`) and evaluated on all 10 `student_test` trips (19,133 scored obs; 0.146% candidate miss rate included in Top-1 denominator).
- **Measured Results Summary**:
  - **Histogram Gradient Boosting (SELECTED)**: CV Macro F1 **0.9035 ± 0.0185**, Student-Test Macro F1 **0.9134**, Top-1 **0.8548**, Log Loss **0.0962**, Route Continuity **0.9362**, Seq LCS F1 **0.7116**, fit **22.29s**, inference **33.7 ms/trip**.
  - **Random Forest**: CV Macro F1 **0.9032 ± 0.0184**, Student-Test Macro F1 **0.9127**, Top-1 **0.8528**, Log Loss **0.0969**, fit **154.08s**, inference **47.1 ms/trip**.
  - **Decision Tree**: CV Macro F1 **0.8983 ± 0.0190**, Student-Test Macro F1 **0.9083**, Top-1 **0.8449**, Log Loss **0.1061**, fit **14.47s**, inference **0.8 ms/trip**.
  - **K-Nearest Neighbours** (exact full dataset, batched): CV Macro F1 **0.8927 ± 0.0179**, Student-Test Macro F1 **0.9048**, Top-1 **0.8446**, Log Loss **0.1503**, inference **11,117.9 ms/trip**.
  - **Logistic Regression**: CV Macro F1 **0.8846 ± 0.0169**, Student-Test Macro F1 **0.8943**, Top-1 **0.8437**, Log Loss **0.1258**, fit **1.44s**, inference **0.6 ms/trip**.
- **Evidence**:
  - `outputs/milestone3/model_comparison.csv` & `model_comparison.md`
  - `outputs/milestone3/model_selection_summary.json` & `cv_fold_assignments.csv`
  - `outputs/milestone3/model_<name>.json` (all 5 models)
  - `outputs/milestone3/cv_macro_f1_comparison.png`, `student_test_comparison.png`, `log_loss_and_latency_comparison.png`, `confusion_matrices.png`, `hgb_iteration_loss.png`
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.models --resume
  ```

## Milestone 4 — Bayesian Hyperparameter Optimisation & Artifact Export (COMPLETE)
- **Optuna TPE Bayesian optimization**: 12 trials (4 startup + 8 Bayesian surrogate-guided) over 6 hyperparameters (`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_bins`, `max_iter`) on the exact same 5-fold `GroupKFold` splits (80 training trips).
- **Best configuration (Trial #7)**: `learning_rate=0.02185, max_leaf_nodes=95, min_samples_leaf=175, l2_regularization=11.1245, max_bins=255, max_iter=300`.
- **Untuned vs Bayesian-Tuned comparison (same 10 `student_test` trips)**:
  - CV Macro F1: `0.90348 ± 0.01849` -> **`0.90441 ± 0.01801`**
  - CV Log Loss: `0.11264` -> **`0.11197`**
  - Student-Test Top-1 Accuracy: `0.85475` -> **`0.85606`**
  - Student-Test Macro F1: `0.91339` -> **`0.91485`**
  - Student-Test Log Loss: `0.09617` -> **`0.09594`**
  - Inference time (10 trips): `0.337s` -> `0.508s` (`50.8 ms/trip`)
- **Exported self-contained artifacts** (`artifacts/model_bundle.joblib`, `artifacts/model_config.json`, `artifacts/feature_spec.json`) and verified **fresh Python subprocess reload** (`route_loaded=False`, `max_abs_diff=0.0`).
- **Evidence**:
  - `outputs/milestone4/optuna_trials.csv`, `search_space.json`, `tuning_summary.json`
  - `outputs/milestone4/tuned_vs_untuned_comparison.csv`, `tuning_history.png`
  - `artifacts/model_bundle.joblib`, `artifacts/model_config.json`, `artifacts/feature_spec.json`
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.tune --trials 12
  ```

## Milestone 5 — HMM/Viterbi Route Reconstruction (COMPLETE)
- **Decoder validation**: Selected HMM parameters (`beta_m=15.0, u_turn_penalty=4.0, emit_weight=1.0, disconnected_penalty=12.0`, saved in `artifacts/hmm_config.json`) strictly using out-of-fold predictions on 15 training trips.
- **Classifier-Only vs Classifier + HMM/Viterbi on all 10 `student_test` trips** (every trip improved!):
  - Pointwise Top-1 Accuracy: `0.8561` -> **`0.9194`** (**+6.34 pp**)
  - Route Continuity: `0.9348` -> **`0.9761`** (**+4.13 pp**)
  - Order-Aware Sequence LCS F1: `0.7095` -> **`0.9100`** (**+20.05 pp**)
  - Sequence Length-Weighted F1: `0.9123` -> **`0.9761`** (**+6.38 pp**)
  - Net gain: **1,482** observations fixed vs 270 hurt (**+1,212 net**); mean decode latency **191.8 ms/trip**.
- **Evidence**:
  - `outputs/milestone5/milestone5_summary.json` (includes concrete fixed & hurt worked examples)
  - `outputs/milestone5/student_test_hmm_per_trip.csv`
  - `outputs/milestone5/hmm_vs_classifier_comparison.png`
  - `outputs/milestone5/student_test_00000077_hmm_map.html` & `student_test_00000077_hmm_route.png`
  - `artifacts/hmm_config.json`
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.hmm_viterbi
  ```

## Milestone 6 — FastAPI Backend & Laptop 2 Handoff (COMPLETE — Laptop 2 LAN Test Pending)
- **Live FastAPI backend verified** across all 10 `student_test` trips (`GET /health`, `GET /trips`, `POST /predict`, 404 on unknown trip), restarted cleanly from saved `artifacts/model_bundle.joblib`, and verified explicit HTTP 503 + `ready: false` when `PATHSNAP_MODE=evaluation` has a missing `PATHSNAP_EVAL_DIR`.
- **Live API summary (10 `student_test` trips, `development` mode)**:
  - Per-trip mean Top-1 Accuracy: **0.9275** (pooled obs Top-1: **0.9194**), Route Continuity: **0.9761**, Sequence LCS F1: **0.9100**, Mean end-to-end API latency: **338.1 ms/trip**.
  - Active LAN URL: `http://10.1.25.138:8000` (verified locally; **reachability from Laptop 2 pending physical LAN test**).
  - Reference `.route` files never loaded during prediction and never returned over HTTP.
- **Evidence**:
  - `backend/main.py`, `src/inference.py`
  - `outputs/milestone6/api_student_test_results.csv`, `milestone6_api_verification.json`, `uvicorn_run1.log`, `uvicorn_run2_restart.log`, `uvicorn_eval_missing_dir.log`
  - `laptop2_handoff/` (`API_CONTRACT.md`, `example_health_response.json`, `example_trips_response.json`, `example_predict_response.json`)
- **Reproduce / Launch**:
  ```bash
  source .venv/bin/activate
  export PATHSNAP_MODE=development
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
  # Automated verification:
  .venv/bin/python -m src.test_api
  ```

## Milestone 7 — Consolidated Draft Technical Report & README (COMPLETE — Screenshots Pending)
- Generated `report/TECHNICAL_REPORT.md` directly from all measured JSON/CSV outputs following every required section and compulsory question in `Deliverables.pdf` (marked DRAFT pending Laptop 2 UI source/screenshot and Laptop 1 receiving terminal screenshot).
- Updated `README.md` with full setup, reproduction, backend launch, UI launch, and evaluation-mode switching instructions.
- **Evidence**:
  - `report/TECHNICAL_REPORT.md`
  - `README.md`
- **Reproduce**:
  ```bash
  .venv/bin/python -m src.build_report
  ```
