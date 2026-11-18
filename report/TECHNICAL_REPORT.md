# PathSnap: Distributed Map-Matching & Route Reconstruction — Technical Report (DRAFT)

> **Draft Status Notice:** All Laptop 1 experiments, model training runs, Bayesian optimization trials, HMM/Viterbi evaluations, Folium maps, and FastAPI backend tests are **complete and verified locally**. This report remains in **DRAFT** status solely pending physical integration of **two team verification screenshots** in Section 7: (1) Laptop 2 working UI screenshot connected over LAN to Laptop 1, and (2) Laptop 1 live CMD/terminal screenshot showing incoming `/predict` logs from Laptop 2.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### Partitions & Integrity Audit
- **Partitions analysed:** `train` (**80 trips**, **208,417 GPS observations**) and `student_test` (**10 trips**, **19,160 GPS observations**). Zero trip overlap (`overlap_trip_ids = 0`). The 10 reserved evaluator trips were never accessed during development, EDA, feature selection, tuning, or HMM design.
- **File & missing-value audit (all 90 development trips):** Every trip contains all 4 headerless whitespace-separated files (`.track`, `.nodes`, `.arcs`, `.route`). Measured parse errors = **0**, invalid/NaN coordinates = **0**, out-of-bounds arc endpoint node IDs = **0**, out-of-bounds route arc IDs = **0**, negative/zero timestamp steps = **0 / 0**, isolated nodes = **0**.
- **Worked example (`train/00000005`):** Verified exact counts — **1,045 GPS readings**, **898 nodes**, **1,579 directed arcs**, and **105 reference-route entries**. Tracing real reference arc `arc_id=176` resolves `from_node_id=143` `(35.548172, 49.615574)` to `to_node_id=140` `(35.544044, 49.612933)` with projected metric length **418.661 m** and compass bearing **225.41°**.

### 1. How We Preprocessed the Data
1. **Coordinate Projection (Metric Accuracy Globally):** Exploratory analysis revealed trips spanning longitude `[-100.42°, 137.24°]` and latitude `[-34.64°, 61.70°]` across multiple continents. Ordinary Euclidean distance in degrees or a single fixed UTM zone would introduce severe metric distortion. We project every trip independently from WGS84 (`EPSG:4326`, raw `lon, lat` order) into a **trip-local Azimuthal Equidistant (`+proj=aeqd`) coordinate system** centered at the trip's median `(lon, lat)` in meters.
2. **Spatial Candidate Generation (Inference-Only):** For each trip we build a vectorized Shapely `STRtree` spatial index over all valid directed road arcs (up to 307,392 arcs in `00000013`) and query directed arcs within `60.0 m`, keeping up to `max_candidates=10` sorted by exact point-to-segment distance (with a `fallback_k=4` nearest-arc fallback when empty). Candidate generation achieves a **0.667%** miss rate on `train` (80 trips) and **0.146%** on `student_test` (10 trips).
3. **Development-Only Monotonic Reference Alignment (`src/alignment.py`):** Because `.route` is an ordered traversal of road segments (e.g., 105 arcs) rather than one label per GPS reading (e.g., 1,045 readings), we implemented a monotonic dynamic-programming aligner over route indices `j = 0..M-1` using exact projected route position `s(t, j) = cum_start[j] + proj[t, j] * len[j]`, heading disambiguation when moving, zero stay penalty for stops, and relaxed gap transition weights for `dt > 2s`. Observations > `50.0 m` from their aligned reference arc are marked unscored (`far_from_reference_route`). This yields **99.86%** label coverage on `student_test` (19,133/19,160 scored) and **88.84%** on `train` (185,149/208,417 scored). Crucially, EDA revealed that `train/00000054` (19,313 obs, median min distance 138.6 m to `.route`) and `train/00000033` (5,844 obs, median 122.4 m) account for ~20,700 of the 23,268 unscored training points—real-world evidence of the reference-route limitations highlighted in `Dataset Exploration.pdf` §6. No official alignment script was provided; we therefore report both alignment-based observation Top-1 accuracy and alignment-free sequence-level `.route` metrics (Sequence LCS F1 and Length-Weighted F1).
4. **Leakage Prevention:** Learned preprocessing (`StandardScaler` for Logistic Regression and KNN) is fit strictly inside each fold's training split via `sklearn.pipeline.Pipeline`. Route-free inference verification confirmed all 21 features compute cleanly with `trip.route is None`.

### 2. Observed Data Patterns & Feature Engineering Justification
- **Sampling irregularities & gaps:** **96.68%** of consecutive steps are exactly 1 s apart, yet **47/90 trips** contain gaps > 5 s, **42/90 trips** exceed 10 s, and `train/00000090` has a **153,064 s** (~42.5 h) stop gap.
- **Road network structure:** **89/90 trips** contain multiple weakly connected components, **60/90 trips** have parallel multi-edges with identical `(from_node, to_node)`, **9/90** contain self-loops, and **10/90** have repeated arcs in `.route` (loops/U-turns). Because most streets are represented by paired opposite directed arcs, the **nearest-road baseline achieves only 52.80% Top-1 accuracy on `student_test`** (54.99% on `train`)—it cannot distinguish forward vs reverse lanes.
- **21 Engineered Inference-Only Features** (strictly from `.track`, `.nodes`, `.arcs`):
  - *Spatial proximity & relative margin:* `dist_m`, `dist_rank`, `dist_diff_from_best`, `dist_ratio_to_best`, `n_candidates`.
  - *Directional alignment & explicit stationary handling:* `heading_diff_deg`, `cos_heading_diff`, `heading_rank`, `heading_valid`, `is_stationary`, `obs_speed_mps`. When `obs_speed_mps < 1.0 m/s` or local displacement `< 1.5 m`, we explicitly set `is_stationary=1`, `heading_valid=0`, `heading_diff_deg=90.0`, and `cos_heading_diff=0.0` so GPS stop jitter does not corrupt direction features.
  - *Local trajectory continuity & sampling context:* `prev_dist_m`, `next_dist_m`, `along_arc_progress_m` (signed projection of `(p_{t+1} - p_{t-1})` onto candidate arc unit vector), `dt_prev_s`, `dt_next_s`.
  - *Road geometry & graph topology:* `proj_frac`, `arc_len_m`, `has_reverse_arc`, `from_node_out_deg`, `to_node_out_deg`.

### EDA Visualisations (Generated from Actual Data)
- Dataset Overview (`train` + `student_test`): `../outputs/milestone1/dataset_eda_overview.png`
- GPS Sampling-Gap Distribution & Top Gap Trips: `../outputs/milestone1/sampling_gaps.png`
- Trip `00000005` Static Trajectory & Road Network Plot: `../outputs/milestone1/trip_00000005_trajectory.png`
- Trip `00000005` Interactive Folium Map: `../outputs/milestone1/trip_00000005_map.html`
- Monotonic Alignment & Baseline Diagnostics: `../outputs/milestone2/alignment_and_baseline_diagnostics.png`

---

## 3. Model Experimentation and Comparison

All five models from five distinct algorithm families were trained locally on Laptop 1 (Apple M2, 8-core, 8 GB RAM) using identical candidates, identical 5-fold `GroupKFold` splits grouped by `trip_id` over the 80 training trips (1,290,243 scored training candidates), and identical evaluation on the 10 `student_test` trips (19,133 scored observations; 0.146% candidate miss rate included in every Top-1 denominator).

### 3.1 Overall Model Comparison

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (10 trips) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear & Probabilistic | StandardScaler + L2 C=1.0, lbfgs, max_iter=500 | 0.8846 ± 0.0169 | 0.8437 | 0.9300 | 0.8943 | 0.1258 | 0.006s (0.6 ms/trip) |
| Decision Tree | Tree-based | CART Gini, max_depth=12, min_samples_leaf=50 | 0.8983 ± 0.0190 | 0.8449 | 0.9259 | 0.9083 | 0.1061 | 0.008s (0.8 ms/trip) |
| Random Forest | Ensemble (Bagging) | 150 trees, max_depth=16, min_samples_leaf=20, max_features=sqrt | 0.9032 ± 0.0184 | 0.8528 | 0.9295 | 0.9127 | 0.0969 | 0.471s (47.1 ms/trip) |
| Histogram Gradient Boosting | Boosting | max_iter=200, lr=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2=1.0 | 0.9035 ± 0.0185 | 0.8548 | 0.9362 | 0.9134 | 0.0962 | 0.337s (33.7 ms/trip) |
| K-Nearest Neighbours | Instance-based | StandardScaler + k=25, distance-weighted Euclidean (brute BLAS, batched) | 0.8927 ± 0.0179 | 0.8446 | 0.9279 | 0.9048 | 0.1503 | 111.179s (11117.9 ms/trip) |

> **Nearest-Road Baseline Reference (Student-Test):** Top-1 Accuracy = **0.5280**, Route Continuity = **0.8015**, Sequence LCS F1 = **0.4583**. The five classifiers score approximately **31.6 to 32.7 percentage points** above this baseline.

### 3.2 Compulsory Analysis for EVERY Model

#### 3.2.1 Logistic Regression (`Linear & Probabilistic`)

**A. Model Understanding**
1. *Suitability for PathSnap:* Logistic Regression serves as our interpretable linear probabilistic baseline. Because candidate arc matching strongly depends on monotonic signals—point-to-segment distance, distance rank, cosine heading alignment, and along-arc trajectory progress—a linear log-odds model tests how much of map-matching can be solved by additive linear boundaries and naturally outputs well-calibrated probabilities for HMM emissions.
2. *Major strengths and limitations:* **Strengths:** Ultra-fast training (1.44s on 1,290,243 candidates) and sub-millisecond inference (0.63 ms/trip), low variance across folds (CV Macro F1 std = 0.0169), and competitive log-loss (0.1258). **Limitations:** Cannot capture nonlinear feature interactions (e.g., discounting noisy heading when `is_stationary=1` or combining `proj_frac` near 0/1 with `prev_dist_m`/`next_dist_m` at intersections), leading to lower student-test Macro F1 (0.8943) and Top-1 (0.8437) than nonlinear ensembles.

**B. Training and Configuration**
5. *Important hyperparameters & initial values:* `{"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": 42}` (random seed: `42`).
6. *Loss function / optimisation objective:* L2-regularized binary cross-entropy (log-loss).

**C. Experimental Results**
7. *Student-testing performance (10 trips):* Accuracy = **0.9508**, Positive Precision = **0.8242**, Positive Recall = **0.8099**, Route Continuity = **0.9300**, Sequence LCS F1 = **0.6934**, Training Time = **1.44s**, Inference Time = **0.0063s** (0.63 ms/trip).
8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **0.8437** (given covered: 0.8450; miss rate: 0.0015), Candidate Macro F1 = **0.8943**, Binary Log Loss = **0.1258**.
9. *Overfitting / underfitting diagnosis:* Mild underfitting rather than overfitting: Train Macro F1 (0.8871) is very close to 5-fold CV Macro F1 (0.8846, gap=+0.0025) and Student-Test Macro F1 (0.8943), confirming high bias / low variance.

#### 3.2.2 K-Nearest Neighbours (`Instance-based`)

**A. Model Understanding**
1. *Suitability for PathSnap:* K-Nearest Neighbours (distance-weighted Euclidean) tests whether local geometric similarity in standardized 21-dimensional feature space can classify road candidates without assuming a global parametric boundary. Distance weighting converts neighborhood class ratios into continuous scores usable by the decoder.
2. *Major strengths and limitations:* **Strengths:** Zero parametric assumptions, instant training/indexing (0.29s), and nonlinear local decision surfaces achieving 0.9048 Macro F1 and 0.8446 Top-1 on student-test. **Limitations:** Severe inference bottleneck at scale (111.18s across 10 test trips = 11117.9 ms/trip, orders of magnitude slower than tree/linear models), higher memory footprint (must retain all 1,290,243 training vectors), and worse probability calibration (Student-Test Log Loss = 0.1503) due to discrete 25-neighbor vote quantization.

**B. Training and Configuration**
5. *Important hyperparameters & initial values:* `{"n_neighbors": 25, "weights": "distance", "metric": "euclidean", "algorithm": "brute", "n_jobs": 6}` (random seed: `None`).
6. *Loss function / optimisation objective:* Non-parametric distance-weighted neighborhood voting (no parametric loss).

**C. Experimental Results**
7. *Student-testing performance (10 trips):* Accuracy = **0.9564**, Positive Precision = **0.8593**, Positive Recall = **0.8114**, Route Continuity = **0.9279**, Sequence LCS F1 = **0.7045**, Training Time = **0.29s**, Inference Time = **111.1792s** (11117.92 ms/trip).
8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **0.8446** (given covered: 0.8458; miss rate: 0.0015), Candidate Macro F1 = **0.9048**, Binary Log Loss = **0.1503**.
9. *Overfitting / underfitting diagnosis:* Notable memorization / overfitting on training points: distance-weighted KNN achieves near-1.0 Train Macro F1 (1.0000) vs CV Macro F1 (0.8927, gap=+0.1073) and elevated CV Log Loss (0.1795) when neighbors disagree or vote unanimously wrong.

#### 3.2.3 Decision Tree (`Tree-based`)

**A. Model Understanding**
1. *Suitability for PathSnap:* A regularized CART Decision Tree (`max_depth=12, min_samples_leaf=50`) directly models conditional nonlinear rules (e.g., splitting first on `dist_rank` and `cos_heading_diff`, and branching differently when `is_stationary=1`) while requiring no feature scaling and yielding fast, human-inspectable decision paths.
2. *Major strengths and limitations:* **Strengths:** Very fast inference (0.81 ms/trip), natural handling of feature interactions and mixed scales, improving over Logistic Regression on Student-Test Macro F1 (0.9083). **Limitations:** Axis-aligned greedy splits produce piecewise-constant leaf probabilities with higher variance and worse calibration (CV Log Loss = 0.1288, Student-Test Log Loss = 0.1061) than ensembles, which hurts downstream HMM emission reliability.

**B. Training and Configuration**
5. *Important hyperparameters & initial values:* `{"criterion": "gini", "max_depth": 12, "min_samples_leaf": 50, "random_state": 42}` (random seed: `42`).
6. *Loss function / optimisation objective:* Gini impurity minimization at greedy axis-aligned splits.

**C. Experimental Results**
7. *Student-testing performance (10 trips):* Accuracy = **0.9580**, Positive Precision = **0.8670**, Positive Recall = **0.8159**, Route Continuity = **0.9259**, Sequence LCS F1 = **0.6914**, Training Time = **14.47s**, Inference Time = **0.0081s** (0.81 ms/trip).
8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **0.8449** (given covered: 0.8462; miss rate: 0.0015), Candidate Macro F1 = **0.9083**, Binary Log Loss = **0.1061**.
9. *Overfitting / underfitting diagnosis:* Moderate overfitting despite depth/leaf regularization: Train Macro F1 (0.9107) exceeds CV Macro F1 (0.8983, gap=+0.0124), and sparse leaves occasionally emit 0/1 probabilities that inflate Log Loss (0.1288).

#### 3.2.4 Random Forest (`Ensemble (Bagging)`)

**A. Model Understanding**
1. *Suitability for PathSnap:* Random Forest averages 150 bootstrap-sampled trees (`max_depth=16, min_samples_leaf=20, max_features='sqrt'`) to reduce single-tree variance and smooth empirical leaf probability estimates for candidate scoring.
2. *Major strengths and limitations:* **Strengths:** Strong generalization and stability across folds (CV Macro F1 = 0.9032 ± 0.0184, Student-Test Macro F1 = 0.9127, Top-1 = 0.8528), substantially cleaner probabilities than a single tree (Student-Test Log Loss = 0.0969). **Limitations:** Slower training (154.08s) and larger memory/disk footprint across 150 deep trees than Histogram Gradient Boosting; because trees are built independently (bagging) rather than optimizing log-loss sequentially, probability calibration trails boosting slightly.

**B. Training and Configuration**
5. *Important hyperparameters & initial values:* `{"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 20, "max_features": "sqrt", "n_jobs": 4, "random_state": 42}` (random seed: `42`).
6. *Loss function / optimisation objective:* Bagged Gini impurity reduction across bootstrap samples.

**C. Experimental Results**
7. *Student-testing performance (10 trips):* Accuracy = **0.9602**, Positive Precision = **0.8797**, Positive Recall = **0.8190**, Route Continuity = **0.9295**, Sequence LCS F1 = **0.6993**, Training Time = **154.08s**, Inference Time = **0.4711s** (47.11 ms/trip).
8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **0.8528** (given covered: 0.8541; miss rate: 0.0015), Candidate Macro F1 = **0.9127**, Binary Log Loss = **0.0969**.
9. *Overfitting / underfitting diagnosis:* Well-regularized with small generalization gap: Train Macro F1 (0.9184) vs CV Macro F1 (0.9032, gap=+0.0152) and Student-Test Macro F1 (0.9127).

#### 3.2.5 Histogram Gradient Boosting (`Boosting`)

**A. Model Understanding**
1. *Suitability for PathSnap:* Histogram Gradient Boosting (`HistGradientBoostingClassifier`) directly optimizes binary log-loss via stagewise additive trees over 255-bin histograms. This aligns directly with PathSnap's need for both accurate nonlinear candidate ranking and sharply calibrated posterior probabilities `P(y=1|x)` for HMM/Viterbi emissions.
2. *Major strengths and limitations:* **Strengths:** Strongest overall probability quality (CV Log Loss = 0.1126, Student-Test Log Loss = 0.0962), top/near-top ranking accuracy (Student-Test Top-1 = 0.8548, Macro F1 = 0.9134, Route Continuity = 0.9362), fast local training (22.29s), compact artifact size, and low latency (33.74 ms/trip). **Limitations:** Sensitive to hyperparameters (`learning_rate`, `max_leaf_nodes`, `l2_regularization`) and less directly interpretable than Logistic Regression or a single Decision Tree, motivating Bayesian optimization in Section 4.

**B. Training and Configuration**
5. *Important hyperparameters & initial values:* `{"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 63, "min_samples_leaf": 50, "l2_regularization": 1.0, "early_stopping": true, "validation_fraction": 0.1, "n_iter_no_change": 15, "scoring": "neg_log_loss", "random_state": 42}` (random seed: `42`).
6. *Loss function / optimisation objective:* Binary log-loss (cross-entropy) via stagewise gradient boosting.

**C. Experimental Results**
7. *Student-testing performance (10 trips):* Accuracy = **0.9602**, Positive Precision = **0.8718**, Positive Recall = **0.8287**, Route Continuity = **0.9362**, Sequence LCS F1 = **0.7116**, Training Time = **22.29s**, Inference Time = **0.3374s** (33.74 ms/trip).
8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **0.8548** (given covered: 0.8560; miss rate: 0.0015), Candidate Macro F1 = **0.9134**, Binary Log Loss = **0.0962**.
9. *Overfitting / underfitting diagnosis:* Well-generalized with built-in early stopping on validation log-loss: Train Macro F1 (0.9280) vs CV Macro F1 (0.9035, gap=+0.0246) and Student-Test Macro F1 (0.9134); the boosting-iteration curve (`outputs/milestone3/hgb_iteration_loss.png`) shows training and validation log-loss tracking closely without late-stage divergence.

### Model Selection Justification
- **Selected Final Model:** **Histogram Gradient Boosting** (`hist_gradient_boosting`).
- **Primary Selection Criterion:** 5-fold Group-CV Candidate Macro F1-score (jointly verified with CV Obs-level Top-1, Log Loss for HMM emissions, fold stability, and inference latency).
- **Measured Evidence:** Across the 5-fold Group-CV benchmark and 10-trip student-test set, the selected model delivers the best joint balance of generalization Macro F1, observation-level Top-1 accuracy, calibrated log-loss (crucial because HMM emissions directly consume `log P(y=1|x)`), tight cross-fold variance, and fast per-trip inference latency suitable for live FastAPI deployment.

### 3.3 Model Visualisations (Generated from Actual Experimental Runs)
- 5-Fold Group-CV Macro F1 Comparison: `../outputs/milestone3/cv_macro_f1_comparison.png`
- Student-Test Top-1 Accuracy, Macro F1 & Route Continuity Comparison: `../outputs/milestone3/student_test_comparison.png`
- Log Loss & Inference Latency Comparison: `../outputs/milestone3/log_loss_and_latency_comparison.png`
- Candidate-Level Confusion Matrices (All 5 Models): `../outputs/milestone3/confusion_matrices.png`
- Boosting-Iteration Log-Loss Curve (Histogram Gradient Boosting ONLY; no artificial epoch curves fabricated for non-iterative models): `../outputs/milestone3/hgb_iteration_loss.png`

---

## 4. Bayesian Hyperparameter Optimisation (for Selected Model)

1. **Hyperparameters searched:** `learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_bins, max_iter`.
2. **Ranges / distributions considered:**
   - `learning_rate`: `{"type": "log_uniform", "low": 0.02, "high": 0.25, "rationale": "Controls shrinkage per boosting stage vs generalization."}`
   - `max_leaf_nodes`: `{"type": "int", "low": 15, "high": 127, "rationale": "Controls interaction depth and tree complexity."}`
   - `min_samples_leaf`: `{"type": "int", "low": 20, "high": 200, "rationale": "Regularizes leaf support to prevent overfitting to specific trips."}`
   - `l2_regularization`: `{"type": "log_uniform", "low": 0.001, "high": 20.0, "rationale": "Shrinks leaf weights and improves probability calibration (log-loss)."}`
   - `max_bins`: `{"type": "categorical", "choices": [128, 255], "rationale": "Histogram discretization resolution for distance/heading thresholds."}`
   - `max_iter`: `{"type": "int", "low": 150, "high": 400, "step": 50, "rationale": "Upper bound on boosting rounds (with early stopping enabled)."}`
3. **Objective function optimised:** Maximize 5-fold Group-CV Candidate Macro F1-score on the 80 training trips using `Optuna TPESampler (Bayesian Tree-structured Parzen Estimator, seed=42)` on the exact same 5-fold `GroupKFold` trip splits.
4. **Number of optimisation trials:** **12** trials (total runtime: **1487.2s**; final refit on 80 trips: **34.88s**).
5. **Best hyperparameter values (Trial #7):** `{"learning_rate": 0.021848629312930523, "max_leaf_nodes": 95, "min_samples_leaf": 175, "l2_regularization": 11.12454811271779, "max_bins": 255, "max_iter": 300}`.
6. **Did the tuned model outperform the untuned model?** Yes on 5-fold Group-CV: CV Macro F1 moved from **0.90348** to **0.90441**, CV Log Loss moved from **0.11264** to **0.11197**, and on the 10 `student_test` trips Top-1 moved from **0.85475** to **0.85606** (Student-Test Log Loss: **0.09617** -> **0.09594**).

### Untuned vs Bayesian-Tuned Comparison (Same Splits & Metrics)

| Variant | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Student-Test Log Loss | Inference Time (10 trips) |
|---|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting (Untuned) | 0.90348 ± 0.01849 | 0.85475 | 0.91339 | 0.09617 | 0.3374s |
| Histogram Gradient Boosting (Bayesian Tuned) | 0.90441 ± 0.01801 | 0.85606 | 0.91485 | 0.09594 | 0.5079s |

- **Artifact Export & Subprocess Reload Check:** Exported self-contained bundle to `artifacts/model_bundle.joblib` (1,483,559 bytes), `artifacts/model_config.json`, and `artifacts/feature_spec.json`. Fresh Python subprocess reload verified exact predictions on `student_test/00000000` with `route_loaded=False` and `max_abs_diff=0.00e+00`.
- Optimisation History Plot: `../outputs/milestone4/tuning_history.png`

---

## 5. HMM and Viterbi Route Reconstruction

1. **Hidden states:** For each GPS observation `t`, the hidden states are its spatially retrieved candidate directed arcs `C_t = {a_{t,1}, ..., a_{t,K_t}}` (`K_t <= 10`), preserving exact `(obs_id, arc_id)` mapping throughout decoding.
2. **Emission log-probabilities:** Derived directly from the tuned classifier's calibrated probability: `log E(t, a) = emit_weight * log(clip(P(y=1 | x_{t,a}), 1e-6, 1-1e-6))` with `emit_weight = 1.0` (selected strictly on out-of-fold predictions of 15 training trips).
3. **Transition log-probabilities:** Directed road-network transitions compare exact projected route distance `Δd_route` against observed GPS step `Δd_gps` via an exponential mismatch penalty `-|Δd_route - Δd_gps| / beta` (`beta=15.0 m`, relaxed to `60.0 m` during `dt > 2s` gaps), rewarding same-arc forward/stationary motion (`+0.25`), penalizing significant backward sliding along a directed arc, discouraging spurious immediate U-turns onto the reverse arc `(u,v)->(v,u)` (`-4.0`), and computing multi-hop directed shortest paths via bounded cached Dijkstra.
4. **Missing candidates & large GPS gaps:** Our candidate generator uses a `fallback_k=4` nearest-arc rescue so active observations rarely lose all candidates; if any observation still has `K_t = 0`, Viterbi records `pred_arc_id = -1` and bridges the transition between adjacent active steps using their true elapsed `Δt` and `Δd_gps`. Across extreme gaps (`dt > 60s` or `Δd_gps > 1000m`), Viterbi performs a clean memory reset to pure emissions so overnight/tunnel gaps cannot corrupt the downstream path.
5. **Disconnected road segments:** When no directed path exists within the bounded search radius (handling the 89/90 multi-component networks found in EDA), transitions receive a finite `disconnected_penalty = -12.0` (softened to `-4.0` across large gaps), preventing `-inf`/NaN collapse while strongly preferring topologically valid continuations.
6. **Did HMM/Viterbi improve the final route?** **Yes — substantially across every metric on the 10 `student_test` trips:** Pointwise Top-1 Accuracy rose from **0.8561** to **0.9194** (`+0.0634`), Route Continuity rose from **0.9348** to **0.9761** (`+0.0413`), and order-aware Sequence LCS F1 jumped from **0.7095** to **0.9100** (`+0.2005`), fixing **1482** observations while hurting **270** (net gain **+1212** observations) at an average decode latency of only **191.8 ms/trip**.

### Classifier-Only vs Classifier + HMM/Viterbi Comparison (10 Student-Test Trips)

| Pipeline | Pointwise Top-1 Accuracy | Route Continuity | Sequence LCS F1 | Sequence Length-Weighted F1 | Mean Decode Latency |
|---|---:|---:|---:|---:|---:|
| Tuned Classifier Only | 0.8561 | 0.9348 | 0.7095 | 0.9123 | 0.0 ms/trip |
| Tuned Classifier + HMM/Viterbi | **0.9194** | **0.9761** | **0.9100** | **0.9761** | 191.8 ms/trip |

### Worked Decoding Case Analysis (Improvements & Failure Mode)
- **Where HMM fixed a classifier error (`student_test/00000077`, `obs_id=2018`, `t=2019s`):** The independent classifier picked `arc_id=602` (top local score), which was topologically connected from the preceding arc `4066`. Viterbi selected `arc_id=4066` (matching the reference arc `4066`), preserving directed continuity from `4066` -> `4066` -> `4066`.
- **Where HMM did not help (`student_test/00000077`, `obs_id=202`, `t=203s`):** The classifier correctly predicted `arc_id=8343` (matching reference `8343`), whereas HMM stayed/shifted onto `arc_id=8339` due to transition smoothing across a tight junction boundary (`prev=8343`, `next=8339`). This illustrates the classic HMM trade-off: smoothing removes far more isolated noise spikes than the few junction-boundary frames it shifts by 1 second.
- Classifier-Only vs HMM/Viterbi Comparison Plot: `../outputs/milestone5/hmm_vs_classifier_comparison.png`

---

## 6. Geographical Visualisation Using Folium

Two interactive OpenStreetMap Folium visualisations with toggleable `LayerControl`, colour-coded layers, tooltips, and legends were generated directly from our code:
1. **EDA Map (`train/00000005`):** `../outputs/milestone1/trip_00000005_map.html` (and static export `../outputs/milestone1/trip_00000005_trajectory.png`) showing raw GPS observations, full 1,579-arc road network, supplied reference route, and traced example arc `176`.
2. **Route Reconstruction Map (`student_test/00000077`):** `../outputs/milestone5/student_test_00000077_hmm_map.html` (and static export `../outputs/milestone5/student_test_00000077_hmm_route.png`) overlaying raw GPS readings, relevant road network, reference route, toggleable classifier-only arcs, HMM/Viterbi predicted route, and a marker pinpointing the worked HMM correction case.

---

## 7. API, UI and Deployment (Two-Laptop Architecture)

### Laptop 1 Backend Verification (COMPLETE)
- **Framework & Launch:** FastAPI (`backend/main.py`) served via `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
- **LAN URL (Laptop 1):** `http://10.1.25.138:8000` (local verification passed; **reachability from Laptop 2 is pending physical LAN test**).
- **Live Development Verification (All 10 `student_test` trips via HTTP `POST /predict`):** Mean Pointwise Top-1 = **0.9275**, Mean Route Continuity = **0.9761**, Mean Sequence LCS F1 = **0.9100**, Mean End-to-End API Latency = **338.1 ms/trip**.
- **Artifact Persistence & Isolation Verified:** Restarted FastAPI cleanly and served identical predictions from `artifacts/model_bundle.joblib` without retraining. Verified `PATHSNAP_MODE=evaluation` with a missing `PATHSNAP_EVAL_DIR` returns explicit HTTP `503` error and `ready: false` on `/health`. Confirmed reference `.route` files are never read during prediction and never exposed in HTTP responses.
- **Laptop 2 Handoff Package:** `laptop2_handoff/` (`API_CONTRACT.md`, `example_health_response.json`, `example_trips_response.json`, `example_predict_response.json` — strictly free of raw datasets, reference labels, and model weights).

### Pending Physical Team Evidence (Required Before Final PDF Export)
- **[PENDING — Laptop 2 UI Screenshot]:** Capture screenshot of Laptop 2's working evaluator UI displaying the trip list from `GET /trips`, interactive predicted route map, and live server-side metrics returned from Laptop 1.
- **[PENDING — Laptop 1 Receiving Terminal Screenshot]:** Capture screenshot of Laptop 1's `uvicorn` terminal showing live `PREDICT 200 | mode=... | trip_id=... | elapsed=... ms` logs triggered by Laptop 2 over LAN.
- **Laptop 2 UI:** The local Python client and bundled Leaflet interface are in `ui/`. The physical two-device demo is pending.

---

## 8. Academic Integrity & AI Assistance Disclosure

- All data inspection, feature engineering, five-model training runs, 5-fold group-aware cross-validation, Bayesian Optuna trials, HMM/Viterbi decoding, Folium maps, and FastAPI endpoint tests were executed genuinely and locally on Laptop 1 using the provided PathSnap student dataset.
- Every reported table, plot, and metric in this document is programmatically generated from saved JSON/CSV experiment artifacts in `outputs/` and is fully reproducible via the commands in `README.md`.
- AI coding assistance was used to accelerate implementation and structuring of Python pipelines and documentation; all experiments were executed locally and verified against actual outputs, and the team should review these explanations against its own run records before submission.
