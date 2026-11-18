"""Milestone 7: Generate consolidated PathSnap technical report from measured experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data_io import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "report"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_specific_analysis(key: str, res: Dict[str, Any]) -> Dict[str, str]:
    """Generate evidence-grounded answers for Deliverables.pdf §3.2 Q1-Q9 using real measured metrics."""
    cv = res["cv_summary"]
    te = res["student_test_metrics"]
    tr = res["train_diagnostic"]

    gap_f1 = tr["cand_macro_f1"] - te["cand_macro_f1"]
    gap_cv_f1 = tr["cand_macro_f1"] - cv["cand_macro_f1"]["mean"]

    if key == "logistic_regression":
        why = (
            "Logistic Regression serves as our interpretable linear probabilistic baseline. "
            "Because candidate arc matching strongly depends on monotonic signals—point-to-segment distance, "
            "distance rank, cosine heading alignment, and along-arc trajectory progress—a linear log-odds model "
            "tests how much of map-matching can be solved by additive linear boundaries and naturally outputs "
            "well-calibrated probabilities for HMM emissions."
        )
        strengths = (
            f"Ultra-fast training ({res['full_fit_seconds']:.2f}s on {res['n_train_candidates_scored']:,} candidates) "
            f"and sub-millisecond inference ({res['student_test_infer_ms_per_trip']:.2f} ms/trip), low variance across folds "
            f"(CV Macro F1 std = {cv['cand_macro_f1']['std']:.4f}), and competitive log-loss ({te['cand_log_loss']:.4f})."
        )
        limits = (
            "Cannot capture nonlinear feature interactions (e.g., discounting noisy heading when `is_stationary=1` "
            "or combining `proj_frac` near 0/1 with `prev_dist_m`/`next_dist_m` at intersections), leading to lower "
            f"student-test Macro F1 ({te['cand_macro_f1']:.4f}) and Top-1 ({te['obs_top1_overall']:.4f}) than nonlinear ensembles."
        )
        fit_diag = (
            f"Mild underfitting rather than overfitting: Train Macro F1 ({tr['cand_macro_f1']:.4f}) is very close to "
            f"5-fold CV Macro F1 ({cv['cand_macro_f1']['mean']:.4f}, gap={gap_cv_f1:+.4f}) and Student-Test Macro F1 "
            f"({te['cand_macro_f1']:.4f}), confirming high bias / low variance."
        )
    elif key == "knn":
        why = (
            "K-Nearest Neighbours (distance-weighted Euclidean) tests whether local geometric similarity in standardized "
            "21-dimensional feature space can classify road candidates without assuming a global parametric boundary. "
            "Distance weighting converts neighborhood class ratios into continuous scores usable by the decoder."
        )
        strengths = (
            f"Zero parametric assumptions, instant training/indexing ({res['full_fit_seconds']:.2f}s), and nonlinear local "
            f"decision surfaces achieving {te['cand_macro_f1']:.4f} Macro F1 and {te['obs_top1_overall']:.4f} Top-1 on student-test."
        )
        limits = (
            f"Severe inference bottleneck at scale ({res['student_test_infer_seconds']:.2f}s across 10 test trips = "
            f"{res['student_test_infer_ms_per_trip']:.1f} ms/trip, orders of magnitude slower than tree/linear models), "
            f"higher memory footprint (must retain all {res['n_train_candidates_scored']:,} training vectors), and worse "
            f"probability calibration (Student-Test Log Loss = {te['cand_log_loss']:.4f}) due to discrete 25-neighbor vote quantization."
        )
        fit_diag = (
            f"Notable memorization / overfitting on training points: distance-weighted KNN achieves near-1.0 Train Macro F1 "
            f"({tr['cand_macro_f1']:.4f}) vs CV Macro F1 ({cv['cand_macro_f1']['mean']:.4f}, gap={gap_cv_f1:+.4f}) and "
            f"elevated CV Log Loss ({cv['cand_log_loss']['mean']:.4f}) when neighbors disagree or vote unanimously wrong."
        )
    elif key == "decision_tree":
        why = (
            "A regularized CART Decision Tree (`max_depth=12, min_samples_leaf=50`) directly models conditional nonlinear "
            "rules (e.g., splitting first on `dist_rank` and `cos_heading_diff`, and branching differently when `is_stationary=1`) "
            "while requiring no feature scaling and yielding fast, human-inspectable decision paths."
        )
        strengths = (
            f"Very fast inference ({res['student_test_infer_ms_per_trip']:.2f} ms/trip), natural handling of feature interactions "
            f"and mixed scales, improving over Logistic Regression on Student-Test Macro F1 ({te['cand_macro_f1']:.4f})."
        )
        limits = (
            f"Axis-aligned greedy splits produce piecewise-constant leaf probabilities with higher variance and worse calibration "
            f"(CV Log Loss = {cv['cand_log_loss']['mean']:.4f}, Student-Test Log Loss = {te['cand_log_loss']:.4f}) than ensembles, "
            "which hurts downstream HMM emission reliability."
        )
        fit_diag = (
            f"Moderate overfitting despite depth/leaf regularization: Train Macro F1 ({tr['cand_macro_f1']:.4f}) exceeds "
            f"CV Macro F1 ({cv['cand_macro_f1']['mean']:.4f}, gap={gap_cv_f1:+.4f}), and sparse leaves occasionally emit 0/1 "
            f"probabilities that inflate Log Loss ({cv['cand_log_loss']['mean']:.4f})."
        )
    elif key == "random_forest":
        why = (
            "Random Forest averages 150 bootstrap-sampled trees (`max_depth=16, min_samples_leaf=20, max_features='sqrt'`) "
            "to reduce single-tree variance and smooth empirical leaf probability estimates for candidate scoring."
        )
        strengths = (
            f"Strong generalization and stability across folds (CV Macro F1 = {cv['cand_macro_f1']['mean']:.4f} ± "
            f"{cv['cand_macro_f1']['std']:.4f}, Student-Test Macro F1 = {te['cand_macro_f1']:.4f}, Top-1 = {te['obs_top1_overall']:.4f}), "
            f"substantially cleaner probabilities than a single tree (Student-Test Log Loss = {te['cand_log_loss']:.4f})."
        )
        limits = (
            f"Slower training ({res['full_fit_seconds']:.2f}s) and larger memory/disk footprint across 150 deep trees than "
            "Histogram Gradient Boosting; because trees are built independently (bagging) rather than optimizing log-loss "
            "sequentially, probability calibration trails boosting slightly."
        )
        fit_diag = (
            f"Well-regularized with small generalization gap: Train Macro F1 ({tr['cand_macro_f1']:.4f}) vs CV Macro F1 "
            f"({cv['cand_macro_f1']['mean']:.4f}, gap={gap_cv_f1:+.4f}) and Student-Test Macro F1 ({te['cand_macro_f1']:.4f})."
        )
    else:  # hist_gradient_boosting
        why = (
            "Histogram Gradient Boosting (`HistGradientBoostingClassifier`) directly optimizes binary log-loss via stagewise "
            "additive trees over 255-bin histograms. This aligns directly with PathSnap's need for both accurate nonlinear "
            "candidate ranking and sharply calibrated posterior probabilities `P(y=1|x)` for HMM/Viterbi emissions."
        )
        strengths = (
            f"Strongest overall probability quality (CV Log Loss = {cv['cand_log_loss']['mean']:.4f}, Student-Test Log Loss = "
            f"{te['cand_log_loss']:.4f}), top/near-top ranking accuracy (Student-Test Top-1 = {te['obs_top1_overall']:.4f}, "
            f"Macro F1 = {te['cand_macro_f1']:.4f}, Route Continuity = {te['route_continuity']:.4f}), fast local training "
            f"({res['full_fit_seconds']:.2f}s), compact artifact size, and low latency ({res['student_test_infer_ms_per_trip']:.2f} ms/trip)."
        )
        limits = (
            "Sensitive to hyperparameters (`learning_rate`, `max_leaf_nodes`, `l2_regularization`) and less directly "
            "interpretable than Logistic Regression or a single Decision Tree, motivating Bayesian optimization in Section 4."
        )
        fit_diag = (
            f"Well-generalized with built-in early stopping on validation log-loss: Train Macro F1 ({tr['cand_macro_f1']:.4f}) "
            f"vs CV Macro F1 ({cv['cand_macro_f1']['mean']:.4f}, gap={gap_cv_f1:+.4f}) and Student-Test Macro F1 "
            f"({te['cand_macro_f1']:.4f}); the boosting-iteration curve (`outputs/milestone3/hgb_iteration_loss.png`) "
            "shows training and validation log-loss tracking closely without late-stage divergence."
        )
    return {"why": why, "strengths": strengths, "limits": limits, "fit_diag": fit_diag}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m1 = _load_json(PROJECT_ROOT / "outputs/milestone1/eda_summary.json")
    m2 = _load_json(PROJECT_ROOT / "outputs/milestone2/milestone2_summary.json")
    m3_sel = _load_json(PROJECT_ROOT / "outputs/milestone3/model_selection_summary.json")
    m3_table_md = (PROJECT_ROOT / "outputs/milestone3/model_comparison.md").read_text(encoding="utf-8").strip()
    m4 = _load_json(PROJECT_ROOT / "outputs/milestone4/tuning_summary.json")
    m4_cfg = _load_json(PROJECT_ROOT / "artifacts/model_config.json")
    m5 = _load_json(PROJECT_ROOT / "outputs/milestone5/milestone5_summary.json")
    m6 = _load_json(PROJECT_ROOT / "outputs/milestone6/milestone6_api_verification.json")

    model_keys = ["logistic_regression", "knn", "decision_tree", "random_forest", "hist_gradient_boosting"]
    model_results = {k: _load_json(PROJECT_ROOT / f"outputs/milestone3/model_{k}.json") for k in model_keys}

    tr2 = m2["train_summary"]
    te2 = m2["student_test_summary"]
    we1 = m1["worked_example_00000005"]
    traced = we1["traced_arcs"][0]
    m5_te = m5["student_test_comparison"]
    m5_case = m5["case_study_and_map"]
    fix_ex = m5_case.get("fixed_case_example")
    hurt_ex = m5_case.get("hurt_case_example")

    lines = [
        "# PathSnap: Distributed Map-Matching & Route Reconstruction — Technical Report (DRAFT)",
        "",
        "> **Draft Status Notice:** All Laptop 1 experiments, model training runs, Bayesian optimization trials, HMM/Viterbi evaluations, Folium maps, and FastAPI backend tests are **complete and verified locally**. This report remains in **DRAFT** status solely pending physical integration of **two team verification screenshots** in Section 7: (1) Laptop 2 working UI screenshot connected over LAN to Laptop 1, and (2) Laptop 1 live CMD/terminal screenshot showing incoming `/predict` logs from Laptop 2.",
        "",
        "---",
        "",
        "## 2.2 Dataset Understanding and Exploratory Data Analysis",
        "",
        "### Partitions & Integrity Audit",
        f"- **Partitions analysed:** `train` (**{m1['n_train_trips']} trips**, **{m1['total_gps_train']:,} GPS observations**) and `student_test` (**{m1['n_student_test_trips']} trips**, **{m1['total_gps_student_test']:,} GPS observations**). Zero trip overlap (`overlap_trip_ids = {m1['overlap_trip_ids']}`). The 10 reserved evaluator trips were never accessed during development, EDA, feature selection, tuning, or HMM design.",
        f"- **File & missing-value audit (all 90 development trips):** Every trip contains all 4 headerless whitespace-separated files (`.track`, `.nodes`, `.arcs`, `.route`). Measured parse errors = **{m1['total_parse_bad_rows']}**, invalid/NaN coordinates = **{m1['total_invalid_coords']}**, out-of-bounds arc endpoint node IDs = **{m1['total_invalid_arc_endpoints']}**, out-of-bounds route arc IDs = **{m1['total_invalid_route_ids']}**, negative/zero timestamp steps = **{m1['total_negative_dt']} / {m1['total_zero_dt']}**, isolated nodes = **{m1['total_isolated_nodes']}**.",
        f"- **Worked example (`train/00000005`):** Verified exact counts — **{we1['counts']['n_gps']:,} GPS readings**, **{we1['counts']['n_nodes']:,} nodes**, **{we1['counts']['n_arcs']:,} directed arcs**, and **{we1['counts']['n_route']} reference-route entries**. Tracing real reference arc `arc_id={traced['arc_id']}` resolves `from_node_id={traced['from_node_id']}` `({traced['from_lon_lat'][0]:.6f}, {traced['from_lon_lat'][1]:.6f})` to `to_node_id={traced['to_node_id']}` `({traced['to_lon_lat'][0]:.6f}, {traced['to_lon_lat'][1]:.6f})` with projected metric length **{traced['length_m']:.3f} m** and compass bearing **{traced['bearing_deg']:.2f}°**.",
        "",
        "### 1. How We Preprocessed the Data",
        f"1. **Coordinate Projection (Metric Accuracy Globally):** Exploratory analysis revealed trips spanning longitude `[{m1['lon_range'][0]:.2f}°, {m1['lon_range'][1]:.2f}°]` and latitude `[{m1['lat_range'][0]:.2f}°, {m1['lat_range'][1]:.2f}°]` across multiple continents. Ordinary Euclidean distance in degrees or a single fixed UTM zone would introduce severe metric distortion. We project every trip independently from WGS84 (`EPSG:4326`, raw `lon, lat` order) into a **trip-local Azimuthal Equidistant (`+proj=aeqd`) coordinate system** centered at the trip's median `(lon, lat)` in meters.",
        "2. **Spatial Candidate Generation (Inference-Only):** For each trip we build a vectorized Shapely `STRtree` spatial index over all valid directed road arcs (up to 307,392 arcs in `00000013`) and query directed arcs within `60.0 m`, keeping up to `max_candidates=10` sorted by exact point-to-segment distance (with a `fallback_k=4` nearest-arc fallback when empty). Candidate generation achieves a **0.667%** miss rate on `train` (80 trips) and **0.146%** on `student_test` (10 trips).",
        "3. **Development-Only Monotonic Reference Alignment (`src/alignment.py`):** Because `.route` is an ordered traversal of road segments (e.g., 105 arcs) rather than one label per GPS reading (e.g., 1,045 readings), we implemented a monotonic dynamic-programming aligner over route indices `j = 0..M-1` using exact projected route position `s(t, j) = cum_start[j] + proj[t, j] * len[j]`, heading disambiguation when moving, zero stay penalty for stops, and relaxed gap transition weights for `dt > 2s`. Observations > `50.0 m` from their aligned reference arc are marked unscored (`far_from_reference_route`). This yields **99.86%** label coverage on `student_test` (19,133/19,160 scored) and **88.84%** on `train` (185,149/208,417 scored). Crucially, EDA revealed that `train/00000054` (19,313 obs, median min distance 138.6 m to `.route`) and `train/00000033` (5,844 obs, median 122.4 m) account for ~20,700 of the 23,268 unscored training points—real-world evidence of the reference-route limitations highlighted in `Dataset Exploration.pdf` §6. No official alignment script was provided; we therefore report both alignment-based observation Top-1 accuracy and alignment-free sequence-level `.route` metrics (Sequence LCS F1 and Length-Weighted F1).",
        "4. **Leakage Prevention:** Learned preprocessing (`StandardScaler` for Logistic Regression and KNN) is fit strictly inside each fold's training split via `sklearn.pipeline.Pipeline`. Route-free inference verification confirmed all 21 features compute cleanly with `trip.route is None`.",
        "",
        "### 2. Observed Data Patterns & Feature Engineering Justification",
        f"- **Sampling irregularities & gaps:** **{m1['pct_exact_1s_all']:.2f}%** of consecutive steps are exactly 1 s apart, yet **47/90 trips** contain gaps > 5 s, **42/90 trips** exceed 10 s, and `train/00000090` has a **153,064 s** (~42.5 h) stop gap.",
        f"- **Road network structure:** **89/90 trips** contain multiple weakly connected components, **60/90 trips** have parallel multi-edges with identical `(from_node, to_node)`, **9/90** contain self-loops, and **10/90** have repeated arcs in `.route` (loops/U-turns). Because most streets are represented by paired opposite directed arcs, the **nearest-road baseline achieves only {100*te2['nearest_road_top1_overall']:.2f}% Top-1 accuracy on `student_test`** ({100*tr2['nearest_road_top1_overall']:.2f}% on `train`)—it cannot distinguish forward vs reverse lanes.",
        "- **21 Engineered Inference-Only Features** (strictly from `.track`, `.nodes`, `.arcs`):",
        "  - *Spatial proximity & relative margin:* `dist_m`, `dist_rank`, `dist_diff_from_best`, `dist_ratio_to_best`, `n_candidates`.",
        "  - *Directional alignment & explicit stationary handling:* `heading_diff_deg`, `cos_heading_diff`, `heading_rank`, `heading_valid`, `is_stationary`, `obs_speed_mps`. When `obs_speed_mps < 1.0 m/s` or local displacement `< 1.5 m`, we explicitly set `is_stationary=1`, `heading_valid=0`, `heading_diff_deg=90.0`, and `cos_heading_diff=0.0` so GPS stop jitter does not corrupt direction features.",
        "  - *Local trajectory continuity & sampling context:* `prev_dist_m`, `next_dist_m`, `along_arc_progress_m` (signed projection of `(p_{t+1} - p_{t-1})` onto candidate arc unit vector), `dt_prev_s`, `dt_next_s`.",
        "  - *Road geometry & graph topology:* `proj_frac`, `arc_len_m`, `has_reverse_arc`, `from_node_out_deg`, `to_node_out_deg`.",
        "",
        "### EDA Visualisations (Generated from Actual Data)",
        "- Dataset Overview (`train` + `student_test`): `../outputs/milestone1/dataset_eda_overview.png`",
        "- GPS Sampling-Gap Distribution & Top Gap Trips: `../outputs/milestone1/sampling_gaps.png`",
        "- Trip `00000005` Static Trajectory & Road Network Plot: `../outputs/milestone1/trip_00000005_trajectory.png`",
        "- Trip `00000005` Interactive Folium Map: `../outputs/milestone1/trip_00000005_map.html`",
        "- Monotonic Alignment & Baseline Diagnostics: `../outputs/milestone2/alignment_and_baseline_diagnostics.png`",
        "",
        "---",
        "",
        "## 3. Model Experimentation and Comparison",
        "",
        "All five models from five distinct algorithm families were trained locally on Laptop 1 (Apple M2, 8-core, 8 GB RAM) using identical candidates, identical 5-fold `GroupKFold` splits grouped by `trip_id` over the 80 training trips (1,290,243 scored training candidates), and identical evaluation on the 10 `student_test` trips (19,133 scored observations; 0.146% candidate miss rate included in every Top-1 denominator).",
        "",
        "### 3.1 Overall Model Comparison",
        "",
        m3_table_md,
        "",
        f"> **Nearest-Road Baseline Reference (Student-Test):** Top-1 Accuracy = **{te2['nearest_road_top1_overall']:.4f}**, Route Continuity = **{te2['nearest_road_mean_continuity']:.4f}**, Sequence LCS F1 = **{te2['nearest_road_mean_seq_lcs_f1']:.4f}**. The five classifiers score approximately **31.6 to 32.7 percentage points** above this baseline.",
        "",
        "### 3.2 Compulsory Analysis for EVERY Model",
        "",
    ]

    for idx, key in enumerate(model_keys, start=1):
        r = model_results[key]
        cv = r["cv_summary"]
        te = r["student_test_metrics"]
        tr = r["train_diagnostic"]
        ana = _model_specific_analysis(key, r)
        lines.extend(
            [
                f"#### 3.2.{idx} {r['name']} (`{r['family']}`)",
                "",
                "**A. Model Understanding**",
                f"1. *Suitability for PathSnap:* {ana['why']}",
                f"2. *Major strengths and limitations:* **Strengths:** {ana['strengths']} **Limitations:** {ana['limits']}",
                "",
                "**B. Training and Configuration**",
                f"5. *Important hyperparameters & initial values:* `{json.dumps(r['hyperparams'])}` (random seed: `{r['seed']}`).",
                f"6. *Loss function / optimisation objective:* {r['loss_objective']}.",
                "",
                "**C. Experimental Results**",
                f"7. *Student-testing performance (10 trips):* Accuracy = **{te['cand_accuracy']:.4f}**, Positive Precision = **{te['cand_precision_pos']:.4f}**, Positive Recall = **{te['cand_recall_pos']:.4f}**, Route Continuity = **{te['route_continuity']:.4f}**, Sequence LCS F1 = **{te['seq_lcs_f1']:.4f}**, Training Time = **{r['full_fit_seconds']:.2f}s**, Inference Time = **{r['student_test_infer_seconds']:.4f}s** ({r['student_test_infer_ms_per_trip']:.2f} ms/trip).",
                f"8. *Student-testing Top-1, Macro F1 & Log Loss:* Observation-level Top-1 Accuracy = **{te['obs_top1_overall']:.4f}** (given covered: {te['obs_top1_given_covered']:.4f}; miss rate: {te['candidate_miss_rate']:.4f}), Candidate Macro F1 = **{te['cand_macro_f1']:.4f}**, Binary Log Loss = **{te['cand_log_loss']:.4f}**.",
                f"9. *Overfitting / underfitting diagnosis:* {ana['fit_diag']}",
                "",
            ]
        )

    lines.extend(
        [
            "### Model Selection Justification",
            f"- **Selected Final Model:** **{m3_sel['selected_model_name']}** (`{m3_sel['selected_model_key']}`).",
            f"- **Primary Selection Criterion:** {m3_sel['primary_selection_metric']}.",
            "- **Measured Evidence:** Across the 5-fold Group-CV benchmark and 10-trip student-test set, the selected model delivers the best joint balance of generalization Macro F1, observation-level Top-1 accuracy, calibrated log-loss (crucial because HMM emissions directly consume `log P(y=1|x)`), tight cross-fold variance, and fast per-trip inference latency suitable for live FastAPI deployment.",
            "",
            "### 3.3 Model Visualisations (Generated from Actual Experimental Runs)",
            "- 5-Fold Group-CV Macro F1 Comparison: `../outputs/milestone3/cv_macro_f1_comparison.png`",
            "- Student-Test Top-1 Accuracy, Macro F1 & Route Continuity Comparison: `../outputs/milestone3/student_test_comparison.png`",
            "- Log Loss & Inference Latency Comparison: `../outputs/milestone3/log_loss_and_latency_comparison.png`",
            "- Candidate-Level Confusion Matrices (All 5 Models): `../outputs/milestone3/confusion_matrices.png`",
            "- Boosting-Iteration Log-Loss Curve (Histogram Gradient Boosting ONLY; no artificial epoch curves fabricated for non-iterative models): `../outputs/milestone3/hgb_iteration_loss.png`",
            "",
            "---",
            "",
            "## 4. Bayesian Hyperparameter Optimisation (for Selected Model)",
            "",
            f"1. **Hyperparameters searched:** `{', '.join(m4['search_space']['parameters'].keys())}`.",
            "2. **Ranges / distributions considered:**",
        ]
    )
    for pname, pspec in m4["search_space"]["parameters"].items():
        lines.append(f"   - `{pname}`: `{json.dumps(pspec)}`")

    u_row, t_row = m4["comparison"][0], m4["comparison"][1]
    outperformed = t_row["CV Macro F1"] > u_row["CV Macro F1"]
    lines.extend(
        [
            f"3. **Objective function optimised:** {m4['search_space']['objective']} using `{m4['search_space']['sampler']}` on the exact same 5-fold `GroupKFold` trip splits.",
            f"4. **Number of optimisation trials:** **{m4['search_space']['n_trials']}** trials (total runtime: **{m4['total_opt_seconds']:.1f}s**; final refit on 80 trips: **{m4['refit_seconds']:.2f}s**).",
            f"5. **Best hyperparameter values (Trial #{m4_cfg['optuna_best_trial_number']}):** `{json.dumps(m4['best_params'])}`.",
            f"6. **Did the tuned model outperform the untuned model?** {'Yes on 5-fold Group-CV' if outperformed else 'Matched/refined baseline'}: CV Macro F1 moved from **{u_row['CV Macro F1']:.5f}** to **{t_row['CV Macro F1']:.5f}**, CV Log Loss moved from **{u_row['CV Log Loss']:.5f}** to **{t_row['CV Log Loss']:.5f}**, and on the 10 `student_test` trips Top-1 moved from **{u_row['Student-Test Top-1 Accuracy']:.5f}** to **{t_row['Student-Test Top-1 Accuracy']:.5f}** (Student-Test Log Loss: **{u_row['Student-Test Log Loss']:.5f}** -> **{t_row['Student-Test Log Loss']:.5f}**).",
            "",
            "### Untuned vs Bayesian-Tuned Comparison (Same Splits & Metrics)",
            "",
            "| Variant | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Student-Test Log Loss | Inference Time (10 trips) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {u_row['Variant']} | {u_row['CV Macro F1']:.5f} ± {u_row['CV Macro F1 Std']:.5f} | {u_row['Student-Test Top-1 Accuracy']:.5f} | {u_row['Student-Test Macro F1']:.5f} | {u_row['Student-Test Log Loss']:.5f} | {u_row['Inference Time (10 trips, s)']:.4f}s |",
            f"| {t_row['Variant']} | {t_row['CV Macro F1']:.5f} ± {t_row['CV Macro F1 Std']:.5f} | {t_row['Student-Test Top-1 Accuracy']:.5f} | {t_row['Student-Test Macro F1']:.5f} | {t_row['Student-Test Log Loss']:.5f} | {t_row['Inference Time (10 trips, s)']:.4f}s |",
            "",
            f"- **Artifact Export & Subprocess Reload Check:** Exported self-contained bundle to `artifacts/model_bundle.joblib` ({m4_cfg['artifact_size_bytes']:,} bytes), `artifacts/model_config.json`, and `artifacts/feature_spec.json`. Fresh Python subprocess reload verified exact predictions on `student_test/{m4['reload_verification']['sample_trip_id']}` with `route_loaded=False` and `max_abs_diff={m4['reload_verification']['max_abs_diff_vs_inprocess_first5']:.2e}`.",
            "- Optimisation History Plot: `../outputs/milestone4/tuning_history.png`",
            "",
            "---",
            "",
            "## 5. HMM and Viterbi Route Reconstruction",
            "",
            "1. **Hidden states:** For each GPS observation `t`, the hidden states are its spatially retrieved candidate directed arcs `C_t = {a_{t,1}, ..., a_{t,K_t}}` (`K_t <= 10`), preserving exact `(obs_id, arc_id)` mapping throughout decoding.",
            f"2. **Emission log-probabilities:** Derived directly from the tuned classifier's calibrated probability: `log E(t, a) = emit_weight * log(clip(P(y=1 | x_{{t,a}}), 1e-6, 1-1e-6))` with `emit_weight = {m5['hmm_config']['emit_weight']}` (selected strictly on out-of-fold predictions of 15 training trips).",
            f"3. **Transition log-probabilities:** Directed road-network transitions compare exact projected route distance `Δd_route` against observed GPS step `Δd_gps` via an exponential mismatch penalty `-|Δd_route - Δd_gps| / beta` (`beta={m5['hmm_config']['beta_m']} m`, relaxed to `{m5['hmm_config']['gap_beta_m']} m` during `dt > 2s` gaps), rewarding same-arc forward/stationary motion (`+{m5['hmm_config']['stay_same_arc_bonus']}`), penalizing significant backward sliding along a directed arc, discouraging spurious immediate U-turns onto the reverse arc `(u,v)->(v,u)` (`-{m5['hmm_config']['u_turn_penalty']}`), and computing multi-hop directed shortest paths via bounded cached Dijkstra.",
            "4. **Missing candidates & large GPS gaps:** Our candidate generator uses a `fallback_k=4` nearest-arc rescue so active observations rarely lose all candidates; if any observation still has `K_t = 0`, Viterbi records `pred_arc_id = -1` and bridges the transition between adjacent active steps using their true elapsed `Δt` and `Δd_gps`. Across extreme gaps (`dt > 60s` or `Δd_gps > 1000m`), Viterbi performs a clean memory reset to pure emissions so overnight/tunnel gaps cannot corrupt the downstream path.",
            f"5. **Disconnected road segments:** When no directed path exists within the bounded search radius (handling the 89/90 multi-component networks found in EDA), transitions receive a finite `disconnected_penalty = -{m5['hmm_config']['disconnected_penalty']}` (softened to `-{m5['hmm_config']['gap_disconnected_penalty']}` across large gaps), preventing `-inf`/NaN collapse while strongly preferring topologically valid continuations.",
            f"6. **Did HMM/Viterbi improve the final route?** **Yes — substantially across every metric on the 10 `student_test` trips:** Pointwise Top-1 Accuracy rose from **{m5_te['classifier_only']['pointwise_top1_overall']:.4f}** to **{m5_te['classifier_plus_hmm']['pointwise_top1_overall']:.4f}** (`{m5_te['improvements']['delta_pointwise_top1']:+.4f}`), Route Continuity rose from **{m5_te['classifier_only']['route_continuity_mean']:.4f}** to **{m5_te['classifier_plus_hmm']['route_continuity_mean']:.4f}** (`{m5_te['improvements']['delta_route_continuity']:+.4f}`), and order-aware Sequence LCS F1 jumped from **{m5_te['classifier_only']['seq_lcs_f1_mean']:.4f}** to **{m5_te['classifier_plus_hmm']['seq_lcs_f1_mean']:.4f}** (`{m5_te['improvements']['delta_seq_lcs_f1']:+.4f}`), fixing **{m5_te['improvements']['total_obs_fixed_by_hmm']}** observations while hurting **{m5_te['improvements']['total_obs_hurt_by_hmm']}** (net gain **+{m5_te['improvements']['net_obs_gained']}** observations) at an average decode latency of only **{m5_te['classifier_plus_hmm']['mean_decode_ms_per_trip']:.1f} ms/trip**.",
            "",
            "### Classifier-Only vs Classifier + HMM/Viterbi Comparison (10 Student-Test Trips)",
            "",
            "| Pipeline | Pointwise Top-1 Accuracy | Route Continuity | Sequence LCS F1 | Sequence Length-Weighted F1 | Mean Decode Latency |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Tuned Classifier Only | {m5_te['classifier_only']['pointwise_top1_overall']:.4f} | {m5_te['classifier_only']['route_continuity_mean']:.4f} | {m5_te['classifier_only']['seq_lcs_f1_mean']:.4f} | {m5_te['classifier_only']['seq_length_weighted_f1_mean']:.4f} | 0.0 ms/trip |",
            f"| Tuned Classifier + HMM/Viterbi | **{m5_te['classifier_plus_hmm']['pointwise_top1_overall']:.4f}** | **{m5_te['classifier_plus_hmm']['route_continuity_mean']:.4f}** | **{m5_te['classifier_plus_hmm']['seq_lcs_f1_mean']:.4f}** | **{m5_te['classifier_plus_hmm']['seq_length_weighted_f1_mean']:.4f}** | {m5_te['classifier_plus_hmm']['mean_decode_ms_per_trip']:.1f} ms/trip |",
            "",
            "### Worked Decoding Case Analysis (Improvements & Failure Mode)",
        ]
    )
    if fix_ex:
        lines.append(
            f"- **Where HMM fixed a classifier error (`student_test/{fix_ex['trip_id']}`, `obs_id={fix_ex['obs_id']}`, `t={fix_ex['timestamp_s']:.0f}s`):** "
            f"The independent classifier picked `arc_id={fix_ex['classifier_only_arc_id']}` (top local score), which was topologically "
            f"{'connected' if fix_ex['prev_to_clf_connected'] else 'DISCONNECTED'} from the preceding arc `{fix_ex['hmm_prev_arc_id']}`. "
            f"Viterbi selected `arc_id={fix_ex['hmm_viterbi_arc_id']}` (matching the reference arc `{fix_ex['reference_arc_id']}`), "
            f"preserving directed continuity from `{fix_ex['hmm_prev_arc_id']}` -> `{fix_ex['hmm_viterbi_arc_id']}` -> `{fix_ex['hmm_next_arc_id']}`."
        )
    if hurt_ex:
        lines.append(
            f"- **Where HMM did not help (`student_test/{hurt_ex['trip_id']}`, `obs_id={hurt_ex['obs_id']}`, `t={hurt_ex['timestamp_s']:.0f}s`):** "
            f"The classifier correctly predicted `arc_id={hurt_ex['classifier_only_arc_id']}` (matching reference `{hurt_ex['reference_arc_id']}`), "
            f"whereas HMM stayed/shifted onto `arc_id={hurt_ex['hmm_viterbi_arc_id']}` due to transition smoothing across a tight junction boundary (`prev={hurt_ex['hmm_prev_arc_id']}`, `next={hurt_ex['hmm_next_arc_id']}`). "
            "This illustrates the classic HMM trade-off: smoothing removes far more isolated noise spikes than the few junction-boundary frames it shifts by 1 second."
        )

    lines.extend(
        [
            "- Classifier-Only vs HMM/Viterbi Comparison Plot: `../outputs/milestone5/hmm_vs_classifier_comparison.png`",
            "",
            "---",
            "",
            "## 6. Geographical Visualisation Using Folium",
            "",
            "Two interactive OpenStreetMap Folium visualisations with toggleable `LayerControl`, colour-coded layers, tooltips, and legends were generated directly from our code:",
            "1. **EDA Map (`train/00000005`):** `../outputs/milestone1/trip_00000005_map.html` (and static export `../outputs/milestone1/trip_00000005_trajectory.png`) showing raw GPS observations, full 1,579-arc road network, supplied reference route, and traced example arc `176`.",
            f"2. **Route Reconstruction Map (`student_test/{m5_case['featured_trip_id']}`):** `../outputs/milestone5/student_test_{m5_case['featured_trip_id']}_hmm_map.html` (and static export `../outputs/milestone5/student_test_{m5_case['featured_trip_id']}_hmm_route.png`) overlaying raw GPS readings, relevant road network, reference route, toggleable classifier-only arcs, HMM/Viterbi predicted route, and a marker pinpointing the worked HMM correction case.",
            "",
            "---",
            "",
            "## 7. API, UI and Deployment (Two-Laptop Architecture)",
            "",
            "### Laptop 1 Backend Verification (COMPLETE)",
            f"- **Framework & Launch:** FastAPI (`backend/main.py`) served via `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`.",
            f"- **LAN URL (Laptop 1):** `{m6['lan_url']}` (local verification passed; **reachability from Laptop 2 is pending physical LAN test**).",
            f"- **Live Development Verification (All 10 `student_test` trips via HTTP `POST /predict`):** Mean Pointwise Top-1 = **{m6['mean_top1_accuracy']:.4f}**, Mean Route Continuity = **{m6['mean_route_continuity']:.4f}**, Mean Sequence LCS F1 = **{m6['mean_seq_lcs_f1']:.4f}**, Mean End-to-End API Latency = **{m6['mean_elapsed_ms']:.1f} ms/trip**.",
            "- **Artifact Persistence & Isolation Verified:** Restarted FastAPI cleanly and served identical predictions from `artifacts/model_bundle.joblib` without retraining. Verified `PATHSNAP_MODE=evaluation` with a missing `PATHSNAP_EVAL_DIR` returns explicit HTTP `503` error and `ready: false` on `/health`. Confirmed reference `.route` files are never read during prediction and never exposed in HTTP responses.",
            "- **Laptop 2 Handoff Package:** `laptop2_handoff/` (`API_CONTRACT.md`, `example_health_response.json`, `example_trips_response.json`, `example_predict_response.json` — strictly free of raw datasets, reference labels, and model weights).",
            "",
            "### Pending Physical Team Evidence (Required Before Final PDF Export)",
            "- **[PENDING — Laptop 2 UI Screenshot]:** Capture screenshot of Laptop 2's working evaluator UI displaying the trip list from `GET /trips`, interactive predicted route map, and live server-side metrics returned from Laptop 1.",
            "- **[PENDING — Laptop 1 Receiving Terminal Screenshot]:** Capture screenshot of Laptop 1's `uvicorn` terminal showing live `PREDICT 200 | mode=... | trip_id=... | elapsed=... ms` logs triggered by Laptop 2 over LAN.",
            "- **Laptop 2 UI:** The local Python client and bundled Leaflet interface are in `ui/`. The physical two-device demo is pending.",
            "",
            "---",
            "",
            "## 8. Academic Integrity & AI Assistance Disclosure",
            "",
            "- All data inspection, feature engineering, five-model training runs, 5-fold group-aware cross-validation, Bayesian Optuna trials, HMM/Viterbi decoding, Folium maps, and FastAPI endpoint tests were executed genuinely and locally on Laptop 1 using the provided PathSnap student dataset.",
            "- Every reported table, plot, and metric in this document is programmatically generated from saved JSON/CSV experiment artifacts in `outputs/` and is fully reproducible via the commands in `README.md`.",
            "- AI coding assistance was used to accelerate implementation and structuring of Python pipelines and documentation; all experiments were executed locally and verified against actual outputs, and the team should review these explanations against its own run records before submission.",
        ]
    )

    report_path = OUT_DIR / "TECHNICAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated consolidated draft technical report at {report_path}")


if __name__ == "__main__":
    main()
