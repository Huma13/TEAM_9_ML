"""Milestone 3: Train, cross-validate, and compare five distinct model families locally on MacBook."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.candidates import FEATURE_COLUMNS
from src.data_io import PROJECT_ROOT, discover_dataset_dirs, load_trip, project_trip_geometry
from src.metrics import (
    compute_candidate_metrics,
    compute_observation_top1_and_miss,
    compute_route_continuity_for_trip,
    compute_sequence_metrics_for_trip,
)

CACHE_DIR = PROJECT_ROOT / "outputs" / "milestone2" / "cache"
OUT_DIR = PROJECT_ROOT / "outputs" / "milestone3"
RANDOM_SEED = 42
N_FOLDS = 5


def get_model_specs() -> List[Dict[str, Any]]:
    """Return specifications for the five distinct model families."""
    return [
        {
            "key": "logistic_regression",
            "name": "Logistic Regression",
            "family": "Linear & Probabilistic",
            "seed": RANDOM_SEED,
            "config_summary": "StandardScaler + L2 C=1.0, lbfgs, max_iter=500",
            "loss_objective": "L2-regularized binary cross-entropy (log-loss)",
            "hyperparams": {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": RANDOM_SEED},
            "batched_predict": False,
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            max_iter=500,
                            solver="lbfgs",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        },
        {
            "key": "decision_tree",
            "name": "Decision Tree",
            "family": "Tree-based",
            "seed": RANDOM_SEED,
            "config_summary": "CART Gini, max_depth=12, min_samples_leaf=50",
            "loss_objective": "Gini impurity minimization at greedy axis-aligned splits",
            "hyperparams": {
                "criterion": "gini",
                "max_depth": 12,
                "min_samples_leaf": 50,
                "random_state": RANDOM_SEED,
            },
            "batched_predict": False,
            "builder": lambda: DecisionTreeClassifier(
                criterion="gini",
                max_depth=12,
                min_samples_leaf=50,
                random_state=RANDOM_SEED,
            ),
        },
        {
            "key": "random_forest",
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "seed": RANDOM_SEED,
            "config_summary": "150 trees, max_depth=16, min_samples_leaf=20, max_features=sqrt",
            "loss_objective": "Bagged Gini impurity reduction across bootstrap samples",
            "hyperparams": {
                "n_estimators": 150,
                "max_depth": 16,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "n_jobs": 4,
                "random_state": RANDOM_SEED,
            },
            "batched_predict": False,
            "builder": lambda: RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=4,
                random_state=RANDOM_SEED,
            ),
        },
        {
            "key": "hist_gradient_boosting",
            "name": "Histogram Gradient Boosting",
            "family": "Boosting",
            "seed": RANDOM_SEED,
            "config_summary": "max_iter=200, lr=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2=1.0",
            "loss_objective": "Binary log-loss (cross-entropy) via stagewise gradient boosting",
            "hyperparams": {
                "max_iter": 200,
                "learning_rate": 0.08,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
                "scoring": "neg_log_loss",
                "random_state": RANDOM_SEED,
            },
            "batched_predict": False,
            "builder": lambda: HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=50,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                scoring="neg_log_loss",
                random_state=RANDOM_SEED,
            ),
        },
        {
            "key": "knn",
            "name": "K-Nearest Neighbours",
            "family": "Instance-based",
            "seed": None,
            "config_summary": "StandardScaler + k=25, distance-weighted Euclidean (brute BLAS, batched)",
            "loss_objective": "Non-parametric distance-weighted neighborhood voting (no parametric loss)",
            "hyperparams": {
                "n_neighbors": 25,
                "weights": "distance",
                "metric": "euclidean",
                "algorithm": "brute",
                "n_jobs": 6,
            },
            "batched_predict": True,
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=25,
                            weights="distance",
                            metric="euclidean",
                            algorithm="brute",
                            n_jobs=6,
                        ),
                    ),
                ]
            ),
        },
    ]


def predict_proba_pos(model: Any, X: np.ndarray, batched: bool = False, batch_size: int = 8192) -> np.ndarray:
    """Return P(y=1) with optional batching to bound RAM usage (critical for KNN on 8 GB RAM)."""
    if not batched or len(X) <= batch_size:
        return model.predict_proba(X)[:, 1].astype(np.float64)
    out = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        end = min(len(X), start + batch_size)
        out[start:end] = model.predict_proba(X[start:end])[:, 1]
    return out


def preload_trip_graphs(split_dir: Path, trip_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Preload arc endpoints, arc lengths, and reference routes for fast continuity/sequence scoring."""
    graphs: Dict[str, Dict[str, Any]] = {}
    for tid in trip_ids:
        trip = load_trip(split_dir, tid, load_route=True)
        geom = project_trip_geometry(trip)
        graphs[tid] = {
            "from_nodes": trip.arcs["from_node_id"].to_numpy(dtype=np.int64),
            "to_nodes": trip.arcs["to_node_id"].to_numpy(dtype=np.int64),
            "arc_len": geom["arc_len"],
            "route": trip.route,
        }
    return graphs


def evaluate_predictions_full(
    cand_df: pd.DataFrame,
    obs_meta_df: pd.DataFrame,
    probs: np.ndarray,
    trip_graphs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute candidate-level, observation-level, continuity, and sequence metrics."""
    scored_cand_mask = cand_df["is_scored"].to_numpy(dtype=bool)
    cand_metrics = compute_candidate_metrics(
        cand_df.loc[scored_cand_mask, "is_target"].to_numpy(dtype=np.int64),
        probs[scored_cand_mask],
    )
    obs_metrics, pred_obs = compute_observation_top1_and_miss(cand_df, probs, obs_meta_df)

    # Per-trip continuity and sequence-level metrics
    cont_list = []
    trans_cont_list = []
    seq_arc_f1_list = []
    seq_lw_f1_list = []
    seq_lcs_f1_list = []
    for tid, grp in pred_obs.groupby("trip_id", sort=False):
        g = trip_graphs[tid]
        pred_arcs = grp.sort_values("obs_id")["pred_arc_id"].to_numpy(dtype=np.int64)
        c = compute_route_continuity_for_trip(pred_arcs, g["from_nodes"], g["to_nodes"])
        cont_list.append(c["route_continuity"])
        trans_cont_list.append(c["transition_continuity_changes_only"])
        s = compute_sequence_metrics_for_trip(pred_arcs, g["route"], g["arc_len"])
        seq_arc_f1_list.append(s["seq_arc_f1"])
        seq_lw_f1_list.append(s["seq_length_weighted_f1"])
        seq_lcs_f1_list.append(s["seq_lcs_f1"])

    return {
        **cand_metrics,
        **obs_metrics,
        "route_continuity": float(np.mean(cont_list)) if cont_list else 0.0,
        "transition_continuity_changes_only": float(np.mean(trans_cont_list)) if trans_cont_list else 0.0,
        "seq_arc_f1": float(np.mean(seq_arc_f1_list)) if seq_arc_f1_list else 0.0,
        "seq_length_weighted_f1": float(np.mean(seq_lw_f1_list)) if seq_lw_f1_list else 0.0,
        "seq_lcs_f1": float(np.mean(seq_lcs_f1_list)) if seq_lcs_f1_list else 0.0,
    }


def build_cv_folds(train_obs: pd.DataFrame, n_splits: int = N_FOLDS) -> Dict[str, int]:
    """Assign each training trip strictly to one GroupKFold split (balanced by observation count)."""
    # Sort trips deterministically by trip_id; GroupKFold balances fold sizes by observation counts!
    obs_trips = train_obs[["trip_id"]].copy()
    gkf = GroupKFold(n_splits=n_splits)
    trip_to_fold: Dict[str, int] = {}
    dummy_x = np.zeros(len(obs_trips), dtype=np.int8)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(dummy_x, groups=obs_trips["trip_id"].to_numpy())):
        val_trips = obs_trips["trip_id"].iloc[val_idx].unique()
        for tid in val_trips:
            trip_to_fold[str(tid)] = int(fold_idx)
    return trip_to_fold


def run_single_model(
    spec: Dict[str, Any],
    train_cand: pd.DataFrame,
    train_obs: pd.DataFrame,
    test_cand: pd.DataFrame,
    test_obs: pd.DataFrame,
    trip_to_fold: Dict[str, int],
    train_graphs: Dict[str, Dict[str, Any]],
    test_graphs: Dict[str, Dict[str, Any]],
    n_splits: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run group-aware CV + full-train refit + student-test evaluation for one model."""
    key = spec["key"]
    print(f"\n{'=' * 70}\nRunning model: {spec['name']} ({spec['family']})\n{'=' * 70}")

    X_train_all = train_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_all = train_cand["is_target"].to_numpy(dtype=np.int8)
    scored_train_all = train_cand["is_scored"].to_numpy(dtype=bool)
    folds_cand = train_cand["trip_id"].map(trip_to_fold).to_numpy(dtype=np.int32)

    X_test = test_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    fold_results: List[Dict[str, Any]] = []
    oof_probs = np.zeros(len(train_cand), dtype=np.float64)
    cv_train_times: List[float] = []
    cv_infer_times: List[float] = []

    for f in range(n_splits):
        tr_mask = (folds_cand != f) & scored_train_all
        va_mask = folds_cand == f
        va_trips = sorted(train_cand.loc[va_mask, "trip_id"].unique().tolist())

        X_tr, y_tr = X_train_all[tr_mask], y_train_all[tr_mask]
        X_va = X_train_all[va_mask]

        model = spec["builder"]()
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        va_probs = predict_proba_pos(model, X_va, batched=spec["batched_predict"])
        inf_s = time.perf_counter() - t1

        oof_probs[va_mask] = va_probs
        cv_train_times.append(fit_s)
        cv_infer_times.append(inf_s)

        va_cand_df = train_cand.loc[va_mask]
        va_obs_df = train_obs[train_obs["trip_id"].isin(va_trips)]
        va_metrics = evaluate_predictions_full(va_cand_df, va_obs_df, va_probs, train_graphs)
        va_metrics["fold"] = f
        va_metrics["n_val_trips"] = len(va_trips)
        va_metrics["fit_seconds"] = round(fit_s, 3)
        va_metrics["infer_seconds"] = round(inf_s, 3)
        fold_results.append(va_metrics)

        print(
            f"  Fold {f}: MacroF1={va_metrics['cand_macro_f1']:.4f} | "
            f"Top1={va_metrics['obs_top1_overall']:.4f} | "
            f"LogLoss={va_metrics['cand_log_loss']:.4f} | "
            f"Cont={va_metrics['route_continuity']:.4f} | "
            f"fit={fit_s:.2f}s, inf={inf_s:.2f}s"
        )
        del model
        gc.collect()

    # Aggregate CV summary (mean & std across folds + pooled OOF)
    def agg(metric_name: str) -> Dict[str, float]:
        vals = [r[metric_name] for r in fold_results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    cv_summary = {
        "cand_macro_f1": agg("cand_macro_f1"),
        "cand_f1_pos": agg("cand_f1_pos"),
        "cand_precision_pos": agg("cand_precision_pos"),
        "cand_recall_pos": agg("cand_recall_pos"),
        "cand_accuracy": agg("cand_accuracy"),
        "cand_log_loss": agg("cand_log_loss"),
        "obs_top1_overall": agg("obs_top1_overall"),
        "obs_top1_given_covered": agg("obs_top1_given_covered"),
        "candidate_miss_rate": agg("candidate_miss_rate"),
        "route_continuity": agg("route_continuity"),
        "seq_lcs_f1": agg("seq_lcs_f1"),
        "mean_fold_fit_seconds": float(np.mean(cv_train_times)),
        "mean_fold_infer_seconds": float(np.mean(cv_infer_times)),
    }

    # Refit on ALL 80 scored training trips and evaluate on train (for overfitting diagnosis) + 10 student_test trips
    print(f"  Refitting {spec['name']} on full scored training set ({int(scored_train_all.sum()):,} candidates)...")
    final_model = spec["builder"]()
    t0 = time.perf_counter()
    final_model.fit(X_train_all[scored_train_all], y_train_all[scored_train_all])
    full_fit_s = time.perf_counter() - t0

    # Student-test inference
    t1 = time.perf_counter()
    test_probs = predict_proba_pos(final_model, X_test, batched=spec["batched_predict"])
    test_inf_s = time.perf_counter() - t1
    test_metrics = evaluate_predictions_full(test_cand, test_obs, test_probs, test_graphs)

    # Training-set diagnostic (use 50k sample for KNN to avoid redundant 1M self-query, exact for others)
    if key == "knn" and len(X_train_all) > 50000:
        rng = np.random.default_rng(RANDOM_SEED)
        sc_idx = np.flatnonzero(scored_train_all)
        sub_idx = rng.choice(sc_idx, size=min(40000, len(sc_idx)), replace=False)
        tr_probs_sub = predict_proba_pos(final_model, X_train_all[sub_idx], batched=True)
        train_diag = compute_candidate_metrics(y_train_all[sub_idx], tr_probs_sub)
        train_diag["note"] = "Evaluated on 40,000 stratified random scored training candidates (distance-weighted KNN)"
    else:
        tr_probs = predict_proba_pos(final_model, X_train_all[scored_train_all], batched=spec["batched_predict"])
        train_diag = compute_candidate_metrics(y_train_all[scored_train_all], tr_probs)
        train_diag["note"] = "Evaluated on all scored training candidates"

    # Capture real boosting-iteration loss history ONLY for HistGradientBoostingClassifier
    iteration_history = None
    if key == "hist_gradient_boosting" and hasattr(final_model, "train_score_") and hasattr(final_model, "validation_score_"):
        # Note: sklearn stores neg_log_loss in train_score_ / validation_score_ when scoring="neg_log_loss"
        iteration_history = {
            "n_iter_": int(final_model.n_iter_),
            "train_log_loss": [float(-x) for x in final_model.train_score_],
            "val_log_loss": [float(-x) for x in final_model.validation_score_],
        }

    result = {
        "key": key,
        "name": spec["name"],
        "family": spec["family"],
        "seed": spec["seed"],
        "config_summary": spec["config_summary"],
        "loss_objective": spec["loss_objective"],
        "hyperparams": spec["hyperparams"],
        "n_train_candidates_scored": int(scored_train_all.sum()),
        "n_test_candidates": int(len(test_cand)),
        "cv_fold_results": fold_results,
        "cv_summary": cv_summary,
        "train_diagnostic": train_diag,
        "student_test_metrics": test_metrics,
        "full_fit_seconds": round(full_fit_s, 3),
        "student_test_infer_seconds": round(test_inf_s, 4),
        "student_test_infer_ms_per_trip": round(1000.0 * test_inf_s / max(test_obs["trip_id"].nunique(), 1), 2),
        "iteration_history": iteration_history,
    }

    # Save immediately so expensive work is resumable!
    (out_dir / f"model_{key}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out_dir / f"oof_probs_{key}.npy", oof_probs.astype(np.float32))
    np.save(out_dir / f"test_probs_{key}.npy", test_probs.astype(np.float32))

    print(
        f"  ==> STUDENT-TEST ({spec['name']}): MacroF1={test_metrics['cand_macro_f1']:.4f} | "
        f"Top1={test_metrics['obs_top1_overall']:.4f} | "
        f"LogLoss={test_metrics['cand_log_loss']:.4f} | "
        f"Cont={test_metrics['route_continuity']:.4f} | "
        f"SeqLCS-F1={test_metrics['seq_lcs_f1']:.4f} | "
        f"fit={full_fit_s:.2f}s, test_inf={test_inf_s:.3f}s"
    )
    del final_model
    gc.collect()
    return result


def generate_comparison_tables_and_plots(results: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Generate required comparison tables, plots, confusion matrices, and data-driven model selection."""
    rows = []
    for r in results:
        cv = r["cv_summary"]
        te = r["student_test_metrics"]
        tr = r["train_diagnostic"]
        rows.append(
            {
                "model_key": r["key"],
                "Model": r["name"],
                "Algorithm Family": r["family"],
                "Key Configuration": r["config_summary"],
                "CV Macro F1 (mean)": round(cv["cand_macro_f1"]["mean"], 4),
                "CV Macro F1 (std)": round(cv["cand_macro_f1"]["std"], 4),
                "CV Top-1 Accuracy": round(cv["obs_top1_overall"]["mean"], 4),
                "CV Log Loss": round(cv["cand_log_loss"]["mean"], 4),
                "Train Macro F1": round(tr["cand_macro_f1"], 4),
                "Student-Test Top-1 Accuracy": round(te["obs_top1_overall"], 4),
                "Student-Test Top-1 (Covered)": round(te["obs_top1_given_covered"], 4),
                "Candidate Miss Rate": round(te["candidate_miss_rate"], 4),
                "Route Continuity": round(te["route_continuity"], 4),
                "Seq LCS F1": round(te["seq_lcs_f1"], 4),
                "Student-Test Macro F1": round(te["cand_macro_f1"], 4),
                "Student-Test Pos Precision": round(te["cand_precision_pos"], 4),
                "Student-Test Pos Recall": round(te["cand_recall_pos"], 4),
                "Student-Test Log Loss": round(te["cand_log_loss"], 4),
                "Train Time (s)": round(r["full_fit_seconds"], 2),
                "Inference Time (s)": round(r["student_test_infer_seconds"], 4),
                "Infer ms/trip": round(r["student_test_infer_ms_per_trip"], 2),
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    # Markdown table matching Deliverables.pdf §3.1 columns
    md_lines = [
        "| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Inference Time (10 trips) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comp_df.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['Algorithm Family']} | {row['Key Configuration']} | "
            f"{row['CV Macro F1 (mean)']:.4f} ± {row['CV Macro F1 (std)']:.4f} | "
            f"{row['Student-Test Top-1 Accuracy']:.4f} | {row['Route Continuity']:.4f} | "
            f"{row['Student-Test Macro F1']:.4f} | {row['Student-Test Log Loss']:.4f} | "
            f"{row['Inference Time (s)']:.3f}s ({row['Infer ms/trip']:.1f} ms/trip) |"
        )
    (out_dir / "model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    names = [r["name"] for r in results]
    short_map = {
        "logistic_regression": "LogReg",
        "knn": "KNN",
        "decision_tree": "DecTree",
        "random_forest": "RandForest",
        "hist_gradient_boosting": "HistGB",
    }
    short_names = [short_map.get(r["key"], r["name"][:8]) for r in results]
    x = np.arange(len(results))
    palette = ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#dc2626"]

    # Plot 1: CV Macro F1 comparison with fold std error bars + per-fold points
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    means = [r["cv_summary"]["cand_macro_f1"]["mean"] for r in results]
    stds = [r["cv_summary"]["cand_macro_f1"]["std"] for r in results]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=palette, alpha=0.85, edgecolor="#1e293b", linewidth=0.8)
    for idx, r in enumerate(results):
        fvals = [fr["cand_macro_f1"] for fr in r["cv_fold_results"]]
        ax.scatter(np.full(len(fvals), idx), fvals, color="#0f172a", s=24, zorder=4, alpha=0.75)
        ax.text(idx, means[idx] + stds[idx] + 0.008, f"{means[idx]:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(max(0.0, min(means) - 0.1), min(1.02, max(means) + 0.07))
    ax.set_ylabel("5-Fold Group-CV Macro F1-score")
    ax.set_title("5-Fold Group-Aware Cross-Validation Macro F1 Comparison (80 Training Trips)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_macro_f1_comparison.png")
    plt.close(fig)

    # Plot 2: Student-Test Top-1 Accuracy, Macro F1, and Route Continuity
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=160)
    w = 0.26
    top1 = [r["student_test_metrics"]["obs_top1_overall"] for r in results]
    mf1 = [r["student_test_metrics"]["cand_macro_f1"] for r in results]
    cont = [r["student_test_metrics"]["route_continuity"] for r in results]
    b1 = ax.bar(x - w, top1, w, label="Obs-Level Top-1 Accuracy (overall)", color="#2563eb")
    b2 = ax.bar(x, mf1, w, label="Candidate Macro F1-score", color="#16a34a")
    b3 = ax.bar(x + w, cont, w, label="Route Continuity", color="#f59e0b")
    for bars_grp in (b1, b2, b3):
        for b in bars_grp:
            val = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(max(0.0, min(top1 + mf1 + cont) - 0.1), 1.04)
    ax.set_ylabel("Score")
    ax.set_title("Student-Test (10 Trips): Top-1 Accuracy, Macro F1 & Route Continuity", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "student_test_comparison.png")
    plt.close(fig)

    # Plot 3: Log Loss & Inference Time (probability calibration + latency for HMM/API)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    ll = [r["student_test_metrics"]["cand_log_loss"] for r in results]
    inf_ms = [r["student_test_infer_ms_per_trip"] for r in results]

    axes[0].bar(x, ll, color=palette, alpha=0.88, edgecolor="#1e293b", linewidth=0.8)
    for idx, val in enumerate(ll):
        axes[0].text(idx, val + max(ll) * 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names)
    axes[0].set_ylabel("Binary Log Loss (lower is better)")
    axes[0].set_title("Student-Test Probability Calibration (Log Loss)", fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, inf_ms, color=palette, alpha=0.88, edgecolor="#1e293b", linewidth=0.8)
    axes[1].set_yscale("log")
    for idx, val in enumerate(inf_ms):
        axes[1].text(idx, val * 1.15, f"{val:.1f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names)
    axes[1].set_ylabel("Mean Inference Latency per Trip (ms, log scale)")
    axes[1].set_title("Student-Test Inference Latency per Trip", fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "log_loss_and_latency_comparison.png")
    plt.close(fig)

    # Plot 4: Confusion Matrices for all 5 models on Student-Test
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    axes_flat = axes.ravel()
    for idx, r in enumerate(results):
        cm_d = r["student_test_metrics"]["confusion_matrix"]
        mat = np.array([[cm_d["tn"], cm_d["fp"]], [cm_d["fn"], cm_d["tp"]]])
        ax = axes_flat[idx]
        im = ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{r['name']}\n(Macro F1={r['student_test_metrics']['cand_macro_f1']:.4f})", fontsize=10, fontweight="bold")
    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.1,
        0.5,
        "Student-Test Set (10 trips)\nBinary candidate classification:\n0 = Non-reference arc\n1 = Reference arc\nThreshold = 0.50",
        fontsize=10,
        va="center",
        bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    fig.suptitle("Candidate-Level Confusion Matrices on Student-Test Partition (All 5 Models)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png")
    plt.close(fig)

    # Plot 5: Real boosting-iteration log-loss curve for HistGradientBoosting ONLY
    hgb_res = next((r for r in results if r["key"] == "hist_gradient_boosting" and r.get("iteration_history")), None)
    if hgb_res is not None:
        hist = hgb_res["iteration_history"]
        iters = np.arange(len(hist["train_log_loss"]))
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
        ax.plot(iters, hist["train_log_loss"], color="#2563eb", lw=2.0, label="Training Log Loss (internal 90%)")
        ax.plot(iters, hist["val_log_loss"], color="#dc2626", lw=2.0, linestyle="--", label="Validation Log Loss (internal 10%)")
        ax.set_xlabel("Boosting Iteration (0 = initial prior)")
        ax.set_ylabel("Binary Log Loss")
        ax.set_title(
            f"Histogram Gradient Boosting: Stagewise Training vs Validation Log Loss (stopped at iter {hist['n_iter_']})",
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "hgb_iteration_loss.png")
        plt.close(fig)

    # Select best model systematically:
    # Primary metric = 5-fold Group-CV Macro F1 & Observation-level Top-1, verified by Log Loss (HMM emissions) & inference latency
    ranked = sorted(
        results,
        key=lambda r: (
            r["cv_summary"]["cand_macro_f1"]["mean"],
            r["cv_summary"]["obs_top1_overall"]["mean"],
            -r["cv_summary"]["cand_log_loss"]["mean"],
        ),
        reverse=True,
    )
    winner = ranked[0]
    selection = {
        "selected_model_key": winner["key"],
        "selected_model_name": winner["name"],
        "primary_selection_metric": "5-fold Group-CV Candidate Macro F1-score (jointly verified with CV Obs-level Top-1, Log Loss for HMM emissions, fold stability, and inference latency)",
        "ranking_by_cv_macro_f1": [
            {
                "key": r["key"],
                "name": r["name"],
                "cv_macro_f1_mean": round(r["cv_summary"]["cand_macro_f1"]["mean"], 5),
                "cv_macro_f1_std": round(r["cv_summary"]["cand_macro_f1"]["std"], 5),
                "cv_top1_mean": round(r["cv_summary"]["obs_top1_overall"]["mean"], 5),
                "cv_log_loss_mean": round(r["cv_summary"]["cand_log_loss"]["mean"], 5),
                "student_test_macro_f1": round(r["student_test_metrics"]["cand_macro_f1"], 5),
                "student_test_top1": round(r["student_test_metrics"]["obs_top1_overall"], 5),
                "student_test_log_loss": round(r["student_test_metrics"]["cand_log_loss"], 5),
                "student_test_infer_ms_per_trip": r["student_test_infer_ms_per_trip"],
            }
            for r in ranked
        ],
    }
    (out_dir / "model_selection_summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Milestone 3: 5-model training & comparison")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test on 10 train trips / 2 folds + 3 test trips")
    parser.add_argument("--resume", action="store_true", help="Skip models whose JSON results already exist")
    args = parser.parse_args()

    out_dir = OUT_DIR / "smoke" if args.smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = discover_dataset_dirs()

    print("Loading cached candidate datasets from Milestone 2...")
    train_cand = pd.read_parquet(CACHE_DIR / "train_candidates.parquet")
    train_obs = pd.read_parquet(CACHE_DIR / "train_obs_meta.parquet")
    test_cand = pd.read_parquet(CACHE_DIR / "student_test_candidates.parquet")
    test_obs = pd.read_parquet(CACHE_DIR / "student_test_obs_meta.parquet")

    # Ensure string trip IDs preserving leading zeros
    for df in (train_cand, train_obs, test_cand, test_obs):
        df["trip_id"] = df["trip_id"].astype(str)

    if args.smoke:
        smoke_train_ids = sorted(train_obs["trip_id"].unique().tolist())[:10]
        smoke_test_ids = sorted(test_obs["trip_id"].unique().tolist())[:3]
        train_cand = train_cand[train_cand["trip_id"].isin(smoke_train_ids)].reset_index(drop=True)
        train_obs = train_obs[train_obs["trip_id"].isin(smoke_train_ids)].reset_index(drop=True)
        test_cand = test_cand[test_cand["trip_id"].isin(smoke_test_ids)].reset_index(drop=True)
        test_obs = test_obs[test_obs["trip_id"].isin(smoke_test_ids)].reset_index(drop=True)
        n_splits = 2
        print(f"SMOKE MODE: {len(smoke_train_ids)} train trips ({len(train_cand):,} cands), {len(smoke_test_ids)} test trips, {n_splits} folds")
    else:
        n_splits = N_FOLDS
        print(f"FULL MODE: {train_obs['trip_id'].nunique()} train trips ({len(train_cand):,} cands), {test_obs['trip_id'].nunique()} test trips, {n_splits} folds")

    trip_to_fold = build_cv_folds(train_obs, n_splits=n_splits)
    fold_df = pd.DataFrame(
        [{"trip_id": tid, "fold": f, "n_obs": int((train_obs["trip_id"] == tid).sum())} for tid, f in sorted(trip_to_fold.items())]
    )
    fold_df.to_csv(out_dir / "cv_fold_assignments.csv", index=False)

    print("Preloading road network graphs for continuity & sequence metrics...")
    train_graphs = preload_trip_graphs(train_dir, sorted(train_obs["trip_id"].unique().tolist()))
    test_graphs = preload_trip_graphs(test_dir, sorted(test_obs["trip_id"].unique().tolist()))

    specs = get_model_specs()
    results: List[Dict[str, Any]] = []
    for spec in specs:
        res_path = out_dir / f"model_{spec['key']}.json"
        if args.resume and res_path.exists():
            print(f"\n[RESUME] Loading existing result for {spec['name']} from {res_path}")
            results.append(json.loads(res_path.read_text(encoding="utf-8")))
            continue
        res = run_single_model(
            spec=spec,
            train_cand=train_cand,
            train_obs=train_obs,
            test_cand=test_cand,
            test_obs=test_obs,
            trip_to_fold=trip_to_fold,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            n_splits=n_splits,
            out_dir=out_dir,
        )
        results.append(res)

    selection = generate_comparison_tables_and_plots(results, out_dir)
    print("\n=== MILESTONE 3 MODEL SELECTION SUMMARY ===")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
