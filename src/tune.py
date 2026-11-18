"""Milestone 4: Bayesian hyperparameter optimisation (Optuna TPE), refit, comparison, and artifact export."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.candidates import CANDIDATE_CONFIG, FEATURE_COLUMNS
from src.data_io import PROJECT_ROOT, discover_dataset_dirs
from src.metrics import compute_candidate_metrics, compute_observation_top1_and_miss
from src.models import (
    CACHE_DIR,
    RANDOM_SEED,
    evaluate_predictions_full,
    preload_trip_graphs,
)

M3_DIR = PROJECT_ROOT / "outputs" / "milestone3"
OUT_DIR = PROJECT_ROOT / "outputs" / "milestone4"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_search_space_doc(model_key: str) -> Dict[str, Any]:
    if model_key == "hist_gradient_boosting":
        return {
            "model_key": model_key,
            "model_name": "Histogram Gradient Boosting",
            "sampler": "Optuna TPESampler (Bayesian Tree-structured Parzen Estimator, seed=42)",
            "objective": "Maximize 5-fold Group-CV Candidate Macro F1-score on the 80 training trips",
            "parameters": {
                "learning_rate": {"type": "log_uniform", "low": 0.02, "high": 0.25, "rationale": "Controls shrinkage per boosting stage vs generalization."},
                "max_leaf_nodes": {"type": "int", "low": 15, "high": 127, "rationale": "Controls interaction depth and tree complexity."},
                "min_samples_leaf": {"type": "int", "low": 20, "high": 200, "rationale": "Regularizes leaf support to prevent overfitting to specific trips."},
                "l2_regularization": {"type": "log_uniform", "low": 1e-3, "high": 20.0, "rationale": "Shrinks leaf weights and improves probability calibration (log-loss)."},
                "max_bins": {"type": "categorical", "choices": [128, 255], "rationale": "Histogram discretization resolution for distance/heading thresholds."},
                "max_iter": {"type": "int", "low": 150, "high": 400, "step": 50, "rationale": "Upper bound on boosting rounds (with early stopping enabled)."},
            },
        }
    elif model_key == "random_forest":
        return {
            "model_key": model_key,
            "model_name": "Random Forest",
            "sampler": "Optuna TPESampler (Bayesian Tree-structured Parzen Estimator, seed=42)",
            "objective": "Maximize 5-fold Group-CV Candidate Macro F1-score on the 80 training trips",
            "parameters": {
                "n_estimators": {"type": "int", "low": 100, "high": 300, "step": 50},
                "max_depth": {"type": "int", "low": 10, "high": 26},
                "min_samples_leaf": {"type": "int", "low": 5, "high": 80},
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5]},
            },
        }
    else:
        raise ValueError(f"Unsupported selected model key for tuning: {model_key}")


def build_model_from_params(model_key: str, params: Dict[str, Any]) -> Any:
    if model_key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_bins=int(params["max_bins"]),
            max_iter=int(params["max_iter"]),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            scoring="neg_log_loss",
            random_state=RANDOM_SEED,
        )
    elif model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=4,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unsupported model key: {model_key}")


def suggest_params(trial: optuna.Trial, model_key: str) -> Dict[str, Any]:
    if model_key == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 20.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
            "max_iter": trial.suggest_int("max_iter", 150, 400, step=50),
        }
    elif model_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 26),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    raise ValueError(f"Unsupported model key: {model_key}")


def plot_tuning_results(trials_df: pd.DataFrame, untuned_cv_f1: float, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    tnums = trials_df["trial_number"].to_numpy()
    vals = trials_df["cv_macro_f1"].to_numpy()
    best_so_far = np.maximum.accumulate(vals)

    axes[0].plot(tnums, vals, "o-", color="#94a3b8", alpha=0.75, markersize=5, label="Trial CV Macro F1")
    axes[0].plot(tnums, best_so_far, "-", color="#2563eb", lw=2.5, label="Best-so-far CV Macro F1")
    axes[0].axhline(untuned_cv_f1, color="#dc2626", linestyle="--", lw=1.6, label=f"Untuned baseline ({untuned_cv_f1:.4f})")
    axes[0].set_xlabel("Bayesian Optimization Trial #")
    axes[0].set_ylabel("5-Fold Group-CV Macro F1")
    axes[0].set_title("Optuna TPE Optimization History", fontweight="bold")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    # Right: CV Macro F1 vs CV Log Loss tradeoff across trials
    ll = trials_df["cv_log_loss"].to_numpy()
    sc = axes[1].scatter(ll, vals, c=tnums, cmap="viridis", s=55, edgecolor="#1e293b", linewidth=0.5)
    best_idx = int(np.argmax(vals))
    axes[1].scatter([ll[best_idx]], [vals[best_idx]], color="#dc2626", marker="*", s=220, zorder=5, label=f"Best Trial #{int(tnums[best_idx])}")
    axes[1].set_xlabel("5-Fold Group-CV Log Loss (lower is better)")
    axes[1].set_ylabel("5-Fold Group-CV Macro F1 (higher is better)")
    axes[1].set_title("Trial Quality: Macro F1 vs Probability Calibration", fontweight="bold")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[1], label="Trial #")

    fig.tight_layout()
    fig.savefig(out_dir / "tuning_history.png")
    plt.close(fig)


def verify_fresh_subprocess_reload(bundle_path: Path, sample_trip_id: str, expected_probs: np.ndarray) -> Dict[str, Any]:
    """Spawn a brand-new Python process to reload artifacts and verify exact predictions without retraining."""
    check_code = f"""
import json, joblib, numpy as np
from pathlib import Path
from src.data_io import discover_dataset_dirs, load_trip
from src.candidates import generate_trip_candidates

bundle = joblib.load({str(bundle_path)!r})
_, test_dir = discover_dataset_dirs()
trip = load_trip(test_dir, {sample_trip_id!r}, load_route=False)
assert trip.route is None
cands = generate_trip_candidates(trip, cfg=bundle["candidate_config"])
X = cands[bundle["feature_columns"]].to_numpy(dtype=np.float32)
probs = bundle["model"].predict_proba(X)[:, 1].astype(np.float64)
print(json.dumps({{"n_cands": int(len(probs)), "mean_prob": float(probs.mean()), "first5": [round(float(x), 6) for x in probs[:5]]}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", check_code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(proc.stdout.strip())
    max_abs_diff = float(np.max(np.abs(np.array(parsed["first5"]) - np.round(expected_probs[:5], 6))))
    return {
        "subprocess_exit_code": proc.returncode,
        "sample_trip_id": sample_trip_id,
        "route_loaded": False,
        "n_cands": parsed["n_cands"],
        "mean_prob": parsed["mean_prob"],
        "first5_probs": parsed["first5"],
        "max_abs_diff_vs_inprocess_first5": max_abs_diff,
        "verified": max_abs_diff < 1e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Milestone 4: Bayesian Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna TPE trials")
    parser.add_argument("--model-key", type=str, default=None, help="Override selected model key from Milestone 3")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    sel = json.loads((M3_DIR / "model_selection_summary.json").read_text(encoding="utf-8"))
    model_key = args.model_key or sel["selected_model_key"]
    untuned_res = json.loads((M3_DIR / f"model_{model_key}.json").read_text(encoding="utf-8"))
    print(f"Tuning selected model: {untuned_res['name']} ({model_key}) with {args.trials} Optuna TPE trials")

    search_doc = get_search_space_doc(model_key)
    search_doc["n_trials"] = args.trials
    (OUT_DIR / "search_space.json").write_text(json.dumps(search_doc, indent=2), encoding="utf-8")

    train_dir, test_dir = discover_dataset_dirs()
    train_cand = pd.read_parquet(CACHE_DIR / "train_candidates.parquet")
    train_obs = pd.read_parquet(CACHE_DIR / "train_obs_meta.parquet")
    test_cand = pd.read_parquet(CACHE_DIR / "student_test_candidates.parquet")
    test_obs = pd.read_parquet(CACHE_DIR / "student_test_obs_meta.parquet")
    for df in (train_cand, train_obs, test_cand, test_obs):
        df["trip_id"] = df["trip_id"].astype(str)

    fold_df = pd.read_csv(M3_DIR / "cv_fold_assignments.csv", dtype={"trip_id": str})
    trip_to_fold = dict(zip(fold_df["trip_id"], fold_df["fold"].astype(int)))
    n_splits = int(fold_df["fold"].nunique())

    X_train_all = train_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_all = train_cand["is_target"].to_numpy(dtype=np.int8)
    scored_train_all = train_cand["is_scored"].to_numpy(dtype=bool)
    folds_cand = train_cand["trip_id"].map(trip_to_fold).to_numpy(dtype=np.int32)
    X_test = test_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Pre-split fold indices once for fast trials
    fold_slices = []
    for f in range(n_splits):
        tr_mask = (folds_cand != f) & scored_train_all
        va_mask = folds_cand == f
        va_scored_mask = va_mask & scored_train_all
        va_trips = sorted(train_cand.loc[va_mask, "trip_id"].unique().tolist())
        va_cand_sub = train_cand.loc[va_mask, ["trip_id", "obs_id", "arc_id", "is_target", "is_scored"]].reset_index(drop=True)
        va_obs_sub = train_obs[train_obs["trip_id"].isin(va_trips)].reset_index(drop=True)
        fold_slices.append((tr_mask, va_mask, va_scored_mask, va_cand_sub, va_obs_sub))

    trial_records: List[Dict[str, Any]] = []
    t_opt_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_key)
        t0 = time.perf_counter()
        f1s, lls, top1s = [], [], []
        for f, (tr_mask, va_mask, _, va_cand_sub, va_obs_sub) in enumerate(fold_slices):
            model = build_model_from_params(model_key, params)
            model.fit(X_train_all[tr_mask], y_train_all[tr_mask])
            va_probs = model.predict_proba(X_train_all[va_mask])[:, 1].astype(np.float64)
            sc_local = va_cand_sub["is_scored"].to_numpy(dtype=bool)
            cm = compute_candidate_metrics(va_cand_sub.loc[sc_local, "is_target"].to_numpy(), va_probs[sc_local])
            om, _ = compute_observation_top1_and_miss(va_cand_sub, va_probs, va_obs_sub)
            f1s.append(cm["cand_macro_f1"])
            lls.append(cm["cand_log_loss"])
            top1s.append(om["obs_top1_overall"])
            del model

        elapsed = time.perf_counter() - t0
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s, ddof=1))
        mean_ll = float(np.mean(lls))
        mean_top1 = float(np.mean(top1s))

        rec = {
            "trial_number": trial.number,
            "cv_macro_f1": round(mean_f1, 6),
            "cv_macro_f1_std": round(std_f1, 6),
            "cv_top1_overall": round(mean_top1, 6),
            "cv_log_loss": round(mean_ll, 6),
            "elapsed_seconds": round(elapsed, 2),
            **{f"param_{k}": v for k, v in params.items()},
        }
        trial_records.append(rec)
        # Save progressively after every trial
        pd.DataFrame(trial_records).to_csv(OUT_DIR / "optuna_trials.csv", index=False)
        print(
            f"  Trial {trial.number:02d}/{args.trials}: CV MacroF1={mean_f1:.5f}±{std_f1:.5f} | "
            f"Top1={mean_top1:.5f} | LogLoss={mean_ll:.5f} | {elapsed:.1f}s | {params}"
        )
        gc.collect()
        return mean_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=4),
        study_name=f"pathsnap_{model_key}_tpe",
    )
    # Seed trial 0 with the untuned baseline params so Bayesian search strictly explores from a known anchor!
    if model_key == "hist_gradient_boosting":
        study.enqueue_trial(
            {
                "learning_rate": 0.08,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
                "max_bins": 255,
                "max_iter": 200,
            }
        )
    study.optimize(objective, n_trials=args.trials)
    total_opt_s = time.perf_counter() - t_opt_start

    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)

    best_params = study.best_params
    best_trial_row = trials_df.loc[trials_df["trial_number"] == study.best_trial.number].iloc[0].to_dict()
    print(f"\nBest trial #{study.best_trial.number}: CV MacroF1={study.best_value:.5f}, params={best_params}")

    # Full 5-fold CV evaluation of best params (for complete fold-by-fold metrics & OOF probs for HMM tuning!)
    print("Running full 5-fold CV with best hyperparameters to record OOF predictions & full metrics...")
    train_graphs = preload_trip_graphs(train_dir, sorted(train_obs["trip_id"].unique().tolist()))
    test_graphs = preload_trip_graphs(test_dir, sorted(test_obs["trip_id"].unique().tolist()))

    oof_probs = np.zeros(len(train_cand), dtype=np.float64)
    tuned_fold_results = []
    for f in range(n_splits):
        tr_mask = (folds_cand != f) & scored_train_all
        va_mask = folds_cand == f
        va_trips = sorted(train_cand.loc[va_mask, "trip_id"].unique().tolist())
        m = build_model_from_params(model_key, best_params)
        m.fit(X_train_all[tr_mask], y_train_all[tr_mask])
        va_probs = m.predict_proba(X_train_all[va_mask])[:, 1].astype(np.float64)
        oof_probs[va_mask] = va_probs
        fm = evaluate_predictions_full(
            train_cand.loc[va_mask],
            train_obs[train_obs["trip_id"].isin(va_trips)],
            va_probs,
            train_graphs,
        )
        fm["fold"] = f
        tuned_fold_results.append(fm)
        del m
    np.save(OUT_DIR / "tuned_oof_probs.npy", oof_probs.astype(np.float32))

    # Refit final tuned model on all 80 scored training trips
    print("Refitting tuned model on all 80 scored training trips...")
    final_model = build_model_from_params(model_key, best_params)
    t0 = time.perf_counter()
    final_model.fit(X_train_all[scored_train_all], y_train_all[scored_train_all])
    refit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    test_probs = final_model.predict_proba(X_test)[:, 1].astype(np.float64)
    test_inf_s = time.perf_counter() - t1
    np.save(OUT_DIR / "tuned_test_probs.npy", test_probs.astype(np.float32))

    tuned_test_metrics = evaluate_predictions_full(test_cand, test_obs, test_probs, test_graphs)

    tuned_cv_macro_f1 = float(np.mean([r["cand_macro_f1"] for r in tuned_fold_results]))
    tuned_cv_macro_f1_std = float(np.std([r["cand_macro_f1"] for r in tuned_fold_results], ddof=1))
    tuned_cv_top1 = float(np.mean([r["obs_top1_overall"] for r in tuned_fold_results]))
    tuned_cv_ll = float(np.mean([r["cand_log_loss"] for r in tuned_fold_results]))

    # Untuned vs Tuned comparison table
    u_cv = untuned_res["cv_summary"]
    u_te = untuned_res["student_test_metrics"]
    comp_rows = [
        {
            "Variant": f"{untuned_res['name']} (Untuned)",
            "CV Macro F1": round(u_cv["cand_macro_f1"]["mean"], 5),
            "CV Macro F1 Std": round(u_cv["cand_macro_f1"]["std"], 5),
            "CV Top-1 Accuracy": round(u_cv["obs_top1_overall"]["mean"], 5),
            "CV Log Loss": round(u_cv["cand_log_loss"]["mean"], 5),
            "Student-Test Top-1 Accuracy": round(u_te["obs_top1_overall"], 5),
            "Student-Test Macro F1": round(u_te["cand_macro_f1"], 5),
            "Student-Test Log Loss": round(u_te["cand_log_loss"], 5),
            "Student-Test Route Continuity": round(u_te["route_continuity"], 5),
            "Student-Test Seq LCS F1": round(u_te["seq_lcs_f1"], 5),
            "Inference Time (10 trips, s)": round(untuned_res["student_test_infer_seconds"], 4),
        },
        {
            "Variant": f"{untuned_res['name']} (Bayesian Tuned)",
            "CV Macro F1": round(tuned_cv_macro_f1, 5),
            "CV Macro F1 Std": round(tuned_cv_macro_f1_std, 5),
            "CV Top-1 Accuracy": round(tuned_cv_top1, 5),
            "CV Log Loss": round(tuned_cv_ll, 5),
            "Student-Test Top-1 Accuracy": round(tuned_test_metrics["obs_top1_overall"], 5),
            "Student-Test Macro F1": round(tuned_test_metrics["cand_macro_f1"], 5),
            "Student-Test Log Loss": round(tuned_test_metrics["cand_log_loss"], 5),
            "Student-Test Route Continuity": round(tuned_test_metrics["route_continuity"], 5),
            "Student-Test Seq LCS F1": round(tuned_test_metrics["seq_lcs_f1"], 5),
            "Inference Time (10 trips, s)": round(test_inf_s, 4),
        },
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "tuned_vs_untuned_comparison.csv", index=False)

    plot_tuning_results(trials_df, u_cv["cand_macro_f1"]["mean"], OUT_DIR)

    # Export self-contained inference artifacts
    bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
    bundle = {
        "model_key": model_key,
        "model_name": untuned_res["name"],
        "model": final_model,
        "preprocessing": "None required (tree-based native histogram binning; missing/stationary values encoded explicitly in feature extraction)",
        "feature_columns": FEATURE_COLUMNS,
        "candidate_config": CANDIDATE_CONFIG,
        "best_params": best_params,
        "random_seed": RANDOM_SEED,
        "trained_on_trips": sorted(train_obs["trip_id"].unique().tolist()),
    }
    joblib.dump(bundle, bundle_path, compress=3)

    feature_spec = {
        "projection": "Trip-local Azimuthal Equidistant (+proj=aeqd) centered on trip median (lon, lat), WGS84 datum, units=meters",
        "coordinate_order_raw": "longitude, latitude",
        "candidate_config": CANDIDATE_CONFIG,
        "feature_columns": FEATURE_COLUMNS,
        "stationary_handling": "When speed < 1.0 m/s or max adjacent step < 1.5 m, is_stationary=1, heading_valid=0, heading_diff_deg=90.0, cos_heading_diff=0.0.",
        "leakage_guarantee": "All 21 features use exclusively .track, .nodes, and .arcs; .route is never read during inference.",
    }
    (ARTIFACTS_DIR / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")

    # Verify reload in a brand-new subprocess!
    sample_tid = sorted(test_obs["trip_id"].unique().tolist())[0]
    sample_expected = test_probs[test_cand["trip_id"] == sample_tid]
    reload_check = verify_fresh_subprocess_reload(bundle_path, sample_tid, sample_expected)
    assert reload_check["verified"], f"Subprocess reload verification FAILED: {reload_check}"
    print("Fresh subprocess reload verification PASSED:", reload_check)

    model_config = {
        "selected_model_key": model_key,
        "selected_model_name": untuned_res["name"],
        "best_params": best_params,
        "n_optuna_trials": args.trials,
        "optuna_best_trial_number": int(study.best_trial.number),
        "optuna_total_seconds": round(total_opt_s, 2),
        "refit_seconds": round(refit_s, 3),
        "artifact_path": str(bundle_path),
        "artifact_size_bytes": bundle_path.stat().st_size,
        "tuned_cv_macro_f1": round(tuned_cv_macro_f1, 5),
        "tuned_cv_top1": round(tuned_cv_top1, 5),
        "tuned_cv_log_loss": round(tuned_cv_ll, 5),
        "tuned_student_test_metrics": tuned_test_metrics,
        "reload_verification": reload_check,
    }
    (ARTIFACTS_DIR / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (OUT_DIR / "tuning_summary.json").write_text(
        json.dumps(
            {
                "search_space": search_doc,
                "best_trial": best_trial_row,
                "best_params": best_params,
                "total_opt_seconds": round(total_opt_s, 2),
                "refit_seconds": round(refit_s, 3),
                "comparison": comp_rows,
                "reload_verification": reload_check,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== MILESTONE 4 SUMMARY ===")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
