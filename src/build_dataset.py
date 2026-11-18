"""Milestone 2: Build cached candidate datasets, alignment labels, nearest-road baseline, and worked example."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alignment import ALIGNMENT_CONFIG, align_track_to_route
from src.candidates import CANDIDATE_CONFIG, FEATURE_COLUMNS, generate_trip_candidates
from src.data_io import (
    PROJECT_ROOT,
    discover_dataset_dirs,
    list_trip_ids,
    load_trip,
    project_trip_geometry,
)
from src.metrics import (
    compute_observation_top1_and_miss,
    compute_route_continuity_for_trip,
    compute_sequence_metrics_for_trip,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "milestone2"
CACHE_DIR = OUT_DIR / "cache"


def verify_route_free_inference(train_dir: Path, trip_id: str = "00000005") -> Dict[str, Any]:
    """Explicit verification that candidate & feature generation succeeds with route=None."""
    trip_no_route = load_trip(train_dir, trip_id, load_route=False)
    assert trip_no_route.route is None, "Expected route=None when load_route=False!"
    cand_df = generate_trip_candidates(trip_no_route)
    assert len(cand_df) > 0, "No candidates generated!"
    assert list(cand_df.columns) == ["trip_id", "obs_id", "arc_id"] + FEATURE_COLUMNS
    assert not cand_df[FEATURE_COLUMNS].isna().any().any(), "NaN detected in inference features!"
    assert np.isfinite(cand_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all(), "Non-finite value in features!"
    return {
        "trip_id": trip_id,
        "route_is_none": True,
        "n_candidate_rows": int(len(cand_df)),
        "n_features": len(FEATURE_COLUMNS),
        "all_finite": True,
    }


def build_split_dataset(
    split_name: str,
    split_dir: Path,
    trip_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build candidate feature table and observation alignment metadata for one split."""
    t0 = time.perf_counter()
    cand_parts: List[pd.DataFrame] = []
    obs_parts: List[pd.DataFrame] = []
    per_trip_stats: List[Dict[str, Any]] = []

    for idx, tid in enumerate(trip_ids):
        trip = load_trip(split_dir, tid, load_route=True)
        geom = project_trip_geometry(trip)

        # Step A: Inference-only candidates & features (never uses trip.route)
        cands = generate_trip_candidates(trip, geom=geom)

        # Step B: Development-only monotonic alignment to .route
        aligned = align_track_to_route(trip, geom=geom)

        # Join labels onto candidates (strictly preserving trip_id, obs_id, arc_id)
        ref_map = aligned[["obs_id", "ref_arc_id", "is_scored"]].copy()
        cands = cands.merge(ref_map, on="obs_id", how="left")
        cands["is_target"] = (cands["is_scored"] & (cands["arc_id"] == cands["ref_arc_id"])).astype(np.int8)

        # Check candidate coverage per observation
        covered_obs = set(cands.loc[cands["is_target"] == 1, "obs_id"].tolist())
        aligned["candidate_has_ref"] = aligned["obs_id"].isin(covered_obs)

        # Nearest-road baseline metrics for this trip (dist_rank == 1)
        nearest_cands = cands.loc[cands["dist_rank"] == 1, ["obs_id", "arc_id"]].set_index("obs_id")["arc_id"]
        pred_arcs = aligned["obs_id"].map(nearest_cands).fillna(-1).to_numpy(dtype=np.int64)
        cont = compute_route_continuity_for_trip(
            pred_arcs,
            trip.arcs["from_node_id"].to_numpy(dtype=np.int64),
            trip.arcs["to_node_id"].to_numpy(dtype=np.int64),
        )
        seq = compute_sequence_metrics_for_trip(pred_arcs, trip.route, geom["arc_len"])

        scored_mask = aligned["is_scored"].to_numpy()
        n_scored = int(np.sum(scored_mask))
        n_cov = int(np.sum(aligned["candidate_has_ref"].to_numpy() & scored_mask))
        n_nearest_ok = int(np.sum((pred_arcs == aligned["ref_arc_id"].to_numpy()) & scored_mask))

        per_trip_stats.append(
            {
                "split": split_name,
                "trip_id": tid,
                "n_obs": len(aligned),
                "n_scored": n_scored,
                "n_unscored": int(len(aligned) - n_scored),
                "label_coverage_pct": round(100.0 * n_scored / max(len(aligned), 1), 2),
                "n_junction_near_tie": int(aligned["is_junction_near_tie"].sum()),
                "n_candidates": len(cands),
                "avg_cands_per_obs": round(len(cands) / max(len(aligned), 1), 2),
                "candidate_miss_rate": round(1.0 - n_cov / max(n_scored, 1), 4),
                "nearest_top1_overall": round(n_nearest_ok / max(n_scored, 1), 4),
                "nearest_route_continuity": round(cont["route_continuity"], 4),
                "nearest_seq_lcs_f1": round(seq["seq_lcs_f1"], 4),
                "mean_aligned_dist_m": round(float(aligned.loc[aligned["is_scored"], "aligned_dist_m"].mean()), 2)
                if n_scored > 0
                else np.nan,
            }
        )

        # Keep only compact columns for parquet cache
        cands = cands.drop(columns=["ref_arc_id"])
        cand_parts.append(cands)
        obs_parts.append(aligned)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(trip_ids):
            print(f"[{split_name}] Processed {idx + 1}/{len(trip_ids)} trips ({time.perf_counter() - t0:.1f}s)")

    cand_all = pd.concat(cand_parts, ignore_index=True)
    obs_all = pd.concat(obs_parts, ignore_index=True)
    trip_stats_df = pd.DataFrame(per_trip_stats)

    # Aggregate nearest-road baseline using negative dist_m as score
    obs_metrics, _ = compute_observation_top1_and_miss(cand_all, -cand_all["dist_m"].to_numpy(), obs_all)

    summary = {
        "split": split_name,
        "n_trips": len(trip_ids),
        "n_total_obs": int(len(obs_all)),
        "n_scored_obs": int(obs_all["is_scored"].sum()),
        "n_unscored_obs": int((~obs_all["is_scored"]).sum()),
        "label_coverage_pct": round(100.0 * float(obs_all["is_scored"].mean()), 3),
        "n_junction_near_tie": int(obs_all["is_junction_near_tie"].sum()),
        "junction_near_tie_pct": round(100.0 * float(obs_all["is_junction_near_tie"].mean()), 3),
        "n_candidate_rows": int(len(cand_all)),
        "avg_candidates_per_obs": round(float(len(cand_all) / max(len(obs_all), 1)), 3),
        "positive_rate_in_scored_cands": round(
            float(cand_all.loc[cand_all["is_scored"], "is_target"].mean()), 4
        ),
        "median_aligned_dist_m_scored": round(
            float(obs_all.loc[obs_all["is_scored"], "aligned_dist_m"].median()), 3
        ),
        "p95_aligned_dist_m_scored": round(
            float(obs_all.loc[obs_all["is_scored"], "aligned_dist_m"].quantile(0.95)), 3
        ),
        "candidate_miss_rate": round(obs_metrics["candidate_miss_rate"], 5),
        "nearest_road_top1_overall": round(obs_metrics["obs_top1_overall"], 5),
        "nearest_road_top1_given_covered": round(obs_metrics["obs_top1_given_covered"], 5),
        "nearest_road_mean_continuity": round(float(trip_stats_df["nearest_route_continuity"].mean()), 5),
        "nearest_road_mean_seq_lcs_f1": round(float(trip_stats_df["nearest_seq_lcs_f1"].mean()), 5),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    return cand_all, obs_all, summary, trip_stats_df


def save_worked_candidate_example(
    train_cand: pd.DataFrame,
    train_obs: pd.DataFrame,
    trip_id: str = "00000005",
) -> Dict[str, Any]:
    """Save a worked example from trip 00000005 showing candidates, features, and reference assignment.

    Selects one moving multi-candidate observation (at an intersection / bidirectional road where
    nearest distance alone may be ambiguous or close) and one stationary observation.
    """
    t_cands = train_cand[train_cand["trip_id"] == trip_id].copy()
    t_obs = train_obs[train_obs["trip_id"] == trip_id].copy()

    # Pick an interesting moving observation where >= 4 candidates exist and reference is covered
    cand_counts = t_cands.groupby("obs_id").size()
    rich_obs = set(cand_counts[cand_counts >= 4].index.tolist())
    # Prefer an observation where heading distinguishes the true directed arc from its reverse!
    moving_cands = t_cands[
        (t_cands["obs_id"].isin(rich_obs))
        & (t_cands["is_target"] == 1)
        & (t_cands["has_reverse_arc"] == 1.0)
        & (t_cands["heading_valid"] == 1.0)
    ]
    if len(moving_cands) > 0:
        # Prefer one where dist_rank >= 2 if available (demonstrates why distance alone fails!), else first
        rank2 = moving_cands[moving_cands["dist_rank"] >= 2]
        chosen_obs = int(rank2.iloc[0]["obs_id"]) if len(rank2) > 0 else int(moving_cands.iloc[0]["obs_id"])
    else:
        chosen_obs = int(t_cands.loc[t_cands["is_target"] == 1, "obs_id"].iloc[0])

    example_rows = t_cands[t_cands["obs_id"] == chosen_obs].copy()
    obs_info = t_obs[t_obs["obs_id"] == chosen_obs].iloc[0].to_dict()

    # Also find a stationary observation example if present
    stat_obs_ids = t_cands.loc[(t_cands["is_stationary"] == 1.0) & (t_cands["is_target"] == 1), "obs_id"].unique()
    stat_example = None
    if len(stat_obs_ids) > 0:
        soid = int(stat_obs_ids[0])
        stat_example = {
            "obs_id": soid,
            "ref_arc_id": int(t_obs.loc[t_obs["obs_id"] == soid, "ref_arc_id"].iloc[0]),
            "candidates": t_cands[t_cands["obs_id"] == soid].to_dict(orient="records"),
        }

    example_csv = OUT_DIR / "worked_candidate_example_00000005.csv"
    example_rows.to_csv(example_csv, index=False)

    payload = {
        "trip_id": trip_id,
        "moving_example_obs_id": chosen_obs,
        "alignment_record": {k: (v.item() if hasattr(v, "item") else v) for k, v in obs_info.items()},
        "n_candidates": int(len(example_rows)),
        "candidates": [
            {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v))
             for k, v in row.items()}
            for row in example_rows.to_dict(orient="records")
        ],
        "stationary_example": stat_example,
        "csv_path": str(example_csv),
    }
    (OUT_DIR / "worked_candidate_example_00000005.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def plot_alignment_diagnostics(
    train_dir: Path,
    train_obs: pd.DataFrame,
    test_obs: pd.DataFrame,
    trip_stats: pd.DataFrame,
) -> None:
    """Visual inspection of alignment quality across splits and representative trips (incl. repeated route & gaps)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)

    # 1. Distribution of aligned distance (m) for scored observations
    tr_d = train_obs.loc[train_obs["is_scored"], "aligned_dist_m"].to_numpy()
    te_d = test_obs.loc[test_obs["is_scored"], "aligned_dist_m"].to_numpy()
    axes[0, 0].hist(tr_d, bins=40, range=(0, 35), density=True, alpha=0.65, color="#2563eb", label=f"Train (med={np.median(tr_d):.2f}m)")
    axes[0, 0].hist(te_d, bins=40, range=(0, 35), density=True, alpha=0.65, color="#f59e0b", label=f"Student Test (med={np.median(te_d):.2f}m)")
    axes[0, 0].set_title("Point-to-Aligned-Reference-Arc Distance Distribution")
    axes[0, 0].set_xlabel("Distance to Aligned Reference Arc (m)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(True, alpha=0.25)

    # 2. Monotonic alignment progression on worked trip 00000005 (obs_id vs aligned_route_idx)
    o5 = train_obs[train_obs["trip_id"] == "00000005"]
    axes[0, 1].plot(o5["obs_id"], o5["aligned_route_idx"], color="#16a34a", lw=2.0, label="Aligned Route Step j*(t)")
    sc_mask = o5["is_scored"].to_numpy()
    if np.any(~sc_mask):
        axes[0, 1].scatter(o5.loc[~sc_mask, "obs_id"], o5.loc[~sc_mask, "aligned_route_idx"], c="#dc2626", s=16, label="Unscored (>50m)")
    axes[0, 1].set_title("Trip 00000005: Monotonic Route Alignment (Stops = Flat Plateaus)")
    axes[0, 1].set_xlabel("GPS Observation Index (obs_id)")
    axes[0, 1].set_ylabel("Reference Route Index j (0..104)")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.25)

    # 3. Monotonic alignment on a trip with repeated segments or large GPS gap (trip 00000078 or 00000075)
    gap_tid = "00000075" if "00000075" in set(train_obs["trip_id"].unique()) else train_obs["trip_id"].iloc[0]
    og = train_obs[train_obs["trip_id"] == gap_tid]
    axes[1, 0].plot(og["obs_id"], og["aligned_route_idx"], color="#7c3aed", lw=1.8, label=f"Trip {gap_tid} Aligned Route Step")
    sc_g = og["is_scored"].to_numpy()
    if np.any(~sc_g):
        axes[1, 0].scatter(og.loc[~sc_g, "obs_id"], og.loc[~sc_g, "aligned_route_idx"], c="#dc2626", s=16, label="Unscored (>50m)")
    axes[1, 0].set_title(f"Trip {gap_tid}: Alignment with Repeated Arcs & Route Discontinuity")
    axes[1, 0].set_xlabel("GPS Observation Index (obs_id)")
    axes[1, 0].set_ylabel("Reference Route Index j")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, alpha=0.25)

    # 4. Per-trip Nearest-Road Baseline Top-1 vs Candidate Miss Rate
    is_te = trip_stats["split"] == "student_test"
    axes[1, 1].scatter(
        100 * trip_stats.loc[~is_te, "candidate_miss_rate"],
        100 * trip_stats.loc[~is_te, "nearest_top1_overall"],
        alpha=0.75,
        color="#2563eb",
        label="Train trips (80)",
    )
    axes[1, 1].scatter(
        100 * trip_stats.loc[is_te, "candidate_miss_rate"],
        100 * trip_stats.loc[is_te, "nearest_top1_overall"],
        alpha=0.95,
        color="#f59e0b",
        marker="^",
        s=65,
        label="Student Test trips (10)",
    )
    axes[1, 1].set_title("Nearest-Road Baseline Top-1 Accuracy vs Candidate Miss Rate")
    axes[1, 1].set_xlabel("Candidate Miss Rate (%)")
    axes[1, 1].set_ylabel("Nearest-Road Top-1 Accuracy (%)")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle("Milestone 2: Monotonic Reference Alignment & Candidate Baseline Diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alignment_and_baseline_diagnostics.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = discover_dataset_dirs()
    train_ids = list_trip_ids(train_dir)
    test_ids = list_trip_ids(test_dir)

    print("Step 1: Verifying route-free inference feature generation...")
    route_free_check = verify_route_free_inference(train_dir, "00000005")
    print("Route-free verification PASSED:", route_free_check)

    print("\nStep 2: Building candidate features & monotonic alignment labels for train (80) and student_test (10)...")
    train_cand, train_obs, train_summary, train_trip_stats = build_split_dataset("train", train_dir, train_ids)
    test_cand, test_obs, test_summary, test_trip_stats = build_split_dataset("student_test", test_dir, test_ids)

    # Save parquet caches for fast Milestone 3 & 4 reuse
    train_cand.to_parquet(CACHE_DIR / "train_candidates.parquet", index=False)
    train_obs.to_parquet(CACHE_DIR / "train_obs_meta.parquet", index=False)
    test_cand.to_parquet(CACHE_DIR / "student_test_candidates.parquet", index=False)
    test_obs.to_parquet(CACHE_DIR / "student_test_obs_meta.parquet", index=False)

    trip_stats_all = pd.concat([train_trip_stats, test_trip_stats], ignore_index=True)
    trip_stats_all.to_csv(OUT_DIR / "per_trip_alignment_and_baseline.csv", index=False)

    worked = save_worked_candidate_example(train_cand, train_obs, "00000005")
    plot_alignment_diagnostics(train_dir, train_obs, test_obs, trip_stats_all)

    report = {
        "candidate_config": CANDIDATE_CONFIG,
        "alignment_config": ALIGNMENT_CONFIG,
        "feature_columns": FEATURE_COLUMNS,
        "route_free_verification": route_free_check,
        "train_summary": train_summary,
        "student_test_summary": test_summary,
        "worked_example_summary": {
            "trip_id": worked["trip_id"],
            "moving_example_obs_id": worked["moving_example_obs_id"],
            "n_candidates": worked["n_candidates"],
            "ref_arc_id": worked["alignment_record"]["ref_arc_id"],
            "aligned_dist_m": round(float(worked["alignment_record"]["aligned_dist_m"]), 3),
        },
    }
    (OUT_DIR / "milestone2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== MILESTONE 2 SUMMARY ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
