"""Milestone 5: HMM / Viterbi Route Reconstruction using calibrated classifier emissions & road connectivity."""

from __future__ import annotations

import heapq
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.candidates import CANDIDATE_CONFIG, FEATURE_COLUMNS
from src.data_io import PROJECT_ROOT, TripData, discover_dataset_dirs, load_trip, project_trip_geometry
from src.metrics import (
    compute_observation_top1_and_miss,
    compute_route_continuity_for_trip,
    compute_sequence_metrics_for_trip,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
M2_CACHE_DIR = PROJECT_ROOT / "outputs" / "milestone2" / "cache"
M4_DIR = PROJECT_ROOT / "outputs" / "milestone4"
OUT_DIR = PROJECT_ROOT / "outputs" / "milestone5"

DEFAULT_HMM_CONFIG: Dict[str, Any] = {
    "emit_weight": 1.0,
    "prob_clip_eps": 1e-6,
    "beta_m": 15.0,               # Exponential route-vs-GPS distance mismatch scale (meters)
    "gap_beta_m": 60.0,           # Relaxed mismatch scale during GPS gaps (dt > 2s)
    "stay_same_arc_bonus": 0.25,  # Mild preference to stay on current arc vs jittering at boundary
    "backward_pen_per_m": 0.25,   # Penalty per meter of significant backward motion along same directed arc
    "switch_penalty": 0.20,       # Mild penalty for switching arcs on 1-hop transition
    "u_turn_penalty": 4.0,        # Penalty for immediate U-turn onto exact reverse directed arc (u,v)->(v,u)
    "multihop_base_penalty": 0.6, # Base penalty for skipping intermediate arcs between readings
    "disconnected_penalty": 12.0, # Finite penalty when no directed path exists within search cap
    "gap_disconnected_penalty": 4.0,  # Softer penalty across large GPS dropouts (dt > 10s or step > 150m)
    "gap_dt_thresh_s": 2.0,
    "large_gap_dt_thresh_s": 10.0,
    "large_gap_dist_m": 150.0,
    "max_dijkstra_dist_m": 1500.0,
}


def build_directed_adjacency(
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    arc_lengths: np.ndarray,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build compact directed adjacency list node_u -> [(node_v, min_arc_len_m), ...]."""
    edge_min: Dict[Tuple[int, int], float] = {}
    for u, v, l in zip(from_nodes.tolist(), to_nodes.tolist(), arc_lengths.tolist()):
        if u == v or not np.isfinite(l):
            continue
        key = (u, v)
        if key not in edge_min or l < edge_min[key]:
            edge_min[key] = float(l)
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (u, v), l in edge_min.items():
        adj[u].append((v, l))
    return adj


def bounded_shortest_path_to_targets(
    adj: Dict[int, List[Tuple[int, float]]],
    source: int,
    targets: set,
    max_dist: float,
) -> Dict[int, float]:
    """Single-source Dijkstra capped at max_dist, stopping early once all targets are settled."""
    remaining = set(targets)
    dist: Dict[int, float] = {source: 0.0}
    found: Dict[int, float] = {}
    if source in remaining:
        found[source] = 0.0
        remaining.discard(source)
        if not remaining:
            return found

    heap: List[Tuple[float, int]] = [(0.0, source)]
    while heap and remaining:
        d, u = heapq.heappop(heap)
        if d > max_dist:
            break
        if d > dist.get(u, float("inf")):
            continue
        if u in remaining:
            found[u] = d
            remaining.discard(u)
            if not remaining:
                break
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd <= max_dist and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return found


def viterbi_decode_trip(
    trip: TripData,
    trip_cands: pd.DataFrame,
    cand_probs: np.ndarray,
    geom: Optional[Dict[str, np.ndarray]] = None,
    cfg: Dict[str, Any] = DEFAULT_HMM_CONFIG,
) -> Dict[str, Any]:
    """Run HMM/Viterbi decoding for one trip using candidate directed arcs as hidden states.

    Preserves 1-to-1 correspondence between every decoded prediction and its original obs_id.
    Handles same-arc motion, multi-hop transitions, U-turn suppression, missing candidates,
    disconnected road components, and large GPS timestamp gaps without underflow or NaN.
    """
    t0 = time.perf_counter()
    if geom is None:
        geom = project_trip_geometry(trip)

    n_obs = len(trip.track)
    track_x = geom["track_x"]
    track_y = geom["track_y"]
    ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
    from_nodes = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    to_nodes = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
    arc_len = geom["arc_len"]

    adj = build_directed_adjacency(from_nodes, to_nodes, arc_len)

    # Group candidates by obs_id (preserving exact order)
    obs_ids_arr = trip_cands["obs_id"].to_numpy(dtype=np.int64)
    arc_ids_arr = trip_cands["arc_id"].to_numpy(dtype=np.int64)
    proj_arr = trip_cands["proj_frac"].to_numpy(dtype=np.float64)
    probs_arr = np.clip(np.asarray(cand_probs, dtype=np.float64), cfg["prob_clip_eps"], 1.0 - cfg["prob_clip_eps"])

    # Fast CSR-style index per obs_id
    per_obs_arcs: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_obs)]
    per_obs_proj: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(n_obs)]
    per_obs_emit: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(n_obs)]
    per_obs_prob: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(n_obs)]

    emit_w = float(cfg["emit_weight"])
    if len(obs_ids_arr) > 0:
        # trip_cands is already grouped by obs_id, but handle arbitrary ordering safely
        order = np.argsort(obs_ids_arr, kind="mergesort")
        s_obs = obs_ids_arr[order]
        s_arc = arc_ids_arr[order]
        s_proj = proj_arr[order]
        s_prob = probs_arr[order]
        uniq_obs, starts, counts = np.unique(s_obs, return_index=True, return_counts=True)
        for oi, st, cnt in zip(uniq_obs.tolist(), starts.tolist(), counts.tolist()):
            en = st + cnt
            per_obs_arcs[oi] = s_arc[st:en]
            per_obs_proj[oi] = s_proj[st:en]
            per_obs_prob[oi] = s_prob[st:en]
            per_obs_emit[oi] = emit_w * np.log(s_prob[st:en])

    # Identify active observations with >=1 candidate (gracefully handles missing candidates!)
    active_obs = [i for i in range(n_obs) if len(per_obs_arcs[i]) > 0]
    pred_arcs = np.full(n_obs, -1, dtype=np.int64)
    pred_probs = np.zeros(n_obs, dtype=np.float64)

    if not active_obs:
        return {
            "pred_arcs": pred_arcs,
            "pred_probs": pred_probs,
            "n_missing_candidate_obs": n_obs,
            "n_disconnected_steps": 0,
            "n_gap_resets": 0,
            "decode_seconds": time.perf_counter() - t0,
        }

    # Shortest path cache per (source_node, max_dist_bucket) -> {target_node: dist_m}
    sp_cache: Dict[Tuple[int, int], Dict[int, float]] = {}

    def get_sp_dist(src: int, targets: set, max_d: float) -> Dict[int, float]:
        # Bucket max_d to 200m increments to maximize cache hits across adjacent 1Hz points
        bucket = int(np.ceil(max_d / 200.0) * 200)
        ckey = (src, bucket)
        cached = sp_cache.get(ckey)
        if cached is None:
            res = bounded_shortest_path_to_targets(adj, src, targets, float(bucket))
            sp_cache[ckey] = res
            return res
        missing_targets = {t for t in targets if t not in cached}
        if missing_targets:
            res = bounded_shortest_path_to_targets(adj, src, missing_targets, float(bucket))
            for k, v in res.items():
                cached[k] = v
            # Mark unreachable targets as inf so we never re-run Dijkstra for them at this bucket!
            for t in missing_targets:
                if t not in cached:
                    cached[t] = float("inf")
        return cached

    n_active = len(active_obs)
    backptrs: List[np.ndarray] = []

    # Initialize Viterbi log-probability vector at first active observation
    t_first = active_obs[0]
    prev_dp = per_obs_emit[t_first].copy()
    backptrs.append(np.full(len(prev_dp), -1, dtype=np.int32))

    beta_normal = float(cfg["beta_m"])
    beta_gap = float(cfg["gap_beta_m"])
    stay_bonus = float(cfg["stay_same_arc_bonus"])
    back_pen_m = float(cfg["backward_pen_per_m"])
    switch_pen = float(cfg["switch_penalty"])
    uturn_pen = float(cfg["u_turn_penalty"])
    multihop_pen = float(cfg["multihop_base_penalty"])
    disc_pen_normal = float(cfg["disconnected_penalty"])
    disc_pen_gap = float(cfg["gap_disconnected_penalty"])
    gap_dt_thr = float(cfg["gap_dt_thresh_s"])
    lgap_dt_thr = float(cfg["large_gap_dt_thresh_s"])
    lgap_dist_thr = float(cfg["large_gap_dist_m"])
    max_dijk_cap = float(cfg["max_dijkstra_dist_m"])

    n_gap_resets = 0

    for idx in range(1, n_active):
        t_prev = active_obs[idx - 1]
        t_curr = active_obs[idx]

        dt = float(ts[t_curr] - ts[t_prev])
        gps_step = float(np.hypot(track_x[t_curr] - track_x[t_prev], track_y[t_curr] - track_y[t_prev]))

        is_gap = dt > gap_dt_thr
        is_large_gap = (dt > lgap_dt_thr) or (gps_step > lgap_dist_thr)
        beta = beta_gap if is_gap else beta_normal
        disc_pen = disc_pen_gap if is_large_gap else disc_pen_normal

        prev_arcs = per_obs_arcs[t_prev]
        prev_proj = per_obs_proj[t_prev]
        curr_arcs = per_obs_arcs[t_curr]
        curr_proj = per_obs_proj[t_curr]
        curr_emit = per_obs_emit[t_curr]

        k_prev = len(prev_arcs)
        k_curr = len(curr_arcs)

        # Across extreme gaps (>60s or >1000m), reset transition memory cleanly to avoid forcing bogus paths
        if dt > 60.0 or gps_step > 1000.0:
            n_gap_resets += 1
            best_prev_idx = int(np.argmax(prev_dp))
            curr_dp = prev_dp[best_prev_idx] - disc_pen + curr_emit
            # Normalize to prevent numerical drift
            curr_dp = curr_dp - np.max(curr_dp)
            prev_dp = curr_dp
            backptrs.append(np.full(k_curr, best_prev_idx, dtype=np.int32))
            continue

        # Target start nodes for current step
        curr_u = from_nodes[curr_arcs]
        curr_v = to_nodes[curr_arcs]
        curr_l = arc_len[curr_arcs]
        prev_u = from_nodes[prev_arcs]
        prev_v = to_nodes[prev_arcs]
        prev_l = arc_len[prev_arcs]

        target_nodes = set(curr_u.tolist())
        search_cap = min(max(250.0, 2.5 * gps_step + 150.0), max_dijk_cap)

        # Precompute multi-hop shortest paths from each distinct prev_v only when needed
        sp_by_src: Dict[int, Dict[int, float]] = {}
        for sv in set(prev_v.tolist()):
            # Check if there are any non-identical target nodes
            non_direct = target_nodes - {sv}
            if non_direct:
                sp_by_src[sv] = get_sp_dist(sv, non_direct, search_cap)
            else:
                sp_by_src[sv] = {}

        # Compute transition log-prob matrix (k_prev, k_curr)
        trans = np.full((k_prev, k_curr), -disc_pen, dtype=np.float64)

        for i in range(k_prev):
            a1 = int(prev_arcs[i])
            u1 = int(prev_u[i])
            v1 = int(prev_v[i])
            l1 = float(prev_l[i])
            f1 = float(prev_proj[i])
            rem1 = (1.0 - f1) * l1
            sp_map = sp_by_src.get(v1, {})

            for j in range(k_curr):
                a2 = int(curr_arcs[j])
                u2 = int(curr_u[j])
                v2 = int(curr_v[j])
                l2 = float(curr_l[j])
                f2 = float(curr_proj[j])

                if a1 == a2:
                    # Same directed arc
                    delta_r = (f2 - f1) * l1
                    if delta_r >= -3.0:
                        mismatch = abs(max(0.0, delta_r) - gps_step)
                        trans[i, j] = stay_bonus - mismatch / beta
                    else:
                        # Moving backward along the same directed arc
                        back_m = -delta_r - 3.0
                        trans[i, j] = -back_pen_m * back_m - abs(gps_step) / beta
                elif v1 == u2:
                    # Direct 1-hop directed continuation!
                    route_d = rem1 + f2 * l2
                    mismatch = abs(route_d - gps_step)
                    pen = switch_pen + (uturn_pen if u1 == v2 else 0.0)
                    trans[i, j] = -pen - mismatch / beta
                else:
                    # Multi-hop directed path v1 -> ... -> u2
                    d_net = sp_map.get(u2, float("inf"))
                    if np.isfinite(d_net):
                        route_d = rem1 + d_net + f2 * l2
                        mismatch = abs(route_d - gps_step)
                        pen = multihop_pen + (uturn_pen if u1 == v2 else 0.0)
                        trans[i, j] = -pen - mismatch / beta
                    else:
                        # Disconnected: keep finite -disc_pen (already initialized)
                        pass

        # Viterbi recurrence: score[i, j] = prev_dp[i] + trans[i, j]
        scores = prev_dp[:, None] + trans
        best_prev = np.argmax(scores, axis=0).astype(np.int32)
        curr_dp = scores[best_prev, np.arange(k_curr)] + curr_emit

        # Subtract max for numerical stability (invariance under constant shift)
        curr_dp = curr_dp - np.max(curr_dp)
        prev_dp = curr_dp
        backptrs.append(best_prev)

    # Backtrack optimal state sequence
    state_idx = np.empty(n_active, dtype=np.int32)
    state_idx[n_active - 1] = int(np.argmax(prev_dp))
    for idx in range(n_active - 1, 0, -1):
        state_idx[idx - 1] = backptrs[idx][state_idx[idx]]

    for idx, oi in enumerate(active_obs):
        si = int(state_idx[idx])
        pred_arcs[oi] = int(per_obs_arcs[oi][si])
        pred_probs[oi] = float(per_obs_prob[oi][si])

    # Count disconnected steps in final decoded sequence
    n_disc = 0
    for idx in range(1, n_active):
        a1 = int(pred_arcs[active_obs[idx - 1]])
        a2 = int(pred_arcs[active_obs[idx]])
        if a1 != a2 and to_nodes[a1] != from_nodes[a2]:
            n_disc += 1

    return {
        "pred_arcs": pred_arcs,
        "pred_probs": pred_probs,
        "n_missing_candidate_obs": int(n_obs - n_active),
        "n_non_1hop_transitions": int(n_disc),
        "n_gap_resets": int(n_gap_resets),
        "decode_seconds": time.perf_counter() - t0,
    }


def evaluate_hmm_vs_classifier(
    split_dir: Path,
    cand_df: pd.DataFrame,
    obs_meta_df: pd.DataFrame,
    probs: np.ndarray,
    hmm_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Compare Classifier-only vs Classifier+HMM/Viterbi across all trips in a split."""
    cand_with_p = cand_df[["trip_id", "obs_id", "arc_id", "proj_frac", "dist_m"]].copy()
    cand_with_p["prob"] = np.asarray(probs, dtype=np.float64)

    # Classifier-only observation predictions
    clf_obs_metrics, clf_pred_obs = compute_observation_top1_and_miss(cand_df, probs, obs_meta_df)

    trip_ids = sorted(obs_meta_df["trip_id"].unique().tolist())
    per_trip_rows: List[Dict[str, Any]] = []
    trip_details: Dict[str, Dict[str, Any]] = {}

    total_scored = 0
    clf_correct = 0
    hmm_correct = 0
    hmm_total_decode_s = 0.0

    for tid in trip_ids:
        trip = load_trip(split_dir, tid, load_route=True)
        geom = project_trip_geometry(trip)
        t_cands = cand_with_p[cand_with_p["trip_id"] == tid].reset_index(drop=True)
        t_obs = obs_meta_df[obs_meta_df["trip_id"] == tid].sort_values("obs_id").reset_index(drop=True)
        t_clf = clf_pred_obs[clf_pred_obs["trip_id"] == tid].sort_values("obs_id").reset_index(drop=True)

        clf_arcs = t_clf["pred_arc_id"].to_numpy(dtype=np.int64)
        ref_arcs = t_obs["ref_arc_id"].to_numpy(dtype=np.int64)
        scored_mask = t_obs["is_scored"].to_numpy(dtype=bool)

        dec = viterbi_decode_trip(trip, t_cands, t_cands["prob"].to_numpy(), geom=geom, cfg=hmm_cfg)
        hmm_arcs = dec["pred_arcs"]
        hmm_total_decode_s += dec["decode_seconds"]

        fn = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
        tn = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)

        clf_cont = compute_route_continuity_for_trip(clf_arcs, fn, tn)
        hmm_cont = compute_route_continuity_for_trip(hmm_arcs, fn, tn)
        clf_seq = compute_sequence_metrics_for_trip(clf_arcs, trip.route, geom["arc_len"])
        hmm_seq = compute_sequence_metrics_for_trip(hmm_arcs, trip.route, geom["arc_len"])

        n_sc = int(np.sum(scored_mask))
        c_ok = int(np.sum((clf_arcs == ref_arcs) & scored_mask))
        h_ok = int(np.sum((hmm_arcs == ref_arcs) & scored_mask))
        n_changed = int(np.sum((clf_arcs != hmm_arcs) & scored_mask))
        n_fixed = int(np.sum((clf_arcs != ref_arcs) & (hmm_arcs == ref_arcs) & scored_mask))
        n_broken = int(np.sum((clf_arcs == ref_arcs) & (hmm_arcs != ref_arcs) & scored_mask))

        total_scored += n_sc
        clf_correct += c_ok
        hmm_correct += h_ok

        per_trip_rows.append(
            {
                "trip_id": tid,
                "n_obs": len(t_obs),
                "n_scored": n_sc,
                "clf_top1": round(c_ok / max(n_sc, 1), 5),
                "hmm_top1": round(h_ok / max(n_sc, 1), 5),
                "delta_top1": round((h_ok - c_ok) / max(n_sc, 1), 5),
                "clf_continuity": round(clf_cont["route_continuity"], 5),
                "hmm_continuity": round(hmm_cont["route_continuity"], 5),
                "clf_seq_lcs_f1": round(clf_seq["seq_lcs_f1"], 5),
                "hmm_seq_lcs_f1": round(hmm_seq["seq_lcs_f1"], 5),
                "clf_seq_lw_f1": round(clf_seq["seq_length_weighted_f1"], 5),
                "hmm_seq_lw_f1": round(hmm_seq["seq_length_weighted_f1"], 5),
                "n_changed": n_changed,
                "n_fixed_by_hmm": n_fixed,
                "n_hurt_by_hmm": n_broken,
                "hmm_decode_ms": round(1000.0 * dec["decode_seconds"], 2),
            }
        )
        trip_details[tid] = {
            "clf_arcs": clf_arcs,
            "hmm_arcs": hmm_arcs,
            "ref_arcs": ref_arcs,
            "scored_mask": scored_mask,
            "n_fixed": n_fixed,
            "n_hurt": n_broken,
            "decode_ms": 1000.0 * dec["decode_seconds"],
        }

    per_trip_df = pd.DataFrame(per_trip_rows)
    summary = {
        "n_trips": len(trip_ids),
        "n_scored_obs": int(total_scored),
        "candidate_miss_rate": round(clf_obs_metrics["candidate_miss_rate"], 5),
        "classifier_only": {
            "pointwise_top1_overall": round(clf_correct / max(total_scored, 1), 5),
            "route_continuity_mean": round(float(per_trip_df["clf_continuity"].mean()), 5),
            "seq_lcs_f1_mean": round(float(per_trip_df["clf_seq_lcs_f1"].mean()), 5),
            "seq_length_weighted_f1_mean": round(float(per_trip_df["clf_seq_lw_f1"].mean()), 5),
        },
        "classifier_plus_hmm": {
            "pointwise_top1_overall": round(hmm_correct / max(total_scored, 1), 5),
            "route_continuity_mean": round(float(per_trip_df["hmm_continuity"].mean()), 5),
            "seq_lcs_f1_mean": round(float(per_trip_df["hmm_seq_lcs_f1"].mean()), 5),
            "seq_length_weighted_f1_mean": round(float(per_trip_df["hmm_seq_lw_f1"].mean()), 5),
            "total_decode_seconds": round(hmm_total_decode_s, 4),
            "mean_decode_ms_per_trip": round(1000.0 * hmm_total_decode_s / max(len(trip_ids), 1), 2),
        },
        "improvements": {
            "delta_pointwise_top1": round((hmm_correct - clf_correct) / max(total_scored, 1), 5),
            "delta_route_continuity": round(
                float(per_trip_df["hmm_continuity"].mean() - per_trip_df["clf_continuity"].mean()), 5
            ),
            "delta_seq_lcs_f1": round(
                float(per_trip_df["hmm_seq_lcs_f1"].mean() - per_trip_df["clf_seq_lcs_f1"].mean()), 5
            ),
            "total_obs_fixed_by_hmm": int(per_trip_df["n_fixed_by_hmm"].sum()),
            "total_obs_hurt_by_hmm": int(per_trip_df["n_hurt_by_hmm"].sum()),
            "net_obs_gained": int(per_trip_df["n_fixed_by_hmm"].sum() - per_trip_df["n_hurt_by_hmm"].sum()),
        },
    }
    return summary, per_trip_df, trip_details


def generate_hmm_folium_map_and_case_study(
    test_dir: Path,
    test_cand: pd.DataFrame,
    test_probs: np.ndarray,
    trip_details: Dict[str, Dict[str, Any]],
    per_trip_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    """Generate a labelled Folium map for the most illustrative student-test trip & extract a concrete case study."""
    # Pick student-test trip with largest positive improvement from HMM
    best_row = per_trip_df.sort_values(["delta_top1", "n_fixed_by_hmm"], ascending=[False, False]).iloc[0]
    tid = str(best_row["trip_id"])
    det = trip_details[tid]

    trip = load_trip(test_dir, tid, load_route=True)
    geom = project_trip_geometry(trip)
    nodes_lat = trip.nodes["lat"].to_numpy()
    nodes_lon = trip.nodes["lon"].to_numpy()
    u_arr = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    v_arr = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)

    clf_arcs = det["clf_arcs"]
    hmm_arcs = det["hmm_arcs"]
    ref_arcs = det["ref_arcs"]
    scored_mask = det["scored_mask"]

    # Find concrete case where HMM fixed classifier error (and one where HMM hurt, if any, for honest analysis!)
    fixed_indices = np.flatnonzero((clf_arcs != ref_arcs) & (hmm_arcs == ref_arcs) & scored_mask)
    hurt_indices = np.flatnonzero((clf_arcs == ref_arcs) & (hmm_arcs != ref_arcs) & scored_mask)

    t_cands = test_cand[test_cand["trip_id"] == tid].copy()
    t_cands["prob"] = test_probs[test_cand["trip_id"] == tid]

    def describe_obs(oi: int) -> Dict[str, Any]:
        c_oi = t_cands[t_cands["obs_id"] == oi].sort_values("prob", ascending=False)
        ca = int(clf_arcs[oi])
        ha = int(hmm_arcs[oi])
        ra = int(ref_arcs[oi])
        prev_h = int(hmm_arcs[oi - 1]) if oi > 0 else -1
        next_h = int(hmm_arcs[oi + 1]) if oi + 1 < len(hmm_arcs) else -1
        return {
            "trip_id": tid,
            "obs_id": int(oi),
            "timestamp_s": float(trip.track.iloc[oi]["timestamp"]),
            "lon_lat": [float(trip.track.iloc[oi]["lon"]), float(trip.track.iloc[oi]["lat"])],
            "reference_arc_id": ra,
            "classifier_only_arc_id": ca,
            "hmm_viterbi_arc_id": ha,
            "hmm_prev_arc_id": prev_h,
            "hmm_next_arc_id": next_h,
            "prev_to_hmm_connected": bool(prev_h >= 0 and (prev_h == ha or to_nodes_safe(v_arr, prev_h) == from_nodes_safe(u_arr, ha))),
            "prev_to_clf_connected": bool(prev_h >= 0 and (prev_h == ca or to_nodes_safe(v_arr, prev_h) == from_nodes_safe(u_arr, ca))),
            "top_candidates": [
                {
                    "arc_id": int(r.arc_id),
                    "prob": round(float(r.prob), 4),
                    "dist_m": round(float(r.dist_m), 2),
                    "heading_diff_deg": round(float(r.heading_diff_deg), 1),
                    "is_stationary": int(r.is_stationary),
                }
                for r in c_oi.head(5).itertuples(index=False)
            ],
        }

    def to_nodes_safe(arr: np.ndarray, idx: int) -> int:
        return int(arr[idx]) if 0 <= idx < len(arr) else -999

    def from_nodes_safe(arr: np.ndarray, idx: int) -> int:
        return int(arr[idx]) if 0 <= idx < len(arr) else -998

    fixed_case = describe_obs(int(fixed_indices[len(fixed_indices) // 2])) if len(fixed_indices) > 0 else None
    hurt_case = describe_obs(int(hurt_indices[0])) if len(hurt_indices) > 0 else None

    # Build labelled Folium map for student-test trip `tid`
    center_lat = float(trip.track["lat"].median())
    center_lon = float(trip.track["lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # Relevant road network (arcs appearing in candidates + 1-hop neighborhood, keeps HTML snappy & clean)
    rel_arc_ids = set(t_cands["arc_id"].unique().tolist()) | set(trip.route.tolist())
    fg_roads = folium.FeatureGroup(name=f"Relevant Road Network ({len(rel_arc_ids):,} arcs)", show=True)
    for aid in sorted(rel_arc_ids):
        if 0 <= aid < len(u_arr):
            u, v = int(u_arr[aid]), int(v_arr[aid])
            folium.PolyLine(
                locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
                color="#94a3b8",
                weight=1.8,
                opacity=0.65,
            ).add_to(fg_roads)
    fg_roads.add_to(m)

    # Reference route
    fg_ref = folium.FeatureGroup(name=f"Reference Route ({len(trip.route)} arcs)", show=True)
    for idx_r, aid in enumerate(trip.route.tolist()):
        u, v = int(u_arr[aid]), int(v_arr[aid])
        folium.PolyLine(
            locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
            color="#16a34a",
            weight=6,
            opacity=0.7,
            tooltip=f"Reference step {idx_r}: arc_id={aid}",
        ).add_to(fg_ref)
    fg_ref.add_to(m)

    # Classifier-only predicted arcs (toggleable for comparison)
    clf_uniq = list(dict.fromkeys(int(a) for a in clf_arcs if a >= 0))
    fg_clf = folium.FeatureGroup(name=f"Classifier-Only Predicted Arcs ({len(clf_uniq)} unique)", show=False)
    for aid in clf_uniq:
        u, v = int(u_arr[aid]), int(v_arr[aid])
        folium.PolyLine(
            locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
            color="#f59e0b",
            weight=3.5,
            opacity=0.85,
            dash_array="6,6",
            tooltip=f"Classifier-only arc_id={aid}",
        ).add_to(fg_clf)
    fg_clf.add_to(m)

    # HMM/Viterbi predicted route
    hmm_uniq = list(dict.fromkeys(int(a) for a in hmm_arcs if a >= 0))
    fg_hmm = folium.FeatureGroup(name=f"HMM/Viterbi Predicted Route ({len(hmm_uniq)} unique arcs)", show=True)
    for aid in hmm_uniq:
        u, v = int(u_arr[aid]), int(v_arr[aid])
        folium.PolyLine(
            locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
            color="#dc2626",
            weight=3.5,
            opacity=0.9,
            tooltip=f"HMM Predicted arc_id={aid}",
        ).add_to(fg_hmm)
    fg_hmm.add_to(m)

    # Raw GPS observations (subsample markers if >2500 for smooth browser rendering)
    fg_gps = folium.FeatureGroup(name=f"Raw GPS Observations ({len(trip.track):,} pts)", show=True)
    stride = max(1, len(trip.track) // 1800)
    for oi in range(0, len(trip.track), stride):
        row = trip.track.iloc[oi]
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=2.3,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.7,
            weight=0.5,
            tooltip=f"obs={oi}, t={row['timestamp']:.0f}s | ref={ref_arcs[oi]} | clf={clf_arcs[oi]} | hmm={hmm_arcs[oi]}",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Highlight the worked HMM-correction observation!
    if fixed_case is not None:
        fg_fix = folium.FeatureGroup(name=f"HMM Correction Example (obs_id={fixed_case['obs_id']})", show=True)
        folium.Marker(
            location=(fixed_case["lon_lat"][1], fixed_case["lon_lat"][0]),
            popup=(
                f"<b>HMM Fixed Error at obs_id={fixed_case['obs_id']}</b><br>"
                f"Classifier-only arc: {fixed_case['classifier_only_arc_id']}<br>"
                f"HMM/Viterbi arc: {fixed_case['hmm_viterbi_arc_id']}<br>"
                f"Reference arc: {fixed_case['reference_arc_id']}"
            ),
            icon=folium.Icon(color="purple", icon="check", prefix="fa"),
        ).add_to(fg_fix)
        fg_fix.add_to(m)

    m.fit_bounds([
        [float(trip.track["lat"].min()), float(trip.track["lon"].min())],
        [float(trip.track["lat"].max()), float(trip.track["lon"].max())],
    ])
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 2px solid #cbd5e1; border-radius: 6px;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 6px;">PathSnap — Student-Test Trip {tid} (HMM/Viterbi)</div>
      <div><span style="display:inline-block;width:11px;height:11px;background:#2563eb;border-radius:50%;margin-right:6px;"></span>Raw GPS observations ({len(trip.track):,})</div>
      <div><span style="display:inline-block;width:14px;height:3px;background:#94a3b8;margin-right:6px;vertical-align:middle;"></span>Relevant road network ({len(rel_arc_ids):,} arcs)</div>
      <div><span style="display:inline-block;width:14px;height:5px;background:#16a34a;margin-right:6px;vertical-align:middle;"></span>Reference route ({len(trip.route)} arcs)</div>
      <div><span style="display:inline-block;width:14px;height:4px;background:#dc2626;margin-right:6px;vertical-align:middle;"></span>HMM/Viterbi predicted route (Top-1: {best_row['hmm_top1']:.3f}, Cont: {best_row['hmm_continuity']:.3f})</div>
      <div><span style="display:inline-block;width:14px;height:3px;background:#f59e0b;margin-right:6px;vertical-align:middle;"></span>Classifier-only (toggleable, Top-1: {best_row['clf_top1']:.3f})</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_path = out_dir / f"student_test_{tid}_hmm_map.html"
    m.save(str(map_path))

    # Static matplotlib comparison plot for technical report
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for aid in rel_arc_ids:
        if 0 <= aid < len(u_arr):
            u, v = int(u_arr[aid]), int(v_arr[aid])
            ax.plot([nodes_lon[u], nodes_lon[v]], [nodes_lat[u], nodes_lat[v]], color="#cbd5e1", lw=0.7, zorder=1)
    for i_r, aid in enumerate(trip.route.tolist()):
        u, v = int(u_arr[aid]), int(v_arr[aid])
        ax.plot([nodes_lon[u], nodes_lon[v]], [nodes_lat[u], nodes_lat[v]], color="#16a34a", lw=4.0, alpha=0.65, zorder=2, label="Reference Route" if i_r == 0 else None)
    for i_h, aid in enumerate(hmm_uniq):
        u, v = int(u_arr[aid]), int(v_arr[aid])
        ax.plot([nodes_lon[u], nodes_lon[v]], [nodes_lat[u], nodes_lat[v]], color="#dc2626", lw=1.8, zorder=3, label="HMM/Viterbi Predicted Route" if i_h == 0 else None)
    ax.scatter(trip.track["lon"], trip.track["lat"], s=4, c="#2563eb", alpha=0.6, zorder=4, label="Raw GPS Points")
    ax.set_title(f"Student-Test Trip {tid}: Raw GPS, Reference Route & HMM/Viterbi Reconstruction", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"student_test_{tid}_hmm_route.png")
    plt.close(fig)

    return {
        "featured_trip_id": tid,
        "folium_map_path": str(map_path),
        "static_plot_path": str(out_dir / f"student_test_{tid}_hmm_route.png"),
        "fixed_case_example": fixed_case,
        "hurt_case_example": hurt_case,
    }


def plot_hmm_comparison(per_trip_df: pd.DataFrame, summary: Dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)

    # Left: Aggregate bar comparison
    metrics_labels = ["Pointwise Top-1\nAccuracy", "Route Continuity\n(1-hop/same)", "Sequence LCS F1\n(Order-Aware)", "Sequence Length-\nWeighted F1"]
    c_vals = [
        summary["classifier_only"]["pointwise_top1_overall"],
        summary["classifier_only"]["route_continuity_mean"],
        summary["classifier_only"]["seq_lcs_f1_mean"],
        summary["classifier_only"]["seq_length_weighted_f1_mean"],
    ]
    h_vals = [
        summary["classifier_plus_hmm"]["pointwise_top1_overall"],
        summary["classifier_plus_hmm"]["route_continuity_mean"],
        summary["classifier_plus_hmm"]["seq_lcs_f1_mean"],
        summary["classifier_plus_hmm"]["seq_length_weighted_f1_mean"],
    ]
    x = np.arange(len(metrics_labels))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, c_vals, w, label="Tuned Classifier Only", color="#f59e0b")
    b2 = axes[0].bar(x + w / 2, h_vals, w, label="Tuned Classifier + HMM/Viterbi", color="#16a34a")
    for grp in (b1, b2):
        for b in grp:
            v = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_labels, fontsize=8.5)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Student-Test Aggregate: Classifier Only vs +HMM/Viterbi", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: Per-trip Top-1 Accuracy comparison across all 10 student-test trips
    tids = per_trip_df["trip_id"].tolist()
    xt = np.arange(len(tids))
    axes[1].bar(xt - w / 2, per_trip_df["clf_top1"], w, label="Classifier Only", color="#f59e0b", alpha=0.85)
    axes[1].bar(xt + w / 2, per_trip_df["hmm_top1"], w, label="+HMM/Viterbi", color="#16a34a", alpha=0.9)
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels(tids, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylim(max(0.0, float(per_trip_df["clf_top1"].min()) - 0.08), 1.03)
    axes[1].set_ylabel("Pointwise Top-1 Accuracy")
    axes[1].set_title("Per-Trip Top-1 Accuracy Across All 10 Student-Test Trips", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "hmm_vs_classifier_comparison.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = discover_dataset_dirs()

    print("Loading tuned classifier OOF & student-test probabilities from Milestone 4...")
    train_cand = pd.read_parquet(M2_CACHE_DIR / "train_candidates.parquet")
    train_obs = pd.read_parquet(M2_CACHE_DIR / "train_obs_meta.parquet")
    test_cand = pd.read_parquet(M2_CACHE_DIR / "student_test_candidates.parquet")
    test_obs = pd.read_parquet(M2_CACHE_DIR / "student_test_obs_meta.parquet")
    for df in (train_cand, train_obs, test_cand, test_obs):
        df["trip_id"] = df["trip_id"].astype(str)

    oof_probs = np.load(M4_DIR / "tuned_oof_probs.npy")
    test_probs = np.load(M4_DIR / "tuned_test_probs.npy")

    # Select HMM transition parameters strictly on a subset of 15 training trips (using OOF probs) — zero leakage!
    print("Validating HMM transition settings on 15 training trips (out-of-fold probabilities)...")
    sample_train_ids = sorted(train_obs["trip_id"].unique().tolist())[:15]
    sub_tr_cand = train_cand[train_cand["trip_id"].isin(sample_train_ids)].reset_index(drop=True)
    sub_tr_obs = train_obs[train_obs["trip_id"].isin(sample_train_ids)].reset_index(drop=True)
    sub_tr_probs = oof_probs[train_cand["trip_id"].isin(sample_train_ids).to_numpy()]

    grid_configs = [
        {"beta_m": 12.0, "u_turn_penalty": 3.0, "emit_weight": 1.0, "disconnected_penalty": 10.0},
        {"beta_m": 15.0, "u_turn_penalty": 4.0, "emit_weight": 1.0, "disconnected_penalty": 12.0},
        {"beta_m": 20.0, "u_turn_penalty": 5.0, "emit_weight": 1.2, "disconnected_penalty": 14.0},
        {"beta_m": 15.0, "u_turn_penalty": 4.0, "emit_weight": 1.5, "disconnected_penalty": 12.0},
    ]
    grid_log = []
    best_cfg = dict(DEFAULT_HMM_CONFIG)
    best_score = -1.0
    for g_idx, g_overrides in enumerate(grid_configs):
        cfg_try = {**DEFAULT_HMM_CONFIG, **g_overrides}
        s_tr, _, _ = evaluate_hmm_vs_classifier(train_dir, sub_tr_cand, sub_tr_obs, sub_tr_probs, cfg_try)
        # Jointly value pointwise Top-1 + sequence LCS F1
        score = 0.6 * s_tr["classifier_plus_hmm"]["pointwise_top1_overall"] + 0.4 * s_tr["classifier_plus_hmm"]["seq_lcs_f1_mean"]
        grid_log.append(
            {
                "grid_idx": g_idx,
                "overrides": g_overrides,
                "train15_oof_top1": s_tr["classifier_plus_hmm"]["pointwise_top1_overall"],
                "train15_oof_seq_lcs_f1": s_tr["classifier_plus_hmm"]["seq_lcs_f1_mean"],
                "train15_oof_continuity": s_tr["classifier_plus_hmm"]["route_continuity_mean"],
                "composite_score": round(score, 5),
            }
        )
        print(f"  HMM grid {g_idx}: {g_overrides} -> Top1={s_tr['classifier_plus_hmm']['pointwise_top1_overall']:.4f}, LCS-F1={s_tr['classifier_plus_hmm']['seq_lcs_f1_mean']:.4f}, Cont={s_tr['classifier_plus_hmm']['route_continuity_mean']:.4f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg_try

    # Save chosen HMM config to artifacts/ for FastAPI inference
    (ARTIFACTS_DIR / "hmm_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

    # Evaluate locked HMM config on all 10 student-test trips
    print("\nEvaluating Classifier-Only vs Classifier+HMM/Viterbi on all 10 student-test trips...")
    test_summary, per_trip_df, trip_details = evaluate_hmm_vs_classifier(
        test_dir, test_cand, test_obs, test_probs, best_cfg
    )
    per_trip_df.to_csv(OUT_DIR / "student_test_hmm_per_trip.csv", index=False)

    case_info = generate_hmm_folium_map_and_case_study(
        test_dir, test_cand, test_probs, trip_details, per_trip_df, OUT_DIR
    )
    plot_hmm_comparison(per_trip_df, test_summary, OUT_DIR)

    report = {
        "hmm_config": best_cfg,
        "train15_oof_grid_search": grid_log,
        "student_test_comparison": test_summary,
        "case_study_and_map": case_info,
    }
    (OUT_DIR / "milestone5_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== MILESTONE 5 SUMMARY ===")
    print(json.dumps(test_summary, indent=2))
    print("\nPer-trip breakdown:")
    print(per_trip_df.to_string(index=False))


if __name__ == "__main__":
    main()
