"""Milestone 2 (Inference-only): Spatial indexing, candidate arc generation, and feature engineering.

STRICT ISOLATION GUARANTEE:
This module NEVER reads, imports, or accepts reference `.route` labels.
All features are computed exclusively from `.track`, `.nodes`, and `.arcs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from src.data_io import TripData, project_trip_geometry

CANDIDATE_CONFIG: Dict[str, Any] = {
    "search_radius_m": 60.0,
    "max_candidates": 10,
    "fallback_k": 4,
    "stationary_speed_thresh_mps": 1.0,
    "stationary_disp_thresh_m": 1.5,
    "heading_min_disp_m": 2.5,
    "heading_window_max_steps": 5,
    "speed_clip_mps": 60.0,
    "dt_clip_s": 120.0,
}

FEATURE_COLUMNS: List[str] = [
    "dist_m",
    "dist_rank",
    "dist_diff_from_best",
    "dist_ratio_to_best",
    "n_candidates",
    "heading_diff_deg",
    "cos_heading_diff",
    "heading_valid",
    "is_stationary",
    "obs_speed_mps",
    "dt_prev_s",
    "dt_next_s",
    "proj_frac",
    "arc_len_m",
    "has_reverse_arc",
    "heading_rank",
    "prev_dist_m",
    "next_dist_m",
    "along_arc_progress_m",
    "from_node_out_deg",
    "to_node_out_deg",
]


def point_to_segment_metrics(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized point-to-segment distance (m) and fractional projection t in [0, 1]."""
    vx = x2 - x1
    vy = y2 - y1
    seg_len2 = vx * vx + vy * vy
    safe_len2 = np.where(seg_len2 > 1e-12, seg_len2, 1.0)
    t = ((px - x1) * vx + (py - y1) * vy) / safe_len2
    t = np.where(seg_len2 > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
    qx = x1 + t * vx
    qy = y1 + t * vy
    dist = np.hypot(px - qx, py - qy)
    return dist, t


def compute_track_kinematics(
    track_x: np.ndarray,
    track_y: np.ndarray,
    timestamps: np.ndarray,
    cfg: Dict[str, Any] = CANDIDATE_CONFIG,
) -> Dict[str, np.ndarray]:
    """Compute per-observation speed, sampling gaps, stationary flag, and robust travel bearing."""
    n = len(track_x)
    dt_prev = np.zeros(n, dtype=np.float64)
    dt_next = np.zeros(n, dtype=np.float64)
    step_prev = np.zeros(n, dtype=np.float64)
    step_next = np.zeros(n, dtype=np.float64)

    if n > 1:
        dts = np.maximum(np.diff(timestamps), 1e-3)
        dists = np.hypot(np.diff(track_x), np.diff(track_y))
        dt_prev[1:] = dts
        dt_prev[0] = dts[0]
        dt_next[:-1] = dts
        dt_next[-1] = dts[-1]
        step_prev[1:] = dists
        step_prev[0] = dists[0]
        step_next[:-1] = dists
        step_next[-1] = dists[-1]

    # Local instantaneous speed (average of adjacent steps, or single side at endpoints)
    if n > 2:
        dx_c = track_x[2:] - track_x[:-2]
        dy_c = track_y[2:] - track_y[:-2]
        dt_c = np.maximum(timestamps[2:] - timestamps[:-2], 1e-3)
        speed = np.zeros(n, dtype=np.float64)
        speed[1:-1] = np.hypot(dx_c, dy_c) / dt_c
        speed[0] = step_next[0] / max(dt_next[0], 1e-3)
        speed[-1] = step_prev[-1] / max(dt_prev[-1], 1e-3)
    elif n == 2:
        sp = step_next[0] / max(dt_next[0], 1e-3)
        speed = np.array([sp, sp], dtype=np.float64)
    else:
        speed = np.zeros(n, dtype=np.float64)

    local_disp = np.maximum(step_prev, step_next)
    is_stationary = (
        (speed < cfg["stationary_speed_thresh_mps"])
        | (local_disp < cfg["stationary_disp_thresh_m"])
    ).astype(np.float32)

    # Robust heading estimation: expand symmetric window up to heading_window_max_steps
    # until displacement >= heading_min_disp_m, avoiding crossing large GPS gaps (>5s).
    bearing = np.zeros(n, dtype=np.float64)
    heading_valid = np.zeros(n, dtype=np.float32)
    min_disp = float(cfg["heading_min_disp_m"])
    max_w = int(cfg["heading_window_max_steps"])

    for i in range(n):
        if is_stationary[i] > 0.5:
            # Explicit stationary handling: do not trust GPS jitter heading
            continue
        found = False
        for w in range(1, max_w + 1):
            lo = max(0, i - w)
            hi = min(n - 1, i + w)
            if lo == hi:
                continue
            # Do not bridge across huge GPS gaps when estimating local heading
            if (timestamps[hi] - timestamps[lo]) > max(10.0, 3.0 * w):
                # Try one-sided short step instead
                if i + 1 < n and (timestamps[i + 1] - timestamps[i]) <= 5.0:
                    dx = track_x[i + 1] - track_x[i]
                    dy = track_y[i + 1] - track_y[i]
                    if np.hypot(dx, dy) >= min_disp:
                        bearing[i] = np.arctan2(dx, dy)
                        heading_valid[i] = 1.0
                        found = True
                        break
                if i - 1 >= 0 and (timestamps[i] - timestamps[i - 1]) <= 5.0:
                    dx = track_x[i] - track_x[i - 1]
                    dy = track_y[i] - track_y[i - 1]
                    if np.hypot(dx, dy) >= min_disp:
                        bearing[i] = np.arctan2(dx, dy)
                        heading_valid[i] = 1.0
                        found = True
                        break
                break
            dx = track_x[hi] - track_x[lo]
            dy = track_y[hi] - track_y[lo]
            if np.hypot(dx, dy) >= min_disp:
                bearing[i] = np.arctan2(dx, dy)
                heading_valid[i] = 1.0
                found = True
                break
        if not found:
            heading_valid[i] = 0.0

    # Local trajectory motion vector (from i-1 to i+1) for along-arc progress
    prev_idx = np.maximum(np.arange(n) - 1, 0)
    next_idx = np.minimum(np.arange(n) + 1, max(n - 1, 0))

    return {
        "obs_speed_mps": np.clip(speed, 0.0, cfg["speed_clip_mps"]).astype(np.float32),
        "dt_prev_s": np.clip(dt_prev, 0.0, cfg["dt_clip_s"]).astype(np.float32),
        "dt_next_s": np.clip(dt_next, 0.0, cfg["dt_clip_s"]).astype(np.float32),
        "is_stationary": is_stationary,
        "obs_bearing": bearing,
        "heading_valid": heading_valid,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
    }


def generate_trip_candidates(
    trip: TripData,
    geom: Dict[str, np.ndarray] | None = None,
    cfg: Dict[str, Any] = CANDIDATE_CONFIG,
) -> pd.DataFrame:
    """Generate candidate directed arcs and inference-only features for one trip.

    Explicitly ignores trip.route (works identically when trip.route is None).
    """
    if geom is None:
        geom = project_trip_geometry(trip)

    track_x = geom["track_x"]
    track_y = geom["track_y"]
    arc_x1 = geom["arc_x1"]
    arc_y1 = geom["arc_y1"]
    arc_x2 = geom["arc_x2"]
    arc_y2 = geom["arc_y2"]
    arc_len = geom["arc_len"]
    arc_bearing = geom["arc_bearing"]
    valid_arc_mask = geom["valid_arc_mask"]

    n_obs = len(track_x)
    n_arcs = len(arc_x1)
    if n_obs == 0 or n_arcs == 0:
        cols = ["trip_id", "obs_id", "arc_id"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    # Vectorized Shapely LineString creation + STRtree spatial index
    valid_arc_ids = np.flatnonzero(valid_arc_mask)
    seg_coords = np.empty((len(valid_arc_ids), 2, 2), dtype=np.float64)
    seg_coords[:, 0, 0] = arc_x1[valid_arc_ids]
    seg_coords[:, 0, 1] = arc_y1[valid_arc_ids]
    seg_coords[:, 1, 0] = arc_x2[valid_arc_ids]
    seg_coords[:, 1, 1] = arc_y2[valid_arc_ids]
    arc_geoms = shapely.linestrings(seg_coords)
    tree = STRtree(arc_geoms)

    pt_geoms = shapely.points(track_x, track_y)

    radius_m = float(cfg["search_radius_m"])
    max_cands = int(cfg["max_candidates"])
    fallback_k = int(cfg["fallback_k"])

    # Fast batch spatial query in meters
    pairs = tree.query(pt_geoms, predicate="dwithin", distance=radius_m)
    obs_hits = pairs[0]
    tree_arc_hits = pairs[1]
    arc_hits = valid_arc_ids[tree_arc_hits]

    # Identify observations with zero hits inside radius_m and apply nearest-k fallback
    hit_counts = np.bincount(obs_hits, minlength=n_obs)
    missing_obs = np.flatnonzero(hit_counts == 0)
    if len(missing_obs) > 0:
        fb_obs_list: List[np.ndarray] = []
        fb_arc_list: List[np.ndarray] = []
        # Query nearest arcs for each missing observation via expanding radius or query_nearest
        for oi in missing_obs:
            pt = pt_geoms[oi]
            # Expand search radius until at least fallback_k arcs found (capped at 2000m)
            found_arcs = np.array([], dtype=np.int64)
            for r_fb in (150.0, 400.0, 1000.0, 2500.0):
                res = tree.query(pt, predicate="dwithin", distance=r_fb)
                if len(res) >= fallback_k:
                    found_arcs = valid_arc_ids[res]
                    break
                elif len(res) > 0:
                    found_arcs = valid_arc_ids[res]
            if len(found_arcs) == 0:
                nearest_idx = tree.query_nearest(pt, all_matches=True)
                found_arcs = valid_arc_ids[nearest_idx[:fallback_k]]
            fb_obs_list.append(np.full(len(found_arcs), oi, dtype=np.int64))
            fb_arc_list.append(found_arcs.astype(np.int64))
        obs_hits = np.concatenate([obs_hits, np.concatenate(fb_obs_list)])
        arc_hits = np.concatenate([arc_hits, np.concatenate(fb_arc_list)])

    # Compute exact point-to-segment distance and fractional projection for all candidate pairs
    dist_m, proj_frac = point_to_segment_metrics(
        track_x[obs_hits],
        track_y[obs_hits],
        arc_x1[arc_hits],
        arc_y1[arc_hits],
        arc_x2[arc_hits],
        arc_y2[arc_hits],
    )

    # Sort by (obs_id, dist_m, arc_id) and keep top max_candidates per observation
    order = np.lexsort((arc_hits, dist_m, obs_hits))
    obs_hits = obs_hits[order]
    arc_hits = arc_hits[order]
    dist_m = dist_m[order]
    proj_frac = proj_frac[order]

    # Compute within-observation rank & filter to <= max_candidates
    is_new_obs = np.empty(len(obs_hits), dtype=bool)
    is_new_obs[0] = True
    is_new_obs[1:] = obs_hits[1:] != obs_hits[:-1]
    group_starts = np.flatnonzero(is_new_obs)
    group_id = np.cumsum(is_new_obs) - 1
    rank0 = np.arange(len(obs_hits)) - group_starts[group_id]

    keep = rank0 < max_cands
    obs_hits = obs_hits[keep]
    arc_hits = arc_hits[keep]
    dist_m = dist_m[keep]
    proj_frac = proj_frac[keep]
    dist_rank = (rank0[keep] + 1).astype(np.float32)

    # Recompute group boundaries after top-k truncation
    is_new_obs = np.empty(len(obs_hits), dtype=bool)
    is_new_obs[0] = True
    is_new_obs[1:] = obs_hits[1:] != obs_hits[:-1]
    group_starts = np.flatnonzero(is_new_obs)
    group_ends = np.append(group_starts[1:], len(obs_hits))
    group_sizes = group_ends - group_starts
    group_id = np.cumsum(is_new_obs) - 1

    min_dist_per_obs = dist_m[group_starts][group_id]
    n_candidates = group_sizes[group_id].astype(np.float32)
    dist_diff_from_best = (dist_m - min_dist_per_obs).astype(np.float32)
    dist_ratio_to_best = ((dist_m + 1.0) / (min_dist_per_obs + 1.0)).astype(np.float32)

    # Kinematics (strictly from .track)
    kin = compute_track_kinematics(
        track_x,
        track_y,
        trip.track["timestamp"].to_numpy(dtype=np.float64),
        cfg=cfg,
    )

    h_valid = kin["heading_valid"][obs_hits]
    obs_b = kin["obs_bearing"][obs_hits]
    arc_b = arc_bearing[arc_hits]
    # Angular difference in [0, pi]
    raw_diff = np.abs(np.arctan2(np.sin(obs_b - arc_b), np.cos(obs_b - arc_b)))
    heading_diff_deg = np.where(h_valid > 0.5, np.degrees(raw_diff), 90.0).astype(np.float32)
    cos_heading_diff = np.where(h_valid > 0.5, np.cos(raw_diff), 0.0).astype(np.float32)

    # Heading rank within each observation's candidates
    h_order = np.lexsort((dist_m, heading_diff_deg, obs_hits))
    h_rank = np.empty(len(obs_hits), dtype=np.float32)
    h_rank[h_order] = (np.arange(len(obs_hits)) - group_starts[group_id] + 1).astype(np.float32)

    # Adjacent observation distances to the same candidate arc (strictly .track + .arcs)
    prev_obs = kin["prev_idx"][obs_hits]
    next_obs = kin["next_idx"][obs_hits]
    prev_dist_m, _ = point_to_segment_metrics(
        track_x[prev_obs],
        track_y[prev_obs],
        arc_x1[arc_hits],
        arc_y1[arc_hits],
        arc_x2[arc_hits],
        arc_y2[arc_hits],
    )
    next_dist_m, _ = point_to_segment_metrics(
        track_x[next_obs],
        track_y[next_obs],
        arc_x1[arc_hits],
        arc_y1[arc_hits],
        arc_x2[arc_hits],
        arc_y2[arc_hits],
    )

    # Signed progress of trajectory (prev -> next) projected along candidate directed arc unit vector
    safe_arc_len = np.maximum(arc_len[arc_hits], 1e-6)
    ux = (arc_x2[arc_hits] - arc_x1[arc_hits]) / safe_arc_len
    uy = (arc_y2[arc_hits] - arc_y1[arc_hits]) / safe_arc_len
    move_dx = track_x[next_obs] - track_x[prev_obs]
    move_dy = track_y[next_obs] - track_y[prev_obs]
    along_arc_progress_m = np.clip(move_dx * ux + move_dy * uy, -100.0, 100.0).astype(np.float32)

    # Road graph topology cues (strictly from .arcs)
    u_nodes = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    v_nodes = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
    n_nodes = len(trip.nodes)
    out_deg = np.bincount(u_nodes[(u_nodes >= 0) & (u_nodes < n_nodes)], minlength=n_nodes).astype(np.float32)

    u_hit = u_nodes[arc_hits]
    v_hit = v_nodes[arc_hits]
    from_node_out_deg = out_deg[u_hit]
    to_node_out_deg = out_deg[v_hit]

    # Whether the reverse directed arc (v -> u) exists in the road network
    edge_set = set(zip(u_nodes.tolist(), v_nodes.tolist()))
    has_reverse_arc = np.fromiter(
        (1.0 if (int(v), int(u)) in edge_set and u != v else 0.0 for u, v in zip(u_hit, v_hit)),
        dtype=np.float32,
        count=len(arc_hits),
    )

    df = pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "obs_id": obs_hits.astype(np.int32),
            "arc_id": arc_hits.astype(np.int32),
            "dist_m": dist_m.astype(np.float32),
            "dist_rank": dist_rank,
            "dist_diff_from_best": dist_diff_from_best,
            "dist_ratio_to_best": dist_ratio_to_best,
            "n_candidates": n_candidates,
            "heading_diff_deg": heading_diff_deg,
            "cos_heading_diff": cos_heading_diff,
            "heading_valid": h_valid,
            "is_stationary": kin["is_stationary"][obs_hits],
            "obs_speed_mps": kin["obs_speed_mps"][obs_hits],
            "dt_prev_s": kin["dt_prev_s"][obs_hits],
            "dt_next_s": kin["dt_next_s"][obs_hits],
            "proj_frac": proj_frac.astype(np.float32),
            "arc_len_m": arc_len[arc_hits].astype(np.float32),
            "has_reverse_arc": has_reverse_arc,
            "heading_rank": h_rank,
            "prev_dist_m": prev_dist_m.astype(np.float32),
            "next_dist_m": next_dist_m.astype(np.float32),
            "along_arc_progress_m": along_arc_progress_m,
            "from_node_out_deg": from_node_out_deg,
            "to_node_out_deg": to_node_out_deg,
        }
    )
    return df
