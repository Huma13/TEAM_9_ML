"""Milestone 2 (Development/Server-side Scorer ONLY): Monotonic DP alignment of .track to .route.

IMPORTANT SEPARATION:
The `.route` file is an ordered chronological sequence of traversed directed arc_ids, NOT a
1-to-1 label per GPS observation. This module aligns each GPS observation to its corresponding
chronological position in `.route` using monotonic dynamic programming (constrained Viterbi over
route positions j = 0..M-1), explicitly handling:
  - Stops / low-speed segments (zero penalty for staying on the same route index j -> j)
  - Repeated road segments (state is route index j, keeping distinct passes separate)
  - Intersections (emission combines metric distance + directional heading alignment when moving)
  - GPS sampling gaps (step-length consistency relaxes during dt > 2s gaps and allows forward skips)
  - Severe GPS outliers (observations > max_align_dist_m from their aligned arc are marked unscored)

NEVER import this module inside inference pipelines!
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.candidates import compute_track_kinematics, point_to_segment_metrics
from src.data_io import TripData, project_trip_geometry

ALIGNMENT_CONFIG: Dict[str, Any] = {
    "max_align_dist_m": 50.0,        # Observations farther than 50 m from aligned route arc -> unscored
    "heading_penalty_m": 12.0,       # Equivalent meters penalty for opposite heading when moving
    "step_mismatch_weight": 0.15,    # Penalty per meter difference between route advance and GPS step
    "gap_step_mismatch_weight": 0.03,# Relaxed penalty during GPS gaps (dt > 2.0s)
    "junction_tie_dist_m": 3.0,      # Diagnostic flag: within 3m of shared junction & <1m distance tie
    "junction_tie_diff_m": 1.0,
}


def align_track_to_route(
    trip: TripData,
    geom: Dict[str, np.ndarray] | None = None,
    cfg: Dict[str, Any] = ALIGNMENT_CONFIG,
) -> pd.DataFrame:
    """Align GPS observations of one trip to its chronological reference `.route` arc sequence.

    Returns a DataFrame with one row per observation:
      trip_id, obs_id, aligned_route_idx, ref_arc_id, aligned_dist_m, aligned_proj_frac,
      is_scored, unscored_reason, is_junction_near_tie
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference .route available for alignment.")
    if geom is None:
        geom = project_trip_geometry(trip)

    track_x = geom["track_x"]
    track_y = geom["track_y"]
    ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
    route_arcs = trip.route.astype(np.int64)

    T = len(track_x)
    M = len(route_arcs)

    rx1 = geom["arc_x1"][route_arcs]
    ry1 = geom["arc_y1"][route_arcs]
    rx2 = geom["arc_x2"][route_arcs]
    ry2 = geom["arc_y2"][route_arcs]
    rlen = geom["arc_len"][route_arcs]
    rbearing = geom["arc_bearing"][route_arcs]

    # Cumulative route distance at start of each route segment (meters)
    cum_start = np.concatenate([[0.0], np.cumsum(rlen[:-1])])

    # Kinematics for heading disambiguation at turns/two-way overlaps
    kin = compute_track_kinematics(track_x, track_y, ts)
    obs_bearing = kin["obs_bearing"]
    h_valid = kin["heading_valid"]

    # Compute (T, M) distance, fractional projection, and exact cumulative route position s(t, j)
    dist_mat = np.empty((T, M), dtype=np.float32)
    proj_mat = np.empty((T, M), dtype=np.float32)
    emit_mat = np.empty((T, M), dtype=np.float32)

    heading_pen = float(cfg["heading_penalty_m"])
    chunk = 512
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        px = track_x[start:end, None]
        py = track_y[start:end, None]
        d_blk, t_blk = point_to_segment_metrics(px, py, rx1[None, :], ry1[None, :], rx2[None, :], ry2[None, :])
        dist_mat[start:end] = d_blk.astype(np.float32)
        proj_mat[start:end] = t_blk.astype(np.float32)

        # Heading mismatch penalty when vehicle is moving (0 when aligned, heading_pen when opposite)
        hv = h_valid[start:end, None]
        ob = obs_bearing[start:end, None]
        ang_diff = np.abs(np.arctan2(np.sin(ob - rbearing[None, :]), np.cos(ob - rbearing[None, :])))
        h_cost = np.where(hv > 0.5, 0.5 * (1.0 - np.cos(ang_diff)) * heading_pen, 0.0)
        emit_mat[start:end] = (d_blk + h_cost).astype(np.float32)

    # Exact projected cumulative distance along .route: s(t, j) = cum_start[j] + proj[t, j] * rlen[j]
    # Monotonic Dynamic Programming over route index j in [0..M-1]
    dp = np.full(M, np.inf, dtype=np.float64)
    backptr = np.zeros((T, M), dtype=np.int32)

    # Initialization at t=0: tiny tie-breaker preferring earlier route start without overriding distance
    dp[:] = emit_mat[0].astype(np.float64) + 0.002 * cum_start

    step_dists = np.hypot(np.diff(track_x), np.diff(track_y)) if T > 1 else np.array([])
    dts = np.diff(ts) if T > 1 else np.array([])

    w_normal = float(cfg["step_mismatch_weight"])
    w_gap = float(cfg["gap_step_mismatch_weight"])

    s_prev = cum_start + proj_mat[0].astype(np.float64) * rlen

    for t in range(1, T):
        gps_step = float(step_dists[t - 1])
        dt = float(dts[t - 1])
        is_gap = (dt > 2.0) or (gps_step > 40.0)
        w_step = w_gap if is_gap else w_normal

        s_curr = cum_start + proj_mat[t].astype(np.float64) * rlen

        # Max plausible forward route index jump at this step
        if is_gap:
            max_jump = M - 1
        else:
            max_jump = min(M - 1, max(15, int(gps_step / 5.0) + 10))

        # k=0: staying on the same route segment (zero transition penalty for stops/slow crawl)
        delta_s0 = s_curr - s_prev
        # Penalize significant backward sliding along the same arc slightly, zero penalty for staying still
        back_pen0 = w_step * np.maximum(0.0, -delta_s0 - 2.0)
        best_cost = dp + back_pen0
        best_prev = np.arange(M, dtype=np.int32)

        for k in range(1, max_jump + 1):
            # Exact projected advance along route from (t-1, j-k) to (t, j)
            route_advance = s_curr[k:] - s_prev[:-k]
            trans_pen = w_step * np.abs(route_advance - gps_step)
            cand_cost = dp[:-k] + trans_pen
            better = cand_cost < best_cost[k:]
            if np.any(better):
                idx = np.flatnonzero(better) + k
                best_cost[idx] = cand_cost[better]
                best_prev[idx] = (idx - k).astype(np.int32)

        dp = best_cost + emit_mat[t].astype(np.float64)
        backptr[t] = best_prev
        s_prev = s_curr

    # End state: tiny tie-breaker preferring finishing near the end of .route
    end_pen = 0.002 * (cum_start[-1] - cum_start)
    best_j = int(np.argmin(dp + end_pen))

    aligned_j = np.empty(T, dtype=np.int32)
    aligned_j[T - 1] = best_j
    for t in range(T - 1, 0, -1):
        aligned_j[t - 1] = backptr[t, aligned_j[t]]

    aligned_arc = route_arcs[aligned_j]
    obs_idx = np.arange(T)
    aligned_dist = dist_mat[obs_idx, aligned_j].astype(np.float64)
    aligned_proj = proj_mat[obs_idx, aligned_j].astype(np.float64)

    # Check unscored / extreme GPS outliers
    max_dist = float(cfg["max_align_dist_m"])
    is_scored = aligned_dist <= max_dist
    reasons = np.where(is_scored, "scored", "far_from_reference_route")

    # Diagnostic flag: junction near-tie between aligned route index j and adjacent j-1 or j+1
    j_tie = np.zeros(T, dtype=bool)
    tie_d = float(cfg["junction_tie_dist_m"])
    tie_diff = float(cfg["junction_tie_diff_m"])
    for t in range(T):
        j = int(aligned_j[t])
        d0 = float(aligned_dist[t])
        # Near either endpoint of the aligned arc
        end_dist = min(float(aligned_proj[t]), 1.0 - float(aligned_proj[t])) * float(rlen[j])
        if end_dist <= tie_d and d0 <= 15.0:
            neighbors = []
            if j > 0:
                neighbors.append(float(dist_mat[t, j - 1]))
            if j + 1 < M:
                neighbors.append(float(dist_mat[t, j + 1]))
            if neighbors and min(abs(d0 - dn) for dn in neighbors) <= tie_diff:
                j_tie[t] = True

    return pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "obs_id": np.arange(T, dtype=np.int32),
            "aligned_route_idx": aligned_j,
            "ref_arc_id": aligned_arc.astype(np.int32),
            "aligned_dist_m": aligned_dist.astype(np.float32),
            "aligned_proj_frac": aligned_proj.astype(np.float32),
            "is_scored": is_scored,
            "unscored_reason": reasons,
            "is_junction_near_tie": j_tie,
        }
    )
