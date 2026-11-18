"""Milestone 1: Dataset inventory, quality checks, connectivity analysis, worked example, and Folium map."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.data_io import (
    PROJECT_ROOT,
    discover_dataset_dirs,
    list_trip_ids,
    load_trip,
    project_trip_geometry,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "milestone1"


def analyze_trip(split_name: str, split_dir: Path, trip_id: str) -> Dict[str, Any]:
    trip_dir = split_dir / trip_id
    file_info = {}
    for ext in ("track", "nodes", "arcs", "route"):
        p = trip_dir / f"{trip_id}.{ext}"
        file_info[f"has_{ext}"] = p.exists()
        file_info[f"{ext}_bytes"] = p.stat().st_size if p.exists() else 0

    trip = load_trip(split_dir, trip_id, load_route=True)
    geom = project_trip_geometry(trip)

    # Coordinate checks
    t_lon = trip.track["lon"].to_numpy(dtype=np.float64)
    t_lat = trip.track["lat"].to_numpy(dtype=np.float64)
    t_ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
    n_lon = trip.nodes["lon"].to_numpy(dtype=np.float64)
    n_lat = trip.nodes["lat"].to_numpy(dtype=np.float64)

    track_invalid_coords = int(
        np.sum(~(np.isfinite(t_lon) & np.isfinite(t_lat) & (t_lon >= -180) & (t_lon <= 180) & (t_lat >= -90) & (t_lat <= 90)))
    )
    nodes_invalid_coords = int(
        np.sum(~(np.isfinite(n_lon) & np.isfinite(n_lat) & (n_lon >= -180) & (n_lon <= 180) & (n_lat >= -90) & (n_lat <= 90)))
    )
    track_invalid_ts = int(np.sum(~np.isfinite(t_ts)))

    # Sampling intervals
    dt = np.diff(t_ts) if len(t_ts) > 1 else np.array([], dtype=np.float64)
    n_negative_dt = int(np.sum(dt < 0))
    n_zero_dt = int(np.sum(dt == 0))
    n_exact_1s = int(np.sum(np.isclose(dt, 1.0, atol=1e-6)))
    n_gt_1s = int(np.sum(dt > 1.0 + 1e-6))
    n_gt_5s = int(np.sum(dt > 5.0 + 1e-6))
    n_gt_10s = int(np.sum(dt > 10.0 + 1e-6))
    max_dt = float(np.max(dt)) if len(dt) > 0 else 0.0
    median_dt = float(np.median(dt)) if len(dt) > 0 else 0.0
    duration_s = float(t_ts[-1] - t_ts[0]) if len(t_ts) > 1 else 0.0

    # Step distances in meters
    step_dx = np.diff(geom["track_x"]) if len(geom["track_x"]) > 1 else np.array([])
    step_dy = np.diff(geom["track_y"]) if len(geom["track_y"]) > 1 else np.array([])
    step_dist_m = np.hypot(step_dx, step_dy)
    total_gps_path_m = float(np.sum(step_dist_m)) if len(step_dist_m) > 0 else 0.0
    max_step_dist_m = float(np.max(step_dist_m)) if len(step_dist_m) > 0 else 0.0
    n_stationary_steps = int(np.sum(step_dist_m < 1.0))

    # Arc and node ID integrity
    n_nodes = len(trip.nodes)
    n_arcs = len(trip.arcs)
    u = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    v = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
    invalid_arc_endpoints = int(np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes)))
    self_loops = int(np.sum(u == v))
    arc_pairs = list(zip(u.tolist(), v.tolist()))
    duplicate_directed_arcs = int(len(arc_pairs) - len(set(arc_pairs)))
    arc_pair_set = set(arc_pairs)
    bidirectional_arcs = int(sum(1 for (a, b) in arc_pairs if (b, a) in arc_pair_set and a != b))

    # Connectivity via NetworkX
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    valid_edges = [(int(a), int(b)) for a, b in zip(u, v) if 0 <= a < n_nodes and 0 <= b < n_nodes]
    G.add_edges_from(valid_edges)
    isolated_nodes = int(nx.number_of_isolates(G))
    n_wcc = int(nx.number_weakly_connected_components(G)) if n_nodes > 0 else 0
    n_scc = int(nx.number_strongly_connected_components(G)) if n_nodes > 0 else 0
    largest_wcc_frac = (
        float(max(len(c) for c in nx.weakly_connected_components(G)) / n_nodes) if n_nodes > 0 else 0.0
    )

    # Reference route checks
    n_route = int(len(trip.route)) if trip.route is not None else 0
    invalid_route_ids = 0
    route_directed_transitions = 0
    route_total_transitions = 0
    route_repeated_arcs = 0
    route_total_len_m = 0.0
    if trip.route is not None and len(trip.route) > 0:
        r = trip.route
        invalid_route_ids = int(np.sum((r < 0) | (r >= n_arcs)))
        valid_r = r[(r >= 0) & (r < n_arcs)]
        route_repeated_arcs = int(len(valid_r) - len(np.unique(valid_r)))
        route_total_len_m = float(np.nansum(geom["arc_len"][valid_r]))
        if len(valid_r) > 1:
            route_total_transitions = len(valid_r) - 1
            for a_prev, a_curr in zip(valid_r[:-1], valid_r[1:]):
                if v[a_prev] == u[a_curr]:
                    route_directed_transitions += 1

    # Geographic extent
    extent_x_km = float((np.nanmax(geom["track_x"]) - np.nanmin(geom["track_x"])) / 1000.0) if len(t_lon) > 0 else 0.0
    extent_y_km = float((np.nanmax(geom["track_y"]) - np.nanmin(geom["track_y"])) / 1000.0) if len(t_lat) > 0 else 0.0

    rec: Dict[str, Any] = {
        "split": split_name,
        "trip_id": trip_id,
        **file_info,
        "n_gps": len(trip.track),
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route": n_route,
        "track_bad_rows": trip.parse_issues["track_bad_rows"],
        "nodes_bad_rows": trip.parse_issues["nodes_bad_rows"],
        "arcs_bad_rows": trip.parse_issues["arcs_bad_rows"],
        "route_bad_rows": trip.parse_issues["route_bad_rows"],
        "track_invalid_coords": track_invalid_coords,
        "nodes_invalid_coords": nodes_invalid_coords,
        "track_invalid_ts": track_invalid_ts,
        "invalid_arc_endpoints": invalid_arc_endpoints,
        "invalid_route_ids": invalid_route_ids,
        "self_loops": self_loops,
        "duplicate_directed_arcs": duplicate_directed_arcs,
        "bidirectional_arcs": bidirectional_arcs,
        "isolated_nodes": isolated_nodes,
        "n_weakly_connected_components": n_wcc,
        "n_strongly_connected_components": n_scc,
        "largest_wcc_frac": round(largest_wcc_frac, 4),
        "lon_min": round(float(np.nanmin(t_lon)), 6),
        "lon_max": round(float(np.nanmax(t_lon)), 6),
        "lat_min": round(float(np.nanmin(t_lat)), 6),
        "lat_max": round(float(np.nanmax(t_lat)), 6),
        "extent_x_km": round(extent_x_km, 3),
        "extent_y_km": round(extent_y_km, 3),
        "duration_s": round(duration_s, 2),
        "median_dt_s": round(median_dt, 3),
        "max_dt_s": round(max_dt, 2),
        "n_negative_dt": n_negative_dt,
        "n_zero_dt": n_zero_dt,
        "n_exact_1s": n_exact_1s,
        "pct_exact_1s": round(100.0 * n_exact_1s / max(len(dt), 1), 2),
        "n_gt_1s": n_gt_1s,
        "n_gt_5s": n_gt_5s,
        "n_gt_10s": n_gt_10s,
        "total_gps_path_m": round(total_gps_path_m, 1),
        "max_step_dist_m": round(max_step_dist_m, 1),
        "n_stationary_steps": n_stationary_steps,
        "route_total_len_m": round(route_total_len_m, 1),
        "route_directed_transitions": route_directed_transitions,
        "route_total_transitions": route_total_transitions,
        "route_continuity_pct": round(100.0 * route_directed_transitions / max(route_total_transitions, 1), 2),
        "route_repeated_arcs": route_repeated_arcs,
    }
    return rec, dt


def build_worked_example_00000005(train_dir: Path) -> Dict[str, Any]:
    trip_id = "00000005"
    trip = load_trip(train_dir, trip_id, load_route=True)
    geom = project_trip_geometry(trip)

    # Verify expected counts
    counts = {
        "n_gps": len(trip.track),
        "n_nodes": len(trip.nodes),
        "n_arcs": len(trip.arcs),
        "n_route": len(trip.route) if trip.route is not None else 0,
    }
    expected = {"n_gps": 1045, "n_nodes": 898, "n_arcs": 1579, "n_route": 105}
    counts_match = counts == expected

    # Trace real arc: first arc in reference route (arc_id = 176) and also arc_id = 0
    traced_arcs = []
    for arc_id in [int(trip.route[0]), 0]:
        row = trip.arcs.iloc[arc_id]
        u = int(row["from_node_id"])
        v = int(row["to_node_id"])
        nu = trip.nodes.iloc[u]
        nv = trip.nodes.iloc[v]
        traced_arcs.append(
            {
                "arc_id": arc_id,
                "in_reference_route": bool(arc_id in set(trip.route.tolist())),
                "from_node_id": u,
                "to_node_id": v,
                "from_lon_lat": [float(nu["lon"]), float(nu["lat"])],
                "to_lon_lat": [float(nv["lon"]), float(nv["lat"])],
                "length_m": round(float(geom["arc_len"][arc_id]), 3),
                "bearing_deg": round(float(np.degrees(geom["arc_bearing"][arc_id]) % 360.0), 2),
            }
        )

    # Generate Folium map for trip 00000005
    center_lat = float(trip.track["lat"].median())
    center_lon = float(trip.track["lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # Layer 1: Road network
    fg_roads = folium.FeatureGroup(name="Road Network (1,579 directed arcs)", show=True)
    nodes_lat = trip.nodes["lat"].to_numpy()
    nodes_lon = trip.nodes["lon"].to_numpy()
    for row in trip.arcs.itertuples(index=False):
        u, v = int(row.from_node_id), int(row.to_node_id)
        folium.PolyLine(
            locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
            color="#9aa5b1",
            weight=1.5,
            opacity=0.65,
        ).add_to(fg_roads)
    fg_roads.add_to(m)

    # Layer 2: Reference route
    fg_route = folium.FeatureGroup(name="Supplied Reference Route (105 arcs)", show=True)
    for step_idx, arc_id in enumerate(trip.route.tolist()):
        row = trip.arcs.iloc[arc_id]
        u, v = int(row["from_node_id"]), int(row["to_node_id"])
        folium.PolyLine(
            locations=[(float(nodes_lat[u]), float(nodes_lon[u])), (float(nodes_lat[v]), float(nodes_lon[v]))],
            color="#16a34a",
            weight=5,
            opacity=0.85,
            tooltip=f"Route step {step_idx}: arc_id={arc_id} ({u}->{v}, {geom['arc_len'][arc_id]:.1f}m)",
        ).add_to(fg_route)
    fg_route.add_to(m)

    # Layer 3: GPS readings
    fg_gps = folium.FeatureGroup(name="Raw GPS Readings (1,045 observations)", show=True)
    for row in trip.track.itertuples(index=False):
        folium.CircleMarker(
            location=(float(row.lat), float(row.lon)),
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.75,
            weight=0.5,
            tooltip=f"obs_id={row.obs_id}, t={row.timestamp:.1f}s, ({row.lon:.6f}, {row.lat:.6f})",
        ).add_to(fg_gps)
    # Start & end markers
    first_obs = trip.track.iloc[0]
    last_obs = trip.track.iloc[-1]
    folium.Marker(
        location=(float(first_obs["lat"]), float(first_obs["lon"])),
        popup=f"Start: obs_id=0, t={first_obs['timestamp']:.1f}s",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(fg_gps)
    folium.Marker(
        location=(float(last_obs["lat"]), float(last_obs["lon"])),
        popup=f"End: obs_id={len(trip.track)-1}, t={last_obs['timestamp']:.1f}s",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Layer 4: Traced worked-example arc
    traced = traced_arcs[0]
    fg_traced = folium.FeatureGroup(name=f"Traced Arc {traced['arc_id']} ({traced['from_node_id']}->{traced['to_node_id']})", show=True)
    folium.PolyLine(
        locations=[
            (traced["from_lon_lat"][1], traced["from_lon_lat"][0]),
            (traced["to_lon_lat"][1], traced["to_lon_lat"][0]),
        ],
        color="#ea580c",
        weight=7,
        opacity=0.95,
        tooltip=f"Traced arc {traced['arc_id']}: node {traced['from_node_id']} -> {traced['to_node_id']} ({traced['length_m']} m)",
    ).add_to(fg_traced)
    fg_traced.add_to(m)

    # Fit bounds tightly to GPS track + margin
    m.fit_bounds([
        [float(trip.track["lat"].min()), float(trip.track["lon"].min())],
        [float(trip.track["lat"].max()), float(trip.track["lon"].max())],
    ])
    folium.LayerControl(collapsed=False).add_to(m)

    # Legend HTML
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 2px solid #cbd5e1; border-radius: 6px;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 6px;">PathSnap — Training Trip {trip_id}</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#2563eb;border-radius:50%;margin-right:6px;"></span>Raw GPS readings (1,045)</div>
      <div><span style="display:inline-block;width:14px;height:3px;background:#9aa5b1;margin-right:6px;vertical-align:middle;"></span>Road network (1,579 directed arcs, 898 nodes)</div>
      <div><span style="display:inline-block;width:14px;height:4px;background:#16a34a;margin-right:6px;vertical-align:middle;"></span>Supplied reference route (105 arcs)</div>
      <div><span style="display:inline-block;width:14px;height:5px;background:#ea580c;margin-right:6px;vertical-align:middle;"></span>Traced example arc #{traced['arc_id']} ({traced['length_m']} m)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_path = OUT_DIR / "trip_00000005_map.html"
    m.save(str(map_path))

    # Also save a static matplotlib plot of trip 00000005 for PDF report inclusion
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for row in trip.arcs.itertuples(index=False):
        u, v = int(row.from_node_id), int(row.to_node_id)
        ax.plot([nodes_lon[u], nodes_lon[v]], [nodes_lat[u], nodes_lat[v]], color="#cbd5e1", lw=0.7, zorder=1)
    for idx, arc_id in enumerate(trip.route.tolist()):
        row = trip.arcs.iloc[arc_id]
        u, v = int(row["from_node_id"]), int(row["to_node_id"])
        ax.plot(
            [nodes_lon[u], nodes_lon[v]],
            [nodes_lat[u], nodes_lat[v]],
            color="#16a34a",
            lw=2.8,
            zorder=2,
            label="Supplied Reference Route" if idx == 0 else None,
        )
    ax.scatter(trip.track["lon"], trip.track["lat"], s=6, c="#2563eb", alpha=0.75, zorder=3, label="Raw GPS Readings (n=1,045)")
    ax.plot([], [], color="#cbd5e1", lw=1.2, label="Road Network (1,579 arcs)")
    ax.set_title("Training Trip 00000005: Raw GPS Trajectory, Road Network & Reference Route", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trip_00000005_trajectory.png")
    plt.close(fig)

    worked = {
        "trip_id": trip_id,
        "split": "train",
        "counts": counts,
        "expected_counts": expected,
        "counts_verified": counts_match,
        "traced_arcs": traced_arcs,
        "folium_map_path": str(map_path),
        "static_plot_path": str(OUT_DIR / "trip_00000005_trajectory.png"),
    }
    return worked


def plot_sampling_gaps(inv_df: pd.DataFrame, all_dts: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)

    # Left: Distribution of sampling intervals (grouped buckets)
    train_dt = all_dts["train"]
    test_dt = all_dts["student_test"]
    combined = np.concatenate([train_dt, test_dt])

    buckets = ["<1s", "1s (exact)", "(1, 2]s", "(2, 5]s", "(5, 10]s", ">10s"]
    def bucket_pcts(arr: np.ndarray) -> List[float]:
        n = max(len(arr), 1)
        return [
            100.0 * np.sum(arr < 1.0 - 1e-6) / n,
            100.0 * np.sum(np.isclose(arr, 1.0, atol=1e-6)) / n,
            100.0 * np.sum((arr > 1.0 + 1e-6) & (arr <= 2.0 + 1e-6)) / n,
            100.0 * np.sum((arr > 2.0 + 1e-6) & (arr <= 5.0 + 1e-6)) / n,
            100.0 * np.sum((arr > 5.0 + 1e-6) & (arr <= 10.0 + 1e-6)) / n,
            100.0 * np.sum(arr > 10.0 + 1e-6) / n,
        ]

    x = np.arange(len(buckets))
    w = 0.36
    p_train = bucket_pcts(train_dt)
    p_test = bucket_pcts(test_dt)
    b1 = axes[0].bar(x - w / 2, p_train, w, label=f"Train (n={len(train_dt):,} steps)", color="#2563eb")
    b2 = axes[0].bar(x + w / 2, p_test, w, label=f"Student Test (n={len(test_dt):,} steps)", color="#f59e0b")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-3, 150)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets, rotation=15)
    axes[0].set_ylabel("Share of Inter-Observation Steps (%, log scale)")
    axes[0].set_title("GPS Sampling Interval Distribution (90 Dev Trips)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(b1, p_train):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, val * 1.15, f"{val:.2f}%", ha="center", va="bottom", fontsize=7)

    # Right: Top 15 trips by maximum GPS sampling gap
    top_gaps = inv_df.sort_values("max_dt_s", ascending=False).head(15)
    colors = ["#f59e0b" if s == "student_test" else "#2563eb" for s in top_gaps["split"]]
    y = np.arange(len(top_gaps))
    axes[1].barh(y, top_gaps["max_dt_s"], color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{tid} ({sp[:4]})" for tid, sp in zip(top_gaps["trip_id"], top_gaps["split"])], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Maximum Sampling Gap (seconds)")
    axes[1].set_title("Top 15 Trips by Largest GPS Timestamp Gap")
    axes[1].grid(True, axis="x", alpha=0.3)
    for idx, val in enumerate(top_gaps["max_dt_s"]):
        axes[1].text(val + 0.5, idx, f"{val:.0f}s", va="center", fontsize=8)

    fig.suptitle(
        f"PathSnap GPS Sampling Gaps — Exact 1s: {100.0 * np.mean(np.isclose(combined, 1.0, atol=1e-6)):.2f}%, Max Gap: {np.max(combined):.0f}s",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sampling_gaps.png")
    plt.close(fig)


def plot_eda_overview(inv_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)

    train_df = inv_df[inv_df["split"] == "train"]
    test_df = inv_df[inv_df["split"] == "student_test"]

    # 1. GPS observations per trip
    axes[0, 0].hist(train_df["n_gps"], bins=18, alpha=0.75, color="#2563eb", label=f"Train (n=80, med={train_df['n_gps'].median():.0f})")
    axes[0, 0].hist(test_df["n_gps"], bins=10, alpha=0.75, color="#f59e0b", label=f"Student Test (n=10, med={test_df['n_gps'].median():.0f})")
    axes[0, 0].set_title("GPS Observations per Trip")
    axes[0, 0].set_xlabel("Number of GPS Readings")
    axes[0, 0].set_ylabel("Trip Count")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(True, alpha=0.25)

    # 2. Road network size: Nodes vs Directed Arcs
    axes[0, 1].scatter(train_df["n_nodes"], train_df["n_arcs"], alpha=0.75, color="#2563eb", label="Train (80)")
    axes[0, 1].scatter(test_df["n_nodes"], test_df["n_arcs"], alpha=0.9, color="#f59e0b", marker="^", s=55, label="Student Test (10)")
    axes[0, 1].set_title("Road Network Size: Nodes vs Directed Arcs")
    axes[0, 1].set_xlabel("Nodes per Trip")
    axes[0, 1].set_ylabel("Directed Arcs per Trip")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.25)

    # 3. Geographic centroids (Lon vs Lat)
    lon_mid = 0.5 * (inv_df["lon_min"] + inv_df["lon_max"])
    lat_mid = 0.5 * (inv_df["lat_min"] + inv_df["lat_max"])
    is_test = inv_df["split"] == "student_test"
    axes[1, 0].scatter(lon_mid[~is_test], lat_mid[~is_test], alpha=0.75, color="#2563eb", label="Train (80)")
    axes[1, 0].scatter(lon_mid[is_test], lat_mid[is_test], alpha=0.95, color="#f59e0b", marker="^", s=60, label="Student Test (10)")
    axes[1, 0].set_title("Geographic Centroids of All 90 Development Trips")
    axes[1, 0].set_xlabel("Median Longitude (deg)")
    axes[1, 0].set_ylabel("Median Latitude (deg)")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, alpha=0.25)

    # 4. Reference route length vs GPS observations
    axes[1, 1].scatter(train_df["n_gps"], train_df["n_route"], alpha=0.75, color="#16a34a", label="Train (80)")
    axes[1, 1].scatter(test_df["n_gps"], test_df["n_route"], alpha=0.9, color="#dc2626", marker="^", s=55, label="Student Test (10)")
    axes[1, 1].set_title("GPS Readings vs Reference Route Arc Entries")
    axes[1, 1].set_xlabel("GPS Observations (.track rows)")
    axes[1, 1].set_ylabel("Reference Route Segments (.route rows)")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle("PathSnap Dataset Exploratory Analysis (80 Train + 10 Student-Test Trips)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_eda_overview.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir = discover_dataset_dirs()
    train_ids = list_trip_ids(train_dir)
    test_ids = list_trip_ids(test_dir)

    print(f"Discovered train dir: {train_dir} ({len(train_ids)} trips)")
    print(f"Discovered student_test dir: {test_dir} ({len(test_ids)} trips)")
    assert len(train_ids) == 80, f"Expected 80 train trips, got {len(train_ids)}"
    assert len(test_ids) == 10, f"Expected 10 student_test trips, got {len(test_ids)}"
    assert len(set(train_ids).intersection(test_ids)) == 0, "Train and student_test trip IDs overlap!"

    rows: List[Dict[str, Any]] = []
    all_dts: Dict[str, List[np.ndarray]] = {"train": [], "student_test": []}

    for split_name, split_dir, tids in [("train", train_dir, train_ids), ("student_test", test_dir, test_ids)]:
        for tid in tids:
            rec, dt = analyze_trip(split_name, split_dir, tid)
            rows.append(rec)
            all_dts[split_name].append(dt)

    inv_df = pd.DataFrame(rows)
    inv_path = OUT_DIR / "dataset_inventory.csv"
    inv_df.to_csv(inv_path, index=False)

    dt_concat = {k: np.concatenate(v) if v else np.array([]) for k, v in all_dts.items()}
    plot_sampling_gaps(inv_df, dt_concat)
    plot_eda_overview(inv_df)

    worked = build_worked_example_00000005(train_dir)
    worked_path = OUT_DIR / "worked_example_00000005.json"
    worked_path.write_text(json.dumps(worked, indent=2), encoding="utf-8")

    # Build concise summary
    combined_dt = np.concatenate([dt_concat["train"], dt_concat["student_test"]])
    summary = {
        "python_version": sys.version.split()[0],
        "train_dir": str(train_dir),
        "student_test_dir": str(test_dir),
        "n_train_trips": len(train_ids),
        "n_student_test_trips": len(test_ids),
        "overlap_trip_ids": 0,
        "all_files_present": bool(inv_df[["has_track", "has_nodes", "has_arcs", "has_route"]].all().all()),
        "total_gps_train": int(inv_df.loc[inv_df["split"] == "train", "n_gps"].sum()),
        "total_gps_student_test": int(inv_df.loc[inv_df["split"] == "student_test", "n_gps"].sum()),
        "total_nodes_all": int(inv_df["n_nodes"].sum()),
        "total_arcs_all": int(inv_df["n_arcs"].sum()),
        "total_route_entries_all": int(inv_df["n_route"].sum()),
        "total_parse_bad_rows": int(inv_df[["track_bad_rows", "nodes_bad_rows", "arcs_bad_rows", "route_bad_rows"]].sum().sum()),
        "total_invalid_coords": int(inv_df[["track_invalid_coords", "nodes_invalid_coords"]].sum().sum()),
        "total_invalid_arc_endpoints": int(inv_df["invalid_arc_endpoints"].sum()),
        "total_invalid_route_ids": int(inv_df["invalid_route_ids"].sum()),
        "total_negative_dt": int(inv_df["n_negative_dt"].sum()),
        "total_zero_dt": int(inv_df["n_zero_dt"].sum()),
        "pct_exact_1s_all": round(float(100.0 * np.mean(np.isclose(combined_dt, 1.0, atol=1e-6))), 3),
        "max_dt_s_all": float(np.max(combined_dt)),
        "trips_with_dt_gt_5s": int((inv_df["n_gt_5s"] > 0).sum()),
        "trips_with_dt_gt_10s": int((inv_df["n_gt_10s"] > 0).sum()),
        "trips_with_isolated_nodes": int((inv_df["isolated_nodes"] > 0).sum()),
        "total_isolated_nodes": int(inv_df["isolated_nodes"].sum()),
        "trips_with_multiple_wcc": int((inv_df["n_weakly_connected_components"] > 1).sum()),
        "trips_with_self_loops": int((inv_df["self_loops"] > 0).sum()),
        "trips_with_duplicate_arcs": int((inv_df["duplicate_directed_arcs"] > 0).sum()),
        "mean_route_continuity_pct": round(float(inv_df["route_continuity_pct"].mean()), 3),
        "min_route_continuity_pct": round(float(inv_df["route_continuity_pct"].min()), 3),
        "trips_with_repeated_route_arcs": int((inv_df["route_repeated_arcs"] > 0).sum()),
        "lon_range": [float(inv_df["lon_min"].min()), float(inv_df["lon_max"].max())],
        "lat_range": [float(inv_df["lat_min"].min()), float(inv_df["lat_max"].max())],
        "worked_example_00000005": worked,
    }
    summary_path = OUT_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== MILESTONE 1 SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
