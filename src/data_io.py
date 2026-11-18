"""Data discovery, headerless file parsing, and local metric projection utilities for PathSnap."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TripData:
    """In-memory representation of one PathSnap trip."""

    trip_id: str
    trip_dir: Path
    track: pd.DataFrame  # columns: obs_id, lon, lat, timestamp
    nodes: pd.DataFrame  # columns: node_id, lon, lat
    arcs: pd.DataFrame  # columns: arc_id, from_node_id, to_node_id
    route: Optional[np.ndarray]  # 1D int array of chronological reference arc_ids, or None
    parse_issues: Dict[str, int]


def discover_dataset_dirs(search_root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Recursively discover one canonical (train, student_test) directory pair under Guidelines.

    Avoids double-counting duplicate folders by selecting the shortest-path canonical parent
    that contains both 'train' and 'student_test' subdirectories with trip folders.
    """
    root = Path(search_root) if search_root is not None else PROJECT_ROOT / "Guidelines"
    if not root.exists():
        raise FileNotFoundError(f"Guidelines directory not found at {root}")

    candidates: List[Tuple[int, Path, Path]] = []
    for dirpath, dirnames, _ in os.walk(root):
        dset = set(dirnames)
        if "train" in dset and "student_test" in dset:
            parent = Path(dirpath)
            train_dir = parent / "train"
            test_dir = parent / "student_test"
            train_trips = [p for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
            test_trips = [p for p in test_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if len(train_trips) > 0 and len(test_trips) > 0:
                depth = len(parent.relative_to(root).parts)
                candidates.append((depth, train_dir, test_dir))

    if not candidates:
        raise FileNotFoundError(f"No train/student_test directory pair found under {root}")

    candidates.sort(key=lambda x: (x[0], str(x[1])))
    return candidates[0][1], candidates[0][2]


def list_trip_ids(split_dir: Path) -> List[str]:
    """Return sorted trip IDs (preserving leading zeros) in a split directory."""
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Trip directory does not exist: {split_dir}")
    trip_ids = sorted(
        p.name for p in split_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return trip_ids


def _resolve_trip_file(trip_dir: Path, trip_id: str, ext: str) -> Optional[Path]:
    """Find <trip_id>.<ext> (or fallback *.<ext>) inside trip_dir."""
    exact = trip_dir / f"{trip_id}.{ext}"
    if exact.exists():
        return exact
    matches = sorted(trip_dir.glob(f"*.{ext}"))
    return matches[0] if matches else None


def parse_track_file(path: Path) -> Tuple[pd.DataFrame, int]:
    """Parse headerless .track file: longitude, latitude, timestamp(s)."""
    rows: List[Tuple[float, float, float]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                bad_rows += 1
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                ts = float(parts[2])
                rows.append((lon, lat, ts))
            except ValueError:
                bad_rows += 1

    df = pd.DataFrame(rows, columns=["lon", "lat", "timestamp"])
    df.insert(0, "obs_id", np.arange(len(df), dtype=np.int64))
    return df, bad_rows


def parse_nodes_file(path: Path) -> Tuple[pd.DataFrame, int]:
    """Parse headerless .nodes file: longitude, latitude (0-based row index = node_id)."""
    rows: List[Tuple[float, float]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                bad_rows += 1
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                rows.append((lon, lat))
            except ValueError:
                bad_rows += 1

    df = pd.DataFrame(rows, columns=["lon", "lat"])
    df.insert(0, "node_id", np.arange(len(df), dtype=np.int64))
    return df, bad_rows


def parse_arcs_file(path: Path) -> Tuple[pd.DataFrame, int]:
    """Parse headerless .arcs file: from_node_id, to_node_id (0-based row index = arc_id)."""
    rows: List[Tuple[int, int]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                bad_rows += 1
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                rows.append((u, v))
            except ValueError:
                bad_rows += 1

    df = pd.DataFrame(rows, columns=["from_node_id", "to_node_id"])
    df.insert(0, "arc_id", np.arange(len(df), dtype=np.int64))
    return df, bad_rows


def parse_route_file(path: Path) -> Tuple[np.ndarray, int]:
    """Parse headerless .route file: 1 integer column of chronological arc_id values."""
    vals: List[int] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                vals.append(int(parts[0]))
            except ValueError:
                bad_rows += 1
    return np.asarray(vals, dtype=np.int64), bad_rows


def load_trip(
    split_dir: Path,
    trip_id: str,
    load_route: bool = True,
) -> TripData:
    """Load one trip from split_dir/<trip_id>.

    Set load_route=False inside inference pipelines to guarantee reference routes are never touched.
    """
    trip_id = str(trip_id)
    trip_dir = split_dir / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip folder not found: {trip_dir}")

    track_path = _resolve_trip_file(trip_dir, trip_id, "track")
    nodes_path = _resolve_trip_file(trip_dir, trip_id, "nodes")
    arcs_path = _resolve_trip_file(trip_dir, trip_id, "arcs")
    route_path = _resolve_trip_file(trip_dir, trip_id, "route") if load_route else None

    issues: Dict[str, int] = {
        "track_bad_rows": 0,
        "nodes_bad_rows": 0,
        "arcs_bad_rows": 0,
        "route_bad_rows": 0,
    }

    if track_path is None:
        raise FileNotFoundError(f"Missing .track file in {trip_dir}")
    if nodes_path is None:
        raise FileNotFoundError(f"Missing .nodes file in {trip_dir}")
    if arcs_path is None:
        raise FileNotFoundError(f"Missing .arcs file in {trip_dir}")

    track_df, issues["track_bad_rows"] = parse_track_file(track_path)
    nodes_df, issues["nodes_bad_rows"] = parse_nodes_file(nodes_path)
    arcs_df, issues["arcs_bad_rows"] = parse_arcs_file(arcs_path)

    route_arr: Optional[np.ndarray] = None
    if load_route and route_path is not None:
        route_arr, issues["route_bad_rows"] = parse_route_file(route_path)

    return TripData(
        trip_id=trip_id,
        trip_dir=trip_dir,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_arr,
        parse_issues=issues,
    )


def build_local_projector(lon: np.ndarray, lat: np.ndarray) -> Transformer:
    """Build a WGS84 -> local Azimuthal Equidistant (meters) projection centered on median coords.

    Using a trip-local Azimuthal Equidistant projection (+proj=aeqd) centered on the trip's
    median (lon, lat) avoids UTM zone-boundary discontinuities and gives sub-centimeter local
    metric accuracy anywhere on Earth.
    """
    valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    if not np.any(valid):
        lon0, lat0 = 0.0, 0.0
    else:
        lon0 = float(np.median(lon[valid]))
        lat0 = float(np.median(lat[valid]))
    crs_wgs84 = CRS.from_epsg(4326)
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(crs_wgs84, crs_local, always_xy=True)


def project_trip_geometry(trip: TripData) -> Dict[str, np.ndarray]:
    """Project track points and arc endpoints into trip-local meters.

    Returns dictionary with:
      track_x, track_y: (N_obs,) meters
      node_x, node_y: (N_nodes,) meters
      arc_x1, arc_y1, arc_x2, arc_y2: (N_arcs,) meters
      arc_len: (N_arcs,) meters
      arc_bearing: (N_arcs,) radians in [-pi, pi] (atan2(dx, dy) = compass bearing from North)
      valid_arc_mask: (N_arcs,) bool indicating both endpoint node IDs exist and have finite coords
    """
    all_lon = np.concatenate([trip.track["lon"].to_numpy(dtype=np.float64), trip.nodes["lon"].to_numpy(dtype=np.float64)])
    all_lat = np.concatenate([trip.track["lat"].to_numpy(dtype=np.float64), trip.nodes["lat"].to_numpy(dtype=np.float64)])
    transformer = build_local_projector(all_lon, all_lat)

    track_x, track_y = transformer.transform(
        trip.track["lon"].to_numpy(dtype=np.float64),
        trip.track["lat"].to_numpy(dtype=np.float64),
    )
    node_x, node_y = transformer.transform(
        trip.nodes["lon"].to_numpy(dtype=np.float64),
        trip.nodes["lat"].to_numpy(dtype=np.float64),
    )
    track_x = np.asarray(track_x, dtype=np.float64)
    track_y = np.asarray(track_y, dtype=np.float64)
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)

    n_nodes = len(trip.nodes)
    u = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    v = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
    in_bounds = (u >= 0) & (u < n_nodes) & (v >= 0) & (v < n_nodes)

    u_safe = np.where(in_bounds, u, 0)
    v_safe = np.where(in_bounds, v, 0)

    arc_x1 = np.where(in_bounds, node_x[u_safe], np.nan)
    arc_y1 = np.where(in_bounds, node_y[u_safe], np.nan)
    arc_x2 = np.where(in_bounds, node_x[v_safe], np.nan)
    arc_y2 = np.where(in_bounds, node_y[v_safe], np.nan)

    dx = arc_x2 - arc_x1
    dy = arc_y2 - arc_y1
    arc_len = np.hypot(dx, dy)
    # Compass bearing in radians: 0 = North (+y), pi/2 = East (+x)
    arc_bearing = np.arctan2(dx, dy)
    valid_arc_mask = in_bounds & np.isfinite(arc_x1) & np.isfinite(arc_y1) & np.isfinite(arc_x2) & np.isfinite(arc_y2)

    return {
        "track_x": track_x,
        "track_y": track_y,
        "node_x": node_x,
        "node_y": node_y,
        "arc_x1": arc_x1,
        "arc_y1": arc_y1,
        "arc_x2": arc_x2,
        "arc_y2": arc_y2,
        "arc_len": arc_len,
        "arc_bearing": arc_bearing,
        "valid_arc_mask": valid_arc_mask,
    }
