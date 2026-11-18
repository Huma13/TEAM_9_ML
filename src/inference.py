"""Pure inference pipeline and strictly separated server-side scorer for PathSnap."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.alignment import align_track_to_route
from src.candidates import generate_trip_candidates
from src.data_io import PROJECT_ROOT, TripData, load_trip, project_trip_geometry
from src.hmm_viterbi import DEFAULT_HMM_CONFIG, viterbi_decode_trip
from src.metrics import (
    compress_arc_sequence,
    compute_route_continuity_for_trip,
    compute_sequence_metrics_for_trip,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class PathSnapPredictor:
    """Self-contained inference engine backed solely by saved artifacts (never reads .route)."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.bundle_path = self.artifacts_dir / "model_bundle.joblib"
        self.hmm_config_path = self.artifacts_dir / "hmm_config.json"

        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Saved model bundle not found at {self.bundle_path}")

        bundle = joblib.load(self.bundle_path)
        self.model = bundle["model"]
        self.model_key: str = bundle["model_key"]
        self.model_name: str = bundle["model_name"]
        self.feature_columns: List[str] = list(bundle["feature_columns"])
        self.candidate_config: Dict[str, Any] = dict(bundle["candidate_config"])
        self.best_params: Dict[str, Any] = dict(bundle.get("best_params", {}))

        if self.hmm_config_path.exists():
            self.hmm_config: Dict[str, Any] = json.loads(self.hmm_config_path.read_text(encoding="utf-8"))
        else:
            self.hmm_config = dict(DEFAULT_HMM_CONFIG)

    def predict_trip(self, split_dir: Path, trip_id: str) -> Dict[str, Any]:
        """Run end-to-end prediction strictly WITHOUT loading any .route file."""
        t_start = time.perf_counter()

        # STRICT ISOLATION: load_route=False guarantees reference route is never read by predictor
        trip = load_trip(split_dir, trip_id, load_route=False)
        assert trip.route is None, "Reference route must never enter the prediction pipeline!"

        geom = project_trip_geometry(trip)

        t_c0 = time.perf_counter()
        cands = generate_trip_candidates(trip, geom=geom, cfg=self.candidate_config)
        cand_ms = (time.perf_counter() - t_c0) * 1000.0

        t_m0 = time.perf_counter()
        if len(cands) > 0:
            X = cands[self.feature_columns].to_numpy(dtype=np.float32)
            probs = self.model.predict_proba(X)[:, 1].astype(np.float64)
        else:
            probs = np.empty(0, dtype=np.float64)
        clf_ms = (time.perf_counter() - t_m0) * 1000.0

        t_h0 = time.perf_counter()
        dec = viterbi_decode_trip(trip, cands, probs, geom=geom, cfg=self.hmm_config)
        hmm_ms = (time.perf_counter() - t_h0) * 1000.0

        pred_arcs: np.ndarray = dec["pred_arcs"]
        pred_probs: np.ndarray = dec["pred_probs"]

        fn = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
        tn = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
        arc_len = geom["arc_len"]
        cont = compute_route_continuity_for_trip(pred_arcs, fn, tn)

        # Build coordinate payloads (explicit lon/lat fields + documented coordinate_order)
        nodes_lon = trip.nodes["lon"].to_numpy(dtype=np.float64)
        nodes_lat = trip.nodes["lat"].to_numpy(dtype=np.float64)

        gps_points = []
        for row, pa, pp in zip(trip.track.itertuples(index=False), pred_arcs.tolist(), pred_probs.tolist()):
            gps_points.append(
                {
                    "obs_id": int(row.obs_id),
                    "lon": round(float(row.lon), 6),
                    "lat": round(float(row.lat), 6),
                    "timestamp_s": round(float(row.timestamp), 3),
                    "pred_arc_id": int(pa) if pa >= 0 else None,
                    "pred_prob": round(float(pp), 4) if pa >= 0 else None,
                }
            )

        compressed_arcs = compress_arc_sequence(pred_arcs)
        predicted_route_geometry = []
        total_pred_len_m = 0.0
        for step_idx, aid in enumerate(compressed_arcs):
            u, v = int(fn[aid]), int(tn[aid])
            seg_len = float(arc_len[aid])
            total_pred_len_m += seg_len
            predicted_route_geometry.append(
                {
                    "step_index": step_idx,
                    "arc_id": int(aid),
                    "from_node_id": u,
                    "to_node_id": v,
                    "coordinates_lon_lat": [
                        [round(float(nodes_lon[u]), 6), round(float(nodes_lat[u]), 6)],
                        [round(float(nodes_lon[v]), 6), round(float(nodes_lat[v]), 6)],
                    ],
                    "length_m": round(seg_len, 2),
                }
            )

        # Relevant local road network geometry (arcs present in candidate set, capped for payload lightness)
        rel_arcs = sorted(set(int(a) for a in cands["arc_id"].unique().tolist() if 0 <= int(a) < len(fn)))
        road_network_context = []
        for aid in rel_arcs:
            u, v = int(fn[aid]), int(tn[aid])
            road_network_context.append(
                {
                    "arc_id": aid,
                    "coordinates_lon_lat": [
                        [round(float(nodes_lon[u]), 6), round(float(nodes_lat[u]), 6)],
                        [round(float(nodes_lon[v]), 6), round(float(nodes_lat[v]), 6)],
                    ],
                }
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "trip_id": trip.trip_id,
            "gps_points": gps_points,
            "predicted_route_geometry": predicted_route_geometry,
            "road_network_context": road_network_context,
            "pred_arcs_internal": pred_arcs,  # stripped before HTTP response; passed to server-side scorer
            "cands_internal": cands,          # stripped before HTTP response; passed to server-side scorer
            "geom_internal": geom,            # stripped before HTTP response; passed to server-side scorer
            "inference_metrics": {
                "n_gps_points": int(len(trip.track)),
                "n_candidates_evaluated": int(len(cands)),
                "n_predicted_route_segments": int(len(compressed_arcs)),
                "predicted_route_length_m": round(total_pred_len_m, 2),
                "route_continuity": round(float(cont["route_continuity"]), 5),
                "transition_continuity_changes_only": round(float(cont["transition_continuity_changes_only"]), 5),
                "n_missing_candidate_obs": int(dec["n_missing_candidate_obs"]),
                "n_gap_resets": int(dec["n_gap_resets"]),
                "candidate_generation_ms": round(cand_ms, 2),
                "classifier_ms": round(clf_ms, 2),
                "viterbi_decode_ms": round(hmm_ms, 2),
            },
            "elapsed_time_ms": round(total_ms, 2),
        }


def score_prediction_server_side(
    split_dir: Path,
    trip_id: str,
    pred_arcs: np.ndarray,
    cands_df: Any,
    geom: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Separate server-side scorer invoked strictly AFTER predictions are finalized.

    Computes aggregate scalar metrics against server-side `.route` if available.
    NEVER returns reference route IDs, alignments, or reference geometry to the client.
    """
    trip_with_route = load_trip(split_dir, trip_id, load_route=True)
    if trip_with_route.route is None or len(trip_with_route.route) == 0:
        return {
            "evaluation_available": False,
            "pointwise_top1_accuracy": None,
            "candidate_miss_rate": None,
            "sequence_lcs_f1": None,
            "sequence_length_weighted_f1": None,
            "sequence_arc_f1": None,
            "n_scored_observations": None,
            "label_coverage_pct": None,
        }

    aligned = align_track_to_route(trip_with_route, geom=geom)
    ref_arcs = aligned["ref_arc_id"].to_numpy(dtype=np.int64)
    scored_mask = aligned["is_scored"].to_numpy(dtype=bool)
    n_scored = int(np.sum(scored_mask))

    # Candidate coverage check
    cand_pairs = set(zip(cands_df["obs_id"].to_numpy(dtype=np.int64).tolist(), cands_df["arc_id"].to_numpy(dtype=np.int64).tolist()))
    n_covered = int(
        sum(1 for oi, ra, sc in zip(aligned["obs_id"].tolist(), ref_arcs.tolist(), scored_mask.tolist()) if sc and (oi, ra) in cand_pairs)
    )
    n_correct = int(np.sum((pred_arcs == ref_arcs) & scored_mask))

    seq = compute_sequence_metrics_for_trip(pred_arcs, trip_with_route.route, geom["arc_len"])

    return {
        "evaluation_available": True,
        "pointwise_top1_accuracy": round(float(n_correct / max(n_scored, 1)), 5),
        "candidate_miss_rate": round(float((n_scored - n_covered) / max(n_scored, 1)), 5),
        "sequence_lcs_f1": round(float(seq["seq_lcs_f1"]), 5),
        "sequence_length_weighted_f1": round(float(seq["seq_length_weighted_f1"]), 5),
        "sequence_arc_f1": round(float(seq["seq_arc_f1"]), 5),
        "n_scored_observations": n_scored,
        "label_coverage_pct": round(100.0 * n_scored / max(len(aligned), 1), 2),
    }
