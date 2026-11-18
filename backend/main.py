"""PathSnap FastAPI Backend (Laptop 1).

Launch from Pathsnap project root (with .venv activated):
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

Mode configuration via environment variables:
  - PATHSNAP_MODE: "development" (default) or "evaluation"
  - PATHSNAP_EVAL_DIR: required local directory path when PATHSNAP_MODE=evaluation
  - PATHSNAP_ARTIFACTS_DIR: optional override for artifacts directory (default: ./artifacts)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_io import PROJECT_ROOT, discover_dataset_dirs, list_trip_ids
from src.inference import ARTIFACTS_DIR, PathSnapPredictor, score_prediction_server_side

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Trip ID string with leading zeros preserved, from GET /trips")


class AppState:
    mode: str = "development"
    trips_dir: Optional[Path] = None
    trips_dir_error: Optional[str] = None
    available_trips: List[str] = []
    predictor: Optional[PathSnapPredictor] = None
    model_error: Optional[str] = None


state = AppState()


def _configure_runtime() -> None:
    mode = os.environ.get("PATHSNAP_MODE", "development").strip().lower()
    if mode not in ("development", "evaluation"):
        state.mode = mode
        state.trips_dir = None
        state.available_trips = []
        state.trips_dir_error = f"Invalid PATHSNAP_MODE='{mode}'. Must be 'development' or 'evaluation'."
        logger.error(state.trips_dir_error)
        return

    state.mode = mode
    state.trips_dir_error = None

    if mode == "development":
        try:
            _, student_test_dir = discover_dataset_dirs()
            state.trips_dir = student_test_dir
            state.available_trips = list_trip_ids(student_test_dir)
        except Exception as exc:
            state.trips_dir = None
            state.available_trips = []
            state.trips_dir_error = f"Failed to discover development student_test directory: {exc}"
            logger.error(state.trips_dir_error)
    else:
        eval_dir_env = os.environ.get("PATHSNAP_EVAL_DIR", "").strip()
        if not eval_dir_env:
            state.trips_dir = None
            state.available_trips = []
            state.trips_dir_error = (
                "Evaluation mode active, but PATHSNAP_EVAL_DIR is not set. "
                "Set PATHSNAP_EVAL_DIR to the local directory containing the reserved evaluation trips."
            )
            logger.error(state.trips_dir_error)
        else:
            eval_path = Path(eval_dir_env).expanduser().resolve()
            if not eval_path.exists() or not eval_path.is_dir():
                state.trips_dir = None
                state.available_trips = []
                state.trips_dir_error = (
                    f"Configured evaluation directory does not exist or is not a directory: {eval_path}"
                )
                logger.error(state.trips_dir_error)
            else:
                trips = list_trip_ids(eval_path)
                if not trips:
                    state.trips_dir = eval_path
                    state.available_trips = []
                    state.trips_dir_error = f"Evaluation directory exists but contains no trip folders: {eval_path}"
                    logger.error(state.trips_dir_error)
                else:
                    state.trips_dir = eval_path
                    state.available_trips = trips

    # Load saved model artifacts (never retrains!)
    art_dir = Path(os.environ.get("PATHSNAP_ARTIFACTS_DIR", str(ARTIFACTS_DIR))).expanduser().resolve()
    try:
        state.predictor = PathSnapPredictor(artifacts_dir=art_dir)
        state.model_error = None
        logger.info(
            "PathSnap backend ready | mode=%s | model=%s | trips=%d | trips_dir=%s",
            state.mode,
            state.predictor.model_name,
            len(state.available_trips),
            state.trips_dir,
        )
    except Exception as exc:
        state.predictor = None
        state.model_error = f"Failed to load inference artifacts from {art_dir}: {exc}"
        logger.error(state.model_error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_runtime()
    yield


app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    version="1.0.0",
    description="Laptop 1 FastAPI backend serving trained classifier + HMM/Viterbi map matching and server-side evaluation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    model_loaded = state.predictor is not None
    ready = model_loaded and (state.trips_dir_error is None) and (len(state.available_trips) > 0)
    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "model_loaded": model_loaded,
        "active_mode": state.mode,
        "model_name": state.predictor.model_name if state.predictor else None,
        "decoder": "HMM/Viterbi",
        "available_trip_count": len(state.available_trips),
        "error": state.trips_dir_error or state.model_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/trips")
def get_trips() -> Dict[str, Any]:
    if state.trips_dir_error is not None or state.trips_dir is None:
        raise HTTPException(status_code=503, detail=state.trips_dir_error or "Trips directory unavailable.")
    return {
        "mode": state.mode,
        "count": len(state.available_trips),
        "trips": state.available_trips,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if state.trips_dir_error is not None or state.trips_dir is None:
        raise HTTPException(status_code=503, detail=state.trips_dir_error or "Trips directory unavailable.")
    if state.predictor is None:
        raise HTTPException(status_code=503, detail=state.model_error or "Model artifacts not loaded.")

    trip_id = req.trip_id.strip()
    if trip_id not in state.available_trips:
        logger.warning("PREDICT 404 | mode=%s | trip_id=%s not in available trips", state.mode, trip_id)
        raise HTTPException(
            status_code=404,
            detail=f"Trip '{trip_id}' is not available in active mode '{state.mode}'. Call GET /trips for valid IDs.",
        )

    try:
        # 1. Pure inference (strictly reads ONLY .track, .nodes, .arcs + saved artifacts)
        pred_out = state.predictor.predict_trip(state.trips_dir, trip_id)

        # 2. Separate server-side scoring AFTER predictions are finalized (reference labels never leave Laptop 1)
        eval_metrics = score_prediction_server_side(
            split_dir=state.trips_dir,
            trip_id=trip_id,
            pred_arcs=pred_out["pred_arcs_internal"],
            cands_df=pred_out["cands_internal"],
            geom=pred_out["geom_internal"],
        )

        combined_metrics = {
            **pred_out["inference_metrics"],
            **eval_metrics,
        }

        elapsed_ms = pred_out["elapsed_time_ms"]
        logger.info(
            "PREDICT 200 | mode=%s | trip_id=%s | gps=%d | segs=%d | continuity=%.4f | top1=%s | seq_lcs_f1=%s | elapsed=%.1f ms",
            state.mode,
            trip_id,
            combined_metrics["n_gps_points"],
            combined_metrics["n_predicted_route_segments"],
            combined_metrics["route_continuity"],
            f"{eval_metrics['pointwise_top1_accuracy']:.4f}" if eval_metrics["pointwise_top1_accuracy"] is not None else "N/A",
            f"{eval_metrics['sequence_lcs_f1']:.4f}" if eval_metrics["sequence_lcs_f1"] is not None else "N/A",
            elapsed_ms,
        )

        return {
            "trip_id": trip_id,
            "mode": state.mode,
            "model_name": state.predictor.model_name,
            "decoder": "HMM/Viterbi",
            "elapsed_time_ms": elapsed_ms,
            "elapsed_time_s": round(elapsed_ms / 1000.0, 4),
            "conventions": {
                "coordinate_order": {
                    "gps_points": "explicit named fields 'lon' (WGS84 degrees East) and 'lat' (WGS84 degrees North)",
                    "predicted_route_geometry": "coordinates_lon_lat is [[from_lon, from_lat], [to_lon, to_lat]] (GeoJSON lon,lat order)",
                    "road_network_context": "coordinates_lon_lat is [[from_lon, from_lat], [to_lon, to_lat]] (GeoJSON lon,lat order)",
                },
                "units": {
                    "coordinates": "WGS84 decimal degrees (EPSG:4326)",
                    "timestamps": "seconds",
                    "lengths_and_distances": "meters",
                    "timings": "milliseconds (ms) and seconds (s)",
                    "metrics_ratios": "fraction in [0.0, 1.0] (except label_coverage_pct in [0.0, 100.0])",
                },
                "missing_value_conventions": {
                    "unmatched_gps_point": "pred_arc_id=null and pred_prob=null when no road candidate exists",
                    "unavailable_reference_route": "evaluation_available=false and reference-dependent metrics set to null",
                },
            },
            "metrics": combined_metrics,
            "gps_points": pred_out["gps_points"],
            "predicted_route_geometry": pred_out["predicted_route_geometry"],
            "road_network_context": pred_out["road_network_context"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PREDICT 500 | mode=%s | trip_id=%s | error=%s", state.mode, trip_id, exc)
        raise HTTPException(status_code=500, detail=f"Inference failed for trip '{trip_id}': {exc}")
