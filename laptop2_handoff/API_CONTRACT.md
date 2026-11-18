# PathSnap — Laptop 1 ↔ Laptop 2 API Contract

This document defines the live HTTP contract between **Laptop 1** (FastAPI model/HMM/scorer server) and **Laptop 2** (evaluator-facing UI).

> **Important integrity & demonstration rules**
> - The Laptop 2 UI **must call live API responses** from Laptop 1 during demonstration. Never hardcode predictions or cache dummy trips.
> - **No raw datasets, `.route` reference labels, or model weights** are included in this handoff folder or exposed over the API. Server-side evaluation against `.route` happens strictly on Laptop 1 after predictions are finalized.
> - CORS is enabled (`allow_origins=["*"]`) so browser-based UIs (React, Vite, HTML/Leaflet, Streamlit, etc.) on Laptop 2 can call Laptop 1 directly over the local LAN.

---

## Base URL

- **Active Laptop 1 LAN URL:** `http://10.1.25.138:8000` *(reachability from Laptop 2 is pending physical LAN test on the same Wi-Fi/hotspot).*
- During local testing on Laptop 1: `http://127.0.0.1:8000`.
- Interactive OpenAPI docs are also available on Laptop 1 at `/docs`.

---

## Endpoints

### 1. `GET /health`
Returns server readiness, whether saved inference artifacts are loaded, and the active dataset mode (`"development"` or `"evaluation"`).

**Response `200 OK`** (`application/json`):
```json
{
  "status": "ready",
  "ready": true,
  "model_loaded": true,
  "active_mode": "development",
  "model_name": "Histogram Gradient Boosting",
  "decoder": "HMM/Viterbi",
  "available_trip_count": 10,
  "error": null,
  "timestamp_utc": "2026-11-18T06:00:00+00:00"
}
```
If evaluation mode is configured with a missing `PATHSNAP_EVAL_DIR`, `/health` returns `status: "degraded"`, `ready: false`, and an explicit error string in `error`.

---

### 2. `GET /trips`
Returns the sorted list of available trip IDs in the active mode (preserving leading zeros).

**Response `200 OK`** (`application/json`):
```json
{
  "mode": "development",
  "count": 10,
  "trips": [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000052",
    "00000056",
    "00000065",
    "00000076",
    "00000077",
    "00000091"
  ]
}
```
- **Error `503 Service Unavailable`**: Returned with `{"detail": "..."}` if the active mode's trip directory is missing or empty.
- The UI **must** populate its trip selector dynamically from `GET /trips` so switching Laptop 1 from `development` (10 student-test trips) to `evaluation` (10 reserved evaluator trips) works automatically with zero UI code changes.

---

### 3. `POST /predict`
Runs full live inference (spatial candidate generation -> trained classifier -> HMM/Viterbi decoding) for the selected trip and returns GPS points, predicted route geometry, local road-network context, server-side metrics, and elapsed latency.

**Request** (`application/json`):
```json
{
  "trip_id": "00000000"
}
```
- `trip_id` **must** be an exact string from `GET /trips` (leading zeros preserved).
- **Error `404 Not Found`**: Returned if `trip_id` is not in the active mode's trip list.
- **Error `503 Service Unavailable`**: Returned if model artifacts or trip directory are unavailable.

**Response `200 OK`** (`application/json`):
See `example_predict_response.json` in this folder for an actual live development response. Top-level schema:

| Field | Type | Description |
|---|---|---|
| `trip_id` | `string` | Requested trip ID (leading zeros preserved). |
| `mode` | `string` | `"development"` or `"evaluation"`. |
| `model_name` | `string` | Loaded final classifier name. |
| `decoder` | `string` | `"HMM/Viterbi"`. |
| `elapsed_time_ms` | `number` | End-to-end prediction time in milliseconds. |
| `elapsed_time_s` | `number` | End-to-end prediction time in seconds. |
| `conventions` | `object` | Explicit coordinate order, units, and missing-value rules. |
| `metrics` | `object` | Inference stats + server-side evaluation metrics (if `.route` exists on Laptop 1). |
| `gps_points` | `array<object>` | Raw GPS observations with per-point matched `pred_arc_id` and `pred_prob`. |
| `predicted_route_geometry` | `array<object>` | Ordered compressed sequence of predicted directed road segments. |
| `road_network_context` | `array<object>` | Relevant nearby road-network arcs for optional map background layer. |

#### Coordinate Order, Units & Missing-Value Conventions
- **Coordinate order**:
  - `gps_points[i]`: explicit named keys `lon` (WGS84 degrees East) and `lat` (WGS84 degrees North). *(Note: Leaflet/Folium expect `[lat, lon]`!)*
  - `predicted_route_geometry[k].coordinates_lon_lat`: `[[from_lon, from_lat], [to_lon, to_lat]]` (GeoJSON `[lon, lat]` order). Swap to `[lat, lon]` when passing directly to Leaflet `L.polyline`.
  - `road_network_context[m].coordinates_lon_lat`: `[[from_lon, from_lat], [to_lon, to_lat]]` (GeoJSON `[lon, lat]` order).
- **Units**:
  - Coordinates: WGS84 decimal degrees (`EPSG:4326`).
  - `timestamp_s`: seconds.
  - `length_m`, `predicted_route_length_m`: meters.
  - `elapsed_time_ms`, `candidate_generation_ms`, `classifier_ms`, `viterbi_decode_ms`: milliseconds.
  - Accuracy / F1 / Continuity / Miss Rate: ratios in `[0.0, 1.0]` (`label_coverage_pct` is in `[0.0, 100.0]`).
- **Missing-value conventions**:
  - If a GPS observation has no nearby road candidate, `pred_arc_id` is `null` and `pred_prob` is `null`.
  - If no server-side `.route` reference file exists for a trip, `metrics.evaluation_available` is `false` and `pointwise_top1_accuracy`, `candidate_miss_rate`, `sequence_lcs_f1`, `sequence_length_weighted_f1`, `sequence_arc_f1`, `n_scored_observations`, and `label_coverage_pct` are `null`.

#### `metrics` Object Fields
- Always populated (route-free):
  - `n_gps_points` (`int`)
  - `n_candidates_evaluated` (`int`)
  - `n_predicted_route_segments` (`int`)
  - `predicted_route_length_m` (`float`, meters)
  - `route_continuity` (`float`, 0–1)
  - `transition_continuity_changes_only` (`float`, 0–1)
  - `n_missing_candidate_obs` (`int`)
  - `n_gap_resets` (`int`)
  - `candidate_generation_ms`, `classifier_ms`, `viterbi_decode_ms` (`float`, ms)
- Populated by Laptop 1's server-side scorer when `.route` is present (`evaluation_available: true`):
  - `pointwise_top1_accuracy` (`float | null`)
  - `candidate_miss_rate` (`float | null`)
  - `sequence_lcs_f1` (`float | null`)
  - `sequence_length_weighted_f1` (`float | null`)
  - `sequence_arc_f1` (`float | null`)
  - `n_scored_observations` (`int | null`)
  - `label_coverage_pct` (`float | null`)

---

## UI Checklist for Laptop 2
1. Provide a configurable API Base URL input (defaulting to Laptop 1's LAN URL) and a status badge backed by `GET /health` showing readiness, active mode, and loaded model.
2. Load trip dropdown/list dynamically from `GET /trips` (displays all 10 evaluation trips automatically when Laptop 1 switches to `evaluation` mode).
3. On trip selection + button click, `POST /predict` with `{"trip_id": selectedTrip}` and render:
   - Interactive map (e.g., Leaflet) with distinct layers + legend + toggle controls for:
     - Raw GPS observations (`gps_points`: `[p.lat, p.lon]`)
     - Predicted route (`predicted_route_geometry`: swap `[lon, lat]` -> `[lat, lon]`)
     - Optional road network context (`road_network_context`)
   - Metrics summary panel showing `pointwise_top1_accuracy`, `sequence_lcs_f1`, `route_continuity`, `candidate_miss_rate`, `predicted_route_length_m`, `n_gps_points`, and `elapsed_time_ms` (gracefully rendering `"N/A"` when `null`).
4. Return your verified UI source folder, dependency file (`package.json` or `requirements.txt`), launch instructions, and a screenshot of the working UI connected to Laptop 1 so they can be integrated into the final repository and technical report.
