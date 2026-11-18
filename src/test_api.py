"""Milestone 6 verification: exercise all 10 student-test trips through live FastAPI & verify restart."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd

from src.data_io import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "outputs" / "milestone6"
HANDOFF_DIR = PROJECT_ROOT / "laptop2_handoff"


def get_lan_ip() -> str:
    """Detect local LAN IP address for Laptop 1."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def wait_for_server(base_url: str, timeout_s: float = 20.0) -> Dict[str, Any]:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Server did not become healthy at {base_url} within {timeout_s}s")


def start_uvicorn(port: int, extra_env: Dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_uvicorn(proc: subprocess.Popen) -> str:
    proc.send_signal(signal.SIGTERM)
    try:
        out, _ = proc.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    lan_ip = get_lan_ip()

    # 1. Start backend in development mode and exercise /health, /trips, and all 10 student-test trips
    print(f"Starting FastAPI backend in development mode on port {port} (LAN IP: {lan_ip})...")
    proc1 = start_uvicorn(port, {"PATHSNAP_MODE": "development"})
    try:
        health1 = wait_for_server(base_url)
        print("GET /health:", health1)
        assert health1["ready"] is True and health1["model_loaded"] is True and health1["active_mode"] == "development"

        trips_resp = httpx.get(f"{base_url}/trips", timeout=10.0)
        trips_resp.raise_for_status()
        trips_data = trips_resp.json()
        print("GET /trips:", trips_data)
        assert trips_data["count"] == 10

        # Test invalid trip_id returns 404
        bad_resp = httpx.post(f"{base_url}/predict", json={"trip_id": "99999999"}, timeout=10.0)
        assert bad_resp.status_code == 404, f"Expected 404 for unknown trip, got {bad_resp.status_code}"

        rows: List[Dict[str, Any]] = []
        example_saved = False
        for tid in trips_data["trips"]:
            r = httpx.post(f"{base_url}/predict", json={"trip_id": tid}, timeout=60.0)
            r.raise_for_status()
            payload = r.json()

            # Verify zero reference route leakage in payload
            payload_keys = set(payload.keys())
            assert "reference_route" not in payload_keys and "route" not in payload_keys
            assert payload["trip_id"] == tid
            assert len(payload["gps_points"]) > 0
            assert len(payload["predicted_route_geometry"]) > 0

            m = payload["metrics"]
            rows.append(
                {
                    "trip_id": tid,
                    "n_gps_points": m["n_gps_points"],
                    "n_predicted_route_segments": m["n_predicted_route_segments"],
                    "predicted_route_length_m": m["predicted_route_length_m"],
                    "route_continuity": m["route_continuity"],
                    "pointwise_top1_accuracy": m["pointwise_top1_accuracy"],
                    "candidate_miss_rate": m["candidate_miss_rate"],
                    "sequence_lcs_f1": m["sequence_lcs_f1"],
                    "sequence_length_weighted_f1": m["sequence_length_weighted_f1"],
                    "elapsed_time_ms": payload["elapsed_time_ms"],
                }
            )
            print(
                f"  POST /predict {tid}: Top1={m['pointwise_top1_accuracy']:.4f} | "
                f"Cont={m['route_continuity']:.4f} | LCS-F1={m['sequence_lcs_f1']:.4f} | "
                f"elapsed={payload['elapsed_time_ms']:.1f}ms"
            )

            # Save actual development response example for laptop2_handoff (first trip!)
            if not example_saved:
                (HANDOFF_DIR / "example_health_response.json").write_text(json.dumps(health1, indent=2), encoding="utf-8")
                (HANDOFF_DIR / "example_trips_response.json").write_text(json.dumps(trips_data, indent=2), encoding="utf-8")
                # Save full actual response in outputs/milestone6 and a schema-complete trimmed-array version OR full compact version in handoff
                (OUT_DIR / f"sample_predict_{tid}_full.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                handoff_example = dict(payload)
                # Keep real live response fields; truncate huge coordinate arrays only with clear note if > 3MB
                if len(json.dumps(payload)) > 1_000_000:
                    handoff_example["gps_points"] = payload["gps_points"][:10]
                    handoff_example["predicted_route_geometry"] = payload["predicted_route_geometry"][:10]
                    handoff_example["road_network_context"] = payload["road_network_context"][:10]
                    handoff_example["_handoff_note"] = (
                        f"Actual live API response for development trip {tid}; gps_points ({len(payload['gps_points'])}), "
                        f"predicted_route_geometry ({len(payload['predicted_route_geometry'])}), and "
                        f"road_network_context ({len(payload['road_network_context'])}) truncated to first 10 items for readability."
                    )
                (HANDOFF_DIR / "example_predict_response.json").write_text(json.dumps(handoff_example, indent=2), encoding="utf-8")
                example_saved = True
    finally:
        logs1 = stop_uvicorn(proc1)
        (OUT_DIR / "uvicorn_run1.log").write_text(logs1, encoding="utf-8")

    # 2. Restart backend and verify it serves predictions strictly from saved artifacts without retraining!
    print("\nRestarting FastAPI backend to verify artifact-backed persistence...")
    proc2 = start_uvicorn(port, {"PATHSNAP_MODE": "development"})
    try:
        health2 = wait_for_server(base_url)
        assert health2["ready"] is True and health2["model_loaded"] is True
        tid0 = trips_data["trips"][0]
        r2 = httpx.post(f"{base_url}/predict", json={"trip_id": tid0}, timeout=30.0)
        r2.raise_for_status()
        p2 = r2.json()
        print(f"Restart verification PASSED for {tid0}: Top1={p2['metrics']['pointwise_top1_accuracy']:.4f}, elapsed={p2['elapsed_time_ms']:.1f}ms")
    finally:
        logs2 = stop_uvicorn(proc2)
        (OUT_DIR / "uvicorn_run2_restart.log").write_text(logs2, encoding="utf-8")

    # 3. Verify evaluation mode produces a clear error when PATHSNAP_EVAL_DIR is missing!
    print("\nTesting evaluation mode error handling with missing PATHSNAP_EVAL_DIR...")
    proc3 = start_uvicorn(port, {"PATHSNAP_MODE": "evaluation", "PATHSNAP_EVAL_DIR": "/tmp/nonexistent_pathsnap_eval_dir"})
    try:
        health3 = wait_for_server(base_url)
        assert health3["active_mode"] == "evaluation"
        assert health3["ready"] is False
        assert "does not exist" in (health3["error"] or "")
        r_trips3 = httpx.get(f"{base_url}/trips", timeout=10.0)
        assert r_trips3.status_code == 503
        eval_error_check = {
            "health_response": health3,
            "trips_status_code": r_trips3.status_code,
            "trips_detail": r_trips3.json(),
        }
        print("Evaluation missing-dir check PASSED:", eval_error_check["trips_detail"])
    finally:
        logs3 = stop_uvicorn(proc3)
        (OUT_DIR / "uvicorn_eval_missing_dir.log").write_text(logs3, encoding="utf-8")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "api_student_test_results.csv", index=False)

    summary = {
        "lan_ip": lan_ip,
        "lan_url": f"http://{lan_ip}:{port}",
        "active_mode_tested": "development",
        "n_student_test_trips_verified": len(rows),
        "mean_top1_accuracy": round(float(res_df["pointwise_top1_accuracy"].mean()), 5),
        "mean_route_continuity": round(float(res_df["route_continuity"].mean()), 5),
        "mean_seq_lcs_f1": round(float(res_df["sequence_lcs_f1"].mean()), 5),
        "mean_elapsed_ms": round(float(res_df["elapsed_time_ms"].mean()), 2),
        "restart_verified": True,
        "missing_eval_dir_verified": eval_error_check,
        "laptop2_reachability": "pending (must be tested physically from Laptop 2 on same LAN)",
    }
    (OUT_DIR / "milestone6_api_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== MILESTONE 6 VERIFICATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
