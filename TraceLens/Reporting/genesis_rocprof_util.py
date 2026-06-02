###############################################################################
# Genesis extension for TraceLens — rocprof capture utilities
###############################################################################

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _find_file(roots: List[Path], pattern: str) -> Optional[Path]:
    for root in roots:
        if root.is_dir():
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def convert_rocprof_csv_to_json(trace_dir: str, output_path: str, include_api: bool = False) -> str:
    """Convert rocprofv3 CSV kernel trace to rocprofiler-sdk JSON for TraceLens."""
    trace_dir_p = Path(trace_dir)
    kernel_csv = trace_dir_p / "kernel_kernel_trace.csv"
    agent_csv = trace_dir_p / "kernel_agent_info.csv"
    hip_csv = trace_dir_p / "kernel_hip_api_trace.csv"
    hsa_csv = trace_dir_p / "kernel_hsa_api_trace.csv"

    if not kernel_csv.exists():
        raise FileNotFoundError(f"{kernel_csv} not found")

    kernel_rows = _read_csv(str(kernel_csv))
    symbols: Dict[int, dict] = {}
    for row in kernel_rows:
        kid = int(row["Kernel_Id"])
        if kid in symbols:
            continue
        name = row["Kernel_Name"]
        symbols[kid] = {
            "kernel_id": kid,
            "kernel_name": name,
            "formatted_kernel_name": name,
            "truncated_kernel_name": name,
        }

    dispatches = []
    for row in kernel_rows:
        dispatches.append({
            "start_timestamp": int(row["Start_Timestamp"]),
            "end_timestamp": int(row["End_Timestamp"]),
            "dispatch_info": {
                "dispatch_id": int(row["Dispatch_Id"]),
                "kernel_id": int(row["Kernel_Id"]),
                "agent_id": {"handle": 0},
                "grid_size": {
                    "x": int(row["Grid_Size_X"]),
                    "y": int(row["Grid_Size_Y"]),
                    "z": int(row["Grid_Size_Z"]),
                },
                "workgroup_size": {
                    "x": int(row["Workgroup_Size_X"]),
                    "y": int(row["Workgroup_Size_Y"]),
                    "z": int(row["Workgroup_Size_Z"]),
                },
                "lds_block_size_v": int(row.get("LDS_Block_Size", 0)),
                "scratch_size": int(row.get("Scratch_Size", 0)),
                "arch_vgpr_count": int(row.get("VGPR_Count", 0)),
                "accum_vgpr_count": int(row.get("Accum_VGPR_Count", 0)),
                "sgpr_count": int(row.get("SGPR_Count", 0)),
            },
            "correlation_id": {"internal": int(row["Correlation_Id"])},
            "thread_id": int(row["Thread_Id"]),
            "stream_id": {"handle": int(row.get("Stream_Id", 0))},
        })

    timestamps = [int(r["Start_Timestamp"]) for r in kernel_rows]
    end_timestamps = [int(r["End_Timestamp"]) for r in kernel_rows]
    pid = int(kernel_rows[0]["Thread_Id"]) if kernel_rows else 0

    agents: List[dict] = []
    if agent_csv.exists():
        for row in _read_csv(str(agent_csv)):
            agents.append({
                "node_id": int(row.get("Node_Id", 0)),
                "type": row.get("Agent_Type", "UNKNOWN"),
                "name": row.get("Name", ""),
                "product_name": row.get("Product_Name", ""),
                "vendor_name": row.get("Vendor_Name", ""),
            })

    def _api_events(path: Path) -> List[dict]:
        if not path.exists():
            return []
        return [
            {
                "start_timestamp": int(r["Start_Timestamp"]),
                "end_timestamp": int(r["End_Timestamp"]),
                "operation": r["Function"],
                "thread_id": int(r["Thread_Id"]),
                "correlation_id": {"internal": int(r["Correlation_Id"])},
            }
            for r in _read_csv(str(path))
        ]

    hip_events = _api_events(hip_csv) if include_api else []
    hsa_events = _api_events(hsa_csv) if include_api else []

    result: Dict[str, Any] = {
        "rocprofiler-sdk-tool": [{
            "metadata": {
                "pid": pid,
                "init_time": min(timestamps) if timestamps else 0,
                "fini_time": max(end_timestamps) if end_timestamps else 0,
                "node": {"hostname": "genesis-benchmark"},
                "command": [],
            },
            "agents": agents,
            "kernel_symbols": list(symbols.values()),
            "buffer_records": {
                "kernel_dispatch": dispatches,
                "memory_copy": [],
                "hip_api": hip_events,
                "hsa_api": hsa_events,
            },
        }],
    }

    with open(output_path, "w") as f:
        json.dump(result, f)
    return output_path


def infer_benchmark_window_s(capture_dir: Path) -> Optional[float]:
    """Parse timed benchmark wall_time from run.log (e.g. wall_time=3.98s)."""
    run_log = capture_dir / "run.log"
    if not run_log.exists():
        return None
    text = run_log.read_text(errors="replace")
    match = re.search(r"wall_time=([\d.]+)s", text)
    if not match:
        return None
    # Small buffer so the window covers the full timed burst.
    return float(match.group(1)) * 1.05


def load_capture(capture_dir: str) -> dict:
    """Load a run_combined_trace.sh / run_profile.sh capture directory."""
    run_dir = Path(capture_dir).resolve()
    manifest = None
    for name in ("capture_manifest.json", "combined_manifest.json"):
        p = run_dir / name
        if p.exists():
            with open(p) as f:
                manifest = json.load(f)
            break

    rocprof_dir = run_dir / "kernel_trace"
    if not rocprof_dir.is_dir():
        rocprof_dir = run_dir

    search = [rocprof_dir, run_dir]
    pftrace = _find_file(search, "*_results.pftrace")
    profile_json = _find_file(search, "*_results.json")
    kernel_csv = rocprof_dir / "kernel_kernel_trace.csv"
    if not kernel_csv.exists():
        kernel_csv = _find_file(search, "*_kernel_trace.csv")
    viztracer = run_dir / "viztracer_trace.json"
    if not viztracer.exists():
        viztracer = None

    return {
        "capture_dir": run_dir,
        "rocprof_dir": rocprof_dir,
        "pftrace": pftrace,
        "profile_json": profile_json,
        "kernel_csv_dir": kernel_csv.parent if kernel_csv else rocprof_dir,
        "has_kernel_csv": kernel_csv is not None and kernel_csv.exists(),
        "viztracer_json": viztracer,
        "manifest": manifest,
    }


def resolve_profile_json(capture: dict, output_dir: Path, include_api: bool) -> Path:
    """Prefer CSV→JSON (clean UTF-8); native rocprof JSON often breaks TraceLens parser."""
    out = output_dir / "kernel_results.json"
    if capture["has_kernel_csv"]:
        if not out.exists() or out.stat().st_size == 0:
            convert_rocprof_csv_to_json(str(capture["kernel_csv_dir"]), str(out), include_api)
        return out
    if capture["profile_json"] and capture["profile_json"].exists():
        return capture["profile_json"]
    raise FileNotFoundError("No kernel CSV or *_results.json found")


def ensure_traceconv(output_dir: Path, user_path: Optional[str]) -> str:
    if user_path and Path(user_path).exists():
        return user_path
    on_path = shutil.which("traceconv")
    if on_path:
        return on_path
    from TraceLens.Reporting.pftrace_utils import acquire_traceconv

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "traceconv"
    if target.exists():
        return str(target)
    return str(acquire_traceconv(None, output_dir))


def pftrace_to_json(pftrace: Path, output_dir: Path, traceconv: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "pftrace_events.json"
    if out_json.exists() and out_json.stat().st_size > 0:
        return out_json
    subprocess.run(
        [traceconv, "json", str(pftrace), str(out_json)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_json
