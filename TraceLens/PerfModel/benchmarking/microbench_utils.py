###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared utilities for the microbenchmark scripts.

- resolve_physical_device: logical torch index -> physical id used by smi tools.
- check_gpu_idle: pre-flight idle check via amd-smi or nvidia-smi.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Tuple


def resolve_physical_device(logical: int) -> Tuple[int, str]:
    """Map a logical torch device index to the physical id used by amd-smi /
    nvidia-smi via HIP/ROCR/CUDA_VISIBLE_DEVICES. Returns (phys, source)."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            mapped = [int(p) for p in parts]
        except ValueError:
            return logical, f"{var}={val} (non-numeric, identity fallback)"
        if 0 <= logical < len(mapped):
            return mapped[logical], var
        return (
            logical,
            f"{var}={val} (logical {logical} out of range, identity fallback)",
        )
    return logical, "identity"


def _as_dict(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else {}


def _metric_value(block: Any, default: int = 0) -> int:
    """Extract an integer metric from an amd-smi ``{value, unit}`` block."""
    if not isinstance(block, dict):
        return default
    value = block.get("value", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _load_json(stdout: str) -> Any:
    if not stdout.strip():
        return None
    return json.loads(stdout)


def _check_amd_gpu_idle(
    phys: int,
    dev_tag: str,
    *,
    util_threshold: int,
    mem_threshold_mib: int,
) -> Tuple[bool, str]:
    metric = subprocess.run(
        ["amd-smi", "metric", "-g", str(phys), "-u", "-m", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    metric_data = _load_json(metric.stdout)
    if not isinstance(metric_data, dict):
        return True, f"amd-smi {dev_tag}: unexpected metric JSON; check skipped"

    gpu_data = metric_data.get("gpu_data") or []
    if not gpu_data or not isinstance(gpu_data[0], dict):
        return True, f"amd-smi {dev_tag}: no gpu_data; check skipped"

    gpu = gpu_data[0]
    usage = _as_dict(gpu.get("usage"))
    mem_usage = _as_dict(gpu.get("mem_usage"))
    util = _metric_value(usage.get("gfx_activity"))
    mem_mib = _metric_value(mem_usage.get("used_vram"))

    proc = subprocess.run(
        ["amd-smi", "process", "-g", str(phys), "-G", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    proc_data = _load_json(proc.stdout)
    other_pids: list[int] = []
    if isinstance(proc_data, list) and proc_data:
        gpu_proc = _as_dict(proc_data[0])
        for entry in gpu_proc.get("process_list") or []:
            info = _as_dict(entry.get("process_info") if isinstance(entry, dict) else None)
            pid = info.get("pid")
            if pid is not None and str(pid) != str(os.getpid()):
                other_pids.append(int(pid))
                
    if util > util_threshold or mem_mib > mem_threshold_mib or other_pids:
        return False, (
            f"amd-smi {dev_tag}: util={util}% mem={mem_mib}MiB "
            f"other_pids={other_pids}"
        )
    return True, f"amd-smi {dev_tag}: idle (util={util}% mem={mem_mib}MiB)"


def check_gpu_idle(
    device: int, *, util_threshold: int = 5, mem_threshold_mib: int = 256
) -> Tuple[bool, str]:
    """Pre-flight idle check via amd-smi or nvidia-smi. Returns (is_idle, msg).
    Skipped (returns True) if neither tool is on PATH."""
    phys, src = resolve_physical_device(device)
    dev_tag = f"logical {device} -> physical {phys} (via {src})"

    if shutil.which("amd-smi"):
        try:
            return _check_amd_gpu_idle(
                phys,
                dev_tag,
                util_threshold=util_threshold,
                mem_threshold_mib=mem_threshold_mib,
            )
        except Exception as e:
            return True, f"amd-smi check skipped ({e})"

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(phys),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [p.strip() for p in (r.stdout or "").split(",")]
            util = int(parts[0]) if parts and parts[0].isdigit() else 0
            mem_mib = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            r2 = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                    "-i",
                    str(phys),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            other_pids = [
                p.strip()
                for p in (r2.stdout or "").splitlines()
                if p.strip() and p.strip() != str(os.getpid())
            ]
            if util > util_threshold or mem_mib > mem_threshold_mib or other_pids:
                return False, (
                    f"nvidia-smi {dev_tag}: util={util}% mem={mem_mib}MiB "
                    f"other_pids={other_pids}"
                )
            return True, f"nvidia-smi {dev_tag}: idle (util={util}% mem={mem_mib}MiB)"
        except Exception as e:
            return True, f"nvidia-smi check skipped ({e})"

    return True, "no amd-smi/nvidia-smi on PATH; skipping idle check"
