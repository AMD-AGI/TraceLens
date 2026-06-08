###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared utilities for the microbenchmark scripts.

- resolve_physical_device: logical torch index -> physical id used by smi tools.
- check_gpu_idle: pre-flight idle check via amdsmi or nvidia-smi.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Tuple


def resolve_physical_device(logical: int) -> Tuple[int, str]:
    """Map a logical torch device index to the physical id used by amdsmi /
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


def _int_metric(value: Any, default: int = 0) -> int:
    """Coerce an amdsmi metric to int; treat ``N/A`` and non-numeric as *default*."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _check_amd_gpu_idle(
    phys: int,
    dev_tag: str,
    *,
    util_threshold: int,
    mem_threshold_mib: int,
) -> Tuple[bool, str]:
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if phys < 0 or phys >= len(devices):
            return (
                True,
                f"amdsmi {dev_tag}: physical id {phys} out of range "
                f"(found {len(devices)} GPU(s)); check skipped",
            )

        device = devices[phys]
        activity = amdsmi.amdsmi_get_gpu_activity(device)
        util = _int_metric(activity.get("gfx_activity"))

        vram = amdsmi.amdsmi_get_gpu_vram_usage(device)
        mem_mib = _int_metric(vram.get("vram_used"))

        other_pids: list[int] = []
        for proc in amdsmi.amdsmi_get_gpu_process_list(device):
            if not isinstance(proc, dict):
                continue
            pid = proc.get("pid")
            if pid is not None and int(pid) != os.getpid():
                other_pids.append(int(pid))

        if util > util_threshold or mem_mib > mem_threshold_mib or other_pids:
            return False, (
                f"amdsmi {dev_tag}: util={util}% mem={mem_mib}MiB "
                f"other_pids={other_pids}"
            )
        return True, f"amdsmi {dev_tag}: idle (util={util}% mem={mem_mib}MiB)"
    finally:
        amdsmi.amdsmi_shut_down()


def check_gpu_idle(
    device: int, *, util_threshold: int = 5, mem_threshold_mib: int = 256
) -> Tuple[bool, str]:
    """Pre-flight idle check via amdsmi or nvidia-smi. Returns (is_idle, msg).
    Skipped (returns True) if neither library/tool is available."""
    phys, src = resolve_physical_device(device)
    dev_tag = f"logical {device} -> physical {phys} (via {src})"

    try:
        import amdsmi  # noqa: F401
    except ImportError:
        amdsmi = None

    if amdsmi is not None:
        try:
            return _check_amd_gpu_idle(
                phys,
                dev_tag,
                util_threshold=util_threshold,
                mem_threshold_mib=mem_threshold_mib,
            )
        except Exception as e:
            return True, f"amdsmi check skipped ({e})"

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

    return True, "no amdsmi/nvidia-smi available; skipping idle check"
