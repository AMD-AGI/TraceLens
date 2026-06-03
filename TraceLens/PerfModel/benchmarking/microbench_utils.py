###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared utilities for the microbenchmark scripts.

- resolve_physical_device: logical torch index -> physical id used by smi tools.
- check_gpu_idle: pre-flight idle check via rocm-smi or nvidia-smi.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Tuple


def resolve_physical_device(logical: int) -> Tuple[int, str]:
    """Map a logical torch device index to the physical id used by rocm-smi /
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


def check_gpu_idle(
    device: int, *, util_threshold: int = 5, mem_threshold_mib: int = 256
) -> Tuple[bool, str]:
    """Pre-flight idle check via rocm-smi or nvidia-smi. Returns (is_idle, msg).
    Skipped (returns True) if neither tool is on PATH."""
    phys, src = resolve_physical_device(device)
    dev_tag = f"logical {device} -> physical {phys} (via {src})"

    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                [
                    "rocm-smi",
                    "-d",
                    str(phys),
                    "--showuse",
                    "--showmemuse",
                    "--showpids",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            data = json.loads(r.stdout) if r.stdout.strip() else {}
            card = data.get(f"card{phys}", {})
            util = int(str(card.get("GPU use (%)", "0")).strip() or 0)
            vram_used = int(
                str(card.get("GPU Memory Allocated (VRAM%)", "0")).strip() or 0
            )
            pids = data.get("system", {}).get("PIDs", []) or []
            other_pids = [p for p in pids if str(p) != str(os.getpid())]
            if util > util_threshold or vram_used > 5 or other_pids:
                return False, (
                    f"rocm-smi {dev_tag}: util={util}% vram_used={vram_used}% "
                    f"other_pids={other_pids}"
                )
            return True, f"rocm-smi {dev_tag}: idle (util={util}% vram={vram_used}%)"
        except Exception as e:
            return True, f"rocm-smi check skipped ({e})"

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

    return True, "no rocm-smi/nvidia-smi on PATH; skipping idle check"
