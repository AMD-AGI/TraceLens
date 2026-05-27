#!/usr/bin/env python3
"""Generate TraceLens/PerfModel/_known_aten_ops.json from the current PyTorch installation.

Collects all op names from torch.ops.* namespaces (e.g. aten, prims) and
torch.distributed, then writes them as a sorted JSON list. This vendored
list is used by _parse_kernel_name() in triton_compiled_perf_model.py to
segment fused op names in Triton kernels without requiring torch at
analysis time.

Run this script on a developer machine whenever the target PyTorch version
is bumped, then commit the updated JSON.

Usage:
    python scripts/generate_known_aten_ops.py
"""

import json
import os

import torch

ops: set[str] = set()

for ns_name in dir(torch.ops):
    if ns_name.startswith("__"):
        continue
    try:
        ns = getattr(torch.ops, ns_name)
        ops.update(n for n in dir(ns) if not n.startswith("__"))
    except Exception:
        pass

if hasattr(torch, "distributed"):
    ops.update(
        n for n in dir(torch.distributed) if not n.startswith("_") and "_" in n
    )

out_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "TraceLens",
    "PerfModel",
    "_known_aten_ops.json",
)
out_path = os.path.normpath(out_path)

with open(out_path, "w") as f:
    json.dump(sorted(ops), f, indent=2)

print(f"Wrote {len(ops)} ops to {out_path} (torch {torch.__version__})")
