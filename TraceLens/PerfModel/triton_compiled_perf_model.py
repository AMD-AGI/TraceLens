###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Perf model for torch.compile-generated Triton kernels (triton_poi_*, triton_red_*,
triton_per_*).

Strategy (Solution A): parse the Inductor wrapper .py files that torch.compile
writes to its cache directory.  Each wrapper contains:
  - "Original ATen: [aten.add, aten.mul, ...]" comments  → which ops were fused
  - size_hints=[xnumel] or [xnumel, rnumel]             → element counts
  - triton_meta signature {0: '*bf16', ...}              → pointer dtypes

Cache dirs searched (first wins):
  1. $TORCHINDUCTOR_CACHE_DIR
  2. ~/.cache/torchinductor
  3. /tmp/torchinductor_<username>

Set TORCHINDUCTOR_CACHE_DIR to a persistent path so artifacts survive process
exit (otherwise they live only in /tmp and are lost on reboot).
"""

import getpass
import glob
import os
import re

# ---------------------------------------------------------------------------
# FLOPs per element for ATen ops that commonly appear in fused Triton kernels.
# Counts arithmetic operations; memory-only ops are 0.
# ---------------------------------------------------------------------------
_FLOPS_PER_ELEM: dict[str, int] = {
    "aten.add": 1, "aten.sub": 1, "aten.mul": 1, "aten.div": 1,
    "aten.pow": 2, "aten.sqrt": 2, "aten.rsqrt": 2,
    "aten.exp": 4, "aten.log": 4,
    "aten.sigmoid": 4, "aten.tanh": 4,
    "aten.relu": 1, "aten.silu": 4, "aten.gelu": 8,
    "aten.mean": 1, "aten.sum": 1, "aten.var": 2,
    "aten.abs": 1, "aten.neg": 1, "aten.reciprocal": 2,
    "aten.ne": 1, "aten.eq": 1, "aten.lt": 1, "aten.gt": 1,
    # memory-only
    "aten._to_copy": 0, "aten.copy_": 0, "aten.clone": 0, "aten.embedding": 0,
}

_PTR_DTYPE_BYTES: dict[str, int] = {
    "*bf16": 2, "*fp16": 2, "*f16": 2,
    "*fp32": 4, "*f32": 4,
    "*i8": 1, "*u8": 1, "*i32": 4, "*i64": 8,
}

# ---------------------------------------------------------------------------
# Inductor artifact parsing
# ---------------------------------------------------------------------------

def _cache_dirs() -> list[str]:
    dirs = []
    env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if env:
        dirs.append(env)
    dirs.append(os.path.expanduser("~/.cache/torchinductor"))
    try:
        dirs.append(f"/tmp/torchinductor_{getpass.getuser()}")
    except Exception:
        pass
    return [d for d in dirs if os.path.isdir(d)]


def _parse_wrapper(content: str) -> dict[str, dict]:
    """Return {kernel_name: metadata} for every triton kernel in a wrapper file."""
    results: dict[str, dict] = {}

    for m in re.finditer(
        r"^(triton_\w+)\s*=\s*async_compile\.triton\(", content, re.MULTILINE
    ):
        name = m.group(1)

        block_end = content.find("''', device_str=", m.start())
        if block_end == -1:
            continue
        block = content[m.start() : block_end]

        # ATen ops from the comment block immediately above the kernel definition
        comment_region = content[max(0, m.start() - 600) : m.start()]
        aten_m = re.search(r"Original ATen:\s*\[([^\]]+)\]", comment_region)
        aten_ops = (
            [o.strip() for o in aten_m.group(1).split(",")] if aten_m else []
        )

        # size_hints=[xnumel] for pointwise, [xnumel, rnumel] for reductions
        hints_m = re.search(r"size_hints=\[([^\]]+)\]", block)
        if hints_m:
            nums = [int(x.strip()) for x in hints_m.group(1).split(",")]
            xnumel = nums[0]
            rnumel = nums[1] if len(nums) > 1 else 1
        else:
            xnumel, rnumel = 0, 1

        # Pointer element sizes from triton_meta signature
        sig_m = re.search(r"'signature':\s*\{([^}]+)\}", block)
        ptr_bytes: list[int] = []
        if sig_m:
            for dm in re.finditer(r"'(\*\w+)'", sig_m.group(1)):
                b = _PTR_DTYPE_BYTES.get(dm.group(1))
                if b is not None:
                    ptr_bytes.append(b)

        results[name] = {
            "aten_ops": aten_ops,
            "xnumel": xnumel,
            "rnumel": rnumel,
            "ptr_bytes": ptr_bytes,
        }

    return results


# Module-level caches so each file is parsed at most once per process.
_kernel_meta_cache: dict[str, dict] = {}
_scanned_wrappers: set[str] = set()


def _lookup(name: str) -> dict | None:
    """Return parsed metadata for a kernel name, scanning cache dirs as needed."""
    if name in _kernel_meta_cache:
        return _kernel_meta_cache[name]

    for cache_dir in _cache_dirs():
        for path in glob.glob(
            os.path.join(cache_dir, "**", "*.py"), recursive=True
        ):
            if path in _scanned_wrappers:
                continue
            _scanned_wrappers.add(path)
            try:
                with open(path) as f:
                    content = f.read()
                if "async_compile.triton" not in content:
                    continue
                _kernel_meta_cache.update(_parse_wrapper(content))
            except Exception:
                pass
            if name in _kernel_meta_cache:
                return _kernel_meta_cache[name]

    return None


# ---------------------------------------------------------------------------
# Perf model class
# ---------------------------------------------------------------------------

class TritonCompiledPerfModel:
    """
    Perf model for torch.compile-generated Triton kernels.

    FLOPs  = sum(flops_per_elem[op] for op in fused_aten_ops) * total_elements
    Bytes  = num_ptr_args * xnumel * bytes_per_elem   (pointwise)
           = (n-1) * xnumel * rnumel * bpe + xnumel * bpe  (reduction)

    Raises NotImplementedError if no Inductor artifacts are found, so
    TraceLens silently skips the kernel (same behaviour as unmodelled ATen ops).
    """

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        self.name = event["name"]
        self._meta = _lookup(self.name)
        self.param_details: dict = {}
        if self._meta is not None:
            self.param_details = {
                "fused_ops": ", ".join(self._meta["aten_ops"]),
                "xnumel": self._meta["xnumel"],
                "rnumel": self._meta["rnumel"],
            }

    def flops(self) -> float:
        if self._meta is None:
            raise NotImplementedError(f"No Inductor artifacts found for {self.name}")
        total = self._meta["xnumel"] * self._meta["rnumel"]
        fpe = sum(_FLOPS_PER_ELEM.get(op, 1) for op in self._meta["aten_ops"])
        return float(fpe * total)

    def bytes(self) -> float:
        if self._meta is None:
            raise NotImplementedError(f"No Inductor artifacts found for {self.name}")
        xnumel = self._meta["xnumel"]
        rnumel = self._meta["rnumel"]
        ptr_bytes = self._meta["ptr_bytes"]
        if not ptr_bytes:
            return 0.0
        bpe = ptr_bytes[0]
        n = len(ptr_bytes)
        # Reduction: (n-1) inputs span xnumel*rnumel; 1 output spans xnumel
        if rnumel > 1:
            return float((n - 1) * xnumel * rnumel * bpe + xnumel * bpe)
        return float(n * xnumel * bpe)

    def flops_bwd(self) -> float:
        raise NotImplementedError

    def bytes_bwd(self) -> float:
        raise NotImplementedError
