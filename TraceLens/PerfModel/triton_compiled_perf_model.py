###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Perf model for torch.compile-generated Triton kernels (triton_poi_*, triton_red_*,
triton_per_*).

Two strategies, tried in order:

  V2 (Trace-Intrinsic): extract metadata from Chrome trace event["args"].
      Available for PyTorch 2.4+ traces.  Exact element counts (not rounded),
      per-tensor byte calculation, no disk I/O.

  V1 (Cache-Based, fallback): parse the Inductor wrapper .py files that
      torch.compile writes to its cache directory.  Used when trace args are
      missing (older PyTorch traces).

Cache dirs searched by V1 (first wins):
  1. $TORCHINDUCTOR_CACHE_DIR
  2. ~/.cache/torchinductor
  3. /tmp/torchinductor_<username>
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
    "aten.add": 1,
    "aten.sub": 1,
    "aten.mul": 1,
    "aten.div": 1,
    "aten.pow": 2,
    "aten.sqrt": 2,
    "aten.rsqrt": 2,
    "aten.exp": 4,
    "aten.log": 4,
    "aten.sigmoid": 4,
    "aten.tanh": 4,
    "aten.relu": 1,
    "aten.silu": 4,
    "aten.gelu": 8,
    "aten.mean": 1,
    "aten.sum": 1,
    "aten.var": 2,
    "aten.abs": 1,
    "aten.neg": 1,
    "aten.reciprocal": 2,
    "aten.ne": 1,
    "aten.eq": 1,
    "aten.lt": 1,
    "aten.gt": 1,
    # memory-only
    "aten._to_copy": 0,
    "aten.copy_": 0,
    "aten.clone": 0,
    "aten.embedding": 0,
}

_PTR_DTYPE_BYTES: dict[str, int] = {
    "*bf16": 2,
    "*fp16": 2,
    "*f16": 2,
    "*fp32": 4,
    "*f32": 4,
    "*i8": 1,
    "*u8": 1,
    "*i32": 4,
    "*i64": 8,
}

_C10_DTYPE_BYTES: dict[str, int] = {
    "c10::BFloat16": 2,
    "c10::Half": 2,
    "c10::Float": 4,
    "c10::Double": 8,
    "c10::Byte": 1,
    "c10::Char": 1,
    "c10::Int": 4,
    "c10::Long": 8,
}

# Map bare op names (from kernel name) to their aten.* keys in _FLOPS_PER_ELEM.
# Sorted longest-first so multi-word ops like "_unsafe_view" match before "view".
_BARE_OPS = sorted(
    [(op.replace("aten.", ""), op) for op in _FLOPS_PER_ELEM],
    key=lambda x: len(x[0]),
    reverse=True,
)


# ---------------------------------------------------------------------------
# V2: Trace-intrinsic metadata extraction (PyTorch 2.4+)
# ---------------------------------------------------------------------------


def _parse_kernel_name(name: str) -> list[str]:
    """Extract fused ATen ops from a Triton kernel name."""
    m = re.match(r"triton_(?:poi|red|per)_fused_(.+)_(\d+)$", name)
    if not m:
        return []
    remaining = m.group(1)
    ops: list[str] = []
    while remaining:
        for bare, full in _BARE_OPS:
            if remaining == bare or remaining.startswith(bare + "_"):
                ops.append(full)
                remaining = remaining[len(bare) :].lstrip("_")
                break
        else:
            remaining = remaining.split("_", 1)[-1] if "_" in remaining else ""
    return ops


def _meta_from_trace_args(event: dict) -> dict | None:
    """Extract kernel metadata from trace event args (V2).

    Returns the same dict schema as _lookup() so downstream code is unchanged,
    or None if the trace event lacks the required fields.
    """
    args = event.get("args", {})
    concrete = args.get("Concrete Inputs")
    input_dims = args.get("Input Dims")
    input_type = args.get("Input type")
    if not concrete or not input_dims or not input_type:
        return None

    name = event.get("name", "")
    is_reduction = name.startswith("triton_red_") or name.startswith("triton_per_")

    scalars = [int(v) for v in concrete if v != ""]
    if is_reduction and len(scalars) >= 2:
        xnumel, rnumel = scalars[-2], scalars[-1]
    elif scalars:
        xnumel, rnumel = scalars[-1], 1
    else:
        return None

    aten_ops = _parse_kernel_name(name)

    ptr_bytes: list[int] = []
    ptr_shapes: list[list[int]] = []
    dtype: str | None = None
    from math import prod

    for dims, dt in zip(input_dims, input_type):
        if dt == "Scalar" or not dims:
            continue
        bpe = _C10_DTYPE_BYTES.get(dt, 0)
        ptr_bytes.append(bpe)
        ptr_shapes.append(dims)
        if dtype is None and bpe > 0:
            dtype = dt.split("::")[-1].lower()

    total_bytes = float(sum(prod(s) * b for s, b in zip(ptr_shapes, ptr_bytes)))
    if not is_reduction and ptr_bytes:
        total_bytes += ptr_bytes[0] * xnumel

    return {
        "aten_ops": aten_ops,
        "xnumel": xnumel,
        "rnumel": rnumel,
        "ptr_bytes": ptr_bytes,
        "total_bytes": total_bytes,
        "dtype": dtype,
    }


# ---------------------------------------------------------------------------
# V1: Inductor cache file parsing (fallback)
# ---------------------------------------------------------------------------


def _cache_dirs() -> list[str]:
    env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if env:
        return [env] if os.path.isdir(env) else []
    dirs = [
        os.path.expanduser("~/.cache/torchinductor"),
        f"/tmp/torchinductor_{getpass.getuser()}",
    ]
    return [d for d in dirs if os.path.isdir(d)]


def _parse_wrapper(content: str) -> dict[str, dict]:
    """Return {kernel_name: metadata} for every triton kernel in a wrapper file."""
    results: dict[str, dict] = {}

    kernel_matches = list(
        re.finditer(
            r"^(triton_\w+)\s*=\s*async_compile\.triton\(", content, re.MULTILINE
        )
    )

    for i, m in enumerate(kernel_matches):
        name = m.group(1)

        block_end = content.find("''', device_str=", m.start())
        if block_end == -1:
            continue
        block = content[m.start() : block_end]

        # ATen ops from the comment block above the kernel definition.
        # Look back to the end of the previous kernel block (or start of file)
        # to handle newer PyTorch versions with larger graph fragment comments.
        prev_end = kernel_matches[i - 1].start() if i > 0 else 0
        comment_region = content[prev_end : m.start()]
        aten_m = re.search(r"Original ATen:\s*\[([^\]]+)\]", comment_region)
        aten_ops = [o.strip() for o in aten_m.group(1).split(",")] if aten_m else []

        # size_hints: list format [xnumel, rnumel] (older PyTorch) or
        # dict format {'x': xnumel, 'r0_': rnumel} (PyTorch 2.7+)
        xnumel, rnumel = 0, 1
        hints_list_m = re.search(r"size_hints=\[([^\]]+)\]", block)
        hints_dict_m = re.search(r"size_hints=\{([^}]+)\}", block)
        if hints_list_m:
            nums = [int(x.strip()) for x in hints_list_m.group(1).split(",")]
            xnumel = nums[0]
            rnumel = nums[1] if len(nums) > 1 else 1
        elif hints_dict_m:
            hint_pairs = re.findall(r"'(\w+)':\s*(\d+)", hints_dict_m.group(1))
            for key, val in hint_pairs:
                if key == "x":
                    xnumel = int(val)
                elif key.startswith("r"):
                    rnumel = int(val)

        # Pointer element sizes and dtype from triton_meta signature
        sig_m = re.search(r"'signature':\s*\{([^}]+)\}", block)
        ptr_bytes: list[int] = []
        dtype: str | None = None
        if sig_m:
            for dm in re.finditer(r"'(\*\w+)'", sig_m.group(1)):
                b = _PTR_DTYPE_BYTES.get(dm.group(1))
                if b is not None:
                    ptr_bytes.append(b)
                    if dtype is None:
                        dtype = dm.group(1)[1:]  # strip leading '*', e.g. 'bf16'

        results[name] = {
            "aten_ops": aten_ops,
            "xnumel": xnumel,
            "rnumel": rnumel,
            "ptr_bytes": ptr_bytes,
            "dtype": dtype,
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
        for path in glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True):
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
        self._meta = _meta_from_trace_args(event)  # V2: trace args
        if self._meta is None:
            self._meta = _lookup(self.name)  # V1: cache fallback
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
        fpe = sum(_FLOPS_PER_ELEM.get(op, 0) for op in self._meta["aten_ops"])
        return float(fpe * total)

    def bytes(self) -> float:
        if self._meta is None:
            raise NotImplementedError(f"No Inductor artifacts found for {self.name}")
        total_bytes = self._meta.get("total_bytes")
        if total_bytes is not None:
            return total_bytes
        # V1 fallback: no per-tensor shapes available
        xnumel = self._meta["xnumel"]
        rnumel = self._meta["rnumel"]
        ptr_bytes = self._meta["ptr_bytes"]
        if not ptr_bytes:
            return 0.0
        if rnumel > 1:
            return float(sum(ptr_bytes[:-1]) * xnumel * rnumel + ptr_bytes[-1] * xnumel)
        return float(sum(ptr_bytes) * xnumel)

    def get_maf_type(self):
        return "vector" if self._meta is not None else None

    def get_compute_precision(self):
        if self._meta is None:
            return None
        return self._meta.get("dtype")

    def flops_bwd(self) -> float:
        raise NotImplementedError

    def bytes_bwd(self) -> float:
        raise NotImplementedError
