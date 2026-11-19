###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Utils. for perf. model.
"""


def optional_int(value, default=None):
    """Parse *value* as int, returning *default* when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def optional_float(value, default=0.0):
    """Parse *value* as float, returning *default* when conversion fails."""
    if value in ("", "None", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rates_for_duration(gflops, bytes_moved, time_us):
    """Return ``(tflops_per_s, tb_per_s)`` implied by a duration in microseconds.

    ``time_us`` is any duration estimate: measured kernel time, the roofline
    ceiling, or an Origami simulation. A missing or non-positive duration
    yields NaN for both rates. A missing ``bytes_moved`` yields NaN TB/s.
    """
    if time_us is None or not (time_us > 0):
        return float("nan"), float("nan")
    tflops_per_s = (gflops / 1e3) / (time_us / 1e6)
    if bytes_moved is None:
        tb_per_s = float("nan")
    else:
        tb_per_s = (bytes_moved / 1e12) / (time_us / 1e6)
    return tflops_per_s, tb_per_s


def add_duration_rate_columns(dict_metrics, gflops, bytes_moved, time_us, prefix):
    """Write ``"{prefix} TFLOPS/s"`` and ``"{prefix} TB/s"`` for ``time_us``.

    ``prefix`` names the estimate, for example ``"Roofline"`` or ``"Origami"``.
    """
    tflops_per_s, tb_per_s = rates_for_duration(gflops, bytes_moved, time_us)
    dict_metrics[f"{prefix} TFLOPS/s"] = tflops_per_s
    dict_metrics[f"{prefix} TB/s"] = tb_per_s
    return tflops_per_s, tb_per_s


def add_simulation_time_columns(
    dict_metrics,
    simulated_time,
    gflops,
    bytes_moved,
    busy_kernel_time,
    label="Origami",
):
    """
    Add simulated time columns (Origami, or ``label`` for other op projections).
    For dict outputs with extra columns, use
    ``TraceLens.PerfModel.op_projections.add_op_projection_columns``.
    """
    if not simulated_time:
        return
    dict_metrics[f"{label} Time (µs)"] = simulated_time
    add_duration_rate_columns(
        dict_metrics, gflops, bytes_moved, simulated_time, prefix=label
    )
    dict_metrics[f"Pct {label}"] = (
        (simulated_time / busy_kernel_time) * 100
        if busy_kernel_time > 0
        else float("nan")
    )


def build_perf_metrics_dict(gflops, bytes_moved, busy_kernel_time):
    """
    Build the standard GFLOPS/TFLOPS/TB-per-s metrics dict shared by the
    PyTorch and JAX perf-metric code paths.
    """
    tflops_per_s, tb_per_s = rates_for_duration(gflops, bytes_moved, busy_kernel_time)
    dict_metrics = {
        "GFLOPS": gflops,
        "Kernel Time (µs)": busy_kernel_time,
        "TFLOPS/s": tflops_per_s,
    }
    if bytes_moved is not None:
        dict_metrics["Data Moved (MB)"] = bytes_moved / (1024 * 1024)
        dict_metrics["FLOPS/Byte"] = (
            (gflops * 1e9) / bytes_moved if bytes_moved > 0 else float("nan")
        )
        dict_metrics["TB/s"] = tb_per_s
    else:
        dict_metrics["Data Moved (MB)"] = float("nan")
        dict_metrics["FLOPS/Byte"] = float("nan")
        dict_metrics["TB/s"] = float("nan")
    return dict_metrics


def gemm_tflops(M, N, K, time_ms):
    """Achieved TFLOPS for a dense GEMM of shape (M, N, K) given elapsed time in ms."""
    return (2 * M * N * K) / (time_ms * 1e-3) / 1e12


def dense_layout(shape, stride):
    """True when ``stride`` is dense, including column-major; a missing stride counts as dense."""
    if not shape or not stride or len(stride) != len(shape):
        return True
    pairs = [(size, step) for size, step in zip(shape, stride) if size > 1]
    if any(step <= 0 for _, step in pairs):
        return False
    pairs.sort(key=lambda item: item[1])
    expect = 1
    for size, step in pairs:
        if step != expect:
            return False
        expect *= size
    return True


def dense_counted_tensor(event, indexes):
    """False when a counted tensor's recorded stride is not dense."""
    args = event.get("args") or {}
    dims = args.get("Input Dims") or []
    strides = args.get("Input Strides") or []
    if not strides:
        return True
    for index in indexes:
        if index >= len(dims) or index >= len(strides):
            continue
        if not dense_layout(dims[index], strides[index]):
            return False
    return True


# Bytes per traced element of each ``c10::ScalarType``, keyed by the type's
# own name. ``Float4_e2m1fn_x2`` packs two FP4 values into one byte, and a
# traced shape already counts those packed elements (K_packed), the same way
# ``QUInt4x2`` and ``Bits4x2`` do. Callers that count unpacked logical
# elements divide by two themselves.
_DTYPE_BPE = {
    "Byte": 1,
    "Char": 1,
    "Short": 2,
    "Int": 4,
    "Long": 8,
    "Half": 2,
    "Float": 4,
    "Double": 8,
    "ComplexHalf": 4,
    "ComplexFloat": 8,
    "ComplexDouble": 16,
    "Bool": 1,
    "QInt8": 1,
    "QUInt8": 1,
    "QInt32": 4,
    "BFloat16": 2,
    "QUInt4x2": 1,
    "QUInt2x4": 1,
    "Bits1x8": 1,
    "Bits2x4": 1,
    "Bits4x2": 1,
    "Bits8": 1,
    "Bits16": 2,
    "Float8_e5m2": 1,
    "Float8_e4m3fn": 1,
    "Float8_e5m2fnuz": 1,
    "Float8_e4m3fnuz": 1,
    "UInt16": 2,
    "UInt32": 4,
    "UInt64": 8,
    "Float8_e8m0fnu": 1,
    "Float4_e2m1fn_x2": 1,
    "BComplex32": 4,
}

# Spellings that are not a ``ScalarType`` name. Profiler traces say "long int"
# for Long, and the roofline says "bf16" for BFloat16. "scalar" is the
# profiler's non-tensor argument, which has always counted as 4 bytes.
_DTYPE_NAME_ALIASES = {
    "long int": "Long",
    "unsigned char": "Byte",
    "signed char": "Char",
    "float16": "Half",
    "float32": "Float",
    "fp16": "Half",
    "fp32": "Float",
    "fp64": "Double",
    "bf16": "BFloat16",
    "fp8": "Float8_e4m3fn",
    # One logical FP4 value. gemm_afp4wfp4 and the roofline count unpacked
    # elements, so this width stays half a byte. The traced container
    # Float4_e2m1fn_x2 is one byte.
    "fp4": 0.5,
    "scalar": 4,
}

_NAME_TO_BPE = {name.lower(): bpe for name, bpe in _DTYPE_BPE.items()}
for _alias, _target in _DTYPE_NAME_ALIASES.items():
    _NAME_TO_BPE[_alias] = _DTYPE_BPE[_target] if isinstance(_target, str) else _target

# ``aten::to`` records the destination dtype as a ``ScalarType`` ordinal in
# ``Concrete Inputs``, not as a name. This is the ordinal order of
# ``AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS``.
_SCALAR_TYPE_ORDINAL_TO_NAME = {
    0: "Byte",
    1: "Char",
    2: "Short",
    3: "Int",
    4: "Long",
    5: "Half",
    6: "Float",
    7: "Double",
    8: "ComplexHalf",
    9: "ComplexFloat",
    10: "ComplexDouble",
    11: "Bool",
    12: "QInt8",
    13: "QUInt8",
    14: "QInt32",
    15: "BFloat16",
    16: "QUInt4x2",
    17: "QUInt2x4",
    18: "Bits1x8",
    19: "Bits2x4",
    20: "Bits4x2",
    21: "Bits8",
    22: "Bits16",
    23: "Float8_e5m2",
    24: "Float8_e4m3fn",
    25: "Float8_e5m2fnuz",
    26: "Float8_e4m3fnuz",
    27: "UInt16",
    28: "UInt32",
    29: "UInt64",
    44: "Float8_e8m0fnu",
    45: "Float4_e2m1fn_x2",
    46: "BComplex32",
}


def name2bpe(name):
    """Bytes per element for a dtype name, or ``None`` when it is not one.

    Accepts a ``c10::ScalarType`` name (``BFloat16`` or ``c10::BFloat16``),
    a profiler spelling (``long int``, ``unsigned char``), and a simulation
    name (``bf16``, ``fp32``). ``Float4_e2m1fn_x2`` is one byte per traced
    element (two FP4 values). ``fp4`` is one logical value, half a byte, for
    callers that have already unpacked K.
    """
    if name is None:
        return None
    key = str(name).strip().lower().split("::")[-1]
    if not key:
        return None
    return _NAME_TO_BPE.get(key)


def scalar_type_bpe(token):
    """Bytes of a profiler ``ScalarType`` ordinal, or ``None`` when it is not one.

    The ordinal is only a way to recover the type name. The width comes from
    ``name2bpe``.
    """
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    try:
        code = int(text)
    except ValueError:
        return None
    name = _SCALAR_TYPE_ORDINAL_TO_NAME.get(code)
    if name is None:
        return None
    return name2bpe(name)


# Keys are stored without a namespace, because torch_dtype_map strips one off
# its input before looking it up. That way "c10::Half" and the already-stripped
# "half" take the same path, and a new c10 spelling of a type already listed
# here needs no entry of its own.
_DTYPE2SIMULATION = {
    "float": "fp32",
    "double": "fp64",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "half": "fp16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "fp8",
    "float8_e4m3fnuz": "fp8",
    "float4_e2m1fn_x2": "fp4",
    "mxfp4": "fp4",
    "unsigned char": "fp8",
}

# Canonical values map to themselves, so normalising an already-normalised
# value is a no-op. Derived rather than written out, so adding a type above is
# enough.
_DTYPE2SIMULATION.update({value: value for value in set(_DTYPE2SIMULATION.values())})


def torch_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a simulation data type.

    Accepts the c10 spelling ("c10::BFloat16"), the same name with the
    namespace already stripped ("bfloat16"), and the canonical simulation name
    ("bf16").

    Args:
        dtype (str): The name of the PyTorch data type.
    Returns:
        str: The name of the simulation data type, or None if the type has no
            simulation equivalent — integer, boolean and complex types among
            them, which have no place in a floating-point roofline.
    """
    if dtype is None:
        return None
    return _DTYPE2SIMULATION.get(str(dtype).lower().split("::")[-1], None)


def parse_bool(input):
    if isinstance(input, bool):
        return input
    if input is None:
        return False
    if isinstance(input, str):
        value = input.strip().lower()
        if value in {"true", "1"}:
            return True
        if value in {"false", "0", ""}:
            return False
    return bool(input)
