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
    Add simulated time columns (Origami, or ``label`` for other op models).
    For dict outputs with extra columns, use
    ``TraceLens.PerfModel.op_models.add_op_model_columns``.
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


def name2bpe(name):
    """
    This function maps a data type name to the number of bytes per element.
    Args:
        name (str): The name of the data type.
    Returns:
        int: The number of bytes per element.
    """
    dict_bpe2dtype = {
        8: ["double", "long int"],
        4: ["float", "scalar", "int"],
        2: ["c10::half", "c10::bfloat16"],
        1: [
            "c10::float8_e4m3fnuz",
            "c10::float8_e4m3fn",
            "c10::float8_e5m2",
            "c10::float8_e8m0fnu",
            "unsigned char",
            "signed char",
            "fp8",
            # Float4_e2m1fn_x2 packs two FP4 values into one byte. Trace tensor
            # shapes already reflect the packed layout (K_packed = K/2), so we
            # use bpe=1 for the packed-pair element and let callers apply the
            # ×2 K-unpacking explicitly when modelling FLOPs.
            "c10::float4_e2m1fn_x2",
            "fp4",
        ],
    }
    dict_dtype2bpe = {
        dtype: bpe for bpe, dtypes in dict_bpe2dtype.items() for dtype in dtypes
    }
    if name is None:
        return None
    return dict_dtype2bpe.get(name.lower(), None)


def simulation_dtype_map(dtype):
    """
    This function maps a PyTorch data type to a simulation data type.
    Args:
        dtype (str): The name of the pytorch data type.
    Returns:
        str: The name of the PyTorch data type.
    """
    dict_dtype2simulation = {
        "fp32": "float",
        "fp64": "double",
        "fp16": "c10::half",
        "bf16": "c10::bfloat16",
        "fp8": "c10::float8_e4m3fnuz",
    }
    return dict_dtype2simulation.get(dtype.lower(), None)


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
