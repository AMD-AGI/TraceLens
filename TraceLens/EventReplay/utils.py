###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Dict, Tuple, Any
import time

_torch_module = None


def _get_torch_or_raise() -> Any:  # Changed return type to Any for flexibility
    """Lazily imports and returns the torch module."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch

            _torch_module = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for EventReplayer functionality that is being used. "
                "Please install PyTorch."
            )
    return _torch_module


list_profile_tensor_types = [
    "double",
    "float",
    "c10::Half",
    "c10::BFloat16",
    "long",
    "long int",
    "int",
    "bool",
    "unsigned char",
    "char",
    "short",
    "c10::Float8_e4m3fnuz",
    "c10::Float8_e5m2fnuz",
    "c10::Float8_e4m3fn",
    "c10::Float8_e5m2",
]

from dataclasses import dataclass


@dataclass
class TensorCfg:
    """
    A class to represent a dummy tensor.
    """

    shape: List[int]
    dtype: str
    strides: List[int]
    init: str = "normal"


def build_tensor(cfg: TensorCfg, device: str = "cuda") -> "torch.Tensor":

    torch = _get_torch_or_raise()
    dict_profile2torchdtype = {
        "bool": torch.bool,
        "int": torch.int,
        "long": torch.long,
        "long int": torch.long,
        "short": torch.short,
        "char": torch.int8,
        "unsigned char": torch.uint8,
        "double": torch.float64,
        "float": torch.float32,
        "c10::Half": torch.float16,
        "c10::BFloat16": torch.bfloat16,
    }
    # FP8 types (available in PyTorch >= 2.1)
    for fp8_name in (
        "c10::Float8_e4m3fnuz",
        "c10::Float8_e5m2fnuz",
        "c10::Float8_e4m3fn",
        "c10::Float8_e5m2",
    ):
        attr = fp8_name.replace("c10::", "")
        torch_dtype = getattr(torch, attr.lower(), None)
        if torch_dtype is not None:
            dict_profile2torchdtype[fp8_name] = torch_dtype

    if cfg.dtype not in dict_profile2torchdtype:
        raise ValueError(
            f"Unknown profiled dtype '{cfg.dtype}'. "
            f"Known types: {list(dict_profile2torchdtype.keys())}"
        )
    dtype = dict_profile2torchdtype[cfg.dtype]
    size = cfg.shape
    stride = cfg.strides
    t = torch.empty_strided(size, stride, dtype=dtype, device=device)
    is_floating = t.is_floating_point() or t.is_complex()
    init = cfg.init
    if init == "normal":
        if not is_floating:
            raise ValueError(
                f"Cannot initialize tensor of type {cfg.dtype} with 'normal' init."
            )
        t.normal_()
    elif init == "zeros":
        t.zero_()
    elif init is not None:
        raise ValueError(f"Unsupported tensor initialization: {init}")
    return t


def summarize_tensor(tensor: "torch.Tensor") -> str:
    """
    Summarize the tensor information.

    Args:
        tensor (torch.Tensor): The tensor to summarize.

    Returns:
        str: The summary string.
    """
    return f"Tensor(shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, strides={tensor.stride()})"


_L2_FLUSH_BUFFER = None
_L2_FLUSH_SIZE = 256 * 1024 * 1024  # 256 MB -- larger than any GPU's L2


def _flush_l2(device: str):
    """Force-evict GPU L2 cache by reading a large buffer."""
    global _L2_FLUSH_BUFFER
    torch = _get_torch_or_raise()
    if _L2_FLUSH_BUFFER is None or str(_L2_FLUSH_BUFFER.device) != device:
        _L2_FLUSH_BUFFER = torch.empty(
            _L2_FLUSH_SIZE // 4, dtype=torch.float32, device=device
        )
    _L2_FLUSH_BUFFER.sum()


def benchmark_func(
    func,
    device,
    warmup=50,
    avg_steps=100,
    flush_l2=False,
):
    """Benchmark a function with warmup and per-iteration CUDA event timing.

    Args:
        func: Callable to benchmark.
        device: CUDA device string.
        warmup: Number of warmup iterations.
        avg_steps: Number of measured iterations.
        flush_l2: If True, flush the GPU L2 cache before each measured iteration
            to simulate cold-cache conditions (more representative of real
            inference where other kernels pollute L2 between invocations).

    Returns:
        dict with keys: median_us, mean_us, std_us, min_us, max_us,
        all_us (list of per-iteration timings in microseconds).
    """
    torch = _get_torch_or_raise()

    for _ in range(warmup):
        func()
    torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    for _ in range(avg_steps):
        if flush_l2:
            _flush_l2(device)
            torch.cuda.synchronize(device)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        func()
        end_evt.record()
        torch.cuda.synchronize(device)
        timings_ms.append(start_evt.elapsed_time(end_evt))

    timings_us = [t * 1000.0 for t in timings_ms]
    sorted_us = sorted(timings_us)
    n = len(sorted_us)
    median = (sorted_us[n // 2] + sorted_us[(n - 1) // 2]) / 2.0
    mean = sum(timings_us) / n
    variance = sum((t - mean) ** 2 for t in timings_us) / n
    std = variance ** 0.5
    return {
        "median_us": median,
        "mean_us": mean,
        "std_us": std,
        "min_us": sorted_us[0],
        "max_us": sorted_us[-1],
        "all_us": timings_us,
    }
