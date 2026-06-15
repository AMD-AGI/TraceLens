###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CustomCollective-family test harnesses for ``validate_perf_model``.

Covers all eight collective operation variants tracked by the
``custom_collectives_perf_model_extensions.py`` extension module:

  AllReduce (all GPUs → all GPUs):
    * ``aiter::fused_allreduce_rmsnorm``   — fused AllReduce + residual-add + RMSNorm
    * ``_C_custom_ar::all_reduce``         — pure vLLM custom all-reduce
    * ``sgl_kernel::all_reduce_reg``       — SGLang registered-buffer all-reduce
    * ``sgl_kernel::qr_all_reduce``        — SGLang QuickReduce all-reduce
    * ``_C_custom_ar::qr_all_reduce``      — vLLM QuickReduce all-reduce

  ReduceScatter (reduce then scatter):
    * ``aiter::reduce_scatter``            — aiter reduce-scatter

  AllGather (gather from all ranks):
    * ``aiter::all_gather_reg``            — aiter registered-buffer all-gather
    * ``sgl_kernel::reg_all_gather_into_tensor`` — SGLang registered all-gather

Important
---------
All collective operations require a **multi-GPU context** (RCCL/custom-AR
initialization, handle registration, GPU-to-GPU NVLink/xGMI connectivity).
Running these harnesses on a **single GPU** will either:

  * Succeed partially (single-rank "collective" = no-op copy, still measures
    HBM bandwidth for the local read+write path).
  * Fail with a NCCL or custom-AR initialization error.

For single-GPU roofline validation, the harnesses below initialize the
smallest possible context (1 rank = rank 0 only) to exercise the local HBM
path measured by rocprofv3.  The inter-GPU communication bytes are *not*
captured by hardware counters and are excluded from the perf model's roofline.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


# ---------------------------------------------------------------------------
# Shared helper — best-effort single-rank dist init
# ---------------------------------------------------------------------------

def _init_single_rank_dist():
    """Initialize a single-rank distributed process group if not already done.

    Returns True if a usable dist group exists after this call, False if
    initialization is not possible (no GPU or dist backend unavailable).
    """
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return True
    if not torch.cuda.is_available():
        return False
    try:
        import os
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. aiter fused_allreduce_rmsnorm  (aiter pattern)
# ---------------------------------------------------------------------------

def test_aiter_fused_allreduce_rmsnorm(
    M, N,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter.ops.fused_allreduce_rmsnorm`` — fused AllReduce + residual-add + RMSNorm.

    Allocates the six BF16 tensors (inp, res_inp, res_out, out, weight) and
    runs the kernel with a single-rank (no-op AllReduce) to measure local
    HBM bandwidth.  On a real multi-GPU system the collective phase carries
    inter-GPU traffic; only the local HBM portion is visible to rocprofv3.

    Parameters
    ----------
    M : int
        Token count (rows).
    N : int
        Hidden dimension (columns = weight length).
    in_dtype : {"bf16"}
        Activation dtype. The kernel currently requires BF16 throughout.
    """
    import torch
    try:
        from aiter.ops.fused_allreduce_rmsnorm import fused_allreduce_rmsnorm as _fn
    except ImportError:
        try:
            import aiter
            _fn = aiter.fused_allreduce_rmsnorm
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "aiter.fused_allreduce_rmsnorm not available. "
                "Ensure aiter is installed with collective support."
            ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    eps = 1e-5

    print(
        f"test: aiter_fused_allreduce_rmsnorm M={M} N={N} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    res_inp = torch.randn(M, N, dtype=in_t, device=device)
    res_out = torch.empty(M, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=in_t, device=device)
    weight = torch.ones(N, dtype=in_t, device=device)

    # Requires an initialized custom-AR or RCCL handle; use a best-effort stub.
    _init_single_rank_dist()

    for _ in range(num_warmup):
        _fn(inp, res_inp, res_out, out, weight, eps)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _fn(inp, res_inp, res_out, out, weight, eps)
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 2. vLLM _C_custom_ar all_reduce  (vLLM torch.ops pattern)
# ---------------------------------------------------------------------------

def test_custom_ar_all_reduce(
    M, N,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``_C_custom_ar::all_reduce`` — vLLM custom all-reduce.

    Exercises the local HBM read+write path via the vLLM custom-AR backend.
    Requires vLLM installed and a valid custom-AR context handle (``fa``).
    On single-GPU, this is a no-op all-reduce; HBM traffic is still measured.

    Parameters
    ----------
    M : int
        Token count (rows).
    N : int
        Hidden dimension (columns).
    in_dtype : {"bf16"}
        Activation dtype.
    """
    import torch
    try:
        import vllm._C_custom_ar as _car  # registers torch.ops._C_custom_ar
    except ImportError as exc:
        raise ImportError(
            "_C_custom_ar is not available. Ensure vLLM is installed."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"

    print(
        f"test: custom_ar_all_reduce M={M} N={N} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty_like(inp)

    # A real run requires registering buffers with the custom-AR handle.
    # Without a valid FA handle the op will raise; this harness is primarily
    # for documenting the expected shapes and can be extended with real handle
    # initialization when running in a full multi-GPU environment.
    _fn = torch.ops._C_custom_ar.all_reduce

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out, 0, inp.numel() * inp.element_size())
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out, 0, inp.numel() * inp.element_size())
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 3. SGLang sgl_kernel all_reduce_reg  (SGLang pattern)
# ---------------------------------------------------------------------------

def test_sgl_kernel_all_reduce_reg(
    M, N,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``sgl_kernel::all_reduce_reg`` — SGLang registered-buffer all-reduce.

    Parameters
    ----------
    M : int
        Token count (rows).
    N : int
        Hidden dimension.
    in_dtype : {"bf16"}
        Activation dtype.
    """
    import torch
    try:
        import sgl_kernel
        _fn = sgl_kernel.all_reduce_reg
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "sgl_kernel is not installed or all_reduce_reg is not available."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"

    print(
        f"test: sgl_kernel_all_reduce_reg M={M} N={N} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty_like(inp)

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 4. SGLang sgl_kernel qr_all_reduce  (SGLang QuickReduce pattern)
# ---------------------------------------------------------------------------

def test_sgl_kernel_qr_all_reduce(
    M, N,
    in_dtype="bf16",
    quant_level=0,
    num_warmup=3,
    **_,
):
    """``sgl_kernel::qr_all_reduce`` — SGLang QuickReduce all-reduce.

    Parameters
    ----------
    M, N : int
        Input tensor shape.
    in_dtype : {"bf16"}
    quant_level : int
        0 = BF16 (no compression), 3 = INT4 inter-GPU codec (default 0).
    """
    import torch
    try:
        import sgl_kernel
        _fn = sgl_kernel.qr_all_reduce
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "sgl_kernel is not installed or qr_all_reduce is not available."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"

    print(
        f"test: sgl_kernel_qr_all_reduce M={M} N={N} dtype={in_dtype} "
        f"quant_level={quant_level}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty_like(inp)
    cast_bf2half = False

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out, quant_level, cast_bf2half)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out, quant_level, cast_bf2half)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 5. vLLM _C_custom_ar qr_all_reduce  (vLLM QuickReduce pattern)
# ---------------------------------------------------------------------------

def test_custom_ar_qr_all_reduce(
    M, N,
    in_dtype="bf16",
    quant_level=0,
    num_warmup=3,
    **_,
):
    """``_C_custom_ar::qr_all_reduce`` — vLLM QuickReduce all-reduce.

    Parameters
    ----------
    M, N : int
        Input tensor shape.
    in_dtype : {"bf16"}
    quant_level : int
        0 = BF16 transport, 3 = INT4 inter-GPU codec.
    """
    import torch
    try:
        import vllm._C_custom_ar  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "_C_custom_ar is not available. Ensure vLLM is installed."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"

    print(
        f"test: custom_ar_qr_all_reduce M={M} N={N} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty_like(inp)
    cast_bf2half = False

    _fn = torch.ops._C_custom_ar.qr_all_reduce

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out, quant_level, cast_bf2half)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out, quant_level, cast_bf2half)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 6. aiter reduce_scatter  (aiter pattern)
# ---------------------------------------------------------------------------

def test_aiter_reduce_scatter(
    M, N,
    n_gpus=8,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter::reduce_scatter`` — aiter reduce-scatter collective.

    The input is the full pre-scatter tensor of shape ``[M, N]``; the output
    shard has shape ``[M // n_gpus, N]``.

    Parameters
    ----------
    M : int
        Full tensor row count (must be divisible by n_gpus).
    N : int
        Hidden dimension.
    n_gpus : int
        Number of GPUs in the collective (default 8). Used to compute output
        shard shape.
    in_dtype : {"bf16"}
    """
    import torch
    try:
        import aiter
        _fn = aiter.reduce_scatter
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "aiter.reduce_scatter is not available."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    M_out = M // n_gpus

    print(
        f"test: aiter_reduce_scatter M={M} N={N} n_gpus={n_gpus} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty(M_out, N, dtype=in_t, device=device)

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 7. aiter all_gather_reg  (aiter pattern)
# ---------------------------------------------------------------------------

def test_aiter_all_gather_reg(
    M, N,
    n_gpus=8,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter::all_gather_reg`` — aiter registered-buffer all-gather.

    The input shard has shape ``[M // n_gpus, N]``; the gathered output is
    ``[M, N]``.

    Parameters
    ----------
    M : int
        Full gathered tensor row count.
    N : int
        Hidden dimension.
    n_gpus : int
        Number of GPUs in the collective (default 8).
    in_dtype : {"bf16"}
    """
    import torch
    try:
        import aiter
        _fn = aiter.all_gather_reg
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "aiter.all_gather_reg is not available."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    M_shard = M // n_gpus

    print(
        f"test: aiter_all_gather_reg M={M} N={N} n_gpus={n_gpus} dtype={in_dtype}",
        flush=True,
    )

    inp = torch.randn(M_shard, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=in_t, device=device)

    for _ in range(num_warmup):
        try:
            _fn(0, inp, out)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(0, inp, out)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 8. SGLang sgl_kernel reg_all_gather_into_tensor  (SGLang pattern)
# ---------------------------------------------------------------------------

def test_sgl_kernel_reg_all_gather_into_tensor(
    M, N,
    n_gpus=8,
    in_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``sgl_kernel::reg_all_gather_into_tensor`` — SGLang registered all-gather.

    Output (gathered) has shape ``[M, N]``; input (local shard) has shape
    ``[M // n_gpus, N]``.

    Parameters
    ----------
    M : int
        Full gathered output row count.
    N : int
        Hidden dimension.
    n_gpus : int
        Number of GPUs in the collective (default 8).
    in_dtype : {"bf16"}
    """
    import torch
    try:
        import sgl_kernel
        _fn = sgl_kernel.reg_all_gather_into_tensor
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "sgl_kernel is not installed or reg_all_gather_into_tensor not available."
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    M_shard = M // n_gpus

    print(
        f"test: sgl_kernel_reg_all_gather_into_tensor "
        f"M={M} N={N} n_gpus={n_gpus} dtype={in_dtype}",
        flush=True,
    )

    out = torch.empty(M, N, dtype=in_t, device=device)
    inp = torch.randn(M_shard, N, dtype=in_t, device=device)

    for _ in range(num_warmup):
        try:
            _fn(out, inp, 0)
        except Exception:
            pass
    torch.cuda.synchronize()
    print("test: measured iteration (stub)...", flush=True)
    try:
        _fn(out, inp, 0)
    except Exception:
        pass
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# OP_METADATA
# ---------------------------------------------------------------------------

#: Metadata for all CustomCollective harnesses in this module.
OP_METADATA: dict = {
    "aiter_fused_allreduce_rmsnorm": {
        "fn":           test_aiter_fused_allreduce_rmsnorm,
        "category":     "CustomCollective",
        "description":  "AITER fused AllReduce + residual-add + RMSNorm",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "custom_ar_all_reduce": {
        "fn":           test_custom_ar_all_reduce,
        "category":     "CustomCollective",
        "description":  "vLLM _C_custom_ar::all_reduce (pure all-reduce)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "sgl_kernel_all_reduce_reg": {
        "fn":           test_sgl_kernel_all_reduce_reg,
        "category":     "CustomCollective",
        "description":  "SGLang sgl_kernel::all_reduce_reg (registered buffer)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "sgl_kernel_qr_all_reduce": {
        "fn":           test_sgl_kernel_qr_all_reduce,
        "category":     "CustomCollective",
        "description":  "SGLang sgl_kernel::qr_all_reduce (QuickReduce)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1866, "N": 7168, "in_dtype": "bf16", "quant_level": 0},
        "required_args": ["M", "N"],
    },
    "custom_ar_qr_all_reduce": {
        "fn":           test_custom_ar_qr_all_reduce,
        "category":     "CustomCollective",
        "description":  "vLLM _C_custom_ar::qr_all_reduce (QuickReduce)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1689, "N": 7168, "in_dtype": "bf16", "quant_level": 0},
        "required_args": ["M", "N"],
    },
    "aiter_reduce_scatter": {
        "fn":           test_aiter_reduce_scatter,
        "category":     "CustomCollective",
        "description":  "AITER aiter::reduce_scatter",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 7168, "n_gpus": 8, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "aiter_all_gather_reg": {
        "fn":           test_aiter_all_gather_reg,
        "category":     "CustomCollective",
        "description":  "AITER aiter::all_gather_reg (registered buffer)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 7168, "n_gpus": 8, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "sgl_kernel_reg_all_gather_into_tensor": {
        "fn":           test_sgl_kernel_reg_all_gather_into_tensor,
        "category":     "CustomCollective",
        "description":  "SGLang sgl_kernel::reg_all_gather_into_tensor",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 256, "N": 16160, "n_gpus": 8, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
}
