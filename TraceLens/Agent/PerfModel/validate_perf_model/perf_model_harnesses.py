###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Registry of the ops that ``validate_perf_model.py`` can validate.

Each op is declared once with :func:`register`. The decorated builder turns
CLI dimensions (an ``argparse.Namespace``) into a synthetic TraceLens event,
and the registry pairs it with the perf-model class that scores that event.
``OP_REGISTRY[key]["model_fn"](args)`` returns ``(flops, bytes, precision)``.

Adding an op
------------
1. Add a ``@register("<key>", ...)`` builder below, in the section for its
   op family. ``model=(source, name)``: ``source`` is a module constant
   (``PERF_MODEL_EXT``, ``ATTN_EXT``, ``CORE_OP_MAP``, ...); ``name`` is the
   class name or its trace op name. ``trace_names`` lists the trace op names
   that map to the op in ``--cases`` mode.
2. Add ``test_<key>(**kwargs)`` to the matching ``tests/<module>.py``. The
   runner finds it by name and rocprofv3 profiles it.
3. Ops validated only from ``--cases`` have no builder: add them to
   the ``_CSV_ONLY_OPS`` table and to ``tests/_generic.OP_CALL_SPEC``.
"""

import functools
import importlib
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path

# Constants
PERF_MODEL_EXT = "TraceLens.PerfModel.extensions.perf_model_extensions"
MOE_EXT = "TraceLens.PerfModel.extensions.moe_perf_model_extensions"
RMSNORM_EXT = "TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions"
ATTN_EXT = "TraceLens.PerfModel.extensions.attention_perf_model_extensions"
PERF_MODEL = "TraceLens.PerfModel.perf_model"
CORE_OP_MAP = "TraceLens.PerfModel.torch_op_mapping"

_PARENTS = Path(__file__).resolve().parents
# <repo>/TraceLens/Agent/PerfModel/validate_perf_model/; staging copies of the
# skill may sit fewer than five levels deep.
REPO_ROOT = str(_PARENTS[4] if len(_PARENTS) > 4 else _PARENTS[-1])
if importlib.util.find_spec("TraceLens") is None:
    sys.path.insert(0, REPO_ROOT)

_BF16 = "c10::BFloat16"
_FP8 = "c10::Float8_e4m3fn"
_E8M0 = "c10::Float8_e8m0fnu"
_TRACE_DTYPE_MAP = {
    "bf16": "c10::BFloat16",
    "fp16": "c10::Half",
    "fp32": "float",
    "fp64": "double",
    "fp8": "c10::Float8_e4m3fnuz",
    "fp8_e4m3fn": "c10::Float8_e4m3fn",
    "fp8_e4m3fnuz": "c10::Float8_e4m3fnuz",
    "fp8_e5m2": "c10::Float8_e5m2",
    "fp4": "c10::Float4_e2m1fn_x2",
    "fp4x2": "c10::Float4_e2m1fn_x2",
    "i8": "signed char",
    "u8": "unsigned char",
    "i32": "int",
    "u16": "unsigned short",
    "u32": "unsigned int",
}
_TRACE_DTYPE_NAMES = (
    "unsigned short",
    "unsigned char",
    "unsigned int",
    "signed char",
    "int",
    "float",
    "double",
    "Half",
    "BFloat16",
)

# (fn_name, description, trace_names); all share _activation_event.
_ACTIVATION_OPS = (
    (
        "silu_and_mul",
        "SiLU-gated activation (SwiGLU gate*up)",
        ("aiter::silu_and_mul", "_C::silu_and_mul"),
    ),
    ("gelu_and_mul", "GELU-gated activation", ("aiter::gelu_and_mul",)),
    ("gelu_tanh_and_mul", "GELU-tanh-gated activation", ("aiter::gelu_tanh_and_mul",)),
)
# (key, category, description, trace_name) for ops with no CLI builder. In
# --cases mode the prediction comes from the traced event and
# tests/_generic.OP_CALL_SPEC replays the traced call.
_CSV_ONLY_OPS = (
    (
        "batched_gemm_a16wfp4_",
        "gemm",
        "AITER batched BF16 x MXFP4 GEMM (triton)",
        "aiter::batched_gemm_a16wfp4_",
    ),
    (
        "fused_qk_rmsnorm",
        "norm",
        "AITER fused Q/K RMSNorm",
        "aiter::_fused_qk_rmsnorm_kernel",
    ),
    (
        "biased_grouped_topk_hip",
        "moe",
        "AITER biased grouped top-k MoE routing (HIP)",
        "aiter::biased_grouped_topk_hip",
    ),
    (
        "moe_sorting_fwd",
        "moe",
        "AITER MoE token sorting / expert bucketing",
        "aiter::moe_sorting_fwd",
    ),
    (
        "mxfp4_moe_sort_hip",
        "moe",
        "AITER MXFP4 MoE scale sorting (HIP)",
        "aiter::mxfp4_moe_sort_hip",
    ),
    (
        "sgl_kernel_silu_and_mul",
        "activation",
        "SGLang sgl_kernel SiLU-and-mul (needs an SGLang runtime)",
        "sgl_kernel::silu_and_mul",
    ),
    (
        "vllm_concat_and_cache_mla",
        "vllm_attention",
        "vLLM MLA KV-cache concat + write",
        "_C_cache_ops::concat_and_cache_mla",
    ),
    (
        "vllm_concat_mla_q",
        "vllm_attention",
        "vLLM MLA query concat (nope + pe)",
        "_C_cache_ops::concat_mla_q",
    ),
    (
        "vllm_per_token_group_fp8_quant",
        "vllm_quant",
        "vLLM per-token-group FP8 quantization",
        "_C::per_token_group_fp8_quant",
    ),
    (
        "flash_attn_varlen_forward",
        "attention",
        "flash-attn varlen forward (packed prefill)",
        "flash_attn::_flash_attn_varlen_forward",
    ),
)

OP_REGISTRY = {}
TRACE_NAME_TO_OP = {}


def register(
    key,
    *,
    category,
    description,
    model=None,
    defaults=None,
    required_args=None,
    trace_names=(),
    csv_harness=False,
    perf_model_only=False,
):
    """Register op ``key`` in :data:`OP_REGISTRY`; decorates its event builder.

    ``model`` is ``(source, name)`` as accepted by :func:`resolve_model_class`,
    or ``None`` for ops validated only from ``--cases`` (decorate
    ``None`` for those). ``defaults`` seed CLI args that the user leaves
    unset; ``required_args`` (default: all ``defaults`` keys) are the ones
    reported for the op. ``csv_harness`` replays report rows through
    ``test_<key>`` with dimensions extracted from the row instead of the
    generic runner. ``perf_model_only`` skips profiling for ops whose kernel
    cannot run standalone.
    """
    if key in OP_REGISTRY:
        raise ValueError(f"Op '{key}' is registered twice")
    defaults = dict(defaults or {})

    def decorator(build_event):
        OP_REGISTRY[key] = {
            "category": category,
            "description": description,
            "model": model,
            "build_event": build_event,
            "model_fn": functools.partial(_evaluate, key),
            "defaults": defaults,
            "required_args": list(defaults if required_args is None else required_args),
            "trace_names": tuple(trace_names),
            "csv_harness": csv_harness,
            "perf_model_only": perf_model_only,
        }
        for trace_name in trace_names:
            if trace_name in TRACE_NAME_TO_OP:
                raise ValueError(
                    f"Trace name '{trace_name}' maps to both "
                    f"'{TRACE_NAME_TO_OP[trace_name]}' and '{key}'"
                )
            TRACE_NAME_TO_OP[trace_name] = key
        return build_event

    return decorator


def resolve_model_class(source, name):
    """Return the perf-model class ``name`` from ``source``.

    ``source`` is a TraceLens module path. ``name`` is looked up in its
    ``op_to_perf_model_class_map`` first, then as an attribute, so both trace
    op names and class names resolve.
    """
    module = importlib.import_module(source)
    mapping = getattr(module, "op_to_perf_model_class_map", {})
    if name in mapping:
        return mapping[name]
    return getattr(module, name)


def model_source(key):
    """Return ``(class_name, source_file, source_line)`` for op ``key``'s model.

    ``source_file`` is relative to the repo root. The line is 0 when the class
    cannot be loaded.
    """
    model = OP_REGISTRY[key]["model"] if key in OP_REGISTRY else None
    if model is None:
        return (key, "unknown", 0)
    source, name = model
    try:
        cls = resolve_model_class(source, name)
        src_file = os.path.relpath(inspect.getfile(cls), REPO_ROOT)
        _, src_line = inspect.getsourcelines(cls)
    except (ImportError, AttributeError, KeyError, OSError, TypeError):
        return (name, source.replace(".", "/") + ".py", 0)
    return (cls.__name__, src_file, src_line)


def tuple_to_list(obj):
    """Recursively convert tuples to lists for event dict compatibility."""
    if isinstance(obj, tuple):
        return [tuple_to_list(x) for x in obj]
    return obj


def run_perf_model_from_event(
    trace_name,
    input_dims,
    input_types,
    input_strides,
    concrete_inputs,
    attn_params=None,
):
    """Evaluate the perf model for a ``unified_perf_summary.csv`` row.

    Builds the event from the row's columns and resolves the class from
    TraceLens's ``op_to_perf_model_class_map``.
    """
    event = {
        "name": trace_name,
        "args": {
            "Input Dims": tuple_to_list(input_dims),
            "Input type": list(input_types) if input_types else [],
        },
    }
    if input_strides:
        event["args"]["Input Strides"] = tuple_to_list(input_strides)
    if concrete_inputs:
        event["args"]["Concrete Inputs"] = list(concrete_inputs)
    if attn_params:
        event["annotation"] = chunk_stats_annotation(attn_params)
    event["kernel_names"] = ["placeholder_kernel"]
    event["kernel_details"] = [{"name": "placeholder_kernel"}]
    try:
        from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
    except ImportError:
        from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import (
            get_pseudo_op_mappings,
        )

        op_to_perf_model_class_map = get_pseudo_op_mappings()
    model_cls = op_to_perf_model_class_map.get(trace_name)
    if model_cls is None:
        raise ValueError(f"No perf model class found for trace name '{trace_name}'")
    model = model_cls(event)
    return (model.flops(), model.bytes(), model.get_compute_precision())


def _evaluate(key, args):
    """Build op ``key``'s event from ``args`` and score it with its perf model."""
    entry = OP_REGISTRY[key]
    if entry["model"] is None:
        raise NotImplementedError(
            "This op is only validated via --cases, which derives the "
            "prediction from the traced event rather than from CLI dimensions."
        )
    cls = resolve_model_class(*entry["model"])
    model = cls(entry["build_event"](args))
    return (model.flops(), model.bytes(), model.get_compute_precision())


def chunk_stats_annotation(params):
    """Rebuild a parseable step annotation from perf_params or attention CSV params.

    ``params`` carries the c_*/g_* sums. Request counts are not stored, so they
    are derived assuming uniform lengths within each phase (requests = sq^2 /
    sqsq), which is what harnesses that size tensors from the annotation need.
    """

    def _int(name):
        value = params.get(name, 0)
        if value is None or str(value) in ("nan", ""):
            return 0
        return int(float(value))

    def _requests(sq, sqsq):
        return max(1, round(sq * sq / sqsq)) if sq and sqsq else (1 if sq else 0)

    c_sq, c_sk, c_sqsq, c_sqsk = (_int(f"c_{s}") for s in ("sq", "sk", "sqsq", "sqsk"))
    g_sq, g_sk, g_sqsq, g_sqsk = (_int(f"g_{s}") for s in ("sq", "sk", "sqsq", "sqsk"))
    return (
        f"execute_0_context_{_requests(c_sq, c_sqsq)}"
        f"(sq{c_sq}sk{c_sk}sqsq{c_sqsq}sqsk{c_sqsk})"
        f"_generation_{_requests(g_sq, g_sqsq)}"
        f"(sq{g_sq}sk{g_sk}sqsq{g_sqsq}sqsk{g_sqsk})"
    )


def _trace_dtype_str(name, default):
    """Map a user-facing dtype string ("bf16", "fp8", ...) to a trace ``Input type``.

    Returns ``default`` when ``name`` is None or empty. A fully-qualified trace
    string (e.g. ``"c10::BFloat16"``) is returned unchanged so callers can pass
    either form.
    """
    if name is None or name == "":
        return default
    key = str(name).strip()
    lower = key.lower()
    if lower in _TRACE_DTYPE_MAP:
        return _TRACE_DTYPE_MAP[lower]
    if "::" in key or key in _TRACE_DTYPE_NAMES:
        return key
    raise ValueError(
        f"Unknown dtype string {name}. Known short names: {sorted(_TRACE_DTYPE_MAP)}"
    )


def _moe_dtypes_from_args(args, defaults):
    """Resolve ``(in_dtype, w_dtype, out_dtype)`` trace strings from ``args``.

    ``defaults`` is a 3-tuple of trace strings used when the corresponding CLI
    flag is unset.
    """
    in_t = _trace_dtype_str(getattr(args, "in_dtype", None), defaults[0])
    w_t = _trace_dtype_str(getattr(args, "w_dtype", None), defaults[1])
    out_t = _trace_dtype_str(getattr(args, "out_dtype", None), defaults[2])
    return (in_t, w_t, out_t)


def _dtype_from_args(args, attr_name, default_trace_str):
    """Resolve a single dtype attribute on ``args`` to a trace ``Input type``."""
    return _trace_dtype_str(getattr(args, attr_name, None), default_trace_str)


def _kv_dtype_from_args(args, default_trace_str):
    """Resolve ``args.kv_dtype`` (paged-attention KV cache dtype)."""
    return _trace_dtype_str(getattr(args, "kv_dtype", None), default_trace_str)


def _parse_unified_attention_annotation(annotation):
    """Parse a detailed step annotation in any format TraceLens parses.

    Accepts the vLLM ``execute_*``, SGLang ``step[...]`` and ATOM detailed
    formats through ``IterationAnnotation``. Returns ctx_req/ctx_sq/ctx_sk/
    ctx_sqsq/ctx_sqsk and gen_req/gen_sq/gen_sk/gen_sqsq/gen_sqsk, or None when
    the annotation is empty or has no sq/sk sums.
    """
    # TraceLens may only be importable after the REPO_ROOT path fix above.
    from TraceLens.TraceUtils.utils.annotation_utils import IterationAnnotation

    if not annotation:
        return None
    try:
        ann = IterationAnnotation(str(annotation))
        stats = ann.chunk_stats()
    except (NotImplementedError, ValueError):
        return None
    parsed = {"ctx_req": ann.context_requests, "gen_req": ann.generation_requests}
    for phase, prefix in (("ctx", "c"), ("gen", "g")):
        for term in ("sq", "sk", "sqsq", "sqsk"):
            parsed[f"{phase}_{term}"] = stats[f"{prefix}_{term}"]
    return parsed


def _a8w8_blockscale_event(name, args):
    """Event for an FP8 / INT8 block-scaled GEMM laid out like ``gemm_a8w8_blockscale``."""
    K = args.K
    N = args.N
    M = args.M
    block_k = getattr(args, "block_k", None) or 128
    nkb = (K + block_k - 1) // block_k
    in_t = _dtype_from_args(args, "in_dtype", "unsigned char")
    w_t = _dtype_from_args(args, "w_dtype", "unsigned char")
    scl_t = _dtype_from_args(args, "scale_dtype", "float")
    return {
        "name": name,
        "args": {
            "Input Dims": [[M, K], [N, K], [M, nkb], [N, nkb]],
            "Input type": [in_t, w_t, scl_t, scl_t],
            "Input Strides": [[K, 1], [K, 1], [nkb, 1], [nkb, 1]],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }


def _activation_event(fn_name, args):
    """Event for ``aiter::<fn_name>`` gated activations (``out = act(gate) * up``).

    Reads ``in_dtype`` (slot ``[1]``, the gate||up input) and ``out_dtype``
    (slot ``[0]``). Defaults to BF16 for both.
    """
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    out_t = _dtype_from_args(args, "out_dtype", "c10::BFloat16")
    return {
        "name": f"aiter::{fn_name}",
        "args": {
            "Input Dims": [[M, N], [M, 2 * N]],
            "Input type": [out_t, in_t],
            "Input Strides": [[N, 1], [2 * N, 1]],
        },
    }


def _varlen_annotation(args):
    """Return ``(annotation, total_q, total_kv)`` for a varlen prefill call.

    The ``_varlen_*`` attributes are filled in by
    ``validate_perf_model._populate_varlen_attention_args`` from the same seed
    and scenario the test uses; without them the call is one ``seq_len``
    sequence. The generation block is zero because the kernels are
    prefill-only; ``c_sqsk`` equals ``c_sqsq`` for self-attention.
    """
    S = args.seq_len
    c_sq = getattr(args, "_varlen_c_sq", S)
    c_sqsq = getattr(args, "_varlen_c_sqsq", S * S)
    total_q = getattr(args, "_varlen_total_q", S)
    total_kv = getattr(args, "_varlen_total_kv", S)
    c_sqsk = getattr(args, "_varlen_c_sqsk", c_sqsq)
    n_ctx = getattr(args, "_varlen_n_ctx", 1)
    annotation = (
        f"execute_0_context_{n_ctx}(sq{c_sq}sk{total_kv}sqsq{c_sqsq}sqsk{c_sqsk})"
        f"_generation_0(sq0sk0sqsq0sqsk0)"
    )
    return (annotation, total_q, total_kv)


def _build_unified_attention_event(name, args):
    """Event for a unified (paged-KV) attention call named ``name``.

    A valid ``execute_*`` ``args.annotation`` is passed through verbatim.
    Otherwise one is synthesized from ``num_decode_seqs`` / ``ctx_len`` /
    ``prefill_seq_len`` (falling back to ``seq_len`` single-token decodes).
    Q / KV token totals are then read back from the annotation, the same way
    real vLLM traces are handled.
    """
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    kv_t = _kv_dtype_from_args(args, "c10::Float8_e4m3fnuz")
    annotation_str = getattr(args, "annotation", None)
    if not _parse_unified_attention_annotation(annotation_str):
        num_decode_seqs = getattr(args, "num_decode_seqs", None)
        ctx_len_arg = getattr(args, "ctx_len", None)
        p_len = int(getattr(args, "prefill_seq_len", 0) or 0)
        _seq_fallback = int(getattr(args, "seq_len", None) or 256)
        n_decode = (
            int(num_decode_seqs) if num_decode_seqs is not None else _seq_fallback
        )
        kv = int(ctx_len_arg) if ctx_len_arg is not None else _seq_fallback
        g_sqsk = n_decode * kv
        generation = (
            f"_generation_{n_decode}(sq{n_decode}sk{g_sqsk}sqsq{n_decode}sqsk{g_sqsk})"
        )
        if p_len > 0:
            c_sqsk = p_len * p_len
            context = f"context_1(sq{p_len}sk{p_len}sqsq{c_sqsk}sqsk{c_sqsk})"
        else:
            context = "context_0(sq0sk0sqsq0sqsk0)"
        annotation_str = f"execute_0_{context}{generation}"
    parsed = _parse_unified_attention_annotation(annotation_str)
    if parsed is None:
        raise ValueError(f"Cannot parse {name} annotation: {annotation_str!r}")
    sum_q = parsed["ctx_sq"] + parsed["gen_sq"]
    sum_kv = parsed["ctx_sk"] + parsed["gen_sk"]
    return {
        "name": name,
        "annotation": annotation_str,
        "args": {
            "Input Dims": [[sum_q, H_Q, d], [sum_kv, H_KV, d]],
            "Input type": [in_t, kv_t],
        },
    }


def _gdn_event(name, args, H_K, annotation):
    """Event for a Gated Delta Network call ``(mixed_qkv, b, a, core_attn_out, layer)``.

    Slot ``[0]`` sets the compute precision; ``gdn_attention_core.bytes()``
    hardcodes 2 BPE, so the dtype only changes the test runner side.
    """
    T = args.seq_len
    H_V = args.num_heads_kv
    d_k = args.head_dim
    d_v = args.head_dim
    D = 2 * H_K * d_k + H_V * d_v
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    return {
        "name": name,
        "annotation": annotation,
        "args": {
            "Input Dims": [[T, D], [T, H_V], [T, H_V], [T, H_V, d_v], []],
            "Input type": [in_t, in_t, in_t, in_t, ""],
        },
    }


def _dsv3_flydsl_moe_event(name, M, K, N, E, topk):
    """Build a parent ``aiter::fused_moe_`` style event for the flydsl pseudo ops.

    Layout matches ``_flydsl_extract_param_details``:
      dims[0] = [tokens, hidden]
      dims[1] = [E, 2*inter_dim, hidden/2]   (w1, fp4-packed K)
      dims[2] = [E, hidden, inter_dim/2]     (w2, fp4-packed K)
      dims[3] = [tokens, topk]
    """
    token, hidden, inter = M, K, N
    return {
        "name": name,
        "args": {
            "Input Dims": [
                [token, hidden],
                [E, 2 * inter, hidden // 2],
                [E, hidden, inter // 2],
                [token, topk],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::Float4_e2m1fn_x2",
                "c10::Float4_e2m1fn_x2",
                "float",
            ],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }


# ---------------------------------------------------------------------------
# GEMM and MoE
# ---------------------------------------------------------------------------
@register(
    "gemm_a8w8_blockscale",
    category="gemm",
    description="FP8 block-scaled GEMM (aiter CK)",
    model=(PERF_MODEL_EXT, "gemm_a8w8_blockscale"),
    defaults={"M": 2048, "N": 4096, "K": 8192},
    trace_names=(
        "aiter::gemm_a8w8_blockscale_ck",
        "aiter::gemm_a8w8_blockscale_cktile",
        "aiter::gemm_a8w8_blockscale_bpreshuffle_ck",
        "aiter::gemm_a8w8_blockscale_bpreshuffle_cktile",
        "aiter::flatmm_a8w8_blockscale_asm",
        "aiter::gfx950_a8w8_blockscale_asm",
        "aiter::gemm_a8w8_ck",
        "aiter::gemm_a8w8_bpreshuffle_ck",
        "sglang_profiler::fp8_utils_gemm_a8w8_blockscale_7",
    ),
)
def _gemm_a8w8_blockscale_event(args):
    """Event for ``aiter::gemm_a8w8_blockscale_ck`` (FP8 / INT8 block-scaled GEMM).

    Dtype kwargs read from ``args`` (defaults match
    :func:`tests.gemm.test_gemm_a8w8_blockscale`):

    * ``in_dtype`` -- activation storage (``"i8"``, ``"u8"``, ``"fp8"`` -- all 1 BPE).
    * ``w_dtype``  -- weight storage (``"i8"``, ``"u8"``, ``"fp8"`` -- all 1 BPE).
    * ``scale_dtype`` -- block-scale dtype (``"fp32"``).
    * ``block_k`` -- per-K-block tile size for scales (default 128).

    ``out_dtype`` is accepted for symmetry but the perf-model class hardcodes
    BF16 output (see ``perf_model_extensions.py:gemm_a8w8_blockscale``); it is
    forwarded to the test runner only.
    """
    return _a8w8_blockscale_event("aiter::gemm_a8w8_blockscale_ck", args)


@register(
    "gemm_a16w16_atomic_",
    category="gemm",
    description="BF16 GEMM (aiter ASM, atomic accumulation)",
    model=(PERF_MODEL_EXT, "gemm_a16w16_atomic_"),
    defaults={"M": 2048, "N": 4096, "K": 8192},
    trace_names=("aiter::gemm_a16w16_atomic_", "aiter::gemm_a16w16"),
)
def _gemm_a16w16_atomic__event(args):
    """Event for ``aiter::gemm_a16w16_atomic_`` (BF16 / FP16 GEMM).

    Dtype kwargs (``in_dtype`` / ``w_dtype`` / ``out_dtype``); kernel requires
    all three to match. Defaults to ``"bf16"``.
    """
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    out_t = _dtype_from_args(args, "out_dtype", "c10::BFloat16")
    event = {
        "name": "aiter::gemm_a16w16_atomic_",
        "args": {
            "Input Dims": [[M, K], [N, K], [], [M, N]],
            "Input type": [in_t, w_t, "", out_t],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "fmoe_fp8_blockscale_g1u1",
    category="moe",
    description="Fused FP8 block-scale MoE (g1u1 SwiGLU)",
    model=(MOE_EXT, "moe_aiter_fused_blockscale"),
    defaults={"M": 512, "N": 2048, "K": 7168, "E": 8, "topk": 2},
    trace_names=("aiter::fmoe_fp8_blockscale_g1u1",),
    csv_harness=True,
)
def _fmoe_fp8_blockscale_g1u1_event(args):
    """Event for ``aiter::fmoe_fp8_blockscale_g1u1``.

    Dtype kwargs read from ``args`` (all optional; defaults match
    :func:`tests.moe.test_fmoe_fp8_blockscale_g1u1`):

    * ``in_dtype``    -- activation dtype. ``"bf16"`` exercises the in-kernel
      quant path; ``"fp8"`` does caller-side ``pertoken_quant``. Default ``"bf16"``.
    * ``w_dtype``     -- expert weight dtype. Kernel only supports ``"fp8"``.
    * ``out_dtype``   -- output buffer dtype. Kernel only supports ``"bf16"``.
    * ``block_n`` / ``block_k`` -- block-scale tile sizes (default ``128``).
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::Float8_e4m3fnuz", "c10::BFloat16")
    )
    block_n = getattr(args, "block_n", None) or 128
    block_k = getattr(args, "block_k", None) or 128
    concrete = [None] * 15
    concrete[8] = topk
    concrete[13] = block_n
    concrete[14] = block_k
    event = {
        "name": "aiter::fmoe_fp8_blockscale_g1u1",
        "args": {
            "Input Dims": [[M, K], [M, K], [E, N * 2, K], [E, K, N]],
            "Input type": [out_t, in_t, w_t, w_t],
            "Concrete Inputs": concrete,
        },
    }
    return event


@register(
    "moe_cktile2stages_gemm1_ck",
    category="moe",
    description="CKTile MoE stage-1 GEMM (up projection)",
    model=(MOE_EXT, "moe_aiter_unfused_up"),
    defaults={"M": 512, "N": 2048, "K": 7168, "E": 8, "topk": 2},
    trace_names=("aiter::moe_cktile2stages_gemm1_ck",),
    csv_harness=True,
)
def _moe_cktile2stages_gemm1_ck_event(args):
    """Event for ``aiter::moe_cktile2stages_gemm1_ck`` (CK-Tile MoE up).

    Dtype kwargs (defaults reflect the BF16 / FP4-weight a16w4 path used by
    the unparameterized harness):

    * ``in_dtype``   -- activation dtype (default ``"bf16"``).
    * ``w_dtype``    -- packed FP4 weight dtype (default ``"fp4x2"``).
    * ``out_dtype``  -- output dtype (default ``"bf16"``).
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::Float4_e2m1fn_x2", "c10::BFloat16")
    )
    event = {
        "name": "aiter::moe_cktile2stages_gemm1_ck",
        "args": {
            "Input Dims": [[M, K], [E, N * 2, K], [M, topk, N]],
            "Input type": [in_t, w_t, out_t],
        },
    }
    return event


@register(
    "moe_cktile2stages_gemm2_ck",
    category="moe",
    description="CKTile MoE stage-2 GEMM (down projection)",
    model=(MOE_EXT, "moe_aiter_unfused_down"),
    defaults={"M": 512, "N": 2048, "K": 7168, "E": 8, "topk": 2},
    trace_names=("aiter::moe_cktile2stages_gemm2_ck",),
    csv_harness=True,
)
def _moe_cktile2stages_gemm2_ck_event(args):
    """Event for ``aiter::moe_cktile2stages_gemm2_ck`` (CK-Tile MoE down).

    Same dtype kwargs as ``moe_cktile2stages_gemm1_ck``.
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::Float4_e2m1fn_x2", "c10::BFloat16")
    )
    event = {
        "name": "aiter::moe_cktile2stages_gemm2_ck",
        "args": {
            "Input Dims": [[M, topk, N], [E, K, N], [M, K]],
            "Input type": [in_t, w_t, out_t],
        },
    }
    return event


@register(
    "ck_moe_stage1",
    category="moe",
    description="CK MoE stage-1 (up projection, fused activation)",
    model=(MOE_EXT, "moe_aiter_ck_stage1"),
    defaults={"M": 512, "N": 2048, "K": 7168, "E": 8, "topk": 2},
    trace_names=("aiter::ck_moe_stage1",),
    csv_harness=True,
)
def _ck_moe_stage1_event(args):
    """Event for ``aiter::ck_moe_stage1`` (CK MoE up + fused activation).

    Dtype kwargs:

    * ``in_dtype``   -- activation dtype. ``"bf16"``/``"fp16"`` (No quant) or
      ``"fp8"``/``"i8"``/``"fp4x2"`` (quantized A). Default ``"bf16"``.
    * ``w_dtype``    -- weight dtype (``"bf16"``/``"fp16"``/``"fp8"``/``"i8"``/
      ``"fp4x2"``). Default ``"bf16"``.
    * ``out_dtype``  -- output dtype (``"bf16"`` / ``"fp16"``). Default
      ``"bf16"``.
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::BFloat16", "c10::BFloat16")
    )
    event = {
        "name": "aiter::ck_moe_stage1",
        "args": {
            "Input Dims": [
                [M, K],
                [E, N * 2, K],
                [E, K, N],
                [M * topk],
                [E],
                [E],
                [M, topk, N],
            ],
            "Input type": [in_t, w_t, out_t],
        },
    }
    return event


@register(
    "ck_moe_stage2",
    category="moe",
    description="CK MoE stage-2 (down projection)",
    model=(MOE_EXT, "moe_aiter_ck_stage2"),
    defaults={"M": 512, "N": 2048, "K": 7168, "E": 8, "topk": 2},
    trace_names=("aiter::ck_moe_stage2",),
    csv_harness=True,
)
def _ck_moe_stage2_event(args):
    """Event for ``aiter::ck_moe_stage2`` (CK MoE down + topk reduce).

    Same dtype kwargs as ``ck_moe_stage1``.
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::BFloat16", "c10::BFloat16")
    )
    event = {
        "name": "aiter::ck_moe_stage2",
        "args": {
            "Input Dims": [
                [M, topk, N],
                [E, N * 2, K],
                [E, K, N],
                [M * topk],
                [E],
                [E],
                [M, K],
            ],
            "Input type": [in_t, w_t, out_t],
        },
    }
    return event


# Qwen3-30B-A3B gate/up shape: hidden=2048, 2*inter=1536, E=128, topk=8.
@register(
    "sglang_fused_moe_triton_invoke",
    category="moe",
    description="SGLang Triton fused-MoE grouped GEMM (invoke_fused_moe_kernel)",
    model=(MOE_EXT, "moe_triton_invoke_grouped_gemm"),
    defaults={"M": 15360, "N": 1536, "K": 2048, "E": 128, "topk": 8},
    trace_names=(
        "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel",
        "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427",
    ),
)
def _sglang_fused_moe_triton_invoke_event(args):
    """Event for ``sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel``.

    A single SGLang Triton ``invoke_fused_moe_kernel`` launch (one grouped GEMM).
    This same op name covers both the gate/up and the down projection; the
    perf-model class derives ``(M_work, N, K)`` generically from the args, so a
    single representative shape validates either direction.

    Args read:

    * ``M``     -- number of input tokens (pre-expansion).
    * ``N``     -- weight output dim (for gate/up pass this is ``2*inter_dim``).
    * ``K``     -- weight contraction dim (hidden_dim for gate/up).
    * ``E``     -- number of experts.
    * ``topk``  -- experts per token.
    * ``in_dtype``  -- activation dtype (default ``"bf16"``; quantized to fp8 in-kernel).
    * ``w_dtype``   -- expert weight dtype (default ``"fp8"``).
    * ``out_dtype`` -- output dtype (default ``"bf16"``).

    The event layout mirrors the trace ``Input Dims``:
        [0] A            = (M, K)
        [1] B            = (E, N, K)
        [2] C            = ()                (output, dims unrecorded)
        [3] C buffer     = (M*topk, N)
        [4] ()
        [5] B_scale      = (E,)
        [6] ()
        [7] topk_weights = (M, topk)
        [8] topk_ids     = (M, topk)
    """
    M, K, N, E, topk = (args.M, args.K, args.N, args.E, args.topk)
    in_t, w_t, out_t = _moe_dtypes_from_args(
        args, defaults=("c10::BFloat16", "c10::Float8_e4m3fnuz", "c10::BFloat16")
    )
    event = {
        "name": "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel",
        "args": {
            "Input Dims": [
                [M, K],
                [E, N, K],
                [],
                [M * topk, N],
                [],
                [E],
                [],
                [M, topk],
                [M, topk],
                [M * topk + E * 64],
                [E * 2],
                [1],
                [],
            ],
            "Input type": [
                in_t,
                w_t,
                "",
                out_t,
                "",
                "float",
                "",
                "float",
                "int",
                "int",
                "int",
                "int",
                "",
            ],
        },
    }
    return event


# ---------------------------------------------------------------------------
# Gated activations
# ---------------------------------------------------------------------------
for _fn_name, _description, _trace_names in _ACTIVATION_OPS:
    register(
        _fn_name,
        category="activation",
        description=_description,
        model=(PERF_MODEL_EXT, f"aiter_{_fn_name}"),
        defaults={"M": 2048, "N": 7168},
        trace_names=_trace_names,
    )(functools.partial(_activation_event, _fn_name))


# ---------------------------------------------------------------------------
# Norm and quant
# ---------------------------------------------------------------------------
@register(
    "rms_norm",
    category="norm",
    description="RMSNorm (CK, aiter::rms_norm)",
    model=(RMSNORM_EXT, "aiter_rms_norm"),
    defaults={"M": 2048, "N": 7168},
    trace_names=("aiter::rms_norm", "aiter::rmsnorm2d_fwd_ck"),
)
def _rms_norm_event(args):
    """Event for ``aiter::rms_norm`` (CK).

    Reads ``in_dtype`` (slot ``[0]``), ``w_dtype`` (slot ``[1]``). Output
    traffic is computed using ``in_dtype`` (kernel out matches input).
    Defaults match :func:`tests.rmsnorm.test_rms_norm`.
    """
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    event = {
        "name": "aiter::rms_norm",
        "args": {
            "Input Dims": [[M, N], [N], [], []],
            "Input type": [in_t, w_t, "Scalar", "Scalar"],
            "Input Strides": [[N, 1], [1], [], []],
        },
    }
    return event


@register(
    "rmsnorm",
    category="norm",
    description="RMSNorm (aiter::rmsnorm, out-first API)",
    model=(RMSNORM_EXT, "aiter_rmsnorm"),
    defaults={"M": 2048, "N": 7168},
    trace_names=("aiter::rmsnorm",),
)
def _rmsnorm_event(args):
    """Event for ``aiter::rmsnorm`` (out-first API).

    Note that the perf model class reads slot ``[1]`` for the input dtype
    (the API is ``rmsnorm(out, inp, weight, eps)`` so slot ``[0]`` is the
    out buffer). Both slots therefore receive ``in_dtype``; ``w_dtype`` goes
    to slot ``[2]``.
    """
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    out_t = _dtype_from_args(args, "out_dtype", in_t)
    event = {
        "name": "aiter::rmsnorm",
        "args": {
            "Input Dims": [[M, N], [M, N], [N], []],
            "Input type": [out_t, in_t, w_t, "Scalar"],
            "Input Strides": [[N, 1], [N, 1], [1], []],
        },
    }
    return event


@register(
    "add_rmsnorm",
    category="norm",
    description="Fused residual-add + RMSNorm",
    model=(RMSNORM_EXT, "aiter_rmsnorm2d_fwd_with_add_ck"),
    defaults={"M": 2048, "N": 7168},
    trace_names=("aiter::add_rmsnorm", "aiter::rmsnorm2d_fwd_with_add_ck"),
)
def _add_rmsnorm_event(args):
    """Event for ``aiter::add_rmsnorm`` (fused residual-add + RMSNorm).

    Slot ``[1]`` is the input (read by the perf-model class for byte-counting
    and compute precision). All BF16 traffic uses ``in_dtype``.
    """
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    event = {
        "name": "aiter::add_rmsnorm",
        "args": {
            "Input Dims": [[M, N], [M, N], [M, N], [M, N], [N], [], []],
            "Input type": [in_t, in_t, in_t, in_t, w_t, "Scalar", "Scalar"],
            "Input Strides": [[N, 1], [N, 1], [N, 1], [N, 1], [1], [], []],
        },
    }
    return event


@register(
    "rmsnorm_dynamicquant",
    category="norm",
    description="Fused RMSNorm + dynamic FP8 quantization",
    model=(RMSNORM_EXT, "aiter_rmsnorm2d_fwd_with_dynamicquant_ck"),
    defaults={"M": 2048, "N": 7168},
    trace_names=("aiter::rmsnorm2d_fwd_with_dynamicquant_ck",),
)
def _rmsnorm_dynamicquant_event(args):
    """Event for ``aiter::rmsnorm2d_fwd_with_dynamicquant_ck``.

    Reads slot ``[1]`` for the input dtype. FP8 output bytes (slot ``[0]``)
    and per-token FP32 scale bytes (slot ``[2]``) are hardcoded by the
    perf-model class, so changing ``out_dtype`` only affects the test runner
    side.
    """
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    out_t = _dtype_from_args(args, "out_dtype", "unsigned char")
    scl_t = _dtype_from_args(args, "scale_dtype", "float")
    event = {
        "name": "aiter::rmsnorm2d_fwd_with_dynamicquant_ck",
        "args": {
            "Input Dims": [[M, N], [M, N], [M, 1], [N], [], []],
            "Input type": [out_t, in_t, scl_t, w_t, "Scalar", "Scalar"],
            "Input Strides": [[N, 1], [N, 1], [1, 1], [1], [], []],
        },
    }
    return event


@register(
    "dynamic_per_token_scaled_quant",
    category="quant",
    description="Dynamic per-token FP8 quantization",
    model=(PERF_MODEL_EXT, "per_group_quant"),
    defaults={"M": 2048, "N": 7168},
    trace_names=("aiter::dynamic_per_token_scaled_quant",),
)
def _dynamic_per_token_scaled_quant_event(args):
    """Event for ``aiter::dynamic_per_token_scaled_quant``.

    Reads slot ``[0]`` for output dtype, slot ``[1]`` for input, slot ``[2]``
    for scales. Compute precision is taken from input.
    """
    N = args.N
    M = args.M
    out_t = _dtype_from_args(args, "out_dtype", "c10::Float8_e4m3fnuz")
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    scl_t = _dtype_from_args(args, "scale_dtype", "float")
    event = {
        "name": "aiter::dynamic_per_token_scaled_quant",
        "args": {
            "Input Dims": [[M, N], [M, N], [M, 1]],
            "Input type": [out_t, in_t, scl_t],
            "Input Strides": [[N, 1], [N, 1], [1, 1]],
        },
    }
    return event


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
@register(
    "_flash_attn_forward",
    category="attention",
    description="Flash Attention forward (CK)",
    model=(PERF_MODEL, "aiter__flash_attn_forward"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("aiter::_flash_attn_forward",),
)
def _flash_attn_forward_event(args):
    """Event for ``aiter::_flash_attn_forward`` (CK flash attention).

    Reads ``in_dtype`` for Q/K/V slots (kernel requires matching dtypes).

    Note
    ----
    The upstream perf-model class (``aiter__flash_attn_forward``) hardcodes
    ``bytes_per_element=2`` in its ``bytes()`` method, so changing
    ``in_dtype`` does not affect predicted bytes; it only flows through to
    the test runner. ``get_compute_precision()`` returns ``None`` for this
    class (it never populates ``param_details["dtype_A_B"]``).
    """
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    concrete = [""] * 12
    concrete[3] = "0.0"
    concrete[5] = "true"
    event = {
        "name": "aiter::_flash_attn_forward",
        "args": {
            "Input Dims": [[1, S, H_Q, d], [1, S, H_KV, d], [1, S, H_KV, d]],
            "Input type": [in_t, in_t, in_t],
            "Concrete Inputs": concrete,
        },
    }
    return event


@register(
    "wrapper_fmha_v3_fwd",
    category="attention",
    description="FMHA v3 forward",
    model=(PERF_MODEL, "aiter__fmha_v3_forward"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("aiter::wrapper_fmha_v3_fwd",),
    csv_harness=True,
)
def _wrapper_fmha_v3_fwd_event(args):
    """Event for ``aiter::wrapper_fmha_v3_fwd`` (FMHA v3 dense forward).

    Same dtype constraints + same hardcoded-bytes caveat as
    ``_flash_attn_forward``.
    """
    S = args.seq_len
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    concrete = [""] * 9
    concrete[4] = "0.0"
    concrete[6] = "true"
    event = {
        "name": "aiter::wrapper_fmha_v3_fwd",
        "args": {
            "Input Dims": [[], [1, S, H_Q, d], [1, S, H_KV, d], [1, S, H_KV, d]],
            "Input type": ["", in_t, in_t, in_t],
            "Concrete Inputs": concrete,
        },
    }
    return event


@register(
    "mha_varlen_fwd",
    category="attention",
    description="MHA variable-length forward (CK)",
    model=(ATTN_EXT, "mha_varlen_fwd"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("aiter::mha_varlen_fwd",),
    csv_harness=True,
)
def _mha_varlen_fwd_event(args):
    """Event for ``aiter::mha_varlen_fwd`` (CK varlen flash attention).

    Reads slot ``[0]`` for Q dtype. Bytes are computed as
    ``name2bpe(dtype_Q) or 2``, so passing ``--in_dtype fp16`` yields BF16
    -> FP16 byte counts (both 2 BPE) and equivalent compute precision.

    Annotation format
    -----------------
    Uses the canonical ``execute_*`` format shared by all attention harnesses::

        execute_0_context_N(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                            _generation_0(sq0sk0sqsq0sqsk0)

    where:

    * ``c_sq``  = total Q tokens across all context sequences
    * ``c_sk``  = total KV tokens across all context sequences (= ``total_kv``)
    * ``c_sqsq``= Σ sq_i²  (causal correction)
    * ``c_sqsk``= Σ sq_i·sk_i  (= c_sqsq for self-attention where sq_i = sk_i)

    The generation block is always zero because this is a prefill-only kernel.
    """
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    annotation, total_q, total_kv = _varlen_annotation(args)
    event = {
        "name": "aiter::mha_varlen_fwd",
        "annotation": annotation,
        "args": {
            "Input Dims": [[total_q, H_Q, d], [total_kv, H_KV, d]],
            "Input type": [in_t, in_t],
        },
    }
    return event


@register(
    "fmha_v3_varlen_fwd",
    category="attention",
    description="FMHA v3 variable-length forward",
    model=(ATTN_EXT, "aiter_fmha_v3_varlen_fwd"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("aiter::fmha_v3_varlen_fwd",),
    csv_harness=True,
)
def _fmha_v3_varlen_fwd_event(args):
    """Event for ``aiter::fmha_v3_varlen_fwd`` (FMHA v3 varlen forward).

    Same dtype + annotation semantics as ``mha_varlen_fwd``:
    uses the canonical ``execute_*`` format with a zero-generation block.
    """
    d = args.head_dim
    H_KV = args.num_heads_kv
    H_Q = args.num_heads_q
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    annotation, total_q, total_kv = _varlen_annotation(args)
    event = {
        "name": "aiter::fmha_v3_varlen_fwd",
        "annotation": annotation,
        "args": {
            "Input Dims": [[total_q, H_Q, d], [total_kv, H_KV, d], [total_kv, H_KV, d]],
            "Input type": [in_t, in_t, in_t],
        },
    }
    return event


# ---------------------------------------------------------------------------
# vLLM ops
# ---------------------------------------------------------------------------
@register(
    "vllm_unquantized_gemm",
    category="vllm_gemm",
    description="vLLM BF16 GEMM (rocm_unquantized_gemm)",
    model=(PERF_MODEL_EXT, "vllm_rocm_unquantized_gemm"),
    defaults={"M": 2048, "N": 4096, "K": 8192},
    trace_names=("vllm::rocm_unquantized_gemm",),
)
def _vllm_unquantized_gemm_event(args):
    """Event for ``vllm::rocm_unquantized_gemm`` (BF16 / FP16 GEMM).

    Dtype kwargs (defaults match :func:`tests.gemm.test_vllm_unquantized_gemm`):

    * ``in_dtype`` / ``w_dtype`` -- activation / weight (default ``"bf16"``).
    * ``bias_dtype`` -- bias dtype if a bias is supplied (default unset =
      no bias). Set this to enable the bias-traffic bytes term.
    """
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    bias_t = _dtype_from_args(args, "bias_dtype", "")
    event = {
        "name": "vllm::rocm_unquantized_gemm",
        "args": {
            "Input Dims": [[M, K], [N, K]],
            "Input type": [in_t, w_t, bias_t],
            "Input Strides": [[K, 1], [K, 1]],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "vllm_triton_gemm_a8w8_blockscale",
    category="vllm_gemm",
    description="vLLM FP8 block-scaled GEMM (wraps aiter gemm_a8w8_blockscale)",
    model=(PERF_MODEL_EXT, "gemm_a8w8_blockscale"),
    defaults={"M": 2048, "N": 4096, "K": 8192},
    trace_names=("vllm::rocm_aiter_triton_gemm_a8w8_blockscale",),
)
def _vllm_triton_gemm_a8w8_blockscale_event(args):
    """Event for the vLLM Triton wrapper around ``gemm_a8w8_blockscale``.

    Same dtype semantics as ``gemm_a8w8_blockscale``.
    """
    return _a8w8_blockscale_event("vllm::rocm_aiter_triton_gemm_a8w8_blockscale", args)


@register(
    "vllm_gemm_with_dynamic_quant",
    category="vllm_gemm",
    description="vLLM FP4 GEMM with dynamic quantization (Quark OCP MX)",
    model=(PERF_MODEL, "vllm_gemm_with_dynamic_quant"),
    defaults={"M": 2048, "N": 4096, "K": 8192},
    trace_names=("vllm::gemm_with_dynamic_quant",),
    csv_harness=True,
)
def _vllm_gemm_with_dynamic_quant_event(args):
    """Event for ``vllm::gemm_with_dynamic_quant`` (Quark OCP MX FP4 GEMM).

    Dtype kwargs:

    * ``in_dtype`` -- activation dtype (``"bf16"`` / ``"fp16"``). The perf
      model uses this for both the activation traffic AND the output traffic
      (the class only reads slot ``[0]``).
    * ``w_dtype`` -- weight dtype (``"fp4x2"``, default). Forwarded to the
      test runner; perf-model bytes for the weight is computed implicitly
      from ``in_dtype`` (limitation of the upstream class).
    * ``scale_dtype`` -- per-32 OCP MX scale dtype (default ``"u8"``).
    """
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "unsigned char")
    scl_t = _dtype_from_args(args, "scale_dtype", "unsigned char")
    event = {
        "name": "vllm::gemm_with_dynamic_quant",
        "args": {
            "Input Dims": [[M, K], [N, K // 2], [N, K // 32]],
            "Input type": [in_t, w_t, scl_t],
            "Input Strides": [[K, 1], [K // 2, 1], [K // 32, 1]],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "vllm_triton_group_quant_fp8",
    category="vllm_quant",
    description="vLLM Triton FP8 per-group quantization",
    model=(PERF_MODEL_EXT, "vllm_triton_per_token_group_quant_fp8"),
    defaults={"M": 2048, "N": 7168, "group_size": 128},
    required_args=["M", "N"],
    trace_names=("vllm::triton_per_token_group_quant_fp8",),
)
def _vllm_triton_group_quant_fp8_event(args):
    """Event for ``vllm::triton_per_token_group_quant_fp8``.

    Reads slot ``[0]`` for input dtype. FP8 output (1 BPE) and FP32 scales
    (4 BPE) are hardcoded by the upstream class.
    """
    N = args.N
    M = args.M
    group_size = getattr(args, "group_size", None) or 128
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    event = {
        "name": "vllm::triton_per_token_group_quant_fp8",
        "args": {
            "Input Dims": [(M, N), ()],
            "Input type": [in_t, "Scalar"],
            "Concrete Inputs": ["", str(group_size)],
        },
    }
    return event


@register(
    "vllm_rmsnorm_fp8_group_quant",
    category="vllm_norm",
    description="vLLM fused RMSNorm + FP8 group quantization",
    model=(RMSNORM_EXT, "vllm_rocm_aiter_rmsnorm_fp8_group_quant"),
    defaults={"M": 2048, "N": 7168, "group_size": 128},
    required_args=["M", "N"],
    trace_names=("vllm::rocm_aiter_rmsnorm_fp8_group_quant",),
)
def _vllm_rmsnorm_fp8_group_quant_event(args):
    """Event for ``vllm::rocm_aiter_rmsnorm_fp8_group_quant``.

    Reads slot ``[0]`` for x dtype (input). The FP8 quant output and FP32
    scales are hardcoded by the upstream class.
    """
    N = args.N
    M = args.M
    group_size = getattr(args, "group_size", None) or 128
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    event = {
        "name": "vllm::rocm_aiter_rmsnorm_fp8_group_quant",
        "args": {
            "Input Dims": [(M, N), (N,), (), ()],
            "Input type": [in_t, w_t, "Scalar", "Scalar"],
            "Input Strides": [(N, 1), (1,), (), ()],
            "Concrete Inputs": ["", "", "", str(group_size)],
        },
    }
    return event


@register(
    "vllm_rmsnorm_add_fp8_group_quant",
    category="vllm_norm",
    description="vLLM fused residual-add + RMSNorm + FP8 group quantization",
    model=(RMSNORM_EXT, "vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant"),
    defaults={"M": 2048, "N": 7168, "group_size": 128},
    required_args=["M", "N"],
    trace_names=("vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant",),
)
def _vllm_rmsnorm_add_fp8_group_quant_event(args):
    """Event for ``vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant``."""
    N = args.N
    M = args.M
    group_size = getattr(args, "group_size", None) or 128
    in_t = _dtype_from_args(args, "in_dtype", "c10::BFloat16")
    w_t = _dtype_from_args(args, "w_dtype", "c10::BFloat16")
    event = {
        "name": "vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant",
        "args": {
            "Input Dims": [(M, N), (M, N), (N,), (), ()],
            "Input type": [in_t, in_t, w_t, "Scalar", "Scalar"],
            "Input Strides": [(N, 1), (N, 1), (1,), (), ()],
            "Concrete Inputs": ["", "", "", "", str(group_size)],
        },
    }
    return event


@register(
    "vllm_unified_attention",
    category="vllm_attention",
    description="vLLM unified attention (paged decode via aiter.paged_attention_v1, FP8 KV cache)",
    model=(ATTN_EXT, "vllm_unified_attention_with_output"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("vllm::unified_attention_with_output",),
    csv_harness=True,
)
def _vllm_unified_attention_event(args):
    """Event for ``vllm::unified_attention_with_output`` (paged decode).

    Reads slot ``[0]`` for Q dtype (via ``InferenceAttention.get_param_details``).
    The KV-cache dtype is conveyed via slot ``[1]`` for documentation, even
    though the upstream class does not consume it -- bytes are
    ``name2bpe(dtype_Q) or 2``.

    Annotation flow
    ---------------
    All scenarios converge on a canonical vLLM-format annotation:

        execute_<iter>_context_<c_n>(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                      _generation_<g_n>(sq<g_sq>sk<g_sk>sqsq<g_sqsq>sqsk<g_sqsk>)

    ``vllm_unified_attention_with_output`` is a paged-decode-only kernel; any
    prefill in the same scheduler step runs through a separate flash-attention
    kernel.  Therefore c_* are always 0 here and all decode statistics live in
    the g_* fields:

        g_sq   = n_decode              (one Q token per decode sequence)
        g_sk   = n_decode * ctx_len    (total KV tokens accessed by the kernel)
        g_sqsq = g_sq  (same; no causal masking across sequences)
        g_sqsk = g_sk  (same)

    ``sum_q`` and ``sum_kv`` are then derived by parsing this annotation through
    ``_parse_unified_attention_annotation``, matching the identical path used for
    real vLLM traces.  This ensures the perf model's FLOPs and bytes calculations
    are consistent regardless of the annotation source.
    """
    return _build_unified_attention_event("vllm::unified_attention_with_output", args)


@register(
    "unified_attention",
    category="attention",
    description="aiter triton unified_attention (paged KV / varlen, accepts --annotation)",
    model=(ATTN_EXT, "vllm_unified_attention_with_output"),
    defaults={"seq_len": 2048, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128},
    trace_names=("aiter::unified_attention",),
    csv_harness=True,
)
def _unified_attention_event(args):
    """Event for aiter triton ``unified_attention`` (paged KV / varlen).

    Reuses ``vllm_unified_attention_with_output`` (same kernel under the hood).

    Annotation flow
    ---------------
    Identical to ``vllm_unified_attention``: all scenarios
    converge on the canonical ``execute_*`` format::

        execute_<iter>_context_<c_n>(sq<c_sq>sk<c_sk>sqsq<c_sqsq>sqsk<c_sqsk>)
                      _generation_<g_n>(sq<g_sq>sk<g_sk>sqsq<g_sqsq>sqsk<g_sqsk>)

    If ``args.annotation`` already carries a valid ``execute_*`` string it is
    passed through verbatim.  Otherwise a synthetic annotation is built from
    ``args.num_decode_seqs`` / ``args.ctx_len`` / ``args.prefill_seq_len``
    (falling back to ``args.seq_len`` as a single-sequence decode scenario).

    Dtype kwargs ``in_dtype`` / ``kv_dtype`` mirror
    ``vllm_unified_attention``.
    """
    return _build_unified_attention_event("aiter::unified_attention", args)


@register(
    "vllm_gdn_attention_core",
    category="vllm_attention",
    description="vLLM Gated Delta Network attention (perf-model-only)",
    model=(ATTN_EXT, "gdn_attention_core"),
    defaults={"seq_len": 2048, "num_heads_kv": 16, "head_dim": 128},
    trace_names=("vllm::gdn_attention_core",),
    csv_harness=True,
    perf_model_only=True,
)
def _vllm_gdn_attention_core_event(args):
    """Event for ``vllm::gdn_attention_core`` (Gated Delta Network).

    Reads slot ``[0]`` for compute precision (mixed_qkv dtype). The ``bytes()``
    method hardcodes 2 BPE so changing dtype only affects the test runner
    side.
    """
    return _gdn_event(
        "vllm::gdn_attention_core",
        args,
        H_K=args.num_heads_kv // 2,
        annotation=f"attn_0_0_{args.seq_len}_{args.seq_len * args.seq_len}_0_0_0",
    )


# ---------------------------------------------------------------------------
# DSV3 dominant pseudo ops (perf models in TraceLens core op_to_perf_model_class_map)
# ---------------------------------------------------------------------------
@register(
    "dsv3_mla_decode_fwd",
    category="dsv3_mla_decode",
    description="AITER paged MLA attention (pseudo_mla_decode_fwd: core ASM + reduce)",
    model=(CORE_OP_MAP, "pseudo_mla_decode_fwd"),
    defaults={
        "seq_len": 8677,
        "E": 64,
        "num_heads_q": 16,
        "head_dim": 576,
        "kv_dtype": "fp8",
        "n_ctx": 0,
        "ctx_qlen": 0,
    },
    required_args=["seq_len", "E", "num_heads_q", "head_dim"],
    trace_names=("pseudo_mla_decode_fwd",),
    csv_harness=True,
)
def _dsv3_mla_decode_fwd_event(args):
    """Event for ``pseudo_mla_decode_fwd`` (FP8 paged MLA decode).

    Resolved from the TraceLens core map (InferenceAttention subclass). Builds
    the chunk-stats annotation the model expects from the decode geometry:
    ``E`` decode sequences (q_len==1), each attending ``seq_len`` KV tokens.
    """
    batch = getattr(args, "E", None) or 64
    ctx = args.seq_len
    nhead = getattr(args, "num_heads_q", None) or 16
    qk = getattr(args, "head_dim", None) or 576
    # Decode: sq_i == 1 per sequence -> g_sq = batch, g_sqsq = batch,
    # g_sk = g_sqsk = batch * ctx.
    g_sq = batch
    g_sk = batch * ctx
    g_sqsq = batch
    g_sqsk = batch * ctx
    ann = (
        f"execute_0_context_0(sq0sk0sqsq0sqsk0)"
        f"_generation_{batch}(sq{g_sq}sk{g_sk}sqsq{g_sqsq}sqsk{g_sqsk})"
    )
    event = {
        "name": "pseudo_mla_decode_fwd",
        "args": {
            "Input Dims": [[batch, nhead, qk], [batch, 1, qk]],
            "Input type": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
        },
        "annotation": ann,
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "dsv3_moe_flydsl_stage1",
    category="dsv3_moe_flydsl",
    description="AITER FlyDSL FP4 MoE stage-1 GEMM (moe_gemm1_0)",
    model=(CORE_OP_MAP, "pseudo_op::moe_flydsl_stage1"),
    defaults={"M": 64, "K": 7168, "N": 256, "E": 257, "topk": 9, "block_m": 32},
    required_args=["M", "K", "N", "E", "topk"],
    trace_names=("pseudo_op::moe_flydsl_stage1",),
    csv_harness=True,
)
def _dsv3_moe_flydsl_stage1_event(args):
    """Event for ``pseudo_op::moe_flydsl_stage1`` (FP4 MoE gate+up GEMM)."""
    M = args.M
    K = getattr(args, "K", None) or 7168
    N = args.N
    E = getattr(args, "E", None) or 257
    topk = getattr(args, "topk", None) or 9
    event = _dsv3_flydsl_moe_event("pseudo_op::moe_flydsl_stage1", M, K, N, E, topk)
    return event


@register(
    "dsv3_moe_flydsl_stage2",
    category="dsv3_moe_flydsl",
    description="AITER FlyDSL FP4 MoE stage-2 GEMM (moe_gemm2)",
    model=(CORE_OP_MAP, "pseudo_op::moe_flydsl_stage2"),
    defaults={"M": 64, "K": 7168, "N": 256, "E": 257, "topk": 9, "block_m": 32},
    required_args=["M", "K", "N", "E", "topk"],
    trace_names=("pseudo_op::moe_flydsl_stage2",),
    csv_harness=True,
)
def _dsv3_moe_flydsl_stage2_event(args):
    """Event for ``pseudo_op::moe_flydsl_stage2`` (FP4 MoE down GEMM)."""
    M = args.M
    K = getattr(args, "K", None) or 7168
    N = args.N
    E = getattr(args, "E", None) or 257
    topk = getattr(args, "topk", None) or 9
    event = _dsv3_flydsl_moe_event("pseudo_op::moe_flydsl_stage2", M, K, N, E, topk)
    return event


# ---------------------------------------------------------------------------
# MXFP4 ops
# ---------------------------------------------------------------------------
@register(
    "gemm_afp4wfp4",
    category="ext_mxfp4",
    description="AITER MXFP4 GEMM (gemm_afp4wfp4_)",
    model=(PERF_MODEL_EXT, "gemm_afp4wfp4"),
    defaults={"M": 822, "N": 2112, "K": 3584},
    trace_names=("aiter::gemm_afp4wfp4_",),
)
def _gemm_afp4wfp4_event(args):
    """Event for ``aiter::gemm_afp4wfp4_`` (MXFP4 GEMM).

    Inputs are FP4 (packed as ``unsigned char`` in the trace). Output is BF16.
    """
    K = args.K
    N = args.N
    M = args.M
    SCALE_GROUP_SIZE = 32
    event = {
        "name": "aiter::gemm_afp4wfp4_",
        "args": {
            "Input Dims": [
                [M, K // 2],
                [N, K // 2],
                [M, K // SCALE_GROUP_SIZE],
                [N, K // SCALE_GROUP_SIZE],
                [],
                [M, N],
                [],
                [],
            ],
            "Input type": [
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "Scalar",
                "c10::BFloat16",
                "",
                "Scalar",
            ],
            "Input Strides": [
                [K // 2, 1],
                [K // 2, 1],
                [K // SCALE_GROUP_SIZE, 1],
                [K // SCALE_GROUP_SIZE, 1],
                [],
                [N, 1],
                [],
                [],
            ],
            "Concrete Inputs": ["", "", "", "", "15", "", "", "False"],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "rope_cached_positions_2c_fwd_impl",
    category="ext_mxfp4",
    description="AITER 2-channel cached-positions forward RoPE",
    model=(PERF_MODEL_EXT, "aiter_rope_cached_positions_2c_fwd_impl"),
    defaults={"M": 822, "num_heads_q": 16, "num_heads_kv": 1, "head_dim": 64},
    required_args=["M"],
    trace_names=("aiter::rope_cached_positions_2c_fwd_impl",),
)
def _rope_cached_positions_2c_fwd_impl_event(args):
    """Event for ``aiter::rope_cached_positions_2c_fwd_impl``.

    Inputs:
      Input Dims[2] = input_x  (s, b, H_q,  d) BF16
      Input Dims[3] = input_y  (s, b, H_kv, d) BF16
    """
    M = args.M
    H_q = getattr(args, "num_heads_q", None) or 16
    H_kv = getattr(args, "num_heads_kv", None) or 1
    d = getattr(args, "head_dim", None) or 64
    d_cs = d // 2
    max_pos = max(M * 16, 4096)
    event = {
        "name": "aiter::rope_cached_positions_2c_fwd_impl",
        "args": {
            "Input Dims": [
                [1, M, H_q, d],
                [1, M, H_kv, d],
                [1, M, H_q, d],
                [1, M, H_kv, d],
                [max_pos, 1, 1, d_cs],
                [max_pos, 1, 1, d_cs],
                [1, M],
                [],
                [],
                [],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "long int",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
        },
    }
    return event


@register(
    "fused_flatten_mxfp4_quant",
    category="ext_mxfp4",
    description="AITER triton fused flatten + MXFP4 quant",
    model=(PERF_MODEL_EXT, "fused_flatten_mxfp4_quant"),
    defaults={"M": 822, "N": 7168, "group_size": 128},
    required_args=["M", "N"],
    trace_names=("sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant",),
)
def _fused_flatten_mxfp4_quant_event(args):
    """Event for ``sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant``."""
    M = args.M
    N = args.N
    group_size = getattr(args, "group_size", None) or 128
    N2 = group_size
    N1 = N // group_size
    event = {
        "name": "sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant",
        "args": {
            "Input Dims": [[M, N1, N2]],
            "Input type": ["c10::BFloat16"],
            "Input Strides": [[N1 * N2, N2, 1]],
        },
    }
    return event


@register(
    "fused_rms_mxfp4_quant",
    category="ext_mxfp4",
    description="AITER triton fused RMSNorm + MXFP4 quant",
    model=(RMSNORM_EXT, "fused_rms_mxfp4_quant"),
    defaults={"M": 822, "N": 7168},
    trace_names=("sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant",),
)
def _fused_rms_mxfp4_quant_event(args):
    """Event for ``sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant``.

    Single-input (no x2, no res1) variant, matching the simplest call shape:
        x1:        (M, N1) BF16
        x1_weight: (N1,)   BF16
    """
    N = args.N
    M = args.M
    event = {
        "name": "sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant",
        "args": {
            "Input Dims": [[M, N], [N], []],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
            "Input Strides": [[N, 1], [1], []],
        },
    }
    return event


# ---------------------------------------------------------------------------
# ATOM FlyDSL ops (tests/ has no test_<key> for these yet, so only the perf
# model is exercised)
# ---------------------------------------------------------------------------
@register(
    "atom_flydsl_preshuffle_gemm_a8",
    category="atom_flydsl",
    description="AITER FlyDSL A8W8 preshuffle GEMM (flydsl_preshuffle_gemm_a8)",
    model=(PERF_MODEL_EXT, "gemm_a8w8_blockscale"),
    defaults={"M": 512, "N": 4096, "K": 7168},
    trace_names=(
        "aiter::gemm_a8w8_bpreshuffle",
        "vllm::_rocm_aiter_preshuffled_per_token_w8a8_gemm",
    ),
    csv_harness=True,
)
def _atom_flydsl_preshuffle_gemm_a8_event(args):
    """Event for ``aiter::gemm_a8w8_bpreshuffle`` (FlyDSL A8W8 GEMM).

    Maps to ``gemm_a8w8_blockscale``: FLOPs ``2*M*N*K``, bytes A/B = 1 byte
    (FP8), output = 2 bytes (BF16). The bpreshuffle layout is layout-only, so
    flops/bytes match the block-scale model.
    """
    K = args.K
    N = args.N
    M = args.M
    in_t = _dtype_from_args(args, "in_dtype", "c10::Float8_e4m3fn")
    w_t = _dtype_from_args(args, "w_dtype", "c10::Float8_e4m3fn")
    scl_t = _dtype_from_args(args, "scale_dtype", "float")
    event = {
        "name": "aiter::gemm_a8w8_bpreshuffle",
        "args": {
            "Input Dims": [[M, K], [N, K], [M, 1], [N, 1]],
            "Input type": [in_t, w_t, scl_t, scl_t],
            "Input Strides": [[K, 1], [K, 1], [1, 1], [1, 1]],
        },
        "kernel_names": ["placeholder_kernel"],
        "kernel_details": [{"name": "placeholder_kernel"}],
    }
    return event


@register(
    "atom_flydsl_gdr_decode",
    category="atom_flydsl",
    description="AITER FlyDSL gated-delta-rule decode (flydsl_gdr_decode)",
    model=(ATTN_EXT, "gdn_attention_core"),
    defaults={"seq_len": 128, "num_heads_kv": 8, "head_dim": 128},
    required_args=["seq_len"],
    trace_names=("aiter::linear_attention_with_output_base",),
    csv_harness=True,
)
def _atom_flydsl_gdr_decode_event(args):
    """Event for ``aiter::linear_attention_with_output_base`` (FlyDSL GDR).

    Maps to ``gdn_attention_core`` -- the op signature
    ``(mixed_qkv, b, a, core_attn_out, layer_name)`` matches the layout the
    model parses. ``--seq_len`` is the GDN total-token count (decode: one token
    per sequence), injected via the GDN ``c_sq`` annotation field.
    """
    return _gdn_event(
        "aiter::linear_attention_with_output_base",
        args,
        H_K=max(1, args.num_heads_kv // 2),
        annotation=f"attn_0_0_{args.seq_len}_0_0_0_0",
    )


# ---------------------------------------------------------------------------
# Ops validated only from --cases (see _CSV_ONLY_OPS)
# ---------------------------------------------------------------------------
for _key, _category, _description, _trace_name in _CSV_ONLY_OPS:
    register(
        _key,
        category=_category,
        description=_description,
        trace_names=(_trace_name,),
    )(None)
