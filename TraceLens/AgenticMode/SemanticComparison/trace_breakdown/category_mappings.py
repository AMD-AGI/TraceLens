###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared category mappings and shape derivation formulas for semantic trace breakdown.

Every semantic_block maps to a perf model category (GEMM, SDPA, Normalization,
or Elementwise) that has a roofline formula. There are no "Other" or "Memory"
categories -- the purpose of semantic labeling is to enable roofline analysis
on graph-mode traces, so every label must be computable.

Defines:
  - SEMANTIC_BLOCK_TO_GROUP: low-level block -> high-level functional group
  - SEMANTIC_BLOCK_TO_PERF_CATEGORY: low-level block -> TraceLens perf model type
  - derive_block_shapes(): compute theoretical FLOPS/bytes per block from HF config
"""

# ---------------------------------------------------------------------------
# High-level functional groups
# ---------------------------------------------------------------------------

SEMANTIC_BLOCK_TO_GROUP = {
    # --- Preamble (optional -- may not be in graph mode traces) ---
    "Preamble: Embedding": "Preamble",
    "Preamble: Input Norm": "Preamble",
    # --- Self-Attention ---
    "Pre-Attn Norm": "Self-Attention",
    "QKV Projection": "Self-Attention",
    "Q Projection": "Self-Attention",
    "KV Projection": "Self-Attention",
    "Rotary Embedding": "Self-Attention",
    "Attention": "Self-Attention",
    "KV Cache Store": "Self-Attention",
    "Output Projection": "Self-Attention",
    "Attention Output Gate": "Self-Attention",
    "Post-Attn Residual Add": "Self-Attention",
    # --- MoE FFN (models with mixture of experts) ---
    "Post-Attn Norm": "MoE / FFN",
    "Router Gate": "MoE / FFN",
    "MoE Routing": "MoE / FFN",
    "MoE GateUp+SwiGLU": "MoE / FFN",
    "MoE Quantize": "MoE / FFN",
    "MoE Down Projection": "MoE / FFN",
    "MoE Finalize": "MoE / FFN",
    "Shared Expert GateUp": "MoE / FFN",
    "Shared Expert Down": "MoE / FFN",
    "Post-MoE Residual Add": "MoE / FFN",
    # --- Dense FFN (models without MoE) ---
    "FFN Norm": "Dense FFN",
    "GateUp Projection": "Dense FFN",
    "Gate Projection": "Dense FFN",
    "Up Projection": "Dense FFN",
    "Activation": "Dense FFN",
    "Down Projection": "Dense FFN",
    "Post-FFN Residual Add": "Dense FFN",
    # --- Epilogue (optional -- may not be in graph mode traces) ---
    "Epilogue: Final Norm": "Epilogue",
    "Epilogue: LM Head": "Epilogue",
}

# ---------------------------------------------------------------------------
# Perf-model categories (parallel to TraceLens PerfModel classes)
#
# EVERY block maps to one of: GEMM, SDPA, Normalization, Elementwise.
# No "Other" or "Memory" -- everything has a roofline formula.
# ---------------------------------------------------------------------------

SEMANTIC_BLOCK_TO_PERF_CATEGORY = {
    # GEMM -- all projection / matmul blocks
    "QKV Projection": "GEMM",
    "Q Projection": "GEMM",
    "KV Projection": "GEMM",
    "Output Projection": "GEMM",
    "Router Gate": "GEMM",
    "MoE GateUp+SwiGLU": "GEMM",
    "MoE Down Projection": "GEMM",
    "Shared Expert GateUp": "GEMM",
    "Shared Expert Down": "GEMM",
    "GateUp Projection": "GEMM",
    "Gate Projection": "GEMM",
    "Up Projection": "GEMM",
    "Down Projection": "GEMM",
    "Epilogue: LM Head": "GEMM",
    # SDPA -- attention kernels (includes any splitK reduce as part of SDPA)
    "Attention": "SDPA",
    # Normalization -- all norm variants
    "Pre-Attn Norm": "Normalization",
    "Post-Attn Norm": "Normalization",
    "FFN Norm": "Normalization",
    "Epilogue: Final Norm": "Normalization",
    "Preamble: Input Norm": "Normalization",
    # Elementwise -- residual adds, activations, data movement, overhead
    "Post-Attn Residual Add": "Elementwise",
    "Post-FFN Residual Add": "Elementwise",
    "Post-MoE Residual Add": "Elementwise",
    "Activation": "Elementwise",
    "Attention Output Gate": "Elementwise",
    "Rotary Embedding": "Elementwise",
    "KV Cache Store": "Elementwise",
    "Preamble: Embedding": "Elementwise",
    "MoE Routing": "Elementwise",
    "MoE Quantize": "Elementwise",
    "MoE Finalize": "Elementwise",
}

DEFAULT_GROUP = "Unknown"
DEFAULT_PERF_CATEGORY = "Elementwise"

# ---------------------------------------------------------------------------
# Timeline categories (aligned with TraceLens rocprof_analysis categories)
# Maps internal perf_category names to the report-facing category names.
# ---------------------------------------------------------------------------

PERF_CATEGORY_TO_TIMELINE_CATEGORY = {
    "GEMM": "GEMM",
    "SDPA": "Attention",
    "Normalization": "Normalization",
    "Elementwise": "Elementwise",
}

DEFAULT_TIMELINE_CATEGORY = "Other"

# ---------------------------------------------------------------------------
# Standalone-compatible op categories
# Maps perf_category to the `op category` values that
# orchestrator_prepare.py's get_enhanced_category() expects.
# ---------------------------------------------------------------------------

PERF_CATEGORY_TO_OP_CATEGORY = {
    "GEMM": "GEMM",
    "SDPA": "SDPA_fwd",
    "Normalization": "Norm",
    "Elementwise": "Elementwise",
}

DEFAULT_OP_CATEGORY = "Elementwise"


def get_timeline_category(semantic_block):
    pc = get_perf_category(semantic_block)
    return PERF_CATEGORY_TO_TIMELINE_CATEGORY.get(pc, DEFAULT_TIMELINE_CATEGORY)


def get_op_category(semantic_block):
    """Return the standalone-compatible op category for a semantic block."""
    pc = get_perf_category(semantic_block)
    return PERF_CATEGORY_TO_OP_CATEGORY.get(pc, DEFAULT_OP_CATEGORY)


def format_input_dims(perf_params, perf_category):
    """Format perf_params as an Input Dims string matching TraceLens format.

    GEMM:  ((M, K), (K, N))
    SDPA:  ((B, H_Q, N_Q, d), (B, H_KV, N_KV, d))
    Norm/Elementwise:  ((num_elems,),)
    """
    if not perf_params:
        return ""
    if perf_category == "GEMM":
        M = perf_params.get("M", 0)
        N = perf_params.get("N", 0)
        K = perf_params.get("K", 0)
        return f"(({M}, {K}), ({K}, {N}))"
    if perf_category == "SDPA":
        B = perf_params.get("B", 1)
        H_Q = perf_params.get("H_Q", 0)
        N_Q = perf_params.get("N_Q", 0)
        H_KV = perf_params.get("H_KV", 0)
        N_KV = perf_params.get("N_KV", 0)
        d = perf_params.get("d_h_qk", perf_params.get("d_h_v", 0))
        return f"(({B}, {H_Q}, {N_Q}, {d}), ({B}, {H_KV}, {N_KV}, {d}))"
    ne = perf_params.get("num_elems", perf_params.get("num_channels", 0))
    return f"(({ne},),)"


DTYPE_TO_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "int4": 0.5,
    "fp4": 0.5,
}


def get_group(semantic_block):
    for prefix, group in SEMANTIC_BLOCK_TO_GROUP.items():
        if semantic_block == prefix or semantic_block.startswith(prefix):
            return group
    return DEFAULT_GROUP


def get_perf_category(semantic_block):
    # Exact match first to avoid "Attention" matching "Attention Output Gate"
    if semantic_block in SEMANTIC_BLOCK_TO_PERF_CATEGORY:
        return SEMANTIC_BLOCK_TO_PERF_CATEGORY[semantic_block]
    for prefix, cat in SEMANTIC_BLOCK_TO_PERF_CATEGORY.items():
        if semantic_block.startswith(prefix):
            return cat
    return DEFAULT_PERF_CATEGORY


# ---------------------------------------------------------------------------
# Shape derivation from HuggingFace config.json
# ---------------------------------------------------------------------------


def parse_model_config(config):
    """Extract relevant dimensions from a HuggingFace config dict.

    Supports both flat configs and nested text_config (multimodal models).
    """
    tc = config.get("text_config", config)
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc.get("num_key_value_heads", tc["num_attention_heads"]),
        "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        "num_hidden_layers": tc["num_hidden_layers"],
        "vocab_size": tc["vocab_size"],
        "dtype": tc.get("dtype", tc.get("torch_dtype", "bfloat16")),
        # MoE fields (optional, with common aliases)
        "num_experts": tc.get("num_experts", tc.get("num_local_experts", 0)),
        "num_experts_per_tok": tc.get(
            "num_experts_per_tok", tc.get("experts_per_token", 0)
        ),
        "moe_intermediate_size": tc.get("moe_intermediate_size", 0),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
        # Dense FFN (non-MoE)
        "intermediate_size": tc.get("intermediate_size", 0),
        # Alternating attention types
        "layer_types": tc.get("layer_types", []),
        # Linear attention specific dims
        "linear_key_head_dim": tc.get("linear_key_head_dim", 0),
        "linear_num_key_heads": tc.get("linear_num_key_heads", 0),
        "linear_num_value_heads": tc.get("linear_num_value_heads", 0),
        "linear_value_head_dim": tc.get("linear_value_head_dim", 0),
    }
    cfg["bpe"] = DTYPE_TO_BYTES.get(cfg["dtype"], 2)
    # If model is MoE but moe_intermediate_size is not set, use intermediate_size
    if cfg["num_experts"] > 0 and cfg["moe_intermediate_size"] == 0:
        cfg["moe_intermediate_size"] = cfg["intermediate_size"]
    return cfg


# ---------------------------------------------------------------------------
# FLOPS / bytes formulas (mirroring TraceLens PerfModel)
# ---------------------------------------------------------------------------


def _gemm_flops(M, N, K, bias=False):
    """2*M*N*K  (+M*N if bias)"""
    return 2 * M * N * K + (M * N if bias else 0)


def _gemm_bytes(M, N, K, bpe_A, bpe_B, bpe_out, bias=False, bpe_bias=None):
    b = M * K * bpe_A + K * N * bpe_B + M * N * bpe_out
    if bias and bpe_bias:
        b += N * bpe_bias
    return b


def _sdpa_flops(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, causal):
    flops_qk = B * H_Q * (2 * N_Q * N_KV * d_qk)
    flops_pv = B * H_Q * (2 * N_Q * d_v * N_KV)
    total = flops_qk + flops_pv
    if causal and N_Q == N_KV:
        total /= 2
    return total


def _sdpa_bytes(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, bpe):
    q = B * N_Q * H_Q * d_qk
    k = B * N_KV * H_KV * d_qk
    v = B * N_KV * H_KV * d_v
    o = B * N_Q * H_Q * d_v
    return (q + k + v + o) * bpe


def _rmsnorm_flops(num_elems, num_channels, affine=True):
    groups = num_elems // num_channels
    f = groups * (2 * num_channels + 2)
    f += num_elems * (2 if affine else 1)
    return f


def _rmsnorm_bytes(num_elems, num_channels, bpe, affine=True):
    b = num_elems * bpe * 2  # read + write
    if affine:
        b += num_channels * bpe  # weight
    return b


def _elementwise_flops(num_elems):
    return num_elems


def _unary_elementwise_bytes(num_elems, bpe_in, bpe_out=None):
    """Unary elementwise: read input + write output (2 transfers)."""
    if bpe_out is None:
        bpe_out = bpe_in
    return num_elems * bpe_in + num_elems * bpe_out


def _binary_elementwise_bytes(num_elems, bpe):
    """Binary elementwise with uniform dtype: read A + read B + write C (3 transfers)."""
    return num_elems * bpe * 3


def _elementwise_bytes(num_elems, bpe, *, unary=False):
    """Convenience wrapper: 2 transfers for unary, 3 for binary."""
    if unary:
        return _unary_elementwise_bytes(num_elems, bpe)
    return _binary_elementwise_bytes(num_elems, bpe)


# ---------------------------------------------------------------------------
# Per-block shape derivation
# ---------------------------------------------------------------------------


def derive_block_shapes(
    semantic_block, model_cfg, num_tokens, context_length=None, layer_idx=None
):
    """Compute theoretical FLOPS and bytes for a semantic block.

    Args:
        semantic_block: low-level block name
        model_cfg: dict from parse_model_config()
        num_tokens: number of active tokens (T)
        context_length: KV cache length for decode SDPA (defaults to num_tokens)
        layer_idx: layer index (used for alternating attention types)

    Returns:
        dict with keys: flops, bytes, perf_params (M/N/K etc.), or None if
        no formula is available for this block.
    """
    T = num_tokens
    bpe = model_cfg["bpe"]
    h = model_cfg["hidden_size"]
    n_heads = model_cfg["num_attention_heads"]
    n_kv = model_cfg["num_key_value_heads"]
    d = model_cfg["head_dim"]
    ctx = context_length if context_length is not None else T

    is_linear_attn = False
    if layer_idx is not None and model_cfg["layer_types"]:
        lt = model_cfg["layer_types"]
        if layer_idx < len(lt) and lt[layer_idx] == "linear_attention":
            is_linear_attn = True

    if semantic_block == "QKV Projection":
        if is_linear_attn:
            n_q = n_heads * d
            n_k = model_cfg["linear_num_key_heads"] * model_cfg["linear_key_head_dim"]
            n_v = (
                model_cfg["linear_num_value_heads"] * model_cfg["linear_value_head_dim"]
            )
            N = n_q + n_k + n_v
        else:
            N = n_heads * d + 2 * n_kv * d
        return {
            "flops": _gemm_flops(T, N, h),
            "bytes": _gemm_bytes(T, N, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": N, "K": h},
        }

    if semantic_block == "Output Projection":
        K_dim = n_heads * d
        return {
            "flops": _gemm_flops(T, h, K_dim),
            "bytes": _gemm_bytes(T, h, K_dim, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": h, "K": K_dim},
        }

    if semantic_block == "Router Gate":
        ne = model_cfg["num_experts"]
        return {
            "flops": _gemm_flops(T, ne, h),
            "bytes": _gemm_bytes(T, ne, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": ne, "K": h},
        }

    if semantic_block == "MoE GateUp+SwiGLU":
        topk = model_cfg["num_experts_per_tok"]
        mi = model_cfg["moe_intermediate_size"]
        M_eff = T * topk
        N = 2 * mi  # gate + up fused
        return {
            "flops": _gemm_flops(M_eff, N, h),
            "bytes": _gemm_bytes(M_eff, N, h, bpe, bpe, bpe),
            "perf_params": {"M": M_eff, "N": N, "K": h},
        }

    if semantic_block == "MoE Down Projection":
        topk = model_cfg["num_experts_per_tok"]
        mi = model_cfg["moe_intermediate_size"]
        M_eff = T * topk
        return {
            "flops": _gemm_flops(M_eff, h, mi),
            "bytes": _gemm_bytes(M_eff, h, mi, bpe, bpe, bpe),
            "perf_params": {"M": M_eff, "N": h, "K": mi},
        }

    if semantic_block == "Epilogue: LM Head":
        vs = model_cfg["vocab_size"]
        M_lm = 1 if T <= 1 else T
        return {
            "flops": _gemm_flops(M_lm, vs, h),
            "bytes": _gemm_bytes(M_lm, vs, h, bpe, bpe, bpe),
            "perf_params": {"M": M_lm, "N": vs, "K": h},
        }

    # --- Dense FFN blocks ---

    if semantic_block == "GateUp Projection":
        inter = model_cfg["intermediate_size"]
        if inter == 0:
            return None
        N = 2 * inter  # gate + up fused
        return {
            "flops": _gemm_flops(T, N, h),
            "bytes": _gemm_bytes(T, N, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": N, "K": h},
        }

    if semantic_block == "Gate Projection":
        inter = model_cfg["intermediate_size"]
        if inter == 0:
            return None
        return {
            "flops": _gemm_flops(T, inter, h),
            "bytes": _gemm_bytes(T, inter, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": inter, "K": h},
        }

    if semantic_block == "Up Projection":
        inter = model_cfg["intermediate_size"]
        if inter == 0:
            return None
        return {
            "flops": _gemm_flops(T, inter, h),
            "bytes": _gemm_bytes(T, inter, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": inter, "K": h},
        }

    if semantic_block == "Down Projection":
        inter = model_cfg["intermediate_size"]
        if inter == 0:
            return None
        return {
            "flops": _gemm_flops(T, h, inter),
            "bytes": _gemm_bytes(T, h, inter, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": h, "K": inter},
        }

    # --- Shared Expert (MoE models with a shared expert alongside routed experts) ---

    if semantic_block == "Shared Expert GateUp":
        si = model_cfg["shared_expert_intermediate_size"]
        if si == 0:
            return None
        N = 2 * si
        return {
            "flops": _gemm_flops(T, N, h),
            "bytes": _gemm_bytes(T, N, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": N, "K": h},
        }

    if semantic_block == "Shared Expert Down":
        si = model_cfg["shared_expert_intermediate_size"]
        if si == 0:
            return None
        return {
            "flops": _gemm_flops(T, h, si),
            "bytes": _gemm_bytes(T, h, si, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": h, "K": si},
        }

    # --- Separate Q / KV projections ---

    if semantic_block == "Q Projection":
        N = n_heads * d
        return {
            "flops": _gemm_flops(T, N, h),
            "bytes": _gemm_bytes(T, N, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": N, "K": h},
        }

    if semantic_block == "KV Projection":
        N = 2 * n_kv * d
        return {
            "flops": _gemm_flops(T, N, h),
            "bytes": _gemm_bytes(T, N, h, bpe, bpe, bpe),
            "perf_params": {"M": T, "N": N, "K": h},
        }

    # --- SDPA (includes any splitK reduce as part of the attention operation) ---

    if semantic_block == "Attention":
        causal = T == ctx
        flops = _sdpa_flops(1, T, n_heads, ctx, n_kv, d, d, causal)
        bts = _sdpa_bytes(1, T, n_heads, ctx, n_kv, d, d, bpe)
        return {
            "flops": flops,
            "bytes": bts,
            "perf_params": {
                "B": 1,
                "N_Q": T,
                "H_Q": n_heads,
                "N_KV": ctx,
                "H_KV": n_kv,
                "d_h_qk": d,
                "d_h_v": d,
                "causal": causal,
            },
        }

    # --- Rotary Embedding (elementwise on Q/K head dims) ---

    if semantic_block == "Rotary Embedding":
        ne = T * n_heads * d
        return {
            "flops": _elementwise_flops(ne),
            "bytes": _elementwise_bytes(ne, bpe, unary=True),
            "perf_params": {"num_elems": ne},
        }

    # --- KV Cache Store (write KV to cache: 2 * T * n_kv * d) ---

    if semantic_block == "KV Cache Store":
        ne = 2 * T * n_kv * d  # K and V
        return {
            "flops": ne,
            "bytes": ne * bpe * 2,  # read + write
            "perf_params": {"num_elems": ne},
        }

    # --- Preamble: Embedding (lookup: T rows of hidden_size) ---

    if semantic_block == "Preamble: Embedding":
        ne = T * h
        return {
            "flops": 0,
            "bytes": ne * bpe * 2,  # read embedding + write output
            "perf_params": {"num_elems": ne},
        }

    # --- MoE Routing (small overhead: topk + renormalize on T * num_experts) ---

    if semantic_block == "MoE Routing":
        ne = T * model_cfg["num_experts"]
        return {
            "flops": _elementwise_flops(ne),
            "bytes": _elementwise_bytes(ne, bpe, unary=True),
            "perf_params": {"num_elems": ne},
        }

    # --- MoE Quantize (quantize T*topk activations of hidden_size) ---

    if semantic_block == "MoE Quantize":
        topk = model_cfg["num_experts_per_tok"]
        ne = T * topk * h
        return {
            "flops": _elementwise_flops(ne),
            "bytes": ne * bpe + ne * 1,  # read fp16/bf16, write int8/fp8
            "perf_params": {"num_elems": ne},
        }

    # --- MoE Finalize (scatter/reduce T*topk results back to T*hidden) ---

    if semantic_block == "MoE Finalize":
        topk = model_cfg["num_experts_per_tok"]
        ne = T * topk * h
        return {
            "flops": ne,  # reduction
            "bytes": ne * bpe + T * h * bpe,  # read expert outputs, write reduced
            "perf_params": {"num_elems": ne},
        }

    # --- Fallback: use perf_category to pick the right formula ---

    perf_cat = get_perf_category(semantic_block)

    if perf_cat == "Normalization":
        ne = T * h
        return {
            "flops": _rmsnorm_flops(ne, h),
            "bytes": _rmsnorm_bytes(ne, h, bpe),
            "perf_params": {"num_elems": ne, "num_channels": h},
        }

    if perf_cat == "Elementwise":
        ne = T * h
        # Residual adds and gating are binary (read A + read B + write C).
        # Activations (SiLU, GELU, etc.) are unary (read + write).
        _BINARY_BLOCKS = {
            "Post-Attn Residual Add",
            "Post-FFN Residual Add",
            "Post-MoE Residual Add",
            "Attention Output Gate",
        }
        is_unary = semantic_block not in _BINARY_BLOCKS
        return {
            "flops": _elementwise_flops(ne),
            "bytes": _elementwise_bytes(ne, bpe, unary=is_unary),
            "perf_params": {"num_elems": ne},
        }

    # Should not be reached -- every block has a perf_category with a formula
    return None
