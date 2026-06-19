"""
Perf-model validation script.

Generates a GPU test harness for a TraceLens perf-model op, runs it under
rocprofv3 with hardware counters, then compares measured ops/bytes against
the perf model's predictions.

Usage (single op):
    python validate_perf_model.py \
        --op gemm_a8w8_blockscale \
        --M 2048 --N 4096 --K 8192 \
        --arch gfx942 --output-dir /tmp/perf_validation

Usage (batch):
    python validate_perf_model.py --all --arch gfx942 --output-dir /tmp/perf_val
    python validate_perf_model.py --category activation --arch gfx942 --output-dir /tmp/perf_val

Usage (from TraceLens report directory):
    python validate_perf_model.py \
        --from-report-dir /path/to/report_folder \
        --arch gfx942 --output-dir /tmp/perf_val

Usage (discover missing coverage):
    python validate_perf_model.py --discover
"""

import argparse
import ast
import csv
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from perf_model_harnesses import (
    _ensure_tracelens_importable,
    _make_activation_model_fn,
    _tuple_to_list,
    run_perf_model_add_rmsnorm,
    run_perf_model_ck_moe_stage1,
    run_perf_model_ck_moe_stage2,
    run_perf_model_dynamic_quant,
    run_perf_model_flash_attn,
    run_perf_model_fmha_v3,
    run_perf_model_fmha_v3_varlen,
    run_perf_model_fmoe,
    run_perf_model_from_event,
    run_perf_model_gemm,
    run_perf_model_gemm_a16w16,
    run_perf_model_mha_varlen,
    run_perf_model_moe_unfused_down,
    run_perf_model_moe_unfused_up,
    run_perf_model_sglang_fused_moe_invoke,
    run_perf_model_rms_norm,
    run_perf_model_rmsnorm,
    run_perf_model_rmsnorm_dynamicquant,
    run_perf_model_unified_attention,
    run_perf_model_vllm_gdn_attention_core,
    run_perf_model_vllm_gemm_with_dynamic_quant,
    run_perf_model_vllm_rmsnorm_add_fp8_group_quant,
    run_perf_model_vllm_rmsnorm_fp8_group_quant,
    run_perf_model_vllm_triton_gemm_a8w8_blockscale,
    run_perf_model_vllm_triton_group_quant_fp8,
    run_perf_model_vllm_unified_attention,
    run_perf_model_vllm_unquantized_gemm,
    run_perf_model_dsv3_flydsl_hgemm,
    run_perf_model_dsv3_batched_gemm_a8w8,
    run_perf_model_dsv3_fused_flatten_fp8_group_quant,
    run_perf_model_dsv3_fused_qk_rope_cat_and_cache_mla,
    run_perf_model_dsv3_fused_append_shared_experts,
    run_perf_model_dsv3_mla_prefill_ps_asm_fwd,
    run_perf_model_dsv3_mla_reduce_v1,
    run_perf_model_gemm_afp4wfp4,
    run_perf_model_rope_cached_positions_2c_fwd_impl,
    run_perf_model_fused_flatten_mxfp4_quant,
    run_perf_model_fused_rms_mxfp4_quant,
    run_perf_model_atom_flydsl_preshuffle_gemm_a8,
    run_perf_model_atom_flydsl_gdr_decode,
    run_perf_model_dsv4_mhc_pre_gemm_sqrsum,
    run_perf_model_dsv4_mhc_pre_big_fuse,
    run_perf_model_dsv4_mhc_post,
    run_perf_model_dsv4_pa_sparse_prefill_opus,
    run_perf_model_dsv4_opus_gemm_a16w16,
    run_perf_model_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm,
    run_perf_model_dsv4_dynamic_per_group_scaled_quant,
    run_perf_model_dsv4_topk_softplus,
    run_perf_model_dsv4_fused_dynamic_mx_quant_moe_sort,
)

# *** CORRECTED memory-read counters: the raw TCC RDREQ set is collected instead
# of FETCH_SIZE so read bytes can be computed exactly from the request-size mix
# (see generate_report for the 32B/64B/128B reconstruction). ***
ARCH_COUNTER_CONFIGS = {
    'gfx942': {
        'ops_counters': ['SQ_INSTS_VALU_MFMA_MOPS_I8', 'SQ_INSTS_VALU_MFMA_MOPS_F8'],
        'ops_derived_16': ['TOTAL_16_OPS'],
        'ops_derived_32': ['TOTAL_32_OPS'],
        'mem_read_counters': ['TCC_EA0_RDREQ_sum', 'TCC_EA0_RDREQ_32B_sum', 'TCC_EA0_RDREQ_64B_sum', 'TCC_BUBBLE_sum'],
        'mem_write_counters': ['WRITE_SIZE'],
    },
    'gfx950': {
        'ops_counters': ['SQ_INSTS_VALU_MFMA_MOPS_I8', 'SQ_INSTS_VALU_MFMA_MOPS_F8', 'SQ_INSTS_VALU_MFMA_MOPS_F6F4'],
        'ops_derived_16': ['TOTAL_16_OPS'],
        'ops_derived_32': ['TOTAL_32_OPS'],
        'mem_read_counters': ['TCC_EA0_RDREQ_sum', 'TCC_EA0_RDREQ_32B_sum', 'TCC_EA0_RDREQ_64B_sum', 'TCC_BUBBLE_sum'],
        'mem_write_counters': ['WRITE_SIZE'],
    },
}

PRECISION_TO_COUNTERS = {
    'fp8': ['SQ_INSTS_VALU_MFMA_MOPS_F8'],
    'fp4': ['SQ_INSTS_VALU_MFMA_MOPS_F6F4'],
    'int8': ['SQ_INSTS_VALU_MFMA_MOPS_I8'],
    'fp16': ['TOTAL_16_OPS'],
    'bf16': ['TOTAL_16_OPS'],
    'fp32': ['TOTAL_32_OPS'],
}

COUNTER_DISPLAY = {
    'SQ_INSTS_VALU_MFMA_MOPS_I8': (512, 'INT8 MFMA Ops', 'ops'),
    'SQ_INSTS_VALU_MFMA_MOPS_F8': (512, 'FP8 MFMA Ops', 'ops'),
    'TOTAL_16_OPS': (1, 'Total 16-bit Ops', 'ops'),
    'TOTAL_32_OPS': (1, 'Total 32-bit Ops', 'ops'),
    'SQ_INSTS_VALU_MFMA_MOPS_F6F4': (512, 'FP4/FP6 MFMA Ops', 'ops'),
    'FETCH_SIZE': (1024, 'HBM Read', 'bytes'),
    'WRITE_SIZE': (1024, 'HBM Write', 'bytes'),
}

# Raw TCC read-request counters kept by parse_counter_csv even though they are
# not part of COUNTER_DISPLAY; generate_report reconstructs read bytes from them.
_MEM_READ_RAW_COUNTERS = {
    'TCC_EA0_RDREQ_sum',
    'TCC_EA0_RDREQ_32B_sum',
    'TCC_EA0_RDREQ_64B_sum',
    'TCC_BUBBLE_sum',
}

_OP_CLASS_MAP = {
    'gemm_a8w8_blockscale': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'gemm_a8w8_blockscale'),
    'gemm_a16w16_atomic_': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'gemm_a16w16_atomic_'),
    'fmoe_fp8_blockscale_g1u1': ('TraceLens.PerfModel.extensions.moe_perf_model_extensions', 'moe_aiter_fused_blockscale'),
    'moe_cktile2stages_gemm1_ck': ('TraceLens.PerfModel.extensions.moe_perf_model_extensions', 'moe_aiter_unfused_up'),
    'moe_cktile2stages_gemm2_ck': ('TraceLens.PerfModel.extensions.moe_perf_model_extensions', 'moe_aiter_unfused_down'),
    'ck_moe_stage1': ('TraceLens.PerfModel.extensions.moe_perf_model_extensions', 'moe_aiter_ck_stage1'),
    'ck_moe_stage2': ('TraceLens.PerfModel.extensions.moe_perf_model_extensions', 'moe_aiter_ck_stage2'),
    'silu_and_mul': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'aiter_silu_and_mul'),
    'gelu_and_mul': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'aiter_gelu_and_mul'),
    'gelu_tanh_and_mul': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'aiter_gelu_tanh_and_mul'),
    'rms_norm': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'aiter_rms_norm'),
    'rmsnorm': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'aiter_rmsnorm'),
    'add_rmsnorm': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'aiter_rmsnorm2d_fwd_with_add_ck'),
    'rmsnorm_dynamicquant': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'aiter_rmsnorm2d_fwd_with_dynamicquant_ck'),
    'dynamic_per_token_scaled_quant': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'per_group_quant'),
    '_flash_attn_forward': ('TraceLens.PerfModel.perf_model', 'aiter__flash_attn_forward'),
    'wrapper_fmha_v3_fwd': ('TraceLens.PerfModel.perf_model', 'aiter__fmha_v3_forward'),
    'mha_varlen_fwd': ('TraceLens.PerfModel.extensions.attention_perf_model_extensions', 'mha_varlen_fwd'),
    'fmha_v3_varlen_fwd': ('TraceLens.PerfModel.extensions.attention_perf_model_extensions', 'aiter_fmha_v3_varlen_fwd'),
    'vllm_unquantized_gemm': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'vllm_rocm_unquantized_gemm'),
    'vllm_triton_gemm_a8w8_blockscale': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'gemm_a8w8_blockscale'),
    'vllm_gemm_with_dynamic_quant': ('TraceLens.PerfModel.perf_model', 'vllm_gemm_with_dynamic_quant'),
    'vllm_triton_group_quant_fp8': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'vllm_triton_per_token_group_quant_fp8'),
    'vllm_rmsnorm_fp8_group_quant': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'vllm_rocm_aiter_rmsnorm_fp8_group_quant'),
    'vllm_rmsnorm_add_fp8_group_quant': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'vllm_rocm_aiter_rmsnorm_with_add_fp8_group_quant'),
    'vllm_unified_attention': ('TraceLens.PerfModel.extensions.attention_perf_model_extensions', 'vllm_unified_attention_with_output'),
    'vllm_gdn_attention_core': ('TraceLens.PerfModel.extensions.attention_perf_model_extensions', 'gdn_attention_core'),
    'dsv3_flydsl_hgemm': ('__dsv3_ext__', 'sglang_flydsl_hgemm'),
    'dsv3_batched_gemm_a8w8': ('__dsv3_ext__', 'sglang_batched_gemm_a8w8'),
    'dsv3_fused_flatten_fp8_group_quant': ('__dsv3_ext__', 'sglang_fused_flatten_fp8_group_quant'),
    'dsv3_fused_qk_rope_cat_and_cache_mla': ('__dsv3_ext__', 'aiter_fused_qk_rope_cat_and_cache_mla'),
    'dsv3_fused_append_shared_experts': ('__dsv3_ext__', 'sglang_fused_append_shared_experts'),
    'dsv3_mla_prefill_ps_asm_fwd': ('__dsv3_ext__', 'aiter_mla_prefill_ps_asm_fwd'),
    'dsv3_mla_reduce_v1': ('__dsv3_ext__', 'aiter_mla_reduce_v1'),
    'gemm_afp4wfp4': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'gemm_afp4wfp4'),
    'rope_cached_positions_2c_fwd_impl': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'aiter_rope_cached_positions_2c_fwd_impl'),
    'fused_flatten_mxfp4_quant': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'fused_flatten_mxfp4_quant'),
    'fused_rms_mxfp4_quant': ('TraceLens.PerfModel.extensions.rmsnorm_perf_model_extensions', 'fused_rms_mxfp4_quant'),
    'atom_flydsl_preshuffle_gemm_a8': ('TraceLens.PerfModel.extensions.perf_model_extensions', 'gemm_a8w8_blockscale'),
    'atom_flydsl_gdr_decode': ('TraceLens.PerfModel.extensions.attention_perf_model_extensions', 'gdn_attention_core'),
}


def _resolve_perf_model_source(op_name):
    """Return (class_name, source_file, source_line) for the perf model class."""
    if op_name not in _OP_CLASS_MAP:
        return (op_name, 'unknown', 0)
    _ensure_tracelens_importable()
    mod_path, cls_name = _OP_CLASS_MAP[op_name]
    try:
        if mod_path == '__dsv3_ext__':
            from perf_model_harnesses import _load_dsv3_extension
            mod = _load_dsv3_extension()
        else:
            mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        src_file = inspect.getfile(cls)
        repo_root = str(Path(__file__).resolve().parents[3])
        src_file = os.path.relpath(src_file, repo_root)
        _, src_line = inspect.getsourcelines(cls)
        return (cls_name, src_file, src_line)
    except Exception:
        return (cls_name, mod_path.replace('.', '/') + '.py', 0)


TESTS_RUNNER = Path(__file__).resolve().parent / 'tests' / '_runner.py'

_RUNNER_INT_FLAGS = ('M', 'N', 'K', 'E', 'topk', 'group_size', 'seq_len',
                     'num_heads_q', 'num_heads_kv', 'head_dim', 'block_n',
                     'block_k', 'block_m', 'split_k')

_RUNNER_STR_FLAGS = ('in_dtype', 'w_dtype', 'out_dtype', 'scale_dtype',
                     'quant_dtype', 'quant_type', 'activation', 'kv_dtype',
                     'bias_dtype')


def _build_runner_argv(op_name, effective_args, num_warmup=3):
    """Build the argv list that rocprofv3 wraps for op ``op_name``.

    Returns ``(argv, num_warmup_iters)`` -- num_warmup_iters is also forwarded
    on the command line and is returned separately so ``discover_kernel_name``
    can compute the expected dispatch count.
    """
    cmd = [sys.executable, str(TESTS_RUNNER), '--op', op_name, '--num-warmup', str(num_warmup)]
    for k in _RUNNER_INT_FLAGS:
        v = getattr(effective_args, k, None)
        if v is not None:
            cmd += [f'--{k}', str(v)]
    for k in _RUNNER_STR_FLAGS:
        v = getattr(effective_args, k, None)
        if v is not None and v != '':
            cmd += [f'--{k}', str(v)]
    varlen_seed = getattr(effective_args, 'varlen_seed', None)
    if varlen_seed is not None:
        cmd += ['--varlen-seed', str(varlen_seed)]
    varlen_num_seqs = getattr(effective_args, 'varlen_num_seqs', None)
    if varlen_num_seqs is not None:
        cmd += ['--varlen-num-seqs', str(varlen_num_seqs)]
    varlen_scenario = getattr(effective_args, 'varlen_scenario', None)
    if varlen_scenario:
        cmd += ['--varlen-scenario', str(varlen_scenario)]
    ann = getattr(effective_args, 'annotation', None)
    if ann:
        cmd += ['--annotation', ann]
    return (cmd, num_warmup)


def _build_csv_runner_argv(registry_key, effective_args, num_warmup=3, input_dims=None, input_types=None):
    """Build a runner argv for the generic CSV path.

    ``registry_key`` is the OP_REGISTRY key (also used by
    ``test_generic_simple_op`` to look up the call dispatch); ``input_dims``
    and ``input_types`` are the raw lists from ``unified_perf_summary.csv``.
    """
    cmd = [sys.executable, str(TESTS_RUNNER), '--op', '__generic__',
           '--registry-key', registry_key, '--num-warmup', str(num_warmup)]
    if input_dims is not None:
        cmd += ['--input-dims-json', json.dumps(_tuple_to_list(input_dims))]
    if input_types is not None:
        cmd += ['--input-types-json', json.dumps(list(input_types) if input_types else [])]
    return (cmd, num_warmup)


def _populate_varlen_attention_args(op_name, effective_args):
    """Pre-compute c_sq/c_sqsk on the parent side for varlen attention ops.

    The harness used to mutate ``args._varlen_c_sq`` / ``args._varlen_c_sqsq``
    in-process so the perf-model runner could read the per-batch aggregates.
    Now that the harness runs in a rocprof'd subprocess, the parent computes
    them itself from the same deterministic seed + scenario used by
    ``tests/attention.make_varlen_seqlens``. Also pre-computes total Q / KV
    token counts so the perf-model harnesses can size Input Dims correctly
    for non-self-attention scenarios (mixed prefill+decode).
    """
    if op_name not in ('mha_varlen_fwd', 'fmha_v3_varlen_fwd'):
        return None
    if getattr(effective_args, '_varlen_c_sq', None) is not None:
        return None
    try:
        from tests.attention import compute_varlen_annotation_stats, make_varlen_seqlens
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tests.attention import compute_varlen_annotation_stats, make_varlen_seqlens
    num_seqs = getattr(effective_args, 'varlen_num_seqs', None) or 4
    seed = getattr(effective_args, 'varlen_seed', None) or 42
    scenario = getattr(effective_args, 'varlen_scenario', None) or 'random'
    sq, cu_q, sk, cu_k = make_varlen_seqlens(
        effective_args.seq_len, num_seqs=num_seqs, seed=seed, scenario=scenario)
    c_sq, c_sqsk = compute_varlen_annotation_stats(sq, sk)
    effective_args._varlen_c_sq = c_sq
    effective_args._varlen_c_sqsq = c_sqsk
    effective_args._varlen_max_seqlen = max(sq + sk)
    effective_args._varlen_total_q = cu_q[-1]
    effective_args._varlen_total_kv = cu_k[-1]
    return None


OP_REGISTRY = {
    'gemm_a8w8_blockscale': {
        'category': 'gemm',
        'model_fn': run_perf_model_gemm,
        'defaults': {'M': 2048, 'N': 4096, 'K': 8192},
        'required_args': ['M', 'N', 'K'],
        'description': 'FP8 block-scaled GEMM (aiter CK)',
    },
    'gemm_a16w16_atomic_': {
        'category': 'gemm',
        'model_fn': run_perf_model_gemm_a16w16,
        'defaults': {'M': 2048, 'N': 4096, 'K': 8192},
        'required_args': ['M', 'N', 'K'],
        'description': 'BF16 GEMM (aiter ASM, atomic accumulation)',
    },
    'fmoe_fp8_blockscale_g1u1': {
        'category': 'moe',
        'model_fn': run_perf_model_fmoe,
        'defaults': {'M': 512, 'N': 2048, 'K': 7168, 'E': 8, 'topk': 2},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'Fused FP8 block-scale MoE (g1u1 SwiGLU)',
    },
    'moe_cktile2stages_gemm1_ck': {
        'category': 'moe',
        'model_fn': run_perf_model_moe_unfused_up,
        'defaults': {'M': 512, 'N': 2048, 'K': 7168, 'E': 8, 'topk': 2},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'CKTile MoE stage-1 GEMM (up projection)',
    },
    'moe_cktile2stages_gemm2_ck': {
        'category': 'moe',
        'model_fn': run_perf_model_moe_unfused_down,
        'defaults': {'M': 512, 'N': 2048, 'K': 7168, 'E': 8, 'topk': 2},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'CKTile MoE stage-2 GEMM (down projection)',
    },
    'ck_moe_stage1': {
        'category': 'moe',
        'model_fn': run_perf_model_ck_moe_stage1,
        'defaults': {'M': 512, 'N': 2048, 'K': 7168, 'E': 8, 'topk': 2},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'CK MoE stage-1 (up projection, fused activation)',
    },
    'ck_moe_stage2': {
        'category': 'moe',
        'model_fn': run_perf_model_ck_moe_stage2,
        'defaults': {'M': 512, 'N': 2048, 'K': 7168, 'E': 8, 'topk': 2},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'CK MoE stage-2 (down projection)',
    },
    'sglang_fused_moe_triton_invoke': {
        'category': 'moe',
        'model_fn': run_perf_model_sglang_fused_moe_invoke,
        # Qwen3-30B-A3B gate/up shape: hidden=2048, 2*inter=1536, E=128, topk=8.
        'defaults': {'M': 15360, 'N': 1536, 'K': 2048, 'E': 128, 'topk': 8},
        'required_args': ['M', 'N', 'K', 'E', 'topk'],
        'description': 'SGLang Triton fused-MoE grouped GEMM (invoke_fused_moe_kernel)',
    },
    'silu_and_mul': {
        'category': 'activation',
        'model_fn': _make_activation_model_fn('silu_and_mul', 'aiter_silu_and_mul'),
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'SiLU-gated activation (SwiGLU gate*up)',
    },
    'gelu_and_mul': {
        'category': 'activation',
        'model_fn': _make_activation_model_fn('gelu_and_mul', 'aiter_gelu_and_mul'),
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'GELU-gated activation',
    },
    'gelu_tanh_and_mul': {
        'category': 'activation',
        'model_fn': _make_activation_model_fn('gelu_tanh_and_mul', 'aiter_gelu_tanh_and_mul'),
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'GELU-tanh-gated activation',
    },
    'rms_norm': {
        'category': 'norm',
        'model_fn': run_perf_model_rms_norm,
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'RMSNorm (CK, aiter::rms_norm)',
    },
    'rmsnorm': {
        'category': 'norm',
        'model_fn': run_perf_model_rmsnorm,
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'RMSNorm (aiter::rmsnorm, out-first API)',
    },
    'add_rmsnorm': {
        'category': 'norm',
        'model_fn': run_perf_model_add_rmsnorm,
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'Fused residual-add + RMSNorm',
    },
    'rmsnorm_dynamicquant': {
        'category': 'norm',
        'model_fn': run_perf_model_rmsnorm_dynamicquant,
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'Fused RMSNorm + dynamic FP8 quantization',
    },
    'dynamic_per_token_scaled_quant': {
        'category': 'quant',
        'model_fn': run_perf_model_dynamic_quant,
        'defaults': {'M': 2048, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'Dynamic per-token FP8 quantization',
    },
    '_flash_attn_forward': {
        'category': 'attention',
        'model_fn': run_perf_model_flash_attn,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'Flash Attention forward (CK)',
    },
    'wrapper_fmha_v3_fwd': {
        'category': 'attention',
        'model_fn': run_perf_model_fmha_v3,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'FMHA v3 forward',
    },
    'mha_varlen_fwd': {
        'category': 'attention',
        'model_fn': run_perf_model_mha_varlen,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'MHA variable-length forward (CK)',
    },
    'fmha_v3_varlen_fwd': {
        'category': 'attention',
        'model_fn': run_perf_model_fmha_v3_varlen,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'FMHA v3 variable-length forward',
    },
    'vllm_unquantized_gemm': {
        'category': 'vllm_gemm',
        'model_fn': run_perf_model_vllm_unquantized_gemm,
        'defaults': {'M': 2048, 'N': 4096, 'K': 8192},
        'required_args': ['M', 'N', 'K'],
        'description': 'vLLM BF16 GEMM (rocm_unquantized_gemm)',
    },
    'vllm_triton_gemm_a8w8_blockscale': {
        'category': 'vllm_gemm',
        'model_fn': run_perf_model_vllm_triton_gemm_a8w8_blockscale,
        'defaults': {'M': 2048, 'N': 4096, 'K': 8192},
        'required_args': ['M', 'N', 'K'],
        'description': 'vLLM FP8 block-scaled GEMM (wraps aiter gemm_a8w8_blockscale)',
    },
    'vllm_gemm_with_dynamic_quant': {
        'category': 'vllm_gemm',
        'model_fn': run_perf_model_vllm_gemm_with_dynamic_quant,
        'defaults': {'M': 2048, 'N': 4096, 'K': 8192},
        'required_args': ['M', 'N', 'K'],
        'description': 'vLLM FP4 GEMM with dynamic quantization (Quark OCP MX)',
        'perf_model_only': True,
    },
    'vllm_triton_group_quant_fp8': {
        'category': 'vllm_quant',
        'model_fn': run_perf_model_vllm_triton_group_quant_fp8,
        'defaults': {'M': 2048, 'N': 7168, 'group_size': 128},
        'required_args': ['M', 'N'],
        'description': 'vLLM Triton FP8 per-group quantization',
    },
    'vllm_rmsnorm_fp8_group_quant': {
        'category': 'vllm_norm',
        'model_fn': run_perf_model_vllm_rmsnorm_fp8_group_quant,
        'defaults': {'M': 2048, 'N': 7168, 'group_size': 128},
        'required_args': ['M', 'N'],
        'description': 'vLLM fused RMSNorm + FP8 group quantization',
    },
    'vllm_rmsnorm_add_fp8_group_quant': {
        'category': 'vllm_norm',
        'model_fn': run_perf_model_vllm_rmsnorm_add_fp8_group_quant,
        'defaults': {'M': 2048, 'N': 7168, 'group_size': 128},
        'required_args': ['M', 'N'],
        'description': 'vLLM fused residual-add + RMSNorm + FP8 group quantization',
    },
    'vllm_unified_attention': {
        'category': 'vllm_attention',
        'model_fn': run_perf_model_vllm_unified_attention,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'vLLM unified attention (paged decode via aiter.paged_attention_v1, FP8 KV cache)',
    },
    'unified_attention': {
        'category': 'attention',
        'model_fn': run_perf_model_unified_attention,
        'defaults': {'seq_len': 2048, 'num_heads_q': 32, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'],
        'description': 'aiter triton unified_attention (paged KV / varlen, accepts --annotation)',
    },
    'vllm_gdn_attention_core': {
        'category': 'vllm_attention',
        'model_fn': run_perf_model_vllm_gdn_attention_core,
        'defaults': {'seq_len': 2048, 'num_heads_kv': 16, 'head_dim': 128},
        'required_args': ['seq_len', 'num_heads_kv', 'head_dim'],
        'description': 'vLLM Gated Delta Network attention (perf-model-only)',
        'perf_model_only': True,
    },
    'dsv3_flydsl_hgemm': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_flydsl_hgemm,
        'defaults': {'M': 32, 'N': 256, 'K': 7168},
        'required_args': ['M', 'N', 'K'],
        'description': 'AITER FlyDSL BF16 GEMM (sglang_profiler::gemm_kernels_flydsl_hgemm_54)',
    },
    'dsv3_batched_gemm_a8w8': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_batched_gemm_a8w8,
        'defaults': {'M': 32, 'N': 128, 'K': 512, 'E': 16, 'group_size': 128},
        'required_args': ['M', 'N', 'K', 'E'],
        'description': 'AITER batched A8W8 GEMM (per-token-group prequant)',
    },
    'dsv3_fused_flatten_fp8_group_quant': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_fused_flatten_fp8_group_quant,
        'defaults': {'M': 32, 'N': 2048, 'group_size': 128},
        'required_args': ['M', 'N'],
        'description': 'AITER triton fused flatten + FP8 group quant',
    },
    'dsv3_fused_qk_rope_cat_and_cache_mla': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_fused_qk_rope_cat_and_cache_mla,
        'defaults': {'M': 32, 'num_heads_q': 16, 'head_dim': 576, 'group_size': 64},
        'required_args': ['M'],
        'description': 'AITER triton fused QK RoPE + KV cache write (MLA)',
    },
    'dsv3_fused_append_shared_experts': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_fused_append_shared_experts,
        'defaults': {'M': 32, 'K': 8, 'E': 256},
        'required_args': ['M'],
        'description': 'sglang triton MoE fused-append-shared-experts',
    },
    'dsv3_mla_prefill_ps_asm_fwd': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_mla_prefill_ps_asm_fwd,
        'defaults': {'seq_len': 1024, 'E': 16, 'num_heads_q': 16, 'head_dim': 192, 'block_n': 1},
        'required_args': ['seq_len'],
        'description': 'AITER ASM MLA prefill (persistent scheduler, FP8)',
    },
    'dsv3_mla_reduce_v1': {
        'category': 'dsv3',
        'model_fn': run_perf_model_dsv3_mla_reduce_v1,
        'defaults': {'seq_len': 1024, 'E': 16, 'num_heads_q': 16, 'head_dim': 192, 'block_n': 1},
        'required_args': ['seq_len'],
        'description': 'AITER MLA cross-split reduce (paired with mla_prefill_ps_asm_fwd)',
    },
    'gemm_afp4wfp4': {
        'category': 'ext_mxfp4',
        'model_fn': run_perf_model_gemm_afp4wfp4,
        'defaults': {'M': 822, 'N': 2112, 'K': 3584},
        'required_args': ['M', 'N', 'K'],
        'description': 'AITER MXFP4 GEMM (gemm_afp4wfp4_)',
    },
    'rope_cached_positions_2c_fwd_impl': {
        'category': 'ext_mxfp4',
        'model_fn': run_perf_model_rope_cached_positions_2c_fwd_impl,
        'defaults': {'M': 822, 'num_heads_q': 16, 'num_heads_kv': 1, 'head_dim': 64},
        'required_args': ['M'],
        'description': 'AITER 2-channel cached-positions forward RoPE',
    },
    'fused_flatten_mxfp4_quant': {
        'category': 'ext_mxfp4',
        'model_fn': run_perf_model_fused_flatten_mxfp4_quant,
        'defaults': {'M': 822, 'N': 7168, 'group_size': 128},
        'required_args': ['M', 'N'],
        'description': 'AITER triton fused flatten + MXFP4 quant',
    },
    'fused_rms_mxfp4_quant': {
        'category': 'ext_mxfp4',
        'model_fn': run_perf_model_fused_rms_mxfp4_quant,
        'defaults': {'M': 822, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'AITER triton fused RMSNorm + MXFP4 quant',
    },
    'atom_flydsl_preshuffle_gemm_a8': {
        'category': 'atom_flydsl',
        'model_fn': run_perf_model_atom_flydsl_preshuffle_gemm_a8,
        'defaults': {'M': 512, 'N': 4096, 'K': 7168},
        'required_args': ['M', 'N', 'K'],
        'description': 'AITER FlyDSL A8W8 preshuffle GEMM (flydsl_preshuffle_gemm_a8)',
    },
    'atom_flydsl_gdr_decode': {
        'category': 'atom_flydsl',
        'model_fn': run_perf_model_atom_flydsl_gdr_decode,
        'defaults': {'seq_len': 128, 'num_heads_kv': 8, 'head_dim': 128},
        'required_args': ['seq_len'],
        'description': 'AITER FlyDSL gated-delta-rule decode (flydsl_gdr_decode)',
    },
    'dsv4_mhc_pre_gemm_sqrsum': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_mhc_pre_gemm_sqrsum,
        'defaults': {'M': 1819, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'DSV4 mHC pre GEMM + sum-of-squares (aiter)',
    },
    'dsv4_mhc_pre_big_fuse': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_mhc_pre_big_fuse,
        'defaults': {'M': 1819, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'DSV4 mHC pre RMS+Sinkhorn+mix fuse (aiter)',
    },
    'dsv4_mhc_post': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_mhc_post,
        'defaults': {'M': 1819, 'N': 7168},
        'required_args': ['M', 'N'],
        'description': 'DSV4 mHC post stream-merge (aiter)',
    },
    'dsv4_pa_sparse_prefill_opus': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_pa_sparse_prefill_opus,
        'defaults': {'M': 1819, 'num_heads_q': 32, 'head_dim': 512},
        'required_args': ['M', 'num_heads_q', 'head_dim'],
        'description': 'DSV4 sparse paged prefill MLA attention (aiter)',
    },
    'dsv4_opus_gemm_a16w16': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_opus_gemm_a16w16,
        'defaults': {'M': 1819, 'N': 384, 'K': 7168},
        'required_args': ['M', 'N', 'K'],
        'description': 'DSV4 batched BF16 GEMM (aiter opus split-K)',
    },
    'dsv4_gemm_a8w8_blockscale_bpreshuffle_asm': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm,
        'defaults': {'M': 1819, 'N': 2048, 'K': 7168},
        'required_args': ['M', 'N', 'K'],
        'description': 'DSV4 FP8 block-scaled GEMM, B preshuffle (aiter ASM)',
    },
    'dsv4_dynamic_per_group_scaled_quant': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_dynamic_per_group_scaled_quant,
        'defaults': {'M': 1819, 'N': 7168, 'group_size': 32},
        'required_args': ['M', 'N'],
        'description': 'DSV4 per-group dynamic MX-FP8 quant (aiter)',
    },
    'dsv4_topk_softplus': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_topk_softplus,
        'defaults': {'M': 1819, 'N': 384, 'topk': 6},
        'required_args': ['M', 'N', 'topk'],
        'description': 'DSV4 MoE router topk + sqrt(softplus) (aiter)',
    },
    'dsv4_fused_dynamic_mx_quant_moe_sort': {
        'category': 'dsv4',
        'model_fn': run_perf_model_dsv4_fused_dynamic_mx_quant_moe_sort,
        'defaults': {'M': 32, 'N': 7168, 'E': 384, 'topk': 6, 'group_size': 32},
        'required_args': ['M', 'N', 'E', 'topk'],
        'description': 'DSV4 fused MX-FP8 quant + MoE sort (aiter)',
    },
}

CATEGORIES = sorted(set(v['category'] for v in OP_REGISTRY.values()))

CSV_NAME_TO_REGISTRY = {
    'aiter::gemm_a8w8_blockscale_ck': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_blockscale_cktile': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_blockscale_bpreshuffle_ck': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_blockscale_bpreshuffle_cktile': 'gemm_a8w8_blockscale',
    'aiter::flatmm_a8w8_blockscale_asm': 'gemm_a8w8_blockscale',
    'aiter::gfx950_a8w8_blockscale_asm': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_ck': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_bpreshuffle_ck': 'gemm_a8w8_blockscale',
    'aiter::gemm_a8w8_bpreshuffle': 'atom_flydsl_preshuffle_gemm_a8',
    'vllm::_rocm_aiter_preshuffled_per_token_w8a8_gemm': 'atom_flydsl_preshuffle_gemm_a8',
    'sglang_profiler::fp8_utils_gemm_a8w8_blockscale_7': 'gemm_a8w8_blockscale',
    'aiter::gemm_a16w16_atomic_': 'gemm_a16w16_atomic_',
    'aiter::fmoe_fp8_blockscale_g1u1': 'fmoe_fp8_blockscale_g1u1',
    'aiter::moe_cktile2stages_gemm1_ck': 'moe_cktile2stages_gemm1_ck',
    'aiter::moe_cktile2stages_gemm2_ck': 'moe_cktile2stages_gemm2_ck',
    'aiter::ck_moe_stage1': 'ck_moe_stage1',
    'aiter::ck_moe_stage2': 'ck_moe_stage2',
    'sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel': 'sglang_fused_moe_triton_invoke',
    'sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel_427': 'sglang_fused_moe_triton_invoke',
    'aiter::silu_and_mul': 'silu_and_mul',
    '_C::silu_and_mul': 'silu_and_mul',
    'aiter::gelu_and_mul': 'gelu_and_mul',
    'aiter::gelu_tanh_and_mul': 'gelu_tanh_and_mul',
    'aiter::rms_norm': 'rms_norm',
    'aiter::rmsnorm2d_fwd_ck': 'rms_norm',
    'aiter::rmsnorm2d_fwd_with_add_ck': 'add_rmsnorm',
    'aiter::rmsnorm2d_fwd_with_dynamicquant_ck': 'rmsnorm_dynamicquant',
    'aiter::dynamic_per_token_scaled_quant': 'dynamic_per_token_scaled_quant',
    'vllm::rocm_unquantized_gemm': 'vllm_unquantized_gemm',
    'vllm::rocm_aiter_triton_gemm_a8w8_blockscale': 'vllm_triton_gemm_a8w8_blockscale',
    'vllm::triton_per_token_group_quant_fp8': 'vllm_triton_group_quant_fp8',
    'vllm::rocm_aiter_rmsnorm_fp8_group_quant': 'vllm_rmsnorm_fp8_group_quant',
    'vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant': 'vllm_rmsnorm_add_fp8_group_quant',
    'vllm::unified_attention_with_output': 'vllm_unified_attention',
    'aiter::mha_varlen_fwd': 'mha_varlen_fwd',
    'aiter::fmha_v3_varlen_fwd': 'fmha_v3_varlen_fwd',
    'aiter::_flash_attn_forward': '_flash_attn_forward',
    'aiter::wrapper_fmha_v3_fwd': 'wrapper_fmha_v3_fwd',
    'aiter::unified_attention': 'unified_attention',
    'aiter::linear_attention_with_output_base': 'atom_flydsl_gdr_decode',
    'aiter::gemm_afp4wfp4_': 'gemm_afp4wfp4',
    'aiter::rope_cached_positions_2c_fwd_impl': 'rope_cached_positions_2c_fwd_impl',
    'sglang_profiler::fused_mxfp4_quant_fused_flatten_mxfp4_quant': 'fused_flatten_mxfp4_quant',
    'sglang_profiler::fused_mxfp4_quant_fused_rms_mxfp4_quant': 'fused_rms_mxfp4_quant',
}

USE_EXISTING_HARNESS_FOR_CSV = frozenset({
    'ck_moe_stage1', 'ck_moe_stage2', 'mha_varlen_fwd', 'unified_attention',
    'fmha_v3_varlen_fwd', 'wrapper_fmha_v3_fwd', 'atom_flydsl_gdr_decode',
    'vllm_unified_attention', 'vllm_gdn_attention_core', 'fmoe_fp8_blockscale_g1u1',
    'moe_cktile2stages_gemm1_ck', 'moe_cktile2stages_gemm2_ck',
    'atom_flydsl_preshuffle_gemm_a8',
})


def discover_missing_coverage():
    """Scan TraceLens perf model mappings for aiter:: and vllm:: ops and report coverage."""
    _ensure_tracelens_importable()
    try:
        from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import get_pseudo_op_mappings
        from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
    except ImportError as e:
        print(f'  WARNING: Could not import perf model mappings: {e}')
        return None
    all_ops = {}
    for name, cls in get_pseudo_op_mappings().items():
        if name.startswith('aiter::') or name.startswith('vllm::'):
            all_ops[name] = cls.__name__
    for name, cls in op_to_perf_model_class_map.items():
        if name.startswith('aiter::') or name.startswith('vllm::'):
            all_ops[name] = cls.__name__
    SKIP_PATTERNS = ['backward', 'bwd', 'reduce_scatter', 'all_gather', 'fused_allreduce']
    registry_trace_names = set()
    for op_key in OP_REGISTRY:
        registry_trace_names.add(f'aiter::{op_key}')
    _VLLM_REGISTRY_TO_TRACE = {
        'vllm::rocm_unquantized_gemm': 'vllm_unquantized_gemm',
        'vllm::rocm_aiter_triton_gemm_a8w8_blockscale': 'vllm_triton_gemm_a8w8_blockscale',
        'vllm::gemm_with_dynamic_quant': 'vllm_gemm_with_dynamic_quant',
        'vllm::triton_per_token_group_quant_fp8': 'vllm_triton_group_quant_fp8',
        'vllm::rocm_aiter_rmsnorm_fp8_group_quant': 'vllm_rmsnorm_fp8_group_quant',
        'vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant': 'vllm_rmsnorm_add_fp8_group_quant',
        'vllm::unified_attention_with_output': 'vllm_unified_attention',
        'vllm::gdn_attention_core': 'vllm_gdn_attention_core',
    }
    for trace_name, reg_key in _VLLM_REGISTRY_TO_TRACE.items():
        if reg_key in OP_REGISTRY:
            registry_trace_names.add(trace_name)
    _VARIANT_MAP = {
        'aiter::gemm_a8w8_blockscale_ck': 'gemm_a8w8_blockscale',
        'aiter::gemm_a8w8_blockscale_cktile': 'gemm_a8w8_blockscale',
        'aiter::gemm_a8w8_blockscale_bpreshuffle_ck': 'gemm_a8w8_blockscale',
        'aiter::gemm_a8w8_blockscale_bpreshuffle_cktile': 'gemm_a8w8_blockscale',
        'aiter::flatmm_a8w8_blockscale_asm': 'gemm_a8w8_blockscale',
        'aiter::gfx950_a8w8_blockscale_asm': 'gemm_a8w8_blockscale',
        'aiter::gemm_a8w8_ck': 'gemm_a8w8_blockscale',
        'aiter::gemm_a8w8_bpreshuffle_ck': 'gemm_a8w8_blockscale',
        'aiter::rmsnorm2d_fwd_ck': 'rms_norm',
        'aiter::rmsnorm2d_fwd_with_add_ck': 'add_rmsnorm',
        'aiter::rmsnorm2d_fwd_with_dynamicquant_ck': 'rmsnorm_dynamicquant',
    }
    for trace_name in _VARIANT_MAP:
        registry_trace_names.add(trace_name)
    print('\n======================================================================')
    print('AITER:: & VLLM:: OP COVERAGE REPORT')
    print('======================================================================')
    covered = []
    missing = []
    skipped = []
    for trace_name in sorted(all_ops):
        cls_name = all_ops[trace_name]
        if any(p in trace_name.lower() for p in SKIP_PATTERNS):
            skipped.append((trace_name, cls_name, 'skipped (collective/backward)'))
            continue
        short_name = trace_name.replace('aiter::', '').replace('vllm::', '')
        if trace_name in registry_trace_names or short_name in OP_REGISTRY:
            covered.append((trace_name, cls_name))
        else:
            missing.append((trace_name, cls_name))
    print(f'\nCovered ({len(covered)}):')
    for name, cls in covered:
        print(f'  [OK]   {name:50s}  -> {cls}')
    if missing:
        print(f'\nMissing ({len(missing)}):')
        for name, cls in missing:
            print(f'  [--]   {name:50s}  -> {cls}')
    if skipped:
        print(f'\nSkipped ({len(skipped)}):')
        for name, cls, reason in skipped:
            print(f'  [skip] {name:50s}  ({reason})')
    print('======================================================================')
    return missing


def run_rocprofv3(rocprofv3_path, counters, runner_argv, out_name, out_dir, timeout=300):
    """Wrap a tests/_runner.py invocation under rocprofv3 PMC counters.

    ``runner_argv`` is the full argv list to execute inside rocprofv3 (e.g.
    ``[python, /abs/tests/_runner.py, --op, gemm_a8w8_blockscale, --M, ...]``).
    """
    counter_dir = os.path.join(out_dir, out_name)
    os.makedirs(counter_dir, exist_ok=True)
    cmd = [rocprofv3_path, '--pmc']
    cmd.extend(counters)
    cmd.append('--output-format')
    cmd.append('csv')
    cmd.append('-o')
    cmd.append(out_name)
    cmd.append('-d')
    cmd.append(counter_dir)
    cmd.append('--')
    cmd.extend(runner_argv)
    stdout_log = os.path.join(out_dir, f'{out_name}_stdout.log')
    stderr_log = os.path.join(out_dir, f'{out_name}_stderr.log')
    print(f'\n{"============================================================"}')
    print(f'Running: {" ".join(cmd)}')
    print(f'Logs:    {stdout_log}')
    print(f'         {stderr_log}')
    print(f'Timeout: {timeout}s')
    print(f'{"============================================================"}')
    with open(stdout_log, 'w') as f_out, open(stderr_log, 'w') as f_err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            f_out.write(stdout or '')
            f_err.write(stderr or '')
            raise RuntimeError(
                f'rocprofv3 timed out after {timeout}s. Check logs:\n  {stdout_log}\n  {stderr_log}')
        f_out.write(stdout or '')
        f_err.write(stderr or '')
    if stdout:
        print(f'[{out_name} stdout]:\n{stdout}')
    if stderr:
        print(f'[{out_name} stderr]:\n{stderr}')
    if proc.returncode != 0:
        raise RuntimeError(
            f'rocprofv3 failed with exit code {proc.returncode}. Check logs:\n  {stdout_log}\n  {stderr_log}')
    csv_path = os.path.join(counter_dir, f'{out_name}_counter_collection.csv')
    if not os.path.exists(csv_path):
        candidates = list(Path(counter_dir).rglob('*counter_collection.csv'))
        if candidates:
            csv_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f'Counter CSV not found in {counter_dir}. Directory contents: '
                f'{os.listdir(counter_dir)}. Check logs:\n  {stdout_log}\n  {stderr_log}')
    print(f'Counter CSV: {csv_path}')
    return csv_path


_INFRA_KERNEL_PATTERNS = ['at::native::', 'at::cuda::', 'void hip', 'Cijk_', 'void at::native']


def discover_kernel_name(rocprofv3_path, runner_argv, num_warmup_iters, out_dir, timeout=120):
    """Trace one execution of ``runner_argv`` to auto-discover the target kernel."""
    disc_dir = os.path.join(out_dir, 'kernel_discovery')
    os.makedirs(disc_dir, exist_ok=True)
    cmd = [rocprofv3_path, '--kernel-trace', '--output-format', 'csv',
           '-o', 'kernel_discovery', '-d', disc_dir, '--']
    cmd.extend(runner_argv)
    stdout_log = os.path.join(out_dir, 'kernel_discovery_stdout.log')
    stderr_log = os.path.join(out_dir, 'kernel_discovery_stderr.log')
    print(f'\n{"============================================================"}')
    print(f'Running: {" ".join(cmd)}')
    print(f'Logs:    {stdout_log}')
    print(f'         {stderr_log}')
    print(f'Timeout: {timeout}s')
    print(f'{"============================================================"}')
    with open(stdout_log, 'w') as f_out, open(stderr_log, 'w') as f_err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            f_out.write(stdout or '')
            f_err.write(stderr or '')
            print(f'  WARNING: kernel-trace timed out after {timeout}s')
            return None
        f_out.write(stdout or '')
        f_err.write(stderr or '')
    if stdout:
        print(f'[kernel_discovery stdout]:\n{stdout}')
    if stderr:
        print(f'[kernel_discovery stderr]:\n{stderr}')
    if proc.returncode != 0:
        print(f'  WARNING: kernel-trace failed (exit {proc.returncode})')
        return None
    trace_csv = os.path.join(disc_dir, 'kernel_discovery_kernel_trace.csv')
    if not os.path.exists(trace_csv):
        candidates = list(Path(disc_dir).rglob('*kernel_trace.csv'))
        if candidates:
            trace_csv = str(candidates[0])
        else:
            print(f'  WARNING: kernel_trace CSV not found in {disc_dir}')
            return None
    kernel_info = {}
    with open(trace_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kn = row.get('Kernel_Name', '')
            start = int(row.get('Start_Timestamp', 0))
            end = int(row.get('End_Timestamp', 0))
            if kn not in kernel_info:
                kernel_info[kn] = {'count': 0, 'total_ns': 0}
            kernel_info[kn]['count'] += 1
            kernel_info[kn]['total_ns'] += (end - start)
    print(f'  Discovered {len(kernel_info)} unique kernels:')
    for kn, info in sorted(kernel_info.items(), key=lambda x: -x[1]['total_ns']):
        dur_ms = info['total_ns'] / 1e6
        print(f'    {info["count"]}x ({dur_ms:.2f} ms total)  {kn[:120]}...')
    expected_count = num_warmup_iters + 1

    def is_infra(name):
        return any(pat in name for pat in _INFRA_KERNEL_PATTERNS)

    candidates = {kn: info for kn, info in kernel_info.items() if not is_infra(kn)}
    if not candidates:
        print('  WARNING: all kernels matched infrastructure patterns')
        return None
    exact_match = {kn: info for kn, info in candidates.items() if info['count'] == expected_count}
    pool = exact_match if exact_match else candidates
    best_kernel = max(pool, key=lambda kn: pool[kn]['total_ns'])
    best_info = pool[best_kernel]
    print(f'\n  Selected kernel ({best_info["count"]} dispatches, {best_info["total_ns"] / 1e6:.2f} ms):')
    print(f'    {best_kernel[:200]}')
    return best_kernel


def parse_kernel_runtime_ns(trace_csv, kernel_name_filter=None):
    """Return ``(last_dispatch_ns, mean_dispatch_ns, num_dispatches)`` for matching kernels.

    Reads a rocprofv3 ``--kernel-trace`` CSV (columns include ``Kernel_Name``,
    ``Start_Timestamp``, ``End_Timestamp``) and computes per-dispatch durations
    in nanoseconds. The "last" dispatch corresponds to the measured iteration
    used for counter readings (warmups precede it). Returns ``(None, None, 0)``
    if no matching dispatches are found or the file is unreadable.
    """
    if not trace_csv or not os.path.exists(trace_csv):
        return (None, None, 0)
    durations_by_dispatch = {}
    try:
        with open(trace_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kn = row.get('Kernel_Name', '')
                if kernel_name_filter and kernel_name_filter != kn and kernel_name_filter not in kn:
                    continue
                try:
                    start = int(row.get('Start_Timestamp', 0))
                    end = int(row.get('End_Timestamp', 0))
                    disp_id = int(row.get('Dispatch_Id', 0))
                except (TypeError, ValueError):
                    continue
                durations_by_dispatch[disp_id] = end - start
    except (OSError, csv.Error):
        return (None, None, 0)
    if not durations_by_dispatch:
        return (None, None, 0)
    n = len(durations_by_dispatch)
    last_id = max(durations_by_dispatch.keys())
    last_ns = durations_by_dispatch[last_id]
    mean_ns = sum(durations_by_dispatch.values()) / n
    return (last_ns, mean_ns, n)


def parse_counter_csv(csv_path, kernel_name_filter=None):
    dispatch_counters = {}
    matched_rows = 0
    total_rows = 0
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            kernel_col = None
            for col in ('Kernel_Name', 'kernel_name', 'KernelName'):
                if col in row:
                    kernel_col = col
                    break
            if kernel_name_filter and kernel_col:
                kn = row[kernel_col]
                if kernel_name_filter != kn and kernel_name_filter not in kn:
                    continue
            matched_rows += 1
            dispatch_id = row.get('Dispatch_Id', row.get('dispatch_id', '0'))
            counter_name_col = None
            counter_value_col = None
            for col in ('Counter_Name', 'counter_name'):
                if col in row:
                    counter_name_col = col
                    break
            for col in ('Counter_Value', 'counter_value'):
                if col in row:
                    counter_value_col = col
                    break
            if counter_name_col and counter_value_col:
                cname = row[counter_name_col]
                # Keep counters with a known display mapping, plus the raw TCC
                # read-request counters used for the corrected read-byte formula.
                if cname in COUNTER_DISPLAY or cname in _MEM_READ_RAW_COUNTERS:
                    if dispatch_id not in dispatch_counters:
                        dispatch_counters[dispatch_id] = {}
                    try:
                        dispatch_counters[dispatch_id][cname] = float(row[counter_value_col])
                    except (ValueError, TypeError):
                        continue
    if dispatch_counters:
        last_dispatch_id = max(dispatch_counters.keys(), key=int)
        counters = dispatch_counters[last_dispatch_id]
    else:
        counters = {}
    print(f"  Parsed {csv_path}: {total_rows} rows, {matched_rows} matched filter '{kernel_name_filter}'")
    print(f'  Found {len(dispatch_counters)} matching dispatches, using last (dispatch '
          f'{last_dispatch_id if dispatch_counters else "N/A"})')
    print(f'  Counters: {counters}')
    return counters


def format_number(val, unit):
    if val is None:
        return 'N/A'
    if unit == 'bytes':
        if val >= 1e9:
            return f'{val / 1e9:.2f} GB'
        if val >= 1e6:
            return f'{val / 1e6:.2f} MB'
        if val >= 1000:
            return f'{val / 1000:.2f} KB'
        return f'{val:.0f} B'
    if val >= 1e12:
        return f'{val / 1e12:.2f} T'
    if val >= 1e9:
        return f'{val / 1e9:.2f} G'
    if val >= 1e6:
        return f'{val / 1e6:.2f} M'
    if val >= 1000:
        return f'{val / 1000:.2f} K'
    return f'{val:.0f}'


def generate_report(op_name, arch, dims_str, predicted_flops, predicted_bytes,
                    predicted_precision, ops_counters, mem_counters, output_dir,
                    kernel_filter=None, kernel_runtime_ns=None,
                    kernel_runtime_mean_ns=None, kernel_runtime_dispatches=0):
    """Generate the validation report.

    Returns a dict with flops_ratio, bytes_ratio, precision_result, and all
    individual scaled counter values for the rich CSV/text report.
    """
    lines = []
    lines.append('======================================================================')
    lines.append('PERF MODEL VALIDATION REPORT')
    lines.append('======================================================================')
    lines.append(f'Op:   {op_name}')
    lines.append(f'Arch: {arch}')
    lines.append(dims_str)
    if kernel_filter:
        lines.append(f'Kernel: {kernel_filter[:120]}')
    lines.append('')
    lines.append('--- Perf Model Predictions ---')
    lines.append(f'  FLOPs:               {format_number(predicted_flops, "ops")}')
    lines.append(f'  Bytes (roofline min): {format_number(predicted_bytes, "bytes")}')
    lines.append(f'  Compute precision:   {predicted_precision or "unknown"}')
    lines.append('')
    lines.append('--- Measured Ops (rocprofv3 --pmc) ---')
    for counter_name, raw_val in ops_counters.items():
        multiplier, label, unit = COUNTER_DISPLAY[counter_name]
        total = raw_val * multiplier
        lines.append(f'  {label:25s}: {format_number(total, unit):>12s}  (raw counter: {raw_val:.0f})')
    lines.append('')
    lines.append('--- Measured Memory (rocprofv3 --pmc) ---')
    # *** CORRECTED read-byte reconstruction from the raw TCC request mix. ***
    rd_total = mem_counters.get('TCC_EA0_RDREQ_sum', 0)
    rd32 = mem_counters.get('TCC_EA0_RDREQ_32B_sum', 0)
    rd64 = mem_counters.get('TCC_EA0_RDREQ_64B_sum', 0)
    measured_read_bytes = 32 * rd32 + 64 * rd64 + 128 * (rd_total - rd32 - rd64)
    write_raw = mem_counters.get('WRITE_SIZE', 0)
    measured_write_bytes = write_raw * 1024
    lines.append(f'  {"HBM Read":25s}: {format_number(measured_read_bytes, "bytes"):>12s}  (raw counter: {rd_total:.0f})')
    lines.append(f'  {"HBM Write":25s}: {format_number(measured_write_bytes, "bytes"):>12s}  (raw counter: {write_raw:.0f})')
    measured_total_bytes = measured_read_bytes + measured_write_bytes
    lines.append(f'  {"Total HBM Traffic":25s}: {format_number(measured_total_bytes, "bytes"):>12s}')
    lines.append('')
    lines.append('--- Comparison ---')
    lines.append(f'  {"Metric":30s} {"Predicted":>14s} {"Measured":>14s} {"Ratio":>10s}')
    lines.append(f'  {"------------------------------"} {"--------------"} {"--------------"} {"----------"}')
    flops_ratio = None
    i8_ops = ops_counters.get('SQ_INSTS_VALU_MFMA_MOPS_I8', 0)
    f8_ops = ops_counters.get('SQ_INSTS_VALU_MFMA_MOPS_F8', 0)
    f4_ops = ops_counters.get('SQ_INSTS_VALU_MFMA_MOPS_F6F4', 0)
    t16_ops = ops_counters.get('TOTAL_16_OPS', 0)
    t32_ops = ops_counters.get('TOTAL_32_OPS', 0)

    def _actual_ops(counter_name, raw_val):
        return raw_val * COUNTER_DISPLAY.get(counter_name, (1, '', ''))[0]

    best_counter = max(
        [
            ('SQ_INSTS_VALU_MFMA_MOPS_F8', f8_ops),
            ('SQ_INSTS_VALU_MFMA_MOPS_I8', i8_ops),
            ('SQ_INSTS_VALU_MFMA_MOPS_F6F4', f4_ops),
            ('TOTAL_16_OPS', t16_ops),
            ('TOTAL_32_OPS', t32_ops),
        ],
        key=lambda x: _actual_ops(x[0], x[1]),
    )
    relevant_ops_counter = best_counter[0] if best_counter[1] > 0 else None
    if relevant_ops_counter and predicted_flops and predicted_flops > 0:
        multiplier, label, unit = COUNTER_DISPLAY[relevant_ops_counter]
        measured_ops = ops_counters[relevant_ops_counter] * multiplier
        flops_ratio = measured_ops / predicted_flops
        lines.append(
            f'  {label:30s} {format_number(predicted_flops, "ops"):>14s} '
            f'{format_number(measured_ops, "ops"):>14s} {flops_ratio:>10.4f}')
    bytes_ratio = None
    if predicted_bytes and predicted_bytes > 0 and measured_total_bytes > 0:
        bytes_ratio = measured_total_bytes / predicted_bytes
        lines.append(
            f'  {"Total HBM Bytes":30s} {format_number(predicted_bytes, "bytes"):>14s} '
            f'{format_number(measured_total_bytes, "bytes"):>14s} {bytes_ratio:>10.4f}')
    precision_result = 'N/A'
    lines.append('')
    lines.append('--- Compute Precision Check ---')
    if predicted_precision and predicted_precision in PRECISION_TO_COUNTERS:
        expected_counters = PRECISION_TO_COUNTERS[predicted_precision]
        expected_label = ', '.join(expected_counters)
        mfma_candidates = {
            'SQ_INSTS_VALU_MFMA_MOPS_F8',
            'SQ_INSTS_VALU_MFMA_MOPS_I8',
            'SQ_INSTS_VALU_MFMA_MOPS_F6F4',
        }
        all_ops = dict(ops_counters)
        dominant_counter = None
        dominant_val = 0
        for cname, cval in all_ops.items():
            if cname in mfma_candidates and cval > dominant_val:
                dominant_counter = cname
                dominant_val = cval
        if dominant_counter is None and all_ops:
            for cname, cval in all_ops.items():
                if cval > dominant_val:
                    dominant_counter = cname
                    dominant_val = cval
        lines.append(f'  Predicted precision:  {predicted_precision}')
        lines.append(f'  Expected counter(s):  {expected_label}')
        if dominant_counter:
            mult, label, _ = COUNTER_DISPLAY.get(dominant_counter, (1, dominant_counter, 'ops'))
            lines.append(f'  Dominant counter:     {dominant_counter}  ({format_number(dominant_val * mult, "ops")})')
            if dominant_counter in expected_counters:
                precision_result = 'PASS'
                lines.append('  Result:               PASS - dominant MFMA type matches predicted precision')
            else:
                precision_result = 'MISMATCH'
                lines.append(f'  Result:               MISMATCH - expected {expected_label} but dominant is {dominant_counter}')
                lines.append('  All ops counters:')
                for cname, cval in sorted(all_ops.items()):
                    if cval > 0:
                        m, l, _ = COUNTER_DISPLAY.get(cname, (1, cname, 'ops'))
                        lines.append(f'    {l:25s}: {format_number(cval * m, "ops"):>12s}')
        else:
            precision_result = 'NO DATA'
            lines.append('  Result:               NO DATA - no ops counters collected')
    elif predicted_precision:
        lines.append(f'  Predicted precision:  {predicted_precision}')
        lines.append(f"  Result:               SKIPPED - no counter mapping for '{predicted_precision}'")
    else:
        lines.append('  Result:               SKIPPED - perf model returned no compute precision')
    measured_throughput_ops = None
    measured_throughput_label = None
    if relevant_ops_counter:
        mult = COUNTER_DISPLAY[relevant_ops_counter][0]
        measured_throughput_label = COUNTER_DISPLAY[relevant_ops_counter][1]
        measured_throughput_ops = ops_counters[relevant_ops_counter] * mult
    pred_tflops_s = None
    meas_tflops_s = None
    pred_tb_s = None
    meas_tb_s = None
    runtime_us = None
    runtime_mean_us = None
    if kernel_runtime_ns and kernel_runtime_ns > 0:
        runtime_s = kernel_runtime_ns / 1e9
        runtime_us = kernel_runtime_ns / 1000
        if kernel_runtime_mean_ns:
            runtime_mean_us = kernel_runtime_mean_ns / 1000
        if predicted_flops:
            pred_tflops_s = predicted_flops / 1e12 / runtime_s
        if predicted_bytes:
            pred_tb_s = predicted_bytes / 1e12 / runtime_s
        if measured_throughput_ops:
            meas_tflops_s = measured_throughput_ops / 1e12 / runtime_s
        if measured_total_bytes:
            meas_tb_s = measured_total_bytes / 1e12 / runtime_s
        lines.append('')
        lines.append('--- Achieved Throughput (last-dispatch runtime from kernel-trace) ---')
        runtime_line = f'  Kernel runtime (last dispatch): {runtime_us:9.3f} us'
        if runtime_mean_us is not None and kernel_runtime_dispatches > 1:
            runtime_line += f'   (mean over {kernel_runtime_dispatches} dispatches: {runtime_mean_us:.3f} us)'
        lines.append(runtime_line)
        lines.append(f'  {"Metric":32s} {"Predicted":>14s} {"Measured":>14s}')
        lines.append(f'  {"--------------------------------"} {"--------------"} {"--------------"}')
        pf_s = f'{pred_tflops_s:>10.3f} TF/s' if pred_tflops_s is not None else '       N/A'
        mf_s = f'{meas_tflops_s:>10.3f} TF/s' if meas_tflops_s is not None else '       N/A'
        meas_label = f'Compute (rocprof {measured_throughput_label})' if measured_throughput_label else 'Compute (rocprof)'
        lines.append(f'  {meas_label:32s} {pf_s:>14s} {mf_s:>14s}')
        pb_s = f'{pred_tb_s:>10.3f} TB/s' if pred_tb_s is not None else '       N/A'
        mb_s = f'{meas_tb_s:>10.3f} TB/s' if meas_tb_s is not None else '       N/A'
        lines.append(f'  {"HBM (rocprof FETCH+WRITE)":32s} {pb_s:>14s} {mb_s:>14s}')
        if pred_tflops_s and pred_tb_s and pred_tb_s > 0:
            lines.append(f'  Arithmetic intensity (predicted): {predicted_flops / max(predicted_bytes, 1):.2f} FLOP/byte')
        if meas_tflops_s and meas_tb_s and meas_tb_s > 0:
            ai_meas = measured_throughput_ops / max(measured_total_bytes, 1) if measured_throughput_ops else 0
            lines.append(f'  Arithmetic intensity (measured):  {ai_meas:.2f} FLOP/byte')
    else:
        lines.append('')
        lines.append('--- Achieved Throughput ---')
        lines.append('  Kernel runtime not available (no kernel-trace CSV); throughput skipped.')
    lines.append('')
    lines.append('Notes:')
    lines.append('  - Ratio = Measured / Predicted.  1.0 = perfect match.')
    lines.append('  - Ops ratio < 1.0 may indicate the kernel uses a different MFMA dtype')
    lines.append('    than assumed. Check other ops counters above.')
    lines.append('  - Bytes ratio > 1.0 is expected due to cache-line granularity,')
    lines.append('    L2 replays, scale tensor traffic, and alignment padding.')
    lines.append('======================================================================')
    report_text = '\n'.join(lines)
    print(report_text)
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f'\nReport saved to: {report_path}')
    mfma_f8 = ops_counters.get('SQ_INSTS_VALU_MFMA_MOPS_F8', 0) + ops_counters.get('SQ_INSTS_VALU_MFMA_MOPS_I8', 0)
    return {
        'flops_ratio': flops_ratio,
        'bytes_ratio': bytes_ratio,
        'precision_result': precision_result,
        'rocprof_f4_ops': f4_ops * COUNTER_DISPLAY.get('SQ_INSTS_VALU_MFMA_MOPS_F6F4', (1,))[0],
        'rocprof_f8_ops': mfma_f8 * 512,
        'rocprof_total_16': t16_ops * COUNTER_DISPLAY.get('TOTAL_16_OPS', (1,))[0],
        'rocprof_total_32': t32_ops * COUNTER_DISPLAY.get('TOTAL_32_OPS', (1,))[0],
        'rocprof_fetch_bytes': measured_read_bytes,
        'rocprof_write_bytes': measured_write_bytes,
        'kernel_runtime_us': runtime_us,
        'kernel_runtime_mean_us': runtime_mean_us,
        'kernel_runtime_dispatches': kernel_runtime_dispatches,
        'predicted_tflops_s': pred_tflops_s,
        'measured_tflops_s': meas_tflops_s,
        'predicted_tb_s': pred_tb_s,
        'measured_tb_s': meas_tb_s,
        'throughput_ops_counter': measured_throughput_label,
    }


def run_single_op_validation(op_name, args, output_dir):
    """Run full validation for a single op. Returns result dict."""
    reg = OP_REGISTRY[op_name]
    arch_cfg = ARCH_COUNTER_CONFIGS[args.arch]
    effective = argparse.Namespace(**vars(args))
    effective.op = op_name
    for key, default_val in reg['defaults'].items():
        if getattr(effective, key, None) is None:
            setattr(effective, key, default_val)
    parts = []
    for key in ('M', 'N', 'K', 'E', 'topk', 'group_size', 'seq_len', 'num_heads_q', 'num_heads_kv', 'head_dim'):
        val = getattr(effective, key, None)
        if val is not None and key in reg['required_args']:
            parts.append(f'{key}={val}')
    ann_val = getattr(effective, 'annotation', None)
    if ann_val:
        parts.append(f'annotation={ann_val[:60]}{"..." if len(ann_val) > 60 else ""}')
    dims_str = ', '.join(parts)
    print(f'\n{"######################################################################"}')
    print(f'# Validating: {op_name} ({reg["description"]})')
    print(f'# Dims: {dims_str}')
    print(f'{"######################################################################"}')
    if reg.get('perf_model_only'):
        print('[1/1] Running TraceLens perf model (perf-model-only mode) ...')
        predicted_flops, predicted_bytes, predicted_precision = reg['model_fn'](effective)
        print(f'       Predicted FLOPs: {format_number(predicted_flops, "ops")}')
        print(f'       Predicted Bytes: {format_number(predicted_bytes, "bytes")}')
        print(f'       Compute precision: {predicted_precision or "unknown"}')
        lines = [
            '======================================================================',
            'PERF MODEL VALIDATION REPORT (perf-model-only)',
            '======================================================================',
            f'Op:   {op_name}',
            f'Arch: {args.arch}',
            dims_str,
            '',
            '--- Perf Model Predictions ---',
            f'  FLOPs:               {format_number(predicted_flops, "ops")}',
            f'  Bytes (roofline min): {format_number(predicted_bytes, "bytes")}',
            f'  Compute precision:   {predicted_precision or "unknown"}',
            '',
            'Note: This op requires model-level context (ForwardContext, KV cache)',
            'and cannot be tested standalone with rocprofv3. Only the perf model',
            'predictions are shown.',
            '======================================================================',
        ]
        report_text = '\n'.join(lines)
        print(report_text)
        report_path = os.path.join(output_dir, 'validation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f'\nReport saved to: {report_path}')
        return {
            'op': op_name,
            'category': reg['category'],
            'dims': dims_str,
            'predicted_flops': predicted_flops,
            'predicted_bytes': predicted_bytes,
            'predicted_precision': predicted_precision or 'N/A',
            'rocprof_f4_ops': 0,
            'rocprof_f8_ops': 0,
            'rocprof_total_16': 0,
            'rocprof_total_32': 0,
            'rocprof_fetch_bytes': 0,
            'rocprof_write_bytes': 0,
            'kernel_name': '',
            'flops_ratio': None,
            'bytes_ratio': None,
            'precision': predicted_precision or 'N/A',
            'status': 'MODEL_ONLY',
        }
    step = 1
    total_steps = 8
    print(f'[{step}/{total_steps}] Building test runner command for {op_name} ...')
    _populate_varlen_attention_args(op_name, effective)
    runner_argv, num_warmup_iters = _build_runner_argv(op_name, effective)
    print(f'       Runner cmd: {" ".join(runner_argv)}')
    step += 1
    if args.kernel_filter:
        kernel_filter = args.kernel_filter
        print(f'[{step}/{total_steps}] Using manual kernel filter: {kernel_filter}')
    else:
        print(f'[{step}/{total_steps}] Discovering target kernel via rocprofv3 --kernel-trace ...')
        kernel_filter = discover_kernel_name(
            args.rocprofv3_path, runner_argv, num_warmup_iters, output_dir, timeout=args.timeout)
        if kernel_filter:
            print(f'       Auto-discovered kernel: {kernel_filter[:120]}...')
        else:
            print('       WARNING: auto-discovery failed, no kernel filter applied')
    step += 1
    print(f'[{step}/{total_steps}] Running TraceLens perf model ...')
    predicted_flops, predicted_bytes, predicted_precision = reg['model_fn'](effective)
    print(f'       Predicted FLOPs: {format_number(predicted_flops, "ops")}')
    print(f'       Predicted Bytes: {format_number(predicted_bytes, "bytes")}')
    print(f'       Compute precision: {predicted_precision or "unknown"}')
    step += 1
    print(f'[{step}/{total_steps}] Running rocprofv3 (MFMA ops counters) ...')
    ops_csv = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_counters'], runner_argv,
                            'ops_counters', output_dir, timeout=args.timeout)
    ops_counters = parse_counter_csv(ops_csv, kernel_filter)
    step += 1
    if not args.skip_derived and arch_cfg.get('ops_derived_16'):
        print(f'[{step}/{total_steps}] Running rocprofv3 (TOTAL_16_OPS) ...')
        try:
            csv_16 = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_derived_16'], runner_argv,
                                   'ops_derived_16', output_dir, timeout=args.timeout)
            ops_counters.update(parse_counter_csv(csv_16, kernel_filter))
        except (RuntimeError, FileNotFoundError) as e:
            print(f'  WARNING: TOTAL_16_OPS collection failed: {e}')
    else:
        print(f'[{step}/{total_steps}] Skipping TOTAL_16_OPS')
    step += 1
    if not args.skip_derived and arch_cfg.get('ops_derived_32'):
        print(f'[{step}/{total_steps}] Running rocprofv3 (TOTAL_32_OPS) ...')
        try:
            csv_32 = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_derived_32'], runner_argv,
                                   'ops_derived_32', output_dir, timeout=args.timeout)
            ops_counters.update(parse_counter_csv(csv_32, kernel_filter))
        except (RuntimeError, FileNotFoundError) as e:
            print(f'  WARNING: TOTAL_32_OPS collection failed: {e}')
    else:
        print(f'[{step}/{total_steps}] Skipping TOTAL_32_OPS')
    step += 1
    mem_counters = {}
    print(f'[{step}/{total_steps}] Running rocprofv3 (FETCH_SIZE) ...')
    try:
        fetch_csv = run_rocprofv3(args.rocprofv3_path, arch_cfg['mem_read_counters'], runner_argv,
                                  'mem_read_counters', output_dir, timeout=args.timeout)
        mem_counters.update(parse_counter_csv(fetch_csv, kernel_filter))
    except (RuntimeError, FileNotFoundError) as e:
        print(f'  WARNING: FETCH_SIZE collection failed: {e}')
    step += 1
    print(f'[{step}/{total_steps}] Running rocprofv3 (WRITE_SIZE) ...')
    try:
        write_csv = run_rocprofv3(args.rocprofv3_path, arch_cfg['mem_write_counters'], runner_argv,
                                  'mem_write_counters', output_dir, timeout=args.timeout)
        mem_counters.update(parse_counter_csv(write_csv, kernel_filter))
    except (RuntimeError, FileNotFoundError) as e:
        print(f'  WARNING: WRITE_SIZE collection failed: {e}')
    kernel_runtime_ns = None
    kernel_runtime_mean_ns = None
    kernel_runtime_dispatches = 0
    trace_csv_path = os.path.join(output_dir, 'kernel_discovery', 'kernel_discovery_kernel_trace.csv')
    if not os.path.exists(trace_csv_path):
        candidates = list(Path(output_dir).rglob('*kernel_trace.csv'))
        if candidates:
            trace_csv_path = str(candidates[0])
    if os.path.exists(trace_csv_path):
        kernel_runtime_ns, kernel_runtime_mean_ns, kernel_runtime_dispatches = parse_kernel_runtime_ns(
            trace_csv_path, kernel_filter)
    report_data = generate_report(
        op_name, args.arch, dims_str, predicted_flops, predicted_bytes,
        predicted_precision, ops_counters, mem_counters, output_dir,
        kernel_filter=kernel_filter, kernel_runtime_ns=kernel_runtime_ns,
        kernel_runtime_mean_ns=kernel_runtime_mean_ns,
        kernel_runtime_dispatches=kernel_runtime_dispatches)
    return {
        'op': op_name,
        'category': reg['category'],
        'dims': dims_str,
        'predicted_flops': predicted_flops,
        'predicted_bytes': predicted_bytes,
        'predicted_precision': predicted_precision or 'N/A',
        'rocprof_f4_ops': report_data['rocprof_f4_ops'],
        'rocprof_f8_ops': report_data['rocprof_f8_ops'],
        'rocprof_total_16': report_data['rocprof_total_16'],
        'rocprof_total_32': report_data['rocprof_total_32'],
        'rocprof_fetch_bytes': report_data['rocprof_fetch_bytes'],
        'rocprof_write_bytes': report_data['rocprof_write_bytes'],
        'kernel_name': kernel_filter or '',
        'kernel_runtime_us': report_data.get('kernel_runtime_us'),
        'kernel_runtime_mean_us': report_data.get('kernel_runtime_mean_us'),
        'kernel_runtime_dispatches': report_data.get('kernel_runtime_dispatches', 0),
        'predicted_tflops_s': report_data.get('predicted_tflops_s'),
        'measured_tflops_s': report_data.get('measured_tflops_s'),
        'predicted_tb_s': report_data.get('predicted_tb_s'),
        'measured_tb_s': report_data.get('measured_tb_s'),
        'throughput_ops_counter': report_data.get('throughput_ops_counter'),
        'flops_ratio': report_data['flops_ratio'],
        'bytes_ratio': report_data['bytes_ratio'],
        'precision': report_data['precision_result'],
        'status': 'OK',
    }


def _generate_comment(row):
    """Auto-generate a discrepancy comment from rule-based analysis."""
    if row.get('status', '').startswith('FAIL'):
        return f'Validation failed: {row["status"]}'
    if row.get('status') == 'MODEL_ONLY':
        return 'Perf-model-only; no GPU measurement available'
    parts = []
    fr = row.get('flops_ratio')
    br = row.get('bytes_ratio')
    if fr is not None:
        if 0.95 <= fr <= 1.05:
            parts.append('FLOPs: exact match')
        elif 0.8 <= fr < 0.95:
            parts.append(f'FLOPs {fr:.2f}x: minor under-count, likely rounding or tile residuals')
        elif fr < 0.5:
            parts.append(f'FLOPs {fr:.2f}x: kernel may use different MFMA dtype than predicted')
        elif fr < 0.8:
            parts.append(f'FLOPs {fr:.2f}x: possible mixed-precision or partial tile usage')
        elif 1.05 < fr <= 1.5:
            parts.append(f'FLOPs {fr:.2f}x: minor over-count from tile padding or epilogue')
        elif fr > 2:
            parts.append(f'FLOPs {fr:.1f}x: likely tile padding, SwiGLU fusion, or extra VALU ops')
        else:
            parts.append(f'FLOPs {fr:.2f}x: moderate over-count, check tile dimensions')
    if br is not None:
        if 0.9 <= br <= 1.2:
            parts.append('Bytes: good match')
        elif br < 0.9:
            parts.append(f'Bytes {br:.2f}x: model may over-estimate (caching?)')
        elif 1.2 < br <= 2:
            parts.append(f'Bytes {br:.2f}x: L2 replay or scale tensor overhead')
        elif br > 2:
            parts.append(f'Bytes {br:.1f}x: significant extra traffic (L2 replays, scales, workspace)')
    prec = row.get('precision', '')
    if prec == 'MISMATCH':
        parts.append('Precision mismatch: dominant HW counter differs from predicted dtype')
    elif prec == 'NO DATA':
        parts.append('No ops counters collected for precision check')
    if parts:
        return '; '.join(parts)
    return 'OK'


CSV_COLUMNS = [
    'op', 'trace_name', 'perf_model_class', 'source_file', 'source_line',
    'predicted_precision', 'predicted_flops', 'predicted_bytes',
    'csv_reported_flops', 'csv_reported_bytes', 'rocprof_f4_ops', 'rocprof_f8_ops',
    'rocprof_total_16', 'rocprof_total_32', 'rocprof_fetch_bytes', 'rocprof_write_bytes',
    'kernel_name', 'kernel_runtime_us', 'kernel_runtime_mean_us', 'kernel_runtime_dispatches',
    'throughput_ops_counter', 'predicted_tflops_s', 'measured_tflops_s', 'predicted_tb_s',
    'measured_tb_s', 'flops_ratio', 'bytes_ratio', 'comment',
]


def _enrich_results(results):
    """Add perf-model source metadata and auto-generated comments to results."""
    for r in results:
        cls_name, src_file, src_line = _resolve_perf_model_source(r['op'])
        r['perf_model_class'] = cls_name
        r['source_file'] = src_file
        r['source_line'] = src_line
        r['comment'] = _generate_comment(r)
    return results


def write_results_csv(results, output_path):
    """Write the enriched results table as a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {}
            for col in CSV_COLUMNS:
                val = r.get(col, '')
                if val is None:
                    val = ''
                elif isinstance(val, float):
                    if col in ('flops_ratio', 'bytes_ratio'):
                        val = f'{val:.4f}'
                    elif col in ('predicted_tflops_s', 'measured_tflops_s', 'predicted_tb_s', 'measured_tb_s'):
                        val = f'{val:.4f}'
                    elif col in ('kernel_runtime_us', 'kernel_runtime_mean_us'):
                        val = f'{val:.3f}'
                    else:
                        val = f'{val:.0f}'
                row[col] = val
            writer.writerow(row)
    print(f'\nCSV report saved to: {output_path}')
    return None


def print_batch_summary(results):
    """Print an enriched summary table for batch validation runs."""
    print('\n====================================================================================================================================================================================')
    print('VALIDATION REPORT TABLE')
    print('====================================================================================================================================================================================')
    hdr = (f'  {"Op":<32s} {"Class":<32s} {"Source":<42s} {"Prec":>5s} {"Pred FLOPs":>12s} '
           f'{"Pred Bytes":>12s} {"F4 Ops":>12s} {"F8 Ops":>12s} {"16-bit":>12s} {"32-bit":>12s} '
           f'{"Fetch":>12s} {"Write":>12s} {"FRatio":>8s} {"BRatio":>8s} {"Status":>10s}')
    print(hdr)
    sep = (f'  {"--------------------------------"} {"--------------------------------"} '
           f'{"------------------------------------------"} {"-----"} {"------------"} {"------------"} '
           f'{"------------"} {"------------"} {"------------"} {"------------"} {"------------"} '
           f'{"------------"} {"--------"} {"--------"} {"----------"}')
    print(sep)
    for r in results:
        fr_s = f'{r["flops_ratio"]:.4f}' if r.get('flops_ratio') else '---'
        br_s = f'{r["bytes_ratio"]:.4f}' if r.get('bytes_ratio') else '---'
        pf = format_number(r.get('predicted_flops'), 'ops') if r.get('predicted_flops') else '---'
        pb = format_number(r.get('predicted_bytes'), 'bytes') if r.get('predicted_bytes') else '---'
        f4 = format_number(r.get('rocprof_f4_ops', 0), 'ops') if r.get('rocprof_f4_ops') else '---'
        f8 = format_number(r.get('rocprof_f8_ops', 0), 'ops') if r.get('rocprof_f8_ops') else '---'
        t16 = format_number(r.get('rocprof_total_16', 0), 'ops') if r.get('rocprof_total_16') else '---'
        t32 = format_number(r.get('rocprof_total_32', 0), 'ops') if r.get('rocprof_total_32') else '---'
        fetch = format_number(r.get('rocprof_fetch_bytes', 0), 'bytes') if r.get('rocprof_fetch_bytes') else '---'
        write = format_number(r.get('rocprof_write_bytes', 0), 'bytes') if r.get('rocprof_write_bytes') else '---'
        prec = str(r.get('predicted_precision', '---'))[:5]
        src = f'{r.get("source_file", "?")}:{r.get("source_line", "?")}'
        cls_name = str(r.get('perf_model_class', '?'))[:32]
        status = str(r.get('status', '---'))[:10]
        op_name = r['op'][:32]
        line = (f'  {op_name:<32s} {cls_name:<32s} {src:<42s} {prec:>5s} {pf:>12s} {pb:>12s} '
                f'{f4:>12s} {f8:>12s} {t16:>12s} {t32:>12s} {fetch:>12s} {write:>12s} '
                f'{fr_s:>8s} {br_s:>8s} {status:>10s}')
        print(line)
    print('====================================================================================================================================================================================')
    print('\nDISCREPANCY COMMENTS:')
    print('----------------------------------------------------------------------------------------------------')
    for r in results:
        comment = r.get('comment', '')
        if comment and comment != 'OK':
            print(f'  {r["op"]:<35s}  {comment}')
    print('----------------------------------------------------------------------------------------------------')
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    model_only_count = sum(1 for r in results if r['status'] == 'MODEL_ONLY')
    fail_count = len(results) - ok_count - model_only_count
    parts = [f'{ok_count} passed']
    if model_only_count:
        parts.append(f'{model_only_count} model-only')
    if fail_count:
        parts.append(f'{fail_count} failed')
    print(f'\n  {", ".join(parts)} out of {len(results)} ops')
    return None


def _parse_csv_field(value_str):
    """Parse a CSV field that contains a Python literal (tuple/list/string).

    Returns the parsed Python object, or the original string on failure.
    """
    if not value_str or value_str.strip() == '':
        return None
    try:
        parsed = ast.literal_eval(value_str)
        return parsed
    except (ValueError, SyntaxError):
        return value_str


def _extract_dims_for_existing_harness(registry_key, input_dims, input_types, concrete_inputs, attn_params=None):
    """Extract M/N/K/etc. from Input Dims for ops that use existing harness generators."""
    ns = argparse.Namespace()
    reg = OP_REGISTRY[registry_key]
    cat = reg['category']
    if cat in ('gemm', 'vllm_gemm'):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        ns.K = input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 8192
        if len(input_dims) > 1 and input_dims[1]:
            ns.N = input_dims[1][0]
        else:
            ns.N = 4096
        return ns
    if cat == 'moe':
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 512
        ns.K = input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 7168
        if len(input_dims) > 1 and input_dims[1] and len(input_dims[1]) >= 3:
            ns.E = input_dims[1][0]
            ns.N = input_dims[1][2]
        else:
            ns.E = 8
            ns.N = 2048
        if concrete_inputs:
            for ci in concrete_inputs:
                if isinstance(ci, (int, float)) and ci not in (0, 1) and ci < 100:
                    ns.topk = int(ci)
                    break
            else:
                ns.topk = 2
            return ns
        ns.topk = 2
        return ns
    if cat in ('activation',):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        if len(input_dims) > 0 and len(input_dims[0]) > 1:
            ns.N = input_dims[0][1]
            return ns
        ns.N = 7168
        return ns
    if cat in ('norm', 'vllm_norm'):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        ns.N = input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 7168
        ns.group_size = 128
        return ns
    if cat in ('quant', 'vllm_quant'):
        if len(input_dims) > 0 and input_dims[0]:
            if len(input_dims[0]) == 3:
                ns.M = input_dims[0][0] * input_dims[0][1]
                ns.N = input_dims[0][2]
            else:
                ns.M = input_dims[0][0]
                ns.N = input_dims[0][1] if len(input_dims[0]) > 1 else 7168
        else:
            ns.M, ns.N = (2048, 7168)
        ns.group_size = 128
        return ns
    if cat in ('attention', 'vllm_attention'):
        if attn_params:
            ns.seq_len = int(attn_params.get('N_Q', 2048))
            ns.num_heads_q = int(attn_params.get('H_Q', 32))
            ns.num_heads_kv = int(attn_params.get('H_KV', 8))
            ns.head_dim = int(attn_params.get('d_h_qk', 128))
            c_sq = attn_params.get('c_sq')
            c_sqsq = attn_params.get('c_sqsq')
            if c_sq is not None and str(c_sq) not in ('nan', ''):
                ns._varlen_c_sq = int(float(c_sq))
            if c_sqsq is not None and str(c_sqsq) not in ('nan', ''):
                ns._varlen_c_sqsq = int(float(c_sqsq))
            return ns
        if len(input_dims) > 0 and input_dims[0] and len(input_dims[0]) >= 3:
            dims = input_dims[0]
            if len(dims) == 4:
                ns.seq_len = dims[1]
                ns.num_heads_q = dims[2]
                ns.head_dim = dims[3]
            elif len(dims) == 3:
                ns.seq_len = dims[0]
                ns.num_heads_q = dims[1]
                ns.head_dim = dims[2]
            else:
                ns.seq_len, ns.num_heads_q, ns.head_dim = (2048, 32, 128)
        else:
            ns.seq_len, ns.num_heads_q, ns.head_dim = (2048, 32, 128)
        ns.num_heads_kv = 8
        if len(input_dims) > 1 and input_dims[1] and len(input_dims[1]) >= 3:
            kv_dims = input_dims[1]
            if len(kv_dims) == 4:
                ns.num_heads_kv = kv_dims[2]
                return ns
            if len(kv_dims) == 3:
                ns.num_heads_kv = kv_dims[1]
        return ns
    for key, default_val in reg.get('defaults', {}).items():
        setattr(ns, key, default_val)
    return ns


def _build_runner_argv_from_csv_entry(registry_key, input_dims, input_types, concrete_inputs, attn_params=None, num_warmup=3):
    """Decide which runner invocation to use for a CSV-driven validation entry.

    Returns ``(argv, num_warmup_iters)``.

    If ``registry_key`` is in :data:`USE_EXISTING_HARNESS_FOR_CSV`, dimensions
    are extracted from ``input_dims`` / ``input_types`` (and the optional
    ``attn_params``) and the standard per-op test_<op> is invoked. Otherwise
    the generic ``__generic__`` test is dispatched with the raw CSV shapes.
    """
    if registry_key in USE_EXISTING_HARNESS_FOR_CSV:
        ns = _extract_dims_for_existing_harness(registry_key, input_dims, input_types, concrete_inputs, attn_params)
        return _build_runner_argv(registry_key, ns, num_warmup=num_warmup)
    return _build_csv_runner_argv(registry_key, argparse.Namespace(), num_warmup=num_warmup,
                                  input_dims=input_dims, input_types=input_types)


def load_report_dir(report_dir):
    """Load ops from a TraceLens report directory for validation.

    Reads unified_perf_summary.csv, filters to ops with perf models and
    matching OP_REGISTRY entries, deduplicates by (name, Input Dims, Input type),
    and optionally loads InferenceAttention_fwd.csv for attention parameters.
    """
    unified_path = os.path.join(report_dir, 'unified_perf_summary.csv')
    if not os.path.isfile(unified_path):
        raise FileNotFoundError(f'unified_perf_summary.csv not found in {report_dir}')
    attn_csv_path = os.path.join(report_dir, 'InferenceAttention_fwd.csv')
    attn_params_map = {}
    if os.path.isfile(attn_csv_path):
        with open(attn_csv_path, newline='') as f:
            for row in csv.DictReader(f):
                key = row.get('name', '')
                input_dims_str = row.get('Input Dims_first', '')
                attn_params_map[(key, input_dims_str)] = {
                    'N_Q': row.get('param: N_Q', ''),
                    'H_Q': row.get('param: H_Q', ''),
                    'H_KV': row.get('param: H_KV', ''),
                    'd_h_qk': row.get('param: d_h_qk', ''),
                    'd_h_v': row.get('param: d_h_v', ''),
                    'c_sq': row.get('param: c_sq', ''),
                    'c_sk': row.get('param: c_sk', ''),
                    'c_sqsq': row.get('param: c_sqsq', ''),
                    'c_sqsk': row.get('param: c_sqsk', ''),
                    'g_sq': row.get('param: g_sq', ''),
                    'g_sk': row.get('param: g_sk', ''),
                    'g_sqsq': row.get('param: g_sqsq', ''),
                    'g_sqsk': row.get('param: g_sqsk', ''),
                }
    entries = []
    seen_keys = set()
    skipped_no_registry = set()
    skipped_no_perf_model = 0
    with open(unified_path, newline='') as f:
        for row in csv.DictReader(f):
            name = row.get('name', '')
            has_pm = row.get('has_perf_model', '')
            if has_pm != 'True':
                skipped_no_perf_model += 1
                continue
            registry_key = CSV_NAME_TO_REGISTRY.get(name)
            if registry_key is None:
                skipped_no_registry.add(name)
                continue
            if registry_key not in OP_REGISTRY:
                skipped_no_registry.add(name)
                continue
            dims_str = row.get('Input Dims', '')
            types_str = row.get('Input type', '')
            dedup_key = (name, dims_str, types_str)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            input_dims = _parse_csv_field(dims_str)
            input_types = _parse_csv_field(types_str)
            input_strides = _parse_csv_field(row.get('Input Strides', ''))
            concrete_inputs = _parse_csv_field(row.get('Concrete Inputs', ''))
            if input_dims is None:
                continue
            if isinstance(input_dims, tuple):
                input_dims = list(input_dims)
            if isinstance(input_types, tuple):
                input_types = list(input_types)
            if isinstance(input_strides, tuple):
                input_strides = list(input_strides)
            if isinstance(concrete_inputs, tuple):
                concrete_inputs = list(concrete_inputs)
            gflops_str = row.get('GFLOPS', '0')
            data_moved_str = row.get('Data Moved (MB)', '0')
            try:
                csv_flops = float(gflops_str) * 1e9
            except (ValueError, TypeError):
                csv_flops = 0
            try:
                csv_bytes = float(data_moved_str) * 1024 * 1024
            except (ValueError, TypeError):
                csv_bytes = 0
            ap = None
            reg_cat = OP_REGISTRY[registry_key]['category']
            if reg_cat in ('attention', 'vllm_attention'):
                ap = attn_params_map.get((name, dims_str))
            entries.append({
                'trace_name': name,
                'registry_key': registry_key,
                'input_dims': input_dims,
                'input_types': input_types,
                'input_strides': input_strides,
                'concrete_inputs': concrete_inputs,
                'csv_flops': csv_flops,
                'csv_bytes': csv_bytes,
                'attn_params': ap,
                'dims_str': dims_str,
            })
    print('\n--- Report Dir Summary ---')
    print(f'  Found {len(entries)} unique (name, shape, dtype) combos to validate')
    if skipped_no_registry:
        print(f'  Skipped {len(skipped_no_registry)} op names (no registry entry):')
        for s in sorted(skipped_no_registry)[:10]:
            print(f'    {s}')
        if len(skipped_no_registry) > 10:
            print(f'    ... and {len(skipped_no_registry) - 10} more')
    print(f'  Skipped {skipped_no_perf_model} rows without perf model')
    print()
    return entries


def run_csv_op_validation(entry, args, output_dir):
    """Run validation for a single op entry loaded from a report dir CSV."""
    registry_key = entry['registry_key']
    trace_name = entry['trace_name']
    reg = OP_REGISTRY[registry_key]
    arch_cfg = ARCH_COUNTER_CONFIGS[args.arch]
    dims_display = entry['dims_str'][:80] if entry['dims_str'] else ''
    print(f'\n{"######################################################################"}')
    print(f'# Validating: {trace_name} -> {registry_key}')
    print(f'# Dims: {dims_display}')
    print(f'{"######################################################################"}')
    if reg.get('perf_model_only'):
        print('[1/1] Running perf model from event dict (perf-model-only) ...')
        try:
            predicted_flops, predicted_bytes, predicted_precision = run_perf_model_from_event(
                trace_name, entry['input_dims'], entry['input_types'],
                entry['input_strides'], entry['concrete_inputs'], entry['attn_params'])
        except Exception as e:
            print(f'  WARNING: perf model failed: {e}')
            predicted_flops, predicted_bytes, predicted_precision = (0, 0, 'N/A')
        return {
            'op': registry_key,
            'trace_name': trace_name,
            'category': reg['category'],
            'dims': dims_display,
            'predicted_flops': predicted_flops,
            'predicted_bytes': predicted_bytes,
            'predicted_precision': predicted_precision or 'N/A',
            'csv_reported_flops': entry['csv_flops'],
            'csv_reported_bytes': entry['csv_bytes'],
            'rocprof_f4_ops': 0,
            'rocprof_f8_ops': 0,
            'rocprof_total_16': 0,
            'rocprof_total_32': 0,
            'rocprof_fetch_bytes': 0,
            'rocprof_write_bytes': 0,
            'kernel_name': '',
            'flops_ratio': None,
            'bytes_ratio': None,
            'precision': predicted_precision or 'N/A',
            'status': 'MODEL_ONLY',
        }
    step = 1
    total_steps = 8
    print(f'[{step}/{total_steps}] Building test runner command ...')
    runner_argv, num_warmup_iters = _build_runner_argv_from_csv_entry(
        registry_key, entry['input_dims'], entry['input_types'],
        entry['concrete_inputs'], entry['attn_params'])
    print(f'       Runner cmd: {" ".join(runner_argv)}')
    step += 1
    if args.kernel_filter:
        kernel_filter = args.kernel_filter
        print(f'[{step}/{total_steps}] Using manual kernel filter: {kernel_filter}')
    else:
        print(f'[{step}/{total_steps}] Discovering target kernel ...')
        kernel_filter = discover_kernel_name(
            args.rocprofv3_path, runner_argv, num_warmup_iters, output_dir, timeout=args.timeout)
        if kernel_filter:
            print(f'       Auto-discovered kernel: {kernel_filter[:120]}...')
        else:
            print('       WARNING: auto-discovery failed')
    step += 1
    ops_counters = {}
    print(f'[{step}/{total_steps}] Running rocprofv3 (MFMA ops) ...')
    try:
        csv_out = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_counters'], runner_argv,
                                'ops_counters', output_dir, timeout=args.timeout)
        ops_counters.update(parse_counter_csv(csv_out, kernel_filter))
    except (RuntimeError, FileNotFoundError) as e:
        print(f'  WARNING: MFMA ops collection failed: {e}')
    step += 1
    if not args.skip_derived and arch_cfg.get('ops_derived_16'):
        print(f'[{step}/{total_steps}] Running rocprofv3 (TOTAL_16_OPS) ...')
        try:
            csv_16 = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_derived_16'], runner_argv,
                                   'ops_derived_16', output_dir, timeout=args.timeout)
            ops_counters.update(parse_counter_csv(csv_16, kernel_filter))
        except (RuntimeError, FileNotFoundError) as e:
            print(f'  WARNING: TOTAL_16_OPS collection failed: {e}')
    step += 1
    if not args.skip_derived and arch_cfg.get('ops_derived_32'):
        print(f'[{step}/{total_steps}] Running rocprofv3 (TOTAL_32_OPS) ...')
        try:
            csv_32 = run_rocprofv3(args.rocprofv3_path, arch_cfg['ops_derived_32'], runner_argv,
                                   'ops_derived_32', output_dir, timeout=args.timeout)
            ops_counters.update(parse_counter_csv(csv_32, kernel_filter))
        except (RuntimeError, FileNotFoundError) as e:
            print(f'  WARNING: TOTAL_32_OPS collection failed: {e}')
    step += 1
    mem_counters = {}
    print(f'[{step}/{total_steps}] Running rocprofv3 (FETCH_SIZE) ...')
    try:
        fetch_csv = run_rocprofv3(args.rocprofv3_path, arch_cfg['mem_read_counters'], runner_argv,
                                  'mem_read_counters', output_dir, timeout=args.timeout)
        mem_counters.update(parse_counter_csv(fetch_csv, kernel_filter))
    except (RuntimeError, FileNotFoundError) as e:
        print(f'  WARNING: FETCH_SIZE collection failed: {e}')
    step += 1
    print(f'[{step}/{total_steps}] Running rocprofv3 (WRITE_SIZE) ...')
    try:
        write_csv = run_rocprofv3(args.rocprofv3_path, arch_cfg['mem_write_counters'], runner_argv,
                                  'mem_write_counters', output_dir, timeout=args.timeout)
        mem_counters.update(parse_counter_csv(write_csv, kernel_filter))
    except (RuntimeError, FileNotFoundError) as e:
        print(f'  WARNING: WRITE_SIZE collection failed: {e}')
    step += 1
    print(f'[{step}/{total_steps}] Running perf model from event dict ...')
    try:
        predicted_flops, predicted_bytes, predicted_precision = run_perf_model_from_event(
            trace_name, entry['input_dims'], entry['input_types'],
            entry['input_strides'], entry['concrete_inputs'], entry['attn_params'])
    except Exception as e:
        print(f'  WARNING: perf model from event failed: {e}')
        predicted_flops, predicted_bytes, predicted_precision = (0, 0, 'N/A')
    print(f'       Predicted FLOPs: {format_number(predicted_flops, "ops")}')
    print(f'       Predicted Bytes: {format_number(predicted_bytes, "bytes")}')
    kernel_runtime_ns = None
    kernel_runtime_mean_ns = None
    kernel_runtime_dispatches = 0
    trace_csv_path = os.path.join(output_dir, 'kernel_discovery', 'kernel_discovery_kernel_trace.csv')
    if not os.path.exists(trace_csv_path):
        candidates = list(Path(output_dir).rglob('*kernel_trace.csv'))
        if candidates:
            trace_csv_path = str(candidates[0])
    if os.path.exists(trace_csv_path):
        kernel_runtime_ns, kernel_runtime_mean_ns, kernel_runtime_dispatches = parse_kernel_runtime_ns(
            trace_csv_path, kernel_filter)
    report_data = generate_report(
        registry_key, args.arch, dims_display, predicted_flops, predicted_bytes,
        predicted_precision, ops_counters, mem_counters, output_dir,
        kernel_filter=kernel_filter, kernel_runtime_ns=kernel_runtime_ns,
        kernel_runtime_mean_ns=kernel_runtime_mean_ns,
        kernel_runtime_dispatches=kernel_runtime_dispatches)
    return {
        'op': registry_key,
        'trace_name': trace_name,
        'category': reg['category'],
        'dims': dims_display,
        'predicted_flops': predicted_flops,
        'predicted_bytes': predicted_bytes,
        'predicted_precision': predicted_precision or 'N/A',
        'csv_reported_flops': entry['csv_flops'],
        'csv_reported_bytes': entry['csv_bytes'],
        'rocprof_f4_ops': report_data['rocprof_f4_ops'],
        'rocprof_f8_ops': report_data['rocprof_f8_ops'],
        'rocprof_total_16': report_data['rocprof_total_16'],
        'rocprof_total_32': report_data['rocprof_total_32'],
        'rocprof_fetch_bytes': report_data['rocprof_fetch_bytes'],
        'rocprof_write_bytes': report_data['rocprof_write_bytes'],
        'kernel_name': kernel_filter or '',
        'kernel_runtime_us': report_data.get('kernel_runtime_us'),
        'kernel_runtime_mean_us': report_data.get('kernel_runtime_mean_us'),
        'kernel_runtime_dispatches': report_data.get('kernel_runtime_dispatches', 0),
        'predicted_tflops_s': report_data.get('predicted_tflops_s'),
        'measured_tflops_s': report_data.get('measured_tflops_s'),
        'predicted_tb_s': report_data.get('predicted_tb_s'),
        'measured_tb_s': report_data.get('measured_tb_s'),
        'throughput_ops_counter': report_data.get('throughput_ops_counter'),
        'flops_ratio': report_data['flops_ratio'],
        'bytes_ratio': report_data['bytes_ratio'],
        'precision': report_data['precision_result'],
        'status': 'OK',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate TraceLens perf model against rocprofv3 hardware counters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            '            Available ops:  %(ops)s\n            Categories:     %(cats)s\n        ' % {
                'ops': ', '.join(sorted(OP_REGISTRY.keys())),
                'cats': ', '.join(CATEGORIES),
            }),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--op', choices=list(OP_REGISTRY.keys()), help='Single op to validate.')
    group.add_argument('--all', action='store_true', help='Run all registered ops with default dimensions.')
    group.add_argument('--category', choices=CATEGORIES, help='Run all ops in a category.')
    group.add_argument('--discover', action='store_true',
                       help='List all aiter:: ops and their validation coverage, then exit.')
    group.add_argument('--from-report-dir', metavar='DIR',
                       help='Read a TraceLens report folder and validate all ops with matching '
                            'harnesses using shapes/dtypes from the CSV.')
    parser.add_argument('--M', type=int, default=None, help='M dimension (tokens)')
    parser.add_argument('--N', type=int, default=None, help='N dimension (inter_dim / columns)')
    parser.add_argument('--K', type=int, default=None, help='K dimension (hidden_dim / rows)')
    parser.add_argument('--E', type=int, default=None, help='Number of experts (MoE ops)')
    parser.add_argument('--topk', type=int, default=None, help='Top-k experts per token (MoE ops)')
    parser.add_argument('--group_size', type=int, default=None,
                        help='Quantization group size (vllm quant/norm ops, default 128)')
    parser.add_argument('--seq_len', type=int, default=None, help='Sequence length (attention ops)')
    parser.add_argument('--num_heads_q', type=int, default=None, help='Query heads (attention ops)')
    parser.add_argument('--num_heads_kv', type=int, default=None, help='KV heads (attention ops)')
    parser.add_argument('--head_dim', type=int, default=None, help='Head dimension (attention ops)')
    parser.add_argument('--annotation', default=None,
                        help="vLLM iteration annotation passed through to ops that accept it "
                             "(e.g. unified_attention). Format: 'execute_<i>_context_<N>"
                             "(sq<S>sk<K>sqsq<Q>sqsk<P>)_generation_<M>(sq<S>sk<K>sqsq<Q>sqsk<P>)'.")
    parser.add_argument('--varlen_num_seqs', type=int, default=None, dest='varlen_num_seqs',
                        help='Number of sequences packed into a varlen attention call (default 4).')
    parser.add_argument('--varlen_seed', type=int, default=None, dest='varlen_seed',
                        help='RNG seed for the random varlen partition (default 42).')
    parser.add_argument('--varlen_scenario', default=None, dest='varlen_scenario',
                        choices=['random', 'mixed_prefill_decode'],
                        help="'random' (default): self-attention with random partition of seq_len "
                             "tokens. 'mixed_prefill_decode': 1 prefill seq with Q=K=seq_len + "
                             "(varlen_num_seqs-1) decode seqs with Q=1, K=seq_len.")
    DTYPE_CHOICES = ['bf16', 'fp16', 'fp32', 'fp8', 'fp8_e4m3fn', 'fp8_e4m3fnuz',
                     'fp8_e5m2', 'fp4', 'fp4x2', 'i8', 'u8']
    parser.add_argument('--in_dtype', default=None, choices=DTYPE_CHOICES,
                        help='Activation/input dtype (default per-op; see perf_model_harnesses).')
    parser.add_argument('--w_dtype', default=None, choices=DTYPE_CHOICES, help='Weight dtype (default per-op).')
    parser.add_argument('--out_dtype', default=None, choices=DTYPE_CHOICES, help='Output dtype (default per-op).')
    parser.add_argument('--kv_dtype', default=None, choices=DTYPE_CHOICES,
                        help='Paged-attention KV-cache dtype (default per-op).')
    parser.add_argument('--bias_dtype', default=None, choices=DTYPE_CHOICES,
                        help='GEMM bias dtype (used by vllm_unquantized_gemm bias path).')
    parser.add_argument('--scale_dtype', default=None, choices=DTYPE_CHOICES,
                        help='Block-scale dtype (passed to the GPU test harness).')
    parser.add_argument('--quant_dtype', default=None, choices=DTYPE_CHOICES, help='Quantization storage dtype.')
    parser.add_argument('--quant_type', default=None,
                        choices=['no', 'per_tensor', 'per_token', 'per_1x32', 'per_1x128',
                                 'per_128x128', 'per_256x128', 'per_1024x128'],
                        help='aiter QuantType selector for ck_moe_stage{1,2}.')
    parser.add_argument('--activation', default=None, choices=['silu', 'gelu', 'swiglu', 'no'],
                        help='Activation selector (silu/gelu/swiglu).')
    parser.add_argument('--block_n', type=int, default=None, help='Per-1x128 / per-128x128 N tile size (default 128).')
    parser.add_argument('--block_k', type=int, default=None, help='Per-1x128 / per-128x128 K tile size (default 128).')
    parser.add_argument('--block_m', type=int, default=None, help='MoE M-block size (default 32).')
    parser.add_argument('--split_k', type=int, default=None, help='CK-Tile / CK split-K factor (default 1).')
    parser.add_argument('--arch', default='gfx942', choices=list(ARCH_COUNTER_CONFIGS.keys()),
                        help='GPU architecture (default: gfx942).')
    parser.add_argument('--output-dir', default=None, help='Directory for harness, rocprofv3 output, and report.')
    parser.add_argument('--rocprofv3-path', default='rocprofv3', help='Path to rocprofv3 binary.')
    parser.add_argument('--skip-derived', action='store_true',
                        help='Skip derived counters (TOTAL_16_OPS, TOTAL_32_OPS).')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each rocprofv3 run.')
    parser.add_argument('--kernel-filter', default=None, help='Manual kernel name filter (overrides auto-discovery).')
    args = parser.parse_args()

    if args.discover:
        discover_missing_coverage()
        return None

    from_report_dir = getattr(args, 'from_report_dir', None)
    if from_report_dir:
        if not args.output_dir:
            parser.error('--output-dir is required for validation runs.')
        base_output_dir = os.path.abspath(args.output_dir)
        os.makedirs(base_output_dir, exist_ok=True)
        entries = load_report_dir(from_report_dir)
        if not entries:
            print('No ops to validate from report dir.')
            return None
        results = []
        for idx, entry in enumerate(entries):
            safe_name = re.sub('[^a-zA-Z0-9_]', '_', entry['trace_name'])
            op_output_dir = os.path.join(base_output_dir, f'{idx:03d}_{safe_name}')
            os.makedirs(op_output_dir, exist_ok=True)
            try:
                result = run_csv_op_validation(entry, args, op_output_dir)
                results.append(result)
            except Exception as e:
                print(f'\n  FAILED: {entry["trace_name"]}: {e}')
                results.append({
                    'op': entry['registry_key'],
                    'trace_name': entry['trace_name'],
                    'category': OP_REGISTRY[entry['registry_key']]['category'],
                    'dims': entry.get('dims_str', '')[:80],
                    'predicted_flops': None,
                    'predicted_bytes': None,
                    'predicted_precision': '---',
                    'csv_reported_flops': entry.get('csv_flops', 0),
                    'csv_reported_bytes': entry.get('csv_bytes', 0),
                    'rocprof_f4_ops': 0,
                    'rocprof_f8_ops': 0,
                    'rocprof_total_16': 0,
                    'rocprof_total_32': 0,
                    'rocprof_fetch_bytes': 0,
                    'rocprof_write_bytes': 0,
                    'kernel_name': '',
                    'flops_ratio': None,
                    'bytes_ratio': None,
                    'precision': '---',
                    'status': f'FAIL: {str(e)[:60]}',
                })
        _enrich_results(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'validation_report_{timestamp}.csv'
        csv_path = os.path.join(base_output_dir, csv_filename)
        write_results_csv(results, csv_path)
        skill_dir = Path(__file__).resolve().parent
        skill_csv_path = skill_dir / csv_filename
        write_results_csv(results, str(skill_csv_path))
        print_batch_summary(results)
        return None

    if args.all:
        ops_to_run = list(OP_REGISTRY.keys())
    elif args.category:
        ops_to_run = [k for k, v in OP_REGISTRY.items() if v['category'] == args.category]
    elif args.op:
        ops_to_run = [args.op]
    else:
        parser.error('One of --op, --all, --category, --discover, or --from-report-dir is required.')

    if not args.output_dir:
        parser.error('--output-dir is required for validation runs.')
    base_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    is_batch = len(ops_to_run) > 1
    results = []
    for op_name in ops_to_run:
        if is_batch:
            op_output_dir = os.path.join(base_output_dir, op_name)
        else:
            op_output_dir = base_output_dir
        os.makedirs(op_output_dir, exist_ok=True)
        try:
            result = run_single_op_validation(op_name, args, op_output_dir)
            results.append(result)
        except Exception as e:
            print(f'\n  FAILED: {op_name}: {e}')
            results.append({
                'op': op_name,
                'category': OP_REGISTRY[op_name]['category'],
                'dims': '',
                'predicted_flops': None,
                'predicted_bytes': None,
                'predicted_precision': '---',
                'rocprof_f4_ops': 0,
                'rocprof_f8_ops': 0,
                'rocprof_total_16': 0,
                'rocprof_total_32': 0,
                'rocprof_fetch_bytes': 0,
                'rocprof_write_bytes': 0,
                'kernel_name': '',
                'flops_ratio': None,
                'bytes_ratio': None,
                'precision': '---',
                'status': f'FAIL: {str(e)[:60]}',
            })
    _enrich_results(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'validation_report_{timestamp}.csv'
    csv_path = os.path.join(base_output_dir, csv_filename)
    write_results_csv(results, csv_path)
    skill_dir = Path(__file__).resolve().parent
    skill_csv_path = skill_dir / csv_filename
    write_results_csv(results, str(skill_csv_path))
    print_batch_summary(results)
    return None


if __name__ == '__main__':
    main()
