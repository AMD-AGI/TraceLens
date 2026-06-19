###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI dispatcher that rocprofv3 wraps as a subprocess.

Usage:

    python tests/_runner.py --op gemm_a8w8_blockscale --M 2048 --N 4096 --K 8192

Imports ``test_<op>`` from the appropriate per-category module and calls it
with the supplied dimensional kwargs. Every test function tolerates extra
kwargs via ``**_`` so unrelated CLI flags are silently ignored.
"""

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    _PKG = "tests"
else:
    _PKG = __package__


def _try_import(modname):
    """Import a sibling test module, tolerating ones that are absent/broken.

    The harness suite is shared across checkouts and some op-category modules
    (or their helpers) may be missing in a given environment. A missing module
    should only disable its own ops, not block dispatch of the rest.
    """
    try:
        return importlib.import_module(f"{_PKG}.{modname}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"_runner: note: module '{modname}' unavailable ({exc}); "
              "its ops will be skipped.", flush=True)
        return None


def _add(table, mod, mapping):
    """Add ``{op_name: getattr(mod, fn_name)}`` entries if ``mod`` imported."""
    if mod is None:
        return
    for op_name, fn_name in mapping.items():
        fn = getattr(mod, fn_name, None)
        if fn is not None:
            table[op_name] = fn


def _build_op_table():
    """Map OP_REGISTRY op-names to their ``test_<op>`` callables."""
    gemm = _try_import("gemm")
    moe = _try_import("moe")
    attention = _try_import("attention")
    rmsnorm = _try_import("rmsnorm")
    other = _try_import("other")
    dsv3 = _try_import("dsv3")
    dsv4 = _try_import("dsv4")
    extensions_new = _try_import("extensions_new")
    atom_flydsl = _try_import("atom_flydsl")

    table = {}
    _add(table, gemm, {
        "gemm_a8w8_blockscale": "test_gemm_a8w8_blockscale",
        "gemm_a16w16_atomic_": "test_gemm_a16w16_atomic_",
        "vllm_unquantized_gemm": "test_vllm_unquantized_gemm",
        "vllm_triton_gemm_a8w8_blockscale": "test_gemm_a8w8_blockscale",
        "vllm_gemm_with_dynamic_quant": "test_vllm_gemm_with_dynamic_quant",
    })
    _add(table, moe, {
        "fmoe_fp8_blockscale_g1u1": "test_fmoe_fp8_blockscale_g1u1",
        "moe_cktile2stages_gemm1_ck": "test_moe_cktile2stages_gemm1_ck",
        "moe_cktile2stages_gemm2_ck": "test_moe_cktile2stages_gemm2_ck",
        "ck_moe_stage1": "test_ck_moe_stage1",
        "ck_moe_stage2": "test_ck_moe_stage2",
        "sglang_fused_moe_triton_invoke": "test_sglang_fused_moe_triton_invoke",
    })
    _add(table, attention, {
        "_flash_attn_forward": "test__flash_attn_forward",
        "wrapper_fmha_v3_fwd": "test_wrapper_fmha_v3_fwd",
        "mha_varlen_fwd": "test_mha_varlen_fwd",
        "fmha_v3_varlen_fwd": "test_fmha_v3_varlen_fwd",
        "unified_attention": "test_unified_attention",
        "vllm_unified_attention": "test_vllm_unified_attention",
    })
    _add(table, rmsnorm, {
        "rms_norm": "test_rms_norm",
        "rmsnorm": "test_rmsnorm",
        "add_rmsnorm": "test_add_rmsnorm",
        "rmsnorm_dynamicquant": "test_rmsnorm_dynamicquant",
        "vllm_rmsnorm_fp8_group_quant": "test_vllm_rmsnorm_fp8_group_quant",
        "vllm_rmsnorm_add_fp8_group_quant": "test_vllm_rmsnorm_add_fp8_group_quant",
    })
    _add(table, other, {
        "silu_and_mul": "test_silu_and_mul",
        "gelu_and_mul": "test_gelu_and_mul",
        "gelu_tanh_and_mul": "test_gelu_tanh_and_mul",
        "dynamic_per_token_scaled_quant": "test_dynamic_per_token_scaled_quant",
        "vllm_triton_group_quant_fp8": "test_vllm_triton_group_quant_fp8",
    })
    _add(table, dsv3, {
        "dsv3_flydsl_hgemm": "test_dsv3_flydsl_hgemm",
        "dsv3_batched_gemm_a8w8": "test_dsv3_batched_gemm_a8w8",
        "dsv3_fused_flatten_fp8_group_quant": "test_dsv3_fused_flatten_fp8_group_quant",
        "dsv3_fused_qk_rope_cat_and_cache_mla": "test_dsv3_fused_qk_rope_cat_and_cache_mla",
        "dsv3_fused_append_shared_experts": "test_dsv3_fused_append_shared_experts",
        "dsv3_mla_prefill_ps_asm_fwd": "test_dsv3_mla_prefill_ps_asm_fwd",
        "dsv3_mla_reduce_v1": "test_dsv3_mla_reduce_v1",
    })
    _add(table, extensions_new, {
        "gemm_afp4wfp4": "test_gemm_afp4wfp4",
        "rope_cached_positions_2c_fwd_impl": "test_rope_cached_positions_2c_fwd_impl",
        "fused_flatten_mxfp4_quant": "test_fused_flatten_mxfp4_quant",
        "fused_rms_mxfp4_quant": "test_fused_rms_mxfp4_quant",
    })
    _add(table, atom_flydsl, {
        "atom_flydsl_preshuffle_gemm_a8": "test_atom_flydsl_preshuffle_gemm_a8",
        "atom_flydsl_gdr_decode": "test_atom_flydsl_gdr_decode",
    })
    _add(table, dsv4, {
        "dsv4_mhc_pre_gemm_sqrsum": "test_dsv4_mhc_pre_gemm_sqrsum",
        "dsv4_mhc_pre_big_fuse": "test_dsv4_mhc_pre_big_fuse",
        "dsv4_mhc_post": "test_dsv4_mhc_post",
        "dsv4_pa_sparse_prefill_opus": "test_dsv4_pa_sparse_prefill_opus",
        "dsv4_opus_gemm_a16w16": "test_dsv4_opus_gemm_a16w16",
        "dsv4_gemm_a8w8_blockscale_bpreshuffle_asm": "test_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm",
        "dsv4_dynamic_per_group_scaled_quant": "test_dsv4_dynamic_per_group_scaled_quant",
        "dsv4_topk_softplus": "test_dsv4_topk_softplus",
        "dsv4_fused_dynamic_mx_quant_moe_sort": "test_dsv4_fused_dynamic_mx_quant_moe_sort",
    })

    generic = _try_import("_generic")
    if generic is not None and getattr(generic, "test_generic_simple_op", None):
        table["__generic__"] = generic.test_generic_simple_op
    return table


def main():
    p = argparse.ArgumentParser(
        description="Validate-perf-model test runner (rocprofv3 wraps this)."
    )
    p.add_argument("--op", required=True, help="OP_REGISTRY key selecting which test_<op> to invoke.")
    for k in (
        "M", "N", "K", "E", "topk", "group_size", "seq_len", "num_heads_q",
        "num_heads_kv", "head_dim", "block_n", "block_k", "block_m", "split_k",
        "num_decode_seqs", "ctx_len", "prefill_seq_len",
    ):
        p.add_argument(f"--{k}", type=int, default=None)
    for k in (
        "in_dtype", "w_dtype", "out_dtype", "scale_dtype", "kv_dtype",
        "bias_dtype", "quant_dtype", "quant_type", "activation",
    ):
        p.add_argument(f"--{k}", type=str, default=None)
    p.add_argument("--annotation", default=None)
    p.add_argument("--num-warmup", type=int, default=3, dest="num_warmup")
    p.add_argument(
        "--varlen-seed", type=int, default=42, dest="varlen_seed",
        help="RNG seed for variable-length attention seq partitioning.",
    )
    p.add_argument(
        "--varlen-num-seqs", type=int, default=4, dest="varlen_num_seqs",
        help="Number of sequences in varlen attention harness.",
    )
    p.add_argument(
        "--varlen-scenario", default="random",
        choices=["random", "mixed_prefill_decode"], dest="varlen_scenario",
        help=(
            "Varlen attention layout: 'random' for self-attention with a random "
            "partition of seq_len tokens, or 'mixed_prefill_decode' for one prefill "
            "seq (Q=K=seq_len) plus (varlen_num_seqs-1) decode seqs (Q=1, K=seq_len each)."
        ),
    )
    p.add_argument("--input-dims-json", default=None, dest="input_dims_json")
    p.add_argument("--input-types-json", default=None, dest="input_types_json")
    p.add_argument("--op-namespace", default=None, dest="op_namespace")
    p.add_argument("--op-fn-name", default=None, dest="op_fn_name")
    p.add_argument(
        "--registry-key", default=None, dest="registry_key",
        help="Original OP_REGISTRY key (generic-CSV mode only).",
    )
    args = p.parse_args()

    table = _build_op_table()
    fn = table.get(args.op)
    if fn is None:
        raise SystemExit(f"_runner: unknown --op '{args.op}'. Known: {sorted(table)}")

    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != "op"}
    print(f"_runner: dispatching test for op={args.op}", flush=True)
    fn(**kwargs)


if __name__ == "__main__":
    main()
