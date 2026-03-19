###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for CK grouped GEMM mapping (#540), SSM/Mamba categorization (#541),
MoE dispatch/combine categorization (#542), and RoPE/CrossEntropy
categorization (#543).
"""

from TraceLens.PerfModel.perf_model import (
    primus_turbo_grouped_gemm,
    primus_turbo_grouped_gemm_variable_k,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    op_to_perf_model_class_map,
)

# ---------------------------------------------------------------------------
# #540 — CK grouped GEMM mapping
# ---------------------------------------------------------------------------


def test_ck_grouped_gemm_mapped():
    op = "primus_turbo_cpp_extension::ck_grouped_gemm"
    assert op_to_perf_model_class_map[op] is primus_turbo_grouped_gemm


def test_ck_grouped_gemm_variable_k_mapped():
    op = "primus_turbo_cpp_extension::ck_grouped_gemm_variable_k"
    assert op_to_perf_model_class_map[op] is primus_turbo_grouped_gemm_variable_k


def test_ck_grouped_gemm_categorizes_as_gemm():
    row = {"name": "primus_turbo_cpp_extension::ck_grouped_gemm", "kernel_details": []}
    assert categorize_torch_op(row) == "GEMM"


def test_ck_grouped_gemm_variable_k_categorizes_as_gemm():
    row = {
        "name": "primus_turbo_cpp_extension::ck_grouped_gemm_variable_k",
        "kernel_details": [],
    }
    assert categorize_torch_op(row) == "GEMM"


def test_ck_grouped_gemm_flops():
    """CK grouped GEMM reuses primus_turbo_grouped_gemm's perf model."""
    event = {
        "name": "primus_turbo_cpp_extension::ck_grouped_gemm",
        "args": {
            "Input Dims": [[32768, 7168], [8, 7168, 4096]],
            "Input type": ["c10::BFloat16", "c10::BFloat16"],
        },
    }
    model = primus_turbo_grouped_gemm(event)
    M, K, N = 32768, 7168, 4096
    assert model.flops() == 2 * M * K * N


# ---------------------------------------------------------------------------
# #541 — SSM/Mamba categorization
# ---------------------------------------------------------------------------


def test_mamba_fwd_categorizes_as_ssm_fwd():
    row = {"name": "MambaSplitConv1dScanCombinedFn", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_fwd"


def test_mamba_bwd_categorizes_as_ssm_bwd():
    row = {"name": "MambaSplitConv1dScanCombinedFnBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_bwd"


def test_causal_conv1d_fwd_categorizes_as_ssm_fwd():
    row = {"name": "DaoAILab::_causal_conv1d_fwd_cpp", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_fwd"


def test_causal_conv1d_bwd_categorizes_as_ssm_bwd():
    row = {"name": "DaoAILab::_causal_conv1d_bwd_cpp", "kernel_details": []}
    assert categorize_torch_op(row) == "SSM_bwd"


# ---------------------------------------------------------------------------
# #542 — MoE dispatch/combine categorization
# ---------------------------------------------------------------------------


def test_moe_dispatch_categorizes_as_moe_comm_fwd():
    row = {"name": "MoEDispatch", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_fwd"


def test_moe_dispatch_bwd_categorizes_as_moe_comm_bwd():
    row = {"name": "MoEDispatchBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_bwd"


def test_moe_combine_categorizes_as_moe_comm_fwd():
    row = {"name": "MoECombine", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_fwd"


def test_moe_combine_bwd_categorizes_as_moe_comm_bwd():
    row = {"name": "MoECombineBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_bwd"


def test_token_permute_categorizes_as_moe_comm_fwd():
    row = {"name": "TokenPermuteMaskMap", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_fwd"


def test_token_permute_bwd_categorizes_as_moe_comm_bwd():
    row = {"name": "TokenPermuteMaskMapBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_bwd"


def test_operation_fuser_fwd_categorizes_as_moe_comm_fwd():
    row = {"name": "_OperationFuserAutogradFunction", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_fwd"


def test_operation_fuser_bwd_categorizes_as_moe_comm_bwd():
    row = {"name": "_OperationFuserAutogradFunctionBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "MoE_comm_bwd"


# ---------------------------------------------------------------------------
# #543 — RoPE and CrossEntropy categorization
# ---------------------------------------------------------------------------


def test_fused_rope_categorizes_as_rope_fwd():
    row = {"name": "FusedRoPEFunc", "kernel_details": []}
    assert categorize_torch_op(row) == "RoPE_fwd"


def test_fused_rope_bwd_categorizes_as_rope_bwd():
    row = {"name": "FusedRoPEFuncBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "RoPE_bwd"


def test_cross_entropy_categorizes_as_ce_fwd():
    row = {"name": "CrossEntropyFunction", "kernel_details": []}
    assert categorize_torch_op(row) == "CrossEntropy_fwd"


def test_cross_entropy_bwd_categorizes_as_ce_bwd():
    row = {"name": "CrossEntropyFunctionBackward", "kernel_details": []}
    assert categorize_torch_op(row) == "CrossEntropy_bwd"
