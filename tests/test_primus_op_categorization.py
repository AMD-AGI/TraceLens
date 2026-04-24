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
    MoEComm,
    moe_dispatch,
    moe_combine,
    CausalConv1d,
    causal_conv1d_fwd,
    FusedRoPE,
    fused_rope_fwd,
    CrossEntropy,
    cross_entropy_fwd,
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


# ---------------------------------------------------------------------------
# MoE comm — perf model tests
# ---------------------------------------------------------------------------


def _moe_dispatch_event(num_tokens=4096, hidden=7168, dtype="c10::BFloat16"):
    return {
        "name": "MoEDispatch",
        "args": {
            "Input Dims": [
                [num_tokens, hidden],
                [num_tokens, 8],
                [num_tokens, 8],
                [],
                [],
                [],
                [],
            ],
            "Input type": [
                dtype,
                "long int",
                "float",
                "Scalar",
                "Scalar",
                "Scalar",
                "Scalar",
            ],
            "Concrete Inputs": ["", "", "", "64", "True", "True", "32768"],
        },
    }


def _moe_combine_event(num_tokens=32768, hidden=7168, dtype="c10::BFloat16"):
    return {
        "name": "MoECombine",
        "args": {
            "Input Dims": [[num_tokens, hidden], [], []],
            "Input type": [dtype, "Scalar", "Scalar"],
            "Concrete Inputs": ["", "True", "True"],
        },
    }


def test_moe_dispatch_mapped():
    assert op_to_perf_model_class_map["MoEDispatch"] is moe_dispatch


def test_moe_combine_mapped():
    assert op_to_perf_model_class_map["MoECombine"] is moe_combine


def test_moe_dispatch_flops_zero():
    model = moe_dispatch(_moe_dispatch_event())
    assert model.flops() == 0


def test_moe_dispatch_bytes():
    model = moe_dispatch(_moe_dispatch_event(num_tokens=4096, hidden=7168))
    expected = 4096 * 7168 * 2  # bf16
    assert model.bytes() == expected


def test_moe_combine_bytes():
    model = moe_combine(_moe_combine_event(num_tokens=32768, hidden=7168))
    expected = 32768 * 7168 * 2
    assert model.bytes() == expected


def test_moe_combine_larger_than_dispatch():
    dispatch = moe_dispatch(_moe_dispatch_event(num_tokens=4096, hidden=7168))
    combine = moe_combine(_moe_combine_event(num_tokens=32768, hidden=7168))
    assert combine.bytes() > dispatch.bytes()


def test_moe_inherits_moecomm():
    assert issubclass(moe_dispatch, MoEComm)
    assert issubclass(moe_combine, MoEComm)


# ---------------------------------------------------------------------------
# Causal Conv1D — perf model tests
# ---------------------------------------------------------------------------


def _conv1d_event(batch=8, channels=3072, seq_len=8192, kernel_size=4):
    return {
        "name": "DaoAILab::_causal_conv1d_fwd_cpp",
        "args": {
            "Input Dims": [
                [batch, channels, seq_len],
                [channels, kernel_size],
                [channels],
                [],
                [],
                [batch, channels, seq_len],
                [],
                [],
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "",
                "",
                "c10::BFloat16",
                "",
                "Scalar",
            ],
        },
    }


def test_causal_conv1d_mapped():
    assert (
        op_to_perf_model_class_map["DaoAILab::_causal_conv1d_fwd_cpp"]
        is causal_conv1d_fwd
    )


def test_causal_conv1d_flops():
    model = causal_conv1d_fwd(
        _conv1d_event(batch=8, channels=3072, seq_len=8192, kernel_size=4)
    )
    expected = 2 * 8 * 3072 * 8192 * 4
    assert model.flops() == expected


def test_causal_conv1d_bytes():
    model = causal_conv1d_fwd(
        _conv1d_event(batch=8, channels=3072, seq_len=8192, kernel_size=4)
    )
    bpe = 2
    input_bytes = 8 * 3072 * 8192 * bpe
    weight_bytes = 3072 * 4 * bpe
    output_bytes = 8 * 3072 * 8192 * bpe
    bias_bytes = 3072 * bpe  # has bias (Input Dims[2] = [3072])
    assert model.bytes() == input_bytes + weight_bytes + output_bytes + bias_bytes


def test_causal_conv1d_inherits():
    assert issubclass(causal_conv1d_fwd, CausalConv1d)


# ---------------------------------------------------------------------------
# RoPE — perf model tests
# ---------------------------------------------------------------------------


def _rope_event(seq_len=4096, batch=4, heads=32, head_dim=128):
    return {
        "name": "FusedRoPEFunc",
        "args": {
            "Input Dims": [
                [seq_len, batch, heads, head_dim],
                [seq_len, 1, 1, head_dim],
                [],
                [],
                [],
                [],
                [],
            ],
            "Input type": [
                "c10::BFloat16",
                "float",
                "",
                "Scalar",
                "",
                "Scalar",
                "Scalar",
            ],
        },
    }


def test_rope_mapped():
    assert op_to_perf_model_class_map["FusedRoPEFunc"] is fused_rope_fwd


def test_rope_flops():
    model = fused_rope_fwd(_rope_event(seq_len=4096, batch=4, heads=32, head_dim=128))
    num_elements = 4096 * 4 * 32 * 128
    expected = 3 * num_elements
    assert model.flops() == expected


def test_rope_bytes():
    model = fused_rope_fwd(_rope_event(seq_len=4096, batch=4, heads=32, head_dim=128))
    num_elements = 4096 * 4 * 32 * 128
    bpe = 2
    expected = 2 * num_elements * bpe
    assert model.bytes() == expected


def test_rope_inherits():
    assert issubclass(fused_rope_fwd, FusedRoPE)


# ---------------------------------------------------------------------------
# CrossEntropy — perf model tests
# ---------------------------------------------------------------------------


def _ce_event(batch=4096, vocab_size=163840):
    return {
        "name": "CrossEntropyFunction",
        "args": {
            "Input Dims": [
                [batch, 1, vocab_size],
                [batch, 1],
                [],
                [],
            ],
            "Input type": ["c10::BFloat16", "long int", "Scalar", "Scalar"],
        },
    }


def test_ce_mapped():
    assert op_to_perf_model_class_map["CrossEntropyFunction"] is cross_entropy_fwd


def test_ce_flops():
    model = cross_entropy_fwd(_ce_event(batch=4096, vocab_size=163840))
    expected = 5 * 4096 * 163840
    assert model.flops() == expected


def test_ce_bytes():
    model = cross_entropy_fwd(_ce_event(batch=4096, vocab_size=163840))
    bpe = 2
    logits_bytes = 4096 * 1 * 163840 * bpe
    target_bytes = 4096 * 8  # long int
    output_bytes = 4096 * 4  # float32 loss
    assert model.bytes() == logits_bytes + target_bytes + output_bytes


def test_ce_inherits():
    assert issubclass(cross_entropy_fwd, CrossEntropy)
