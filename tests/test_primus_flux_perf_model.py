###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for Primus Turbo / Flux MXFP4 ops:
  - primus_turbo::gemm_fp4_impl -> GEMM (Flux trace leaf)
  - primus_flux::a6w4_mm_qa -> GEMM
  - primus_flux::a6w4_mm_qa_prequant_w -> GEMM
  - primus_flux::quantize_mxfp4_h16_dual -> UnaryElementwise
  - tree_postprocess_extension (fwd/bwd pseudo-op labeling)
"""


from TraceLens.PerfModel.perf_model import gemm_fp4_impl, GEMM
from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
    a6w4_mm_qa,
    a6w4_mm_qa_prequant_w,
    quantize_mxfp4_h16_dual,
    _QuantizeMxfp4Base,
    _suffix_for_op,
    tree_postprocess_extension,
)
from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
from TraceLens.PerfModel.utils import MXFP4_GROUP_SIZE, name2bpe

# ---------------------------------------------------------------------------
# Helpers — synthetic trace events
# ---------------------------------------------------------------------------


def _gemm_fp4_impl_event(M=3072, N=3072, K=3072, has_bias=False):
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    dims = [
        [M, K_packed],  # A (FP4 packed)
        [M, K_scale],  # A scales (E8M0)
        [],  # Scalar
        [N, K_packed],  # B (FP4 packed)
        [N, K_scale],  # B scales (E8M0)
        [],  # Scalar
        [],  # out_dtype
        [],  # bias_flag
        [],  # granularity
        [],  # backend
        [],  # preshuffled
        [N] if has_bias else [],  # bias
        [],  # out_scale_A (optional)
        [],  # out_scale_B (optional)
    ]
    types = [
        "c10::Float4_e2m1fn_x2",
        "c10::Float8_e8m0fnu",
        "Scalar",
        "c10::Float4_e2m1fn_x2",
        "c10::Float8_e8m0fnu",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "c10::BFloat16" if has_bias else "",
        "",
        "",
    ]
    return {
        "name": "primus_turbo::gemm_fp4_impl",
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Input Strides": [[] for _ in dims],
            "Concrete Inputs": [""] * len(dims),
        },
    }


def _a6w4_mm_qa_event(M=3072, N=3072, K=3072, has_bias=True):
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    return {
        "name": "primus_flux::a6w4_mm_qa",
        "args": {
            "Input Dims": [
                [M, K],  # aq (u8 activation)
                [scale_elems],  # sa (activation scales)
                [N, K],  # b (BF16 weight)
                [N] if has_bias else [],  # bias
            ],
            "Input type": [
                "unsigned char",
                "unsigned char",
                "c10::BFloat16",
                "c10::BFloat16" if has_bias else "",
            ],
            "Input Strides": [[], [], [], []],
            "Concrete Inputs": ["", "", "", ""],
        },
    }


def _a6w4_mm_qa_prequant_w_event(M=3072, N=3072, K=3072, has_bias=True):
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    scale_elems = M * K_scale
    return {
        "name": "primus_flux::a6w4_mm_qa_prequant_w",
        "args": {
            "Input Dims": [
                [M, K],  # aq (u8 activation)
                [scale_elems],  # sa (activation scales)
                [N, K_packed],  # wq (FP4 packed weight)
                [N, K_scale],  # sb (weight scales)
                [],  # Scalar (n)
                [N] if has_bias else [],  # bias
            ],
            "Input type": [
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "unsigned char",
                "Scalar",
                "c10::BFloat16" if has_bias else "",
            ],
            "Input Strides": [[], [], [], [], [], []],
            "Concrete Inputs": ["", "", "", "", "", ""],
        },
    }


def _quantize_mxfp4_event(M=3072, N=3072):
    return {
        "name": "primus_flux::quantize_mxfp4_h16_dual",
        "args": {
            "Input Dims": [[M, N], [], []],
            "Input type": ["c10::BFloat16", "Scalar", "Scalar"],
            "Input Strides": [[N, 1], [], []],
            "Concrete Inputs": ["", "", ""],
        },
    }


# ===========================================================================
# Op registration
# ===========================================================================


def test_gemm_fp4_impl_registered():
    assert "primus_turbo::gemm_fp4_impl" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["primus_turbo::gemm_fp4_impl"] is gemm_fp4_impl


def test_a6w4_mm_qa_registered():
    assert "primus_flux::a6w4_mm_qa" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["primus_flux::a6w4_mm_qa"] is a6w4_mm_qa


def test_a6w4_mm_qa_prequant_w_registered():
    assert "primus_flux::a6w4_mm_qa_prequant_w" in op_to_perf_model_class_map
    assert (
        op_to_perf_model_class_map["primus_flux::a6w4_mm_qa_prequant_w"]
        is a6w4_mm_qa_prequant_w
    )


def test_quantize_mxfp4_h16_dual_registered():
    assert "primus_flux::quantize_mxfp4_h16_dual" in op_to_perf_model_class_map
    assert (
        op_to_perf_model_class_map["primus_flux::quantize_mxfp4_h16_dual"]
        is quantize_mxfp4_h16_dual
    )


def test_pseudo_ops_registered():
    for prefix in [
        "_MXFP4Linear",
        "_MXFP4GatedLinear",
        "_MXFP4LinearGelu",
        "_MXFP4LinearGeluLegs",
    ]:
        for role in [
            "_yfwd_gemm",
            "Backward_xgrad_gemm",
            "Backward_wgrad_gemm",
            "_yfwd_a6w4",
            "Backward_xgrad_a6w4",
            "Backward_wgrad_a6w4",
        ]:
            name = f"{prefix}{role}"
            assert name in op_to_perf_model_class_map, f"missing: {name}"


# ===========================================================================
# gemm_fp4_impl
# ===========================================================================


def test_gemm_fp4_impl_inherits_gemm():
    assert issubclass(gemm_fp4_impl, GEMM)


def test_gemm_fp4_impl_flops():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert model.flops() == 2 * M * N * K


def test_gemm_fp4_impl_param_details():
    M, N, K = 4096, 3072, 6144
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_gemm_fp4_impl_k_unpacked_from_packed_dim():
    """K_logical = K_packed * 2 (Float4_e2m1fn_x2 packs two values per byte)."""
    M, N, K = 1024, 2048, 4096
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    assert model.K == K
    assert model.K == model.param_details["a_scale_shape"][1] * MXFP4_GROUP_SIZE


def test_gemm_fp4_impl_bytes_includes_scales():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K))
    bpe_fp4_packed = 1  # c10::Float4_e2m1fn_x2 = 1 byte per packed pair
    bpe_scale = 1
    bpe_out = 2
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    expected = (
        M * K_packed * bpe_fp4_packed  # A
        + N * K_packed * bpe_fp4_packed  # B
        + M * N * bpe_out  # Y
        + N * K_scale * bpe_scale  # B scales
    )
    assert model.bytes() == expected


def test_gemm_fp4_impl_bytes_with_bias():
    M, N, K = 3072, 3072, 3072
    model = gemm_fp4_impl(_gemm_fp4_impl_event(M, N, K, has_bias=True))
    bpe_fp4_packed = 1
    bpe_scale = 1
    bpe_out = 2
    K_packed = K // 2
    K_scale = K // MXFP4_GROUP_SIZE
    expected = (
        M * K_packed * bpe_fp4_packed
        + N * K_packed * bpe_fp4_packed
        + M * N * bpe_out
        + N * bpe_out  # bias (BF16)
        + N * K_scale * bpe_scale
    )
    assert model.bytes() == expected


def test_gemm_fp4_impl_compute_precision():
    from TraceLens.PerfModel.utils import torch_dtype_map

    model = gemm_fp4_impl(_gemm_fp4_impl_event())
    assert model.get_compute_precision() == torch_dtype_map("mxfp4")


# ===========================================================================
# a6w4_mm_qa
# ===========================================================================


def test_a6w4_mm_qa_inherits_gemm():
    assert issubclass(a6w4_mm_qa, GEMM)


def test_a6w4_mm_qa_flops():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=False))
    assert model.flops() == 2 * M * N * K


def test_a6w4_mm_qa_param_details():
    M, N, K = 4096, 2048, 6144
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_a6w4_mm_qa_bytes():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=True))
    bpe_u8 = 1
    bpe_bf16 = 2
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    expected = (
        M * K * bpe_u8  # activation
        + N * K * bpe_bf16  # weight
        + M * N * bpe_bf16  # output
        + N * bpe_bf16  # bias
        + scale_elems * bpe_u8  # activation scales
    )
    assert model.bytes() == expected


def test_a6w4_mm_qa_no_bias():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa(_a6w4_mm_qa_event(M, N, K, has_bias=False))
    bpe_u8 = 1
    bpe_bf16 = 2
    scale_elems = M * (K // MXFP4_GROUP_SIZE)
    expected = (
        M * K * bpe_u8 + N * K * bpe_bf16 + M * N * bpe_bf16 + scale_elems * bpe_u8
    )
    assert model.bytes() == expected


# ===========================================================================
# a6w4_mm_qa_prequant_w
# ===========================================================================


def test_a6w4_mm_qa_prequant_w_inherits_gemm():
    assert issubclass(a6w4_mm_qa_prequant_w, GEMM)


def test_a6w4_mm_qa_prequant_w_flops():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K, has_bias=False))
    assert model.flops() == 2 * M * N * K


def test_a6w4_mm_qa_prequant_w_param_details():
    M, N, K = 4096, 2048, 6144
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K))
    assert (model.M, model.N, model.K) == (M, N, K)


def test_a6w4_mm_qa_prequant_w_bytes():
    M, N, K = 3072, 3072, 3072
    model = a6w4_mm_qa_prequant_w(_a6w4_mm_qa_prequant_w_event(M, N, K, has_bias=True))
    bpe_u8 = 1
    bpe_bf16 = 2
    expected = (
        M * K * bpe_u8  # activation (u8)
        + N * K * bpe_u8  # weight (u8 packed, bpe from trace = unsigned char = 1)
        + M * N * bpe_bf16  # output
        + N * bpe_bf16  # bias
    )
    assert model.bytes() == expected


# ===========================================================================
# quantize_mxfp4_h16_dual
# ===========================================================================


def test_quantize_mxfp4_dual_inherits_base():
    assert issubclass(quantize_mxfp4_h16_dual, _QuantizeMxfp4Base)


def test_quantize_mxfp4_dual_flops():
    M, N = 3072, 3072
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M, N))
    assert model.flops() == 4 * M * N


def test_quantize_mxfp4_dual_bytes():
    M, N = 3072, 3072
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M, N))
    bpe_in = 2  # BF16
    bpe_fp4 = 0.5
    bpe_scale = 1
    expected = (
        M * N * bpe_in  # read
        + 2 * M * N * bpe_fp4  # write FP4 (dual)
        + 2 * max(1, M * N // MXFP4_GROUP_SIZE) * bpe_scale  # write scales (dual)
    )
    assert model.bytes() == expected


# ===========================================================================
# name2bpe for FP4 types
# ===========================================================================


def test_name2bpe_fp4_returns_half():
    assert name2bpe("fp4") == 0.5


def test_name2bpe_float4_e2m1fn_x2_returns_one():
    assert name2bpe("c10::Float4_e2m1fn_x2") == 1


def test_name2bpe_e8m0_scale():
    assert name2bpe("c10::Float8_e8m0fnu") == 1


def test_mxfp4_group_size_is_32():
    assert MXFP4_GROUP_SIZE == 32


# ===========================================================================
# _suffix_for_op helper
# ===========================================================================


def test_suffix_for_gemm():
    assert _suffix_for_op("primus_turbo::gemm_fp4_impl") == "gemm"


def test_suffix_for_a6w4():
    assert _suffix_for_op("primus_flux::a6w4_mm_qa") == "a6w4"
    assert _suffix_for_op("primus_flux::a6w4_mm_qa_prequant_w") == "a6w4"


# ===========================================================================
# tree_postprocess_extension — pseudo-op injection
# ===========================================================================


class _FakeTree:
    """Minimal mock of TraceToTree for testing tree_postprocess_extension."""

    def __init__(self):
        self.events = []
        self.events_by_uid = {}
        self.name2event_uids = {}
        self.seq_num2event_uids_map = {}

    def add_event(self, event):
        uid = len(self.events)
        event["UID"] = uid
        self.events.append(event)
        self.events_by_uid[uid] = event
        name = event.get("name", "")
        self.name2event_uids.setdefault(name, []).append(uid)
        return uid

    def get_UID2event(self, uid):
        return self.events_by_uid[uid]

    def get_subtree_bwd_events(self, uid):
        return self.events_by_uid[uid].get("bwd_events", [])


def _make_tree_with_fwd_bwd():
    """Build a minimal tree with one _MXFP4LinearFunction forward event
    containing a gemm_fp4_impl child, and a linked backward with two gemms."""
    tree = _FakeTree()

    # Forward parent
    fwd_parent = {
        "name": "_MXFP4LinearFunction",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 100,
        "dur": 50,
        "args": {"Sequence number": 1},
        "children": [],
        "bwd_events": [],
    }
    fwd_uid = tree.add_event(fwd_parent)

    # Forward GEMM child
    fwd_gemm = {
        "name": "primus_turbo::gemm_fp4_impl",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 110,
        "dur": 20,
        "args": {
            "Input Dims": [
                [1024, 512],
                [1024, 32],
                [],
                [2048, 512],
                [2048, 32],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            "Input type": [
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "Scalar",
                "c10::Float4_e2m1fn_x2",
                "c10::Float8_e8m0fnu",
                "Scalar",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "Input Strides": [[] for _ in range(14)],
            "Concrete Inputs": [""] * 14,
            "Sequence number": 1,
            "External id": 1,
        },
        "parent": fwd_uid,
        "children": [],
        "gpu_events": [],
    }
    gemm_uid = tree.add_event(fwd_gemm)
    fwd_parent["children"].append(gemm_uid)

    # Backward parent
    bwd_parent = {
        "name": "_MXFP4LinearFunctionBackward",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 200,
        "dur": 80,
        "args": {"Sequence number": 2},
        "children": [],
    }
    bwd_uid = tree.add_event(bwd_parent)
    fwd_parent["bwd_events"] = [bwd_uid]

    # Backward xgrad GEMM
    bwd_xgrad = {
        "name": "primus_turbo::gemm_fp4_impl",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 210,
        "dur": 20,
        "args": dict(fwd_gemm["args"]),
        "parent": bwd_uid,
        "children": [],
        "gpu_events": [],
    }
    xgrad_uid = tree.add_event(bwd_xgrad)
    bwd_parent["children"].append(xgrad_uid)

    # Backward wgrad GEMM
    bwd_wgrad = {
        "name": "primus_turbo::gemm_fp4_impl",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 240,
        "dur": 20,
        "args": dict(fwd_gemm["args"]),
        "parent": bwd_uid,
        "children": [],
        "gpu_events": [],
    }
    wgrad_uid = tree.add_event(bwd_wgrad)
    bwd_parent["children"].append(wgrad_uid)

    return tree, fwd_uid, gemm_uid, bwd_uid, xgrad_uid, wgrad_uid


def test_tree_postprocess_creates_fwd_pseudo_op():
    tree, fwd_uid, gemm_uid, _, _, _ = _make_tree_with_fwd_bwd()
    tree_postprocess_extension(tree)

    # The forward gemm should now be a child of a pseudo op
    gemm = tree.get_UID2event(gemm_uid)
    pseudo_uid = gemm["parent"]
    pseudo = tree.get_UID2event(pseudo_uid)
    assert pseudo["name"] == "_MXFP4Linear_yfwd_gemm"
    assert pseudo["args"]["Pseudo op"] is True


def test_tree_postprocess_creates_bwd_pseudo_ops():
    tree, _, _, _, xgrad_uid, wgrad_uid = _make_tree_with_fwd_bwd()
    tree_postprocess_extension(tree)

    xgrad = tree.get_UID2event(xgrad_uid)
    xgrad_pseudo = tree.get_UID2event(xgrad["parent"])
    assert xgrad_pseudo["name"] == "_MXFP4LinearBackward_xgrad_gemm"
    assert xgrad_pseudo["args"]["Pseudo op"] is True

    wgrad = tree.get_UID2event(wgrad_uid)
    wgrad_pseudo = tree.get_UID2event(wgrad["parent"])
    assert wgrad_pseudo["name"] == "_MXFP4LinearBackward_wgrad_gemm"
    assert wgrad_pseudo["args"]["Pseudo op"] is True


def test_tree_postprocess_pseudo_op_inherits_dims():
    tree, _, gemm_uid, _, _, _ = _make_tree_with_fwd_bwd()
    orig_dims = tree.get_UID2event(gemm_uid)["args"]["Input Dims"]
    tree_postprocess_extension(tree)

    gemm = tree.get_UID2event(gemm_uid)
    pseudo = tree.get_UID2event(gemm["parent"])
    assert pseudo["args"]["Input Dims"] == orig_dims


def test_tree_postprocess_no_mxfp4_events_is_noop():
    tree = _FakeTree()
    tree.add_event({"name": "aten::mm", "args": {}, "children": []})
    tree_postprocess_extension(tree)
    assert len(tree.events) == 1


def test_tree_postprocess_rewires_parent():
    """After injection, the fwd parent's children list contains the pseudo op UID,
    not the original gemm UID."""
    tree, fwd_uid, gemm_uid, _, _, _ = _make_tree_with_fwd_bwd()
    tree_postprocess_extension(tree)
    fwd_event = tree.get_UID2event(fwd_uid)
    assert gemm_uid not in fwd_event["children"]
    gemm = tree.get_UID2event(gemm_uid)
    pseudo_uid = gemm["parent"]
    assert pseudo_uid in fwd_event["children"]
