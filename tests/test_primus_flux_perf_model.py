###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for Primus Turbo / Flux MXFP4 ops:
  - primus_flux::a6w4_mm_qa -> GEMM
  - primus_flux::a6w4_mm_qa_prequant_w -> GEMM
  - primus_flux::quantize_mxfp4_h16_dual -> UnaryElementwise
  - tree_postprocess_extension (fwd/bwd pseudo-op labeling)
"""

from TraceLens.PerfModel.perf_model import GEMM, SDPA
from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
    flux_flydsl_attn_fwd,
    flux_flydsl_attn_fwd_split,
    flux_flydsl_attn_bwd,
    flux_flydsl_attn_bwd_split,
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


def test_quantize_mxfp4_base_flops():
    """_QuantizeMxfp4Base.flops() returns 2 * nelems (single quant)."""
    model = quantize_mxfp4_h16_dual(_quantize_mxfp4_event(M=32, N=64))
    base_flops = _QuantizeMxfp4Base.flops(model)
    assert base_flops == 2 * 32 * 64


def test_quantize_mxfp4_base_bytes_none_bpe():
    """_QuantizeMxfp4Base.bytes() returns None when bpe_in is None."""
    event = _quantize_mxfp4_event()
    event["args"]["Input type"][0] = "unknown_dtype"
    model = quantize_mxfp4_h16_dual(event)
    assert _QuantizeMxfp4Base.bytes(model) is None


def test_quantize_mxfp4_dual_bytes_none_bpe():
    """quantize_mxfp4_h16_dual.bytes() returns None when bpe_in is None."""
    event = _quantize_mxfp4_event()
    event["args"]["Input type"][0] = "unknown_dtype"
    model = quantize_mxfp4_h16_dual(event)
    assert model.bytes() is None


def test_inject_pseudo_op_no_parent():
    """_inject_pseudo_op is a no-op when the event has no parent."""
    from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
        _inject_pseudo_op,
    )

    tree = _FakeTree()
    evt = {
        "name": "primus_turbo::gemm_fp4_impl",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 100,
        "dur": 10,
        "args": {
            "Input Dims": [],
            "Input type": [],
            "Input Strides": [],
            "Concrete Inputs": [],
        },
        "children": [],
        "gpu_events": [],
    }
    tree.add_event(evt)
    before = len(tree.events)
    _inject_pseudo_op(tree, evt, "test_pseudo")
    assert len(tree.events) == before


def test_inject_pseudo_op_child_not_in_parent_children():
    """_inject_pseudo_op appends to parent children when orig UID is not in the list."""
    from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
        _inject_pseudo_op,
    )

    tree = _FakeTree()
    parent = {
        "name": "parent",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 0,
        "dur": 100,
        "args": {},
        "children": [],
    }
    parent_uid = tree.add_event(parent)
    child = {
        "name": "primus_turbo::gemm_fp4_impl",
        "ph": "X",
        "cat": "cpu_op",
        "pid": 0,
        "tid": 0,
        "ts": 10,
        "dur": 10,
        "args": {
            "Input Dims": [],
            "Input type": [],
            "Input Strides": [],
            "Concrete Inputs": [],
            "Sequence number": 0,
            "External id": 0,
        },
        "parent": parent_uid,
        "children": [],
        "gpu_events": [],
    }
    tree.add_event(child)
    # Deliberately don't add child to parent["children"]
    _inject_pseudo_op(tree, child, "test_pseudo")
    assert len(parent["children"]) == 1


def test_collect_gemm_children_empty():
    """_collect_gemm_children returns empty list when no GEMM children."""
    from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
        _collect_gemm_children,
    )

    tree = _FakeTree()
    parent = {"name": "parent", "children": []}
    tree.add_event(parent)
    non_gemm = {"name": "aten::add", "ts": 1}
    uid = tree.add_event(non_gemm)
    parent["children"].append(uid)
    assert _collect_gemm_children(tree, parent) == []


def test_collect_gemm_descendants_nested():
    """_collect_gemm_descendants finds GEMMs nested multiple levels deep."""
    from TraceLens.PerfModel.extensions.primus_flux_perf_model_extensions import (
        _collect_gemm_descendants,
    )

    tree = _FakeTree()
    root = {"name": "root", "children": [], "ts": 0}
    tree.add_event(root)
    mid = {"name": "aten::view", "children": [], "ts": 1}
    mid_uid = tree.add_event(mid)
    root["children"].append(mid_uid)
    gemm = {"name": "primus_turbo::gemm_fp4_impl", "children": [], "ts": 2}
    gemm_uid = tree.add_event(gemm)
    mid["children"].append(gemm_uid)
    result = _collect_gemm_descendants(tree, root)
    assert len(result) == 1
    assert result[0]["name"] == "primus_turbo::gemm_fp4_impl"


# ===========================================================================
# FlydSL attention helpers
# ===========================================================================


def _flydsl_attn_fwd_event(B=32, S=512, H=24, D=128):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    return {
        "name": "flux_flydsl::attn_fwd",
        "args": {
            "Input Dims": [shape, shape, shape],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [stride, stride, stride],
            "Concrete Inputs": ["", "", ""],
        },
    }


def _flydsl_attn_fwd_split_event(B=32, S=512, H=24, D=128, split=256):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    return {
        "name": "flux_flydsl::attn_fwd_split",
        "args": {
            "Input Dims": [shape, shape, shape, []],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"],
            "Input Strides": [stride, stride, stride, []],
            "Concrete Inputs": ["", "", "", str(split)],
        },
    }


def _flydsl_attn_bwd_event(B=32, S=512, H=24, D=128):
    shape = [B, S, H, D]
    stride = [S * H * D, H * D, D, 1]
    lse_shape = [B, H, S]
    lse_stride = [H * S, S, 1]
    return {
        "name": "flux_flydsl::attn_bwd",
        "args": {
            "Input Dims": [shape, shape, shape, shape, shape, lse_shape],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "float",
            ],
            "Input Strides": [stride, stride, stride, stride, stride, lse_stride],
            "Concrete Inputs": ["", "", "", "", "", ""],
        },
    }


def _flydsl_attn_bwd_split_event(B=32, S=512, H=24, D=128):
    half_S = S // 2
    full_shape = [B, S, H, D]
    half_shape = [B, half_S, H, D]
    full_stride = [S * H * D, H * D, D, 1]
    half_stride = [half_S * H * D, H * D, D, 1]
    lse_shape = [B, H, S]
    lse_stride = [H * S, S, 1]
    return {
        "name": "flux_flydsl::attn_bwd_split",
        "args": {
            "Input Dims": [
                half_shape,
                half_shape,  # dOut_x, dOut_img
                full_shape,
                full_shape,
                full_shape,  # K, V, Out
                half_shape,
                half_shape,  # Q_x, Q_img
                lse_shape,  # LSE
            ],
            "Input type": [
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "c10::BFloat16",
                "float",
            ],
            "Input Strides": [
                half_stride,
                half_stride,
                full_stride,
                full_stride,
                full_stride,
                half_stride,
                half_stride,
                lse_stride,
            ],
            "Concrete Inputs": [""] * 8,
        },
    }


# ===========================================================================
# FlydSL attention registration
# ===========================================================================


def test_flydsl_attn_fwd_registered():
    assert "flux_flydsl::attn_fwd" in op_to_perf_model_class_map
    assert op_to_perf_model_class_map["flux_flydsl::attn_fwd"] is flux_flydsl_attn_fwd


def test_flydsl_attn_fwd_split_registered():
    assert "flux_flydsl::attn_fwd_split" in op_to_perf_model_class_map


def test_flydsl_attn_bwd_registered():
    assert "flux_flydsl::attn_bwd" in op_to_perf_model_class_map


def test_flydsl_attn_bwd_split_registered():
    assert "flux_flydsl::attn_bwd_split" in op_to_perf_model_class_map


# ===========================================================================
# FlydSL attn_fwd
# ===========================================================================


def test_flydsl_attn_fwd_inherits_sdpa():
    assert issubclass(flux_flydsl_attn_fwd, SDPA)


def test_flydsl_attn_fwd_category():
    assert flux_flydsl_attn_fwd.category == "SDPA_fwd"


def test_flydsl_attn_fwd_params():
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B=32, S=512, H=24, D=128))
    assert (
        model.B,
        model.N_Q,
        model.H_Q,
        model.N_KV,
        model.H_KV,
        model.d_h_qk,
        model.d_h_v,
    ) == (32, 512, 24, 512, 24, 128, 128)


def test_flydsl_attn_fwd_flops():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    expected = 4 * B * H * S * S * D  # non-causal
    assert model.flops() == expected


def test_flydsl_attn_fwd_bytes():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    bpe = 2  # BF16
    expected = (3 * B * S * H * D + B * S * H * D) * bpe  # Q+K+V read + O write
    assert model.bytes() == expected


# ===========================================================================
# FlydSL attn_fwd_split
# ===========================================================================


def test_flydsl_attn_fwd_split_same_flops_as_fwd():
    B, S, H, D = 32, 512, 24, 128
    fwd = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    split = flux_flydsl_attn_fwd_split(_flydsl_attn_fwd_split_event(B, S, H, D))
    assert fwd.flops() == split.flops()


def test_flydsl_attn_fwd_split_same_bytes_as_fwd():
    B, S, H, D = 32, 512, 24, 128
    fwd = flux_flydsl_attn_fwd(_flydsl_attn_fwd_event(B, S, H, D))
    split = flux_flydsl_attn_fwd_split(_flydsl_attn_fwd_split_event(B, S, H, D))
    assert fwd.bytes() == split.bytes()


# ===========================================================================
# FlydSL attn_bwd
# ===========================================================================


def test_flydsl_attn_bwd_inherits_sdpa():
    assert issubclass(flux_flydsl_attn_bwd, SDPA)


def test_flydsl_attn_bwd_category():
    assert flux_flydsl_attn_bwd.category == "SDPA_bwd"


def test_flydsl_attn_bwd_params():
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B=32, S=512, H=24, D=128))
    assert (model.B, model.N_Q, model.H_Q, model.N_KV) == (32, 512, 24, 512)


def test_flydsl_attn_bwd_flops():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    fwd_flops = 4 * B * H * S * S * D
    expected = int(fwd_flops * 2.5)  # bwd = 5/2 * fwd (5 matmuls with flash recompute)
    assert model.flops() == expected


def test_flydsl_attn_bwd_bytes():
    B, S, H, D = 32, 512, 24, 128
    model = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    bpe = 2
    expected = 7 * B * S * H * D * bpe  # Q,K,V,dO reads + dQ,dK,dV writes
    assert model.bytes() == expected


# ===========================================================================
# FlydSL attn_bwd_split
# ===========================================================================


def test_flydsl_attn_bwd_split_category():
    assert flux_flydsl_attn_bwd_split.category == "SDPA_bwd"


def test_flydsl_attn_bwd_split_uses_full_seq():
    model = flux_flydsl_attn_bwd_split(
        _flydsl_attn_bwd_split_event(B=32, S=512, H=24, D=128)
    )
    assert model.N_Q == 512
    assert model.N_KV == 512


def test_flydsl_attn_bwd_split_same_flops_as_bwd():
    B, S, H, D = 32, 512, 24, 128
    bwd = flux_flydsl_attn_bwd(_flydsl_attn_bwd_event(B, S, H, D))
    split = flux_flydsl_attn_bwd_split(_flydsl_attn_bwd_split_event(B, S, H, D))
    assert bwd.flops() == split.flops()
