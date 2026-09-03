###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the patchability gate and symbol demangling.

Covers the cheap, filesystem-free classification layer
(:func:`classify_patchability`) and the symbol-name normalization
(:func:`base_symbol`) it relies on, including the PR #1154 cases: Tensile,
MIOpen, inductor Triton, and CK (demangled and mangled forms).
"""

from TraceLens.TraceUtils.kernel_source import classify_patchability
from TraceLens.TraceUtils.kernel_source.demangle import base_symbol


# --- base_symbol ------------------------------------------------------------
def test_base_symbol_plain_name():
    assert base_symbol("my_fused_kernel") == "my_fused_kernel"


def test_base_symbol_strips_return_type_template_and_namespace():
    assert base_symbol("void ns::sub::foo<int, 4>(float*, int)") == "foo"


def test_base_symbol_handles_anonymous_namespace():
    # PR #1154: an (anonymous namespace) kernel must not collapse to "void".
    assert (
        base_symbol("void (anonymous namespace)::softmax_kernel(float*)")
        == "softmax_kernel"
    )


def test_base_symbol_mangled_roundtrip():
    # Works via itanium-demangler or the c++filt / length-prefix fallbacks.
    assert base_symbol("_Z6kernelv") == "kernel"


def test_base_symbol_empty():
    assert base_symbol("") == ""
    assert base_symbol(None) == ""  # type: ignore[arg-type]


# --- classify_patchability: non-patchable categories ------------------------
def test_gate_tensile():
    verdict = classify_patchability("Cijk_Ailk_Bljk_SB_MT128x128")
    assert verdict.patchable is False
    assert verdict.kind == "tensile_precompiled"


def test_gate_miopen_by_op_name():
    verdict = classify_patchability(
        "some_device_kernel", op_name="aten::miopen_convolution"
    )
    assert verdict.patchable is False
    assert verdict.kind == "miopen_precompiled"


def test_gate_inductor_by_call_stack():
    verdict = classify_patchability(
        "triton_kernel",
        call_stack=["/root/.cache/torchinductor_x/abc/xyz.py(42): forward"],
    )
    assert verdict.patchable is False
    assert verdict.kind == "triton_inductor_generated"


def test_gate_inductor_by_name():
    for name in ("triton_poi_fused_add_0", "triton_red_sum_1", "triton_tem_mm_2"):
        assert classify_patchability(name).kind == "triton_inductor_generated"


def test_gate_ck_demangled_form():
    verdict = classify_patchability("ck_tile::fmha::kernel<Traits, 128>")
    assert verdict.patchable is False
    assert verdict.kind == "aiter_ck"


def test_gate_ck_does_not_misfire_on_lookalikes():
    # Names that merely end in "ck" must NOT be classified as CK.
    for name in ("block::foo", "unpack::bar", "flashck::baz"):
        assert classify_patchability(name).patchable is None


# --- classify_patchability: unknown (proceed) -------------------------------
def test_gate_unknown_returns_none():
    verdict = classify_patchability("elementwise_add_kernel")
    assert verdict.patchable is None
    assert verdict.kind == ""
