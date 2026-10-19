###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Top-k perf models seen in a GLM-5.3 SGLang decode trace."""

import pytest

from TraceLens.PerfModel.extensions import (
    attention_perf_model_extensions as attn_ext,
    moe_perf_model_extensions as moe_ext,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    resolve_perf_model_class,
)

_DECODE_ANNOTATION = (
    "execute_8_context_0(sq0sk0sqsq0sqsk0)" "_generation_8(sq8sk815046sqsq8sqsk815046)"
)


def _event(name, dims, types, annotation=None):
    event = {
        "name": name,
        "args": {
            "Input Dims": dims,
            "Input type": types,
            "Input Strides": [[1] * len(d) for d in dims],
            "Concrete Inputs": [""] * len(dims),
        },
    }
    if annotation is not None:
        event["annotation"] = annotation
    return event


def _fast_topk_event(annotation=_DECODE_ANNOTATION):
    return _event(
        "sgl_kernel::fast_topk_transform_fused",
        [(8, 1048640), (8,), (8, 2048), (8, 1048580), (9,), ()],
        ["float", "int", "int", "int", "int", ""],
        annotation=annotation,
    )


@pytest.mark.parametrize(
    "op_name,model_class,category",
    [
        (
            "sglang_profiler::topk__post_process_topk_ids",
            moe_ext.sglang_fused_append_shared_experts,
            "MoE_aux",
        ),
        (
            "sgl_kernel::fast_topk_transform_fused",
            attn_ext.sgl_kernel_fast_topk_transform_fused,
            "InferenceAttention",
        ),
    ],
)
def test_op_mapping_and_category(op_name, model_class, category):
    assert resolve_perf_model_class(op_name) is model_class
    assert categorize_torch_op({"name": op_name}) == category


def test_post_process_topk_ids_uses_ids_and_weights():
    event = _event(
        "sglang_profiler::topk__post_process_topk_ids",
        [(8, 8), (8, 8), (8, 256), ()],
        ["int", "float", "c10::BFloat16", ""],
    )
    model = resolve_perf_model_class(event["name"])(event)
    assert model.flops() == 8 * 8
    assert model.bytes() == 2 * (8 * 8 * 4 + 8 * 8 * 4)
    assert model.get_compute_precision() == "fp32"


def test_fast_topk_uses_annotation_context_not_padded_width():
    model = attn_ext.sgl_kernel_fast_topk_transform_fused(_fast_topk_event())
    assert model.param_details["valid_scores"] == 815046
    assert model.flops() == 815046
    assert model.bytes() == 815046 * 4 + 8 * (2048 + 2048) * 4
    assert model.get_compute_precision() == "fp32"
    assert model.get_maf_type() == "vector"


def test_fast_topk_short_rows_gather_only_valid_entries():
    annotation = (
        "execute_2_context_0(sq0sk0sqsq0sqsk0)" "_generation_2(sq2sk1000sqsq2sqsk1000)"
    )
    event = _fast_topk_event(annotation)
    event["args"]["Input Dims"] = [(2, 4160), (2,), (2, 2048), (2, 4100), (3,), ()]
    model = attn_ext.sgl_kernel_fast_topk_transform_fused(event)
    assert model.bytes() == 1000 * 4 + 2 * (500 + 2048) * 4


def test_fast_topk_counts_causal_prefill_rows():
    annotation = (
        "execute_4_context_1(sq4sk10sqsq16sqsk40)" "_generation_0(sq0sk0sqsq0sqsk0)"
    )
    model = attn_ext.sgl_kernel_fast_topk_transform_fused(_fast_topk_event(annotation))
    assert model.param_details["valid_scores"] == 7 + 8 + 9 + 10


def test_fast_topk_without_annotation_has_no_estimate():
    model = attn_ext.sgl_kernel_fast_topk_transform_fused(_fast_topk_event("NA"))
    assert model.flops() is None
    assert model.bytes() is None
