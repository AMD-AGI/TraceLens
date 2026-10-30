###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A model's repeated structure is not always one uniform stack.

ConvNeXt is four stages (``depths=[3,3,27,3]``) of a shared ``ConvNextLayer`` at
growing channel widths -- a NON-uniform nesting the single-decoder-stack model
cannot express. These tests pin the reconstruction of that nesting from the flat
live ``MetaModuleGroup`` list, and the conservative predicate that decides when a
model is non-uniform, WITHOUT instantiating anything: the tree is rebuilt from
dotted paths alone, and the predicate must never misread a uniform (even MoE)
transformer as non-uniform.
"""

from __future__ import annotations

from TraceLens.ModelUtils.extract import (
    ArchitectureSpec,
    _build_nested_groups,
    _is_non_uniform,
    reconcile_live_module_groups,
)
from TraceLens.ModelUtils.meta_trace import MetaModuleGroup


def _grp(path: str, length: int, element_class: str) -> MetaModuleGroup:
    return MetaModuleGroup(
        path=path, length=length, element_class=element_class, signatures=()
    )


# What walk_meta_module_tree reports for facebook/convnext-base-224-22k.
CONVNEXT_GROUPS = [
    _grp("convnext.encoder.stages", 4, "ConvNextStage"),
    _grp("convnext.encoder.stages.0.layers", 3, "ConvNextLayer"),
    _grp("convnext.encoder.stages.1.layers", 3, "ConvNextLayer"),
    _grp("convnext.encoder.stages.2.layers", 27, "ConvNextLayer"),
    _grp("convnext.encoder.stages.3.layers", 3, "ConvNextLayer"),
    _grp("convnext.encoder.stages.1.downsampling_layer", 2, "ConvNextLayerNorm"),
    _grp("convnext.encoder.stages.2.downsampling_layer", 2, "ConvNextLayerNorm"),
    _grp("convnext.encoder.stages.3.downsampling_layer", 2, "ConvNextLayerNorm"),
]


def test_build_nested_groups_reconstructs_the_convnext_tree():
    roots = _build_nested_groups(CONVNEXT_GROUPS)
    # One root: the 4-stage container.
    assert len(roots) == 1
    stages = roots[0]
    assert stages.path == "convnext.encoder.stages"
    assert stages.count == 4
    assert stages.element_class == "ConvNextStage"

    children = {child.path: child for child in stages.children}
    # The four per-stage `layers` lists fold into ONE collapsed child, carrying each
    # stage's own depth in per_index_counts.
    layers = children["convnext.encoder.stages.*.layers"]
    assert layers.element_class == "ConvNextLayer"
    assert layers.per_index_counts == [3, 3, 27, 3]
    # The downsampling list is absent from stage 0 -> leading 0, sized to the parent's
    # own repeat count (4).
    downs = children["convnext.encoder.stages.*.downsampling_layer"]
    assert downs.per_index_counts == [0, 2, 2, 2]


def test_is_non_uniform_fires_for_convnext():
    assert _is_non_uniform(_build_nested_groups(CONVNEXT_GROUPS)) is True


def test_is_non_uniform_ignores_a_uniform_llm():
    # A plain decoder stack: one ModuleList, no nesting.
    groups = [_grp("model.layers", 32, "LlamaDecoderLayer")]
    nested = _build_nested_groups(groups)
    assert len(nested) == 1 and nested[0].per_index_counts is None
    assert _is_non_uniform(nested) is False


def test_is_non_uniform_ignores_a_uniform_moe_stack():
    # Every layer owns an identical 256-expert ModuleList -> uniform per-position count.
    groups = [_grp("model.layers", 32, "DecoderLayer")] + [
        _grp(f"model.layers.{i}.mlp.experts", 256, "Expert") for i in range(32)
    ]
    assert _is_non_uniform(_build_nested_groups(groups)) is False


def test_is_non_uniform_ignores_dense_replace_moe():
    # Experts absent in the first two (dense) layers, identical where present:
    # per_index_counts = [0, 0, 256, ...] -> non-zero counts all equal -> uniform.
    groups = [_grp("model.layers", 8, "DecoderLayer")] + [
        _grp(f"model.layers.{i}.mlp.experts", 256, "Expert") for i in range(2, 8)
    ]
    nested = _build_nested_groups(groups)
    experts = nested[0].children[0]
    assert experts.per_index_counts == [0, 0, 256, 256, 256, 256, 256, 256]
    assert _is_non_uniform(nested) is False


def test_reconcile_routes_convnext_to_nested_and_leaves_single_stack_empty():
    spec = ArchitectureSpec(name="convnext-base", model_type="convnext")
    reconcile_live_module_groups(spec, CONVNEXT_GROUPS)
    assert spec.nested_module_groups  # populated
    assert spec.nested_module_groups[0].path == "convnext.encoder.stages"
    # The single uniform decoder-stack fields are left untouched.
    assert spec.layer_variants == []
    assert spec.decoder_class is None


def test_reconcile_leaves_a_uniform_llm_on_the_single_stack_path():
    spec = ArchitectureSpec(name="llama", model_type="llama")
    spec.num_hidden_layers = 32
    reconcile_live_module_groups(spec, [_grp("model.layers", 32, "LlamaDecoderLayer")])
    # No nested tree -> the existing single-stack reconcile ran.
    assert spec.nested_module_groups == []
    assert spec.decoder_class == "LlamaDecoderLayer"
