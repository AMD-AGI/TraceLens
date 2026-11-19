###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A decoder class is one class; its layers are not one thing.

DeepSeek-V4 builds a different compressor for layer 2 than for layer 3 and
none at all for layer 0, and a different router for the first three layers than
for the rest. Which it builds is settled entirely by the constructor:

    self.layer_type = config.layer_types[layer_idx]
    self.compressor = (
        COMPRESSOR_CLASSES[self.layer_type](config)
        if self.layer_type != "sliding_attention" else None
    )
    self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
    self.gate = HashRouter(config) if self.is_hash else TopKRouter(config)

yet the split could only be had by instantiating the whole model on a meta
device, because `layer_types` and `mlp_layer_types` arrive as ``None`` -- the
checkpoint omits them and the config CLASS derives them. With those lists in
hand, asking each class what IT constructs for one ``layer_idx`` gives the same
answer with nothing instantiated.

These tests pin that answer WITHOUT a live module tree, which is the point: a
graph whose per-layer truth depends on being able to build the model is a graph
that is wrong whenever the model cannot be built.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.extract import reconcile_live_module_groups
from TraceLens.ModelUtils.loader import load_model_spec

MODEL = "deepseek-ai/DeepSeek-V4-Flash"

#: What the live meta tree reports for this checkpoint, independently derived.
EXPECTED = {
    ("DeepseekV4HCACompressor", "DeepseekV4TopKRouter"): 20,
    ("DeepseekV4CSACompressor", "DeepseekV4TopKRouter"): 20,
    ("no compressor", "DeepseekV4HashRouter"): 2,
    ("DeepseekV4CSACompressor", "DeepseekV4HashRouter"): 1,
}


@pytest.fixture(scope="module")
def static_variants():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec(MODEL, detailed=True)
    # No live tree AT ALL -- the parse has to answer on its own.
    reconcile_live_module_groups(spec, None)
    return spec.layer_variants


def _key(variant) -> tuple[str, str]:
    return tuple(variant.label.split(" + "))


class TestTheParseSplitsTheStackOnItsOwn:
    def test_every_distinct_layer_is_its_own_variant(self, static_variants) -> None:
        assert len(static_variants) == len(EXPECTED), [v.label for v in static_variants]

    def test_each_one_covers_the_layers_it_should(self, static_variants) -> None:
        assert {_key(v): v.count for v in static_variants} == EXPECTED

    def test_the_first_three_layers_route_by_hash(self, static_variants) -> None:
        # The config class defaults to three `hash_moe` layers, then `moe`.
        hashed = {
            index
            for variant in static_variants
            if "HashRouter" in variant.label
            for index in variant.layer_indices
        }
        assert hashed == {0, 1, 2}

    def test_the_first_two_build_no_compressor_at_all(self, static_variants) -> None:
        absent = {
            index
            for variant in static_variants
            if variant.label.startswith("no compressor")
            for index in variant.layer_indices
        }
        assert absent == {0, 1}

    def test_the_rest_alternate_compressors(self, static_variants) -> None:
        hca = next(v for v in static_variants if "HCACompressor" in v.label)
        csa = next(
            v
            for v in static_variants
            if "CSACompressor" in v.label and "TopKRouter" in v.label
        )
        assert hca.layer_indices[:3] == [3, 5, 7]
        assert csa.layer_indices[:3] == [4, 6, 8]

    def test_every_layer_is_covered_exactly_once(self, static_variants) -> None:
        seen: list[int] = []
        for variant in static_variants:
            seen.extend(variant.layer_indices)
        assert sorted(seen) == list(range(43))

    def test_each_variant_carries_its_own_subtrees(self, static_variants) -> None:
        # Without these the renderer would draw one layer's submodules for all.
        for variant in static_variants:
            assert sorted(variant.component_trees) == [
                "mlp",
                "self_attn",
            ], variant.label
