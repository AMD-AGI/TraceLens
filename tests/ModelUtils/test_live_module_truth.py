###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The live module decides which submodules exist; the parse only says what COULD.

An ``__init__`` guarded by a config flag constructs one arm or the other, and a
forward that picks between two submodules parses as both calls. The checkpoint in
front of us was built exactly one way, so the arm it does not have is not drawn.

Kimi is the case this was found on: ``use_full_rank_gate`` is true, so every one of
its 93 layers owns ``self_attn.g_proj`` and none owns ``g_a_proj``/``g_b_proj`` --
yet both were drawn, and the phi over them reported an arm of a rank the model never
produces. The same question one level up decides ``mlp`` vs ``block_sparse_moe``.

Two things are deliberately NOT treated as missing modules, both measured rather
than guessed -- the first draft of this rule got both wrong:

* a METHOD the forward calls (Kimi's ``moe_infer``). ``named_modules()`` lists
  modules, so it never lists a method; its absence from the live tree is no
  evidence, and pruning on it deleted 22 live nodes.
* a scalar the constructor assigns (``self.use_full_rank_gate = config...get(...)``).
  It is recorded like any other ``__init__`` assignment and is not in the live tree
  because it is not a module.
"""

from __future__ import annotations

import textwrap
import types

import pytest

from TraceLens.ModelUtils.ast_analyze import analyze_sources
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import SUBMODULE_OMITTED, build_block_node
from TraceLens.ModelUtils.extract import (
    _absent_submodule_paths,
    _live_module_overrides,
    _live_path_set,
)

_SOURCE = textwrap.dedent("""
    class Gate(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            return self.proj(x)

    class Attn(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.use_full_rank_gate = config.get("use_full_rank_gate", False)
            self.q_proj = nn.Linear(8, 8)
            if self.use_full_rank_gate:
                self.g_proj = nn.Linear(8, 8)
            else:
                self.g_a_proj = nn.Linear(8, 2)
                self.g_b_proj = nn.Linear(2, 8)

        def forward(self, x):
            q = self.q_proj(x)
            if self.use_full_rank_gate:
                g = self.g_proj(x)
            else:
                g = self.g_b_proj(self.g_a_proj(x))
            return q * g

    class Moe(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gate = Gate(dim)

        def moe_infer(self, x):
            scaled = x * 2
            return scaled + 1

        def forward(self, x):
            g = self.gate(x)
            return self.moe_infer(g)
""")


@pytest.fixture(scope="module")
def registry():
    return analyze_sources({"m.py": _SOURCE}).class_registry


@pytest.fixture(scope="module")
def spec(registry):
    return types.SimpleNamespace(class_registry=registry, block_components=[])


@pytest.fixture(scope="module")
def basic_ops():
    return BasicOpFilter.for_detailed()


def _absent(spec, basic_ops, comp_attr, comp_class, live):
    return _absent_submodule_paths(
        spec, comp_attr, comp_class, _live_path_set(live), basic_ops
    )


class TestAnArmTheCheckpointDoesNotHave:
    def test_the_unbuilt_gate_pair_is_omitted(self, spec, basic_ops) -> None:
        live = {"self_attn": "Attn", "self_attn.q_proj": "L", "self_attn.g_proj": "L"}
        absent = _absent(spec, basic_ops, "self_attn", "Attn", live)
        assert absent == {
            "g_a_proj": SUBMODULE_OMITTED,
            "g_b_proj": SUBMODULE_OMITTED,
        }

    def test_the_other_build_omits_the_other_arm(self, spec, basic_ops) -> None:
        """Flip which arm the checkpoint was built with; the answer flips too."""
        live = {
            "self_attn": "Attn",
            "self_attn.q_proj": "L",
            "self_attn.g_a_proj": "L",
            "self_attn.g_b_proj": "L",
        }
        absent = _absent(spec, basic_ops, "self_attn", "Attn", live)
        assert absent == {"g_proj": SUBMODULE_OMITTED}

    def test_a_module_that_is_there_is_left_alone(self, spec, basic_ops) -> None:
        live = {
            "self_attn": "Attn",
            "self_attn.q_proj": "L",
            "self_attn.g_proj": "L",
            "self_attn.g_a_proj": "L",
            "self_attn.g_b_proj": "L",
        }
        assert _absent(spec, basic_ops, "self_attn", "Attn", live) == {}


class TestWhatIsNotAMissingModule:
    def test_a_method_the_forward_calls(self, spec, basic_ops, registry) -> None:
        """``moe_infer`` is a method: named_modules() never lists one."""
        tree = build_block_node(
            attr_name="block_sparse_moe",
            class_name="Moe",
            registry=registry,
            basic_ops=basic_ops,
            infer_init_steps=True,
        )
        drawn = {getattr(c, "attr_name", "") for c in tree.children or []}
        assert "moe_infer" in drawn, f"fixture no longer exercises the guard: {drawn}"

        live = {
            "block_sparse_moe": "Moe",
            "block_sparse_moe.gate": "Gate",
            "block_sparse_moe.gate.proj": "L",
        }
        assert _absent(spec, basic_ops, "block_sparse_moe", "Moe", live) == {}

    def test_a_scalar_the_constructor_assigns(self, spec, basic_ops) -> None:
        """``use_full_rank_gate`` is an __init__ assignment, but not a module."""
        live = {"self_attn": "Attn", "self_attn.q_proj": "L", "self_attn.g_proj": "L"}
        absent = _absent(spec, basic_ops, "self_attn", "Attn", live)
        assert "use_full_rank_gate" not in absent


class TestAModuleListIsAddressedByIndex:
    def test_an_indexed_element_matches_the_bare_attr(self) -> None:
        """``named_modules()`` says ``experts.0.w1``; the parsed tree says ``experts.w1``."""
        paths = _live_path_set({"experts.0.w1": "L", "experts.1.w1": "L"})
        assert "experts.w1" in paths
        assert "experts.0.w1" in paths

    def test_a_name_that_merely_looks_numeric_is_untouched(self) -> None:
        assert "block2.w" in _live_path_set({"block2.w": "L"})


class TestWholeComponents:
    def test_a_component_the_layer_was_not_built_with(
        self, registry, basic_ops
    ) -> None:
        components = [
            types.SimpleNamespace(attr_name="self_attn", class_name="Attn"),
            types.SimpleNamespace(attr_name="block_sparse_moe", class_name="Moe"),
        ]
        spec = types.SimpleNamespace(
            class_registry=registry, block_components=components
        )
        live = {"self_attn": "Attn", "self_attn.q_proj": "L", "self_attn.g_proj": "L"}
        omitted, _overrides = _live_module_overrides(spec, live, basic_ops)
        assert omitted == frozenset({"block_sparse_moe"})

    def test_no_live_tree_means_no_claim(self, registry, basic_ops) -> None:
        """Without descriptors we know nothing, so nothing is pruned."""
        spec = types.SimpleNamespace(
            class_registry=registry,
            block_components=[
                types.SimpleNamespace(attr_name="self_attn", class_name="Attn")
            ],
        )
        assert _live_module_overrides(spec, {}, basic_ops) == (frozenset(), {})
