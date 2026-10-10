###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A constructor parameter nobody overrides has the value it defaults to.

A default is NOT a parameter's value in general -- any caller may pass something
else, which is why inferring one was previously refused outright. But when every
construction site in the model leaves a parameter out, the default is what the
model builds with. That is evidence, not a guess.

GPT-2 turns on a whole arm of its attention this way::

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        ...
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

The arm that WAS drawn took the 2x-wide projection with it, which is how a
12-head attention came to report 36 heads.

The subtle part: the one call passing ``is_cross_attention=True`` sits inside
``if config.add_cross_attention:`` -- an arm the checkpoint switches off.
Counting it would make the parameter look overridden by a call that never runs,
which is the very question being asked, so unreachable construction sites are
skipped.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import _settled_ctor_defaults

SOURCE = """
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.is_cross_attention = is_cross_attention


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config=config, layer_idx=0)
        if config.add_cross_attention:
            self.crossattention = Attention(config, True, layer_idx=0)
"""


def _settled(config: dict) -> dict:
    return _settled_ctor_defaults(ast.parse(SOURCE), config)


class TestAParameterNobodyPasses:
    def test_it_takes_its_default(self) -> None:
        """``layer_idx`` is passed at every site, so only the other settles."""
        assert _settled({"add_cross_attention": False})["Attention"] == {
            "is_cross_attention": False
        }

    def test_an_unreachable_call_does_not_override_it(self) -> None:
        """The only caller passing it is one the config switches off."""
        settled = _settled({"add_cross_attention": False})
        assert settled["Attention"]["is_cross_attention"] is False

    def test_a_reachable_call_does_override_it(self) -> None:
        """With cross-attention on, a caller really does pass it."""
        settled = _settled({"add_cross_attention": True})
        assert "is_cross_attention" not in settled.get("Attention", {})


class TestWhatIsNotSettled:
    def test_a_positional_argument_counts_as_passed(self) -> None:
        source = """
class A:
    def __init__(self, config, flag=False):
        self.flag = flag


class B:
    def __init__(self, config):
        self.a = A(config, True)
"""
        settled = _settled_ctor_defaults(ast.parse(source), {})
        assert "flag" not in settled.get("A", {}), settled

    def test_a_class_nobody_builds_says_nothing(self) -> None:
        source = """
class A:
    def __init__(self, config, flag=False):
        self.flag = flag
"""
        assert _settled_ctor_defaults(ast.parse(source), {}) == {}

    def test_a_default_that_is_not_a_literal_is_skipped(self) -> None:
        source = """
class A:
    def __init__(self, config, flag=compute()):
        self.flag = flag


class B:
    def __init__(self, config):
        self.a = A(config)
"""
        settled = _settled_ctor_defaults(ast.parse(source), {})
        assert "flag" not in settled.get("A", {}), settled
