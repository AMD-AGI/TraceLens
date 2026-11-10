###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A ``self.<attr>`` built from a scalar LOCAL stays off the tensor edges.

SGLang's attention settles its split sizes through an intermediate local
before storing them on ``self``::

    attn_tp_size = get_parallel().attn_tp_size
    self.num_heads = self.total_num_heads // attn_tp_size
    self.q_size = self.num_heads * self.head_dim
    ...
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

``attn_tp_size`` can never be resolved to a concrete value (``get_parallel()``
is an external, unresolvable call), and it is a bare local, not a
``self.<attr>`` -- so ``self.num_heads``, and in turn ``self.q_size``, never
qualified as a "scalar setting" and fell through as an unresolved TENSOR
attribute, materialized as a Constant input wired straight onto the ``split``.
The type-check then reported the split fed two tensor operands instead of one.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _SCALAR_SETTING,
    _self_config_values,
)


def _init(source: str) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(source))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )


_SPLIT_SIZE_VIA_LOCAL = """
    class Attention:
        def __init__(self, config):
            attn_tp_size = get_parallel().attn_tp_size
            self.total_num_heads = config.num_attention_heads
            self.num_heads = self.total_num_heads // attn_tp_size
            self.head_dim = config.head_dim
            self.q_size = self.num_heads * self.head_dim
            self.weight = weight
"""


class TestAScalarThreadedThroughAnUnresolvableLocal:
    def test_the_local_dependent_attr_is_recorded_as_a_setting(self) -> None:
        values = _self_config_values(_init(_SPLIT_SIZE_VIA_LOCAL), {})
        assert values["num_heads"] is _SCALAR_SETTING

    def test_an_attr_derived_from_that_attr_is_also_a_setting(self) -> None:
        values = _self_config_values(_init(_SPLIT_SIZE_VIA_LOCAL), {})
        assert values["q_size"] is _SCALAR_SETTING

    def test_a_concretely_resolved_attr_still_resolves(self) -> None:
        values = _self_config_values(
            _init(_SPLIT_SIZE_VIA_LOCAL), {"num_attention_heads": 32, "head_dim": 128}
        )
        assert values["total_num_heads"] == 32

    def test_a_bare_passed_in_tensor_is_never_fabricated_as_a_setting(self) -> None:
        """``self.weight = weight`` must stay a genuine tensor attribute."""
        values = _self_config_values(_init(_SPLIT_SIZE_VIA_LOCAL), {})
        assert "weight" not in values
