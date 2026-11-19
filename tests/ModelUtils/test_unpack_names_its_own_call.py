###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A tuple unpack names the call it unpacks, not some earlier op.

Kimi's MLA attention splits once and then reassigns the same names from a call
that produces no op of its own::

    k_pass, value_states = torch.split(k_pass, [nope, v], dim=-1)   # line 432
    ...
    key_states = torch.cat((k_pass, k_rot), dim=-1)                 # line 440
    if past_key_values is not None:                                 # line 442
        key_states, value_states = past_key_values.update(key_states, value_states, i)

``past_key_values`` is a forward PARAMETER, so ``.update(...)`` is not captured
as an op. Resolving a producer for that unpack walked back through its arguments
and landed on the split -- whose first slice was then relabelled ``key_states``.
The split then looked like a producer of ``key_states``, so the branch phi joined
it, and the attention read a key from before the rotary halves were concatenated.

An inline op's slices are named by the unpack of ITS OWN call, which is a span of
lines rather than one: a chained expression can put the multi-output call several
lines below the assignment it belongs to.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source


def _split_output_names(source: str, cls: str = "M") -> dict[str, tuple[str, ...]]:
    analysis = analyze_source(textwrap.dedent(source), config={"hidden_size": 8})
    structure = analysis.class_registry[cls]
    return {
        op.attr_name: tuple(op.output_names)
        for op in structure.forward_operations.values()
        if op.label.strip().lower() in {"split", "chunk", "unbind"}
    }


class TestUnpackNaming:
    def test_a_later_unpack_does_not_rename_an_earlier_split(self) -> None:
        names = _split_output_names("""
            class M(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.proj = nn.Linear(8, 8)

                def forward(self, x, cache):
                    a = self.proj(x)
                    k_pass, value_states = torch.split(a, [4, 4], dim=-1)
                    key_states = torch.cat((k_pass, k_pass), dim=-1)
                    if cache is not None:
                        key_states, value_states = cache.update(key_states, value_states)
                    return key_states
            """)
        assert names, "no split op captured"
        for attr, output_names in names.items():
            assert output_names in ((), ("k_pass", "value_states")), (
                attr,
                output_names,
            )

    def test_an_unpack_of_the_split_itself_still_names_it(self) -> None:
        names = _split_output_names("""
            class M(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.proj = nn.Linear(8, 8)

                def forward(self, x):
                    a = self.proj(x)
                    first, second = torch.split(a, [4, 4], dim=-1)
                    return first + second
            """)
        assert ("first", "second") in names.values(), names

    def test_a_multi_line_chained_unpack_still_names_its_own_call(self) -> None:
        """GLM's vision qkv: the ``unbind`` sits lines below the assignment."""
        names = _split_output_names("""
            class M(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.qkv = nn.Linear(8, 24)

                def forward(self, x):
                    query_states, key_states, value_states = (
                        self.qkv(x)
                        .reshape(1, 3, 2, 4)
                        .permute(1, 0, 2, 3)
                        .unbind(0)
                    )
                    return query_states + key_states + value_states
            """)
        assert ("query_states", "key_states", "value_states") in names.values(), names
