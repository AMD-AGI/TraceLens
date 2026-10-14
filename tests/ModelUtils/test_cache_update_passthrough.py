###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A call that hands back the names it was given changes nothing we can draw.

Every attention in the gate set does this::

    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx
    )

``past_key_values`` is a forward PARAMETER, so ``.update`` has no body here and
produces no op. The generic assignment path still had to give both targets some
producer, and for an untraced call that is whichever one the expression walk
settled on -- for Kimi a ``torch.split`` ten lines earlier. That split then
looked like a second, competing source of ``key_states``, so a branch phi was
built over it and the attention read a key from before the rotary half was
concatenated on (head dim 256 where the source says 192).

A call that legitimately reassigns its arguments looks different:
``query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos,
sin)`` resolves to a node of its own. Three things separate them -- the receiver
is a forward parameter, no operation was emitted, and what came back is only an
echo of an argument.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source

_HEADER = """class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, hidden_states, past_key_values):
"""


def _ops(body: str) -> dict[str, str]:
    source = _HEADER + textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8)
    analysis = analyze_source(source, config={"hidden_size": 8})
    return {
        op.attr_name: op.label
        for op in analysis.class_registry["M"].forward_operations.values()
    }


class TestCacheUpdateIsAPassthrough:
    def test_no_phi_is_built_over_a_cache_update(self) -> None:
        ops = _ops("""
            a = self.proj(hidden_states)
            k_pass, value_states = torch.split(a, [4, 4], dim=-1)
            key_states = torch.cat((k_pass, k_pass), dim=-1)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states
                )
            return key_states + value_states
            """)
        merges = [attr for attr, label in ops.items() if label == "Merge"]
        assert merges == [], (merges, ops)

    def test_a_call_that_renames_what_it_is_given_still_merges(self) -> None:
        """Not the same shape: the targets are not the arguments."""
        ops = _ops("""
            a = self.proj(hidden_states)
            k_pass, value_states = torch.split(a, [4, 4], dim=-1)
            key_states = torch.cat((k_pass, k_pass), dim=-1)
            if past_key_values is not None:
                other_key, other_value = past_key_values.update(
                    key_states, value_states
                )
                key_states = other_key
            return key_states + value_states
            """)
        assert "Merge" in ops.values(), ops

    def test_a_method_on_self_is_not_treated_as_opaque(self) -> None:
        """``self.<attr>`` is resolvable, so it must keep its normal wiring."""
        source = textwrap.dedent("""
            class M(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.proj = nn.Linear(8, 8)
                    self.cache = nn.Identity()

                def forward(self, hidden_states, flag):
                    a = self.proj(hidden_states)
                    k, v = torch.split(a, [4, 4], dim=-1)
                    if flag is not None:
                        k, v = self.cache.update(k, v)
                    return k + v
            """)
        analysis = analyze_source(source, config={"hidden_size": 8})
        structure = analysis.class_registry["M"]
        # The point is only that the self-call path is left alone -- it must not
        # be short-circuited by the forward-parameter rule.
        assert structure.forward_operations
