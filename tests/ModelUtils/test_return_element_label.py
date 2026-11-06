###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A returned value is named for itself, never for the module that computed it.

``_return_element_label`` sees through a trailing housekeeping call so that
``return cos.to(dtype=x.dtype), sin.to(...)`` still names its two slots -- without
that, a tuple of casts has no bare-``Name`` element at all and neither becomes an
output port.

``torch.where(...)`` has exactly the same shape, and the receiver is a MODULE.
DeepSeek's indexer ends with::

    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

and published an ``@output`` labelled ``torch`` -- a boundary standing for a
module rather than for the indices it carries.
"""

from __future__ import annotations

import textwrap

from TraceLens.ModelUtils.ast_analyze import analyze_source


def _labels(source: str) -> set[str]:
    analysis = analyze_source(source, config={"hidden_size": 8})
    return set(analysis.class_registry["M"].forward_return_slots or {})


class TestReturnSlotNaming:
    def test_a_module_call_is_not_named_after_the_module(self) -> None:
        source = """
class M(nn.Module):
    def forward(self, scores, invalid):
        picks = scores.topk(4, dim=-1).indices
        return torch.where(invalid, picks, picks)
"""
        labels = _labels(source)
        assert "torch" not in labels, labels

    def test_it_is_named_for_the_call_instead(self) -> None:
        source = """
class M(nn.Module):
    def forward(self, scores, invalid):
        picks = scores.topk(4, dim=-1).indices
        return torch.where(invalid, picks, picks)
"""
        assert "where" in _labels(source)

    def test_a_trailing_cast_still_names_the_value(self) -> None:
        """``cos.to(...)`` is housekeeping ON a value; the value names the slot."""
        source = """
class M(nn.Module):
    def forward(self, x):
        cos = x * 2
        sin = x * 3
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
"""
        labels = _labels(source)
        assert {"cos", "sin"} <= labels, labels

    def test_a_scaled_tensor_keeps_the_tensor_name(self) -> None:
        """``weights * self.scale`` is still ``weights``; the scale has no name."""
        source = """
class M(nn.Module):
    def forward(self, weights):
        return (weights * self.routed_scaling_factor).to(weights.dtype)
"""
        assert "weights" in _labels(source)

    def test_a_sum_of_two_named_tensors_is_not_named_for_one_factor(self) -> None:
        """``alpha * residual + beta * hidden`` is the sum, not ``alpha``."""
        source = """
class M(nn.Module):
    def forward(self, residual, hidden_states):
        alpha = residual.sigmoid()
        beta = hidden_states.sigmoid()
        return (alpha * residual + beta * hidden_states).to(residual.dtype)
"""
        labels = _labels(source)
        assert "alpha" not in labels, labels
        assert "residual" not in labels, labels
        assert "hidden_states" not in labels, labels


def test_forwarded_tuple_return_publishes_both_slots() -> None:
    """``return add_rmsnorm(...)`` is the callee's tuple, and both arms are live.

    ``_sconv_add_norm`` returns ``None, hidden + full`` from one arm and
    ``add_rmsnorm(...)`` (itself ``(y, res)``) from the other. Neither arm is a
    bare-name tuple the name-based pass can read, so without a positional merge
    the add and the forwarded call are extracted and then left unconsumed.
    """
    source = textwrap.dedent("""
        import torch.nn as nn

        def add_rmsnorm(residual, delta, weight, eps):
            y = residual + delta
            res = y * weight
            return y, res

        def _sconv_add_norm(delta, hidden, sconv, norm, positions):
            full = hidden + delta
            if norm is None:
                return None, hidden + full
            return add_rmsnorm(hidden, full, norm, 1e-6)

        class Block(nn.Module):
            def forward(self, hidden, delta, positions, norm):
                normed, hidden = _sconv_add_norm(
                    delta, hidden, hidden, norm, positions
                )
                return normed + hidden
        """)
    analysis = analyze_source(source, config={"hidden_size": 8})
    block = analysis.class_registry["Block"]
    returns = {
        key: value
        for key, value in block.multi_op_method_returns.items()
        if "sconv_add_norm" in key
    }
    assert len(returns) == 1, list(block.multi_op_method_returns)
    slots, order, _primary = next(iter(returns.values()))
    assert len(order) == 2, order
    assert all(slots.get(name) for name in order), slots
    # The caller's unpack docks both positions, so neither slot is dropped.
    unpacked = [
        names
        for names in block.forward_step_output_names.values()
        if names[:2] == ["normed", "hidden"] or set(names) >= {"normed", "hidden"}
    ]
    assert unpacked, block.forward_step_output_names
