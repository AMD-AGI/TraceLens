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
