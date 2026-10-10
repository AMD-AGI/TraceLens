###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which attention implementation a checkpoint runs has exactly one answer.

A checkpoint that configures nothing runs ``"sdpa"`` -- the transformers default
-- and three separate passes need to know it: which kernel the dispatch variable
resolves to, whether a flash-request predicate is true, and whether a forward
branch guarded on the implementation is live. The first two defaulted it
themselves; the third read the raw config, found the key absent, and treated the
branch as unresolvable -- so Kimi kept the padding and slicing that only the
flash path performs, and GLM kept an eager mask tail its early
``if ... == "sdpa": return`` skips.
"""

from __future__ import annotations

from TraceLens.ModelUtils.ast_analyze import (
    _with_resolved_attn_implementation,
    resolved_attn_implementation,
)


class TestResolvedImplementation:
    def test_an_unconfigured_checkpoint_runs_sdpa(self) -> None:
        assert resolved_attn_implementation({}) == "sdpa"
        assert resolved_attn_implementation(None) == "sdpa"

    def test_a_blank_value_is_not_an_implementation(self) -> None:
        assert resolved_attn_implementation({"_attn_implementation": "   "}) == "sdpa"

    def test_a_configured_value_is_normalised_and_kept(self) -> None:
        assert (
            resolved_attn_implementation(
                {"_attn_implementation": " Flash_Attention_2 "}
            )
            == "flash_attention_2"
        )


class TestConfigCarriesTheAnswer:
    def test_the_default_is_written_in_so_branches_can_read_it(self) -> None:
        """A branch predicate reads the config dict, not the resolver."""
        filled = _with_resolved_attn_implementation({"hidden_size": 8})
        assert filled["_attn_implementation"] == "sdpa"
        # ...without disturbing anything else.
        assert filled["hidden_size"] == 8

    def test_a_configured_checkpoint_is_left_exactly_alone(self) -> None:
        config = {"_attn_implementation": "flash_attention_2"}
        assert _with_resolved_attn_implementation(config) is config

    def test_the_callers_config_is_not_mutated(self) -> None:
        config: dict = {}
        _with_resolved_attn_implementation(config)
        assert config == {}

    def test_none_stays_none(self) -> None:
        assert _with_resolved_attn_implementation(None) is None
