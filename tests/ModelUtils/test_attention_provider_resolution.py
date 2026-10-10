###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Whether attention is torch's own is a question about WHERE it is defined.

Node colouring used to answer it from two name lists -- ``"sdpa"``,
``"flex_attention"``, ``"flash_attn"``, ``"xformers"`` ... -- kept on the grounds
that ``flex_attention`` reaches its torch op through a ``torch.compile``
singleton behind a conditional import, which "no AST walk can follow without
fragile guesswork".

It can be followed. The conditional import is an ordinary binding the import
table already records, and the singleton is picked by a plain ternary whose other
arm is the torch symbol itself. These tests resolve all three transformers
attention integrations against the REAL installed source, so the claim is checked
rather than asserted -- including ``flex_attention``, which none of the four
models under test uses and which therefore has no graph-level coverage at all.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.ast_analyze import (
    attention_kernel_provider_module,
    is_torch_provided_attention,
)
from TraceLens.ModelUtils.attention_wrapper import attention_kernel_provider

_INTEGRATIONS = "transformers.integrations."


def _provider(module: str, symbol: str) -> str:
    resolved = attention_kernel_provider(module, symbol)
    if resolved is None:
        pytest.skip(f"{module}#{symbol} is not present in this transformers")
    return resolved


class TestProviderIsResolvedFromSource:
    def test_sdpa_resolves_to_torch_functional(self) -> None:
        """``attn_output = torch.nn.functional.scaled_dot_product_attention(...)``."""
        assert _provider(
            _INTEGRATIONS + "sdpa_attention", "sdpa_attention_forward"
        ) == ("torch.nn.functional")

    def test_flash_resolves_outside_torch(self) -> None:
        """Flash-attn is recognisable attention but is not a torch operation."""
        provider = _provider(
            _INTEGRATIONS + "flash_attention", "flash_attention_forward"
        )
        assert provider.split(".")[0] != "torch", provider

    def test_flex_resolves_through_the_compile_singleton_to_torch(self) -> None:
        """The case the name lists were kept for.

        ``flex_attention_forward`` assigns from a transformers-local shim, whose
        own body binds ``flex_attention_compiled`` by a ternary -- one arm a
        ``torch.compile`` singleton, the other the bare ``flex_attention``
        imported, inside an ``if is_torch_flex_attn_available():``, from
        ``torch.nn.attention.flex_attention``.
        """
        provider = _provider(_INTEGRATIONS + "flex_attention", "flex_attention_forward")
        assert provider.split(".")[0] == "torch", provider


class TestColourFollowsTheProvider:
    def test_a_resolved_torch_provider_is_native(self) -> None:
        assert is_torch_provided_attention(["kernel_provider: torch.nn.functional"])

    def test_a_resolved_library_provider_is_not(self) -> None:
        assert not is_torch_provided_attention(
            ["kernel_provider: transformers.modeling_flash_attention_utils"]
        )

    def test_the_kernel_name_does_not_decide(self) -> None:
        """Both directions: a torch-sounding library kernel and the reverse."""
        assert not is_torch_provided_attention(
            ["kernel: scaled_dot_product_attention", "kernel_provider: xformers.ops"]
        )
        assert is_torch_provided_attention(
            [
                "kernel: some_fused_flash_attn_thing",
                "kernel_provider: torch.nn.functional",
            ]
        )

    def test_an_unresolved_step_is_not_assumed_into_torch(self) -> None:
        assert not is_torch_provided_attention(["kernel: mystery_attention"])

    def test_a_direct_import_supplies_the_provider(self) -> None:
        assert (
            attention_kernel_provider_module(["import: fla.ops.kda#chunk_kda_fwd"])
            == "fla.ops.kda"
        )
        assert not is_torch_provided_attention(["import: fla.ops.kda#chunk_kda_fwd"])
