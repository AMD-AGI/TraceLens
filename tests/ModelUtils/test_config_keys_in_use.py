###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A config key is read under the names models actually state, not guessed ones.

``_get(config, "post_norm", "use_post_norm")`` and its kin carried a second (and
sometimes third) spelling nobody had evidence for. That is the same flaw as the
dimension-alias table removed earlier: a name claimed on the chance some model
might use it.

Measured across the five pinned models, these groups had every model on the
FIRST name, so the alternates answered for nothing:

    sliding_window_size, use_bias, bias, expert_intermediate_size,
    block_types, hybrid_block_types, num_dense_layers, moe_layer_interval,
    use_post_norm, num_parameters

Three reads keep more than one name, each for a stated reason:

* ``qk_norm`` / ``use_qk_norm`` -- MiniMax-M3 states the second, and no config
  class declares a rename, so it is that model's own key.
* ``hidden_act`` / ``activation_function`` -- GPT-2 states the second, likewise
  undeclared.
* ``rope_parameters`` / ``rope_scaling`` / ``rope_theta`` -- three DIFFERENT
  keys, not spellings: a parameters dict, a scaling dict, and a bare theta. Any
  of them present means the model has rope, which is the question asked.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path("TraceLens/ModelUtils/extract.py")

# Reads that legitimately take more than one name, with why, from the docstring.
IN_USE = {
    ("qk_norm", "use_qk_norm"),
    ("hidden_act", "activation_function"),
    ("rope_parameters", "rope_scaling", "rope_theta"),
}

RETIRED = {
    "sliding_window_size",
    "use_bias",
    "expert_intermediate_size",
    "block_types",
    "hybrid_block_types",
    "num_dense_layers",
    "moe_layer_interval",
    "use_post_norm",
    "num_parameters",
}


def _multi_key_reads() -> set[tuple[str, ...]]:
    text = SOURCE.read_text(encoding="utf-8")
    found: set[tuple[str, ...]] = set()
    for match in re.finditer(r"_get\(config,\s*((?:\"[^\"]+\"\s*,?\s*)+)\)", text):
        names = tuple(re.findall(r'"([^"]+)"', match.group(1)))
        if len(names) > 1:
            found.add(names)
    return found


class TestOnlyNamesInUseAreRead:
    def test_every_multi_name_read_is_accounted_for(self) -> None:
        """A new one needs evidence that some model states it."""
        assert _multi_key_reads() == IN_USE, _multi_key_reads()

    def test_none_of_them_stands_in_for_another_key(self) -> None:
        """Retired as an ALTERNATE. ``moe_layer_interval`` is still read on its
        own, paired with ``moe_layer_start`` -- a key in its own right, which is
        a different thing from a second spelling of ``moe_layer_freq``."""
        alternates = {name for names in _multi_key_reads() for name in names}
        assert alternates & RETIRED == set(), alternates & RETIRED
