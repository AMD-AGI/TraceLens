###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model and library versions the graph tests are written against.

**Tests pin; generation does not.** Exporting a model pulls that model's current
code and whatever libraries it currently needs -- that is the product, and it has
to draw the model as it is today. A test is the opposite: it asserts that one
revision of one model produces one specific graph, so it has to read the same
source every run.

Nothing here is consulted by the library. ``load_model_spec(revision=...)`` and
``ensure_model_dependencies(extra_packages=...)`` take these as arguments, and
only these tests pass them; left unset, both resolve to current.

Why this exists: ``deepseek-ai/DeepSeek-V4-Flash`` published an ``inference/``
reference implementation and an ``encoding/`` chat-template helper mid-session.
The export began reading them, 2267 nodes became 1472, and 20 tests went red for
a reason that had nothing to do with this repo -- indistinguishable from a real
regression until someone went looking.

Two kinds of version matter, and a test needs both:

* the **checkpoint revision**, which decides what the model's own repo ships;
* the **library versions**, which decide what its modeling code resolves to. A
  natively-supported model is read from installed ``transformers``, so upgrading
  that package changes the graph as surely as the checkpoint changing does.

To move a pin: change it here, rerun the four-model gate, and account for every
node-count delta the way any other graph change is accounted for. Never refresh
one to silence a failure -- a moved pin and a real regression look identical in
the diff, which is exactly what the pin exists to tell apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPin:
    """One model's checkpoint revision and the libraries its graph was read with."""

    revision: str
    """Full commit SHA. A branch or tag moves, which is the thing being guarded."""

    transformers: str
    """Release the assertions were written against.

    For a model built in this interpreter this is checked, not installed: a test
    cannot re-exec itself, so it says plainly that it is describing different
    source rather than asserting on it.
    """

    packages: tuple[str, ...] = field(default_factory=tuple)
    """Pinned ``name==version`` requirements beyond ``transformers``.

    Only for a model provisioned into its own environment. Installed before the
    dependency probe runs, so the probe cannot fetch an unpinned latest in their
    place -- which is precisely what an export wants it to do.
    """

    own_environment: bool = False
    """Whether this model is built in a provisioned environment of its own."""


MODEL_PINS: dict[str, ModelPin] = {
    # Read from upstream transformers at the SHA in ``source.TRANSFORMERS_GITHUB_SOURCE``.
    "zai-org/GLM-5.3-Flash": ModelPin(
        revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        transformers="5.16.1",
    ),
    "MiniMaxAI/MiniMax-M3": ModelPin(
        revision="f0e1c1e04d40177e4673a22097036854f536e9c0",
        transformers="5.16.1",
    ),
    # Read from INSTALLED transformers, so its graph moves with that package.
    "deepseek-ai/DeepSeek-V4-Flash": ModelPin(
        revision="60d8d70770c6776ff598c94bb586a859a38244f1",
        transformers="5.16.1",
    ),
    # A decoder-only model of the pre-Llama generation, and the only one here
    # that is not a recent MoE. It covers what the others cannot: `Conv1D`
    # projections (`addmm` against a `[in, out]` weight, the transpose of
    # `F.linear`'s layout), a shape restored by tuple CONCATENATION rather than
    # written as one tuple, and a config that states none of its own switches --
    # `add_cross_attention` and `reorder_and_upcast_attn` live in the config
    # CLASS, and reading them as `None` rather than `False` drew a whole
    # cross-attention tower the checkpoint cannot build.
    "openai-community/gpt2": ModelPin(
        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        transformers="5.16.1",
    ),
    # Kimi's code does not run under the transformers installed here, so its
    # graph is built in an environment of its own. Those libraries are pinned
    # too: ``fla-core`` supplies the linear-attention kernels the model imports
    # at module scope, and without them the class cannot reach the meta device,
    # so every meta-derived shape silently disappears.
    "moonshotai/Kimi-K3": ModelPin(
        revision="f831ab66814297da540d832a5235f8e904f29d06",
        transformers="4.56.2",
        packages=("einops==0.8.2", "fla-core==0.5.2"),
        own_environment=True,
    ),
}


def pin_for(model_id: str | None) -> ModelPin | None:
    """The pin for ``model_id``, or ``None`` when it is not pinned."""
    if not model_id:
        return None
    return MODEL_PINS.get(str(model_id).strip())


def is_pinned(model_id: str | None) -> bool:
    """Whether this suite reads ``model_id`` at a fixed revision."""
    return pin_for(model_id) is not None
