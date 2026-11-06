###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model and library versions the graph tests are written against.

Tests pin a checkpoint revision and a transformers release so a graph assertion
describes one source tree. Export does not read these pins; it draws the model
as it is today.

The four-model gate is GLM-5.3-Flash, MiniMax-M3, DeepSeek-V4-Flash, and
Kimi-K3. A checkpoint revision decides what the model's own repo ships. The
transformers pin decides what a natively supported model's modeling code
resolves to. Move a pin here and account for the graph delta. Do not move a
pin only to silence a failure.
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
    # Read from installed transformers, so its graph moves with that package.
    "deepseek-ai/DeepSeek-V4-Flash": ModelPin(
        revision="60d8d70770c6776ff598c94bb586a859a38244f1",
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
