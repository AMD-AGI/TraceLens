###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Checkpoint revisions this analysis reads.

A Hugging Face repo is mutable. ``deepseek-ai/DeepSeek-V4-Flash`` published two
new directories mid-session -- a standalone reference implementation under
``inference/`` and a chat-template encoder under ``encoding/`` -- and the export
silently started reading them: 2267 nodes became 1472, with 68 type-check
warnings, 12 frames gone and 20 tests failing. Nothing in this repo changed, and
nothing in the checkpoint's ``config.json`` changed either.

Tests that assert on a model's graph are assertions about one revision of that
model's source. Left floating they describe whatever the repo happens to hold
that morning, so a red suite means "upstream published something", which is
indistinguishable from "we broke it" until someone goes looking.

Pinning the checkpoint is the same move the repo already makes for upstream
transformers (``source.TRANSFORMERS_GITHUB_SOURCE`` names a commit, not
``@main``, so hard-coded op ids do not drift under line shifts).

A model id absent from this table is not pinned and resolves to whatever the hub
currently serves -- someone exporting their own checkpoint gets their checkpoint.
The pin applies wherever the id IS listed, including the subprocess that builds a
graph under a model's own ``transformers``, so a pinned model is read the same
way everywhere.

To move a pin: change the SHA here, rerun the four-model gate, and account for
every node-count delta the way any other graph change is accounted for. Do not
refresh it to silence a failure -- a moved pin and a real regression look alike
in the diff, and the point of the pin is to tell them apart.
"""

from __future__ import annotations

# Checkpoint revisions the graph tests and the four-model gate are written
# against. Full commit SHAs: a branch or tag name moves, which is the thing being
# guarded against.
MODEL_REVISIONS: dict[str, str] = {
    "zai-org/GLM-5.3-Flash": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "moonshotai/Kimi-K3": "f831ab66814297da540d832a5235f8e904f29d06",
    "MiniMaxAI/MiniMax-M3": "f0e1c1e04d40177e4673a22097036854f536e9c0",
    "deepseek-ai/DeepSeek-V4-Flash": "60d8d70770c6776ff598c94bb586a859a38244f1",
}


def pinned_revision(model_id: str | None) -> str | None:
    """The revision ``model_id`` is pinned to, or ``None`` if it is not pinned.

    Anything that is not a plain ``owner/name`` hub id -- a local directory, a
    GitHub reference -- is not a hub checkout and has no revision to pin.
    """
    if not model_id:
        return None
    return MODEL_REVISIONS.get(str(model_id).strip())


def is_pinned(model_id: str | None) -> bool:
    """Whether ``model_id`` is read at a fixed revision."""
    return pinned_revision(model_id) is not None
