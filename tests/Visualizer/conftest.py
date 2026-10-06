###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build a model's graph in the environment that model's own code needs.

A checkpoint whose modeling code cannot be imported here is still *parsed* --
from whatever same-named module this interpreter can reach. For Kimi-K3 that is
a different ``get_unpad_data`` than the one the model ships, so a test asserting
on its expansion was describing a file the model never runs. Worse, the model
cannot be built on the meta device either, so every meta-derived shape is
missing and the graph is quietly a degraded one.

The exporter already solves this at run time by re-executing under a pinned
``transformers`` (``model_env.reexec_in_model_env``). These fixtures do the same
for tests: ask whether this interpreter can import the checkpoint, and when it
cannot, build the graph in a subprocess that can. Results are cached for the
whole session, so the expensive build happens once however many tests ask.

When the environment cannot be provisioned at all (no cached copy and no
network), the test SKIPS with the reason. Asserting on the degraded graph
instead would be asserting on the wrong model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_BUILDER = Path(__file__).with_name("build_graph_in_env.py")


def _build_here(model_id: str) -> list[dict]:
    from TraceLens.ModelUtils.loader import load_model_spec
    from TraceLens.ModelUtils.shape_inference import ShapeInferencer
    from TraceLens.Visualizer.model_explorer_export.merge import (
        build_merged_model_graph,
    )

    spec = load_model_spec(model_id, detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _build_in_env(model_id: str, reason: str) -> list[dict]:
    from TraceLens.ModelUtils import model_env

    version = model_env.required_transformers_version(model_id)
    if version is None:
        pytest.skip(
            f"{model_id} needs different model code ({reason}) but its config "
            "names no transformers version to pin"
        )
    target = model_env.ensure_model_dependencies(model_id, version)
    if target is None:
        pytest.skip(
            f"could not provision transformers=={version} for {model_id}; "
            "the graph here would be built from different source than the "
            "model actually runs"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nodes.json"
        result = subprocess.run(
            [sys.executable, str(_BUILDER), model_id, str(out)],
            env=model_env._child_env(target),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not out.exists():
            pytest.skip(
                f"building {model_id} under transformers=={version} failed: "
                f"{result.stderr.strip()[-400:]}"
            )
        return json.loads(out.read_text())


@pytest.fixture(scope="session")
def model_graph_nodes():
    """``nodes(model_id)`` -> merged-graph nodes, built where the model belongs."""
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils import model_env
    from TraceLens.ModelUtils.model_pins import is_pinned

    cache: dict[str, list[dict]] = {}

    def nodes(model_id: str) -> list[dict]:
        # Every graph test funnels through here, so this is where an unpinned
        # model is caught. Asserting on a floating checkpoint means asserting on
        # whatever its repo holds today: when DeepSeek-V4-Flash published two
        # extra directories, 20 tests went red for a reason that was nothing to
        # do with this repo and took a while to tell apart from a real
        # regression.
        assert is_pinned(model_id), (
            f"{model_id} is not pinned. Graph assertions describe one revision "
            "of a model's source; add its commit SHA to "
            "TraceLens/ModelUtils/model_pins.py so this suite reads the same "
            "code every run."
        )
        if model_id not in cache:
            # Already inside a provisioned environment (the exporter's own
            # re-exec flag): this interpreter IS the right one.
            import os

            if os.environ.get(model_env.REEXEC_ENV_FLAG):
                cache[model_id] = _build_here(model_id)
            else:
                reason = model_env.detect_env_mismatch(model_id)
                cache[model_id] = (
                    _build_here(model_id)
                    if reason is None
                    else _build_in_env(model_id, reason)
                )
        return cache[model_id]

    return nodes


@pytest.fixture(scope="session")
def kimi_nodes(model_graph_nodes):
    """Kimi-K3's merged graph, built under the transformers its code needs."""
    return model_graph_nodes("moonshotai/Kimi-K3")
