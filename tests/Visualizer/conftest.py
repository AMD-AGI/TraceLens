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


def _build_here(model_id: str, revision: str | None = None) -> list[dict]:
    from TraceLens.ModelUtils.loader import load_model_spec
    from TraceLens.ModelUtils.shape_inference import ShapeInferencer
    from TraceLens.Visualizer.model_explorer_export.merge import (
        build_merged_model_graph,
    )

    spec = load_model_spec(model_id, detailed=True, revision=revision)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    return graph["nodes"]


def _build_in_env(model_id: str, reason: str, pin) -> list[dict]:
    from TraceLens.ModelUtils import model_env

    # The pin names the versions these assertions were written against, rather
    # than whatever the checkpoint's config happens to say or the probe happens
    # to find. An export does the opposite and takes the model's current
    # requirements -- that is the point of the split.
    version = pin.transformers
    target = model_env.ensure_model_dependencies(
        model_id, version, extra_packages=list(pin.packages)
    )
    if target is None:
        pytest.skip(
            f"could not provision transformers=={version} for {model_id}; "
            "the graph here would be built from different source than the "
            "model actually runs"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nodes.json"
        result = subprocess.run(
            [sys.executable, str(_BUILDER), model_id, str(out), pin.revision],
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


def _require_pinned_transformers(model_id: str, pin) -> None:
    """Skip rather than assert on a graph read from a different transformers.

    A natively-supported model is read from the INSTALLED package, so its graph
    moves with that package just as surely as with the checkpoint. A test cannot
    re-exec itself into another release, so the honest thing is to say plainly
    that it would be describing different source.
    """
    import transformers

    installed = getattr(transformers, "__version__", "")
    if installed != pin.transformers:
        pytest.skip(
            f"{model_id} is pinned to transformers=={pin.transformers} but "
            f"{installed} is installed; this graph would be read from different "
            "source than these assertions describe"
        )


@pytest.fixture(scope="session")
def model_graph_nodes():
    """``nodes(model_id)`` -> merged-graph nodes, built where the model belongs."""
    pytest.importorskip("huggingface_hub")
    from TraceLens.ModelUtils import model_env
    from model_pins import pin_for

    cache: dict[str, list[dict]] = {}

    def nodes(model_id: str) -> list[dict]:
        # Every graph test funnels through here, so this is where an unpinned
        # model is caught. Asserting on a floating checkpoint means asserting on
        # whatever its repo holds today: when DeepSeek-V4-Flash published two
        # extra directories, 20 tests went red for a reason that had nothing to
        # do with this repo, and that took a while to tell apart from a real
        # regression.
        pin = pin_for(model_id)
        assert pin is not None, (
            f"{model_id} is not pinned. A graph assertion describes one revision "
            "of one model's source; add its commit SHA and library versions to "
            "tests/model_pins.py so this suite reads the same code every run."
        )
        if model_id not in cache:
            # Already inside a provisioned environment (the exporter's own
            # re-exec flag): this interpreter IS the right one.
            import os

            if os.environ.get(model_env.REEXEC_ENV_FLAG):
                cache[model_id] = _build_here(model_id, pin.revision)
            elif pin.own_environment:
                reason = model_env.detect_env_mismatch(model_id)
                cache[model_id] = _build_in_env(
                    model_id, reason or "pinned to its own environment", pin
                )
            else:
                _require_pinned_transformers(model_id, pin)
                cache[model_id] = _build_here(model_id, pin.revision)
        return cache[model_id]

    return nodes


@pytest.fixture(scope="session")
def kimi_nodes(model_graph_nodes):
    """Kimi-K3's merged graph, built under the transformers its code needs."""
    return model_graph_nodes("moonshotai/Kimi-K3")
