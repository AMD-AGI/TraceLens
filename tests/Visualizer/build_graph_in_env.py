###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build one model's merged graph and dump its nodes as JSON.

Run as a script, so a test can build a model inside the environment that model's
own code needs rather than in whatever interpreter pytest happens to be. See
``tests/Visualizer/conftest.py``.

    python tests/Visualizer/build_graph_in_env.py <model-id> <out.json> [revision] [regime-json]

``regime-json`` states which RUN to draw (see ``RuntimeRegime``); omitted, the
defaults apply, exactly as an export with no flags would.
"""

from __future__ import annotations

import copy
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

_SPECS: dict[tuple, object] = {}
_SPEC_KEYS: dict[int, tuple] = {}
_MERGED: dict[tuple, dict] = {}


def _graph_cache_dir() -> Path:
    root = os.environ.get("TRACELENS_GRAPH_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "tracelens" / "graph-nodes"


def _regime_payload(regime) -> object:
    if regime is None:
        return None
    if dataclasses.is_dataclass(regime):
        return dataclasses.asdict(regime)
    return regime


def pinned_model_spec(model_id, *, detailed=True, revision=None, regime=None):
    """Load a spec at the pin, reusing the one already loaded in this process.

    Callers that pass ``revision`` keep it. Callers that do not get the
    revision in ``tests/model_pins.py`` when the model is pinned.
    """
    tests_dir = str(Path(__file__).resolve().parents[1])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from model_pins import pin_for
    from TraceLens.ModelUtils.loader import load_model_spec

    pin = pin_for(model_id)
    resolved = revision if revision is not None else (pin.revision if pin else None)
    key = (
        str(model_id),
        bool(detailed),
        resolved,
        json.dumps(_regime_payload(regime), sort_keys=True, default=str),
    )
    if key not in _SPECS:
        _SPECS[key] = load_model_spec(
            model_id, detailed=detailed, revision=resolved, regime=regime
        )
    spec = _SPECS[key]
    _SPEC_KEYS[id(spec)] = key
    return spec


def cached_graph_nodes(model_id, revision, transformers, regime, build):
    """Return merged graph nodes, reading and writing the on-disk cache.

    The key is the model, the pin, and the regime. A lock keeps two pytest
    workers from writing the same file. ``build`` runs only on a miss.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "revision": revision,
            "transformers": transformers,
            "regime": _regime_payload(regime),
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    directory = _graph_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    lock_path = directory / f"{digest}.lock"
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.is_file():
                return json.loads(path.read_text())
            nodes = build()
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(nodes, default=str))
            temporary.replace(path)
            return nodes
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def pinned_merged_graph(spec, *, shape_inferencer=None):
    """Return the merged graph for a pinned spec, built once per process.

    A spec loaded through ``pinned_model_spec`` is cached in memory and on
    disk, keyed by the same pin and by whether shapes were inferred. The
    caller gets a copy. Any other spec is built immediately and not cached.
    """
    from TraceLens.Visualizer.model_explorer_export.merge import (
        build_merged_model_graph,
    )

    def build():
        if shape_inferencer is None:
            return build_merged_model_graph(spec)
        return build_merged_model_graph(spec, shape_inferencer=shape_inferencer)

    meta = _SPEC_KEYS.get(id(spec))
    if meta is None:
        return build()
    shaped = shape_inferencer is not None
    cache_key = (meta, shaped)
    if cache_key not in _MERGED:
        payload = json.dumps(
            {"spec": meta, "shaped": shaped}, sort_keys=True, default=str
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        directory = _graph_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"graph-{digest}.json"
        lock_path = directory / f"graph-{digest}.lock"
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if path.is_file():
                    _MERGED[cache_key] = json.loads(path.read_text())
                else:
                    graph = build()
                    temporary = path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(graph, default=str))
                    temporary.replace(path)
                    _MERGED[cache_key] = graph
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return copy.deepcopy(_MERGED[cache_key])


def main(argv: list[str]) -> int:
    model_id, out_path = argv[1], argv[2]
    # The caller passes the revision these assertions were written against; a
    # bare invocation reads the checkpoint's head, like an export does.
    revision = argv[3] if len(argv) > 3 else None
    regime_json = argv[4] if len(argv) > 4 else None

    from TraceLens.ModelUtils.ast_analyze import RuntimeRegime
    from TraceLens.ModelUtils.loader import load_model_spec
    from TraceLens.ModelUtils.shape_inference import ShapeInferencer
    from TraceLens.Visualizer.model_explorer_export.merge import (
        build_merged_model_graph,
    )

    regime = RuntimeRegime(**json.loads(regime_json)) if regime_json else None
    spec = load_model_spec(model_id, detailed=True, revision=revision, regime=regime)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    with open(out_path, "w") as handle:
        json.dump(graph["nodes"], handle, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
