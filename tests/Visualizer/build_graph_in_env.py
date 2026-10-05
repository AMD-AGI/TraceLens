###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build one model's merged graph and dump its nodes as JSON.

Run as a script, so a test can build a model inside the environment that model's
own code needs rather than in whatever interpreter pytest happens to be. See
``tests/Visualizer/conftest.py``.

    python tests/Visualizer/build_graph_in_env.py <model-id> <out.json>
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    model_id, out_path = argv[1], argv[2]

    from TraceLens.ModelUtils.loader import load_model_spec
    from TraceLens.ModelUtils.shape_inference import ShapeInferencer
    from TraceLens.Visualizer.model_explorer_export.merge import (
        build_merged_model_graph,
    )

    spec = load_model_spec(model_id, detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    with open(out_path, "w") as handle:
        json.dump(graph["nodes"], handle, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
