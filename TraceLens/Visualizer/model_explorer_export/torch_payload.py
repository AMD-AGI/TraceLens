###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn an FX inventory into a Model Explorer payload.

The inventory itself is built in ``TraceLens.ModelUtils``. This module is the
only place that turns it into Model Explorer nodes, so ModelUtils does not
import the Visualizer.
"""

from __future__ import annotations

from typing import Any

from TraceLens.ModelUtils.shape_inference import TensorSpec
from TraceLens.ModelUtils.torch_introspect import (
    FxNodeInfo,
    InputSpec,
    computation_graph_from_fx_inventory,
    export_module_inventory,
    fx_module_inventory,
)
from TraceLens.Visualizer.model_explorer_export.adapter import (
    computation_graph_to_explorer_graph,
)
from TraceLens.Visualizer.model_explorer_export.shapes import apply_shape_attrs


def explorer_graph_from_fx_inventory(
    inventory: list[FxNodeInfo],
    *,
    graph_id: str = "torch_module",
    label: str | None = None,
    include_shapes: bool = True,
) -> dict[str, Any]:
    """Build a single Model Explorer ``Graph`` dict from an FX inventory."""
    computation = computation_graph_from_fx_inventory(inventory)
    graph = computation_graph_to_explorer_graph(
        computation, graph_id=graph_id, label=label
    )
    if include_shapes:
        specs = {
            info.name: TensorSpec(shape=tuple(info.shape), dtype=info.dtype or "")
            for info in inventory
            if info.shape is not None
        }
        for node in graph["nodes"]:
            spec = specs.get(node.get("id"))
            if spec is not None:
                apply_shape_attrs(node, spec)
    return graph


def payload_from_fx_inventory(
    inventory: list[FxNodeInfo],
    *,
    name: str = "torch_module",
    include_shapes: bool = True,
) -> dict[str, Any]:
    """Wrap an FX inventory in a full Model Explorer payload envelope."""
    graph = explorer_graph_from_fx_inventory(
        inventory, graph_id=name, label=name, include_shapes=include_shapes
    )
    return {
        "name": name,
        "model_type": "torch-module",
        "source": "tracelens-torch-introspection",
        "tracelensViewer": {},
        "graphCollections": [{"label": name, "graphs": [graph]}],
    }


def build_torch_module_payload(
    module: Any,
    input_shape: tuple[int, ...],
    *,
    include_shapes: bool = True,
    name: str | None = None,
    model_dtype: Any = None,
) -> dict[str, Any]:  # pragma: no cover
    """Trace an ``nn.Module`` and build a Model Explorer payload.

    ``input_shape`` is the shape of the module's primary input tensor
    (batch included), e.g. ``(2, 128)`` for token ids or ``(1, 3, 224, 224)``
    for an image. Shapes are filled in by an FX ``ShapeProp`` pass.
    """
    resolved_name = name or type(module).__name__
    inventory = fx_module_inventory(module, tuple(input_shape), model_dtype=model_dtype)
    return payload_from_fx_inventory(
        inventory, name=resolved_name, include_shapes=include_shapes
    )


def build_exported_module_payload(
    module: Any,
    input_specs: list[InputSpec],
    *,
    include_shapes: bool = True,
    name: str | None = None,
    model_dtype: Any = None,
) -> dict[str, Any]:  # pragma: no cover
    """Trace an ``nn.Module`` with ``torch.export`` and build a Model Explorer payload."""
    inventory = export_module_inventory(module, input_specs, model_dtype=model_dtype)
    return payload_from_fx_inventory(
        inventory, name=name or type(module).__name__, include_shapes=include_shapes
    )
