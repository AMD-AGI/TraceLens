###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Introspect a plain ``torch.nn.Module`` into a Model Explorer payload.

Unlike the AST backend (which statically parses a Hugging Face checkpoint's
``modeling_*.py`` source), this path handles an arbitrary in-memory
``nn.Module`` by symbolically tracing it with ``torch.fx`` and propagating
shapes with ``ShapeProp``.

Two layers keep the torch dependency isolated:

* **Layer A (torch-free)** — turns a plain FX *inventory* (a list of
  :class:`FxNodeInfo` records: op kind, target, module namespace, input node
  names, shape/dtype) into a :class:`~TraceLens.ModelUtils.computation_graph.ComputationGraph`
  and then a Model Explorer payload, reusing the existing
  ``computation_graph_to_explorer_graph`` adapter and ``apply_shape_attrs``.
  This layer has no torch import and is unit-tested with hand-built fixtures.

* **Layer B (torch shim)** — runs ``torch.fx.symbolic_trace`` + ``ShapeProp``
  on a real module to produce the inventory Layer A consumes. It executes
  torch, so it is excluded from coverage (``# pragma: no cover``); the
  torch-free CI cannot run it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from TraceLens.ModelUtils.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    InlineFrameSpec,
)
from TraceLens.ModelUtils.shape_inference import TensorSpec

# NOTE: ``adapter`` / ``shapes`` live under ``TraceLens.Visualizer.model_explorer_export``,
# whose package ``__init__`` re-exports this module — importing them at module load
# would create a circular import. They are imported lazily inside the one function
# that needs them.


# ── Layer A: torch-free inventory → payload ──────────────────────────────────


@dataclass
class FxNodeInfo:
    """One FX node, reduced to the fields the graph builder needs.

    ``op`` mirrors ``torch.fx.Node.op`` (``placeholder``, ``call_function``,
    ``call_method``, ``call_module``, ``get_attr``, ``output``). ``namespace``
    is the dotted path of the owning submodule (``encoder.layer.0``); nodes
    sharing a prefix are grouped into expandable Model Explorer subgraphs.
    """

    name: str
    op: str
    label: str
    inputs: list[str] = field(default_factory=list)
    namespace: str = ""
    shape: tuple[Any, ...] | None = None
    dtype: str | None = None


def _dotted_prefixes(path: str) -> list[str]:
    """``"a.b.c"`` → ``["a", "a.b", "a.b.c"]`` (empty path → ``[]``)."""
    segments = [seg for seg in str(path or "").split(".") if seg]
    return [".".join(segments[: index + 1]) for index in range(len(segments))]


def computation_graph_from_fx_inventory(
    inventory: list[FxNodeInfo],
) -> ComputationGraph:
    """Build a :class:`ComputationGraph` from a torch-free FX inventory."""
    graph = ComputationGraph()
    index_by_name: dict[str, int] = {}

    for info in inventory:
        if info.op == "placeholder":
            synthetic = "@input"
        elif info.op == "output":
            synthetic = "@output"
        else:
            synthetic = None
        graph.nodes.append(
            GraphNodeSpec(
                key=info.name,
                block=None,
                label=info.label or info.name,
                synthetic=synthetic,
            )
        )
        index_by_name[info.name] = len(graph.nodes) - 1

    # Data-dependency edges. The ``output`` node is wired via ``output_ports``
    # (the adapter reads those separately and skips normal links to it).
    output_index: int | None = None
    output_source: int | None = None
    for info in inventory:
        target_index = index_by_name[info.name]
        if info.op == "output":
            output_index = target_index
            for source_name in info.inputs:
                source_index = index_by_name.get(source_name)
                if source_index is not None:
                    output_source = source_index
                    break
            continue
        for source_name in info.inputs:
            source_index = index_by_name.get(source_name)
            if source_index is None:
                continue
            link = (source_index, target_index)
            if link not in graph.links:
                graph.links.append(link)

    if output_index is not None and output_source is not None:
        graph.output_node_index = output_index
        graph.output_ports = {"result": output_source}
        graph.primary_output_port = "result"

    _add_namespace_frames(graph, inventory, index_by_name)
    return graph


def _add_namespace_frames(
    graph: ComputationGraph,
    inventory: list[FxNodeInfo],
    index_by_name: dict[str, int],
) -> None:
    """Group nodes that share a dotted-namespace prefix into inline frames.

    One frame per distinct prefix; the adapter applies larger frames first, so
    a node at ``a.b.c`` lands in the nested namespace ``a/b/c``.
    """
    members: dict[str, list[int]] = {}
    for info in inventory:
        index = index_by_name[info.name]
        for prefix in _dotted_prefixes(info.namespace):
            members.setdefault(prefix, []).append(index)
    for prefix, node_indices in members.items():
        graph.inline_frames.append(
            InlineFrameSpec(
                frame_id=prefix,
                label=prefix.rsplit(".", 1)[-1],
                node_indices=node_indices,
            )
        )


def explorer_graph_from_fx_inventory(
    inventory: list[FxNodeInfo],
    *,
    graph_id: str = "torch_module",
    label: str | None = None,
    include_shapes: bool = True,
) -> dict[str, Any]:
    """Build a single Model Explorer ``Graph`` dict from an FX inventory."""
    from TraceLens.Visualizer.model_explorer_export.adapter import (
        computation_graph_to_explorer_graph,
    )
    from TraceLens.Visualizer.model_explorer_export.shapes import apply_shape_attrs

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


# ── Layer B: torch shim (excluded from torch-free CI coverage) ───────────────


def _module_namespace(node: Any) -> str:  # pragma: no cover
    """Deepest owning-module path from an FX node's ``nn_module_stack``."""
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    last = list(stack.values())[-1]
    if isinstance(last, (tuple, list)) and last:
        return str(last[0])
    # Older/newer torch may key the OrderedDict by the module path itself.
    return str(list(stack.keys())[-1])


def _node_label(node: Any, module: Any) -> str:  # pragma: no cover
    """Human-readable tile label for one FX node."""
    if node.op == "placeholder":
        return str(node.name)
    if node.op == "output":
        return "Output"
    if node.op == "call_module":
        try:
            return type(module.get_submodule(str(node.target))).__name__
        except Exception:
            return str(node.target).rsplit(".", 1)[-1]
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    if node.op == "call_method":
        return str(node.target)
    if node.op == "get_attr":
        return str(node.target).rsplit(".", 1)[-1]
    return str(node.name)


def fx_module_inventory(
    module: Any,
    input_shape: tuple[int, ...],
    *,
    model_dtype: Any = None,
) -> list[FxNodeInfo]:  # pragma: no cover
    """Symbolically trace ``module`` and propagate shapes into an inventory."""
    import torch
    import torch.fx
    from torch.fx.passes.shape_prop import ShapeProp

    traced = torch.fx.symbolic_trace(module)

    dtype = model_dtype
    if dtype is None:
        dtype = next(
            (p.dtype for p in module.parameters() if p.dtype.is_floating_point),
            torch.float32,
        )
    example = torch.zeros(*input_shape, dtype=dtype)
    try:
        ShapeProp(traced).propagate(example)
    except Exception:
        # Shapes are best-effort; the graph structure is still useful without them.
        pass

    inventory: list[FxNodeInfo] = []
    for node in traced.graph.nodes:
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
        dtype_meta = getattr(tensor_meta, "dtype", None)
        # For call_module leaves the tracer's stack is the parent scope; for
        # traced-through custom submodules it is that submodule — both are the
        # grouping we want.
        namespace = _module_namespace(node)
        if node.op == "call_module" and namespace == str(node.target):
            namespace = str(node.target).rsplit(".", 1)[0] if "." in str(node.target) else ""
        inventory.append(
            FxNodeInfo(
                name=node.name,
                op=node.op,
                label=_node_label(node, module),
                inputs=[dep.name for dep in node.all_input_nodes],
                namespace=namespace,
                shape=tuple(int(d) for d in shape) if shape is not None else None,
                dtype=str(dtype_meta).replace("torch.", "") if dtype_meta else None,
            )
        )
    return inventory


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
    inventory = fx_module_inventory(
        module, tuple(input_shape), model_dtype=model_dtype
    )
    return payload_from_fx_inventory(
        inventory, name=resolved_name, include_shapes=include_shapes
    )
