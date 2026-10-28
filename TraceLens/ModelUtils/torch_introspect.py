###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Introspect a plain ``torch.nn.Module`` into a Model Explorer payload.

Unlike the AST backend (which statically parses a Hugging Face checkpoint's
``modeling_*.py`` source), this path handles an arbitrary in-memory
``nn.Module``. Given one input shape it symbolically traces the module with
``torch.fx`` and propagates shapes with ``ShapeProp``; given named
:class:`InputSpec` inputs it traces with ``torch.export``, which runs through
shape-dependent Python control flow and keeps symbolic dims (``B``, ``S``)
symbolic in every output shape.

Two layers keep the torch dependency isolated:

* **Layer A (torch-free)** — turns a plain FX *inventory* (a list of
  :class:`FxNodeInfo` records: op kind, target, module namespace, input node
  names, shape/dtype) into a :class:`~TraceLens.ModelUtils.computation_graph.ComputationGraph`.
  The Model Explorer payload is built by the Visualizer, so this package does
  not import it. This layer has no torch import and is unit-tested with
  hand-built fixtures.

* **Layer B (torch shim)** — runs ``torch.fx.symbolic_trace`` + ``ShapeProp``
  (or ``torch.export``) on a real module to produce the inventory Layer A
  consumes. It executes torch, so it is excluded from coverage
  (``# pragma: no cover``); the torch-free CI cannot run it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from TraceLens.ModelUtils.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    InlineFrameSpec,
)
from TraceLens.ModelUtils.size_provenance import (
    SizeReadResolver,
    TracedNode,
    parse_stack_trace,
)

_log = logging.getLogger(__name__)

# Constants
# Example size of the first symbolic dim; later ones double it. Never 0 or 1
# (torch.export specializes those to constants) and pairwise distinct, so each
# traced symbol is identified by its example size alone.
_SYMBOLIC_EXAMPLE_SIZE_BASE = 16
# torch.export nodes that only assert; they carry no tensor.
_EXPORT_SKIPPED_OPS = frozenset({"_assert_tensor_metadata", "_assert_scalar"})
# torch.export reads every symbolic size off a model input, whatever tensor the
# source read it from; SizeReadResolver recovers the real one.
_EXPORT_SIZE_READ_OPS = frozenset({"sym_size", "sym_numel", "sym_stride"})


# ── Layer A: torch-free inventory → payload ──────────────────────────────────


@dataclass(frozen=True)
class InputSpec:
    """One named forward input for ``torch.export`` tracing.

    ``dims`` holds an ``int`` per static axis and a name per symbolic one; a
    name shared across inputs is one dimension (``W=B,S`` and ``GI=B,S,48``).
    ``dtype`` is a ``torch`` dtype name (``int64``); ``None`` means the
    module's floating dtype.
    """

    name: str
    dims: tuple[int | str, ...]
    dtype: str | None = None


def symbolic_example_sizes(input_specs: list[InputSpec]) -> dict[str, int]:
    """A distinct example size for every symbolic dim, in first-use order."""
    sizes: dict[str, int] = {}
    for spec in input_specs:
        for dim in spec.dims:
            if isinstance(dim, str) and dim not in sizes:
                sizes[dim] = _SYMBOLIC_EXAMPLE_SIZE_BASE << len(sizes)
    return sizes


@dataclass
class FxNodeInfo:
    """One FX node, reduced to the fields the graph builder needs.

    ``op`` mirrors ``torch.fx.Node.op`` (``placeholder``, ``call_function``,
    ``call_method``, ``call_module``, ``get_attr``, ``output``). ``namespace``
    is the dotted path of the owning submodule (``encoder.layer.0``); nodes
    sharing a prefix are grouped into expandable Model Explorer subgraphs.
    ``reads_sizes`` marks a node that reads a symbolic size (``arange(S)``),
    so it varies with the input even when it has no input edge.
    """

    name: str
    op: str
    label: str
    inputs: list[str] = field(default_factory=list)
    namespace: str = ""
    shape: tuple[Any, ...] | None = None
    dtype: str | None = None
    reads_sizes: bool = False


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
    # A node no input reaches (a weight read, a literal table) is constant.
    varies: set[str] = set()

    for info in inventory:
        if info.op == "placeholder":
            synthetic = "@input"
        elif info.op == "output":
            synthetic = "@output"
        else:
            synthetic = None
        if (
            info.op == "placeholder"
            or info.reads_sizes
            or any(name in varies for name in info.inputs)
        ):
            varies.add(info.name)
        graph.nodes.append(
            GraphNodeSpec(
                key=info.name,
                block=None,
                label=info.label or info.name,
                synthetic=synthetic,
                constant=synthetic is None and info.name not in varies,
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
        # An exported aten overload (``mul.Tensor``) is labelled by its op name.
        packet = getattr(node.target, "_overloadpacket", node.target)
        return getattr(packet, "__name__", str(node.target))
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
            namespace = (
                str(node.target).rsplit(".", 1)[0] if "." in str(node.target) else ""
            )
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


def export_module_inventory(
    module: Any,
    input_specs: list[InputSpec],
    *,
    model_dtype: Any = None,
) -> list[FxNodeInfo]:  # pragma: no cover
    """Trace ``module`` with ``torch.export`` into an inventory with symbolic shapes.

    Every input is zero-filled (a valid index for embedding/gather inputs) and
    passed by keyword. Symbolic dims stay symbolic: a node's shape reads
    ``[B, S, 512]`` or ``[B*S, 64]``, never the example sizes.
    """
    import torch
    from torch.export import Dim

    float_dtype = model_dtype or next(
        (p.dtype for p in module.parameters() if p.dtype.is_floating_point),
        torch.float32,
    )
    sizes = symbolic_example_sizes(input_specs)
    dims = {name: Dim(name) for name in sizes}
    kwargs: dict[str, Any] = {}
    dynamic_shapes: dict[str, Any] = {}
    for spec in input_specs:
        dtype = float_dtype
        if spec.dtype is not None:
            dtype = getattr(torch, spec.dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"input {spec.name!r}: unknown dtype {spec.dtype!r}")
        kwargs[spec.name] = torch.zeros(
            *(sizes.get(dim, dim) for dim in spec.dims), dtype=dtype
        )
        axes = {i: dims[dim] for i, dim in enumerate(spec.dims) if isinstance(dim, str)}
        dynamic_shapes[spec.name] = axes or None

    anchors = _model_code(module)
    program = _export_refining_ranges(module, kwargs, dynamic_shapes, anchors)
    signature = program.graph_signature
    user_inputs = set(signature.user_inputs)
    # A buffer the forward writes (a cache) is state the next call reads, so
    # it stays an input; every other lifted weight, buffer or constant goes.
    mutated = set(signature.buffers_to_mutate.values())
    state_inputs = {
        name: buffer
        for name, buffer in signature.inputs_to_buffers.items()
        if buffer in mutated
    }
    placeholders = [
        node
        for node in program.graph.nodes
        if node.op == "placeholder" and node.name in user_inputs
    ]
    symbols = _symbol_names(placeholders, sizes)
    specialized = sorted(set(sizes) - {str(name) for name in symbols.values()})
    if specialized:
        _log.warning(
            "the model fixed symbolic dim(s) %s to their example size(s) %s",
            ", ".join(specialized),
            ", ".join(str(sizes[name]) for name in specialized),
        )

    files = {code.co_filename for code in anchors}
    inventory: list[FxNodeInfo] = []
    traced: list[TracedNode] = []
    sized: list[FxNodeInfo] = []
    # A node that yields no tensor stands for what it reads: the tensor behind
    # an ``.item()``, or a size whose source tensor the resolver finds.
    stand_in: dict[str, tuple[list[str], bool]] = {}
    for node in program.graph.nodes:
        if node.op == "placeholder" and not (
            node.name in user_inputs or node.name in state_inputs
        ):
            continue
        inputs: list[str] = []
        reads_sizes = False
        for dep in node.all_input_nodes:
            edges, dep_reads_sizes = stand_in.get(dep.name, ([dep.name], False))
            inputs.extend(edges)
            reads_sizes |= dep_reads_sizes
        inputs = list(dict.fromkeys(inputs))
        label = _node_label(node, module)
        if node.op == "placeholder" and node.name in state_inputs:
            label = state_inputs[node.name]
        value = node.meta.get("val")
        if node.op == "call_function" and label in _EXPORT_SIZE_READ_OPS:
            stand_in[node.name] = ([], True)
            continue
        if node.op == "call_function" and (
            label in _EXPORT_SKIPPED_OPS or not _is_tensor_valued(value)
        ):
            stand_in[node.name] = (inputs, reads_sizes)
            continue
        if node.op == "output":
            # Export returns a tuple; the graph's output tile shows the first.
            value = next((v for v in node.args[0] if v is not None), None)
            value = getattr(value, "meta", {}).get("val")
        shape = dtype_name = None
        if isinstance(value, torch.Tensor):
            shape = tuple(_format_dim(dim, symbols) for dim in value.shape)
            dtype_name = str(value.dtype).replace("torch.", "")
        info = FxNodeInfo(
            name=node.name,
            op=node.op,
            label=label,
            inputs=inputs,
            namespace=_module_namespace(node),
            shape=shape,
            dtype=dtype_name,
            reads_sizes=reads_sizes,
        )
        inventory.append(info)
        traced.append(
            TracedNode(
                name=node.name,
                frames=parse_stack_trace(node.meta.get("stack_trace"), files),
                module_calls=tuple(node.meta.get("nn_module_stack", {})),
            )
        )
        if reads_sizes:
            sized.append(info)

    root_inputs = dict(zip(kwargs, signature.user_inputs))
    resolver = SizeReadResolver(traced, root_inputs)
    unresolved = []
    for info in sized:
        # Sizes are host values to an op with a tensor operand (``view``);
        # only an op with none (``arange``) docks on what it is sized by.
        if info.inputs:
            continue
        info.inputs = resolver.sources_of(info.name)
        if not info.inputs:
            unresolved.append(info.name)
    if unresolved:
        _log.warning(
            "%d op(s) read a size whose source tensor was not found in the "
            "model source; drawn with no input: %s",
            len(unresolved),
            ", ".join(unresolved),
        )
    return inventory


def _export_refining_ranges(
    module: Any,
    kwargs: dict[str, Any],
    dynamic_shapes: dict[str, Any],
    anchors: set[Any],
) -> Any:  # pragma: no cover
    """``torch.export`` the module, narrowing symbolic ranges the model itself guards."""
    import torch
    import torch.fx.proxy
    from torch._dynamo.exc import UserError, UserErrorType
    from torch.export.dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes

    # Export records only ``forward`` frames in a node's stack trace, which
    # loses the line of an op in a helper (``mix``, ``rope_tables``); frames
    # of registered anchors are kept too. This is a private torch hook.
    registry = torch.fx.proxy._STACK_TRACE_ANCHORS
    added = {(code.co_filename, code.co_name) for code in anchors} - registry
    registry |= added
    # Under grad mode each no_grad region becomes a nested graph, which
    # decomposing inlines with the stack trace of torch's generated code.
    with torch.no_grad():
        try:
            program = torch.export.export(
                module, (), kwargs, dynamic_shapes=dynamic_shapes, strict=False
            )
        except UserError as exc:
            if exc.error_type != UserErrorType.CONSTRAINT_VIOLATION:
                raise
            # The message trails prose after the fixes, which the refiner
            # would misread as one more fix; hand it the fix block alone.
            fixes = str(exc).partition("Suggested fixes:")[2].strip().split("\n\n")[0]
            refined = refine_dynamic_shapes_from_suggested_fixes(
                f"Suggested fixes:\n{fixes}", dynamic_shapes
            )
            _log.warning("the model constrains its symbolic dims; using %s", refined)
            program = torch.export.export(
                module, (), kwargs, dynamic_shapes=refined, strict=False
            )
        finally:
            registry -= added
        # Decomposing functionalizes an in-place write, which otherwise has no
        # edge to the op that reads it.
        return program.run_decompositions({})


def _model_code(module: Any) -> set[Any]:  # pragma: no cover
    """Code of every function in the source files that define the module's classes."""
    import inspect
    import sys

    code = set()
    names = {type(sub).__module__ for sub in module.modules()}
    for python_module in filter(None, (sys.modules.get(name) for name in names)):
        for value in vars(python_module).values():
            members = vars(value).values() if inspect.isclass(value) else [value]
            for member in members:
                function = inspect.unwrap(getattr(member, "__func__", member))
                if (
                    inspect.isfunction(function)
                    and function.__module__ == python_module.__name__
                ):
                    code.add(function.__code__)
    return code


def _is_tensor_valued(value: Any) -> bool:  # pragma: no cover
    import torch

    if isinstance(value, (tuple, list)):
        return any(isinstance(item, torch.Tensor) for item in value)
    return isinstance(value, torch.Tensor)


def _symbol_names(
    placeholders: list[Any], sizes: dict[str, int]
) -> dict[Any, Any]:  # pragma: no cover
    """Map each traced input symbol to its spec name, matched by example size."""
    import sympy

    by_size = {size: sympy.Symbol(name) for name, size in sizes.items()}
    symbols: dict[Any, Any] = {}
    for node in placeholders:
        for dim in getattr(node.meta.get("val"), "shape", ()):
            expr = getattr(getattr(dim, "node", None), "expr", None)
            if isinstance(expr, sympy.Symbol) and dim.node.hint in by_size:
                symbols[expr] = by_size[dim.node.hint]
    return symbols


def _format_dim(dim: Any, symbols: dict[Any, Any]) -> int | str:  # pragma: no cover
    if isinstance(dim, int):
        return dim
    expr = dim.node.expr.xreplace(symbols)
    return int(expr) if expr.is_number else str(expr).replace(" ", "")
