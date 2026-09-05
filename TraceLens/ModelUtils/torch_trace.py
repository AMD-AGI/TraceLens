###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PyTorch-based model graph builder.

Replaces the AST-based pipeline with runtime tracing:

1. Instantiate model on meta device (zero memory, no weights)
2. Walk ``named_modules()`` for the module hierarchy
3. ``torch.fx.symbolic_trace`` per composite module for tensor ops
4. Forward hooks for ground-truth output shapes
5. Convert to Model Explorer graph format
"""

from __future__ import annotations

import logging
import re
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.fx
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

_log = logging.getLogger(__name__)

# ── Styling constants ────────────────────────────────────────────────────────

_DARK_TEXT = "#1a1a1a"
_WHITE_TEXT = "#ffffff"

_STYLE_INPUT = {"backgroundColor": "#d9e8f5", "textColor": _DARK_TEXT}
_STYLE_EMBEDDING = {"backgroundColor": "#27ae60", "textColor": _WHITE_TEXT}
_STYLE_LINEAR = {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT}
_STYLE_NORM = {"backgroundColor": "#f0e68c", "textColor": _DARK_TEXT}
_STYLE_ATTENTION = {"backgroundColor": "#5dade2", "textColor": _WHITE_TEXT}
_STYLE_ACTIVATION = {"backgroundColor": "#e67e22", "textColor": _WHITE_TEXT}
_STYLE_DEFAULT = {"backgroundColor": "#bdc3c7", "textColor": _DARK_TEXT}
_STYLE_OP = {"backgroundColor": "#ecf0f1", "textColor": _DARK_TEXT}


# ── Config patching ──────────────────────────────────────────────────────────

def _patch_config(config) -> None:
    """Ensure custom HF configs have all attributes the model code expects."""
    raw = config.to_dict()
    for key, val in raw.items():
        if not hasattr(config, key):
            setattr(config, key, val)
    for attr, default in [("max_length", 131072), ("use_cache", True)]:
        if not hasattr(config, attr):
            setattr(config, attr, default)


# ── Meta-device model instantiation ──────────────────────────────────────────

def _instantiate_meta(checkpoint: str | Path) -> tuple[Any, Any]:
    """Load config and instantiate model on meta device.

    Returns (model, config).
    """
    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
    _patch_config(config)

    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except ValueError:
            model = AutoModel.from_config(config, trust_remote_code=True)
    model.eval()
    return model, config


# ── Rotary embedding patching ────────────────────────────────────────────────

def _patch_rotary_embeddings(model: torch.nn.Module) -> None:
    """Patch rotary embedding modules to stay on meta device.

    Many HF models create CPU tensors inside rotary embedding forward().
    This causes meta-device forward passes to fail.
    """
    def _rotary_forward_meta(self, *args, **kwargs):
        dim = getattr(self, "dim", 64)
        max_seq = args[0] if args and isinstance(args[0], int) else 4096
        return torch.empty(max_seq, dim // 2, 2, device="meta", dtype=torch.float32)

    for _name, mod in model.named_modules():
        if "rotary" in type(mod).__name__.lower():
            mod.forward = types.MethodType(_rotary_forward_meta, mod)

    # Patch apply_rotary_pos_emb if present (shape-preserving)
    model_module = sys.modules.get(type(model).__module__)
    if model_module is None:
        # Try the inner transformer module
        for child in model.children():
            model_module = sys.modules.get(type(child).__module__)
            if model_module:
                break
    if model_module and hasattr(model_module, "apply_rotary_pos_emb"):
        model_module.apply_rotary_pos_emb = lambda x, *a, **k: x


# ── Shape capture via forward hooks ──────────────────────────────────────────

def _capture_shapes(
    model: torch.nn.Module,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> dict[str, tuple[int, ...]]:
    """Run a meta-device forward pass and capture per-module output shapes."""
    shapes: dict[str, tuple[int, ...]] = {}

    def _make_hook(name: str):
        def hook(_mod, _inp, output):
            try:
                if isinstance(output, torch.Tensor):
                    shapes[name] = tuple(output.shape)
                elif isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            shapes[name] = tuple(item.shape)
                            break
            except Exception:
                pass
        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(_make_hook(name)))

    dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
    try:
        with torch.no_grad():
            model(dummy, use_cache=False)
    except Exception:
        pass
    finally:
        for h in handles:
            h.remove()

    return shapes


def _symbolise(
    shape: tuple[int, ...], *, batch_size: int = 1, seq_len: int = 128
) -> str:
    """Convert shape tuple to symbolic string like ``B x S x 4096``."""
    parts = []
    for d in shape:
        if d == batch_size:
            parts.append("B")
        elif d == seq_len:
            parts.append("S")
        else:
            parts.append(str(d))
    return " x ".join(parts)


# ── torch.fx per-module tracing ──────────────────────────────────────────────

def _fx_trace_module(mod: torch.nn.Module) -> torch.fx.Graph | None:
    """Try to symbolically trace a module.  Returns None on failure."""
    try:
        traced = torch.fx.symbolic_trace(mod)
        return traced.graph
    except Exception:
        return None


_FX_OP_LABELS = {
    "silu": "SiLU",
    "relu": "ReLU",
    "gelu": "GELU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "chunk": "Chunk",
    "split": "Split",
    "cat": "Concat",
    "stack": "Stack",
    "mul": "Multiply",
    "add": "Add",
    "matmul": "MatMul",
    "bmm": "BatchMatMul",
    "baddbmm": "BatchMatMul",
    "scaled_dot_product_attention": "SDPA",
    "layer_norm": "LayerNorm",
    "dropout": "Dropout",
    "getitem": "Index",
    "view": "View",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "permute": "Permute",
    "contiguous": "Contiguous",
    "unsqueeze": "Unsqueeze",
    "squeeze": "Squeeze",
    "expand": "Expand",
    "mean": "Mean",
    "sum": "Sum",
}


def _fx_op_label(node: torch.fx.Node) -> str:
    """Human-readable label for an FX graph node."""
    if node.op == "call_module":
        return str(node.target).rsplit(".", 1)[-1]
    name = node.name
    # Strip trailing _N suffixes (getitem_1 → getitem)
    base = re.sub(r"_\d+$", "", name)
    return _FX_OP_LABELS.get(base, base.replace("_", " ").title())


def _is_interesting_op(node: torch.fx.Node) -> bool:
    """Filter out housekeeping ops, keeping only meaningful tensor operations."""
    if node.op in ("placeholder", "output"):
        return False
    if node.op == "call_module":
        return True
    name = re.sub(r"_\d+$", "", node.name)
    # Skip pure indexing/accessor ops
    if name in ("getitem",):
        return False
    return True


# ── Module classification ────────────────────────────────────────────────────

def _classify_module(mod: torch.nn.Module) -> str:
    """Classify a module for styling and labeling."""
    name = type(mod).__name__
    if isinstance(mod, torch.nn.Embedding):
        return "embedding"
    if isinstance(mod, (torch.nn.Linear,)):
        return "linear"
    if isinstance(mod, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
        return "norm"
    if re.search(r"(?i)(RMSNorm|LayerNorm|GroupNorm)", name):
        return "norm"
    if re.search(r"(?i)(Attention)", name):
        return "attention"
    if re.search(r"(?i)(Embedding)", name):
        return "embedding"
    if isinstance(mod, (torch.nn.SiLU, torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh)):
        return "activation"
    return "default"


def _style_for(category: str) -> dict[str, str]:
    return {
        "embedding": _STYLE_EMBEDDING,
        "linear": _STYLE_LINEAR,
        "norm": _STYLE_NORM,
        "attention": _STYLE_ATTENTION,
        "activation": _STYLE_ACTIVATION,
        "op": _STYLE_OP,
        "input": _STYLE_INPUT,
    }.get(category, _STYLE_DEFAULT)


def _module_label(mod: torch.nn.Module) -> str:
    """Friendly label for a module."""
    cls = type(mod).__name__
    if isinstance(mod, torch.nn.Linear):
        return f"Linear({mod.in_features}, {mod.out_features})"
    if isinstance(mod, torch.nn.Embedding):
        return f"Embedding({mod.num_embeddings}, {mod.embedding_dim})"
    return cls


# ── Layer deduplication ──────────────────────────────────────────────────────

def _detect_repeated_layers(model: torch.nn.Module) -> dict[str, int]:
    """Find repeated layer blocks (e.g. 40 identical GLMBlock layers).

    Returns dict mapping the container path to repeat count.
    """
    repeats: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 1:
            # Check if all children have the same class
            classes = {type(child).__name__ for child in mod}
            if len(classes) == 1:
                repeats[name] = len(mod)
    return repeats


# ── Graph building ───────────────────────────────────────────────────────────

def _shape_attrs(shape_str: str, dtype: str = "bfloat16") -> list[dict]:
    """Build output shape attributes for a node."""
    compact = shape_str.replace(" x ", "x")
    return [
        {"key": "output_shape", "value": f"{shape_str} {dtype}"},
        {"key": "output_dtype", "value": dtype},
    ]


def _output_metadata(shape_str: str, dtype: str = "bfloat16") -> list[dict]:
    compact = shape_str.replace(" x ", "x")
    return [
        {
            "id": "0",
            "attrs": [
                {"key": "shape", "value": f"{shape_str} {dtype}"},
                {"key": "tensor_shape", "value": f"{compact} {dtype}"},
                {"key": "dtype", "value": dtype},
            ],
        }
    ]


def build_graph(
    checkpoint: str | Path,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
    title: str | None = None,
) -> dict[str, Any]:
    """Build a complete Model Explorer payload from a checkpoint.

    This is the main entry point, replacing the entire AST pipeline.
    """
    model, config = _instantiate_meta(checkpoint)
    _patch_rotary_embeddings(model)

    # ── Capture shapes ───────────────────────────────────────────────────
    raw_shapes = _capture_shapes(model, seq_len=seq_len, batch_size=batch_size)
    shapes: dict[str, str] = {}
    for path, shape in raw_shapes.items():
        shapes[path] = _symbolise(shape, batch_size=batch_size, seq_len=seq_len)

    # ── Detect repeated layers ───────────────────────────────────────────
    repeats = _detect_repeated_layers(model)

    # ── Determine dtype ──────────────────────────────────────────────────
    dtype = str(getattr(config, "torch_dtype", "bfloat16")).replace("torch.", "")

    # ── Build model name ─────────────────────────────────────────────────
    model_name = title or getattr(config, "_name_or_path", str(checkpoint))
    model_type = getattr(config, "model_type", "unknown")

    # ── Walk module tree and build nodes ──────────────────────────────────
    nodes: list[dict[str, Any]] = []
    group_attrs: list[dict[str, Any]] = []
    group_configs: list[dict[str, Any]] = []
    edges_from: dict[str, str] = {}  # module_path → node_id

    # Input node
    nodes.append({
        "id": "@input",
        "label": "Tokenized text",
        "namespace": "",
        "attrs": [
            {"key": "synthetic", "value": "@input"},
            *_shape_attrs(f"B x S", "int64"),
        ],
        "style": _STYLE_INPUT,
        "outputsMetadata": _output_metadata("B x S", "int64"),
    })

    # Track which ModuleLists are repeated and which layer index to show
    skip_layers: set[str] = set()  # paths to skip (layers 1..N-1)
    layer_containers: dict[str, int] = {}  # container path → count
    for container_path, count in repeats.items():
        layer_containers[container_path] = count
        # Skip all but layer 0
        for i in range(1, count):
            skip_layers.add(f"{container_path}.{i}")

    # Collect all module paths and their children (preserving registration order)
    module_children: dict[str, list[str]] = defaultdict(list)
    module_map: dict[str, torch.nn.Module] = {}
    module_order: list[str] = []  # preserve named_modules() order
    for name, mod in model.named_modules():
        module_map[name] = mod
        if name:
            module_order.append(name)
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            module_children[parent].append(name)
        elif name:
            module_children[""].append(name)

    # Identify leaf vs composite modules
    leaf_modules: set[str] = set()
    composite_modules: set[str] = set()
    for name, mod in model.named_modules():
        if not name:
            continue
        children = list(mod.children())
        if children:
            composite_modules.add(name)
        else:
            leaf_modules.add(name)

    def _should_skip(path: str) -> bool:
        """Check if this path is in a skipped repeated layer."""
        for skip in skip_layers:
            if path == skip or path.startswith(skip + "."):
                return True
        return False

    def _namespace_for(path: str) -> str:
        """Compute the Model Explorer namespace (parent group) for a path."""
        parts = path.split(".")
        if len(parts) <= 1:
            return ""
        # Build namespace, collapsing repeated layers
        ns_parts = []
        for i, part in enumerate(parts[:-1]):
            prefix = ".".join(parts[: i + 1])
            if prefix in layer_containers:
                count = layer_containers[prefix]
                mod = module_map.get(f"{prefix}.0")
                cls_name = type(mod).__name__ if mod else part
                ns_parts.append(f"{count}x_{cls_name}")
                # Skip the layer index part
                continue
            # If this part is a digit (layer index), skip it
            if part.isdigit():
                continue
            mod = module_map.get(prefix)
            if mod and prefix in composite_modules:
                ns_parts.append(type(mod).__name__)
            else:
                ns_parts.append(part)
        return "/".join(ns_parts)

    def _node_id(path: str) -> str:
        """Build a node id, replacing layer indices with representative."""
        return path.replace(".", "/")

    # ── Try torch.fx on composite modules to get internal ops ────────────
    fx_graphs: dict[str, list[dict]] = {}  # module_path → list of op nodes

    for path in module_order:
        if path not in composite_modules:
            continue
        if _should_skip(path):
            continue
        mod = module_map[path]
        # Only trace modules that have at least one leaf child
        has_leaf = any(
            not list(child.children())
            for child in mod.children()
        )
        if not has_leaf:
            continue

        graph = _fx_trace_module(mod)
        if graph is None:
            continue

        # Convert FX graph nodes to Model Explorer nodes
        namespace = _namespace_for(path) or type(mod).__name__
        # If this is inside a repeated layer, adjust namespace
        op_nodes = []
        prev_id = None
        node_map: dict[str, str] = {}  # fx node name → our node id

        for fx_node in graph.nodes:
            if not _is_interesting_op(fx_node):
                # Track placeholders for edge wiring
                if fx_node.op == "placeholder":
                    node_map[fx_node.name] = "@input"
                continue

            if fx_node.op == "call_module":
                # This is a child module — will be added as a regular node
                target = str(fx_node.target)
                child_path = f"{path}.{target}"
                node_map[fx_node.name] = _node_id(child_path)
                continue

            # This is a tensor op (call_function, call_method)
            label = _fx_op_label(fx_node)
            op_id = f"{_node_id(path)}/{fx_node.name}"
            node_map[fx_node.name] = op_id

            # Build incoming edges from predecessors
            incoming = []
            for arg in fx_node.args:
                if isinstance(arg, torch.fx.Node) and arg.name in node_map:
                    incoming.append({
                        "sourceNodeId": node_map[arg.name],
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": str(len(incoming)),
                    })
                elif isinstance(arg, (tuple, list)):
                    for item in arg:
                        if isinstance(item, torch.fx.Node) and item.name in node_map:
                            incoming.append({
                                "sourceNodeId": node_map[item.name],
                                "sourceNodeOutputId": "0",
                                "targetNodeInputId": str(len(incoming)),
                            })

            op_node = {
                "id": op_id,
                "label": label,
                "namespace": namespace,
                "attrs": [{"key": "operation", "value": "tensor_op"}],
                "style": _STYLE_OP,
            }
            if incoming:
                op_node["incomingEdges"] = incoming
            op_nodes.append(op_node)

        if op_nodes:
            fx_graphs[path] = op_nodes

    # ── Add module nodes ─────────────────────────────────────────────────
    # Process in registration order (named_modules() order = forward order)
    prev_top_level_id = "@input"

    for path in module_order:
        if not path:
            continue
        if _should_skip(path):
            continue

        mod = module_map[path]
        category = _classify_module(mod)
        namespace = _namespace_for(path)
        node_id = _node_id(path)
        label = _module_label(mod)
        style = _style_for(category)

        # For composite modules that were fx-traced, they become namespaces
        if path in composite_modules and path not in fx_graphs:
            # Just a group — add group attributes
            group_label = type(mod).__name__
            # Check if this is a repeated layer container
            if path in layer_containers:
                continue  # ModuleList itself is not a node
            # Add namespace label
            if namespace:
                group_attrs.append({
                    "nodeId": namespace + "/" + type(mod).__name__,
                    "attrs": [
                        {"key": "class", "value": type(mod).__name__},
                    ],
                })
            continue

        if path in composite_modules and path in fx_graphs:
            # This composite was traced — its children + ops are in the graph
            # Add group attributes
            continue

        # Leaf module — add as a node
        attrs = [
            {"key": "class", "value": type(mod).__name__},
        ]
        shape_str = shapes.get(path)
        if shape_str:
            attrs.extend(_shape_attrs(shape_str, dtype))

        node: dict[str, Any] = {
            "id": node_id,
            "label": label,
            "namespace": namespace,
            "attrs": attrs,
            "style": style,
        }
        if shape_str:
            node["outputsMetadata"] = _output_metadata(shape_str, dtype)

        nodes.append(node)
        edges_from[path] = node_id

    # ── Add fx op nodes ──────────────────────────────────────────────────
    for path, op_nodes in fx_graphs.items():
        namespace = _namespace_for(path)
        mod = module_map[path]
        parent_ns = namespace + "/" + type(mod).__name__ if namespace else type(mod).__name__

        for op_node in op_nodes:
            # Adjust namespace to be under the parent module
            if not op_node["namespace"].startswith(parent_ns):
                op_node["namespace"] = parent_ns
            nodes.append(op_node)

    # ── Wire edges between sequential modules ────────────────────────────
    # For modules without fx-traced edges, wire sequentially based on
    # forward_calls order (which is the order they appear in named_modules)
    _wire_sequential_edges(nodes, model, module_map, shapes, layer_containers, skip_layers)

    # ── Add repeated layer group configs ─────────────────────────────────
    for container_path, count in layer_containers.items():
        mod_0 = module_map.get(f"{container_path}.0")
        if mod_0:
            cls_name = type(mod_0).__name__
            ns = _namespace_for(f"{container_path}.0")
            group_id = ns + f"/{count}x_{cls_name}" if ns else f"{count}x_{cls_name}"
            group_attrs.append({
                "nodeId": group_id,
                "attrs": [
                    {"key": "class", "value": cls_name},
                    {"key": "count", "value": str(count)},
                ],
            })

    # ── Build fact sheet ─────────────────────────────────────────────────
    raw_config = config.to_dict()
    fact_lines = [f"<b>{model_name}</b>"]
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "vocab_size", "max_position_embeddings",
                "model_type", "torch_dtype"):
        val = raw_config.get(key)
        if val is not None:
            fact_lines.append(f"  {key}: {val}")

    return {
        "name": model_name,
        "model_type": model_type,
        "source": "tracelens-torch-trace",
        "tracelensViewer": {
            "factSheet": "\n".join(fact_lines),
            "dtype": dtype,
        },
        "graphCollections": [
            {
                "label": model_name,
                "graphs": [
                    {
                        "id": "model",
                        "nodes": nodes,
                        "groupNodeAttributes": group_attrs,
                        "groupNodeConfigs": group_configs,
                    }
                ],
            }
        ],
    }


def _wire_sequential_edges(
    nodes: list[dict],
    model: torch.nn.Module,
    module_map: dict[str, torch.nn.Module],
    shapes: dict[str, str],
    layer_containers: dict[str, int],
    skip_layers: set[str],
) -> None:
    """Wire edges between nodes that don't already have incoming edges.

    Strategy: process nodes in insertion order (which matches forward
    execution order from named_modules). Track the last node emitted
    globally and per namespace level. A node without edges connects to
    the previous node in the same namespace, or (for the first node in
    a namespace) to the most recent node overall that shares a common
    ancestor namespace.
    """
    node_ids = {n["id"] for n in nodes}
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Track the last node seen at each namespace level.
    # When a node in ns="A/B/C" is emitted, we update last_in_ns for
    # "A/B/C", "A/B", "A", and "".
    last_in_ns: dict[str, str] = {}
    prev_in_ns: dict[str, str] = {}  # last node in *exactly* that ns

    def _update_last(ns: str, node_id: str) -> None:
        prev_in_ns[ns] = node_id
        # Bubble up to all ancestor namespaces
        parts = ns.split("/") if ns else []
        while True:
            key = "/".join(parts)
            last_in_ns[key] = node_id
            if not parts:
                break
            parts.pop()

    for node in nodes:
        if node["id"] == "@input":
            _update_last("", "@input")
            continue
        if "incomingEdges" in node:
            _update_last(node.get("namespace", ""), node["id"])
            continue

        ns = node.get("namespace", "")
        source_id = None

        # Check if there's a preceding node visible at this namespace level
        # (includes nodes from child namespaces, via bubble-up)
        if ns in last_in_ns:
            source_id = last_in_ns[ns]
        else:
            # First node in this namespace — find predecessor by walking
            # up the namespace hierarchy
            parts = ns.split("/") if ns else []
            while parts:
                parts.pop()
                parent = "/".join(parts)
                if parent in last_in_ns:
                    source_id = last_in_ns[parent]
                    break
            if source_id is None:
                source_id = last_in_ns.get("", "@input")

        if source_id and source_id in node_by_id:
            node["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ]
        _update_last(ns, node["id"])
