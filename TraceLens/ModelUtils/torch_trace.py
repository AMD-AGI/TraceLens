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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.fx
import transformers
from transformers import AutoConfig

_log = logging.getLogger(__name__)

# ── Styling constants ────────────────────────────────────────────────────────

_DARK_TEXT = "#1a1a1a"
_WHITE_TEXT = "#ffffff"

_STYLE_INPUT = {"backgroundColor": "#d9e8f5", "textColor": _DARK_TEXT}
_STYLE_OUTPUT = {"backgroundColor": "#d5f5d9", "textColor": _DARK_TEXT}
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

def _resolve_auto_classes(config) -> list[type]:
    """Return a ranked list of Auto classes to try, based on model card metadata.

    Checks ``config.auto_map`` first (explicit mapping from the repo),
    then falls back to the ``config.architectures`` class-name suffix.
    """
    # 1. auto_map: explicit mapping in the repo's config.json
    auto_map = getattr(config, "auto_map", None) or {}
    _PREFERRED_ORDER = [
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForConditionalGeneration",
        "AutoModel",
    ]
    for cls_name in _PREFERRED_ORDER:
        if cls_name in auto_map:
            return [getattr(transformers, cls_name)]

    # 2. architectures: infer candidates from class name suffix
    architectures = getattr(config, "architectures", None) or []
    if architectures:
        arch = architectures[0]
        _SUFFIX_CANDIDATES: dict[str, list[str]] = {
            "ForCausalLM": [
                "AutoModelForCausalLM",
            ],
            "ForConditionalGeneration": [
                "AutoModelForCausalLM",
                "AutoModelForSeq2SeqLM",
                "AutoModel",
            ],
            "ForSeq2SeqLM": [
                "AutoModelForSeq2SeqLM",
            ],
        }
        for suffix, candidates in _SUFFIX_CANDIDATES.items():
            if arch.endswith(suffix):
                return [getattr(transformers, c) for c in candidates]

    # 3. Last resort
    return [transformers.AutoModel]


def _instantiate_meta(checkpoint: str | Path) -> tuple[Any, Any]:
    """Load config and instantiate model on meta device.

    Inspects the model card (``auto_map`` / ``architectures``) to pick
    the correct Auto class, then instantiates on the meta device.

    Returns (model, config).
    """
    # Suppress the "torch_dtype is deprecated" warning from transformers
    _tf_logger = logging.getLogger("transformers.configuration_utils")
    _prev_level = _tf_logger.level
    _tf_logger.setLevel(logging.ERROR)

    try:
        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
        _patch_config(config)

        auto_classes = _resolve_auto_classes(config)
        last_err: Exception | None = None

        for auto_cls in auto_classes:
            try:
                _log.info("Trying %s for %s", auto_cls.__name__, checkpoint)
                with torch.device("meta"):
                    model = auto_cls.from_config(config, trust_remote_code=True)
                model.eval()
                _log.info("Instantiated with %s", auto_cls.__name__)
                return model, config
            except (ValueError, KeyError) as exc:
                last_err = exc
                continue

        raise ValueError(
            f"Could not instantiate {checkpoint} with any Auto class "
            f"({', '.join(c.__name__ for c in auto_classes)}): {last_err}"
        )
    finally:
        _tf_logger.setLevel(_prev_level)


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


def _infer_shapes_from_weights(
    model: torch.nn.Module,
    captured: dict[str, tuple[int, ...]],
    *,
    batch_size: int = 1,
    seq_len: int = 128,
) -> dict[str, tuple[int, ...]]:
    """Infer output shapes from module weight dimensions for modules
    that didn't get shapes from the forward pass (e.g. vision encoder)."""
    shapes = dict(captured)
    for name, mod in model.named_modules():
        if name in shapes or not name:
            continue
        if isinstance(mod, torch.nn.Linear):
            shapes[name] = (batch_size, seq_len, mod.out_features)
        elif isinstance(mod, torch.nn.Embedding):
            shapes[name] = (batch_size, seq_len, mod.embedding_dim)
        elif isinstance(mod, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            ns = getattr(mod, "normalized_shape", None)
            if ns:
                shapes[name] = (batch_size, seq_len, *ns)
        elif isinstance(mod, torch.nn.Conv2d):
            shapes[name] = (batch_size, mod.out_channels, seq_len, seq_len)
        elif isinstance(mod, torch.nn.Conv1d):
            shapes[name] = (batch_size, mod.out_channels, seq_len)
        elif isinstance(mod, torch.nn.Conv3d):
            shapes[name] = (batch_size, mod.out_channels, seq_len, seq_len, seq_len)
        elif hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
            # Generic: use last dim of weight as output feature dim
            w_shape = mod.weight.shape
            if len(w_shape) >= 1:
                shapes[name] = (batch_size, seq_len, w_shape[0])
        # For norm-like modules, try to infer from the hidden_size attr
        elif hasattr(mod, "hidden_size"):
            shapes[name] = (batch_size, seq_len, mod.hidden_size)
    return shapes


# ── Call-graph capture via forward hooks ─────────────────────────────────────

def _tensor_ids(x: Any) -> list[int]:
    """Extract Python id()s of all tensors in a nested structure."""
    if isinstance(x, torch.Tensor):
        return [id(x)]
    if isinstance(x, (tuple, list)):
        ids = []
        for item in x:
            ids.extend(_tensor_ids(item))
        return ids
    if isinstance(x, dict):
        ids = []
        for v in x.values():
            ids.extend(_tensor_ids(v))
        return ids
    return []


def _capture_call_graph(
    model: torch.nn.Module,
    composite_modules: set[str],
    *,
    seq_len: int = 128,
    batch_size: int = 1,
) -> dict[str, list[tuple[str, str]]]:
    """Capture dataflow edges between child modules of each composite.

    Runs a forward pass, tracking which tensor objects flow between
    children of each composite module. Returns a dict mapping composite
    module path → list of (source_child, target_child) edges.

    Children that receive the composite's own input (not a sibling's
    output) are marked as receiving from a virtual "@input" source.
    """
    # For each module, record pre-hook input tensor IDs and post-hook output IDs
    pre_inputs: dict[str, list[int]] = {}   # module path → input tensor IDs
    post_outputs: dict[str, list[int]] = {}  # module path → output tensor IDs
    call_order: dict[str, int] = {}  # module path → call order
    counter = [0]

    def _pre_hook(name: str):
        def hook(_mod, args, kwargs):
            pre_inputs[name] = _tensor_ids(args) + _tensor_ids(kwargs)
            call_order[name] = counter[0]
            counter[0] += 1
        return hook

    def _post_hook(name: str):
        def hook(_mod, _inp, output):
            post_outputs[name] = _tensor_ids(output)
        return hook

    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue
        handles.append(mod.register_forward_pre_hook(_pre_hook(name), with_kwargs=True))
        handles.append(mod.register_forward_hook(_post_hook(name)))

    dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device="meta")
    try:
        with torch.no_grad():
            model(dummy, use_cache=False)
    except Exception:
        pass
    finally:
        for h in handles:
            h.remove()

    # Build per-composite edge lists.
    edges: dict[str, list[tuple[str, str]]] = {}
    for comp_path in composite_modules:
        # Collect direct children
        children: list[str] = []
        for name in pre_inputs:
            if name.startswith(comp_path + "."):
                suffix = name[len(comp_path) + 1:]
                if "." not in suffix:
                    children.append(name)

        if len(children) < 2:
            continue

        children.sort(key=lambda c: call_order.get(c, 0))

        # Map tensor id → producing child (or "@input" for composite's own input)
        producer: dict[int, str] = {}
        # The composite's own input tensors are the "root" source
        comp_inputs = set(pre_inputs.get(comp_path, []))
        for tid in comp_inputs:
            producer[tid] = "@input"

        child_edges: list[tuple[str, str]] = []
        # Track untracked tensor IDs shared by multiple children
        untracked_consumers: dict[int, list[str]] = defaultdict(list)

        for child in children:
            # Check which producer this child's inputs come from
            sources: set[str] = set()
            for tid in pre_inputs.get(child, []):
                if tid in producer:
                    sources.add(producer[tid])
                else:
                    # Untracked tensor — record for shared-input detection
                    untracked_consumers[tid].append(child)
            for src in sorted(sources):
                if src != child:
                    child_edges.append((src, child))
            # Register this child's outputs
            for tid in post_outputs.get(child, []):
                producer[tid] = child

        # Children sharing the same untracked input tensor are parallel.
        # Treat them as all coming from "@input".
        children_with_edges = {tgt for _, tgt in child_edges}
        for _tid, consumers in untracked_consumers.items():
            if len(consumers) > 1:
                for child in consumers:
                    if child not in children_with_edges:
                        child_edges.append(("@input", child))
                        children_with_edges.add(child)

        if child_edges:
            # Deduplicate
            seen: set[tuple[str, str]] = set()
            unique = []
            for e in child_edges:
                if e not in seen:
                    seen.add(e)
                    unique.append(e)
            edges[comp_path] = unique

    return edges


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
    """Keep all tensor operations (fully expanded view)."""
    if node.op in ("placeholder", "output"):
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
        return "Linear"
    if isinstance(mod, torch.nn.Embedding):
        return "Embedding"
    # For trivial activation wrappers (e.g. transformers SiLUActivation),
    # try FX tracing to get the real op name
    if not list(mod.children()):
        cls_mod = type(mod).__module__ or ""
        if not cls_mod.startswith("torch."):
            try:
                graph = torch.fx.symbolic_trace(mod).graph
                ops = [n for n in graph.nodes if n.op not in ("placeholder", "output")]
                if len(ops) == 1:
                    return _fx_op_label(ops[0])
            except Exception:
                pass
    return cls


# ── Layer deduplication ──────────────────────────────────────────────────────

@dataclass
class LayerGroup:
    """A group of structurally identical layers inside a ModuleList."""
    class_name: str
    indices: list[int]
    representative: int  # index to keep

    @property
    def count(self) -> int:
        return len(self.indices)


def _detect_repeated_layers(
    model: torch.nn.Module,
) -> dict[str, list[LayerGroup]]:
    """Find repeated layer blocks, grouping by structural type.

    For heterogeneous ModuleLists (e.g. 34 linear-attn + 11 sparse-attn
    layers), creates separate groups for each distinct layer type.

    Returns dict mapping container path to list of LayerGroups.
    """
    repeats: dict[str, list[LayerGroup]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) <= 1:
            continue

        # Group children by their structural signature (class + child names)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, child in enumerate(mod):
            # Signature = class name + sorted child module class names
            child_sig = type(child).__name__
            child_structure = tuple(
                (cname, type(cmod).__name__)
                for cname, cmod in child.named_children()
            )
            sig = f"{child_sig}:{child_structure}"
            groups[sig].append(i)

        layer_groups = []
        for _sig, indices in groups.items():
            cls_name = type(mod[indices[0]]).__name__
            layer_groups.append(LayerGroup(
                class_name=cls_name,
                indices=indices,
                representative=indices[0],
            ))

        if any(g.count > 1 for g in layer_groups):
            repeats[name] = layer_groups

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


def _build_fact_sheet(model_name: str, config) -> str:
    """Build an HTML fact sheet from the config, including sub-configs."""
    lines = [f"<b>{model_name}</b>"]

    # Collect all config dicts (top-level + sub-configs like text_config)
    top = config.to_dict()
    sub_configs: dict[str, dict] = {}
    for key, val in top.items():
        if isinstance(val, dict) and any(
            k in val for k in ("hidden_size", "num_hidden_layers", "vocab_size")
        ):
            sub_configs[key] = val

    _FACT_KEYS = [
        "model_type", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "vocab_size", "max_position_embeddings",
        "dtype", "num_local_experts", "num_experts_per_tok",
        "moe_intermediate_size",
    ]

    def _emit(d: dict, prefix: str = "") -> None:
        for key in _FACT_KEYS:
            val = d.get(key)
            if val is not None:
                label = f"{prefix}{key}" if prefix else key
                lines.append(f"  {label}: {val}")

    if sub_configs:
        # Multi-modal: show sub-configs with headers
        # Top-level keys first
        for key in ("model_type", "dtype"):
            val = top.get(key)
            if val is not None:
                lines.append(f"  {key}: {val}")
        for section, d in sub_configs.items():
            section_label = section.replace("_config", "").replace("_", " ").title()
            lines.append(f"\n<b>{section_label}</b>")
            _emit(d)
    else:
        _emit(top)

    # Layer type breakdown
    return "\n".join(lines)


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
    hook_shapes = _capture_shapes(model, seq_len=seq_len, batch_size=batch_size)
    raw_shapes = _infer_shapes_from_weights(
        model, hook_shapes, batch_size=batch_size, seq_len=seq_len
    )
    shapes: dict[str, str] = {}
    for path, shape in raw_shapes.items():
        shapes[path] = _symbolise(shape, batch_size=batch_size, seq_len=seq_len)

    # ── Detect repeated layers ───────────────────────────────────────────
    repeats = _detect_repeated_layers(model)

    # ── Determine dtype ──────────────────────────────────────────────────
    dtype = getattr(config, "dtype", None)
    # Search sub-configs if top-level is None
    if dtype is None:
        for attr in ("text_config", "language_config", "decoder_config"):
            sub = getattr(config, attr, None)
            if sub is not None:
                dtype = getattr(sub, "dtype", None)
                if dtype is not None:
                    break
    dtype = str(dtype or "bfloat16").replace("torch.", "")

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
            *_shape_attrs("B x S", "int64"),
        ],
        "style": _STYLE_INPUT,
        "outputsMetadata": _output_metadata("B x S", "int64"),
    })

    # ── Build skip set from repeated layer groups ────────────────────────
    # Keep one representative per *distinct layer type* in each ModuleList.
    # When there are multiple types (e.g. LinearAttn vs SparseAttn), show
    # them as parallel branches so the viewer renders them side-by-side.
    skip_layers: set[str] = set()
    layer_group_map: dict[str, list[LayerGroup]] = repeats
    container_total: dict[str, int] = {}
    # Set of representative layer paths to keep
    representative_paths: set[str] = set()

    for container_path, groups in layer_group_map.items():
        total = sum(g.count for g in groups)
        container_total[container_path] = total
        for group in groups:
            representative_paths.add(f"{container_path}.{group.representative}")
        # Skip all layers that aren't a representative
        for i in range(total):
            path = f"{container_path}.{i}"
            if path not in representative_paths:
                skip_layers.add(path)

    # ── Collect module info (preserving registration order) ──────────────
    module_children: dict[str, list[str]] = defaultdict(list)
    module_map: dict[str, torch.nn.Module] = {}
    module_order: list[str] = []
    # Also build attr_name map: path → attribute name used in parent
    attr_names: dict[str, str] = {}

    for name, mod in model.named_modules():
        module_map[name] = mod
        if name:
            module_order.append(name)
            # The last part of the dot-path is the attr name
            attr_names[name] = name.rsplit(".", 1)[-1]
        if "." in name:
            parent = name.rsplit(".", 1)[0]
            module_children[parent].append(name)
        elif name:
            module_children[""].append(name)

    # Identify leaf vs composite modules
    composite_modules: set[str] = set()
    for name, mod in model.named_modules():
        if name and list(mod.children()):
            composite_modules.add(name)

    # ── Capture call graph for dataflow-aware edge wiring ────────────────
    call_graph = _capture_call_graph(
        model, composite_modules, seq_len=seq_len, batch_size=batch_size
    )

    def _should_skip(path: str) -> bool:
        for skip in skip_layers:
            if path == skip or path.startswith(skip + "."):
                return True
        return False

    # ── Build namespace labels using attr names + layer group labels ─────
    container_set: set[str] = set(container_total.keys())

    # Map each representative path → its group label (e.g. "3x LinearAttn+MLP")
    rep_group_label: dict[str, str] = {}
    for container_path, groups in layer_group_map.items():
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            # Build a descriptive label for this layer type
            rep_mod = module_map.get(rep_path)
            if rep_mod:
                attn_type = ""
                mlp_type = ""
                for cname, cmod in rep_mod.named_children():
                    if "attn" in cname and not attn_type:
                        attn_type = type(cmod).__name__
                    if "mlp" in cname and not mlp_type:
                        mlp_type = type(cmod).__name__
                parts_label = attn_type
                if mlp_type:
                    parts_label += f" + {mlp_type}"
                rep_group_label[rep_path] = (
                    f"{group.count}x {group.class_name} ({parts_label})"
                )
            else:
                rep_group_label[rep_path] = f"{group.count}x {group.class_name}"

    def _namespace_for(path: str) -> str:
        """Build a namespace using attr names, with layer group labels."""
        parts = path.split(".")
        if len(parts) <= 1:
            return ""
        ns_parts = []
        i = 0
        while i < len(parts) - 1:
            prefix = ".".join(parts[: i + 1])

            if prefix in container_set:
                # ModuleList — next part is the layer index
                if i + 1 < len(parts) - 1:
                    layer_prefix = ".".join(parts[: i + 2])
                    # Use per-group label if this is a representative
                    label = rep_group_label.get(layer_prefix)
                    if label:
                        ns_parts.append(label)
                    else:
                        # Fallback: total count
                        total = container_total[prefix]
                        mod = module_map.get(layer_prefix)
                        cls_name = type(mod).__name__ if mod else "Layer"
                        ns_parts.append(f"{total}x {cls_name}")
                    i += 2  # skip container + index
                    continue
                i += 1
                continue

            # Use attr name for composites, class name for clarity
            if prefix in composite_modules:
                attr = attr_names.get(prefix, parts[i])
                mod = module_map.get(prefix)
                cls_name = type(mod).__name__ if mod else attr
                # Use attr name if it's informative, class name otherwise
                if attr.isdigit():
                    ns_parts.append(cls_name)
                else:
                    ns_parts.append(f"{attr} ({cls_name})")
            i += 1
        return "/".join(ns_parts)

    def _node_id(path: str) -> str:
        return path.replace(".", "/")

    # ── Try torch.fx on composite modules to get internal ops ────────────
    fx_graphs: dict[str, list[dict]] = {}

    for path in module_order:
        if path not in composite_modules:
            continue
        if _should_skip(path):
            continue
        mod = module_map[path]
        has_leaf = any(not list(c.children()) for c in mod.children())
        if not has_leaf:
            continue

        graph = _fx_trace_module(mod)
        if graph is None:
            continue

        namespace = _namespace_for(path) or type(mod).__name__
        op_nodes = []
        node_map: dict[str, str] = {}

        for fx_node in graph.nodes:
            if not _is_interesting_op(fx_node):
                if fx_node.op == "placeholder":
                    node_map[fx_node.name] = "@input"
                continue

            if fx_node.op == "call_module":
                target = str(fx_node.target)
                child_path = f"{path}.{target}"
                node_map[fx_node.name] = _node_id(child_path)
                continue

            label = _fx_op_label(fx_node)
            op_id = f"{_node_id(path)}/{fx_node.name}"
            node_map[fx_node.name] = op_id

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

    # ── Try torch.fx on custom leaf modules to expand their ops ──────────
    for path in module_order:
        if path in composite_modules or path in fx_graphs:
            continue
        if _should_skip(path):
            continue
        # Skip if parent composite already has an FX graph (this leaf is
        # already referenced via call_module in the parent's graph)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        if parent in fx_graphs:
            continue
        mod = module_map[path]
        cls = type(mod)
        # Only expand non-standard modules (not torch.nn builtins)
        if cls.__module__ and cls.__module__.startswith("torch."):
            continue
        if list(mod.children()):
            continue  # has children — handled as composite

        graph = _fx_trace_module(mod)
        if graph is None:
            continue

        namespace = _namespace_for(path)
        # Build a parent namespace for the expanded ops
        attr = attr_names.get(path, cls.__name__)
        parent_ns = (
            namespace + f"/{attr} ({cls.__name__})"
            if namespace
            else f"{attr} ({cls.__name__})"
        )

        op_nodes = []
        node_map: dict[str, str] = {}

        for fx_node in graph.nodes:
            if not _is_interesting_op(fx_node):
                if fx_node.op == "placeholder":
                    node_map[fx_node.name] = "@input"
                continue

            if fx_node.op == "call_module":
                target = str(fx_node.target)
                child_path = f"{path}.{target}"
                node_map[fx_node.name] = _node_id(child_path)
                continue

            label = _fx_op_label(fx_node)
            op_id = f"{_node_id(path)}/{fx_node.name}"
            node_map[fx_node.name] = op_id

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
                "namespace": parent_ns,
                "attrs": [{"key": "operation", "value": "tensor_op"}],
                "style": _STYLE_OP,
            }
            if incoming:
                op_node["incomingEdges"] = incoming
            op_nodes.append(op_node)

        if op_nodes:
            fx_graphs[path] = op_nodes

    # ── Add module nodes ─────────────────────────────────────────────────
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

        if path in composite_modules and path not in fx_graphs:
            if path in container_set:
                continue  # ModuleList itself is not a node
            if namespace:
                group_attrs.append({
                    "nodeId": namespace + "/" + f"{attr_names.get(path, '')} ({type(mod).__name__})",
                    "attrs": [
                        {"key": "class", "value": type(mod).__name__},
                    ],
                })
            continue

        if path in composite_modules and path in fx_graphs:
            continue

        # Custom leaf module that was FX-expanded — skip the single node
        if path in fx_graphs:
            continue

        # Leaf module — add as a node
        attrs = [
            {"key": "class", "value": type(mod).__name__},
            {"key": "attr_name", "value": attr_names.get(path, "")},
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
        attr = attr_names.get(path, type(mod).__name__)
        parent_ns = (
            namespace + f"/{attr} ({type(mod).__name__})"
            if namespace
            else f"{attr} ({type(mod).__name__})"
        )

        for op_node in op_nodes:
            if not op_node["namespace"].startswith(parent_ns):
                op_node["namespace"] = parent_ns
            nodes.append(op_node)

    # ── Build parallel branch info for edge wiring ─────────────────────
    # For containers with multiple layer types, collect the namespace
    # prefixes of each parallel branch so the wiring can fan them out
    # from the same predecessor instead of chaining them sequentially.
    parallel_ns_groups: list[set[str]] = []
    for container_path, groups in layer_group_map.items():
        if len(groups) <= 1:
            continue
        branch_namespaces = set()
        for group in groups:
            rep_path = f"{container_path}.{group.representative}"
            # _namespace_for(rep_path + ".dummy") gives the branch NS
            # because ".dummy" is the leaf (excluded from namespace)
            branch_ns = _namespace_for(rep_path + ".dummy")
            branch_namespaces.add(branch_ns)
        parallel_ns_groups.append(branch_namespaces)

    # ── Output node ──────────────────────────────────────────────────────
    # Determine output shape from the model's last module
    output_shape = "B x S x V"
    # Check for lm_head or tied embeddings to get vocab size
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        output_shape = f"B x S x {model.lm_head.out_features}"
    elif hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
        vocab = model.language_model.embed_tokens.num_embeddings
        output_shape = f"B x S x {vocab}"
    elif hasattr(model, "embed_tokens"):
        vocab = model.embed_tokens.num_embeddings
        output_shape = f"B x S x {vocab}"

    nodes.append({
        "id": "@output",
        "label": "Logits",
        "namespace": "",
        "attrs": [
            {"key": "synthetic", "value": "@output"},
            *_shape_attrs(output_shape, dtype),
        ],
        "style": _STYLE_OUTPUT,
        "outputsMetadata": _output_metadata(output_shape, dtype),
    })

    # ── Wire edges ───────────────────────────────────────────────────────
    _wire_sequential_edges(
        nodes, model, module_map, shapes, {}, skip_layers,
        parallel_ns_groups=parallel_ns_groups,
        call_graph=call_graph,
    )

    # ── Add layer group attributes ───────────────────────────────────────
    for container_path, groups in layer_group_map.items():
        total = container_total[container_path]
        rep_path = f"{container_path}.0"
        mod_0 = module_map.get(rep_path)
        cls_name = type(mod_0).__name__ if mod_0 else "Layer"
        ns = _namespace_for(rep_path + ".dummy").rsplit("/", 1)[0] if "." in rep_path else ""
        group_id = ns if ns else f"{total}x {cls_name}"
        # Summarize layer types
        type_summary = ", ".join(
            f"{g.count}x {g.class_name}" for g in groups
        ) if len(groups) > 1 else f"{total}x {cls_name}"
        group_attrs.append({
            "nodeId": group_id,
            "attrs": [
                {"key": "class", "value": cls_name},
                {"key": "count", "value": str(total)},
                {"key": "layer_types", "value": type_summary},
            ],
        })

    # ── Build fact sheet ─────────────────────────────────────────────────
    fact_sheet = _build_fact_sheet(model_name, config)

    # Add layer type breakdown
    for container_path, groups in layer_group_map.items():
        if len(groups) > 1:
            fact_sheet += "\n\n<b>Layer Types</b>"
            for group in groups:
                rep = module_map.get(f"{container_path}.{group.representative}")
                if rep:
                    attn_type = ""
                    for cname, cmod in rep.named_children():
                        if "attn" in cname:
                            attn_type = type(cmod).__name__
                            break
                    mlp_type = ""
                    for cname, cmod in rep.named_children():
                        if "mlp" in cname:
                            mlp_type = type(cmod).__name__
                            break
                    desc = f"{attn_type}"
                    if mlp_type:
                        desc += f" + {mlp_type}"
                    fact_sheet += f"\n  {group.count}x {group.class_name}: {desc}"
                    fact_sheet += f"\n    layers: {group.indices}"

    return {
        "name": model_name,
        "model_type": model_type,
        "source": "tracelens-torch-trace",
        "tracelensViewer": {
            "factSheet": {
                "title": model_name,
                "body": fact_sheet,
                "bodyHtml": fact_sheet.replace("  ", "&nbsp;&nbsp;").replace("\n", "<br>\n"),
            },
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
    layer_containers: dict,
    skip_layers: set[str],
    *,
    parallel_ns_groups: list[set[str]] | None = None,
    call_graph: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    """Wire edges between nodes that don't already have incoming edges.

    Uses call_graph (captured during forward pass) to wire dataflow-aware
    edges between child modules of composite modules, falling back to
    sequential wiring when call_graph data is unavailable.

    For parallel branches (multiple layer types in the same container),
    all branches fan out from the same predecessor instead of being
    chained sequentially.
    chained sequentially.
    """
    node_ids = {n["id"] for n in nodes}
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}
    parallel_ns_groups = parallel_ns_groups or []
    call_graph = call_graph or {}

    # Build call-graph predecessor lookup at the node-ID level.
    # call_graph has paths like "a.b.c" → [("a.b.c.child1", "a.b.c.child2")].
    # We need to map node IDs (which use "/" and may be deeper) to their
    # dataflow predecessors.
    #
    # Strategy: for each composite's child-level edge (src_child → tgt_child),
    # find all nodes that belong to tgt_child (or are tgt_child itself),
    # and that are the FIRST node in tgt_child's subtree. Those should
    # have edges from the LAST node in src_child's subtree.
    #
    # We'll build this as: node_id → set of source child paths, and resolve
    # actual source node IDs during wiring.

    # Map: dotted child path → node ID prefix (using "/")
    def _path_to_id_prefix(dotted: str) -> str:
        return dotted.replace(".", "/")

    # For each composite, build set of children that have dataflow sources
    # (children that appear as targets in call_graph edges)
    cg_sources: dict[str, list[str]] = {}  # child_path → [source_child_paths]
    cg_children_with_no_sources: dict[str, set[str]] = {}  # comp → children with no incoming
    for comp_path, edges in call_graph.items():
        targets_seen: dict[str, list[str]] = defaultdict(list)
        all_children: set[str] = set()
        for src, tgt in edges:
            targets_seen[tgt].append(src)
            all_children.add(src)
            all_children.add(tgt)
        for tgt, srcs in targets_seen.items():
            cg_sources[tgt] = srcs
        # Children that only produce (never consume from a sibling)
        cg_children_with_no_sources[comp_path] = all_children - set(targets_seen.keys())

    # Build a lookup: namespace → set of sibling namespaces in parallel
    parallel_siblings: dict[str, set[str]] = {}
    for group in parallel_ns_groups:
        for ns in group:
            parallel_siblings[ns] = group

    # Track the last node seen at each namespace level.
    last_in_ns: dict[str, str] = {}

    # For parallel branches, snapshot the predecessor before the first
    # branch starts so all branches can connect from it.
    parallel_entry_point: dict[str, str] = {}  # branch_ns → predecessor_id
    parallel_entered: set[str] = set()  # branch namespaces we've seen

    def _update_last(ns: str, node_id: str, *, bubble: bool = True) -> None:
        if not bubble:
            last_in_ns[ns] = node_id
            return
        parts = ns.split("/") if ns else []
        while True:
            key = "/".join(parts)
            # Don't bubble up past a parallel branch boundary into sibling
            # territory — otherwise later parallel branches would see the
            # last node of a prior branch as their predecessor.
            if key in parallel_siblings and key != ns:
                # Only update this level, don't bubble further
                last_in_ns[key] = node_id
                break
            last_in_ns[key] = node_id
            if not parts:
                break
            parts.pop()

    def _find_branch_ns(ns: str) -> str | None:
        """Check if this namespace is inside a parallel branch."""
        for branch_ns in parallel_siblings:
            if ns == branch_ns or ns.startswith(branch_ns + "/"):
                return branch_ns
        return None

    # Track last node of each parallel branch for fan-in after branches
    branch_last_node: dict[str, str] = {}  # branch_ns → last node id

    # Track last node emitted per dotted module path (for call-graph wiring)
    last_node_for_path: dict[str, str] = {}
    # For @input call-graph sources, track the entry point of each composite
    composite_entry: dict[str, str] = {}  # comp_path → predecessor node ID

    def _node_id_to_path(nid: str) -> str:
        return nid.replace("/", ".")

    def _cg_find_sources(node_id: str, fallback_source: str | None) -> list[str] | None:
        """Check if this node is the first node of a child module that has
        call-graph predecessors. Returns list of source node IDs or None.

        ``fallback_source`` is the sequential predecessor — used when a
        call-graph edge points to "@input" (composite's own input).
        """
        path = _node_id_to_path(node_id)
        # Walk up the path to find a child that has call-graph sources
        parts = path.split(".")
        for depth in range(len(parts), 0, -1):
            child_path = ".".join(parts[:depth])
            if child_path in cg_sources:
                # Only apply to the FIRST node in this child's subtree
                if child_path in last_node_for_path:
                    return None  # already wired a node in this child

                # Determine the @input fallback: use the saved composite
                # entry point (predecessor before the composite started)
                comp_path = child_path.rsplit(".", 1)[0]
                input_source = composite_entry.get(comp_path, fallback_source)

                # Find the last node of each source child
                src_ids = []
                for src_path in cg_sources[child_path]:
                    if src_path == "@input":
                        if input_source:
                            src_ids.append(input_source)
                    else:
                        if src_path in last_node_for_path:
                            src_ids.append(last_node_for_path[src_path])
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique = []
                for s in src_ids:
                    if s not in seen:
                        seen.add(s)
                        unique.append(s)
                return unique if unique else None
        return None

    def _record_composite_entry(node_id: str, source_id: str | None) -> None:
        """Record the entry point for composites containing this node."""
        if not source_id:
            return
        path = _node_id_to_path(node_id)
        parts = path.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            comp_path = ".".join(parts[:depth])
            if comp_path in call_graph and comp_path not in composite_entry:
                composite_entry[comp_path] = source_id

    for node in nodes:
        if node["id"] == "@input":
            _update_last("", "@input")
            continue
        if "incomingEdges" in node:
            ns = node.get("namespace", "")
            branch_ns = _find_branch_ns(ns)
            if branch_ns is not None:
                branch_last_node[branch_ns] = node["id"]
            # Track for call-graph
            path = _node_id_to_path(node["id"])
            parts = path.split(".")
            for depth in range(1, len(parts) + 1):
                last_node_for_path[".".join(parts[:depth])] = node["id"]
            _update_last(ns, node["id"])
            continue

        ns = node.get("namespace", "")
        source_id = None

        # Check if this node is in a parallel branch
        branch_ns = _find_branch_ns(ns)

        if branch_ns is not None and branch_ns not in parallel_entered:
            # First time entering this parallel branch.
            # If this is the first branch in the group, snapshot the
            # current predecessor. For subsequent branches, reuse it.
            siblings = parallel_siblings[branch_ns]
            if not any(s in parallel_entered for s in siblings):
                # First branch in the group — find predecessor normally
                parts = ns.split("/") if ns else []
                while parts:
                    parts.pop()
                    parent = "/".join(parts)
                    if parent in last_in_ns:
                        source_id = last_in_ns[parent]
                        break
                if source_id is None:
                    source_id = last_in_ns.get("", "@input")
                # Save this as the entry point for all sibling branches
                for s in siblings:
                    parallel_entry_point[s] = source_id
            else:
                # Subsequent branch — use the saved entry point
                source_id = parallel_entry_point.get(branch_ns)
            parallel_entered.add(branch_ns)
        elif branch_ns is not None and ns in last_in_ns:
            # Inside an already-entered parallel branch — wire sequentially
            source_id = last_in_ns[ns]
        elif branch_ns is not None:
            # Inside an already-entered branch, but first node in a new
            # sub-namespace — walk up to find predecessor within the branch
            parts = ns.split("/") if ns else []
            while parts:
                parts.pop()
                parent = "/".join(parts)
                if parent in last_in_ns:
                    source_id = last_in_ns[parent]
                    break
            if source_id is None:
                source_id = last_in_ns.get("", "@input")
        else:
            # Not in a parallel branch. Check if we just left one —
            # if so, fan in from all branches.
            fan_in_sources = []
            for group in parallel_ns_groups:
                # Check: all branches entered AND this node is past them
                if group <= parallel_entered:
                    for sibling_ns in group:
                        if sibling_ns in branch_last_node:
                            fan_in_sources.append(branch_last_node[sibling_ns])
            if fan_in_sources:
                # Wire from all branch endpoints
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": src,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": "0",
                    }
                    for src in fan_in_sources
                    if src in node_by_id
                ]
                # Clear the groups so we don't fan-in again for next node
                for group in parallel_ns_groups:
                    if group <= parallel_entered:
                        for sibling_ns in group:
                            branch_last_node.pop(sibling_ns, None)
                _update_last(ns, node["id"])
                continue

            if ns in last_in_ns:
                source_id = last_in_ns[ns]
            else:
                # First node in this namespace — walk up
                parts = ns.split("/") if ns else []
                while parts:
                    parts.pop()
                    parent = "/".join(parts)
                    if parent in last_in_ns:
                        source_id = last_in_ns[parent]
                        break
                if source_id is None:
                    source_id = last_in_ns.get("", "@input")

        if branch_ns is not None:
            branch_last_node[branch_ns] = node["id"]

        # ── Call-graph-aware wiring: override source_id if we have
        #    dataflow information for this node's parent composite ────────
        _record_composite_entry(node["id"], source_id)
        cg_srcs = _cg_find_sources(node["id"], source_id)
        if cg_srcs is not None:
            if len(cg_srcs) == 1:
                source_id = cg_srcs[0]
            elif len(cg_srcs) > 1:
                # Multiple dataflow predecessors
                node["incomingEdges"] = [
                    {
                        "sourceNodeId": src,
                        "sourceNodeOutputId": "0",
                        "targetNodeInputId": str(i),
                    }
                    for i, src in enumerate(cg_srcs)
                    if src in node_by_id
                ]
                # Update tracking and continue
                path = _node_id_to_path(node["id"])
                parts = path.split(".")
                for depth in range(1, len(parts) + 1):
                    last_node_for_path[".".join(parts[:depth])] = node["id"]
                _update_last(ns, node["id"])
                continue

        if source_id and source_id in node_by_id:
            node["incomingEdges"] = [
                {
                    "sourceNodeId": source_id,
                    "sourceNodeOutputId": "0",
                    "targetNodeInputId": "0",
                }
            ]

        # Update last_node_for_path for call-graph tracking
        path = _node_id_to_path(node["id"])
        parts = path.split(".")
        for depth in range(1, len(parts) + 1):
            last_node_for_path[".".join(parts[:depth])] = node["id"]

        _update_last(ns, node["id"])
