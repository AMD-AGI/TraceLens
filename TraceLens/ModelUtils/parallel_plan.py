###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read a model's declared device-split plans and apply partition counts.

Hugging Face configs carry three plans:

* ``base_model_tp_plan`` — how each weight is sharded across tensor-parallel ranks
* ``base_model_pp_plan`` — which tensors cross a pipeline-stage boundary
* ``base_model_ep_plan`` — how experts are split

The causal-LM class adds ``_tp_plan`` / ``_pp_plan`` for modules that sit outside
the base stack (the output head). The number of partitions is a runtime choice;
the plan says which axis splits and which collective the style implies. A width
the model declares (heads, intermediate size, expert count, layer count) either
divides by that count or it does not.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# What each Hugging Face parallel style does to the activation. The mesh the
# style counts against is "ep" for expert-axis styles and "tp" otherwise.
# ``moe_tp_experts`` is declared on the expert plan but the collective it names
# is an all-reduce across the tensor-parallel ranks.
_STYLE_EFFECTS: dict[str, tuple[str, str, str]] = {
    "colwise": ("output features", "sharded output", "tp"),
    "packed_colwise": ("output features", "sharded output", "tp"),
    "colwise_gather_output": ("output features", "gather", "tp"),
    "colwise_rep": ("output features", "gather", "tp"),
    "rowwise": ("input features", "all-reduce", "tp"),
    "packed_rowwise": ("input features", "all-reduce", "tp"),
    "rowwise_split_input": ("input features", "local output", "tp"),
    "rowwise_rep": ("input features", "replicated output", "tp"),
    "embedding_rowwise": ("embedding rows", "replicated output", "tp"),
    "sequence_parallel": ("sequence", "sharded sequence", "tp"),
    "replicated_with_grad_allreduce": ("replicated", "gradient all-reduce", "tp"),
    "grouped_gemm": ("experts", "sharded experts", "ep"),
    "ep_router": ("replicated router", "dispatch", "ep"),
    "megamoe_router": ("replicated router", "dispatch", "ep"),
    "megamoe_experts": ("experts", "sharded experts", "ep"),
    "moe_identity_expert": ("replicated", "none", "ep"),
    "moe_tp_experts": ("expert output", "all-reduce", "tp"),
    "all_reduce": ("output", "all-reduce", "tp"),
    "mla_kv_a_proj": ("output features", "sharded output", "tp"),
}

_TP_WIDTH_KEYS = {"intermediate_size", "moe_intermediate_size"}
_EP_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_PLAN_FIELDS = {
    "base_model_tp_plan": "tp",
    "base_model_pp_plan": "pp",
    "base_model_ep_plan": "ep",
    "_tp_plan": "tp_head",
    "_pp_plan": "pp_head",
}


@dataclass(frozen=True)
class ParallelDegree:
    """How many ways each axis is split. ``1`` leaves that axis whole."""

    tp: int = 1
    pp: int = 1
    ep: int = 1

    def requested(self) -> bool:
        return self.tp > 1 or self.pp > 1 or self.ep > 1


@dataclass(frozen=True)
class ClassPlans:
    """Plans declared on one config or model class."""

    name: str
    model_type: str | None = None
    tp: dict[str, str] | None = None
    ep: dict[str, str] | None = None
    pp: list[tuple[str, list[str], list[str]]] | None = None
    tp_head: dict[str, str] | None = None
    pp_head: list[tuple[str, list[str], list[str]]] | None = None
    sub_configs: dict[str, str] = field(default_factory=dict)
    tp_declared: bool = False
    pp_declared: bool = False
    ep_declared: bool = False


@dataclass(frozen=True)
class ParallelRule:
    """One module pattern and the split its style describes."""

    pattern: str
    style: str
    owner: str
    axis: str
    collective: str
    partitions: int
    mesh: str
    plan: str


@dataclass(frozen=True)
class WidthCheck:
    """Whether a declared width divides by the partition count."""

    name: str
    size: int
    partitions: int
    divides: bool
    mesh: str
    detail: str = ""


@dataclass(frozen=True)
class PipelineStage:
    """One pipeline partition: a half-open layer range and the modules that sit with it."""

    index: int
    layer_start: int
    layer_end: int
    modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuGroup:
    """GPUs that draw as one graph: the same stage structure and the same shapes.

    ``experts`` is every expert owned by those GPUs. ``expert_sets`` lists the
    distinct slices when the GPUs do not all hold the same experts. ``shards``
    lists tensor widths that do not divide evenly, at this group's size.
    """

    index: int
    gpus: tuple[int, ...]
    experts: tuple[int, ...] | None = None
    expert_sets: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    stage_indexes: tuple[int, ...] = ()
    shards: tuple[tuple[str, int], ...] = ()


@dataclass
class ParallelLayout:
    """The plans evaluated at one :class:`ParallelDegree`."""

    degree: ParallelDegree
    rules: list[ParallelRule] = field(default_factory=list)
    boundaries: list[tuple[str, list[str], list[str]]] = field(default_factory=list)
    stages: list[PipelineStage] = field(default_factory=list)
    checks: list[WidthCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mode: str = "inference"
    experts_per_tok: int | None = None
    hidden_size: int | None = None
    gpu_group: GpuGroup | None = None

    def fact_lines(self) -> list[str]:
        """Fact-sheet lines. Nested lines keep the four-space subline indent."""
        lines: list[str] = []
        lines.extend(_gpu_group_fact_lines(self))
        if self.degree.tp > 1:
            lines.append(f"Tensor parallel: {self.degree.tp} partitions")
            lines.extend(_rule_lines(self.rules, "tp"))
            lines.extend(_check_lines(self.checks, "tp"))
        if self.degree.pp > 1:
            lines.append(f"Pipeline parallel: {self.degree.pp} partitions")
            lines.extend(_pipeline_lines(self))
            lines.extend(_check_lines(self.checks, "pp"))
        if self.degree.ep > 1:
            lines.append(f"Expert parallel: {self.degree.ep} partitions")
            lines.extend(_rule_lines(self.rules, "ep"))
            lines.extend(_check_lines(self.checks, "ep"))
        communication = _communication_fact_lines(self)
        if communication:
            lines.append("Communications:")
            lines.extend(communication)
        lines.extend(self.notes)
        return lines


def parse_class_plans(source: str) -> list[ClassPlans]:
    """Read plan assignments from a configuration or modeling module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    plans: list[ClassPlans] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            parsed = _class_plans(node)
            if parsed is not None:
                plans.append(parsed)
    return plans


def build_parallel_layout(spec: Any, degree: ParallelDegree) -> ParallelLayout:
    """Evaluate ``spec``'s declared plans at ``degree``."""
    classes: list[ClassPlans] = []
    for path in _plan_sources(spec):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _log.warning("Could not read parallel plans from %s: %s", path, exc)
            continue
        classes.extend(parse_class_plans(text))
    raw = spec.raw_config if isinstance(getattr(spec, "raw_config", None), dict) else {}
    layout = layout_from_plans(classes, spec, degree, raw)
    for note in layout.notes:
        _log.warning("%s", note)
    for check in layout.checks:
        if not check.divides:
            _log.warning(
                "%s=%s does not divide by %s %s partitions%s",
                check.name,
                check.size,
                check.partitions,
                check.mesh,
                f" ({check.detail})" if check.detail else "",
            )
    return layout


def layout_from_plans(
    classes: list[ClassPlans],
    spec: Any,
    degree: ParallelDegree,
    raw_config: dict[str, Any] | None = None,
) -> ParallelLayout:
    """Build a layout from plans already parsed out of the model source."""
    raw = raw_config or {}
    layout = ParallelLayout(degree=degree)
    if not classes:
        if degree.requested():
            layout.notes.append(
                "No Hugging Face parallel plan was found in the model source."
            )
        return layout

    groups = _select_groups(classes, raw)
    _apply_tied_embedding(groups, raw)
    head = _architecture_head(classes, raw)
    regime = getattr(spec, "runtime_regime", None)
    layout.mode = getattr(regime, "mode", None) or "inference"
    topk = getattr(spec, "num_experts_per_tok", None)
    hidden = getattr(spec, "hidden_size", None)
    layout.experts_per_tok = topk if isinstance(topk, int) and topk > 0 else None
    layout.hidden_size = hidden if isinstance(hidden, int) and hidden > 0 else None
    if degree.tp > 1:
        tp_rules = _style_rules(groups, "tp", degree)
        if head is not None and head.tp_head:
            tp_rules.extend(_rules_from_styles(head.tp_head, head.name, degree, "tp"))
        if any(rule.plan == "tp" for rule in tp_rules):
            layout.rules.extend(tp_rules)
        else:
            layout.notes.append("The model declares no tensor-parallel plan.")
        layout.checks.extend(_tp_checks(spec, raw, degree.tp, tp_rules))
    if degree.ep > 1:
        ep_rules = _style_rules(groups, "ep", degree)
        if ep_rules:
            layout.rules.extend(ep_rules)
        elif not any(group.ep_declared and group.ep for group in groups):
            layout.notes.append("The model declares no expert-parallel plan.")
        layout.checks.extend(_ep_checks(spec, raw, degree.ep))
    if degree.pp > 1:
        boundaries = _pipeline_boundaries(groups, head)
        if boundaries:
            layout.boundaries = boundaries
            layout.stages = _pipeline_stages(boundaries, _layer_count(spec), degree.pp)
        else:
            layout.notes.append("The model declares no pipeline-parallel plan.")
        layout.checks.extend(_pp_checks(spec, degree.pp, layout.stages))
    return layout


def unique_gpu_groups(layout: ParallelLayout, spec: Any) -> list[GpuGroup]:
    """Group devices that would draw the same graph.

    Device ``0`` is pipeline rank 0, expert rank 0, tensor rank 0. The tensor
    rank changes fastest, then the expert rank, then the pipeline rank, matching
    a mesh ordered ``(pp, ep, tp)``. Two devices share a group when their
    pipeline stage has the same layer count and the same side modules, and every
    sharded width (including the local expert count) has the same size on both.
    """
    degree = layout.degree
    world = degree.tp * degree.pp * degree.ep
    expert_count = _declared_expert_count(spec, layout)
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for gpu in range(world):
        pp_rank, ep_rank, tp_rank = _gpu_coord(gpu, degree)
        key = (
            _stage_structure(layout, pp_rank),
            _shape_signature(layout, tp_rank, ep_rank, expert_count),
        )
        buckets.setdefault(key, []).append(gpu)
    ordered = sorted(
        (tuple(members) for members in buckets.values()), key=lambda item: item[0]
    )
    return [
        _make_gpu_group(index, gpus, layout, expert_count)
        for index, gpus in enumerate(ordered)
    ]


def gpu_group_report(groups: list[GpuGroup], degree: ParallelDegree) -> list[str]:
    """Lines for stdout: which GPUs landed in which subgraph."""
    lines: list[str] = []
    axes = [
        name
        for name, size in (
            ("pipeline", degree.pp),
            ("expert", degree.ep),
            ("tensor", degree.tp),
        )
        if size > 1
    ]
    if len(axes) > 1:
        lines.append(
            "GPU numbering: "
            + ", then ".join(axes)
            + " (the last rank changes fastest)"
        )
    lines.extend(
        f"subgraph {group.index}: {format_device_ids(group.gpus)}" for group in groups
    )
    return lines


def scope_payload_to_gpu_group(payload: dict[str, Any], layout: ParallelLayout) -> None:
    """Keep the stage body this group draws, when it is not the whole pipeline."""
    for collection in payload.get("graphCollections") or []:
        if not isinstance(collection, dict):
            continue
        for graph in collection.get("graphs") or []:
            if isinstance(graph, dict):
                scope_graph_to_gpu_group(graph, layout)


def scope_graph_to_gpu_group(graph: dict[str, Any], layout: ParallelLayout) -> None:
    """Drop modules and pipeline hops that this group's stage does not run."""
    group = layout.gpu_group
    if group is None or layout.degree.pp <= 1 or not layout.stages:
        return
    if set(group.stage_indexes) == set(range(len(layout.stages))):
        return
    representative = layout.stages[min(group.stage_indexes)]
    total = sum(stage.layer_end - stage.layer_start for stage in layout.stages)
    count = representative.layer_end - representative.layer_start
    if total > 0 and count != total:
        _rename_repeat(graph, total, count)
    keep = set(representative.modules)
    excluded = {
        module
        for stage in layout.stages
        for module in stage.modules
        if module not in keep
    }
    excluded -= _stack_names(layout)
    stack = _stack_names(layout)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return
    kept: list[Any] = []
    for node in nodes:
        if not isinstance(node, dict):
            kept.append(node)
            continue
        if _node_owned_by(node, excluded, stack):
            continue
        hop = _pipeline_hop(node)
        if hop is not None and representative.index not in hop:
            continue
        kept.append(node)
    graph["nodes"] = kept
    alive = {str(node.get("id")) for node in kept if isinstance(node, dict)}
    for node in kept:
        if not isinstance(node, dict):
            continue
        edges = node.get("incomingEdges")
        if isinstance(edges, list):
            node["incomingEdges"] = [
                edge for edge in edges if str(edge.get("sourceNodeId")) in alive
            ]
    _prune_empty_groups(graph, kept)


def annotate_parallel_nodes(graph: dict[str, Any], layout: ParallelLayout) -> None:
    """Stamp matching modules, and place each collective on the edge that leaves one."""
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in list(nodes):
        if not isinstance(node, dict):
            continue
        rule = _rule_for_module(node, layout.rules)
        if rule is None:
            continue
        attrs = node.setdefault("attrs", [])
        if any(item.get("key") == "parallel" for item in attrs):
            continue
        attrs.append(
            {
                "key": "parallel",
                "value": (
                    f"{rule.mesh} x{rule.partitions} {rule.style}; "
                    f"{rule.axis}; {rule.collective}"
                ),
            }
        )
    _add_communications(nodes, layout)


# Collectives that move an activation. Anything else (a sharded output, a
# replicated norm, local expert compute) stays on the module that produces it.
_ACTIVATION_COLLECTIVES = {
    "all-reduce": "all-reduce",
    "gather": "gather",
    "dispatch": "dispatch",
    "gradient all-reduce": "gradient all-reduce",
}

_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}

_COMM_STYLE = {
    "backgroundColor": "#fdebd0",
    "textColor": "#1a1a1a",
    "borderColor": "#e67e22",
}


def _communication_fact_lines(layout: ParallelLayout) -> list[str]:
    grouped: dict[tuple[str, str, int], list[str]] = {}
    for rule in layout.rules:
        kind = _rule_collective(rule, layout.mode)
        if kind is None:
            continue
        grouped.setdefault((kind, rule.mesh, rule.partitions), []).append(rule.pattern)
    lines: list[str] = []
    for (kind, mesh, partitions), patterns in grouped.items():
        lines.append(f"    {kind} {mesh} x{partitions}: {', '.join(patterns)}")
    if layout.degree.ep > 1 and any(
        rule.style in {"grouped_gemm", "megamoe_experts"}
        or rule.collective == "dispatch"
        for rule in layout.rules
    ):
        fanout = (
            f" x {layout.experts_per_tok} experts/token"
            if layout.experts_per_tok
            else ""
        )
        lines.append(f"    combine ep x{layout.degree.ep}: all-to-all{fanout}")
    hops = _visible_pipeline_hops(layout)
    if hops:
        carried = ", ".join(_stack_outputs(layout))
        lines.append(
            f"    send/recv {carried} across {hops} pipeline "
            f"boundar{'y' if hops == 1 else 'ies'}"
        )
    return lines


def _add_communications(nodes: list[Any], layout: ParallelLayout) -> None:
    planned: dict[str, dict[str, Any]] = {}
    module_keys: set[tuple[str, ...]] = set()
    for node in list(nodes):
        if not isinstance(node, dict):
            continue
        rule = _rule_for_module(node, layout.rules)
        kind = _rule_collective(rule, layout.mode) if rule is not None else None
        if kind is None or rule is None:
            continue
        key = _instance_key(_module_path(node), _pattern_tail(rule.pattern))
        if key is not None:
            module_keys.add(key)
        _plan_kind(planned, node, kind, _leaving_consumer_ids(nodes, node, key))
    _plan_inlined_collectives(nodes, layout, planned, module_keys)
    for host in _combine_hosts(nodes, layout):
        key = _instance_key(_module_path(host), ["experts"]) or tuple(
            _module_path(host)
        )
        # Tokens return from the experts before a tensor-parallel all-reduce
        # of that same output.
        _plan_kind(
            planned,
            host,
            "combine",
            _leaving_consumer_ids(nodes, host, key),
            front=True,
        )
    for node in list(nodes):
        if not isinstance(node, dict):
            continue
        entry = planned.get(str(node.get("id")))
        if not entry:
            continue
        _insert_collective_chain(
            nodes, node, entry["kinds"], layout, allowed=entry["allowed"]
        )
    _add_pipeline_transfers(nodes, layout)


def _rule_collective(rule: ParallelRule | None, mode: str) -> str | None:
    if rule is None or rule.partitions <= 1:
        return None
    if rule.collective == "gradient all-reduce":
        return "gradient all-reduce" if mode == "training" else None
    return _ACTIVATION_COLLECTIVES.get(rule.collective)


def _combine_hosts(nodes: list[Any], layout: ParallelLayout) -> list[dict[str, Any]]:
    """Where a routed-expert combine sits: the experts output, once per instance."""
    if layout.degree.ep <= 1:
        return []
    if not any(
        rule.style in {"grouped_gemm", "megamoe_experts"}
        or rule.collective == "dispatch"
        for rule in layout.rules
    ):
        return []
    hosts = [
        node for node in nodes if isinstance(node, dict) and _is_experts_module(node)
    ]
    seen = {tuple(_module_path(node)) for node in hosts}
    for node in _inlined_exits(nodes, ["experts"]):
        key = _instance_key(_module_path(node), ["experts"])
        if key is None or key in seen:
            continue
        hosts.append(node)
        seen.add(key)
    if hosts:
        return hosts
    gemms = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        rule = _rule_for_module(node, layout.rules)
        if rule is not None and rule.style in {"grouped_gemm", "megamoe_experts"}:
            gemms.append(node)
    downs = [node for node in gemms if _module_path(node)[-1:] == ["down_proj"]]
    if downs:
        return [downs[0]]
    return [gemms[-1]] if gemms else []


def _insert_collective_chain(
    nodes: list[Any],
    producer: dict[str, Any],
    kinds: list[str],
    layout: ParallelLayout,
    allowed: set[str] | None = None,
) -> None:
    current = producer
    restrict = allowed
    for kind in kinds:
        targets = _consumers(nodes, str(current.get("id")))
        if restrict is not None:
            targets = [node for node in targets if str(node.get("id")) in restrict]
        if not targets:
            return
        comm = _communication_node(
            current, kind, layout, partitions=_partitions_for(kind, layout)
        )
        nodes.append(comm)
        _retarget(nodes, str(current.get("id")), str(comm.get("id")), restrict)
        restrict = None
        current = comm


def _partitions_for(kind: str, layout: ParallelLayout) -> int:
    if kind in {"dispatch", "combine"}:
        return layout.degree.ep
    if kind == "send":
        return layout.degree.pp
    return layout.degree.tp


def _communication_node(
    producer: dict[str, Any],
    kind: str,
    layout: ParallelLayout,
    *,
    partitions: int,
    sink: bool = False,
    detail: str = "",
    repeated: bool = True,
) -> dict[str, Any]:
    repeats = _repeat_count(producer) if repeated else None
    label = kind if repeats is None else f"{kind} ×{repeats}"
    if detail:
        label = f"{label} {detail}"
    node_id = f"{producer.get('id')}/@comm:{kind}{':' + detail if detail else ''}"
    payload = _payload_text(producer, kind, layout, partitions)
    value = f"{_mesh_for(kind)} x{partitions} {kind}"
    if payload:
        value = f"{value}; {payload}"
    if repeats is not None:
        value = f"{value}; ×{repeats} layers"
    attrs = [
        {"key": "parallel", "value": value},
        {
            "key": "synthetic",
            "value": "communication_sink" if sink else "communication",
        },
    ]
    shape = _attr(producer, "output_shape")
    if shape:
        attrs.append({"key": "output_shape", "value": shape})
    return {
        "id": node_id,
        "label": label,
        "namespace": producer.get("namespace") or "",
        "attrs": attrs,
        "style": dict(_COMM_STYLE),
        "incomingEdges": [
            {
                "sourceNodeId": producer.get("id"),
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ],
    }


def _add_pipeline_transfers(nodes: list[Any], layout: ParallelLayout) -> None:
    hops = max(len(layout.stages) - 1, 0)
    if hops == 0:
        return
    producer = _hidden_producer(nodes)
    if producer is None:
        return
    carried = _stack_outputs(layout)
    for hop in range(hops):
        for name in carried:
            detail = f"stage {hop}→{hop + 1} {name}"
            send = _communication_node(
                producer,
                "send",
                layout,
                partitions=layout.degree.pp,
                detail=detail,
                repeated=False,
            )
            recv = _communication_node(
                send,
                "recv",
                layout,
                partitions=layout.degree.pp,
                sink=True,
                detail=detail,
                repeated=False,
            )
            nodes.append(send)
            nodes.append(recv)


def _consumers(nodes: list[Any], source_id: str) -> list[dict[str, Any]]:
    found = []
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("id")) == source_id:
            continue
        for edge in node.get("incomingEdges") or []:
            if str(edge.get("sourceNodeId")) == source_id:
                found.append(node)
                break
    return found


def _retarget(
    nodes: list[Any],
    source_id: str,
    comm_id: str,
    allowed: set[str] | None = None,
) -> None:
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("id")) == comm_id:
            continue
        if allowed is not None and str(node.get("id")) not in allowed:
            continue
        for edge in node.get("incomingEdges") or []:
            if str(edge.get("sourceNodeId")) == source_id:
                edge["sourceNodeId"] = comm_id


def _mesh_for(kind: str) -> str:
    if kind in {"dispatch", "combine"}:
        return "ep"
    if kind in {"send", "recv"}:
        return "pp"
    return "tp"


def _stack_outputs(layout: ParallelLayout) -> list[str]:
    for _key, inputs, outputs in layout.boundaries:
        if "hidden_states" in inputs and outputs:
            return list(outputs)
    return ["hidden_states"]


def _hidden_producer(nodes: list[Any]) -> dict[str, Any] | None:
    ranked: list[tuple[int, dict[str, Any]]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        label = str(node.get("label") or "")
        if label != "hidden_states" and not node_id.endswith("hidden_states"):
            continue
        score = 0
        if _parsed_shape(node)[0] is not None:
            score += 4
        if "@output" in node_id:
            score += 2
        if re.search(r"(?:^|/)\d+x_", str(node.get("namespace") or "")):
            score += 1
        ranked.append((score, node))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _payload_text(
    producer: dict[str, Any], kind: str, layout: ParallelLayout, partitions: int
) -> str | None:
    dims, dtype = _parsed_shape(producer)
    if kind in {"dispatch", "combine"}:
        hidden = layout.hidden_size
        if hidden and (not dims or str(dims[-1]) != str(hidden)):
            # The producer is an index or score, not the activation that moves.
            dims = ["B", "S", str(hidden)]
            if not _is_activation_dtype(dtype):
                dtype = None
        if dims and layout.experts_per_tok:
            dims = [*dims, str(layout.experts_per_tok)]
        return _byte_text(dims, dtype)
    if kind == "gather" and dims:
        return _byte_text(_shard_last(dims, partitions), dtype)
    return _byte_text(dims, dtype)


def _shard_last(dims: list[str], partitions: int) -> list[str]:
    if partitions <= 1 or not dims:
        return dims
    last = dims[-1]
    if last.isdigit() and int(last) % partitions == 0:
        return [*dims[:-1], str(int(last) // partitions)]
    return dims


def _parsed_shape(node: dict[str, Any]) -> tuple[list[str] | None, str | None]:
    value = _attr(node, "output_shape")
    if not value:
        return None, None
    start = value.find("[")
    end = value.find("]", start + 1)
    if start == -1 or end == -1:
        return None, None
    inner = value[start + 1 : end].strip()
    dims = [part.strip() for part in inner.split(",") if part.strip()] or None
    dtype = value[end + 1 :].strip() or None
    return dims, dtype


def _byte_text(dims: list[str] | None, dtype: str | None) -> str | None:
    if not dims:
        return None
    width = _dtype_bytes(dtype)
    if width is None:
        return None
    factors: list[int | str] = []
    for dim in dims:
        pieces = dim.split("*")
        if not pieces:
            return None
        for piece in pieces:
            if piece.isdigit():
                factors.append(int(piece))
            elif re.fullmatch(r"[A-Za-z_][\w]*", piece):
                factors.append(piece)
            else:
                return None
    numeric = 1
    symbols: list[str] = []
    for factor in factors:
        if isinstance(factor, int):
            numeric *= factor
        else:
            symbols.append(factor)
    if width is not None:
        numeric *= width
    if symbols:
        body = "*".join([str(numeric)] + symbols) if numeric != 1 else "*".join(symbols)
        return f"{body} bytes"
    return f"{numeric} bytes"


def _is_activation_dtype(dtype: str | None) -> bool:
    if not dtype:
        return False
    key = dtype.lower()
    return key in {"half"} or key.startswith(("float", "fp", "bf"))


def _dtype_bytes(dtype: str | None) -> int | None:
    if not dtype:
        return None
    key = dtype.lower()
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    if key.startswith(("float8", "fp8")):
        return 1
    return None


def _repeat_count(node: dict[str, Any]) -> int | None:
    text = f"{node.get('namespace') or ''}/{node.get('id') or ''}"
    counts = [int(match) for match in re.findall(r"(?:^|/)(\d+)x_", text)]
    return counts[-1] if counts else None


def _attr(node: dict[str, Any], key: str) -> str | None:
    for item in node.get("attrs") or []:
        if isinstance(item, dict) and item.get("key") == key:
            value = item.get("value")
            if isinstance(value, str):
                return value
    return None


def _visible_pipeline_hops(layout: ParallelLayout) -> int:
    total = max(len(layout.stages) - 1, 0)
    group = layout.gpu_group
    if group is None or not layout.stages or total == 0:
        return total
    if set(group.stage_indexes) == set(range(len(layout.stages))):
        return total
    representative = min(group.stage_indexes)
    last = len(layout.stages) - 1
    return (1 if representative > 0 else 0) + (1 if representative < last else 0)


def _gpu_group_fact_lines(layout: ParallelLayout) -> list[str]:
    group = layout.gpu_group
    if group is None:
        return []
    lines = [f"Subgraph {group.index}: {format_device_ids(group.gpus)}"]
    if (
        layout.degree.pp > 1
        and layout.stages
        and set(group.stage_indexes) != set(range(len(layout.stages)))
    ):
        shown = ", ".join(str(index) for index in group.stage_indexes)
        noun = "stage" if len(group.stage_indexes) == 1 else "stages"
        lines.append(f"    pipeline {noun} {shown}")
    if group.experts is not None:
        lines.append(f"    experts {_format_ranges(group.experts)}")
        for gpus, experts in group.expert_sets:
            lines.append(
                f"    {format_device_ids(gpus)}: experts {_format_ranges(experts)}"
            )
    for name, size in group.shards:
        lines.append(f"    {name} shard {size}")
    return lines


def _make_gpu_group(
    index: int,
    gpus: tuple[int, ...],
    layout: ParallelLayout,
    expert_count: int | None,
) -> GpuGroup:
    degree = layout.degree
    stage_indexes = tuple(sorted({_gpu_coord(gpu, degree)[0] for gpu in gpus}))
    experts: tuple[int, ...] | None = None
    expert_sets: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    if expert_count is not None:
        by_set: dict[tuple[int, ...], list[int]] = {}
        for gpu in gpus:
            ep_rank = _gpu_coord(gpu, degree)[1]
            by_set.setdefault(_expert_ids(expert_count, degree.ep, ep_rank), []).append(
                gpu
            )
        ordered = tuple(
            (tuple(members), owned)
            for owned, members in sorted(by_set.items(), key=lambda item: item[1][0])
        )
        experts = tuple(
            sorted({expert for _members, owned in ordered for expert in owned})
        )
        if len(ordered) > 1:
            expert_sets = ordered
    uneven = [
        check for check in layout.checks if check.mesh == "tp" and not check.divides
    ]
    shards: tuple[tuple[str, int], ...] = ()
    if uneven:
        tp_rank = _gpu_coord(gpus[0], degree)[2]
        shards = tuple(
            (check.name, _share(check.size, degree.tp, tp_rank)) for check in uneven
        )
    return GpuGroup(
        index=index,
        gpus=gpus,
        experts=experts,
        expert_sets=expert_sets,
        stage_indexes=stage_indexes,
        shards=shards,
    )


def _stage_structure(layout: ParallelLayout, pp_rank: int) -> tuple[Any, ...]:
    if layout.degree.pp <= 1 or not layout.stages or pp_rank >= len(layout.stages):
        return ("model",)
    stage = layout.stages[pp_rank]
    return (stage.layer_end - stage.layer_start, stage.modules)


def _shape_signature(
    layout: ParallelLayout, tp_rank: int, ep_rank: int, expert_count: int | None
) -> tuple[Any, ...]:
    shards = tuple(
        (check.name, _share(check.size, layout.degree.tp, tp_rank))
        for check in layout.checks
        if check.mesh == "tp"
    )
    local = (
        None
        if expert_count is None
        else len(_expert_ids(expert_count, layout.degree.ep, ep_rank))
    )
    return (shards, local)


def _declared_expert_count(spec: Any, layout: ParallelLayout) -> int | None:
    value = getattr(spec, "num_experts", None)
    if isinstance(value, int) and value > 0:
        return value
    for check in layout.checks:
        if check.mesh == "ep" and check.size > 0:
            return check.size
    return None


def _gpu_coord(gpu: int, degree: ParallelDegree) -> tuple[int, int, int]:
    block = degree.tp * degree.ep
    pp_rank = gpu // block
    rem = gpu % block
    return pp_rank, rem // degree.tp, rem % degree.tp


def _share(size: int, parts: int, rank: int) -> int:
    if parts <= 1:
        return size
    base, extra = divmod(size, parts)
    return base + (1 if rank < extra else 0)


def _expert_ids(count: int, parts: int, rank: int) -> tuple[int, ...]:
    if parts <= 1:
        return tuple(range(count))
    base, extra = divmod(count, parts)
    start = rank * base + min(rank, extra)
    owned = base + (1 if rank < extra else 0)
    return tuple(range(start, start + owned))


def format_device_ids(gpus: tuple[int, ...] | list[int]) -> str:
    """Compact ``GPUs 0-3, 5`` form. A single device reads ``GPU 0``."""
    return f"{'GPU' if len(set(gpus)) == 1 else 'GPUs'} {_format_ranges(gpus)}"


def _format_ranges(values: tuple[int, ...] | list[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return ""
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ", ".join(
        str(begin) if begin == end else f"{begin}-{end}" for begin, end in ranges
    )


def _rename_repeat(graph: dict[str, Any], total: int, count: int) -> None:
    pattern = re.compile(rf"(^|/){total}x_")

    def replace(text: str) -> str:
        return pattern.sub(lambda match: f"{match.group(1)}{count}x_", text)

    renamed = _map_strings(graph, replace)
    graph.clear()
    graph.update(renamed)


def _map_strings(value: Any, replace: Any) -> Any:
    if isinstance(value, str):
        return replace(value)
    if isinstance(value, list):
        return [_map_strings(item, replace) for item in value]
    if isinstance(value, dict):
        return {
            (_map_strings(key, replace) if isinstance(key, str) else key): _map_strings(
                item, replace
            )
            for key, item in value.items()
        }
    return value


def _stack_names(layout: ParallelLayout) -> set[str]:
    """The repeated layer stack, the first plan entry fed by ``hidden_states``."""
    for key, inputs, _outputs in layout.boundaries:
        if "hidden_states" in inputs:
            return {key}
    return set()


def _node_owned_by(node: dict[str, Any], excluded: set[str], stack: set[str]) -> bool:
    if not excluded:
        return False
    id_segments = [part for part in str(node.get("id") or "").split("/") if part]
    return _path_owned(id_segments, excluded, stack) or _path_owned(
        _node_segments(node), excluded, stack
    )


def _path_owned(segments: list[str], excluded: set[str], stack: set[str]) -> bool:
    for segment in segments:
        if segment in stack or re.fullmatch(r"\d+x_.*", segment):
            return False
        if segment in excluded:
            return True
    return False


def _pipeline_hop(node: dict[str, Any]) -> tuple[int, int] | None:
    text = f"{node.get('id') or ''} {node.get('label') or ''}"
    match = re.search(r"stage (\d+)→(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _prune_empty_groups(graph: dict[str, Any], nodes: list[Any]) -> None:
    alive = {""}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        parts: list[str] = []
        for part in str(node.get("namespace") or "").split("/"):
            if not part:
                continue
            parts.append(part)
            alive.add("/".join(parts))
    attrs = graph.get("groupNodeAttributes")
    if not isinstance(attrs, dict):
        return
    for key in list(attrs):
        if key not in alive:
            del attrs[key]


def _plan_sources(spec: Any) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in getattr(spec, "code_paths", None) or []:
        path = Path(str(raw))
        candidates = [path]
        if path.name.startswith("modeling_") and path.suffix == ".py":
            candidates.append(
                path.with_name("configuration_" + path.name[len("modeling_") :])
            )
        for candidate in candidates:
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            found.append(candidate)
    return found


def _class_plans(node: ast.ClassDef) -> ClassPlans | None:
    values: dict[str, Any] = {}
    declared: set[str] = set()
    model_type: str | None = None
    sub_configs: dict[str, str] = {}
    for stmt in node.body:
        target, value = _assignment(stmt)
        if target is None or value is None:
            continue
        if target == "model_type":
            literal = _literal(value)
            if isinstance(literal, str):
                model_type = literal
            continue
        if target == "sub_configs":
            sub_configs = _sub_config_names(value)
            continue
        field_name = _PLAN_FIELDS.get(target)
        if field_name is None:
            continue
        declared.add(field_name)
        values[field_name] = _plan_value(field_name, value)
    if not declared and model_type is None and not sub_configs:
        return None
    return ClassPlans(
        name=node.name,
        model_type=model_type,
        tp=values.get("tp"),
        ep=values.get("ep"),
        pp=values.get("pp"),
        tp_head=values.get("tp_head"),
        pp_head=values.get("pp_head"),
        sub_configs=sub_configs,
        tp_declared="tp" in declared,
        pp_declared="pp" in declared,
        ep_declared="ep" in declared,
    )


def _assignment(stmt: ast.stmt) -> tuple[str | None, ast.AST | None]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                continue
            parsed_key = _literal(key)
            if isinstance(parsed_key, str):
                result[parsed_key] = _literal(value)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand)
        if isinstance(value, int):
            return -value
    return None


def _sub_config_names(node: ast.AST) -> dict[str, str]:
    if not isinstance(node, ast.Dict):
        return {}
    names: dict[str, str] = {}
    for key, value in zip(node.keys, node.values):
        if key is None or not isinstance(value, ast.Name):
            continue
        parsed = _literal(key)
        if isinstance(parsed, str):
            names[parsed] = value.id
    return names


def _plan_value(field_name: str, node: ast.AST) -> Any:
    literal = _literal(node)
    if field_name in {"pp", "pp_head"}:
        return _pipeline_entries(literal)
    if isinstance(literal, dict):
        return {key: value for key, value in literal.items() if isinstance(value, str)}
    return None


def _pipeline_entries(
    literal: Any,
) -> list[tuple[str, list[str], list[str]]] | None:
    if not isinstance(literal, dict):
        return None
    entries: list[tuple[str, list[str], list[str]]] = []
    for key, value in literal.items():
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(item, list) for item in value)
            and all(isinstance(name, str) for side in value for name in side)
        ):
            entries.append((key, list(value[0]), list(value[1])))
    return entries


def _select_groups(classes: list[ClassPlans], raw: dict[str, Any]) -> list[ClassPlans]:
    by_name = {item.name: item for item in classes}
    model_type = raw.get("model_type")
    top = next((item for item in classes if item.model_type == model_type), None)
    if top is not None and (top.tp_declared or top.pp_declared or top.ep_declared):
        return [top]
    if top is not None and top.sub_configs:
        nested = [
            by_name[name]
            for name in top.sub_configs.values()
            if name in by_name
            and (
                by_name[name].tp_declared
                or by_name[name].pp_declared
                or by_name[name].ep_declared
            )
        ]
        if nested:
            return nested
    return [
        item
        for item in classes
        if item.tp_declared or item.pp_declared or item.ep_declared
    ]


def _architecture_head(
    classes: list[ClassPlans], raw: dict[str, Any]
) -> ClassPlans | None:
    by_name = {item.name: item for item in classes}
    for name in raw.get("architectures") or []:
        found = by_name.get(str(name))
        if found is not None and (found.tp_head or found.pp_head):
            return found
    return None


def _apply_tied_embedding(groups: list[ClassPlans], raw: dict[str, Any]) -> None:
    """Hugging Face adds ``embed_tokens: embedding_rowwise`` when the embedding is tied.

    ``PreTrainedConfig.__post_init__`` does this whenever the config ties the
    word embedding and already has a tensor-parallel plan. The key and the style
    are that rule, copied so a tied embedding is sharded the way the runtime
    shards it.
    """
    if not _tied(raw):
        return
    for index, group in enumerate(groups):
        if not group.tp:
            continue
        if "embed_tokens" in group.tp:
            return
        groups[index] = ClassPlans(
            name=group.name,
            model_type=group.model_type,
            tp={**group.tp, "embed_tokens": "embedding_rowwise"},
            ep=group.ep,
            pp=group.pp,
            tp_head=group.tp_head,
            pp_head=group.pp_head,
            sub_configs=group.sub_configs,
            tp_declared=group.tp_declared,
            pp_declared=group.pp_declared,
            ep_declared=group.ep_declared,
        )
        return


def _tied(raw: dict[str, Any]) -> bool:
    if raw.get("tie_word_embeddings") is True:
        return True
    for value in raw.values():
        if isinstance(value, dict) and value.get("tie_word_embeddings") is True:
            return True
    return False


def _style_rules(
    groups: list[ClassPlans], plan: str, degree: ParallelDegree
) -> list[ParallelRule]:
    rules: list[ParallelRule] = []
    for group in groups:
        styles = group.tp if plan == "tp" else group.ep
        if not styles:
            continue
        rules.extend(_rules_from_styles(styles, group.name, degree, plan))
    return rules


def _rules_from_styles(
    styles: dict[str, str], owner: str, degree: ParallelDegree, plan: str
) -> list[ParallelRule]:
    rules: list[ParallelRule] = []
    for pattern, style in styles.items():
        axis, collective, mesh = _STYLE_EFFECTS.get(style, ("unspecified", style, plan))
        partitions = degree.ep if mesh == "ep" else degree.tp
        rules.append(
            ParallelRule(
                pattern=pattern,
                style=style,
                owner=owner,
                axis=axis,
                collective=collective,
                partitions=partitions,
                mesh=mesh,
                plan=plan,
            )
        )
    return rules


def _pipeline_boundaries(
    groups: list[ClassPlans], head: ClassPlans | None
) -> list[tuple[str, list[str], list[str]]]:
    entries: list[tuple[str, list[str], list[str]]] = []
    seen: set[str] = set()
    for group in groups:
        for entry in group.pp or []:
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            entries.append(entry)
    if head is not None:
        for entry in head.pp_head or []:
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            entries.append(entry)
    return entries


def _layer_count(spec: Any) -> int | None:
    value = getattr(spec, "num_hidden_layers", None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _pipeline_stages(
    boundaries: list[tuple[str, list[str], list[str]]],
    layers: int | None,
    partitions: int,
) -> list[PipelineStage]:
    if layers is None or partitions < 1:
        return []
    stack = next(
        (key for key, inputs, _outputs in boundaries if "hidden_states" in inputs),
        None,
    )
    before: list[str] = []
    after: list[str] = []
    if stack is None:
        after = [key for key, _inputs, _outputs in boundaries]
    else:
        seen_stack = False
        for key, _inputs, _outputs in boundaries:
            if key == stack:
                seen_stack = True
                continue
            (after if seen_stack else before).append(key)
    ranges = _even_ranges(layers, partitions)
    stages: list[PipelineStage] = []
    for index, (start, end) in enumerate(ranges):
        modules: list[str] = []
        if index == 0:
            modules.extend(before)
        if index == len(ranges) - 1:
            modules.extend(after)
        stages.append(
            PipelineStage(
                index=index,
                layer_start=start,
                layer_end=end,
                modules=tuple(modules),
            )
        )
    return stages


def _even_ranges(total: int, partitions: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, partitions)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(partitions):
        count = base + (1 if index < extra else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def _tp_checks(
    spec: Any, raw: dict[str, Any], partitions: int, rules: list[ParallelRule]
) -> list[WidthCheck]:
    checks: list[WidthCheck] = []
    seen: set[str] = set()
    for name, size in (
        ("num_attention_heads", getattr(spec, "num_attention_heads", None)),
        ("num_key_value_heads", getattr(spec, "num_key_value_heads", None)),
        ("intermediate_size", getattr(spec, "intermediate_size", None)),
        ("moe_intermediate_size", getattr(spec, "moe_intermediate_size", None)),
    ):
        if isinstance(size, int) and size > 0:
            checks.append(_check(name, size, partitions, "tp"))
            seen.add(name)
    _collect_config_widths(raw, checks, seen, partitions, "", 0)
    if any(rule.style == "embedding_rowwise" for rule in rules) or any(
        rule.style.startswith("colwise") and "*" not in rule.pattern for rule in rules
    ):
        vocab = getattr(spec, "vocab_size", None)
        if isinstance(vocab, int) and vocab > 0 and "vocab_size" not in seen:
            checks.append(_check("vocab_size", vocab, partitions, "tp"))
    return checks


def _collect_config_widths(
    config: dict[str, Any],
    checks: list[WidthCheck],
    seen: set[str],
    partitions: int,
    prefix: str,
    depth: int,
) -> None:
    if depth > 2:
        return
    for key, value in config.items():
        if isinstance(value, dict):
            _collect_config_widths(
                value, checks, seen, partitions, f"{prefix}{key}.", depth + 1
            )
            continue
        if not isinstance(value, int) or value <= 0:
            continue
        if key in seen:
            continue
        if key.endswith("_heads") or key in _TP_WIDTH_KEYS:
            name = f"{prefix}{key}"
            checks.append(_check(name, value, partitions, "tp"))
            seen.add(key)


def _ep_checks(spec: Any, raw: dict[str, Any], partitions: int) -> list[WidthCheck]:
    experts = getattr(spec, "num_experts", None)
    if isinstance(experts, int) and experts > 0:
        return [_check("num_experts", experts, partitions, "ep")]
    for key in _EP_COUNT_KEYS:
        value = _find_int(raw, key)
        if value is not None:
            return [_check(key, value, partitions, "ep")]
    return []


def _pp_checks(
    spec: Any, partitions: int, stages: list[PipelineStage]
) -> list[WidthCheck]:
    layers = _layer_count(spec)
    if layers is None:
        return []
    sizes = [stage.layer_end - stage.layer_start for stage in stages]
    detail = ""
    if sizes:
        detail = "stages " + ", ".join(str(size) for size in sizes)
    return [_check("num_hidden_layers", layers, partitions, "pp", detail)]


def _find_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if isinstance(value, int) and value > 0:
        return value
    for nested in config.values():
        if isinstance(nested, dict):
            found = _find_int(nested, key)
            if found is not None:
                return found
    return None


def _check(
    name: str, size: int, partitions: int, mesh: str, detail: str = ""
) -> WidthCheck:
    return WidthCheck(
        name=name,
        size=size,
        partitions=partitions,
        divides=partitions > 0 and size % partitions == 0,
        mesh=mesh,
        detail=detail,
    )


def _rule_lines(rules: list[ParallelRule], plan: str) -> list[str]:
    grouped: dict[tuple[str, str, str, int, str], list[str]] = {}
    for rule in rules:
        if rule.plan != plan:
            continue
        key = (rule.owner, rule.style, rule.axis, rule.partitions, rule.collective)
        grouped.setdefault(key, []).append(rule.pattern)
    lines: list[str] = []
    owners = {key[0] for key in grouped}
    for (owner, style, axis, partitions, collective), patterns in grouped.items():
        owner_prefix = f"{owner}: " if len(owners) > 1 else ""
        joined = ", ".join(patterns)
        lines.append(
            f"    {owner_prefix}{style} x{partitions} ({axis}, {collective}): {joined}"
        )
    return lines


def _check_lines(checks: list[WidthCheck], mesh: str) -> list[str]:
    lines: list[str] = []
    for check in checks:
        if check.mesh != mesh:
            continue
        if check.divides:
            text = f"{check.name} {check.size} divides by {check.partitions}"
        else:
            text = f"{check.name} {check.size} does not divide by {check.partitions}"
        if check.detail:
            text = f"{text} ({check.detail})"
        lines.append(f"    {text}")
    return lines


def _pipeline_lines(layout: ParallelLayout) -> list[str]:
    lines: list[str] = []
    for stage in layout.stages:
        extras = f"; {', '.join(stage.modules)}" if stage.modules else ""
        lines.append(
            f"    stage {stage.index}: layers [{stage.layer_start}, {stage.layer_end})"
            f"{extras}"
        )
    for key, inputs, outputs in layout.boundaries:
        lines.append(f"    {key}: {', '.join(inputs)} -> {', '.join(outputs)}")
    return lines


_TILE_HEADS = {"seq", "sidefeed", "sideproducer", "branch"}


def _plan_kind(
    planned: dict[str, dict[str, Any]],
    node: dict[str, Any],
    kind: str,
    allowed: set[str],
    *,
    front: bool = False,
) -> None:
    if not allowed:
        return
    entry = planned.setdefault(str(node.get("id")), {"kinds": [], "allowed": set()})
    if kind not in entry["kinds"]:
        if front:
            entry["kinds"].insert(0, kind)
        else:
            entry["kinds"].append(kind)
    entry["allowed"].update(allowed)


def _plan_inlined_collectives(
    nodes: list[Any],
    layout: ParallelLayout,
    planned: dict[str, dict[str, Any]],
    module_keys: set[tuple[str, ...]],
) -> None:
    """A module expanded into ops still has one collective, on the edge that leaves it."""
    for rule in layout.rules:
        kind = _rule_collective(rule, layout.mode)
        if kind is None:
            continue
        tail = _pattern_tail(rule.pattern)
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            key = _instance_key(_module_path(node), tail)
            if key is None or key in module_keys:
                continue
            groups.setdefault(key, []).append(node)
        for key, members in groups.items():
            member_ids = {str(node.get("id")) for node in members}
            for node in members:
                allowed = _leaving_consumer_ids(nodes, node, key, member_ids=member_ids)
                _plan_kind(planned, node, kind, allowed)


def _inlined_exits(nodes: list[Any], tail: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for node in nodes:
        if not isinstance(node, dict) or _is_experts_module(node):
            continue
        key = _instance_key(_module_path(node), tail)
        if key is None:
            continue
        groups.setdefault(key, []).append(node)
    exits: list[dict[str, Any]] = []
    for key, members in groups.items():
        member_ids = {str(node.get("id")) for node in members}
        for node in members:
            if _leaving_consumer_ids(nodes, node, key, member_ids=member_ids):
                exits.append(node)
    return exits


def _leaving_consumer_ids(
    nodes: list[Any],
    producer: dict[str, Any],
    key: tuple[str, ...] | None,
    member_ids: set[str] | None = None,
) -> set[str]:
    """Consumers outside the module. Input boundaries stay on the producer."""
    found: set[str] = set()
    for consumer in _consumers(nodes, str(producer.get("id"))):
        consumer_id = str(consumer.get("id"))
        if member_ids is not None and consumer_id in member_ids:
            continue
        if _is_input_boundary(consumer):
            continue
        if key is not None and _in_instance(consumer, key):
            continue
        found.add(consumer_id)
    return found


def _is_input_boundary(node: dict[str, Any]) -> bool:
    for segment in str(node.get("id") or "").split("/"):
        if segment == "@input" or segment.startswith(("@input:", "@input_mirror")):
            return True
    return _attr(node, "synthetic") in {"@input", "@input_mirror"}


def _in_instance(node: dict[str, Any], key: tuple[str, ...]) -> bool:
    path = _module_path(node)
    return len(path) >= len(key) and tuple(path[: len(key)]) == key


def _instance_key(path: list[str], tail: list[str]) -> tuple[str, ...] | None:
    if not tail or len(path) < len(tail):
        return None
    width = len(tail)
    for index in range(len(path) - width + 1):
        if path[index : index + width] == tail:
            return tuple(path[: index + width])
    return None


def _is_experts_module(node: dict[str, Any]) -> bool:
    return _attr(node, "attr_name") == "experts" and _module_path(node)[-1:] == [
        "experts"
    ]


def _rule_for_module(
    node: dict[str, Any], rules: list[ParallelRule]
) -> ParallelRule | None:
    """The plan rule for this node when the node is the module the plan names."""
    leaf = _attr(node, "attr_name") or ""
    path = _module_path(node)
    if not leaf or not path or path[-1] != leaf:
        return None
    return _matching_rule(path, rules)


def _module_path(node: dict[str, Any]) -> list[str]:
    """Attribute path a plan pattern matches.

    Graph ids carry that path (``self_attn/seq:0:q_proj:q_proj:0``) while the
    namespace is the class that owns the op. A node whose id has no attribute
    path still matches through its namespace and ``attr_name``.
    """
    leaf = _attr(node, "attr_name") or ""
    from_id = _attributes_in_id(str(node.get("id") or ""))
    if from_id and (not leaf or from_id[-1] == leaf or leaf.startswith("@")):
        return from_id
    return _node_segments(node)


def _attributes_in_id(node_id: str) -> list[str]:
    path: list[str] = []
    for segment in node_id.split("/"):
        for name in _segment_attributes(segment):
            if not path or path[-1] != name:
                path.append(name)
    return path


def _segment_attributes(segment: str) -> list[str]:
    if not segment or segment.startswith("@"):
        return []
    parts = segment.split(":")
    if len(parts) >= 3 and parts[0] in _TILE_HEADS and parts[1].isdigit():
        names: list[str] = []
        for part in parts[2:]:
            name = _attribute_name(part)
            if name is None:
                break
            if not names or names[-1] != name:
                names.append(name)
        return names
    name = _attribute_name(segment)
    return [name] if name else []


def _attribute_name(part: str) -> str | None:
    if not part or part.startswith("@") or part.isdigit():
        return None
    # A source location is glued on with @ (``indexer@l466``). The attribute
    # is the name in front of that marker.
    name, _separator, _rest = part.partition("@")
    return name or None


def _node_segments(node: dict[str, Any]) -> list[str]:
    segments: list[str] = []
    for piece in re.split(r"[./]", str(node.get("namespace") or "")):
        if piece:
            segments.append(piece)
    attr = ""
    for item in node.get("attrs") or []:
        if isinstance(item, dict) and item.get("key") == "attr_name":
            attr = str(item.get("value") or "")
    leaf = attr or str(node.get("label") or "")
    for piece in re.split(r"[./]", leaf):
        if piece:
            segments.append(piece)
    return segments


def _matching_rule(
    segments: list[str], rules: list[ParallelRule]
) -> ParallelRule | None:
    best: ParallelRule | None = None
    best_len = -1
    for rule in rules:
        tail = _pattern_tail(rule.pattern)
        if not tail or len(segments) < len(tail):
            continue
        if segments[-len(tail) :] == tail and len(tail) > best_len:
            best = rule
            best_len = len(tail)
    return best


def _pattern_tail(pattern: str) -> list[str]:
    parts = pattern.split(".")
    if "*" in parts:
        star = max(index for index, part in enumerate(parts) if part == "*")
        parts = parts[star + 1 :]
    return [part for part in parts if part and part != "*"]
