#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel fusion: from kernel facts to candidates, kinds and impact estimates.

Dependencies flow one way: ``classify_kernels`` (kernel facts) <- this module <-
``orchestrator_prepare`` (builds candidates) and ``kernel_fusion_analysis`` (prices
them). Each class owns one decision:

- ``FusionForms``: the fused forms derived from the taxonomy's ``covers``, and the
  matrix and fusable predicates.
- ``FusionLabeler``: a group's kernel names to a fusion kind and confidence, or None
  when the group is not a candidate.
- ``FusionCandidateBuilder``: a trace to ``FusionCandidate`` records.
- ``FusionPricer``: a candidate to a ``FusionEstimate`` and the compute rows its
  finding suppresses.
"""

import ast
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from math import prod

import pandas as pd

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    PERF_ARG_COLS,
    PERF_THREAD_COLS,
    TARGET_HIGH,
    TARGET_LOW,
    TARGET_MID,
    calculate_efficiency,
    format_args,
    format_kernel_names,
    get_matrix_maf_tflops,
    perf_float,
    perf_row_key,
)
from TraceLens.Agent.Analysis.utils.classify_kernels import (
    COMPILED,
    LAYOUT_TRANSFORM,
    PRIMITIVES,
    UNKNOWN,
    KernelClass,
    KernelTaxonomy,
)
from TraceLens.PerfModel.perf_model import GEMM, BinaryElementwise, Reduce
from TraceLens.PerfModel.utils import name2bpe
from TraceLens.TreePerf.tree_perf import event_process_thread_names
from TraceLens.util import TraceEventUtils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The generic kinds of a group that matches no form.
GEMM_EPILOGUE = "gemm_epilogue"
ELEMENTWISE_CHAIN = "elementwise_chain"
# Kind names that differ from the joined elements; the rest are named by joining.
KIND_NAMES = {
    ("gemm", "softmax", "gemm"): "attention",
    ("collective", "norm"): "allreduce_norm",
    ("softmax", "topk"): "routing_topk",
}
# Fused types that are not fusion targets: MoE GEMMs and MoE routing, found by
# primitive, and compiled kernels, found by type (their pointwise/reduce mix is
# unknown, so no form can describe them).
NON_TARGET_COVERS = frozenset({"moe_gemm", "route"})
MATRIX_COVERS = frozenset({"gemm", "moe_gemm"})
# Copy-engine kernels and library collectives never sit in a same-stream seam.
NON_FUSABLE_COVERS = frozenset({"copy", "library_collective"})
NORM_ONLY_COVERS = frozenset({"norm", "batchnorm"})

MAX_FUSION_KERNEL_COUNT = 15
# Unrelated kernels allowed between two elements of a form.
MAX_GAP = 1

NN_MODULE_PREFIX = "nn.Module: "
# Case-A gates (comparative mode): trace 1 has more kernels than trace 2, by at
# most CASE_A_MAX_DELTA, and has at most CASE_A_MAX_N1 kernels in all.
CASE_A_MAX_DELTA = 15
CASE_A_MAX_N1 = 30
# Event categories that are never fusion candidate containers.
SKIP_CATEGORIES = frozenset(
    {
        *TraceEventUtils.GpuEventCategories,
        TraceEventUtils.CpuEventCategories.Runtime,
        TraceEventUtils.CpuEventCategories.Driver,
    }
)
# Confidence, bound and fusion-type labels used at more than one site.
CONFIDENCE_HIGH = "high"
BOUND_COMPUTE = "compute"
BOUND_MEMORY = "memory"
FUSION_TYPE_MATRIX = "matrix_compute"
FUSION_TYPE_MEMORY = "memory_bound"

# Memory/compute pipeline overlap fraction (0 = no overlap, 1 = perfect overlap).
OVERLAP_EFFICIENCY = 0.85
# Drop estimates whose best-case impact_score is below this threshold.
MIN_IMPACT_SCORE = 2.0
MIN_SAVINGS_MS = 0.1
# Fraction of a candidate's kernels that must have perf models: short and long.
MIN_MODELED_FRAC = 0.5
MIN_MODELED_FRAC_LONG = 0.75
LONG_GROUP_KERNELS = 5
NO_VALUE = "—"  # Table cell for a value the perf report does not provide

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _record_dict(record) -> dict:
    """A record as a JSON-ready dict: unset (None) fields are omitted, kernel tuples
    become lists."""
    out = {}
    for f in fields(record):
        value = getattr(record, f.name)
        if isinstance(value, tuple):
            value = [_record_dict(k) for k in value]
        if value is not None:
            out[f.name] = value
    return out


@dataclass(frozen=True)
class FusedForm:
    """The ordered unfused form of one fused kernel type."""

    kind: str
    elements: tuple[str, ...]
    replaces_span: bool


@dataclass(frozen=True)
class Label:
    """A group's fusion kind and confidence; ``span`` is the half-open range of the
    kernels a ``replaces_span`` kind replaces (None for every other kind)."""

    kind: str
    confidence: str
    span: tuple[int, int] | None


@dataclass(frozen=True)
class KernelRecord:
    """One kernel launch of a fusion candidate, in either pass and either trace."""

    name: str
    type: str
    dur_us: float
    perf_key: list | None = None
    gpu_op_uid: int | None = None
    data_in_mb: float | None = None
    data_out_mb: float | None = None


@dataclass(frozen=True, kw_only=True)
class FusionCandidate:
    """A group of kernels that one fused kernel could replace.

    A standalone candidate fills the ``kernel_count``, ``kernels``, ``ops`` and
    ``total_kernel_time_us`` fields; a comparative one fills the ``_trace1`` and
    ``_trace2`` fields instead.
    """

    module_name: str
    base_name: str
    parent_chain: list
    instance_count: int
    kernel_count: int | None = None
    kernel_count_trace1: int | None = None
    kernel_count_trace2: int | None = None
    delta: int | None = None
    kernels: tuple[KernelRecord, ...] | None = None
    kernels_trace1: tuple[KernelRecord, ...] | None = None
    kernels_trace2: tuple[KernelRecord, ...] | None = None
    kernel_type_signature: list
    kernel_type_summary: dict
    fusion_kind: str
    confidence: str
    source: str | None = None
    total_kernel_time_us: float | None = None
    total_kernel_time_us_trace1: float | None = None
    total_kernel_time_us_trace2: float | None = None
    ops: list | None = None
    ops_trace1: list | None = None
    ops_trace2: list | None = None

    @property
    def comparative(self) -> bool:
        return self.kernels_trace1 is not None

    @property
    def primary_kernels(self) -> tuple[KernelRecord, ...]:
        """The kernels this candidate is about: trace 1's when comparative."""
        return self.kernels_trace1 if self.comparative else self.kernels

    @property
    def primary_ops(self) -> list:
        return self.ops_trace1 if self.comparative else self.ops

    @property
    def time_us(self) -> float:
        return (
            self.total_kernel_time_us_trace1
            if self.comparative
            else self.total_kernel_time_us
        )

    def to_dict(self) -> dict:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FusionCandidate":
        kernels = {
            f.name: tuple(KernelRecord(**k) for k in data[f.name])
            for f in fields(cls)
            if f.type == tuple[KernelRecord, ...] | None and f.name in data
        }
        return cls(**{**data, **kernels})


@dataclass(frozen=True, kw_only=True)
class FusionEstimate:
    """One kernel_fusion row of ``kernel_fusion_metrics.json``."""

    operation: str
    category: str = "kernel_fusion"
    type: str = "kernel_fusion"
    impact_score: float
    impact_score_low: float
    impact_score_high: float
    estimation: str
    bound_type: str
    time_ms: float
    instance_count: int
    kernel_count: int
    modeled_kernel_count: int
    delta_kernel_count: int | None = None
    confidence: str
    fusion_kind: str
    fusion_type: str
    warning: str | None = None
    rows: list | None = None
    rows_trace1: list | None = None
    rows_trace2: list | None = None
    op_keys: list

    def to_dict(self) -> dict:
        return _record_dict(self)


# ---------------------------------------------------------------------------
# Forms and labeling
# ---------------------------------------------------------------------------


class FusionForms:
    """Fusion facts derived from the taxonomy: one form per fused type, plus the
    matrix and fusable predicates."""

    def __init__(self, taxonomy: KernelTaxonomy):
        named = {p for covers in KIND_NAMES for p in covers}
        unknown = (
            NON_TARGET_COVERS
            | MATRIX_COVERS
            | NON_FUSABLE_COVERS
            | NORM_ONLY_COVERS
            | named
        ) - PRIMITIVES
        if unknown:
            raise ValueError(f"Cover names not in PRIMITIVES: {sorted(unknown)}")
        self.taxonomy = taxonomy
        by_covers = {}
        for kc in taxonomy.types():
            if (
                self.is_fused(kc)
                and kc.type != COMPILED
                and NON_TARGET_COVERS.isdisjoint(kc.covers)
            ):
                kind = KIND_NAMES.get(kc.covers, "_".join(kc.covers))
                # A form with a matrix element replaces every kernel between its first
                # and last element, matrix kernels included (a flash kernel replaces
                # QK, mask, softmax, PV).
                by_covers.setdefault(
                    kc.covers,
                    FusedForm(kind, kc.covers, not MATRIX_COVERS.isdisjoint(kc.covers)),
                )
        self.forms = tuple(
            sorted(by_covers.values(), key=lambda f: len(f.elements), reverse=True)
        )
        self.kinds = frozenset(
            {f.kind for f in self.forms} | {GEMM_EPILOGUE, ELEMENTWISE_CHAIN}
        )

    @staticmethod
    def is_fused(kc: KernelClass) -> bool:
        return len(kc.covers) >= 2

    def is_matrix(self, kc: KernelClass | str) -> bool:
        if isinstance(kc, str):
            kc = self.taxonomy.by_type(kc)
        return not MATRIX_COVERS.isdisjoint(kc.covers)

    def matrix_mask(
        self, classes: Sequence[KernelClass | str], op_keys: Sequence[list | None]
    ) -> list[bool]:
        """Per kernel: a matrix kernel, or one launched by the same op (perf key) as a
        matrix kernel. An op that launches a GEMM or convolution also launches its
        layout and bookkeeping kernels; they are part of the matrix op, not an
        epilogue. A kernel without a perf key stands alone."""
        own = [self.is_matrix(kc) for kc in classes]
        matrix_ops = {
            tuple(key) for key, m in zip(op_keys, own, strict=True) if m and key
        }
        return [
            m or bool(key and tuple(key) in matrix_ops)
            for key, m in zip(op_keys, own, strict=True)
        ]

    def is_fusable(self, kc: KernelClass) -> bool:
        return NON_FUSABLE_COVERS.isdisjoint(kc.covers)

    def replaces_span(self, kind: str) -> bool:
        return any(f.replaces_span for f in self.forms if f.kind == kind)


class FusionLabeler:
    """Group of kernel names, in stream order, to a fusion kind and confidence."""

    def __init__(self, taxonomy: KernelTaxonomy, forms: FusionForms):
        self.taxonomy = taxonomy
        self.forms = forms

    def label(
        self, names: Sequence[str], op_keys: Sequence[list | None]
    ) -> Label | None:
        """The label of the group, or None when it is not a fusion candidate.

        ``op_keys`` are the kernels' launching-op perf keys, in the same order.
        """
        classes = [self.taxonomy.classify(n) for n in names]
        if sum(self.forms.is_fusable(kc) for kc in classes) < 2:
            return None
        if any(self.forms.is_fused(kc) for kc in classes):
            return None
        if any(self.taxonomy.is_compiled(n) for n in names):
            return None
        # One op's own multi-kernel implementation is not a seam between ops.
        if all(op_keys) and len({tuple(key) for key in op_keys}) == 1:
            return None

        form, span = self._seam(classes)
        if span:
            classes, op_keys = classes[span[0] : span[1]], op_keys[span[0] : span[1]]
        if len(classes) > MAX_FUSION_KERNEL_COUNT:
            return None
        matrix = self.forms.matrix_mask(classes, op_keys)
        # Also true when every kernel is matrix: nothing is left to fuse.
        if any(matrix) and all(
            kc.covers and NORM_ONLY_COVERS.issuperset(kc.covers)
            for kc, is_matrix in zip(classes, matrix)
            if not is_matrix
        ):
            return None

        if form is not None:
            kind = form.kind
        elif any(matrix):
            kind = GEMM_EPILOGUE
        else:
            kind = ELEMENTWISE_CHAIN
        if any(kc.type == UNKNOWN for kc in classes):
            confidence = "low"
        elif kind == GEMM_EPILOGUE:
            confidence = "medium"
        else:
            confidence = CONFIDENCE_HIGH
        return Label(kind, confidence, span)

    def _seam(self, classes: list) -> tuple[FusedForm | None, tuple | None]:
        """The first form (longest first) that matches the kernels, with its span."""
        for form in self.forms.forms:
            for start in range(len(classes)):
                stop = self._match(form, classes, start)
                if stop is not None:
                    return form, ((start, stop) if form.replaces_span else None)
        return None, None

    def _match(self, form: FusedForm, classes: list, start: int) -> int | None:
        """The stop of ``form`` matched from ``start``, or None.

        A kernel matches an element only if it covers that element alone.
        """
        if classes[start].covers != (form.elements[0],):
            return None
        pos = start
        for element in form.elements[1:]:
            window = range(pos + 1, len(classes))
            if not form.replaces_span:
                window = window[: 1 + MAX_GAP]
            for nxt in window:
                if classes[nxt].covers == (element,):
                    pos = nxt
                    break
                if form.replaces_span and self.forms.is_matrix(classes[nxt]):
                    return None
            else:
                return None
        return pos + 1


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------


class OpResolver:
    """Resolves a GPU kernel UID to its launching perf-report op and perf row key."""

    def __init__(self, tree, perf_ops: list):
        self.tree = tree
        self.kernel_op = {
            uid: ev for ev in perf_ops for uid in ev.get("gpu_events", [])
        }

    def perf_key(self, ev) -> list:
        """The perf-summary row key (op name, args, thread) of a perf-report op event."""
        args = ev.get("args", {})
        thread = event_process_thread_names(self.tree, ev)
        return list(
            perf_row_key(
                ev.get("name"),
                *(args.get(c) for c in PERF_ARG_COLS),
                *(thread[c] for c in PERF_THREAD_COLS),
            )
        )

    def fields(self, uid) -> dict:
        """Launching-op identity of kernel ``uid``: its op UID and perf row key."""
        op = self.kernel_op.get(uid)
        if op is None:
            return {"op_uid": uid, "perf_key": None}
        return {"op_uid": op.get("UID", uid), "perf_key": self.perf_key(op)}


def _efficiency_parts(eff: dict) -> tuple:
    """(pct, peak, unit, bound) of a ``calculate_efficiency`` result.

    ``bound`` reads ``compute-bound`` / ``memory-bound``; pct, peak and unit are
    None when the row has no efficiency or no bound to pick the peak from.
    """
    bound = eff["bound_type"]
    bound_label = f"{bound}-bound" if bound else None
    pct = eff["efficiency_percent"]
    if pct is None or bound is None:
        return None, None, None, bound_label
    peak, unit = (
        (eff["resolved_peak_maf"], "TFLOPS")
        if bound == "compute"
        else (eff["resolved_peak_mem_bw"], "TB/s")
    )
    return pct, peak, unit, bound_label


@dataclass
class _Group:
    """The instances of one module (or sibling parent) found in the trace.

    Each instance is a list of (kernel record, launching-op UID) pairs, in stream
    order; ``instances2`` holds trace 2's kernels of a comparative group.
    """

    module_name: str
    base_name: str
    parent_chain: list
    instances: list
    instances2: list = field(default_factory=list)


class FusionCandidateBuilder:
    """Trace to fusion candidates: walks the event tree, groups kernels by module,
    labels each group and dedups."""

    def __init__(self, tree, analyzer, perf_rows: dict, labeler: FusionLabeler):
        self.tree = tree
        self.perf_rows = perf_rows
        self.labeler = labeler
        self.taxonomy = labeler.taxonomy
        self.categorizer = analyzer.event_to_category
        self.perf_ops = analyzer.collect_unified_perf_events()
        self.resolver = OpResolver(tree, self.perf_ops)
        self._launches: dict[int, tuple[KernelRecord, int] | None] = {}

    # -- standalone ---------------------------------------------------------

    def standalone(self) -> list[FusionCandidate]:
        """Candidates from the live trace tree.

        Walks ``tree.events`` for modules that launch two or more kernels, then adds
        the single-kernel children of a parent whose sequence is not already inside
        a module candidate.
        """
        modules = self._module_groups()
        candidates = [c for g in modules for c in [self._candidate(g)] if c]
        accepted = [tuple(k.name for k in c.kernels) for c in candidates]
        for g in self._sibling_groups():
            names = tuple(k.name for k, _ in g.instances[0])
            if any(self._contains_run(names, hay) for hay in accepted):
                continue
            if c := self._candidate(g, "sibling_sequence"):
                candidates.append(c)
        candidates = self._dedup_by_kernel_set(candidates)
        print(
            f"  ✓ Fusion candidates: {len(modules)} unique module types, "
            f"{len(candidates)} candidates"
        )
        return candidates

    def _module_groups(self) -> list[_Group]:
        groups: dict = {}
        for ev in self.tree.events:
            if self.categorizer(ev) in SKIP_CATEGORIES:
                continue
            gpu_uids = ev.get("gpu_events", [])
            if len(gpu_uids) < 2:
                continue
            name = ev.get("name", "")
            base = self._strip_module_index(name)
            if not base:
                continue
            launches = [x for uid in gpu_uids if (x := self._launch(uid))]
            if base in groups:
                groups[base].instances.append(launches)
            elif len(launches) >= 2:
                groups[base] = _Group(name, base, self._parent_chain(ev), [launches])
        return list(groups.values())

    def _sibling_groups(self) -> list[_Group]:
        """Single-kernel ops under one parent, grouped by the parent's base name."""
        children: dict = defaultdict(list)
        for ev in self.perf_ops:
            gpu_uids = ev.get("gpu_events", [])
            parent_uid = ev.get("parent")
            if len(gpu_uids) != 1 or parent_uid is None:
                continue
            if launch := self._launch(gpu_uids[0]):
                children[parent_uid].append(launch)

        groups: dict = {}
        for parent_uid, launches in children.items():
            if len(launches) < 2:
                continue
            try:
                name = self.tree.get_UID2event(parent_uid).get("name", "")
            except (KeyError, IndexError):
                continue
            base = self._strip_module_index(name)
            if base in groups:
                groups[base].instances.append(launches)
            else:
                groups[base] = _Group(name, base, [], [launches])
        return list(groups.values())

    def _launch(self, uid) -> tuple[KernelRecord, int] | None:
        """Kernel ``uid`` as (record, launching-op UID); None if it is not a kernel.

        Memoized: the module walk meets each kernel once per ancestor.
        """
        if uid not in self._launches:
            self._launches[uid] = self._resolve_launch(uid)
        return self._launches[uid]

    def _resolve_launch(self, uid) -> tuple[KernelRecord, int] | None:
        try:
            k = self.tree.get_UID2event(uid)
            if self.categorizer(k) != TraceEventUtils.GpuEventCategories.Kernel:
                return None
            name = k.get("name", "")
            op = self.resolver.fields(uid)
            record = KernelRecord(
                name=name,
                type=self.taxonomy.classify(name).type,
                dur_us=k.get("dur", 0),
                perf_key=op["perf_key"],
                gpu_op_uid=uid,
            )
            return record, op["op_uid"]
        except (KeyError, IndexError):
            return None

    def _candidate(self, group: _Group, source: str | None = None):
        """The group as a standalone candidate, or None if the labeler excludes it."""
        label = self._label(group.instances[0])
        if label is None:
            return None
        instances = group.instances
        if label.span is not None:
            labels = [self._label(i) for i in instances]
            instances = [
                inst[lab.span[0] : lab.span[1]]
                for inst, lab in zip(instances, labels)
                if lab is not None and lab.span is not None
            ]
        kernels = tuple(self._with_data_moved(k) for k, _ in instances[0])
        signature, summary = self._type_summary(kernels)
        return FusionCandidate(
            module_name=group.module_name,
            base_name=group.base_name,
            parent_chain=group.parent_chain,
            instance_count=len(instances),
            kernel_count=len(kernels),
            kernels=kernels,
            kernel_type_signature=signature,
            kernel_type_summary=summary,
            fusion_kind=label.kind,
            confidence=label.confidence,
            source=source,
            total_kernel_time_us=sum(k.dur_us for inst in instances for k, _ in inst),
            ops=self._group_ops(instances),
        )

    def _label(self, instance: list) -> Label | None:
        """The label of one instance's (kernel, op UID) pairs."""
        return self.labeler.label(
            [k.name for k, _ in instance], [k.perf_key for k, _ in instance]
        )

    def _with_data_moved(self, k: KernelRecord) -> KernelRecord:
        """``k`` with the data in/out of its perf row, when it has one."""
        row = self.perf_rows.get(tuple(k.perf_key or ()))
        if row is None:
            return k
        data_in_mb, data_out_mb = self._row_data_in_out(row)
        return replace(k, data_in_mb=data_in_mb, data_out_mb=data_out_mb)

    @staticmethod
    def _contains_run(needle: tuple, hay: tuple) -> bool:
        return any(
            hay[i : i + len(needle)] == needle
            for i in range(len(hay) - len(needle) + 1)
        )

    # -- comparative --------------------------------------------------------

    def comparative(self, diff_stats: pd.DataFrame) -> list[FusionCandidate]:
        """Candidates from a diff_stats table (comparative mode).

        Walks trace 1's event tree at every level (nn.Module, aten::, etc.),
        cross-references descendant GPU kernel UIDs against ``diff_stats`` to find
        trace 2's counterparts via LCA, and deduplicates by ``base_name`` with
        instance accumulation (same as standalone).
        """
        if diff_stats.empty or "lowest_common_ancestor_id" not in diff_stats.columns:
            print("  ⚠️  diff_stats.csv is empty or missing expected columns")
            return []
        diff_stats = diff_stats.dropna(subset=["lowest_common_ancestor_id"])
        uid_to_t1, lca_to_t2 = self._diff_stats_lookups(diff_stats)
        if not uid_to_t1:
            print("  ⚠️  No trace1 kernel UIDs found in diff_stats.csv")
            return []

        groups: dict = {}
        for ev in self.tree.events:
            if self.categorizer(ev) in SKIP_CATEGORIES:
                continue
            gpu_uids = list(dict.fromkeys(ev.get("gpu_events", [])))
            if len(gpu_uids) < 2:
                continue
            name = ev.get("name", "")
            base = self._strip_module_index(name)
            if not base:
                continue

            t1, lca_ids = [], set()
            for uid in gpu_uids:
                if uid in uid_to_t1:
                    record, lca_id = uid_to_t1[uid]
                    op = self.resolver.fields(uid)
                    t1.append((replace(record, perf_key=op["perf_key"]), op["op_uid"]))
                    lca_ids.add(lca_id)
            if len(t1) < 2:
                continue
            t2 = [pair for lca_id in lca_ids for pair in lca_to_t2.get(lca_id, [])]
            if not self._passes_gates([k for k, _ in t1], [k for k, _ in t2]):
                continue

            if base in groups:
                groups[base].instances.append(t1)
                groups[base].instances2.append(t2)
            else:
                groups[base] = _Group(name, base, self._parent_chain(ev), [t1], [t2])

        candidates = [c for g in groups.values() for c in [self._comparative(g)] if c]
        return self._dedup_by_kernel_set(candidates)

    def _comparative(self, group: _Group):
        """The group as a comparative candidate, or None if trace 1 is not the slower
        side or the labeler excludes its kernels."""
        # Per-instance += keeps the totals reproducible against earlier runs.
        total1 = total2 = 0
        for inst1, inst2 in zip(group.instances, group.instances2, strict=True):
            total1 += sum(k.dur_us for k, _ in inst1)
            total2 += sum(k.dur_us for k, _ in inst2)
        if total1 < total2:
            return None
        label = self._label(group.instances[0])
        if label is None:
            return None
        kernels1 = tuple(k for k, _ in group.instances[0])
        kernels2 = tuple(k for k, _ in group.instances2[0])
        signature, summary = self._type_summary(kernels1)
        return FusionCandidate(
            module_name=group.module_name,
            base_name=group.base_name,
            parent_chain=group.parent_chain,
            instance_count=len(group.instances),
            kernel_count_trace1=len(kernels1),
            kernel_count_trace2=len(kernels2),
            delta=len(kernels1) - len(kernels2),
            kernels_trace1=kernels1,
            kernels_trace2=kernels2,
            kernel_type_signature=signature,
            kernel_type_summary=summary,
            fusion_kind=label.kind,
            confidence=label.confidence,
            total_kernel_time_us_trace1=total1,
            total_kernel_time_us_trace2=total2,
            ops_trace1=self._group_ops(group.instances),
            ops_trace2=self._group_ops(group.instances2),
        )

    def _diff_stats_lookups(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """UID to (trace 1 kernel, LCA id) and LCA id to trace 2 (kernel, op UID) pairs."""
        uid_to_t1 = {}
        lca_to_t2: dict = defaultdict(list)

        has_uid = "gpu_op_uid" in df.columns
        for _, row in df.iterrows():
            name = row["name"]
            ktype = self.taxonomy.classify(name).type
            lca_id = row["lowest_common_ancestor_id"]
            dur_us = float(row.get("kernel_time", 0) or 0)

            if (
                row["source"] == "trace1"
                and has_uid
                and pd.notna(row.get("gpu_op_uid"))
            ):
                uid = int(row["gpu_op_uid"])
                uid_to_t1[uid] = (
                    KernelRecord(name, ktype, dur_us, gpu_op_uid=uid),
                    lca_id,
                )
            elif row["source"] == "trace2":
                # Trace 2 has no tree here: its launching op comes from the row itself.
                op_uid = row.get("cpu_op_uid")
                perf_key = list(
                    perf_row_key(
                        row.get("cpu_op_name"),
                        *(row.get(c) for c in PERF_ARG_COLS),
                    )
                )
                lca_to_t2[lca_id].append(
                    (
                        KernelRecord(name, ktype, dur_us, perf_key),
                        row.get("gpu_op_uid") if pd.isna(op_uid) else op_uid,
                    )
                )
        return uid_to_t1, dict(lca_to_t2)

    def _passes_gates(self, t1: list, t2: list) -> bool:
        """Case-A gates and delta filters on the kernel records of one instance."""
        n1, n2 = len(t1), len(t2)
        if n1 <= n2:  # focus on trace 1 inefficiencies only
            return False
        if n1 - n2 > CASE_A_MAX_DELTA:  # large deltas are structural divergence
            return False
        if n1 > CASE_A_MAX_N1:  # oversized sub-trees are structural divergence
            return False
        if n2 < 1:  # no gap analysis against nothing
            return False

        delta = sorted(t1, key=lambda k: k.dur_us)[: n1 - n2]
        forms = self.labeler.forms
        # Skip if all delta kernels are infrastructure (copy engine, library collective).
        if all(not forms.is_fusable(self.taxonomy.by_type(k.type)) for k in delta):
            return False
        # Skip if all delta kernels are layout transforms (tile/autotuner choice).
        return not all(k.type == LAYOUT_TRANSFORM for k in delta)

    # -- shared -------------------------------------------------------------

    @staticmethod
    def _strip_module_index(name: str) -> str:
        return re.sub(r"_(\d+)$", "", name.removeprefix(NN_MODULE_PREFIX))

    def _parent_chain(self, ev) -> list:
        """Walk ancestors and return a list of cleaned parent names (outermost last)."""
        parent_chain = []
        ancestor = ev
        while self.tree.get_parent_event(ancestor) is not None:
            ancestor = self.tree.get_parent_event(ancestor)
            pname = ancestor.get("name", "")
            if pname:
                if pname.startswith(NN_MODULE_PREFIX):
                    pname = pname.removeprefix(NN_MODULE_PREFIX)
                elif "/" in pname:
                    pname = pname.rsplit("/", 1)[-1]
                parent_chain.append(pname)
        return parent_chain

    @staticmethod
    def _type_summary(kernels: Sequence[KernelRecord]) -> tuple[list, dict]:
        """The kernel types in order, and their counts."""
        signature = [k.type for k in kernels]
        return signature, dict(Counter(signature))

    @staticmethod
    def _group_ops(instances: list) -> list:
        """Group kernels from every instance by launching op (perf row key).

        ``time_us`` sums the kernels' durations over all instances; ``count`` is the
        number of distinct op invocations. Kernel names stack in first-seen order.
        Kernels with no perf row are not merged: each kernel name gets its own group.
        """
        groups: dict = {}
        op_uids: dict = defaultdict(set)
        for pairs in instances:
            for k, op_uid in pairs:
                # A kernel with no perf row is its own row, keyed by kernel name.
                key = tuple(k.perf_key) if k.perf_key else (None, k.name)
                g = groups.setdefault(
                    key,
                    {"perf_key": k.perf_key, "kernel_names": [], "time_us": 0.0},
                )
                if k.name not in g["kernel_names"]:
                    g["kernel_names"].append(k.name)
                g["time_us"] += k.dur_us
                op_uids[key].add(op_uid)
        return [{**g, "count": len(op_uids[key])} for key, g in groups.items()]

    @staticmethod
    def _dedup_by_kernel_set(candidates: list) -> list:
        """Keep one candidate per ordered kernel-name tuple: the one that is an
        nn.Module, then the one with the deeper parent chain."""

        def score(c):
            return c.module_name.startswith(NN_MODULE_PREFIX), len(c.parent_chain)

        seen: dict = {}
        for c in candidates:
            kset = tuple(k.name for k in c.primary_kernels)
            prev = seen.get(kset)
            if prev is None or score(c) > score(prev):
                seen[kset] = c
        return list(seen.values())

    @classmethod
    def _row_data_in_out(cls, row):
        """(data_in_mb, data_out_mb) of a perf-summary row."""
        dm = perf_float(row, "Data Moved (MB)")
        pp = row.get("perf_params", "")
        return cls._compute_data_in_out(
            row.get("op category", ""), "" if pd.isna(pp) else pp, dm
        )

    @classmethod
    def _compute_data_in_out(cls, op_category, perf_params_str, data_moved_mb):
        """Split Data Moved into (data_in_mb, data_out_mb) in proportion to the op's
        input and output sizes; evenly when they are unknown."""
        if not data_moved_mb:
            return None, None
        try:
            params = ast.literal_eval(str(perf_params_str)) if perf_params_str else {}
        except (ValueError, SyntaxError, TypeError):
            params = {}
        sizes = cls._in_out_sizes(op_category, params)
        if sizes is None:
            return data_moved_mb / 2, data_moved_mb / 2
        size_in, size_out = sizes
        total = size_in + size_out
        return data_moved_mb * size_in / total, data_moved_mb * size_out / total

    @staticmethod
    def _in_out_sizes(op_category, params: dict) -> tuple | None:
        """(input, output) size of a perf-summary op, in elements or bytes, or None
        when its category and params do not give them."""
        if op_category == Reduce.category and "num_input_elems" in params:
            dtype_in, dtype_out = params["dtype_in_out"]
            bpe_in = name2bpe(dtype_in)
            # Like Reduce: an output without its own dtype keeps the input's.
            bpe_out = name2bpe(dtype_out) if dtype_out else bpe_in
            if None in (bpe_in, bpe_out):
                return None
            return (
                params["num_input_elems"] * bpe_in,
                params["num_output_elems"] * bpe_out,
            )
        if op_category == GEMM.category:
            M, N, K = params.get("M"), params.get("N"), params.get("K")
            return (M * K + K * N, M * N) if all((M, N, K)) else None
        if op_category == BinaryElementwise.category and "shape_in1" in params:
            shape1, shape2 = params["shape_in1"], params["shape_in2"]
            dtype1, dtype2, dtype_out = params["dtype_in1_in2_out"]
            bpe1, bpe2 = name2bpe(dtype1), name2bpe(dtype2)
            bpe_out = BinaryElementwise.output_bpe(
                shape1, shape2, bpe1, bpe2, dtype_out
            )
            if None in (bpe1, bpe2, bpe_out):
                return None
            nelems_out = prod(BinaryElementwise.get_broadcast_shape(shape1, shape2))
            return prod(shape1) * bpe1 + prod(shape2) * bpe2, nelems_out * bpe_out
        return None


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class FusionPricer:
    """Candidate to impact estimate: roofline projection (standalone) or the measured
    gap to trace 2 (comparative), as ``impact_score`` (% of E2E GPU time)."""

    def __init__(
        self,
        forms: FusionForms,
        peaks: tuple,
        perf_rows: dict,
        baseline_ms: float,
        *,
        display_ms: float | None = None,
        perf_rows2: dict | None = None,
        baseline2_ms: float = 0,
        same_platform: bool = False,
    ):
        """``peaks`` is (peak_bw_tbs, peak_maf_tflops); the trace 2 arguments price
        comparative candidates. Trace 2 Efficiency and Bound are filled only when
        ``same_platform``. ``display_ms`` is the denominator of the detail rows'
        %E2E (default ``baseline_ms``); multi_trace passes the focus node's time."""
        self.forms = forms
        self.peak_bw_tbs, self.peak_maf_tflops = peaks
        self.peak_bw_bytes_s = self.peak_bw_tbs * 1e12
        self.matrix_maf = get_matrix_maf_tflops(self.peak_maf_tflops)
        self.vector_maf = self.peak_maf_tflops.get("vector_bf16", self.matrix_maf)
        self.perf_rows = perf_rows
        self.baseline_ms = baseline_ms
        self.display_ms = baseline_ms if display_ms is None else display_ms
        self.perf_rows2 = perf_rows2
        self.baseline2_ms = baseline2_ms
        self.same_platform = same_platform

    @staticmethod
    def shortlist(
        candidates: list[FusionCandidate], baseline_ms: float
    ) -> list[FusionCandidate]:
        """Drop candidates whose total time cannot reach MIN_IMPACT_SCORE of baseline,
        to avoid pricing them. It is a strict upper bound, as savings never exceed
        the group's time."""
        # Microseconds per millisecond over percent.
        min_time_us = baseline_ms * (1000 / 100) * MIN_IMPACT_SCORE
        return [c for c in candidates if c.time_us >= min_time_us]

    def price_all(
        self, candidates: list[FusionCandidate]
    ) -> list[tuple[FusionCandidate, FusionEstimate]]:
        """(candidate, estimate) for every priced candidate, best impact first.

        If ``baseline_ms`` is missing or non-positive, ``impact_score`` is undefined;
        this returns an empty list and emits a stderr warning.
        """
        if self.baseline_ms <= 0:
            print(
                f"[kernel_fusion.FusionPricer.price_all] "
                f"baseline_ms={self.baseline_ms!r} is invalid; skipping impact_score "
                f"computation",
                file=sys.stderr,
            )
            return []
        pairs = []
        for c in candidates:
            estimate = self.price_comparative(c) if c.comparative else self.price(c)
            if estimate is not None:
                pairs.append((c, estimate))
        return sorted(pairs, key=lambda p: p[1].impact_score, reverse=True)

    def suppressed_op_keys(self, pairs: list) -> list:
        """Perf keys of the ops whose every kernel a high-confidence finding removes.

        A fusion removes every non-matrix op; a matrix op only when the kind
        replaces its span (attention). A GEMM that keeps running keeps its own
        tuning finding.
        """
        keys = []
        for c, est in pairs:
            if est.confidence != CONFIDENCE_HIGH:
                continue
            matrix_keys = {
                tuple(k.perf_key)
                for k in c.primary_kernels
                if k.perf_key and self.forms.is_matrix(k.type)
            }
            replaces = self.forms.replaces_span(est.fusion_kind)
            keys += [k for k in est.op_keys if replaces or tuple(k) not in matrix_keys]
        return keys

    def price_comparative(self, c: FusionCandidate) -> FusionEstimate | None:
        """Impact of a comparative candidate: the measured GPU time gap between trace
        1 and trace 2 kernels. None if the candidate should be skipped."""
        gap_us = c.total_kernel_time_us_trace1 - c.total_kernel_time_us_trace2
        if gap_us <= 0:
            return None

        savings_high = gap_us / 1000
        if savings_high < MIN_SAVINGS_MS:
            return None

        savings_low = (TARGET_LOW / TARGET_HIGH) * savings_high
        savings_mid = (TARGET_MID / TARGET_HIGH) * savings_high
        time_ms = c.total_kernel_time_us_trace1 / 1000

        impact_score_high = savings_high / self.baseline_ms * 100
        impact_score_low = savings_low / self.baseline_ms * 100
        impact_score_mid = savings_mid / self.baseline_ms * 100

        if impact_score_high < MIN_IMPACT_SCORE:
            return None

        has_matrix_ops = any(self.forms.is_matrix(k.type) for k in c.kernels_trace1)
        return FusionEstimate(
            operation=c.base_name,
            impact_score=round(impact_score_mid, 2),
            impact_score_low=round(impact_score_low, 2),
            impact_score_high=round(impact_score_high, 2),
            estimation="measured",
            bound_type=BOUND_COMPUTE if has_matrix_ops else BOUND_MEMORY,
            time_ms=round(time_ms, 3),
            instance_count=c.instance_count,
            kernel_count=len(c.kernels_trace1),
            modeled_kernel_count=len(c.kernels_trace1),
            delta_kernel_count=c.delta,
            confidence=c.confidence,
            fusion_kind=c.fusion_kind,
            fusion_type=FUSION_TYPE_MATRIX if has_matrix_ops else FUSION_TYPE_MEMORY,
            rows_trace1=self.build_rows(c.ops_trace1, self.perf_rows, self.display_ms),
            rows_trace2=self.build_rows(
                c.ops_trace2,
                self.perf_rows2,
                self.baseline2_ms,
                fill_efficiency=self.same_platform,
            ),
            op_keys=[g["perf_key"] for g in c.ops_trace1 if g["perf_key"]],
        )

    def price(self, c: FusionCandidate) -> FusionEstimate | None:
        """Impact of a standalone candidate.

        Savings are projected via roofline model: the elementwise epilogue time saved
        if kernels were fused, or the roofline-capped savings of elementwise-only
        groups. Kernels without perf models use their measured trace time as-is and
        a warning is emitted. None if the candidate should be skipped.
        """
        enriched = []
        matrix = self.forms.matrix_mask(
            [k.type for k in c.kernels], [k.perf_key for k in c.kernels]
        )
        for k, is_matrix in zip(c.kernels, matrix):
            perf = self.perf_rows.get(tuple(k.perf_key)) if k.perf_key else None
            dm = perf_float(perf, "Data Moved (MB)")
            enriched.append(
                {
                    "dur_us": k.dur_us,
                    "is_matrix": is_matrix,
                    "data_moved_mb": dm,
                    "data_in_mb": k.data_in_mb,
                    "data_out_mb": k.data_out_mb,
                    "gflops": perf_float(perf, "GFLOPS"),
                    "has_perf_data": dm is not None,
                }
            )

        modeled = [e for e in enriched if e["has_perf_data"]]
        unmodeled_count = len(enriched) - len(modeled)
        if not modeled:
            return None

        modeled_frac = len(modeled) / len(enriched)
        min_frac = (
            MIN_MODELED_FRAC
            if len(enriched) < LONG_GROUP_KERNELS
            else MIN_MODELED_FRAC_LONG
        )
        if modeled_frac < min_frac:
            return None

        current_per_instance_us = sum(e["dur_us"] for e in enriched)
        if current_per_instance_us <= 0:
            return None

        subgroups = self._split_into_subgroups(enriched)
        has_matrix_ops = any(e["is_matrix"] for e in enriched)

        total_savings_us = 0.0
        for has_epilogue, sg_kernels in subgroups:
            if has_epilogue:
                non_gemm_time = sum(
                    e["dur_us"] for e in sg_kernels if not e["is_matrix"]
                )
                total_savings_us += non_gemm_time
            elif not any(e["is_matrix"] for e in sg_kernels):
                total_savings_us += self._roofline_savings_us(sg_kernels, TARGET_HIGH)

        bound_type = (
            BOUND_COMPUTE
            if any(has_epilogue for has_epilogue, _ in subgroups)
            else BOUND_MEMORY
        )

        gap_high = total_savings_us / current_per_instance_us
        gap_low = (TARGET_LOW / TARGET_HIGH) * gap_high
        gap_mid = (TARGET_MID / TARGET_HIGH) * gap_high

        time_ms = c.total_kernel_time_us / 1000
        savings_high_ms = gap_high * time_ms

        if savings_high_ms < MIN_SAVINGS_MS:
            return None

        impact_score_high = gap_high * time_ms / self.baseline_ms * 100
        impact_score_low = gap_low * time_ms / self.baseline_ms * 100
        impact_score_mid = gap_mid * time_ms / self.baseline_ms * 100

        if impact_score_high < MIN_IMPACT_SCORE:
            return None

        return FusionEstimate(
            operation=c.base_name,
            impact_score=round(impact_score_mid, 2),
            impact_score_low=round(impact_score_low, 2),
            impact_score_high=round(impact_score_high, 2),
            estimation="full" if len(modeled) == len(enriched) else "partial",
            bound_type=bound_type,
            time_ms=round(time_ms, 3),
            instance_count=c.instance_count,
            kernel_count=len(c.kernels),
            modeled_kernel_count=len(modeled),
            confidence=c.confidence,
            fusion_kind=c.fusion_kind,
            fusion_type=FUSION_TYPE_MATRIX if has_matrix_ops else FUSION_TYPE_MEMORY,
            warning=(
                f"{unmodeled_count} of {len(c.kernels)} kernels lack perf models; "
                f"savings estimate uses trace time for unmodeled kernels"
                if unmodeled_count > 0
                else None
            ),
            rows=self.build_rows(c.ops, self.perf_rows, self.display_ms),
            op_keys=[g["perf_key"] for g in c.ops if g["perf_key"]],
        )

    @staticmethod
    def _split_into_subgroups(enriched: list) -> list:
        """Split kernel sequence at GEMM boundaries into fusable sub-groups.

        Returns list of (has_epilogue, kernels): ``has_epilogue`` is True for a GEMM
        followed by elementwise ops; the other sub-groups are consecutive non-GEMM ops
        or a standalone GEMM.
        """
        subgroups = []
        current = []
        for k in enriched:
            if k["is_matrix"] and current:
                subgroups.append(current)
                current = [k]
            else:
                current.append(k)
        if current:
            subgroups.append(current)

        return [
            (
                any(k["is_matrix"] for k in sg) and any(not k["is_matrix"] for k in sg),
                sg,
            )
            for sg in subgroups
        ]

    def _roofline_savings_us(self, enriched: list, target_pct: float) -> float:
        """Roofline-projected savings for an elementwise sub-group.

        Blends between max (perfect overlap) and sum (no overlap) using OVERLAP_EFFICIENCY.
        """
        modeled = [e for e in enriched if e["has_perf_data"]]
        if not modeled:
            return 0.0

        first = modeled[0]
        last = modeled[-1]
        # "Data Moved (MB)" in the perf report is MiB, as in the perf model.
        total_ext_data_bytes = (
            (first["data_in_mb"] or 0.0) + (last["data_out_mb"] or 0.0)
        ) * 2**20

        vector_gflops = sum(
            e["gflops"] for e in modeled if e["gflops"] and not e["is_matrix"]
        )
        matrix_gflops = sum(
            e["gflops"] for e in modeled if e["gflops"] and e["is_matrix"]
        )

        if total_ext_data_bytes <= 0:
            return 0.0

        frac = target_pct / 100.0
        memory_time_us = total_ext_data_bytes / (self.peak_bw_bytes_s * frac) * 1e6
        matrix_time_us = (
            (matrix_gflops * 1e9) / (self.matrix_maf * 1e12 * frac) * 1e6
            if matrix_gflops > 0
            else 0.0
        )
        vector_time_us = (
            (vector_gflops * 1e9) / (self.vector_maf * 1e12 * frac) * 1e6
            if vector_gflops > 0
            else 0.0
        )

        fused_optimal = max(memory_time_us, matrix_time_us, vector_time_us)
        fused_sum = memory_time_us + matrix_time_us + vector_time_us
        fused_time_us = fused_optimal + (1.0 - OVERLAP_EFFICIENCY) * (
            fused_sum - fused_optimal
        )

        unmodeled_time_us = sum(e["dur_us"] for e in enriched if not e["has_perf_data"])
        current_us = sum(e["dur_us"] for e in enriched)
        return max(0.0, current_us - fused_time_us - unmodeled_time_us)

    def _efficiency_cells(self, eff: dict) -> tuple:
        """(Efficiency, Bound) table cells from a ``calculate_efficiency`` result."""
        pct, peak, unit, bound = _efficiency_parts(eff)
        bound_cell = bound or NO_VALUE
        if pct is None:
            return NO_VALUE, bound_cell
        return f"{pct:.2f}% of {peak:g} {unit}", bound_cell

    def build_rows(
        self,
        ops: list[dict],
        perf_rows: dict,
        baseline_ms: float,
        fill_efficiency: bool = True,
    ) -> list[dict]:
        """Detail-table rows, one per launching op + args, as finished cells.

        Time, %E2E, Count and Kernel Name come from the candidate's own ops
        (summed over its instances). Args, Kernel Path, FLOPS/Byte, Efficiency and
        Bound describe the op signature and come from its ``perf_rows`` entry.
        ``fill_efficiency=False`` leaves Efficiency and Bound empty (trace 2 on a
        platform whose peaks are unknown).
        """
        rows = []
        for g in ops:
            key = g["perf_key"]
            operation, dims, dtypes = key[:3] if key else (NO_VALUE, None, None)
            perf = perf_rows.get(tuple(key)) if key else None
            eff = (
                calculate_efficiency(perf, self.peak_bw_tbs, self.peak_maf_tflops)
                if perf is not None
                else None
            )
            efficiency, bound = (
                self._efficiency_cells(eff)
                if eff and fill_efficiency
                else (NO_VALUE,) * 2
            )
            entry_point = perf.get("entry_point") if perf is not None else None
            time_ms = g["time_us"] / 1000
            rows.append(
                {
                    "operation": operation,
                    "args": (key and format_args(dims, dtypes)) or NO_VALUE,
                    "kernel_path": (
                        entry_point if isinstance(entry_point, str) else "Not found"
                    ),
                    "kernel_name": format_kernel_names(g["kernel_names"])[1],
                    "time_ms": f"{time_ms:.3f}",
                    "pct_e2e": (
                        f"{time_ms / baseline_ms * 100:.2f}"
                        if baseline_ms > 0
                        else NO_VALUE
                    ),
                    "count": g["count"],
                    "flops_per_byte": (
                        f"{eff['flops_per_byte']:.2f}"
                        if eff and eff["flops_per_byte"]
                        else NO_VALUE
                    ),
                    "efficiency": efficiency,
                    "bound": bound,
                }
            )
        return rows
