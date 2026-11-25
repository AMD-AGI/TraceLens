###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Inference Kernel Timeline Breakdown -- perf report extension.

Identifies the highest batch-size region of an inference trace (the
steady-state decode plateau) and selects **both** a representative decode
step and its immediately preceding prefill step from that region.  Prunes
the trace tree to the combined subtrees, tagging every event with a
``step_phase`` column (``prefill`` or ``decode``).

Adds a kernel timeline sheet enriched with phase classification, MTP
section tagging, and structural decoder-layer decomposition with template
clustering.

Features:
    - **Phase classification**: auto-detects MTP (EAGLE), non-MTP SGLang, or
      vLLM traces and labels each ``run_batch`` as prefill / decode /
      decode_idle / prefilldecode.  Decode-phase batches are preferred during
      step selection.
    - **Region-based step selection**: groups consecutive same-phase
      ``run_batch`` events into alternating runs, picks the decode run with
      the highest modal batch size, selects a representative decode step from
      it, and a representative prefill step from the immediately preceding
      prefill run (if any).
    - **MTP section tagging**: identifies EAGLE draft / verify / draft_extend
      sections and tags every kernel with its containing section.
    - **Layer decomposition**: detects ``nn.Module: *DecoderLayer*`` events,
      collects each layer's GPU kernel subtree, tokenizes the kernel sequence
      into canonical labels (via the ``TOKEN_RULES`` regex table), classifies
      layers as MoE or dense, and clusters identical sequences into templates.

Multi-architecture tabs (aligned with KernelBreakdown export
``find_unique_arch_layers``):
    Per ``step_phase``, exclude anomalous decoder layers (>10× median kernel
    count), group layers by distinctive ``cpu_op`` fingerprint, assign tab
    stems from discovered ``nn.Module`` child classes (or kernel/cpu_op
    fingerprints), and pick ``min(layer_idx)`` per fingerprint group.

Output sheets:
    - Per-architecture tabs (e.g. ``<arch_stem>_layer1_prefill``) when layer
      types are detected; otherwise ``kernel_timeline`` or phase sheets.
    - ``kernel_timeline``-style rows: every GPU kernel with columns for step_phase,
      timestamp, duration, parent cpu_op, nn.Module, decode_section,
      decoder_layer, layer_kind, template_id, and layer_token.
    - ``mtp_section_summary``: per-section kernel count and duration (MTP only).
    - ``layer_summary``: per-layer stats (kind, template, kernel count,
      duration, percentage of step, compressed token sequence).

Steady-state step selection strategy:
    1. Primary:  vLLM/SGLang ``user_annotation`` events that encode request
       counts in their name (e.g. ``execute_context_N(...)_generation_N(...)``).
       The annotation with the highest ``num_requests`` is mapped to its
       containing ``run_batch`` event.
    2. Fallback: when no such annotations exist, build alternating runs from
       phase-classified ``run_batch`` events, find the decode run with the
       highest modal batch size (the "region"), then pick one decode step
       and one prefill step from that region.  Each uses the same general
       algorithm: modal first dim of ``cpu_op`` input shapes, with
       nearest-average-duration tie-breaking.
    3. Tie-breaker: when multiple steps share the same max score, pick the
       one whose total GPU kernel duration is closest to the average across
       tied candidates.

Tokenization:
    Kernel names are mapped to stable canonical tokens via the ``TOKEN_RULES``
    list -- an ordered sequence of ``(compiled_regex, token_label)`` pairs.
    First match wins.  To add a new token, append a rule to the list.  Order
    matters: more specific patterns must precede general ones (e.g.
    ``fused_qk_rope_cat_and_cache_mla`` before generic ``rope``).

Extension scope arguments (via environment variables, since the report CLI
uses strict argparse and rejects unknown flags):
    ``TRACELENS_EXT_MOE_ONLY=1``        Keep only MoE-classified layers.
    ``TRACELENS_EXT_FIRST_N_LAYERS=N``  Keep only the first N decoder layers
                                        (after MoE filtering).
    ``TRACELENS_EXT_MTP_SCOPE=SECTION`` Keep only kernels in the given MTP
                                        decode_section (e.g. ``verify``).

Usage (recommended -- enables tree-based layer detection):

    TraceLens_generate_perf_report_pytorch_inference \\
        --profile_json_path trace.json.gz \\
        --extension_file examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py \\
        --output_xlsx_path report.xlsx \\
        --group_by_parent_module

With scope filters (via environment variables):

    TRACELENS_EXT_MOE_ONLY=1 TRACELENS_EXT_FIRST_N_LAYERS=1 \\
    TraceLens_generate_perf_report_pytorch_inference \\
        --profile_json_path trace.json.gz \\
        --extension_file examples/custom_workflows/inference_kernel_timeline_breakdown/inference_kernel_timeline_breakdown.py \\
        --output_xlsx_path report.xlsx \\
        --group_by_parent_module

SGLang EAGLE/MTP traces use verify-section + rmsnorm step
selection automatically when EAGLE markers are detected; ``--mtp_scope
verify`` is optional. vLLM traces still use annotation-based prefill/decode
step selection unchanged.

Extension hooks used:
    tree_postprocess_extension(tree)        -- phase classification, region-
                                               based step selection, subtree
                                               filtering, step_phase tagging,
                                               MTP section tagging
    get_additional_dataframes_extension(tree) -- layer decomposition,
                                               kernel_timeline, layer_summary,
                                               mtp_section_summary sheets,
                                               scope filtering
"""

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Extension scope arguments (via environment variables)
# ---------------------------------------------------------------------------
# The report CLI uses strict ``parser.parse_args()``, which rejects unknown
# flags before extensions load.  Extension-specific scope options are
# therefore read from environment variables instead of CLI flags:
#
#   TRACELENS_EXT_MOE_ONLY=1       -- keep only MoE-classified layers
#   TRACELENS_EXT_FIRST_N_LAYERS=N -- keep only the first N decoder layers
#   TRACELENS_EXT_MTP_SCOPE=verify -- keep only kernels in this MTP section
#
@dataclass
class _ExtArgs:
    moe_only: bool = False
    first_n_layers: int = 0
    mtp_scope: str = ""


def _parse_ext_args() -> _ExtArgs:
    return _ExtArgs(
        moe_only=os.environ.get("TRACELENS_EXT_MOE_ONLY", "") in ("1", "true", "True"),
        first_n_layers=int(os.environ.get("TRACELENS_EXT_FIRST_N_LAYERS", "0")),
        mtp_scope=os.environ.get("TRACELENS_EXT_MTP_SCOPE", ""),
    )


_EXT_ARGS = _parse_ext_args()


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

# ---------------------------------------------------------------------------
# Annotation patterns for vLLM/SGLang iteration detection
# ---------------------------------------------------------------------------
ANNOTATION_PATTERNS = [
    re.compile(
        r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
    ),
    re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"execute_new_\d+_cached_\d+"),
    re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
]

# ---------------------------------------------------------------------------
# EAGLE / MTP section detection patterns
# ---------------------------------------------------------------------------
EAGLE_WORKER_FILE_RE = r"eagle_worker(?:_v2)?\.py"
EAGLE_SECTION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "draft",
        re.compile(rf"{EAGLE_WORKER_FILE_RE}\(\d+\):\s*(?:draft)$", re.IGNORECASE),
    ),
    (
        "verify",
        re.compile(rf"{EAGLE_WORKER_FILE_RE}\(\d+\):\s*(?:verify)$", re.IGNORECASE),
    ),
    (
        "draft_extend",
        re.compile(
            rf"{EAGLE_WORKER_FILE_RE}\(\d+\):\s*(?:forward_draft_extend_after_decode|_draft_extend_for_decode)$",
            re.IGNORECASE,
        ),
    ),
]
EAGLE_DRAFT_RE = EAGLE_SECTION_PATTERNS[0][1]

# Pattern for detecting run_batch boundaries
RUN_BATCH_RE = re.compile(r"scheduler\.py.*run_batch", re.IGNORECASE)

# SGLang MTP export alignment (KernelBreakdown SGLang exporter)
DEFAULT_RMSNORM_SUBSTRING = "rmsnorm"

# vLLM gpu_model_runner: forward annotation (narrower step boundary)
VLLM_FORWARD_RE = re.compile(r"^gpu_model_runner:\s*forward$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Phase classification patterns
# ---------------------------------------------------------------------------
FORWARD_IDLE_RE = re.compile(r"model_runner\.py\(\d+\):\s*forward_idle", re.IGNORECASE)
FORWARD_EXTEND_RE = re.compile(
    r"(?:gpu_model_runner\.py|model_runner\.py)\(\d+\):\s*forward_extend",
    re.IGNORECASE,
)
FORWARD_DECODE_RE = re.compile(
    r"(?:gpu_model_runner\.py|model_runner\.py)\(\d+\):\s*forward_decode",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Layer decomposition constants
# ---------------------------------------------------------------------------
DECODER_LAYER_RE = re.compile(
    r"nn\.Module:\s*\w*(?:DecoderLayer|TransformerBlock|Block)_(\d+)"
)
DECODER_CLASS_RE = re.compile(
    r"nn\.Module:\s*(\w*(?:DecoderLayer|TransformerBlock|Block))_\d+"
)

MOE_TOKENS = frozenset(
    {
        "TopK",
        "MoESorting",
        "ck_moe_stage1",
        "ck_moe_stage2",
        "ck_moe_stage",
        "moe_mxfp4_sort",
        "fused_moe",
        "MoEGate(Gemm)",
        "MoEAppendSharedExperts",
        "router",
        "MoERouting",
        "MoEFinalize",
    }
)

# ---------------------------------------------------------------------------
# Multi-architecture fingerprinting constants
# ---------------------------------------------------------------------------
_FINGERPRINT_EXCLUDE_PREFIXES = (
    "aten::",
    "aiter::moe",
    "aiter::topk_softmax",
    "_moe_C",
    "_C_custom_ar::",
)

_INFRA_MODULE_SKIP = frozenset(
    {
        "norm",
        "linear",
        "embedding",
        "siluandmul",
    }
)

# Inner MoE FFN blocks appear in every decoder layer — not architecture types.
_MOE_BLOCK_MODULE_SKIP = frozenset(
    {
        "moeblock",
        "sparsemoe",
        "fusedmoe",
        "sharedfusedmoe",
    }
)


def _is_architecture_module_class(class_name: str) -> bool:
    n = class_name.lower()
    if any(skip in n for skip in _INFRA_MODULE_SKIP):
        return False
    if any(skip in n for skip in _MOE_BLOCK_MODULE_SKIP):
        return False
    return True


_ANOMALY_MULTIPLIER = 10
_ANOMALY_MIN_THRESHOLD = 200

_NN_MODULE_PREFIX = "nn.Module: "


def _is_sglang_mtp(tree) -> bool:
    return bool(getattr(tree, "_inference_ext_sglang_mtp", False))


# ---------------------------------------------------------------------------
# Layer decomposition data classes
# ---------------------------------------------------------------------------
@dataclass
class LayerInfo:
    layer_idx: int
    layer_name: str
    tokens: List[str]
    tokens_compact: List[str]
    ff_kind: str  # "moe" | "dense" | "unknown"
    step_phase: str = ""  # "prefill" | "decode" | ""
    template_id: int = -1
    total_kernel_dur_us: float = 0.0
    kernel_count: int = 0
    kernel_uids: List[int] = field(default_factory=list)
    cpu_op_names: List[str] = field(default_factory=list)
    arch_type: str = ""
    arch_fingerprint: frozenset = field(default_factory=frozenset)


@dataclass
class LayerDecomposition:
    layers: List[LayerInfo]
    templates: Dict[int, List[str]]  # template_id -> compressed token sequence
    kernel_to_layer: Dict[int, int]  # kernel UID -> layer_idx
    layer_to_arch: Dict[Tuple[str, int], str] = field(
        default_factory=dict
    )  # (phase, layer_idx) -> arch
    # (step_phase, arch_name) -> representative decoder_layer index for export tabs
    arch_reps: Dict[Tuple[str, str], int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------
def _classify_run_batch_phases(events: list) -> Dict[int, str]:
    """Classify each run_batch event UID as prefill/decode/decode_idle/prefilldecode.

    Auto-detects which strategy to use based on available markers.
    """
    run_batches = [
        e
        for e in events
        if e.get("cat") == "python_function" and RUN_BATCH_RE.search(e.get("name", ""))
    ]
    if not run_batches:
        return {}

    run_batches.sort(key=lambda e: e.get("ts", 0))

    rb_intervals = [
        (rb, rb.get("ts", 0), rb.get("ts", 0) + rb.get("dur", 0)) for rb in run_batches
    ]

    rb_uid_set = {rb.get("UID") for rb in run_batches}
    pyfunc_events = [
        e
        for e in events
        if e.get("cat") == "python_function" and e.get("UID") not in rb_uid_set
    ]
    pyfunc_events.sort(key=lambda e: e.get("ts", 0))

    has_mtp_markers = False
    has_sglang_markers = False
    has_annotations = False

    rb_uid_to_children_names: Dict[int, List[str]] = {
        rb.get("UID"): [] for rb in run_batches
    }

    for pf in pyfunc_events:
        pf_ts = pf.get("ts", 0)
        pf_name = pf.get("name", "")
        for rb, rb_start, rb_end in rb_intervals:
            if rb_start <= pf_ts <= rb_end:
                rb_uid_to_children_names[rb.get("UID")].append(pf_name)
                break

    for names in rb_uid_to_children_names.values():
        for n in names:
            if FORWARD_IDLE_RE.search(n) or EAGLE_DRAFT_RE.search(n):
                has_mtp_markers = True
                break
        if has_mtp_markers:
            break

    if not has_mtp_markers:
        for names in rb_uid_to_children_names.values():
            for n in names:
                if FORWARD_EXTEND_RE.search(n) or FORWARD_DECODE_RE.search(n):
                    has_sglang_markers = True
                    break
            if has_sglang_markers:
                break

    if not has_mtp_markers and not has_sglang_markers:
        for e in events:
            if e.get("cat") == "user_annotation":
                name = e.get("name", "")
                if any(p.match(name) for p in ANNOTATION_PATTERNS):
                    has_annotations = True
                    break

    if has_mtp_markers:
        result: Dict[int, str] = {}
        for rb in run_batches:
            uid = rb.get("UID")
            names = rb_uid_to_children_names.get(uid, [])
            if any(FORWARD_IDLE_RE.search(n) for n in names):
                result[uid] = "decode_idle"
            elif any(EAGLE_DRAFT_RE.search(n) for n in names):
                result[uid] = "decode"
            else:
                result[uid] = "prefill"
        return result

    if has_sglang_markers:
        result = {}
        for rb in run_batches:
            uid = rb.get("UID")
            names = rb_uid_to_children_names.get(uid, [])
            has_extend = any(FORWARD_EXTEND_RE.search(n) for n in names)
            has_decode = any(FORWARD_DECODE_RE.search(n) for n in names)
            if has_extend and has_decode:
                result[uid] = "prefilldecode"
            elif has_extend:
                result[uid] = "prefill"
            else:
                result[uid] = "decode"
        return result

    if has_annotations:
        annotations = [
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and any(p.match(e.get("name", "")) for p in ANNOTATION_PATTERNS)
        ]
        annotations.sort(key=lambda e: e.get("ts", 0))
        result = {}
        for rb, rb_start, rb_end in rb_intervals:
            uid = rb.get("UID")
            matched_kind = "decode"
            for ann in annotations:
                ann_ts = ann.get("ts", 0)
                if rb_start <= ann_ts <= rb_end:
                    details = _get_iter_details_from_name(ann.get("name", ""))
                    cr = int(details.get("context_requests") or 0)
                    gr = int(details.get("generation_requests") or 0)
                    if cr > 0 and gr > 0:
                        matched_kind = "prefilldecode"
                    elif cr > 0:
                        matched_kind = "prefill"
                    else:
                        matched_kind = "decode"
                    break
            result[uid] = matched_kind
        return result

    return {rb.get("UID"): "decode" for rb in run_batches}


def _tag_run_batch_phases(tree) -> Dict[int, str]:
    """Classify run_batch phases, tag events in-place, print summary."""
    phase_map = _classify_run_batch_phases(tree.events)
    for event in tree.events:
        uid = event.get("UID")
        if uid in phase_map:
            event["phase"] = phase_map[uid]
    if phase_map:
        counts = Counter(phase_map.values())
        parts = [f"{count} {phase}" for phase, count in sorted(counts.items())]
        print(f"[inference_ext] Phase classification: {', '.join(parts)}")
    return phase_map


# ---------------------------------------------------------------------------
# Tokenization -- data-driven regex lookup table
# ---------------------------------------------------------------------------
TOKEN_RULES: List[Tuple["re.Pattern[str]", str]] = [
    # Norms
    (re.compile(r"rmsnorm|layernorm"), "RMSNorm"),
    # MLA decode path (before generic rope -- fused_qk_rope contains "rope")
    (re.compile(r"get_mla_metadata"), "get_mla_metadata"),
    (re.compile(r"fused_qk_rope_cat_and_cache_mla"), "fused_qk_rope_cat_and_cache_mla"),
    (re.compile(r"mla_decode_stage"), "mla_decode"),
    (re.compile(r"mla_reduce"), "mla_reduce"),
    (re.compile(r"set_mla_kv_buffer"), "set_mla_kv_buffer"),
    # Generic rope (after MLA-specific rules)
    (re.compile(r"rope"), "RoPE"),
    # GatedDeltaNet / linear attention
    (re.compile(r"causal_conv1d"), "causal_conv1d"),
    (re.compile(r"fused_gdn_gating|gdn_attention_core"), "GDN_attention"),
    (
        re.compile(r"fused_recurrent.*gated_delta|FusedRecurrentFunction"),
        "FusedRecurrent_GDR",
    ),
    (re.compile(r"chunk.*gated_delta|ChunkGatedDeltaRule"), "ChunkGatedDeltaRule"),
    (re.compile(r"chunk_local_cumsum"), "chunk_local_cumsum"),
    (re.compile(r"l2norm_fwd"), "l2norm"),
    # Attention
    (re.compile(r"(?=.*concat)(?=.*mha)"), "concat_cast_mha"),
    (re.compile(r"fmha|flash|mha_varlen_fwd"), "fmha"),
    (re.compile(r"paged_attention"), "paged_attention"),
    (re.compile(r"reshape_and_cache"), "reshape_and_cache"),
    (re.compile(r"(?=.*attention)(?=.*self)"), "fmha"),
    # Communication
    (re.compile(r"all_reduce"), "all_reduce"),
    # Quantization / activations
    (
        re.compile(
            r"dynamic_per_group_scaled_quant|per_token_group_quant|dynamic_mxfp4_quant"
        ),
        "quant",
    ),
    (re.compile(r"silu_and_mul|swiglu|act_and_mul"), "silu_and_mul"),
    # MoE path (order matters -- more specific first)
    (re.compile(r"biased_grouped_topk|grouped_topk|gathertopk"), "TopK"),
    (re.compile(r"moe_?sorting"), "MoESorting"),
    (re.compile(r"fused_append_shared_experts"), "MoEAppendSharedExperts"),
    (re.compile(r"ck_moe_stage1"), "ck_moe_stage1"),
    (re.compile(r"ck_moe_stage2"), "ck_moe_stage2"),
    (re.compile(r"ck_moe_stage"), "ck_moe_stage"),
    (re.compile(r"kernel_moe_gemm|moe_gemm"), "MoEGate(Gemm)"),
    (re.compile(r"moe_mxfp4_sort"), "moe_mxfp4_sort"),
    (re.compile(r"fused_moe"), "fused_moe"),
    (re.compile(r"gemm_a16w16_atomic|gemm_a16_w16_atomic_kernel"), "router"),
    (re.compile(r"moe::dev::routing|routingdeepseek"), "MoERouting"),
    (re.compile(r"moe::dev::finalize"), "MoEFinalize"),
    # Elementwise bucket
    (
        re.compile(r"^aten::(?:ge|lt|bitwise_and|copy_|masked_fill_|fill_)"),
        "elementwise",
    ),
    # Gemm variants (order: specific before general)
    (re.compile(r"^aten::mm$"), "Gemm"),
    (re.compile(r"gemm_afp4wfp4"), "Gemm(mxfp4)"),
    (re.compile(r"batched_gemm_a16wfp4"), "BatchedGemm(a16wfp4)"),
    (re.compile(r"aten::addmm|aten::linear"), "Gemm"),
]


def _canonical_token(cpu_op_name: str, kernel_name: str) -> str:
    """Map a kernel's cpu_op/kernel name to a stable canonical token.

    Walks TOKEN_RULES in order; first regex match wins.
    """
    name = cpu_op_name if cpu_op_name else kernel_name
    if not name:
        return "UNKNOWN"
    nlow = name.lower()
    for pattern, token in TOKEN_RULES:
        if pattern.search(nlow):
            return token
    return name[:60] if len(name) > 60 else name


DENSE_TOKENS = frozenset(
    {
        "silu_and_mul",
        "GDN_attention",
        "FusedRecurrent_GDR",
        "ChunkGatedDeltaRule",
        "paged_attention",
    }
)


def _classify_layer_kind(tokens: Sequence[str]) -> str:
    """Classify a layer as 'moe', 'dense', or 'unknown' from its token set."""
    token_set = set(tokens)
    if token_set & MOE_TOKENS:
        return "moe"
    if token_set & DENSE_TOKENS:
        return "dense"
    return "unknown"


def _compress_runs(tokens: Sequence[str]) -> List[str]:
    """Collapse consecutive duplicate tokens into Token*N form."""
    if not tokens:
        return []
    out: List[str] = []
    cur = tokens[0]
    cnt = 1
    for t in tokens[1:]:
        if t == cur:
            cnt += 1
            continue
        out.append(cur if cnt == 1 else f"{cur}*{cnt}")
        cur = t
        cnt = 1
    out.append(cur if cnt == 1 else f"{cur}*{cnt}")
    return out


# ---------------------------------------------------------------------------
# Layer decomposition
# ---------------------------------------------------------------------------
def _collect_layer_subtree_uids(tree, layer_event: dict) -> set:
    """Collect all event UIDs in the subtree rooted at a decoder layer event."""
    uids: set = set()
    if layer_event.get("children"):
        stack = [layer_event]
        while stack:
            ev = stack.pop()
            uid = ev.get("UID")
            if uid is None or uid in uids:
                continue
            uids.add(uid)
            for child_uid in ev.get("children", []):
                try:
                    stack.append(tree.get_UID2event(child_uid))
                except (KeyError, TypeError):
                    pass
            for gpu_uid in ev.get("gpu_events", []):
                uids.add(gpu_uid)
    else:
        ts_start = layer_event.get("ts", 0)
        ts_end = ts_start + layer_event.get("dur", 0)
        root_uid = layer_event.get("UID")
        if root_uid is not None:
            uids.add(root_uid)
        for ev in tree.events:
            cat = ev.get("cat", "")
            if cat in GPU_CATEGORIES:
                continue
            ev_ts = ev.get("ts", 0)
            if ev_ts < ts_start or ev_ts > ts_end:
                continue
            uid = ev.get("UID")
            if uid is not None:
                uids.add(uid)
            for gpu_uid in ev.get("gpu_events", []):
                uids.add(gpu_uid)
    return uids


def _op_is_distinctive(cpu_op: str) -> bool:
    """Match KernelBreakdown export: ops that distinguish architecture types."""
    if not cpu_op:
        return False
    for prefix in _FINGERPRINT_EXCLUDE_PREFIXES:
        if cpu_op.startswith(prefix):
            return False
    if "topk_softmax" in cpu_op:
        return False
    return True


def _compute_layer_fingerprint(cpu_op_names: Sequence[str]) -> frozenset:
    """Distinctive cpu_op set for one decoder layer (same rules as vLLM export)."""
    return frozenset(
        op.strip() for op in cpu_op_names if _op_is_distinctive((op or "").strip())
    )


def _exclude_anomalous_layers_export_style(
    phase_layers: List["LayerInfo"],
    decoder_class: str = "",
) -> set:
    """
    Port of export ``find_unique_arch_layers`` anomaly filter (per step_phase).

    Drops layers whose kernel count exceeds max(median * 10, 200) when that
    layer absorbs the whole forward (e.g. ``<DecoderClass>_0`` with 290k
    kernel_seq rows at batch_idx=0).  Returns set of (step_phase, layer_idx)
    keys to remove.
    """
    if len(phase_layers) <= 2:
        return set()
    counts = sorted(li.kernel_count for li in phase_layers)
    median_count = counts[len(counts) // 2]
    threshold = max(_ANOMALY_MIN_THRESHOLD, median_count * _ANOMALY_MULTIPLIER)
    phase = phase_layers[0].step_phase if phase_layers else ""
    anomalous = [li for li in phase_layers if li.kernel_count > threshold]
    if not anomalous or len(anomalous) >= len(phase_layers):
        return set()
    for li in anomalous:
        label = (
            f"{decoder_class}_{li.layer_idx}" if decoder_class else str(li.layer_idx)
        )
        print(
            f"[inference_ext] Excluding anomalous layer {label!r} "
            f"({li.kernel_count} kernels vs median {median_count}, phase={phase!r})"
        )
    return {(phase, li.layer_idx) for li in anomalous}


def _arch_representatives_from_fingerprints(
    phase: str,
    phase_layers: List["LayerInfo"],
    tree,
    decoder_class: str,
    layer_ev_by_idx: Dict[Tuple[str, int], dict],
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, int], str]]:
    """
    Match export ``find_unique_arch_layers``: group by fingerprint, assign arch
    tab stem, pick ``min(layer_idx)`` per fingerprint group as representative.
    """
    fp_to_layers: Dict[frozenset, List[LayerInfo]] = {}
    for li in phase_layers:
        if not li.arch_fingerprint and li.tokens_compact:
            li.arch_fingerprint = frozenset(("__sig__",) + tuple(li.tokens_compact[:8]))
        fp_to_layers.setdefault(li.arch_fingerprint, []).append(li)

    if len(phase_layers) > 1 and len(fp_to_layers) < 2:
        for li in phase_layers:
            tok = _arch_from_token_signature(li.tokens_compact, phase)
            if tok:
                li.arch_fingerprint = frozenset((("__tok__", tok),))
        fp_to_layers = {}
        for li in phase_layers:
            fp_to_layers.setdefault(li.arch_fingerprint, []).append(li)

    phase_fp_groups = [
        (fp, min(x.layer_idx for x in grp)) for fp, grp in fp_to_layers.items()
    ]
    fp_name_map = _discover_inner_module_names_from_tree(
        tree,
        decoder_class,
        phase_fp_groups,
    )
    phase_fps = list(fp_to_layers.keys())

    for fp, group in fp_to_layers.items():
        rep_li = min(group, key=lambda x: x.layer_idx)
        arch_name = _resolve_arch_type_name(
            fp_name_map,
            fp,
            phase_fps,
            phase,
        )
        if not arch_name:
            arch_name = _infer_arch_name_from_fingerprint(fp, phase_fps, phase)
        if not arch_name:
            arch_name = _arch_from_token_signature(rep_li.tokens_compact, phase)
        if not arch_name:
            layer_ev = layer_ev_by_idx.get((rep_li.step_phase, rep_li.layer_idx))
            if layer_ev is not None:
                arch_name = _arch_from_layer_module(tree, layer_ev, phase)
        arch_name = _normalize_arch_for_phase(arch_name, phase)
        if not arch_name and decoder_class:
            arch_name = decoder_class
        for li in group:
            li.arch_type = arch_name

    arch_reps: Dict[Tuple[str, str], int] = {}
    layer_to_arch: Dict[Tuple[str, int], str] = {}
    for fp, group in sorted(
        fp_to_layers.items(),
        key=lambda item: min(x.layer_idx for x in item[1]),
    ):
        arch_name = next((li.arch_type for li in group if li.arch_type), "")
        if not arch_name or _is_infra_arch_stem(arch_name):
            continue
        rep_idx = min(li.layer_idx for li in group)
        arch_reps[(phase, arch_name)] = rep_idx
        for li in group:
            layer_to_arch[(phase, int(li.layer_idx))] = arch_name

    return arch_reps, layer_to_arch


def _cpu_op_belongs_to_decoder_layer(tree, cpu_op_ev: dict, layer_ev: dict) -> bool:
    """True if *cpu_op_ev* is attributed to *layer_ev* (not a sibling layer)."""
    layer_uid = layer_ev.get("UID")
    current = cpu_op_ev
    for _ in range(300):
        parent = tree.get_parent_event(current)
        if parent is None:
            return False
        if parent.get("UID") == layer_uid:
            return True
        pname = parent.get("name", "")
        if parent.get("cat") == "python_function" and DECODER_LAYER_RE.search(pname):
            return False
        current = parent
    return False


def _collect_layer_cpu_op_names(
    tree,
    layer_ev: dict,
    subtree_uids: set,
    layer_kernels: Optional[List[dict]] = None,
) -> List[str]:
    """cpu_op names attributed to this decoder layer (matches kernel_seq layer rows)."""
    names: List[str] = []
    seen: set = set()
    layer_mod = layer_ev.get("name", "").replace(_NN_MODULE_PREFIX, "").strip()
    layer_class = re.sub(r"_\d+$", "", layer_mod)

    def _add(name: str) -> None:
        n = (name or "").strip()
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    for uid in subtree_uids:
        try:
            ev = tree.get_UID2event(uid)
        except (KeyError, TypeError):
            continue
        if ev.get("cat") == "cpu_op":
            stack = ev.get("nn_module_stack") or []
            if layer_class and any(layer_class in str(s) for s in stack):
                _add(ev.get("name", ""))
            elif _cpu_op_belongs_to_decoder_layer(tree, ev, layer_ev):
                _add(ev.get("name", ""))

    for kev in layer_kernels or []:
        parent_uid = kev.get("parent")
        if parent_uid is None:
            continue
        try:
            parent_ev = tree.get_UID2event(parent_uid)
        except (KeyError, TypeError):
            continue
        if parent_ev.get("cat") == "cpu_op":
            _add(parent_ev.get("name", ""))

    return names


def _decoder_class_from_events(layer_events: List[Tuple[int, dict]]) -> str:
    """Pick the outer decoder class, not inner MoE blocks (e.g. *SparseMoeBlock*)."""
    counts: Dict[str, int] = {}
    for _, ev in layer_events:
        m = DECODER_CLASS_RE.search(ev.get("name", ""))
        if not m:
            continue
        cls = m.group(1)
        if cls.endswith("DecoderLayer") or "TransformerBlock" in cls:
            counts[cls] = counts.get(cls, 0) + 1
    if counts:
        return max(counts, key=counts.get)
    for _, ev in layer_events:
        m = DECODER_CLASS_RE.search(ev.get("name", ""))
        if m:
            return m.group(1)
    return ""


def _decoder_class_from_tree(tree, events: Optional[list] = None) -> str:
    """Discover outer ``*DecoderLayer`` class from the pruned trace."""
    evs = events if events is not None else tree.events
    candidates: List[Tuple[int, dict]] = []
    for ev in evs:
        if ev.get("cat") != "python_function":
            continue
        m = DECODER_LAYER_RE.search(ev.get("name", ""))
        if m:
            candidates.append((int(m.group(1)), ev))
    return _decoder_class_from_events(candidates)


def _filter_outer_decoder_layer_events(
    layer_events: List[Tuple[int, dict]],
    decoder_class: str,
) -> List[Tuple[int, dict]]:
    """
    Keep only outer decoder layers (``Qwen3_5DecoderLayer_N``), matching
    KernelBreakdown export ``layer_label.startswith(decoder_class + '_')``.
    Drops inner blocks like ``Qwen3NextSparseMoeBlock_0`` that also match
    ``DECODER_LAYER_RE`` because of the ``Block`` suffix.
    """
    if not decoder_class:
        return layer_events
    outer_re = re.compile(rf"^nn\.Module:\s*{re.escape(decoder_class)}_(\d+)$")
    by_key: Dict[Tuple[int, str], dict] = {}
    for _layer_idx, ev in layer_events:
        name = (ev.get("name") or "").strip()
        m = outer_re.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        phase = (ev.get("step_phase") or "").strip()
        key = (idx, phase)
        prev = by_key.get(key)
        if prev is None or float(ev.get("dur", 0) or 0) >= float(
            prev.get("dur", 0) or 0
        ):
            by_key[key] = ev
    if not by_key:
        return layer_events
    dropped = len(layer_events) - len(by_key)
    if dropped:
        print(
            f"[inference_ext] Outer-layer filter ({decoder_class!r}): "
            f"kept {len(by_key)} layer events, dropped {dropped} inner/duplicate events"
        )
    return sorted(
        [(idx, ev) for (idx, _phase), ev in by_key.items()],
        key=lambda x: (x[1].get("step_phase", ""), x[0], x[1].get("ts", 0)),
    )


def _is_infra_arch_stem(arch_name: str) -> bool:
    """Generic gemm/linear stems are not multi-architecture tab types."""
    if not arch_name:
        return True
    lower = arch_name.lower()
    return bool(
        re.search(
            r"unquantized_gemm|^(?:rocm_)?gemm\b|^linear$|^norm$|router",
            lower,
        )
    )


# Kernel/cpu_op substring -> tab stem (model-agnostic; order = most specific first).
_ARCH_OP_TAB_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"paged_attention_v1", re.I), "paged_attention_v1"),
    (re.compile(r"paged_attention", re.I), "paged_attention_v1"),
    (re.compile(r"reshape_and_cache", re.I), "reshape_and_cache_flash"),
    (
        re.compile(r"gdn_attention|ChunkGatedDelta|FusedRecurrent|GatedDelta", re.I),
        "gated_delta_net",
    ),
]

_EXCEL_SHEET_NAME_MAX = 31

_GDN_TOKEN_MARKERS = frozenset(
    {
        "ChunkGatedDeltaRule",
        "GDN_attention",
        "FusedRecurrent_GDR",
        "causal_conv1d",
    }
)
_ATTN_TOKEN_MARKERS = frozenset(
    {
        "reshape_and_cache",
        "paged_attention",
        "fmha",
    }
)


def _arch_stem_from_op_rules(text: str) -> str:
    """First matching tab stem from ``_ARCH_OP_TAB_RULES`` on *text*."""
    for pattern, stem in _ARCH_OP_TAB_RULES:
        if pattern.search(text):
            return stem
    return ""


def _attention_stem_for_phase(step_phase: str) -> str:
    """Attention-block tab stem for decode vs prefill (from op rules, not model names)."""
    if step_phase == "prefill":
        return (
            _arch_stem_from_op_rules("reshape_and_cache") or "reshape_and_cache_flash"
        )
    return _arch_stem_from_op_rules("paged_attention") or "paged_attention_v1"


def _gated_delta_stem() -> str:
    return _arch_stem_from_op_rules("ChunkGatedDelta") or "gated_delta_net"


def _stem_from_module_class(class_name: str, step_phase: str) -> str:
    """Tab stem from a discovered ``nn.Module`` child class name."""
    base = re.sub(r"_\d+$", "", (class_name or "").strip())
    if not base or not _is_architecture_module_class(base):
        return ""
    lower = base.lower()
    if "gateddelta" in lower or re.search(r"\bgdn\b", lower):
        return base
    if "attention" in lower:
        return _attention_stem_for_phase(step_phase)
    return base


def _normalize_arch_for_phase(arch_name: str, step_phase: str) -> str:
    """Map decode-style attention stems to prefill cache stems when needed."""
    if not arch_name:
        return ""
    if step_phase == "prefill" and re.search(r"paged_attention", arch_name, re.I):
        return _attention_stem_for_phase("prefill")
    return arch_name


def _arch_from_layer_module(
    tree,
    layer_ev: dict,
    step_phase: str,
) -> str:
    """Primary arch label from non-infra nn.Module children (matches export discovery)."""
    for child_uid in layer_ev.get("children", []):
        try:
            child = tree.get_UID2event(child_uid)
        except (KeyError, TypeError):
            continue
        name = child.get("name", "")
        if not name.startswith(_NN_MODULE_PREFIX):
            continue
        class_name = re.sub(
            r"_\d+$",
            "",
            name[len(_NN_MODULE_PREFIX) :].strip(),
        )
        stem = _stem_from_module_class(class_name, step_phase)
        if stem:
            return stem
    return ""


def _resolve_arch_type_name(
    fp_name_map: Dict[frozenset, str],
    fingerprint: frozenset,
    all_fingerprints: Sequence[frozenset],
    step_phase: str,
) -> str:
    """Map a layer fingerprint to a multi-arch tab stem (discovered module or ops)."""
    discovered = (fp_name_map.get(fingerprint, "") or "").strip()
    if discovered:
        stem = _stem_from_module_class(discovered, step_phase) or discovered
        return _normalize_arch_for_phase(stem, step_phase)
    inferred = _infer_arch_name_from_fingerprint(
        fingerprint,
        all_fingerprints,
        step_phase,
    )
    if inferred:
        return _normalize_arch_for_phase(inferred, step_phase)
    return ""


def _arch_from_token_signature(tokens_compact: Sequence[str], step_phase: str) -> str:
    """Classify architecture from compressed kernel tokens (per-layer)."""
    if not tokens_compact:
        return ""
    blob = " ".join(tokens_compact)
    token_set = set(tokens_compact)
    stem = _arch_stem_from_op_rules(blob)
    if stem:
        return _normalize_arch_for_phase(stem, step_phase)
    if step_phase == "prefill":
        if token_set & _GDN_TOKEN_MARKERS:
            return _gated_delta_stem()
        if token_set & _ATTN_TOKEN_MARKERS or re.search(
            r"reshape_and_cache|mha_varlen", blob, re.I
        ):
            return _attention_stem_for_phase("prefill")
    if step_phase == "decode":
        if token_set & {"paged_attention"}:
            return _attention_stem_for_phase("decode")
        if token_set & _GDN_TOKEN_MARKERS:
            return _gated_delta_stem()
    return ""


def _shorten_excel_tab_stem(arch_name: str, suffix_len: int) -> str:
    """Shorten *arch_name* so ``stem + suffix`` fits Excel's sheet name limit."""
    budget = max(8, _EXCEL_SHEET_NAME_MAX - suffix_len)
    if len(arch_name) <= budget:
        return arch_name
    trimmed = re.sub(r"DecoderLayer$", "", arch_name, flags=re.I)
    if trimmed and len(trimmed) <= budget:
        return trimmed
    block_suffix = re.search(
        r"((?:Gated)?DeltaNet|Attention\w*|SparseMoe\w*|MoeBlock)$",
        arch_name,
        re.I,
    )
    if block_suffix and len(block_suffix.group(1)) <= budget:
        return block_suffix.group(1)
    if "_" in arch_name:
        parts_us = arch_name.split("_")
        for n in (3, 2, 1):
            cand = "_".join(parts_us[-n:])
            if cand and len(cand) <= budget:
                return cand
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", arch_name)
    if len(parts) >= 2:
        cand = "_".join(p.lower() for p in parts[-2:])
        if len(cand) <= budget:
            return cand
    if parts and len(parts[-1]) <= budget:
        return parts[-1]
    return arch_name[:budget]


def _excel_arch_tab_name(arch_name: str, layer_idx: int, step_phase: str) -> str:
    """Build ``{stem}_layer{N}_{decode|prefill}`` within Excel's 31-char limit."""
    phase_suffix = "decode" if step_phase == "decode" else "prefill"
    suffix = f"_layer{layer_idx}_{phase_suffix}"
    stem = _shorten_excel_tab_stem(arch_name, len(suffix))
    return (stem + suffix)[:_EXCEL_SHEET_NAME_MAX]


def _infer_arch_name_from_fingerprint(
    fingerprint: frozenset,
    all_fingerprints: Sequence[frozenset],
    step_phase: str = "",
) -> str:
    """Fallback tab stem from distinctive cpu_ops (matches export phase 4)."""
    other_fps = [f for f in all_fingerprints if f != fingerprint]
    common = fingerprint.intersection(*other_fps) if other_fps else frozenset()
    distinctive = fingerprint - common
    if not distinctive:
        distinctive = fingerprint
    blob = " ".join(str(x) for x in distinctive)
    if step_phase == "prefill" and re.search(r"reshape_and_cache", blob, re.I):
        return "reshape_and_cache_flash"
    if step_phase == "decode" and re.search(r"paged_attention", blob, re.I):
        return "paged_attention_v1"
    for pattern, tab_stem in _ARCH_OP_TAB_RULES:
        if pattern.search(blob):
            return tab_stem
    stripped_all = [op.rsplit("::", 1)[-1] if "::" in op else op for op in distinctive]
    skip = (
        "copy",
        "reshape",
        "silu_and_mul",
        "topk_softmax",
        "ck_moe",
        "fused_moe",
        "moe_sorting",
    )
    candidates = [s for s in stripped_all if not any(x in s.lower() for x in skip)]
    if not candidates:
        candidates = list(stripped_all)
    return max(candidates, key=len) if candidates else ""


def _discover_inner_module_names_from_tree(
    tree,
    decoder_class: str,
    fp_groups: List[Tuple[frozenset, int]],
) -> Dict[frozenset, str]:
    """Port of export ``_discover_inner_module_names`` using the in-memory tree.

    Reads nn.Module children of ``<decoder_class>_0`` and matches cpu_ops in
    each child's span to architecture fingerprints.
    """
    if not decoder_class or not fp_groups:
        return {}

    decoder_0_label = f"{decoder_class}_0"
    decoder_0_span: Optional[Tuple[float, float]] = None
    child_events: Dict[str, Tuple[float, float]] = {}

    for ev in tree.events:
        if ev.get("cat") != "python_function":
            continue
        name = ev.get("name", "")
        if not name.startswith(_NN_MODULE_PREFIX):
            continue
        raw = name[len(_NN_MODULE_PREFIX) :].strip()
        m = re.match(r"^(.+?)_(\d+)$", raw)
        mtype = m.group(1) if m else raw
        try:
            ts = float(ev.get("ts", 0))
            dur = float(ev.get("dur", 0))
        except (ValueError, TypeError):
            continue

        if raw == decoder_0_label and decoder_0_span is None:
            decoder_0_span = (ts, ts + dur)
            continue

        if decoder_0_span is not None:
            span_ts, span_end = decoder_0_span
            if span_ts <= ts and ts + dur <= span_end:
                if mtype != decoder_class and mtype not in child_events:
                    child_events[mtype] = (ts, dur)

    if not child_events:
        return {}

    direct_children: List[str] = []
    for cls_a, (ts_a, dur_a) in child_events.items():
        is_grandchild = False
        for cls_b, (ts_b, dur_b) in child_events.items():
            if cls_a == cls_b:
                continue
            if ts_b <= ts_a and ts_a + dur_a <= ts_b + dur_b:
                is_grandchild = True
                break
        if not is_grandchild:
            direct_children.append(cls_a)

    candidates_mod = [dc for dc in direct_children if _is_architecture_module_class(dc)]
    if not candidates_mod:
        return {}

    span_ts0, span_end0 = decoder_0_span  # type: ignore[misc]
    cand_spans = {dc: child_events[dc] for dc in candidates_mod}
    cand_ops: Dict[str, set] = {dc: set() for dc in candidates_mod}
    _MAX_OPS = 50

    for ev in tree.events:
        if ev.get("cat") != "cpu_op":
            continue
        name = ev.get("name")
        if not isinstance(name, str):
            continue
        try:
            ts = float(ev.get("ts", 0))
            dur = float(ev.get("dur", 0))
        except (ValueError, TypeError):
            continue
        if ts + dur < span_ts0 or ts > span_end0:
            continue
        for dc, (dc_ts, dc_dur) in cand_spans.items():
            if dc_ts <= ts and ts + dur <= dc_ts + dc_dur:
                cand_ops[dc].add(name)
        if all(len(ops) >= _MAX_OPS for ops in cand_ops.values()):
            break

    all_fps = [fp for fp, _ in fp_groups]
    type_distinctive: Dict[frozenset, frozenset] = {}
    for fp, _ in fp_groups:
        others = [f for f in all_fps if f != fp]
        common = fp.intersection(*others) if others else frozenset()
        type_distinctive[fp] = fp - common

    result: Dict[frozenset, str] = {}
    for dc, ops in cand_ops.items():
        if not _is_architecture_module_class(dc):
            continue
        for fp, distinctive in type_distinctive.items():
            if fp in result:
                continue
            if ops & distinctive:
                result[fp] = dc

    for fp, _ in fp_groups:
        if fp in result:
            continue
        name = _infer_arch_name_from_fingerprint(fp, all_fps, "")
        if name:
            result[fp] = name

    if result:
        print(
            "[inference_ext] Discovered arch names: "
            + ", ".join(f"{name}" for name in sorted(set(result.values())))
        )
    return result


def _decompose_layers(
    tree, events: Optional[list] = None
) -> Optional[LayerDecomposition]:
    """Detect decoder layers, assign kernels via subtree, tokenize, and cluster.

    Includes anomaly filtering (layers with >10x median kernel count are
    excluded), fingerprint-based architecture grouping, and architecture
    naming from inner nn.Module children.
    """
    if events is None:
        events = tree.events

    layer_events: List[Tuple[int, dict]] = []
    for ev in events:
        if ev.get("cat") != "python_function":
            continue
        name = ev.get("name", "")
        m = DECODER_LAYER_RE.search(name)
        if m:
            layer_events.append((int(m.group(1)), ev))

    if not layer_events:
        return None

    decoder_class = _decoder_class_from_tree(tree, events)
    layer_events = _filter_outer_decoder_layer_events(layer_events, decoder_class)
    if not layer_events:
        return None
    if not decoder_class:
        decoder_class = _decoder_class_from_events(layer_events)

    layer_events.sort(key=lambda x: (x[0], x[1].get("ts", 0)))

    # --- First pass: collect kernel info per layer ---
    raw_layer_infos: List[Tuple[LayerInfo, dict]] = []  # (info, layer_ev)
    raw_kernel_to_layer: Dict[int, int] = {}

    for layer_idx, layer_ev in layer_events:
        subtree_uids = _collect_layer_subtree_uids(tree, layer_ev)

        verify_ranges = getattr(tree, "_inference_ext_verify_ranges", []) or []

        layer_kernels: List[dict] = []
        for uid in subtree_uids:
            try:
                ev = tree.get_UID2event(uid)
            except (KeyError, TypeError):
                continue
            if ev.get("cat") in GPU_CATEGORIES:
                if (
                    _is_sglang_mtp(tree)
                    and verify_ranges
                    and layer_ev.get("step_phase", "") == "decode"
                    and not _event_ts_in_ranges(ev.get("ts", 0), verify_ranges)
                ):
                    continue
                layer_kernels.append(ev)

        layer_kernels.sort(key=lambda e: e.get("ts", 0))

        tokens: List[str] = []
        total_dur = 0.0
        kernel_uids: List[int] = []

        subtree_cpu_ops = _collect_layer_cpu_op_names(
            tree,
            layer_ev,
            subtree_uids,
            layer_kernels,
        )

        for kev in layer_kernels:
            kuid = kev.get("UID")
            kernel_uids.append(kuid)
            raw_kernel_to_layer[kuid] = layer_idx
            total_dur += kev.get("dur", 0)

            cpu_op_name = ""
            parent_uid = kev.get("parent")
            if parent_uid is not None:
                try:
                    parent_ev = tree.get_UID2event(parent_uid)
                    if parent_ev.get("cat") == "cpu_op":
                        cpu_op_name = parent_ev.get("name", "")
                except (KeyError, TypeError):
                    pass

            tokens.append(_canonical_token(cpu_op_name, kev.get("name", "")))

        tokens_compact = _compress_runs(tokens)
        ff_kind = _classify_layer_kind(tokens)

        step_phase = layer_ev.get("step_phase", "")
        if not step_phase and layer_kernels:
            step_phase = layer_kernels[0].get("step_phase", "")

        info = LayerInfo(
            layer_idx=layer_idx,
            layer_name=layer_ev.get("name", ""),
            tokens=tokens,
            tokens_compact=tokens_compact,
            ff_kind=ff_kind,
            step_phase=step_phase,
            total_kernel_dur_us=total_dur,
            kernel_count=len(layer_kernels),
            kernel_uids=kernel_uids,
            cpu_op_names=subtree_cpu_ops,
        )
        raw_layer_infos.append((info, layer_ev))

    layer_ev_by_idx: Dict[Tuple[str, int], dict] = {}
    for idx, ev in layer_events:
        phase = (ev.get("step_phase") or "").strip()
        layer_ev_by_idx[(phase, int(idx))] = ev

    for li, _ in raw_layer_infos:
        li.arch_fingerprint = _compute_layer_fingerprint(li.cpu_op_names)

    phase_to_layers: Dict[str, List[LayerInfo]] = {}
    for li, _ in raw_layer_infos:
        if li.step_phase:
            phase_to_layers.setdefault(li.step_phase, []).append(li)

    # KernelBreakdown export: per-phase anomaly filter, fingerprint groups,
    # min(layer_idx) representative per group (find_unique_arch_layers).
    arch_reps: Dict[Tuple[str, str], int] = {}
    layer_to_arch: Dict[Tuple[str, int], str] = {}
    for phase, phase_layers in phase_to_layers.items():
        if not phase:
            continue
        excluded = _exclude_anomalous_layers_export_style(
            phase_layers,
            decoder_class,
        )
        if excluded:
            excluded_uids: set = set()
            for li in phase_layers:
                if (li.step_phase, li.layer_idx) in excluded:
                    excluded_uids.update(li.kernel_uids)
            for uid in excluded_uids:
                raw_kernel_to_layer.pop(uid, None)
            phase_layers = [
                li
                for li in phase_layers
                if (li.step_phase, li.layer_idx) not in excluded
            ]
            raw_layer_infos[:] = [
                (li, ev)
                for li, ev in raw_layer_infos
                if (li.step_phase, li.layer_idx) not in excluded
            ]
            phase_to_layers[phase] = phase_layers

        phase_reps, phase_arch_map = _arch_representatives_from_fingerprints(
            phase,
            phase_layers,
            tree,
            decoder_class,
            layer_ev_by_idx,
        )
        arch_reps.update(phase_reps)
        layer_to_arch.update(phase_arch_map)

    multi_arch = len(arch_reps) >= 2
    if multi_arch:
        tab_preview = [
            f"{arch}_layer{idx}({phase})"
            for (phase, arch), idx in sorted(
                arch_reps.items(), key=lambda x: (x[0][0], x[1])
            )
        ]
        print(
            f"[inference_ext] Detected {len(arch_reps)} architecture tabs: "
            + ", ".join(tab_preview)
        )

    # Layer summary: one row per representative tab; timeline keeps full mapping.
    all_layer_infos: List[LayerInfo] = []
    kernel_to_layer = dict(raw_kernel_to_layer)

    if multi_arch:
        seen_phase_arch: set = set()
        for li, _ in raw_layer_infos:
            key = (li.step_phase, li.arch_type)
            if key in seen_phase_arch:
                continue
            if li.layer_idx != arch_reps.get(key):
                continue
            seen_phase_arch.add(key)
            all_layer_infos.append(li)
    else:
        for li, _ in raw_layer_infos:
            all_layer_infos.append(li)

    # Template clustering
    key_counts: Dict[Tuple[str, ...], int] = {}
    for layer in all_layer_infos:
        key = tuple(layer.tokens_compact)
        key_counts[key] = key_counts.get(key, 0) + 1

    keys_sorted = sorted(key_counts.keys(), key=lambda k: (-key_counts[k], -len(k), k))
    key_to_id: Dict[Tuple[str, ...], int] = {k: i for i, k in enumerate(keys_sorted)}

    templates: Dict[int, List[str]] = {}
    for layer in all_layer_infos:
        tid = key_to_id[tuple(layer.tokens_compact)]
        layer.template_id = tid
        if tid not in templates:
            templates[tid] = layer.tokens_compact

    print(
        f"[inference_ext] Decomposed {len(all_layer_infos)} decoder layers "
        f"({len(templates)} templates, "
        f"{sum(1 for l in all_layer_infos if l.ff_kind == 'moe')} MoE, "
        f"{sum(1 for l in all_layer_infos if l.ff_kind == 'dense')} dense)"
    )

    return LayerDecomposition(
        layers=all_layer_infos,
        templates=templates,
        kernel_to_layer=kernel_to_layer,
        layer_to_arch=layer_to_arch,
        arch_reps=arch_reps,
    )


# ---------------------------------------------------------------------------
# Annotation parsing helpers
# ---------------------------------------------------------------------------
def _get_iter_details_from_name(name: str) -> dict:
    """Parse iteration details (batch size, request counts) from a
    ``user_annotation`` event name."""
    name = name.replace("(", "_").replace(")", "_")
    parts = re.sub(r"[sqk]+", "_", name).split("_")
    try:
        if len(parts) < 10:
            ctx_req, ctx_sum, gen_req, gen_sum = (
                parts[2],
                parts[3],
                parts[6],
                parts[7],
            )
        elif len(parts) < 12:
            ctx_req, ctx_sum, gen_req, gen_sum = (
                parts[2],
                parts[3],
                parts[7],
                parts[8],
            )
        else:
            ctx_req, ctx_sum, gen_req, gen_sum = (
                parts[3],
                parts[5],
                parts[11],
                parts[13],
            )
        return {
            "batch_size": int(ctx_sum) + int(gen_sum),
            "num_requests": int(ctx_req) + int(gen_req),
            "context_requests": int(ctx_req),
            "generation_requests": int(gen_req),
        }
    except (IndexError, ValueError):
        return {
            "batch_size": 0,
            "num_requests": 0,
            "context_requests": 0,
            "generation_requests": 0,
        }


def _find_annotation_roots(events: list) -> List[dict]:
    """Find user_annotation events that match vLLM/SGLang iteration patterns."""
    roots = []
    for event in events:
        if event.get("cat") != "user_annotation":
            continue
        name = event.get("name", "")
        if any(p.match(name) for p in ANNOTATION_PATTERNS):
            roots.append(event)
    roots.sort(key=lambda e: e.get("ts", 0))
    return roots


def _find_forward_annotation_within(
    events: list,
    outer_event: dict,
) -> Optional[dict]:
    """Find the ``gpu_model_runner: forward`` annotation nested within
    *outer_event*'s time interval.

    vLLM traces have broad ``execute_context_*`` annotations that span
    the entire iteration (preprocess + forward + sample + postprocess).
    The ``gpu_model_runner: forward`` annotation is the narrower boundary
    that contains only the model execution kernels we care about.

    Returns the forward annotation event if found, otherwise None.
    """
    outer_ts = outer_event.get("ts", 0)
    outer_end = outer_ts + outer_event.get("dur", 0)

    for ev in events:
        if ev.get("cat") != "user_annotation":
            continue
        if not VLLM_FORWARD_RE.match(ev.get("name", "")):
            continue
        ev_ts = ev.get("ts", 0)
        if outer_ts <= ev_ts <= outer_end:
            return ev
    return None


# ---------------------------------------------------------------------------
# Batch-size inference from cpu_op input shapes
# ---------------------------------------------------------------------------
def _find_run_batches(events: list) -> List[dict]:
    """Return ``python_function`` events whose name matches
    ``scheduler.py.*run_batch``, sorted by timestamp."""
    batches = [
        e
        for e in events
        if e.get("cat") == "python_function" and RUN_BATCH_RE.search(e.get("name", ""))
    ]
    batches.sort(key=lambda e: e.get("ts", 0))
    return batches


def _infer_batch_size_for_interval(
    cpu_ops: list, ts_start: float, ts_end: float
) -> int:
    """Infer the batch size from ``cpu_op`` events within a time interval.

    Collects the first dimension of every ``Input Dims`` tensor shape from
    all ``cpu_op`` events whose timestamp falls within ``[ts_start, ts_end]``
    and returns the **mode** (most frequent value).  This matches the approach
    used by the batch-size inference in the reporting pipeline.

    ``Input Dims`` is accessed as a native Python list (already parsed from
    JSON by the trace loader), e.g. ``[[128, 7168], [7168]]``.
    """
    first_dims: List[int] = []
    for e in cpu_ops:
        ts = e.get("ts", 0)
        if ts < ts_start or ts > ts_end:
            continue
        input_dims = e.get("args", {}).get("Input Dims")
        if not input_dims:
            continue
        for dim_list in input_dims:
            if isinstance(dim_list, list) and dim_list:
                if isinstance(dim_list[0], int):
                    first_dims.append(dim_list[0])
    if not first_dims:
        return 0
    return Counter(first_dims).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Subtree collection & GPU duration helpers
# ---------------------------------------------------------------------------
def _collect_subtree_uids(tree, root_event: dict) -> set:
    """Collect all event UIDs belonging to the subtree rooted at *root_event*.

    When the root has ``children`` links (tree built with python functions),
    a BFS through the tree is used.  Otherwise falls back to collecting
    every event whose timestamp falls within the root's time interval,
    plus any GPU events reachable via ``gpu_events`` links on those events
    (which captures trailing kernels that finish after the CPU interval).
    """
    uids: set = set()

    if root_event.get("children"):
        stack = [root_event]
        while stack:
            ev = stack.pop()
            uid = ev.get("UID")
            if uid is None or uid in uids:
                continue
            uids.add(uid)
            for child_uid in ev.get("children", []):
                try:
                    stack.append(tree.get_UID2event(child_uid))
                except KeyError:
                    pass
        for gpu_uid in root_event.get("gpu_events", []):
            uids.add(gpu_uid)
    else:
        ts_start = root_event.get("ts", 0)
        ts_end = ts_start + root_event.get("dur", 0)
        root_uid = root_event.get("UID")
        if root_uid is not None:
            uids.add(root_uid)
        gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
        for ev in tree.events:
            cat = ev.get("cat", "")
            if cat in gpu_cats:
                continue
            ev_ts = ev.get("ts", 0)
            if ev_ts < ts_start or ev_ts > ts_end:
                continue
            uid = ev.get("UID")
            if uid is not None:
                uids.add(uid)
            for gpu_uid in ev.get("gpu_events", []):
                uids.add(gpu_uid)

    return uids


def _total_gpu_dur(tree, run_batch_event: dict) -> float:
    """Sum ``dur`` of all GPU kernel events associated with *run_batch_event*.

    Uses the ``gpu_events`` list when available (tree built with python
    functions).  Otherwise collects GPU durations from ``gpu_events`` of
    all non-GPU events within the run_batch time interval.
    """
    gpu_uids = run_batch_event.get("gpu_events")
    if gpu_uids:
        total = 0.0
        for gpu_uid in gpu_uids:
            try:
                total += tree.get_UID2event(gpu_uid).get("dur", 0)
            except KeyError:
                pass
        return total

    ts_start = run_batch_event.get("ts", 0)
    ts_end = ts_start + run_batch_event.get("dur", 0)
    seen: set = set()
    total = 0.0
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    for ev in tree.events:
        cat = ev.get("cat", "")
        if cat in gpu_cats:
            continue
        ev_ts = ev.get("ts", 0)
        if ev_ts < ts_start or ev_ts > ts_end:
            continue
        for uid in ev.get("gpu_events", []):
            if uid not in seen:
                seen.add(uid)
                try:
                    total += tree.get_UID2event(uid).get("dur", 0)
                except KeyError:
                    pass
    return total


def _pick_by_nearest_avg_duration(
    tree,
    candidates: List[dict],
) -> dict:
    """Among tied *candidates*, return the one whose total GPU kernel
    duration is closest to the average across all candidates.

    When the candidates lack direct ``gpu_events`` links (tree built
    without python functions), GPU durations are computed in a single
    pass over all events rather than per-candidate to avoid O(N*M).
    """
    if len(candidates) == 1:
        return candidates[0]

    need_interval_fallback = not candidates[0].get("gpu_events")
    if need_interval_fallback:
        cand_ranges = [
            (rb, rb.get("ts", 0), rb.get("ts", 0) + rb.get("dur", 0))
            for rb in candidates
        ]
        cand_durs = {id(rb): 0.0 for rb in candidates}
        gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
        for ev in tree.events:
            if ev.get("cat", "") in gpu_cats:
                continue
            ev_gpu = ev.get("gpu_events")
            if not ev_gpu:
                continue
            ev_ts = ev.get("ts", 0)
            for rb, ts_s, ts_e in cand_ranges:
                if ts_s <= ev_ts <= ts_e:
                    for uid in ev_gpu:
                        try:
                            cand_durs[id(rb)] += tree.get_UID2event(uid).get("dur", 0)
                        except KeyError:
                            pass
                    break
        durs = [(rb, cand_durs[id(rb)]) for rb in candidates]
    else:
        durs = [(rb, _total_gpu_dur(tree, rb)) for rb in candidates]

    avg = mean(d for _, d in durs)
    return min(durs, key=lambda x: (abs(x[1] - avg), x[0].get("ts", 0)))[0]


# ---------------------------------------------------------------------------
# SGLang EAGLE/MTP detection and verify-step selection (KernelBreakdown export)
# ---------------------------------------------------------------------------
def _detect_mtp_markers(events: list) -> bool:
    """True when the trace has EAGLE/MTP ``run_batch`` child markers (SGLang)."""
    run_batches = _find_run_batches(events)
    if not run_batches:
        return False
    rb_intervals = [
        (rb, rb.get("ts", 0), rb.get("ts", 0) + rb.get("dur", 0)) for rb in run_batches
    ]
    rb_uid_set = {rb.get("UID") for rb in run_batches}
    for pf in events:
        if pf.get("cat") != "python_function" or pf.get("UID") in rb_uid_set:
            continue
        pf_name = pf.get("name", "")
        pf_ts = pf.get("ts", 0)
        for _rb, rb_start, rb_end in rb_intervals:
            if rb_start <= pf_ts <= rb_end:
                if FORWARD_IDLE_RE.search(pf_name) or EAGLE_DRAFT_RE.search(pf_name):
                    return True
                break
    return False


def _kernel_name_indicates_moe(kernel_name: str) -> bool:
    return "moe" in (kernel_name or "").lower()


def _first_dim_from_input_dims(input_dims) -> Optional[int]:
    if not input_dims:
        return None
    for dim_list in input_dims:
        if isinstance(dim_list, list) and dim_list:
            if isinstance(dim_list[0], int) and dim_list[0] > 0:
                return int(dim_list[0])
    return None


def _rmsnorm_batch_dim0_from_cpu_op(
    cpu_op_ev: dict, rmsnorm_substring: str = DEFAULT_RMSNORM_SUBSTRING
) -> Optional[int]:
    """Batch dim0 from cpu_ops whose name contains *rmsnorm_substring* (export rule)."""
    needle = (rmsnorm_substring or "").strip().lower()
    if not needle:
        return None
    name = (cpu_op_ev.get("name") or "").strip()
    if needle not in name.lower():
        return None
    return _first_dim_from_input_dims(cpu_op_ev.get("args", {}).get("Input Dims"))


def _intervals_overlap(
    start_a: float, end_a: float, start_b: float, end_b: float
) -> bool:
    return start_a < end_b and start_b < end_a


def _verify_ranges_for_run_batch(
    run_batch: dict,
    mtp_sections: List[Tuple[str, float, float]],
) -> List[Tuple[float, float]]:
    """Intersect *run_batch* interval with MTP ``verify`` section intervals."""
    rb_start = run_batch.get("ts", 0)
    rb_end = rb_start + run_batch.get("dur", 0)
    ranges: List[Tuple[float, float]] = []
    for section_name, sec_start, sec_end in mtp_sections:
        if section_name != "verify":
            continue
        lo = max(rb_start, sec_start)
        hi = min(rb_end, sec_end)
        if lo < hi:
            ranges.append((lo, hi))
    return ranges


def _event_ts_in_ranges(ts: float, ranges: List[Tuple[float, float]]) -> bool:
    return any(lo <= ts < hi for lo, hi in ranges)


def _total_gpu_dur_in_ranges(tree, ranges: List[Tuple[float, float]]) -> float:
    total = 0.0
    for ev in tree.events:
        if ev.get("cat") not in GPU_CATEGORIES:
            continue
        if _event_ts_in_ranges(ev.get("ts", 0), ranges):
            total += ev.get("dur", 0)
    return total


def _select_sglang_mtp_verify_decode_step(
    tree,
    run_batches: List[dict],
    phase_map: Dict[int, str],
    mtp_sections: List[Tuple[str, float, float]],
    rmsnorm_substring: str = DEFAULT_RMSNORM_SUBSTRING,
) -> Tuple[Optional[dict], List[Tuple[float, float]]]:
    """
    Pick the decode ``run_batch`` using the SGLang KernelBreakdown exporter rules:
    - phase is ``decode`` only (not ``decode_idle``)
    - score verify-section rows by max rmsnorm cpu_op batch dim0
    - require at least one verify kernel with ``moe`` in kernel_name
    - tie-break: nearest_avg_duration on verify kernel time
    """
    cpu_ops = [e for e in tree.events if e.get("cat") == "cpu_op"]
    rb_sorted = sorted(run_batches, key=lambda e: e.get("ts", 0))

    batch_to_max_batch: Dict[int, int] = {}
    batch_verify_ranges: Dict[int, List[Tuple[float, float]]] = {}
    batch_has_moe: set = set()

    for batch_idx, rb in enumerate(rb_sorted):
        uid = rb.get("UID")
        if phase_map.get(uid) != "decode":
            continue
        verify_ranges = _verify_ranges_for_run_batch(rb, mtp_sections)
        if not verify_ranges:
            continue
        batch_verify_ranges[batch_idx] = verify_ranges

        for cop in cpu_ops:
            ts = cop.get("ts", 0)
            if not _event_ts_in_ranges(ts, verify_ranges):
                continue
            b = _rmsnorm_batch_dim0_from_cpu_op(cop, rmsnorm_substring)
            if b is not None:
                batch_to_max_batch[batch_idx] = max(
                    batch_to_max_batch.get(batch_idx, 0), b
                )

        for ev in tree.events:
            if ev.get("cat") not in GPU_CATEGORIES:
                continue
            if not _event_ts_in_ranges(ev.get("ts", 0), verify_ranges):
                continue
            if _kernel_name_indicates_moe(ev.get("name", "")):
                batch_has_moe.add(batch_idx)

    eligible = {
        bid: dim for bid, dim in batch_to_max_batch.items() if bid in batch_has_moe
    }
    if not eligible:
        print(
            "[inference_ext] SGLang MTP: no decode verify step with both "
            f"rmsnorm-scored batch dim0 and MoE in kernel_name."
        )
        return None, []

    max_dim0 = max(eligible.values())
    candidates_idx = [bid for bid, dim in eligible.items() if dim == max_dim0]
    tied_count = len(candidates_idx)

    if len(candidates_idx) == 1:
        chosen_idx = candidates_idx[0]
    else:
        cand_durs = [
            (bid, _total_gpu_dur_in_ranges(tree, batch_verify_ranges[bid]))
            for bid in candidates_idx
        ]
        avg_dur = sum(d for _, d in cand_durs) / len(cand_durs)
        chosen_idx = min(cand_durs, key=lambda x: (abs(x[1] - avg_dur), x[0]))[0]

    chosen_rb = rb_sorted[chosen_idx]
    verify_ranges = batch_verify_ranges[chosen_idx]
    if tied_count > 1:
        print(
            f"[inference_ext] SGLang MTP: {tied_count} steps tied at "
            f"max rmsnorm batch dim0={max_dim0}; chose batch_idx={chosen_idx} "
            f"(tie-breaker: nearest_avg_duration)."
        )
    print(
        f"[inference_ext] SGLang MTP verify export: selected run_batch "
        f"batch_idx={chosen_idx} (ts={chosen_rb.get('ts', 0):.0f}, "
        f"dur={chosen_rb.get('dur', 0):.0f}), max rmsnorm batch dim0={max_dim0}"
    )
    return chosen_rb, verify_ranges


def _select_sglang_mtp_prefill_step(
    tree,
    run_batches: List[dict],
    phase_map: Dict[int, str],
    rmsnorm_substring: str = DEFAULT_RMSNORM_SUBSTRING,
) -> Optional[dict]:
    """Pick the prefill ``run_batch`` with the largest rmsnorm batch dim0."""
    cpu_ops = [e for e in tree.events if e.get("cat") == "cpu_op"]
    rb_sorted = sorted(run_batches, key=lambda e: e.get("ts", 0))

    batch_to_max_batch: Dict[int, int] = {}
    for batch_idx, rb in enumerate(rb_sorted):
        uid = rb.get("UID")
        if phase_map.get(uid) != "prefill":
            continue
        rb_ts = rb.get("ts", 0)
        rb_end = rb_ts + rb.get("dur", 0)
        max_b = 0
        for cop in cpu_ops:
            ts = cop.get("ts", 0)
            if rb_ts <= ts < rb_end:
                b = _rmsnorm_batch_dim0_from_cpu_op(cop, rmsnorm_substring)
                if b is not None:
                    max_b = max(max_b, b)
        if max_b > 0:
            batch_to_max_batch[batch_idx] = max_b

    if not batch_to_max_batch:
        print(
            "[inference_ext] SGLang MTP: no prefill step with "
            "rmsnorm-scored batch dim0."
        )
        return None

    max_dim0 = max(batch_to_max_batch.values())
    candidates_idx = [bid for bid, dim in batch_to_max_batch.items() if dim == max_dim0]
    tied_count = len(candidates_idx)

    if len(candidates_idx) == 1:
        chosen_idx = candidates_idx[0]
    else:
        cand_durs = [
            (bid, _total_gpu_dur(tree, rb_sorted[bid])) for bid in candidates_idx
        ]
        avg_dur = sum(d for _, d in cand_durs) / len(cand_durs)
        chosen_idx = min(cand_durs, key=lambda x: (abs(x[1] - avg_dur), x[0]))[0]

    chosen_rb = rb_sorted[chosen_idx]
    if tied_count > 1:
        print(
            f"[inference_ext] SGLang MTP prefill: {tied_count} steps tied at "
            f"max rmsnorm batch dim0={max_dim0}; chose batch_idx={chosen_idx} "
            f"(tie-breaker: nearest_avg_duration)."
        )
    print(
        f"[inference_ext] SGLang MTP prefill export: selected run_batch "
        f"batch_idx={chosen_idx} (ts={chosen_rb.get('ts', 0):.0f}, "
        f"dur={chosen_rb.get('dur', 0):.0f}), "
        f"max rmsnorm batch dim0={max_dim0}"
    )
    return chosen_rb


def _filter_subtree_uids_to_ranges(
    tree, uids: set, ranges: List[Tuple[float, float]]
) -> set:
    """Keep GPU events in *ranges* and decoder-layer ``python_function`` overlaps."""
    if not ranges:
        return uids
    kept: set = set()
    for uid in uids:
        try:
            ev = tree.get_UID2event(uid)
        except (KeyError, TypeError):
            continue
        ts = ev.get("ts", 0)
        end = ts + ev.get("dur", 0)
        cat = ev.get("cat", "")
        if cat == "python_function" and DECODER_LAYER_RE.search(ev.get("name", "")):
            if any(_intervals_overlap(ts, end, lo, hi) for lo, hi in ranges):
                kept.add(uid)
            continue
        if _event_ts_in_ranges(ts, ranges):
            kept.add(uid)
    for uid in list(kept):
        try:
            ev = tree.get_UID2event(uid)
        except (KeyError, TypeError):
            continue
        for gpu_uid in ev.get("gpu_events", []):
            if gpu_uid not in uids or gpu_uid in kept:
                continue
            try:
                gev = tree.get_UID2event(gpu_uid)
            except (KeyError, TypeError):
                continue
            if _event_ts_in_ranges(gev.get("ts", 0), ranges):
                kept.add(gpu_uid)
    return kept


def _decoder_layer_sort_key(layer_val) -> Tuple[int, object]:
    """Numeric layer index sort (avoids ``'10' < '4'`` string ordering)."""
    try:
        return (0, int(layer_val))
    except (TypeError, ValueError):
        return (1, str(layer_val))


_MOE_ROUTING_CPU_OPS = ("aiter::moe", "_moe_C")


def _cpu_op_matches_moe_routing(cpu_op_name: str, needle: str) -> bool:
    """Match vLLM export ``_cpu_op_matches_moe_routing`` (KernelBreakdown)."""
    cpu = (cpu_op_name or "").strip()
    n = (needle or "").strip()
    if not n:
        return False
    if n == "aiter::moe":
        return cpu.startswith("aiter::moe")
    if "::" in n:
        return cpu == n
    return cpu == n or cpu.startswith(n + "::")


def _first_moe_layer_from_routing_cpu_op(df: pd.DataFrame) -> Optional[int]:
    """
    vLLM ``select_export_rows_for_step`` parity: first chronologically sorted
    kernel whose parent cpu_op matches MoE routing (``aiter::moe`` / ``_moe_C``).
    """
    if df.empty or "Parent cpu_op" not in df.columns:
        return None
    sort_df = df
    if "Timestamp (us)" in df.columns:
        sort_df = df.sort_values("Timestamp (us)")
    for needle in _MOE_ROUTING_CPU_OPS:
        for _, row in sort_df.iterrows():
            if not _cpu_op_matches_moe_routing(row.get("Parent cpu_op", ""), needle):
                continue
            layer = row.get("decoder_layer")
            if layer == "" or pd.isna(layer):
                continue
            try:
                return int(layer)
            except (TypeError, ValueError):
                return layer  # type: ignore[return-value]
    return None


def _decoder_layers_with_moe_kernel_name(df: pd.DataFrame) -> List:
    """Layer indices with at least one row whose Kernel name contains ``moe``."""
    if df.empty or "Kernel name" not in df.columns:
        return []
    layers: List = []
    for layer in sorted(
        df["decoder_layer"].dropna().unique(), key=_decoder_layer_sort_key
    ):
        if layer == "":
            continue
        sub = df[df["decoder_layer"] == layer]
        if sub["Kernel name"].str.lower().str.contains("moe", regex=False).any():
            layers.append(layer)
    return layers


def _apply_sglang_mtp_first_moe_export_scope(
    df: pd.DataFrame,
    layer_decomp: Optional[LayerDecomposition],
    tree,
) -> Tuple[pd.DataFrame, Optional[LayerDecomposition]]:
    """
    SGLang MTP: per phase, export one MoE decoder layer.
    Decode: lowest layer with ``moe`` in kernel name (SGLang verify export).
    Prefill: vLLM routing parity — layer of the first ``aiter::moe`` / ``_moe_C``
    parent cpu_op (chronological), not lowest layer with moe in kernel name.
    """
    decoder_class = _decoder_class_from_tree(tree) or ""
    arch_name = decoder_class or "decode"
    kept_parts: List[pd.DataFrame] = []
    arch_reps: Dict[Tuple[str, str], int] = {}
    layer_to_arch: Dict[Tuple[str, int], str] = {}

    for phase in ("prefill", "decode"):
        phase_df = df[df["step_phase"] == phase]
        if phase_df.empty:
            continue
        pre_count = len(phase_df)
        first_moe: Optional[int] = None
        if phase == "prefill":
            first_moe = _first_moe_layer_from_routing_cpu_op(phase_df)
        if first_moe is None:
            moe_layers = _decoder_layers_with_moe_kernel_name(phase_df)
            if not moe_layers:
                print(
                    f"[inference_ext] SGLang MTP first-MoE export ({phase}): "
                    "no MoE layer found"
                )
                kept_parts.append(phase_df)
                continue
            try:
                first_moe = int(moe_layers[0])
            except (TypeError, ValueError):
                first_moe = moe_layers[0]  # type: ignore[assignment]

        layer_col = pd.to_numeric(phase_df["decoder_layer"], errors="coerce")
        phase_kept = phase_df[layer_col == first_moe].copy()
        if "arch_type" in phase_kept.columns:
            phase_kept["arch_type"] = arch_name

        label = f"{decoder_class}_{first_moe}" if decoder_class else str(first_moe)
        print(
            f"[inference_ext] SGLang MTP first-MoE export ({phase}): {label!r} "
            f"({len(phase_kept)} / {pre_count} kernels)"
        )
        kept_parts.append(phase_kept)
        arch_reps[(phase, arch_name)] = first_moe
        layer_to_arch[(phase, first_moe)] = arch_name

    if kept_parts:
        df = pd.concat(kept_parts, ignore_index=True)

    if layer_decomp is not None and arch_reps:
        layer_decomp.arch_reps = arch_reps
        layer_decomp.layer_to_arch.update(layer_to_arch)

    return df, layer_decomp


# ---------------------------------------------------------------------------
# Single-step selection: annotation path
# ---------------------------------------------------------------------------
def _find_best_step_annotation(
    tree,
    annotation_roots: List[dict],
    run_batches: List[dict],
) -> Optional[dict]:
    """Pick the single step event that contains the annotation with
    the highest ``num_requests``.

    When ``run_batches`` is non-empty, each annotation is mapped to its
    containing ``run_batch`` and the best ``run_batch`` is returned.

    When ``run_batches`` is empty (typical vLLM traces without SGLang's
    ``scheduler.py.*run_batch``), the **annotations themselves** are used
    as step boundaries and the best annotation event is returned directly.
    """
    if not annotation_roots:
        return None

    # Score every annotation
    scored: List[Tuple[dict, int]] = []
    for ann in annotation_roots:
        det = _get_iter_details_from_name(ann.get("name", ""))
        scored.append((ann, det["num_requests"]))

    if run_batches:
        rb_sorted = sorted(run_batches, key=lambda e: e.get("ts", 0))
        rb_starts = [rb.get("ts", 0) for rb in rb_sorted]
        rb_ends = [rb.get("ts", 0) + rb.get("dur", 0) for rb in rb_sorted]

        best_score = -1
        candidates: List[dict] = []

        for ann, score in scored:
            ann_ts = ann.get("ts", 0)
            containing_rb = None
            for idx, rb in enumerate(rb_sorted):
                if rb_starts[idx] <= ann_ts <= rb_ends[idx]:
                    containing_rb = rb
                    break
            if containing_rb is None:
                continue

            if score > best_score:
                best_score = score
                candidates = [containing_rb]
            elif score == best_score:
                if containing_rb.get("UID") not in {c.get("UID") for c in candidates}:
                    candidates.append(containing_rb)

        if not candidates:
            return None

        chosen = _pick_by_nearest_avg_duration(tree, candidates)
        print(
            f"[inference_ext] Annotation path: selected run_batch "
            f"(ts={chosen.get('ts', 0):.0f}, dur={chosen.get('dur', 0):.0f}) "
            f"with num_requests={best_score}"
        )
        return chosen

    # No run_batches -- use annotations directly as step events.
    # Prefer steps that contain nn.Module DecoderLayer events, since
    # those are needed for proper layer decomposition.
    best_score = max(s for _, s in scored)
    candidates = [ann for ann, s in scored if s == best_score]

    # Try to find candidates with nn.Module: *DecoderLayer* events
    # by checking if there are python_function decoder events in each
    # candidate's time window.
    candidates_with_layers: List[dict] = []
    for ann in candidates:
        fwd = _find_forward_annotation_within(tree.events, ann)
        check_ev = fwd if fwd else ann
        ts_start = check_ev.get("ts", 0)
        ts_end = ts_start + check_ev.get("dur", 0)
        has_layers = False
        for ev in tree.events:
            if ev.get("cat") != "python_function":
                continue
            if not DECODER_LAYER_RE.search(ev.get("name", "")):
                continue
            ev_ts = ev.get("ts", 0)
            if ts_start <= ev_ts <= ts_end:
                has_layers = True
                break
        if has_layers:
            candidates_with_layers.append(ann)

    if candidates_with_layers:
        print(
            f"[inference_ext] {len(candidates_with_layers)}/{len(candidates)} "
            f"annotations have DecoderLayer events"
        )
        candidates = candidates_with_layers

    # Pick the candidate with median duration (avoids expensive GPU duration
    # computation on annotation-only traces which can be very large).
    candidates.sort(key=lambda e: e.get("dur", 0))
    chosen = candidates[len(candidates) // 2]

    det = _get_iter_details_from_name(chosen.get("name", ""))
    kind = "decode"
    cr = det.get("context_requests", 0)
    gr = det.get("generation_requests", 0)
    if cr > 0 and gr > 0:
        kind = "prefilldecode"
    elif cr > 0:
        kind = "prefill"
    print(
        f"[inference_ext] Annotation path (no run_batch): selected annotation "
        f"(ts={chosen.get('ts', 0):.0f}, dur={chosen.get('dur', 0):.0f}) "
        f"with num_requests={best_score}, phase={kind}"
    )
    return chosen


# ---------------------------------------------------------------------------
# Single-step selection: cpu_op batch-size fallback path
# ---------------------------------------------------------------------------
def _find_best_step_batch_size(
    tree,
    events: list,
    candidates: List[dict],
) -> Optional[dict]:
    """Pick the single best ``run_batch`` from *candidates* using modal
    cpu_op dim0.

    For each candidate, the batch size is inferred as the most frequent
    first dimension across all ``cpu_op`` ``Input Dims`` tensors within
    that interval (matching the batch-size inference approach in the
    reporting pipeline).  The caller is responsible for providing the
    appropriate candidate set (e.g. decode-only, prefill-only, or all).
    """
    if not candidates:
        return None

    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]

    rb_batch_sizes: List[Tuple[dict, int]] = []
    for rb in candidates:
        ts_start = rb.get("ts", 0)
        ts_end = ts_start + rb.get("dur", 0)
        bs = _infer_batch_size_for_interval(cpu_ops, ts_start, ts_end)
        rb_batch_sizes.append((rb, bs))

    nonzero = [(rb, bs) for rb, bs in rb_batch_sizes if bs > 0]
    if not nonzero:
        return None

    bs_counts = Counter(bs for _, bs in nonzero)
    modal_bs = bs_counts.most_common(1)[0][0]

    modal_candidates = [rb for rb, bs in nonzero if bs == modal_bs]

    chosen = _pick_by_nearest_avg_duration(tree, modal_candidates)
    tied_msg = ""
    if len(modal_candidates) > 1:
        tied_msg = (
            f" ({len(modal_candidates)} steps at batch_size={modal_bs}; "
            f"tie-breaker: nearest_avg_duration)"
        )
    print(
        f"[inference_ext] Selected run_batch "
        f"(ts={chosen.get('ts', 0):.0f}, dur={chosen.get('dur', 0):.0f}) "
        f"with modal batch_size={modal_bs}{tied_msg}"
    )
    return chosen


# ---------------------------------------------------------------------------
# Alternating runs grouping
# ---------------------------------------------------------------------------
def _build_alternating_runs(
    run_batches: List[dict],
    phase_map: Dict[int, str],
) -> List[Tuple[str, List[dict]]]:
    """Group consecutive ``run_batch`` events with the same phase into runs.

    Returns ``[(phase, [rb_events...]), ...]`` sorted by timestamp, e.g.
    ``[("prefill", [rb0, rb1]), ("decode", [rb2, ..., rb12]), ...]``.
    """
    if not run_batches:
        return []
    rb_sorted = sorted(run_batches, key=lambda e: e.get("ts", 0))
    runs: List[Tuple[str, List[dict]]] = []
    cur_phase = phase_map.get(rb_sorted[0].get("UID"), "decode")
    cur_group: List[dict] = [rb_sorted[0]]

    for rb in rb_sorted[1:]:
        ph = phase_map.get(rb.get("UID"), "decode")
        if ph == cur_phase:
            cur_group.append(rb)
        else:
            runs.append((cur_phase, cur_group))
            cur_phase = ph
            cur_group = [rb]
    runs.append((cur_phase, cur_group))
    return runs


def _modal_batch_size_for_run(
    run_batches: List[dict],
    cpu_ops: list,
) -> int:
    """Compute the modal batch size across a list of run_batch events."""
    dims: List[int] = []
    for rb in run_batches:
        ts_start = rb.get("ts", 0)
        ts_end = ts_start + rb.get("dur", 0)
        bs = _infer_batch_size_for_interval(cpu_ops, ts_start, ts_end)
        if bs > 0:
            dims.append(bs)
    if not dims:
        return 0
    return Counter(dims).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# MTP section detection & tagging
# ---------------------------------------------------------------------------
def _detect_mtp_sections(events: list) -> List[Tuple[str, float, float]]:
    """Scan ``python_function`` events for EAGLE section markers.

    Returns a list of ``(section_name, ts_start, ts_end)`` sorted by
    timestamp, or an empty list for non-MTP traces.
    """
    sections: List[Tuple[str, float, float]] = []
    for event in events:
        if event.get("cat") != "python_function":
            continue
        name = event.get("name", "")
        for section_name, pattern in EAGLE_SECTION_PATTERNS:
            if pattern.search(name):
                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                sections.append((section_name, ts, ts + dur))
                break
    sections.sort(key=lambda s: s[1])
    return sections


def _tag_decode_sections(
    events: list, sections: List[Tuple[str, float, float]]
) -> None:
    """Tag every event with a ``decode_section`` field based on which
    MTP section interval contains its timestamp.  Events outside all
    sections receive an empty string."""
    for event in events:
        ts = event.get("ts", 0)
        event["decode_section"] = ""
        for section_name, sec_start, sec_end in sections:
            if sec_start <= ts < sec_end:
                event["decode_section"] = section_name
                break


# ---------------------------------------------------------------------------
# Extension hook 1: tree_postprocess_extension
# ---------------------------------------------------------------------------
def _select_steps_from_region(
    tree,
    run_batches: List[dict],
    phase_map: Dict[int, str],
) -> Tuple[Optional[dict], Optional[dict]]:
    """Select a decode step and (optionally) its preceding prefill step
    from the same highest batch-size region.

    Returns ``(decode_rb, prefill_rb)`` -- *prefill_rb* is ``None`` when
    there is no prefill run immediately before the best decode run.
    """
    # Exclude decode_idle (MTP-only); vLLM/non-MTP SGLang use "decode" / prefill labels.
    decode_phases = {"decode"}
    prefill_phases = {"prefill", "prefilldecode"}
    cpu_ops = [e for e in tree.events if e.get("cat") == "cpu_op"]

    # Build alternating runs
    runs = _build_alternating_runs(run_batches, phase_map)
    if not runs:
        return None, None

    # Find decode runs and their modal batch size
    decode_run_indices: List[Tuple[int, int]] = []  # (run_index, modal_bs)
    for i, (phase, rbs) in enumerate(runs):
        if phase in decode_phases:
            mbs = _modal_batch_size_for_run(rbs, cpu_ops)
            decode_run_indices.append((i, mbs))

    if not decode_run_indices:
        # No decode runs at all -- fall back to all run_batches
        return _find_best_step_batch_size(tree, tree.events, run_batches), None

    # Pick the decode run with the highest modal batch size
    best_run_idx, best_mbs = max(decode_run_indices, key=lambda x: x[1])
    best_decode_rbs = runs[best_run_idx][1]

    print(
        f"[inference_ext] Best decode region: run {best_run_idx} "
        f"({len(best_decode_rbs)} steps, modal batch_size={best_mbs})"
    )

    # Select representative decode step
    decode_rb = _find_best_step_batch_size(
        tree,
        tree.events,
        best_decode_rbs,
    )

    # Look at the immediately preceding run -- if it's a prefill run, pick from it
    prefill_rb: Optional[dict] = None
    if best_run_idx > 0:
        prev_phase, prev_rbs = runs[best_run_idx - 1]
        if prev_phase in prefill_phases and prev_rbs:
            print(
                f"[inference_ext] Preceding prefill region: run {best_run_idx - 1} "
                f"({len(prev_rbs)} steps)"
            )
            prefill_rb = _find_best_step_batch_size(
                tree,
                tree.events,
                prev_rbs,
            )

    return decode_rb, prefill_rb


def tree_postprocess_extension(tree) -> None:
    """Filter the trace tree to the subtree(s) rooted at the representative
    decode (and optionally prefill) ``run_batch`` step(s) from the highest
    batch-size region.

    Called by ``apply_extension`` in the report generator before
    any DataFrames are built.
    """
    # Phase classification (before step selection so we can filter)
    phase_map = _tag_run_batch_phases(tree)

    run_batches = _find_run_batches(tree.events)
    decode_rb: Optional[dict] = None
    prefill_rb: Optional[dict] = None
    mtp_markers = _detect_mtp_markers(tree.events)

    # SGLang EAGLE/MTP: KernelBreakdown verify + rmsnorm step selection (not vLLM).
    if mtp_markers and run_batches:
        mtp_sections_full = _detect_mtp_sections(tree.events)
        if mtp_sections_full:
            decode_rb, verify_ranges = _select_sglang_mtp_verify_decode_step(
                tree,
                run_batches,
                phase_map,
                mtp_sections_full,
            )
            if decode_rb is not None:
                tree._inference_ext_sglang_mtp = True
                tree._inference_ext_verify_ranges = verify_ranges
                prefill_rb = _select_sglang_mtp_prefill_step(
                    tree,
                    run_batches,
                    phase_map,
                )

    # Primary: annotation-based selection (vLLM and non-MTP traces)
    annotation_roots = _find_annotation_roots(tree.events)
    if decode_rb is None and annotation_roots:
        print(f"[inference_ext] Found {len(annotation_roots)} iteration annotations")
        decode_rb = _find_best_step_annotation(tree, annotation_roots, run_batches)

        # When annotations are used directly (no run_batches), narrow the
        # step boundaries to ``gpu_model_runner: forward`` sub-annotations
        # (which contain only the model forward pass, not preprocess/sample).
        # Also look for the best prefill annotation preceding decode.
        if decode_rb is not None and not run_batches:
            fwd_ann = _find_forward_annotation_within(tree.events, decode_rb)
            if fwd_ann is not None:
                print(
                    f"[inference_ext] Narrowed decode to 'gpu_model_runner: forward' "
                    f"(ts={fwd_ann.get('ts', 0):.0f}, "
                    f"dur={fwd_ann.get('dur', 0):.0f})"
                )
                decode_rb = fwd_ann

            decode_ts = decode_rb.get("ts", 0)
            prefill_anns = []
            for ann in annotation_roots:
                det = _get_iter_details_from_name(ann.get("name", ""))
                cr = det.get("context_requests", 0)
                ann_ts = ann.get("ts", 0)
                if ann_ts < decode_ts and cr > 0:
                    prefill_anns.append(ann)
            if prefill_anns:
                prefill_anns.sort(key=lambda e: e.get("ts", 0))
                prefill_rb = prefill_anns[-1]
                # Narrow prefill to its forward annotation too
                prefill_fwd = _find_forward_annotation_within(tree.events, prefill_rb)
                if prefill_fwd is not None:
                    print(
                        f"[inference_ext] Narrowed prefill to 'gpu_model_runner: forward' "
                        f"(ts={prefill_fwd.get('ts', 0):.0f}, "
                        f"dur={prefill_fwd.get('dur', 0):.0f})"
                    )
                    prefill_rb = prefill_fwd
                else:
                    print(
                        f"[inference_ext] Annotation path: found preceding prefill "
                        f"annotation (ts={prefill_rb.get('ts', 0):.0f}, "
                        f"dur={prefill_rb.get('dur', 0):.0f})"
                    )

    # Fallback: region-based cpu_op batch-size scoring (non-MTP SGLang, etc.)
    if decode_rb is None:
        if not annotation_roots:
            print(
                "[inference_ext] No vLLM/SGLang iteration annotations found. "
                "Trying cpu_op batch-size region selection."
            )
        else:
            print(
                "[inference_ext] Could not map annotations to run_batch events. "
                "Trying cpu_op batch-size region selection."
            )

        if phase_map and run_batches:
            decode_rb, prefill_rb = _select_steps_from_region(
                tree,
                run_batches,
                phase_map,
            )
        elif run_batches:
            decode_rb = _find_best_step_batch_size(
                tree,
                tree.events,
                run_batches,
            )

    if decode_rb is None:
        print("[inference_ext] Could not determine a representative step.")
        return

    # Determine the real phase of the primary selected step.  When the
    # step is actually a prefill or mixed (prefilldecode) step that landed
    # in ``decode_rb`` via the fallback path, we must label it correctly.
    _primary_phase = phase_map.get(decode_rb.get("UID"), "")
    if not _primary_phase:
        # Annotation-based path: infer from the annotation name.
        det = _get_iter_details_from_name(decode_rb.get("name", ""))
        cr = det.get("context_requests", 0)
        gr = det.get("generation_requests", 0)
        if cr > 0 and gr > 0:
            _primary_phase = "prefilldecode"
        elif cr > 0:
            _primary_phase = "prefill"
        else:
            _primary_phase = "decode"

    # Collect subtrees for selected steps
    decode_uids = _collect_subtree_uids(tree, decode_rb)
    if _is_sglang_mtp(tree):
        verify_ranges = getattr(tree, "_inference_ext_verify_ranges", [])
        decode_uids = _filter_subtree_uids_to_ranges(tree, decode_uids, verify_ranges)
    prefill_uids: set = set()
    if prefill_rb is not None:
        prefill_uids = _collect_subtree_uids(tree, prefill_rb)

    keep_uids = decode_uids | prefill_uids

    original_count = len(tree.events)
    kept = [e for e in tree.events if e.get("UID") in keep_uids]

    tree.events = kept
    tree.events_by_uid = {
        event.get("UID", i): event for i, event in enumerate(tree.events)
    }

    # Prune tree metadata so downstream traversals don't reference
    # UIDs that were filtered out.
    if hasattr(tree, "cpu_root_nodes"):
        tree.cpu_root_nodes = [uid for uid in tree.cpu_root_nodes if uid in keep_uids]
    for ev in tree.events:
        if "children" in ev:
            ev["children"] = [c for c in ev["children"] if c in keep_uids]
        if "gpu_events" in ev:
            ev["gpu_events"] = [u for u in ev["gpu_events"] if u in keep_uids]
        if "parent" in ev and ev["parent"] not in keep_uids:
            del ev["parent"]

    # Tag every event with step_phase.  When both a decode and prefill step
    # were selected, events are tagged by which subtree they belong to.
    # When only a single step was selected (no prefill_rb), use the real
    # phase of that step instead of hardcoding "decode".
    for ev in tree.events:
        uid = ev.get("UID")
        if uid in prefill_uids:
            ev["step_phase"] = "prefill"
        elif uid in decode_uids:
            ev["step_phase"] = _primary_phase if prefill_rb is None else "decode"
        else:
            ev["step_phase"] = ""

    phase_counts = Counter(ev.get("step_phase", "") for ev in tree.events)
    phase_msg = ", ".join(
        f"{cnt} {ph}" for ph, cnt in sorted(phase_counts.items()) if ph
    )

    print(
        f"[inference_ext] Kept {len(kept)} / {original_count} events "
        f"({100 * len(kept) / max(1, original_count):.1f}%) "
        f"from run_batch subtree(s)"
    )
    if prefill_rb is not None or _primary_phase != "decode":
        print(f"[inference_ext] Step phases: {phase_msg}")

    # --- MTP section detection & tagging ---
    mtp_sections = _detect_mtp_sections(tree.events)
    if mtp_sections:
        _tag_decode_sections(tree.events, mtp_sections)
        section_names = [s[0] for s in mtp_sections]
        print(f"[inference_ext] Detected MTP sections: {section_names}")
        if _is_sglang_mtp(tree):
            print(
                "[inference_ext] SGLang MTP mode: decode step pruned to "
                "verify section (prefill/decode region logic skipped)."
            )
    else:
        for event in tree.events:
            event["decode_section"] = ""


# ---------------------------------------------------------------------------
# Extension hook 2: get_additional_dataframes_extension
# ---------------------------------------------------------------------------
def get_additional_dataframes_extension(tree) -> Dict[str, pd.DataFrame]:
    """Build a kernel timeline DataFrame listing every GPU kernel
    in chronological order, enriched with layer decomposition info.

    Called by the report generator after all standard sheets are built.
    """
    additional_dfs: Dict[str, pd.DataFrame] = {}

    kernel_events = [e for e in tree.events if e.get("cat") in GPU_CATEGORIES]

    if not kernel_events:
        print("[inference_ext] No GPU kernel events found for timeline sheet.")
        return additional_dfs

    kernel_events.sort(key=lambda e: e.get("ts", 0))

    # Run layer decomposition
    layer_decomp: Optional[LayerDecomposition] = _decompose_layers(tree)

    rows = []
    for event in kernel_events:
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)
        args = event.get("args", {})

        row = {
            "Kernel name": event.get("name", ""),
            "Category": event.get("cat", ""),
            "Timestamp (us)": ts,
            "Duration (us)": dur,
            "Stream": args.get("stream", ""),
        }

        # Walk up the tree to find the parent cpu_op and nn.Module
        parent_cpu_op = _find_ancestor_by_category(tree, event, "cpu_op")
        if parent_cpu_op:
            row["Parent cpu_op"] = parent_cpu_op.get("name", "")
            row["Parent cpu_op UID"] = parent_cpu_op.get("UID", "")
            row["Input Dims"] = str(parent_cpu_op.get("args", {}).get("Input Dims", ""))
        else:
            row["Parent cpu_op"] = ""
            row["Parent cpu_op UID"] = ""
            row["Input Dims"] = ""

        parent_module = _find_ancestor_by_category(tree, event, "python_function")
        if parent_module and parent_module.get("name", "").startswith("nn.Module:"):
            row["nn.Module"] = parent_module.get("name", "").replace(
                _NN_MODULE_PREFIX, ""
            )
        else:
            row["nn.Module"] = ""

        row["step_phase"] = event.get("step_phase", "")
        row["decode_section"] = event.get("decode_section", "")

        # Layer decomposition columns
        uid = event.get("UID")
        if layer_decomp and uid in layer_decomp.kernel_to_layer:
            layer_idx = int(layer_decomp.kernel_to_layer[uid])
            step_phase = event.get("step_phase", "")
            layer_info = next(
                (
                    l
                    for l in layer_decomp.layers
                    if l.layer_idx == layer_idx and l.step_phase == step_phase
                ),
                None,
            )
            row["decoder_layer"] = layer_idx
            row["layer_kind"] = layer_info.ff_kind if layer_info else ""
            row["template_id"] = layer_info.template_id if layer_info else ""
            row["arch_type"] = layer_decomp.layer_to_arch.get(
                (step_phase, layer_idx), ""
            )
            cpu_op_name = row.get("Parent cpu_op", "")
            row["layer_token"] = _canonical_token(cpu_op_name, event.get("name", ""))
        else:
            row["decoder_layer"] = ""
            row["layer_kind"] = ""
            row["template_id"] = ""
            row["arch_type"] = ""
            row["layer_token"] = ""

        row["UID"] = uid
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Apply CLI scope filters ---
    pre_filter_count = len(df)

    if _EXT_ARGS.mtp_scope:
        scope = _EXT_ARGS.mtp_scope
        df = df[df["decode_section"] == scope].copy()
        print(
            f"[inference_ext] --mtp_scope={scope!r}: "
            f"{len(df)} / {pre_filter_count} kernels kept"
        )

    sglang_mtp = _is_sglang_mtp(tree)
    if sglang_mtp and not _EXT_ARGS.mtp_scope:
        df = df[
            (df["step_phase"] != "decode") | (df["decode_section"] == "verify")
        ].copy()
        print(
            f"[inference_ext] SGLang MTP: kept verify decode + prefill kernels "
            f"({len(df)} / {pre_filter_count})"
        )
        pre_filter_count = len(df)

    # SGLang export: single tab for lowest-index MoE layer (KernelBreakdown parity).
    if sglang_mtp and not _EXT_ARGS.moe_only and _EXT_ARGS.first_n_layers <= 0:
        df, layer_decomp = _apply_sglang_mtp_first_moe_export_scope(
            df,
            layer_decomp,
            tree,
        )
        pre_filter_count = len(df)

    if _EXT_ARGS.moe_only:
        if sglang_mtp or _EXT_ARGS.mtp_scope == "verify":
            moe_layers = _decoder_layers_with_moe_kernel_name(df)
            if moe_layers:
                df = df[df["decoder_layer"].isin(moe_layers)].copy()
            else:
                df = df[df["layer_kind"] == "moe"].copy()
        else:
            df = df[df["layer_kind"] == "moe"].copy()
        print(
            f"[inference_ext] --moe_only: "
            f"{len(df)} / {pre_filter_count} kernels kept"
        )

    if _EXT_ARGS.first_n_layers > 0:
        use_moe_kernel_rule = (
            sglang_mtp or _EXT_ARGS.moe_only or _EXT_ARGS.mtp_scope == "verify"
        )
        if use_moe_kernel_rule:
            moe_layers = _decoder_layers_with_moe_kernel_name(df)
            keep_layers = set(moe_layers[: _EXT_ARGS.first_n_layers])
        else:
            unique_layers = sorted(
                df["decoder_layer"].dropna().unique(),
                key=_decoder_layer_sort_key,
            )
            unique_layers = [l for l in unique_layers if l != ""]
            keep_layers = set(unique_layers[: _EXT_ARGS.first_n_layers])
        df = df[df["decoder_layer"].isin(keep_layers)].copy()
        print(
            f"[inference_ext] --first_n_layers={_EXT_ARGS.first_n_layers}: "
            f"keeping layers {sorted(keep_layers)}, "
            f"{len(df)} / {pre_filter_count} kernels kept"
        )

    # --- Decide tab layout: layer-named tabs vs phase-only vs single ---
    use_arch_tabs = layer_decomp is not None and bool(layer_decomp.arch_reps)
    phases = [p for p in df["step_phase"].unique() if p]

    if use_arch_tabs:
        arch_counts = (
            df.groupby(["step_phase", "arch_type"]).size().to_dict()
            if "arch_type" in df.columns
            else {}
        )
        print(f"[inference_ext] Kernel arch_type counts: {arch_counts}")
        # One tab per (step_phase, arch) representative layer.
        for (phase, arch_name), rep_layer_idx in sorted(
            layer_decomp.arch_reps.items(), key=lambda x: (x[0][0], x[0][1])
        ):
            layer_col = pd.to_numeric(df["decoder_layer"], errors="coerce")
            arch_df = df[
                (df["step_phase"] == phase)
                & (df["arch_type"] == arch_name)
                & (layer_col == rep_layer_idx)
            ].copy()
            if arch_df.empty:
                n_arch = int(
                    ((df["step_phase"] == phase) & (df["arch_type"] == arch_name)).sum()
                )
                n_layer = int(
                    ((df["step_phase"] == phase) & (layer_col == rep_layer_idx)).sum()
                )
                tab_preview = _excel_arch_tab_name(arch_name, rep_layer_idx, phase)
                print(
                    f"[inference_ext] Tab {tab_preview} ({phase}): "
                    f"0 kernels (arch_type matches={n_arch}, layer matches={n_layer})"
                )
                continue
            tab_name = _excel_arch_tab_name(arch_name, rep_layer_idx, phase)
            additional_dfs[tab_name] = arch_df
            print(
                f"[inference_ext] Built tab {tab_name!r}: "
                f"{len(arch_df)} kernels ({phase})"
            )
    elif len(phases) > 1:
        for phase in sorted(phases):
            phase_df = df[df["step_phase"] == phase].copy()
            additional_dfs[phase] = phase_df
        phase_counts = ", ".join(
            f"{len(df[df['step_phase'] == p])} {p}" for p in sorted(phases)
        )
        print(f"[inference_ext] Built per-phase sheets: {phase_counts}")
    elif len(phases) == 1:
        additional_dfs[phases[0]] = df
        print(f"[inference_ext] Built {phases[0]} sheet: " f"{len(df)} kernels")
    else:
        additional_dfs["kernel_timeline"] = df
        print(f"[inference_ext] Built kernel_timeline sheet: " f"{len(df)} kernels")

    # --- Optional MTP section summary sheet ---
    has_mtp = df["decode_section"].astype(bool).any()
    if has_mtp:
        total_step_dur = df["Duration (us)"].sum()
        summary_rows = []
        for section_name in df["decode_section"].unique():
            if not section_name:
                continue
            sec_df = df[df["decode_section"] == section_name]
            sec_dur = sec_df["Duration (us)"].sum()
            summary_rows.append(
                {
                    "Section": section_name,
                    "Kernel count": len(sec_df),
                    "Total kernel duration (us)": sec_dur,
                    "% of step duration": (
                        100.0 * sec_dur / total_step_dur if total_step_dur else 0.0
                    ),
                }
            )
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.sort_values(
                "Total kernel duration (us)", ascending=False, inplace=True
            )
            additional_dfs["mtp_section_summary"] = summary_df
            print(
                f"[inference_ext] Built mtp_section_summary sheet: "
                f"{len(summary_df)} sections"
            )

    # --- Layer summary sheet ---
    if layer_decomp and layer_decomp.layers:
        # Only summarize layers that survived scope filtering
        kept_layer_idxs = set(df["decoder_layer"].dropna().unique()) - {""}
        filtered_layers = [
            li for li in layer_decomp.layers if li.layer_idx in kept_layer_idxs
        ]

        total_step_kernel_dur = df["Duration (us)"].sum()
        layer_rows = []
        for layer_info in filtered_layers:
            pct = (
                100.0 * layer_info.total_kernel_dur_us / total_step_kernel_dur
                if total_step_kernel_dur
                else 0.0
            )
            layer_rows.append(
                {
                    "step_phase": layer_info.step_phase,
                    "layer_idx": layer_info.layer_idx,
                    "layer_name": layer_info.layer_name,
                    "layer_kind": layer_info.ff_kind,
                    "template_id": layer_info.template_id,
                    "kernel_count": layer_info.kernel_count,
                    "total_dur_us": round(layer_info.total_kernel_dur_us, 2),
                    "pct_of_step": round(pct, 2),
                    "token_sequence": " | ".join(layer_info.tokens_compact),
                }
            )
        layer_summary_df = pd.DataFrame(layer_rows)
        additional_dfs["layer_summary"] = layer_summary_df
        print(
            f"[inference_ext] Built layer_summary sheet: "
            f"{len(layer_summary_df)} layers"
        )

    return additional_dfs


def _find_ancestor_by_category(
    tree, event: dict, target_category: str
) -> Optional[dict]:
    """Walk up the parent chain to find the nearest ancestor with the
    given event category."""
    current = event
    for _ in range(100):  # guard against infinite loops
        parent = tree.get_parent_event(current)
        if parent is None:
            return None
        cat = parent.get("cat", "")
        name = parent.get("name", "")
        if target_category == "python_function":
            if cat == "python_function" and name.startswith("nn.Module:"):
                return parent
        elif cat == target_category:
            return parent
        current = parent
    return None
