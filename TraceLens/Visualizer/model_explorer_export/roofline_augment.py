###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Augment the analysis-orchestrator's compute findings with an analytical,
config-derived roofline for kernels the trace could not model.

The trace-based perf report models an op only when the trace carries its Input
Dims (``record_shapes=True``) *and* PerfModel has a formula for it. A model's
bespoke kernels -- a custom attention or fused-MoE -- therefore fall through as
"heuristic / unmodeled", and the orchestrator can only rank them by E2E
GPU-time share, with no bound classification or roofline floor.

This utility closes that gap *without* needing trace shapes. It runs TraceLens's
model-explorer roofline (:func:`...roofline_cli.run_projection`), which derives
op shapes from the model config, at one or more runtime regimes (decode/prefill,
batch, context). It maps the dominant unmodeled kernel of each roofline category
(Attention, MoE, ...) to the matching orchestrator finding by kernel name, then:

- writes ``cache/roofline_augmentation.json`` (the full record), and
- merges an additive ``roofline`` block onto the matching
  ``priority_data.json::findings[]`` entry (:func:`merge_into_priority_data`),
  leaving every existing field untouched.

The orchestrator does not invent the regime. A sub-agent writes
``cache/roofline_augment_args.json`` from the trace artifacts, and this
script projects each phase in that file. A direct CLI remains for a single
regime. Must run after Step 8.5 (``generate_priority_data``), which regenerates
``priority_data.json``.

CLI::

    python -m TraceLens.Visualizer.model_explorer_export.roofline_augment \\
      --output-dir <orchestrator_output_dir> \\
      --args-json <orchestrator_output_dir>/cache/roofline_augment_args.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

# Substrings (lowercased) in a KERNEL NAME that identify its roofline op
# category. Name-based, not module-based: a bare projection GEMM launched from
# an attention module is a GEMM, not the attention core, so matching on the
# module chain would mis-map it. A GEMM kernel maps to "GEMM" (and is dropped
# when the roofline has no standalone GEMM category, i.e. projections are folded
# into the attention/MoE boundary ops).
_CATEGORY_MATCHERS: dict[str, tuple[str, ...]] = {
    "Attention": (
        "rel_attention",
        "flash_attn",
        "flashmla",
        "fmha",
        "sdpa",
        "_attn_",
        "attention",
        # Fused latent attention. ``mla`` covers ``aiter::mla_*``,
        # ``opus_mla_v4_*`` and ``flashmla``; it is not a module-name match.
        "mla",
    ),
    "MoE": ("fused_moe", "moe_", "_moe", "grouped_gemm", "expert"),
    "GEMM": ("cijk", "tensile", "cutlass", "::mm", "addmm", "gemm"),
}


def classify_kernel(kernel_name: str | None, module_chain: Any = None) -> str | None:
    """Map a trace kernel to a roofline op category by its NAME.

    Attention/MoE win over the generic GEMM bucket. ``module_chain`` is accepted
    for signature compatibility but intentionally unused — matching on it would
    mis-map a projection GEMM (launched from an attention module) to Attention."""
    name = (kernel_name or "").lower()
    for category in ("Attention", "MoE", "GEMM"):
        if any(token in name for token in _CATEGORY_MATCHERS[category]):
            return category
    return None


def summarize_projection(projection, tp: int, ridge: float | None) -> dict[str, dict]:
    """Collapse a RooflineProjection into per-category roofline facts.

    Sums FLOPs/bytes/roofline-time across the modeled ops of each category and
    derives arithmetic intensity and the per-GPU roofline-time floor. The
    projection already costs each op on one GPU: tensor-parallel widths and
    expert-parallel tokens are applied when the op is built, and a graph leaf's
    shape is the rank-local one. ``tp`` is the degree the caller projected
    with; it is not applied again.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    by_cat: dict[str, dict] = {}
    for op in projection.ops:
        if not op.modeled or op.category is None:
            continue
        acc = by_cat.setdefault(
            op.category,
            {
                "flops": 0.0,
                "bytes": 0.0,
                "roofline_time_us": 0.0,
                "bound_time": {},
                # Per-invocation (one layer, one forward) FLOPs/bytes of the
                # category's dominant op, used to scale against a trace kernel's
                # measured per-invocation cost.
                "flops_per_invocation": 0.0,
                "bytes_per_invocation": 0.0,
                "_dominant_flops": -1.0,
            },
        )
        acc["flops"] += op.total_flops or 0.0
        acc["bytes"] += op.total_bytes or 0.0
        acc["roofline_time_us"] += op.total_roofline_time_us or 0.0
        if op.roofline_bound:
            acc["bound_time"][op.roofline_bound] = acc["bound_time"].get(
                op.roofline_bound, 0.0
            ) + (op.total_roofline_time_us or 0.0)
        if (op.total_flops or 0.0) > acc["_dominant_flops"]:
            acc["_dominant_flops"] = op.total_flops or 0.0
            acc["flops_per_invocation"] = op.flops or 0.0
            acc["bytes_per_invocation"] = op.bytes or 0.0
    out: dict[str, dict] = {}
    for cat, acc in by_cat.items():
        ai = acc["flops"] / acc["bytes"] if acc["bytes"] else None
        # The dominant-time bound classifies the category.
        bound = (
            max(acc["bound_time"], key=acc["bound_time"].get)
            if acc["bound_time"]
            else None
        )
        out[cat] = {
            "bound": bound,
            "arithmetic_intensity": ai,
            "roofline_time_us": acc["roofline_time_us"],
            "roofline_us_per_gpu": acc["roofline_time_us"],
            "flops": acc["flops"],
            "bytes": acc["bytes"],
            "flops_per_invocation": acc["flops_per_invocation"],
            "bytes_per_invocation": acc["bytes_per_invocation"],
            "vs_ridge": (
                None
                if ai is None or ridge is None
                else ("below" if ai < ridge else "above")
            ),
        }
    return out


def achieved_from_measurement(
    facts: dict, time_ms: float | None, count: int | None, arch: dict | None
) -> dict | None:
    """Combine a kernel's measured time with the analytical per-invocation
    FLOPs/bytes to report achieved throughput vs peak -- the measured side of a
    roofline that the main report would compute from trace Input Dims.

    Each of ``count`` invocations is assumed to run the projected per-invocation
    (one-layer) shape. That holds for a single-regime capture; a mixed-regime
    one (e.g. an agentic prefill+decode replay) will skew it, which the caller
    must flag."""
    if not time_ms or not count:
        return None
    fpi = facts.get("flops_per_invocation") or 0.0
    bpi = facts.get("bytes_per_invocation") or 0.0
    seconds = time_ms / 1000.0
    achieved_tflops = (fpi * count) / seconds / 1e12 if fpi else None
    achieved_gbps = (bpi * count) / seconds / 1e9 if bpi else None
    peak_tflops = (arch or {}).get("peak_tflops")
    mem_bw_gbps = (arch or {}).get("mem_bw_gbps")
    # Report utilisation against whichever ceiling the roofline bound selects.
    if facts.get("bound") == "MEMORY_BOUND":
        pct_of_peak = (
            100.0 * achieved_gbps / mem_bw_gbps
            if achieved_gbps and mem_bw_gbps
            else None
        )
        ceiling = "memory bandwidth"
    else:
        pct_of_peak = (
            100.0 * achieved_tflops / peak_tflops
            if achieved_tflops and peak_tflops
            else None
        )
        ceiling = "compute peak"
    return {
        "achieved_tflops": achieved_tflops,
        "achieved_gbps": achieved_gbps,
        "pct_of_peak": pct_of_peak,
        "ceiling": ceiling,
    }


def _dominant_ops(output_dir: Path) -> list[dict]:
    """Every op across the orchestrator's per-category metrics, tagged with its
    finding category, newest-first by GPU time."""
    ops: list[dict] = []
    for metrics_path in sorted(
        glob.glob(str(output_dir / "category_data" / "*_metrics.json"))
    ):
        category = Path(metrics_path).name.removesuffix("_metrics.json")
        try:
            data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for op in data.get("operations", []):
            if isinstance(op, dict):
                ops.append({**op, "_finding_category": category})
    ops.sort(key=lambda o: o.get("time_ms") or 0.0, reverse=True)
    return ops


def _note(
    kernel: str,
    roofline_category: str,
    facts: dict,
    ridge: float | None,
    achieved: dict | None,
    regime: dict,
) -> str:
    ai = facts.get("arithmetic_intensity")
    bound = (facts.get("bound") or "").replace("_", "-").lower() or "unclassified"
    us = facts.get("roofline_us_per_gpu")
    ridge_txt = f" vs the BF16 ridge of {ridge:.0f}" if ridge else ""
    ai_txt = f"{ai:.1f} FLOP/byte{ridge_txt}" if ai is not None else "n/a"
    us_txt = f"; ~{us/1000:.2f} ms/GPU roofline floor" if us else ""
    phase = regime.get("phase") or "the captured"
    achieved_txt = ""
    if achieved and achieved.get("pct_of_peak") is not None:
        thr = (
            f"{achieved['achieved_gbps']:.0f} GB/s"
            if achieved.get("ceiling") == "memory bandwidth"
            and achieved.get("achieved_gbps")
            else (
                f"{achieved['achieved_tflops']:.1f} TFLOP/s"
                if achieved.get("achieved_tflops")
                else "n/a"
            )
        )
        achieved_txt = (
            f" Measured throughput ≈ {thr} = **{achieved['pct_of_peak']:.1f}% of "
            f"{achieved['ceiling']}** (from the kernel's trace time against the "
            f"config-derived FLOPs/bytes; assumes every invocation runs the "
            f"projected {phase}-phase shape, so a mixed-regime capture skews it)."
        )
    return (
        f"> **Analytical roofline (model-explorer, config-derived):** the trace "
        f"carried no Input Dims for this kernel, so its shapes are taken from the "
        f"model config and costed with the same PerfModel roofline used elsewhere. "
        f"Modeled as **{roofline_category}**, it is **{bound}** at the {phase} regime "
        f"(arithmetic intensity ≈ {ai_txt}{us_txt}).{achieved_txt}"
    )


def _is_modeled(op: dict) -> bool:
    return (op.get("impact_estimate_method") or op.get("estimate_method")) == "modeled"


def _kernel_name(op: dict) -> str:
    return op.get("kernel_name") or op.get("name") or ""


def _pick_dominant(ops: list[dict], categories: set[str]) -> dict[str, dict]:
    """Highest-GPU-time unmodeled op per roofline category.

    ``ops`` is already sorted by GPU time descending, so the first match wins.
    Trace-modeled ops already carry a roofline and are skipped."""
    picked: dict[str, dict] = {}
    for op in ops:
        if _is_modeled(op):
            continue
        category = classify_kernel(_kernel_name(op))
        if category is None or category not in categories or category in picked:
            continue
        picked[category] = op
    return picked


def _phase_record(
    op: dict, kernel: str, category: str, proj: dict, *, allow_achieved: bool
) -> dict:
    facts = proj["summary"][category]
    regime = proj["regime"]
    achieved = (
        achieved_from_measurement(
            facts, op.get("time_ms"), op.get("count"), proj.get("arch")
        )
        if allow_achieved
        else None
    )
    return {
        "phase": regime.get("phase"),
        "batch_size": regime.get("batch_size"),
        "seq_len": regime.get("seq_len"),
        "bound": facts["bound"],
        "arithmetic_intensity": facts["arithmetic_intensity"],
        "roofline_us_per_gpu": facts["roofline_us_per_gpu"],
        "flops": facts["flops"],
        "bytes": facts["bytes"],
        "measured_time_ms": op.get("time_ms"),
        "measured_count": op.get("count"),
        "achieved": achieved,
        "note": _note(kernel, category, facts, proj.get("ridge"), achieved, regime),
    }


def _combined_note(category: str, phases: list[dict]) -> str:
    """One blockquote covering every projected phase.

    Achieved % is omitted: the kernel's measured time is not split by phase,
    so scoring it against one phase's shapes would mis-state the result."""
    bits = []
    for phase in phases:
        ai = phase.get("arithmetic_intensity")
        bound = (phase.get("bound") or "").replace("_", "-").lower() or "unclassified"
        us = phase.get("roofline_us_per_gpu")
        ai_txt = f"{ai:.1f} FLOP/byte" if ai is not None else "n/a"
        us_txt = f", ~{us / 1000:.2f} ms/GPU roofline floor" if us else ""
        bits.append(
            f"**{phase.get('phase') or 'phase'}** {bound} (AI ≈ {ai_txt}{us_txt})"
        )
    return (
        "> **Analytical roofline (model-explorer, config-derived):** the trace "
        "carried no Input Dims for this kernel, so its shapes are taken from the "
        "model config and costed with the same PerfModel roofline used elsewhere. "
        f"Modeled as **{category}** at each captured phase — {'; '.join(bits)}. "
        "Measured time is not split by phase, so no achieved-% is reported."
    )


def _finalize_entry(acc: dict) -> dict:
    phases = acc["phases"]
    primary = phases[0]
    multi = len(phases) > 1
    return {
        **acc,
        "bound": primary["bound"],
        "arithmetic_intensity": primary["arithmetic_intensity"],
        "roofline_us_per_gpu": primary["roofline_us_per_gpu"],
        "flops": primary["flops"],
        "bytes": primary["bytes"],
        "measured_time_ms": primary["measured_time_ms"],
        "measured_count": primary["measured_count"],
        "achieved": None if multi else primary.get("achieved"),
        "note": (
            _combined_note(acc["roofline_category"], phases)
            if multi
            else primary["note"]
        ),
    }


def augment_phases(output_dir: Path, projections: list[dict]) -> dict:
    """Attach roofline facts for one or more projected regimes.

    Each item of ``projections`` is ``{summary, regime, ridge, arch}``. Only the
    highest-GPU-time unmodeled kernel per roofline category is augmented.

    When the metrics ops carry a ``phase``, each regime is matched to the ops of
    that phase and the achieved % uses that phase's measured time. When they do
    not, every regime is attached to the same dominant kernel and the achieved %
    is reported only for a single regime — a mixed capture's measured time is
    not split by phase."""
    ops = _dominant_ops(output_dir)
    phased = any(op.get("phase") not in (None, "") for op in ops)
    grouped: dict[str, dict] = {}

    def add(op: dict, proj: dict, *, allow_achieved: bool) -> None:
        category = classify_kernel(_kernel_name(op))
        if category is None or category not in proj["summary"]:
            return
        kernel = _kernel_name(op)
        record = _phase_record(
            op, kernel, category, proj, allow_achieved=allow_achieved
        )
        acc = grouped.setdefault(
            kernel,
            {
                "kernel": kernel,
                "finding_category": op.get("_finding_category"),
                "roofline_category": category,
                "phases": [],
            },
        )
        if any(existing.get("phase") == record["phase"] for existing in acc["phases"]):
            return
        acc["phases"].append(record)

    if phased:
        for proj in projections:
            phase_name = str(proj["regime"].get("phase") or "").lower()
            phase_ops = [
                op for op in ops if str(op.get("phase") or "").lower() == phase_name
            ]
            for op in _pick_dominant(phase_ops, set(proj["summary"])).values():
                add(op, proj, allow_achieved=True)
    else:
        categories: set[str] = set()
        for proj in projections:
            categories |= set(proj["summary"])
        picked = _pick_dominant(ops, categories)
        for category, op in picked.items():
            covering = [proj for proj in projections if category in proj["summary"]]
            # Achieved % divides one measured time by one phase's FLOPs/bytes.
            # A kernel scored at two phases has no per-phase measured time.
            for proj in covering:
                add(op, proj, allow_achieved=len(covering) == 1)

    entries = [_finalize_entry(acc) for acc in grouped.values()]
    single = len(projections) == 1
    payload = {
        "regime": projections[0]["regime"] if single else None,
        "regimes": [proj["regime"] for proj in projections],
        "ridge_flops_per_byte": projections[0].get("ridge") if single else None,
        "roofline_by_category": (
            projections[0]["summary"]
            if single
            else [proj["summary"] for proj in projections]
        ),
        "entries": entries,
    }
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "roofline_augmentation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    payload["merged_findings"] = merge_into_priority_data(output_dir, entries)
    return payload


def augment(
    output_dir: Path,
    summary: dict[str, dict],
    regime: dict,
    ridge: float | None,
    arch: dict | None = None,
) -> dict:
    """Attach one regime's roofline facts to the dominant unmodeled kernel of
    each roofline category. See :func:`augment_phases`."""
    return augment_phases(
        output_dir,
        [{"summary": summary, "regime": regime, "ridge": ridge, "arch": arch}],
    )


def _roofline_block(entry: dict, phase: dict | None = None) -> dict:
    """An additive finding block. ``phase`` selects one regime of a multi-phase
    entry; otherwise the entry's own (possibly combined) note is used."""
    source = phase or entry
    block = {
        "source": "analytical-model-explorer",
        "roofline_category": entry["roofline_category"],
        "bound": source["bound"],
        "arithmetic_intensity": source["arithmetic_intensity"],
        "roofline_us_per_gpu": source["roofline_us_per_gpu"],
        "achieved": source.get("achieved"),
        "note": source["note"],
    }
    if phase is not None and phase.get("phase"):
        block["phase"] = phase["phase"]
    elif len(entry.get("phases") or []) > 1:
        block["phases"] = [
            {
                "phase": item.get("phase"),
                "bound": item["bound"],
                "arithmetic_intensity": item["arithmetic_intensity"],
                "roofline_us_per_gpu": item["roofline_us_per_gpu"],
                "achieved": item.get("achieved"),
            }
            for item in entry["phases"]
        ]
    return block


def merge_into_priority_data(output_dir: Path, entries: list[dict]) -> int:
    """Fold each roofline entry into the matching ``priority_data.json`` finding.

    Matches an entry to a finding by its kernel name appearing in a member's
    ``operation`` string. A finding that carries a ``phase`` receives only the
    matching phase; a finding with no phase receives the combined note. The
    merge attaches an additive ``roofline`` block and leaves every existing
    field untouched -- in particular ``bound_type``, ``efficiency_pct`` and
    ``estimate_method``. Report generation renders ``findings[i].roofline.note``
    verbatim. Returns the number of findings annotated."""
    path = output_dir / "priority_data.json"
    if not path.is_file() or not entries:
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    findings = data.get("findings", [])
    merged = 0
    for entry in entries:
        kernel = entry["kernel"]
        phases = entry.get("phases") or []
        for finding in findings:
            if "roofline" in finding:
                continue
            members = finding.get("members") or []
            if not any(
                kernel in str(member.get("operation", "")) for member in members
            ):
                continue
            finding_phase = finding.get("phase")
            if finding_phase and phases:
                match = next(
                    (
                        phase
                        for phase in phases
                        if str(phase.get("phase") or "").lower()
                        == str(finding_phase).lower()
                    ),
                    None,
                )
                if match is None:
                    continue
                finding["roofline"] = _roofline_block(entry, match)
            else:
                finding["roofline"] = _roofline_block(entry)
            merged += 1
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return merged


def report_notes(output_dir: Path) -> dict[str, str]:
    """``{kernel_name: markdown_note}`` for report generation to inject beside
    each kernel in the Detailed Analysis. Empty if augmentation was not run."""
    path = Path(output_dir) / "cache" / "roofline_augmentation.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {e["kernel"]: e["note"] for e in payload.get("entries", [])}


def _ridge(gpu_arch: dict | None) -> float | None:
    if not gpu_arch:
        return None
    peak = (gpu_arch.get("max_achievable_tflops") or {}).get("matrix_bf16")
    bw = gpu_arch.get("mem_bw_gbps")
    if not peak or not bw:
        return None
    return peak * 1e12 / (bw * 1e9)


_PHASES = ("prefill", "decode")


def phase_argvs(spec: dict) -> list[list[str]]:
    """One roofline-CLI argv per phase in a ``roofline_augment_args`` spec.

    Returns an empty list when ``source`` is missing or ``phases`` is empty:
    the caller skips. Raises ``ValueError`` or ``TypeError`` when a present spec
    is invalid.
    """
    source = spec.get("source")
    phases = spec.get("phases") or []
    if not source or not phases:
        return []
    if not isinstance(phases, list):
        raise TypeError("phases must be a list")
    base = [str(source)]
    if spec.get("code_path"):
        base += ["--code-path", str(spec["code_path"])]
    if spec.get("engine"):
        base += ["--engine", str(spec["engine"])]
    for key, flag in (("tp", "--tp"), ("pp", "--pp"), ("ep", "--ep")):
        if spec.get(key) is not None:
            base += [flag, str(int(spec[key]))]
    if spec.get("gpu_arch_name"):
        base += ["--gpu-arch-name", str(spec["gpu_arch_name"])]
    if spec.get("gpu_arch_json_path"):
        base += ["--gpu-arch-json-path", str(spec["gpu_arch_json_path"])]
    argvs = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise TypeError("each phase must be an object")
        name = str(phase.get("phase") or "")
        if name not in _PHASES:
            raise ValueError(f"phase must be prefill or decode, got {name!r}")
        try:
            batch = int(phase["batch_size"])
            seq = int(phase["seq_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"phase {name} needs integer batch_size and seq_len"
            ) from exc
        argv = base + [
            "--phase",
            name,
            "--batch-size",
            str(batch),
            "--seq-len",
            str(seq),
        ]
        dims = phase.get("dims") or {}
        if not isinstance(dims, dict):
            raise TypeError("phase dims must be an object")
        for key, value in dims.items():
            argv += ["--dim", f"{key}={int(value)}"]
        argvs.append(argv)
    return argvs


def _projection_record(proj_args, projection, gpu_arch) -> dict:
    ridge = _ridge(gpu_arch)
    return {
        "summary": summarize_projection(projection, proj_args.tp, ridge),
        "regime": {
            "phase": proj_args.phase,
            "batch_size": proj_args.batch_size,
            "seq_len": proj_args.seq_len,
            "dims": dict(item.split("=", 1) for item in (proj_args.dim or [])),
            "tp": proj_args.tp,
            "pp": proj_args.pp,
            "ep": proj_args.ep,
            "engine": proj_args.engine,
        },
        "ridge": ridge,
        "arch": {
            "peak_tflops": (
                (gpu_arch.get("max_achievable_tflops") or {}).get("matrix_bf16")
                if gpu_arch
                else None
            ),
            "mem_bw_gbps": gpu_arch.get("mem_bw_gbps") if gpu_arch else None,
        },
    }


def _report(output_dir: Path, payload: dict) -> None:
    categories = payload.get("roofline_by_category")
    n_categories = len(categories) if isinstance(categories, dict) else 0
    if not n_categories:
        n_categories = len({e["roofline_category"] for e in payload["entries"]})
    print(
        f"Wrote {output_dir / 'cache' / 'roofline_augmentation.json'}: "
        f"{len(payload['entries'])} kernel(s) augmented across "
        f"{n_categories} roofline categor(ies)."
    )
    for entry in payload["entries"]:
        ai = entry["arithmetic_intensity"]
        print(
            f"  {entry['kernel']}: {entry['roofline_category']} {entry['bound']} "
            f"AI={ai:.1f} FLOP/byte  "
            f"~{(entry['roofline_us_per_gpu'] or 0) / 1000:.2f} ms/GPU"
            if ai is not None
            else f"  {entry['kernel']}: {entry['roofline_category']} {entry['bound']}"
        )


def main(argv: list[str] | None = None) -> int:
    # Import here so the heavy Visualizer extra is only needed when augmenting.
    from TraceLens.Visualizer.model_explorer_export.roofline_cli import (
        build_parser,
        run_projection,
    )

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--output-dir",
        required=True,
        help="Orchestrator output directory (holds category_data/, cache/).",
    )
    pre.add_argument(
        "--args-json",
        type=Path,
        help=(
            "roofline_augment_args.json written by the roofline-augment sub-agent. "
            "Projects every phase in the file. Skips when source or phases is empty."
        ),
    )
    ns, rest = pre.parse_known_args(argv)
    if ns.args_json and rest:
        print(
            "roofline_augment: pass --args-json or projection flags, not both",
            file=sys.stderr,
        )
        return 1

    if ns.args_json:
        try:
            spec = json.loads(ns.args_json.read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise TypeError("roofline_augment_args.json must be an object")
            argvs = phase_argvs(spec)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"roofline_augment: {exc}", file=sys.stderr)
            return 1
        if not argvs:
            reason = spec.get("skipped") or "source or phases unresolved"
            print(f"roofline_augment: skipped ({reason})")
            return 0
        cli_argvs = argvs
    else:
        # The remaining args are the model-source + runtime-regime flags shared
        # with the roofline-projection CLI (checkpoint, --engine, --tp, --phase, ...).
        cli_argvs = [rest]

    projections = []
    for cli_argv in cli_argvs:
        proj_args = build_parser().parse_args(cli_argv)
        try:
            projection, gpu_arch, layout_warnings = run_projection(proj_args)
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"roofline_augment: {exc}", file=sys.stderr)
            return 1
        for text in layout_warnings:
            print(f"Warning: {text}", file=sys.stderr)
        projections.append(_projection_record(proj_args, projection, gpu_arch))

    output_dir = Path(ns.output_dir)
    payload = augment_phases(output_dir, projections)
    _report(output_dir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
