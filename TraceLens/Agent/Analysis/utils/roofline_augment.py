# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""Augment the analysis-orchestrator's compute findings with an analytical,
config-derived roofline for kernels the trace could not model.

The trace-based perf report models an op only when the trace carries its Input
Dims (``record_shapes=True``) *and* PerfModel has a formula for it. A model's
bespoke kernels -- a custom attention or fused-MoE -- therefore fall through as
"heuristic / unmodeled", and the orchestrator can only rank them by E2E
GPU-time share, with no bound classification or roofline floor.

This utility closes that gap *without* needing trace shapes: it runs TraceLens's
model-explorer roofline (:func:`...roofline_cli.run_projection`), which derives
op shapes from the model config, at the trace's runtime regime (decode/prefill,
batch, context). It then maps each modeled roofline op (Attention, MoE, GEMM,
...) onto the matching orchestrator finding by kernel name / module path, and
writes ``metadata/roofline_augmentation.json`` plus ready-to-embed Markdown
notes. Report generation (reference.md Step 12) injects each note beside its
kernel in the Detailed Analysis.

CLI::

    python -m TraceLens.Agent.Analysis.utils.roofline_augment \\
      --output-dir <orchestrator_output_dir> \\
      <checkpoint> --code-path <modeling.py> --engine vllm --tp 8 \\
      --phase decode --batch-size 32 --seq-len 1 --dim past_len=65536 \\
      --gpu-arch-json-path <platform>.json
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
    "Attention": ("rel_attention", "flash_attn", "fmha", "sdpa", "_attn_", "attention"),
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
    derives arithmetic intensity + a per-GPU roofline-time floor (the model
    declares no TP plan for many custom archs, so FLOPs/bytes are whole-model;
    dividing by ``tp`` approximates the per-GPU shard)."""
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
            acc["bound_time"][op.roofline_bound] = (
                acc["bound_time"].get(op.roofline_bound, 0.0)
                + (op.total_roofline_time_us or 0.0)
            )
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
            "roofline_us_per_gpu": acc["roofline_time_us"] / tp if tp else acc["roofline_time_us"],
            "flops": acc["flops"],
            "bytes": acc["bytes"],
            "flops_per_invocation": acc["flops_per_invocation"],
            "bytes_per_invocation": acc["bytes_per_invocation"],
            "vs_ridge": (
                None if ai is None or ridge is None else ("below" if ai < ridge else "above")
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
    for metrics_path in sorted(glob.glob(str(output_dir / "category_data" / "*_metrics.json"))):
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
            if achieved.get("ceiling") == "memory bandwidth" and achieved.get("achieved_gbps")
            else f"{achieved['achieved_tflops']:.1f} TFLOP/s"
            if achieved.get("achieved_tflops")
            else "n/a"
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


def _entry(op: dict, kernel: str, roofline_cat: str, facts: dict, ridge, arch, regime) -> dict:
    achieved = achieved_from_measurement(
        facts, op.get("time_ms"), op.get("count"), arch
    )
    return {
        "kernel": kernel,
        "finding_category": op.get("_finding_category"),
        "roofline_category": roofline_cat,
        "bound": facts["bound"],
        "arithmetic_intensity": facts["arithmetic_intensity"],
        "roofline_us_per_gpu": facts["roofline_us_per_gpu"],
        "flops": facts["flops"],
        "bytes": facts["bytes"],
        "measured_time_ms": op.get("time_ms"),
        "measured_count": op.get("count"),
        "achieved": achieved,
        "note": _note(kernel, roofline_cat, facts, ridge, achieved, regime),
    }


def augment(
    output_dir: Path,
    summary: dict[str, dict],
    regime: dict,
    ridge: float | None,
    arch: dict | None = None,
) -> dict:
    """Attach roofline facts to the dominant unmodeled trace kernel of each
    roofline category.

    Only the single highest-GPU-time heuristic/unmodeled kernel per category is
    augmented -- that is the one the orchestrator surfaces as a top finding and
    the one the analytical roofline most directly explains. Trace-modeled ops
    already carry a roofline and are skipped. When ``arch`` carries
    ``peak_tflops``/``mem_bw_gbps``, each entry also reports the achieved
    throughput (measured kernel time vs the config-derived FLOPs/bytes)."""
    picked: dict[str, dict] = {}
    for op in _dominant_ops(output_dir):  # already sorted by GPU time, desc
        if (op.get("impact_estimate_method") or op.get("estimate_method")) == "modeled":
            continue  # already roofline-modeled from the trace; nothing to add
        kernel = op.get("kernel_name") or op.get("name") or ""
        roofline_cat = classify_kernel(kernel)
        if roofline_cat is None or roofline_cat not in summary or roofline_cat in picked:
            continue
        picked[roofline_cat] = _entry(
            op, kernel, roofline_cat, summary[roofline_cat], ridge, arch, regime
        )
    entries = list(picked.values())
    payload = {
        "regime": regime,
        "ridge_flops_per_byte": ridge,
        "roofline_by_category": summary,
        "entries": entries,
    }
    meta = output_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "roofline_augmentation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def report_notes(output_dir: Path) -> dict[str, str]:
    """``{kernel_name: markdown_note}`` for report generation to inject beside
    each kernel in the Detailed Analysis. Empty if augmentation was not run."""
    path = Path(output_dir) / "metadata" / "roofline_augmentation.json"
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
        help="Orchestrator output directory (holds category_data/, metadata/).",
    )
    ns, rest = pre.parse_known_args(argv)

    # The remaining args are the model-source + runtime-regime flags shared with
    # the roofline-projection CLI (checkpoint, --engine, --tp, --phase, --dim, ...).
    proj_args = build_parser().parse_args(rest)
    try:
        projection, gpu_arch, layout_warnings = run_projection(proj_args)
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"roofline_augment: {exc}", file=sys.stderr)
        return 1
    for text in layout_warnings:
        print(f"Warning: {text}", file=sys.stderr)

    ridge = _ridge(gpu_arch)
    arch = {
        "peak_tflops": (gpu_arch.get("max_achievable_tflops") or {}).get("matrix_bf16")
        if gpu_arch
        else None,
        "mem_bw_gbps": gpu_arch.get("mem_bw_gbps") if gpu_arch else None,
    }
    summary = summarize_projection(projection, proj_args.tp, ridge)
    regime = {
        "phase": proj_args.phase,
        "batch_size": proj_args.batch_size,
        "seq_len": proj_args.seq_len,
        "dims": dict(d.split("=", 1) for d in (proj_args.dim or [])),
        "tp": proj_args.tp,
        "engine": proj_args.engine,
    }
    payload = augment(Path(ns.output_dir), summary, regime, ridge, arch)

    n = len(payload["entries"])
    print(
        f"Wrote {Path(ns.output_dir) / 'metadata' / 'roofline_augmentation.json'}: "
        f"{n} kernel(s) augmented across {len(summary)} roofline categor(ies)."
    )
    for e in payload["entries"]:
        ai = e["arithmetic_intensity"]
        print(
            f"  {e['kernel']}: {e['roofline_category']} {e['bound']} "
            f"AI={ai:.1f} FLOP/byte  ~{(e['roofline_us_per_gpu'] or 0)/1000:.2f} ms/GPU"
            if ai is not None
            else f"  {e['kernel']}: {e['roofline_category']} {e['bound']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
