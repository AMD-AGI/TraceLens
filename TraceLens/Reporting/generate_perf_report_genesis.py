###############################################################################
# TraceLens Genesis extension — rocprof/pftrace reports for physics sim workloads
#
# Extends standard TraceLens rocprof + pftrace tools with:
#   - Steady-state GPU timeline (post-JIT simulation burst)
#   - Genesis physics kernel categorization
#   - Capture directory loading (run_combined_trace.sh output)
#   - CSV→JSON fallback when native rocprof JSON has invalid UTF-8
###############################################################################

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from TraceLens.Reporting.generate_perf_report_rocprof import generate_perf_report_rocprof
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Reporting.generate_perf_report_pftrace_memory_copy import (
    generate_perf_report_pftrace_memory_copy,
)
from TraceLens.Reporting.genesis_analysis import (
    apply_genesis_categories_to_rocprof,
    compute_steady_state_timeline,
    fix_rocprof_kernel_summary_units,
)
from TraceLens.Reporting.genesis_rocprof_util import (
    ensure_traceconv,
    infer_benchmark_window_s,
    load_capture,
    pftrace_to_json,
    resolve_profile_json,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_sheet(name: str, used: set) -> str:
    base = name[:31]
    n = 0
    while base in used:
        n += 1
        suffix = f"_{n}"
        base = name[: 31 - len(suffix)] + suffix
    used.add(base)
    return base


def write_excel(path: Path, sections: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    used: set = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for prefix, dfs in sections.items():
            for sheet, df in dfs.items():
                if df is None or df.empty:
                    continue
                # rocprof sheets use short names (no prefix); pftrace keeps prefix for clarity
                sheet_label = sheet if prefix == "rocprof" else f"{prefix}_{sheet}"
                df.to_excel(writer, sheet_name=_safe_sheet(sheet_label, used), index=False)
    logger.info("Wrote %s", path)


def _rocprof_sheets_for_excel(rocprof: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Keep only canonical rocprof sheets (no full-process timeline duplicate)."""
    keep = (
        "gpu_timeline",
        "kernel_summary",
        "kernel_summary_by_category",
        "short_kernels_summary",
        "short_kernel_histogram",
        "kernel_details",
    )
    return {k: v for k, v in rocprof.items() if k in keep and v is not None and not v.empty}


def _cleanup_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
        logger.info("Removed intermediate work dir %s", work_dir)


def write_genesis_summary_md(
    path: Path,
    capture: dict,
    reports: Dict[str, Dict[str, pd.DataFrame]],
    steady_meta: dict,
) -> None:
    lines = [
        "# TraceLens Genesis Performance Report",
        "",
        "Extension report for physics-simulation workloads (Genesis/Taichi-style kernels).",
        "",
    ]
    m = capture.get("manifest") or {}
    if m:
        lines.append(
            f"**Capture:** {m.get('timestamp', capture['capture_dir'].name)} — "
            f"n_envs={m.get('n_envs')}, steps={m.get('num_steps')}, fp{m.get('precision')}"
        )
    lines.append("")

    tl = reports.get("rocprof", {}).get("gpu_timeline")
    if tl is not None:
        lines += [
            "## GPU Timeline (timed benchmark window)",
            "",
            f"Steady-state window: `{steady_meta.get('method', 'n/a')}` — "
            f"{steady_meta.get('dispatch_count', 0):,} dispatches, "
            f"**{steady_meta.get('gpu_util_pct', 0):.1f}%** GPU busy over "
            f"{steady_meta.get('window_ms', 0):.1f} ms",
            "",
            "> Full rocprof session idle (JIT/build) is excluded from this report.",
            "",
        ]
        for _, row in tl.iterrows():
            lines.append(f"- **{row['type']}**: {row['time ms']:.2f} ms ({row['percent']:.1f}%)")
        lines.append("")

    by_cat = reports.get("rocprof", {}).get("kernel_summary_by_category")
    if by_cat is not None and not by_cat.empty:
        lines += ["## Kernel Categories", ""]
        try:
            lines.append(by_cat.to_markdown(index=False))
        except Exception:
            lines.append(by_cat.to_string(index=False))
        lines.append("")

    ks = reports.get("rocprof", {}).get("kernel_summary")
    if ks is not None:
        lines += ["## Top 10 Kernels", ""]
        lines += ["| Kernel | Count | Total (ms) | % | Category |", "|--------|-------|------------|---|----------|"]

        for _, row in ks.head(10).iterrows():
            name = row["name"] if len(row["name"]) <= 52 else row["name"][:49] + "..."
            cat = row.get("Category", "Other")
            lines.append(
                f"| `{name}` | {int(row['Count'])} | {row['Total Kernel Time (ms)']:.1f} | "
                f"{row['Percentage (%)']:.2f} | {cat} |"
            )
        lines.append("")

    hip = reports.get("pftrace_hip_activity", {}).get("hip_summary")
    if hip is not None and not hip.empty:
        lines += ["## Top HIP API (pftrace)", ""]
        try:
            lines.append(hip.head(10).to_markdown(index=False))
        except Exception:
            lines.append(hip.head(10).to_string(index=False))
        lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Wrote %s", path)


def _resolve_steady_state_fallback_s(capture: dict, cli_value: Optional[float]) -> float:
    if cli_value is not None:
        return cli_value
    inferred = infer_benchmark_window_s(capture["capture_dir"])
    if inferred is not None:
        logger.info("Auto-detected benchmark window %.2fs from run.log", inferred)
        return inferred
    return 5.0


def generate_perf_report_genesis(
    capture_dir: str,
    output_dir: str,
    include_api: bool = False,
    kernel_details: bool = False,
    short_kernel_study: bool = True,
    traceconv_path: Optional[str] = None,
    steady_state_gap_ms: float = 1000.0,
    steady_state_fallback_s: Optional[float] = None,
    keep_work: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run TraceLens rocprof + pftrace reports with Genesis-specific supplements.

    Returns dict with keys: rocprof, pftrace_hip_api, pftrace_hip_activity,
    pftrace_memory_copy, genesis (steady-state timeline, categories, ...).
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work_dir = out / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    capture = load_capture(capture_dir)
    fallback_s = _resolve_steady_state_fallback_s(capture, steady_state_fallback_s)
    profile_json = resolve_profile_json(capture, work_dir, include_api)
    traceconv = ensure_traceconv(work_dir, traceconv_path)

    reports: Dict[str, Dict[str, pd.DataFrame]] = {}
    steady_meta: dict = {}

    # Intermediate CSV dir — avoids auto-xlsx generation in upstream TraceLens
    # functions (which write an xlsx when both output paths are None).
    # Cleaned up with the rest of .work/ at the end.
    scratch_csvs = str(work_dir / "scratch_csvs")

    logger.info("TraceLens rocprof kernel report")
    reports["rocprof"] = generate_perf_report_rocprof(
        profile_json_path=str(profile_json),
        output_xlsx_path=None,
        output_csvs_dir=scratch_csvs,
        kernel_details=kernel_details,
        short_kernel_study=short_kernel_study,
    )

    if reports["rocprof"].get("kernel_summary") is not None:
        reports["rocprof"]["kernel_summary"] = fix_rocprof_kernel_summary_units(
            reports["rocprof"]["kernel_summary"]
        )
        logger.info("Applying Genesis kernel categories to rocprof summary sheets")
        apply_genesis_categories_to_rocprof(reports["rocprof"])

    rocprof_dir = Path(capture["rocprof_dir"])
    trace_csv = rocprof_dir / "kernel_kernel_trace.csv"

    if trace_csv.exists():
        logger.info("Genesis steady-state GPU timeline (fallback window %.2fs)", fallback_s)
        ss_tl, steady_meta = compute_steady_state_timeline(
            str(trace_csv),
            gap_threshold_ns=int(steady_state_gap_ms * 1e6),
            fallback_window_ns=int(fallback_s * 1e9),
        )
        steady_meta["fallback_window_s"] = fallback_s
        reports["rocprof"]["gpu_timeline"] = ss_tl

    pftrace = capture.get("pftrace")
    if pftrace is not None and Path(pftrace).exists():
        pf_json = pftrace_to_json(Path(pftrace), work_dir, traceconv)
        trace = str(pf_json)

        logger.info("TraceLens pftrace HIP activity")
        reports["pftrace_hip_activity"] = generate_perf_report_pftrace_hip_activity(
            trace_path=trace,
            output_xlsx_path=None,
            output_csvs_dir=scratch_csvs,
            output_md_path=None,
            traceconv_path=traceconv,
        )

        logger.info("TraceLens pftrace memory copy")
        try:
            reports["pftrace_memory_copy"] = generate_perf_report_pftrace_memory_copy(
                trace_path=trace,
                output_xlsx_path=None,
                output_csvs_dir=scratch_csvs,
                traceconv_path=traceconv,
            )
        except Exception as exc:
            logger.warning("Memory copy report skipped: %s", exc)

    excel_sections: Dict[str, Dict[str, pd.DataFrame]] = {
        "rocprof": _rocprof_sheets_for_excel(reports.get("rocprof", {})),
    }
    for key in ("pftrace_hip_activity", "pftrace_memory_copy"):
        if key in reports:
            excel_sections[key] = reports[key]

    write_excel(out / "genesis_perf_report.xlsx", excel_sections)
    write_genesis_summary_md(out / "genesis_summary.md", capture, reports, steady_meta)

    if not keep_work:
        _cleanup_work_dir(work_dir)

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TraceLens Genesis extension: rocprof/pftrace reports for physics sim workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  TraceLens_generate_perf_report_genesis \\
      --capture-dir profile_output/20260529_181047 \\
      --output-dir analysis_output/20260529_181047
        """,
    )
    parser.add_argument(
        "--capture-dir", "--combined-run-dir", dest="capture_dir", required=True,
        help="profile_output/<timestamp>/ from run_combined_trace.sh or run_profile.sh",
    )
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--include-api", action="store_true")
    parser.add_argument("--kernel-details", action="store_true", default=False)
    parser.add_argument("--no-short-kernel-study", action="store_false", dest="short_kernel_study")
    parser.add_argument("--traceconv", default=None)
    parser.add_argument(
        "--steady-state-gap-ms", type=float, default=1000.0,
        help="Min gap (ms) to split JIT/build from simulation burst",
    )
    parser.add_argument(
        "--steady-state-fallback-s", type=float, default=None,
        help="Timed benchmark window in seconds (default: auto from run.log wall_time, else 5s)",
    )
    parser.add_argument(
        "--keep-work", action="store_true",
        help="Keep .work/ intermediates (kernel_results.json, pftrace_events.json) for debugging",
    )
    args = parser.parse_args()

    generate_perf_report_genesis(
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        include_api=args.include_api,
        kernel_details=args.kernel_details,
        short_kernel_study=args.short_kernel_study,
        traceconv_path=args.traceconv,
        steady_state_gap_ms=args.steady_state_gap_ms,
        steady_state_fallback_s=args.steady_state_fallback_s,
        keep_work=args.keep_work,
    )

    out = Path(args.output_dir).resolve()
    print("\n=== Done ===")
    print(f"  Excel   : {out / 'genesis_perf_report.xlsx'}")
    print(f"  Summary : {out / 'genesis_summary.md'}")


if __name__ == "__main__":
    main()
