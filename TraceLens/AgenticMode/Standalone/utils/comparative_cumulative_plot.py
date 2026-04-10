###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Comparative cumulative projection plot for standalone + TraceDiff extension.

Builds per-operation Trace 1 / Trace 2 kernel times from
``perf_report_csvs/unified_perf_summary.csv``, then uses the same stacked
**Baseline → Projection** (optional **Target**) chart as
``AgenticMode/Comparative/Analysis/plotting_manual.py`` (`CumulativeProjectionChart`):

- **Baseline**: sum of Trace 1 kernel time per category (``parent_module`` or
  ``op category``).
- **Projection**: per category, ``min(baseline, target)`` — the optimistic
  “if we matched the faster side per category” stack (equivalently, per-row
  opportunity ``max(0, t1 - t2)`` aggregates consistently when comparing sums).

Embeds into markdown via ``{{COMPARATIVE_CUMULATIVE_PLOT}}``.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from TraceLens.AgenticMode.Standalone.utils.cumulative_projection_chart import (
    CumulativeProjectionChart,
)

_KERNEL_TIME_COL = "Kernel Time (µs)_sum"
_SPEEDUP_COL = "speedup (trace2/trace1)"
_DELTA_COL = "delta_us (trace2 - trace1)"
_TIME_MS_COL = "kernel_time_ms"
_PLACEHOLDER = "{{COMPARATIVE_CUMULATIVE_PLOT}}"


def _pick_agg_column(df: pd.DataFrame) -> str:
    if "parent_module" in df.columns:
        return "parent_module"
    if "op category" in df.columns:
        return "op category"
    raise ValueError(
        "unified_perf_summary needs parent_module or op category for aggregation"
    )


def _row_trace2_us(row: pd.Series) -> float:
    t1 = row.get(_KERNEL_TIME_COL)
    if t1 is None or (isinstance(t1, float) and np.isnan(t1)):
        return float("nan")
    t1 = float(t1)
    d = row.get(_DELTA_COL)
    if d is not None and not (isinstance(d, float) and np.isnan(d)):
        return t1 + float(d)
    sp = row.get(_SPEEDUP_COL)
    if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
        s = float(sp)
        if s >= 0:
            return t1 * s
    return t1


def _validate_unified_df(df: pd.DataFrame, source: str) -> None:
    if _KERNEL_TIME_COL not in df.columns:
        raise ValueError(f"Expected column {_KERNEL_TIME_COL!r} in {source}")
    if _DELTA_COL not in df.columns and _SPEEDUP_COL not in df.columns:
        raise ValueError(
            "No TraceDiff comparative columns "
            f"({_DELTA_COL} / {_SPEEDUP_COL}); generate perf report with "
            "tracediff_comparison_extension.py and --extension_args <trace2>."
        )
    _pick_agg_column(df)  # raises if no category column


def load_unified_comparative_frame(
    perf_csvs_dir: str,
) -> Tuple[pd.DataFrame, str]:
    path = os.path.join(perf_csvs_dir, "unified_perf_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing unified perf summary: {path}")
    df = pd.read_csv(path)
    _validate_unified_df(df, path)
    agg_col = _pick_agg_column(df)
    return df, agg_col


def load_unified_comparative_from_excel(
    xlsx_path: str,
    sheet_name: str = "unified_perf_summary",
) -> Tuple[pd.DataFrame, str]:
    """Load the unified perf sheet from a perf_report.xlsx (or similar)."""
    path = os.path.abspath(xlsx_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Excel report not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    _validate_unified_df(df, f"{path}::{sheet_name}")
    agg_col = _pick_agg_column(df)
    return df, agg_col


def _debug_category_projection_ms(
    df: pd.DataFrame, agg_col: str
) -> pd.DataFrame:
    """Per-category Trace1, Trace2, and projection (min) in ms — matches chart math."""
    t1 = pd.to_numeric(df[_KERNEL_TIME_COL], errors="coerce").fillna(0.0)
    t2 = df.apply(_row_trace2_us, axis=1)
    t2 = pd.to_numeric(t2, errors="coerce").fillna(t1)
    tmp = df[[agg_col]].copy()
    tmp["_t1_us"] = t1
    tmp["_t2_us"] = t2
    g1 = tmp.groupby(agg_col, dropna=False)["_t1_us"].sum() / 1000.0
    g2 = tmp.groupby(agg_col, dropna=False)["_t2_us"].sum() / 1000.0
    out = pd.DataFrame(
        {
            "baseline_ms": g1,
            "target_ms": g2,
        }
    )
    out["projection_ms"] = out[["baseline_ms", "target_ms"]].min(axis=1)
    out["gain_vs_baseline_ms"] = out["baseline_ms"] - out["projection_ms"]
    return out.sort_values("gain_vs_baseline_ms", ascending=False)


def _build_baseline_target_frames(
    df: pd.DataFrame, agg_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One row per unified_perf_summary row; same categories for baseline vs target."""
    t1 = pd.to_numeric(df[_KERNEL_TIME_COL], errors="coerce").fillna(0.0)
    t2 = df.apply(_row_trace2_us, axis=1)
    t2 = pd.to_numeric(t2, errors="coerce").fillna(t1)

    baseline_df = pd.DataFrame(
        {
            agg_col: df[agg_col],
            _TIME_MS_COL: t1 / 1000.0,
        }
    )
    target_df = pd.DataFrame(
        {
            agg_col: df[agg_col],
            _TIME_MS_COL: t2 / 1000.0,
        }
    )
    return baseline_df, target_df


def generate_comparative_cumulative_plot(
    output_dir: str,
    trace1_label: str,
    trace2_label: str,
    *,
    title: Optional[str] = None,
    output_filename: str = "comparative_cumulative_kernel_time.png",
    write_base64: bool = True,
    include_target_bar: bool = False,
    excel_path: Optional[str] = None,
    debug: bool = False,
) -> bool:
    """
    Stacked projection chart (plotting_manual-style).

    Args:
        include_target_bar: If True, third bar shows Trace 2 category totals
            (same legend/stack colors). Default False: Baseline + Projection only.
        excel_path: If set, read ``unified_perf_summary`` from this workbook instead
            of ``<output_dir>/perf_report_csvs/unified_perf_summary.csv``.
        debug: Print per-category baseline / target / projection (ms) and row counts.
    """
    try:
        if excel_path:
            df, agg_col = load_unified_comparative_from_excel(excel_path)
        else:
            perf_csvs = os.path.join(output_dir, "perf_report_csvs")
            df, agg_col = load_unified_comparative_frame(perf_csvs)
        baseline_df, target_df = _build_baseline_target_frames(df, agg_col)
    except (FileNotFoundError, ValueError) as e:
        print(f"[comparative_cumulative_plot] Skip: {e}")
        return False

    if debug:
        print(
            f"[comparative_cumulative_plot] DEBUG rows={len(df)} "
            f"agg={agg_col!r} source={'excel' if excel_path else 'csv'}"
        )
        tbl = _debug_category_projection_ms(df, agg_col)
        with pd.option_context("display.max_rows", 50, "display.width", 120):
            print(tbl.to_string())
        b_tot = tbl["baseline_ms"].sum()
        p_tot = tbl["projection_ms"].sum()
        t_tot = tbl["target_ms"].sum()
        print(
            f"[comparative_cumulative_plot] DEBUG totals (ms): "
            f"baseline={b_tot:.3f} projection={p_tot:.3f} target={t_tot:.3f} "
            f"(projection==baseline for a category iff target is slower there)"
        )

    chart = CumulativeProjectionChart(Path(output_dir))
    plot_path = chart.generate_chart(
        baseline_df=baseline_df,
        target_df=target_df,
        time_column=_TIME_MS_COL,
        category_column=agg_col,
        baseline_label=trace1_label,
        projection_label="Projection",
        target_label=trace2_label,
        include_target_bar=include_target_bar,
        filename=output_filename,
    )

    if plot_path is None:
        return False

    # Optional suptitle: reopen and add (CumulativeProjectionChart has its own title)
    if title:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        img = mpimg.imread(str(plot_path))
        fig, ax = plt.subplots(figsize=(15, 10.5))
        ax.imshow(img)
        ax.axis("off")
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.subplots_adjust(top=0.92)
        plt.savefig(
            str(plot_path),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

    print(f"[comparative_cumulative_plot] Wrote {plot_path}")

    if write_base64:
        with open(plot_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        b64_path = os.path.join(output_dir, "comparative_cumulative_base64.txt")
        with open(b64_path, "w") as f:
            f.write(b64)
        print(f"[comparative_cumulative_plot] Wrote {b64_path}")

    return True


def embed_comparative_cumulative_plot(
    output_dir: str,
    report_filename: str = "standalone_analysis.md",
    placeholder: str = _PLACEHOLDER,
) -> bool:
    report_path = os.path.join(output_dir, report_filename)
    b64_path = os.path.join(output_dir, "comparative_cumulative_base64.txt")

    if not os.path.isfile(report_path):
        print(f"[comparative_cumulative_plot] Report missing: {report_path}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    if placeholder not in text:
        return False

    if os.path.isfile(b64_path):
        with open(b64_path, "r", encoding="utf-8") as f:
            b64 = f.read().strip()
        block = (
            "### Cumulative performance projection (TraceDiff)\n\n"
            f"![Comparative cumulative projection]"
            f"(data:image/png;base64,{b64})\n"
        )
        text = text.replace(placeholder, block)
        embedded = True
    else:
        text = text.replace(placeholder, "")
        embedded = False

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)

    return embedded


def generate_and_embed_comparative_cumulative_plot(
    output_dir: str,
    trace1_label: str,
    trace2_label: str,
    title: Optional[str] = None,
    report_filename: str = "standalone_analysis.md",
    *,
    include_target_bar: bool = False,
    excel_path: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """Generate PNG + base64 and embed into ``standalone_analysis.md``."""
    out = {"plot": False, "embed": False}
    out["plot"] = generate_comparative_cumulative_plot(
        output_dir,
        trace1_label,
        trace2_label,
        title=title,
        include_target_bar=include_target_bar,
        excel_path=excel_path,
        debug=debug,
    )
    if out["plot"]:
        out["embed"] = embed_comparative_cumulative_plot(
            output_dir, report_filename=report_filename
        )
    else:
        embed_comparative_cumulative_plot(
            output_dir, report_filename=report_filename
        )
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Generate comparative cumulative projection chart from perf_report_csvs "
            "or from perf_report.xlsx (plotting_manual CumulativeProjectionChart)."
        )
    )
    p.add_argument(
        "--excel",
        metavar="XLSX",
        default=None,
        help="Read sheet unified_perf_summary from this workbook; default output dir is its folder",
    )
    p.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory for PNG/base64 (and perf_report_csvs/ when not using --excel)",
    )
    p.add_argument("trace1_label", help="Label for Trace 1 (primary profile)")
    p.add_argument("trace2_label", help="Label for Trace 2 (extension_args trace)")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-category baseline/target/projection (ms) before plotting",
    )
    p.add_argument("--title", default=None, help="Optional extra suptitle")
    p.add_argument(
        "--report",
        default="standalone_analysis.md",
        help="Markdown file under output_dir to embed",
    )
    p.add_argument(
        "--include-target",
        action="store_true",
        help="Add third stacked bar for Trace 2 category totals",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        help="Only write PNG/base64; do not modify markdown",
    )
    args = p.parse_args()
    if args.excel:
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.excel))
        excel_path = os.path.abspath(args.excel)
    else:
        if not args.output_dir:
            p.error("output_dir is required unless --excel is set")
        out_dir = args.output_dir
        excel_path = None

    ok = generate_comparative_cumulative_plot(
        out_dir,
        args.trace1_label,
        args.trace2_label,
        title=args.title,
        include_target_bar=args.include_target,
        excel_path=excel_path,
        debug=args.debug,
    )
    if ok and not args.no_embed:
        embed_comparative_cumulative_plot(out_dir, report_filename=args.report)
    elif not ok and not args.no_embed:
        embed_comparative_cumulative_plot(out_dir, report_filename=args.report)


if __name__ == "__main__":
    main()
