###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional analysis extension: detailed plot + ms-savings.

CLI usage:

    python TraceLens/Agent/Analysis/utils/agent_extension.py \\
        --output-dir <dir> \\
        --title '<Model> on <Platform> -- Kernel Tuning Potential'
"""

import argparse
import base64
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TraceLens.Agent.Analysis.utils.plot_utils import (
    _CAT_PALETTE,
    _REST_KEY,
    _short_name,
)
from TraceLens.Agent.Analysis.utils.report_utils import (
    generate_priority_data,
)
from TraceLens.Agent.Analysis.utils.report_utils import load_manifest


class ImpactPlot:
    """Detailed 2-panel cumulative-savings plot (impact_score -> ms)."""

    def __init__(
        self,
        output_dir: str,
        title: str,
        output_filename: str = "perf_improvement.png",
        write_base64: bool = False,
        show_error_bars: bool = True,
    ):
        self.output_dir = output_dir
        self.title = title
        self.output_filename = output_filename
        self.write_base64 = write_base64
        self.show_error_bars = show_error_bars

    def run(self) -> bool:
        """Generate the figure; returns False if inputs are missing."""
        if not self._load_inputs():
            return False
        self._compute_projections()
        self._build_segments()
        self._build_color_map()

        fig, (ax_stack, ax_cone) = plt.subplots(
            1,
            2,
            figsize=(14, 5.5),
            gridspec_kw={"width_ratios": [1.3, 1.0]},
        )
        self._render_stacked_bars(ax_stack)
        fig.text(
            0.5,
            -0.02,
            "Gray bars represent categories without performance models — not reflected in savings projections.",
            ha="center",
            fontsize=8.5,
            color="#888888",
            style="italic",
        )
        self._render_throughput_cone(ax_cone)
        fig.suptitle(self.title, fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save(fig)
        return True

    def _load_inputs(self) -> bool:
        plot_data_path = os.path.join(self.output_dir, "priority_data.json")
        if not os.path.exists(plot_data_path):
            try:
                generate_priority_data(self.output_dir)
            except Exception as e:
                print(f"priority_data generation failed: {e}")
                return False

        if not os.path.exists(plot_data_path):
            print(f"priority_data.json not found at {plot_data_path} - skipping plot")
            return False

        with open(plot_data_path, "r") as f:
            plot_data = json.load(f)

        self.baseline_ms = float(plot_data.get("baseline_ms", 0))
        self.recommendations = plot_data.get("recommendations", [])
        if not self.recommendations or self.baseline_ms <= 0:
            print(
                "No kernel tuning recommendations or invalid baseline - skipping plot"
            )
            return False

        try:
            self.manifest = load_manifest(self.output_dir)
        except FileNotFoundError:
            print("category_manifest.json not found - skipping plot")
            return False
        return True

    def _compute_projections(self) -> None:
        """Accumulate per-step latency, savings, and throughput."""
        baseline_ms = self.baseline_ms
        cum_mid = cum_lo = cum_hi = 0.0
        steps = ["Baseline"]
        e2e_ms = [baseline_ms]
        savings = [0]
        rel = [100.0]
        err_lo = [0.0]
        err_hi = [0.0]
        rel_lo = [100.0]
        rel_hi = [100.0]

        for rec in self.recommendations:
            score_mid = float(rec.get("impact_score", 0))
            score_lo = float(rec.get("impact_score_low", score_mid))
            score_hi = float(rec.get("impact_score_high", score_mid))
            sm = score_mid * baseline_ms / 100.0
            sl = score_lo * baseline_ms / 100.0
            sh = score_hi * baseline_ms / 100.0
            cum_mid += sm
            cum_lo += sl
            cum_hi += sh
            lat_mid = max(0.01, baseline_ms - cum_mid)
            lat_best = max(0.01, baseline_ms - cum_hi)
            lat_worst = max(0.01, baseline_ms - cum_lo)
            steps.append(_short_name(rec["category"], max_len=20))
            e2e_ms.append(lat_mid)
            savings.append(sm)
            rel.append(round(baseline_ms / lat_mid * 100, 1))
            err_lo.append(lat_mid - lat_best)
            err_hi.append(lat_worst - lat_mid)
            rel_lo.append(
                round(baseline_ms / lat_worst * 100 if lat_worst > 0 else 100.0, 1)
            )
            rel_hi.append(
                round(baseline_ms / lat_best * 100 if lat_best > 0 else 100.0, 1)
            )

        self.proj = {
            "steps": steps,
            "e2e_ms": np.array(e2e_ms, dtype=float),
            "savings": savings,
            "rel": rel,
            "err_lo": np.array(err_lo, dtype=float),
            "err_hi": np.array(err_hi, dtype=float),
            "rel_lo": rel_lo,
            "rel_hi": rel_hi,
        }

    def _build_segments(self) -> None:
        """Build per-category segment data for the stacked bar chart."""
        plotted_cats = {r["category"] for r in self.recommendations}
        baseline_by_cat: Dict[str, float] = {}
        for cat in self.manifest["categories"]:
            if cat.get("tier") != "compute_kernel":
                continue
            gt = float(cat.get("gpu_kernel_time_ms", 0) or 0)
            if gt > 0:
                baseline_by_cat[cat["name"]] = gt

        kernel_sum = sum(baseline_by_cat.values())
        rest_e2e = max(0.0, self.baseline_ms - kernel_sum)
        if rest_e2e > 0:
            baseline_by_cat[_REST_KEY] = rest_e2e

        analyzed = sorted(
            [
                {"name": n, "time_ms": t}
                for n, t in baseline_by_cat.items()
                if n != _REST_KEY and n in plotted_cats
            ],
            key=lambda x: x["time_ms"],
        )
        unanalyzed = sorted(
            [
                {"name": n, "time_ms": t}
                for n, t in baseline_by_cat.items()
                if n != _REST_KEY and n not in plotted_cats
            ],
            key=lambda x: x["time_ms"],
        )
        segment_order = [x["name"] for x in analyzed + unanalyzed]
        if _REST_KEY in baseline_by_cat:
            segment_order.append(_REST_KEY)

        self.baseline_by_cat = baseline_by_cat
        self.segment_order = segment_order

    def _build_color_map(self) -> None:
        self.cat_color_map = {
            rec["category"]: _CAT_PALETTE[i % len(_CAT_PALETTE)]
            for i, rec in enumerate(self.recommendations)
        }

    def _render_stacked_bars(self, ax) -> None:
        """Render the left-panel stacked E2E bars with labels and error bars."""
        proj = self.proj
        recommendations = self.recommendations
        baseline_by_cat = self.baseline_by_cat
        segment_order = self.segment_order
        cat_color_map = self.cat_color_map
        show_error_bars = self.show_error_bars
        baseline_ms = self.baseline_ms

        n_bars = len(proj["steps"])
        x = np.arange(n_bars, dtype=float)
        width = 0.62
        rest_label = "Non-computing"
        rest_color = "#aab7c4"

        def _segment_heights(num_recs: int) -> Dict[str, float]:
            savings_by_cat: Dict[str, float] = {}
            for j in range(num_recs):
                r = recommendations[j]
                c = r["category"]
                sv = float(r.get("impact_score", 0)) * baseline_ms / 100.0
                if c in baseline_by_cat and c != _REST_KEY:
                    savings_by_cat[c] = savings_by_cat.get(c, 0.0) + sv
                elif _REST_KEY in baseline_by_cat:
                    savings_by_cat[_REST_KEY] = savings_by_cat.get(_REST_KEY, 0.0) + sv
            return {
                name: max(
                    0.0, baseline_by_cat.get(name, 0.0) - savings_by_cat.get(name, 0.0)
                )
                for name in segment_order
            }

        bottom = np.zeros(n_bars, dtype=float)
        for seg_name in segment_order:
            h = np.array(
                [_segment_heights(k)[seg_name] for k in range(n_bars)], dtype=float
            )
            if np.all(h <= 0):
                continue
            color = (
                rest_color
                if seg_name == _REST_KEY
                else cat_color_map.get(seg_name, "#bdc3c7")
            )
            lbl = rest_label if seg_name == _REST_KEY else _short_name(seg_name)
            ax.bar(
                x,
                h,
                width,
                bottom=bottom,
                label=lbl,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.95,
            )
            bottom += h

        for k in range(n_bars):
            total_h = float(bottom[k])
            accent = "#333333"
            if k > 0:
                cat = recommendations[k - 1].get("category", "")
                accent = cat_color_map.get(cat, "#333333")
            label_y = total_h + (proj["err_hi"][k] + 1.2 if show_error_bars else 1.2)
            ax.text(
                x[k],
                label_y,
                f"{proj['e2e_ms'][k]:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=accent,
            )
            if k > 0 and proj["savings"][k] > 0:
                err_lo_k = proj["err_lo"][k] if show_error_bars else 0.0
                text_y = total_h - err_lo_k - 1.0
                ax.text(
                    x[k],
                    text_y,
                    f"-{proj['savings'][k]:.1f} ms",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=accent,
                    fontweight="bold",
                    zorder=5,
                )
            if show_error_bars and (proj["err_lo"][k] > 0 or proj["err_hi"][k] > 0):
                ax.errorbar(
                    [x[k]],
                    [total_h],
                    yerr=[[proj["err_lo"][k]], [proj["err_hi"][k]]],
                    fmt="none",
                    ecolor=accent,
                    elinewidth=1.1,
                    capsize=3,
                    zorder=6,
                )

        ax.set_xticks(x)
        tick_colors = ["#333333"] + [
            cat_color_map.get(recommendations[k]["category"], "#333333")
            for k in range(len(recommendations))
        ]
        tick_labels = ax.set_xticklabels(proj["steps"], fontsize=9)
        for tick, color in zip(tick_labels, tick_colors):
            tick.set_color(color)
        ax.set_ylabel("E2E time stacked by category (ms)", fontsize=11)
        ax.set_title(
            "Projected E2E Latency (stacked by kernel category)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ymax = (
            float(np.max(bottom + proj["err_hi"]) + 8)
            if show_error_bars
            else float(np.max(bottom) + 8)
        )
        ax.set_ylim(0, ymax * 1.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _render_throughput_cone(self, ax) -> None:
        """Render the cumulative throughput cone on the given axes."""
        proj = self.proj
        x_vals = np.arange(len(proj["steps"]))
        cum_arr = np.array(proj["rel"], dtype=float)
        cum_lo = np.array(proj["rel_lo"], dtype=float)
        cum_hi = np.array(proj["rel_hi"], dtype=float)
        cum_lo[0] = cum_hi[0] = cum_arr[0]

        ax.fill_between(x_vals, cum_lo, cum_hi, color="#2ecc71", alpha=0.15, zorder=1)
        ax.plot(
            x_vals, cum_hi, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2
        )
        ax.plot(
            x_vals, cum_lo, "-", color="#27ae60", linewidth=1.0, alpha=0.4, zorder=2
        )
        ax.plot(
            x_vals,
            proj["rel"],
            "o-",
            color="#2ecc71",
            linewidth=2.5,
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2.5,
            zorder=4,
        )
        last_idx = len(proj["rel"]) - 1
        ax.annotate(
            f"{proj['rel'][last_idx]:.0f}%",
            (last_idx, proj["rel"][last_idx]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#1a7a3a",
            zorder=5,
        )
        ax.set_xticks(range(len(proj["steps"])))
        ax.set_xticklabels(proj["steps"], fontsize=9)
        ax.set_ylabel("% Relative Throughput (Baseline = 100)", fontsize=11)
        ax.set_title(
            "Cumulative Throughput Improvement\n(75\u2013100% roofline potential)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_ylim(80, max(cum_hi) * 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    def _save(self, fig) -> None:
        out_path = os.path.join(self.output_dir, self.output_filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Detailed impact-savings plot saved to {out_path}")

        if self.write_base64:
            with open(out_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("ascii")
            b64_path = os.path.join(
                self.output_dir, self.output_filename.replace(".png", "_base64.txt")
            )
            with open(b64_path, "w") as f:
                f.write(b64_str)
            print(f"Base64 written to {b64_path}")


class MarkdownRehydrator:
    """Rewrite marker-wrapped post-Phase-1 markdown back to legacy ms-savings form."""

    _RE_MARKER_BEGIN = re.compile(r"<!--\s*impact-begin\s+(.*?)\s*-->")
    _RE_MARKER_END = re.compile(r"<!--\s*impact-end\s*-->")
    _RE_TOP_OPS_ROW_TRAILER = re.compile(r"\s*<!--\s*top-ops-row\s+(.*?)\s*-->\s*$")

    _LEGACY_TOP_OPS_HEADER = (
        "| Rank | Category | Time (ms) | % of Compute Time | Ops | "
        "Potential improvement (time, E2E %) |\n"
        "|------|----------|-----------|-------------------|-----|"
        "-------------------------------------|"
    )

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.baseline_ms = self._load_baseline_ms(output_dir)

    def run(self) -> Dict[str, str]:
        """Walk analysis markdown files and splice ms-form text in place of markers."""
        if self.baseline_ms <= 0:
            print(
                f"[agent_extension.rehydrate_reports_to_ms] baseline_ms={self.baseline_ms!r} "
                f"in {self.output_dir}; cannot convert impact_score back to ms.",
                file=sys.stderr,
            )
            return {}

        targets: List[str] = []
        analysis = os.path.join(self.output_dir, "analysis.md")
        if os.path.isfile(analysis):
            targets.append(analysis)
        targets.extend(
            sorted(
                glob.glob(os.path.join(self.output_dir, "category_findings", "*.md"))
            )
        )
        targets.extend(
            sorted(glob.glob(os.path.join(self.output_dir, "system_findings", "*.md")))
        )

        results: Dict[str, str] = {}
        for path in targets:
            try:
                results[path] = self._rehydrate_one_file(path)
            except Exception as e:
                results[path] = f"error: {e}"

        n_rewritten = sum(1 for v in results.values() if v == "rewritten")
        n_skipped = len(results) - n_rewritten
        print(
            f"Rehydration complete: {n_rewritten} file(s) rewritten, "
            f"{n_skipped} skipped (no markers or already rehydrated)."
        )
        return results

    def _rehydrate_one_file(self, path: str) -> str:
        with open(path, "r") as f:
            original = f.read()

        out_parts: List[str] = []
        cursor = 0
        rewrote_any = False

        for begin in self._RE_MARKER_BEGIN.finditer(original):
            if begin.start() < cursor:
                continue
            end = self._RE_MARKER_END.search(original, begin.end())
            if end is None:
                break
            attrs = self._parse_attrs(begin.group(1))
            kind = attrs.pop("kind", None) or ""
            body = original[begin.end() : end.start()].strip("\n")
            rendered = self._render_legacy(kind, attrs, body, self.baseline_ms)
            if rendered is None:
                continue
            out_parts.append(original[cursor : begin.start()])
            out_parts.append(rendered)
            cursor = end.end()
            rewrote_any = True

        if not rewrote_any:
            return "skipped_no_match"

        out_parts.append(original[cursor:])
        text = "".join(out_parts)
        if text == original:
            return "skipped_no_match"

        with open(path, "w") as f:
            f.write(text)
        return "rewritten"

    @staticmethod
    def _load_baseline_ms(output_dir: str) -> float:
        try:
            manifest = load_manifest(output_dir)
        except FileNotFoundError:
            return 0.0
        return float(manifest.get("gpu_utilization", {}).get("total_time_ms", 0) or 0)

    @staticmethod
    def _parse_attrs(attr_str: str) -> Dict[str, Optional[str]]:
        """Parse ``key=value`` pairs (with optional quoting and ``null``)."""
        attrs: Dict[str, Optional[str]] = {}
        for match in re.finditer(r'(\w+)=("([^"]*)"|(\S+))', attr_str):
            key = match.group(1)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            attrs[key] = None if value == "null" else value
        return attrs

    @staticmethod
    def _attr_float(attrs: Dict[str, Optional[str]], key: str) -> Optional[float]:
        raw = attrs.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _render_p_item(attrs, body, baseline_ms):
        """Render a P-item Impact line in legacy ms-savings form."""
        lo = MarkdownRehydrator._attr_float(attrs, "low")
        hi = MarkdownRehydrator._attr_float(attrs, "high")
        if lo is None or hi is None:
            return "**Impact**: Not quantifiable from trace data"

        lo_ms = lo * baseline_ms / 100.0
        hi_ms = hi * baseline_ms / 100.0
        rendered = (
            f"**Impact**: ~{lo_ms:.1f}\u2013{hi_ms:.1f} ms savings "
            f"({lo:.1f}\u2013{hi:.1f}% of E2E)"
        )
        category = attrs.get("category")
        if category:
            rendered += (
                f" from closing efficiency gaps to 75\u2013100% of roofline "
                f"(pre-computed from `{category}_metrics.json` impact_estimates)."
            )
        return rendered

    @staticmethod
    def _render_detail_estimate(attrs, body, baseline_ms):
        """Render the two-bullet Low/High Impact estimate block in ms+E2E% form."""
        lo = MarkdownRehydrator._attr_float(attrs, "low")
        hi = MarkdownRehydrator._attr_float(attrs, "high")
        if lo is None or hi is None:
            return body
        lo_ms = lo * baseline_ms / 100.0
        hi_ms = hi * baseline_ms / 100.0
        return (
            f"- Low end (75% roofline): {lo_ms:.3f} ms savings ({lo:.2f}% E2E)\n"
            f"- High end (100% roofline): {hi_ms:.3f} ms savings ({hi:.2f}% E2E)"
        )

    @staticmethod
    def _render_top_ops_row(row: str, baseline_ms: float) -> str:
        trailer = MarkdownRehydrator._RE_TOP_OPS_ROW_TRAILER.search(row)
        if trailer is None:
            return row.rstrip() + " -- |"
        base = row[: trailer.start()].rstrip()
        attrs = MarkdownRehydrator._parse_attrs(trailer.group(1))
        lo = MarkdownRehydrator._attr_float(attrs, "low")
        hi = MarkdownRehydrator._attr_float(attrs, "high")
        if lo is None or hi is None or hi <= 0:
            opportunity = "--"
        else:
            lo_ms = lo * baseline_ms / 100.0
            hi_ms = hi * baseline_ms / 100.0
            opportunity = (
                f"~{lo_ms:.1f}\u2013{hi_ms:.1f} ms " f"({lo:.1f}\u2013{hi:.1f}%)"
            )
        return f"{base} {opportunity} |"

    @staticmethod
    def _render_top_ops(attrs, body, baseline_ms):
        """Restore the dropped Potential-improvement column on the Top Ops table."""
        lines = body.split("\n")
        out_lines: List[str] = []
        swapped_header = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not swapped_header and stripped.startswith(
                "| Rank | Category | Time (ms) | % of Compute Time | Ops |"
            ):
                out_lines.append(MarkdownRehydrator._LEGACY_TOP_OPS_HEADER)
                swapped_header = True
                i += 2
                continue
            if line.startswith("|") and line.rstrip().endswith("|"):
                out_lines.append(
                    MarkdownRehydrator._render_top_ops_row(line, baseline_ms)
                )
            elif MarkdownRehydrator._RE_TOP_OPS_ROW_TRAILER.search(line):
                out_lines.append(
                    MarkdownRehydrator._render_top_ops_row(line, baseline_ms)
                )
            else:
                out_lines.append(line)
            i += 1
        return "\n".join(out_lines)

    @staticmethod
    def _render_legacy(kind, attrs, body, baseline_ms):
        """Dispatch to the per-kind renderer; ``None`` for unknown kinds."""
        renderers = {
            "p_item": MarkdownRehydrator._render_p_item,
            "detail_estimate": MarkdownRehydrator._render_detail_estimate,
            "top_ops": MarkdownRehydrator._render_top_ops,
        }
        fn = renderers.get(kind)
        if fn is None:
            return None
        return fn(attrs, body, baseline_ms)


# Module-level shims preserved for external callers / tests.


def generate_impact_savings_plot(
    output_dir: str,
    title: str,
    output_filename: str = "perf_improvement.png",
    write_base64: bool = False,
    show_error_bars: bool = True,
) -> bool:
    return ImpactPlot(
        output_dir, title, output_filename, write_base64, show_error_bars
    ).run()


def rehydrate_reports_to_ms(output_dir: str) -> Dict[str, str]:
    return MarkdownRehydrator(output_dir).run()


_parse_attrs = MarkdownRehydrator._parse_attrs
_render_legacy = MarkdownRehydrator._render_legacy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detailed impact-savings plot + markdown rehydration "
            "(reverses the impact_score convention to pre-Phase-1 ms savings)."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Analysis output directory")
    parser.add_argument("--title", required=True, help="Plot suptitle")
    args = parser.parse_args()

    generate_impact_savings_plot(
        args.output_dir,
        args.title,
        output_filename="perf_improvement.png",
        write_base64=True,
    )
    rehydrate_reports_to_ms(args.output_dir)


if __name__ == "__main__":
    main()
