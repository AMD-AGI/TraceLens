###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Cumulative projection chart (stacked bars: Baseline → Projection [→ Target]).

Ported from AgenticMode/Comparative/Analysis/plotting_manual.py for reuse in
Standalone comparative reports (TraceDiff-enriched perf CSVs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CumulativeProjectionChart:
    """Generate cumulative optimization projection charts from operation data."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_chart(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        time_column: str = "total_direct_kernel_time_ms",
        category_column: str = "op category",
        baseline_label: str = "BASELINE",
        projection_label: str = "PROJECTION",
        target_label: str = "TARGET",
        include_target_bar: bool = False,
        filename: str = "cumulative_projection.png",
    ) -> Optional[Path]:
        try:
            baseline_categories = self._aggregate_by_category(
                baseline_df, time_column, category_column
            )
            target_categories = self._aggregate_by_category(
                target_df, time_column, category_column
            )

            if not baseline_categories or not target_categories:
                print("⚠️  No category data available")
                return None

            if time_column == "total_direct_kernel_time_sum":
                baseline_categories = {
                    k: v / 1000 for k, v in baseline_categories.items()
                }
                target_categories = {k: v / 1000 for k, v in target_categories.items()}

            projection_categories = self._calculate_projection(
                baseline_categories, target_categories
            )

            baseline_total = sum(baseline_categories.values())
            projection_total = sum(projection_categories.values())
            target_total = sum(target_categories.values())

            print(f"\n📊 Chart data:")
            print(
                f"  Baseline:   {baseline_total:.2f}ms ({len(baseline_categories)} categories)"
            )
            print(
                f"  Projection: {projection_total:.2f}ms ({len(projection_categories)} categories)"
            )
            print(
                f"  Target:     {target_total:.2f}ms ({len(target_categories)} categories)"
            )

            plot_path = self._create_chart(
                baseline_categories=baseline_categories,
                projection_categories=projection_categories,
                target_categories=target_categories if include_target_bar else None,
                baseline_total=baseline_total,
                projection_total=projection_total,
                target_total=target_total,
                baseline_label=baseline_label,
                projection_label=projection_label,
                target_label=target_label,
                filename=filename,
            )

            return plot_path

        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _aggregate_by_category(
        self, df: pd.DataFrame, time_column: str, category_column: str
    ) -> Dict[str, float]:
        if df is None or df.empty:
            return {}

        if time_column not in df.columns:
            print(
                f"⚠️  Time column '{time_column}' not found. Available: {df.columns.tolist()}"
            )
            return {}

        if category_column not in df.columns:
            print(
                f"⚠️  Category column '{category_column}' not found. Available: {df.columns.tolist()}"
            )
            return {}

        df_copy = df.copy()
        category_dict: Dict[str, float] = {}
        grouped = df_copy.groupby(category_column)[time_column].sum()

        for category, total_time in grouped.items():
            if pd.notna(total_time):
                category_dict[str(category)] = float(total_time)

        print(
            f"  Aggregated {len(category_dict)} categories from {len(df)} operations"
        )
        return category_dict

    def _calculate_projection(
        self, baseline_categories: Dict[str, float], target_categories: Dict[str, float]
    ) -> Dict[str, float]:
        """Per category, ``min(baseline_ms, target_ms)`` with missing side as 0.

        Aligns with “take the better of the two runs per category.” Categories
        where baseline is 0 but target has time contribute 0 to projection (no
        baseline cost to replace), not the full target time.
        """
        projection: Dict[str, float] = {}
        all_categories = set(baseline_categories.keys()) | set(target_categories.keys())
        for category in all_categories:
            b = float(baseline_categories.get(category, 0) or 0)
            t = float(target_categories.get(category, 0) or 0)
            projection[category] = min(b, t)
        return projection

    def _create_chart(
        self,
        baseline_categories: Dict[str, float],
        projection_categories: Dict[str, float],
        target_categories: Optional[Dict[str, float]],
        baseline_total: float,
        projection_total: float,
        target_total: float,
        baseline_label: str,
        projection_label: str,
        target_label: str,
        filename: str,
    ) -> Path:
        all_categories = sorted(
            set(baseline_categories.keys()) | set(projection_categories.keys())
        )

        if target_categories is not None:
            all_categories = sorted(
                set(all_categories) | set(target_categories.keys())
            )
            states = [baseline_label, projection_label, target_label]
            data_dict = {
                baseline_label: baseline_categories,
                projection_label: projection_categories,
                target_label: target_categories,
            }
            totals = [baseline_total, projection_total, target_total]
        else:
            states = [baseline_label, projection_label]
            data_dict = {
                baseline_label: baseline_categories,
                projection_label: projection_categories,
            }
            totals = [baseline_total, projection_total]

        fig, ax = plt.subplots(figsize=(15, 10))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        bottom_values = np.zeros(len(states))
        bar_width = 0.45
        x_pos = np.arange(len(states))

        cat_colors = {}
        for i, cat in enumerate(all_categories):
            cat_colors[cat] = plt.cm.tab20(i % 20)

        for cat in all_categories:
            values = [data_dict[state].get(cat, 0) for state in states]

            bars = ax.bar(
                x_pos,
                values,
                bar_width,
                label=cat,
                bottom=bottom_values,
                color=cat_colors[cat],
                edgecolor="white",
                linewidth=2.5,
                alpha=0.95,
            )

            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    height = bar.get_height()
                    y_center = bottom_values[i] + height / 2

                    fontsize = (
                        8
                        if height < max(totals) * 0.10
                        else 9 if height < max(totals) * 0.18 else 10
                    )

                    if height > max(totals) * 0.06:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y_center,
                            f"{val:.0f}",
                            ha="center",
                            va="center",
                            fontsize=fontsize,
                            fontweight="bold",
                            color="white",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="black",
                                alpha=0.65,
                            ),
                        )

            bottom_values += values

        ax.set_ylabel(
            "Total Execution Time (ms)", fontsize=14, fontweight="bold", labelpad=12
        )
        ax.set_title(
            "Cumulative Performance Optimization Projection",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)

        for i, total in enumerate(totals):
            ax.text(
                i,
                total + max(totals) * 0.008,
                f"{total:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#1a1a1a",
            )

            if i > 0:
                improvement_ms = totals[0] - total
                improvement_pct = (
                    (improvement_ms / totals[0] * 100) if totals[0] > 0 else 0
                )

                ax.text(
                    i,
                    total + max(totals) * 0.035,
                    f"↓ {improvement_pct:.1f}% ({improvement_ms:.0f}ms saved)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#27AE60",
                    style="italic",
                )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=11,
            framealpha=0.98,
            edgecolor="#cccccc",
            fancybox=True,
            shadow=True,
        )
        ax.set_ylim(0, max(totals) * 1.18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(
            str(output_path),
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"✓ Chart saved: {output_path}")
        return output_path
