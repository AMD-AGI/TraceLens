#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Complete Plotting Module
Handles all chart and visualization generation for Jarvis analysis

Extracted from Reporting/jarvis_analysis.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re
from datetime import datetime

import data_extractors


class JarvisPlotter:
    """Complete implementation of all Jarvis plotting functionality"""

    def __init__(
        self,
        output_dir: Path,
        gpu1_name: str,
        gpu2_name: str,
        gpu1_data: Dict,
        gpu2_data: Dict,
        use_critical_path: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save plots
            gpu1_name: Name of GPU 1
            gpu2_name: Name of GPU 2
            gpu1_data: GPU 1 performance data
            gpu2_data: GPU 2 performance data
            use_critical_path: Whether to use critical path data
        """
        self.output_dir = output_dir
        self.gpu1_name = gpu1_name
        self.gpu2_name = gpu2_name
        self.gpu1_data = gpu1_data
        self.gpu2_data = gpu2_data
        self.use_critical_path = use_critical_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for operation categories
        self.category_colors = {
            "GEMM": "#1f77b4",
            "CONV": "#ff7f0e",
            "Convolution Operations": "#ff7f0e",
            "Batch Normalization": "#2ca02c",
            "Pooling Operations": "#d62728",
            "Flash Attention": "#9467bd",
            "Element-wise": "#e377c2",
            "NCCL": "#7f7f7f",
            "NCCL Operations": "#7f7f7f",
            "Memory": "#bcbd22",
            "Memory Operations": "#bcbd22",
            "GPU Memory": "#e377c2",
            "GPU Kernels": "#1f77b4",
            "CUDA Runtime": "#ff7f0e",
            "CPU Operations": "#8c564b",
            "Idle": "#8c564b",
            "Other": "#aec7e8",
        }

    def generate_all_plots(self) -> List[Path]:
        """
        Generate all plots for the analysis.

        Returns:
            List of paths to generated plots
        """
        plot_paths = []

        print("\n📊 Generating plots...")

        # Gap analysis plots
        gap_plots = self.generate_gap_analysis_plots()
        plot_paths.extend(gap_plots)

        # Optimization opportunity plots
        opt_plots = self.generate_optimization_opportunity_plots()
        plot_paths.extend(opt_plots)

        # Overall improvement plot - DISABLED
        # overall_plot = self.generate_overall_improvement_plot()
        # if overall_plot:
        #     plot_paths.append(overall_plot)

        return plot_paths

    def generate_gap_analysis_plots(self) -> List[Path]:
        """Generate gap analysis plots highlighting performance differences"""
        plot_paths = []

        try:
            # Determine baseline/target assignment
            baseline_data, target_data, baseline_name, target_name = (
                self._determine_baseline_target()
            )

            print(f"\n{'='*60}")
            if self.use_critical_path:
                print("CRITICAL PATH MODE: Using critical path operations")
            else:
                print("TIMELINE MODE: Using all operations from timeline")

            # Get data based on critical path flag
            baseline_cat, target_cat, data_source_label = self._get_category_data(
                baseline_data, target_data
            )
            print(f"Data source: {data_source_label}")
            print(f"{'='*60}\n")

            # Plot 1: Gap Analysis by Operation Category
            if not baseline_cat.empty and not target_cat.empty:
                plot_path = self._generate_category_gap_plot(
                    baseline_cat,
                    target_cat,
                    baseline_name,
                    target_name,
                    data_source_label,
                )
                if plot_path:
                    plot_paths.append(plot_path)

            # Plot 2: Top Operations Gap Analysis
            if "gpu_timeline" in baseline_data and "gpu_timeline" in target_data:
                plot_path = self._generate_operations_gap_plot(
                    baseline_data["gpu_timeline"],
                    target_data["gpu_timeline"],
                    baseline_name,
                    target_name,
                )
                if plot_path:
                    plot_paths.append(plot_path)

        except Exception as e:
            print(f"    ⚠️  Gap analysis plots failed: {e}")
            import traceback

            traceback.print_exc()

        return plot_paths

    def generate_optimization_opportunity_plots(self) -> List[Path]:
        """Generate plots showing estimated performance improvements"""
        plot_paths = []

        try:
            baseline_data, target_data, baseline_label, target_label = (
                self._determine_baseline_target()
            )

            # Enhanced Category Analysis with Gap and Potential Gains
            if (
                "ops_summary_by_category" in baseline_data
                and "ops_summary_by_category" in target_data
            ):
                plot_path = self._generate_category_optimization_plot(
                    baseline_data["ops_summary_by_category"],
                    target_data["ops_summary_by_category"],
                    baseline_label,
                    target_label,
                )
                if plot_path:
                    plot_paths.append(plot_path)

        except Exception as e:
            print(f"    ⚠️  Optimization opportunity plots failed: {e}")
            import traceback

            traceback.print_exc()

        return plot_paths

    def generate_overall_improvement_plot(self) -> Optional[Path]:
        """Generate overall end-to-end improvement potential chart"""
        try:
            baseline_data, target_data, baseline_label, target_label = (
                self._determine_baseline_target()
            )

            # Get total times
            baseline_timeline = baseline_data.get("gpu_timeline")
            target_timeline = target_data.get("gpu_timeline")

            if baseline_timeline is None or target_timeline is None:
                return None

            time_col = (
                "time ms" if "time ms" in baseline_timeline.columns else "duration_ms"
            )

            # Get total time from 'total_time' row if available
            if "type" in baseline_timeline.columns:
                total_time_row = baseline_timeline[
                    baseline_timeline["type"] == "total_time"
                ]
                baseline_total = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                baseline_total = 0.0

            if "type" in target_timeline.columns:
                total_time_row = target_timeline[
                    target_timeline["type"] == "total_time"
                ]
                target_total = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                target_total = 0.0

            current_gap = baseline_total - target_total

            fig, ax = plt.subplots(figsize=(14, 8))

            # Create scenarios
            scenarios = [
                "Current State",
                "25% Improvement",
                "50% Improvement",
                "75% Improvement",
                "Match Target",
            ]
            times = [
                baseline_total,
                baseline_total - (current_gap * 0.25),
                baseline_total - (current_gap * 0.5),
                baseline_total - (current_gap * 0.75),
                target_total,
            ]
            improvements = [
                0,
                current_gap * 0.25,
                current_gap * 0.5,
                current_gap * 0.75,
                current_gap,
            ]
            colors = ["#FF6B35", "#FFB84D", "#2ECC71", "#27AE60", "#1A9B5E"]

            bars = ax.bar(
                range(len(scenarios)),
                times,
                color=colors,
                alpha=0.85,
                edgecolor="black",
                linewidth=2,
            )

            # Add Target reference line
            ax.axhline(
                y=target_total,
                color="#004E89",
                linestyle="--",
                linewidth=3,
                label=f"Target Reference: {target_total:.0f}ms",
                zorder=1,
            )

            # Add annotations
            for i, (bar, time, improvement) in enumerate(
                zip(bars, times, improvements)
            ):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 300,
                    f"{time:.0f}ms",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

                if i == 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,
                        f"Current\\nGap\\n⚠️ +{current_gap:.0f}ms",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.8", facecolor="#E74C3C", alpha=0.9
                        ),
                    )
                else:
                    improvement_pct = improvement / baseline_total * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,
                        f"Gain\\n✓ -{improvement:.0f}ms\\n({improvement_pct:.1f}%)",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.7", facecolor="#27AE60", alpha=0.9
                        ),
                    )

            ax.set_ylabel("Total Execution Time (ms)", fontsize=13, fontweight="bold")
            ax.set_title(
                "Overall End-to-End Performance Improvement Potential\\nBaseline Optimization Roadmap",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
            ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(times) * 1.2)

            plt.tight_layout()
            overall_plot = self.output_dir / "baseline_target_overall_improvement.png"
            plt.savefig(str(overall_plot), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Overall improvement plot: {overall_plot}")
            return overall_plot

        except Exception as e:
            print(f"    ⚠️  Overall improvement plot failed: {e}")
            import traceback

            traceback.print_exc()

        return None

    def generate_optimization_opportunities_table(
        self, ai_analysis: Optional[str]
    ) -> Optional[Path]:
        """
        Generate optimization opportunities table with category breakdowns and key operations.
        Saves as CSV with columns: Category, Current Time (ms), Projected Time (ms),
        Potential Gain (ms), Impact (%), Key Candidate Operations
        """
        try:
            baseline_data, target_data, baseline_label, target_label = (
                self._determine_baseline_target()
            )

            baseline_timeline = baseline_data.get("gpu_timeline")
            target_timeline = target_data.get("gpu_timeline")

            if baseline_timeline is None or target_timeline is None:
                return None

            time_col = (
                "time ms" if "time ms" in baseline_timeline.columns else "duration_ms"
            )

            if "type" in baseline_timeline.columns:
                total_time_row = baseline_timeline[
                    baseline_timeline["type"] == "total_time"
                ]
                baseline_total_ms = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                baseline_total_ms = 0.0

            if "type" in target_timeline.columns:
                total_time_row = target_timeline[
                    target_timeline["type"] == "total_time"
                ]
                target_total_ms = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                target_total_ms = 0.0

            baseline_categories = self._extract_category_breakdown(
                baseline_data, baseline_total_ms
            )
            target_categories = self._extract_category_breakdown(
                target_data, target_total_ms
            )

            category_gains_ms = self._extract_optimization_gains(
                ai_analysis, baseline_categories, baseline_total_ms, target_total_ms
            )

            projection_categories = self._calculate_projection(
                baseline_categories, category_gains_ms
            )

            opportunities = []
            for category in sorted(baseline_categories.keys()):
                current_time = baseline_categories[category]
                projected_time = projection_categories.get(category, current_time)
                gain_ms = current_time - projected_time
                impact_pct = (
                    (gain_ms / baseline_total_ms * 100) if baseline_total_ms > 0 else 0
                )

                # Extract key operations from TARGET data
                extraction_category = category
                try:
                    if extraction_category:
                        ops_df = data_extractors.extract_data_by_category(
                            extraction_category, target_data, top_ops=5
                        )

                        if ops_df is not None and not ops_df.empty:
                            key_ops = "; ".join(
                                [
                                    f"{row['name']} shape:{row['shape']} ({row['Kernel Time (µs)_sum']/1000:.1f}ms)"
                                    for _, row in ops_df.iterrows()
                                ]
                            )
                        else:
                            key_ops = "Automated operation breakdown not yet supported"
                    else:
                        key_ops = "Automated operation breakdown not yet supported"
                except ValueError:
                    key_ops = "Automated operation breakdown not yet supported"

                # import pdb
                # pdb.set_trace()

                opportunities.append(
                    {
                        "Category": category,
                        "Current Time (ms)": round(current_time, 2),
                        "Projected Optimized Time (ms)": round(projected_time, 2),
                        "Potential Gain (ms)": round(gain_ms, 2),
                        "Impact (%)": round(impact_pct, 2),
                        "Key Candidate Operations": key_ops,
                    }
                )

            df = pd.DataFrame(opportunities)
            df = df.sort_values("Potential Gain (ms)", ascending=False)
            df = df.reset_index(drop=True)

            csv_path = self.output_dir / "optimization_opportunities_table.csv"
            df.to_csv(csv_path, index=False)

            print(f"\n    ✓ Optimization opportunities table: {csv_path}")
            print(f"    Total potential gain: {df['Potential Gain (ms)'].sum():.2f}ms")

            return csv_path

        except Exception as e:
            print(f"    ⚠️  Failed to generate opportunities table: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_cumulative_optimization_progression_chart(
        self, ai_analysis: Optional[str]
    ) -> Optional[Path]:
        """
        Generate cumulative optimization progression chart showing Baseline → Projection → Target.
        Shows composition of time savings across operation categories with AI-extracted gains.
        """
        try:
            baseline_data, target_data, baseline_label, target_label = (
                self._determine_baseline_target()
            )
            # Get timelines and totals

            baseline_timeline = baseline_data.get("gpu_timeline")
            target_timeline = target_data.get("gpu_timeline")

            if baseline_timeline is None or target_timeline is None:
                return None

            time_col = (
                "time ms" if "time ms" in baseline_timeline.columns else "duration_ms"
            )

            # Get total time from 'total_time' row if available
            if "type" in baseline_timeline.columns:
                total_time_row = baseline_timeline[
                    baseline_timeline["type"] == "total_time"
                ]
                baseline_total_ms = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                baseline_total_ms = 0.0

            if "type" in target_timeline.columns:
                total_time_row = target_timeline[
                    target_timeline["type"] == "total_time"
                ]
                target_total_ms = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                target_total_ms = 0.0

            # Extract category breakdowns
            baseline_categories = self._extract_category_breakdown(
                baseline_data, baseline_total_ms
            )
            target_categories = self._extract_category_breakdown(
                target_data, target_total_ms
            )

            # Extract optimization gains from AI analysis
            category_gains_ms = self._extract_optimization_gains(
                ai_analysis, baseline_categories, baseline_total_ms, target_total_ms
            )

            # Calculate projection (baseline with optimizations applied)
            projection_categories = self._calculate_projection(
                baseline_categories, category_gains_ms
            )
            projection_total_ms = sum(projection_categories.values())

            print(f"\\n  Chart data:")
            print(
                f"    Baseline: {baseline_total_ms:.2f}ms, {len(baseline_categories)} categories"
            )
            print(
                f"    Projection: {projection_total_ms:.2f}ms, {len(projection_categories)} categories"
            )
            print(
                f"    Target: {target_total_ms:.2f}ms, {len(target_categories)} categories"
            )

            # Generate the chart
            plot_path = self._generate_cumulative_chart(
                baseline_categories,
                projection_categories,
                target_categories,
                baseline_total_ms,
                projection_total_ms,
                target_total_ms,
            )

            return plot_path

        except Exception as e:
            print(f"    ⚠️  Cumulative optimization progression chart failed: {e}")
            import traceback

            traceback.print_exc()

        return None

    def generate_optimization_opportunities_chart(
        self, ai_analysis: Optional[str]
    ) -> Optional[Path]:
        """
        Generate bar chart showing optimization opportunities with priority levels.
        Extracts data from the "OPPORTUNITIES AND IMPACT" table in AI analysis.
        """
        try:
            if not ai_analysis:
                print("    ⚠️  No AI analysis provided for opportunities chart")
                return None

            baseline_data, target_data, baseline_label, target_label = (
                self._determine_baseline_target()
            )

            # Get baseline total time
            baseline_timeline = baseline_data.get("gpu_timeline")
            if baseline_timeline is None:
                return None

            time_col = (
                "time ms" if "time ms" in baseline_timeline.columns else "duration_ms"
            )
            if "type" in baseline_timeline.columns:
                total_time_row = baseline_timeline[
                    baseline_timeline["type"] == "total_time"
                ]
                baseline_total_ms = (
                    float(total_time_row[time_col].values[0])
                    if not total_time_row.empty
                    else 0.0
                )
            else:
                baseline_total_ms = 0.0

            # Extract opportunities from AI analysis
            opportunities = self._extract_opportunities_table(
                ai_analysis, baseline_total_ms
            )

            if not opportunities:
                print("    ⚠️  No opportunities found in AI analysis")
                return None

            # Generate the chart
            plot_path = self._generate_opportunities_chart(
                opportunities, baseline_total_ms
            )
            return plot_path

        except Exception as e:
            print(f"    ⚠️  Optimization opportunities chart failed: {e}")
            import traceback

            traceback.print_exc()

        return None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _determine_baseline_target(self) -> Tuple[Dict, Dict, str, str]:
        """Determine which GPU is baseline and which is target"""
        mi300_patterns = ["MI300", "mi300", "MI300X", "mi300x"]
        h200_patterns = ["H200", "h200", "GH200"]

        is_gpu1_mi300 = any(p in self.gpu1_name for p in mi300_patterns)
        is_gpu2_mi300 = any(p in self.gpu2_name for p in mi300_patterns)
        is_gpu1_h200 = any(p in self.gpu1_name for p in h200_patterns)
        is_gpu2_h200 = any(p in self.gpu2_name for p in h200_patterns)

        # Default: GPU2 as baseline, GPU1 as target
        baseline_data = self.gpu2_data
        target_data = self.gpu1_data
        baseline_name = "Baseline"
        target_name = "Target"

        # Adjust based on GPU types
        if is_gpu1_mi300:
            baseline_data = self.gpu1_data
            target_data = self.gpu2_data
        elif is_gpu2_mi300:
            baseline_data = self.gpu2_data
            target_data = self.gpu1_data
        elif is_gpu1_h200:
            baseline_data = self.gpu2_data
            target_data = self.gpu1_data
        elif is_gpu2_h200:
            baseline_data = self.gpu1_data
            target_data = self.gpu2_data

        return baseline_data, target_data, baseline_name, target_name

    def _get_category_data(
        self, baseline_data: Dict, target_data: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Get category data based on critical path flag"""
        if self.use_critical_path:
            baseline_cp_nodes = baseline_data.get("critical_path_nodes")
            target_cp_nodes = target_data.get("critical_path_nodes")

            if (
                baseline_cp_nodes is not None
                and not baseline_cp_nodes.empty
                and target_cp_nodes is not None
                and not target_cp_nodes.empty
            ):
                baseline_cat = self._extract_categories_from_cp_nodes(
                    baseline_cp_nodes, "baseline"
                )
                target_cat = self._extract_categories_from_cp_nodes(
                    target_cp_nodes, "target"
                )
                return baseline_cat, target_cat, "Critical Path"
            else:
                print(
                    "  Warning: Critical path data not available, falling back to ops_summary"
                )

        # Fallback to ops_summary_by_category
        baseline_cat = baseline_data.get("ops_summary_by_category", pd.DataFrame())
        target_cat = target_data.get("ops_summary_by_category", pd.DataFrame())
        return (
            baseline_cat,
            target_cat,
            "Timeline" if not self.use_critical_path else "Timeline (Fallback)",
        )

    def _extract_categories_from_cp_nodes(
        self, cp_nodes: pd.DataFrame, label: str
    ) -> pd.DataFrame:
        """Extract category summary from critical path nodes"""
        # This would aggregate critical path nodes by category
        # For now, return as-is (would need more implementation)
        return cp_nodes

    def _generate_category_gap_plot(
        self,
        baseline_cat: pd.DataFrame,
        target_cat: pd.DataFrame,
        baseline_name: str,
        target_name: str,
        data_source_label: str,
    ) -> Optional[Path]:
        """Generate category-level gap analysis plot"""
        try:
            if (
                "op category" not in baseline_cat.columns
                or "total_direct_kernel_time_ms" not in baseline_cat.columns
            ):
                return None

            fig, ax = plt.subplots(figsize=(14, 8))

            all_cats = sorted(
                set(
                    baseline_cat["op category"].tolist()
                    + target_cat["op category"].tolist()
                )
            )

            baseline_times = []
            target_times = []
            gaps = []

            for cat in all_cats:
                b_time = baseline_cat[baseline_cat["op category"] == cat][
                    "total_direct_kernel_time_ms"
                ].sum()
                t_time = target_cat[target_cat["op category"] == cat][
                    "total_direct_kernel_time_ms"
                ].sum()
                baseline_times.append(b_time)
                target_times.append(t_time)
                gaps.append(b_time - t_time)

            x = range(len(all_cats))
            width = 0.35

            bars1 = ax.bar(
                [i - width / 2 for i in x],
                baseline_times,
                width,
                label=baseline_name,
                color="#FF6B35",
                alpha=0.8,
            )
            bars2 = ax.bar(
                [i + width / 2 for i in x],
                target_times,
                width,
                label=target_name,
                color="#004E89",
                alpha=0.8,
            )

            # Highlight gaps
            for i, gap in enumerate(gaps):
                if gap > 0:
                    ax.text(
                        i,
                        max(baseline_times[i], target_times[i]) + 50,
                        f"⚠️ +{gap:.0f}ms",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="red",
                    )
                elif gap < 0:
                    ax.text(
                        i,
                        max(baseline_times[i], target_times[i]) + 50,
                        f"✓ {gap:.0f}ms",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="green",
                    )

            ax.set_ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
            ax.set_title(
                f"GAP Analysis with Potential Optimization Opportunities ({data_source_label} View)\\n"
                f"Baseline vs Target",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [cat[:20] + "..." if len(cat) > 20 else cat for cat in all_cats],
                rotation=45,
                ha="right",
            )
            ax.legend(fontsize=11)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            gap_plot = self.output_dir / "baseline_target_gap_analysis_categories.png"
            plt.savefig(str(gap_plot), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Gap analysis categories plot: {gap_plot}")
            return gap_plot

        except Exception as e:
            print(f"    ⚠️  Category gap plot failed: {e}")
            return None

    def _generate_operations_gap_plot(
        self,
        baseline_timeline: pd.DataFrame,
        target_timeline: pd.DataFrame,
        baseline_name: str,
        target_name: str,
    ) -> Optional[Path]:
        """Generate top operations gap analysis plot"""
        try:
            time_col = (
                "time ms" if "time ms" in baseline_timeline.columns else "duration_ms"
            )
            name_col = None
            for col in ["name", "Name", "op_name", "operation"]:
                if col in baseline_timeline.columns:
                    name_col = col
                    break

            if not name_col or not time_col:
                return None

            fig, ax = plt.subplots(figsize=(14, 8))

            # Get top operations
            baseline_top = baseline_timeline.nlargest(8, time_col)[
                [name_col, time_col]
            ].copy()
            target_top = target_timeline.nlargest(8, time_col)[
                [name_col, time_col]
            ].copy()
            all_ops = list(
                set(baseline_top[name_col].tolist() + target_top[name_col].tolist())
            )[:8]

            baseline_ops_time = []
            target_ops_time = []
            op_gaps = []

            for op in all_ops:
                b_time = baseline_timeline[baseline_timeline[name_col] == op][
                    time_col
                ].sum()
                t_time = target_timeline[target_timeline[name_col] == op][
                    time_col
                ].sum()
                baseline_ops_time.append(b_time)
                target_ops_time.append(t_time)
                op_gaps.append(b_time - t_time)

            y_pos = range(len(all_ops))
            height = 0.35

            bars1 = ax.barh(
                [i - height / 2 for i in y_pos],
                baseline_ops_time,
                height,
                label=baseline_name,
                color="#FF6B35",
                alpha=0.8,
            )
            bars2 = ax.barh(
                [i + height / 2 for i in y_pos],
                target_ops_time,
                height,
                label=target_name,
                color="#004E89",
                alpha=0.8,
            )

            # Highlight gaps
            for i, gap in enumerate(op_gaps):
                max_time = max(baseline_ops_time[i], target_ops_time[i])
                if gap > 0:
                    ax.text(
                        max_time + 20,
                        i,
                        f" ⚠️ +{gap:.0f}ms",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="red",
                    )
                elif gap < 0:
                    ax.text(
                        max_time + 20,
                        i,
                        f" ✓ {gap:.0f}ms",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="green",
                    )

            ax.set_xlabel("Execution Time (ms)", fontsize=12, fontweight="bold")
            ax.set_title(
                "Top Operations GAP Analysis\\nBaseline vs Target",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [op[:35] + "..." if len(op) > 35 else op for op in all_ops], fontsize=9
            )
            ax.legend(fontsize=11)
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            ops_plot = self.output_dir / "baseline_target_gap_analysis_operations.png"
            plt.savefig(str(ops_plot), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Gap analysis operations plot: {ops_plot}")
            return ops_plot

        except Exception as e:
            print(f"    ⚠️  Operations gap plot failed: {e}")
            return None

    def _generate_category_optimization_plot(
        self,
        baseline_cat: pd.DataFrame,
        target_cat: pd.DataFrame,
        baseline_label: str,
        target_label: str,
    ) -> Optional[Path]:
        """Generate category optimization opportunity plot"""
        try:
            if (
                "op category" not in baseline_cat.columns
                or "total_direct_kernel_time_ms" not in baseline_cat.columns
            ):
                return None

            fig, ax = plt.subplots(figsize=(16, 9))

            all_cats = sorted(
                set(
                    baseline_cat["op category"].tolist()
                    + target_cat["op category"].tolist()
                )
            )

            baseline_times = []
            target_times = []
            potential_optimized = []
            gaps = []

            for cat in all_cats:
                b_time = baseline_cat[baseline_cat["op category"] == cat][
                    "total_direct_kernel_time_ms"
                ].sum()
                t_time = target_cat[target_cat["op category"] == cat][
                    "total_direct_kernel_time_ms"
                ].sum()
                gap = b_time - t_time
                potential_opt = b_time - (gap * 0.5) if gap > 0 else b_time

                baseline_times.append(b_time)
                target_times.append(t_time)
                gaps.append(gap)
                potential_optimized.append(potential_opt)

            x = range(len(all_cats))
            width = 0.2

            bars1 = ax.bar(
                [i - 1.5 * width for i in x],
                baseline_times,
                width,
                label=f"{baseline_label} (Current)",
                color="#FF6B35",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )
            bars2 = ax.bar(
                [i - 0.5 * width for i in x],
                target_times,
                width,
                label=f"{target_label} (Reference)",
                color="#004E89",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )
            bars3 = ax.bar(
                [i + 0.5 * width for i in x],
                potential_optimized,
                width,
                label=f"{baseline_label} (50% Gap Closure)",
                color="#2ECC71",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add annotations
            for i, (b_t, t_t, gap, pot_opt) in enumerate(
                zip(baseline_times, target_times, gaps, potential_optimized)
            ):
                if gap > 0:
                    gap_y = max(b_t, t_t) + 200
                    ax.annotate(
                        "",
                        xy=(i - 1.5 * width, gap_y),
                        xytext=(i - 0.5 * width, gap_y),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=2.5),
                    )
                    ax.text(
                        i - width,
                        gap_y + 100,
                        f"GAP\\n⚠️ +{gap:.0f}ms",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="red",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7
                        ),
                    )

                improvement = b_t - pot_opt
                if improvement > 0:
                    impr_y = pot_opt + 150
                    ax.annotate(
                        "",
                        xy=(i - 1.5 * width, impr_y),
                        xytext=(i + 0.5 * width, impr_y),
                        arrowprops=dict(arrowstyle="<->", color="green", lw=2.5),
                    )
                    ax.text(
                        i - 0.5 * width,
                        impr_y + 100,
                        f"POTENTIAL GAIN\\n✓ -{improvement:.0f}ms",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="green",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7
                        ),
                    )

            ax.set_ylabel("Execution Time (ms)", fontsize=13, fontweight="bold")
            ax.set_title(
                "Category Performance Analysis: Gap Identification & Optimization Potential\\n"
                "Baseline vs Target with 50% Gap Closure Target",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [cat[:18] + "..." if len(cat) > 18 else cat for cat in all_cats],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            ax.legend(fontsize=12, loc="upper left", framealpha=0.95)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

            plt.tight_layout()
            opt_gain_plot = (
                self.output_dir / "baseline_target_category_optimization_analysis.png"
            )
            plt.savefig(str(opt_gain_plot), dpi=150, bbox_inches="tight")
            plt.close()
            print(
                f"    ✓ Enhanced category optimization analysis plot: {opt_gain_plot}"
            )
            return opt_gain_plot

        except Exception as e:
            print(f"    ⚠️  Category optimization plot failed: {e}")
            return None

    def _extract_category_breakdown(
        self, gpu_data: Dict, total_ms: float
    ) -> Dict[str, float]:
        """Extract category breakdown from GPU data"""
        categories = {}

        # Try critical path first if enabled
        # if self.use_critical_path and 'critical_path_nodes' in gpu_data:
        #     categories = self._extract_categories_from_critical_path(
        #         gpu_data['critical_path_nodes'], total_ms
        #     )
        #     if categories:
        #         return categories

        # Try ops_summary_by_category
        ops_summary = gpu_data.get("ops_summary_by_category")
        if ops_summary is not None and not ops_summary.empty:
            print(f"\n  DEBUG: Extracting categories from ops_summary_by_category")
            print(f"  Total rows in ops_summary_by_category: {len(ops_summary)}")

            for idx, row in ops_summary.iterrows():
                cat = row.get("op category", row.get("category", "Other"))
                duration = row.get(
                    "total_direct_kernel_time_ms", row.get("duration_ms", 0)
                )
                categories[cat] = float(duration)
                print(f"    Row {idx}: category='{cat}', duration={duration:.2f}ms")

            print(f"  Total categories extracted: {len(categories)}")
            print(f"  Categories: {list(categories.keys())}")
            print(f"  Total duration: {sum(categories.values()):.2f}ms")
        else:
            print(f"  DEBUG: ops_summary_by_category not found or empty")

        # # Normalize if needed
        # cat_sum = sum(categories.values())
        # if cat_sum > 0 and abs(cat_sum - total_ms) > total_ms * 0.05:
        #     print(f"  DEBUG: Normalizing categories (cat_sum={cat_sum:.2f}ms, total_ms={total_ms:.2f}ms)")
        #     scale = total_ms / cat_sum
        #     categories = {k: v * scale for k, v in categories.items()}
        #     print(f"  After normalization: {sum(categories.values()):.2f}ms")

        return categories

    def _extract_categories_from_critical_path(
        self, cp_df: pd.DataFrame, total_ms: float
    ) -> Dict[str, float]:
        """Extract categories from critical path dataframe"""
        categories = {}

        if cp_df is None or cp_df.empty:
            return categories

        # Find time column
        time_col = None
        for col in ["duration_us", "duration_ms", "time ms", "duration"]:
            if col in cp_df.columns:
                time_col = col
                break

        if not time_col:
            return categories

        # Find category/name column
        cat_col = "category" if "category" in cp_df.columns else None
        name_col = "name" if "name" in cp_df.columns else None

        # Aggregate by category
        for _, row in cp_df.iterrows():
            duration = float(row.get(time_col, 0))
            if time_col == "duration_us":
                duration = duration / 1000.0  # Convert to ms

            cat = row.get(cat_col, "Other") if cat_col else "Other"
            categories[cat] = categories.get(cat, 0) + duration

        return categories

    def _extract_optimization_gains(
        self,
        ai_analysis: Optional[str],
        baseline_categories: Dict[str, float],
        baseline_total_ms: float,
        target_total_ms: float,
    ) -> Dict[str, float]:
        """Extract optimization gains from AI analysis text by calculating from % Impact on Overall"""
        category_gains_ms = {}

        if not ai_analysis:
            print("  No AI analysis provided")
            return category_gains_ms

        lines = ai_analysis.split("\n")
        in_optimization_section = False

        print(f"\n  Extracting optimization gains from AI analysis...")
        print(f"  Baseline total time: {baseline_total_ms:.2f}ms")
        print(f"  Available baseline categories: {list(baseline_categories.keys())}")

        for line_num, line in enumerate(lines):
            # Look for OPTIMIZATION OPPORTUNITIES section (section 2)
            # Accept variations: "OPPORTUNITIES AND IMPACT" or "HIGH-IMPACT OPTIMIZATION OPPORTUNITIES"
            if (
                not in_optimization_section
                and (
                    ("OPPORTUNITIES" in line.upper() and "IMPACT" in line.upper())
                    or (
                        "HIGH-IMPACT" in line.upper() and "OPTIMIZATION" in line.upper()
                    )
                )
                and not line.strip().startswith("|")
            ):  # Not a table cell
                in_optimization_section = True
                print(
                    f"  Found optimization opportunities section at line {line_num}: {line.strip()}"
                )
                continue

            # Stop at next major section (##) but not if it still has OPTIMIZATION in it
            if in_optimization_section:
                # Debug: show every line we're processing in the section
                if line.strip():
                    print(f"  [Line {line_num}] Processing: {line[:80]}")

                if line.startswith("##"):
                    if "OPTIMIZATION" not in line.upper():
                        print(
                            f"  End of OPTIMIZATION section at line {line_num}: {line.strip()}"
                        )
                        break
                    else:
                        # Still in optimization section, just a subsection
                        print(
                            f"  Skipping optimization subsection header at line {line_num}"
                        )
                        continue

                # Parse table rows (only when we're in the section and line has pipes)
                if "|" in line and len(line.strip()) > 5:
                    parts = [p.strip() for p in line.split("|")]
                # Parse table rows (only when we're in the section and line has pipes)
                if "|" in line and len(line.strip()) > 5:
                    parts = [p.strip() for p in line.split("|")]

                    # Skip separator rows (---) and header rows
                    line_lower = line.lower()
                    is_separator = "---" in line
                    is_header = (
                        "priority" in line_lower
                        and "category" in line_lower
                        and "current time" in line_lower
                    )
                    is_cumulative_row = (
                        "cumulative impact" in line_lower or "tier" in line_lower
                    )

                    # Only process data rows
                    if (
                        len(parts) >= 7
                        and not is_separator
                        and not is_header
                        and not is_cumulative_row
                    ):
                        print(f"\n  Parsing line {line_num}: {line.strip()}")
                        print(f"    Parts ({len(parts)}): {parts}")

                        # Table format: | Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall |
                        # After split:  ['', 'Priority', 'Category', 'Current Time', 'Target Time', 'Potential Gain', '% Impact', '']
                        # Indices:        0      1          2            3                 4                    5              6        7

                        category_name = parts[2].strip() if len(parts) > 2 else ""
                        potential_gain_str = (
                            parts[5].strip() if len(parts) > 5 else ""
                        )  # Potential Gain column
                        impact_pct_str = (
                            parts[6].strip() if len(parts) > 6 else ""
                        )  # % Impact column

                        # Extract gain directly from "Potential Gain" column (this is the actual gap in ms)
                        gain_ms = 0

                        if potential_gain_str and category_name:
                            # Remove markdown formatting and "ms" suffix
                            potential_gain_str = (
                                potential_gain_str.replace("**", "")
                                .replace("*", "")
                                .replace("ms", "")
                                .replace(",", "")
                                .strip()
                            )

                            # Extract number
                            ms_match = re.search(r"(\d+(?:\.\d+)?)", potential_gain_str)
                            if ms_match:
                                gain_ms = float(ms_match.group(1))

                                # Validate it makes sense
                                if gain_ms > 0 and gain_ms < baseline_total_ms:
                                    impact_pct = (gain_ms / baseline_total_ms) * 100
                                    print(
                                        f"    Extracted Potential Gain: {gain_ms:.2f}ms"
                                    )
                                    print(
                                        f"    Calculated % Impact: {impact_pct:.2f}% (from {impact_pct_str})"
                                    )
                                else:
                                    gain_ms = 0

                        if gain_ms > 0 and category_name:
                            # Clean up category name (remove emoji and markdown, but keep underscores and hyphens)
                            # Remove: *, `, #, and emoji symbols
                            # Keep: _, - (these are part of category names like BN_bwd, CONV_fwd)
                            category_name = re.sub(
                                r"[\*\`#🎯⚡⚠️✓⚪]", "", category_name
                            ).strip()

                            # Map to actual category
                            category = self._map_operation_to_category(
                                category_name, baseline_categories
                            )

                            if category:
                                category_gains_ms[category] = (
                                    category_gains_ms.get(category, 0) + gain_ms
                                )
                                print(
                                    f"    ✓ Mapped '{category_name}' -> '{category}': {gain_ms:.2f}ms ({impact_pct:.2f}%)"
                                )
                            else:
                                print(
                                    f"    ✗ Could not map '{category_name}' to any baseline category"
                                )
                                print(
                                    f"       Available categories: {list(baseline_categories.keys())}"
                                )

        print(f"\n  ===== EXTRACTION SUMMARY =====")
        print(f"  Total extracted gains: {category_gains_ms}")
        print(
            f"  Total optimization potential: {sum(category_gains_ms.values()):.2f}ms"
        )
        print(
            f"  Potential improvement: {(sum(category_gains_ms.values()) / baseline_total_ms * 100):.2f}%"
        )
        print(f"  ==============================\n")

        return category_gains_ms

    def _map_operation_to_category(
        self, operation: str, baseline_categories: Dict[str, float]
    ) -> Optional[str]:
        """Map an operation name to a category"""
        operation_lower = operation.lower()

        # Try exact match first
        for cat_name in baseline_categories.keys():
            if cat_name.lower() == operation_lower:
                return cat_name

        # Try partial match
        for cat_name in baseline_categories.keys():
            cat_lower = cat_name.lower()
            if cat_lower in operation_lower or operation_lower in cat_lower:
                return cat_name

            # Try without "operations" suffix
            cat_base = (
                cat_lower.replace(" operations", "").replace("operations", "").strip()
            )
            op_base = (
                operation_lower.replace(" operations", "")
                .replace("operations", "")
                .strip()
            )
            if cat_base in op_base or op_base in cat_base:
                return cat_name

        # Try keyword matching for common patterns
        keyword_map = {
            "batch_norm": ["batchnorm", "batch norm", "bn_", "bn ", "_bn"],
            "conv": ["convolution", "conv_", "conv ", "_conv"],
            "pool": ["pooling", "maxpool", "avgpool", "pool_"],
            "gemm": ["matmul", "linear", "addmm", "mm_", "mm ", "matrix"],
            "elementwise": [
                "element_wise",
                "element-wise",
                "relu",
                "sigmoid",
                "add",
                "mul",
            ],
            "memory": ["memcpy", "copy", "mem_"],
            "nccl": ["allreduce", "broadcast", "c10d", "collective"],
            "flash": ["attention", "attn"],
            "optimizer": ["adam", "sgd", "rmsprop"],
            "loss": ["cross_entropy", "softmax", "nll"],
        }

        for base_keyword, variants in keyword_map.items():
            if base_keyword in operation_lower or any(
                v in operation_lower for v in variants
            ):
                # Find matching category
                for cat_name in baseline_categories.keys():
                    cat_lower = cat_name.lower()
                    if base_keyword in cat_lower or any(
                        v in cat_lower for v in variants
                    ):
                        return cat_name

        return None

    def _extract_opportunities_table(
        self, ai_analysis: str, baseline_total_ms: float
    ) -> List[Dict]:
        """Extract opportunities table data from AI analysis"""
        opportunities = []
        lines = ai_analysis.split("\n")
        in_opportunities_section = False

        print(f"\n  Extracting opportunities table...")

        for line_num, line in enumerate(lines):
            # Look for OPPORTUNITIES section (HIGH-IMPACT OPTIMIZATION OPPORTUNITIES or OPPORTUNITIES AND IMPACT)
            if (
                not in_opportunities_section
                and ("OPPORTUNITIES" in line.upper() or "HIGH-IMPACT" in line.upper())
                and "IMPACT" in line.upper()
                and not line.strip().startswith("|")
            ):
                in_opportunities_section = True
                print(
                    f"  Found opportunities section at line {line_num}: {line.strip()}"
                )
                continue

            # Stop at next major section
            if in_opportunities_section:
                if (
                    line.startswith("##")
                    and "OPPORTUNITIES" not in line.upper()
                    and "IMPACT" not in line.upper()
                ):
                    print(f"  End of opportunities section at line {line_num}")
                    break

                # Parse table rows
                if "|" in line and len(line.strip()) > 5:
                    parts = [p.strip() for p in line.split("|")]

                    # Skip separator and header rows
                    line_lower = line.lower()
                    is_separator = "---" in line
                    is_header = "priority" in line_lower or "category" in line_lower
                    is_cumulative = "cumulative" in line_lower

                    # Parse data rows: | Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall |
                    if (
                        len(parts) >= 7
                        and not is_separator
                        and not is_header
                        and not is_cumulative
                    ):
                        priority = parts[1].strip()
                        category = re.sub(r"[\*\`#🎯⚡⚠️✓⚪]", "", parts[2]).strip()
                        current_time_str = parts[3].strip()
                        target_time_str = parts[4].strip()
                        potential_gain_str = parts[5].strip()
                        impact_pct_str = parts[6].strip()

                        # Extract numeric values
                        current_time = self._extract_ms_value(current_time_str)
                        target_time = self._extract_ms_value(target_time_str)
                        potential_gain = self._extract_ms_value(potential_gain_str)
                        impact_pct = self._extract_percentage(impact_pct_str)

                        if category and potential_gain > 0:
                            opportunities.append(
                                {
                                    "priority": priority,
                                    "category": category,
                                    "current_time": current_time,
                                    "target_time": target_time,
                                    "potential_gain": potential_gain,
                                    "impact_pct": impact_pct,
                                }
                            )
                            print(
                                f"    Added: {category} - {potential_gain:.2f}ms ({impact_pct:.2f}%)"
                            )

        print(f"  Total opportunities extracted: {len(opportunities)}")
        return opportunities

    def _extract_ms_value(self, text: str) -> float:
        """Extract millisecond value from text"""
        text = (
            text.replace("**", "")
            .replace("*", "")
            .replace("ms", "")
            .replace(",", "")
            .strip()
        )
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else 0.0

    def _extract_percentage(self, text: str) -> float:
        """Extract percentage value from text"""
        text = text.replace("**", "").replace("*", "").replace("%", "").strip()
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        return float(match.group(1)) if match else 0.0

    def _generate_opportunities_chart(
        self, opportunities: List[Dict], baseline_total_ms: float
    ) -> Optional[Path]:
        """Generate bar chart for optimization opportunities"""
        try:
            if not opportunities:
                return None

            # Sort by potential gain (descending)
            opportunities = sorted(
                opportunities, key=lambda x: x["potential_gain"], reverse=True
            )

            categories = [opp["category"] for opp in opportunities]
            gains = [opp["potential_gain"] for opp in opportunities]
            impacts = [opp["impact_pct"] for opp in opportunities]
            priorities = [opp["priority"] for opp in opportunities]

            # Color mapping for priorities
            color_map = {
                "🎯 High": "#e74c3c",  # Red
                "⚡ Medium": "#f39c12",  # Orange
                "⚪ Low": "#95a5a6",  # Gray
            }

            colors = []
            for priority in priorities:
                matched = False
                for key in color_map:
                    if key in priority or priority in key:
                        colors.append(color_map[key])
                        matched = True
                        break
                if not matched:
                    # Default color based on emoji
                    if "🎯" in priority:
                        colors.append("#e74c3c")
                    elif "⚡" in priority:
                        colors.append("#f39c12")
                    else:
                        colors.append("#95a5a6")

            fig, ax = plt.subplots(figsize=(14, max(8, len(categories) * 0.5)))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#f8f9fa")

            # Create horizontal bar chart
            y_pos = np.arange(len(categories))
            bars = ax.barh(
                y_pos, gains, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2
            )

            # Add value labels on bars
            for i, (bar, gain, impact) in enumerate(zip(bars, gains, impacts)):
                width = bar.get_width()
                ax.text(
                    width + max(gains) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{gain:.1f} ms\n({impact:.2f}%)",
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories, fontsize=11)
            ax.set_xlabel("Potential Gain (ms)", fontsize=12, fontweight="bold")
            ax.set_title(
                "High-Impact Optimization Opportunities",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            # Add legend for priority levels
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#e74c3c", label="🎯 High Priority (>5% impact)"),
                Patch(facecolor="#f39c12", label="⚡ Medium Priority (1-5% impact)"),
                Patch(facecolor="#95a5a6", label="⚪ Low Priority (<1% impact)"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

            # Add cumulative impact text
            total_gain = sum(gains)
            total_impact = (total_gain / baseline_total_ms) * 100
            ax.text(
                0.02,
                0.98,
                f"Cumulative Impact: {total_gain:.1f} ms ({total_impact:.2f}%)",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="#3498db", alpha=0.3),
                verticalalignment="top",
            )

            ax.grid(axis="x", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()

            # Save plot
            plot_path = self.output_dir / "baseline_optimization_opportunities.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"    ✓ Optimization opportunities chart: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"    ⚠️  Failed to generate opportunities chart: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _calculate_projection(
        self, baseline_categories: Dict[str, float], category_gains_ms: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate projected categories after applying optimization gains"""
        projection = {}

        print(f"\\n  Calculating projection:")
        print(f"    Baseline categories: {baseline_categories}")
        print(f"    Optimization gains (ms): {category_gains_ms}")

        for cat, baseline_val in baseline_categories.items():
            gain_ms = category_gains_ms.get(cat, 0)

            # Simply subtract the gain in milliseconds
            if gain_ms > 0:
                # Ensure we don't go negative
                projected_val = max(0, baseline_val - gain_ms)
                projection[cat] = projected_val
                print(
                    f"    {cat}: {baseline_val:.2f}ms -> {projected_val:.2f}ms (gain: {gain_ms:.2f}ms)"
                )
            else:
                projection[cat] = baseline_val

        return projection

    def _generate_cumulative_chart(
        self,
        baseline_cat: Dict[str, float],
        projection_cat: Dict[str, float],
        target_cat: Dict[str, float],
        baseline_total: float,
        projection_total: float,
        target_total: float,
    ) -> Optional[Path]:
        """Generate the cumulative stacked bar chart"""
        try:
            # COMMENTED OUT TARGET BAR - Keep for later use
            # all_categories = sorted(set(baseline_cat.keys()) | set(projection_cat.keys()) | set(target_cat.keys()))
            all_categories = sorted(
                set(baseline_cat.keys()) | set(projection_cat.keys())
            )

            fig, ax = plt.subplots(figsize=(15, 10))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#f8f9fa")

            # COMMENTED OUT TARGET BAR - Keep for later use
            # states = ['BASELINE', 'PROJECTION', 'TARGET']
            # data_dict = {
            #     'BASELINE': baseline_cat,
            #     'PROJECTION': projection_cat,
            #     'TARGET': target_cat
            # }
            # totals = [baseline_total, projection_total, target_total]

            states = ["BASELINE", "PROJECTION"]
            data_dict = {
                "BASELINE": baseline_cat,
                "PROJECTION": projection_cat,
            }
            # totals = [baseline_total, projection_total]
            totals = [sum(baseline_cat.values()), sum(projection_cat.values())]

            # COMMENTED OUT TARGET BAR - Keep for later use
            # bottom_values = np.zeros(3)
            bottom_values = np.zeros(2)
            bar_width = 0.45
            x_pos = np.arange(len(states))

            # Assign colors
            cat_colors = {}
            for i, cat in enumerate(all_categories):
                cat_colors[cat] = self.category_colors.get(cat, plt.cm.tab20(i % 20))

            # Draw stacked bars
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

                # Add value labels
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

            # Customize
            ax.set_ylabel(
                "Total Execution Time (ms)", fontsize=14, fontweight="bold", labelpad=12
            )
            ax.set_title(
                "Cumulative Performance Optimization Projection (idle time not shown)",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(states, fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
            ax.set_axisbelow(True)

            # Add total labels
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
                    # Calculate improvement excluding idle time
                    baseline_active = sum(baseline_cat.values())
                    projection_active = sum(projection_cat.values())
                    improvement_pct_active = (
                        ((baseline_active - projection_active) / baseline_active * 100)
                        if baseline_active > 0
                        else 0
                    )

                    # Calculate total improvement (with idle)
                    improvement_pct_total = (
                        ((baseline_active - projection_active) / baseline_total * 100)
                        if baseline_total > 0
                        else 0
                    )

                    ax.text(
                        i,
                        total + max(totals) * 0.035,
                        f"↓ {improvement_pct_active:.1f}% (sans idle) | ↓ {improvement_pct_total:.1f}% total",
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

            # Spine adjustments
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")

            plt.tight_layout()
            cumulative_plot = (
                self.output_dir / "baseline_cumulative_optimization_progression.png"
            )
            plt.savefig(
                str(cumulative_plot),
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()
            print(f"    ✓ Cumulative optimization progression chart: {cumulative_plot}")
            return cumulative_plot

        except Exception as e:
            print(f"    ⚠️  Cumulative chart generation failed: {e}")
            import traceback

            traceback.print_exc()

        return None

    @staticmethod
    def normalize_category_name(name: str) -> str:
        """Normalize category names for consistent coloring"""
        name_map = {
            "CONV": "Convolution Operations",
            "Conv": "Convolution Operations",
            "Convolution": "Convolution Operations",
            "GEMM": "GEMM",
            "MatMul": "GEMM",
            "Linear": "GEMM",
            "Batch Norm": "Batch Normalization",
            "BatchNorm": "Batch Normalization",
            "BN": "Batch Normalization",
            "Pool": "Pooling Operations",
            "Pooling": "Pooling Operations",
            "NCCL": "NCCL Operations",
            "Communication": "NCCL Operations",
            "Memory": "Memory Operations",
            "MemCpy": "Memory Operations",
            "Element": "Element-wise",
            "Elementwise": "Element-wise",
            "Activation": "Element-wise",
        }
        return name_map.get(name, name)
