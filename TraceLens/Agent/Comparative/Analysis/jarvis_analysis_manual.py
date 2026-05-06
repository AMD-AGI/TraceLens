#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
JARVIS UNIFIED ANALYZER - Main Orchestrator
Complete workflow: Raw traces → Analysis → Report generation

This is the main entry point that coordinates all analysis modules.

Usage:
  python3 Analysis/jarvis_analysis.py \\
    --gpu1-kineto trace1.json \\
    --gpu1-et et1.json \\
    --gpu2-kineto trace2.json \\
    --gpu2-et et2.json \\
    --api-key YOUR_KEY
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Jarvis modules
from tracelens_runner import TraceLensRunner
from llm_prompts import LLMPromptManager
from plotting_complete import JarvisPlotter
from report_generator import ReportGenerator
import data_extractors
from sharepoint_loader import load_trace_from_url, is_sharepoint_url
import visualizations
from TraceLens import TraceDiff, TreePerfAnalyzer

from tracelens_diff_analyzer import *  # analyze_lca_differences, print_lca_warnings, save_lca_results


import pdb

# Import AI clients
try:
    from slodels import SLAIOpenAI, SLAIAnthropic

    AI_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: AI modules not available")
    AI_AVAILABLE = False


def add_op_cat_column(data_dict: Dict):
    """
    Normalize op category columns across all sheets.
    If a sheet has 'parent_module' but no 'op category', add 'op category' = 'parent_module'
    """

    for sheet_name, df in data_dict.items():
        # Skip non-DataFrame entries (like 'trace_file')
        if not isinstance(df, pd.DataFrame):
            continue

        # Check if sheet has 'parent_module' but no 'op category'
        if "parent_module" in df.columns and "op category" not in df.columns:
            df["op category"] = df["parent_module"]
            print(
                f"  ✓ Added 'op category' column to '{sheet_name}' (copied from 'parent_module')"
            )


class JarvisAnalyzer:
    """
    Main orchestrator for Jarvis GPU comparison analysis.

    Coordinates:
    - TraceLens report generation (tracelens_runner.py)
    - LLM analysis (llm_prompts.py + AI client)
    - Plotting (plotting.py)
    - Report generation (report_generator.py)
    """

    def __init__(
        self,
        gpu1_kineto: str,
        gpu1_et: str,
        gpu2_kineto: str,
        gpu2_et: str,
        gpu1_name: Optional[str] = None,
        gpu2_name: Optional[str] = None,
        output_dir: str = "trace_reports",
        api_key: Optional[str] = None,
        save_intermediates: bool = True,
        generate_plots: bool = False,
        use_critical_path: bool = True,
        enable_inference_phase_analysis: bool = False,
    ):
        """
        Args:
            gpu1_kineto: Path or URL to GPU1 kineto trace
            gpu1_et: Path or URL to GPU1 execution trace
            gpu2_kineto: Path or URL to GPU2 kineto trace
            gpu2_et: Path or URL to GPU2 execution trace
            gpu1_name: Name for GPU1 (auto-extracted if None)
            gpu2_name: Name for GPU2 (auto-extracted if None)
            output_dir: Base output directory
            api_key: OpenAI/Anthropic API key
            save_intermediates: Whether to save intermediate files
            generate_plots: Enable plotting
            use_critical_path: Use critical path data in analysis
            enable_inference_phase_analysis: Enable prefill/decode phase analysis for inference traces
        """
        # Download traces from URLs if needed
        print("🔍 Checking trace paths...")

        # Set up temporary output directory first (needed for cache)
        # We'll refine it later with timestamp
        self._temp_output_base = Path(output_dir)

        self.gpu1_kineto = self._resolve_trace_path(gpu1_kineto, "GPU1 kineto")
        self.gpu1_et = self._resolve_trace_path(gpu1_et, "GPU1 ET") if gpu1_et else None
        self.gpu2_kineto = self._resolve_trace_path(gpu2_kineto, "GPU2 kineto")
        self.gpu2_et = self._resolve_trace_path(gpu2_et, "GPU2 ET") if gpu2_et else None

        # Auto-extract GPU names if not provided
        self.gpu1_name = gpu1_name or self._extract_gpu_name(self.gpu1_kineto)
        self.gpu2_name = gpu2_name or self._extract_gpu_name(self.gpu2_kineto)

        # Setup output directory with timestamp
        # Check if output_dir already looks like a complete report directory
        # Pattern: ends with _YYYYMMDD_HHMMSS or contains vs in the final component
        import re

        output_path = self._temp_output_base
        final_dir_name = output_path.name

        # Check if the final directory already has timestamp and GPU names
        has_timestamp = re.search(r"_\d{8}_\d{6}$", final_dir_name)
        has_vs = "_vs_" in final_dir_name

        if has_timestamp and has_vs:
            # Output dir already looks like a complete report directory, use it as-is
            self.output_dir = output_path
            print(f"📁 Using existing output directory: {self.output_dir}")
        else:
            # Create new timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = (
                output_path / f"{self.gpu1_name}_vs_{self.gpu2_name}_{timestamp}"
            )
            print(f"📁 Creating new output directory: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key
        self.save_intermediates = save_intermediates
        self.generate_plots = generate_plots
        self.use_critical_path = use_critical_path
        self.enable_inference_phase_analysis = enable_inference_phase_analysis

        # Initialize modules
        self.tracelens_runner = TraceLensRunner(
            save_intermediates, use_critical_path, enable_inference_phase_analysis
        )
        self.llm_prompt_manager = LLMPromptManager(use_critical_path)

        # AI client setup
        self.client_anthropic = None
        if api_key and AI_AVAILABLE:
            self.client_anthropic = SLAIAnthropic(api_key=api_key)

        # Data storage
        self.gpu1_data = {}
        self.gpu2_data = {}

    def expand_table(self, markdown_context, table):

        PROMPT = """
            You are analyzing GPU workload performance to enhance a projection table with context-aware comments.

    ### MARKDOWN DATA STRUCTURE REFERENCE

    The analysis text below is based on a comprehensive markdown comparison report with the following structure:

    **Key Sections Available:**
    1. **GPU_TIMELINE** - High-level GPU utilization metrics (computation_time, exposed_comm_time, exposed_memcpy_time, busy_time, idle_time, total_time)
    2. **OPS_SUMMARY** - Operation-level aggregation grouped by operation name only (e.g., all aten::addmm calls together)
    3. **OPS_SUMMARY_BY_CATEGORY** - Category-level aggregation (GEMM, CONV_fwd, CONV_bwd, BN_fwd, BN_bwd, etc.)
    4. **OPS_UNIQUE_ARGS** - Most detailed view: operations grouped by unique combinations of (name + input shapes + dtypes + strides)
    5. **OPTIMIZATION PROJECTIONS** - Pre-calculated projection tables showing:
    - Current Time (ms) - Baseline performance
    - Projected Optimized Time (ms) - Best achievable (min of baseline/target)
    - Potential Gain (ms) - Time savings
    - Impact (%) - Percentage impact on overall runtime
    - Key Candidate Operations - Top operations to optimize from target trace

    **Important Context:**
    - Gap Formula: Gap = Baseline - Target
    * POSITIVE gap = Baseline SLOWER (took more time)
    * NEGATIVE gap = Baseline FASTER (took less time)
    - Categories may be aggregated differently between baseline and target (e.g., fused kernels vs separate ops)
    - Kernel naming conventions differ between AMD (rocBLAS, MIOpen) and NVIDIA (CUTLASS, cuBLAS, cuDNN)
    - Operations may fall under different categories due to implementation differences

    ### YOUR TASK

    Your task is: for each row in the table, if there are special things to consider based on the information provided, add it as a brief comment. For example, if you see that in the target workload, a given category includes fused kernels while the baseline doesn't, you can point it out. If you spot something that might create misleading predictions (e.g. equivalent kernels falling under different groups target and baseline, or the other way around), you can point it out.

    The output should be in the following form:
    comments for row 1***comments for row 2***comments for row 3 etc.

    Do not use the triple * (***) anywhere else in the output. If the analysis text had triple *** in it, escape or substitute something appropriate instead.

    You can use markdown.

    Be mindful of target and baseline architectures.
    In general, we are trying to improve baseline. So if, for example, baseline is mi300, make sure that recommendations are relevant to AMD architectures. Rephrase or remove recommendations that are not appropriate to the manufacturer. For example, a change is needed when a recommendation mentions CUTLASS when the baseline is AMD, or when a recommendation mention RocBLAS when the baseline is NVIDIA.

    If you don't have an insight to share, enter "no comment" for the corresponding rows.
    Output ONLY the comments, start right away.
    """
        PROMPT += f"### MARKDOWN COMPARISON DATA ###\n{markdown_context}\n### END OF MARKDOWN DATA ###\n\n"
        PROMPT += f"### PREDICTION TABLE ### {table} ### END OF PREDICTION TABLE ### \nExtracted comments:\n"

        try:
            response = self.client_anthropic.messages.create(
                model="claude-sonnet-4",
                max_tokens=5000,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": """You are helping to aggregate different sources of a GPU workload analysis and expand the projection table with relevant comments.""",
                    },
                    {"role": "user", "content": PROMPT},
                ],
            )
            new_row = response.content[0].text.strip().split("***")
            return new_row

        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return None

    # def _analyze_tracelens_diff_lcas(self):
    #     """Analyze TraceLens diff for unique LCAs and multi-module LCAs"""
    #     diff_csv = self.output_dir / "tracelens_diff" / "diff_stats.csv"

    #     if not diff_csv.exists():
    #         print("  ⚠️  TraceDiff CSV not found, skipping LCA analysis")
    #         return None

    #     try:
    #         # Analyze unique LCAs
    #         print("\n🔍 Analyzing TraceDiff LCAs...")
    #         unique_results = analyze_lca_differences(
    #             str(diff_csv),
    #             self.gpu1_name,
    #             self.gpu2_name
    #         )

    #         print_lca_warnings(unique_results, self.gpu1_name, self.gpu2_name)

    #         # Analyze multi-module LCAs
    #         print("\n🔍 Analyzing multi-module LCAs...")
    #         multi_results = analyze_multi_module_lcas(
    #             str(diff_csv),
    #             self.gpu1_name,
    #             self.gpu2_name
    #         )

    #         print_multi_module_warnings(multi_results, self.gpu1_name, self.gpu2_name)

    #         # Save all results
    #         output_dir = self.output_dir / "lca_analysis"
    #         save_lca_results(unique_results, output_dir, self.gpu1_name, self.gpu2_name)
    #         save_multi_module_results(multi_results, output_dir, self.gpu1_name, self.gpu2_name)

    #         return {
    #             'unique_lcas': unique_results,
    #             'multi_module': multi_results
    #         }

    #     except Exception as e:
    #         print(f"  ⚠️  LCA analysis failed: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    def _analyze_tracelens_diff_lcas(self):
        """Analyze TraceLens diff for unique LCAs, multi-module LCAs, and problematic groups"""
        diff_csv = self.output_dir / "tracelens_diff" / "diff_stats.csv"

        if not diff_csv.exists():
            print("  ⚠️  TraceDiff CSV not found, skipping LCA analysis")
            return None

        try:
            from tracelens_diff_analyzer import (
                analyze_lca_differences,
                print_lca_warnings,
                save_lca_results,
                analyze_multi_module_lcas,
                print_multi_module_warnings,
                save_multi_module_results,
                prepare_cleaned_diff_data,
                analyze_problematic_groups,
                save_problematic_results,
                calculate_lca_optimization_opportunities,
                save_lca_optimization_opportunities,
            )

            # Step 1: Analyze unique LCAs
            # gpu2 = baseline (trace2), gpu1 = target (trace1)
            print("\n🔍 Analyzing unique LCAs...")
            unique_results = analyze_lca_differences(
                diff_csv_path=str(diff_csv),
                baseline_name=self.gpu2_name,
                target_name=self.gpu1_name,
                baseline_source="trace2",
                target_source="trace1",
            )
            print_lca_warnings(unique_results, self.gpu2_name, self.gpu1_name)

            # Step 2: Analyze multi-module LCAs
            print("\n🔍 Analyzing multi-module LCAs...")
            multi_results = analyze_multi_module_lcas(
                diff_csv_path=str(diff_csv),
                baseline_name=self.gpu2_name,
                target_name=self.gpu1_name,
                baseline_source="trace2",
                target_source="trace1",
            )
            print_multi_module_warnings(multi_results, self.gpu2_name, self.gpu1_name)

            # Step 3: Clean data and analyze problematic groups
            print("\n🔍 Analyzing problematic groups...")
            df_cleaned = prepare_cleaned_diff_data(
                diff_csv_path=str(diff_csv),
                baseline_name=self.gpu2_name,
                target_name=self.gpu1_name,
                unique_lca_results=unique_results,
                multi_module_results=multi_results,
            )

            problematic_results = analyze_problematic_groups(df_cleaned)

            # Step 4: Calculate optimization opportunities
            print("\n🔍 Calculating optimization opportunities...")
            lca_opportunities = calculate_lca_optimization_opportunities(
                df_with_corrections=problematic_results["df_with_corrections"],
                baseline_name=self.gpu2_name,
                target_name=self.gpu1_name,
                baseline_source="trace2",
                target_source="trace1",
            )

            # Save all results
            output_dir = self.output_dir / "lca_analysis"
            save_lca_results(unique_results, output_dir, self.gpu2_name, self.gpu1_name)
            save_multi_module_results(
                multi_results, output_dir, self.gpu2_name, self.gpu1_name
            )
            save_problematic_results(problematic_results, output_dir)
            # save_lca_optimization_opportunities(lca_opportunities, output_dir, self.gpu2_name, self.gpu1_name)

            save_lca_optimization_opportunities(
                lca_opportunities,
                output_dir,
                self.gpu2_name,
                self.gpu1_name,
                df_with_corrections=problematic_results[
                    "df_with_corrections"
                ],  # ADD this
            )

            kernel_summary_path = output_dir / "kernel_optimization_summary.csv"
            if kernel_summary_path.exists():
                visualizations.generate_kernel_optimization_html(kernel_summary_path)

            return {
                "unique_lcas": unique_results,
                "multi_module": multi_results,
                "problematic_groups": problematic_results,
                "lca_opportunities": lca_opportunities,
            }

        except Exception as e:
            print(f"  ⚠️  LCA analysis failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _resolve_trace_path(self, path: str, description: str) -> str:
        """
        Resolve trace path - download from URL if needed, otherwise use local path.

        Args:
            path: Local path or URL to trace file
            description: Description for logging (e.g., "GPU1 kineto")

        Returns:
            Local file path to the trace
        """
        if not path:
            return path

        # Check if it's a URL
        if path.startswith(("http://", "https://")):
            print(f"  📥 Downloading {description} from URL...")
            if is_sharepoint_url(path):
                print(f"     SharePoint URL detected")

            # Create cache directory for downloaded traces
            cache_dir = self._temp_output_base / ".trace_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            local_path = load_trace_from_url(path, cache_dir)
            print(f"     ✓ Downloaded to: {local_path}")
            return local_path
        else:
            # Local path - verify it exists
            if not Path(path).exists():
                raise FileNotFoundError(f"{description} not found: {path}")
            print(f"  ✓ Using local {description}: {path}")
            return path

    def run(self) -> bool:
        """
        Execute the complete analysis workflow.

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 70)
        print("JARVIS UNIFIED ANALYZER")
        print("=" * 70)
        print(f"GPU 1: {self.gpu1_name}")
        print(f"GPU 2: {self.gpu2_name}")
        print(f"Output: {self.output_dir}")
        print(f"Critical Path: {'Enabled' if self.use_critical_path else 'Disabled'}")
        print("=" * 70 + "\n")

        try:
            # Step 1: Process GPU 1
            print("🔄 Processing GPU 1...")
            self._process_gpu(
                self.gpu1_kineto, self.gpu1_et, self.gpu1_name, self.gpu1_data
            )

            # Step 2: Process GPU 2
            print("\n🔄 Processing GPU 2...")
            self._process_gpu(
                self.gpu2_kineto, self.gpu2_et, self.gpu2_name, self.gpu2_data
            )

            # Step 3: add tracediff-based info

            self._generate_tracelens_diff_report()

            # Step 3: Generate comparison report (SKIPPED - not needed for current workflow)
            # print("\n📊 Generating comparison report...")
            # gpu1_report = self.output_dir / self.gpu1_name / f"{self.gpu1_name}_tracelens_report.xlsx"
            # gpu2_report = self.output_dir / self.gpu2_name / f"{self.gpu2_name}_tracelens_report.xlsx"
            #
            # success = self.tracelens_runner.generate_comparison_report(
            #     gpu1_report,
            #     gpu2_report,
            #     self.output_dir,
            #     self.gpu1_name,
            #     self.gpu2_name
            # )

            # Step 3.1
            self._analyze_tracelens_diff_lcas()

            # # Step 3.5: Generate unified comparison markdown (lightweight for LLM)
            # print("\n📝 Generating unified comparison markdown (lightweight)...")
            # from markdown_generator import MarkdownGenerator

            # if self.gpu1_data and self.gpu2_data:
            #     # Use whitelist mode to only include essential sheets for LLM
            #     markdown_gen = MarkdownGenerator(
            #         included_sheets=['gpu_timeline', 'ops_summary', 'ops_summary_by_category', 'ops_unique_args']
            #     )
            #     comparison_md_path = self.output_dir / "comparison_report.md"

            #     markdown_gen.generate_comparison_markdown(
            #         gpu1_dfs=self.gpu2_data,  # baseline
            #         gpu2_dfs=self.gpu1_data,  # target
            #         gpu1_name=self.gpu2_name,  # baseline
            #         gpu2_name=self.gpu1_name,  # target
            #         output_path=comparison_md_path,
            #         max_rows_per_sheet=30,  # Reduced from 50
            #         max_rows_ops_unique_args=10  # ops_unique_args is extremely verbose
            #     )
            #     print(f"✓ Unified comparison markdown: {comparison_md_path}")

            # # Step 4: AI Analysis
            # ai_analysis = None
            # if self.client_anthropic:
            #     print("\n🤖 Performing AI analysis...")
            #     ai_analysis = self._run_ai_analysis()

            # Step 3: Generate plots & projection tables
            plot_paths = []
            table_paths = {}

            if self.generate_plots:
                print("\n📈 Generating plots...")

                # Generate cumulative progression chart
                from plotting_manual import CumulativeProjectionChart

                for agg_over in ["op category", "parent_module"]:

                    baseline_df = self.gpu2_data.get(
                        "ops_unique_args"
                    )  # can be ops_summary
                    target_df = self.gpu1_data.get("ops_unique_args")

                    if baseline_df is not None and target_df is not None:
                        print(
                            f"  Generating cumulative progression chart for '{agg_over}'..."
                        )

                        # Create modified dataframes with unmatched categories
                        baseline_df_modified, target_df_modified, unmatched_info = (
                            self._handle_unmatched_categories(
                                baseline_df=baseline_df,
                                target_df=target_df,
                                category_column=agg_over,
                            )
                        )

                        # Print unmatched category info
                        if (
                            unmatched_info["baseline_unmatched"]
                            or unmatched_info["target_unmatched"]
                        ):
                            print(f"\n  ℹ️  Unmatched categories for '{agg_over}':")
                            if unmatched_info["baseline_unmatched"]:
                                print(
                                    f"    Baseline only: {unmatched_info['baseline_unmatched']}"
                                )
                            if unmatched_info["target_unmatched"]:
                                print(
                                    f"    Target only: {unmatched_info['target_unmatched']}"
                                )

                        chart_gen = CumulativeProjectionChart(self.output_dir)

                        chart_path = chart_gen.generate_chart(
                            baseline_df=baseline_df_modified,
                            target_df=target_df_modified,
                            time_column="total_direct_kernel_time_sum",
                            category_column=agg_over,
                            baseline_label=f"BASELINE ({self.gpu2_name})",
                            projection_label="PROJECTION",
                            target_label=f"TARGET ({self.gpu1_name})",
                            include_target_bar=False,
                            filename=f"cumulative_projection_{agg_over}.png",
                        )

                        if chart_path:
                            plot_paths.append(chart_path)
                            print(f"  ✓ Cumulative progression chart: {chart_path}")

                        table_path = (
                            chart_gen.generate_optimization_opportunities_table(
                                baseline_df=baseline_df_modified,
                                target_df=target_df_modified,
                                time_column="total_direct_kernel_time_sum",
                                category_column=agg_over,
                                filename=f"optimization_opportunities_{agg_over}.csv",
                            )
                        )
                        if table_path:
                            table_paths[agg_over] = table_path

                    else:
                        print("  ⚠️  Category data not available for chart generation")

                        # Get totals for report
                        # baseline_total = self._get_total_time(self.gpu2_data)
                        # target_total = self._get_total_time(self.gpu1_data)

                        # executive_summary = self._generate_executive_summary(
                        #     baseline_total,
                        #     target_total
                        # )

            # Step 4: Generate markdown comparison with projection tables
            print("\n📝 Generating comparison markdown with projections...")
            from markdown_generator_manual import MarkdownGenerator

            if self.gpu1_data and self.gpu2_data:
                # Use whitelist mode to only include essential sheets for LLM
                markdown_gen = MarkdownGenerator(
                    included_sheets=[
                        "gpu_timeline",
                        "ops_summary",
                        "ops_summary_by_category",
                        "ops_unique_args",
                    ]
                )
                comparison_md_path = self.output_dir / "comparison_report.md"

                # Pass all projection tables as a dictionary
                markdown_gen.generate_comparison_markdown(
                    gpu1_dfs=self.gpu2_data,  # baseline
                    gpu2_dfs=self.gpu1_data,  # target
                    gpu1_name=self.gpu2_name,  # baseline
                    gpu2_name=self.gpu1_name,  # target
                    output_path=comparison_md_path,
                    projection_table_paths=table_paths,  # Dict[str, Path] with all tables
                    max_rows_per_sheet=30,  # 30,
                    max_rows_ops_unique_args=10,
                )
                print(
                    f"✓ Unified comparison markdown with projections: {comparison_md_path}"
                )

                final_table_agg = "parent_module"
                cumulative_table = table_paths[final_table_agg]
                df = pd.read_csv(cumulative_table)

                # Load markdown comparison report for context
                comparison_md_path = self.output_dir / "comparison_report.md"

                if comparison_md_path.exists():
                    print(f"\n📋 Loading markdown for table expansion...")
                    with open(comparison_md_path, "r", encoding="utf-8") as f:
                        markdown_context = f.read()
                    print(f"✓ Loaded markdown: {len(markdown_context)} chars")

                    try:
                        # Pass markdown context instead of ai_analysis
                        comments = self.expand_table(markdown_context, df.to_string())

                        if comments and len(comments) == len(df):
                            df["Comments"] = comments
                            cumulative_table_expanded = (
                                cumulative_table.parent
                                / "optimization_opportunities_table_expanded.csv"
                            )
                            df.to_csv(cumulative_table_expanded, index=False)
                            plot_paths.append(cumulative_table_expanded)
                            print(
                                f"✓ Expanded table with markdown context: {cumulative_table_expanded}"
                            )

                            saved_png_path = (
                                "cumulative_projection_" + final_table_agg + ".png"
                            )
                            html_path = (
                                visualizations.generate_interactive_optimization_html(
                                    cumulative_table_expanded, saved_png_path
                                )
                            )
                            if html_path:
                                plot_paths.append(html_path)
                                print(f"✓ Generated interactive HTML: {html_path}")
                        else:
                            print(
                                f"⚠️  Comment count mismatch: got {len(comments) if comments else 0}, expected {len(df)}\nComments:{comments}"
                            )

                    except Exception as e:
                        print(f"⚠️  Failed to expand table: {e}")
                        import traceback

                        traceback.print_exc()
                else:
                    print(
                        f"⚠️  Markdown comparison not found at {comparison_md_path}, skipping table expansion"
                    )

            print("\n" + "=" * 70)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 70)
            print("=" * 70 + "\n")

            return True

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _handle_unmatched_categories(
        self, baseline_df: pd.DataFrame, target_df: pd.DataFrame, category_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Create modified dataframes where categories that exist in only one trace
        are relabeled as 'unmatched_baseline' or 'unmatched_target'.

        Args:
            baseline_df: Baseline operations dataframe
            target_df: Target operations dataframe
            category_column: Column name containing categories

        Returns:
            Tuple of (modified_baseline_df, modified_target_df, unmatched_info)
            where unmatched_info contains lists of original unmatched category names
        """
        # Work on copies to avoid modifying originals
        baseline_df = baseline_df.copy()
        target_df = target_df.copy()

        baseline_df[category_column] = baseline_df[category_column].replace(
            "NA", "Root"
        )
        baseline_df[category_column] = baseline_df[category_column].fillna("Root")

        target_df[category_column] = target_df[category_column].replace("NA", "Root")
        target_df[category_column] = target_df[category_column].fillna("Root")

        # Get unique categories from each
        baseline_categories = set(baseline_df[category_column].unique())
        target_categories = set(target_df[category_column].unique())

        # Find unmatched categories
        baseline_only = baseline_categories - target_categories
        target_only = target_categories - baseline_categories

        # Track which categories were unmatched
        unmatched_info = {
            "baseline_unmatched": sorted(list(baseline_only)),
            "target_unmatched": sorted(list(target_only)),
        }

        # Relabel unmatched categories
        if baseline_only:
            baseline_df.loc[
                baseline_df[category_column].isin(baseline_only), category_column
            ] = "Unmatched"

        if target_only:
            target_df.loc[
                target_df[category_column].isin(target_only), category_column
            ] = "Unmatched"

        return baseline_df, target_df, unmatched_info

    def _process_gpu(self, kineto: str, et: str, name: str, data_dict: Dict):
        """Process a single GPU's traces"""
        print(f"\n{'='*70}")
        print(f"DEBUG: _process_gpu called for {name}")
        print(f"  Kineto trace: {kineto}")
        print(f"  ET trace: {et}")
        print(f"  GPU name: {name}")
        print(f"  Data dict keys before: {list(data_dict.keys())}")
        print(f"{'='*70}\n")

        gpu_dir = self.output_dir / name
        gpu_dir.mkdir(parents=True, exist_ok=True)

        # Check if TraceLens report already exists
        report_path = gpu_dir / f"{name}_tracelens_report.xlsx"
        print(f"\n📁 Checking for existing report: {report_path}")
        print(f"   Report exists: {report_path.exists()}")

        if report_path.exists():
            print(f"\n⚠️  TraceLens report already exists: {report_path}")
            print(f"   File size: {report_path.stat().st_size / 1024:.2f} KB")

            # Check if running interactively (has stdin) or in background (like web dashboard)
            import sys

            is_interactive = sys.stdin.isatty()

            if is_interactive:
                response = (
                    input(
                        "Do you want to (o)verwrite it or (u)se the existing one? [o/u]: "
                    )
                    .strip()
                    .lower()
                )
                print(f"   Your response: '{response}'")
            else:
                # Non-interactive mode (web dashboard): use existing by default
                print("   Running in non-interactive mode, using existing report")
                response = "u"

            if response == "u":
                print(f"✓ Using existing report: {report_path}")
                # Load existing report
                try:
                    result_dfs = pd.read_excel(
                        report_path, sheet_name=None, engine="openpyxl"
                    )
                    print(f"   Loaded {len(result_dfs)} sheets from existing report")

                    # Debug: Show what data was loaded
                    if "ops_summary_by_category" in result_dfs:
                        ops_df = result_dfs["ops_summary_by_category"]
                        print(
                            f"\n   DEBUG: ops_summary_by_category has {len(ops_df)} rows"
                        )
                        if len(ops_df) > 0:
                            total_time = (
                                ops_df["total_direct_kernel_time_sum"].sum()
                                if "total_direct_kernel_time_sum" in ops_df.columns
                                else 0
                            )
                            print(
                                f"   DEBUG: Total time from categories: {total_time:.2f}ms"
                            )
                            print(
                                f"   DEBUG: Categories: {ops_df['category'].tolist() if 'category' in ops_df.columns else 'N/A'}"
                            )

                    # Set the trace file path
                    if self.use_critical_path:
                        linked_trace = gpu_dir / f"{name}_linked.json"
                    else:
                        linked_trace = Path(kineto)

                    # Store data
                    data_dict.update(result_dfs)
                    data_dict["trace_file"] = linked_trace

                    # For the case when we have nn module classification but no category
                    add_op_cat_column(data_dict)

                    print(
                        f"   DEBUG: Data dict keys after loading: {list(data_dict.keys())}"
                    )
                    print(
                        f"✓ Successfully loaded existing report, skipping regeneration\n"
                    )
                    return
                except Exception as e:
                    print(f"⚠️  Error loading existing report: {e}")
                    print(f"   Falling back to regeneration")
            elif response != "o":
                print(f"⚠️  Invalid input '{response}'. Defaulting to overwrite.")
        else:
            print(f"   No existing report found, will generate new report")

        # Run critical path analysis (if enabled)
        extension_path = None
        if self.use_critical_path:
            linked_trace, cp_pickle = self.tracelens_runner.run_critical_path_analysis(
                kineto, et, gpu_dir, name
            )
            extension_path = (
                Path(__file__).parent.parent
                / "CritPath"
                / "tracelens_critical_path_extension.py"
            )
        else:
            # Use original kineto trace directly (no linking needed)
            linked_trace = Path(kineto)

        # Generate TraceLens report
        result_dfs = self.tracelens_runner.generate_tracelens_report(
            linked_trace, gpu_dir, extension_path, name
        )

        # Store data
        data_dict.update(result_dfs)
        data_dict["trace_file"] = linked_trace
        # For the case when we have nn module classification but no category
        add_op_cat_column(data_dict)

    def _generate_tracelens_diff_report(self):
        """Generate TraceLens TreeDiff comparison report with caching"""

        try:
            print("\n📊 Generating TraceLens TreeDiff report...")

            # Define output directories
            diff_output_dir = self.output_dir / "tracelens_diff"
            diff_pruned_dir = self.output_dir / "tracelens_diff_pruned"

            # Check if cached reports exist
            diff_summary_exists = (diff_output_dir / "diff_stats.csv").exists()
            pruned_summary_exists = (diff_pruned_dir / "diff_stats.csv").exists()

            if diff_summary_exists and pruned_summary_exists:
                print(f"\n⚠️  TraceDiff reports already exist:")
                print(f"   - {diff_output_dir}/")
                print(f"   - {diff_pruned_dir}/")

                # Check if running interactively
                import sys

                is_interactive = sys.stdin.isatty()

                if is_interactive:
                    response = (
                        input("Do you want to (o)verwrite or (u)se existing? [o/u]: ")
                        .strip()
                        .lower()
                    )
                else:
                    # Non-interactive mode: use existing by default
                    print("   Running in non-interactive mode, using existing reports")
                    response = "u"

                if response == "u":
                    print(f"✓ Using existing TraceDiff reports\n")
                    return
                elif response != "o":
                    print(f"⚠️  Invalid input '{response}'. Defaulting to overwrite.")

            # Get trace file paths
            trace_file1 = str(self.gpu1_data.get("trace_file"))
            trace_file2 = str(self.gpu2_data.get("trace_file"))

            if not trace_file1 or not trace_file2:
                print("  ⚠️  Trace files not found in data, skipping TreeDiff")
                return

            print(f"  Loading tree 1: {trace_file1}")
            print(f"  Loading tree 2: {trace_file2}")

            # Build trees
            from TraceLens import TraceDiff, TreePerfAnalyzer

            perf_analyzer1 = TreePerfAnalyzer.from_file(
                trace_file1, add_python_func=True
            )
            perf_analyzer2 = TreePerfAnalyzer.from_file(
                trace_file2, add_python_func=True
            )
            tree1 = perf_analyzer1.tree
            tree2 = perf_analyzer2.tree

            # Generate diff
            td = TraceDiff(tree1, tree2)
            td.generate_diff_stats()
            td.generate_tracediff_report()

            # Write reports
            diff_output_dir.mkdir(parents=True, exist_ok=True)
            td.print_tracediff_report_files(str(diff_output_dir))
            print(f"  ✓ TraceDiff reports written to: {diff_output_dir}/")

            # Also generate pruned (GPU-only) version
            diff_pruned_dir.mkdir(parents=True, exist_ok=True)
            td.print_tracediff_report_files(str(diff_pruned_dir), prune_non_gpu=True)
            print(
                f"  ✓ Pruned TraceDiff reports (GPU only) written to: {diff_pruned_dir}/"
            )

        except Exception as e:
            print(f"  ⚠️  TraceLens TreeDiff generation failed: {e}")
            import traceback

            traceback.print_exc()

    def _extract_gpu_name(self, trace_path: str) -> str:
        """Extract GPU name from trace filename"""
        filename = Path(trace_path).stem

        # Common GPU patterns
        patterns = [
            "MI300",
            "MI300X",
            "MI300A",
            "H100",
            "H200",
            "A100",
            "A6000",
            "V100",
            "GH200",
        ]

        for pattern in patterns:
            if pattern.lower() in filename.lower():
                return pattern

        return "GPU"

    def _extract_summary_data(self) -> Dict:
        """Extract summary data from GPU data"""
        return {
            "gpu1_name": self.gpu1_name,
            "gpu2_name": self.gpu2_name,
            "gpu1_total_latency": self._get_total_time(self.gpu1_data),
            "gpu2_total_latency": self._get_total_time(self.gpu2_data),
        }

    def _get_total_time(self, data_dict: Dict) -> float:
        """Get total execution time from data dictionary"""
        if "gpu_timeline" in data_dict:
            timeline = data_dict["gpu_timeline"]
            # Look for the special 'total_time' row
            if "type" in timeline.columns:
                total_time_row = timeline[timeline["type"] == "total_time"]
                if not total_time_row.empty:
                    time_col = (
                        "time ms" if "time ms" in timeline.columns else "duration_ms"
                    )
                    return float(total_time_row[time_col].values[0])
            # # Fallback to sum if no 'total_time' row exists
            # time_col = 'time ms' if 'time ms' in timeline.columns else 'duration_ms'
            # return float(timeline[time_col].sum())
        return 0.0

    def _generate_executive_summary(
        self, baseline_total: float, target_total: float
    ) -> str:
        """Generate executive summary"""
        gap_ms = abs(baseline_total - target_total)
        gap_pct = (
            (gap_ms / min(baseline_total, target_total) * 100)
            if min(baseline_total, target_total) > 0
            else 0
        )

        if baseline_total > target_total:
            status = f"Baseline is **trailing** by {gap_ms:.2f} ms ({gap_pct:.2f}%)"
        else:
            status = f"Baseline is **leading** by {gap_ms:.2f} ms ({gap_pct:.2f}%)"

        return f"""
**Performance Comparison: {self.gpu2_name} (Baseline) vs {self.gpu1_name} (Target)**

- Baseline Total Time: {baseline_total:.2f} ms
- Target Total Time: {target_total:.2f} ms
- {status}

Analysis mode: {'Critical Path' if self.use_critical_path else 'Timeline'}
"""


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Jarvis Unified GPU Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--target-gpu-kineto",
        required=True,
        help="Target kineto trace (local path or URL)",
    )
    parser.add_argument(
        "--target-gpu-et",
        required=False,
        help="Target execution trace (local path or URL, required for critical path analysis)",
    )
    parser.add_argument(
        "--baseline-gpu-kineto",
        required=True,
        help="Baseline kineto trace (local path or URL)",
    )
    parser.add_argument(
        "--baseline-gpu-et",
        required=False,
        help="Baseline execution trace (local path or URL, required for critical path analysis)",
    )

    # Optional arguments
    parser.add_argument("--target-gpu-name", help="Name for target GPU")
    parser.add_argument("--baseline-gpu-name", help="Name for baseline GPU")
    parser.add_argument(
        "--output-dir", default="trace_reports", help="Output directory"
    )
    parser.add_argument("--api-key", help="AI API key")
    parser.add_argument(
        "--no-save-intermediates",
        action="store_true",
        help="Do not save intermediate files",
    )
    parser.add_argument(
        "--generate-plots",
        "--enable-plots",
        action="store_true",
        dest="generate_plots",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--enable-tracelens-diff",
        action="store_true",
        help="(Legacy flag - TraceLens diff now generated automatically)",
    )
    parser.add_argument(
        "--disable-critical-path",
        action="store_true",
        help="Disable critical path analysis",
    )
    parser.add_argument(
        "--enable-inference-phase-analysis",
        action="store_true",
        help="Enable prefill/decode phase analysis for inference traces",
    )

    args = parser.parse_args()

    # Validate execution traces are provided if critical path is enabled
    use_critical_path = not args.disable_critical_path
    if use_critical_path and (not args.target_gpu_et or not args.baseline_gpu_et):
        print(
            "⚠️  Warning: Execution traces (--target-gpu-et, --baseline-gpu-et) are required for critical path analysis"
        )
        print(
            "   Automatically disabling critical path analysis. Use --disable-critical-path to suppress this warning."
        )
        use_critical_path = False
        args.disable_critical_path = True

    # Create analyzer
    analyzer = JarvisAnalyzer(
        gpu1_kineto=args.target_gpu_kineto,
        gpu1_et=args.target_gpu_et,
        gpu2_kineto=args.baseline_gpu_kineto,
        gpu2_et=args.baseline_gpu_et,
        gpu1_name=args.target_gpu_name,
        gpu2_name=args.baseline_gpu_name,
        output_dir=args.output_dir,
        api_key=args.api_key,
        save_intermediates=not args.no_save_intermediates,
        generate_plots=args.generate_plots,
        use_critical_path=not args.disable_critical_path,
        enable_inference_phase_analysis=args.enable_inference_phase_analysis,
    )

    # Run analysis
    success = analyzer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
