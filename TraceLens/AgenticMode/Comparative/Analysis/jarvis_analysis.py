#!/usr/bin/env python3
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
    import pandas as pd
    
    for sheet_name, df in data_dict.items():
        # Skip non-DataFrame entries (like 'trace_file')
        if not isinstance(df, pd.DataFrame):
            continue
        
        # Check if sheet has 'parent_module' but no 'op category'
        if 'parent_module' in df.columns and 'op category' not in df.columns:
            df['op category'] = df['parent_module']
            print(f"  ✓ Added 'op category' column to '{sheet_name}' (copied from 'parent_module')")


class JarvisAnalyzer:
    """
    Main orchestrator for Jarvis GPU comparison analysis.
    
    Coordinates:
    - TraceLens report generation (tracelens_runner.py)
    - LLM analysis (llm_prompts.py + AI client)
    - Plotting (plotting.py)
    - Report generation (report_generator.py)
    """
    
    def __init__(self,
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
                 enable_inference_phase_analysis: bool = False):
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
        has_timestamp = re.search(r'_\d{8}_\d{6}$', final_dir_name)
        has_vs = '_vs_' in final_dir_name
        
        if has_timestamp and has_vs:
            # Output dir already looks like a complete report directory, use it as-is
            self.output_dir = output_path
            print(f"📁 Using existing output directory: {self.output_dir}")
        else:
            # Create new timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = output_path / f"{self.gpu1_name}_vs_{self.gpu2_name}_{timestamp}"
            print(f"📁 Creating new output directory: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key
        self.save_intermediates = save_intermediates
        self.generate_plots = generate_plots
        self.use_critical_path = use_critical_path
        self.enable_inference_phase_analysis = enable_inference_phase_analysis
        
        # Initialize modules
        self.tracelens_runner = TraceLensRunner(save_intermediates, use_critical_path, enable_inference_phase_analysis)
        self.llm_prompt_manager = LLMPromptManager(use_critical_path)
        
        # AI client setup
        self.client_anthropic = None
        if api_key and AI_AVAILABLE:
            self.client_anthropic = SLAIAnthropic(
                api_key=api_key
            )
        
        # Data storage
        self.gpu1_data = {}
        self.gpu2_data = {}
    
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
        if path.startswith(('http://', 'https://')):
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
        print("\n" + "="*70)
        print("JARVIS UNIFIED ANALYZER")
        print("="*70)
        print(f"GPU 1: {self.gpu1_name}")
        print(f"GPU 2: {self.gpu2_name}")
        print(f"Output: {self.output_dir}")
        print(f"Critical Path: {'Enabled' if self.use_critical_path else 'Disabled'}")
        print("="*70 + "\n")
        
        try:
            # Step 1: Process GPU 1
            print("🔄 Processing GPU 1...")
            self._process_gpu(
                self.gpu1_kineto,
                self.gpu1_et,
                self.gpu1_name,
                self.gpu1_data
            )
            
            # Step 2: Process GPU 2
            print("\n🔄 Processing GPU 2...")
            self._process_gpu(
                self.gpu2_kineto,
                self.gpu2_et,
                self.gpu2_name,
                self.gpu2_data
            )
            
            # Step 3: add tracediff-based info
            #self._generate_tracelens_diff_report()            

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
            
            # Step 3.5: Generate unified comparison markdown (lightweight for LLM)
            print("\n📝 Generating unified comparison markdown (lightweight)...")
            from markdown_generator import MarkdownGenerator
            
            if self.gpu1_data and self.gpu2_data:
                # Use whitelist mode to only include essential sheets for LLM
                markdown_gen = MarkdownGenerator(
                    included_sheets=['gpu_timeline', 'ops_summary', 'ops_summary_by_category', 'ops_unique_args']
                )
                comparison_md_path = self.output_dir / "comparison_report.md"
                
                markdown_gen.generate_comparison_markdown(
                    gpu1_dfs=self.gpu2_data,  # baseline
                    gpu2_dfs=self.gpu1_data,  # target
                    gpu1_name=self.gpu2_name,  # baseline
                    gpu2_name=self.gpu1_name,  # target
                    output_path=comparison_md_path,
                    max_rows_per_sheet=30,  # Reduced from 50
                    max_rows_ops_unique_args=10  # ops_unique_args is extremely verbose
                )
                print(f"✓ Unified comparison markdown: {comparison_md_path}")
            
            

            # Step 4: AI Analysis
            ai_analysis = None
            if self.client_anthropic:
                print("\n🤖 Performing AI analysis...")
                ai_analysis = self._run_ai_analysis()
            

            
        

            # Step 5: Generate plots (if enabled)
            plot_paths = []
            if self.generate_plots:
                print("\n📈 Generating plots...")
                plotter = JarvisPlotter(
                    self.output_dir,
                    self.gpu1_name,
                    self.gpu2_name,
                    self.gpu1_data,
                    self.gpu2_data,
                    self.use_critical_path
                )
                plot_paths = plotter.generate_all_plots()
                
                # Generate cumulative optimization chart if AI analysis is available
                if ai_analysis:
                    cumulative_plot = plotter.generate_cumulative_optimization_progression_chart(ai_analysis)
                    if cumulative_plot:
                        plot_paths.append(cumulative_plot)

                    cumulative_table = plotter.generate_optimization_opportunities_table(ai_analysis)
                    if cumulative_table:
                        plot_paths.append(cumulative_table)

                        import pandas as pd
                        df = pd.read_csv(cumulative_table)

                        try:
                            comments = self.expand_table(ai_analysis, df.to_string())
                            if comments and len(comments) == len(df):
                                df['Comments'] = comments
                                cumulative_table_expanded = cumulative_table.parent / "optimization_opportunities_table_expanded.csv"
                                df.to_csv(cumulative_table_expanded, index=False)
                                plot_paths.append(cumulative_table_expanded)
                                print(f"Expanded table.")

                                
                                html_path = visualizations.generate_interactive_optimization_html(cumulative_table_expanded)
                                if html_path:
                                    plot_paths.append(html_path)

                        except Exception as e:
                            print(f"Failed to expand table: {e}")



            # Step 6: Generate final report
            print("\n📄 Generating final report...")
            report_gen = ReportGenerator(
                self.output_dir,
                self.gpu1_name,
                self.gpu2_name,
                self.use_critical_path
            )
            
            # Get totals for report
            baseline_total = self._get_total_time(self.gpu2_data)
            target_total = self._get_total_time(self.gpu1_data)
            
            executive_summary = self._generate_executive_summary(
                baseline_total,
                target_total
            )
            
            report_path = report_gen.generate_report(
                ai_analysis,
                executive_summary,
                plot_paths,
                baseline_total,
                target_total
            )
            
            print("\n" + "="*70)
            print("✅ ANALYSIS COMPLETE")
            print("="*70)
            print(f"Report: {report_path}")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
                response = input("Do you want to (o)verwrite it or (u)se the existing one? [o/u]: ").strip().lower()
                print(f"   Your response: '{response}'")
            else:
                # Non-interactive mode (web dashboard): use existing by default
                print("   Running in non-interactive mode, using existing report")
                response = 'u'
            
            if response == 'u':
                print(f"✓ Using existing report: {report_path}")
                # Load existing report
                import pandas as pd
                try:
                    result_dfs = pd.read_excel(report_path, sheet_name=None, engine='openpyxl')
                    print(f"   Loaded {len(result_dfs)} sheets from existing report")
                    
                    # Debug: Show what data was loaded
                    if 'ops_summary_by_category' in result_dfs:
                        ops_df = result_dfs['ops_summary_by_category']
                        print(f"\n   DEBUG: ops_summary_by_category has {len(ops_df)} rows")
                        if len(ops_df) > 0:
                            total_time = ops_df['total_direct_kernel_time_sum'].sum() if 'total_direct_kernel_time_sum' in ops_df.columns else 0
                            print(f"   DEBUG: Total time from categories: {total_time:.2f}ms")
                            print(f"   DEBUG: Categories: {ops_df['category'].tolist() if 'category' in ops_df.columns else 'N/A'}")
                    
                    # Set the trace file path
                    if self.use_critical_path:
                        linked_trace = gpu_dir / f"{name}_linked.json"
                    else:
                        linked_trace = Path(kineto)
                    
                    # Store data
                    data_dict.update(result_dfs)
                    data_dict['trace_file'] = linked_trace


                    # For the case when we have nn module classification but no category
                    add_op_cat_column(data_dict)

                    print(f"   DEBUG: Data dict keys after loading: {list(data_dict.keys())}")
                    print(f"✓ Successfully loaded existing report, skipping regeneration\n")
                    return
                except Exception as e:
                    print(f"⚠️  Error loading existing report: {e}")
                    print(f"   Falling back to regeneration")
            elif response != 'o':
                print(f"⚠️  Invalid input '{response}'. Defaulting to overwrite.")
        else:
            print(f"   No existing report found, will generate new report")
        
        # Run critical path analysis (if enabled)
        extension_path = None
        if self.use_critical_path:
            linked_trace, cp_pickle = self.tracelens_runner.run_critical_path_analysis(
                kineto, et, gpu_dir, name
            )
            extension_path = Path(__file__).parent.parent / "CritPath" / "tracelens_critical_path_extension.py"
        else:
            # Use original kineto trace directly (no linking needed)
            linked_trace = Path(kineto)
        
        # Generate TraceLens report
        result_dfs = self.tracelens_runner.generate_tracelens_report(
            linked_trace,
            gpu_dir,
            extension_path,
            name
        )
        
        # Store data
        data_dict.update(result_dfs)
        data_dict['trace_file'] = linked_trace
        # For the case when we have nn module classification but no category
        add_op_cat_column(data_dict)
    
    def _generate_tracelens_diff_report(self):
        """Generate TraceLens TreeDiff comparison report"""
        try:
            print("\n📊 Generating TraceLens TreeDiff report...")
           
            
            # Get trace file paths
            trace_file1 = str(self.gpu1_data.get('trace_file'))
            trace_file2 = str(self.gpu2_data.get('trace_file'))
            
            if not trace_file1 or not trace_file2:
                print("  ⚠️  Trace files not found in data, skipping TreeDiff")
                return
            
            print(f"  Loading tree 1: {trace_file1}")
            print(f"  Loading tree 2: {trace_file2}")
            
            # Build trees
            perf_analyzer1 = TreePerfAnalyzer.from_file(trace_file1)
            perf_analyzer2 = TreePerfAnalyzer.from_file(trace_file2)
            tree1 = perf_analyzer1.tree
            tree2 = perf_analyzer2.tree
            
            # Generate diff
            td = TraceDiff(tree1, tree2)
            td.generate_tracediff_report()
            
            # Write reports
            diff_output_dir = self.output_dir / "tracelens_diff"
            diff_output_dir.mkdir(parents=True, exist_ok=True)
            
            td.print_tracediff_report_files(str(diff_output_dir))
            print(f"  ✓ TraceDiff reports written to: {diff_output_dir}/")
            
            # Also generate pruned (GPU-only) version
            diff_pruned_dir = self.output_dir / "tracelens_diff_pruned"
            diff_pruned_dir.mkdir(parents=True, exist_ok=True)
            
            td.print_tracediff_report_files(str(diff_pruned_dir), prune_non_gpu=True)
            print(f"  ✓ Pruned TraceDiff reports (GPU only) written to: {diff_pruned_dir}/")
            
        except Exception as e:
            print(f"  ⚠️  TraceLens TreeDiff generation failed: {e}")
            import traceback
            traceback.print_exc()
        
    def expand_table(self, ai_analysis, table):
        
        PROMPT = '''You will get a written analysis report and a table generated based on it.
        Your task is: for each row in the table, find key written recommendations/comments from the report and move them into a list, with minimal changes (except as specified below).
        
        The output should be in the following form:
        comments for row 1***comments for row 2***comments for row 3 etc.
        
        Do not add new comments or expand on existing ones. If no relevant comment is present, output "No comment" for that particular row. Do not use the triple * (***) anywhere else in the output. If the analysis text had triple *** in it, escape or substitute something appropriate instead.

        If recommendations for certain operations are distributed across multiple parts of the report (for example, a brief recommendation in a table, and a verbal comment in a separate place), integrate the recommendations in one brief paragraph, staying as close to the original as reasonably possible.
        You can use markdown.
        
        Be mindful of target and baseline architectures.
        In general, we are trying to improve baseline. So if, for example, baseline is mi300, make sure that recommendations are relevant to AMD architectures. Rephrase or remove recommendations that are not appropriate to the manufacturer. For example, a change is needed when a recommendation mentions CUTLASS when the baseline is AMD, or when a recommendation mention RocBLAS when the baseline is NVIDIA.

        Output ONLY the comments, start right away.

        '''
        PROMPT += f"### ANALYSIS TEXT ###\n{ai_analysis}\n### END OF ANALYSIS TEXT ###"
        PROMPT += f"### RESULT TABLE ### {table} ### END OF RESULT TABLE ### \nExtracted comments:\n"

        try:
            response = self.client_anthropic.messages.create(
                model="claude-sonnet-4",
                max_tokens=5000,
                temperature=0,
                messages=[
                    {"role": "system", "content": """You are helping to aggregate different sources of a GPU workload analysis."""},
                    {"role": "user", "content": PROMPT}
                ]
            )
            new_row = response.content[0].text.strip().split("***")
            return new_row
        
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return None

    def _run_ai_analysis(self) -> Optional[str]:
        """Run AI analysis on the comparison data"""
        # Get totals first
        baseline_total = self._get_total_time(self.gpu2_data)
        target_total = self._get_total_time(self.gpu1_data)
        
        # Extract all data structures
        summary_data = self._extract_summary_data()
        gemm_data = data_extractors.extract_gemm_data(self.gpu1_data, self.gpu2_data, 
                                                       self.gpu1_name, self.gpu2_name)
        conv_data = data_extractors.extract_conv_data(self.gpu1_data, self.gpu2_data,
                                                      self.gpu1_name, self.gpu2_name)
        ops_summary = data_extractors.extract_ops_summary(self.gpu1_data, self.gpu2_data,
                                                                self.gpu1_name, self.gpu2_name)
        overall_data = data_extractors.extract_overall_data(self.gpu1_data, self.gpu2_data)
        
        detailed_comparison = data_extractors.extract_detailed_comparison(self.gpu1_data, self.gpu2_data)
        
        ops_unique_args = data_extractors.extract_ops_unique_args_comparison(self.gpu1_data, self.gpu2_data)

        ops_summary_by_category = data_extractors.extract_ops_summary_by_category(self.gpu1_data, self.gpu2_data)
        
        # Build critical path section
        critical_path_section = ""
        if self.use_critical_path:
            critical_path_section = "Critical path data available - analyze operations on the critical path from the markdown tables above"
        else:
            critical_path_section = "Critical path analysis disabled - focus on timeline data and category-level aggregations from the markdown tables above"
        
        # Identify slow categories (where baseline is trailing) and extract detailed operations
        # Extract category comparison data directly from DataFrames for this analysis
        detailed_slow_ops = {}
        slow_categories = ['GEMM', 'CONV_fwd', 'CONV_bwd', 'BN_fwd', 'BN_bwd', 'flash_attention', 
                          'elementwise', 'reduction', 'NCCL']
        
        # Get category comparison from DataFrames
        if 'ops_summary_by_category' in self.gpu2_data and 'ops_summary_by_category' in self.gpu1_data:
            baseline_cats = self.gpu2_data['ops_summary_by_category']
            target_cats = self.gpu1_data['ops_summary_by_category']
            
            if not baseline_cats.empty and not target_cats.empty:
                # Merge on category to compare
                import pandas as pd
                merged = pd.merge(
                    baseline_cats[['op category', 'total_direct_kernel_time_ms']],
                    target_cats[['op category', 'total_direct_kernel_time_ms']],
                    on='op category',
                    suffixes=('_baseline', '_target'),
                    how='outer'
                ).fillna(0)
                
                # Calculate gaps
                merged['gap_ms'] = merged['total_direct_kernel_time_ms_baseline'] - merged['total_direct_kernel_time_ms_target']
                
                # Identify slow categories (positive gap = baseline slower)
                slow_cats_df = merged[merged['gap_ms'] > 0]
                
                for _, row in slow_cats_df.iterrows():
                    category = row['op category']
                    gap_ms = row['gap_ms']
                    
                    if category in slow_categories:
                        print(f"  Detected slow category: {category} (gap: {gap_ms:.2f}ms)")
                        # Extract top 10 operations for this category
                        ops_details = data_extractors.extract_detailed_operations_for_category(
                            self.gpu2_data, self.gpu1_data, category, top_n=10
                        )
                        if ops_details:
                            detailed_slow_ops[category] = ops_details
                            print(f"    Extracted {len(ops_details)} detailed operations for {category}")
        
        # Check if unified comparison markdown is available (REQUIRED for prompt)
        comparison_md_path = self.output_dir / "comparison_report.md"
        
        if not comparison_md_path.exists():
            raise FileNotFoundError(f"Comparison markdown not found: {comparison_md_path}. This is required for LLM analysis.")
        
        print("\n📋 Loading comparison markdown for detailed data...")
        with open(comparison_md_path, 'r', encoding='utf-8') as f:
            markdown_data_section = f.read()
        print(f"✓ Loaded comparison markdown: {comparison_md_path} ({len(markdown_data_section)} chars)")
        


        # # Load TraceLens diff summary if available
        # diff_summary_path = self.output_dir / "tracelens_diff" / "diff_stats_summary.csv"
        # if diff_summary_path.exists():
        #     import pandas as pd
        #     diff_df = pd.read_csv(diff_summary_path)
        #     diff_summary = f"\n\n## TRACE DIFF SUMMARY \n\n```\n{diff_df.to_string()}\n```\n"
        #     markdown_data_section += diff_summary
        #     print(f"✓ Added TraceLens diff summary to analysis")

        # Build prompt with markdown data
        print("\n📋 Building analysis prompt...")
        prompt = self.llm_prompt_manager.build_analysis_prompt(
            baseline_gpu=self.gpu2_name,
            target_gpu=self.gpu1_name,
            baseline_total_time=baseline_total,
            target_total_time=target_total,
            critical_path_section=critical_path_section,
            markdown_data_section=markdown_data_section,
        )
        print(f"✓ Built prompt ({len(prompt)} chars)")
        
        # Save prompt for debugging
        prompt_file = self.output_dir / "llm_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ Prompt saved to: {prompt_file}")

        # Get system message
        system_message = self.llm_prompt_manager.get_system_message()
        
        # Call LLM
        try:
            response = self.client_anthropic.messages.create(
                model="claude-sonnet-4",
                max_tokens=5000,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return None
    
    def _extract_gpu_name(self, trace_path: str) -> str:
        """Extract GPU name from trace filename"""
        filename = Path(trace_path).stem
        
        # Common GPU patterns
        patterns = [
            'MI300', 'MI300X', 'MI300A',
            'H100', 'H200', 'A100', 'A6000',
            'V100', 'GH200'
        ]
        
        for pattern in patterns:
            if pattern.lower() in filename.lower():
                return pattern
        
        return "GPU"
    
    def _extract_summary_data(self) -> Dict:
        """Extract summary data from GPU data"""
        return {
            'gpu1_name': self.gpu1_name,
            'gpu2_name': self.gpu2_name,
            'gpu1_total_latency': self._get_total_time(self.gpu1_data),
            'gpu2_total_latency': self._get_total_time(self.gpu2_data)
        }
    
    def _get_total_time(self, data_dict: Dict) -> float:
        """Get total execution time from data dictionary"""
        if 'gpu_timeline' in data_dict:
            timeline = data_dict['gpu_timeline']
            # Look for the special 'total_time' row
            if 'type' in timeline.columns:
                total_time_row = timeline[timeline['type'] == 'total_time']
                if not total_time_row.empty:
                    time_col = 'time ms' if 'time ms' in timeline.columns else 'duration_ms'
                    return float(total_time_row[time_col].values[0])
            # # Fallback to sum if no 'total_time' row exists
            # time_col = 'time ms' if 'time ms' in timeline.columns else 'duration_ms'
            # return float(timeline[time_col].sum())
        return 0.0
    
    def _generate_executive_summary(self, baseline_total: float, target_total: float) -> str:
        """Generate executive summary"""
        gap_ms = abs(baseline_total - target_total)
        gap_pct = (gap_ms / min(baseline_total, target_total) * 100) if min(baseline_total, target_total) > 0 else 0
        
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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--gpu1-kineto', required=True, 
                       help='GPU1 kineto trace (local path or URL)')
    parser.add_argument('--gpu1-et', required=False, 
                       help='GPU1 execution trace (local path or URL, required for critical path analysis)')
    parser.add_argument('--gpu2-kineto', required=True, 
                       help='GPU2 kineto trace (local path or URL)')
    parser.add_argument('--gpu2-et', required=False, 
                       help='GPU2 execution trace (local path or URL, required for critical path analysis)')
    
    # Optional arguments
    parser.add_argument('--gpu1-name', help='Name for GPU1')
    parser.add_argument('--gpu2-name', help='Name for GPU2')
    parser.add_argument('--output-dir', default='trace_reports', help='Output directory')
    parser.add_argument('--api-key', help='AI API key')
    parser.add_argument('--no-save-intermediates', action='store_true', 
                       help='Do not save intermediate files')
    parser.add_argument('--generate-plots', '--enable-plots', action='store_true', 
                       dest='generate_plots',
                       help='Generate visualization plots')
    parser.add_argument('--enable-tracelens-diff', action='store_true',
                       help='(Legacy flag - TraceLens diff now generated automatically)')
    parser.add_argument('--disable-critical-path', action='store_true',
                       help='Disable critical path analysis')
    parser.add_argument('--enable-inference-phase-analysis', action='store_true',
                       help='Enable prefill/decode phase analysis for inference traces')
    
    args = parser.parse_args()
    
    # Validate execution traces are provided if critical path is enabled
    use_critical_path = not args.disable_critical_path
    if use_critical_path and (not args.gpu1_et or not args.gpu2_et):
        print("⚠️  Warning: Execution traces (--gpu1-et, --gpu2-et) are required for critical path analysis")
        print("   Automatically disabling critical path analysis. Use --disable-critical-path to suppress this warning.")
        use_critical_path = False
        args.disable_critical_path = True
    
    # Create analyzer
    analyzer = JarvisAnalyzer(
        gpu1_kineto=args.gpu1_kineto,
        gpu1_et=args.gpu1_et,
        gpu2_kineto=args.gpu2_kineto,
        gpu2_et=args.gpu2_et,
        gpu1_name=args.gpu1_name,
        gpu2_name=args.gpu2_name,
        output_dir=args.output_dir,
        api_key=args.api_key,
        save_intermediates=not args.no_save_intermediates,
        generate_plots=args.generate_plots,
        use_critical_path=not args.disable_critical_path,
        enable_inference_phase_analysis=args.enable_inference_phase_analysis
    )
    
    # Run analysis
    success = analyzer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
