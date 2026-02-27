#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Report Generator Module
Handles final markdown report generation
"""

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class ReportGenerator:
    """Generates final markdown report from analysis results"""
    
    def __init__(self, output_dir: Path, gpu1_name: str, gpu2_name: str,
                 use_critical_path: bool = True):
        """
        Args:
            output_dir: Output directory for reports
            gpu1_name: Name of GPU 1
            gpu2_name: Name of GPU 2
            use_critical_path: Whether critical path analysis is enabled
        """
        self.output_dir = output_dir
        self.gpu1_name = gpu1_name
        self.gpu2_name = gpu2_name
        self.use_critical_path = use_critical_path
        self.summary_file = output_dir / "analysis_summary.md"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self,
                       ai_analysis: Optional[str],
                       executive_summary: str,
                       plot_paths: List[Path],
                       baseline_total_ms: float,
                       target_total_ms: float) -> Path:
        """
        Generate the final markdown report.
        
        Args:
            ai_analysis: AI-generated analysis text
            executive_summary: Executive summary text
            plot_paths: List of paths to plots to embed
            baseline_total_ms: Baseline total execution time
            target_total_ms: Target total execution time
            
        Returns:
            Path to generated report
        """
        with open(self.summary_file, 'w') as f:
            # Write header
            self._write_header(f)
            
            # # Write executive summary
            # f.write("## Executive Summary\n\n")
            # f.write(executive_summary)
            # f.write("\n\n")

            # Embed plots - show gap analysis chart first, then cumulative optimization progression
            if plot_paths:
                # Find the gap analysis categories chart
                gap_chart = None
                cumulative_plot = None
                
                for plot_path in plot_paths:
                    if plot_path and plot_path.exists():
                        if 'gap_analysis_categories' in plot_path.stem.lower():
                            gap_chart = plot_path
                        elif 'cumulative' in plot_path.stem.lower():
                            cumulative_plot = plot_path
                
                # Show gap analysis chart first
                if gap_chart:
                    f.write("## Category Performance Gap Analysis\n\n")
                    rel_path = gap_chart.relative_to(self.output_dir) if gap_chart.is_relative_to(self.output_dir) else gap_chart
                    f.write(f"![{gap_chart.stem}]({rel_path})\n\n")
                
                # Then show cumulative optimization progression chart
                if cumulative_plot:
                    f.write("## Cumulative Performance Optimization Progression \n\n")
                    rel_path = cumulative_plot.relative_to(self.output_dir) if cumulative_plot.is_relative_to(self.output_dir) else cumulative_plot
                    f.write(f"![{cumulative_plot.stem}]({rel_path})\n\n")

            # Write AI analysis if available
            if ai_analysis:
                f.write(ai_analysis)
                f.write("\n\n")
            
            # Write footer
            # self._write_footer(f, baseline_total_ms, target_total_ms)
        
        print(f"\n✅ Report generated: {self.summary_file}")
        return self.summary_file
    
    def _write_header(self, f):
        """Write report header"""
        analysis_mode = "Critical Path Analysis" if self.use_critical_path else "Timeline Analysis"
        
        f.write(f"# GAP Analysis with Potential Optimization Opportunities ({analysis_mode})\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Analysis Mode:** {analysis_mode}\n\n")
        
        if not self.use_critical_path:
            f.write("*Note: Critical path analysis is disabled. Analysis based on timeline and operation-level data.*\n\n")
        
        f.write("---\n\n")
    
    def _write_footer(self, f, baseline_total_ms: float, target_total_ms: float):
        """Write report footer"""
        f.write("\n---\n\n")
        f.write("## Performance Summary\n\n")
        f.write(f"- **Baseline Total Time:** {baseline_total_ms:.2f} ms\n")
        f.write(f"- **Target Total Time:** {target_total_ms:.2f} ms\n")
        
        gap_ms = abs(baseline_total_ms - target_total_ms)
        gap_pct = (gap_ms / min(baseline_total_ms, target_total_ms) * 100) if min(baseline_total_ms, target_total_ms) > 0 else 0
        
        if baseline_total_ms > target_total_ms:
            f.write(f"- **Gap:** {gap_ms:.2f} ms ({gap_pct:.2f}% - Baseline trailing)\n")
        else:
            f.write(f"- **Gap:** {gap_ms:.2f} ms ({gap_pct:.2f}% - Baseline leading)\n")
        
        f.write(f"\n*Analysis Mode: {self.analysis_mode}*\n")
        f.write(f"\n*Report generated by Jarvis Unified Analyzer*\n")
    
    @property
    def analysis_mode(self) -> str:
        """Get analysis mode string"""
        return "Critical Path Analysis" if self.use_critical_path else "Timeline Analysis"
    
    def add_section(self, title: str, content: str):
        """
        Append a section to the report.
        
        Args:
            title: Section title
            content: Section content
        """
        with open(self.summary_file, 'a') as f:
            f.write(f"\n## {title}\n\n")
            f.write(content)
            f.write("\n\n")
    
    def embed_plot(self, plot_path: Path, title: Optional[str] = None):
        """
        Embed a single plot in the report.
        
        Args:
            plot_path: Path to plot image
            title: Optional title for the plot section
        """
        if not plot_path or not plot_path.exists():
            return
        
        with open(self.summary_file, 'a') as f:
            if title:
                f.write(f"\n### {title}\n\n")
            else:
                f.write(f"\n### {plot_path.stem.replace('_', ' ').title()}\n\n")
            
            rel_path = plot_path.relative_to(self.output_dir) if plot_path.is_relative_to(self.output_dir) else plot_path
            f.write(f"![{plot_path.stem}]({rel_path})\n\n")
