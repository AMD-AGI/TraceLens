#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Markdown Generator Module
Converts TraceLens DataFrames to formatted markdown for LLM consumption
"""

from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path


class MarkdownGenerator:
    """Generates formatted markdown from TraceLens report DataFrames"""
    
    # Sheets to exclude from markdown generation
    EXCLUDED_SHEETS = ['GEMM', 'CONV_fwd', 'CONV_bwd']
    
    # Sheets to include for LLM prompt (lightweight version)
    LLM_INCLUDED_SHEETS = ['gpu_timeline', 'ops_summary', 'ops_summary_by_category', 'ops_unique_args']
    
    def __init__(self, excluded_sheets: Optional[List[str]] = None, 
                 included_sheets: Optional[List[str]] = None):
        """
        Args:
            excluded_sheets: List of sheet names to exclude (defaults to GEMM, CONV_fwd, CONV_bwd)
            included_sheets: If provided, ONLY these sheets will be included (overrides excluded_sheets)
        """
        self.excluded_sheets = excluded_sheets or self.EXCLUDED_SHEETS
        self.included_sheets = included_sheets  # If set, use whitelist instead of blacklist
    
    def generate_markdown(self, result_dfs: Dict[str, pd.DataFrame], 
                         output_path: Optional[Path] = None,
                         gpu_name: str = "GPU",
                         max_rows_per_sheet: int = 100,
                         max_rows_ops_unique_args: int = 20) -> str:
        """
        Generate markdown from TraceLens result DataFrames.
        
        Args:
            result_dfs: Dictionary of sheet_name -> DataFrame from TraceLens
            output_path: Optional path to save markdown file
            gpu_name: Name of the GPU for the report
            max_rows_per_sheet: Maximum rows to include per sheet (to avoid token overflow)
            max_rows_ops_unique_args: Special limit for ops_unique_args (default 20, it's very verbose)
            
        Returns:
            Generated markdown string
        """
        markdown_parts = []
        
        # Header
        markdown_parts.append(f"# TraceLens Performance Report: {gpu_name}\n")
        markdown_parts.append("=" * 80)
        markdown_parts.append("\n")
        
        # Determine which sheets to include
        if self.included_sheets:
            # Whitelist mode: only include specified sheets
            included_sheets = [name for name in result_dfs.keys() 
                             if name in self.included_sheets and 
                             isinstance(result_dfs[name], pd.DataFrame) and 
                             not result_dfs[name].empty]
        else:
            # Blacklist mode: exclude specified sheets
            included_sheets = [name for name in result_dfs.keys() 
                             if name not in self.excluded_sheets and 
                             isinstance(result_dfs[name], pd.DataFrame) and 
                             not result_dfs[name].empty]
        
        if included_sheets:
            markdown_parts.append("## Table of Contents\n")
            for i, sheet_name in enumerate(included_sheets, 1):
                markdown_parts.append(f"{i}. [{sheet_name}](#{sheet_name.lower().replace(' ', '-').replace('_', '-')})")
            markdown_parts.append("\n")
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n\n")
        
        # Generate markdown for each sheet
        for sheet_name in included_sheets:
            df = result_dfs[sheet_name]
            
            # Add sheet header
            markdown_parts.append(f"## {sheet_name}\n")
            markdown_parts.append("-" * 80)
            markdown_parts.append("\n")
            
            # Add metadata
            markdown_parts.append(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}\n\n")
            
            # Special handling for ops_unique_args - it's extremely verbose
            if sheet_name == 'ops_unique_args':
                row_limit = max_rows_ops_unique_args
            else:
                row_limit = max_rows_per_sheet
            
            # Truncate if too many rows
            df_display = df.head(row_limit) if len(df) > row_limit else df
            
            if len(df) > row_limit:
                markdown_parts.append(f"*Showing first {row_limit} of {len(df)} rows*\n\n")
            
            # Convert DataFrame to markdown table
            try:
                # Format numeric columns to reasonable precision
                df_formatted = df_display.copy()
                
                # Truncate very long string columns (like kernel names)
                for col in df_formatted.select_dtypes(include=['object']).columns:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: (str(x)[:100] + '...') if isinstance(x, str) and len(str(x)) > 100 else x
                    )
                
                # Format numeric columns
                for col in df_formatted.select_dtypes(include=['float64', 'float32']).columns:
                    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                
                # Drop columns with list/dict data for ops_unique_args (too verbose)
                if sheet_name == 'ops_unique_args':
                    cols_to_drop = [col for col in df_formatted.columns 
                                   if df_formatted[col].apply(lambda x: isinstance(x, (list, dict))).any()]
                    if cols_to_drop:
                        df_formatted = df_formatted.drop(columns=cols_to_drop)
                        markdown_parts.append(f"*Note: Dropped {len(cols_to_drop)} verbose columns: {', '.join(cols_to_drop[:3])}{'...' if len(cols_to_drop) > 3 else ''}*\n\n")
                
                table_md = df_formatted.to_markdown(index=False, tablefmt='github')
                markdown_parts.append(table_md)
            except Exception as e:
                # Fallback if markdown conversion fails
                markdown_parts.append(f"*Error converting table: {e}*\n")
                markdown_parts.append(f"```\n{df_display.to_string()}\n```")
            
            markdown_parts.append("\n\n")
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n\n")
        
        # Footer
        markdown_parts.append("---\n")
        markdown_parts.append(f"*Generated for {gpu_name}*\n") #  | Excluded sheets: {', '.join(self.excluded_sheets)}*\n")
        # TODO: rm excluded sheets mention

        # Join all parts
        markdown_content = "\n".join(markdown_parts)
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✓ Markdown report saved: {output_path}")
        
        return markdown_content
    
    def generate_comparison_markdown(self, 
                                 gpu1_dfs: Dict[str, pd.DataFrame],
                                 gpu2_dfs: Dict[str, pd.DataFrame],
                                 gpu1_name: str,
                                 gpu2_name: str,
                                 output_path: Optional[Path] = None,
                                 projection_table_paths: Optional[Dict[str, Path]] = None,
                                 max_rows_per_sheet: int = 50,
                                 max_rows_ops_unique_args: int = 10) -> str:
        """
        Generate side-by-side comparison markdown for two GPUs in a single file.
        
        Args:
            gpu1_dfs: DataFrames for GPU1 (baseline)
            gpu2_dfs: DataFrames for GPU2 (target)
            gpu1_name: Name of GPU1 (baseline)
            gpu2_name: Name of GPU2 (target)
            output_path: Optional path to save markdown
            projection_table_paths: Optional dict mapping category names to CSV paths with optimization opportunities
            max_rows_per_sheet: Max rows per sheet
            max_rows_ops_unique_args: Special limit for ops_unique_args (very verbose)
            
        Returns:
            Generated markdown string
        """
        markdown_parts = []
        
        # Header
        markdown_parts.append(f"# TraceLens Comparison: {gpu1_name} (Baseline) vs {gpu2_name} (Target)\n")
        markdown_parts.append("=" * 80)
        markdown_parts.append("\n\n")
        
        # Get all sheets from both GPUs - use whitelist if provided, otherwise blacklist
        gpu1_sheets = set()
        for key, value in gpu1_dfs.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                if self.included_sheets:
                    # Whitelist mode
                    if key in self.included_sheets:
                        gpu1_sheets.add(key)
                else:
                    # Blacklist mode
                    if key not in self.excluded_sheets:
                        gpu1_sheets.add(key)
        
        gpu2_sheets = set()
        for key, value in gpu2_dfs.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                if self.included_sheets:
                    # Whitelist mode
                    if key in self.included_sheets:
                        gpu2_sheets.add(key)
                else:
                    # Blacklist mode
                    if key not in self.excluded_sheets:
                        gpu2_sheets.add(key)
        
        # Get all unique sheets (union)
        all_sheets = sorted(gpu1_sheets.union(gpu2_sheets))
        common_sheets = sorted(gpu1_sheets.intersection(gpu2_sheets))
        gpu1_only = sorted(gpu1_sheets - gpu2_sheets)
        gpu2_only = sorted(gpu2_sheets - gpu1_sheets)
        
        # Summary section
        markdown_parts.append("## Summary\n")
        markdown_parts.append(f"- **Total sheets for comparison:** {len(all_sheets)}\n")
        markdown_parts.append(f"- **Common sheets:** {len(common_sheets)}\n")
        markdown_parts.append(f"- **{gpu1_name} only:** {len(gpu1_only)}\n")
        markdown_parts.append(f"- **{gpu2_name} only:** {len(gpu2_only)}\n")
        #markdown_parts.append(f"- **Excluded sheets:** {', '.join(self.excluded_sheets)}\n")
        markdown_parts.append("\n")
        markdown_parts.append("=" * 80)
        markdown_parts.append("\n\n")
        
        # Table of contents
        if all_sheets:
            markdown_parts.append("## Table of Contents\n")
            for i, sheet_name in enumerate(all_sheets, 1):
                markdown_parts.append(f"{i}. [{sheet_name}](#{sheet_name.lower().replace(' ', '-').replace('_', '-')})")
            if projection_table_paths:
                markdown_parts.append(f"{len(all_sheets) + 1}. [Optimization Projections](#optimization-projections)")
            markdown_parts.append("\n")
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n\n")
        
        # Generate comparison for each sheet
        for sheet_name in all_sheets:
            markdown_parts.append(f"## {sheet_name}\n")
            markdown_parts.append("-" * 80)
            markdown_parts.append("\n")
            
            has_gpu1 = sheet_name in gpu1_sheets
            has_gpu2 = sheet_name in gpu2_sheets
            
            # Show availability status
            if has_gpu1 and has_gpu2:
                markdown_parts.append(f"**Status:** Available in both {gpu1_name} and {gpu2_name}\n\n")
            elif has_gpu1:
                markdown_parts.append(f"**Status:** Available only in {gpu1_name}\n\n")
            else:
                markdown_parts.append(f"**Status:** Available only in {gpu2_name}\n\n")
            
            # GPU1/Baseline section
            if has_gpu1:
                markdown_parts.append(f"### 📊 {gpu1_name} (Baseline)\n")
                df1 = gpu1_dfs[sheet_name]
                
                # Special row limit for ops_unique_args
                row_limit = max_rows_ops_unique_args if sheet_name == 'ops_unique_args' else max_rows_per_sheet
                df1_display = df1.head(row_limit) if len(df1) > row_limit else df1
                
                

                markdown_parts.append(f"**Rows:** {len(df1)} | **Columns:** {len(df1.columns)}")
                if len(df1) > row_limit:
                    markdown_parts.append(f" | *Showing first {row_limit} rows*")
                markdown_parts.append("\n\n")
                
                try:
                    # Format data
                    df1_formatted = df1_display.copy()
                    
                    # Truncate long strings
                    for col in df1_formatted.select_dtypes(include=['object']).columns:
                        df1_formatted[col] = df1_formatted[col].apply(
                            lambda x: (str(x)[:100] + '...') if isinstance(x, str) and len(str(x)) > 100 else x
                        )
                    
                    # Format numerics
                    for col in df1_formatted.select_dtypes(include=['float64', 'float32']).columns:
                        df1_formatted[col] = df1_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                    
                    # Drop verbose columns for ops_unique_args
                    if sheet_name == 'ops_unique_args':
                        cols_to_drop = [col for col in df1_formatted.columns 
                                    if df1_formatted[col].apply(lambda x: isinstance(x, (list, dict))).any()]
                        if cols_to_drop:
                            df1_formatted = df1_formatted.drop(columns=cols_to_drop)
                    
                    table_md = df1_formatted.to_markdown(index=False, tablefmt='github')
                    markdown_parts.append(table_md)
                except Exception as e:
                    markdown_parts.append(f"*Error converting table: {e}*\n")
                    markdown_parts.append(f"```\n{df1_display.to_string()}\n```")
                markdown_parts.append("\n\n")
            else:
                markdown_parts.append(f"### 📊 {gpu1_name} (Baseline)\n")
                markdown_parts.append(f"*No data available for {sheet_name}*\n\n")
            
            # GPU2/Target section
            if has_gpu2:
                markdown_parts.append(f"### 📊 {gpu2_name} (Target)\n")
                df2 = gpu2_dfs[sheet_name]
                
                # Special row limit for ops_unique_args
                row_limit = max_rows_ops_unique_args if sheet_name == 'ops_unique_args' else max_rows_per_sheet
                df2_display = df2.head(row_limit) if len(df2) > row_limit else df2
                
                markdown_parts.append(f"**Rows:** {len(df2)} | **Columns:** {len(df2.columns)}")
                if len(df2) > row_limit:
                    markdown_parts.append(f" | *Showing first {row_limit} rows*")
                markdown_parts.append("\n\n")
                
                try:
                    # Format data
                    df2_formatted = df2_display.copy()
                    
                    # Truncate long strings
                    for col in df2_formatted.select_dtypes(include=['object']).columns:
                        df2_formatted[col] = df2_formatted[col].apply(
                            lambda x: (str(x)[:100] + '...') if isinstance(x, str) and len(str(x)) > 100 else x
                        )
                    
                    # Format numerics
                    for col in df2_formatted.select_dtypes(include=['float64', 'float32']).columns:
                        df2_formatted[col] = df2_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                    
                    # Drop verbose columns for ops_unique_args
                    if sheet_name == 'ops_unique_args':
                        cols_to_drop = [col for col in df2_formatted.columns 
                                    if df2_formatted[col].apply(lambda x: isinstance(x, (list, dict))).any()]
                        if cols_to_drop:
                            df2_formatted = df2_formatted.drop(columns=cols_to_drop)
                    
                    table_md = df2_formatted.to_markdown(index=False, tablefmt='github')
                    markdown_parts.append(table_md)
                except Exception as e:
                    markdown_parts.append(f"*Error converting table: {e}*\n")
                    markdown_parts.append(f"```\n{df2_display.to_string()}\n```")
                markdown_parts.append("\n\n")
            else:
                markdown_parts.append(f"### 📊 {gpu2_name} (Target)\n")
                markdown_parts.append(f"*No data available for {sheet_name}*\n\n")
            
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n\n")
        
        # Add optimization projections section if table paths provided
        if projection_table_paths:
            markdown_parts.append("## 📊 OPTIMIZATION PROJECTIONS\n")
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n")
            markdown_parts.append(f"**Projection:** Optimized {gpu1_name} performance based on {gpu2_name} target\n\n")
            markdown_parts.append("These tables show the potential performance improvement for each category:\n")
            markdown_parts.append("- **Current Time:** Baseline performance\n")
            markdown_parts.append("- **Projected Optimized Time:** Best achievable performance (min of baseline/target)\n")
            markdown_parts.append("- **Potential Gain:** Time savings from optimization\n")
            markdown_parts.append("- **Impact:** Percentage impact on overall runtime\n")
            markdown_parts.append("- **Key Candidate Operations:** Top operations to optimize from target trace\n\n")
            
            # Process each projection table
            for category_name, table_path in sorted(projection_table_paths.items()):
                if not table_path or not table_path.exists():
                    continue
                
                markdown_parts.append(f"### Projections by {category_name}\n")
                markdown_parts.append("-" * 60)
                markdown_parts.append("\n\n")
                
                try:
                    proj_df = pd.read_csv(table_path)
                    table_md = proj_df.to_markdown(index=False, tablefmt='github')
                    markdown_parts.append(table_md)
                    markdown_parts.append("\n\n")
                    
                    # Add summary stats for this category breakdown
                    total_gain = proj_df['Potential Gain (ms)'].sum()
                    total_impact = proj_df['Impact (%)'].sum()
                    markdown_parts.append(f"**Total Potential Gain ({category_name}):** {total_gain:.2f} ms ({total_impact:.2f}% of baseline)\n\n")
                    
                except Exception as e:
                    markdown_parts.append(f"*Error loading projection table for {category_name}: {e}*\n\n")
                
                markdown_parts.append("\n")
            
            markdown_parts.append("=" * 80)
            markdown_parts.append("\n\n")
        
        # Footer
        markdown_parts.append("---\n")
        markdown_parts.append(f"*Comparison: {gpu1_name} (Baseline) vs {gpu2_name} (Target)*\n") #  | Excluded: {', '.join(self.excluded_sheets)}
        
        markdown_content = "\n".join(markdown_parts)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✓ Comparison markdown saved: {output_path}")
        
        return markdown_content


def generate_tracelens_markdown(result_dfs: Dict[str, pd.DataFrame],
                                output_path: Path,
                                gpu_name: str = "GPU",
                                excluded_sheets: Optional[List[str]] = None,
                                max_rows: int = 100) -> str:
    """
    Convenience function to generate markdown from TraceLens results.
    
    Args:
        result_dfs: Dictionary of DataFrames from TraceLens
        output_path: Path to save the markdown file
        gpu_name: Name of the GPU
        excluded_sheets: Sheets to exclude (defaults to GEMM, CONV_fwd, CONV_bwd)
        max_rows: Maximum rows per sheet
        
    Returns:
        Generated markdown content
    """
    generator = MarkdownGenerator(excluded_sheets=excluded_sheets)
    return generator.generate_markdown(
        result_dfs=result_dfs,
        output_path=output_path,
        gpu_name=gpu_name,
        max_rows_per_sheet=max_rows
    )