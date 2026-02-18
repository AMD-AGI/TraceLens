#!/usr/bin/env python3
"""
Cumulative Projection Chart Generator
Generates Baseline → Projection → Target visualization based on operation categories

This module creates a stacked bar chart showing performance projection across
operation categories without requiring AI analysis.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class CumulativeProjectionChart:
    """Generate cumulative optimization projection charts from operation data"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save the chart
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_chart(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        time_column: str = 'total_direct_kernel_time_ms',
        category_column: str = 'op category',
        baseline_label: str = 'BASELINE',
        projection_label: str = 'PROJECTION',
        target_label: str = 'TARGET',
        include_target_bar: bool = False,
        filename: str = 'cumulative_projection.png'
    ) -> Optional[Path]:
        """
        Generate cumulative projection chart from operation dataframes.
        
        Args:
            baseline_df: DataFrame with baseline operations (must have time_column and category_column)
            target_df: DataFrame with target operations (must have time_column and category_column)
            time_column: Column name containing operation duration in milliseconds
            category_column: Column name containing operation categories
            baseline_label: Label for baseline bar
            projection_label: Label for projection bar
            target_label: Label for target bar
            include_target_bar: Whether to include the target bar in the chart
            filename: Output filename
            
        Returns:
            Path to generated chart, or None if generation fails
        """
        try:
            
            # Extract category breakdowns
            baseline_categories = self._aggregate_by_category(baseline_df, time_column, category_column)
            target_categories = self._aggregate_by_category(target_df, time_column, category_column)
            
            if not baseline_categories or not target_categories:
                print("⚠️  No category data available")
                return None
            

            # Currently reported in microseconds in ops unique args
            if time_column == 'total_direct_kernel_time_sum':
                baseline_categories = {k: v / 1000 for k, v in baseline_categories.items()}
                target_categories = {k: v / 1000 for k, v in target_categories.items()}
            
            # Calculate projection (optimized baseline based on target performance)
            projection_categories = self._calculate_projection(baseline_categories, target_categories)

            # Calculate totals
            baseline_total = sum(baseline_categories.values())
            projection_total = sum(projection_categories.values())
            target_total = sum(target_categories.values())


            
            print(f"\n📊 Chart data:")
            print(f"  Baseline:   {baseline_total:.2f}ms ({len(baseline_categories)} categories)")
            print(f"  Projection: {projection_total:.2f}ms ({len(projection_categories)} categories)")
            print(f"  Target:     {target_total:.2f}ms ({len(target_categories)} categories)")
            
            # Generate the chart
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
                filename=filename
            )
            
            return plot_path
            
        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_by_category(
        self,
        df: pd.DataFrame,
        time_column: str,
        category_column: str
    ) -> Dict[str, float]:
        """
        Aggregate operation times by category.
        
        Args:
            df: DataFrame with operations
            time_column: Column containing duration
            category_column: Column containing categories
            
        Returns:
            Dictionary mapping category names to total duration in ms
        """
        if df is None or df.empty:
            return {}
        
        # Validate columns exist
        if time_column not in df.columns:
            print(f"⚠️  Time column '{time_column}' not found. Available: {df.columns.tolist()}")
            return {}
        
        if category_column not in df.columns:
            print(f"⚠️  Category column '{category_column}' not found. Available: {df.columns.tolist()}")
            return {}
        
        df_copy = df.copy()

        # df_copy[category_column] = df_copy[category_column].replace('NA', 'Root')
        # df_copy[category_column] = df_copy[category_column].fillna('Root')

        # Group by category and sum durations
        category_dict = {}
        grouped = df_copy.groupby(category_column)[time_column].sum()
        
        for category, total_time in grouped.items():
            if pd.notna(total_time):
                category_dict[str(category)] = float(total_time)
        
        print(f"  Aggregated {len(category_dict)} categories from {len(df)} operations")
        
        print(f"Category dict: {category_dict}")
        return category_dict
    

    def _extract_key_operations(
        self,
        df: pd.DataFrame,
        category: str,
        category_column: str,
        time_column: str,
        top_n: int = 5
    ) -> str:
        """
        Extract key operations for a given category.
        
        Args:
            df: Operations dataframe
            category: Category to filter by
            category_column: Column name for categories
            time_column: Column name for time
            top_n: Number of top operations to extract
            
        Returns:
            Semicolon-separated string of key operations
        """
        try:
            # Filter by category
            cat_df = df[df[category_column] == category].copy()
            
            if cat_df.empty:
                return "No operations found"
            
            # Sort by time and get top N
            cat_df = cat_df.nlargest(top_n, time_column)
            
            # Build operation strings
            ops_list = []
            for _, row in cat_df.iterrows():
                # Try to get name and shape
                name = row.get('name', row.get('op_name', 'Unknown'))
                shape = row.get('shape', row.get('Input Dims', ''))
                
                # Get time value (convert from microseconds if needed)
                time_val = row[time_column]
                if time_column == 'total_direct_kernel_time_sum':
                    time_val = time_val / 1000  # Convert to ms
                
                if shape:
                    op_str = f"{name} shape:{shape} ({time_val:.1f}ms)"
                else:
                    op_str = f"{name} ({time_val:.1f}ms)"
                
                ops_list.append(op_str)
            
            return '; '.join(ops_list)
            
        except Exception as e:
            return f"Error extracting operations: {str(e)}"

    def generate_optimization_opportunities_table(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        time_column: str = 'total_direct_kernel_time_ms',
        category_column: str = 'op category',
        filename: str = 'optimization_opportunities_table.csv',
        top_n = 5
    ) -> Optional[Path]:
        """
        Generate optimization opportunities table with category breakdowns.
        
        Args:
            baseline_df: Baseline operations dataframe
            target_df: Target operations dataframe
            time_column: Column name containing operation duration
            category_column: Column name containing operation categories
            filename: Output CSV filename
            
        Returns:
            Path to generated CSV, or None if generation fails
        """
        try:
            # Extract category breakdowns
            baseline_categories = self._aggregate_by_category(baseline_df, time_column, category_column)
            target_categories = self._aggregate_by_category(target_df, time_column, category_column)
            
            if not baseline_categories or not target_categories:
                print("⚠️  No category data available for table")
                return None
            
            # Convert from microseconds if needed
            if time_column == 'total_direct_kernel_time_sum':
                baseline_categories = {k: v / 1000 for k, v in baseline_categories.items()}
                target_categories = {k: v / 1000 for k, v in target_categories.items()}
            
            # Calculate projection
            projection_categories = self._calculate_projection(baseline_categories, target_categories)
            
            # Calculate totals
            baseline_total = sum(baseline_categories.values())
            
            # Build opportunities list
            opportunities = []
            for category in sorted(baseline_categories.keys()):
                current_time = baseline_categories[category]
                projected_time = projection_categories.get(category, current_time)
                gain_ms = current_time - projected_time
                impact_pct = (gain_ms / baseline_total * 100) if baseline_total > 0 else 0
                
                key_ops = self._extract_key_operations(
                    target_df, category, category_column, time_column, top_n=top_n
                )

                opportunities.append({
                    'Category': category,
                    'Current Time (ms)': round(current_time, 2),
                    'Projected Optimized Time (ms)': round(projected_time, 2),
                    'Potential Gain (ms)': round(gain_ms, 2),
                    'Impact (%)': round(impact_pct, 2),
                    'Key Candidate Operations': key_ops
                })
            
            # Create dataframe and save
            df = pd.DataFrame(opportunities)
            df = df.sort_values('Potential Gain (ms)', ascending=False)
            df = df.reset_index(drop=True)
            
            csv_path = self.output_dir / filename
            df.to_csv(csv_path, index=False)
            
            print(f"✓ Optimization opportunities table: {csv_path}")
            print(f"  Total potential gain: {df['Potential Gain (ms)'].sum():.2f}ms")
            
            return csv_path
            
        except Exception as e:
            print(f"⚠️  Failed to generate opportunities table: {e}")
            import traceback
            traceback.print_exc()
            return None


    

    def _calculate_projection(
        self,
        baseline_categories: Dict[str, float],
        target_categories: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate projection by taking the minimum time for each category.
        
        This represents an optimized baseline where each category achieves
        the better performance between baseline and target.
        
        Args:
            baseline_categories: Baseline category times
            target_categories: Target category times
            
        Returns:
            Projected category times (optimized baseline)
        """
        projection = {}
        
        # Get all categories from both baseline and target
        all_categories = set(baseline_categories.keys()) | set(target_categories.keys())
        
        for category in all_categories:
            baseline_time = baseline_categories.get(category, 0)
            target_time = target_categories.get(category, 0)
            
            # If category exists in both, take the minimum (best performance)
            if baseline_time > 0 and target_time > 0:
                projection[category] = min(baseline_time, target_time)
            # If only in baseline, use baseline value
            elif baseline_time > 0:
                print(f"Category only in baseline! {category}")
                projection[category] = baseline_time
            # If only in target, use target value
            elif target_time > 0:
                print(f"Category only in target! {category}")
                projection[category] = target_time
        
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
        filename: str
    ) -> Path:
        """Create the stacked bar chart"""
        
        # Determine all categories and states
        all_categories = sorted(set(baseline_categories.keys()) | set(projection_categories.keys()))
        
        if target_categories is not None:
            all_categories = sorted(set(all_categories) | set(target_categories.keys()))
            states = [baseline_label, projection_label, target_label]
            data_dict = {
                baseline_label: baseline_categories,
                projection_label: projection_categories,
                target_label: target_categories
            }
            totals = [baseline_total, projection_total, target_total]
        else:
            states = [baseline_label, projection_label]
            data_dict = {
                baseline_label: baseline_categories,
                projection_label: projection_categories,
            }
            totals = [baseline_total, projection_total]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        
        bottom_values = np.zeros(len(states))
        bar_width = 0.45
        x_pos = np.arange(len(states))
        
        # Assign colors to categories using matplotlib's color cycle
        cat_colors = {}
        for i, cat in enumerate(all_categories):
            cat_colors[cat] = plt.cm.tab20(i % 20)
        
        # Draw stacked bars
        for cat in all_categories:
            values = [data_dict[state].get(cat, 0) for state in states]
            
            bars = ax.bar(
                x_pos, values, bar_width,
                label=cat,
                bottom=bottom_values,
                color=cat_colors[cat],
                edgecolor='white',
                linewidth=2.5,
                alpha=0.95
            )
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    height = bar.get_height()
                    y_center = bottom_values[i] + height / 2
                    
                    # Adjust font size based on bar height
                    fontsize = 8 if height < max(totals) * 0.10 else 9 if height < max(totals) * 0.18 else 10
                    
                    # Only show label if bar is large enough
                    if height > max(totals) * 0.06:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, y_center,
                            f'{val:.0f}',
                            ha='center', va='center',
                            fontsize=fontsize, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65)
                        )
            
            bottom_values += values
        
        # Customize axes and labels
        ax.set_ylabel('Total Execution Time (ms)', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title(
            'Cumulative Performance Optimization Projection',
            fontsize=16, fontweight='bold', pad=20
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(states, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Add total time labels above bars
        for i, total in enumerate(totals):
            ax.text(
                i, total + max(totals) * 0.008,
                f'{total:.0f}ms',
                ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='#1a1a1a'
            )
            
            # Add improvement percentage for non-baseline bars
            if i > 0:
                improvement_ms = totals[0] - total
                improvement_pct = (improvement_ms / totals[0] * 100) if totals[0] > 0 else 0
                
                ax.text(
                    i, total + max(totals) * 0.035,
                    f'↓ {improvement_pct:.1f}% ({improvement_ms:.0f}ms saved)',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    color='#27AE60', style='italic'
                )
        
        # Legend
        ax.legend(
            loc='upper left', bbox_to_anchor=(1.01, 1),
            fontsize=11, framealpha=0.98,
            edgecolor='#cccccc', fancybox=True, shadow=True
        )
        ax.set_ylim(0, max(totals) * 1.18)
        
        # Spine adjustments
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        # Save
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(
            str(output_path), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none'
        )
        plt.close()
        
        print(f"✓ Chart saved: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # Example: Create sample data
    baseline_data = pd.DataFrame({
        'op category': ['GEMM', 'GEMM', 'CONV', 'CONV', 'Batch Normalization', 'Memory'],
        'duration_ms': [100, 150, 80, 90, 50, 30]
    })
    
    target_data = pd.DataFrame({
        'op category': ['GEMM', 'GEMM', 'CONV', 'CONV', 'Batch Normalization', 'Memory'],
        'duration_ms': [80, 120, 60, 70, 40, 25]
    })
    
    # Generate chart
    output_dir = Path('.')
    chart_gen = CumulativeProjectionChart(output_dir)
    
    chart_path = chart_gen.generate_chart(
        baseline_df=baseline_data,
        target_df=target_data,
        time_column='duration_ms',
        category_column='op category',
        filename='example_cumulative_projection.png'
    )
    
    if chart_path:
        print(f"\n✓ Example chart generated: {chart_path}")