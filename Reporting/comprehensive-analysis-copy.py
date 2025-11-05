import pandas as pd
import numpy as np
from openai import OpenAI
from slodels import SLAIAzureOpenAI
import json
from pathlib import Path
import logging
from datetime import datetime
import re

class ExcelComparisonReportGenerator:
    """
    Excel file comparison tool with professional report generation
    """
    
    def __init__(self, api_key=None, log_level=logging.INFO):
        self.client = SLAIAzureOpenAI(api_key=api_key) if api_key else None
        self.model = "gpt-4o"
        self.loaded_files = {}
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_files(self, file_paths_list):
        """Load Excel files"""
        self.loaded_files = {}
        
        for file_path in file_paths_list:
            file_key = Path(file_path).stem
            self.loaded_files[file_key] = {}
            
            try:
                xl_file = pd.ExcelFile(file_path)
                
                print(f"Loading {file_key}...")
                for sheet_name in xl_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        self.loaded_files[file_key][sheet_name] = {
                            'dataframe': df,
                            'file_path': file_path
                        }
                        print(f"  Sheet '{sheet_name}': {df.shape}")
                    except Exception as e:
                        print(f"  Error loading sheet '{sheet_name}': {e}")
                        self.logger.error(f"Error loading sheet {sheet_name}: {e}")
                        
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                self.logger.error(f"Error loading file {file_path}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.loaded_files)} files")
        return self.loaded_files
    
    def compare_two_files(self, file1_key, file2_key, sheet1_name=None, sheet2_name=None):
        """Detailed comparison between two files with gain/lag calculations"""
        print("\n" + "="*80)
        print("DETAILED FILE COMPARISON")
        print("="*80)
        
        if file1_key not in self.loaded_files or file2_key not in self.loaded_files:
            print("One or both files not found")
            return None
        
        # Get sheets
        if sheet1_name is None:
            sheet1_name = list(self.loaded_files[file1_key].keys())[0]
        if sheet2_name is None:
            sheet2_name = list(self.loaded_files[file2_key].keys())[0]
        
        if sheet1_name not in self.loaded_files[file1_key]:
            print(f"Sheet '{sheet1_name}' not found in {file1_key}")
            return None
        if sheet2_name not in self.loaded_files[file2_key]:
            print(f"Sheet '{sheet2_name}' not found in {file2_key}")
            return None
        
        df1 = self.loaded_files[file1_key][sheet1_name]['dataframe']
        df2 = self.loaded_files[file2_key][sheet2_name]['dataframe']
        
        print(f"\nComparing:")
        print(f"   Baseline: {file1_key} -> {sheet1_name} ({df1.shape})")
        print(f"   Comparison: {file2_key} -> {sheet2_name} ({df2.shape})")
        
        comparison_results = {
            'file1': file1_key,
            'file2': file2_key,
            'sheet1': sheet1_name,
            'sheet2': sheet2_name,
            'cell_comparisons': [],
            'column_summaries': {},
            'summary_statistics': {}
        }
        
        print("\nAnalyzing differences...")
        
        min_rows = min(len(df1), len(df2))
        min_cols = min(len(df1.columns), len(df2.columns))
        
        total_cells = 0
        matching_cells = 0
        different_cells = 0
        
        # Cell-by-cell comparison
        for row_idx in range(min_rows):
            for col_idx in range(min_cols):
                val1 = df1.iloc[row_idx, col_idx]
                val2 = df2.iloc[row_idx, col_idx]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if not (pd.isna(val1) or pd.isna(val2)):
                        total_cells += 1
                        
                        if val1 == val2:
                            matching_cells += 1
                        else:
                            different_cells += 1
                            
                            if val1 != 0:
                                pct_change = ((val2 - val1) / abs(val1)) * 100
                            else:
                                pct_change = float('inf') if val2 != 0 else 0
                            
                            gain_or_lag = "GAIN" if val2 > val1 else "LAG"
                            col_name = str(df1.columns[col_idx]) if col_idx < len(df1.columns) else f"Col_{col_idx}"
                            
                            comparison_results['cell_comparisons'].append({
                                'cell': self._index_to_excel_ref(row_idx, col_idx),
                                'row': row_idx + 1,
                                'col': col_idx + 1,
                                'column_name': col_name,
                                'value1': val1,
                                'value2': val2,
                                'difference': val2 - val1,
                                'pct_change': pct_change,
                                'status': gain_or_lag
                            })
        
        # Column-level summaries
        for col_idx in range(min_cols):
            col_name = str(df1.columns[col_idx]) if col_idx < len(df1.columns) else f"Col_{col_idx}"
            
            if pd.api.types.is_numeric_dtype(df1.iloc[:, col_idx]) and pd.api.types.is_numeric_dtype(df2.iloc[:, col_idx]):
                sum1 = df1.iloc[:, col_idx].sum()
                sum2 = df2.iloc[:, col_idx].sum()
                mean1 = df1.iloc[:, col_idx].mean()
                mean2 = df2.iloc[:, col_idx].mean()
                
                if not (pd.isna(sum1) or pd.isna(sum2)):
                    diff = sum2 - sum1
                    pct_change = ((sum2 - sum1) / abs(sum1) * 100) if sum1 != 0 else (0 if sum2 == 0 else float('inf'))
                    
                    comparison_results['column_summaries'][col_name] = {
                        'sum1': sum1,
                        'sum2': sum2,
                        'mean1': mean1,
                        'mean2': mean2,
                        'difference': diff,
                        'pct_change': pct_change,
                        'status': 'GAIN' if diff > 0 else ('LAG' if diff < 0 else 'EQUAL')
                    }
        
        # Summary statistics
        comparison_results['summary_statistics'] = {
            'total_numeric_cells_compared': total_cells,
            'matching_cells': matching_cells,
            'different_cells': different_cells,
            'match_percentage': (matching_cells / total_cells * 100) if total_cells > 0 else 0,
            'total_gains': sum(1 for c in comparison_results['cell_comparisons'] if c['status'] == 'GAIN'),
            'total_lags': sum(1 for c in comparison_results['cell_comparisons'] if c['status'] == 'LAG')
        }
        
        print(f"Analysis complete: {different_cells} differences found")
        
        return comparison_results
    
    def load_two_sheets(self, file1_key, file2_key, sheet1_name=None, sheet2_name=None):
        """Detailed comparison between two files with gain/lag calculations"""
        print("\n" + "="*80)
        print("DETAILED FILE COMPARISON")
        print("="*80)
        
        if file1_key not in self.loaded_files or file2_key not in self.loaded_files:
            print("One or both files not found")
            return None
        
        # Get sheets
        if sheet1_name is None:
            sheet1_name = list(self.loaded_files[file1_key].keys())[0]
        if sheet2_name is None:
            sheet2_name = list(self.loaded_files[file2_key].keys())[0]
        
        if sheet1_name not in self.loaded_files[file1_key]:
            print(f"Sheet '{sheet1_name}' not found in {file1_key}")
            return None
        if sheet2_name not in self.loaded_files[file2_key]:
            print(f"Sheet '{sheet2_name}' not found in {file2_key}")
            return None
        
        df1 = self.loaded_files[file1_key][sheet1_name]['dataframe']
        df2 = self.loaded_files[file2_key][sheet2_name]['dataframe']
        
        print(f"\nComparing:")
        print(f"   Baseline: {file1_key} -> {sheet1_name} ({df1.shape})")
        print(f"   Comparison: {file2_key} -> {sheet2_name} ({df2.shape})")
        
        return df1, df2
    
    def generate_report(self, comparison_results, report_title=None, custom_sections=None):
        """Generate comprehensive markdown report
        
        Args:
            comparison_results: Results from compare_two_files
            report_title: Custom title for the report
            custom_sections: Dict with custom section configurations
        """
        if not comparison_results:
            return "No comparison results available"
        
        file1 = comparison_results['file1']
        file2 = comparison_results['file2']
        stats = comparison_results['summary_statistics']
        cell_comps = comparison_results['cell_comparisons']
        col_summaries = comparison_results['column_summaries']
        
        # Calculate overall metrics
        overall_worse = stats['total_lags'] > stats['total_gains']
        overall_pct = 100 - stats['match_percentage']
        
        # Group by categories
        categories = self._categorize_differences(cell_comps, col_summaries)
        
        # Build report
        report = []
        
        # Title
        if report_title:
            report.append(f"# {report_title}\n")
        else:
            report.append(f"# Comparison Report: {file1} vs {file2}\n")
        
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary section
        report.append("## Summary")
        
        # Overall comparison
        if overall_worse:
            report.append(f"- Overall **{overall_pct:.1f}% worse than {file1}**. "
                         f"({stats['matching_cells']} matching vs {stats['different_cells']} different)")
        else:
            report.append(f"- Overall **{overall_pct:.1f}% better than {file1}**. "
                         f"({stats['matching_cells']} matching vs {stats['different_cells']} different)")
        
        # Key insights
        if categories:
            top_category = categories[0]
            report.append(f"- Most of this diff comes from **{top_category['name']}** "
                         f"– causing **{abs(top_category['total_impact']):.0f} units** difference.")
        
        report.append(f"- Total changes: **{stats['total_gains']} gains** and **{stats['total_lags']} lags**")
        report.append("")
        
        # Detailed sections for each category
        for idx, category in enumerate(categories, 1):
            report.append("---\n")
            report.append(f"### {idx}. {category['name']}")
            
            # Status indicator
            if category['avg_pct_change'] < -10:
                report.append(f"- Causing **~{abs(category['total_impact']):.0f} units regression**.")
            elif category['avg_pct_change'] > 10:
                report.append(f"- Showing **~{abs(category['total_impact']):.0f} units improvement**.")
            else:
                report.append(f"- Minor changes: **{abs(category['total_impact']):.0f} units** difference.")
            
            # Performance comparison
            if category['status'] == 'LAG':
                multiplier = abs(category['avg_pct_change']) / 100 + 1
                if multiplier > 2:
                    report.append(f"- **{multiplier:.1f}× slower** on {file2}.")
                else:
                    report.append(f"- **{abs(category['avg_pct_change']):.1f}% slower** on {file2}.")
            else:
                if category['avg_pct_change'] > 20:
                    report.append(f"- **{category['avg_pct_change']:.1f}% faster** on {file2}.")
            
            # Extreme cases
            if category['extreme_cases']:
                extreme = category['extreme_cases'][0]
                if extreme['pct_change'] != float('inf'):
                    report.append(f"- **Extreme case**: `{extreme['column_name']}` is "
                                f"**{abs(extreme['pct_change']):.1f}× {'slower' if extreme['status'] == 'LAG' else 'faster'}**, "
                                f"but not major portion of workload.")
            
            # Opportunity
            if category['status'] == 'LAG' and abs(category['total_impact']) > 10:
                report.append(f"- Optimization opportunity is **{abs(category['total_impact']):.0f} units**.")
            
            report.append("")
        
        # The Good section
        gains = [cat for cat in categories if cat['status'] == 'GAIN' and cat['avg_pct_change'] > 5]
        if gains:
            report.append("---\n")
            report.append("### The Good")
            for gain in gains[:5]:
                report.append(f"- The **{gain['name']}** operations are "
                            f"**{gain['avg_pct_change']:.0f}% faster** than {file1}.")
            report.append("")
        
        # Overall comments
        report.append("---\n")
        report.append("### Workload Comments")
        
        # Calculate workload distribution
        total_values = sum(abs(cat['total_impact']) for cat in categories)
        if categories and total_values > 0:
            top_pct = (abs(categories[0]['total_impact']) / total_values * 100)
            if top_pct > 50:
                report.append(f"- Overall, **>{top_pct:.0f}% of changes** concentrated in **{categories[0]['name']}**.")
        
        # Optimization suggestions
        if overall_worse:
            report.append(f"- Do you plan to use **optimization techniques** here?")
            report.append(f"- Consider investigating the top {min(3, len(categories))} regression areas.")
        
        report.append("")
        
        return "\n".join(report)
    
    def _categorize_differences(self, cell_comps, col_summaries):
        """Categorize differences into logical groups"""
        categories = {}
        
        # Group by column name
        for comp in cell_comps:
            col_name = comp['column_name']
            if col_name not in categories:
                categories[col_name] = {
                    'name': col_name,
                    'cells': [],
                    'total_impact': 0,
                    'extreme_cases': []
                }
            
            categories[col_name]['cells'].append(comp)
            categories[col_name]['total_impact'] += comp['difference']
        
        # Calculate metrics for each category
        result = []
        for cat_name, cat_data in categories.items():
            cells = cat_data['cells']
            
            # Filter out infinite values for averaging
            valid_pcts = [c['pct_change'] for c in cells if c['pct_change'] != float('inf')]
            avg_pct = np.mean(valid_pcts) if valid_pcts else 0
            
            # Determine status
            status = 'GAIN' if cat_data['total_impact'] > 0 else 'LAG'
            
            # Find extreme cases
            extreme = sorted(cells, key=lambda x: abs(x['pct_change']) if x['pct_change'] != float('inf') else 0, reverse=True)[:3]
            
            result.append({
                'name': cat_name,
                'total_impact': cat_data['total_impact'],
                'avg_pct_change': avg_pct,
                'status': status,
                'num_cells': len(cells),
                'extreme_cases': extreme
            })
        
        # Sort by absolute impact
        result.sort(key=lambda x: abs(x['total_impact']), reverse=True)
        
        return result
    
    def save_report(self, report_content, filename=None, output_dir="./reports"):
        """Save report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.md"
        
        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'
        
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nReport saved to: {filepath}")
        return filepath
    
    # def generate_ai_insights(self, comparison_results):
    #     """Generate AI-powered insights"""
    #     if not self.client:
    #         return "AI analysis not available (no API key provided)"
        
    #     try:
    #         prompt = f"""
    #         Analyze this Excel file comparison and provide strategic insights:
            
    #         Files: {comparison_results['file1']} vs {comparison_results['file2']}
            
    #         Statistics:
    #         {json.dumps(comparison_results['summary_statistics'], indent=2)}
            
    #         Column Summaries:
    #         {json.dumps(comparison_results['column_summaries'], indent=2, default=str)}
            
    #         Top Differences:
    #         {json.dumps(comparison_results['cell_comparisons'][:20], indent=2, default=str)}
            
    #         Provide:
    #         1. Root cause analysis of major differences
    #         2. Business impact assessment
    #         3. Specific actionable recommendations
    #         4. Risk areas to investigate
    #         5. Opportunities for improvement
    #         """
            
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=2000,
    #             temperature=0.7
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         self.logger.error(f"AI analysis error: {e}")
    #         return f"❌ Error in AI analysis: {e}"
    def generate_ai_insights(self, sheet1, sheet2):
        """Generate AI-powered insights"""
        if not self.client:
            return "AI analysis not available (no API key provided)"

        try:
            # Convert dataframes to readable string format with better structure
            df1_str = sheet1.to_string(max_rows=100, max_cols=20)  # Adjust limits as needed
            df2_str = sheet2.to_string(max_rows=100, max_cols=20)
            
            # Get summary statistics
            df1_stats = sheet1.describe().to_string()
            df2_stats = sheet2.describe().to_string()
            
            prompt = f"""
            You are analyzing GPU performance data comparing MI300 and H100 accelerators.
            
            **MI300 Performance Data:**
            {df1_str}
            
            **MI300 Summary Statistics:**
            {df1_stats}
            
            **H100 Performance Data:**
            {df2_str}
            
            **H100 Summary Statistics:**
            {df2_stats}
            
            Please provide a comprehensive quantitative analysis with specific numbers, percentages, and metrics.
            
            **Required Analysis:**
            
            ## Summary
            1. Provide a quantitative comparison of MI300 vs H100 overall performance:
            - Calculate percentage difference in total execution time
            - Identify which accelerator is faster and by how much (in ms and %)
            - Provide specific numeric metrics
            
            2. Component-level breakdown:
            - Identify the top 3-5 components contributing most to performance differences
            - Provide actual time differences in milliseconds for each component
            - Calculate percentage contribution of each component to the total gap
            
            3. Overall time gap:
            - Report total time difference in milliseconds
            - Report as percentage difference
            - Indicate if MI300 is faster or slower
            
            ## GEMMs Performance
            1. Optimization opportunity:
            - Calculate exact millisecond difference in GEMM operations between MI300 and H100
            - Express as percentage of total execution time
            - Identify specific GEMM operations with largest gaps
            
            2. Compute bound analysis:
            - Analyze if GEMMs are compute-bound based on utilization metrics
            - Provide evidence from the data (specific metrics and values)
            
            3. Improvement potential:
            - Estimate potential improvement in milliseconds if MI300 matched H100 GEMM performance
            - Calculate theoretical speedup percentage
            - Identify specific optimization opportunities
            
            **Format Requirements:**
            - Use actual numbers from the data
            - Include units (ms, %, TFLOPS, etc.)
            - Show calculations where relevant
            - Highlight key findings with specific metrics
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a GPU performance analysis expert. Always provide specific quantitative metrics with numbers, not general observations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,  # Increased for detailed analysis
                temperature=0.3  # Lower temperature for more factual responses
            )
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            return f"Error in AI analysis: {e}"
    
    def _index_to_excel_ref(self, row, col):
        """Convert 0-based indices to Excel cell reference"""
        col_str = ''
        col_num = col + 1
        
        while col_num > 0:
            col_num -= 1
            col_str = chr(col_num % 26 + ord('A')) + col_str
            col_num //= 26
        
        return f"{col_str}{row + 1}"


def main():
    """Main execution function"""
    
    print("="*80)
    print("EXCEL COMPARISON REPORT GENERATOR")
    print("="*80)

    api_key = ""
    
    generator = ExcelComparisonReportGenerator(api_key=api_key)
    
    # Check for files
    trace_dir = Path("./traces-diff-gpus")
    
    if not trace_dir.exists():
        print(f"\nDirectory '{trace_dir}' not found. Creating it...")
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created '{trace_dir}'. Please place Excel files there and run again.")
        return
    
    excel_files = list(trace_dir.glob("*.xlsx")) + list(trace_dir.glob("*.xls"))
    
    if len(excel_files) < 2:
        print(f"\nNeed at least 2 Excel files in '{trace_dir}'")
        print(f"Found: {len(excel_files)} file(s)")
        return
    
    print(f"\nFound {len(excel_files)} Excel file(s):")
    for i, file in enumerate(excel_files, 1):
        print(f"   {i}. {file.name}")
    
    # Select files to compare
    print("\nSelect files to compare:")
    try:
        file1_idx = int(input("Select baseline file (number): ")) - 1
        file2_idx = int(input("Select comparison file (number): ")) - 1
        
        if not (0 <= file1_idx < len(excel_files) and 0 <= file2_idx < len(excel_files)):
            print("Invalid selection")
            return
        
        baseline_file = excel_files[file1_idx]
        comparison_file = excel_files[file2_idx]
        
    except (ValueError, IndexError):
        print("Invalid input")
        return
    
    # Load files
    print("\nLoading files...")
    generator.load_files([str(baseline_file), str(comparison_file)])
    
    if len(generator.loaded_files) < 2:
        print("Failed to load both files")
        return
    
    # Get file keys
    file_keys = list(generator.loaded_files.keys())
    file1_key = file_keys[0]
    file2_key = file_keys[1]
    
    # Select sheets
    print(f"\nSheets in {file1_key}: {list(generator.loaded_files[file1_key].keys())}")
    sheet1 = input(f"Enter sheet name for {file1_key} (or press Enter for first): ").strip()
    
    print(f"\nSheets in {file2_key}: {list(generator.loaded_files[file2_key].keys())}")
    sheet2 = input(f"Enter sheet name for {file2_key} (or press Enter for first): ").strip()
    
    sheet1 = sheet1 if sheet1 else None
    sheet2 = sheet2 if sheet2 else None
    
    # Perform comparison
    print("\nComparing files...")
    df1, df2 = generator.load_two_sheets(file1_key, file2_key, sheet1, sheet2)

    
    # if not results:
    #     print("❌ Comparison failed")
    #     return
    
    # # Generate report
    # print("\n📝 Generating report...")
    # report_title = input("Enter report title (or press Enter for default): ").strip()
    # report_title = report_title if report_title else None
    
    # report = generator.generate_report(results, report_title=report_title)
    
    # # Display report
    # print("\n" + "="*80)
    # print("📊 COMPARISON REPORT")
    # print("="*80)
    # print(report)
    
    # # Save report
    # save_choice = input("\n💾 Save report to file? (y/n): ").lower()
    # if save_choice == 'y':
    #     filename = input("Enter filename (or press Enter for auto-generated): ").strip()
    #     filename = filename if filename else None
    #     generator.save_report(report, filename)
    
    # AI insights
    if api_key:
        # ai_choice = input("\n🤖 Generate AI insights? (y/n): ").lower()
        # if ai_choice == 'y':
        # print("\n⏳ Generating AI insights...")
        # insights = generator.generate_ai_insights(results)
        # print("\n" + "="*80)
        # print("🤖 AI INSIGHTS")
        # print("="*80)
        # print(insights)

        print("\nGenerating AI insights...")
        insights = generator.generate_ai_insights(df1, df2)
        print("\n" + "="*80)
        print("AI INSIGHTS")
        print("="*80)
        print(insights)
        
        # Save insights
        # save_ai = input("\n💾 Save AI insights to file? (y/n): ").lower()
        # if save_ai == 'y':
        # insights_filename = input("Enter filename (or press Enter for auto): ").strip()
        insights_filename = "GEMM comparison of MI300 and H100"
        if not insights_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            insights_filename = f"ai_insights_{timestamp}.md"
        generator.save_report(insights, insights_filename)
    
    print("\nProcess complete!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()