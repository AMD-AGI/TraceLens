import pandas as pd
import numpy as np
from openai import OpenAI
from slodels import SLAIAzureOpenAI
import os
import json
from pathlib import Path
import logging
from datetime import datetime
from itertools import combinations
import re

class MultiFileExcelAnalyzer:
    """
    Complete Excel file analyzer with automatic cell and range detection for multi-file comparison
    """
    
    def __init__(self, api_key, log_level=logging.INFO):
        self.client = SLAIAzureOpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.loaded_files = {}
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'excel_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_multiple_files(self, file_paths_list, selected_sheets=None):
        """Load multiple Excel files with their sheets"""
        self.loaded_files = {}
        
        for file_path in file_paths_list:
            file_key = Path(file_path).stem
            self.loaded_files[file_key] = {}
            
            try:
                xl_file = pd.ExcelFile(file_path)
                sheets_to_load = selected_sheets.get(file_key, xl_file.sheet_names) if selected_sheets else xl_file.sheet_names
                
                print(f"Loading {file_key}...")
                for sheet_name in sheets_to_load:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        self.loaded_files[file_key][sheet_name] = {
                            'dataframe': df,
                            'file_path': file_path,
                            'summary': self.get_data_summary(df)
                        }
                        print(f"  Sheet '{sheet_name}': {df.shape}")
                    except Exception as e:
                        print(f"  Error loading sheet '{sheet_name}': {e}")
                        self.logger.error(f"Error loading sheet {sheet_name}: {e}")
                        
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                self.logger.error(f"Error loading file {file_path}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.loaded_files)} files with {sum(len(sheets) for sheets in self.loaded_files.values())} sheets")
        return self.loaded_files
    
    def get_data_summary(self, df):
        """Generate comprehensive data summary"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
            "sample_data": df.head(3).to_dict('records'),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_info = {}
            for col in categorical_cols:
                cat_info[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
            summary["categorical_summary"] = cat_info
        
        return summary
    
    # ==================== CELL VALUE METHODS ====================
    
    def get_cell_value(self, file_key, sheet_name, cell_reference):
        """Get value from specific cell using Excel-style reference"""
        try:
            if file_key not in self.loaded_files:
                return f"File '{file_key}' not found"
            
            if sheet_name not in self.loaded_files[file_key]:
                return f"Sheet '{sheet_name}' not found in file '{file_key}'"
            
            df = self.loaded_files[file_key][sheet_name]['dataframe']
            
            col_letters = ''.join(filter(str.isalpha, cell_reference)).upper()
            row_number = int(''.join(filter(str.isdigit, cell_reference))) - 1
            
            col_index = 0
            for char in col_letters:
                col_index = col_index * 26 + (ord(char) - ord('A') + 1)
            col_index -= 1
            
            if row_number >= len(df) or col_index >= len(df.columns):
                return f"Cell {cell_reference} is out of range"
            
            return df.iloc[row_number, col_index]
            
        except Exception as e:
            return f"Error reading cell: {e}"
    
    def compare_specific_cells(self, cell_mappings):
        """Compare specific cells across multiple files"""
        comparison_results = {
            'cell_values': [],
            'summary': {},
            'differences': []
        }
        
        print("\n" + "="*80)
        print("SPECIFIC CELL COMPARISON")
        print("="*80)
        
        grouped_by_label = {}
        for mapping in cell_mappings:
            label = mapping.get('label', f"{mapping['file']}_{mapping['sheet']}_{mapping['cell']}")
            if label not in grouped_by_label:
                grouped_by_label[label] = []
            
            value = self.get_cell_value(mapping['file'], mapping['sheet'], mapping['cell'])
            
            cell_info = {
                'file': mapping['file'],
                'sheet': mapping['sheet'],
                'cell': mapping['cell'],
                'value': value,
                'label': label
            }
            grouped_by_label[label].append(cell_info)
            comparison_results['cell_values'].append(cell_info)
        
        for label, cells in grouped_by_label.items():
            print(f"\n{label}:")
            values = []
            
            for cell in cells:
                print(f"   {cell['file']} -> {cell['sheet']}[{cell['cell']}] = {cell['value']}")
                if isinstance(cell['value'], (int, float)) and not isinstance(cell['value'], str):
                    values.append(cell['value'])
            
            if len(values) > 1:
                comparison_results['summary'][label] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': max(values) - min(values),
                    'all_equal': len(set(values)) == 1
                }
                
                if not comparison_results['summary'][label]['all_equal']:
                    diff_pct = (max(values) - min(values)) / min(values) * 100 if min(values) != 0 else 0
                    print(f"   Min: {min(values)}, Max: {max(values)}, Difference: {diff_pct:.2f}%")
                    comparison_results['differences'].append({
                        'label': label,
                        'difference_amount': max(values) - min(values),
                        'difference_percentage': diff_pct
                    })
                else:
                    print(f"   All values are identical")
        
        return comparison_results
    
    def compare_cell_ranges(self, range_mappings):
        """Compare cell ranges across files"""
        comparison_results = {}
        
        print("\n" + "="*80)
        print("CELL RANGE COMPARISON")
        print("="*80)
        
        for mapping in range_mappings:
            file_key = mapping['file']
            sheet_name = mapping['sheet']
            cell_range = mapping['range']
            label = mapping.get('label', f"{file_key}_{sheet_name}_{cell_range}")
            
            try:
                if file_key not in self.loaded_files or sheet_name not in self.loaded_files[file_key]:
                    print(f"{label}: File or sheet not found")
                    continue
                
                df = self.loaded_files[file_key][sheet_name]['dataframe']
                
                start_cell, end_cell = cell_range.split(':')
                
                start_col = ''.join(filter(str.isalpha, start_cell)).upper()
                start_row = int(''.join(filter(str.isdigit, start_cell))) - 1
                
                end_col = ''.join(filter(str.isalpha, end_cell)).upper()
                end_row = int(''.join(filter(str.isdigit, end_cell))) - 1
                
                def col_to_index(col_letters):
                    index = 0
                    for char in col_letters:
                        index = index * 26 + (ord(char) - ord('A') + 1)
                    return index - 1
                
                start_col_idx = col_to_index(start_col)
                end_col_idx = col_to_index(end_col)
                
                range_data = df.iloc[start_row:end_row+1, start_col_idx:end_col_idx+1]
                
                print(f"\n{label} ({file_key} - {sheet_name}):")
                print(f"   Shape: {range_data.shape}")
                print(f"   Preview:\n{range_data.head()}")
                
                comparison_results[label] = {
                    'data': range_data,
                    'shape': range_data.shape,
                    'numeric_summary': range_data.select_dtypes(include=[np.number]).describe().to_dict() if len(range_data.select_dtypes(include=[np.number]).columns) > 0 else {}
                }
                
            except Exception as e:
                print(f"Error processing {label}: {e}")
                self.logger.error(f"Error processing range {label}: {e}")
        
        return comparison_results
    
    # ==================== TWO-FILE COMPARISON METHODS ====================
    
    def compare_two_files_detailed(self, file1_key, file2_key, sheet1_name=None, sheet2_name=None):
        """Detailed comparison between two files with % gain/lag calculations"""
        print("\n" + "="*80)
        print("DETAILED FILE COMPARISON WITH GAIN/LAG ANALYSIS")
        print("="*80)
        
        if file1_key not in self.loaded_files or file2_key not in self.loaded_files:
            print("One or both files not found")
            return None
        
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
        print(f"   File 1 (Baseline): {file1_key} -> {sheet1_name} ({df1.shape})")
        print(f"   File 2 (Comparison): {file2_key} -> {sheet2_name} ({df2.shape})")
        
        comparison_results = {
            'file1': file1_key,
            'file2': file2_key,
            'sheet1': sheet1_name,
            'sheet2': sheet2_name,
            'cell_comparisons': [],
            'summary_statistics': {},
            'sheet_level_comparison': {}
        }
        
        print("\n" + "-"*80)
        print("CELL-BY-CELL COMPARISON")
        print("-"*80)
        
        min_rows = min(len(df1), len(df2))
        min_cols = min(len(df1.columns), len(df2.columns))
        
        total_cells = 0
        matching_cells = 0
        different_cells = 0
        
        for row_idx in range(min_rows):
            for col_idx in range(min_cols):
                val1 = df1.iloc[row_idx, col_idx]
                val2 = df2.iloc[row_idx, col_idx]
                
                cell_ref = self._index_to_excel_ref(row_idx, col_idx)
                
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
                            
                            comparison_results['cell_comparisons'].append({
                                'cell': cell_ref,
                                'row': row_idx + 1,
                                'col': col_idx + 1,
                                'column_name': df1.columns[col_idx] if col_idx < len(df1.columns) else f"Col_{col_idx}",
                                'value1': val1,
                                'value2': val2,
                                'difference': val2 - val1,
                                'pct_change': pct_change,
                                'status': gain_or_lag
                            })
        
        if comparison_results['cell_comparisons']:
            sorted_diffs = sorted(comparison_results['cell_comparisons'], 
                                 key=lambda x: abs(x['pct_change']) if x['pct_change'] != float('inf') else 0, 
                                 reverse=True)
            
            print(f"\nTop 10 Largest Changes:")
            print(f"{'Cell':<8} {'Column':<20} {'File1 Value':<15} {'File2 Value':<15} {'Change':<12} {'% Change':<12} {'Status':<8}")
            print("-" * 100)
            
            for diff in sorted_diffs[:10]:
                pct_str = f"{diff['pct_change']:.2f}%" if diff['pct_change'] != float('inf') else "∞"
                print(f"{diff['cell']:<8} {str(diff['column_name'])[:19]:<20} {diff['value1']:<15.2f} {diff['value2']:<15.2f} "
                      f"{diff['difference']:<12.2f} {pct_str:<12} {diff['status']:<8}")
        
        comparison_results['summary_statistics'] = {
            'total_numeric_cells_compared': total_cells,
            'matching_cells': matching_cells,
            'different_cells': different_cells,
            'match_percentage': (matching_cells / total_cells * 100) if total_cells > 0 else 0,
            'total_gains': sum(1 for c in comparison_results['cell_comparisons'] if c['status'] == 'GAIN'),
            'total_lags': sum(1 for c in comparison_results['cell_comparisons'] if c['status'] == 'LAG')
        }
        
        print(f"\nSummary Statistics:")
        print(f"   Total numeric cells compared: {total_cells}")
        print(f"   Matching cells: {matching_cells} ({comparison_results['summary_statistics']['match_percentage']:.2f}%)")
        print(f"   Different cells: {different_cells}")
        print(f"   Gains (File2 > File1): {comparison_results['summary_statistics']['total_gains']}")
        print(f"   Lags (File2 < File1): {comparison_results['summary_statistics']['total_lags']}")
        
        return comparison_results
    
    def compare_sheets_all_metrics(self, file1_key, file2_key):
        """Compare all sheets between two files with comprehensive metrics"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SHEET-BY-SHEET COMPARISON")
        print("="*80)
        
        if file1_key not in self.loaded_files or file2_key not in self.loaded_files:
            print("One or both files not found")
            return None
        
        sheets1 = set(self.loaded_files[file1_key].keys())
        sheets2 = set(self.loaded_files[file2_key].keys())
        
        common_sheets = sheets1.intersection(sheets2)
        unique_to_file1 = sheets1 - sheets2
        unique_to_file2 = sheets2 - sheets1
        
        print(f"\nSheet Overview:")
        print(f"   Common sheets: {len(common_sheets)} - {list(common_sheets)}")
        print(f"   Unique to {file1_key}: {len(unique_to_file1)} - {list(unique_to_file1)}")
        print(f"   Unique to {file2_key}: {len(unique_to_file2)} - {list(unique_to_file2)}")
        
        all_comparisons = {}
        
        for sheet_name in common_sheets:
            print(f"\n{'='*80}")
            print(f"Analyzing Sheet: {sheet_name}")
            print(f"{'='*80}")
            
            df1 = self.loaded_files[file1_key][sheet_name]['dataframe']
            df2 = self.loaded_files[file2_key][sheet_name]['dataframe']
            
            sheet_comparison = {
                'sheet_name': sheet_name,
                'shape1': df1.shape,
                'shape2': df2.shape,
                'column_comparison': {},
                'numeric_summary': {}
            }
            
            print(f"\nShape Comparison:")
            print(f"   {file1_key}: {df1.shape[0]} rows x {df1.shape[1]} columns")
            print(f"   {file2_key}: {df2.shape[0]} rows x {df2.shape[1]} columns")
            
            row_diff = df2.shape[0] - df1.shape[0]
            col_diff = df2.shape[1] - df1.shape[1]
            
            if row_diff != 0:
                row_pct = (row_diff / df1.shape[0] * 100) if df1.shape[0] > 0 else 0
                print(f"   Row difference: {row_diff:+d} ({row_pct:+.2f}%)")
            
            if col_diff != 0:
                col_pct = (col_diff / df1.shape[1] * 100) if df1.shape[1] > 0 else 0
                print(f"   Column difference: {col_diff:+d} ({col_pct:+.2f}%)")
            
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            common_cols = cols1.intersection(cols2)
            
            print(f"\nColumn Comparison:")
            print(f"   Common columns: {len(common_cols)}")
            if cols1 - cols2:
                print(f"   Only in {file1_key}: {list(cols1 - cols2)}")
            if cols2 - cols1:
                print(f"   Only in {file2_key}: {list(cols2 - cols1)}")
            
            print(f"\nNumeric Column Analysis:")
            print(f"{'Column':<25} {'File1 Sum':<15} {'File2 Sum':<15} {'Difference':<15} {'% Change':<12} {'Status':<8}")
            print("-" * 100)
            
            for col in common_cols:
                if col in df1.columns and col in df2.columns:
                    if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                        sum1 = df1[col].sum()
                        sum2 = df2[col].sum()
                        
                        if not (pd.isna(sum1) or pd.isna(sum2)):
                            diff = sum2 - sum1
                            pct_change = ((sum2 - sum1) / abs(sum1) * 100) if sum1 != 0 else (0 if sum2 == 0 else float('inf'))
                            status = "GAIN" if sum2 > sum1 else ("LAG" if sum2 < sum1 else "EQUAL")
                            
                            sheet_comparison['numeric_summary'][col] = {
                                'sum1': sum1,
                                'sum2': sum2,
                                'difference': diff,
                                'pct_change': pct_change,
                                'status': status,
                                'mean1': df1[col].mean(),
                                'mean2': df2[col].mean()
                            }
                            
                            pct_str = f"{pct_change:.2f}%" if pct_change != float('inf') else "∞"
                            col_name = str(col)[:24]
                            print(f"{col_name:<25} {sum1:<15,.2f} {sum2:<15,.2f} {diff:<15,.2f} {pct_str:<12} {status:<8}")
            
            all_comparisons[sheet_name] = sheet_comparison
        
        return all_comparisons
    
    # ==================== AUTO-DETECTION METHODS ====================
    
    def auto_detect_key_cells(self, keywords=None, numeric_only=True):
        """Automatically detect key cells based on keywords and patterns"""
        if keywords is None:
            keywords = ['total', 'sum', 'revenue', 'cost', 'profit', 'sales', 
                       'amount', 'price', 'average', 'count', 'balance', 
                       'budget', 'forecast', 'actual', 'variance', 'net',
                       'gross', 'subtotal', 'grand total', 'expense']
        
        detected_cells = []
        
        print("\n" + "="*80)
        print("AUTO-DETECTING KEY CELLS")
        print("="*80)
        print(f"Searching for keywords: {keywords}")
        
        for file_key, sheets in self.loaded_files.items():
            print(f"\n📁 Analyzing {file_key}...")
            
            for sheet_name, sheet_data in sheets.items():
                df = sheet_data['dataframe']
                print(f"  📋 Sheet: {sheet_name}")
                
                for row_idx in range(min(50, len(df))):
                    for col_idx in range(min(10, len(df.columns))):
                        cell_value = df.iloc[row_idx, col_idx]
                        
                        if isinstance(cell_value, str):
                            cell_lower = cell_value.lower().strip()
                            
                            for keyword in keywords:
                                if keyword in cell_lower:
                                    adjacent_cells = [
                                        (row_idx, col_idx + 1),
                                        (row_idx + 1, col_idx),
                                        (row_idx, col_idx + 2),
                                    ]
                                    
                                    for adj_row, adj_col in adjacent_cells:
                                        if adj_row < len(df) and adj_col < len(df.columns):
                                            adj_value = df.iloc[adj_row, adj_col]
                                            
                                            if isinstance(adj_value, (int, float)) and not pd.isna(adj_value):
                                                if not numeric_only or adj_value != 0:
                                                    cell_ref = self._index_to_excel_ref(adj_row, adj_col)
                                                    label_ref = self._index_to_excel_ref(row_idx, col_idx)
                                                    
                                                    detected_cells.append({
                                                        'file': file_key,
                                                        'sheet': sheet_name,
                                                        'cell': cell_ref,
                                                        'label': cell_value.strip(),
                                                        'value': adj_value,
                                                        'label_cell': label_ref,
                                                        'keyword_matched': keyword
                                                    })
                                                    
                                                    print(f"    Found: '{cell_value.strip()}' at {label_ref}, value {adj_value} at {cell_ref}")
                                                    break
        
        unique_cells = self._deduplicate_detected_cells(detected_cells)
        
        print(f"\nTotal unique key cells detected: {len(unique_cells)}")
        return unique_cells
    
    def auto_detect_comparable_cells(self, min_files=2):
        """Automatically find cells that can be compared across files"""
        all_detected = self.auto_detect_key_cells()
        
        grouped = {}
        for cell in all_detected:
            normalized_label = self._normalize_label(cell['label'])
            
            if normalized_label not in grouped:
                grouped[normalized_label] = []
            grouped[normalized_label].append(cell)
        
        comparable = {}
        comparison_mappings = []
        
        print("\n" + "="*80)
        print("AUTO-DETECTED COMPARABLE CELLS")
        print("="*80)
        
        for label, cells in grouped.items():
            unique_files = set(cell['file'] for cell in cells)
            
            if len(unique_files) >= min_files:
                comparable[label] = cells
                print(f"\n'{label}' found in {len(unique_files)} files:")
                
                for cell in cells:
                    print(f"   {cell['file']} -> {cell['sheet']}[{cell['cell']}] = {cell['value']}")
                    comparison_mappings.append({
                        'file': cell['file'],
                        'sheet': cell['sheet'],
                        'cell': cell['cell'],
                        'label': label
                    })
        
        return comparison_mappings
    
    def auto_detect_data_ranges(self, min_rows=5, min_cols=2):
        """Automatically detect continuous data ranges in sheets"""
        detected_ranges = []
        
        print("\n" + "="*80)
        print("AUTO-DETECTING DATA RANGES")
        print("="*80)
        
        for file_key, sheets in self.loaded_files.items():
            print(f"\n📁 Analyzing {file_key}...")
            
            for sheet_name, sheet_data in sheets.items():
                df = sheet_data['dataframe']
                print(f"  📋 Sheet: {sheet_name}")
                
                ranges = self._find_continuous_blocks(df, min_rows, min_cols)
                
                for range_info in ranges:
                    start_ref = self._index_to_excel_ref(range_info['start_row'], range_info['start_col'])
                    end_ref = self._index_to_excel_ref(range_info['end_row'], range_info['end_col'])
                    range_str = f"{start_ref}:{end_ref}"
                    
                    label = self._generate_range_label(df, range_info)
                    
                    detected_ranges.append({
                        'file': file_key,
                        'sheet': sheet_name,
                        'range': range_str,
                        'label': label,
                        'shape': (range_info['end_row'] - range_info['start_row'] + 1,
                                 range_info['end_col'] - range_info['start_col'] + 1),
                        'has_header': range_info['has_header']
                    })
                    
                    print(f"    Range: {range_str} ({label}) - Shape: {detected_ranges[-1]['shape']}")
        
        return detected_ranges
    
    def auto_detect_comparable_ranges(self):
        """Find ranges that can be compared across files based on similar structure"""
        all_ranges = self.auto_detect_data_ranges()
        
        comparable_groups = {}
        
        print("\n" + "="*80)
        print("AUTO-DETECTED COMPARABLE RANGES")
        print("="*80)
        
        for range_info in all_ranges:
            key = f"shape_{range_info['shape'][0]}x{range_info['shape'][1]}_header_{range_info['has_header']}"
            
            if key not in comparable_groups:
                comparable_groups[key] = []
            comparable_groups[key].append(range_info)
        
        comparable_ranges = []
        for key, ranges in comparable_groups.items():
            unique_files = set(r['file'] for r in ranges)
            
            if len(unique_files) >= 2:
                print(f"\nSimilar ranges found in {len(unique_files)} files:")
                for r in ranges:
                    print(f"   {r['file']} -> {r['sheet']}[{r['range']}] - {r['label']}")
                comparable_ranges.extend(ranges)
        
        return comparable_ranges
    
    # ==================== STRUCTURE COMPARISON METHODS ====================
    
    def compare_sheets_by_structure(self, sheet_selections):
        """Compare sheets by structure across different files"""
        print("\n" + "="*80)
        print("SHEET STRUCTURE COMPARISON")
        print("="*80)
        
        structures = {}
        
        for selection in sheet_selections:
            file_key = selection['file']
            sheet_name = selection['sheet']
            key = f"{file_key}_{sheet_name}"
            
            if file_key in self.loaded_files and sheet_name in self.loaded_files[file_key]:
                summary = self.loaded_files[file_key][sheet_name]['summary']
                structures[key] = {
                    'file': file_key,
                    'sheet': sheet_name,
                    'columns': summary['columns'],
                    'shape': summary['shape'],
                    'data_types': summary['data_types']
                }
                
                print(f"\n{file_key} - {sheet_name}:")
                print(f"   Shape: {summary['shape']}")
                print(f"   Columns: {summary['columns']}")
        
        if len(structures) > 1:
            all_column_sets = [set(s['columns']) for s in structures.values()]
            common_cols = set.intersection(*all_column_sets)
            
            print(f"\nCommon columns across selected sheets: {list(common_cols)}")
            
            for key, struct in structures.items():
                unique = set(struct['columns']) - common_cols
                if unique:
                    print(f"Unique to {key}: {list(unique)}")
        
        return structures
    
    def compare_file_structures(self):
        """Compare structures across all loaded files"""
        comparison = {
            "file_overview": {},
            "common_columns": set(),
            "unique_columns": {},
            "column_type_differences": {},
            "size_comparison": {}
        }
        
        all_columns = set()
        file_columns = {}
        
        for file_key, sheets in self.loaded_files.items():
            comparison["file_overview"][file_key] = {
                "sheets": list(sheets.keys()),
                "total_rows": sum(sheet_data['summary']['shape'][0] for sheet_data in sheets.values()),
                "total_columns": sum(sheet_data['summary']['shape'][1] for sheet_data in sheets.values())
            }
            
            file_cols = set()
            for sheet_name, sheet_data in sheets.items():
                sheet_cols = set(sheet_data['summary']['columns'])
                file_cols.update(sheet_cols)
                all_columns.update(sheet_cols)
            
            file_columns[file_key] = file_cols
        
        if file_columns:
            comparison["common_columns"] = set.intersection(*file_columns.values()) if len(file_columns) > 1 else set()
            
            for file_key, cols in file_columns.items():
                other_files_cols = set.union(*[file_columns[k] for k in file_columns if k != file_key]) if len(file_columns) > 1 else set()
                comparison["unique_columns"][file_key] = cols - other_files_cols
        
        return comparison
    
    # ==================== AI ANALYSIS METHODS ====================
    
    def analyze_with_ai(self, prompt):
        """Send prompt to OpenAI and get analysis"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            return f"Error in AI analysis: {e}"
    
    def create_comparison_analysis_prompt(self, comparison_results):
        """Create AI prompt for detailed comparison analysis"""
        prompt = f"""
        Analyze this detailed comparison between two Excel files:
        
        Files Compared:
        - Baseline: {comparison_results['file1']} - {comparison_results['sheet1']}
        - Comparison: {comparison_results['file2']} - {comparison_results['sheet2']}
        
        Summary Statistics:
        {json.dumps(comparison_results['summary_statistics'], indent=2, default=str)}
        
        Top Differences (sample):
        {json.dumps(comparison_results['cell_comparisons'][:20], indent=2, default=str)}
        
        Please provide:
        1. Analysis of the overall changes between the files
        2. Interpretation of gains and lags
        3. Identification of patterns in the differences
        4. Potential reasons for the observed changes
        5. Areas of concern or noteworthy improvements
        6. Recommendations for further investigation
        7. Business implications of the changes
        """
        return prompt
    
    def create_cell_comparison_prompt(self, comparison_results):
        """Create AI prompt for cell comparison analysis"""
        prompt = f"""
        Analyze the following cell-level comparison across multiple Excel files:
        
        Cell Values Compared:
        {json.dumps(comparison_results['cell_values'], indent=2, default=str)}
        
        Statistical Summary:
        {json.dumps(comparison_results['summary'], indent=2, default=str)}
        
        Identified Differences:
        {json.dumps(comparison_results['differences'], indent=2, default=str)}
        
        Please provide:
        1. Analysis of the differences found
        2. Potential reasons for discrepancies
        3. Data quality assessment
        4. Recommendations for reconciliation
        5. Risk assessment if differences are significant
        """
        return prompt
    
    def create_multi_file_comparison_prompt(self, comparison_data, analysis_type="structural"):
        """Create prompts for multi-file analysis"""
        
        # Convert sets to lists for JSON serialization
        comparison_data_serializable = {
            'file_overview': comparison_data['file_overview'],
            'common_columns': list(comparison_data['common_columns']),
            'unique_columns': {k: list(v) for k, v in comparison_data['unique_columns'].items()},
            'column_type_differences': comparison_data.get('column_type_differences', {}),
            'size_comparison': comparison_data.get('size_comparison', {})
        }
        
        prompts = {
            "structural": f"""
            I have multiple Excel files loaded for comparison. Here's the structural analysis:
            
            File Overview:
            {json.dumps(comparison_data_serializable['file_overview'], indent=2)}
            
            Common columns across all files: {comparison_data_serializable['common_columns']}
            
            Unique columns per file:
            {json.dumps(comparison_data_serializable['unique_columns'], indent=2)}
            
            Please analyze:
            1. Data structure consistency across files
            2. Potential data integration challenges
            3. Recommendations for data harmonization
            4. Common analytical opportunities
            5. Data quality assessment across files
            6. Schema standardization suggestions
            """,
            
            "business_comparison": f"""
            Business analysis across multiple datasets:
            
            File Overview:
            {json.dumps(comparison_data_serializable['file_overview'], indent=2)}
            
            Common columns: {comparison_data_serializable['common_columns']}
            
            Analyze:
            1. Performance trends across different datasets
            2. Comparative business metrics and KPIs
            3. Cross-dataset correlations and insights
            4. Strategic recommendations based on combined data
            5. Risk assessment across all datasets
            6. Revenue and growth opportunities
            """
        }
        
        return prompts.get(analysis_type, prompts["structural"])
    
    def create_analysis_prompt(self, data_summary, analysis_type="general"):
        """Create targeted prompts based on analysis type"""
        
        prompts = {
            "general": f"""
            I have an Excel dataset with the following characteristics:
            - Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
            - Columns: {', '.join(data_summary['columns'])}
            - Data types: {data_summary['data_types']}
            - Missing values: {data_summary['missing_values']}
            
            Sample data (first few rows):
            {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Please provide:
            1. A comprehensive analysis of this dataset
            2. Key insights and patterns you can identify
            3. Potential data quality issues
            4. Recommendations for further analysis
            5. Interesting questions this data could answer
            """,
            
            "business": f"""
            As a business analyst, examine this dataset:
            - Columns: {', '.join(data_summary['columns'])}
            - {data_summary['shape'][0]} records with {data_summary['shape'][1]} attributes
            
            Sample data:
            {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Provide:
            1. Business insights and KPIs that can be derived
            2. Revenue/performance patterns if applicable
            3. Customer/market trends if relevant
            4. Actionable recommendations
            5. Risk factors or opportunities identified
            """,
            
            "statistical": f"""
            Perform statistical analysis on this dataset:
            - Shape: {data_summary['shape']}
            - Numeric columns summary: {data_summary.get('numeric_summary', {})}
            - Missing data: {data_summary['missing_values']}
            
            Please analyze:
            1. Statistical distributions and central tendencies
            2. Correlations and relationships between variables
            3. Outliers and anomalies
            4. Data completeness and quality assessment
            5. Recommended statistical tests or models
            """,
            
            "data_quality": f"""
            Assess the data quality of this dataset:
            - Columns and types: {data_summary['data_types']}
            - Missing values: {data_summary['missing_values']}
            - Sample records: {json.dumps(data_summary['sample_data'][:3], indent=2, default=str)}
            
            Evaluate:
            1. Data completeness and consistency
            2. Potential data entry errors or inconsistencies
            3. Standardization issues
            4. Duplicate detection recommendations
            5. Data cleaning and preprocessing steps needed
            """
        }
        
        return prompts.get(analysis_type, prompts["general"])
    
    def perform_comprehensive_analysis(self, analysis_types=None):
        """Perform comprehensive multi-file analysis"""
        if not self.loaded_files:
            print("No files loaded. Please load files first.")
            return
        
        if analysis_types is None:
            analysis_types = ["structural"]
        
        comparison_data = self.compare_file_structures()
        
        print("\n" + "="*60)
        print("MULTI-FILE ANALYSIS OVERVIEW")
        print("="*60)
        
        for file_key, info in comparison_data['file_overview'].items():
            print(f"{file_key}:")
            print(f"   Sheets: {', '.join(info['sheets'])}")
            print(f"   Total rows: {info['total_rows']:,}")
            print(f"   Total columns: {info['total_columns']}")
        
        print(f"\nCommon columns across all files ({len(comparison_data['common_columns'])}): {list(comparison_data['common_columns'])}")
        
        for analysis_type in analysis_types:
            print(f"\n" + "="*60)
            print(f"AI ANALYSIS - {analysis_type.upper()}")
            print("="*60)
            
            prompt = self.create_multi_file_comparison_prompt(comparison_data, analysis_type)
            analysis = self.analyze_with_ai(prompt)
            print(analysis)
            print("\n" + "-"*40)
    
    # ==================== HELPER METHODS ====================
    
    def _index_to_excel_ref(self, row, col):
        """Convert 0-based indices to Excel cell reference"""
        col_str = ''
        col_num = col + 1
        
        while col_num > 0:
            col_num -= 1
            col_str = chr(col_num % 26 + ord('A')) + col_str
            col_num //= 26
        
        return f"{col_str}{row + 1}"
    
    def _normalize_label(self, label):
        """Normalize label for comparison"""
        normalized = re.sub(r'[^a-z0-9\s]', '', label.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _deduplicate_detected_cells(self, cells):
        """Remove duplicate detections"""
        seen = set()
        unique = []
        
        for cell in cells:
            key = (cell['file'], cell['sheet'], cell['cell'])
            if key not in seen:
                seen.add(key)
                unique.append(cell)
        
        return unique
    
    def _find_continuous_blocks(self, df, min_rows, min_cols):
        """Find continuous data blocks in dataframe"""
        ranges = []
        
        for start_row in range(len(df)):
            for start_col in range(len(df.columns)):
                if pd.notna(df.iloc[start_row, start_col]):
                    end_row = start_row
                    end_col = start_col
                    
                    while end_col < len(df.columns) - 1:
                        if pd.notna(df.iloc[start_row, end_col + 1]):
                            end_col += 1
                        else:
                            break
                    
                    while end_row < len(df) - 1:
                        row_has_data = all(pd.notna(df.iloc[end_row + 1, c]) 
                                          for c in range(start_col, end_col + 1))
                        if row_has_data:
                            end_row += 1
                        else:
                            break
                    
                    if (end_row - start_row + 1) >= min_rows and (end_col - start_col + 1) >= min_cols:
                        overlaps = any(
                            self._ranges_overlap(start_row, end_row, start_col, end_col, r)
                            for r in ranges
                        )
                        
                        if not overlaps:
                            has_header = self._row_looks_like_header(df, start_row, start_col, end_col)
                            
                            ranges.append({
                                'start_row': start_row,
                                'end_row': end_row,
                                'start_col': start_col,
                                'end_col': end_col,
                                'has_header': has_header
                            })
        
        return ranges
    
    def _ranges_overlap(self, start_row, end_row, start_col, end_col, existing_range):
        """Check if two ranges overlap"""
        return not (end_row < existing_range['start_row'] or 
                   start_row > existing_range['end_row'] or
                   end_col < existing_range['start_col'] or 
                   start_col > existing_range['end_col'])
    
    def _row_looks_like_header(self, df, row, start_col, end_col):
        """Check if a row looks like headers"""
        for col in range(start_col, end_col + 1):
            val = df.iloc[row, col]
            if not isinstance(val, str):
                return False
        return True
    
    def _generate_range_label(self, df, range_info):
        """Generate descriptive label for a range"""
        first_cell = df.iloc[range_info['start_row'], range_info['start_col']]
        
        if isinstance(first_cell, str) and len(first_cell) > 0:
            return first_cell[:50]
        
        rows = range_info['end_row'] - range_info['start_row'] + 1
        cols = range_info['end_col'] - range_info['start_col'] + 1
        return f"Data Block ({rows}x{cols})"
    
    def get_sheet_names(self, file_path):
        """Get all sheet names from Excel file"""
        try:
            xl_file = pd.ExcelFile(file_path)
            return xl_file.sheet_names
        except Exception as e:
            self.logger.error(f"Error getting sheet names: {e}")
            return []


def main():
    """Main execution function with interactive menu"""
    
    print("="*80)
    print("MULTI-FILE EXCEL ANALYZER WITH AUTO-DETECTION")
    print("="*80)
    
    # Initialize analyzer
    api_key = input("\nEnter your API key (or press Enter to skip AI analysis): ").strip()
    if not api_key:
        api_key = "dummy_key"
        print("Running without AI analysis features")
    
    analyzer = MultiFileExcelAnalyzer(api_key=api_key)
    
    # Load files from directory
    trace_dir = Path("./traces-diff-gpus")
    
    if not trace_dir.exists():
        print(f"\nDirectory '{trace_dir}' not found. Creating it...")
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created '{trace_dir}'. Please place Excel files there and run again.")
        return
    
    excel_files = list(trace_dir.glob("*.xlsx")) + list(trace_dir.glob("*.xls"))
    
    if not excel_files:
        print(f"\nNo Excel files found in '{trace_dir}'")
        print("Please place Excel files (.xlsx or .xls) in the directory and run again.")
        return
    
    print(f"\nFound {len(excel_files)} Excel file(s):")
    for i, file in enumerate(excel_files, 1):
        print(f"   {i}. {file.name}")
    
    # Ask user which files to load
    print("\nSelect files to analyze:")
    print("   Enter numbers separated by commas (e.g., 1,2,3)")
    print("   Or press Enter to load all files")
    
    selection = input("\nYour selection: ").strip()
    
    if selection:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_files = [excel_files[i] for i in indices if 0 <= i < len(excel_files)]
        except:
            print("Invalid selection, loading all files")
            selected_files = excel_files
    else:
        selected_files = excel_files
    
    if not selected_files:
        print("No files selected")
        return
    
    # Load the selected files
    file_paths = [str(f) for f in selected_files]
    analyzer.load_multiple_files(file_paths)
    
    if not analyzer.loaded_files:
        print("No files were successfully loaded")
        return
    
    # Main menu
    while True:
        print("\n" + "="*80)
        print("ANALYSIS OPTIONS")
        print("="*80)
        print("1. Auto-detect and compare key cells")
        print("2. Auto-detect and compare data ranges")
        print("3. Both (cells and ranges)")
        print("4. Compare file structures")
        print("5. Manual cell comparison")
        print("6. Manual range comparison")
        print("7. Run AI-powered comprehensive analysis")
        print("8. Compare two files (detailed gain/lag analysis)")
        print("9. Compare all sheets between two files")
        print("10. Reload files")
        print("11. Exit")
        
        choice = input("\nSelect option (1-11): ").strip()
        
        if choice == "1":
            # Auto-detect and compare cells
            print("\n" + "="*80)
            print("AUTO-DETECTING KEY CELLS")
            print("="*80)
            
            customize = input("\nUse default keywords? (y/n): ").lower()
            keywords = None
            
            if customize == 'n':
                keywords_input = input("Enter keywords separated by commas: ")
                keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
            
            cell_mappings = analyzer.auto_detect_comparable_cells(min_files=2)
            
            if cell_mappings:
                print(f"\nFound {len(cell_mappings)} comparable cells")
                results = analyzer.compare_specific_cells(cell_mappings)
                
                if results['differences'] and api_key != "dummy_key":
                    do_ai = input("\nRun AI analysis on differences? (y/n): ").lower()
                    if do_ai == 'y':
                        prompt = analyzer.create_cell_comparison_prompt(results)
                        print("\nAnalyzing with AI...")
                        ai_analysis = analyzer.analyze_with_ai(prompt)
                        print("\n" + "="*80)
                        print("AI ANALYSIS")
                        print("="*80)
                        print(ai_analysis)
            else:
                print("\nNo comparable cells found across files")
        
        elif choice == "2":
            # Auto-detect and compare ranges
            print("\n" + "="*80)
            print("AUTO-DETECTING DATA RANGES")
            print("="*80)
            
            range_mappings = analyzer.auto_detect_comparable_ranges()
            
            if range_mappings:
                print(f"\nFound {len(range_mappings)} comparable ranges")
                print("\nSelect ranges to compare:")
                print("   Enter numbers separated by commas")
                print("   Or type 'all' to compare all ranges")
                
                for i, r in enumerate(range_mappings, 1):
                    print(f"   {i}. {r['file']} -> {r['sheet']}[{r['range']}] - {r['label']}")
                
                selection = input("\nYour selection: ").strip()
                
                if selection.lower() == 'all':
                    selected_ranges = range_mappings
                else:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        selected_ranges = [range_mappings[i] for i in indices if 0 <= i < len(range_mappings)]
                    except:
                        print("Invalid selection, using all ranges")
                        selected_ranges = range_mappings
                
                if selected_ranges:
                    range_results = analyzer.compare_cell_ranges(selected_ranges)
            else:
                print("\nNo comparable ranges found across files")
        
        elif choice == "3":
            # Both cells and ranges
            print("\nRunning comprehensive auto-detection...")
            
            # Cells
            print("\n" + "-"*80)
            cell_mappings = analyzer.auto_detect_comparable_cells(min_files=2)
            if cell_mappings:
                results = analyzer.compare_specific_cells(cell_mappings)
            
            # Ranges
            print("\n" + "-"*80)
            range_mappings = analyzer.auto_detect_comparable_ranges()
            if range_mappings:
                selected_ranges = range_mappings[:min(5, len(range_mappings))]
                range_results = analyzer.compare_cell_ranges(selected_ranges)
        
        elif choice == "4":
            # Compare file structures
            print("\nAnalyzing file structures...")
            analyzer.perform_comprehensive_analysis(analysis_types=["structural"])
        
        elif choice == "5":
            # Manual cell comparison
            print("\n" + "="*80)
            print("MANUAL CELL COMPARISON")
            print("="*80)
            print("\nFormat: file_name,sheet_name,cell_reference,label")
            print("Example: file1,Sheet1,B5,Total Revenue")
            print("Enter one mapping per line. Press Enter on empty line to finish.\n")
            
            cell_mappings = []
            while True:
                line = input("➜ ").strip()
                if not line:
                    break
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    cell_mappings.append({
                        'file': parts[0],
                        'sheet': parts[1],
                        'cell': parts[2],
                        'label': parts[3] if len(parts) > 3 else f"{parts[0]}_{parts[2]}"
                    })
                    print(f"   Added: {parts[0]} -> {parts[1]}[{parts[2]}]")
                else:
                    print("   Invalid format, skipping")
            
            if cell_mappings:
                results = analyzer.compare_specific_cells(cell_mappings)
            else:
                print("No cell mappings entered")
        
        elif choice == "6":
            # Manual range comparison
            print("\n" + "="*80)
            print("MANUAL RANGE COMPARISON")
            print("="*80)
            print("\nFormat: file_name,sheet_name,range,label")
            print("Example: file1,Sheet1,A1:D10,Q1 Sales Data")
            print("Enter one mapping per line. Press Enter on empty line to finish.\n")
            
            range_mappings = []
            while True:
                line = input("➜ ").strip()
                if not line:
                    break
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    range_mappings.append({
                        'file': parts[0],
                        'sheet': parts[1],
                        'range': parts[2],
                        'label': parts[3] if len(parts) > 3 else f"{parts[0]}_{parts[2]}"
                    })
                    print(f"   Added: {parts[0]} -> {parts[1]}[{parts[2]}]")
                else:
                    print("   Invalid format, skipping")
            
            if range_mappings:
                range_results = analyzer.compare_cell_ranges(range_mappings)
            else:
                print("No range mappings entered")
        
        elif choice == "7":
            # AI comprehensive analysis
            if api_key == "dummy_key":
                print("\nAI analysis requires an API key. Please restart with a valid key.")
            else:
                print("\nRunning AI-powered comprehensive analysis...")
                print("\nSelect analysis types:")
                print("1. Structural analysis")
                print("2. Business comparison")
                print("3. Both")
                
                ai_choice = input("\nYour choice (1-3): ").strip()
                
                if ai_choice == "1":
                    analysis_types = ["structural"]
                elif ai_choice == "2":
                    analysis_types = ["business_comparison"]
                else:
                    analysis_types = ["structural", "business_comparison"]
                
                analyzer.perform_comprehensive_analysis(analysis_types=analysis_types)
        
        elif choice == "8":
            # Compare two files with gain/lag analysis
            print("\nDETAILED TWO-FILE COMPARISON")
            print("\nAvailable files:")
            file_keys = list(analyzer.loaded_files.keys())
            for i, fk in enumerate(file_keys, 1):
                sheets = list(analyzer.loaded_files[fk].keys())
                print(f"{i}. {fk} (Sheets: {', '.join(sheets)})")
            
            try:
                file1_idx = int(input("\nSelect baseline file (number): ")) - 1
                file2_idx = int(input("Select comparison file (number): ")) - 1
                
                if 0 <= file1_idx < len(file_keys) and 0 <= file2_idx < len(file_keys):
                    file1_key = file_keys[file1_idx]
                    file2_key = file_keys[file2_idx]
                    
                    # Ask for sheet names
                    print(f"\nSheets in {file1_key}: {list(analyzer.loaded_files[file1_key].keys())}")
                    sheet1 = input(f"Enter sheet name for {file1_key} (or press Enter for first sheet): ").strip()
                    
                    print(f"\nSheets in {file2_key}: {list(analyzer.loaded_files[file2_key].keys())}")
                    sheet2 = input(f"Enter sheet name for {file2_key} (or press Enter for first sheet): ").strip()
                    
                    sheet1 = sheet1 if sheet1 else None
                    sheet2 = sheet2 if sheet2 else None
                    
                    # Perform comparison
                    results = analyzer.compare_two_files_detailed(file1_key, file2_key, sheet1, sheet2)
                    
                    if results and api_key != "dummy_key":
                        do_ai = input("\nRun AI analysis on comparison? (y/n): ").lower()
                        if do_ai == 'y':
                            prompt = analyzer.create_comparison_analysis_prompt(results)
                            print("\nAnalyzing with AI...")
                            ai_analysis = analyzer.analyze_with_ai(prompt)
                            print("\n" + "="*80)
                            print("AI COMPARISON ANALYSIS")
                            print("="*80)
                            print(ai_analysis)
                else:
                    print("Invalid file selection")
            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}")
        
        elif choice == "9":
            # Compare all sheets between two files
            print("\nSHEET-BY-SHEET COMPARISON")
            print("\nAvailable files:")
            file_keys = list(analyzer.loaded_files.keys())
            for i, fk in enumerate(file_keys, 1):
                print(f"{i}. {fk}")
            
            try:
                file1_idx = int(input("\nSelect first file (number): ")) - 1
                file2_idx = int(input("Select second file (number): ")) - 1
                
                if 0 <= file1_idx < len(file_keys) and 0 <= file2_idx < len(file_keys):
                    file1_key = file_keys[file1_idx]
                    file2_key = file_keys[file2_idx]
                    
                    results = analyzer.compare_sheets_all_metrics(file1_key, file2_key)
                else:
                    print("Invalid file selection")
            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}")
        
        elif choice == "10":
            # Reload files
            print("\nReloading files...")
            analyzer.load_multiple_files(file_paths)
            print("Files reloaded successfully")
        
        elif choice == "11":
            # Exit
            print("\nThank you for using Multi-File Excel Analyzer!")
            print("="*80)
            break
        
        else:
            print("\nInvalid option. Please select 1-11.")
        
        # Ask if user wants to continue
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            continue_choice = input("\nPress Enter to return to main menu...").strip()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()