###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple


def analyze_lca_differences(
    diff_csv_path: str,
    baseline_name: str,
    target_name: str,
    baseline_source: str = "trace2",
    target_source: str = "trace1"
) -> Dict[str, Any]:
    """
    Find LCAs unique to each trace and compute statistics.
    
    Args:
        diff_csv_path: Path to TraceDiff CSV
        baseline_name: Display name for baseline (for output)
        target_name: Display name for target (for output)
        baseline_source: Source column value for baseline (default: "trace2")
        target_source: Source column value for target (default: "trace1")
        
    Returns:
        Dict with baseline_only_stats, target_only_stats, and total times
    """
    df = pd.read_csv(diff_csv_path)
    
    # Validate columns
    required = ['source', 'lowest_common_ancestor_id', 'kernel_time']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Need: {required}")
    
    # Inspect actual source values
    actual_sources = df['source'].unique()
    print(f"  📋 Sources in CSV: {list(actual_sources)}")
    
    if len(actual_sources) != 2:
        raise ValueError(f"Expected 2 sources, found {len(actual_sources)}: {actual_sources}")
    
    # Validate source values exist
    if baseline_source not in actual_sources:
        raise ValueError(f"Baseline source '{baseline_source}' not found. Available: {actual_sources}")
    if target_source not in actual_sources:
        raise ValueError(f"Target source '{target_source}' not found. Available: {actual_sources}")
    
    # Split by source (don't sort!)
    baseline_df = df[df['source'] == baseline_source].copy()
    target_df = df[df['source'] == target_source].copy()
    
    print(f"  📊 {baseline_name} ({baseline_source}): {len(baseline_df)} rows")
    print(f"  📊 {target_name} ({target_source}): {len(target_df)} rows")
    
    # Get unique LCA IDs
    baseline_lcas = set(baseline_df['lowest_common_ancestor_id'].unique())
    target_lcas = set(target_df['lowest_common_ancestor_id'].unique())
    
    print(f"  🔍 {baseline_name}: {len(baseline_lcas)} unique LCAs")
    print(f"  🔍 {target_name}: {len(target_lcas)} unique LCAs")
    
    # Find unique LCAs
    baseline_only = sorted(baseline_lcas - target_lcas)
    target_only = sorted(target_lcas - baseline_lcas)
    
    print(f"  ⚡ Unique to {baseline_name}: {len(baseline_only)} LCAs")
    print(f"  ⚡ Unique to {target_name}: {len(target_only)} LCAs")
    
    # Total times
    baseline_total = baseline_df['kernel_time'].sum()
    target_total = target_df['kernel_time'].sum()
    
    # Compute stats
    baseline_stats = _compute_stats(baseline_df, baseline_only, baseline_total)
    target_stats = _compute_stats(target_df, target_only, target_total)
    
    return {
        'baseline_only_stats': baseline_stats,
        'target_only_stats': target_stats,
        'baseline_total_time': baseline_total,
        'target_total_time': target_total,
        'baseline_source': baseline_source,
        'target_source': target_source
    }


def _compute_stats(df: pd.DataFrame, lca_ids: List, total_time: float) -> Dict:
    """Compute statistics for operations with given LCA IDs"""
    if not lca_ids:
        return {
            'lca_ids': [],
            'num_kernels': 0,
            'cumulative_runtime': 0.0,
            'percentage_of_total': 0.0,
            'details_df': pd.DataFrame()
        }
    
    lca_df = df[df['lowest_common_ancestor_id'].isin(lca_ids)]
    runtime = lca_df['kernel_time'].sum()
    pct = (runtime / total_time * 100) if total_time > 0 else 0.0
    
    return {
        'lca_ids': lca_ids,
        'num_kernels': len(lca_df),
        'cumulative_runtime': runtime,
        'percentage_of_total': pct,
        'details_df': lca_df
    }


def print_lca_warnings(results: Dict, baseline_name: str, target_name: str):
    """Print compact warnings for unique LCAs"""
    baseline_stats = results['baseline_only_stats']
    target_stats = results['target_only_stats']
    
    print(f"\n{'='*70}")
    print("TRACELENS DIFF - LCA ANALYSIS")
    print(f"{'='*70}")
    
    if baseline_stats['lca_ids']:
        print(f"\n⚠️  LCAs in {baseline_name} only:")
        print(f"   IDs: {baseline_stats['lca_ids'][:10]}{'...' if len(baseline_stats['lca_ids']) > 10 else ''}")
        print(f"   Kernels: {baseline_stats['num_kernels']} | "
              f"Runtime: {baseline_stats['cumulative_runtime']:.2f} ms | "
              f"Percent: {baseline_stats['percentage_of_total']:.2f}%")
    else:
        print(f"\n✓ No unique LCAs in {baseline_name}")
    
    if target_stats['lca_ids']:
        print(f"\n⚠️  LCAs in {target_name} only:")
        print(f"   IDs: {target_stats['lca_ids'][:10]}{'...' if len(target_stats['lca_ids']) > 10 else ''}")
        print(f"   Kernels: {target_stats['num_kernels']} | "
              f"Runtime: {target_stats['cumulative_runtime']:.2f} ms | "
              f"Percent: {target_stats['percentage_of_total']:.2f}%")
    else:
        print(f"\n✓ No unique LCAs in {target_name}")
    
    print(f"\n{'='*70}\n")


def save_lca_results(
    results: Dict,
    output_dir: Path,
    baseline_name: str,
    target_name: str
):
    """Save LCA analysis results to CSV files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_stats = results['baseline_only_stats']
    target_stats = results['target_only_stats']
    
    # Save individual trace details
    if baseline_stats['num_kernels'] > 0:
        csv_path = output_dir / f"{baseline_name}_only_operations.csv"
        baseline_stats['details_df'].to_csv(csv_path, index=False)
        print(f"  ✓ {baseline_name}-only ops: {csv_path}")
    
    if target_stats['num_kernels'] > 0:
        csv_path = output_dir / f"{target_name}_only_operations.csv"
        target_stats['details_df'].to_csv(csv_path, index=False)
        print(f"  ✓ {target_name}-only ops: {csv_path}")
    
    # Save summary
    summary = pd.DataFrame({
        'Trace': [baseline_name, target_name],
        'Unique LCAs': [len(baseline_stats['lca_ids']), len(target_stats['lca_ids'])],
        'Num Kernels': [baseline_stats['num_kernels'], target_stats['num_kernels']],
        'Runtime (ms)': [baseline_stats['cumulative_runtime'], target_stats['cumulative_runtime']],
        'Percentage (%)': [baseline_stats['percentage_of_total'], target_stats['percentage_of_total']]
    })
    
    summary_path = output_dir / "lca_analysis_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  ✓ Summary: {summary_path}")


def analyze_multi_module_lcas(
    diff_csv_path: str,
    baseline_name: str,
    target_name: str,
    baseline_source: str = "trace2",
    target_source: str = "trace1"
) -> Dict[str, Any]:
    """
    Find LCAs that contain kernels from multiple nn_module_parents.
    
    Args:
        diff_csv_path: Path to TraceDiff CSV
        baseline_name: Display name for baseline
        target_name: Display name for target
        baseline_source: Source column value for baseline
        target_source: Source column value for target
        
    Returns:
        Dict with multi_module_lcas info for each trace
    """
    df = pd.read_csv(diff_csv_path)
    
    # Validate columns
    required = ['source', 'lowest_common_ancestor_id', 'nn_module_parent', 'kernel_time']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Need: {required}")
    
    results = {}
    
    for source, display_name in [(baseline_source, baseline_name), (target_source, target_name)]:
        trace_df = df[df['source'] == source].copy()
        
        # Group by LCA and count unique nn_module_parents
        lca_groups = trace_df.groupby('lowest_common_ancestor_id').agg({
            'nn_module_parent': lambda x: x.nunique(),
            'kernel_time': 'sum',
            'name': 'count'
        }).reset_index()
        
        lca_groups.columns = ['lca_id', 'num_modules', 'total_runtime', 'num_kernels']
        
        # Filter to LCAs with multiple modules
        multi_module = lca_groups[lca_groups['num_modules'] > 1].copy()
        multi_module = multi_module.sort_values('total_runtime', ascending=False)
        
        # Get detailed info for these LCAs
        if len(multi_module) > 0:
            multi_lca_ids = multi_module['lca_id'].tolist()
            details_df = trace_df[trace_df['lowest_common_ancestor_id'].isin(multi_lca_ids)].copy()
            
            # Get module breakdown for each multi-module LCA
            module_breakdown = []
            for lca_id in multi_lca_ids:
                lca_data = trace_df[trace_df['lowest_common_ancestor_id'] == lca_id]
                modules = lca_data.groupby('nn_module_parent').agg({
                    'kernel_time': 'sum',
                    'name': 'count'
                }).reset_index()
                modules.columns = ['module', 'runtime', 'kernel_count']
                modules['lca_id'] = lca_id
                module_breakdown.append(modules)
            
            module_breakdown_df = pd.concat(module_breakdown, ignore_index=True)
        else:
            details_df = pd.DataFrame()
            module_breakdown_df = pd.DataFrame()
        
        total_time = trace_df['kernel_time'].sum()
        multi_time = multi_module['total_runtime'].sum()
        pct = (multi_time / total_time * 100) if total_time > 0 else 0.0
        
        results[source] = {
            'display_name': display_name,
            'num_multi_module_lcas': len(multi_module),
            'total_kernels': multi_module['num_kernels'].sum() if len(multi_module) > 0 else 0,
            'total_runtime': multi_time,
            'percentage_of_total': pct,
            'summary_df': multi_module,
            'details_df': details_df,
            'module_breakdown_df': module_breakdown_df
        }
    
    return {
        'baseline_stats': results[baseline_source],
        'target_stats': results[target_source],
        'baseline_source': baseline_source,
        'target_source': target_source
    }


def print_multi_module_warnings(results: Dict, baseline_name: str, target_name: str):
    """Print warnings for LCAs with multiple nn_module_parents"""
    baseline_stats = results['baseline_stats']
    target_stats = results['target_stats']
    
    print(f"\n{'='*70}")
    print("MULTI-MODULE LCA ANALYSIS")
    print(f"{'='*70}")
    
    if baseline_stats['num_multi_module_lcas'] > 0:
        print(f"\n⚠️  {baseline_name}: LCAs spanning multiple modules:")
        print(f"   Multi-module LCAs: {baseline_stats['num_multi_module_lcas']} | "
              f"Kernels: {baseline_stats['total_kernels']} | "
              f"Runtime: {baseline_stats['total_runtime']:.2f} ms | "
              f"Percent: {baseline_stats['percentage_of_total']:.2f}%")
        
        # Show top 5 by runtime
        top5 = baseline_stats['summary_df'].head(5)
        print(f"\n   Top 5 by runtime:")
        for _, row in top5.iterrows():
            print(f"     LCA {row['lca_id']}: {row['num_modules']} modules, "
                  f"{row['num_kernels']} kernels, {row['total_runtime']:.2f} ms")
    else:
        print(f"\n✓ {baseline_name}: No multi-module LCAs found")
    
    if target_stats['num_multi_module_lcas'] > 0:
        print(f"\n⚠️  {target_name}: LCAs spanning multiple modules:")
        print(f"   Multi-module LCAs: {target_stats['num_multi_module_lcas']} | "
              f"Kernels: {target_stats['total_kernels']} | "
              f"Runtime: {target_stats['total_runtime']:.2f} ms | "
              f"Percent: {target_stats['percentage_of_total']:.2f}%")
        
        # Show top 5 by runtime
        top5 = target_stats['summary_df'].head(5)
        print(f"\n   Top 5 by runtime:")
        for _, row in top5.iterrows():
            print(f"     LCA {row['lca_id']}: {row['num_modules']} modules, "
                  f"{row['num_kernels']} kernels, {row['total_runtime']:.2f} ms")
    else:
        print(f"\n✓ {target_name}: No multi-module LCAs found")
    
    print(f"\n{'='*70}\n")


def save_multi_module_results(
    results: Dict,
    output_dir: Path,
    baseline_name: str,
    target_name: str
):
    """Save multi-module LCA analysis results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_stats = results['baseline_stats']
    target_stats = results['target_stats']
    
    # Save summary
    if baseline_stats['num_multi_module_lcas'] > 0:
        summary_path = output_dir / f"{baseline_name}_multi_module_lcas.csv"
        baseline_stats['summary_df'].to_csv(summary_path, index=False)
        print(f"  ✓ {baseline_name} multi-module summary: {summary_path}")
        
        # Save module breakdown
        breakdown_path = output_dir / f"{baseline_name}_module_breakdown.csv"
        baseline_stats['module_breakdown_df'].to_csv(breakdown_path, index=False)
        print(f"  ✓ {baseline_name} module breakdown: {breakdown_path}")
    
    if target_stats['num_multi_module_lcas'] > 0:
        summary_path = output_dir / f"{target_name}_multi_module_lcas.csv"
        target_stats['summary_df'].to_csv(summary_path, index=False)
        print(f"  ✓ {target_name} multi-module summary: {summary_path}")
        
        # Save module breakdown
        breakdown_path = output_dir / f"{target_name}_module_breakdown.csv"
        target_stats['module_breakdown_df'].to_csv(breakdown_path, index=False)
        print(f"  ✓ {target_name} module breakdown: {breakdown_path}")


def prepare_cleaned_diff_data(
    diff_csv_path: str,
    baseline_name: str,
    target_name: str,
    unique_lca_results: Dict,
    multi_module_results: Dict
) -> pd.DataFrame:
    """
    Clean diff data by:
    1. Assign special LCA IDs to unique and multi-module LCAs
    2. Remove those kernels from analysis
    3. Return cleaned dataframe
    
    Args:
        diff_csv_path: Path to TraceDiff CSV
        baseline_name: Display name for baseline
        target_name: Display name for target
        unique_lca_results: Results from analyze_lca_differences()
        multi_module_results: Results from analyze_multi_module_lcas()
        
    Returns:
        Cleaned DataFrame with special LCA IDs assigned and excluded kernels removed
    """
    df = pd.read_csv(diff_csv_path)
    
    baseline_source = unique_lca_results['baseline_source']
    target_source = unique_lca_results['target_source']
    
    # Get LCA IDs to exclude
    unique_baseline = set(unique_lca_results['baseline_only_stats']['lca_ids'])
    unique_target = set(unique_lca_results['target_only_stats']['lca_ids'])
    
    multi_baseline = set(multi_module_results['baseline_stats']['summary_df']['lca_id'].tolist()) \
                     if multi_module_results['baseline_stats']['num_multi_module_lcas'] > 0 else set()
    multi_target = set(multi_module_results['target_stats']['summary_df']['lca_id'].tolist()) \
                   if multi_module_results['target_stats']['num_multi_module_lcas'] > 0 else set()
    
    all_excluded = unique_baseline | unique_target | multi_baseline | multi_target
    
    print(f"\n📋 Cleaning diff data:")
    print(f"   Total rows: {len(df)}")
    print(f"   Excluded LCAs: {len(all_excluded)}")
    
    # Assign special LCA IDs and mark for removal
    df = df.copy()
    df['_exclude'] = False
    
    # Unique LCAs
    df.loc[(df['source'] == baseline_source) & (df['lowest_common_ancestor_id'].isin(unique_baseline)), 
           'lowest_common_ancestor_id'] = -1  # Special ID for unique baseline
    df.loc[(df['source'] == baseline_source) & (df['lowest_common_ancestor_id'] == -1), '_exclude'] = True
    
    df.loc[(df['source'] == target_source) & (df['lowest_common_ancestor_id'].isin(unique_target)), 
           'lowest_common_ancestor_id'] = -2  # Special ID for unique target
    df.loc[(df['source'] == target_source) & (df['lowest_common_ancestor_id'] == -2), '_exclude'] = True
    
    # Multi-module LCAs
    df.loc[(df['source'] == baseline_source) & (df['lowest_common_ancestor_id'].isin(multi_baseline)), 
           'lowest_common_ancestor_id'] = -3  # Special ID for multi-module baseline
    df.loc[(df['source'] == baseline_source) & (df['lowest_common_ancestor_id'] == -3), '_exclude'] = True
    
    df.loc[(df['source'] == target_source) & (df['lowest_common_ancestor_id'].isin(multi_target)), 
           'lowest_common_ancestor_id'] = -4  # Special ID for multi-module target
    df.loc[(df['source'] == target_source) & (df['lowest_common_ancestor_id'] == -4), '_exclude'] = True
    
    # Remove excluded rows
    excluded_count = df['_exclude'].sum()
    df_cleaned = df[~df['_exclude']].drop(columns=['_exclude'])
    
    print(f"   Excluded rows: {excluded_count}")
    print(f"   Remaining rows: {len(df_cleaned)}")
    
    return df_cleaned


def find_nn_lca(stack1: str, stack2: str) -> str:
    """
    Find the lowest common ancestor in nn_module_stack.
    
    Args:
        stack1: nn_module_stack from trace 1
        stack2: nn_module_stack from trace 2
        
    Returns:
        Lowest common nn module or "root"
    """
    if pd.isna(stack1) or pd.isna(stack2):
        return "root"
    
    # Parse stacks - format is semicolon-separated
    parts1 = str(stack1).split(';')
    parts2 = str(stack2).split(';')
    
    # Find common prefix
    common = []
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            common.append(p1)
        else:
            break
    
    return common[-1] if common else "root"


def analyze_problematic_groups(df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """
    Group by LCA ID (across both traces).
    Find LCAs with multiple nn_module_parents - these are "problematic".
    For problematic LCAs, compute nn_module LCA from the stacks.
    
    Args:
        df_cleaned: Cleaned DataFrame from prepare_cleaned_diff_data()
        
    Returns:
        Dict with problematic groups info and corrected nn_parent column data
    """
    print(f"\n{'='*70}")
    print("PROBLEMATIC GROUP ANALYSIS")
    print(f"{'='*70}\n")
    
    # Group by LCA ID only (across both traces)
    grouped = df_cleaned.groupby('lowest_common_ancestor_id').agg({
        'nn_module_parent': lambda x: list(x.unique()),
        'nn_module_stack': lambda x: list(x.unique()),
        'source': lambda x: list(x.unique()),
        'kernel_time': 'sum',
        'name': 'count'
    }).reset_index()
    
    grouped.columns = ['lca_id', 'nn_modules', 'nn_stacks', 'sources', 'total_runtime', 'kernel_count']
    
    # Count unique nn_module_parents per LCA
    grouped['num_modules'] = grouped['nn_modules'].apply(len)
    
    # Problematic LCAs have multiple nn_module_parents (should be exactly 2)
    problematic = grouped[grouped['num_modules'] > 1].copy()
    
    print(f"📊 Group statistics:")
    print(f"   Total LCAs: {len(grouped)}")
    print(f"   Problematic LCAs: {len(problematic)}")
    
    if len(problematic) == 0:
        print(f"\n✓ No problematic LCAs found\n")
        print(f"{'='*70}\n")
        
        # All kernels keep their original nn_module_parent
        df_result = df_cleaned.copy()
        df_result['corrected_nn_parent'] = df_result['nn_module_parent']
        
        return {
            'num_problematic': 0,
            'problematic_df': pd.DataFrame(),
            'df_with_corrections': df_result
        }
    
    # Compute nn_module LCA for problematic groups
    def compute_nn_lca_for_group(row):
        stacks = row['nn_stacks']
        if len(stacks) < 2:
            return "root"
        
        # Find LCA between the two stacks (should be exactly 2)
        if len(stacks) == 2:
            return find_nn_lca(stacks[0], stacks[1])
        else:
            # More than 2 stacks - find LCA across all
            lca = stacks[0]
            for stack in stacks[1:]:
                lca = find_nn_lca(lca, stack)
            return lca
    
    problematic['nn_module_lca'] = problematic.apply(compute_nn_lca_for_group, axis=1)
    
    print(f"\n⚠️  Top 10 problematic LCAs by runtime:")
    top10 = problematic.nlargest(10, 'total_runtime')
    for idx, row in top10.iterrows():
        print(f"   LCA {row['lca_id']}: {row['num_modules']} modules, "
              f"{row['kernel_count']} kernels, {row['total_runtime']:.2f} ms")
        print(f"      Modules: {row['nn_modules']}")
        print(f"      NN LCA: '{row['nn_module_lca']}'")
    
    print(f"\n{'='*70}\n")
    
    # Create corrected_nn_parent column
    df_result = df_cleaned.copy()
    
    # Start with original values
    df_result['corrected_nn_parent'] = df_result['nn_module_parent']
    
    # Build lookup for problematic LCAs
    problematic_lookup = {}
    for _, row in problematic.iterrows():
        lca_id = row['lca_id']
        nn_lca = row['nn_module_lca']
        problematic_lookup[lca_id] = nn_lca
    
    # Apply corrections: replace nn_module_parent with nn_module_lca for problematic LCAs
    def get_corrected_parent(row):
        lca_id = row['lowest_common_ancestor_id']
        if lca_id in problematic_lookup:
            return problematic_lookup[lca_id]
        return row['nn_module_parent']
    
    df_result['corrected_nn_parent'] = df_result.apply(get_corrected_parent, axis=1)
    
    corrections_made = (df_result['corrected_nn_parent'] != df_result['nn_module_parent']).sum()
    print(f"✓ Applied corrections to {corrections_made} rows")
    
    return {
        'num_problematic': len(problematic),
        'problematic_df': problematic,
        'df_with_corrections': df_result,
        'corrections_made': corrections_made
    }


def save_problematic_results(
    results: Dict,
    output_dir: Path
):
    """Save problematic group analysis results"""
    if results['num_problematic'] == 0:
        print(f"  ℹ️  No problematic groups to save")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save problematic groups summary
    prob_df = results['problematic_df'].copy()
    
    # Convert list columns to string for CSV compatibility
    if 'sources' in prob_df.columns:
        prob_df['sources'] = prob_df['sources'].apply(lambda x: ', '.join(map(str, x)))
    if 'nn_stacks' in prob_df.columns:
        prob_df['nn_stacks'] = prob_df['nn_stacks'].apply(lambda x: ' | '.join(map(str, x)))
    if 'nn_modules' in prob_df.columns:
        prob_df['nn_modules'] = prob_df['nn_modules'].apply(lambda x: ', '.join(map(str, x)))
    
    prob_path = output_dir / "problematic_groups.csv"
    prob_df.to_csv(prob_path, index=False)
    print(f"  ✓ Problematic groups: {prob_path}")
    
    # Save corrected diff data
    corrected_path = output_dir / "diff_corrected.csv"
    results['df_with_corrections'].to_csv(corrected_path, index=False)
    print(f"  ✓ Corrected diff data: {corrected_path}")


def calculate_lca_optimization_opportunities(
    df_with_corrections: pd.DataFrame,
    baseline_name: str,
    target_name: str,
    baseline_source: str = "trace2",
    target_source: str = "trace1"
) -> pd.DataFrame:
    """
    Calculate optimization opportunities for baseline.
    Predicted optimized performance = min(baseline_time, target_time) for each LCA.
    
    Args:
        df_with_corrections: DataFrame with corrected_nn_parent column
        baseline_name: Display name for baseline (being optimized)
        target_name: Display name for target (performance goal)
        baseline_source: Source column value for baseline
        target_source: Source column value for target
        
    Returns:
        DataFrame with LCA-level optimization opportunities for baseline
        
    Raises:
        ValueError: If any LCAs are only present in one trace (data not properly cleaned)
    """
    print(f"\n{'='*70}")
    print(f"LCA OPTIMIZATION OPPORTUNITIES ({baseline_name})")
    print(f"{'='*70}\n")
    
    # Validate source values exist
    sources = set(df_with_corrections['source'].unique())
    if baseline_source not in sources:
        raise ValueError(f"Baseline source '{baseline_source}' not found in data. Available: {sources}")
    if target_source not in sources:
        raise ValueError(f"Target source '{target_source}' not found in data. Available: {sources}")
    
    # Validate: Check for LCAs only in one trace
    baseline_lcas = set(df_with_corrections[df_with_corrections['source'] == baseline_source]['lowest_common_ancestor_id'].unique())
    target_lcas = set(df_with_corrections[df_with_corrections['source'] == target_source]['lowest_common_ancestor_id'].unique())
    
    only_in_baseline = baseline_lcas - target_lcas
    only_in_target = target_lcas - baseline_lcas
    
    if only_in_baseline or only_in_target:
        error_msg = "Data not properly cleaned - found LCAs in only one trace:\n"
        if only_in_baseline:
            error_msg += f"  Only in {baseline_name} ({baseline_source}): {sorted(list(only_in_baseline))}\n"
        if only_in_target:
            error_msg += f"  Only in {target_name} ({target_source}): {sorted(list(only_in_target))}\n"
        error_msg += "These should have been filtered out in prepare_cleaned_diff_data()"
        raise ValueError(error_msg)
    
    print(f"✓ Data validation passed: All {len(baseline_lcas)} LCAs present in both traces\n")
    
    # Group by LCA and source to get totals per trace
    lca_by_source = df_with_corrections.groupby(['lowest_common_ancestor_id', 'source']).agg({
        'kernel_time': 'sum',
        'corrected_nn_parent': 'first',  # Should be same for all rows in LCA
        'name': 'count'
    }).reset_index()
    
    # Pivot to get baseline and target side by side
    lca_pivot = lca_by_source.pivot(
        index='lowest_common_ancestor_id',
        columns='source',
        values='kernel_time'
    ).reset_index()
    
    # Get corrected_nn_parent (same for both traces)
    lca_modules = df_with_corrections.groupby('lowest_common_ancestor_id')['corrected_nn_parent'].first().reset_index()
    lca_pivot = lca_pivot.merge(lca_modules, on='lowest_common_ancestor_id')
    
    # Rename columns for clarity
    lca_pivot = lca_pivot.rename(columns={
        baseline_source: 'baseline_time',
        target_source: 'target_time'
    })
    
    # Calculate predicted optimized time (min of both traces)
    lca_pivot['predicted_optimized'] = lca_pivot[['baseline_time', 'target_time']].min(axis=1)
    
    # Calculate opportunity (only for baseline)
    lca_pivot['opportunity'] = lca_pivot['baseline_time'] - lca_pivot['predicted_optimized']
    lca_pivot['opportunity_pct'] = (
        lca_pivot['opportunity'] / lca_pivot['baseline_time'] * 100
    ).fillna(0.0)
    
    # Sort by opportunity
    lca_pivot = lca_pivot.sort_values('opportunity', ascending=False)
    
    # Calculate totals
    baseline_total = lca_pivot['baseline_time'].sum()
    target_total = lca_pivot['target_time'].sum()
    total_opportunity = lca_pivot['opportunity'].sum()
    
    print(f"📊 Summary:")
    print(f"   Total LCAs: {len(lca_pivot)}")
    print(f"   Baseline ({baseline_name}) time: {baseline_total:.2f} ms")
    print(f"   Target ({target_name}) time: {target_total:.2f} ms")
    print(f"   Total opportunity: {total_opportunity:.2f} ms ({total_opportunity/baseline_total*100:.2f}%)")
    
    # Show top opportunities
    print(f"\n⚡ Top 10 optimization opportunities:")
    top10 = lca_pivot.head(10)
    for idx, row in top10.iterrows():
        print(f"   LCA {row['lowest_common_ancestor_id']} ({row['corrected_nn_parent']}): "
              f"{row['opportunity']:.2f} microseconds ({row['opportunity_pct']:.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    return lca_pivot
def save_lca_optimization_opportunities(
    lca_opportunities: pd.DataFrame,
    output_dir: Path,
    baseline_name: str,
    target_name: str,
    df_with_corrections: pd.DataFrame = None
):
    """Save LCA optimization opportunities to CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    opp_path = output_dir / "lca_optimization_opportunities.csv"
    lca_opportunities.to_csv(opp_path, index=False)
    print(f"  ✓ LCA optimization opportunities: {opp_path}")
    
    # Get kernel names and CPU op names from both baseline and target if df_with_corrections is provided
    if df_with_corrections is not None:
        # Get baseline and target sources from df_with_corrections
        sources = df_with_corrections['source'].unique()
        baseline_source = 'trace2' if 'trace2' in sources else sorted(sources)[0]
        target_source = 'trace1' if 'trace1' in sources else sorted(sources)[1]
        
        # Extract from baseline
        baseline_df = df_with_corrections[df_with_corrections['source'] == baseline_source].copy()
        lca_baseline_info = baseline_df.groupby('lowest_common_ancestor_id').agg({
            'name': lambda x: '***'.join(sorted(set(x))),  # kernel names
            'cpu_op_name': lambda x: '***'.join(sorted(set(x))) if 'cpu_op_name' in baseline_df.columns else ''  # cpu op names
        }).reset_index()
        lca_baseline_info.columns = ['lowest_common_ancestor_id', 'baseline_kernel_names', 'baseline_cpu_op_names']
        
        # Extract from target
        target_df = df_with_corrections[df_with_corrections['source'] == target_source].copy()
        lca_target_info = target_df.groupby('lowest_common_ancestor_id').agg({
            'name': lambda x: '***'.join(sorted(set(x))),  # kernel names
            'cpu_op_name': lambda x: '***'.join(sorted(set(x))) if 'cpu_op_name' in target_df.columns else ''  # cpu op names
        }).reset_index()
        lca_target_info.columns = ['lowest_common_ancestor_id', 'target_kernel_names', 'target_cpu_op_names']
        
        # Merge both into lca_opportunities
        lca_opportunities = lca_opportunities.merge(lca_baseline_info, on='lowest_common_ancestor_id', how='left')
        lca_opportunities = lca_opportunities.merge(lca_target_info, on='lowest_common_ancestor_id', how='left')
        
        # Save updated opportunities with all info
        opp_with_kernels_path = output_dir / "lca_optimization_opportunities_with_kernels.csv"
        lca_opportunities.to_csv(opp_with_kernels_path, index=False)
        print(f"  ✓ LCA opportunities with kernel names: {opp_with_kernels_path}")
        
        # Create kernel-level summary (aggregated by baseline kernel name)
        kernel_summary = lca_opportunities.groupby('baseline_kernel_names').agg({
            'lowest_common_ancestor_id': 'count',
            'baseline_time': 'sum',
            'target_time': 'sum',
            'opportunity': 'sum',
            'corrected_nn_parent': lambda x: ', '.join(sorted(set(x))),
            'target_kernel_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***')))),
            'baseline_cpu_op_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***')))),
            'target_cpu_op_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***'))))
        }).reset_index()
        
        kernel_summary.columns = [
            'baseline_kernel_names',
            'num_aggregated_LCAs',
            'baseline_time',
            'target_time',
            'opportunity',
            'nn_modules',
            'target_kernel_names',
            'baseline_cpu_op_names',
            'target_cpu_op_names'
        ]
        
        # Calculate percentage
        kernel_summary['opportunity_pct'] = (
            kernel_summary['opportunity'] / kernel_summary['baseline_time'] * 100
        ).fillna(0.0)
        
        kernel_summary = kernel_summary.sort_values('opportunity', ascending=False)
        
        kernel_path = output_dir / "kernel_optimization_summary.csv"
        kernel_summary.to_csv(kernel_path, index=False)
        print(f"  ✓ Kernel-level summary: {kernel_path}")
    
    # Also save a summary grouped by corrected_nn_parent
    module_summary = lca_opportunities.groupby('corrected_nn_parent').agg({
        'lowest_common_ancestor_id': 'count',
        'baseline_time': 'sum',
        'target_time': 'sum',
        'opportunity': 'sum'
    }).reset_index()
    
    # Add kernel and cpu op names aggregation if available
    if 'baseline_kernel_names' in lca_opportunities.columns:
        module_agg = lca_opportunities.groupby('corrected_nn_parent').agg({
            'baseline_kernel_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***')))),
            'target_kernel_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***')))),
            'baseline_cpu_op_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***')))),
            'target_cpu_op_names': lambda x: '***'.join(sorted(set('***'.join(x.dropna()).split('***'))))
        }).reset_index()
        module_summary = module_summary.merge(module_agg, on='corrected_nn_parent', how='left')
    
    module_summary.columns = [
        'nn_module',
        'num_aggregated_LCAs',
        'baseline_time',
        'target_time',
        'opportunity'
    ] + (['baseline_kernel_names', 'target_kernel_names', 'baseline_cpu_op_names', 'target_cpu_op_names'] 
         if 'baseline_kernel_names' in lca_opportunities.columns else [])
    
    # Calculate percentage
    module_summary['opportunity_pct'] = (
        module_summary['opportunity'] / module_summary['baseline_time'] * 100
    ).fillna(0.0)
    
    module_summary = module_summary.sort_values('opportunity', ascending=False)
    
    module_path = output_dir / "module_optimization_summary.csv"
    module_summary.to_csv(module_path, index=False)
    print(f"  ✓ Module-level summary: {module_path}")