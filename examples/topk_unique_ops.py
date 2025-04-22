import argparse
import pandas as pd
from TraceLens import TreePerfAnalyzer
from typing import List, Dict, Any

def generate_unique_ops_view(df_kernel_launchers: pd.DataFrame, sort_by_time: bool = True) -> pd.DataFrame:
    df_grouped = df_kernel_launchers.copy()
    grouping_cols_original = ['name', 'Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs']
    str_col_names, actual_grouping_cols = [], []

    for col in grouping_cols_original:
        if col not in df_grouped.columns:
            continue
        actual_grouping_cols.append(col)
        str_col_name = f"{col}_str_repr_for_grouping"
        df_grouped[str_col_name] = df_grouped[col].apply(str)
        str_col_names.append(str_col_name)

    if not str_col_names:
        raise ValueError("No valid columns found to group by.")

    agg_dict = {}
    columns_to_keep_first = []

    if 'total_direct_kernel_time' in df_grouped.columns:
        agg_dict['total_direct_kernel_time'] = ['mean', 'sum']
    if 'UID' in df_grouped.columns:
        agg_dict['UID'] = ['first', 'count']
        columns_to_keep_first.append('UID')
    if 'kernel_names' in df_grouped.columns:
        agg_dict['kernel_names'] = 'first'
        columns_to_keep_first.append('kernel_names')

    for col in actual_grouping_cols:
        agg_dict[col] = 'first'
        columns_to_keep_first.append(col)

    df_unique_ops = df_grouped.groupby(str_col_names, dropna=False, sort=False).agg(agg_dict)

    def flatten_cols(col_obj):
        if isinstance(col_obj, tuple):
            return '_'.join(filter(None, col_obj)).strip('_')
        return str(col_obj)

    df_unique_ops.columns = [flatten_cols(col) for col in df_unique_ops.columns.values]
    df_unique_ops.reset_index(inplace=True)

    rename_map = {}
    if 'UID_count' in df_unique_ops.columns:
        rename_map['UID_count'] = 'operation_count'
    for col in columns_to_keep_first:
        col_first_name = f'{col}_first'
        if col_first_name in df_unique_ops.columns:
            rename_map[col_first_name] = col

    df_unique_ops.rename(columns=rename_map, inplace=True)

    sum_time_col_name = 'total_direct_kernel_time_sum'
    final_columns = [col for col in grouping_cols_original if col in df_unique_ops.columns]
    final_columns += [col for col in ['UID', 'operation_count', 'kernel_names', 'total_direct_kernel_time_mean', sum_time_col_name] if col in df_unique_ops.columns]
    seen = set()
    final_columns_ordered = []
    for col in final_columns:
        if col in df_unique_ops.columns and col not in seen:
            final_columns_ordered.append(col)
            seen.add(col)
    for col in df_unique_ops.columns:
        if col not in seen and not col.endswith("_str_repr_for_grouping"):
            final_columns_ordered.append(col)
            seen.add(col)

    df_unique_ops = df_unique_ops[final_columns_ordered]

    if sort_by_time and sum_time_col_name in df_unique_ops.columns:
        df_unique_ops = df_unique_ops.sort_values(by=sum_time_col_name, ascending=False).reset_index(drop=True)

    return df_unique_ops

def main():
    parser = argparse.ArgumentParser(description="Generate Unique Ops View from PyTorch Profile")
    parser.add_argument('--input', type=str, required=True, help='Input profile path (JSON/trace)')
    parser.add_argument('--output', type=str, required=True, help='Output Excel path')
    parser.add_argument('--topk', type=int, default=50, help='Number of top ops to save (default: 50)')
    args = parser.parse_args()

    print(f"Loading profile from: {args.input}")
    perf_analyzer = TreePerfAnalyzer.from_file(args.input)
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)

    print(f"Generating unique ops view...")
    df_ops_view = generate_unique_ops_view(df_kernel_launchers)

    topk = args.topk
    print(f"Saving top-{topk} ops to Excel at: {args.output}")
    df_ops_view.head(topk).to_excel(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
