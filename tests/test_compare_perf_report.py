import os
import subprocess
import json
from datetime import datetime
import argparse
import pandas as pd
from pandas.api.types import is_float_dtype

import TraceLens

def compare_cols(df_test, df_ref, cols, tol=1e-6):
    """Compare columns in two dataframes, skipping rows where ref is None/NaN.
    Args:
        df_test: test dataframe
        df_ref: reference dataframe
        cols: list of column names to compare
        tol: tolerance for comparison (used only for float comparisons)
    Returns:
        list of column names that are different
    """
    diff_cols = []
    for col in cols:
        # Filter valid rows: ref is not NaN/None
        valid_mask = df_ref[col].notna()
        if not valid_mask.any():
            continue  # skip if ref is empty for this column

        ref_col = df_ref.loc[valid_mask, col]
        test_col = df_test.loc[df_test.index.intersection(ref_col.index), col]

        # Align indices (important!)
        test_col, ref_col = test_col.align(ref_col, join="right")  # align to ref

        if is_float_dtype(df_test[col]):
            diff = test_col - ref_col
            if not diff.abs().max() < tol:
                diff_cols.append(col)
        else:
            if not (test_col == ref_col).all():
                diff_cols.append(col)

    return diff_cols

def test_perf_report_regression(profile_path, ref_report_path, fn_report_path):
    # cols = ['GFLOPS_first', 'FLOPS/Byte_first', 'TFLOPS/s_mean', 'TB/s_mean']

    # generate script cmd construction
    tracelens_repo_path = os.path.dirname(TraceLens.__path__[0])
    # generate_script_path = "/home/ajassani/TraceLens/examples/generate_perf_report.py"
    generate_script_path = os.path.join(tracelens_repo_path, "examples", "generate_perf_report.py")
    cmd = f"python {generate_script_path} --profile_json_path {profile_path} --output_xlsx_path {fn_report_path}"
    # run the script
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    sheets = pd.ExcelFile(ref_report_path).sheet_names
    # this map is required for older format and will be removed 
    ref2fn_sheet_map = {"FLASH_ATTN_fwd": "SDPA_fwd", "FLASH_ATTN_bwd": "SDPA_bwd",
                 "CONV_fwd": "CONV_fwd", "CONV_bwd": "CONV_bwd",
                 "UNARY_ELEMWISE": "UnaryElementwise",
                 "BINARY_ELEMWISE": "BinaryElementwise"}
    cols_ignore = ['Non-Data-Mov TFLOPS/s_mean', 'Non-Data-Mov Kernel Time (µs)_sum', 'Non-Data-Mov Kernel Time (µs)_mean']
    case_passed = True
    for sheet in sheets:
        # skip gpu_timeline sheet was required as there was change in the calculation methodology - again, this skip will be removed
        if sheet == "gpu_timeline":
            continue
        df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
        df_fn = pd.read_excel(fn_report_path, sheet_name=ref2fn_sheet_map.get(sheet, sheet))

        # if df_ref is empty, skip
        if df_ref.empty:
            continue
        cols = df_ref.columns
        cols = [col for col in cols if col not in cols_ignore]
        # rename foll cols in fn report to match ref report, again this is required for older format and will be removed in future
        rename_cols = {"param: N_KV": "param: N_K", "param: H_Q": "param: H", "param: d_h": "param: d_k"}
        for col in rename_cols:
            if col in df_fn.columns:
                df_fn.rename(columns={col: rename_cols[col]}, inplace=True)
        try:
            diff_cols = compare_cols(df_fn, df_ref, cols)
        except Exception as e:
            print(f"Error comparing sheet {sheet}: {e}")
            return False
        if diff_cols:
            print(f"Sheet {sheet}: {diff_cols} are different")
            return False
    os.remove(fn_report_path)  # delete the fn report if all sheets passed
    return True

def main(args):
    ref_root = args.ref_root
    fn_root = args.fn_root
    profile_data_json = args.profile_data_json
    with open(profile_data_json, 'r') as f:
        list_ref = json.load(f)
    # name fn report based on curr timestamp
    fn_root = fn_root + datetime.now().strftime("%Y%m%d%H%M%S")
    # make fn root
    os.makedirs(fn_root)
    failed_cases = []
    for ref in list_ref:
        print(f"Testing profile: {ref['profile_path']}")
        profile_path = ref["profile_path"]
        report_filename = ref["report_filename"]
        ref_report_path = os.path.join(ref_root, report_filename)
        fn_report_path = os.path.join(fn_root, report_filename)
        passed = test_perf_report_regression(profile_path, ref_report_path, fn_report_path)
        if not passed:
            failed_cases.append({"profile_path": profile_path, "report_filename": report_filename})
            print(f"Failed for this case")

    num_failed = len(failed_cases)
    num_passed = len(list_ref) - num_failed
    print(f"Passed: {num_passed}, Failed: {num_failed}")

    if num_failed == 0:
        print("All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perf report regression test")
    parser.add_argument("--ref_root", type=str, required=True, help="Root directory for reference reports")
    parser.add_argument("--fn_root", type=str, required=True, help="Root directory for function reports")
    parser.add_argument("--profile_data_json", type=str, required=True, help="Path to profile data JSON file. This contains the profile path and report filename.")
    # example {
    #    "profile_path": "/path/to/profile.json",
    #    "report_filename": "report.xlsx"
    # }
    args = parser.parse_args()
    main(args)
