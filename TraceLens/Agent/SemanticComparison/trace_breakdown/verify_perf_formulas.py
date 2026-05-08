#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Verify our perf model formulas against TraceLens reference CSVs.

Reads TraceLens-generated per-category CSVs (GEMM, SDPA_fwd, UnaryElementwise,
BinaryElementwise) and checks that our category_mappings.py formulas produce
identical GFLOPS and Data Moved (MB) values.

Usage:
    python verify_perf_formulas.py --ref-dir <path_to_perf_report_csvs>
"""

import argparse
import ast
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from category_mappings import (
    _gemm_flops,
    _gemm_bytes,
    _sdpa_flops,
    _sdpa_bytes,
    _elementwise_flops,
    _elementwise_bytes,
    _unary_elementwise_bytes,
    _binary_elementwise_bytes,
)

MiB = 1024 * 1024

TRACELENS_DTYPE_TO_BPE = {
    "double": 8,
    "long int": 8,
    "float": 4,
    "scalar": 4,
    "int": 4,
    "c10::half": 2,
    "c10::bfloat16": 2,
    "c10::float8_e4m3fnuz": 1,
    "c10::float8_e4m3fn": 1,
    "c10::float8_e5m2": 1,
    "unsigned char": 1,
    "signed char": 1,
    "fp8": 1,
}


def _bpe(dtype_str):
    """Map a TraceLens dtype string to bytes-per-element, or None."""
    if dtype_str is None or dtype_str == "None":
        return None
    return TRACELENS_DTYPE_TO_BPE.get(dtype_str.lower().strip(), None)


def _parse_tuple(s):
    """Parse a string representation of a Python tuple."""
    if pd.isna(s) or s == "":
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def _product(shape):
    """Product of a shape tuple, treating empty tuple as 1."""
    if not shape:
        return 1
    r = 1
    for x in shape:
        r *= x
    return r


def _broadcast_shape(s1, s2):
    """Compute numpy-style broadcast output shape."""
    if not s1:
        return s2
    if not s2:
        return s1
    ndim = max(len(s1), len(s2))
    s1 = (1,) * (ndim - len(s1)) + tuple(s1)
    s2 = (1,) * (ndim - len(s2)) + tuple(s2)
    out = []
    for a, b in zip(s1, s2):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            out.append(max(a, b))
    return tuple(out)


class Result:
    def __init__(self, category):
        self.category = category
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def record_pass(self):
        self.total += 1
        self.passed += 1

    def record_skip(self, reason=""):
        self.total += 1
        self.skipped += 1

    def record_fail(self, row_desc, detail):
        self.total += 1
        self.failed += 1
        self.failures.append((row_desc, detail))


def _close(a, b, rel_tol=1e-6, abs_tol=1e-3):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# -- GEMM -------------------------------------------------------------------


def verify_gemm(csv_path):
    res = Result("GEMM")
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        M = int(row["param: M"])
        N = int(row["param: N"])
        K = int(row["param: K"])
        bias = row["param: bias"]
        if isinstance(bias, str):
            bias = bias.strip().lower() == "true"
        bias = bool(bias)

        dtype_str = str(row["param: dtype_A_B"])
        dtypes = _parse_tuple(dtype_str)
        if dtypes is None or len(dtypes) < 2:
            res.record_skip("cannot parse dtype_A_B")
            continue

        bpe_A = _bpe(dtypes[0])
        bpe_B = _bpe(dtypes[1])
        if bpe_A is None or bpe_B is None:
            res.record_skip("unknown dtype")
            continue
        bpe_out = bpe_A

        ref_gflops = row["GFLOPS_first"]
        ref_data_mb = row["Data Moved (MB)_first"]

        our_flops = _gemm_flops(M, N, K, bias)
        our_gflops = our_flops / 1e9

        our_bytes = _gemm_bytes(
            M,
            N,
            K,
            bpe_A,
            bpe_B,
            bpe_out,
            bias=bias,
            bpe_bias=bpe_out if bias else None,
        )
        our_data_mb = our_bytes / MiB

        desc = f"Row {idx}: aten::mm M={M} N={N} K={K} bias={bias}"

        ok_flops = _close(our_gflops, ref_gflops)
        ok_bytes = True
        if not pd.isna(ref_data_mb):
            ok_bytes = _close(our_data_mb, ref_data_mb)

        if ok_flops and ok_bytes:
            res.record_pass()
        else:
            detail = []
            if not ok_flops:
                detail.append(f"GFLOPS: ours={our_gflops:.6f} ref={ref_gflops:.6f}")
            if not ok_bytes:
                detail.append(f"Data MB: ours={our_data_mb:.6f} ref={ref_data_mb:.6f}")
            res.record_fail(desc, "; ".join(detail))

    return res


# -- SDPA -------------------------------------------------------------------


def verify_sdpa(csv_path):
    res = Result("SDPA_fwd")
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        B = int(row["param: B"])
        N_Q = int(row["param: N_Q"])
        H_Q = int(row["param: H_Q"])
        N_KV = int(row["param: N_KV"])
        H_KV = int(row["param: H_KV"])
        d_qk = int(row["param: d_h_qk"])
        d_v = int(row["param: d_h_v"])
        causal = row["param: causal"]
        if isinstance(causal, str):
            causal = causal.strip().lower() == "true"
        causal = bool(causal)

        input_type_str = str(row.get("Input type_first", ""))
        dtypes = _parse_tuple(input_type_str)
        if dtypes and len(dtypes) > 0:
            bpe = _bpe(dtypes[0])
        else:
            bpe = 2
        if bpe is None:
            bpe = 2

        ref_gflops = row["GFLOPS_first"]
        ref_data_mb = row["Data Moved (MB)_first"]

        our_flops = _sdpa_flops(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, causal)
        our_gflops = our_flops / 1e9

        our_bytes = _sdpa_bytes(B, N_Q, H_Q, N_KV, H_KV, d_qk, d_v, bpe)
        our_data_mb = our_bytes / MiB

        desc = (
            f"Row {idx}: SDPA B={B} N_Q={N_Q} H_Q={H_Q} N_KV={N_KV} "
            f"H_KV={H_KV} d_qk={d_qk} d_v={d_v} causal={causal}"
        )

        ok_flops = _close(our_gflops, ref_gflops)
        ok_bytes = True
        if not pd.isna(ref_data_mb):
            ok_bytes = _close(our_data_mb, ref_data_mb)

        if ok_flops and ok_bytes:
            res.record_pass()
        else:
            detail = []
            if not ok_flops:
                detail.append(f"GFLOPS: ours={our_gflops:.6f} ref={ref_gflops:.6f}")
            if not ok_bytes:
                detail.append(f"Data MB: ours={our_data_mb:.6f} ref={ref_data_mb:.6f}")
            res.record_fail(desc, "; ".join(detail))

    return res


# -- Unary Elementwise ------------------------------------------------------


def verify_unary_elementwise(csv_path):
    res = Result("UnaryElementwise")
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        shape_str = str(row["param: op_shape"])
        shape = _parse_tuple(shape_str)
        if shape is None:
            res.record_skip("cannot parse op_shape")
            continue
        nelems = _product(shape)

        dtype_str = str(row["param: dtype_in_out"])
        dtypes = _parse_tuple(dtype_str)
        if dtypes is None or len(dtypes) < 2:
            res.record_skip("cannot parse dtype_in_out")
            continue

        bpe_in = _bpe(dtypes[0])
        bpe_out = _bpe(dtypes[1])

        ref_gflops = row["GFLOPS_first"]
        ref_data_mb = row.get("Data Moved (MB)_first", float("nan"))

        our_flops = _elementwise_flops(nelems)
        our_gflops = our_flops / 1e9

        ok_flops = _close(our_gflops, ref_gflops)

        ok_bytes = True
        bytes_detail = ""
        if not pd.isna(ref_data_mb) and bpe_in is not None and bpe_out is not None:
            our_bytes = _unary_elementwise_bytes(nelems, bpe_in, bpe_out)
            our_data_mb = our_bytes / MiB
            ok_bytes = _close(our_data_mb, ref_data_mb)
            if not ok_bytes:
                bytes_detail = (
                    f"Data MB: our_unary={our_data_mb:.6f} " f"ref={ref_data_mb:.6f}"
                )

        desc = f"Row {idx}: {row['name']} shape={shape} dtypes={dtypes}"

        if ok_flops and ok_bytes:
            res.record_pass()
        else:
            detail = []
            if not ok_flops:
                detail.append(f"GFLOPS: ours={our_gflops:.6f} ref={ref_gflops:.6f}")
            if bytes_detail:
                detail.append(bytes_detail)
            res.record_fail(desc, "; ".join(detail))

    return res


# -- Binary Elementwise -----------------------------------------------------


def verify_binary_elementwise(csv_path):
    res = Result("BinaryElementwise")
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        shape1_str = str(row["param: shape_in1"])
        shape2_str = str(row["param: shape_in2"])
        shape1 = _parse_tuple(shape1_str)
        shape2 = _parse_tuple(shape2_str)
        if shape1 is None:
            res.record_skip("cannot parse shape_in1")
            continue

        nelems_in1 = _product(shape1)
        nelems_in2 = _product(shape2) if shape2 else 1

        out_shape = _broadcast_shape(shape1, shape2)
        nelems_out = _product(out_shape)

        dtype_str = str(row["param: dtype_in1_in2_out"])
        dtypes = _parse_tuple(dtype_str)
        if dtypes is None or len(dtypes) < 3:
            res.record_skip("cannot parse dtype_in1_in2_out")
            continue

        bpe_in1 = _bpe(str(dtypes[0]) if dtypes[0] is not None else None)
        bpe_in2 = _bpe(str(dtypes[1]) if dtypes[1] is not None else None)
        bpe_o = _bpe(str(dtypes[2]) if dtypes[2] is not None else None)

        ref_gflops = row["GFLOPS_first"]
        ref_data_mb = row.get("Data Moved (MB)_first", float("nan"))

        our_flops = nelems_out
        our_gflops = our_flops / 1e9

        ok_flops = _close(our_gflops, ref_gflops)

        ok_bytes = True
        bytes_detail = ""
        if (
            not pd.isna(ref_data_mb)
            and bpe_in1 is not None
            and bpe_in2 is not None
            and bpe_o is not None
        ):
            our_bytes = nelems_in1 * bpe_in1 + nelems_in2 * bpe_in2 + nelems_out * bpe_o
            our_data_mb = our_bytes / MiB
            ok_bytes = _close(our_data_mb, ref_data_mb)
            if not ok_bytes:
                bytes_detail = f"Data MB: ours={our_data_mb:.6f} ref={ref_data_mb:.6f}"

        desc = (
            f"Row {idx}: {row['name']} shape1={shape1} shape2={shape2} "
            f"dtypes={dtypes}"
        )

        if ok_flops and ok_bytes:
            res.record_pass()
        else:
            detail = []
            if not ok_flops:
                detail.append(f"GFLOPS: ours={our_gflops:.6f} ref={ref_gflops:.6f}")
            if bytes_detail:
                detail.append(bytes_detail)
            res.record_fail(desc, "; ".join(detail))

    return res


# -- Main -------------------------------------------------------------------


def print_results(results):
    print("\n" + "=" * 80)
    print("PERF FORMULA VERIFICATION SUMMARY")
    print("=" * 80)

    all_pass = True
    for r in results:
        status = "PASS" if r.failed == 0 else "FAIL"
        if r.failed > 0:
            all_pass = False
        print(
            f"\n  {r.category:25s}  {status}  "
            f"(passed={r.passed}, failed={r.failed}, "
            f"skipped={r.skipped}, total={r.total})"
        )
        for desc, detail in r.failures[:10]:
            print(f"    FAIL: {desc}")
            print(f"          {detail}")
        if len(r.failures) > 10:
            print(f"    ... and {len(r.failures) - 10} more failures")

    print("\n" + "=" * 80)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- see details above")
    print("=" * 80 + "\n")
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Verify perf formulas against TraceLens reference CSVs"
    )
    parser.add_argument(
        "--ref-dir", required=True, help="Path to TraceLens perf_report_csvs directory"
    )
    args = parser.parse_args()

    ref = args.ref_dir
    results = []

    gemm_csv = os.path.join(ref, "GEMM.csv")
    if os.path.exists(gemm_csv):
        results.append(verify_gemm(gemm_csv))
    else:
        print(f"WARNING: {gemm_csv} not found, skipping GEMM verification")

    sdpa_csv = os.path.join(ref, "SDPA_fwd.csv")
    if os.path.exists(sdpa_csv):
        results.append(verify_sdpa(sdpa_csv))
    else:
        print(f"WARNING: {sdpa_csv} not found, skipping SDPA verification")

    unary_csv = os.path.join(ref, "UnaryElementwise.csv")
    if os.path.exists(unary_csv):
        results.append(verify_unary_elementwise(unary_csv))
    else:
        print(
            f"WARNING: {unary_csv} not found, "
            "skipping Unary Elementwise verification"
        )

    binary_csv = os.path.join(ref, "BinaryElementwise.csv")
    if os.path.exists(binary_csv):
        results.append(verify_binary_elementwise(binary_csv))
    else:
        print(
            f"WARNING: {binary_csv} not found, "
            "skipping Binary Elementwise verification"
        )

    if not results:
        print("ERROR: No reference CSVs found. Nothing to verify.")
        sys.exit(1)

    all_pass = print_results(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
