###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run validate_perf_model harnesses and collect rocprofv3 hardware counters.

---------------------------------------------------------------------------
PART 3 of 3 in the validate-perf-model workflow
---------------------------------------------------------------------------

This script is Step 3: it reads the CSV test cases produced by Step 1
(``generate_test_cases.py``), runs each harness under ``rocprofv3`` with
GPU cache flushed between runs, compares measured hardware counters to the
perf model's FLOPs/bytes predictions, and writes per-category summary CSVs
to the ``output/`` directory.

Usage
-----
  # Run all ops whose test cases exist under test_cases/
  python run_validation.py --output-dir output/

  # Run a single op
  python run_validation.py --op gemm_a8w8_blockscale --output-dir output/

  # Run inside Docker
  python run_validation.py --docker <image> --output-dir output/

  # Run on a remote host via SSH
  python run_validation.py --ssh user@host --output-dir output/

  # Override GPU arch (default: auto-detect)
  python run_validation.py --arch gfx950 --output-dir output/

  # Dry-run: print commands without executing
  python run_validation.py --dry-run --output-dir output/

Execution model
---------------
For each test case row, the script calls ``validate_perf_model.py --op ...``
with the appropriate dimension kwargs.  The subprocess can be wrapped in a
Docker ``docker exec`` command or an SSH call by passing ``--docker`` /
``--ssh``.  ``validate_perf_model.py`` flushes the GPU cache between its
rocprofv3 sub-runs to clear L2 cache state.

Output
------
``output/<timestamp>_<op>.csv``  -- per-op result rows (base + extra columns)
``output/<timestamp>_summary.csv`` -- merged summary across all ops run
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class SubprocessRunner:
    """Runs shell commands with optional Docker or SSH wrapping.

    Parameters
    ----------
    docker_image : str or None
        If set, prefix every command with ``docker exec <image>``.
    ssh_target : str or None
        If set, wrap every command with ``ssh <target> '<cmd>'``.
    dry_run : bool
        If True, print commands instead of executing them.
    timeout : int
        Per-command timeout in seconds.
    """

    def __init__(self, docker_image=None, ssh_target=None, dry_run=False, timeout=600, docker_workdir=None):
        self.docker_image = docker_image
        self.ssh_target = ssh_target
        self.dry_run = dry_run
        self.timeout = timeout
        self.docker_workdir = docker_workdir

    def _wrap(self, cmd):
        if self.docker_image:
            prefix = ["docker", "exec"]
            if self.docker_workdir:
                prefix += ["-w", self.docker_workdir]
            return prefix + [self.docker_image] + cmd
        if self.ssh_target:
            return ["ssh", self.ssh_target, " ".join(f"'{c}'" if " " in c else c for c in cmd)]
        return cmd

    def run(self, cmd, check=True, capture_output=False):
        wrapped = self._wrap(cmd)
        if self.dry_run:
            print(f"[dry-run] {' '.join(wrapped)}")
            return subprocess.CompletedProcess(wrapped, 0, stdout="", stderr="")
        return subprocess.run(
            wrapped, check=check, capture_output=capture_output, text=True, timeout=self.timeout
        )

    def check_output(self, cmd):
        wrapped = self._wrap(cmd)
        if self.dry_run:
            print(f"[dry-run] {' '.join(wrapped)}")
            return ""
        return subprocess.check_output(wrapped, stderr=subprocess.DEVNULL, timeout=30, text=True)


def detect_gpu_arch(runner):
    """Detect GPU arch from ``rocm-smi`` via the runner (supports docker/ssh)."""
    for cmd in (["rocm-smi", "--showproductname"], ["rocminfo"]):
        try:
            out = runner.check_output(cmd)
        except Exception:
            continue
        for arch in ("gfx950", "gfx942", "gfx941", "gfx940"):
            if arch in out:
                print(f"[arch] Detected: {arch}")
                return arch
    print("[arch] WARNING: could not detect GPU arch; defaulting to gfx942", file=sys.stderr)
    return "gfx942"


def _find_test_case_csvs(test_cases_dir, op_filter=None, category_filter=None):
    """Return sorted list of test case CSV paths matching the filters."""
    result = []
    for csv_path in sorted(test_cases_dir.glob("*.csv")):
        op = csv_path.stem
        if op_filter and op not in op_filter:
            continue
        result.append(csv_path)
    return result


def _read_test_case_csv(csv_path):
    """Read a test case CSV and return a list of row dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


_INT_ARGS = {
    "E", "K", "M", "N", "topk", "block_k", "block_m", "block_n", "ctx_len",
    "seq_len", "split_k", "head_dim", "group_size", "num_heads_q", "num_heads_kv",
    "num_decode_seqs", "prefill_seq_len", "varlen_num_seqs",
}
_STR_ARGS = {
    "w_dtype", "in_dtype", "kv_dtype", "out_dtype", "activation", "annotation",
    "bias_dtype", "quant_type", "quant_dtype", "scale_dtype", "varlen_scenario",
}


def _row_to_argv(op, row, validate_script, arch, output_dir, rocprofv3_path, timeout, extra_flags):
    """Convert a test-case CSV row into a validate_perf_model.py argv list."""
    cmd = [
        sys.executable,
        str(validate_script),
        "--op", row.get("op") or op,
        "--arch", arch,
        "--output-dir", str(output_dir),
        "--rocprofv3-path", rocprofv3_path,
        "--timeout", str(timeout),
    ]
    for k, v in row.items():
        if k in ("op", "category", "source", "trace_name", "registry_key"):
            continue
        if k in _INT_ARGS:
            try:
                cmd += [f"--{k}", str(int(float(v)))]
            except (ValueError, TypeError):
                pass
        elif k in _STR_ARGS:
            cmd += [f"--{k}", str(v)]
    cmd += extra_flags
    return cmd


def _collect_result_csv(op_output_dir):
    """Read the validation_report_*.csv from an op output dir."""
    candidates = sorted(op_output_dir.glob("validation_report_*.csv"))
    if not candidates:
        return []
    rows = []
    with open(candidates[-1], newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def run_all(args):
    """Main entry point: iterate over test case CSVs and run validations."""
    repo_dir = Path(__file__).resolve().parent
    runner = SubprocessRunner(
        docker_image=args.docker,
        ssh_target=args.ssh,
        dry_run=args.dry_run,
        timeout=args.timeout + 120,
        docker_workdir=str(repo_dir) if args.docker else None,
    )
    arch = args.arch
    if not arch:
        arch = detect_gpu_arch(runner)
    print(f"[run_validation] GPU arch: {arch}")
    validate_script = repo_dir / "validate_perf_model.py"
    test_cases_dir = Path(args.test_cases_dir)
    base_output_dir = Path(args.output_dir).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    op_filter = set(o.strip() for o in args.op.split(",")) if args.op else None
    csv_paths = _find_test_case_csvs(test_cases_dir, op_filter, args.category)
    if not csv_paths:
        print(
            f"No test case CSVs found in {test_cases_dir} (op_filter={op_filter}, "
            f"category={args.category}). Run generate_test_cases.py first."
        )
        return
    print(f"[run_validation] {len(csv_paths)} op CSV(s) to process.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    extra_flags = args.extra or []
    for csv_path in csv_paths:
        op = csv_path.stem
        rows = _read_test_case_csv(csv_path)
        if not rows:
            print(f"  [skip] {op}: empty test case CSV")
            continue
        print("\n" + "=" * 70)
        print(f"  OP: {op}   ({len(rows)} test cases)")
        op_output_dir = base_output_dir / f"{timestamp}_{op}"
        op_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {op_output_dir}")
        for i, row in enumerate(rows):
            case_dir = op_output_dir / f"case_{i:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            argv = _row_to_argv(
                op, row, validate_script, arch, case_dir, args.rocprofv3_path, args.timeout, extra_flags
            )
            desc = ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("op", "category", "source"))
            print(f"\n  [{i + 1}/{len(rows)}] {desc}")
            print("  CMD: " + " ".join(argv))
            status = "OK"
            try:
                proc = runner.run(argv, check=False, capture_output=True)
                if proc.returncode != 0:
                    print(f"  FAILED: {op}: exit {proc.returncode}")
                    if proc.stderr:
                        print(proc.stderr[-2000:])
                    status = f"FAIL:exit{proc.returncode}"
                elif proc.stdout:
                    print(proc.stdout[-1000:])
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT: {op} exceeded {args.timeout}s")
                status = "TIMEOUT"
            result_rows = _collect_result_csv(case_dir)
            if result_rows:
                for r in result_rows:
                    r = dict(r)
                    r["op"] = op
                    r["status"] = status
                    all_results.append(r)
            else:
                all_results.append({"op": op, "status": status})

    if all_results:
        summary_path = base_output_dir / f"{timestamp}_summary.csv"
        fieldnames = []
        for r in all_results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_results:
                w.writerow(r)
        print(f"\n[run_validation] Wrote summary: {summary_path}")


def main():
    p = argparse.ArgumentParser(
        description="Run validate_perf_model harnesses from test_cases/ CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--test-cases-dir", default="test_cases",
                   help="Directory containing <op>.csv test case files (default: test_cases/).")
    p.add_argument("--output-dir", default="output",
                   help="Directory for per-op result CSVs and merged summary (default: output/).")
    p.add_argument("--op", default=None, help="Restrict to a single op name or comma-separated list.")
    p.add_argument("--category", default=None,
                   help="Restrict to a single category (e.g. GEMM, MoE, InferenceAttention).")
    p.add_argument("--arch", default=None, help="GPU architecture (default: auto-detect via rocm-smi).")
    p.add_argument("--rocprofv3-path", default="rocprofv3", help="Path to rocprofv3 binary (default: rocprofv3).")
    p.add_argument("--timeout", type=int, default=300, help="Per-rocprofv3-run timeout in seconds (default: 300).")
    wrap_group = p.add_mutually_exclusive_group()
    wrap_group.add_argument("--docker", metavar="IMAGE", default=None, help="Wrap commands in `docker exec <IMAGE>`.")
    wrap_group.add_argument("--ssh", metavar="USER@HOST", default=None, help="Wrap commands in `ssh <USER@HOST>`.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=None,
                   help="Extra flags appended to every validate_perf_model.py invocation.")
    args = p.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
