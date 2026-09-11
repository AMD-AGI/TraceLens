###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for trace splitting.

Runs all three splitting modes in a single invocation per trace:
- ``--store-single-iteration``: first, middle, last iterations (split_traces/)
- ``--find-steady-state``: steady-state window extraction (steady_state_traces/)
- ``--divide-phases``: phase-divided extraction (phase_split_traces/)

Plus an annotation-stripped equivalence test on multi-iteration traces.

Run with --update-references to generate or refresh reference directories.
"""

import gzip, json, os, re, shutil, sys, pytest

from TraceLens.TraceUtils import split_inference_trace_annotation as split
from TraceLens.util import DataLoader

MULTI_ITER_TRACES_ROOT = "tests/traces/trace_splitter_traces"
SMALL_TRACES_ROOT = "tests/traces/trace_splitter_traces/small_traces"

_BOOKEND_NAMES = {"warmup", "wrapup"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_from_root(root):
    if not os.path.isdir(root):
        return []
    cases = []
    for entry in sorted(os.listdir(root)):
        dirpath = os.path.join(root, entry)
        if not os.path.isdir(dirpath):
            continue
        gz_files = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz_files:
            continue
        cases.append(pytest.param(dirpath, gz_files[0], id=entry))
    return cases


def _discover_multi_iter_cases():
    return _discover_from_root(MULTI_ITER_TRACES_ROOT)


def _discover_small_trace_cases():
    return _discover_from_root(SMALL_TRACES_ROOT)


def _list_gz(directory):
    """Return sorted list of .json.gz filenames in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".json.gz"))


def _list_gz_recursive(directory):
    """Return sorted list of .json.gz paths relative to *directory*."""
    if not os.path.isdir(directory):
        return []
    result = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".json.gz"):
                result.append(os.path.relpath(os.path.join(dirpath, f), directory))
    return sorted(result)


def _run_main(trace_path, out_dir, extra_args):
    old_argv = sys.argv
    sys.argv = [
        "split_inference_trace_annotation",
        trace_path,
        "-o",
        out_dir,
        *extra_args,
    ]
    try:
        split.main()
    finally:
        sys.argv = old_argv


def _find_roots(trace_path):
    """Load trace, find iteration roots. Returns (roots, N) or skips."""
    trace_data = DataLoader.load_data(trace_path)
    events = trace_data["traceEvents"]
    result = split.find_iteration_roots(events)
    if result.status.name == "NOT_SPLITTABLE":
        pytest.skip("trace is not splittable")
    roots = result.roots
    if not roots or len(roots) < 1:
        pytest.skip("no iteration roots found")
    return roots, len(roots)


def _compare_gz_files(gen_dir, ref_dir, context_label, recursive=False):
    """Compare all .json.gz files in ref_dir against generated ones in gen_dir."""
    ref_files = _list_gz_recursive(ref_dir) if recursive else _list_gz(ref_dir)
    assert ref_files, f"{ref_dir} contains no .json.gz references"

    for rel_path in ref_files:
        gen_path = os.path.join(gen_dir, rel_path)
        ref_path = os.path.join(ref_dir, rel_path)
        assert os.path.exists(gen_path), (
            f"{context_label}: generated file '{rel_path}' not found in output"
        )

        gen_trace = DataLoader.load_data(gen_path)
        ref_trace = DataLoader.load_data(ref_path)

        gen_events = gen_trace["traceEvents"]
        ref_events = ref_trace["traceEvents"]

        assert len(gen_events) == len(ref_events), (
            f"{context_label} '{rel_path}': event count mismatch — "
            f"generated {len(gen_events)}, reference {len(ref_events)}"
        )
        assert gen_events == ref_events, (
            f"{context_label} '{rel_path}': traceEvents content mismatch"
        )


def _update_ref_dir(gen_dir, ref_dir, recursive=False):
    """Copy .json.gz files from gen_dir to ref_dir, preserving subdirectory structure."""
    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.makedirs(ref_dir)
    rel_paths = _list_gz_recursive(gen_dir) if recursive else _list_gz(gen_dir)
    for rel_path in rel_paths:
        dest = os.path.join(ref_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(gen_dir, rel_path), dest)


def _find_gen_file_for_iter(out_dir, idx, total):
    """Find the generated .json.gz for iteration *idx* in the output directory."""
    for f in _list_gz(out_dir):
        if re.search(rf"_iteration_{idx}[_.]", f):
            return f
    if idx == 0:
        for f in _list_gz(out_dir):
            if "_warmup." in f:
                return f
    if idx == total - 1:
        for f in _list_gz(out_dir):
            if "_wrapup." in f:
                return f
    return None


def _compute_targets(roots, N):
    """Compute which iterations to extract: first/middle/last + warmup/wrapup."""
    non_bookend = [i for i, r in enumerate(roots) if r.get("name") not in _BOOKEND_NAMES]
    targets = {}
    if non_bookend:
        targets["first"] = non_bookend[0]
        targets["middle"] = non_bookend[len(non_bookend) // 2]
        targets["last"] = non_bookend[-1]
    if roots and roots[0].get("name") == "warmup":
        targets["warmup"] = 0
    if roots and roots[-1].get("name") == "wrapup":
        targets["wrapup"] = N - 1
    return targets


# ---------------------------------------------------------------------------
# Test 1: all three splitting modes in one invocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_multi_iter_cases())
def test_trace_split(dirpath, trace_gz, tmp_path, update_references):
    trace_path = os.path.join(dirpath, trace_gz)
    split_ref = os.path.join(dirpath, "split_traces")
    ss_ref = os.path.join(dirpath, "steady_state_traces")
    phase_ref = os.path.join(dirpath, "phase_split_traces")

    roots, N = _find_roots(trace_path)
    targets = _compute_targets(roots, N)
    is_llm = any(x in os.path.basename(dirpath) for x in ("sglang", "vllm"))

    # Run all three modes in a single main() call.
    out_dir = str(tmp_path / "output")
    os.makedirs(out_dir, exist_ok=True)
    flags = [
        "--store-single-iteration", "--iterations", "all",
        "--find-steady-state", "--divide-phases",
    ]
    if is_llm:
        flags.append("--llm-inference")
    _run_main(trace_path, out_dir, flags)

    # Categorise output: top-level iteration files = splits,
    # top-level non-iteration files = steady-state, subdirs = phases.
    all_top = _list_gz(out_dir)
    ss_gen_files = [f for f in all_top if "steady_state" in f]
    phase_gen_files = [
        f for f in _list_gz_recursive(out_dir)
        if os.sep in f or "/" in f
    ]

    if update_references:
        # split_traces: copy only the target iteration files
        if os.path.isdir(split_ref):
            shutil.rmtree(split_ref)
        os.makedirs(split_ref)
        for label, idx in targets.items():
            matching = _find_gen_file_for_iter(out_dir, idx, N)
            if matching:
                shutil.copy2(
                    os.path.join(out_dir, matching),
                    os.path.join(split_ref, matching),
                )
        metadata = {"total_iterations": N, "targets": targets}
        with open(os.path.join(split_ref, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # steady_state_traces
        if ss_gen_files:
            if os.path.isdir(ss_ref):
                shutil.rmtree(ss_ref)
            os.makedirs(ss_ref)
            for f in ss_gen_files:
                shutil.copy2(os.path.join(out_dir, f), os.path.join(ss_ref, f))

        # phase_split_traces
        if phase_gen_files:
            if os.path.isdir(phase_ref):
                shutil.rmtree(phase_ref)
            os.makedirs(phase_ref)
            for rel_path in phase_gen_files:
                dest = os.path.join(phase_ref, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(os.path.join(out_dir, rel_path), dest)

        pytest.skip(f"Updated references ({N} iterations)")
        return

    # --- Compare split_traces ---
    if os.path.isdir(split_ref):
        ref_gz = _list_gz(split_ref)
        for ref_name in ref_gz:
            gen_path = os.path.join(out_dir, ref_name)
            ref_path = os.path.join(split_ref, ref_name)
            assert os.path.exists(gen_path), (
                f"Generated file '{ref_name}' not found in output"
            )
            gen_trace = DataLoader.load_data(gen_path)
            ref_trace = DataLoader.load_data(ref_path)
            assert gen_trace["traceEvents"] == ref_trace["traceEvents"], (
                f"split '{ref_name}': traceEvents content mismatch"
            )

    # --- Compare steady_state_traces ---
    if os.path.isdir(ss_ref) and ss_gen_files:
        _compare_gz_files(out_dir, ss_ref, "--find-steady-state")

    # --- Compare phase_split_traces ---
    if os.path.isdir(phase_ref) and phase_gen_files:
        _compare_gz_files(out_dir, phase_ref, "--divide-phases", recursive=True)


# ---------------------------------------------------------------------------
# Test 2: annotation-stripped traces should produce the same splits
# ---------------------------------------------------------------------------


def _strip_annotations(events):
    """Remove user_annotation events from an event list."""
    return [e for e in events if e.get("cat") != "user_annotation"]


def _ref_file_for_iter(ref_dir, idx, total):
    """Find the reference .json.gz for iteration *idx*.

    Matches ``_iteration_{idx}`` in the filename, or ``_warmup`` / ``_wrapup``
    for the first / last index when those bookend files exist.
    """
    for f in _list_gz(ref_dir):
        if re.search(rf"_iteration_{idx}[_.]", f):
            return f
    if idx == 0:
        for f in _list_gz(ref_dir):
            if "_warmup." in f:
                return f
    if idx == total - 1:
        for f in _list_gz(ref_dir):
            if "_wrapup." in f:
                return f
    return None


def _collect_events(directory, recursive=False):
    """Load all .json.gz files and return the combined traceEvents list."""
    paths = _list_gz_recursive(directory) if recursive else _list_gz(directory)
    all_events = []
    for rel in paths:
        trace = DataLoader.load_data(os.path.join(directory, rel))
        all_events.extend(trace["traceEvents"])
    return all_events


_KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _kernel_events(events):
    """Return only GPU kernel events from an event list."""
    return [e for e in events if e.get("cat") in _KERNEL_CATS]


@pytest.mark.parametrize("dirpath,trace_gz", _discover_multi_iter_cases())
def test_trace_split_no_annotations(dirpath, trace_gz, tmp_path):
    """Stripping annotations and re-splitting should match the annotated references."""
    trace_path = os.path.join(dirpath, trace_gz)
    split_ref = os.path.join(dirpath, "split_traces")

    if not os.path.isdir(split_ref):
        pytest.skip("no split_traces/ reference directory")

    # Load trace and strip annotations
    trace_data = DataLoader.load_data(trace_path)
    original_events = trace_data["traceEvents"]
    annotations = [e for e in original_events if e.get("cat") == "user_annotation"]
    if not annotations:
        pytest.skip("trace has no annotations to strip")

    stripped_trace = {**trace_data, "traceEvents": _strip_annotations(original_events)}
    stripped_path = str(tmp_path / "stripped.json.gz")
    with gzip.open(stripped_path, "wt", encoding="utf-8") as f:
        json.dump(stripped_trace, f)

    is_llm = any(x in os.path.basename(dirpath) for x in ("sglang", "vllm"))
    extra = ["--llm-inference"] if is_llm else []

    # Verify the generic detector finds the same iteration count
    _, N = _find_roots(stripped_path)
    with open(os.path.join(split_ref, "metadata.json")) as f:
        meta = json.load(f)
    N_ref = meta["total_iterations"]
    assert N == N_ref, (
        f"Iteration count mismatch: stripped={N}, annotated={N_ref}"
    )

    targets = meta["targets"]

    # --- store-single-iteration: compare kernel counts ---
    total_gen_kernels = 0
    total_ref_kernels = 0
    for label, idx in targets.items():
        out = str(tmp_path / f"split_{label}")
        os.makedirs(out, exist_ok=True)
        _run_main(
            stripped_path, out,
            ["--store-single-iteration", "--iterations", str(idx)] + extra,
        )
        gen_gz = _list_gz(out)
        assert gen_gz, f"No output for stripped iteration {idx} ({label})"
        gen_events = DataLoader.load_data(os.path.join(out, gen_gz[0]))["traceEvents"]

        ref_file = _ref_file_for_iter(split_ref, idx, N_ref)
        assert ref_file, f"No reference file for iteration {idx}"
        ref_events = _strip_annotations(
            DataLoader.load_data(os.path.join(split_ref, ref_file))["traceEvents"]
        )
        total_gen_kernels += len(_kernel_events(gen_events))
        total_ref_kernels += len(_kernel_events(ref_events))

    tolerance = max(1, int(total_ref_kernels * 0.05))
    assert total_gen_kernels >= total_ref_kernels - tolerance, (
        f"store-single-iteration: stripped path has significantly fewer total "
        f"kernels ({total_gen_kernels}) than annotated ({total_ref_kernels}), "
        f"tolerance={tolerance}"
    )

    # --- find-steady-state + divide-phases: compare kernel counts ---
    ss_ref = os.path.join(dirpath, "steady_state_traces")
    dp_ref = os.path.join(dirpath, "phase_split_traces")
    run_ss = os.path.isdir(ss_ref) and not is_llm
    run_dp = os.path.isdir(dp_ref) and not is_llm

    if run_ss or run_dp:
        out = str(tmp_path / "ss_and_phase")
        os.makedirs(out, exist_ok=True)
        flags = []
        if run_ss:
            flags.append("--find-steady-state")
        if run_dp:
            flags.append("--divide-phases")
        _run_main(stripped_path, out, flags + extra)

        if run_ss:
            gen_kernels = _kernel_events(_collect_events(out))
            ref_kernels = _kernel_events(
                _strip_annotations(_collect_events(ss_ref))
            )
            assert len(gen_kernels) >= len(ref_kernels), (
                f"find-steady-state: stripped has fewer kernels ({len(gen_kernels)}) "
                f"than annotated ({len(ref_kernels)})"
            )

        if run_dp:
            gen_kernels = _kernel_events(_collect_events(out, recursive=True))
            ref_kernels = _kernel_events(
                _strip_annotations(_collect_events(dp_ref, recursive=True))
            )
            assert len(gen_kernels) >= len(ref_kernels), (
                f"divide-phases: stripped has fewer kernels ({len(gen_kernels)}) "
                f"than annotated ({len(ref_kernels)})"
            )


# ---------------------------------------------------------------------------
# Test 3: small traces must be NOT_SPLITTABLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_small_trace_cases())
def test_trace_not_splittable(dirpath, trace_gz):
    """Small/simple traces should be detected as NOT_SPLITTABLE."""
    trace_path = os.path.join(dirpath, trace_gz)
    trace_data = DataLoader.load_data(trace_path)
    events = trace_data["traceEvents"]
    result = split.find_iteration_roots(events)
    assert result.status.name == "NOT_SPLITTABLE", (
        f"Expected NOT_SPLITTABLE but got {result.status.name} "
        f"with {len(result.roots) if result.roots else 0} roots"
    )
