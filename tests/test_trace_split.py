###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for trace splitting.

Covers three splitting modes against two trace collections:
- ``--store-single-iteration``: first, middle, last iterations (split_traces/)
- ``--find-steady-state``: steady-state window extraction (steady_state_traces/)
- ``--divide-phases``: phase-divided extraction (phase_split_traces/)

Plus an annotation-stripped equivalence test on multi-iteration traces.

Run with --update-references to generate or refresh reference directories.
"""

import gzip, json, os, re, shutil, sys, pytest

from TraceLens.TraceUtils import split_inference_trace_annotation as split
from TraceLens.util import DataLoader

INFERENCE_TRACES_ROOT = "tests/traces/inference"
MULTI_ITER_TRACES_ROOT = "tests/traces/trace_splitter_traces"
SMALL_TRACES_ROOT = "tests/traces/trace_splitter_traces/small_traces"


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


def _discover_inference_cases():
    return _discover_from_root(INFERENCE_TRACES_ROOT)


def _discover_multi_iter_cases():
    return _discover_from_root(MULTI_ITER_TRACES_ROOT)


def _discover_small_trace_cases():
    return _discover_from_root(SMALL_TRACES_ROOT)


def _discover_all_cases():
    return _discover_from_root(INFERENCE_TRACES_ROOT) + _discover_from_root(
        MULTI_ITER_TRACES_ROOT
    )


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


# ---------------------------------------------------------------------------
# Test 1: --store-single-iteration (first / middle / last)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_all_cases())
def test_trace_split(dirpath, trace_gz, tmp_path, update_references):
    trace_path = os.path.join(dirpath, trace_gz)
    ref_dir = os.path.join(dirpath, "split_traces")

    _, N = _find_roots(trace_path)
    targets = {"first": 0, "middle": N // 2, "last": N - 1}

    generated_files = {}
    for label, idx in targets.items():
        out_dir = str(tmp_path / label)
        os.makedirs(out_dir, exist_ok=True)
        _run_main(trace_path, out_dir, ["--store-single-iteration", "--iterations", str(idx)])
        gz_files = _list_gz(out_dir)
        assert gz_files, f"No .json.gz output for iteration {idx} ({label})"
        generated_files[label] = os.path.join(out_dir, gz_files[0])

    if update_references:
        if os.path.isdir(ref_dir):
            shutil.rmtree(ref_dir)
        os.makedirs(ref_dir)
        for label, gen_path in generated_files.items():
            shutil.copy2(gen_path, os.path.join(ref_dir, os.path.basename(gen_path)))
        metadata = {"total_iterations": N, "targets": targets}
        with open(os.path.join(ref_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        pytest.skip(f"Updated split_traces/ references ({N} total iterations)")
        return

    if not os.path.isdir(ref_dir):
        pytest.skip("no split_traces/ reference directory")

    for label, gen_path in generated_files.items():
        gen_name = os.path.basename(gen_path)
        ref_path = os.path.join(ref_dir, gen_name)
        if not os.path.exists(ref_path):
            pytest.fail(
                f"Generated file '{gen_name}' ({label} iteration) has no matching "
                f"reference in split_traces/"
            )
        gen_trace = DataLoader.load_data(gen_path)
        ref_trace = DataLoader.load_data(ref_path)
        gen_events = gen_trace["traceEvents"]
        ref_events = ref_trace["traceEvents"]
        assert len(gen_events) == len(ref_events), (
            f"{label} iteration (idx {targets[label]}): event count mismatch — "
            f"generated {len(gen_events)}, reference {len(ref_events)}"
        )
        assert gen_events == ref_events, (
            f"{label} iteration (idx {targets[label]}): traceEvents content mismatch"
        )


# ---------------------------------------------------------------------------
# Test 2: --find-steady-state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_all_cases())
def test_trace_split_steady_state(dirpath, trace_gz, tmp_path, update_references):
    trace_path = os.path.join(dirpath, trace_gz)
    ref_dir = os.path.join(dirpath, "steady_state_traces")

    _find_roots(trace_path)

    out_dir = str(tmp_path / "steady_state")
    os.makedirs(out_dir, exist_ok=True)
    _run_main(trace_path, out_dir, ["--find-steady-state"])

    gen_files = _list_gz(out_dir)
    if not gen_files:
        pytest.skip("--find-steady-state produced no output for this trace")

    if update_references:
        _update_ref_dir(out_dir, ref_dir)
        pytest.skip(f"Updated steady_state_traces/ references ({len(gen_files)} files)")
        return

    if not os.path.isdir(ref_dir):
        pytest.skip("no steady_state_traces/ reference directory")

    _compare_gz_files(out_dir, ref_dir, "--find-steady-state")


# ---------------------------------------------------------------------------
# Test 3: --divide-phases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirpath,trace_gz", _discover_all_cases())
def test_trace_split_divide_phases(dirpath, trace_gz, tmp_path, update_references):
    trace_path = os.path.join(dirpath, trace_gz)
    ref_dir = os.path.join(dirpath, "phase_split_traces")

    _find_roots(trace_path)

    out_dir = str(tmp_path / "divide_phases")
    os.makedirs(out_dir, exist_ok=True)
    _run_main(trace_path, out_dir, ["--divide-phases"])

    gen_files = _list_gz_recursive(out_dir)
    if not gen_files:
        pytest.skip("--divide-phases produced no output for this trace")

    if update_references:
        _update_ref_dir(out_dir, ref_dir, recursive=True)
        pytest.skip(f"Updated phase_split_traces/ references ({len(gen_files)} files)")
        return

    if not os.path.isdir(ref_dir):
        pytest.skip("no phase_split_traces/ reference directory")

    _compare_gz_files(out_dir, ref_dir, "--divide-phases", recursive=True)


# ---------------------------------------------------------------------------
# Test 4: annotation-stripped traces should produce the same splits
#   (only on multi-iteration traces where generic detection is meaningful)
# ---------------------------------------------------------------------------


def _strip_annotations(events):
    """Remove user_annotation events from an event list."""
    return [e for e in events if e.get("cat") != "user_annotation"]


def _event_sort_key(e):
    return (e.get("ts", 0), e.get("dur", 0), e.get("name", ""), e.get("cat", ""))


def _ref_file_for_iter(ref_dir, idx):
    """Find the reference .json.gz whose name contains _iteration_{idx}."""
    for f in _list_gz(ref_dir):
        if re.search(rf"_iteration_{idx}[_.]", f):
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

    # --- store-single-iteration ---
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

        ref_file = _ref_file_for_iter(split_ref, idx)
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

    # --- find-steady-state ---
    # LLM overlapped schedulers (SGLang, vLLM) have an inherent offset
    # between annotation and branch-descent iteration boundaries, so the
    # steady-state window may differ when annotations are stripped.
    ss_ref = os.path.join(dirpath, "steady_state_traces")
    if os.path.isdir(ss_ref) and not is_llm:
        out = str(tmp_path / "steady_state")
        os.makedirs(out, exist_ok=True)
        _run_main(stripped_path, out, ["--find-steady-state"] + extra)
        gen_kernels = _kernel_events(_collect_events(out))
        ref_kernels = _kernel_events(
            _strip_annotations(_collect_events(ss_ref))
        )
        assert len(gen_kernels) >= len(ref_kernels), (
            f"find-steady-state: stripped has fewer kernels ({len(gen_kernels)}) "
            f"than annotated ({len(ref_kernels)})"
        )

    # --- divide-phases ---
    dp_ref = os.path.join(dirpath, "phase_split_traces")
    if os.path.isdir(dp_ref) and not is_llm:
        out = str(tmp_path / "divide_phases")
        os.makedirs(out, exist_ok=True)
        _run_main(stripped_path, out, ["--divide-phases"] + extra)
        gen_kernels = _kernel_events(_collect_events(out, recursive=True))
        ref_kernels = _kernel_events(
            _strip_annotations(_collect_events(dp_ref, recursive=True))
        )
        assert len(gen_kernels) >= len(ref_kernels), (
            f"divide-phases: stripped has fewer kernels ({len(gen_kernels)}) "
            f"than annotated ({len(ref_kernels)})"
        )


# ---------------------------------------------------------------------------
# Test 5: small traces must be NOT_SPLITTABLE
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
