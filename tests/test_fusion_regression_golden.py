###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Strict golden regression for the deterministic agentic pipeline on bundled traces.

Each case runs the chain the analysis orchestrator runs before any LLM step: the
perf report with the agent flag set, ``orchestrator_prepare``, the system and
compute category scripts named by the manifest, then ``priority_data.json``.
Every artifact under ``category_data/``, ``metadata/`` and ``priority_data.json``
must match ``tests/traces/fusion_golden/<case>.json`` exactly, floats included.
``kernel_fusion_metrics.json`` (the fusion contract) is stored in full and
``fusion_candidates.json`` as a per-candidate projection, so a failure names
what moved; every other artifact is pinned by a sha256 of its bytes after the
output dir and repo root are replaced by placeholders and numpy scalar reprs are
unwrapped (the only difference observed between numpy 1.x and 2.x environments).

Regenerate with ``--update-references`` only for an intended change, and list
each regenerated case and the reason in the PR description.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

from TraceLens.Agent.Analysis.utils.arch_utils import _collect_arch_jsons
from TraceLens.Agent.Analysis.utils.report_utils import generate_priority_data

# Constants
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "traces", "fusion_golden")
AGENTS_DIR = os.path.join(
    REPO_ROOT,
    "TraceLens",
    "Agent",
    "Analysis",
    "skills",
    "analysis-orchestrator",
    "agents",
)
_MI300 = os.path.join(REPO_ROOT, "tests", "traces", "mi300")
_H100 = os.path.join(REPO_ROOT, "tests", "traces", "h100")
_OWLV2 = "google_owlv2-large-patch14-ensemble__1016001.json.gz"
_QWEN = "Qwen_Qwen1.5-0.5B-Chat__1016005.json.gz"
_BERT = "gaunernst_bert-small-uncased__1016001.json.gz"
_NSFW = "Falconsai_nsfw_image_detection__1016002.json.gz"
_TIMESFORMER = "facebook_timesformer-base-finetuned-k400__1016002.json.gz"
# case -> (trace1, trace2 or None). trace1 is MI300X; trace2 is the H100 comparison trace.
CASES = {
    "owlv2_mi300": (os.path.join(_MI300, _OWLV2), None),
    "qwen1.5_mi300": (os.path.join(_MI300, _QWEN), None),
    "act_checkpoint_mi300": (
        os.path.join(_MI300, "resnet_act_checkpoint.json.gz"),
        None,
    ),
    "bert_small_mi300": (os.path.join(_MI300, _BERT), None),
    "nsfw_mi300": (os.path.join(_MI300, _NSFW), None),
    "timesformer_mi300": (os.path.join(_MI300, _TIMESFORMER), None),
    "cmp_timesformer": (
        os.path.join(_MI300, _TIMESFORMER),
        os.path.join(_H100, _TIMESFORMER),
    ),
    "cmp_bert_small": (os.path.join(_MI300, _BERT), os.path.join(_H100, _BERT)),
}
GATED_DIRS = ("category_data", "metadata")
FULL_ARTIFACT = "category_data/kernel_fusion_metrics.json"
CANDIDATES_ARTIFACT = "category_data/fusion_candidates.json"
CANDIDATE_FIELDS = (
    "module_name",
    "base_name",
    "source",
    "instance_count",
    "kernel_count",
    "eligible_kernel_count",
    "kernel_type_signature",
    "has_fused_kernel",
    "total_kernel_time_us",
)
SCRIPT_RE = re.compile(r"category_analyses/(\w+)\.py")
# The kernel_details columns store str() of a list of dicts holding numpy scalars, which
# numpy>=2 renders as np.float64(1.5) and numpy<2 as 1.5. Same value, so strip the wrapper.
NUMPY_REPR_RE = re.compile(rb"np\.(?:float|int|uint|bool_|str_)\w*\(([^()]*)\)")
PERF_REPORT_FLAGS = (
    "--enable_pseudo_ops",
    "--group_by_num_kernels",
    "--include_call_stack",
)
PIPELINE_TIMEOUT_S = 900


@pytest.mark.parametrize("case", sorted(CASES))
def test_fusion_pipeline_matches_golden(case, tmp_path, update_references):
    trace1, trace2 = CASES[case]
    out = str(tmp_path)
    _run_pipeline(trace1, trace2, out)
    actual = _snapshot(out)
    golden_path = os.path.join(GOLDEN_DIR, f"{case}.json")
    if update_references:
        with open(golden_path, "w") as f:
            json.dump(actual, f, indent=1, sort_keys=True)
            f.write("\n")
        return
    with open(golden_path) as f:
        golden = json.load(f)
    assert (
        actual["kernel_fusion_metrics"] == golden["kernel_fusion_metrics"]
    ), _leaf_diffs(golden["kernel_fusion_metrics"], actual["kernel_fusion_metrics"])
    assert actual["fusion_candidates"] == golden["fusion_candidates"], _leaf_diffs(
        golden["fusion_candidates"], actual["fusion_candidates"]
    )
    moved = sorted(
        a
        for a in set(golden["sha256"]) | set(actual["sha256"])
        if golden["sha256"].get(a) != actual["sha256"].get(a)
    )
    assert not moved, f"artifacts differ from golden: {moved}"


def _run_pipeline(trace1, trace2, out):
    """Run the orchestrator's deterministic steps into ``out``."""
    arch = _collect_arch_jsons()["MI300X"]
    scope = "comparative" if trace2 else "standalone"
    if trace2:
        _check_call(
            "TraceLens.Reporting.generate_perf_report_pytorch",
            [
                "--profile_json_path",
                trace2,
                "--gpu_arch_json_path",
                arch,
                *PERF_REPORT_FLAGS,
            ],
            ["--output_csvs_dir", os.path.join(out, "perf_report_trace2_csvs")],
        )
        csvs = ["--output_csvs_dir", os.path.join(out, "perf_report_trace1_csvs")]
        csvs += ["--comparison_json_path", trace2]
    else:
        csvs = ["--output_csvs_dir", os.path.join(out, "perf_report_csvs")]
    _check_call(
        "TraceLens.Reporting.generate_perf_report_pytorch",
        [
            "--profile_json_path",
            trace1,
            "--gpu_arch_json_path",
            arch,
            *PERF_REPORT_FLAGS,
        ],
        csvs,
    )
    prep = ["--trace-path", trace1, "--platform", "MI300X", "--output-dir", out]
    prep += ["--comparison-scope", scope]
    if trace2:
        prep += ["--platform2", "H100"]
    _check_call("TraceLens.Agent.Analysis.utils.orchestrator_prepare", prep)
    with open(os.path.join(out, "category_data", "category_manifest.json")) as f:
        categories = json.load(f)["categories"]
    # System scripts first: compute metrics read fusion_flagged from kernel_fusion_metrics.json.
    for tier in ("system", "compute_kernel"):
        for cat in (c for c in categories if c["tier"] == tier):
            module = _category_module(cat["skill"])
            _check_call(module, _category_args(module, out, scope, cat["name"]))
    generate_priority_data(out)


def _check_call(module, *arg_groups):
    args = [a for group in arg_groups for a in group]
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=PIPELINE_TIMEOUT_S,
    )
    assert (
        proc.returncode == 0
    ), f"{module} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


def _category_module(skill):
    """Resolve a manifest skill to its script the way the orchestrator does: via its agent file."""
    path = os.path.join(AGENTS_DIR, f"{skill}.md")
    if not os.path.isfile(path):
        path = os.path.join(AGENTS_DIR, "generic-op-analyzer.md")
    with open(path) as f:
        script = SCRIPT_RE.search(f.read()).group(1)
    return f"TraceLens.Agent.Analysis.category_analyses.{script}"


def _category_args(module, out, scope, category):
    # The category scripts spell their flags differently; pass only the ones each declares.
    with open(importlib.util.find_spec(module).origin) as f:
        src = f.read()
    args = ["--output-dir", out]
    for flag in ("--comparison_scope", "--comparison-scope"):
        if f'"{flag}"' in src:
            args += [flag, scope]
    if '"--category"' in src:
        args += ["--category", category]
    return args


def _snapshot(out):
    digests = {}
    contents = {}
    paths = [
        os.path.join(d, f)
        for d in GATED_DIRS
        for f in sorted(os.listdir(os.path.join(out, d)))
    ]
    for rel in paths + ["priority_data.json"]:
        with open(os.path.join(out, rel), "rb") as f:
            data = (
                f.read()
                .replace(out.encode(), b"<OUT>")
                .replace(REPO_ROOT.encode(), b"<REPO>")
            )
        data = NUMPY_REPR_RE.sub(rb"\1", data)
        digests[rel] = hashlib.sha256(data).hexdigest()
        contents[rel] = data
    candidates = json.loads(contents[CANDIDATES_ARTIFACT])
    return {
        "kernel_fusion_metrics": (
            json.loads(contents[FULL_ARTIFACT]) if FULL_ARTIFACT in contents else None
        ),
        "fusion_candidates": [
            {k: c[k] for k in CANDIDATE_FIELDS if k in c} for c in candidates
        ],
        "sha256": digests,
    }


def _leaf_diffs(expected, actual, path="", out=None, limit=20):
    """First ``limit`` differing leaf paths, for a readable assertion message."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in sorted(set(expected) | set(actual)):
            _leaf_diffs(expected.get(k), actual.get(k), f"{path}.{k}", out, limit)
    elif (
        isinstance(expected, list)
        and isinstance(actual, list)
        and len(expected) == len(actual)
    ):
        for i, (e, a) in enumerate(zip(expected, actual)):
            _leaf_diffs(e, a, f"{path}[{i}]", out, limit)
    elif expected != actual:
        out.append(
            f"{path}: {json.dumps(expected)[:120]} -> {json.dumps(actual)[:120]}"
        )
    return out
