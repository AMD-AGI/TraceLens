#!/usr/bin/env python3
"""Reorganize coverage-named tests into module-aligned test files."""

from __future__ import annotations

import ast
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

SKIP_SOURCES = {
    "test_push95_coverage.py",
    "test_coverage_final.py",
}

WHOLE_FILE_TARGETS = {
    "test_agent_coverage.py": "test_analysis_agent_utils.py",
    "test_reporting_coverage.py": "test_reporting_utils.py",
    "test_perfmodel_coverage.py": "test_perfmodel_extensions.py",
    "test_treeperf_coverage.py": "test_treeperf.py",
    "test_nccl_analyser_coverage.py": "test_nccl_analyser.py",
    "test_trace_to_tree_coverage.py": "test_trace2tree.py",
    "test_trace2tree_mla_coverage.py": "test_pseudo_ops_extension.py",
    "test_split_annotation_coverage.py": "test_split_inference_trace_annotation.py",
    "test_custom_collectives_coverage.py": "test_perfmodel_extensions.py",
    "test_inference_trace_coverage.py": "test_inference_perf_report.py",
}

CLASS_ROUTES = [
    (r"Nccl|nccl", "test_nccl_analyser.py"),
    (r"PseudoOp|MLA|TraceToTree|Trace2Tree", "test_pseudo_ops_extension.py"),
    (r"TraceDiff|Tracediff|tracediff", "test_tracediff.py"),
    (
        r"SplitAnnotation|split_inference|CaptureMerge|GraphCapture",
        "test_split_inference_trace_annotation.py",
    ),
    (r"CompareTraces|JaxLlama", "test_compare_traces_jax_llama.py"),
    (r"ComparePerf", "test_compare_perf_reports.py"),
    (r"Rocprof|rocprof", "test_rocprof_perf_report.py"),
    (r"Pftrace|pftrace|HIPActivity|HipActivity", "test_pftrace_hip_activity_report.py"),
    (r"Collective|MultiRank", "test_multi_rank_collective_report.py"),
    (r"Inference|GraphMode", "test_inference_perf_report.py"),
    (r"Genesis|genesis", "test_genesis.py"),
    (
        r"Category|Arch|Convolution|Elementwise|Gemm|Norm|Reduce|TritonAnalysis",
        "test_analysis_agent_category_utils.py",
    ),
    (
        r"Orchestrator|Agent|Validation|PlotUtils|ReportUtils|Classify|KernelFusion|Prepare|Fusion",
        "test_analysis_agent_utils.py",
    ),
    (
        r"PerfModel|Moe|Attention|RmsNorm|Triton|KernelName|Conv|Softmax|Sdpa|CustomCollective|Bulk",
        "test_perfmodel_extensions.py",
    ),
    (
        r"TreePerf|JaxTree|GPUEvent|Subtree|KernelLauncher|JaxAnalyses",
        "test_treeperf.py",
    ),
    (r"Reporting|PytorchReport|Pytorch|Reporting|Cli", "test_reporting_utils.py"),
    (r"Util|RocprofParser|TraceEvent", "test_util.py"),
    (r"JAX|Jax", "test_jax_perf_report.py"),
]

CATCH_ALL_CLASS = re.compile(r"TestCoveragePush95Phase|TestCoverage.*Phase\d+$")

METHOD_ROUTES = [
    (
        r"orchestrator|fusion_candidates|_StubTree|_StubAnalyzer|classify_kernels|validation_utils|plot_utils|report_utils|kernel_fusion|MarkerValidator|validate_findings|validate_report",
        "test_analysis_agent_utils.py",
    ),
    (
        r"perf_model|moe_ext|attn_ext|rms_ext|pext\.|TritonCompiled|kernel_name_parser|custom_collectives|InferenceAttention|moe_aiter|gemm_a",
        "test_perfmodel_extensions.py",
    ),
    (
        r"TreePerf|GPUEvent|JaxTree|JaxAnalyses|summarize_df|get_kernel_launchers|build_df_unified",
        "test_treeperf.py",
    ),
    (
        r"generate_inference|capture_merge|merge_capture|align_streams|verify_subtree|load_capture|find_closest_batch|execution_details",
        "test_inference_perf_report.py",
    ),
    (
        r"generate_perf_report_pytorch[^_]|generate_perf_report_genesis|compare_perf_reports|reporting_utils",
        "test_reporting_utils.py",
    ),
    (
        r"pftrace|HIPActivity|HIPEvent|build_hip_summary",
        "test_pftrace_hip_activity_report.py",
    ),
    (r"genesis|Genesis", "test_genesis.py"),
    (r"rocprof|Rocprof", "test_rocprof_perf_report.py"),
    (r"compare_traces_jax|jax_llama", "test_compare_traces_jax_llama.py"),
    (r"TraceDiff|trace_diff|tracediff", "test_tracediff.py"),
    (r"TraceToTree|trace_to_tree|pseudo_ops|JaxTraceToTree", "test_trace2tree.py"),
    (
        r"split_inference|annotation_utils|inference_iteration",
        "test_split_inference_trace_annotation.py",
    ),
    (
        r"analysis_utils|arch_utils|CategoryAnalysis|convolution_analysis|elementwise_analysis",
        "test_analysis_agent_category_utils.py",
    ),
    (r"NcclAnalyser|nccl_analyser", "test_nccl_analyser.py"),
    (r"RocprofParser|TraceEventUtils", "test_util.py"),
    (r"generate_perf_report_jax|JaxAnalyses", "test_jax_perf_report.py"),
    (r"collective_report|multi_rank", "test_multi_rank_collective_report.py"),
]

IMPORT_REPLACEMENTS = [
    (
        r"from tests\.test_treeperf_coverage import",
        "from tests.fixtures.treeperf import",
    ),
    (
        r"from tests\.test_reporting_coverage import",
        "from tests.fixtures.reporting import",
    ),
    (r"from tests\.test_agent_coverage import", "from tests.fixtures.agent import"),
    (
        r"from tests\.test_perfmodel_coverage import",
        "from tests.fixtures.perfmodel import",
    ),
    (
        r"from tests\.test_coverage_95_final import",
        "from tests.fixtures.reporting import",
    ),
    (r"from tests\.test_push95_coverage import", "from tests.fixtures.traces import"),
]

COPYRIGHT_RE = re.compile(r"^#+\s*\n(?:# .*\n)+#\s*\n\n", re.MULTILINE)
MODULE_DOCSTRING_RE = re.compile(r'^(\s*"""[\s\S]*?"""\s*\n)', re.MULTILINE)


def route_class(name: str) -> str | None:
    for pat, target in CLASS_ROUTES:
        if re.search(pat, name):
            return target
    return None


def route_method(source: str) -> str | None:
    for pat, target in METHOD_ROUTES:
        if re.search(pat, source, re.IGNORECASE):
            return target
    return None


def get_node_source(lines: list[str], node: ast.AST) -> str:
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def strip_header(text: str) -> str:
    text = COPYRIGHT_RE.sub("", text, count=1)
    text = MODULE_DOCSTRING_RE.sub("", text, count=1)
    return text.lstrip("\n")


def rewrite_imports(text: str) -> str:
    for old, new in IMPORT_REPLACEMENTS:
        text = re.sub(old, new, text)
    return text


def collect_test_names(block: str) -> list[str]:
    tree = ast.parse(block)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return names


def append_block(
    target_path: str, source_file: str, block: str, seen: set[str]
) -> bool:
    block = rewrite_imports(strip_header(block))
    names = collect_test_names(block)
    if not names:
        return False
    if any(name in seen for name in names):
        return False
    seen.update(names)

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n# --- migrated from {source_file} ---\n")
        f.write(block)
        if not block.endswith("\n"):
            f.write("\n")
    return True


def extract_module_imports(lines: list[str], tree: ast.Module) -> str:
    parts = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parts.append(get_node_source(lines, node))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            break
    return rewrite_imports("".join(parts))


def migrate_whole_file(source: str, target: str, seen: set[str]) -> None:
    path = os.path.join(TESTS_DIR, source)
    text = open(path, encoding="utf-8").read()
    if append_block(os.path.join(TESTS_DIR, target), source, text, seen):
        print(f"merged whole: {source} -> {target}")


def migrate_split_file(source: str, seen: set[str]) -> None:
    path = os.path.join(TESTS_DIR, source)
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    tree = ast.parse("".join(lines))
    imports_block = extract_module_imports(lines, tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if CATCH_ALL_CLASS.search(node.name):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith(
                        "test_"
                    ):
                        method_src = get_node_source(lines, item)
                        target = route_method(method_src) or route_method(
                            imports_block + method_src
                        )
                        if not target:
                            print(
                                f"WARN unrouted method {source}::{node.name}.{item.name}"
                            )
                            target = "test_reporting_utils.py"
                        dedented = []
                        for line in lines[item.lineno - 1 : item.end_lineno]:
                            dedented.append(
                                line[4:] if line.startswith("    ") else line
                            )
                        block = imports_block + "\n\n" + "".join(dedented)
                        label = f"{source}::{node.name}.{item.name}"
                        if append_block(
                            os.path.join(TESTS_DIR, target), label, block, seen
                        ):
                            print(f"  method -> {target}: {item.name}")
            else:
                target = route_class(node.name)
                if not target:
                    print(f"WARN unrouted class {source}::{node.name}")
                    target = "test_reporting_utils.py"
                block = imports_block + "\n\n" + get_node_source(lines, node)
                if append_block(os.path.join(TESTS_DIR, target), source, block, seen):
                    print(f"  class -> {target}: {node.name}")

        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            func_src = get_node_source(lines, node)
            target = route_method(func_src) or route_method(imports_block + func_src)
            if not target:
                print(f"WARN unrouted function {source}::{node.name}")
                target = "test_reporting_utils.py"
            block = imports_block + "\n\n" + func_src
            if append_block(os.path.join(TESTS_DIR, target), source, block, seen):
                print(f"  function -> {target}: {node.name}")


def create_fixtures() -> None:
    fixtures_dir = os.path.join(TESTS_DIR, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    mappings = {
        "treeperf.py": "test_treeperf_coverage.py",
        "reporting.py": "test_reporting_coverage.py",
        "agent.py": "test_agent_coverage.py",
        "perfmodel.py": "test_perfmodel_coverage.py",
    }

    for fixture_name, source_name in mappings.items():
        src_path = os.path.join(TESTS_DIR, source_name)
        src = open(src_path, encoding="utf-8").read()
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        helper_parts = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                break
            if isinstance(node, ast.ClassDef) and node.name.startswith("_"):
                helper_parts.append(get_node_source(lines, node))
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
                helper_parts.append(get_node_source(lines, node))
            elif isinstance(node, ast.Assign):
                helper_parts.append(get_node_source(lines, node))

        imports = extract_module_imports(lines, tree)
        header = (
            "###############################################################################\n"
            "# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.\n"
            "#\n"
            "# See LICENSE for license information.\n"
            "###############################################################################\n\n"
            f'"""Shared test helpers migrated from {source_name}."""\n\n'
        )
        open(os.path.join(fixtures_dir, fixture_name), "w", encoding="utf-8").write(
            header + imports + "\n" + "".join(helper_parts)
        )

    # treeperf extras that lived after Test class in source file
    treeperf_extra = """
def _mk_pytorch_trace():
    corr = 100
    return [
        _make_gpu_event(
            "cpu", 1000, 100, "cpu_op", "aten::mm", pid=100,
            args={"Input Dims": [[32, 64], [64, 128]], "Input type": ["fp16", "fp16"]},
        ),
        _make_gpu_event(
            "rt", 1010, 5, "cuda_runtime", "hipLaunchKernel", pid=100,
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "kern", 1050, 50, "kernel", "gemm_kernel", pid=0, tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1050, "s"),
        _mk_ac2g(corr, 0, 7, 1100, "f"),
    ]


def _sweep_treeperf_analyzer(analyzer):
    assert analyzer.tree is not None
    analyzer.check_gpu_only()
    timeline = analyzer.get_df_gpu_timeline(micro_idle_thresh_us=0)
    assert isinstance(timeline, pd.DataFrame)
    launchers = analyzer.get_df_kernel_launchers(
        include_args=True,
        include_kernel_details=True,
        include_call_stack=analyzer.add_python_func,
        id_cols=True,
    )
    assert isinstance(launchers, pd.DataFrame)
    if not launchers.empty:
        TreePerfAnalyzer.get_df_kernel_launchers_summary(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_unique_args(launchers, include_pct=True)
    unified = analyzer.build_df_unified_perf_table(include_nccl=True)
    if not unified.empty:
        try:
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified, include_pct=True, tree=analyzer.tree,
                agg_metrics=["mean", "median", "max", "min", "std", "sum", "count"],
            )
        except (ValueError, KeyError):
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified, include_pct=True, tree=analyzer.tree
            )
"""
    with open(os.path.join(fixtures_dir, "treeperf.py"), "a", encoding="utf-8") as f:
        f.write(treeperf_extra)

    final_path = os.path.join(TESTS_DIR, "test_coverage_95_final.py")
    if os.path.isfile(final_path):
        src = open(final_path, encoding="utf-8").read()
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        extra = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "_jax_llama_trace_events",
                "_write_gz_trace",
            }:
                extra.append(get_node_source(lines, node))
        if extra:
            with open(
                os.path.join(fixtures_dir, "reporting.py"), "a", encoding="utf-8"
            ) as f:
                f.write("\n\n# --- from test_coverage_95_final ---\n")
                f.write("".join(extra))

    traces_src = '''###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared trace paths and discovery helpers for TraceLens tests."""

from __future__ import annotations

import os

import pytest

_TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACES_ROOT = os.path.join(_TESTS_DIR, "traces")
INFERENCE_ROOT = os.path.join(TRACES_ROOT, "inference")
TESTS_DIR = _TESTS_DIR
ROCprof_FILE = os.path.join(_TESTS_DIR, "rocprof/908_results.json.gz")
NORM_TRACE = os.path.join(
    TRACES_ROOT, "perf_model/normalization/normalization_layer_test.json.gz"
)
RESNET_TRACE = os.path.join(TRACES_ROOT, "mi300/resnet_act_checkpoint.json.gz")
RESNET = RESNET_TRACE
TIMESFORMER1 = os.path.join(
    TRACES_ROOT, "mi300/facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)
TIMESFORMER2 = os.path.join(
    TRACES_ROOT, "h100/facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)
COMPARE_DIR = os.path.join(_TESTS_DIR, "traces/compare_test_ops")
JAX_PB = os.path.join(
    TRACES_ROOT,
    "mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
)


def _discover_trace_gz_files():
    cases = []
    for root, _dirs, files in os.walk(TRACES_ROOT):
        for name in sorted(files):
            if not name.endswith(".json.gz"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, TESTS_DIR)
            cases.append(pytest.param(path, id=rel.replace(os.sep, "/")))
    return cases


def _discover_inference_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        cases.append(pytest.param(dirpath, gz[0], id=entry))
    return cases
'''
    open(os.path.join(fixtures_dir, "traces.py"), "w", encoding="utf-8").write(
        traces_src
    )
    open(os.path.join(fixtures_dir, "__init__.py"), "w", encoding="utf-8").write(
        '"""Shared pytest helpers for TraceLens tests."""\n'
    )


TRACE_NAMES = {
    "TRACES_ROOT",
    "INFERENCE_ROOT",
    "TIMESFORMER1",
    "TIMESFORMER2",
    "NORM_TRACE",
    "ROCprof_FILE",
    "RESNET_TRACE",
    "RESNET",
    "JAX_PB",
    "COMPARE_DIR",
    "_discover_trace_gz_files",
    "_discover_inference_cases",
}


def inject_trace_imports() -> None:
    for fn in os.listdir(TESTS_DIR):
        if not fn.startswith("test_") or not fn.endswith(".py"):
            continue
        path = os.path.join(TESTS_DIR, fn)
        text = open(path, encoding="utf-8").read()
        parts = text.split("\n# --- migrated from ")
        if len(parts) <= 1:
            continue
        new_parts = [parts[0]]
        changed = False
        for chunk in parts[1:]:
            header_line, _, body = chunk.partition("\n")
            used = {
                n for n in TRACE_NAMES if re.search(r"\b" + re.escape(n) + r"\b", body)
            }
            if used and "from tests.fixtures.traces import" not in body:
                imp = (
                    "from tests.fixtures.traces import "
                    + ", ".join(sorted(used))
                    + "\n"
                )
                body = imp + body
                changed = True
            new_parts.append(header_line + "\n" + body)
        if changed:
            open(path, "w", encoding="utf-8").write(
                new_parts[0]
                + "".join("\n# --- migrated from " + c for c in new_parts[1:])
            )
            print(f"injected trace imports: {fn}")


def main() -> None:
    create_fixtures()

    split_sources = sorted(
        f
        for f in os.listdir(TESTS_DIR)
        if (
            f.startswith("test_coverage")
            or (f.endswith("_coverage.py") and f not in WHOLE_FILE_TARGETS)
        )
        and f not in SKIP_SOURCES
        and f not in WHOLE_FILE_TARGETS
    )

    seen: set[str] = set()
    for source, target in WHOLE_FILE_TARGETS.items():
        if os.path.isfile(os.path.join(TESTS_DIR, source)):
            migrate_whole_file(source, target, seen)

    for source in split_sources + ["test_reporting_cli_coverage.py"]:
        if os.path.isfile(os.path.join(TESTS_DIR, source)):
            print(f"split: {source}")
            migrate_split_file(source, seen)

    inject_trace_imports()

    to_delete = set(WHOLE_FILE_TARGETS) | set(SKIP_SOURCES) | set(split_sources)
    to_delete.add("test_reporting_cli_coverage.py")
    for fn in sorted(to_delete):
        path = os.path.join(TESTS_DIR, fn)
        if os.path.isfile(path):
            os.remove(path)
            print(f"deleted: {fn}")


if __name__ == "__main__":
    main()
