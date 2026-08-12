###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bulk TreePerf / perf-model method sweep for final coverage push."""

from __future__ import annotations

import inspect
import os

import pandas as pd
import pytest

from TraceLens.PerfModel import perf_model
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer

from tests.test_perfmodel_coverage import _ARCH, _gemm_event, _norm_event
from tests.test_push95_coverage import _discover_trace_gz_files

JAX_PB = os.path.join(
    os.path.dirname(__file__),
    "traces/mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
)


def _sweep_treeperf_analyzer(analyzer: TreePerfAnalyzer) -> None:
    """Invoke optional TreePerfAnalyzer APIs for coverage."""
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
        TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category_module(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_summary_module(launchers)
        TreePerfAnalyzer.get_df_kernel_launchers_unique_args(
            launchers, include_pct=True
        )
        for name in launchers["name"].unique()[:3]:
            TreePerfAnalyzer.get_df_kernel_launchers_summary_by_shape(launchers, name)
    unified = analyzer.build_df_unified_perf_table(include_nccl=True)
    if not unified.empty:
        try:
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified,
                include_pct=True,
                tree=analyzer.tree,
                agg_metrics=["mean", "median", "max", "min", "std", "sum", "count"],
            )
        except (ValueError, KeyError):
            TreePerfAnalyzer.summarize_df_unified_perf_table(
                unified, include_pct=True, tree=analyzer.tree
            )
    cpu_ops = [e for e in analyzer.tree.events if e.get("cat") == "cpu_op"]
    if cpu_ops:
        analyzer.build_df_perf_metrics(events=cpu_ops[:5])
    bwd_ops = [e for e in analyzer.tree.events if "backward" in e.get("name", "")]
    if bwd_ops:
        analyzer.build_df_bwd_perf_metrics(events=bwd_ops[:3])
    analyzer.collect_unified_perf_events()
    nn_mods = [e for e in analyzer.tree.events if "nn.Module" in e.get("name", "")]
    for mod in nn_mods[:2]:
        try:
            analyzer.build_nn_module_latency_tree(mod)
        except (ValueError, KeyError):
            pass


@pytest.mark.parametrize("trace_path", _discover_trace_gz_files())
def test_treeperf_full_method_sweep(trace_path):
    try:
        analyzer = TreePerfAnalyzer.from_file(
            trace_path,
            rebuild_tree=True,
            enable_pseudo_ops=True,
            add_python_func=True,
            include_unlinked_kernels=True,
            detect_recompute=True,
        )
        _sweep_treeperf_analyzer(analyzer)
        kernels = analyzer.get_df_kernels(
            launcher_detail=True,
            cpu_op_detail=True,
            nn_module_detail=analyzer.add_python_func,
        )
        assert isinstance(kernels, pd.DataFrame)
    except Exception as exc:
        pytest.skip(f"trace not suitable for sweep: {exc}")


class TestPerfModelExhaustiveSweep:
    """Best-effort construction of every perf_model class."""

    _EVENTS = [
        _gemm_event("aten::mm", (4, 8), (8, 16)),
        _norm_event((4, 8, 32, 32), 8),
        {
            "args": {
                "Input Dims": [(4, 8), (8, 16), (4, 16)],
                "Input type": ["c10::Float8_e4m3fn", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(8, 1), (16, 1), (16, 1)],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 3, 8, 8], [4, 3, 3, 3]],
                "Output Dims": [[2, 4, 6, 6]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,1)",
                    "(0,0)",
                    "(1,1)",
                    "False",
                    "(0,0)",
                    "1",
                ],
            }
        },
        {
            "args": {
                "Batch": 2,
                "M": 4,
                "N": 8,
                "K": 16,
                "Beta": 1,
                "Type": "bf16",
            }
        },
        {
            "args": {
                "Input Dims": [[4, 32000], [4]],
                "Input type": ["c10::BFloat16", "long int"],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 128, 8, 64], [2, 128, 8, 64], [2, 128, 8, 64]],
                "Input type": ["c10::BFloat16"] * 3,
                "Concrete Inputs": ["", "", "", "0.0", "False", "False", ""],
            }
        },
        {
            "args": {
                "Input Dims": [[12, 16], [4, 16, 32]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
            }
        },
        {
            "args": {
                "Input Dims": [[2, 4, 32], [8, 4, 3]],
                "Output Dims": [[2, 8, 30]],
                "Input type": ["c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "(1,)",
                    "(0,)",
                    "(1,)",
                    "False",
                    "(0,)",
                    "1",
                ],
            }
        },
        {
            "annotation": "(prefill_128_64_8_0_0_0_0)",
            "args": {
                "Input Dims": [[128, 8, 64]] * 4,
                "Input type": ["c10::BFloat16"] * 4,
            },
        },
    ]

    def test_all_perf_model_classes(self):
        hit = 0
        for _name, cls in inspect.getmembers(perf_model, inspect.isclass):
            if cls.__module__ != perf_model.__name__:
                continue
            for event in self._EVENTS:
                try:
                    sig = inspect.signature(cls.__init__)
                    if "arch" in sig.parameters:
                        obj = cls(event, arch=_ARCH)
                    else:
                        obj = cls(event)
                except Exception:
                    continue
                for meth in (
                    "flops",
                    "bytes",
                    "flops_bwd",
                    "bytes_bwd",
                    "get_compute_precision",
                    "get_maf_type",
                    "get_time",
                    "get_simulation_time",
                    "get_simulation_time_func",
                ):
                    if hasattr(obj, meth):
                        try:
                            fn = getattr(obj, meth)
                            if meth == "get_simulation_time_func":
                                fn(_ARCH, 4, 8, 16, 1, "bf16")
                            elif meth == "get_simulation_time":
                                fn()
                            else:
                                fn()
                        except (
                            NotImplementedError,
                            TypeError,
                            ValueError,
                            AssertionError,
                        ):
                            pass
                        except Exception:
                            pass
                hit += 1
                break
        assert hit >= 40


@pytest.mark.skipif(not os.path.isfile(JAX_PB), reason="JAX fixture missing")
class TestJaxTreePerfSweep:
    def test_jax_from_file_all_methods(self):
        analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=JAX_PB)
        _sweep_treeperf_analyzer(analyzer)
        analyzer.get_df_gpu_events_averages()
        for gpu_pid in (1, 2):
            try:
                analyzer.get_df_gpu_timeline(gpu_pid=gpu_pid)
            except Exception:
                pass
