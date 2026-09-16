###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""vLLM AsyncLLM frontend sheets on the inference performance report."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch,
)
from TraceLens.TraceUtils.vllm_async_llm import (
    SHEET_ENGINE_STEPS,
    SHEET_INGRESS,
    SHEET_OUTPUT_GROUPS,
    build_vllm_async_llm_report_dfs,
    prefilter_async_llm_events,
    python_display_name,
)
from tests.fixtures.reporting import _build_synthetic_trace, _mk_event

pytestmark = pytest.mark.filterwarnings(
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:Source column 'kernel_details__summarize_kernel_stats' not found.*:UserWarning",
    "ignore:Found .* events with failed performance metric computation.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
)


def _py(name, ts, dur, pid=1, tid=1, args=None):
    return _mk_event("python_function", name, ts, dur, pid, tid, args=args)


def _write_chrome(path: Path, events) -> str:
    path.write_text(json.dumps({"traceEvents": events}))
    return str(path)


def _async_llm_events():
    """Two arrivals, two engine output groups (1 then 2 finishes), idle noise."""
    return [
        _py(
            "vllm/entrypoints/openai/api_router.py(34): create_completion",
            ts=1000,
            dur=2,
        ),
        _py(
            "vllm/entrypoints/openai/serving.py(113): _create_completion",
            ts=1001,
            dur=1,
        ),
        _py("vllm/v1/engine/core_client.py(10): assign_request_id", ts=1010, dur=1),
        _py("vllm/v1/engine/core_client.py(1104): _send_input", ts=1020, dur=5),
        _py(
            "vllm/v1/engine/core_client.py(200): _send_input_message",
            ts=1021,
            dur=1,
        ),
        _py(
            "vllm/v1/engine/output_processor.py(80): process_outputs",
            ts=2200,
            dur=50,
            args={"Python id": 10},
        ),
        _py(
            "vllm/v1/engine/output_processor.py(200): _finish_request",
            ts=2210,
            dur=5,
            args={"Python id": 11, "Python parent id": 10},
        ),
        _py(
            "vllm/entrypoints/openai/api_router.py(34): create_completion",
            ts=3000,
            dur=2,
        ),
        _py("vllm/v1/engine/core_client.py(10): assign_request_id", ts=3010, dur=1),
        _py("vllm/v1/engine/core_client.py(1104): _send_input", ts=3020, dur=5),
        _py(
            "vllm/v1/engine/output_processor.py(80): process_outputs",
            ts=4100,
            dur=80,
            args={"Python id": 20},
        ),
        _py(
            "vllm/v1/engine/output_processor.py(200): _finish_request",
            ts=4110,
            dur=5,
            args={"Python id": 21, "Python parent id": 20},
        ),
        _py(
            "vllm/v1/engine/output_processor.py(200): _finish_request",
            ts=4120,
            dur=5,
            args={"Python id": 22, "Python parent id": 20},
        ),
    ]


def _worker_events():
    return [
        _mk_event(
            "user_annotation",
            "execute_context_0(0)_generation_0(0)",
            ts=1500,
            dur=10,
            pid=0,
            tid=7,
        ),
        _mk_event(
            "user_annotation",
            "execute_context_1(801)_generation_0(0)",
            ts=2000,
            dur=200,
            pid=0,
            tid=7,
        ),
        _mk_event(
            "user_annotation",
            "execute_context_1(10)_generation_1(1)",
            ts=4000,
            dur=100,
            pid=0,
            tid=7,
        ),
    ]


def test_python_display_name():
    assert (
        python_display_name("vllm/v1/engine/core_client.py(1104): _send_input")
        == "_send_input"
    )
    assert python_display_name("_finish_request") == "_finish_request"


def test_async_llm_three_grains(tmp_path):
    async_path = _write_chrome(tmp_path / "async.json", _async_llm_events())
    dfs = build_vllm_async_llm_report_dfs(async_path, worker_events=_worker_events())

    ingress = dfs[SHEET_INGRESS]
    assert list(ingress.columns) == [
        "event_ord",
        "t_http_ms",
        "t_assign_ms",
        "t_send_ms",
    ]
    assert len(ingress) == 2
    assert list(ingress["event_ord"]) == [1, 2]
    # t0 is first HTTP (1000 us). send 1020 -> 0.020 ms
    assert ingress.loc[0, "t_send_ms"] == pytest.approx(0.020)
    assert ingress.loc[1, "t_send_ms"] == pytest.approx(2.020)

    groups = dfs[SHEET_OUTPUT_GROUPS]
    assert len(groups) == 3
    assert list(groups["group"]) == [1, 2, 2]
    assert list(groups["n_finished"]) == [1, 2, 2]
    assert list(groups["finish_ord_in_group"]) == [1, 1, 2]

    steps = dfs[SHEET_ENGINE_STEPS]
    assert len(steps) == 2
    assert "0(0)_generation_0(0)" not in "".join(steps["annotation"].astype(str))
    assert steps.loc[0, "n_arrivals"] == 1
    assert steps.loc[1, "n_arrivals"] == 1
    assert steps.loc[0, "n_finished"] == 1
    assert steps.loc[1, "n_finished"] == 2


def test_finish_via_python_parent_id_when_not_nested(tmp_path):
    events = [
        _py(
            "vllm/entrypoints/openai/api_router.py(34): create_completion",
            ts=1000,
            dur=1,
        ),
        _py("vllm/v1/engine/core_client.py(10): assign_request_id", ts=1010, dur=1),
        _py("vllm/v1/engine/core_client.py(1104): _send_input", ts=1020, dur=1),
        _py(
            "vllm/v1/engine/output_processor.py(80): process_outputs",
            ts=2000,
            dur=5,
            args={"Python id": 7},
        ),
        # Outside the process_outputs duration so TraceToTree will not nest it.
        _py(
            "vllm/v1/engine/output_processor.py(200): _finish_request",
            ts=3000,
            dur=1,
            args={"Python parent id": 7},
        ),
    ]
    async_path = _write_chrome(tmp_path / "async.json", events)
    worker = [
        _mk_event(
            "user_annotation",
            "execute_context_1(8)_generation_0(0)",
            ts=1500,
            dur=100,
            pid=0,
            tid=7,
        )
    ]
    dfs = build_vllm_async_llm_report_dfs(async_path, worker_events=worker)
    groups = dfs[SHEET_OUTPUT_GROUPS]
    assert len(groups) == 1
    assert groups.loc[0, "n_finished"] == 1
    assert dfs[SHEET_ENGINE_STEPS].loc[0, "n_finished"] == 1


def test_prefilter_drops_unrelated_python_spans(tmp_path):
    noise = [
        _py(f"asyncio/base_events.py({i}): _run_once", ts=1500 + i, dur=1)
        for i in range(200)
    ]
    events = _async_llm_events() + noise
    kept = prefilter_async_llm_events(events)
    assert all(
        python_display_name(e.get("name") or "")
        in {
            "_send_input",
            "assign_request_id",
            "create_completion",
            "process_outputs",
            "_finish_request",
        }
        for e in kept
    )
    assert len(kept) == len(_async_llm_events()) - 2  # drops _create_completion + _send_input_message
    async_path = _write_chrome(tmp_path / "async.json", events)
    dfs = build_vllm_async_llm_report_dfs(async_path, worker_events=_worker_events())
    assert len(dfs[SHEET_INGRESS]) == 2
    assert len(dfs[SHEET_OUTPUT_GROUPS]) == 3
    assert len(dfs[SHEET_ENGINE_STEPS]) == 2


def test_keeps_decode_only_engine_step(tmp_path):
    events = [
        _py(
            "vllm/entrypoints/openai/api_router.py(34): create_completion",
            ts=1000,
            dur=1,
        ),
        _py("vllm/v1/engine/core_client.py(10): assign_request_id", ts=1010, dur=1),
        _py("vllm/v1/engine/core_client.py(1104): _send_input", ts=1020, dur=1),
        _py(
            "vllm/v1/engine/output_processor.py(80): process_outputs",
            ts=2500,
            dur=20,
        ),
        _py(
            "vllm/v1/engine/output_processor.py(200): _finish_request",
            ts=2510,
            dur=1,
        ),
    ]
    async_path = _write_chrome(tmp_path / "async.json", events)
    worker = [
        _mk_event(
            "user_annotation",
            "execute_context_0(0)_generation_8(8)",
            ts=2000,
            dur=100,
            pid=0,
            tid=7,
        )
    ]
    dfs = build_vllm_async_llm_report_dfs(async_path, worker_events=worker)
    assert len(dfs[SHEET_ENGINE_STEPS]) == 1
    assert "generation_8(8)" in dfs[SHEET_ENGINE_STEPS].loc[0, "annotation"]


def test_report_writes_async_sheets_to_csv_dir(tmp_path):
    worker_trace = _build_synthetic_trace([("aten::mm", "gemm_kernel", 100)])
    worker_trace["traceEvents"].extend(_worker_events())
    profile = tmp_path / "worker.json"
    profile.write_text(json.dumps(worker_trace))
    async_path = _write_chrome(tmp_path / "async.json", _async_llm_events())
    csv_dir = tmp_path / "csvs"
    result = generate_perf_report_pytorch(
        profile_json_path=str(profile),
        output_csvs_dir=str(csv_dir),
        output_xlsx_path=str(tmp_path / "report.xlsx"),
        collective_analysis=False,
        engine="vllm",
        async_llm_trace=async_path,
    )
    assert SHEET_INGRESS in result
    assert SHEET_OUTPUT_GROUPS in result
    assert SHEET_ENGINE_STEPS in result
    assert (csv_dir / f"{SHEET_INGRESS}.csv").exists()
    assert (csv_dir / f"{SHEET_OUTPUT_GROUPS}.csv").exists()
    assert (csv_dir / f"{SHEET_ENGINE_STEPS}.csv").exists()
    xls = pd.ExcelFile(tmp_path / "report.xlsx")
    assert SHEET_INGRESS in xls.sheet_names


def test_async_sheets_skipped_for_non_vllm_engine(tmp_path):
    worker_trace = _build_synthetic_trace([("aten::mm", "gemm_kernel", 100)])
    profile = tmp_path / "worker.json"
    profile.write_text(json.dumps(worker_trace))
    async_path = _write_chrome(tmp_path / "async.json", _async_llm_events())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = generate_perf_report_pytorch(
            profile_json_path=str(profile),
            output_csvs_dir=str(tmp_path / "csvs"),
            output_xlsx_path=None,
            collective_analysis=False,
            engine="sglang",
            async_llm_trace=async_path,
        )
    assert SHEET_INGRESS not in result
    assert any("async_llm_trace" in str(w.message) for w in caught)
