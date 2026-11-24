###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from TraceLens.Agent.PerfModel.validate_perf_model import harnesses
from TraceLens.Agent.PerfModel.validate_perf_model.harnesses import (
    _CSV_ONLY_OPS,
    OP_REGISTRY,
    _trace_dtype_str,
    chunk_stats_annotation,
    model_source,
    register,
    run_perf_model_from_event,
)
from TraceLens.Agent.PerfModel.validate_perf_model.validator import (
    _enrich_results,
    _extract_dims_for_existing_harness,
    _read_csv_rows,
    format_number,
    main,
    parse_counter_csv,
    parse_kernel_runtime_ns,
)

# Warmup, L2 flush and marker, then a measured call that launches two kernels.
_DISPATCHES = [
    (1, "gemm_a"),
    (2, "gemm_b"),
    (3, "void at::native::FillFunctor<unsigned char>"),
    (4, "void at::native::bitwise_not_kernel_cuda"),
    (5, "gemm_a"),
    (6, "void at::native::copy_kernel"),
    (7, "gemm_b"),
]


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_counters_sum_over_the_measured_call(tmp_path):
    rows = [
        (d, k, name, d * 10)
        for d, k in _DISPATCHES
        for name in ("TOTAL_16_OPS", "WRITE_SIZE")
    ]
    path = _write(
        tmp_path / "pmc.csv",
        ["Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value"],
        rows,
    )
    assert parse_counter_csv(path) == {"TOTAL_16_OPS": 120.0, "WRITE_SIZE": 120.0}
    assert parse_counter_csv(path, "gemm_b") == {
        "TOTAL_16_OPS": 70.0,
        "WRITE_SIZE": 70.0,
    }


def test_runtime_sums_the_measured_call(tmp_path):
    path = _write(
        tmp_path / "trace.csv",
        ["Dispatch_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"],
        [(d, k, d * 100, d * 100 + d) for d, k in _DISPATCHES],
    )
    assert parse_kernel_runtime_ns(path) == (12, 6.0, 2)


def test_trace_without_marker_fails(tmp_path):
    path = _write(
        tmp_path / "pmc.csv",
        ["Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value"],
        [(1, "gemm_a", "TOTAL_16_OPS", 1)],
    )
    with pytest.raises(RuntimeError, match="marker"):
        parse_counter_csv(path)


def test_validator_imports_without_torch():
    code = (
        "import sys\n"
        "import TraceLens.Agent.PerfModel.validate_perf_model.validator\n"
        "assert 'torch' not in sys.modules, 'validator parent imported torch'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_csv_only_op_has_a_call_spec():
    pytest.importorskip("torch")
    from TraceLens.Agent.PerfModel.validate_perf_model.replay.generic import (
        OP_CALL_SPEC,
    )

    missing = [key for key, *_ in _CSV_ONLY_OPS if key not in OP_CALL_SPEC]
    assert not missing, f"no OP_CALL_SPEC entry in replay/generic.py for {missing}"


def _row(op, **fields):
    return {
        "op": op,
        "run_status": "OK",
        "flops_ratio": 1.0,
        "bytes_ratio": 1.0,
        "predicted_flops": 100.0,
        "predicted_bytes": 10.0,
        **fields,
    }


def test_precision_mismatch_blocks_only_mfma_ops():
    gemm, norm, absent = _enrich_results(
        [
            _row("gemm_a8w8_blockscale", precision="MISMATCH"),
            _row("rms_norm", precision="MISMATCH"),
            _row(
                "gemm_a8w8_blockscale",
                precision="MISMATCH",
                precision_reason="fp4 counter is not available on gfx942",
            ),
        ]
    )
    assert gemm["precision_check"] == "MISMATCH"
    assert gemm["match_status"] == "precision-mismatch"
    assert norm["match_status"] == "match"
    assert absent["match_status"] == "match"


def test_report_mismatch_fails_the_row():
    same, differs = _enrich_results(
        [
            _row("rms_norm", csv_reported_flops=100.0, csv_reported_bytes=10.0),
            _row("rms_norm", csv_reported_flops=150.0, csv_reported_bytes=10.0),
        ]
    )
    assert same["run_status"] == "OK"
    assert differs["run_status"].startswith("FAIL: prediction FLOPs")
    assert differs["match_status"] == ""


def test_kernel_match_compares_kernel_families():
    rows = _enrich_results(
        [
            _row(
                "rms_norm",
                trace_kernels=["_k_BLOCK_SIZE_M_32"],
                kernel_name="_k_BLOCK_SIZE_M_64; void other<int>(x)",
            ),
            _row("rms_norm", trace_kernels=["_k"], kernel_name="void other<int>(x)"),
            _row("rms_norm", kernel_name="_k"),
        ]
    )
    assert [r["kernel_match"] for r in rows] == ["match", "mismatch", ""]


def test_arch_is_required():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "TraceLens.Agent.PerfModel.validate_perf_model.validator",
            "--op",
            "rms_norm",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "--arch is required" in proc.stderr


_GDN_OPS = {
    "vllm_gdn_attention_core": "vllm::gdn_attention_core",
    "atom_flydsl_gdr_decode": "aiter::linear_attention_with_output_base",
}


@pytest.mark.parametrize(
    "key",
    [k for k, e in OP_REGISTRY.items() if e["model"] is not None and k not in _GDN_OPS],
)
def test_default_event_scores_with_its_perf_model(key):
    entry = OP_REGISTRY[key]
    flops, nbytes, _ = entry["model_fn"](argparse.Namespace(**entry["defaults"]))
    assert flops > 0 and nbytes > 0


@pytest.mark.parametrize("key, trace_name", _GDN_OPS.items())
def test_gdn_event_lays_out_one_row_per_token(key, trace_name):
    args = argparse.Namespace(seq_len=16, num_heads_kv=4, head_dim=8)
    event = OP_REGISTRY[key]["build_event"](args)
    assert event["name"] == trace_name
    assert event["args"]["Input Dims"] == [[16, 64], [16, 4], [16, 4], [16, 4, 8], []]


@pytest.mark.parametrize(
    "extra, annotation, dims",
    [
        (
            {
                "annotation": "execute_0_context_1(sq4sk4sqsq16sqsk16)_generation_0(sq0sk0sqsq0sqsk0)"
            },
            "execute_0_context_1(sq4sk4sqsq16sqsk16)_generation_0(sq0sk0sqsq0sqsk0)",
            [[4, 8, 64], [4, 2, 64]],
        ),
        (
            {
                "annotation": "unparsable",
                "prefill_seq_len": 5,
                "num_decode_seqs": 3,
                "ctx_len": 10,
            },
            "execute_0_context_1(sq5sk5sqsq25sqsk25)_generation_3(sq3sk30sqsq3sqsk30)",
            [[8, 8, 64], [35, 2, 64]],
        ),
    ],
)
def test_unified_attention_sizes_tensors_from_the_annotation(extra, annotation, dims):
    args = argparse.Namespace(head_dim=64, num_heads_kv=2, num_heads_q=8, **extra)
    event = OP_REGISTRY["unified_attention"]["build_event"](args)
    assert event["annotation"] == annotation
    assert event["args"]["Input Dims"] == dims


def test_register_rejects_duplicate_keys_and_trace_names(monkeypatch):
    monkeypatch.setattr(harnesses, "OP_REGISTRY", {})
    monkeypatch.setattr(harnesses, "TRACE_NAME_TO_OP", {})
    register("op_a", category="c", description="d", trace_names=("trace_a",))(None)
    with pytest.raises(ValueError, match="registered twice"):
        register("op_a", category="c", description="d")
    with pytest.raises(ValueError, match="maps to both 'op_a' and 'op_b'"):
        register("op_b", category="c", description="d", trace_names=("trace_a",))(None)


def test_model_source_and_cases_only_ops(monkeypatch):
    cls_name, path, line = model_source("rms_norm")
    assert cls_name == "aiter_rms_norm"
    assert path == "TraceLens/PerfModel/extensions/rmsnorm_perf_model_extensions.py"
    assert line > 0
    assert model_source("moe_sorting_fwd") == ("moe_sorting_fwd", "unknown", 0)
    monkeypatch.setitem(
        OP_REGISTRY, "op_x", {"model": ("TraceLens.PerfModel.perf_model", "no_class")}
    )
    assert model_source("op_x") == ("no_class", "TraceLens/PerfModel/perf_model.py", 0)
    with pytest.raises(NotImplementedError, match="--cases"):
        OP_REGISTRY["moe_sorting_fwd"]["model_fn"](argparse.Namespace())


@pytest.mark.parametrize(
    "trace_name, dims, types, strides, concrete, attn_params, expected",
    [
        (
            "aiter::rms_norm",
            ((16, 64), (64,), (), ()),
            ["c10::BFloat16", "c10::BFloat16", "Scalar", "Scalar"],
            ((64, 1), (1,), (), ()),
            ["", "", "", ""],
            None,
            (4176.0, 4224, "bf16"),
        ),
        (
            "aiter::mha_varlen_fwd",
            [[16, 4, 64], [16, 2, 64]],
            ["c10::BFloat16", "c10::BFloat16"],
            None,
            None,
            {"c_sq": 16, "c_sk": 16, "c_sqsq": 256, "c_sqsk": 256},
            (131072, 24576, "bf16"),
        ),
    ],
)
def test_perf_model_runs_from_report_columns(
    trace_name, dims, types, strides, concrete, attn_params, expected
):
    result = run_perf_model_from_event(
        trace_name, dims, types, strides, concrete, attn_params
    )
    assert result == expected


def test_unknown_trace_name_has_no_perf_model():
    with pytest.raises(ValueError, match="No perf model class"):
        run_perf_model_from_event("unknown::op", [], [], None, None)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"c_sq": 16, "c_sk": 16, "c_sqsq": 128, "c_sqsk": 128, "g_sq": "nan"},
            "execute_0_context_2(sq16sk16sqsq128sqsk128)_generation_0(sq0sk0sqsq0sqsk0)",
        ),
        (
            {"g_sq": 4, "g_sk": 8, "c_sq": None},
            "execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq4sk8sqsq0sqsk0)",
        ),
    ],
)
def test_chunk_stats_annotation_derives_request_counts(params, expected):
    assert chunk_stats_annotation(params) == expected


def test_trace_dtype_str_resolves_short_and_traced_names():
    assert _trace_dtype_str(None, "dflt") == "dflt"
    assert _trace_dtype_str("BF16", "dflt") == "c10::BFloat16"
    assert _trace_dtype_str("c10::Half", "dflt") == "c10::Half"
    with pytest.raises(ValueError, match="Unknown dtype string"):
        _trace_dtype_str("bogus", "dflt")


@pytest.fixture
def rocprof(monkeypatch):
    """Replace the rocprofv3 subprocess with one that writes small fake CSVs.

    ``modes`` maps a pass name (``-o``) to how it fails: ``timeout``, ``exit``,
    ``no_csv`` or ``marker_only``. ``counters`` are the raw values reported for
    every dispatch.
    """
    fake = argparse.Namespace(
        modes={},
        counters={
            "SQ_INSTS_VALU_MFMA_MOPS_F8": 128,
            "TOTAL_16_OPS": 4,
            "TCC_EA0_RDREQ_sum": 16,
            "WRITE_SIZE": 2,
        },
        cmds=[],
    )

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.returncode = 0
            fake.cmds.append(cmd)

        def kill(self):
            pass

        def communicate(self, timeout=None):
            mode = fake.modes.get(self.cmd[self.cmd.index("-o") + 1])
            if mode == "timeout":
                if timeout is not None:
                    raise subprocess.TimeoutExpired(self.cmd, timeout)
                return "", ""
            if mode == "exit":
                self.returncode = 1
                return "", "rocprofv3 error"
            if mode != "no_csv":
                self._write_csv(marker_only=mode == "marker_only")
            return "rocprofv3 output", ""

        def _write_csv(self, marker_only):
            result_dir = Path(self.cmd[self.cmd.index("-d") + 1])
            if "--kernel-trace" in self.cmd:
                header = [
                    "Dispatch_Id",
                    "Kernel_Name",
                    "Start_Timestamp",
                    "End_Timestamp",
                ]
                rows = [(d, k, d * 100, d * 100 + d) for d, k in _DISPATCHES]
                # rocprofv3 names the file after the process id.
                path = result_dir / "host" / "1_kernel_trace.csv"
            else:
                names = self.cmd[
                    self.cmd.index("--pmc") + 1 : self.cmd.index("--output-format")
                ]
                header = ["Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value"]
                rows = [
                    (d, k, c, fake.counters.get(c, 0))
                    for d, k in _DISPATCHES
                    for c in names
                ]
                path = (
                    result_dir
                    / f"{self.cmd[self.cmd.index('-o') + 1]}_counter_collection.csv"
                )
            if marker_only:
                rows = [r for r in rows if "bitwise_not" in r[1]]
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, header, rows)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return fake


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["validator", *argv])
    return main()


def test_single_op_validation_reports_counters_against_the_prediction(
    rocprof, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    assert (
        _run_main(
            monkeypatch,
            *"--op gemm_a8w8_blockscale --M 16 --N 32 --K 64 --arch gfx942".split(),
            "--kernel-filter",
            "gemm_a",
            "--rocprofv3-path",
            str(tmp_path / "rocprofv3"),
            "--output-dir",
            str(out),
        )
        is None
    )
    (row,) = _read_csv_rows(out / "validation_report.csv")
    assert {
        k: row[k]
        for k in (
            "run_status",
            "flops_ratio",
            "bytes_ratio",
            "precision_check",
            "match_status",
            "kernel_name",
            "kernel_runtime_us",
            "rocprof_f8_ops",
            "rocprof_total_16",
            "rocprof_write_bytes",
        )
    } == {
        "run_status": "OK",
        "flops_ratio": "1.0000",
        "bytes_ratio": "1.0000",
        "precision_check": "PASS",
        "match_status": "match",
        "kernel_name": "gemm_a",
        "kernel_runtime_us": "0.005",
        "rocprof_f8_ops": "65536",
        "rocprof_total_16": "4",
        "rocprof_write_bytes": "2048",
    }
    assert (
        "PASS - dominant MFMA type matches"
        in (out / "validation_report.txt").read_text()
    )
    runner = rocprof.cmds[0][rocprof.cmds[0].index("--") + 3 :]
    assert (
        runner
        == "--op gemm_a8w8_blockscale --num-warmup 3 --M 16 --N 32 --K 64".split()
    )


@pytest.mark.parametrize(
    "pass_name, mode, status",
    [
        ("kernel_discovery", "timeout", "FAIL: rocprofv3 timed out after 300s"),
        ("kernel_discovery", "exit", "FAIL: rocprofv3 failed with exit code 1"),
        ("kernel_discovery", "no_csv", "FAIL: No *kernel_trace.csv in"),
        ("kernel_discovery", "marker_only", "FAIL: no kernel after the marker"),
        ("ops_derived_16", "exit", "OK"),
    ],
)
def test_rocprofv3_failures_fail_the_row_unless_the_pass_is_optional(
    rocprof, tmp_path, monkeypatch, pass_name, mode, status
):
    rocprof.modes[pass_name] = mode
    out = tmp_path / "out"
    _run_main(
        monkeypatch,
        *"--op rms_norm --M 16 --N 64 --arch gfx942".split(),
        "--output-dir",
        str(out),
    )
    (row,) = _read_csv_rows(out / "validation_report.csv")
    assert row["run_status"].startswith(status)
    if status == "OK":
        assert row["rocprof_total_16"] == "0"


@pytest.mark.parametrize(
    "argv, counters, precision_check, report_line",
    [
        (
            "--op rms_norm --skip-derived",
            {},
            "NO DATA",
            "NO DATA - no ops counters collected",
        ),
        (
            "--op rms_norm",
            {"TOTAL_16_OPS": 4},
            "PASS",
            "PASS - dominant MFMA type matches predicted precision",
        ),
        (
            "--op gemm_afp4wfp4",
            {"SQ_INSTS_VALU_MFMA_MOPS_F8": 128},
            "MISMATCH",
            "Reason:               fp4 counter (SQ_INSTS_VALU_MFMA_MOPS_F6F4) is not available on gfx942",
        ),
        (
            "--op _flash_attn_forward",
            {},
            "N/A",
            "SKIPPED - perf model returned no compute precision",
        ),
    ],
)
def test_precision_check_compares_the_dominant_counter(
    rocprof, tmp_path, monkeypatch, argv, counters, precision_check, report_line
):
    rocprof.counters = counters
    out = tmp_path / "out"
    _run_main(
        monkeypatch,
        *argv.split(),
        *"--arch gfx942 --output-dir".split(),
        str(out),
    )
    (row,) = _read_csv_rows(out / "validation_report.csv")
    assert row["precision_check"] == precision_check
    assert report_line in (out / "validation_report.txt").read_text()


def test_category_batch_reports_each_op_and_marks_failures(
    rocprof, tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(OP_REGISTRY["rms_norm"], "perf_model_only", True)
    out = tmp_path / "out"
    _run_main(
        monkeypatch,
        *"--category norm --arch gfx942 --output-dir".split(),
        str(out),
    )
    statuses = {
        r["op"]: r["run_status"].split(":")[0]
        for r in _read_csv_rows(out / "validation_report.csv")
    }
    assert statuses == {
        "rms_norm": "MODEL_ONLY",
        "rmsnorm": "OK",
        "add_rmsnorm": "OK",
        "rmsnorm_dynamicquant": "OK",
        "fused_qk_rmsnorm": "FAIL",
    }
    assert "perf-model-only" in (out / "rms_norm" / "validation_report.txt").read_text()
    assert "3 passed, 1 model-only, 1 failed out of 5 ops" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, runner_tail",
    [
        (
            "--op mha_varlen_fwd --varlen_seed 7 --varlen_num_seqs 2 "
            "--varlen_scenario mixed_prefill_decode",
            "--varlen-seed 7 --varlen-num-seqs 2 --varlen-scenario mixed_prefill_decode",
        ),
        (
            "--op unified_attention --annotation "
            "execute_0_context_1(sq4sk4sqsq16sqsk16)_generation_0(sq0sk0sqsq0sqsk0)",
            "--annotation "
            "execute_0_context_1(sq4sk4sqsq16sqsk16)_generation_0(sq0sk0sqsq0sqsk0)",
        ),
    ],
)
def test_attention_options_reach_the_replay_runner(
    rocprof, tmp_path, monkeypatch, argv, runner_tail
):
    out = tmp_path / "out"
    _run_main(
        monkeypatch,
        *argv.split(),
        *"--arch gfx942 --output-dir".split(),
        str(out),
    )
    (row,) = _read_csv_rows(out / "validation_report.csv")
    assert row["run_status"] == "OK"
    assert " ".join(rocprof.cmds[0]).endswith(runner_tail)


@pytest.mark.parametrize(
    "only_ops, exit_code, ops",
    [
        (None, 1, ["rms_norm", "mha_varlen_fwd", "moe_sorting_fwd"]),
        ("aiter::rms_norm", 0, ["rms_norm"]),
        ("absent_op", None, None),
    ],
)
def test_cases_mode_validates_traced_cases(
    rocprof, tmp_path, monkeypatch, only_ops, exit_code, ops
):
    bf16 = "c10::BFloat16"
    common = {
        "input_strides": None,
        "dims_str": "",
        "csv_bytes": 1,
        "trace_kernels": [],
    }
    cases = [
        {
            **common,
            "trace_name": "aiter::rms_norm",
            "input_dims": [[16, 64], [64], [], []],
            "input_strides": [[64, 1], [1], [], []],
            "input_types": [bf16, bf16, "Scalar", "Scalar"],
            "concrete_inputs": ["", "", "", ""],
            "attn_params": None,
            "csv_flops": 4176.0,
            "csv_bytes": 4224,
            "trace_kernels": ["gemm_a"],
        },
        {
            **common,
            "trace_name": "aiter::mha_varlen_fwd",
            "input_dims": [[16, 4, 64], [16, 2, 64]],
            "input_types": [bf16, bf16],
            "concrete_inputs": None,
            "attn_params": {"c_sq": 16, "c_sk": 16, "c_sqsq": 256, "c_sqsk": 256},
            "csv_flops": 1.0,
        },
        {
            **common,
            "trace_name": "aiter::moe_sorting_fwd",
            "input_dims": [],
            "input_types": [],
            "concrete_inputs": [],
            "attn_params": None,
            "csv_flops": 1.0,
        },
        {**common, "trace_name": "unknown::op"},
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    out = tmp_path / "out"
    args = ["--cases", str(cases_path), "--arch", "gfx942", "--output-dir", str(out)]
    if only_ops:
        args += ["--only-ops", only_ops]
    assert _run_main(monkeypatch, *args) == exit_code
    if ops:
        rows = _read_csv_rows(out / "validation_report.csv")
        assert [r["op"] for r in rows] == ops
        assert rows[0]["run_status"] == "OK"
        assert rows[0]["kernel_match"] == "match"
        assert all(r["run_status"].startswith("FAIL") for r in rows[1:])


def test_discover_reports_covered_missing_and_skipped_ops(monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules,
        "TraceLens.PerfModel.extensions.pseudo_ops_perf_utils",
        argparse.Namespace(
            get_pseudo_op_mappings=lambda: {
                "aiter::rms_norm": dict,
                "vllm::unknown_op": list,
                "aiter::attn_bwd": set,
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "TraceLens.PerfModel.torch_op_mapping",
        argparse.Namespace(op_to_perf_model_class_map={"aiter::other_op": tuple}),
    )
    assert _run_main(monkeypatch, "--discover") is None
    out = capsys.readouterr().out
    assert "Covered (1):" in out and "Missing (2):" in out and "Skipped (1):" in out
    assert "[--]   vllm::unknown_op" in out and "[skip] aiter::attn_bwd" in out


@pytest.mark.parametrize(
    "key, dims, types, concrete, perf_params, expected",
    [
        (
            "dsv3_mla_decode_fwd",
            [],
            ["bf16", "c10::Float8_e4m3fn"],
            None,
            {"H_Q": 16, "g_sq": 4, "g_sk": 32, "c_sq": 8, "c_sqsq": 16, "N_Q": 12},
            {
                "n_ctx": 4,
                "ctx_qlen": 2,
                "E": 4,
                "seq_len": 8,
                "num_heads_q": 16,
                "head_dim": 576,
                "kv_dtype": "fp8",
            },
        ),
        (
            "dsv3_mla_decode_fwd",
            [],
            None,
            None,
            None,
            {
                "n_ctx": 0,
                "ctx_qlen": 0,
                "E": 64,
                "seq_len": 8677,
                "num_heads_q": 16,
                "head_dim": 576,
                "kv_dtype": "bf16",
            },
        ),
        (
            "dsv3_moe_flydsl_stage1",
            [],
            [],
            None,
            {
                "num_tokens": 8,
                "hidden_dim": 64,
                "inter_dim": 32,
                "num_experts": 4,
                "topk": 2,
            },
            {"M": 8, "K": 64, "N": 32, "E": 4, "topk": 2, "block_m": 32},
        ),
        (
            "gemm_a8w8_blockscale",
            [[16, 64], [32, 64]],
            [],
            None,
            None,
            {"M": 16, "K": 64, "N": 32},
        ),
        (
            "gemm_a8w8_blockscale",
            [],
            [],
            None,
            None,
            {"M": 2048, "K": 8192, "N": 4096},
        ),
        (
            "ck_moe_stage1",
            [],
            [],
            None,
            {
                "num_tokens": 8,
                "hidden_dim": 64,
                "inter_dim": 32,
                "num_experts": 4,
                "topk": 2,
            },
            {"M": 8, "K": 64, "N": 32, "E": 4, "topk": 2},
        ),
        (
            "ck_moe_stage1",
            [[8, 64], [4, 16, 32]],
            [],
            [3],
            None,
            {"M": 8, "K": 64, "E": 4, "N": 32, "topk": 3},
        ),
        (
            "ck_moe_stage1",
            [[8, 64], [4, 16, 32]],
            [],
            [0],
            None,
            {"M": 8, "K": 64, "E": 4, "N": 32, "topk": 2},
        ),
        (
            "ck_moe_stage1",
            [],
            [],
            None,
            None,
            {"M": 512, "K": 7168, "E": 8, "N": 2048, "topk": 2},
        ),
        ("silu_and_mul", [[16, 64]], [], None, None, {"M": 16, "N": 64}),
        ("silu_and_mul", [[16]], [], None, None, {"M": 16, "N": 7168}),
        (
            "rms_norm",
            [[16, 64]],
            [],
            None,
            None,
            {"M": 16, "N": 64, "group_size": 128},
        ),
        (
            "dynamic_per_token_scaled_quant",
            [[2, 8, 64]],
            [],
            None,
            None,
            {"M": 16, "N": 64, "group_size": 128},
        ),
        (
            "dynamic_per_token_scaled_quant",
            [[16, 64]],
            [],
            None,
            None,
            {"M": 16, "N": 64, "group_size": 128},
        ),
        (
            "dynamic_per_token_scaled_quant",
            [],
            [],
            None,
            None,
            {"M": 2048, "N": 7168, "group_size": 128},
        ),
        (
            "_flash_attn_forward",
            [],
            [],
            None,
            {
                "N_Q": 16,
                "H_Q": 4,
                "H_KV": 2,
                "d_h_qk": 64,
                "c_sq": 16,
                "c_sk": 16,
                "c_sqsq": 128,
                "c_sqsk": 128,
                "g_sq": "nan",
            },
            {
                "seq_len": 16,
                "num_heads_q": 4,
                "num_heads_kv": 2,
                "head_dim": 64,
                "_varlen_c_sq": 16,
                "_varlen_c_sqsq": 128,
                "annotation": "execute_0_context_2(sq16sk16sqsq128sqsk128)_generation_0(sq0sk0sqsq0sqsk0)",
            },
        ),
        (
            "_flash_attn_forward",
            [[2, 16, 4, 64], [2, 16, 2, 64]],
            [],
            None,
            None,
            {"seq_len": 16, "num_heads_q": 4, "head_dim": 64, "num_heads_kv": 2},
        ),
        (
            "_flash_attn_forward",
            [[16, 4, 64], [16, 2, 64]],
            [],
            None,
            None,
            {"seq_len": 16, "num_heads_q": 4, "head_dim": 64, "num_heads_kv": 2},
        ),
        (
            "_flash_attn_forward",
            [],
            [],
            None,
            None,
            {"seq_len": 2048, "num_heads_q": 32, "head_dim": 128, "num_heads_kv": 8},
        ),
        (
            "gemm_afp4wfp4",
            [],
            [],
            None,
            None,
            {"M": 822, "N": 2112, "K": 3584},
        ),
    ],
)
def test_traced_dims_map_onto_harness_args(
    key, dims, types, concrete, perf_params, expected
):
    ns = _extract_dims_for_existing_harness(
        key, dims, types, concrete, perf_params=perf_params
    )
    assert vars(ns) == expected


@pytest.mark.parametrize(
    "val, unit, expected",
    [
        (None, "ops", "N/A"),
        (2.5e9, "bytes", "2.50 GB"),
        (3e6, "bytes", "3.00 MB"),
        (4000, "bytes", "4.00 KB"),
        (5, "bytes", "5 B"),
        (2e12, "ops", "2.00 T"),
        (3e9, "ops", "3.00 G"),
        (4e6, "ops", "4.00 M"),
        (5000, "ops", "5.00 K"),
        (7, "ops", "7"),
    ],
)
def test_format_number_scales_to_the_nearest_unit(val, unit, expected):
    assert format_number(val, unit) == expected


@pytest.mark.parametrize(
    "fields, comment",
    [
        (
            {"flops_ratio": 1.0, "bytes_ratio": 1.1},
            "hint: FLOPs: exact match; Bytes: good match",
        ),
        (
            {"flops_ratio": 0.9, "bytes_ratio": 0.5},
            "hint: FLOPs 0.90x: minor under-count, likely rounding or tile residuals; "
            "Bytes 0.50x: model over-predicts bytes",
        ),
        (
            {"flops_ratio": 0.3, "bytes_ratio": 1.5},
            "hint: FLOPs 0.30x: kernel may use different MFMA dtype than predicted; "
            "Bytes 1.50x: L2 replay or scale tensor overhead",
        ),
        (
            {"flops_ratio": 0.6, "bytes_ratio": 3.0},
            "hint: FLOPs 0.60x: possible mixed-precision or partial tile usage; "
            "Bytes 3.0x: significant extra traffic (L2 replays, scales, workspace)",
        ),
        (
            {"flops_ratio": 1.3, "bytes_ratio": 1.1},
            "hint: FLOPs 1.30x: minor over-count from tile padding or epilogue; "
            "Bytes: good match",
        ),
        (
            {"flops_ratio": 3.0, "bytes_ratio": 1.1},
            "hint: FLOPs 3.0x: likely tile padding, SwiGLU fusion, or extra VALU ops; "
            "Bytes: good match",
        ),
        (
            {"flops_ratio": 1.8, "bytes_ratio": 1.1},
            "hint: FLOPs 1.80x: moderate over-count, check tile dimensions; "
            "Bytes: good match",
        ),
        (
            {"flops_ratio": 1.0, "bytes_ratio": 1.1, "precision": "NO DATA"},
            "hint: FLOPs: exact match; Bytes: good match; "
            "No ops counters collected for precision check",
        ),
    ],
)
def test_comment_describes_the_ratio_band(fields, comment):
    (row,) = _enrich_results([_row("rms_norm", **fields)])
    assert row["comment"] == comment
