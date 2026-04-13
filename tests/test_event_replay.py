###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time
import pandas as pd
import argparse
import json
import subprocess

import pytest
import torch
try:
    import torchvision.models as torchvision_models
except ImportError:
    torchvision_models = None
from torch.profiler import profile, record_function, ProfilerActivity
import os
from TraceLens import EventReplayer, TreePerfAnalyzer, GPUEventAnalyser
from TraceLens.EventReplay.event_replay import (
    SkipReplayError,
    MIOPEN_FUSED_OP_MAP,
    NON_REPLAYABLE_NAMES,
)


def profile_resnet(path=None):
    if torchvision_models is None:
        pytest.skip("torchvision is required for ResNet profiling tests")
    device = "cuda"
    dtype = torch.bfloat16
    model = torchvision_models.resnet18().to(device=device, dtype=dtype)
    batch = 20
    C_IN, H_IN, W_IN = 3, 224, 224
    dummy_input = torch.randn(batch, C_IN, H_IN, W_IN).to(device=device, dtype=dtype)
    dummy_output = torch.randn(batch, 1000).to(device=device, dtype=dtype)
    if path is None:
        path = "/tmp/resnet_trace.json"

    def trace_handler(p):
        p.export_chrome_trace(path)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=10, warmup=5, active=3, repeat=1),
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as p:
        for idx in range(50):
            out = model(dummy_input)
            out.backward(dummy_output)
            p.step()
    return path


def benchmark_func(func, warmup, avg_steps):
    """
    Benchmark a function with warmup and average steps.
    Disclaimer: This method would be inaccurate for very short ops.
    TODO:
        (1) improve this by using a more precise timer
        (2) move to TraceLens.utils
    Args:
        func (callable): The function to benchmark.
        warmup (int): Number of warmup iterations.
        avg_steps (int): Number of iterations to average over.
    Returns:
        float: Average time taken per iteration in microseconds.
    """
    # Warmup phase
    for _ in range(warmup):
        func()

    # Benchmarking phase
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(avg_steps):
        func()
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_sec = elapsed_time / avg_steps
    avg_time_us = avg_time_sec * 1e6

    return avg_time_us


def get_unique_evts(perf_analyzer, event_name):
    def list_to_tuple(obj):
        if isinstance(obj, list):
            return tuple(list_to_tuple(item) for item in obj)
        return obj

    cfg_fields = ["Input Dims", "Input type", "Input Strides", "Concrete Inputs"]
    events = [e for e in perf_analyzer.tree.events if e.get("name") == event_name]
    rows = []
    for evt in events:
        gpu_events = [
            perf_analyzer.tree.get_UID2event(uid) for uid in evt["gpu_events"]
        ]
        gpu_time = GPUEventAnalyser(gpu_events).compute_metrics()["busy_time"]
        row = {
            "name": evt.get("name"),
            "UID": evt.get("UID"),
        }
        for arg in cfg_fields:
            row[arg] = list_to_tuple(evt["args"][arg])
        row["gpu_time"] = gpu_time
        rows.append(row)
    df = pd.DataFrame(rows)

    # summarize the dataframe across args\
    dict_agg = {"name": "first", "UID": "first", "gpu_time": ["median", "max", "min"]}
    df_summary = df.groupby(cfg_fields).agg(dict_agg)
    df_summary.columns = ["_".join(col).strip() for col in df_summary.columns.values]
    # now we return a list of dict where each element of list is "UID first event": event, "stats": stats
    list_unique_evts = []
    for idx, row in df_summary.iterrows():
        evt = perf_analyzer.tree.get_UID2event(row["UID_first"])
        stats = {
            "gpu_time_median": row["gpu_time_median"],
            "gpu_time_max": row["gpu_time_max"],
            "gpu_time_min": row["gpu_time_min"],
        }
        list_unique_evts.append(
            {
                "event": evt,
                "stats": stats,
            }
        )
    return list_unique_evts


def profile_the_replay(replayer):
    """
    Profile the replay of an event.
    I know the name is confusing,
    but what I mean is that we are profiling the replay of the event
    Args:
        replayer (EventReplayer): The EventReplayer object.
        warmup (int): Number of warmup iterations.
        avg_steps (int): Number of iterations to average over.
    Returns:
        str: path of the replayed events trace

    """

    def trace_handler(p):
        p.export_chrome_trace("/tmp/replay_trace.json")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    wait = 10
    warmup = 5
    active = 10
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        ),
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as p:
        for idx in range(wait + warmup + active):
            replayer.replay()
            p.step()

    return "/tmp/replay_trace.json"


def compare_event_replay(perf_analyzer, list_unique_evts, verbose=False):
    list_replay_ir = []
    rows = []
    for evt_dict in list_unique_evts:
        evt = evt_dict["event"]
        stats = evt_dict["stats"]
        try:
            my_replayer = EventReplayer(evt, device="cuda")
        except Exception as e:
            print(f"Error creating EventReplayer: {e}")
            continue
        gpu_time = stats["gpu_time_median"]
        avg_time_us = benchmark_func(my_replayer.replay, warmup=50, avg_steps=100)
        percent_diff = (avg_time_us - gpu_time) / gpu_time * 100

        if verbose:
            print(evt["UID"])
            print(evt["name"])
            print(f"Average time per replay: {avg_time_us:.2f} us")
            print(f"GPU busy time: {gpu_time:.2f} us")
            print(f"Max GPU busy time: {stats['gpu_time_max']:.2f} us")
            print(f"Min GPU busy time: {stats['gpu_time_min']:.2f} us")
            print(f"Percent difference: {percent_diff:.2f}%")
            print(f"Abs difference: {avg_time_us - gpu_time:.2f} us")
        row = {
            "UID": evt["UID"],
            "name": evt["name"],
            "avg_time_replay_us": avg_time_us,
            "gt_time_mean_us": gpu_time,
            "gt_time_max_us": stats["gpu_time_max"],
            "gt_time_min_us": stats["gpu_time_min"],
            "percent_diff": percent_diff,
            "abs_diff": avg_time_us - gpu_time,
        }

        # profile the replay
        replay_path = profile_the_replay(my_replayer)
        replay_analyzer = TreePerfAnalyzer.from_file(replay_path)
        replayed_events = [
            e for e in replay_analyzer.tree.events if e.get("name") == evt["name"]
        ]
        # just taking the centre for now
        replayed_event = replayed_events[len(replayed_events) // 2]
        try:
            replayed_kernels = [
                replay_analyzer.tree.get_UID2event(uid)
                for uid in replayed_event.get("gpu_events")
            ]
        except Exception as e:
            print(f"Error getting replayed kernels: {e}")
            print(f"Event: {evt}")
            print(f"Replayed event: {replayed_event}")
            raise e
        replayed_kernel_names = [evt.get("name") for evt in replayed_kernels]

        # now we need to get the kernel names of the original event
        original_kernels = [
            perf_analyzer.tree.get_UID2event(uid) for uid in evt.get("gpu_events")
        ]
        original_kernel_names = [evt.get("name") for evt in original_kernels]
        # now we need to compare the two lists
        replayed_kernel_names = set(replayed_kernel_names)
        original_kernel_names = set(original_kernel_names)
        # now we need to get the difference of the two lists
        diff = replayed_kernel_names.difference(original_kernel_names)

        row["replayed_kernels"] = replayed_kernel_names
        row["original_kernels"] = original_kernel_names
        row["diff"] = diff
        row["kernels_match"] = len(diff) == 0
        rows.append(row)
        list_replay_ir.append(my_replayer.get_repro_info())
    df = pd.DataFrame(rows)
    return df, list_replay_ir


def test_resnet(full_run_trace_path=None, output_csv_path=None):
    """
    Run and profile a simple ResNet model
    Get events and replay them
    Compare the duration of the replayed events with the original events
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("Requires CUDA/HIP with at least one visible GPU")

    if full_run_trace_path is None or not os.path.exists(full_run_trace_path):
        full_run_trace_path = profile_resnet(full_run_trace_path)
    if output_csv_path is None:
        output_csv_path = "/tmp/resnet_event_replay_comparison.csv"

    perf_analyzer = TreePerfAnalyzer.from_file(full_run_trace_path)

    event_names_to_test = [
        "aten::convolution",
        "aten::convolution_backward",
        "aten::native_batch_norm",
        "aten::native_batch_norm_backward",
        "aten::addmm",
        "aten::mm",
        "aten::clamp_min_",
        "aten::threshold_backward",
        "aten::add_",
        "aten::mean",
        "aten::sum",
        "aten::max_pool2d_with_indices",
        # "aten::max_pool2d_with_indices_backward"
    ]
    list_unique_evts = [
        evt
        for event_name in event_names_to_test
        for evt in get_unique_evts(perf_analyzer, event_name)
    ]

    # now we have a list of unique events
    df_results, list_replay_ir = compare_event_replay(perf_analyzer, list_unique_evts)

    # see how many kernel_match failed
    num_kernels_match = df_results["kernels_match"].sum()
    assert num_kernels_match == len(
        df_results
    ), f"Number of kernels that do not match: {len(df_results) - num_kernels_match} out of {len(df_results)}"
    print("All kernels match between the replayed events and the original events.")
    # save the dataframe to a csv file
    df_results.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # save the replay IRs to a json file
    replay_ir_path = output_csv_path.replace(".csv", "_replay_ir.json")
    with open(replay_ir_path, "w") as f:
        json.dump(list_replay_ir, f, indent=4)

    # now we test the batched replay
    from TraceLens import EventReplay

    dir_batched_replay = os.path.dirname(EventReplay.__file__)
    batched_replay_file = os.path.join(dir_batched_replay, "batched_replay.py")
    print(f"Running batched replay from directory: {dir_batched_replay}")
    cmd = [
        "python",  # run as "python ..."
        batched_replay_file,  # path to the batched replay script
        os.path.abspath(replay_ir_path),
    ]
    result = subprocess.run(cmd, cwd=dir_batched_replay, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running batched replay: {result.stderr}")
    else:
        print("Batched replay completed successfully.")
        print(result.stdout)


# --- Unit tests for MIOpen translation, SkipReplayError, and Tensor[] support ---


def _make_conv_bias_event():
    """Create a minimal ConvBias_ event dict for testing."""
    return {
        "name": "ConvBias_",
        "args": {
            "Input Dims": [[2, 64, 56, 56], [128, 64, 3, 3], [128]],
            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
            "Input Strides": [[200704, 3136, 56, 1], [576, 9, 3, 1], [1]],
            "Concrete Inputs": ["", "", "", "1", "1"],
        },
    }


class TestMiopenTranslation:
    def test_translate_conv_bias_produces_convolution_name(self):
        event = _make_conv_bias_event()
        translated = EventReplayer._translate_miopen_fused_event(event)
        assert translated["name"] == "aten::convolution"

    def test_translate_conv_bias_has_9_args(self):
        event = _make_conv_bias_event()
        translated = EventReplayer._translate_miopen_fused_event(event)
        assert len(translated["args"]["Concrete Inputs"]) == 9
        assert len(translated["args"]["Input type"]) == 9
        assert len(translated["args"]["Input Dims"]) == 9
        assert len(translated["args"]["Input Strides"]) == 9

    def test_translate_conv_bias_preserves_tensor_args(self):
        event = _make_conv_bias_event()
        translated = EventReplayer._translate_miopen_fused_event(event)
        # First 3 args are tensors — dims/strides preserved, concrete inputs empty
        assert translated["args"]["Input Dims"][0] == [2, 64, 56, 56]
        assert translated["args"]["Input Dims"][1] == [128, 64, 3, 3]
        assert translated["args"]["Input Dims"][2] == [128]
        assert translated["args"]["Input type"][0] == "c10::BFloat16"

    def test_translate_conv_bias_pads_convolution_params(self):
        event = _make_conv_bias_event()
        translated = EventReplayer._translate_miopen_fused_event(event)
        concrete = translated["args"]["Concrete Inputs"]
        # stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1
        assert concrete[3] == "[1, 1]"
        assert concrete[4] == "[1, 1]"
        assert concrete[5] == "[1, 1]"
        assert concrete[6] == "false"
        assert concrete[7] == "[0, 0]"
        assert concrete[8] == "1"

    def test_translate_conv_bias_relu(self):
        event = _make_conv_bias_event()
        event["name"] = "ConvBiasReLU_"
        translated = EventReplayer._translate_miopen_fused_event(event)
        assert translated["name"] == "aten::convolution"

    def test_translate_backward_variants(self):
        for bwd_name in ["ConvBias_Backward", "ConvBiasReLU_Backward"]:
            event = _make_conv_bias_event()
            event["name"] = bwd_name
            translated = EventReplayer._translate_miopen_fused_event(event)
            assert translated["name"] == "aten::convolution_backward"

    def test_conv_bias_event_replayer_lazy(self):
        """Test that EventReplayer can process a ConvBias_ event in lazy mode."""
        event = _make_conv_bias_event()
        replayer = EventReplayer(event, lazy=True)
        repro_info = replayer.get_repro_info()
        assert repro_info["op_name"] == "aten::convolution"
        assert len(repro_info["replay_ir"]["list_pos_args"]) > 0


class TestSkipReplayError:
    def test_non_replayable_ops_raise_skip(self):
        for name in ["FlashAttnFunc", "LayerNormFn", "MoEDispatch"]:
            event = {
                "name": name,
                "args": {
                    "Input Dims": [],
                    "Input type": [],
                    "Input Strides": [],
                    "Concrete Inputs": [],
                },
            }
            with pytest.raises(SkipReplayError, match="custom op"):
                EventReplayer(event, lazy=True)

    def test_pseudo_ops_raise_skip(self):
        for name in ["pseudo_op::my_op", "_LinearForward"]:
            event = {
                "name": name,
                "args": {
                    "Input Dims": [],
                    "Input type": [],
                    "Input Strides": [],
                    "Concrete Inputs": [],
                },
            }
            with pytest.raises(SkipReplayError, match="pseudo operation|pseudo operation"):
                EventReplayer(event, lazy=True)

    def test_normal_aten_op_does_not_raise_skip(self):
        """A valid aten:: op should not raise SkipReplayError (it may fail at schema matching)."""
        event = {
            "name": "aten::add",
            "args": {
                "Input Dims": [[2, 3], [2, 3]],
                "Input type": ["float", "float"],
                "Input Strides": [[3, 1], [3, 1]],
                "Concrete Inputs": ["", ""],
            },
        }
        # Should raise ValueError (schema mismatch due to missing alpha arg), not SkipReplayError
        with pytest.raises(ValueError):
            EventReplayer(event, lazy=True)


class TestTensorListSchemaMatch:
    def test_tensor_list_schema_no_longer_raises(self):
        """Verify that _is_schema_match doesn't raise ValueError for Tensor[] types."""
        all_schemas = torch._C._jit_get_all_schemas()
        cat_schemas = [s for s in all_schemas if s.name == "aten::cat"]
        assert len(cat_schemas) > 0, "aten::cat schema not found"

        event = {
            "name": "aten::cat",
            "args": {
                "Input Dims": [[2, 3]],
                "Input type": ["float"],
                "Input Strides": [[3, 1]],
                "Concrete Inputs": [""],
            },
        }
        # Should not raise ValueError — it returns True or False
        for schema in cat_schemas:
            try:
                result = EventReplayer._is_schema_match(event, schema)
                # The result is a bool, not an exception
                assert isinstance(result, bool)
            except ValueError:
                pytest.fail(
                    "Tensor[] schema matching raised ValueError instead of returning bool"
                )

    def test_cat_event_replayer_lazy(self):
        """Test that EventReplayer can process an aten::cat event in lazy mode."""
        # aten::cat(Tensor[] tensors, int dim=0) -> Tensor
        event = {
            "name": "aten::cat",
            "args": {
                "Input Dims": [[2, 3], []],
                "Input type": ["float", "Scalar"],
                "Input Strides": [[3, 1], []],
                "Concrete Inputs": ["", "0"],
            },
        }
        replayer = EventReplayer(event, lazy=True)
        repro_info = replayer.get_repro_info()
        assert repro_info["op_name"] == "aten::cat"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA/HIP GPU"
)
class TestEndToEndConvBiasReplay:
    def test_conv_bias_replay_executes(self):
        """End-to-end test: ConvBias_ event is translated and replayed on GPU."""
        event = _make_conv_bias_event()
        replayer = EventReplayer(event, device="cuda", lazy=False)
        # Should not raise
        replayer.replay()


# # --- Example Usage (remains the same logic) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile ResNet, replay events, and compare."
    )
    parser.add_argument(
        "--full_run_trace_path",
        type=str,
        default="resnet_full_run_trace.json",
        help="Path to the full run trace file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="resnet_event_replay_comparison.csv",
        help="Path to save the output CSV file with comparison results.",
    )
    # You could add more arguments for other configurable paths like replay traces

    args = parser.parse_args()

    test_resnet(
        full_run_trace_path=args.full_run_trace_path, output_csv_path=args.output_csv
    )
