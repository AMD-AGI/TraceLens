###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import argparse
import os
import pytest
import torch
import torch.nn
#import TraceLens
# Normalization layers

default_normalization_layer_trace_file = "traces/perf_model/normalization/normalization_layer_test.json.gz"

@pytest.mark.parameterize("trace_file", [default_normalization_layer_trace_file])
def test_normalization_layers(trace_file: str):
    assert os.path.exists(trace_file), f"Trace file {trace_file} does not exist"

def create_normalization_layer_trace(outfile: str):
    # super simple network with the normalization layers that we care about
    class Net(torch.nn.Module):
        def __init__(self, input_shape):
            super(Net, self).__init__()
            self.bn = torch.nn.BatchNorm2d(input_shape[-3], affine=True)
            self.ln = torch.nn.LayerNorm(input_shape[1:], elementwise_affine=True)
            self.gn = torch.nn.GroupNorm(4, input_shape[1], affine=True)
            self.rmsn = torch.nn.RMSNorm(input_shape[2:], elementwise_affine=True)
            self.inn = torch.nn.InstanceNorm2d(input_shape[-3], affine=True)
            self.bn2 = torch.nn.BatchNorm2d(input_shape[-3], affine=False)
            self.ln2 = torch.nn.LayerNorm(input_shape[1:], elementwise_affine=False)
            self.gn2 = torch.nn.GroupNorm(4, input_shape[1], affine=False)
            self.rmsn2 = torch.nn.RMSNorm(input_shape[2:], elementwise_affine=False)
            self.inn2 = torch.nn.InstanceNorm2d(input_shape[-3], affine=False)
        def forward(self, x):
            with torch.profiler.record_function("BatchNorm affine=True"):
                x = self.bn(x)
            with torch.profiler.record_function("LayerNorm affine=True"):
                x = self.ln(x)
            with torch.profiler.record_function("GroupNorm affine=True"):
                x = self.gn(x)
            with torch.profiler.record_function("RMSNorm affine=True"):
                x = self.rmsn(x)
            with torch.profiler.record_function("InstanceNorm affine=True"):
                x = self.inn(x)
            with torch.profiler.record_function("BatchNorm affine=False"):
                x = self.bn2(x)
            with torch.profiler.record_function("LayerNorm affine=False"):
                x = self.ln2(x)
            with torch.profiler.record_function("GroupNorm affine=False"):
                x = self.gn2(x)
            with torch.profiler.record_function("RMSNorm affine=False"):
                x = self.rmsn2(x)
            with torch.profiler.record_function("InstanceNormNorm affine=False"):
                x = self.inn2(x)
            return x
    torch.set_default_device('cuda')
    input_shape = [8, 16, 32, 32]
    net = Net(input_shape)
    torch.manual_seed(0)
    
    x = torch.randn(input_shape)
    criterion = torch.nn.MSELoss()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
        acc_events=True,
    ) as p:
        for i in range(3):
            outputs = net(x)
            loss = criterion(outputs, torch.randn(input_shape))
            loss.backward()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    p.export_chrome_trace(outfile)

def main():
    parser = argparse.ArgumentParser(
        description="Tests for the perf model. Can run this to generate traces."
    )
    parser.add_argument("--create-normalization-trace", action="store_true")
    parser.add_argument(
        "--normalization-trace-file",
        type=str,
        help="Output file for the normalzation model trace",
    )
    if len(os.sys.argv) == 1:
        parser.print_help()
        os.sys.exit(1)
    args = parser.parse_args()
    if args.normalization_trace_file is None:
        args.normalization_trace_file = default_normalization_layer_trace_file
    else:
        args.create_normalization_trace = True
    if args.create_normalization_trace:
        print(f"Creating normalization layer trace at {args.normalization_trace_file}")
        create_normalization_layer_trace(args.normalization_trace_file)

if __name__ == "__main__":
    main()


