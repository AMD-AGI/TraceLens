#!/usr/bin/env python3
"""Generate real PyTorch profiler traces from small MoE workloads.

Runs tiny MoE forward passes on GPU, captures Chrome Trace Event JSON
via torch.profiler, and saves as .json.gz for TraceLens eval unit tests.

Usage (inside container):
    python3 evals/generate_moe_traces.py --gpu MI300X \
        [--output-dir evals/unit_tests/moe]
"""

import argparse
import gzip
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoE(nn.Module):
    """Minimal MoE layer: gate -> top-k routing -> expert FFNs -> combine."""

    def __init__(self, num_experts, hidden, intermediate, topk):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.topk = topk
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        self.w2 = nn.Parameter(torch.randn(num_experts, intermediate, hidden))

    def forward(self, x):
        # x: (tokens, hidden)
        tokens = x.size(0)
        label = (
            f"MoE[experts={self.num_experts}, tokens={tokens}, "
            f"topk={self.topk}, hidden={self.hidden}, "
            f"intermediate={self.intermediate}]"
        )
        with torch.profiler.record_function(label):
            gate_logits = self.gate(x)  # (tokens, num_experts)
            weights, indices = torch.topk(gate_logits, self.topk, dim=-1)
            weights = F.softmax(weights, dim=-1)  # (tokens, topk)

            flat_indices = indices.reshape(-1)  # (tokens * topk,)
            flat_x = x.unsqueeze(1).expand(-1, self.topk, -1).reshape(-1, x.size(-1))

            expert_w1 = self.w1[flat_indices]  # (tokens*topk, hidden, intermediate)
            expert_w2 = self.w2[flat_indices]  # (tokens*topk, intermediate, hidden)

            with torch.profiler.record_function(
                f"expert_ffn[w1=({self.num_experts},{self.hidden},{self.intermediate}), "
                f"w2=({self.num_experts},{self.intermediate},{self.hidden}), "
                f"active_pairs={tokens * self.topk}]"
            ):
                h = torch.bmm(flat_x.unsqueeze(1), expert_w1).squeeze(1)
                h = F.silu(h)
                out = torch.bmm(h.unsqueeze(1), expert_w2).squeeze(1)

            out = out.reshape(tokens, self.topk, -1)
            out = (out * weights.unsqueeze(-1)).sum(dim=1)
        return out


MOE_CONFIGS = [
    {
        "id": "moe_01_many_experts_few_tokens",
        "num_experts": 64, "tokens": 16, "hidden": 512,
        "intermediate": 1024, "topk": 2,
        "desc": "64 experts, 16 tokens, topk=2 - extreme under-utilization",
    },
    {
        "id": "moe_02_few_experts_many_tokens",
        "num_experts": 8, "tokens": 256, "hidden": 512,
        "intermediate": 1024, "topk": 2,
        "desc": "8 experts, 256 tokens, topk=2 - well-utilized",
    },
    {
        "id": "moe_03_top1_sparse",
        "num_experts": 16, "tokens": 128, "hidden": 512,
        "intermediate": 1024, "topk": 1,
        "desc": "16 experts, 128 tokens, topk=1 - sparse routing",
    },
    {
        "id": "moe_04_top2_dense",
        "num_experts": 16, "tokens": 128, "hidden": 512,
        "intermediate": 1024, "topk": 2,
        "desc": "16 experts, 128 tokens, topk=2 - dense routing",
    },
    {
        "id": "moe_05_large_capacity",
        "num_experts": 8, "tokens": 512, "hidden": 1024,
        "intermediate": 2048, "topk": 2,
        "desc": "8 experts, 512 tokens, large dims - scale test",
    },
    {
        "id": "moe_06_tiny_batch_many_experts",
        "num_experts": 128, "tokens": 4, "hidden": 512,
        "intermediate": 1024, "topk": 8,
        "desc": "128 experts, 4 tokens, topk=8 - launch overhead dominant",
    },
]


def generate_trace(config, output_dir, device, platform, warmup=5, active_iters=3):
    """Run MoE forward pass under torch.profiler and save trace as .json.gz."""
    case_id = config["id"]
    case_dir = os.path.join(output_dir, f"{case_id}_analysis_output")
    ref_dir = os.path.join(case_dir, "analysis_output_ref")
    os.makedirs(ref_dir, exist_ok=True)

    model = SimpleMoE(
        num_experts=config["num_experts"],
        hidden=config["hidden"],
        intermediate=config["intermediate"],
        topk=config["topk"],
    ).to(device).to(torch.bfloat16)

    x = torch.randn(config["tokens"], config["hidden"],
                     device=device, dtype=torch.bfloat16)

    # Warmup outside profiler (ensure kernels are compiled / JIT'd)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Profile with simple settings (no schedule - ROCm roctracer works best this way)
    trace_json_path = os.path.join(case_dir, f"{case_id}.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_modules=True,
    ) as prof:
        for _ in range(active_iters):
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_json_path)

    # Compress to .gz
    gz_path = trace_json_path + ".gz"
    with open(trace_json_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(trace_json_path)

    size_kb = os.path.getsize(gz_path) / 1024
    print(f"  {case_id}: {gz_path} ({size_kb:.1f} KB)")
    print(f"    Config: {config['num_experts']} experts, {config['tokens']} tokens, "
          f"topk={config['topk']}, hidden={config['hidden']}, "
          f"intermediate={config['intermediate']}")

    return {
        "id": case_id,
        "sub_category": "moe",
        "trace_path": os.path.relpath(gz_path, os.path.dirname(output_dir) + "/.."),
        "reference_dir": os.path.relpath(ref_dir, os.path.dirname(output_dir) + "/.."),
        "platform": platform,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate MoE profiler traces")
    parser.add_argument("--gpu", required=True,
                        help="GPU / platform label written into the CSV "
                             "(e.g. MI300X, H100, A100).")
    parser.add_argument("--output-dir", default="evals/unit_tests/moe",
                        help="Output directory for traces")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific case IDs to generate (default: all)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No GPU available. Run inside a ROCm/CUDA container with GPU access.")
        return

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Platform label: {args.gpu}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Output: {args.output_dir}")
    print()

    configs = MOE_CONFIGS
    if args.cases:
        configs = [c for c in configs if c["id"] in args.cases]

    csv_rows = []
    for config in configs:
        print(f"Generating: {config['id']} - {config['desc']}")
        row = generate_trace(config, args.output_dir, device, args.gpu)
        csv_rows.append(row)
        print()

    print("--- CSV entries for unit_test_traces.csv ---")
    print("id,sub_category,trace_path,reference_dir,platform")
    for r in csv_rows:
        print(f"{r['id']},{r['sub_category']},{r['trace_path']},"
              f"{r['reference_dir']},{r['platform']}")

    print(f"\nGenerated {len(csv_rows)} traces.")


if __name__ == "__main__":
    main()
