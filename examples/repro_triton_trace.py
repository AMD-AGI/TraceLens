###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Repro script: capture a Chrome trace of a torch.compile-generated Triton kernel
workload, then run TraceLens to expose the blank GFLOPS/TFLOPS/TB/s rows.

Usage (from repo root, arr_training env):
    python examples/repro_triton_trace.py

Then run TraceLens:
    python -m TraceLens.Reporting.generate_perf_report_pytorch \
        --profile_json_path torch_trace_output_claude/triton_repro/trace.json.gz \
        --output_xlsx_path torch_trace_output_claude/triton_repro/perf_report.xlsx
"""

import gzip
import os
import shutil

import torch
import torch.nn as nn
import torch.profiler

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "torch_trace_output_claude", "triton_repro"
)
DATE_TAG = __import__("datetime").date.today().strftime("%Y%m%d")
TRACE_PATH = os.path.join(OUT_DIR, f"trace_{DATE_TAG}.json.gz")


# ---------------------------------------------------------------------------
# Model: RMSNorm + hand-rolled Attention + SwiGLU MLP
# nn.MultiheadAttention triggers an Inductor AssertionError on ROCm; use
# explicit Q/K/V projections + F.scaled_dot_product_attention instead.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        q = self.q(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v(x).view(B, T, H, Hd).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 2048, n_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T, D = 8, 4096, 2048

    os.makedirs(OUT_DIR, exist_ok=True)

    model = TransformerBlock(dim=D).to(device=device, dtype=dtype)
    model = torch.compile(model, mode="default")

    x = torch.randn(B, T, D, device=device, dtype=dtype)

    # Warmup: let torch.compile finish compiling before we profile
    print("Warming up (torch.compile)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    print("Warmup done.")

    # Profiled run
    tmp_trace = os.path.join(OUT_DIR, "trace_tmp")
    print(f"Profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()

    # Export Chrome trace then gzip it
    raw_path = tmp_trace + ".json"
    prof.export_chrome_trace(raw_path)
    with open(raw_path, "rb") as f_in, gzip.open(TRACE_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(raw_path)

    print(f"Trace saved to: {TRACE_PATH}")
    print()
    print("Next step — run TraceLens:")
    print(f"  python -m TraceLens.Reporting.generate_perf_report_pytorch \\")
    print(f"      --profile_json_path {TRACE_PATH} \\")
    print(f"      --output_xlsx_path {os.path.join(OUT_DIR, 'perf_report.xlsx')}")


if __name__ == "__main__":
    main()
