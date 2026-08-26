<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Profile distributed PyTorch workloads

```{meta}
:description: Learn how to profile distributed PyTorch workloads using DDP and torchrun, with rank-unique trace files and selective rank profiling to manage trace volume.
:keywords: TraceLens, PyTorch distributed, DDP, torchrun, distributed profiling, GPU trace, ROCm, AMD Instinct, multi-GPU, NCCL, performance profiling, CUDA
```

Learn the core concepts behind profiling a distributed workload. It directly builds on the [Configure and run the PyTorch profiler](./torch-profiling.md) that covered profiling a single-GPU run.

This tutorial focuses on what changes when a workload runs under PyTorch Distributed (DDP or otherwise) — not on the mechanics of DDP itself, but on how profiling behaves when multiple processes participate.

## How distributed profiling works

This example uses `torchrun` to launch the job. The key concept is that `torchrun` starts one identical Python process for every GPU you are using.

When you profile this distributed workload, you are profiling multiple identical programs at the same time, each running on its own GPU.

### What rank means

To coordinate, each process needs a unique identity. `torchrun` provides this identity using environment variables, which we call *rank*.

| Variable | Description |
|----------|-------------|
| `WORLD_SIZE` | The total number of processes in the entire job (for example, 2 nodes × 8 GPUs/node = 16). |
| `RANK` | The global ID for this specific process, from `0` to `WORLD_SIZE - 1`. |
| `LOCAL_RANK` | The local GPU index on this specific machine (for example, `0`, `1`, ... `7` for an 8-GPU node). |

Every process runs the same Python program, but uses these rank variables to:

- Select its GPU using `LOCAL_RANK` (for example, `cuda:0`, `cuda:1`).
- Participate in collective operations such as gradient sync using `RANK`.

### Why naive profiling breaks

If every rank writes to the same `trace.json`, the independent processes race and overwrite each other's output.

- To avoid file races (required): Use a unique filename per rank, such as `rank{rank}_iter{start}_{end}.json`. This ensures concurrent processes never write to the same file.

- To reduce trace volume (optional): Enable profiling only for selected ranks (for example, `"0"` or `"0,3"`). Every rank still executes the full workload, but only the chosen ranks record traces—keeping the data manageable and export overhead low.

The `make_profiler_ctx()` helper in the training script below implements both of these ideas.

For more on the DDP setup itself, see the [DDP Tutorial — PyTorch](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and the [`torchrun` launcher docs](https://pytorch.org/docs/stable/elastic/run.html).

## Training script

Save the following as `train_ddp.py`:

```python
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, schedule as profiler_schedule


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=local_rank,
    )

    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def make_model_and_data(rank):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model = models.resnet18().to(device).to(dtype)
    ddp_model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    B, C, H, W = 10, 3, 224, 224
    num_classes = 1000
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
    dummy_target = torch.randn(B, num_classes, device=device, dtype=dtype)

    return ddp_model, optimizer, dummy_input, dummy_target, device


def train_step(ddp_model, optimizer, X, Y):
    optimizer.zero_grad(set_to_none=True)
    out = ddp_model(X)
    loss = F.mse_loss(out, Y)
    loss.backward()
    optimizer.step()
    return loss


def make_profiler_ctx(rank, profile_ranks, traces_dir):
    """
    Returns a real profiler for selected ranks and a no-op profiler for others.
    Uses the same wait=10, warmup=5, active=3, repeat=1 schedule as the
    PyTorch profiling tutorial.
    """

    class NullProfiler:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def step(self):
            pass

    if profile_ranks != "all":
        profile_ranks_int = {int(x) for x in profile_ranks.split(",") if x.strip()}
        if rank not in profile_ranks_int:
            return NullProfiler()

    os.makedirs(traces_dir, exist_ok=True)

    sched = profiler_schedule(wait=10, warmup=5, active=3, repeat=1)

    def trace_handler(p):
        end_iter = p.step_num
        start_iter = end_iter - 3 + 1
        trace_path = os.path.join(traces_dir, f"rank{rank}_iter{start_iter}_{end_iter}.json")
        p.export_chrome_trace(trace_path)

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    )


def ddp_worker(rank, world_size, profile_ranks, traces_dir):
    rank, world_size, local_rank = setup()
    print(f"[rank {rank}] starting worker")

    ddp_model, optimizer, dummy_input, dummy_target, device = make_model_and_data(rank)

    total_steps = 100
    prof = make_profiler_ctx(rank, profile_ranks, traces_dir)
    with prof:
        for step_idx in range(total_steps):
            loss = train_step(ddp_model, optimizer, dummy_input, dummy_target)
            prof.step()
            if rank == 0 and step_idx % 10 == 0:
                print(f"[rank {rank}] step {step_idx}/{total_steps} loss={loss.item():.4f}", flush=True)

    dist.barrier(device_ids=[rank])
    cleanup()
    print(f"[rank {rank}] finished worker")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    profile_ranks = "0, 2, 4"  # or "all"
    traces_dir = "./traces"
    ddp_worker(rank, world_size, profile_ranks, traces_dir)


if __name__ == "__main__":
    main()
```

## Run the job

Launch with `torchrun`, specifying the number of GPUs:

```bash
torchrun --nproc_per_node=8 train_ddp.py
```

After the job completes, each profiled rank writes its own Chrome trace file in `./traces`. For example:

```text
traces/rank0_iter10_12.json
traces/rank2_iter10_12.json
traces/rank4_iter10_12.json
```

## Multi-node profiling

For multi-node runs, nothing changes conceptually from the profiler's point of view. If you have four nodes with eight GPUs each (32 total ranks):

- Global ranks span `0` to `31` across all nodes.
- The same rank-filtering and unique filename logic applies.
- Each process writes its trace to the storage path accessible to that process.

If you write to local disk paths (for example, `./traces`), each node contains only the trace files for its local ranks. If you write to a shared NFS or network-mounted directory, all ranks' traces appear in one place because all nodes share the same storage.

## Related topics

- [Configure and run the PyTorch profiler](./torch-profiling.md)
- [Fuse multi-rank traces](../trace-fusion.md)
- [Analyze collective communication](../nccl-analysis.md)
- [Generate a PyTorch performance report](../generate-perf-report-pytorch.md)
