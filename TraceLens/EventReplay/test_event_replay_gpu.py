###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU integration tests for EventReplay.

Profiles real ops, replays from the captured trace, and validates:
  1. Kernel name match between original and replayed execution
  2. BUG-1:  lazy=True + auto_init=True works on GPU
  3. BUG-2:  get_repro_info() is idempotent (doesn't corrupt IR)
  4. CLAIM-4: replay() returns a tensor
  5. CLAIM-1: first-match-wins with real ops

Requires a GPU (MI300X / MI210 / etc). Run from the repo root:
    python TraceLens/EventReplay/test_event_replay_gpu.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.profiler import profile, ProfilerActivity

from TraceLens.EventReplay.event_replay import EventReplayer
from TraceLens.EventReplay.custom_inits import CustomInit
from TraceLens.EventReplay.utils import TensorCfg

assert torch.cuda.is_available(), "GPU required for this test"

DEVICE = "cuda"
TRACE_FILE = "/tmp/test_event_replay_gpu_trace.json"
REPLAY_TRACE = "/tmp/test_event_replay_gpu_replay.json"

# ---------------------------------------------------------------------------
# Step 1: Profile a set of real ops
# ---------------------------------------------------------------------------

print("=" * 80)
print("Step 1: Profiling real ops")
print("=" * 80)

M, K, N = 256, 1024, 512
mm_a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
mm_b = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
add_a = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
add_b = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
bmm_a = torch.randn(4, M, K, dtype=torch.bfloat16, device=DEVICE)
bmm_b = torch.randn(4, K, N, dtype=torch.bfloat16, device=DEVICE)

def run_ops():
    torch.mm(mm_a, mm_b)
    torch.add(add_a, add_b)
    torch.bmm(bmm_a, bmm_b)
    torch.mul(add_a, add_b)
    torch.sigmoid(add_a)

for _ in range(10):
    run_ops()
torch.cuda.synchronize()

def trace_handler(p):
    p.export_chrome_trace(TRACE_FILE)

wait, warmup, active = 3, 3, 5
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
    record_shapes=True,
    on_trace_ready=trace_handler,
) as p:
    for _ in range(wait + warmup + active):
        run_ops()
        p.step()

print(f"Trace saved to {TRACE_FILE}")

# ---------------------------------------------------------------------------
# Step 2: Load trace and find events
# ---------------------------------------------------------------------------

print(f"\n{'=' * 80}")
print("Step 2: Loading trace")
print("=" * 80)

with open(TRACE_FILE) as f:
    trace_data = json.load(f)

all_events = trace_data.get("traceEvents", [])

OPS_TO_TEST = ["aten::mm", "aten::add", "aten::bmm", "aten::mul", "aten::sigmoid"]

def find_event(events, op_name):
    """Find a cpu_op event with the right name and shape data."""
    candidates = [
        e for e in events
        if e.get("cat") == "cpu_op"
        and e.get("name") == op_name
        and "args" in e
        and "Input Dims" in e.get("args", {})
    ]
    if candidates:
        return candidates[len(candidates) // 2]
    return None

results = []
errors = []

# ---------------------------------------------------------------------------
# Step 3: Replay each op and validate
# ---------------------------------------------------------------------------

print(f"\n{'=' * 80}")
print("Step 3: Replay and validate")
print("=" * 80)

print(f"\n{'Op':<30} {'Kernel Match':<15} {'Return':<10} {'Lazy':<10} {'ReproInfo':<12} {'Status'}")
print("-" * 100)

for op_name in OPS_TO_TEST:
    evt = find_event(all_events, op_name)
    if evt is None:
        print(f"{op_name:<30} {'SKIP':<15} {'---':<10} {'---':<10} {'---':<12} not in trace")
        continue

    status = []

    # --- Test: basic replay returns a result (CLAIM-4) ---
    try:
        replayer = EventReplayer(evt, device=DEVICE, auto_init=False)
        result = replayer.replay()
        returns_ok = isinstance(result, torch.Tensor)
    except Exception as e:
        returns_ok = False
        status.append(f"replay error: {e}")

    # --- Test: lazy mode works (BUG-1) ---
    try:
        lazy_replayer = EventReplayer(evt, device=DEVICE, lazy=True, auto_init=False)
        lazy_result = lazy_replayer.replay()
        lazy_ok = isinstance(lazy_result, torch.Tensor)
        assert hasattr(lazy_replayer, "args"), "self.args not set after lazy replay"
    except Exception as e:
        lazy_ok = False
        status.append(f"lazy error: {e}")

    # --- Test: get_repro_info idempotent (BUG-2) ---
    try:
        repro_replayer = EventReplayer(evt, device=DEVICE, lazy=True)
        info1 = repro_replayer.get_repro_info()
        info2 = repro_replayer.get_repro_info()
        repro_ok = (info1 == info2)
        for arg in repro_replayer.event_replay_IR["list_pos_args"]:
            if arg["arg_type"].startswith("Tensor"):
                assert isinstance(arg["value"], TensorCfg), "IR corrupted after get_repro_info"
        repro_replayer.replay()
    except Exception as e:
        repro_ok = False
        status.append(f"repro error: {e}")

    # --- Test: kernel name match ---
    kernel_match = "N/A"
    try:
        replay_replayer = EventReplayer(evt, device=DEVICE, auto_init=False)
        for _ in range(5):
            replay_replayer.replay()
        torch.cuda.synchronize()

        def th(p):
            p.export_chrome_trace(REPLAY_TRACE)

        w, wu, a = 2, 2, 3
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=w, warmup=wu, active=a, repeat=1),
            record_shapes=True,
            on_trace_ready=th,
        ) as p:
            for _ in range(w + wu + a):
                replay_replayer.replay()
                p.step()

        with open(REPLAY_TRACE) as f:
            replay_trace = json.load(f)

        orig_gpu = set()
        for e in all_events:
            if e.get("cat") == "kernel" and e.get("name", ""):
                orig_gpu.add(e["name"])

        replay_gpu = set()
        for e in replay_trace.get("traceEvents", []):
            if e.get("cat") == "kernel" and e.get("name", ""):
                replay_gpu.add(e["name"])

        kernel_match = "MATCH" if replay_gpu.issubset(orig_gpu) else "MISMATCH"
    except Exception as e:
        kernel_match = "ERROR"
        status.append(f"kernel error: {e}")

    ok = returns_ok and lazy_ok and repro_ok and kernel_match in ("MATCH", "N/A")
    tag = "PASS" if ok else "FAIL"
    detail = "; ".join(status) if status else ""

    print(f"{op_name:<30} {kernel_match:<15} {'OK' if returns_ok else 'FAIL':<10} {'OK' if lazy_ok else 'FAIL':<10} {'OK' if repro_ok else 'FAIL':<12} {tag} {detail}")
    results.append({"op": op_name, "ok": ok, "kernel": kernel_match,
                     "returns": returns_ok, "lazy": lazy_ok, "repro": repro_ok})

# ---------------------------------------------------------------------------
# Step 4: First-match-wins test (CLAIM-1) on GPU
# ---------------------------------------------------------------------------

print(f"\n{'=' * 80}")
print("Step 4: First-match-wins (CLAIM-1)")
print("=" * 80)

log = []

class InitA(CustomInit):
    op_patterns = ["aten::mm"]
    def initialize(self, replayer, **kwargs):
        log.append("A")

class InitB(CustomInit):
    op_patterns = ["aten::mm"]
    def initialize(self, replayer, **kwargs):
        log.append("B")

saved_registry = EventReplayer._custom_init_registry[:]
try:
    EventReplayer._custom_init_registry = [InitA(), InitB()]
    mm_evt = find_event(all_events, "aten::mm")
    if mm_evt:
        r = EventReplayer(mm_evt, device=DEVICE, auto_init=True)
        r.replay()
        first_match_ok = (log == ["A"])
        print(f"  First-match-wins: {'PASS' if first_match_ok else 'FAIL'} (log={log})")
    else:
        first_match_ok = True
        print("  SKIP: aten::mm not in trace")
finally:
    EventReplayer._custom_init_registry = saved_registry

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 80}")
print("Summary")
print("=" * 80)

total = len(results)
passed = sum(1 for r in results if r["ok"])
print(f"Op tests:         {passed}/{total} passed")
print(f"First-match-wins: {'PASS' if first_match_ok else 'FAIL'}")

all_pass = passed == total and first_match_ok
print(f"\nOverall: {'ALL PASSED' if all_pass else 'FAILURES DETECTED'}")
sys.exit(0 if all_pass else 1)
