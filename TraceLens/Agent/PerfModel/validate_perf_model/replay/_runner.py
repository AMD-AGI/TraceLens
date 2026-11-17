###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI dispatcher that rocprofv3 wraps as a subprocess.

Usage:

    python replay/_runner.py --op gemm_a8w8_blockscale --M 2048 --N 4096 --K 8192

Finds ``test_<op>`` in the first of :data:`MODULES` that defines it and calls
it with the supplied dimensional kwargs. Every harness tolerates extra kwargs
via ``**_`` so unrelated CLI flags are silently ignored, builds its inputs and
returns the kernel call; :func:`measure` then runs that call.
``--op __generic__`` dispatches to ``generic.test_generic_simple_op``.
"""

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    _PKG = "replay"
else:
    _PKG = __package__

# Constants

# Test modules searched for ``test_<op>``, in order. Add a module here when a
# new op family gets its own file.
MODULES = ("gemm", "moe", "attention", "rmsnorm", "other", "rope")

# Larger than the L2 and on-package cache of the supported GPUs.
L2_FLUSH_BYTES = 512 * 1024 * 1024

# Flags forwarded to test functions; validate_perf_model builds its runner
# argv from the same lists.
INT_FLAGS = (
    "M",
    "N",
    "K",
    "E",
    "topk",
    "group_size",
    "seq_len",
    "num_heads_q",
    "num_heads_kv",
    "head_dim",
    "block_n",
    "block_k",
    "block_m",
    "split_k",
    "num_decode_seqs",
    "ctx_len",
    "prefill_seq_len",
    "n_ctx",
    "ctx_qlen",
)
STR_FLAGS = (
    "in_dtype",
    "w_dtype",
    "out_dtype",
    "scale_dtype",
    "quant_dtype",
    "quant_type",
    "activation",
    "kv_dtype",
    "bias_dtype",
)


def find_test_fn(op):
    """Return the ``test_<op>`` callable for ``op``, or ``None`` if none exists.

    A module that fails to import only disables its own ops, since kernel
    libraries for some op families may be absent in a given environment.
    """
    if op == "__generic__":
        return importlib.import_module(f"{_PKG}.generic").test_generic_simple_op
    for modname in MODULES:
        try:
            mod = importlib.import_module(f"{_PKG}.{modname}")
        except (ImportError, OSError, RuntimeError) as exc:
            print(
                f"_runner: note: module '{modname}' unavailable ({exc}); "
                "its ops will be skipped.",
                flush=True,
            )
            continue
        fn = getattr(mod, f"test_{op}", None)
        if fn is not None:
            return fn
    return None


def measure(call, num_warmup):
    """Run ``call`` ``num_warmup`` times, then once more as the measured call.

    Before the measured call, an L2 flush makes it start from a cold cache, so
    its read bytes are not lowered by data the warmup calls left in L2, and a
    ``bitwise_not`` marker separates the warmup dispatches from the measured
    ones in the rocprofv3 trace. Both are ``at::native`` kernels, which the
    validator treats as infrastructure and leaves out of the results.
    """
    import torch

    for _ in range(num_warmup):
        call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    flush.fill_(1)
    marker = torch.zeros(1, dtype=torch.int32, device="cuda")
    marker.bitwise_not_()
    torch.cuda.synchronize()
    call()
    torch.cuda.synchronize()
    print("test: done", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Validate-perf-model test runner (rocprofv3 wraps this)."
    )
    p.add_argument(
        "--op",
        required=True,
        help="OP_REGISTRY key selecting which test_<op> to invoke.",
    )
    for k in INT_FLAGS:
        p.add_argument(f"--{k}", type=int, default=None)
    for k in STR_FLAGS:
        p.add_argument(f"--{k}", type=str, default=None)
    p.add_argument("--annotation", default=None)
    p.add_argument("--num-warmup", type=int, default=3, dest="num_warmup")
    p.add_argument(
        "--varlen-seed",
        type=int,
        default=42,
        dest="varlen_seed",
        help="RNG seed for variable-length attention seq partitioning.",
    )
    p.add_argument(
        "--varlen-num-seqs",
        type=int,
        default=4,
        dest="varlen_num_seqs",
        help="Number of sequences in varlen attention harness.",
    )
    p.add_argument(
        "--varlen-scenario",
        default="random",
        choices=["random", "mixed_prefill_decode"],
        dest="varlen_scenario",
        help=(
            "Varlen attention layout: 'random' for self-attention with a random "
            "partition of seq_len tokens, or 'mixed_prefill_decode' for one prefill "
            "seq (Q=K=seq_len) plus (varlen_num_seqs-1) decode seqs (Q=1, K=seq_len each)."
        ),
    )
    p.add_argument("--input-dims-json", default=None, dest="input_dims_json")
    p.add_argument("--input-types-json", default=None, dest="input_types_json")
    p.add_argument(
        "--concrete-inputs-json",
        default=None,
        dest="concrete_inputs_json",
        help="Traced non-tensor arguments (generic-CSV mode only).",
    )
    p.add_argument("--op-namespace", default=None, dest="op_namespace")
    p.add_argument("--op-fn-name", default=None, dest="op_fn_name")
    p.add_argument(
        "--registry-key",
        default=None,
        dest="registry_key",
        help="Original OP_REGISTRY key (generic-CSV mode only).",
    )
    args = p.parse_args()

    fn = find_test_fn(args.op)
    if fn is None:
        raise SystemExit(
            f"_runner: no test_{args.op} in replay/{{{','.join(MODULES)}}}.py"
        )

    kwargs = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ("op", "num_warmup")
    }
    print(f"_runner: dispatching test for op={args.op}", flush=True)
    measure(fn(**kwargs), args.num_warmup)


if __name__ == "__main__":
    main()
