###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression test for TraceLens issue #422 (and related #423):
JAX trace events emitted around ``transformer_engine.jax`` custom calls
must be categorised as ``TE`` instead of ``Uncategorized Events``.

The bug:

- ``__amd_rocclr_fillBufferAligned.kd`` events have no TE substring in
  their name, so the name-keyword scan in
  ``JaxOpKeys.ClassCategories`` misses them and they land in
  ``Uncategorized Events``. Their ``args["hlo_op"]`` *does* identify
  the surrounding XLA op (``te_fused_attn_backward_ffi.*``).
- ``te_fused_attn_{forward,backward}_ffi`` events themselves did not
  match the old ``TEKeys = ["transformer_engine"]`` keyword.
- ``FillBuffer`` substring used to be in ``ConvKeys``, mis-routing any
  TE-adjacent buffer-init fusion to ``Conv``.

The fixture is a small JAX 0.8 / TE 2.8 / ROCm 7.1.1 trace generated
by ``learning/jax-minimal/12_te_fused_attn.py`` (2-layer training
forward + ``jax.value_and_grad``, BS3HD, causal mask, dropout 0.1).
"""

import os

from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.util import DataLoader, TraceEventUtils

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "test_data_jax_te_fused_attn_categorization")


def _trace_path():
    for fname in os.listdir(FIXTURE_DIR):
        if fname.endswith(".xplane.pb"):
            return os.path.join(FIXTURE_DIR, fname)
    raise FileNotFoundError(f"no xplane.pb under {FIXTURE_DIR}")


def _bucket_compute_events():
    data = DataLoader.load_data(filename_path=_trace_path(), save_preprocessed=False)
    events = data["traceEvents"]
    # GPU-stream compute events: pid<=100, tid<100, with a duration.
    compute = [
        e
        for e in events
        if isinstance(e.get("name"), str)
        and "dur" in e
        and e.get("pid") is not None
        and int(e.get("pid")) <= 100
        and e.get("tid") is not None
        and int(e.get("tid")) < 100
    ]
    cat, uncat = JaxAnalyses.breakdown_compute_events(
        compute, group_by_gpu=False, group_by_name=False
    )
    return cat, uncat, compute


def test_te_fused_attn_ffi_events_land_in_te():
    """`te_fused_attn_{forward,backward}_ffi` events should match TEKeys
    via their event name (not via metadata fallback)."""
    cat, uncat, _ = _bucket_compute_events()
    # Both directions of TE FFI host events.
    leaked = [n for n in uncat if n.startswith("te_fused_attn_")]
    assert leaked == [], (
        f"te_fused_attn_* events leaked to uncategorized: {leaked}; "
        "expected to be matched by JaxOpKeys.TEKeys substring"
    )
    assert "TE" in cat, "expected TE bucket to be populated"


def test_fillbuffer_aligned_routed_to_te_via_metadata():
    """`__amd_rocclr_fillBufferAligned.kd` events sit inside TE custom
    calls and must be routed to TE via the args["hlo_op"] fallback."""
    _, uncat, compute = _bucket_compute_events()
    assert "__amd_rocclr_fillBufferAligned.kd" not in uncat, (
        "fillBufferAligned still uncategorized after #422 fix; "
        "expected metadata-aware fallback to route via args['hlo_op']"
    )
    # Sanity: at least one fillBufferAligned event in the fixture, and
    # all of them carry a TE-flavoured hlo_op.
    fbs = [e for e in compute if e["name"] == "__amd_rocclr_fillBufferAligned.kd"]
    assert (
        len(fbs) >= 10
    ), f"expected the fixture to contain fillBufferAligned events, got {len(fbs)}"
    for e in fbs:
        hlo_op = (e.get("args") or {}).get("hlo_op", "")
        assert "te_fused_attn" in hlo_op, (
            f"fixture event has unexpected hlo_op={hlo_op!r}; this test "
            "assumes the TE-fused-attn reproducer trace"
        )


def test_fillbuffer_substring_no_longer_routes_to_conv():
    """`"FillBuffer"` was removed from ConvKeys (issue #423 — those
    events come from XLA TE buffer-init fusions, not from real conv ops)."""
    conv_keys = TraceEventUtils.JaxOpKeys.ConvKeys
    assert (
        "FillBuffer" not in conv_keys
    ), f"FillBuffer should not be in ConvKeys anymore; got {conv_keys}"


def test_te_keys_include_te_fused_attn():
    """Bonus regression for #422: TEKeys must include `te_fused_attn`."""
    te_keys = TraceEventUtils.JaxOpKeys.TEKeys
    assert (
        "te_fused_attn" in te_keys
    ), f"expected 'te_fused_attn' in TEKeys; got {te_keys}"


def test_uncategorized_bucket_is_always_seeded():
    """When every event matches a category (no leftovers), the
    ``Uncategorized Events`` bucket must still be present with ``[0, 0]``
    so downstream consumers (``create_gpu_summary`` etc.) can index it
    unconditionally without ``KeyError``. Regression for review of #655."""
    events_all_te = [
        {"pid": 1, "tid": 1, "name": "te_fused_attn_forward_ffi", "dur": 100.0},
        {"pid": 1, "tid": 1, "name": "Cijk_mm_fp16", "dur": 50.0},
    ]
    cat, _ = JaxAnalyses.breakdown_compute_events(events_all_te, group_by_gpu=False)
    unc = TraceEventUtils.JaxOpKeys.UncategorizedEventKey
    assert (
        unc in cat
    ), f"Uncategorized bucket missing when all events match a category: {list(cat)}"
    assert cat[unc] == [0, 0], f"expected [0, 0], got {cat[unc]}"
