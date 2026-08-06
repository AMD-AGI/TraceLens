###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceUtils.annotation_utils.

Covers every worked annotation examples
- vLLM native / detailed (``execute_...``)
- SGLang native / detailed (``step[...]``)
- ATOM (``prefill[]`` / ``decode[]``)
- capture annotations and non-matching input.

Each case asserts the parsed ``kind``, the populated ``IterationAnnotation``
fields, and (where relevant) ``meta`` and ``has_sqsk`` / ``chunk_stats()``.

The ``iter_details()`` / ``full_details()`` dicts are covered separately, since
both are consumed by key elsewhere and must agree on ``batch_size``.
"""

import pytest

from TraceLens.TraceUtils.annotation_utils import (
    ITERATION_BACKUP_PATTERNS,
    ITERATION_PATTERNS,
    PHASE_DECODE_ONLY,
    PHASE_PREFILL_ONLY,
    PHASE_PREFILLDECODE,
    CaptureAnnotation,
    IterationAnnotation,
    average_detail,
    classify_phase,
    find_events_by_patterns,
    find_iteration_roots_by_priority,
    find_phase_from_window,
    has_context,
    has_generation,
    is_decode_only,
    is_mixed,
    is_prefill_only,
    iteration_details,
)

# Case tables: (name, kind, expected_fields, expected_meta)
VLLM_CASES = [
    (
        "execute_context_2(14721)_generation_0(0)",
        "vllm_native",
        dict(
            batch_size=14721,
            context_requests=2,
            context_sum=14721,
            generation_requests=0,
            generation_sum=0,
            has_sqsk=False,
        ),
        {},
    ),
    (
        "execute_context_0(0)_generation_64(64)",
        "vllm_native",
        dict(
            batch_size=64,
            context_requests=0,
            context_sum=0,
            generation_requests=64,
            generation_sum=64,
            has_sqsk=False,
        ),
        {},
    ),
    (  # spec-decode / MTP: generation_sum (128) exceeds request count (32)
        "execute_context_0(0)_generation_32(128)",
        "vllm_native",
        dict(
            batch_size=128,
            context_requests=0,
            context_sum=0,
            generation_requests=32,
            generation_sum=128,
            has_sqsk=False,
        ),
        {},
    ),
    (  # mixed: both phases populated
        "execute_context_2(6144)_generation_3(3)",
        "vllm_native",
        dict(
            batch_size=6147,
            context_requests=2,
            context_sum=6144,
            generation_requests=3,
            generation_sum=3,
            has_sqsk=False,
        ),
        {},
    ),
    # --- detailed (detailed_trace_annotation) ---
    (  # fresh prefill -> sk == sq, sqsk == sqsq
        "execute_14721_context_2(sq14721sk14721sqsq108745533sqsk108745533)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(
            batch_size=14721,
            context_requests=2,
            c_sq=14721,
            c_sk=14721,
            c_sqsq=108745533,
            c_sqsk=108745533,
            context_sum=14721,
            generation_requests=0,
            g_sq=0,
            g_sk=0,
            g_sqsq=0,
            g_sqsk=0,
            generation_sum=0,
            has_sqsk=True,
        ),
        {},
    ),
    (  # chunked prefill -> sk / sqsk diverge from fresh case
        "execute_14721_context_2(sq14721sk16221sqsq108745533sqsk120007533)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(
            batch_size=14721,
            context_requests=2,
            c_sq=14721,
            c_sk=16221,
            c_sqsq=108745533,
            c_sqsk=120007533,
            context_sum=14721,
            generation_requests=0,
            g_sq=0,
            g_sk=0,
            g_sqsq=0,
            g_sqsk=0,
            generation_sum=0,
            has_sqsk=True,
        ),
        {},
    ),
    (  # chunked 6 requests -> sqsq unchanged, sk / sqsk grow
        "execute_17408_context_6(sq17408sk52224sqsq59768832sqsk179306496)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(
            batch_size=17408,
            context_requests=6,
            c_sq=17408,
            c_sk=52224,
            c_sqsq=59768832,
            c_sqsk=179306496,
            context_sum=17408,
            generation_requests=0,
            g_sq=0,
            g_sk=0,
            g_sqsq=0,
            g_sqsk=0,
            generation_sum=0,
            has_sqsk=True,
        ),
        {},
    ),
    (  # plain decode: query_len == 1 -> g_sqsq == request count, g_sqsk == g_sk
        "execute_64_context_0(sq0sk0sqsq0sqsk0)"
        "_generation_64(sq64sk131072sqsq64sqsk131072)",
        "vllm_detailed",
        dict(
            batch_size=64,
            context_requests=0,
            context_sum=0,
            c_sq=0,
            c_sk=0,
            c_sqsq=0,
            c_sqsk=0,
            generation_requests=64,
            g_sq=64,
            g_sk=131072,
            g_sqsq=64,
            g_sqsk=131072,
            generation_sum=64,
            has_sqsk=True,
        ),
        {},
    ),
    (  # spec-decode / MTP decode: query_len == 4
        "execute_128_context_0(sq0sk0sqsq0sqsk0)"
        "_generation_32(sq128sk65536sqsq512sqsk262144)",
        "vllm_detailed",
        dict(
            batch_size=128,
            context_requests=0,
            context_sum=0,
            c_sq=0,
            c_sk=0,
            c_sqsq=0,
            c_sqsk=0,
            generation_requests=32,
            g_sq=128,
            g_sk=65536,
            g_sqsq=512,
            g_sqsk=262144,
            generation_sum=128,
            has_sqsk=True,
        ),
        {},
    ),
    (  # mixed: both groups populated
        "execute_6147_context_2(sq6144sk7144sqsq20971520sqsk23019520)"
        "_generation_3(sq3sk6144sqsq3sqsk6144)",
        "vllm_detailed",
        dict(
            batch_size=6147,
            context_requests=2,
            c_sq=6144,
            c_sk=7144,
            c_sqsq=20971520,
            c_sqsk=23019520,
            context_sum=6144,
            generation_requests=3,
            g_sq=3,
            g_sk=6144,
            g_sqsq=3,
            g_sqsk=6144,
            generation_sum=3,
            has_sqsk=True,
        ),
        {},
    ),
]

SGLANG_CASES = [
    # --- native (no roofline_annotations) ---
    (
        "step[EXTEND bs=2 toks=14721]",
        "sglang_native",
        dict(
            batch_size=14721,
            context_requests=2,
            context_sum=14721,
            c_sq=14721,
            has_sqsk=False,
        ),
        {},
    ),
    (
        "step[DECODE bs=64]",
        "sglang_native",
        dict(
            batch_size=64,
            generation_requests=64,
            generation_sum=64,
            g_sq=64,
            context_requests=0,
            has_sqsk=False,
        ),
        {},
    ),
    (  # no toks / sq data -> bs is the only usable batch_size proxy
        "step[MIXED bs=2]",
        "sglang_native",
        dict(
            batch_size=2,
            context_requests=2,
            has_sqsk=False,
        ),
        {},
    ),
    # --- detailed (roofline_annotations) ---
    (  # fresh EXTEND -> c_sqsk == c_sqsq
        "step[EXTEND bs=2 toks=14721 c_sq=14721 c_sqsq=108745533 "
        "c_sqsk=108745533 c_sk=14721]",
        "sglang_detailed",
        dict(
            batch_size=14721,
            context_requests=2,
            context_sum=14721,
            c_sq=14721,
            c_sk=14721,
            c_sqsq=108745533,
            c_sqsk=108745533,
            has_sqsk=True,
        ),
        {},
    ),
    (  # chunked EXTEND
        "step[EXTEND bs=2 toks=14721 c_sq=14721 c_sqsq=108745533 "
        "c_sqsk=120007533 c_sk=16221]",
        "sglang_detailed",
        dict(
            batch_size=14721,
            context_requests=2,
            c_sq=14721,
            c_sk=16221,
            c_sqsq=108745533,
            c_sqsk=120007533,
            has_sqsk=True,
        ),
        {},
    ),
    (  # 6 fresh EXTEND
        "step[EXTEND bs=6 toks=17408 c_sq=17408 c_sqsq=59768832 "
        "c_sqsk=59768832 c_sk=17408]",
        "sglang_detailed",
        dict(
            batch_size=17408,
            context_requests=6,
            c_sq=17408,
            c_sk=17408,
            c_sqsq=59768832,
            c_sqsk=59768832,
            has_sqsk=True,
        ),
        {},
    ),
    (  # plain DECODE -> g_sq == bs
        "step[DECODE bs=64 g_sq=64 g_sqsq=64 g_sqsk=131072 g_sk=131072]",
        "sglang_detailed",
        dict(
            batch_size=64,
            generation_requests=64,
            generation_sum=64,
            g_sq=64,
            g_sk=131072,
            g_sqsq=64,
            g_sqsk=131072,
            has_sqsk=True,
        ),
        {},
    ),
    (  # MTP DECODE -> g_sq (128) != bs (64), so batch_size counts tokens
        "step[DECODE bs=64 g_sq=128 g_sqsq=256 g_sqsk=262144 g_sk=131072]",
        "sglang_detailed",
        dict(
            batch_size=128,
            generation_requests=64,
            generation_sum=128,
            g_sq=128,
            g_sk=131072,
            g_sqsq=256,
            g_sqsk=262144,
            has_sqsk=True,
        ),
        {},
    ),
    (  # MIXED -> c=/g= are per-group request counts
        "step[MIXED bs=2 c=1 g=1 c_sq=5 c_sk=8 c_sqsq=25 c_sqsk=40 "
        "g_sq=1 g_sk=12 g_sqsq=1 g_sqsk=12]",
        "sglang_detailed",
        dict(
            batch_size=6,
            context_requests=1,
            generation_requests=1,
            c_sq=5,
            c_sk=8,
            c_sqsq=25,
            c_sqsk=40,
            g_sq=1,
            g_sk=12,
            g_sqsq=1,
            g_sqsk=12,
            context_sum=5,
            generation_sum=1,
            has_sqsk=True,
        ),
        {},
    ),
    (  # MIXED with multiple chunks + decodes
        "step[MIXED bs=5 c=2 g=3 c_sq=6144 c_sk=7144 c_sqsq=20971520 "
        "c_sqsk=23019520 g_sq=3 g_sk=6144 g_sqsq=3 g_sqsk=6144]",
        "sglang_detailed",
        dict(
            batch_size=6147,
            context_requests=2,
            generation_requests=3,
            c_sq=6144,
            c_sk=7144,
            c_sqsq=20971520,
            c_sqsk=23019520,
            g_sq=3,
            g_sk=6144,
            g_sqsq=3,
            g_sqsk=6144,
            has_sqsk=True,
        ),
        {},
    ),
    (  # MIXED all-context edge case (zeros for empty group)
        "step[MIXED bs=1 c=1 g=0 c_sq=3 c_sk=3 c_sqsq=9 c_sqsk=9 "
        "g_sq=0 g_sk=0 g_sqsq=0 g_sqsk=0]",
        "sglang_detailed",
        dict(
            batch_size=3,
            context_requests=1,
            generation_requests=0,
            c_sq=3,
            c_sk=3,
            c_sqsq=9,
            c_sqsk=9,
            g_sq=0,
            has_sqsk=True,
        ),
        {},
    ),
]

ATOM_CASES = [
    # --- prefill: native (no detailed suffix) vs detailed (has_sqsk reflects fields) ---
    (
        "prefill[bs=2 tok=14721 ctx=[7803, 6918]]",
        "atom_native",
        dict(
            batch_size=14721,
            context_requests=2,
            context_sum=14721,
            c_sq=14721,
            generation_requests=0,
            has_sqsk=False,
        ),
        dict(ctx="[7803, 6918]"),
    ),
    (  # fresh prefill detailed -> sqsk == sqsq
        "prefill[bs=2 tok=14721 ctx=[7803, 6918] sqsq=108745533 "
        "sqsk=108745533 sk=14721]",
        "atom_detailed",
        dict(
            batch_size=14721,
            context_requests=2,
            context_sum=14721,
            c_sq=14721,
            c_sk=14721,
            c_sqsq=108745533,
            c_sqsk=108745533,
            has_sqsk=True,
        ),
        dict(ctx="[7803, 6918]"),
    ),
    (  # chunked prefill detailed -> sqsk diverges
        "prefill[bs=2 tok=14721 ctx=[7803, 6918] sqsq=108745533 "
        "sqsk=119025333 sk=16221]",
        "atom_detailed",
        dict(
            batch_size=14721,
            c_sqsq=108745533,
            c_sqsk=119025333,
            c_sk=16221,
            has_sqsk=True,
        ),
        {},
    ),
    (  # 6 requests, ctx truncated (>5)
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3]",
        "atom_native",
        dict(
            batch_size=17408,
            context_requests=6,
            context_sum=17408,
            c_sq=17408,
            has_sqsk=False,
        ),
        dict(ctx="[4096, 4096, 4096]...+3"),
    ),
    (  # 6 requests truncated, detailed
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3 sqsq=59768832 "
        "sqsk=59768832 sk=17408]",
        "atom_detailed",
        dict(
            batch_size=17408,
            context_requests=6,
            c_sq=17408,
            c_sqsq=59768832,
            c_sqsk=59768832,
            c_sk=17408,
            has_sqsk=True,
        ),
        dict(ctx="[4096, 4096, 4096]...+3"),
    ),
    (  # prefill with Two-Batch-Overlap -> meta tbo
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3 sqsq=59768832 "
        "sqsk=59768832 sk=17408 tbo=1]",
        "atom_detailed",
        dict(
            batch_size=17408,
            context_requests=6,
            context_sum=17408,
            c_sq=17408,
            c_sk=17408,
            c_sqsq=59768832,
            c_sqsk=59768832,
            has_sqsk=True,
        ),
        dict(tbo=True),
    ),
    # --- decode ---
    (
        "decode[bs=64 tok=64 d=64]",
        "atom_native",
        dict(
            batch_size=64,
            generation_requests=64,
            generation_sum=64,
            g_sq=64,
            context_requests=0,
            has_sqsk=False,
        ),
        dict(d=64),
    ),
    (  # decode detailed
        "decode[bs=64 tok=64 d=64 sqsq=64 sqsk=131072 sk=131072]",
        "atom_detailed",
        dict(
            batch_size=64,
            generation_requests=64,
            generation_sum=64,
            g_sq=64,
            g_sk=131072,
            g_sqsq=64,
            g_sqsk=131072,
            has_sqsk=True,
        ),
        dict(d=64),
    ),
    (  # CUDAGraph padding bs=117/128 -> real batch 117
        "decode[bs=117/128 tok=117 d=117]",
        "atom_native",
        dict(
            batch_size=117,
            generation_requests=117,
            generation_sum=117,
            g_sq=117,
            has_sqsk=False,
        ),
        dict(d=117),
    ),
    (  # padding, detailed
        "decode[bs=117/128 tok=117 d=117 sqsq=117 sqsk=239616 sk=239616]",
        "atom_detailed",
        dict(
            batch_size=117,
            generation_requests=117,
            g_sq=117,
            g_sk=239616,
            g_sqsq=117,
            g_sqsk=239616,
            has_sqsk=True,
        ),
        dict(d=117),
    ),
    (  # spec-decode / MTP, non-detailed -> meta spec
        "decode[bs=32 tok=128 d=32 spec=3]",
        "atom_native",
        dict(
            batch_size=128,
            generation_requests=32,
            generation_sum=128,
            g_sq=128,
            has_sqsk=False,
        ),
        dict(d=32, spec=3),
    ),
    (  # spec-decode / MTP, detailed
        "decode[bs=32 tok=128 d=32 spec=3 sqsq=512 sqsk=262144 sk=65536]",
        "atom_detailed",
        dict(
            batch_size=128,
            generation_requests=32,
            g_sq=128,
            g_sk=65536,
            g_sqsq=512,
            g_sqsk=262144,
            has_sqsk=True,
        ),
        dict(d=32, spec=3),
    ),
    (  # mixed batch on decode path with TBO -> meta p, d, tbo
        "decode[bs=128 tok=384 p=2 d=126 sqsq=132612 sqsk=1114112 sk=258048 tbo=1]",
        "atom_detailed",
        dict(
            batch_size=384,
            generation_requests=128,
            generation_sum=384,
            g_sq=384,
            g_sk=258048,
            g_sqsq=132612,
            g_sqsk=1114112,
            has_sqsk=True,
        ),
        dict(p=2, d=126, tbo=True),
    ),
]

ALL_ITERATION_CASES = VLLM_CASES + SGLANG_CASES + ATOM_CASES


def _check(name, kind, fields, meta):
    ann = IterationAnnotation(name)
    assert ann.matched, f"{name!r} did not match any format"
    assert ann.kind == kind, f"{name!r}: kind {ann.kind} != {kind}"
    for attr, expected in fields.items():
        got = getattr(ann, attr)
        assert got == expected, f"{name!r}: {attr} {got} != {expected}"
    for key, expected in meta.items():
        assert (
            ann.meta.get(key) == expected
        ), f"{name!r}: meta[{key}] {ann.meta.get(key)} != {expected}"
    return ann


@pytest.mark.parametrize("name,kind,fields,meta", VLLM_CASES)
def test_vllm_annotations(name, kind, fields, meta):
    _check(name, kind, fields, meta)


@pytest.mark.parametrize("name,kind,fields,meta", SGLANG_CASES)
def test_sglang_annotations(name, kind, fields, meta):
    _check(name, kind, fields, meta)


@pytest.mark.parametrize("name,kind,fields,meta", ATOM_CASES)
def test_atom_annotations(name, kind, fields, meta):
    _check(name, kind, fields, meta)


@pytest.mark.parametrize("name,kind,fields,meta", ALL_ITERATION_CASES)
def test_chunk_stats_gated_on_has_sqsk(name, kind, fields, meta):
    """chunk_stats() returns aggregates iff detailed fields are present."""
    ann = _check(name, kind, fields, meta)
    if ann.has_sqsk:
        stats = ann.chunk_stats()
        assert stats["c_sq"] == ann.c_sq and stats["g_sk"] == ann.g_sk
    else:
        with pytest.raises(NotImplementedError):
            ann.chunk_stats()


@pytest.mark.parametrize(
    "name",
    [
        "some_random_op",
        "aten::matmul",
        "ProfilerStep#42",
        "",
    ],
)
def test_non_matching_annotations(name):
    ann = IterationAnnotation(name)
    assert not ann.matched
    assert ann.kind is None


# --------------------------------------------------------------------------- #
# Detail dicts (iter_details / full_details)
# --------------------------------------------------------------------------- #

ITER_DETAIL_KEYS = {
    "batch_size",
    "num_requests",
    "context_requests",
    "context_sum",
    "generation_requests",
    "generation_sum",
}

FULL_DETAIL_KEYS = {
    "name",
    "context_requests",
    "generation_requests",
    "c_sq",
    "c_sk",
    "c_sqsq",
    "c_sqsk",
    "g_sq",
    "g_sk",
    "g_sqsq",
    "g_sqsk",
    "num_requests",
    "batch_size",
    "has_sqsk",
}


@pytest.mark.parametrize("name,kind,fields,meta", ALL_ITERATION_CASES)
def test_detail_dict_keys(name, kind, fields, meta):
    """Both shapes are consumed by key, so the key sets are part of the API."""
    ann = _check(name, kind, fields, meta)
    assert set(ann.iter_details()) == ITER_DETAIL_KEYS
    assert set(ann.full_details()) == FULL_DETAIL_KEYS


@pytest.mark.parametrize("name,kind,fields,meta", ALL_ITERATION_CASES)
def test_detail_dicts_track_parsed_fields(name, kind, fields, meta):
    ann = _check(name, kind, fields, meta)
    iter_d, full_d = ann.iter_details(), ann.full_details()
    # batch_size follows one rule for both shapes (meta override included).
    assert iter_d["batch_size"] == full_d["batch_size"] == ann.batch_size
    assert iter_d["num_requests"] == full_d["num_requests"] == ann.num_requests
    for key in ("context_requests", "generation_requests"):
        assert iter_d[key] == full_d[key] == getattr(ann, key)
    assert iter_d["context_sum"] == ann.context_sum
    assert iter_d["generation_sum"] == ann.generation_sum
    assert full_d["name"] == name
    assert full_d["has_sqsk"] == ann.has_sqsk


@pytest.mark.parametrize(
    "name,expected",
    [
        ("execute_context_2(14721)_generation_0(0)", 14721),
        # Spec-decode / MTP: tokens (128), not the 32/64 requests behind them.
        (
            "execute_128_context_0(sq0sk0sqsq0sqsk0)"
            "_generation_32(sq128sk65536sqsq512sqsk262144)",
            128,
        ),
        ("decode[bs=32 tok=128 d=32 spec=3]", 128),
        ("step[DECODE bs=64 g_sq=128 g_sqsq=256 g_sqsk=262144 g_sk=131072]", 128),
        # SGLang bs= counts requests, so it loses to the sq sums when present.
        (
            "step[MIXED bs=5 c=2 g=3 c_sq=6144 c_sk=7144 c_sqsq=20971520 "
            "c_sqsk=23019520 g_sq=3 g_sk=6144 g_sqsq=3 g_sqsk=6144]",
            6147,
        ),
        ("step[EXTEND bs=2 toks=14721]", 14721),
        # Only labels without any token counts fall back to bs=.
        ("step[MIXED bs=2]", 2),
    ],
)
def test_batch_size_counts_tokens(name, expected):
    ann = IterationAnnotation(name)
    assert ann.batch_size == expected
    assert ann.iter_details()["batch_size"] == expected
    assert ann.full_details()["batch_size"] == expected


def test_details_exact_shape_for_mixed_vllm():
    """Pins both dicts for one mixed step; batch_size (6147) spans both groups
    while context_sum (6144) covers only the context group."""
    name = (
        "execute_6147_context_2(sq6144sk7144sqsq20971520sqsk23019520)"
        "_generation_3(sq3sk6144sqsq3sqsk6144)"
    )
    ann = IterationAnnotation(name)
    assert ann.iter_details() == {
        "batch_size": 6147,
        "num_requests": 5,
        "context_requests": 2,
        "context_sum": 6144,
        "generation_requests": 3,
        "generation_sum": 3,
    }
    assert ann.full_details() == {
        "name": name,
        "context_requests": 2,
        "generation_requests": 3,
        "c_sq": 6144,
        "c_sk": 7144,
        "c_sqsq": 20971520,
        "c_sqsk": 23019520,
        "g_sq": 3,
        "g_sk": 6144,
        "g_sqsq": 3,
        "g_sqsk": 6144,
        "num_requests": 5,
        "batch_size": 6147,
        "has_sqsk": True,
    }


def test_iter_details_fallback_for_unmatched():
    """Unmatched annotations (e.g. generic diffusion) count as one decode step.

    ``full_details()`` reports no requests, since no parser claimed the name,
    but still carries the single-token default batch_size.
    """
    ann = IterationAnnotation("some_random_op")
    assert ann.batch_size == 1
    assert ann.iter_details() == {
        "batch_size": 1,
        "num_requests": 1,
        "context_requests": 0,
        "context_sum": 0,
        "generation_requests": 1,
        "generation_sum": 1,
    }
    full = ann.full_details()
    assert full["batch_size"] == 1
    assert full["num_requests"] == 0
    assert full["has_sqsk"] is False


# --------------------------------------------------------------------------- #
# Capture annotations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,batch_size,mode",
    [
        ("capture_128_decode", 128, "decode"),
        ("capture_1_prefill", 1, "prefill"),
        ("capture_256_mixed_prefill", 256, "mixed_prefill"),
    ],
)
def test_capture_annotations(name, batch_size, mode):
    ann = CaptureAnnotation(name)
    assert ann.matched
    assert ann.kind == "capture"
    assert ann.batch_size == batch_size
    assert ann.mode == mode


@pytest.mark.parametrize(
    "name", ["execute_context_2(10)_generation_0(0)", "not_capture"]
)
def test_capture_non_matching(name):
    ann = CaptureAnnotation(name)
    assert not ann.matched
    assert ann.batch_size is None


# --------------------------------------------------------------------------- #
# Event / root discovery
# --------------------------------------------------------------------------- #

DETAILED_DECODE = (
    "execute_64_context_0(sq0sk0sqsq0sqsk0)_generation_64(sq64sk131072sqsq64sqsk131072)"
)
DETAILED_MIXED = (
    "execute_6147_context_2(sq6144sk7144sqsq20971520sqsk23019520)"
    "_generation_3(sq3sk6144sqsq3sqsk6144)"
)
NATIVE_DECODE = "execute_context_0(0)_generation_64(64)"


def _event(name, ts=0, cat="user_annotation"):
    return {"name": name, "ts": ts, "cat": cat}


def test_find_events_by_patterns_filters_by_category_and_sorts_by_ts():
    events = [
        _event(NATIVE_DECODE, ts=30),
        _event(NATIVE_DECODE, ts=10),
        _event(NATIVE_DECODE, ts=20, cat="cpu_op"),
        _event("unrelated_op", ts=5),
    ]
    found = find_events_by_patterns(events, ITERATION_BACKUP_PATTERNS)
    assert [e["ts"] for e in found] == [10, 30]

    # cat=None keeps the cpu_op copy as well.
    found = find_events_by_patterns(events, ITERATION_BACKUP_PATTERNS, cat=None)
    assert [e["ts"] for e in found] == [10, 20, 30]


def test_find_events_by_patterns_returns_each_event_once():
    """A name matching two patterns in the same tier is not duplicated."""
    events = [_event(DETAILED_DECODE)]
    both = [ITERATION_PATTERNS[0], ITERATION_PATTERNS[0]]
    assert len(find_events_by_patterns(events, both)) == 1


def test_find_events_by_patterns_on_no_match():
    assert find_events_by_patterns([_event("unrelated")], ITERATION_PATTERNS) == []


def test_find_events_by_patterns_logging(capsys):
    events = [_event(NATIVE_DECODE), _event("unrelated")]
    find_events_by_patterns(events, ITERATION_BACKUP_PATTERNS)
    assert capsys.readouterr().out == ""

    find_events_by_patterns(events, ITERATION_BACKUP_PATTERNS, label="iteration")
    assert capsys.readouterr().out == "Found 1 iteration events\n"

    find_events_by_patterns(
        events, ITERATION_BACKUP_PATTERNS, label="iteration", verbose=True
    )
    assert capsys.readouterr().out == f"Found 1 iteration events\n{NATIVE_DECODE}\n"


def test_find_events_by_patterns_with_a_single_pattern():
    events = [
        _event(DETAILED_MIXED, ts=20),
        _event(DETAILED_DECODE, ts=10),
        _event(NATIVE_DECODE, ts=5),
        _event(DETAILED_DECODE, ts=1, cat="cpu_op"),
    ]
    roots = find_events_by_patterns(events, [ITERATION_PATTERNS[0]])
    assert [e["ts"] for e in roots] == [10, 20]


def test_find_iteration_roots_by_priority_prefers_the_detailed_tier():
    events = [
        _event(DETAILED_DECODE, ts=10),
        _event(NATIVE_DECODE, ts=20),
        _event("step[DECODE bs=64]", ts=30),
    ]
    roots = find_iteration_roots_by_priority(events)
    assert [e["name"] for e in roots] == [DETAILED_DECODE]


def test_find_iteration_roots_by_priority_falls_back_to_the_native_tier():
    events = [_event(NATIVE_DECODE, ts=20), _event("step[DECODE bs=64]", ts=10)]
    roots = find_iteration_roots_by_priority(events)
    assert [e["ts"] for e in roots] == [10, 20]


def test_find_iteration_roots_by_priority_on_no_match():
    assert find_iteration_roots_by_priority([_event("unrelated")]) == []
    # Detailed annotations are ignored when they carry the wrong category.
    wrong_cat = [_event(DETAILED_DECODE, cat="cpu_op")]
    assert find_iteration_roots_by_priority(wrong_cat) == []


def test_find_iteration_roots_by_priority_accepts_custom_tiers():
    events = [_event(DETAILED_DECODE, ts=10), _event(NATIVE_DECODE, ts=20)]
    roots = find_iteration_roots_by_priority(
        events, pattern_tiers=[ITERATION_BACKUP_PATTERNS]
    )
    assert [e["name"] for e in roots] == [NATIVE_DECODE]


# --------------------------------------------------------------------------- #
# Phase classification
# --------------------------------------------------------------------------- #

PREFILL_ONLY = {"context_requests": 2, "generation_requests": 0}
MIXED = {"context_requests": 2, "generation_requests": 8}
DECODE_ONLY = {"context_requests": 0, "generation_requests": 8}
IDLE = {"context_requests": 0, "generation_requests": 0}


@pytest.mark.parametrize(
    "detail,context,generation,prefill_only,mixed,decode_only,phase",
    [
        (PREFILL_ONLY, True, False, True, False, False, PHASE_PREFILL_ONLY),
        (MIXED, True, True, False, True, False, PHASE_PREFILLDECODE),
        (DECODE_ONLY, False, True, False, False, True, PHASE_DECODE_ONLY),
        (IDLE, False, False, False, False, False, None),
        ({}, False, False, False, False, False, None),
    ],
)
def test_phase_predicates(
    detail, context, generation, prefill_only, mixed, decode_only, phase
):
    assert has_context(detail) is context
    assert has_generation(detail) is generation
    assert is_prefill_only(detail) is prefill_only
    assert is_mixed(detail) is mixed
    assert is_decode_only(detail) is decode_only
    assert classify_phase(detail) == phase


# --------------------------------------------------------------------------- #
# Per-window aggregation
# --------------------------------------------------------------------------- #


def test_iteration_details_parses_both_shapes():
    roots = [{"name": DETAILED_MIXED}, {"name": DETAILED_DECODE}]

    brief = iteration_details(roots)
    assert [d["batch_size"] for d in brief] == [6147, 64]
    assert "c_sq" not in brief[0]

    full = iteration_details(roots, full=True)
    assert [d["batch_size"] for d in full] == [6147, 64]
    assert full[0]["c_sq"] == 6144
    assert full[0]["has_sqsk"] is True


def test_iteration_details_on_empty_input():
    assert iteration_details([]) == []


def test_average_detail():
    details = [{"batch_size": 10}, {"batch_size": 20}, {"batch_size": 30}]
    assert average_detail(details, "batch_size") == 20.0
    # Missing keys count as zero, and an empty window averages to zero.
    assert average_detail(details, "num_requests") == 0.0
    assert average_detail([], "batch_size") == 0.0


def test_find_phase_from_window_counts_each_phase_separately():
    """A mixed step counts as prefilldecode, never as prefill or decode."""
    details = [
        {**PREFILL_ONLY, "batch_size": 100, "num_requests": 2},
        {**MIXED, "batch_size": 200, "num_requests": 10},
        {**MIXED, "batch_size": 300, "num_requests": 10},
        {**DECODE_ONLY, "batch_size": 400, "num_requests": 8},
        {**IDLE, "batch_size": 0, "num_requests": 0},
    ]
    assert find_phase_from_window(details) == {
        "num_prefill": 1,
        "num_prefilldecode": 2,
        "num_decode": 1,
        "avg_bs": 200,  # int(1000 / 5)
        "avg_conc": 6,  # int(30 / 5)
    }


def test_find_phase_from_window_accepts_full_details():
    roots = [{"name": DETAILED_MIXED}, {"name": DETAILED_DECODE}]
    assert find_phase_from_window(iteration_details(roots, full=True)) == (
        find_phase_from_window(iteration_details(roots))
    )


def test_find_phase_from_window_on_empty_input():
    assert find_phase_from_window([]) == {
        "num_prefill": 0,
        "num_prefilldecode": 0,
        "num_decode": 0,
        "avg_bs": 0,
        "avg_conc": 0,
    }
