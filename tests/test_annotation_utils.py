###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceUtils.annotation_utils.

Covers every worked annotation example documented in the sibling references:
- ``VLLM_annotation.md``   -> vLLM native / detailed (``execute_...``)
- ``SGLANG_annotation.md`` -> SGLang native / detailed (``step[...]``)
- ``ATOM_annotation.md``   -> ATOM (``prefill[]`` / ``decode[]``)
plus capture annotations and non-matching input.

Each case asserts the parsed ``kind``, the populated ``IterationAnnotation``
fields, and (where relevant) ``meta`` and ``has_sqsk`` / ``chunk_stats()``.
"""

import pytest

from TraceLens.TraceUtils.annotation_utils import (
    CaptureAnnotation,
    IterationAnnotation,
)

# --------------------------------------------------------------------------- #
# Case tables: (name, kind, expected_fields, expected_meta)
# Only the listed fields/meta keys are asserted; others are ignored.
# --------------------------------------------------------------------------- #

VLLM_CASES = [
    # --- native (no detailed_trace_annotation) ---
    (
        "execute_context_2(14721)_generation_0(0)",
        "vllm_native",
        dict(context_requests=2, context_sum=14721, generation_requests=0,
             generation_sum=0, has_sqsk=False),
        {},
    ),
    (
        "execute_context_0(0)_generation_64(64)",
        "vllm_native",
        dict(context_requests=0, context_sum=0, generation_requests=64,
             generation_sum=64, has_sqsk=False),
        {},
    ),
    (  # spec-decode / MTP: generation_sum (128) exceeds request count (32)
        "execute_context_0(0)_generation_32(128)",
        "vllm_native",
        dict(generation_requests=32, generation_sum=128, has_sqsk=False),
        {},
    ),
    (  # mixed: both phases populated
        "execute_context_2(6144)_generation_3(3)",
        "vllm_native",
        dict(context_requests=2, context_sum=6144, generation_requests=3,
             generation_sum=3, has_sqsk=False),
        {},
    ),
    # --- detailed (detailed_trace_annotation) ---
    (  # fresh prefill -> sk == sq, sqsk == sqsq
        "execute_14721_context_2(sq14721sk14721sqsq108745533sqsk108745533)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(context_requests=2, c_sq=14721, c_sk=14721, c_sqsq=108745533,
             c_sqsk=108745533, context_sum=14721, generation_requests=0,
             g_sq=0, g_sk=0, g_sqsq=0, g_sqsk=0, generation_sum=0, has_sqsk=True),
        {},
    ),
    (  # chunked prefill -> sk / sqsk diverge from fresh case
        "execute_14721_context_2(sq14721sk16221sqsq108745533sqsk120007533)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(context_requests=2, c_sq=14721, c_sk=16221, c_sqsq=108745533,
             c_sqsk=120007533, has_sqsk=True),
        {},
    ),
    (  # 6 fresh requests
        "execute_17408_context_6(sq17408sk17408sqsq59768832sqsk59768832)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(context_requests=6, c_sq=17408, c_sk=17408, c_sqsq=59768832,
             c_sqsk=59768832, has_sqsk=True),
        {},
    ),
    (  # chunked 6 requests -> sqsq unchanged, sk / sqsk grow
        "execute_17408_context_6(sq17408sk52224sqsq59768832sqsk179306496)"
        "_generation_0(sq0sk0sqsq0sqsk0)",
        "vllm_detailed",
        dict(context_requests=6, c_sq=17408, c_sk=52224, c_sqsq=59768832,
             c_sqsk=179306496, has_sqsk=True),
        {},
    ),
    (  # plain decode: query_len == 1 -> g_sqsq == request count, g_sqsk == g_sk
        "execute_64_context_0(sq0sk0sqsq0sqsk0)"
        "_generation_64(sq64sk131072sqsq64sqsk131072)",
        "vllm_detailed",
        dict(generation_requests=64, g_sq=64, g_sk=131072, g_sqsq=64,
             g_sqsk=131072, generation_sum=64, context_requests=0, has_sqsk=True),
        {},
    ),
    (  # spec-decode / MTP decode: query_len == 4
        "execute_128_context_0(sq0sk0sqsq0sqsk0)"
        "_generation_32(sq128sk65536sqsq512sqsk262144)",
        "vllm_detailed",
        dict(generation_requests=32, g_sq=128, g_sk=65536, g_sqsq=512,
             g_sqsk=262144, generation_sum=128, has_sqsk=True),
        {},
    ),
    (  # mixed: both groups populated
        "execute_6147_context_2(sq6144sk7144sqsq20971520sqsk23019520)"
        "_generation_3(sq3sk6144sqsq3sqsk6144)",
        "vllm_detailed",
        dict(context_requests=2, c_sq=6144, c_sk=7144, c_sqsq=20971520,
             c_sqsk=23019520, generation_requests=3, g_sq=3, g_sk=6144,
             g_sqsq=3, g_sqsk=6144, has_sqsk=True),
        {},
    ),
]

SGLANG_CASES = [
    # --- native (no roofline_annotations) ---
    (
        "step[EXTEND bs=2 toks=14721]",
        "sglang_native",
        dict(context_requests=2, context_sum=14721, c_sq=14721,
             generation_requests=0, has_sqsk=False),
        dict(batch_size=14721),
    ),
    (
        "step[DECODE bs=64]",
        "sglang_native",
        dict(generation_requests=64, generation_sum=64, g_sq=64,
             context_requests=0, has_sqsk=False),
        dict(batch_size=64),
    ),
    (
        "step[MIXED bs=2]",
        "sglang_native",
        dict(context_requests=2, has_sqsk=False),
        dict(batch_size=2),
    ),
    # --- detailed (roofline_annotations) ---
    (  # fresh EXTEND -> c_sqsk == c_sqsq
        "step[EXTEND bs=2 toks=14721 c_sq=14721 c_sqsq=108745533 "
        "c_sqsk=108745533 c_sk=14721]",
        "sglang_detailed",
        dict(context_requests=2, context_sum=14721, c_sq=14721, c_sk=14721,
             c_sqsq=108745533, c_sqsk=108745533, has_sqsk=True),
        dict(batch_size=14721),
    ),
    (  # chunked EXTEND
        "step[EXTEND bs=2 toks=14721 c_sq=14721 c_sqsq=108745533 "
        "c_sqsk=120007533 c_sk=16221]",
        "sglang_detailed",
        dict(context_requests=2, c_sq=14721, c_sk=16221, c_sqsq=108745533,
             c_sqsk=120007533, has_sqsk=True),
        {},
    ),
    (  # 6 fresh EXTEND
        "step[EXTEND bs=6 toks=17408 c_sq=17408 c_sqsq=59768832 "
        "c_sqsk=59768832 c_sk=17408]",
        "sglang_detailed",
        dict(context_requests=6, c_sq=17408, c_sk=17408, c_sqsq=59768832,
             c_sqsk=59768832, has_sqsk=True),
        {},
    ),
    (  # plain DECODE -> g_sq == bs
        "step[DECODE bs=64 g_sq=64 g_sqsq=64 g_sqsk=131072 g_sk=131072]",
        "sglang_detailed",
        dict(generation_requests=64, generation_sum=64, g_sq=64, g_sk=131072,
             g_sqsq=64, g_sqsk=131072, has_sqsk=True),
        dict(batch_size=64),
    ),
    (  # MTP DECODE -> g_sq (128) != bs (64)
        "step[DECODE bs=64 g_sq=128 g_sqsq=256 g_sqsk=262144 g_sk=131072]",
        "sglang_detailed",
        dict(generation_requests=64, generation_sum=128, g_sq=128, g_sk=131072,
             g_sqsq=256, g_sqsk=262144, has_sqsk=True),
        dict(batch_size=64),
    ),
    (  # MIXED -> c=/g= are per-group request counts
        "step[MIXED bs=2 c=1 g=1 c_sq=5 c_sk=8 c_sqsq=25 c_sqsk=40 "
        "g_sq=1 g_sk=12 g_sqsq=1 g_sqsk=12]",
        "sglang_detailed",
        dict(context_requests=1, generation_requests=1, c_sq=5, c_sk=8,
             c_sqsq=25, c_sqsk=40, g_sq=1, g_sk=12, g_sqsq=1, g_sqsk=12,
             context_sum=5, generation_sum=1, has_sqsk=True),
        dict(batch_size=2),
    ),
    (  # MIXED with multiple chunks + decodes
        "step[MIXED bs=5 c=2 g=3 c_sq=6144 c_sk=7144 c_sqsq=20971520 "
        "c_sqsk=23019520 g_sq=3 g_sk=6144 g_sqsq=3 g_sqsk=6144]",
        "sglang_detailed",
        dict(context_requests=2, generation_requests=3, c_sq=6144, c_sk=7144,
             c_sqsq=20971520, c_sqsk=23019520, g_sq=3, g_sk=6144, g_sqsq=3,
             g_sqsk=6144, has_sqsk=True),
        dict(batch_size=5),
    ),
    (  # MIXED all-context edge case (zeros for empty group)
        "step[MIXED bs=1 c=1 g=0 c_sq=3 c_sk=3 c_sqsq=9 c_sqsk=9 "
        "g_sq=0 g_sk=0 g_sqsq=0 g_sqsk=0]",
        "sglang_detailed",
        dict(context_requests=1, generation_requests=0, c_sq=3, c_sk=3,
             c_sqsq=9, c_sqsk=9, g_sq=0, has_sqsk=True),
        dict(batch_size=1),
    ),
]

ATOM_CASES = [
    # --- prefill: native (no detailed suffix) vs detailed (has_sqsk reflects fields) ---
    (
        "prefill[bs=2 tok=14721 ctx=[7803, 6918]]",
        "atom_native",
        dict(context_requests=2, context_sum=14721, c_sq=14721,
             generation_requests=0, has_sqsk=False),
        dict(ctx="[7803, 6918]"),
    ),
    (  # fresh prefill detailed -> sqsk == sqsq
        "prefill[bs=2 tok=14721 ctx=[7803, 6918] sqsq=108745533 "
        "sqsk=108745533 sk=14721]",
        "atom_detailed",
        dict(context_requests=2, context_sum=14721, c_sq=14721, c_sk=14721,
             c_sqsq=108745533, c_sqsk=108745533, has_sqsk=True),
        dict(ctx="[7803, 6918]"),
    ),
    (  # chunked prefill detailed -> sqsk diverges
        "prefill[bs=2 tok=14721 ctx=[7803, 6918] sqsq=108745533 "
        "sqsk=119025333 sk=16221]",
        "atom_detailed",
        dict(c_sqsq=108745533, c_sqsk=119025333, c_sk=16221, has_sqsk=True),
        {},
    ),
    (  # 6 requests, ctx truncated (>5)
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3]",
        "atom_native",
        dict(context_requests=6, context_sum=17408, c_sq=17408, has_sqsk=False),
        dict(ctx="[4096, 4096, 4096]...+3"),
    ),
    (  # 6 requests truncated, detailed
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3 sqsq=59768832 "
        "sqsk=59768832 sk=17408]",
        "atom_detailed",
        dict(context_requests=6, c_sq=17408, c_sqsq=59768832, c_sqsk=59768832,
             c_sk=17408, has_sqsk=True),
        dict(ctx="[4096, 4096, 4096]...+3"),
    ),
    (  # prefill with Two-Batch-Overlap -> meta tbo
        "prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3 sqsq=59768832 "
        "sqsk=59768832 sk=17408 tbo=1]",
        "atom_detailed",
        dict(c_sqsq=59768832, c_sk=17408, has_sqsk=True),
        dict(tbo=True),
    ),
    # --- decode ---
    (
        "decode[bs=64 tok=64 d=64]",
        "atom_native",
        dict(generation_requests=64, generation_sum=64, g_sq=64,
             context_requests=0, has_sqsk=False),
        dict(d=64),
    ),
    (  # decode detailed
        "decode[bs=64 tok=64 d=64 sqsq=64 sqsk=131072 sk=131072]",
        "atom_detailed",
        dict(generation_requests=64, generation_sum=64, g_sq=64, g_sk=131072,
             g_sqsq=64, g_sqsk=131072, has_sqsk=True),
        dict(d=64),
    ),
    (  # CUDAGraph padding bs=117/128 -> real batch 117
        "decode[bs=117/128 tok=117 d=117]",
        "atom_native",
        dict(generation_requests=117, generation_sum=117, g_sq=117,
             has_sqsk=False),
        dict(d=117),
    ),
    (  # padding, detailed
        "decode[bs=117/128 tok=117 d=117 sqsq=117 sqsk=239616 sk=239616]",
        "atom_detailed",
        dict(generation_requests=117, g_sq=117, g_sk=239616, g_sqsq=117,
             g_sqsk=239616, has_sqsk=True),
        dict(d=117),
    ),
    (  # spec-decode / MTP, non-detailed -> meta spec
        "decode[bs=32 tok=128 d=32 spec=3]",
        "atom_native",
        dict(generation_requests=32, generation_sum=128, g_sq=128,
             has_sqsk=False),
        dict(d=32, spec=3),
    ),
    (  # spec-decode / MTP, detailed
        "decode[bs=32 tok=128 d=32 spec=3 sqsq=512 sqsk=262144 sk=65536]",
        "atom_detailed",
        dict(generation_requests=32, g_sq=128, g_sk=65536, g_sqsq=512,
             g_sqsk=262144, has_sqsk=True),
        dict(d=32, spec=3),
    ),
    (  # mixed batch on decode path with TBO -> meta p, d, tbo
        "decode[bs=128 tok=384 p=2 d=126 sqsq=132612 sqsk=1114112 "
        "sk=258048 tbo=1]",
        "atom_detailed",
        dict(generation_requests=128, generation_sum=384, g_sq=384, g_sk=258048,
             g_sqsq=132612, g_sqsk=1114112, has_sqsk=True),
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
        assert ann.meta.get(key) == expected, (
            f"{name!r}: meta[{key}] {ann.meta.get(key)} != {expected}"
        )
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


@pytest.mark.parametrize("name", ["execute_context_2(10)_generation_0(0)", "not_capture"])
def test_capture_non_matching(name):
    ann = CaptureAnnotation(name)
    assert not ann.matched
    assert ann.batch_size is None
