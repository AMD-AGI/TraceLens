###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Central parsing for vLLM/SGLang/ATOM trace annotations.

Two families, each a class that is constructed from a raw annotation string
and populates its fields from whichever registered pattern matches:
- ``IterationAnnotation`` -> iteration/execution annotations
- ``CaptureAnnotation``   -> graph-capture annotations
"""

import re
from typing import Callable, List

# --- patterns --------------------------------------------------------------
# Each block lists matching annotations, prefill/decode/mixed where the format
# has all three. The detailed variants carry the sq/sk roofline aggregates and
# are supersets of their native counterparts; examples too long for one line
# continue on an indented line.

# execute_14721_context_2(sq14721sk14721sqsq108745533sqsk108745533)
#     _generation_0(sq0sk0sqsq0sqsk0)
# execute_64_context_0(sq0sk0sqsq0sqsk0)_generation_64(sq64sk131072sqsq64sqsk131072)
# execute_6147_context_2(sq6144sk7144sqsq20971520sqsk23019520)
#     _generation_3(sq3sk6144sqsq3sqsk6144)
VLLM_DETAILED_PATTERN = re.compile(
    r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
)
# execute_context_2(14721)_generation_0(0)
# execute_context_0(0)_generation_64(64)
# execute_context_2(6144)_generation_3(3)
VLLM_NATIVE_PATTERN = re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)")

# step[EXTEND bs=2 toks=14721 c_sq=14721 c_sqsq=108745533 c_sqsk=108745533 c_sk=14721]
# step[DECODE bs=64 g_sq=128 g_sqsq=256 g_sqsk=262144 g_sk=131072]  <- MTP: g_sq > bs
# step[MIXED bs=2 c=1 g=1 c_sq=5 c_sk=8 c_sqsq=25 c_sqsk=40
#     g_sq=1 g_sk=12 g_sqsq=1 g_sqsk=12]
SGLANG_DETAILED_PATTERN = re.compile(
    r"step\[(?:EXTEND|DECODE|MIXED)\b[^\]]*sqsq=\d+[^\]]*\]"
)
# step[EXTEND bs=2 toks=14721]
# step[DECODE bs=64]
# step[MIXED bs=2]  <- neither toks nor sq/sk, so bs is the only count
SGLANG_NATIVE_PATTERN = re.compile(r"step\[(?:EXTEND|DECODE|MIXED)\b.*\]")

# prefill[bs=2 tok=14721 ctx=[7803, 6918]]
# prefill[bs=6 tok=17408 ctx=[4096, 4096, 4096]...+3]  <- ctx truncated past 5
# decode[bs=64 tok=64 d=64]
# decode[bs=117/128 tok=117 d=117]  <- CUDAGraph padding, real batch is 117
ATOM_NATIVE_PATTERN = re.compile(r"^(prefill|decode)\[(.*)\]")
# prefill[bs=2 tok=14721 ctx=[7803, 6918] sqsq=108745533 sqsk=108745533 sk=14721]
# decode[bs=32 tok=128 d=32 spec=3 sqsq=512 sqsk=262144 sk=65536]
# decode[bs=128 tok=384 p=2 d=126 sqsq=132612 sqsk=1114112 sk=258048 tbo=1]
ATOM_DETAILED_PATTERN = re.compile(r"^(prefill|decode)\[.*sqsq=\d+.*\]")

# capture_128_decode
# capture_1_prefill
# capture_256_mixed_prefill
CAPTURE_PATTERN = re.compile(r"capture_(\d+)_(.*)")

# Root-discovery priority: PRIMARY (detailed) patterns are tried first and
# BACKUP (native) patterns are used only when no primary root is found.
ITERATION_PATTERNS = [
    VLLM_DETAILED_PATTERN,
    ATOM_DETAILED_PATTERN,
    SGLANG_DETAILED_PATTERN,
]
ITERATION_BACKUP_PATTERNS = [
    VLLM_NATIVE_PATTERN,
    SGLANG_NATIVE_PATTERN,
    ATOM_NATIVE_PATTERN,
]


def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _first_int(v) -> int:
    """First integer in a value (handles CUDAGraph padding, e.g. 'bs=117/128')."""
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else 0


def _tokenize(name: str) -> List[str]:
    """Flatten an execute_* annotation to a token list (parens dropped, sq/sk letters removed)."""
    return re.sub(r"[sqk]+", "_", name.replace("(", "_").replace(")", "_")).split("_")


# --- field parsers: each populates the instance and returns False if the
#     candidate pattern turned out not to be parseable ------------------------
def _fill_vllm_detailed(ann, name):
    p = _tokenize(name)
    ann.context_requests = _safe_int(p[3])
    ann.c_sq, ann.c_sk = _safe_int(p[5]), _safe_int(p[6])
    ann.c_sqsq, ann.c_sqsk = _safe_int(p[7]), _safe_int(p[8])
    ann.generation_requests = _safe_int(p[11])
    ann.g_sq, ann.g_sk = _safe_int(p[13]), _safe_int(p[14])
    ann.g_sqsq, ann.g_sqsk = _safe_int(p[15]), _safe_int(p[16])
    ann.context_sum, ann.generation_sum = ann.c_sq, ann.g_sq
    ann.batch_size = ann.c_sq + ann.g_sq
    ann.has_sqsk = True


def _fill_vllm_native(ann, name):
    p = _tokenize(name)
    ann.context_requests, ann.generation_requests = _safe_int(p[2]), _safe_int(p[6])
    ann.context_sum, ann.generation_sum = _safe_int(p[3]), _safe_int(p[7])
    ann.c_sq, ann.g_sq = ann.context_sum, ann.generation_sum
    ann.batch_size = ann.c_sq + ann.g_sq


def _fill_sglang_native(ann, name):
    """Base SGLang label with no roofline suffix (see SGLANG_annotation.md)."""
    m = re.match(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]", name)
    if not m:
        return False
    kind_word, bs = m.group(1), int(m.group(2))
    toks = int(m.group(3) or 0)
    if kind_word == "DECODE":
        ann.generation_requests = ann.generation_sum = ann.g_sq = bs
    else:  # EXTEND / MIXED treated as prefill; toks = total prompt tokens.
        ann.context_requests = bs
        ann.context_sum = ann.c_sq = toks
    ann.batch_size = ann.c_sq + ann.g_sq or bs


def _fill_sglang_detailed(ann, name):
    """SGLang label with roofline suffix (c_*/g_* aggregates; see SGLANG_annotation.md)."""
    m = SGLANG_DETAILED_PATTERN.match(name)
    if not m:
        return False
    body = name[name.index("[") + 1 : name.rindex("]")]
    mode = body.split()[0]
    kv = dict(re.findall(r"(\w+)=(\d+)", body))
    bs, toks = _safe_int(kv.get("bs", 0)), _safe_int(kv.get("toks", 0))
    ann.c_sq, ann.c_sk = _safe_int(kv.get("c_sq", 0)), _safe_int(kv.get("c_sk", 0))
    ann.c_sqsq, ann.c_sqsk = _safe_int(kv.get("c_sqsq", 0)), _safe_int(
        kv.get("c_sqsk", 0)
    )
    ann.g_sq, ann.g_sk = _safe_int(kv.get("g_sq", 0)), _safe_int(kv.get("g_sk", 0))
    ann.g_sqsq, ann.g_sqsk = _safe_int(kv.get("g_sqsq", 0)), _safe_int(
        kv.get("g_sqsk", 0)
    )
    if mode == "DECODE":
        ann.generation_requests, ann.generation_sum = bs, ann.g_sq
    elif mode == "EXTEND":
        ann.context_requests, ann.context_sum = bs, ann.c_sq
    else:  # MIXED: c=/g= are per-group request counts.
        ann.context_requests = _safe_int(kv.get("c", 0))
        ann.generation_requests = _safe_int(kv.get("g", 0))
        ann.context_sum, ann.generation_sum = ann.c_sq, ann.g_sq
    ann.batch_size = ann.c_sq + ann.g_sq or toks or bs
    ann.has_sqsk = True


def _fill_atom(ann, name):
    """ATOM prefill[...]/decode[...] labels (see ATOM_annotation.md).

    Shared by the native and detailed kinds; the sqsq/sqsk/sk fields are present
    only with ``ATOM_ENABLE_DETAILED_ANNOTATION=1``, and ``has_sqsk`` reflects that.
    """
    m = ATOM_NATIVE_PATTERN.match(name)
    if not m:
        return False
    phase, body = m.group(1), m.group(2)
    kv = dict(re.findall(r"(\w+)=(\[[^\]]*\](?:\.\.\.\+\d+)?|\S+)", body))
    bs = _first_int(kv.get("bs", 0))  # real batch; ignore CUDAGraph pad
    tokens = _safe_int(kv.get("tok", 0))
    sk, sqsq, sqsk = (
        _safe_int(kv.get("sk", 0)),
        _safe_int(kv.get("sqsq", 0)),
        _safe_int(kv.get("sqsk", 0)),
    )
    detailed = "sqsq" in kv and "sqsk" in kv

    if phase == "prefill":
        ann.context_requests, ann.context_sum = bs, tokens
        ann.c_sq, ann.c_sk, ann.c_sqsq, ann.c_sqsk = tokens, sk, sqsq, sqsk
        if "ctx" in kv:
            ann.meta["ctx"] = kv["ctx"]
    else:
        ann.generation_requests, ann.generation_sum = bs, tokens
        ann.g_sq, ann.g_sk, ann.g_sqsq, ann.g_sqsk = tokens, sk, sqsq, sqsk
        for k in ("p", "d", "spec"):
            if k in kv:
                ann.meta[k] = _safe_int(kv[k])
    if "tbo" in kv:
        ann.meta["tbo"] = True
    ann.batch_size = tokens  # tok= is already the whole batch dimension
    ann.has_sqsk = detailed


class IterationAnnotation:
    """Iteration/execution annotation.

    Formats are matched by **priority**: ``FORMATS`` is tried top-to-bottom and
    the first entry whose pattern matches. Order is therefore significant.
    """

    # (kind, pattern, parser); tried in priority order at construction time.
    # Each *_detailed entry must precede its *_native superset (see docstring).
    FORMATS: List[tuple] = [
        ("vllm_detailed", VLLM_DETAILED_PATTERN, _fill_vllm_detailed),
        ("vllm_native", VLLM_NATIVE_PATTERN, _fill_vllm_native),
        ("sglang_detailed", SGLANG_DETAILED_PATTERN, _fill_sglang_detailed),
        ("sglang_native", SGLANG_NATIVE_PATTERN, _fill_sglang_native),
        ("atom_detailed", ATOM_DETAILED_PATTERN, _fill_atom),
        ("atom_native", ATOM_NATIVE_PATTERN, _fill_atom),
    ]

    def __init__(self, annotation: str):
        self.name = annotation
        self.kind = None
        self.context_requests = 0
        self.generation_requests = 0
        self.context_sum = 0
        self.generation_sum = 0
        self.batch_size = 1
        self.c_sq = self.c_sk = self.c_sqsq = self.c_sqsk = 0
        self.g_sq = self.g_sk = self.g_sqsq = self.g_sqsk = 0
        self.has_sqsk = False
        self.meta = {}
        for kind, pattern, parser in self.FORMATS:
            if pattern.match(annotation) and parser(self, annotation) is not False:
                self.kind = kind
                break

    @classmethod
    def register_format(
        cls, kind: str, pattern: "re.Pattern", parser: Callable
    ) -> None:
        cls.FORMATS.append((kind, pattern, parser))

    @property
    def matched(self) -> bool:
        return self.kind is not None

    @property
    def num_requests(self) -> int:
        return self.context_requests + self.generation_requests

    def _details(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "num_requests": self.num_requests,
            "context_requests": self.context_requests,
            "context_sum": self.context_sum,
            "generation_requests": self.generation_requests,
            "generation_sum": self.generation_sum,
        }

    def iter_details(self) -> dict:
        """Request/token counts for steady-state and phase logic (legacy shape).

        A matched iteration annotation (vLLM / SGLang / ATOM) always yields its
        real per-request/token details; anything unmatched falls back to a
        single decode-equivalent step (e.g. generic diffusion workloads).
        """
        if not self.matched:
            # Generic workload (e.g. diffusion): one decode-equivalent step.
            return {
                "batch_size": 1,
                "num_requests": 1,
                "context_requests": 0,
                "context_sum": 0,
                "generation_requests": 1,
                "generation_sum": 1,
            }
        return self._details()

    def full_details(self) -> dict:
        """Full per-step detail incl. c_sq/c_sk/g_* aggregates (legacy shape)."""
        return {
            "name": self.name,
            "context_requests": self.context_requests,
            "generation_requests": self.generation_requests,
            "c_sq": self.c_sq,
            "c_sk": self.c_sk,
            "c_sqsq": self.c_sqsq,
            "c_sqsk": self.c_sqsk,
            "g_sq": self.g_sq,
            "g_sk": self.g_sk,
            "g_sqsq": self.g_sqsq,
            "g_sqsk": self.g_sqsk,
            "num_requests": self.num_requests,
            "batch_size": self.batch_size,
            "has_sqsk": self.has_sqsk,
        }

    def chunk_stats(self) -> dict:
        """context/generation sq-sk aggregates; requires a full sq/sk annotation."""
        if not self.has_sqsk:
            raise NotImplementedError(
                "attention without sq/sk annotation is not supported"
            )
        return {
            "c_sq": self.c_sq,
            "c_sk": self.c_sk,
            "c_sqsq": self.c_sqsq,
            "c_sqsk": self.c_sqsk,
            "g_sq": self.g_sq,
            "g_sk": self.g_sk,
            "g_sqsq": self.g_sqsq,
            "g_sqsk": self.g_sqsk,
        }


def _fill_capture(ann, name):
    m = CAPTURE_PATTERN.match(name)
    if not m:
        return False
    ann.batch_size = int(m.group(1))
    ann.mode = m.group(2)


class CaptureAnnotation:
    """Graph-capture annotation (``capture_{batch_size}_{mode}``)."""

    FORMATS: List[tuple] = [("capture", CAPTURE_PATTERN, _fill_capture)]

    def __init__(self, annotation: str):
        self.name = annotation
        self.kind = None
        self.batch_size = None
        self.mode = None
        for kind, pattern, parser in self.FORMATS:
            if pattern.match(annotation) and parser(self, annotation) is not False:
                self.kind = kind
                break

    @classmethod
    def register_format(
        cls, kind: str, pattern: "re.Pattern", parser: Callable
    ) -> None:
        cls.FORMATS.append((kind, pattern, parser))

    @property
    def matched(self) -> bool:
        return self.kind is not None


def annotation_str_from_event(event: dict) -> str:
    return event.get("annotation") or event.get("name", "")


def find_annotation_roots(events: List[dict], pattern: "re.Pattern") -> List[dict]:
    roots = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and pattern.match(e.get("name", ""))
    ]
    roots.sort(key=lambda x: x.get("ts", 0))
    return roots
