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
VLLM_DETAILED_PATTERN = re.compile(
    r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
)
VLLM_NATIVE_PATTERN = re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)")

SGLANG_DETAILED_PATTERN = re.compile(r"step\[(?:EXTEND|DECODE|MIXED)\b[^\]]*sqsq=\d+[^\]]*\]")
SGLANG_NATIVE_PATTERN = re.compile(r"step\[(?:EXTEND|DECODE|MIXED)\b.*\]")

ATOM_NATIVE_PATTERN = re.compile(r"^(prefill|decode)\[(.*)\]")
ATOM_DETAILED_PATTERN = re.compile(r"^(prefill|decode)\[.*sqsq=\d+.*\]")

CAPTURE_PATTERN = re.compile(r"capture_(\d+)_(.*)")

# Root-discovery priority: PRIMARY (detailed) patterns are tried first and
# BACKUP (native) patterns are used only when no primary root is found. 
ITERATION_PATTERNS = [VLLM_DETAILED_PATTERN, ATOM_DETAILED_PATTERN, SGLANG_DETAILED_PATTERN]
ITERATION_BACKUP_PATTERNS = [VLLM_NATIVE_PATTERN, SGLANG_NATIVE_PATTERN, ATOM_NATIVE_PATTERN]


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
    ann.has_sqsk = True


def _fill_vllm_native(ann, name):
    p = _tokenize(name)
    ann.context_requests, ann.generation_requests = _safe_int(p[2]), _safe_int(p[6])
    ann.context_sum, ann.generation_sum = _safe_int(p[3]), _safe_int(p[7])
    ann.c_sq, ann.g_sq = ann.context_sum, ann.generation_sum


def _fill_sglang_native(ann, name):
    """Base SGLang label with no roofline suffix (see SGLANG_annotation.md)."""
    m = re.match(r"step\[(\w+)\s+bs=(\d+)(?:\s+toks=(\d+))?\]", name)
    if not m:
        return False
    kind_word, bs = m.group(1), int(m.group(2))
    toks = int(m.group(3) or 0)
    if kind_word == "DECODE":
        ann.generation_requests = ann.generation_sum = ann.g_sq = bs
        ann.meta["batch_size"] = bs
    else:  # EXTEND / MIXED treated as prefill; toks = total prompt tokens.
        ann.context_requests = bs
        ann.context_sum = ann.c_sq = toks
        ann.meta["batch_size"] = toks or bs


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
    ann.c_sqsq, ann.c_sqsk = _safe_int(kv.get("c_sqsq", 0)), _safe_int(kv.get("c_sqsk", 0))
    ann.g_sq, ann.g_sk = _safe_int(kv.get("g_sq", 0)), _safe_int(kv.get("g_sk", 0))
    ann.g_sqsq, ann.g_sqsk = _safe_int(kv.get("g_sqsq", 0)), _safe_int(kv.get("g_sqsk", 0))
    if mode == "DECODE":
        ann.generation_requests, ann.generation_sum = bs, ann.g_sq
        ann.meta["batch_size"] = bs
    elif mode == "EXTEND":
        ann.context_requests, ann.context_sum = bs, ann.c_sq
        ann.meta["batch_size"] = toks or bs
    else:  # MIXED: c=/g= are per-group request counts.
        ann.context_requests = _safe_int(kv.get("c", 0))
        ann.generation_requests = _safe_int(kv.get("g", 0))
        ann.context_sum, ann.generation_sum = ann.c_sq, ann.g_sq
        ann.meta["batch_size"] = bs
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
    # Values may be bracketed lists with internal spaces (e.g. ctx=[a, b]...+N),
    # so match key=value pairs rather than splitting on whitespace.
    kv = dict(re.findall(r"(\w+)=(\[[^\]]*\](?:\.\.\.\+\d+)?|\S+)", body))
    bs = _first_int(kv.get("bs", 0))  # real batch; ignore CUDAGraph pad
    tokens = _safe_int(kv.get("tok", 0))
    sk, sqsq, sqsk = _safe_int(kv.get("sk", 0)), _safe_int(kv.get("sqsq", 0)), _safe_int(kv.get("sqsk", 0))
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
    ann.has_sqsk = detailed


class IterationAnnotation:
    """Iteration/execution annotation.

    Formats are matched by **priority**: ``FORMATS`` is tried top-to-bottom and
    the first entry whose pattern matches *and* whose parser does not return
    ``False`` wins (``kind`` is set and matching stops). Order is therefore
    significant, because some patterns are supersets of others:

    - ``SGLANG_NATIVE_PATTERN`` / ``ATOM_NATIVE_PATTERN`` (the base labels) also
      match their own *detailed* labels, so each ``*_detailed`` entry MUST be
      listed **before** its ``*_native`` counterpart. Otherwise a detailed label
      would be misclassified as native and its sqsq/sqsk/sk fields dropped.
    - vLLM detailed vs native are mutually exclusive under ``re.match`` (native
      anchors ``execute_context_``, detailed has ``execute_<n>_context_``), but
      detailed is still listed first for consistency.

    ``register_format`` appends to the end (lowest priority); insert into
    ``FORMATS`` manually if a custom format must take precedence.
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

    @property
    def batch_size(self) -> int:
        return self.meta.get("batch_size", self.c_sq + self.g_sq)

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
            "batch_size": self.c_sq + self.g_sq,
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


# --- event helpers ---------------------------------------------------------
def annotation_str_from_event(event: dict) -> str:
    return event.get("annotation") or event.get("name", "")


def find_annotation_roots(
    events: List[dict], pattern: "re.Pattern"
) -> List[dict]:
    roots = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and pattern.match(e.get("name", ""))
    ]
    roots.sort(key=lambda x: x.get("ts", 0))
    return roots
