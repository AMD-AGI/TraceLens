###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom initializers for EventReplayer.

Operations captured by the PyTorch profiler have zeroed-out metadata tensors
(block tables, routing tensors, etc.) because the profiler records shapes and
dtypes but not tensor values.  Custom initializers fill these tensors with
realistic content so the GPU kernel exercises real memory-access and compute
patterns during replay benchmarking.

To add a custom initializer for a new op family:
  1. Subclass ``CustomInit``
  2. Set ``op_patterns`` to one or more substrings that match the op name
  3. Implement ``initialize()`` — mutate replayer.args / replayer.kwargs in-place
  4. Return a one-line summary string (printed by EventReplayer)
  5. Register with ``EventReplayer.register_custom_init(YourInit())``
     or add it to the ``_custom_init_registry`` default list.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass  # EventReplayer imported at runtime to avoid circular dep

# -- Batch context extraction from vLLM profiler annotations ---------------

_BATCH_ANNO_RE = re.compile(
    r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)"
)


def extract_batch_context(analyzer: Any) -> int:
    """Parse vLLM ``user_annotation`` events and attach batch context to ops.

    vLLM annotates each ``execute_model`` step with a ``user_annotation``
    event of the form ``execute_context_N(T)_generation_N(T)`` where
    *N* = number of sequences and *T* = total query tokens for that phase.

    This function:
      1. Collects all such annotations with their ``[ts, ts+dur]`` ranges.
      2. For every ``paged_attention`` cpu_op event, finds the enclosing
         annotation by timestamp and attaches a ``batch_context`` dict::

             event["batch_context"] = {
                 "n_prefill": 2,
                 "prefill_tokens": 18,
                 "n_decode": 2,
                 "decode_tokens": 2,
             }

    Args:
        analyzer: A ``TreePerfAnalyzer`` (or any object whose ``.tree.events``
            yields the trace event list).

    Returns:
        Number of paged_attention events that were annotated.
    """
    annotations = []
    for e in analyzer.tree.events:
        cat = e.get("cat") or ""
        if cat != "user_annotation":
            continue
        m = _BATCH_ANNO_RE.search(e.get("name", ""))
        if not m:
            continue
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        annotations.append({
            "ts": ts,
            "end": ts + dur,
            "n_prefill": int(m.group(1)),
            "prefill_tokens": int(m.group(2)),
            "n_decode": int(m.group(3)),
            "decode_tokens": int(m.group(4)),
        })

    if not annotations:
        return 0

    annotations.sort(key=lambda a: a["ts"])

    annotated = 0
    for e in analyzer.tree.events:
        name = e.get("name", "")
        if "paged_attention" not in name:
            continue
        if not e.get("args", {}).get("Input Dims"):
            continue
        ts = e.get("ts", 0)
        for a in annotations:
            if a["ts"] <= ts <= a["end"]:
                e["batch_context"] = {
                    "n_prefill": a["n_prefill"],
                    "prefill_tokens": a["prefill_tokens"],
                    "n_decode": a["n_decode"],
                    "decode_tokens": a["decode_tokens"],
                }
                annotated += 1
                break

    return annotated


class CustomInit(ABC):
    """Base class for tensor initializers applied before replay."""

    op_patterns: List[str] = []

    def applies_to(self, replayer: Any) -> bool:
        op_name = replayer.event.get("name", "")
        return op_name in self.op_patterns

    @abstractmethod
    def initialize(self, replayer: Any, **kwargs) -> Optional[str]:
        """Mutate replayer.args/kwargs in-place.  Return a summary string."""
        ...


class PagedAttentionInit(CustomInit):
    """Initialize block_tables, seq_lens, and query_start_loc for paged attention.

    When ``batch_context`` is present on the event (attached by
    :func:`extract_batch_context`), uses the exact prefill/decode split from
    vLLM's profiler annotations.  Otherwise falls back to heuristics:
      - query_tokens == num_seqs  →  decode (1 token/seq)
      - query_tokens > num_seqs   →  prefill (tokens distributed uniformly)

    In all cases:
      - ``seq_lens`` set to ``max_seq_len`` for every sequence.
      - Block table entries drawn from a random permutation of the pool.
    """

    op_patterns = ["_rocm_C::paged_attention"]

    def initialize(self, replayer: Any, **kwargs) -> Optional[str]:
        try:
            import numpy as np
        except ImportError:
            return "[custom init] PagedAttentionInit skipped — numpy not available"

        args = replayer.args
        op_name = replayer.event.get("name", "")

        ir = replayer.event_replay_IR
        arg_names = [a["arg_name"] for a in ir["list_pos_args"]]
        def _by_name_or_pos(name, pos):
            if name in arg_names:
                return args[arg_names.index(name)]
            return args[pos]

        block_tables = _by_name_or_pos("block_tables", 9)
        seq_lens = _by_name_or_pos("seq_lens", 10)
        key_cache = _by_name_or_pos("key_cache", 5)
        block_size = int(_by_name_or_pos("block_size", 12))
        max_seq_len = int(_by_name_or_pos("max_seq_len", 13))

        num_seqs = block_tables.shape[0]
        max_blocks_per_seq = block_tables.shape[1]
        num_blocks_total = key_cache.shape[0]

        query = _by_name_or_pos("query", 4)
        num_query_tokens = query.shape[0]

        rng = np.random.default_rng(42)

        # -- Determine per-sequence query token counts -------------------------
        batch_ctx = replayer.event.get("batch_context")
        if batch_ctx is not None:
            n_pf = batch_ctx["n_prefill"]
            pf_tok = batch_ctx["prefill_tokens"]
            n_dec = batch_ctx["n_decode"]
            dec_tok = batch_ctx["decode_tokens"]

            per_seq_queries = []
            if n_pf > 0:
                base_pf = pf_tok // n_pf
                rem_pf = pf_tok % n_pf
                for s in range(n_pf):
                    per_seq_queries.append(base_pf + (1 if s < rem_pf else 0))
            for _ in range(n_dec):
                per_seq_queries.append(1)

            if len(per_seq_queries) != num_seqs:
                per_seq_queries = per_seq_queries[:num_seqs]
                while len(per_seq_queries) < num_seqs:
                    per_seq_queries.append(1)

            phase = ("mixed" if n_pf > 0 and n_dec > 0
                     else "prefill" if n_pf > 0 else "decode")
            source = "annotation"
        else:
            tokens_per_seq = num_query_tokens / num_seqs if num_seqs else 1
            if tokens_per_seq > 1:
                base_q = num_query_tokens // num_seqs
                rem_q = num_query_tokens % num_seqs
                per_seq_queries = [base_q + (1 if s < rem_q else 0)
                                   for s in range(num_seqs)]
                phase = "prefill"
            else:
                per_seq_queries = [1] * num_seqs
                phase = "decode"
            source = "heuristic"

        # -- seq_lens: max_seq_len for every sequence --------------------------
        lengths = np.full(num_seqs, max_seq_len, dtype=np.int32)

        # -- block_tables: permutation of physical block pool ------------------
        bt = np.zeros((num_seqs, max_blocks_per_seq), dtype=np.int32)
        all_block_ids = rng.permutation(num_blocks_total)
        block_cursor = 0
        for s in range(num_seqs):
            blocks_needed = (int(lengths[s]) + block_size - 1) // block_size
            blocks_needed = min(blocks_needed, max_blocks_per_seq)
            for b in range(blocks_needed):
                bt[s, b] = all_block_ids[block_cursor % num_blocks_total]
                block_cursor += 1

        import torch

        block_tables.copy_(torch.from_numpy(bt).to(block_tables.device))
        seq_lens.copy_(torch.from_numpy(lengths).to(seq_lens.device))

        # -- query_start_loc: CSR indptr encoding per-seq query counts ---------
        qsl = _by_name_or_pos("query_start_loc", 11)
        if (qsl is not None
                and hasattr(qsl, "shape")
                and qsl.numel() > 0):
            qloc = np.zeros(num_seqs + 1, dtype=np.int32)
            for s in range(num_seqs):
                qloc[s + 1] = qloc[s] + per_seq_queries[s]
            qloc = qloc[: qsl.numel()]
            qsl.copy_(torch.from_numpy(qloc).to(qsl.device))

        ctx_str = ""
        if batch_ctx is not None:
            ctx_str = (f"  Annotation: {batch_ctx['n_prefill']} prefill "
                       f"({batch_ctx['prefill_tokens']} tok) + "
                       f"{batch_ctx['n_decode']} decode "
                       f"({batch_ctx['decode_tokens']} tok).")

        return (
            f"[custom init] {op_name} — paged attention metadata: "
            f"phase={phase} ({source}), num_seqs={num_seqs}, "
            f"max_seq_len={max_seq_len}, block_size={block_size}, "
            f"num_blocks={num_blocks_total}, "
            f"max_blocks_per_seq={max_blocks_per_seq}.{ctx_str}"
        )


class MoeRoutingInit(CustomInit):
    """Initialize MoE routing tensors (sorted_token_ids, sorted_expert_ids,
    num_valid_ids) so the CK kernel processes real token-to-expert assignments
    instead of short-circuiting on num_valid_ids=0.

    Supported kwargs:
      moe_distribution: "uniform" (default) or "zipf"
      moe_zipf_s: Zipf exponent (default 1.2), only used with "zipf"

    Arg layout for aiter::ck_moe_stage1/2:
      [0]  hidden_states  [M, K]         bf16
      [1]  w1             [E, N, K]      bf16
      [2]  w2             [E, K2, N2]    bf16
      [3]  sorted_token_ids  [padded]    int32  <- init
      [4]  sorted_expert_ids [blocks+1]  int32  <- init
      [5]  num_valid_ids     [2]         int32  <- init
      [6]  output         [M, top_k, N2] bf16
      [7]  top_k          (scalar)
      ...
      [11] block_m        (scalar)
    """

    op_patterns = ["aiter::ck_moe_stage1", "aiter::ck_moe_stage2"]

    def initialize(self, replayer: Any, **kwargs) -> Optional[str]:
        try:
            import numpy as np
        except ImportError:
            return "[custom init] MoeRoutingInit skipped — numpy not available"

        distribution = kwargs.get("moe_distribution", "uniform")
        zipf_s = kwargs.get("moe_zipf_s", 1.2)

        args = replayer.args
        op_name = replayer.event.get("name", "")

        # Locate args by name from the IR when available, fall back to position
        ir = replayer.event_replay_IR
        arg_names = [a["arg_name"] for a in ir["list_pos_args"]]
        def _by_name_or_pos(name, pos):
            if name in arg_names:
                return args[arg_names.index(name)]
            return args[pos]

        sorted_token_ids = _by_name_or_pos("sorted_token_ids", 3)
        sorted_expert_ids = _by_name_or_pos("sorted_expert_ids", 4)
        num_valid_ids = _by_name_or_pos("num_valid_ids", 5)
        top_k = int(_by_name_or_pos("topk", 7))
        block_m_val = _by_name_or_pos("block_m", 11)
        block_m = int(block_m_val) if block_m_val is not None else 32

        hidden = _by_name_or_pos("hidden_states", 0)
        M = hidden.shape[0]
        w1 = _by_name_or_pos("w1", 1)
        E = w1.shape[0]
        num_tokens = M * top_k
        padded_total = sorted_token_ids.shape[0]
        num_blocks = sorted_expert_ids.shape[0]

        rng = np.random.default_rng(42)

        if distribution == "zipf":
            ranks = np.arange(1, E + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, zipf_s)
            probs = weights / weights.sum()
            expert_assignments = rng.choice(E, size=num_tokens, p=probs)
        else:
            expert_assignments = rng.integers(0, E, size=num_tokens)

        token_ids_list: list = []
        expert_ids_list: list = []
        for expert_id in range(E):
            tokens_for_expert = np.where(expert_assignments == expert_id)[0]
            count = len(tokens_for_expert)
            if count == 0:
                continue
            padded_count = ((count + block_m - 1) // block_m) * block_m
            n_blocks_for_expert = padded_count // block_m
            padded_tokens = np.full(padded_count, num_tokens, dtype=np.int32)
            padded_tokens[:count] = tokens_for_expert // top_k
            token_ids_list.append(padded_tokens)
            expert_ids_list.extend([expert_id] * n_blocks_for_expert)

        all_token_ids = (
            np.concatenate(token_ids_list)
            if token_ids_list
            else np.array([], dtype=np.int32)
        )

        if len(all_token_ids) < padded_total:
            padding = np.full(
                padded_total - len(all_token_ids), num_tokens, dtype=np.int32
            )
            all_token_ids = np.concatenate([all_token_ids, padding])
        else:
            all_token_ids = all_token_ids[:padded_total]

        all_expert_ids = np.array(expert_ids_list, dtype=np.int32)
        if len(all_expert_ids) < num_blocks:
            padding = np.zeros(num_blocks - len(all_expert_ids), dtype=np.int32)
            all_expert_ids = np.concatenate([all_expert_ids, padding])
        else:
            all_expert_ids = all_expert_ids[:num_blocks]

        import torch

        sorted_token_ids.copy_(
            torch.from_numpy(all_token_ids).to(sorted_token_ids.device)
        )
        sorted_expert_ids.copy_(
            torch.from_numpy(all_expert_ids).to(sorted_expert_ids.device)
        )

        valid_count = min(
            len(np.concatenate(token_ids_list)) if token_ids_list else 0,
            padded_total,
        )
        if num_valid_ids.numel() >= 1:
            num_valid_ids[0] = valid_count
        if num_valid_ids.numel() >= 2:
            num_valid_ids[1] = valid_count

        dist_label = f"zipf(s={zipf_s})" if distribution == "zipf" else "uniform"
        experts_active = len(set(expert_assignments.tolist()))
        return (
            f"[custom init] {op_name} — initialized MoE routing: "
            f"dist={dist_label}, M={M}, top_k={top_k}, E={E}, block_m={block_m}, "
            f"num_tokens={num_tokens}, active_experts={experts_active}/{E}, "
            f"valid_ids={valid_count}/{padded_total}, "
            f"blocks={len(expert_ids_list)}/{num_blocks}.  "
            f"Assumptions: {dist_label} expert distribution, deterministic seed."
        )
