###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for pseudo-op extensions.
"""

from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.utils import (
    dense_counted_tensor,
    name2bpe,
    torch_dtype_map,
)
from TraceLens.TraceUtils.utils.annotation_utils import (
    IterationAnnotation,
    add_logical_batch_size,
)
import math


class InferenceAttention:
    """
    Base class for Unified Attention operations.

    Unified Attention operations combine the entire attention computation
    (query, key, value, output) into a single kernel launch.

    Subclasses only need to implement ``get_param_details`` to extract
    parameters from their framework-specific event format. The returned
    dict must contain at least the keys listed in ``REQUIRED_PARAM_KEYS``.

    **No perf estimate:** If parsing fails, :meth:`get_param_details` returns
    :meth:`no_perf_param_details` (includes ``_no_perf: True``). In that case
    :meth:`flops`, :meth:`bytes`, and :meth:`get_compute_precision` return
    ``None``. Subclasses that extend the parent dict should return early when
    ``params.get("_no_perf")`` is true.
    """

    category = "InferenceAttention"
    bwd_category = None

    REQUIRED_PARAM_KEYS = (
        "B",
        "N_Q",
        "H_Q",
        "N_KV",
        "H_KV",
        "d_h_qk",
        "d_h_v",
        "c_sq",
        "c_sk",
        "c_sqsq",
        "c_sqsk",
        "g_sq",
        "g_sk",
        "g_sqsq",
        "g_sqsk",
    )

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)
        self.B = self.param_details["B"]
        self.N_Q = self.param_details["N_Q"]
        self.H_Q = self.param_details["H_Q"]
        self.N_KV = self.param_details["N_KV"]
        self.H_KV = self.param_details["H_KV"]
        self.d_h_qk = self.param_details["d_h_qk"]
        self.d_h_v = self.param_details["d_h_v"]

    @staticmethod
    def no_perf_param_details():
        """Placeholder params when annotation/inputs cannot be parsed; disables FLOPs/bytes."""
        return {
            "B": 1,
            "N_Q": 0,
            "H_Q": 0,
            "N_KV": 0,
            "H_KV": 0,
            "d_h_qk": 0,
            "d_h_v": 0,
            "c_sq": 0,
            "c_sk": 0,
            "c_sqsq": 0,
            "c_sqsk": 0,
            "g_sq": 0,
            "g_sk": 0,
            "g_sqsq": 0,
            "g_sqsk": 0,
            "dropout": 0.0,
            "causal": False,
            "flash_impl": True,
            "dtype_Q": None,
            "_no_perf": True,
        }

    @staticmethod
    def _parse_chunk_stats(annotation):
        """context/generation sq-sk aggregates; raises if not a full sq/sk annotation."""
        return IterationAnnotation(annotation).chunk_stats()

    @staticmethod
    def get_param_details(event):
        try:
            annotation = str(event.get("annotation"))
            stats = InferenceAttention._parse_chunk_stats(annotation)
            c_sq = stats["c_sq"]
            c_sk = stats["c_sk"]
            c_sqsq = stats["c_sqsq"]
            c_sqsk = stats["c_sqsk"]
            g_sq = stats["g_sq"]
            g_sk = stats["g_sk"]
            g_sqsq = stats["g_sqsq"]
            g_sqsk = stats["g_sqsk"]

            input_dims = event["args"]["Input Dims"]
            q_shape, k_shape = input_dims[0], input_dims[1]
            N_Q, H_Q, d_h_qk = q_shape
            N_KV, H_KV, d_h_v = k_shape[-3:]
            input_types = event["args"]["Input type"]
            dtype_Q = input_types[0]

            propagated_kv = (event.get("attention_perf_meta") or {}).get(
                "k_cache_dtype"
            )
            dtype_KV = (
                propagated_kv
                if propagated_kv
                else (input_types[1] if len(input_types) > 1 else dtype_Q)
            )
            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H_Q,
                "H_V": H_KV,
                "H_K": H_KV,
                "N_KV": N_KV,
                "H_KV": H_KV,
                "d_h_qk": d_h_qk,
                "d_h_v": d_h_v,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                "c_sq": c_sq,
                "c_sk": c_sk,
                "c_sqsq": c_sqsq,
                "c_sqsk": c_sqsk,
                "g_sq": g_sq,
                "g_sk": g_sk,
                "g_sqsq": g_sqsq,
                "g_sqsk": g_sqsk,
                "dtype_Q": dtype_Q,
                "dtype_KV": dtype_KV,
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()

    # ------------------------------------------------------------------
    # Static helpers – reusable across subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def flops_func(H_Q, d_h_qk, d_h_v, c_sqsk, c_sqsq, g_sqsk):
        """Calculate attention FLOPs for chunked-prefill + generation.

        Considers non-causal attention between the current and previous
        chunks and causal attention within the current chunk.

        Args:
            H_Q: Number of query heads.
            d_h_qk: Head dimension for Q/K.
            d_h_v: Head dimension for V.
            c_sqsk: Sum-of-products sq*sk across context requests.
            c_sqsq: Sum-of-products sq*sq across context requests (causal part).
            g_sqsk: Sum-of-products sq*sk across generation requests.
        """
        ctx_flops_qk = H_Q * (2 * c_sqsk * d_h_qk)
        ctx_flops_pv = H_Q * (2 * c_sqsk * d_h_v)
        ctx_flops_qk -= H_Q * (1 * c_sqsq * d_h_qk)
        ctx_flops_pv -= H_Q * (1 * c_sqsq * d_h_v)

        gen_flops_qk = H_Q * (2 * g_sqsk * d_h_qk)
        gen_flops_pv = H_Q * (2 * g_sqsk * d_h_v)

        return ctx_flops_qk + ctx_flops_pv + gen_flops_qk + gen_flops_pv

    @staticmethod
    def bytes_func(
        B,
        H_Q,
        H_KV,
        d_h_qk,
        d_h_v,
        c_sq,
        c_sk,
        g_sq,
        g_sk,
        bytes_per_element,
        bytes_per_element_KV=None,
    ):
        """Calculate bytes moved for attention (context + generation).

        Args:
            B: Batch size.
            H_Q / H_KV: Number of query / key-value heads.
            d_h_qk / d_h_v: Head dimensions for Q-K / V.
            c_sq / c_sk: Aggregate sequence lengths for context Q / KV.
            g_sq / g_sk: Aggregate sequence lengths for generation Q / KV.
            bytes_per_element: Bytes per Q / output / current-chunk K-V element.
            bytes_per_element_KV: Bytes per cached K / V element. Defaults to
                ``bytes_per_element`` (e.g. when KV-cache dtype matches Q).
        """
        if bytes_per_element_KV is None:
            bytes_per_element_KV = bytes_per_element
        ctx_elems_q = (
            B * c_sq * H_Q * d_h_qk  # Q read
            + B * c_sq * H_Q * d_h_v  # output write
            + B * c_sq * H_KV * d_h_qk  # K read (current chunk, Q-dtype)
            + B * c_sq * H_KV * d_h_v  # V read (current chunk, Q-dtype)
        )
        ctx_elems_kv = (
            B * (c_sk - c_sq) * H_KV * d_h_qk  # K read (cached, KV-dtype)
            + B * (c_sk - c_sq) * H_KV * d_h_v  # V read (cached, KV-dtype)
        )
        gen_elems_q = (
            B * g_sq * H_Q * d_h_qk
            + B * g_sq * H_Q * d_h_v
            + B * g_sq * H_KV * d_h_qk  # K read (current token, Q-dtype)
            + B * g_sq * H_KV * d_h_v  # V read (current token, Q-dtype)
        )
        gen_elems_kv = (
            B * (g_sk - g_sq) * H_KV * d_h_qk  # K read (cached, KV-dtype)
            + B * (g_sk - g_sq) * H_KV * d_h_v  # V read (cached, KV-dtype)
        )

        return (ctx_elems_q + gen_elems_q) * bytes_per_element + (
            ctx_elems_kv + gen_elems_kv
        ) * bytes_per_element_KV

    # ------------------------------------------------------------------
    # Instance methods – work for any subclass with valid param_details
    # ------------------------------------------------------------------

    def _cap_gen_kv(self, value):
        """Cap a generation KV aggregate to the sliding window.

        Opt-in via ``param_details["sliding_window"]`` (W). Generation (decode)
        tokens have ``sq = 1``, so each attends to ``min(ctx_i, W)`` KV tokens.
        With only aggregates available (``g_sk = Σ ctx``, ``g_sq = #requests``),
        ``Σ min(ctx_i, W)`` is approximated by ``min(value, g_sq * W)`` -- exact
        when all contexts are on one side of W (typical in steady-state decode),
        an upper bound otherwise. No-op when ``sliding_window`` is unset or <= 0,
        so non-windowed attention models are unaffected.
        """
        w = self.param_details.get("sliding_window")
        if w and w > 0 and self.param_details.get("g_sq"):
            return min(value, self.param_details["g_sq"] * w)
        return value

    def flops(self):
        if self.param_details.get("_no_perf"):
            return None
        if self.param_details["c_sq"] == 0 and self.param_details["g_sq"] == 0:
            raise NotImplementedError(
                "Attention perf model for decode phase requires custom annotations"
            )
        return self.flops_func(
            self.H_Q,
            self.d_h_qk,
            self.d_h_v,
            self.param_details["c_sqsk"],
            self.param_details["c_sqsq"],
            self._cap_gen_kv(self.param_details["g_sqsk"]),
        )

    def bytes(self, bytes_per_element=None):
        if self.param_details.get("_no_perf"):
            return None
        bpe = bytes_per_element
        if bpe is None:
            bpe = name2bpe(self.param_details.get("dtype_Q")) or 2
        dtype_kv = self.param_details.get("dtype_KV")
        bpe_kv = (name2bpe(dtype_kv) if dtype_kv else None) or bpe
        return self.bytes_func(
            self.B,
            self.H_Q,
            self.H_KV,
            self.d_h_qk,
            self.d_h_v,
            self.param_details["c_sq"],
            self.param_details["c_sk"],
            self.param_details["g_sq"],
            self._cap_gen_kv(self.param_details["g_sk"]),
            bpe,
            bpe_kv,
        )

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for attention is not defined.")

    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for attention is not defined.")

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        dtype = self.param_details.get("dtype_Q", None)
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "matrix"


class _V4PagedDecodeBase(InferenceAttention):
    """Base perf model for the DeepSeek-V4 sparse paged-decode pseudo ops.

    Decode-only kernel: context aggregates (``c_*``) are zeroed so only the
    generation (``g_*``) terms remain, and KV cache is modeled as bf16. Per-mode KV span (``sliding_window``) is set by ``MODE``:
    ``swa`` = window, ``csa`` = window + min(ctx / CSA stride, index_topk)
    compressed slots, ``hca`` = window + hierarchical-compressed history.
    """

    MODE = "swa"

    CSA_COMPRESS_STRIDE = 4
    HCA_COMPRESS_STRIDE = 128

    # Hardcoded per-variant scalars not recoverable from a single decode op:
    #   index_topk = sparse indexer budget, window = sliding-window size,
    #   n_heads = global query-head count (H_Q fallback = n_heads // tp),
    #   d_h = head dim fallback.
    _CONFIGS = {
        "DeepSeek-V4-Pro": {
            "index_topk": 1024,
            "window": 128,
            "n_heads": 128,
            "d_h": 512,
        },
        "DeepSeek-V4-Flash": {
            "index_topk": 512,
            "window": 128,
            "n_heads": 64,
            "d_h": 512,
        },
    }

    @classmethod
    def _config_for(cls, model_name):
        """Config for a DeepSeek-V4 variant (substring match, Flash fallback)."""
        if model_name:
            for key, cfg in cls._CONFIGS.items():
                if key.lower() in str(model_name).lower():
                    return cfg
        return cls._CONFIGS["DeepSeek-V4-Flash"]

    @classmethod
    def get_param_details(cls, event):
        try:
            args = event.get("args") or {}
            annotation = str(event.get("annotation"))
            stats = InferenceAttention._parse_chunk_stats(annotation)

            # Decode-only kernel: drop any context (prefill) aggregates
            stats["c_sq"] = stats["c_sk"] = stats["c_sqsq"] = stats["c_sqsk"] = 0

            model_name = args.get("v4_model_name")
            cfg = cls._config_for(model_name)

            H_Q = args.get("v4_H_Q")
            if not H_Q:
                tp = args.get("v4_tp") or 1
                H_Q = max(1, cfg["n_heads"] // int(tp))
            d_h = args.get("v4_d_h") or cfg["d_h"]
            H_Q = int(H_Q)
            d_h = int(d_h)

            N_Q = stats["g_sq"]
            dims = args.get("Input Dims") or []
            if dims and dims[0] and len(dims[0]) >= 1:
                N_Q = dims[0][0] or N_Q

            window = int(args.get("v4_window") or cfg["window"])
            index_topk = int(args.get("v4_index_topk") or cfg["index_topk"])
            sliding_window = cls._sliding_window(stats, window, index_topk)

            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H_Q,
                "H_V": 1,
                "H_K": 1,
                "N_KV": stats["g_sk"],
                "H_KV": 1,  # single MLA-absorbed latent KV head
                "d_h_qk": d_h,
                "d_h_v": d_h,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                **stats,
                "dtype_Q": "c10::bfloat16",
                "dtype_KV": "c10::bfloat16",  # V4 KV cache modeled as bf16
                "sliding_window": sliding_window,
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        return torch_dtype_map("c10::bfloat16")

    @classmethod
    def _sliding_window(cls, stats, window, index_topk):
        """Per-mode cap on the KV span each decode query attends to."""
        if cls.MODE == "csa":
            # CSA attends window + min(ceil(ctx / stride), index_topk) compressed slots per query.
            g_sq = stats.get("g_sq") or 0
            g_sk = stats.get("g_sk") or 0
            if g_sq <= 0:
                return window + index_topk  # saturated fallback (no annotation)
            mean_ctx = g_sk / g_sq
            n_committed = math.ceil(mean_ctx / cls.CSA_COMPRESS_STRIDE)
            return int(window + min(n_committed, index_topk))
        if cls.MODE == "hca":
            g_sq = stats.get("g_sq") or 0
            g_sk = stats.get("g_sk") or 0
            if g_sq <= 0:
                return 0  # uncapped / dense fallback when annotation missing
            mean_ctx = g_sk / g_sq
            return int(window + math.ceil(mean_ctx / cls.HCA_COMPRESS_STRIDE))
        return window  # swa (dense sliding window)


class pseudo_v4_paged_decode_swa(_V4PagedDecodeBase):
    """DeepSeek-V4 dense / sliding-window sparse paged decode."""

    MODE = "swa"


class pseudo_v4_paged_decode_csa(_V4PagedDecodeBase):
    """DeepSeek-V4 compressed sparse attention (CSA) paged decode."""

    MODE = "csa"


class pseudo_v4_paged_decode_hca(_V4PagedDecodeBase):
    """DeepSeek-V4 hierarchical compressed attention (HCA) paged decode."""

    MODE = "hca"


class vllm_sparse_attn_decode(_V4PagedDecodeBase):
    """HBM bytes of the BF16 query and output, the FP8 KV records and their slot indices, and the fp32 split partials written by the partial kernel and read back by the reduce; the page pool is not read.

    Each query reads the rows in its ``main_indptr`` / ``extra_indptr`` ranges,
    so the attended KV comes from the step annotation: ``min(ctx, window)``
    window rows plus ``min(ctx // 4, index_topk)`` CSA rows. Only CSA layers
    (``v4_mode == "csa"``, stamped by the injector) are modeled; HCA and
    SWA-only layers launch the same kernel and are no-perf.
    """

    _KV_RECORD_BYTES = 576
    _INDEX_BYTES = 4
    WINDOW = 128
    INDEX_TOPK = 1024

    @classmethod
    def get_param_details(cls, event):
        try:
            args = event.get("args") or {}
            dims = args.get("Input Dims") or []
            q, partial = dims[0], dims[1]
            tokens, heads, d_h = int(q[0]), int(q[1]), int(q[2])
            block_k = int(args["v4_block_k"])
            if partial[0] != tokens:
                raise ValueError
            if partial[2] == heads:
                splits = int(partial[1])
            elif partial[1] == heads:
                splits = int(partial[2])
            else:
                raise ValueError
            if tokens <= 0 or heads <= 0 or d_h <= 0 or splits <= 0 or block_k <= 0:
                raise ValueError
            if args.get("v4_mode") != "csa":
                raise ValueError
            ann = IterationAnnotation(str(event.get("annotation")))
            if not ann.has_sqsk or ann.g_sq <= 0 or ann.g_sk <= 0:
                raise ValueError
            window = int(args.get("v4_window") or cls.WINDOW)
            index_topk = int(args.get("v4_index_topk") or cls.INDEX_TOPK)
            mean_ctx = int(ann.g_sk // ann.g_sq)
            attended = min(mean_ctx, window) + min(
                mean_ctx // cls.CSA_COMPRESS_STRIDE, index_topk
            )
            live = min(ann.g_sq, tokens)
            details = {
                "B": 1,
                "N_Q": tokens,
                "H_Q": heads,
                "H_V": 1,
                "H_K": 1,
                "N_KV": attended,
                "H_KV": 1,
                "d_h_qk": d_h,
                "d_h_v": d_h,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                "c_sq": 0,
                "c_sk": 0,
                "c_sqsq": 0,
                "c_sqsk": 0,
                "g_sq": tokens,
                "g_sk": live * attended,
                "g_sqsq": tokens,
                "g_sqsk": live * attended,
                "dtype_Q": "c10::BFloat16",
                "dtype_KV": "c10::Float8_e4m3fn",
                "sliding_window": attended,
                "attended_kv": attended,
                "live_queries": live,
                "num_splits": splits,
                "v4_mode": "csa",
                "window": window,
                "index_topk": index_topk,
            }
            return add_logical_batch_size(details, event)
        except (IndexError, KeyError, TypeError, ValueError):
            return InferenceAttention.no_perf_param_details()

    def bytes(self, bytes_per_element=None):
        if self.param_details.get("_no_perf"):
            return None
        tokens = self.param_details["g_sq"]
        kv_rows = self.param_details["g_sqsk"]
        splits = self.param_details["num_splits"]
        q_bytes = 2 * tokens * self.H_Q * self.d_h_qk * 2
        kv_bytes = kv_rows * (self._KV_RECORD_BYTES + self._INDEX_BYTES)
        # part_m, part_l and part_acc [tokens, splits, H, d]: written, then read by the reduce.
        partial_bytes = 2 * tokens * splits * self.H_Q * (2 + self.d_h_qk) * 4
        return q_bytes + kv_bytes + partial_bytes


class pa_sparse_prefill_opus_fwd(InferenceAttention):
    """Perf model for ``aiter::pa_sparse_prefill_opus_fwd`` (DeepSeek-V4 sparse
    prefill main attention).

        Input Dims[0] = Q                 (N_Q, H_Q, d_h)
        Input Dims[2] = kv_indices_prefix (P_prefix,)   selected history slots
        Input Dims[5] = kv_indices_extend (P_extend,)   in-chunk local-window slots

    ``pairs = P_prefix + P_extend`` is the exact number of (query-token, KV-slot)
    pairs the kernel attends

    """

    @staticmethod
    def get_param_details(event):
        try:
            dims = event["args"]["Input Dims"]
            types = event["args"]["Input type"]
            q_shape = dims[0]
            N_Q, H_Q, d_h = q_shape[0], q_shape[1], q_shape[2]
            pairs = int(dims[2][0]) + int(dims[5][0])
            dtype_Q = types[0]
            dtype_KV = types[1] if len(types) > 1 else dtype_Q
            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H_Q,
                "H_V": 1,
                "H_K": 1,
                "N_KV": pairs,
                "H_KV": 1,  # single MLA-absorbed latent KV head
                "d_h_qk": d_h,
                "d_h_v": d_h,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                "prefill_pairs": pairs,
                "dtype_Q": dtype_Q,
                "dtype_KV": dtype_KV,
            }
        except (ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()

    def flops(self):
        if self.param_details.get("_no_perf"):
            return None
        pairs = self.param_details["prefill_pairs"]
        # per pair, per head: QK (2*d_h_qk) + PV (2*d_h_v)
        return pairs * self.H_Q * 2 * (self.d_h_qk + self.d_h_v)

    def bytes(self, bytes_per_element=None):
        if self.param_details.get("_no_perf"):
            return None
        bpe = bytes_per_element
        if bpe is None:
            bpe = name2bpe(self.param_details.get("dtype_Q")) or 2
        dtype_kv = self.param_details.get("dtype_KV")
        bpe_kv = (name2bpe(dtype_kv) if dtype_kv else None) or bpe
        pairs = self.param_details["prefill_pairs"]
        qo_bytes = 2 * self.N_Q * self.H_Q * self.d_h_qk * bpe
        kv_bytes = pairs * self.d_h_v * bpe_kv
        return qo_bytes + kv_bytes

    def get_compute_precision(self):
        """Compute precision: BF16 (bf16 MFMA), taken from the Q dtype.

        DeepSeek-V4 sparse prefill runs bf16 MFMA (validated against
        ``SQ_INSTS_VALU_MFMA_MOPS_BF16`` counters). Falls back to bf16 if the
        Q dtype is missing from the trace.
        """
        if self.param_details.get("_no_perf"):
            return None
        dtype = self.param_details.get("dtype_Q")
        return torch_dtype_map(dtype) if dtype else torch_dtype_map("c10::bfloat16")


class vllm_unified_attention_with_output(InferenceAttention):
    """Attention perf model for vLLM unified_attention_with_output events."""

    pass


class mha_varlen_fwd(InferenceAttention):
    pass


class aiter_fmha_v3_varlen_fwd(InferenceAttention):
    """
    Annotation-aware perf model for ``aiter::fmha_v3_varlen_fwd`` (inference:
    sglang / vLLM).

    Uses the same chunk statistics as :class:`InferenceAttention` (``annotation``
    on the event). Sets ``d_h_v`` from the **v** tensor (``Input Dims[2]``) so MLA
    shapes with differing Q/K vs V head dims are modeled correctly.

    If annotation parsing fails, this class falls back to the core shape-based
    SDPA model so regular training traces still get FLOPs/bytes coverage while
    annotated inference traces use the more accurate per-request sequence stats.
    """

    category = "SDPA_fwd"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None, enable_origami=False):
        self.enable_origami = enable_origami
        super().__init__(event, arch, python_path)

    @staticmethod
    def _core_param_details(event):
        params = perf_model.aiter__fmha_v3_varlen_fwd.get_param_details(event).copy()
        params["_fallback_core"] = True
        return params

    def _core_model(self):
        return perf_model.aiter__fmha_v3_varlen_fwd(
            self.event,
            self.arch,
            self.python_path,
            enable_origami=self.enable_origami,
        )

    @staticmethod
    def get_param_details(event):
        params = InferenceAttention.get_param_details(event)
        if params.get("_no_perf"):
            try:
                return aiter_fmha_v3_varlen_fwd._core_param_details(event)
            except (ValueError, IndexError, KeyError, TypeError):
                return params
        args = event.get("args") or {}
        dims = args.get("Input Dims") or []
        if len(dims) > 2 and len(dims[2]) >= 1:
            params["d_h_v"] = dims[2][-1]
        return params

    def flops(self):
        if self.param_details.get("_fallback_core"):
            return self._core_model().flops()
        return super().flops()

    def bytes(self, bytes_per_element=None):
        if self.param_details.get("_fallback_core"):
            if bytes_per_element is None:
                return self._core_model().bytes()
            return self._core_model().bytes(bytes_per_element)
        return super().bytes(bytes_per_element)

    def get_compute_precision(self):
        if self.param_details.get("_fallback_core"):
            return self._core_model().get_compute_precision()
        return super().get_compute_precision()

    def get_simulation_time(self):
        if self.param_details.get("_fallback_core"):
            return self._core_model().get_simulation_time()
        return None


class aiter_paged_attention_ragged(InferenceAttention):
    """
    Performance model for ``aiter::paged_attention_ragged`` (inference: sglang),
    surfaced in traces as ``sglang_profiler::attention_paged_attention_ragged``
    (per-layer ``_<idx>`` suffix is stripped upstream).

    TODO: account for fp8 KV-cache dtype (``kv_cache_dtype`` ``"fp8"`` /
    ``"fp8_e4m3"`` stores K/V as 1 B/elem while Q and output remain BF16/FP16);
    we will make that change later.

    Uses the same chunk statistics as :class:`InferenceAttention` (``annotation``
    on the event) for packed variable-length decode requests. Reads ``Q`` from
    ``Input Dims[2]`` and paged ``K``/``V`` caches from ``Input Dims[3]``/``[4]``
    (shape ``[num_pages, page_size, H_KV, head_dim]``); ``d_h_v`` is taken from
    the **v** tensor so MLA-style shapes with differing Q/K vs V head dims are
    modeled correctly.

    Unparseable annotation or unexpected input dims yields
    :meth:`InferenceAttention.no_perf_param_details` (see base class).
    """

    @staticmethod
    def get_param_details(event):
        try:
            annotation = str(event.get("annotation"))
            stats = InferenceAttention._parse_chunk_stats(annotation)

            dims = event["args"]["Input Dims"]
            q_shape = dims[2]
            k_shape = dims[3]
            v_shape = dims[4]
            N_Q, H_Q, d_h_qk = q_shape
            H_KV = k_shape[-2]
            d_h_v = v_shape[-1]
            N_KV = stats["c_sk"] + stats["g_sk"]
            dtype_Q = event["args"]["Input type"][2]

            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H_Q,
                "H_V": H_KV,
                "H_K": H_KV,
                "N_KV": N_KV,
                "H_KV": H_KV,
                "d_h_qk": d_h_qk,
                "d_h_v": d_h_v,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                **stats,
                "dtype_Q": dtype_Q,
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()


class aiter_mha_batch_prefill(InferenceAttention):
    """
    Performance model for ``aiter::mha_batch_prefill`` (inference: sglang) — the
    paged chunked-prefill / extend kernel.

    TODO: account for fp8 (``q_descale`` / ``k_descale`` / ``v_descale`` /
    ``kv_block_descale`` paths store K/V in lower precision than Q/output); we
    will make that change later.

    Uses the same chunk statistics as :class:`InferenceAttention` (``annotation``
    on the event), which naturally cover mixed chunked-prefill + decode batches
    via the ``c_*`` and ``g_*`` aggregates. Sets ``d_h_v`` from the **v** tensor
    (``Input Dims[2]``) so MLA shapes with differing Q/K vs V head dims are
    modeled correctly.

    Unparseable annotation yields :meth:`InferenceAttention.no_perf_param_details`
    (see base class).
    """

    @staticmethod
    def get_param_details(event):
        params = InferenceAttention.get_param_details(event)
        if params.get("_no_perf"):
            return params
        args = event.get("args") or {}
        dims = args.get("Input Dims") or []
        if len(dims) > 2 and len(dims[2]) >= 1:
            params["d_h_v"] = dims[2][-1]
        return params


class mla_decode_fwd(InferenceAttention):
    pass


class pseudo_mla_prefill_fwd(InferenceAttention):
    @staticmethod
    def get_param_details(event):
        params = InferenceAttention.get_param_details(event)
        if params.get("_no_perf"):
            return params
        args = event.get("args") or {}
        dims = args.get("Input Dims") or []
        if len(dims) > 2 and len(dims[2]) >= 1:
            params["d_h_v"] = dims[2][-1]
        return params


class mla_tilelang_sparse_fwd(InferenceAttention):
    """
    Performance model for ``sglang::tilelang_sparse_fwd`` (TileLang SparseMLA;
    inference: sglang).

    Indices topk from ``Input Dims[2][-1]`` → ``sliding_window`` /
    :meth:`InferenceAttention._cap_gen_kv`.
    """

    @staticmethod
    def get_param_details(event):
        params = InferenceAttention.get_param_details(event)
        if params.get("_no_perf"):
            return params
        concrete = event.get("Concrete Inputs", [])
        if len(concrete) < 5:
            params["d_h_v"] = 512
        else:
            params["d_h_v"] = int(concrete[4])
        args = event.get("args") or {}
        dims = args.get("Input Dims") or []
        if len(dims) <= 2 or len(dims[2]) < 1:
            return InferenceAttention.no_perf_param_details()
        topk = int(dims[2][-1])
        if topk <= 0:
            return InferenceAttention.no_perf_param_details()
        params["sliding_window"] = topk
        return params


class vllm_unified_mla_attention_with_output(InferenceAttention):
    pass


class pa_decode_gluon(InferenceAttention):
    """
    Performance model for ``aiter::pa_decode_gluon`` (gluon Triton kernel
    ``paged_attention_decode_sliding_window``; inference: ATOM/vLLM).

    Grouped-query paged decode attention over an FP8 E4M3 KV cache.

    Decode-only kernel — it never processes prefill/context requests (a separate
    extend kernel does). So ``get_param_details`` forces the context aggregates
    (``c_*``) to 0, which collapses the inherited ``flops``/``bytes`` to the
    generation-only roofline, and sets ``dtype_KV`` from the FP8 cache.

    Sliding window: the ``sliding_window`` arg (``Concrete Inputs[19]``) caps the
    KV each query attends to at ``min(ctx_i, W)``. It is surfaced as
    ``param_details["sliding_window"]`` and the base roofline applies the cap
    (aggregate approximation ``min(g_sk, g_sq*W)``). ``-1`` layers (full
    attention) are unaffected.
    """

    @staticmethod
    def get_param_details(event):
        try:
            annotation = str(event.get("annotation"))
            stats = InferenceAttention._parse_chunk_stats(annotation)

            dims = event["args"]["Input Dims"]
            types = event["args"]["Input type"]
            q_shape = dims[0]
            N_Q, H_Q, d_h_qk = q_shape[0], q_shape[1], q_shape[2]
            # k_cache: [num_blocks, H_KV, ...] -> H_KV at index 1
            H_KV = dims[2][1]
            d_h_v = d_h_qk  # logical V head dim matches Q for this kernel
            # Decode-only kernel: it never processes prefill/context requests
            stats["c_sq"] = stats["c_sk"] = stats["c_sqsq"] = stats["c_sqsk"] = 0
            N_KV = stats["g_sk"]
            dtype_Q = types[0]
            dtype_KV = types[2] if len(types) > 2 else types[0]
            # sliding_window is the positional `sliding_window` arg; <= 0 (e.g.
            # -1) means full attention. Capping is applied by the base roofline.
            conc = event["args"].get("Concrete Inputs") or []
            try:
                sliding_window = int(conc[19])
            except (IndexError, ValueError, TypeError):
                sliding_window = 0

            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H_Q,
                "H_V": H_KV,
                "H_K": H_KV,
                "N_KV": N_KV,
                "H_KV": H_KV,
                "d_h_qk": d_h_qk,
                "d_h_v": d_h_v,
                "dropout": 0.0,
                "causal": False,
                "flash_impl": True,
                **stats,
                "dtype_Q": dtype_Q,
                "dtype_KV": dtype_KV,
                "sliding_window": sliding_window,
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()


class gdn_attention_core(InferenceAttention):
    """
    Performance model for vllm::gdn_attention_core (Gated Delta Network).

    Recurrent linear attention used by Qwen3-Next/Qwen3.5.  Each value head
    maintains a state S ∈ R^{d_v × d_k} that is decayed, updated, and queried
    per token.  There is no KV cache — compute per token is O(d_v * d_k)
    regardless of past context length.

    Head relationship: num_v_heads = 2 * num_k_heads.  Multiple v-heads share
    one q/k head.

    Input Dims layout:
        [0] mixed_qkv       [T, 2*H_K*d_k + H_V*d_v]
        [1] b                [T, H_V]
        [2] a                [T, H_V]
        [3] core_attn_out    [T, H_V, d_v]
        [4] layer_name       ()

    Per token per v-head FLOPs (recurrent delta rule):
        decay S:         1 * d_v * d_k
        retrieval S^T k: 2 * d_v * d_k
        state update:    2 * d_v * d_k
        output S^T q:    2 * d_v * d_k
        total:           7 * d_v * d_k
    """

    def __init__(self, event, arch=None, python_path=None):
        super().__init__(event, arch, python_path)
        self.H_V = self.param_details["H_V"]
        self.d_k = self.param_details["d_h_qk"]
        self.d_v = self.param_details["d_h_v"]

    @staticmethod
    def get_param_details(event):
        stats = IterationAnnotation(str(event.get("annotation"))).chunk_stats()
        c_sq = stats["c_sq"]
        g_sq = stats["g_sq"]

        input_dims = event["args"]["Input Dims"]
        D = input_dims[0][1]  # 2*H_K*d_k + H_V*d_v
        H_V = input_dims[1][1]  # num_v_heads / tp
        d_v = input_dims[3][2]  # head_v_dim

        H_K = H_V // 2
        key_dim_tp = (D - H_V * d_v) // 2
        d_k = key_dim_tp // H_K

        dtype_Q = event["args"]["Input type"][0]

        return {
            "B": 1,
            "N_Q": c_sq + g_sq,
            "H_Q": H_K,
            "N_KV": 0,
            "H_KV": H_K,
            "H_V": H_V,
            "H_K": H_K,
            "d_h_qk": d_k,
            "d_h_v": d_v,
            "c_sq": c_sq,
            "c_sk": c_sq,
            "c_sqsq": 0,
            "c_sqsk": 0,
            "g_sq": g_sq,
            "g_sk": g_sq,
            "g_sqsq": 0,
            "g_sqsk": 0,
            "dtype_Q": dtype_Q,
        }

    @staticmethod
    def _gdn_flops_func(H_V, d_k, d_v, total_tokens):
        """GDN recurrent delta rule FLOPs.

        Per token per v-head: 7 * d_v * d_k
        (decay: 1, retrieval: 2, state update: 2, output query: 2)
        """
        return total_tokens * H_V * 7 * d_v * d_k

    @staticmethod
    def _gdn_bytes_func(H_V, d_k, d_v, total_tokens, bytes_per_element):
        """GDN memory traffic.  State S stays in registers during recurrence.

        Per token read:  q(d_k) + k(d_k) shared across 2 v-heads → H_V*d_k
                         v(d_v) per v-head → H_V*d_v
                         a(1) + b(1) per v-head → 2*H_V
        Per token write: o(d_v) per v-head → H_V*d_v
        """
        elems_per_token = H_V * (d_k + 2 * d_v + 2)
        return total_tokens * elems_per_token * bytes_per_element

    def flops(self):
        total_tokens = self.param_details["c_sq"] + self.param_details["g_sq"]
        if total_tokens == 0:
            raise NotImplementedError(
                "GDN perf model requires annotation with non-zero c_sq or g_sq"
            )
        return self._gdn_flops_func(self.H_V, self.d_k, self.d_v, total_tokens)

    def bytes(self, bytes_per_element=2):
        total_tokens = self.param_details["c_sq"] + self.param_details["g_sq"]
        return self._gdn_bytes_func(
            self.H_V, self.d_k, self.d_v, total_tokens, bytes_per_element
        )

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for GDN attention is not defined.")

    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for GDN attention is not defined.")

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_Q", None)
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "matrix"


class mla_decode_v4_asm(InferenceAttention):
    """HBM bytes of the gathered KV pages, not the page pool; a multi-split launch leaves the BF16 output to stage 2.

    ``kv_page_indices`` is sized for the maximum context under CUDA graphs, so
    with a sq/sk step annotation the attended KV rows come from the annotated
    context: ``min(ctx, window)`` plus ``ctx / stride`` compressed rows, where
    the layer (SWA / CSA / HCA) is inferred from the per-request index
    capacity. The buffer length remains the upper bound and the fallback.
    ``v4_window`` / ``v4_index_topk`` event args override the defaults.
    """

    WINDOW = 128
    # Upper bound on the CSA index budget; only separates CSA from HCA capacity.
    MAX_INDEX_TOPK = 1024
    INDEX_ALIGN = 64
    COMPRESS_STRIDE = {
        "swa": None,
        "csa": _V4PagedDecodeBase.CSA_COMPRESS_STRIDE,
        "hca": _V4PagedDecodeBase.HCA_COMPRESS_STRIDE,
    }

    @classmethod
    def _layer_mode(cls, capacity_per_req, window, index_topk):
        """SWA / CSA / HCA from the per-request ``kv_page_indices`` capacity."""
        if capacity_per_req <= window + cls.INDEX_ALIGN:
            return "swa"
        if capacity_per_req <= window + index_topk + cls.INDEX_ALIGN:
            return "csa"
        return "hca"

    @classmethod
    def _live_kv(cls, event, capacity, num_seqs):
        """Attended KV rows from the step annotation, or None without one."""
        annotation = event.get("annotation")
        if not annotation or num_seqs <= 0:
            return None
        ann = IterationAnnotation(str(annotation))
        if not ann.has_sqsk or ann.g_sq <= 0 or ann.g_sk <= 0:
            return None
        args = event.get("args") or {}
        window = int(args.get("v4_window") or cls.WINDOW)
        index_topk = int(args.get("v4_index_topk") or cls.MAX_INDEX_TOPK)
        capacity_per_req = capacity // num_seqs
        mode = cls._layer_mode(capacity_per_req, window, index_topk)
        mean_ctx = ann.g_sk / ann.g_sq
        per_req = min(math.ceil(mean_ctx), window)
        stride = cls.COMPRESS_STRIDE[mode]
        if stride:
            per_req += math.ceil(mean_ctx / stride)
        per_req = min(per_req, capacity_per_req)
        return mode, min(per_req * min(ann.g_sq, num_seqs), capacity)

    @staticmethod
    def _buffer_base(dims):
        """Index of the split partials: 13 when the scalars precede the output
        buffers (older aiter), 9 when the buffers come first (current aiter)."""
        for base in (13, 9):
            if (
                len(dims) > base + 2
                and len(dims[base]) == 4
                and len(dims[base + 1]) == 4
                and len(dims[base + 2]) == 3
            ):
                return base
        return None

    @classmethod
    def get_param_details(cls, event):
        args = event.get("args", {})
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        base = cls._buffer_base(dims)
        if base is None or len(dims[0]) != 3 or len(dims[2]) != 4 or not dims[6]:
            return add_logical_batch_size(
                InferenceAttention.no_perf_param_details(), event
            )
        page_size = dims[2][1]
        h_kv = dims[2][2]
        kv_capacity = dims[6][0] * page_size
        q_tokens, h_q, d_qk = dims[0]
        d_v = dims[base + 2][-1]
        ints = []
        for item in args.get("Concrete Inputs") or []:
            if item in ("", None):
                continue
            try:
                ints.append(int(item))
            except (TypeError, ValueError):
                continue
        max_seqlen_q = ints[0] if ints else 1
        num_kv_splits = ints[2] if len(ints) > 2 else 1
        write_valid = len(ints) > 3 and ints[3] == 1
        if (
            page_size != 1
            or h_kv != 1
            or max_seqlen_q != 1
            or q_tokens <= 0
            or kv_capacity <= 0
            or d_qk != 512
        ):
            return add_logical_batch_size(
                InferenceAttention.no_perf_param_details(), event
            )
        num_seqs = dims[5][0] - 1 if dims[5] else 0
        live = cls._live_kv(event, kv_capacity, num_seqs)
        num_kv, kv_mode = kv_capacity, None
        if live is not None:
            kv_mode, num_kv = live
        # One query token per attended KV row, so g_sqsk equals the KV row count.
        details = {
            "B": 1,
            "N_Q": q_tokens,
            "H_Q": h_q,
            "N_KV": num_kv,
            "H_KV": h_kv,
            "d_h_qk": d_qk,
            "d_h_v": d_v,
            "c_sq": 0,
            "c_sk": 0,
            "c_sqsq": 0,
            "c_sqsk": 0,
            "g_sq": q_tokens,
            "g_sk": num_kv,
            "g_sqsq": q_tokens,
            "g_sqsk": num_kv,
            "dtype_Q": types[0] if types else None,
            "q_tokens": q_tokens,
            "num_kv": num_kv,
            "kv_capacity": kv_capacity,
            "kv_mode": kv_mode,
            "num_kv_splits": num_kv_splits,
            "dtype_rope": types[1] if len(types) > 1 else None,
            "dtype_kv": types[2] if len(types) > 2 else None,
            "dtype_kvrope": types[3] if len(types) > 3 else None,
            "dtype_index": types[6] if len(types) > 6 else "int",
            "buffer_base": base,
            "split_shape": dims[base],
            "lse_shape": dims[base + 1],
            "out_shape": dims[base + 2],
            "indptr_elems": sum(dims[i][0] for i in (4, 5, 7) if dims[i]),
            "sink_elems": dims[8][0] if dims[8] else 0,
            "valid_elems": (
                dims[base + 3][0]
                if write_valid and len(dims) > base + 3 and dims[base + 3]
                else 0
            ),
            "rope_elems": math.prod(dims[1]) if len(dims) > 1 and dims[1] else 0,
            "kv_rope_dim": dims[3][-1] if len(dims) > 3 and dims[3] else 64,
        }
        return add_logical_batch_size(details, event)

    def flops(self):
        # QK + PV via InferenceAttention.flops_func; FlashAttention arXiv:2205.14135, Section 2.
        return super().flops()

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        base = p["buffer_base"]
        if not dense_counted_tensor(
            self.event, (0, 1, 2, 3, 6, 8, base, base + 1, base + 2)
        ):
            return None
        bpe_q = name2bpe(str(p["dtype_Q"]).lower())
        bpe_kv = name2bpe(str(p["dtype_kv"]).lower())
        bpe_rope = name2bpe(str(p["dtype_rope"]).lower())
        bpe_kvrope = name2bpe(str(p["dtype_kvrope"]).lower())
        bpe_index = name2bpe(str(p["dtype_index"]).lower())
        if None in (bpe_q, bpe_kv, bpe_rope, bpe_kvrope, bpe_index):
            return None
        read = (
            p["q_tokens"] * p["H_Q"] * p["d_h_qk"] * bpe_q
            + p["rope_elems"] * bpe_rope
            + p["num_kv"] * p["d_h_qk"] * bpe_kv
            + p["num_kv"] * p["kv_rope_dim"] * bpe_kvrope
            + p["num_kv"] * bpe_index
            + p["indptr_elems"] * 4
            + p["sink_elems"] * 4
        )
        if p["num_kv_splits"] == 1:
            written = math.prod(p["out_shape"]) * 2
        else:
            written = math.prod(p["split_shape"]) * 4 + math.prod(p["lse_shape"]) * 4
        return read + written + p["valid_elems"] * 4

    def get_maf_type(self):
        if self.param_details.get("_no_perf"):
            return None
        return super().get_maf_type()


class sgl_kernel_deepseek_v4_topk_transform_512:
    """
    Perf model for sgl_kernel::deepseek_v4_topk_transform_512 (Flash indexer
    top-k + page-table transform; topk fixed at 512 by the op name).

    Expected Input Dims:
        [0] score (rows, L_max)   [1] lengths (rows,)
        [2] src/aux indices      [3] dst / topk indices (rows, 512)
    """

    category = "InferenceAttention"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        try:
            stats = InferenceAttention._parse_chunk_stats(str(event.get("annotation")))
            dims = event["args"]["Input Dims"]
            types = event["args"]["Input type"]
            rows = dims[0][0]
            topk = dims[3][1] if len(dims) > 3 and dims[3] and len(dims[3]) > 1 else 512
            params = {
                "rows": rows,
                "topk": topk,
                "dtype_score": types[0],
                "dtype_page_table": types[3] if len(types) > 3 else "int",
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return {"_no_perf": True}
        params.update(stats)
        params["valid_scores"] = sum(
            stats[f"{p}_sqsk"] - (stats[f"{p}_sqsq"] - stats[f"{p}_sq"]) // 2
            for p in ("c", "g")
        )
        return params

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        return p["valid_scores"]

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe_score = name2bpe(p["dtype_score"])
        bpe_idx = name2bpe(p["dtype_page_table"]) or 4
        if bpe_score is None:
            return None
        rows, topk = p["rows"], p["topk"]
        gathered = min(topk, p["valid_scores"] / rows) if rows else 0
        read_scores = p["valid_scores"] * bpe_score
        return read_scores + rows * (gathered + topk) * bpe_idx

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for indexer top-k is not defined.")

    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for indexer top-k is not defined.")

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_score")
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "vector"


class sgl_kernel_fast_topk:
    """
    Perf model for sgl_kernel::fast_topk (Flash decode indexer selection).

    Expected Input Dims:
        [0] score (rows, L_max)  [1] topk_idx (rows, topk)  [2] lengths (rows,)
    """

    category = "InferenceAttention"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        try:
            stats = InferenceAttention._parse_chunk_stats(str(event.get("annotation")))
            dims = event["args"]["Input Dims"]
            types = event["args"]["Input type"]
            rows, l_max = dims[0][0], dims[0][1]
            topk = dims[1][1] if len(dims) > 1 and dims[1] and len(dims[1]) > 1 else 0
            params = {
                "rows": rows,
                "l_max": l_max,
                "topk": topk,
                "dtype_score": types[0],
                "dtype_idx": types[1] if len(types) > 1 else "int",
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return {"_no_perf": True}
        params.update(stats)
        params["valid_scores"] = sum(
            stats[f"{p}_sqsk"] - (stats[f"{p}_sqsq"] - stats[f"{p}_sq"]) // 2
            for p in ("c", "g")
        )
        return params

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        return p["valid_scores"]

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe_score = name2bpe(p["dtype_score"])
        bpe_idx = name2bpe(p["dtype_idx"]) or 4
        if bpe_score is None:
            return None
        rows, topk = p["rows"], p["topk"]
        return p["valid_scores"] * bpe_score + rows * topk * bpe_idx

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for fast_topk is not defined.")

    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for fast_topk is not defined.")

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_score")
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "vector"


class _FlashIndexerConfig:
    """Hardcoded Flash indexer geometry when Input Dims are absent on Synthetic Ops."""

    _CONFIGS = {
        "DeepSeek-V4-Pro": {
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 1024,
        },
        "DeepSeek-V4-Flash": {
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
        },
    }

    @classmethod
    def config_for(cls, model_name):
        if model_name:
            for key, cfg in cls._CONFIGS.items():
                if key.lower() in str(model_name).lower():
                    return cfg
        return cls._CONFIGS["DeepSeek-V4-Flash"]


class pseudo_pa_mqa_logits_fp4(InferenceAttention):
    """
    Flash SGLang FP4 MQA lightning-indexer logits (pa_mqa_logits_fp4_*).

    QK-only MQA score path (no PV). Prefer annotation sq/sk; fall back to
    Flash indexer head/dim defaults stamped by the Trace2Tree injector.
    """

    @staticmethod
    def get_param_details(event):
        try:
            args = event.get("args") or {}
            annotation = str(event.get("annotation"))
            stats = InferenceAttention._parse_chunk_stats(annotation)
            cfg = _FlashIndexerConfig.config_for(args.get("v4_model_name"))
            H = int(args.get("index_n_heads") or cfg["index_n_heads"])
            d = int(args.get("index_head_dim") or cfg["index_head_dim"])
            index_topk = int(args.get("index_topk") or cfg["index_topk"])
            N_Q = stats["c_sq"] + stats["g_sq"]
            dims = args.get("Input Dims") or []
            types = args.get("Input type") or []
            if dims and len(dims) > 3 and dims[3] and len(dims[3]) >= 3:
                N_Q, H, d = dims[3][0], dims[3][1], dims[3][2]
            dtype_Q = types[3] if len(types) > 3 else "c10::Float4_e2m1fn_x2"
            dtype_KV = types[2] if len(types) > 2 else "c10::Float4_e2m1fn_x2"
            return {
                "B": 1,
                "N_Q": N_Q,
                "H_Q": H,
                "H_KV": 1,
                "N_KV": stats["c_sk"] + stats["g_sk"],
                "d_h_qk": d,
                "d_h_v": d,
                "dropout": 0.0,
                "causal": True,
                "flash_impl": True,
                "index_topk": index_topk,
                "dtype_Q": dtype_Q,
                "dtype_KV": dtype_KV,
                **stats,
            }
        except (NotImplementedError, ValueError, IndexError, KeyError, TypeError):
            return InferenceAttention.no_perf_param_details()

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        if p["c_sq"] == 0 and p["g_sq"] == 0:
            return None
        H, d = self.H_Q, self.d_h_qk
        ctx_qk = H * (2 * p["c_sqsk"] - p["c_sqsq"]) * d
        gen_qk = H * (2 * p["g_sqsk"]) * d
        return ctx_qk + gen_qk

    def bytes(self, bytes_per_element=None):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe_q = name2bpe(p.get("dtype_Q")) or 1
        bpe_kv = name2bpe(p.get("dtype_KV")) or 1
        N_Q, H, d = self.N_Q, self.H_Q, self.d_h_qk
        total_sk = p["c_sk"] + p["g_sk"]
        read_q = N_Q * H * d * bpe_q
        # MQA: keys shared across index heads.
        read_k = total_sk * d * bpe_kv
        avg_ctx = total_sk / N_Q if N_Q else 0
        topk = min(p["index_topk"], avg_ctx) if p["index_topk"] else avg_ctx
        write_logits = N_Q * topk * 4  # streamed logits / top candidates
        return read_q + read_k + write_logits

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        return "fp4"

    def get_maf_type(self):
        return "matrix"


class pseudo_quantize_fp4_indexer:
    """
    Flash indexer FP4 quant (`quantize_fp4_indexer_tensor` / `_quantize_fp4_indexer`).

    Models elementwise FP4 pack of indexer keys. Geometry from injector stamps
    or annotation + Flash indexer defaults.
    """

    category = "GroupQuant"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        try:
            args = event.get("args") or {}
            cfg = _FlashIndexerConfig.config_for(args.get("v4_model_name"))
            d = int(args.get("index_head_dim") or cfg["index_head_dim"])
            annotation = str(event.get("annotation", "NA"))
            try:
                stats = InferenceAttention._parse_chunk_stats(annotation)
                n_tok = stats["c_sk"] + stats["g_sk"]
            except (NotImplementedError, ValueError, TypeError):
                stats = None
                n_tok = int(args.get("num_tokens") or 0)
            dims = args.get("Input Dims") or []
            if dims and dims[0]:
                # Prefer explicit tensor numel when present.
                shape = dims[0]
                numel = 1
                for x in shape:
                    numel *= int(x)
                n_tok = numel // max(d, 1)
            if n_tok <= 0:
                return {"_no_perf": True}
            return {
                "num_elems": n_tok * d,
                "dtype_in": "c10::BFloat16",
                "stats": stats,
            }
        except (ValueError, IndexError, KeyError, TypeError):
            return {"_no_perf": True}

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        # max-abs + scale per element (same order as other quant models)
        return 2 * p["num_elems"]

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe_in = name2bpe(p["dtype_in"]) or 2
        return p["num_elems"] * bpe_in + p["num_elems"] * 1  # bf16 read + fp4 write

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        return "fp4"

    def get_maf_type(self):
        return "vector"


class pseudo_flash_indexer_topk:
    """
    Pseudo parent for Flash PD indexer selection that currently lands on
    aten::topk (+ optional repeat_interleave). Uses score matrix dims from the
    wrapped aten::topk event rather than inventing a generic aten::topk formula
    for unrelated call sites.
    """

    category = "InferenceAttention"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        try:
            dims = event["args"]["Input Dims"]
            types = event["args"].get("Input type") or []
            rows, cols = dims[0][0], dims[0][1]
            # Guard: only Flash indexer-scale score matrices (avoid generic aten::topk).
            if cols is None or int(cols) < 1024:
                return {"_no_perf": True}
            # k may be Concrete Inputs or a trailing empty dim; default Flash 512.
            k = 512
            concrete = event["args"].get("Concrete Inputs") or []
            for item in concrete:
                s = str(item).strip()
                if s.isdigit():
                    k = int(s)
                    break
            annotation = str(event.get("annotation", "NA"))
            try:
                stats = InferenceAttention._parse_chunk_stats(annotation)
            except (NotImplementedError, ValueError, TypeError):
                stats = {
                    "c_sq": 0,
                    "c_sk": 0,
                    "c_sqsq": 0,
                    "c_sqsk": 0,
                    "g_sq": 0,
                    "g_sk": 0,
                    "g_sqsq": 0,
                    "g_sqsk": 0,
                }
            return {
                "rows": rows,
                "cols": cols,
                "k": k,
                "dtype_score": types[0] if types else "float",
                **stats,
            }
        except (ValueError, IndexError, KeyError, TypeError):
            return {"_no_perf": True}

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        # Comparison work proportional to the score buffer scanned.
        return p["rows"] * p["cols"]

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe = name2bpe(p["dtype_score"]) or 4
        return p["rows"] * p["cols"] * bpe + p["rows"] * p["k"] * 4

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        dtype = self.param_details.get("dtype_score")
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "vector"


class fused_k_norm_rope_q_flashmla:
    """
    FlashMLA fused K-RMSNorm + RoPE + Q prep
    (`fused_k_norm_rope_q_flashmla` / `fused_k_norm_rope_flashmla_with_q`).

    Geometry from injector stamps (tokens, head_dim=512, rope_dim=64 defaults).
    """

    category = "FusedRoPE"
    bwd_category = None

    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)

    @staticmethod
    def get_param_details(event):
        try:
            args = event.get("args") or {}
            annotation = str(event.get("annotation", "NA"))
            try:
                stats = InferenceAttention._parse_chunk_stats(annotation)
                tokens = stats["c_sq"] + stats["g_sq"]
            except (NotImplementedError, ValueError, TypeError):
                tokens = int(args.get("num_tokens") or 0)
            d_h = int(args.get("d_h") or 512)
            rope_dim = int(args.get("rope_dim") or 64)
            if tokens <= 0:
                return {"_no_perf": True}
            return {
                "tokens": tokens,
                "d_h": d_h,
                "rope_dim": rope_dim,
                "dtype": "c10::BFloat16",
            }
        except (ValueError, KeyError, TypeError):
            return {"_no_perf": True}

    def flops(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        # RMSNorm ~2*d_h + RoPE ~6*rope_dim per token (sin/cos apply pair)
        return p["tokens"] * (2 * p["d_h"] + 6 * p["rope_dim"])

    def bytes(self):
        p = self.param_details
        if p.get("_no_perf"):
            return None
        bpe = name2bpe(p["dtype"]) or 2
        # read K + write rotated K (+ optional Q touch ~ same width)
        return 3 * p["tokens"] * p["d_h"] * bpe

    def get_compute_precision(self):
        if self.param_details.get("_no_perf"):
            return None
        return "bf16"

    def get_maf_type(self):
        return "vector"
