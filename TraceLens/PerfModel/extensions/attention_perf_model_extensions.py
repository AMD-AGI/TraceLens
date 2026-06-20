###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for pseudo-op extensions.
"""

from TraceLens.PerfModel.utils import torch_dtype_map, name2bpe
import re


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
        """Parse the sglang/vLLM annotation string into context/generation aggregates.

        Returns a dict with ``c_sq``, ``c_sk``, ``c_sqsq``, ``c_sqsk``,
        ``g_sq``, ``g_sk``, ``g_sqsq``, ``g_sqsk``. Raises ``NotImplementedError``
        if the annotation is missing or cannot be parsed.
        """
        if annotation == "NA":
            raise NotImplementedError(
                "VLLM attention without annotation is not supported"
            )

        if "sq" not in annotation:
            requests = annotation.replace("(", "_").replace(")", "_").split("_")
            if len(requests) < 8:
                raise NotImplementedError(
                    "VLLM attention without annotation is not supported"
                )
            c_sq = int(requests[3])
            c_sk = int(requests[3])
            c_sqsq = int(requests[4])
            c_sqsk = int(requests[4])
            g_sq, g_sk, g_sqsq, g_sqsk = 0, 0, 0, 0
        else:
            name = annotation.replace("(", "_").replace(")", "_")
            requests = re.sub(r"[sqk]+", "_", name).split("_")
            if len(requests) < 16:
                raise NotImplementedError(
                    "VLLM attention without annotation is not supported"
                )
            c_sq = int(requests[5])
            c_sk = int(requests[6])
            c_sqsq = int(requests[7])
            c_sqsk = int(requests[8])
            g_sq = int(requests[13])
            g_sk = int(requests[14])
            g_sqsq = int(requests[15])
            g_sqsk = int(requests[16])

        return {
            "c_sq": c_sq,
            "c_sk": c_sk,
            "c_sqsq": c_sqsq,
            "c_sqsk": c_sqsk,
            "g_sq": g_sq,
            "g_sk": g_sk,
            "g_sqsq": g_sqsq,
            "g_sqsk": g_sqsk,
        }

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
            self.param_details["g_sqsk"],
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
            self.param_details["g_sk"],
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


class vllm_unified_attention_with_output(InferenceAttention):
    """Attention perf model for vLLM unified_attention_with_output events."""

    pass


class mha_varlen_fwd(InferenceAttention):
    pass


class aiter_fmha_v3_varlen_fwd(InferenceAttention):
    """
    Performance model for ``aiter::fmha_v3_varlen_fwd`` (inference: sglang / vLLM).

    Uses the same chunk statistics as :class:`InferenceAttention` (``annotation``
    on the event). Sets ``d_h_v`` from the **v** tensor (``Input Dims[2]``) so MLA
    shapes with differing Q/K vs V head dims are modeled correctly.

    Unparseable annotation yields :meth:`InferenceAttention.no_perf_param_details`
    (see base class); no packed-tensor fallback.
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
        return params


class vllm_unified_mla_attention_with_output(InferenceAttention):
    pass


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
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)
        self.H_V = self.param_details["H_V"]
        self.d_k = self.param_details["d_h_qk"]
        self.d_v = self.param_details["d_h_v"]

    @staticmethod
    def get_param_details(event):
        annotation = str(event.get("annotation"))
        if annotation == "NA":
            raise NotImplementedError(
                "GDN attention without annotation is not supported"
            )

        if "sq" not in annotation:
            requests = annotation.replace("(", "_").replace(")", "_").split("_")
            if len(requests) < 8:
                raise NotImplementedError(
                    "GDN attention without annotation is not supported"
                )
            c_sq = int(requests[3])
            g_sq = 0
        else:
            name = annotation.replace("(", "_").replace(")", "_")
            requests = re.sub(r"[sqk]+", "_", name).split("_")
            if len(requests) < 16:
                raise NotImplementedError(
                    "GDN attention without annotation is not supported"
                )
            c_sq = int(requests[5])
            g_sq = int(requests[13])

        input_dims = event["args"]["Input Dims"]
        T = input_dims[0][0]
        D = input_dims[0][1]  # 2*H_K*d_k + H_V*d_v
        H_V = input_dims[1][1]  # num_v_heads / tp
        d_v = input_dims[3][2]  # head_v_dim

        H_K = H_V // 2
        key_dim_tp = (D - H_V * d_v) // 2
        d_k = key_dim_tp // H_K

        dtype_Q = event["args"]["Input type"][0]

        return {
            "H_V": H_V,
            "H_K": H_K,
            "d_h_qk": d_k,
            "d_h_v": d_v,
            "c_sq": c_sq,
            "g_sq": g_sq,
            "dtype_Q": dtype_Q,
        }

    @staticmethod
    def flops_func(H_V, d_k, d_v, total_tokens):
        """GDN recurrent delta rule FLOPs.

        Per token per v-head: 7 * d_v * d_k
        (decay: 1, retrieval: 2, state update: 2, output query: 2)
        """
        return total_tokens * H_V * 7 * d_v * d_k

    @staticmethod
    def bytes_func(H_V, d_k, d_v, total_tokens, bytes_per_element):
        """GDN HBM traffic.  State S stays in registers during recurrence.

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
        return self.flops_func(self.H_V, self.d_k, self.d_v, total_tokens)

    def bytes(self, bytes_per_element=2):
        total_tokens = self.param_details["c_sq"] + self.param_details["g_sq"]
        return self.bytes_func(
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


class aten__efficient_attention_forward:
    """
    Performance model for aten::_efficient_attention_forward.

    Reference implementation:
        torch/_C/_VariableFunctions.pyi / aten/src/ATen/native/transformers/attention.cpp

    xformers / PyTorch memory-efficient attention forward pass (CUDA/HIP cutlass kernel).
    Called via torch.nn.functional.scaled_dot_product_attention when the
    efficient-attention backend is selected.

    Signature: _efficient_attention_forward(
        query, key, value, bias,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, custom_mask_type, causal,
        scale, seqstart_q, seqstart_k
    ) -> (output, logsumexp, ...)

    Expected Input Dims from trace:
        [0] query  — [B, N_Q, H_Q, d_h_qk],   dtype bf16/fp16
        [1] key    — [B, N_KV, H_KV, d_h_qk], dtype bf16/fp16
        [2] value  — [B, N_KV, H_KV, d_h_v],  dtype bf16/fp16
        [3] bias   — [B, H_Q, N_Q, N_KV] or [] when absent
        [4..13]    — scalars / optional tensors (empty dims)

    Expected Input type from trace:
        ['c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', ...]

    Concrete Inputs[8]  = dropout_p        (float, e.g. '0.')
    Concrete Inputs[9]  = custom_mask_type (int: 0=none, 1=causal, 2=anti-causal)
    Concrete Inputs[10] = causal           (bool string 'True'/'False', legacy alias)

    Roofline -- FLOPs:
        Inherits SDPA.flops_func:
            flops_QK = B * H_Q * 2 * N_Q * N_KV * d_h_qk
            flops_PV = B * H_Q * 2 * N_Q * d_h_v * N_KV
            total    = flops_QK + flops_PV  (halved if causal and N_Q == N_KV)

    Roofline -- bytes moved:
        bytes_read_Q  = B * N_Q  * H_Q  * d_h_qk * bpe
        bytes_read_K  = B * N_KV * H_KV * d_h_qk * bpe
        bytes_read_V  = B * N_KV * H_KV * d_h_v  * bpe
        bytes_write_O = B * N_Q  * H_Q  * d_h_v  * bpe
        Total         = sum of all above
        Note: attention bias read omitted (negligible vs Q/K/V for large N).

    Notes:
        Layout is BNHD (batch, seq, heads, head_dim) — bhnd_idx = (0, 1, 2, 3).
        flash_impl=True: xformers uses flash-style tiled recompute in backward.
        output_bpe == input_bpe (no dtype change in fwd).

    Vendor roofline reference:
        xformers/benchmarks/benchmark_mem_eff_attn.py
    """

    category = "SDPA_fwd"

    def __init__(self, event, **kwargs):
        self.param_details = self.get_param_details(event)
        self.B = self.param_details["B"]
        self.N_Q = self.param_details["N_Q"]
        self.H_Q = self.param_details["H_Q"]
        self.N_KV = self.param_details["N_KV"]
        self.H_KV = self.param_details["H_KV"]
        self.d_h_qk = self.param_details["d_h_qk"]
        self.d_h_v = self.param_details["d_h_v"]

    @staticmethod
    def get_param_details(event):
        from TraceLens.PerfModel.perf_model import extract_sdpa_cfg
        input_dims = event["args"]["Input Dims"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        # Inputs: query=0, key=1, value=2, bias=3 (layout BNHD)
        q_shape = input_dims[0]
        k_shape = input_dims[1]
        v_shape = input_dims[2]
        bhnd_idx = (0, 1, 2, 3)
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (
            sdpa_cfg[key]
            for key in ["B", "N_Q", "H_Q", "N_KV", "H_KV", "d_h_qk", "d_h_v"]
        )

        # custom_mask_type: 0=none, 1=causal, 2=anti-causal (Concrete Inputs[9])
        custom_mask_type = 0
        if len(concrete_inputs) > 9 and concrete_inputs[9] not in ("", "None"):
            try:
                custom_mask_type = int(concrete_inputs[9])
            except (ValueError, TypeError):
                pass

        # Legacy causal bool (Concrete Inputs[10])
        causal_flag = False
        if len(concrete_inputs) > 10 and concrete_inputs[10] not in ("", "None"):
            causal_flag = concrete_inputs[10].strip().lower() == "true"

        is_causal = (custom_mask_type == 1) or causal_flag

        dropout_p = 0.0
        if len(concrete_inputs) > 8 and concrete_inputs[8] not in ("", "None"):
            try:
                dropout_p = float(concrete_inputs[8])
            except (ValueError, TypeError):
                pass

        dtype_A_B = tuple(event["args"]["Input type"][:2])

        return {
            "B": B,
            "N_Q": N_Q,
            "H_Q": H_Q,
            "N_KV": N_KV,
            "H_KV": H_KV,
            "d_h_qk": d_h_qk,
            "d_h_v": d_h_v,
            "dropout": dropout_p,
            "causal": is_causal,
            "flash_impl": True,
            "dtype_A_B": dtype_A_B,
        }

    def flops(self):
        from TraceLens.PerfModel.perf_model import SDPA
        return SDPA.flops_func(
            self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV,
            self.d_h_qk, self.d_h_v, self.param_details["causal"],
        )

    def bytes(self, bytes_per_element=2):
        from TraceLens.PerfModel.perf_model import SDPA
        return SDPA.bytes_func(
            self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV,
            self.d_h_qk, self.d_h_v, self.param_details["causal"],
            bytes_per_element,
        )

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_A_B", [None])[0]
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "matrix"


class aten__efficient_attention_backward:
    """
    Performance model for aten::_efficient_attention_backward.

    Reference implementation:
        torch/_C/_VariableFunctions.pyi / aten/src/ATen/native/transformers/attention.cpp

    xformers / PyTorch memory-efficient attention backward pass.

    Signature: _efficient_attention_backward(
        grad_out, query, key, value, bias, out,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        logsumexp, dropout_p, scale,
        num_splits_key, causal, ...
    ) -> (grad_query, grad_key, grad_value, grad_bias)

    Expected Input Dims from trace:
        [0] grad_out — [B, N_Q, H_Q, d_h_v],   dtype bf16/fp16
        [1] query    — [B, N_Q, H_Q, d_h_qk],  dtype bf16/fp16
        [2] key      — [B, N_KV, H_KV, d_h_qk], dtype bf16/fp16
        [3] value    — [B, N_KV, H_KV, d_h_v],  dtype bf16/fp16
        [4] bias     — [B, H_Q, N_Q, N_KV] or []
        [5] out      — [B, N_Q, H_Q, d_h_v]
        [10] logsumexp — [B, H_Q, N_Q]

    Expected Input type from trace:
        ['c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', ...]

    Concrete Inputs[8]  = max_seqlen_q (int)
    Concrete Inputs[9]  = max_seqlen_k (int)
    Concrete Inputs[11] = dropout_p    (float, e.g. '0.')
    Concrete Inputs[15] = causal       (bool string 'True'/'False')

    Roofline -- FLOPs:
        Inherits SDPA.flops_bwd_func (flash_impl=True adds QK recompute):
            recompute_QK + V_grad + P_grad + Q_grad + K_grad matmuls
            (see SDPA.flops_bwd_func for full breakdown with GQA reduce terms)

    Roofline -- bytes moved:
        bytes_read_Q       = B * N_Q  * H_Q  * d_h_qk * bpe
        bytes_read_K       = B * N_KV * H_KV * d_h_qk * bpe
        bytes_read_V       = B * N_KV * H_KV * d_h_v  * bpe
        bytes_read_O_grad  = B * N_Q  * H_Q  * d_h_v  * bpe
        bytes_write_Q_grad = B * N_Q  * H_Q  * d_h_qk * bpe
        bytes_write_K_grad = B * N_KV * H_KV * d_h_qk * bpe
        bytes_write_V_grad = B * N_KV * H_KV * d_h_v  * bpe
        Total              = sum of all above

    Notes:
        Layout is BNHD — bhnd_idx = (0, 1, 2, 3); query/key/value at indices 1/2/3.
        flash_impl=True: xformers bwd recomputes the attention scores (no P stored).
        output_bpe == input_bpe.

    Vendor roofline reference:
        xformers/benchmarks/benchmark_mem_eff_attn.py
    """

    category = "SDPA_bwd"

    def __init__(self, event, **kwargs):
        self.param_details = self.get_param_details(event)
        self.B = self.param_details["B"]
        self.N_Q = self.param_details["N_Q"]
        self.H_Q = self.param_details["H_Q"]
        self.N_KV = self.param_details["N_KV"]
        self.H_KV = self.param_details["H_KV"]
        self.d_h_qk = self.param_details["d_h_qk"]
        self.d_h_v = self.param_details["d_h_v"]

    @staticmethod
    def get_param_details(event):
        from TraceLens.PerfModel.perf_model import extract_sdpa_cfg
        input_dims = event["args"]["Input Dims"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        # Inputs: grad_out=0, query=1, key=2, value=3, bias=4, out=5 (layout BNHD)
        q_shape = input_dims[1]
        k_shape = input_dims[2]
        v_shape = input_dims[3]
        bhnd_idx = (0, 1, 2, 3)
        sdpa_cfg = extract_sdpa_cfg(q_shape, k_shape, v_shape, bhnd_idx)
        B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v = (
            sdpa_cfg[key]
            for key in ["B", "N_Q", "H_Q", "N_KV", "H_KV", "d_h_qk", "d_h_v"]
        )

        # causal (Concrete Inputs[15])
        is_causal = False
        if len(concrete_inputs) > 15 and concrete_inputs[15] not in ("", "None"):
            is_causal = concrete_inputs[15].strip().lower() == "true"

        dropout_p = 0.0
        if len(concrete_inputs) > 11 and concrete_inputs[11] not in ("", "None"):
            try:
                dropout_p = float(concrete_inputs[11])
            except (ValueError, TypeError):
                pass

        # dtype from query (index 1) and key (index 2)
        dtype_A_B = tuple(event["args"]["Input type"][1:3])

        return {
            "B": B,
            "N_Q": N_Q,
            "H_Q": H_Q,
            "N_KV": N_KV,
            "H_KV": H_KV,
            "d_h_qk": d_h_qk,
            "d_h_v": d_h_v,
            "dropout": dropout_p,
            "causal": is_causal,
            "flash_impl": True,
            "dtype_A_B": dtype_A_B,
        }

    def flops(self):
        from TraceLens.PerfModel.perf_model import SDPA
        return SDPA.flops_bwd_func(
            self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV,
            self.d_h_qk, self.d_h_v,
            self.param_details["causal"],
            self.param_details["flash_impl"],
        )

    def bytes(self, bytes_per_element=2):
        from TraceLens.PerfModel.perf_model import SDPA
        return SDPA.bytes_bwd_func(
            self.B, self.N_Q, self.H_Q, self.N_KV, self.H_KV,
            self.d_h_qk, self.d_h_v,
            self.param_details["causal"],
            bytes_per_element,
        )

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_A_B", [None])[0]
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "matrix"
