###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
    def get_param_details(event):
        try:
            annotation = str(event.get("annotation"))
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

            input_dims = event["args"]["Input Dims"]
            q_shape, k_shape = input_dims[0], input_dims[1]
            N_Q, H_Q, d_h_qk = q_shape
            N_KV, H_KV, d_h_v = k_shape[-3:]
            dtype_Q = event["args"]["Input type"][0]
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
        B, H_Q, H_KV, d_h_qk, d_h_v, c_sq, c_sk, g_sq, g_sk, bytes_per_element
    ):
        """Calculate bytes moved for attention (context + generation).

        Args:
            B: Batch size.
            H_Q / H_KV: Number of query / key-value heads.
            d_h_qk / d_h_v: Head dimensions for Q-K / V.
            c_sq / c_sk: Aggregate sequence lengths for context Q / KV.
            g_sq / g_sk: Aggregate sequence lengths for generation Q / KV.
            bytes_per_element: Bytes per tensor element.
        """
        ctx_elems = (
            B * c_sq * H_Q * d_h_qk  # Q read
            + B * c_sk * H_KV * d_h_qk  # K read
            + B * c_sk * H_KV * d_h_v  # V read
            + B * c_sq * H_Q * d_h_v  # output write
        )
        gen_elems = (
            B * g_sq * H_Q * d_h_qk
            + B * g_sk * H_KV * d_h_qk
            + B * g_sk * H_KV * d_h_v
            + B * g_sq * H_Q * d_h_v
        )
        return (ctx_elems + gen_elems) * bytes_per_element

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


class mla_decode_fwd(InferenceAttention):
    pass


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
