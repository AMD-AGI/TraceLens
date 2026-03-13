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
    """

    REQUIRED_PARAM_KEYS = (
        "B", "N_Q", "H_Q", "N_KV", "H_KV", "d_h_qk", "d_h_v",
        "c_sq", "c_sk", "c_sqsq", "c_sqsk",
        "g_sq", "g_sk", "g_sqsq", "g_sqsk",
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
    def get_param_details(event):
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

        return {
            "B": 1,
            "N_Q": N_Q,
            "H_Q": H_Q,
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
        }

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
    def bytes_func(B, H_Q, H_KV, d_h_qk, d_h_v,
                   c_sq, c_sk, g_sq, g_sk, bytes_per_element):
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
            B * c_sq * H_Q * d_h_qk        # Q read
            + B * c_sk * H_KV * d_h_qk     # K read
            + B * c_sk * H_KV * d_h_v       # V read
            + B * c_sq * H_Q * d_h_v        # output write
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
        if self.param_details["c_sq"] == 0 and self.param_details["g_sq"] == 0:
            raise NotImplementedError(
                "Attention perf model for decode phase requires custom annotations"
            )
        return self.flops_func(
            self.H_Q, self.d_h_qk, self.d_h_v,
            self.param_details["c_sqsk"],
            self.param_details["c_sqsq"],
            self.param_details["g_sqsk"],
        )

    def bytes(self, bytes_per_element=2):
        return self.bytes_func(
            self.B, self.H_Q, self.H_KV, self.d_h_qk, self.d_h_v,
            self.param_details["c_sq"], self.param_details["c_sk"],
            self.param_details["g_sq"], self.param_details["g_sk"],
            bytes_per_element,
        )

    def flops_bwd(self):
        raise NotImplementedError("Backward pass for attention is not defined.")

    def bytes_bwd(self):
        raise NotImplementedError("Backward pass for attention is not defined.")

    def get_compute_precision(self):
        dtype = self.param_details.get("dtype_A_B", [None])[0]
        return torch_dtype_map(dtype) if dtype else None

    def get_maf_type(self):
        return "matrix"


class vllm_unified_attention_with_output(InferenceAttention):
    """Attention perf model for vLLM unified_attention_with_output events."""
    pass


class mha_varlen_fwd(InferenceAttention):
    pass

class mla_decode_fwd(InferenceAttention):
    pass
