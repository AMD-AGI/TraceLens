###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hand-written chunk_kda stub. The body only has to parse."""

from fla.ops.kda.chunk_fwd import chunk_kda_fwd


class triton:
    @staticmethod
    def jit(fn):
        return fn


@triton.jit
def _l2norm_kernel(x):
    var = sum(x)
    rstd = 1 / sqrt(var)
    out = x * rstd
    return out


def l2norm_fwd(x):
    return _l2norm_kernel(x)


@triton.jit
def _fused_beta_kernel(beta, scale):
    activated = sigmoid(beta)
    out = activated * scale
    return out


def fused_beta_sigmoid(beta, scale=1.0):
    return _fused_beta_kernel(beta, scale)


def fused_recurrent_kda(q, k, v, g, beta):
    return q, None


@triton.jit
def _kda_gate_kernel(g):
    out = exp(g) * softplus(g)
    activated = sigmoid(g)
    cum = cumsum(activated)
    return out, cum


def kda_gate_chunk_cumsum(g):
    return _kda_gate_kernel(g)


class ChunkKDAFunction:
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta, scale=1.0)
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(g)
        return chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            use_gate_in_kernel=use_gate_in_kernel,
        )

    @staticmethod
    def apply(*args, **kwargs):
        return ChunkKDAFunction.forward(None, *args, **kwargs)


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
):
    return ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
    )
