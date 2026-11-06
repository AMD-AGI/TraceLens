###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Imported forward handoff for the chunk_kda stub."""


def chunk_gated_delta_rule_fwd_h(q, k, v, g, beta):
    return q


def chunk_gla_fwd_o_gk(h, g):
    return h


def chunk_kda_fwd(q, k, v, g, beta, use_gate_in_kernel=False):
    h = chunk_gated_delta_rule_fwd_h(q, k, v, g, beta)
    o = chunk_gla_fwd_o_gk(h, g)
    return o
