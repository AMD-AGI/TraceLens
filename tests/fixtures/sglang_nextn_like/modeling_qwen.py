###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Offline SGLang-style block whose NextN draft is a dense layer."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.o_proj(self.qkv_proj(hidden_states))


class FusedMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False
        )

    def forward(self, hidden_states):
        return hidden_states


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        return self.down_proj(self.gate_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FusedMoE(config)

    def forward(self, hidden_states):
        return self.mlp(self.self_attn(hidden_states))


class NextNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size)
        self.hnorm = RMSNorm(config.hidden_size)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.self_attn = Attention(config)
        self.mlp = DenseMLP(config)

    def forward(self, hidden_states):
        return self.mlp(self.self_attn(self.eh_proj(hidden_states)))


class QwenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.nextn = NextNLayer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.nextn(hidden_states))
