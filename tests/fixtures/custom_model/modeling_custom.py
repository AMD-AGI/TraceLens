###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom modeling file for AST inspection tests."""

import torch
from torch import nn


class CustomRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_scaling = 1.0
        self.register_buffer(
            "inv_freq", torch.ones(config.hidden_size), persistent=False
        )

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq.float() @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(1)


class CustomLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_lora_rank = config.kv_lora_rank
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.kv_proj = nn.Linear(config.hidden_size, config.kv_lora_rank)


class CustomSharedExpertMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        self.shared_expert = nn.Linear(config.hidden_size, config.moe_intermediate_size)


class CustomDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = CustomRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CustomLatentAttention(config)
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.block_sparse_moe = CustomSharedExpertMoE(config)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CustomForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = CustomRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [CustomDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
