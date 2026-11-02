"""Offline vLLM-style DeepSeek block with latent attention and an MTP draft."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class DeepseekAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


class FusedMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.moe_intermediate_size = config.moe_intermediate_size
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states):
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = DeepseekAttention(config)
        self.mlp = FusedMoE(config)

    def forward(self, hidden_states):
        return self.mlp(self.self_attn(self.input_layernorm(hidden_states)))


class MultiTokenPredictorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size)
        self.hnorm = RMSNorm(config.hidden_size)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = DecoderLayer(config)

    def forward(self, hidden_states):
        return self.mtp_block(self.eh_proj(hidden_states))


class DeepseekForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.mtp = MultiTokenPredictorLayer(config)
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.mtp(hidden_states)
        return self.lm_head(self.norm(hidden_states))
