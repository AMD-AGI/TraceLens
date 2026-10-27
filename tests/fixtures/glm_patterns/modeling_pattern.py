###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hand-written model whose forwards exercise the graph rules.

Two decoder variants (linear attention plus a dense MLP, full attention plus
MoE), a gated norm, a forget gate, a hyperconnection, an expert loop, and a
vision tower with a two-value rotary return. This is not an upstream model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pattern_sdpa(query, key, value, cu_seqlens):
    """``cu_seqlens`` is an interface input this implementation does not read."""
    del cu_seqlens
    return F.scaled_dot_product_attention(query, key, value)


def build_attention_mask(attention_mask):
    filled = attention_mask.float()
    return filled.masked_fill(attention_mask == 0, -1.0e4)


class PatternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        squared = hidden_states.float().pow(2)
        variance = squared.mean(-1, keepdim=True)
        normed = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normed.to(dtype)


class PatternGatedNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        dtype = hidden_states.dtype
        squared = hidden_states.float().pow(2)
        variance = squared.mean(-1, keepdim=True)
        normed = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        activated = torch.sigmoid(gate)
        return self.weight * normed.to(dtype) * activated


class PatternLinearAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        del layer_idx
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.g_a_proj = nn.Linear(hidden, hidden, bias=False)
        self.g_b_proj = nn.Linear(hidden, hidden, bias=False)
        self.conv_b = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.f_a_proj = nn.Linear(hidden, hidden, bias=False)
        self.f_b_proj = nn.Linear(hidden, hidden, bias=False)
        self.core_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_norm = PatternGatedNorm(hidden)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.cached_state = nn.Parameter(torch.zeros(hidden))

    def forward(self, hidden_states, attention_mask=None):
        masked = hidden_states * attention_mask
        query = self.q_proj(masked)
        key = self.k_proj(masked)
        value = self.v_proj(masked)
        mixed = self.conv_b(masked)
        joined = torch.cat((mixed, key, value), dim=-1)
        branch_a = self.f_a_proj(joined)
        branch_b = self.f_b_proj(joined)
        gate = (branch_a + branch_b) * torch.sigmoid(branch_a)
        opened = self.g_a_proj(hidden_states) + self.g_b_proj(hidden_states).view(
            hidden_states.shape
        )
        _stored = self.cached_state.to(hidden_states.dtype)
        self.cached_state = _stored
        core = self.core_proj(query)
        return self.out_proj(self.o_norm(core + opened, gate))


class PatternFullAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        del layer_idx
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = PatternRMSNorm(hidden)
        self.k_norm = PatternRMSNorm(hidden)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        query = self.q_norm(self.q_proj(hidden_states))
        key = self.k_norm(self.k_proj(hidden_states))
        value = self.v_proj(hidden_states)
        return self.o_proj(pattern_sdpa(query, key, value, None))


class PatternHyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.mix = nn.Linear(hidden, hidden, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden))

    def forward(self, hidden_states, residual):
        mixed = self.mix(hidden_states)
        closed = self.scale * self.scale
        return residual + mixed * closed


class PatternMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(F.silu(gate) * up)


class PatternExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        activated = F.silu(self.gate_proj(hidden_states))
        return self.down_proj(activated)


class PatternMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = PatternExperts(config)
        self.num_experts = config.num_experts

    def forward(self, hidden_states):
        scores = self.router(hidden_states).sigmoid()
        chosen = scores.topk(2, dim=-1)
        mixed = hidden_states + chosen.values.sum(dim=-1, keepdim=True)
        for _index in range(self.num_experts):
            mixed = mixed + self.experts(hidden_states)
        return mixed + hidden_states


class PatternDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if config.is_linear_attn_layer(layer_idx):
            self.self_attn = PatternLinearAttention(config, layer_idx)
        else:
            self.self_attn = PatternFullAttention(config, layer_idx)
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.ffn = PatternMoE(config)
        else:
            self.ffn = PatternMLP(config)
        hidden = config.hidden_size
        self.attn_hc = PatternHyperConnection(config)
        self.ffn_hc = PatternHyperConnection(config)
        self.input_layernorm = PatternRMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PatternRMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, attention_mask=attention_mask)
        hidden_states = self.attn_hc(attn_out, residual)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        return self.ffn_hc(self.ffn(normed), residual)


class PatternRotary(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = getattr(config, "head_dim", 4)
        self.register_buffer("inv_freq", torch.ones(head_dim), persistent=False)

    def forward(self, hidden_states, position_ids):
        freqs = self.inv_freq.float() * position_ids.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class PatternVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.patch_embed = nn.Linear(hidden, hidden, bias=False)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.q_norm = PatternRMSNorm(hidden)
        self.k_norm = PatternRMSNorm(hidden)
        self.rotary_emb = PatternRotary(config)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, pixel_values, grid_thw):
        hidden_states = self.patch_embed(pixel_values)
        cu_seqlens = F.pad(grid_thw.sum(dim=-1).cumsum(dim=0), (1, 0))
        query, key, value = self.qkv(hidden_states).unbind(dim=-1)
        query = self.q_norm(query)
        key = self.k_norm(key)
        cos, sin = self.rotary_emb(hidden_states, grid_thw)
        query_embed = (query * cos).to(query.dtype).transpose(1, 2)
        key_embed = (key * sin).to(key.dtype).transpose(1, 2)
        attended = pattern_sdpa(query_embed, key_embed, value, cu_seqlens)
        projected = self.o_proj(attended)
        gate = self.gate_proj(projected)
        up = self.up_proj(projected)
        hidden_states = self.down_proj(F.silu(gate) * up)
        state = hidden_states
        positions = grid_thw
        for _step in range(2):
            state = state + positions
            positions = positions + grid_thw
        return state


class PatternTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                PatternDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = PatternRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.visual = PatternVisionModel(config.vision_config)

    def forward(self, input_ids, attention_mask, pixel_values, grid_thw):
        image = self.visual(pixel_values, grid_thw)
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + image.mean(dim=1, keepdim=True)
        mask = build_attention_mask(attention_mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=mask)
        return self.norm(hidden_states)


class PatternForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = PatternTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask, pixel_values, grid_thw):
        hidden_states = self.model(input_ids, attention_mask, pixel_values, grid_thw)
        return self.lm_head(hidden_states)
