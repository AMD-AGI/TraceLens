###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hand-written Kimi-shaped decoder used by source-only graph tests.

Four layers: KDA+MLP, two KDA+MoE, one MLA+MoE. The kernel call matches the
flags and ports the pipeline tests read. This is not the upstream model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.kda import chunk_kda, fused_recurrent_kda


def rearrange(tensor, pattern, **axes):
    return tensor


class FusedRMSNormGated(nn.Module):
    """External gated norm with no Python body, so it stays one RMSNorm node."""

    def __init__(self, hidden_size, eps=1e-6, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = activation


class KimiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class KimiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(self.gate_proj(hidden_states))


class KimiMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_experts))
        self.top_k = config.num_experts_per_token
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_router_activation_func = config.moe_router_activation_func
        self.moe_renormalize = config.moe_renormalize
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden)
        scores = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            raise NotImplementedError
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = KimiMoEGate(config)
        self.experts = KimiMLP(config)

    def forward(self, hidden_states):
        topk_idx, topk_weight = self.gate(hidden_states)
        return self.experts(hidden_states) * topk_weight.sum(dim=-1, keepdim=True)


class KimiMLAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = KimiRMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=False
        )
        self.kv_a_layernorm = KimiRMSNorm(config.kv_lora_rank)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        key = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        return self.o_proj(query + key)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.mode = "chunk"
        self.layer_idx = layer_idx
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_conv1d = nn.Conv1d(hidden, hidden, 3)
        self.k_conv1d = nn.Conv1d(hidden, hidden, 3)
        self.v_conv1d = nn.Conv1d(hidden, hidden, 3)
        self.f_a_proj = nn.Linear(hidden, hidden, bias=False)
        self.f_b_proj = nn.Linear(hidden, hidden, bias=False)
        self.b_proj = nn.Linear(hidden, hidden, bias=False)
        self.A_log = nn.Parameter(torch.empty(hidden))
        self.dt_bias = nn.Parameter(torch.empty(hidden))
        self.o_norm = FusedRMSNormGated(
            hidden, eps=config.rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.head_dim = config.linear_attn_config["head_dim"]

    def forward(self, hidden_states, attention_mask=None, cache_params=None):
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if use_cache and q_len == 1 else self.mode
        q = self.q_conv1d(self.q_proj(hidden_states))
        k = self.k_conv1d(self.k_proj(hidden_states))
        v = self.v_conv1d(self.v_proj(hidden_states))
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states)
        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k)
        )
        if mode == "chunk":
            o, _state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )
        else:
            o, _state = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta)
        return self.o_proj(self.o_norm(o))


class KimiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(config, layer_idx)
        elif config.is_mla:
            self.self_attn = KimiMLAAttention(config, layer_idx)
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states


class KimiLinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class KimiLinearForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = KimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.model(input_ids=input_ids, **kwargs)
        return self.lm_head(hidden_states)
