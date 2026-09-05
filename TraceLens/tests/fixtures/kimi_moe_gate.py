###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            raise NotImplementedError

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1:
            grouped = scores_for_choice.view(
                batch * sequence,
                self.num_expert_group,
                -1,
            )
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            scores_for_choice = grouped.masked_fill(
                ~group_mask.bool().unsqueeze(-1),
                0.0,
            ).flatten(1)

        _, topk_idx = torch.topk(
            scores_for_choice,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight
