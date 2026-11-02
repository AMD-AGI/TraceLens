"""An SGLang-style block that imports FusedMoE from a sibling module."""

from .moe import FusedMoE


class QwenMoeBlock:
    def __init__(self):
        self.experts = FusedMoE(num_experts=64)
