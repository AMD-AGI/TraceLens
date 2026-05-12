###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared helper functions for trace_breakdown scripts.
"""


def build_rle(kernel_indices, cls_by_idx):
    """Run-length encode kernel indices by perf_category.

    Returns list of (perf_category, count, [kernel_indices], [kernel_types]).
    """
    if not kernel_indices:
        return []

    def _cls(idx):
        c = cls_by_idx.get(idx, {})
        return c.get("perf_category", "Others"), c.get("kernel_type", "Unknown")

    first_cat, first_kt = _cls(kernel_indices[0])
    groups = []
    cur_cat = first_cat
    cur_indices = [kernel_indices[0]]
    cur_types = [first_kt]

    for idx in kernel_indices[1:]:
        cat, kt = _cls(idx)
        if cat == cur_cat:
            cur_indices.append(idx)
            cur_types.append(kt)
        else:
            groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
            cur_cat = cat
            cur_indices = [idx]
            cur_types = [kt]

    groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
    return groups


def detect_period(rle_groups):
    """Find the shortest repeating period in an RLE group sequence.

    Returns the period length (number of RLE groups per super-cycle).
    """
    cats = [g[0] for g in rle_groups]
    n = len(cats)
    if n < 6:
        return n

    for p in range(3, n // 2 + 1):
        prefix = cats[:p]
        matches = sum(1 for i in range(p, n) if cats[i] == prefix[i % p])
        total = n - p
        if total > 0 and matches / total > 0.85 and total >= 2 * p:
            return p

    return n


def infer_model_info(cls_by_idx, num_layers, config=None, is_graph_mode=False):
    """Build model_info dict from data, optionally enriched by HF config."""
    architecture = "unknown"
    ffn_type = "unknown"
    layer_types = None

    if config:
        tc = config.get("text_config", config)
        architecture = tc.get("model_type", config.get("model_type", "unknown"))
        num_experts = tc.get("num_experts", tc.get("num_local_experts", 0))
        ffn_type = "moe" if num_experts > 0 else "dense"
        layer_types = tc.get("layer_types")

    if architecture == "unknown":
        moe_indicators = {
            "MoE GEMM",
            "MoE Finalize",
            "MoE Routing",
            "MoE Quantize",
        }
        ktypes = {c.get("kernel_type", "") for c in cls_by_idx.values()}
        ffn_type = "moe" if ktypes & moe_indicators else "dense"

    info = {
        "architecture": architecture,
        "num_layers": num_layers,
        "ffn_type": ffn_type,
        "graph_mode": is_graph_mode,
    }
    if layer_types:
        info["layer_types"] = layer_types
    return info
