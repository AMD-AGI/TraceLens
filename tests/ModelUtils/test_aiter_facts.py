###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MoE launch facts come from the aiter checkout the engine's build pins."""

from dataclasses import replace

from TraceLens.ModelUtils import aiter_facts
from TraceLens.ModelUtils.aiter_facts import (
    aiter_moe_facts,
    call_keywords,
    moe_facts_for_model,
    fallback_stage2_mode,
    fallback_tiles,
    tuned_moe_rows,
)
from TraceLens.ModelUtils.github import GitHubRef

_HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,q_dtype_w,q_type,us,"
    "block_m,kernelName1,kernelName2"
)
_FP4 = "torch.float4_e2m1fn_x2,QuantType.per_1x32"

_TABLES = {
    "aiter/configs/tuned_fmoe.csv": [
        _HEADER,
        f"gfx950,256,1,7168,384,385,7,{_FP4},9.0,32,cktile_1,slow_stage2",
        f"gfx950,256,4,7168,384,385,7,{_FP4},9.0,32,flydsl_1,reduce_stage2",
        # Another device, another shape, another weight format.
        f"gfx942,304,1,7168,384,385,7,{_FP4},1.0,32,k1,other_device",
        f"gfx950,256,1,7168,512,385,7,{_FP4},1.0,32,k1,other_shape",
        "gfx950,256,1,7168,384,385,7,torch.int8,QuantType.per_Token,1.0,32,k1,other_dtype",
    ],
    "aiter/configs/model_configs/dsv4_tuned_fmoe.csv": [
        _HEADER,
        f"gfx950,256,1,7168,384,385,7,{_FP4},5.0,16,cktile_1,opus_moe2_atomic",
    ],
}

_OPUS_COMMON = """
OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT = OpusA8W4KernelContract(
    name="gfx950_a8w4_decode_v1",
    bk_logical=256,
    fp4_values_per_byte=2,
    mfma_m=16,
    scale_groups_per_row_pack=8,
)
"""


_FUSED_MOE = """
_PADDED_M_TIERS = [32768, 131072]


def get_2stage_cfgs(token):
    if token < 2048:
        _tile_m, _s1, _s2 = 16, "", "_bnt2"
    elif token < 4096:
        _tile_m, _s1, _s2 = 64, "", ""
    else:
        _tile_m, _s1, _s2 = 128, "", ""
    _base_kn2 = flydsl_kernel_name(2, "a", "w", "o", _tile_m, 128, 256, "atomic")
"""


def _serve(monkeypatch, stage2_header="from __future__ import annotations\n"):
    texts = {
        "aiter/fused_moe.py": _FUSED_MOE,
        "csrc/opus_moe/opus_moe_common.py": _OPUS_COMMON,
        "aiter/ops/opus/moe_stage2_a8w4.py": stage2_header + "import torch\n",
    }
    texts.update({path: "\n".join(rows) for path, rows in _TABLES.items()})

    class _File:
        def __init__(self, text):
            self._text = text

        def read_text(self, encoding=None):
            return self._text

    monkeypatch.setattr(
        aiter_facts,
        "fetch_github_file",
        lambda ref, path: _File(texts[path]) if path in texts else None,
    )
    monkeypatch.setattr(
        aiter_facts,
        "list_github_files",
        lambda ref, pattern: [path for path in sorted(_TABLES) if "tuned_fmoe" in path],
    )


_REF = GitHubRef("ROCm", "aiter", "abc")
_SHAPE = dict(
    gfx="gfx950", cu_num=256, model_dim=7168, inter_dim=384, experts=385, topk=7
)


def test_call_keywords_reads_the_literal_arguments_of_the_contract():
    contract = call_keywords(_OPUS_COMMON, "OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT")
    assert contract["bk_logical"] == 256
    assert contract["scale_groups_per_row_pack"] == 8


def test_tuned_rows_match_the_device_and_shape_and_keep_the_fastest(monkeypatch):
    _serve(monkeypatch)
    assert tuned_moe_rows(_REF, **_SHAPE) == (
        (1, 16, "cktile_1", "opus_moe2_atomic"),
        (4, 32, "flydsl_1", "reduce_stage2"),
    )
    assert tuned_moe_rows(_REF, **{**_SHAPE, "experts": 999}) == ()


def test_moe_facts_collect_the_table_the_tiers_and_the_opus_contract(monkeypatch):
    _serve(monkeypatch)
    facts = dict(aiter_moe_facts(_REF, **_SHAPE))
    assert facts["moe_tuned_rows"][0] == (1, 16, "cktile_1", "opus_moe2_atomic")
    assert facts["moe_padded_tiers"] == (32768, 131072)
    assert facts["opus_stage2_contract"]["mfma_m"] == 16
    assert facts["opus_stage2_dummy_arg"] is True


def test_a_wrapper_with_the_tensor_class_annotation_takes_no_dummy_argument(
    monkeypatch,
):
    _serve(monkeypatch, stage2_header="")
    assert dict(aiter_moe_facts(_REF, **_SHAPE))["opus_stage2_dummy_arg"] is False


def test_model_facts_need_fp4_experts_a_device_and_a_commit(monkeypatch):
    from tests.test_graph_schedule import _pro

    _serve(monkeypatch)
    device = replace(_pro(), num_cus=256, gfx_arch="gfx950")
    assert dict(moe_facts_for_model(device, "abc"))["moe_tuned_rows"]
    assert moe_facts_for_model(device, None) == ()
    assert moe_facts_for_model(_pro(), "abc") == ()
    assert moe_facts_for_model(replace(device, expert_dtype=""), "abc") == ()


def test_the_heuristic_flydsl_pick_is_read_from_the_dispatch_function():
    assert fallback_tiles(_FUSED_MOE) == ((2048, 16), (4096, 64), (None, 128))
    assert fallback_stage2_mode(_FUSED_MOE) == "atomic"


def test_the_moe_facts_carry_the_heuristic_pick(monkeypatch):
    _serve(monkeypatch)
    facts = dict(aiter_moe_facts(_REF, **_SHAPE))
    assert facts["moe_fallback_tiles"][-1] == (None, 128)
    assert facts["moe_fallback_stage2_mode"] == "atomic"
