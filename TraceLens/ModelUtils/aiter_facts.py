###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Facts the aiter checkout an engine was built against fixes for the MoE launch.

aiter picks the second MoE GEMM per padded token count from its tuned
configuration tables (``get_2stage_cfgs``), and the choice decides which
kernels run and what they are called with. The tables, the padding tiers, and
the Opus decode kernel's tile contract are all in the checkout, so they are read
there. A fact the checkout does not give is absent.
"""

from __future__ import annotations

import ast
import csv
import re

from TraceLens.ModelUtils.engine_facts import (
    _evaluate,
    _Unresolved,
    module_constant,
)
from TraceLens.ModelUtils.github import (
    GitHubRef,
    fetch_github_file,
    list_github_files,
)

_REPO = ("ROCm", "aiter")
_FUSED_MOE = "aiter/fused_moe.py"
_OPUS_COMMON = "csrc/opus_moe/opus_moe_common.py"
_OPUS_STAGE2 = "aiter/ops/opus/moe_stage2_a8w4.py"
_CONTRACT = "OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT"

# ``AITER_CONFIG.get_config_file`` merges ``configs/tuned_fmoe.csv`` with every
# ``configs/model_configs/*tuned_fmoe*.csv`` that is not an untuned list.
_TUNED_TABLES = r"^aiter/configs/(model_configs/.*tuned_fmoe.*|tuned_fmoe)\.csv$"

# aiter's names for the dtypes a checkpoint's expert format selects.
_FP4_WEIGHT = "torch.float4_e2m1fn_x2"
_PER_GROUP_32 = "QuantType.per_1x32"


def _text(ref: GitHubRef, path: str) -> str | None:
    found = fetch_github_file(ref, path)
    return found.read_text(encoding="utf-8") if found is not None else None


def call_keywords(source: str, name: str) -> dict:
    """Literal keyword arguments of the module-level ``name = Class(...)`` call."""
    for statement in ast.parse(source).body:
        if not (
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets)
        ):
            continue
        found = {}
        for keyword in statement.value.keywords:
            try:
                found[keyword.arg] = _evaluate(keyword.value, {})
            except _Unresolved:
                continue
        return found
    raise KeyError(name)


def tuned_moe_rows(
    ref: GitHubRef,
    *,
    gfx: str,
    cu_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
) -> tuple[tuple[int, int, str, str], ...]:
    """``(padded tokens, block_m, stage-1 kernel, stage-2 kernel)`` of the tuned
    rows for FP4 experts of this shape.

    Rows from every merged table that match the key ``get_2stage_cfgs`` builds.
    When tables repeat a shape, the merge keeps the fastest row.
    """
    best: dict[int, tuple[float, int, str, str]] = {}
    for path in list_github_files(ref, _TUNED_TABLES):
        text = _text(ref, path)
        if text is None:
            continue
        for row in csv.DictReader(text.splitlines()):
            if (
                row.get("gfx", gfx) != gfx
                or row.get("q_dtype_w") != _FP4_WEIGHT
                or row.get("q_type") != _PER_GROUP_32
                or (row.get("cu_num"), row.get("model_dim"), row.get("inter_dim"))
                != (str(cu_num), str(model_dim), str(inter_dim))
                or (row.get("expert"), row.get("topk")) != (str(experts), str(topk))
            ):
                continue
            try:
                token, micros = int(row["token"]), float(row.get("us") or "inf")
                block_m = int(row["block_m"])
            except (KeyError, ValueError):
                continue
            if token not in best or micros < best[token][0]:
                best[token] = (micros, block_m, row["kernelName1"], row["kernelName2"])
    return tuple(sorted((token, *rest) for token, (_, *rest) in best.items()))


def fallback_tiles(source: str) -> tuple[tuple[int | None, int], ...]:
    """``(token bound, tile_m)`` of the heuristic FlyDSL pick in ``get_2stage_cfgs``.

    The function chooses ``tile_m`` with an ``if token < N`` chain that assigns a
    tuple whose first target is ``_tile_m``. The last entry has no bound.
    """
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "get_2stage_cfgs"
    )

    def tile(body):
        for statement in body:
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Tuple)
                and isinstance(statement.targets[0].elts[0], ast.Name)
                and statement.targets[0].elts[0].id == "_tile_m"
            ):
                return _evaluate(statement.value, {})[0]
        return None

    def bound(test):
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "token"
            and isinstance(test.ops[0], ast.Lt)
        ):
            return _evaluate(test.comparators[0], {})
        return None

    for node in ast.walk(function):
        if not isinstance(node, ast.If) or tile(node.body) is None:
            continue
        tiers = []
        while True:
            tiers.append((bound(node.test), tile(node.body)))
            rest = node.orelse
            if len(rest) == 1 and isinstance(rest[0], ast.If):
                node = rest[0]
                continue
            tiers.append((None, tile(rest)))
            return tuple(tiers)
    raise KeyError("_tile_m")


def fallback_stage2_mode(source: str) -> str:
    """The reduction mode of the heuristic FlyDSL stage 2: the last argument of
    ``flydsl_kernel_name(2, ...)`` assigned to ``_base_kn2``."""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_base_kn2"
            and isinstance(node.value, ast.Call)
        ):
            return _evaluate(node.value.args[-1], {})
    raise KeyError("_base_kn2")


def aiter_moe_facts(
    ref: GitHubRef,
    *,
    gfx: str,
    cu_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
) -> tuple[tuple[str, object], ...]:
    """The aiter MoE facts for FP4 experts of this shape, as ``(name, value)``."""
    facts: list[tuple[str, object]] = []
    rows = tuned_moe_rows(
        ref,
        gfx=gfx,
        cu_num=cu_num,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
    )
    if rows:
        facts.append(("moe_tuned_rows", rows))
    try:
        fused = _text(ref, _FUSED_MOE)
        if fused is not None:
            facts.append(
                ("moe_padded_tiers", module_constant(fused, "_PADDED_M_TIERS"))
            )
            facts.append(("moe_fallback_tiles", fallback_tiles(fused)))
            facts.append(("moe_fallback_stage2_mode", fallback_stage2_mode(fused)))
    except (KeyError, SyntaxError, StopIteration, _Unresolved):
        pass
    try:
        common = _text(ref, _OPUS_COMMON)
        if common is not None:
            facts.append(("opus_stage2_contract", call_keywords(common, _CONTRACT)))
    except (KeyError, SyntaxError):
        pass
    stage2 = _text(ref, _OPUS_STAGE2)
    if stage2 is not None:
        # ``compile_ops`` registers a custom op whose first parameter is not
        # annotated with the ``Tensor`` class (a postponed annotation is a
        # string), and such an op is called with a one-element dummy tensor.
        postponed = re.search(
            r"^from __future__ import annotations$", stage2, re.MULTILINE
        )
        facts.append(("opus_stage2_dummy_arg", postponed is not None))
    return tuple(facts)


def moe_facts_for_model(model, commit: str | None) -> tuple[tuple[str, object], ...]:
    """The aiter MoE facts for a :class:`GraphModel` with FP4 experts."""
    if commit is None or model.expert_dtype != "fp4" or not model.num_cus:
        return ()
    return aiter_moe_facts(
        GitHubRef(*_REPO, ref=commit),
        gfx=model.gfx_arch,
        cu_num=model.num_cus,
        model_dim=model.hidden_size,
        inter_dim=model.moe_intermediate_size // model.tp,
        experts=model.n_routed_experts + model.n_shared_experts,
        topk=model.num_experts_per_tok + model.n_shared_experts,
    )
