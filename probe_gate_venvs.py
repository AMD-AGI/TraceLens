###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run the 4-model gate with each model in the environment it actually needs.

``probe_gate_reconciled.py`` runs every model in whatever interpreter invoked it,
so a model whose own code cannot import there is silently exercised on a
degraded path -- and any defect that only appears once its real dependencies are
present is invisible. This driver asks ``TraceLens.ModelUtils.model_env`` the same
question the exporter asks at run time, and re-runs the per-model gate in the
provisioned environment when the answer is "not here".

Usage:  python probe_gate_venvs.py [model-id ...]
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
MODELS = [
    "zai-org/GLM-5.3-Flash",
    "moonshotai/Kimi-K3",
    "MiniMaxAI/MiniMax-M3",
    "deepseek-ai/DeepSeek-V4-Flash",
    "openai-community/gpt2",
]

# Node count each model is expected to render, so a silent structural change is
# caught even when every check still reports zero. Counts are stable run to run
# on all four models again as of 18148ab2, which removed the last of the
# hash-order and id-reuse leaks; verify with probe_digest.py under two
# PYTHONHASHSEED values before trusting a count delta.
# A mismatch is not automatically a regression -- it needs explaining, and if the
# new number is right, updating here is part of the change.
# Measured 2026-10-01 after the branch-merge rename + no-op merge/cast elision.
# Deltas from the previous recorded numbers, each accounted for:
#   GLM      1267 -> 1288 (loop ports + body boundaries, 93630ddd/35b61fc9/9788075f)
#            1288 -> 1276 (-12 no-op merges/casts folded onto their producer)
#   MiniMax   650 ->  667 (same boundary commits) ->  660 (-7 folded)
#   DeepSeek 2248 -> 2336 (same boundary commits) -> 2327 (-9 folded)
#   Kimi      612 unchanged
# Then +1 output tile per block a tensor leaves without one (GLM +2, Kimi +13,
# MiniMax +3, DeepSeek +5); `block boundaries` fell 4->2 / 42->26 / 4->0 / 8->0.
# Then the entering side of the same rule (GLM +2, Kimi +10): `block boundaries`
# is 0 on all four.
# Then the tensor constructors that were never labelled at all (GLM +2 for the
# indexer's zeros/full, MiniMax +2 for the mask's zeros).
# Then GLM 1282 -> 1279: the mask builder's two ops (which had no mask to read
# and could not be sized) give way to the one `torch.ones` the model runs.
# Every check is now zero on all four.
EXPECTED_NODES = {
    # 2026-10-16b -- a parameter the model annotates as a plain Python value is
    # not the method's primary TENSOR parameter, and a string-keyed dict lookup
    # is not a tensor index.
    # DeepSeek's KV cache names the entry it addresses first
    # (`store_compression_weights(self, name: str, kv, gate)`), so `name` was
    # taken as the primary, `kv`/`gate` were demoted, and `self.entry_count[
    # name]` / `self.buffer_kv[name]` -- host bookkeeping over a dict -- were
    # emitted as tensor index ops. With no integer index they fell to a MoE
    # fallback that stamps `num_experts_per_tok` on them, which is where the
    # cache's `[B, usable, 6]` came from. It is now the one `kv[:, :usable]`
    # the method really returns, `[B, usable, 1024]`.
    # DeepSeek 2558 -> 2418. 165 nodes removed, 25 added; 25 of the removals are
    # the SAME nodes re-identified when the frame changed shape (13 Slice,
    # 8 Clone, 4 overlap_kv @output), so the real removal is 140:
    #     18  `Indexing` ops fabricated for string-keyed dict lookups
    #     15  constants only those ops read (entry_count/compress_rate/buffer_gate)
    #     50  boundary + mirror tiles for `name` and for the mis-assigned
    #         `hidden_states`/`hidden_states_2` primaries
    #     36  `result_1`/`result_2` return-slot tiles and their mirrors
    #     21  `overlap_kv`/`overlap_gate` state ops/outputs and 5 Multiply
    # No other model calls a method whose first parameter is annotated as a
    # plain value, so none moves.
    # 2026-10-16 -- the live module tree says which submodules a variant HAS.
    # An absent component and a component with no override were the same state
    # (no entry in `component_trees`), and the renderer's fallback for "no
    # override" is to draw the DEFAULT component list -- which contains both
    # arms. So every Kimi variant group drew an `mlp` AND a `block_sparse_moe`,
    # including the one whose own label says "no mlp".
    # Kimi 648 -> 517, every one of the 131 accounted:
    #     66  1x group: block_sparse_moe (that layer has `mlp`, and was never
    #         built with a MoE block at all)
    #     25  24x group: mlp     (those layers have block_sparse_moe)
    #     25  68x group: mlp     (same)
    #      6  SituAndMul frame boundary tiles, 2 per deleted MLP subtree
    #      3  68x self_attn: g_a_proj, g_b_proj, @op_l651_c8_merge
    #      3  1x  self_attn: the same three
    #      3  @op_l965_c8_merge, one per variant group
    # The gate pair goes for the same reason: this checkpoint sets
    # `use_full_rank_gate: true`, so `g_proj` is constructed and `g_a_proj` /
    # `g_b_proj` never are. A phi left with one arm folds onto it rather than
    # being drawn as a choice between one thing.
    # The other four models do not move. DeepSeek takes the synthesize path (its
    # AST variant count disagrees with the live buckets) so it never reached the
    # early return that caused this, and its four variant trees are intact.
    # NOTE, because it cost a wrong first attempt: a child is only evidence of
    # absence if it is BOTH drawn AND in the owner's `init_assignments`.
    # `moe_infer` is a METHOD the forward calls -- `named_modules()` would never
    # list it -- and pruning on live-tree absence alone deleted 22 LIVE nodes
    # from the two variants that really do have a MoE block.
    # NOTE also: the frame census did NOT catch this and correctly reports 0
    # missing, because it tracks `@fn_`/`@positional_` FREE-FUNCTION frames and
    # these were module components. A 66-node module subtree can vanish without
    # it noticing; EXPECTED_NODES is the only guard that covers that case, and
    # it is what fired here.
    # 2026-10-15d -- a method called on a LOCAL object is read like any other.
    # DeepSeek's compressor and indexer run `cache_layer.store_compression_
    # weights(...)` / `update_overlap_state(...)` on their KV cache, both
    # defined in the model's own source. The receiver is not `self`, so no
    # submodule path saw them: the calls vanished, their results had no
    # producer, and whatever read them fell back on the op that happened to
    # precede it -- a `copy_` drawn reading the gate chunk the model only
    # PASSES to the cache, and every phi over a cached value taking the
    # projection as its arm.
    # DeepSeek 2393 -> 2558, every one of the 165 accounted: 44 ops the methods
    # really run, 90 boundary tiles for what crosses in/out, 23 constants they
    # read, and 8 `@state_out` sinks for the values written back into
    # `overlap_kv`/`overlap_gate` (nothing in THIS forward reads those; the
    # next call does). No other model calls a method on a local object whose
    # class this file defines, so none moves.
    # 2026-10-15c -- an op that READS a forward parameter declares it. Only a
    # kernel's `inputs:` detail counted as a declaration, so a plain op naming
    # a parameter (`attention_mask[:, :, None]` records it in `param_inputs`)
    # fell back to the frame's primary input: GLM's
    # `apply_mask_to_padding_states` had ONE tile, named `attention_mask` and
    # carrying the hidden state, so the unsqueeze that should read a `[B, S]`
    # mask read `[B, S, 4096]` and the whole frame came out rank 4.
    # GLM 1311 -> 1319, +4 per linear-attention variant (2 of them): the frame's
    # one mis-named tile becomes two (`hidden_states` + `attention_mask`), and
    # the mask gains a tile on each box it crosses to get there. No other model
    # has a frame reading a second parameter this way, so none moves.
    # 2026-10-15b -- mutually-exclusive layer variants converge in a phi. A
    # repeat stack whose layers are not all alike is drawn as one group per
    # variant, and each group's output was wired straight to whatever follows
    # the stack: GLM's `hc_head`, a `mean`, appeared to read three tensors at
    # once. Exactly one variant is what any layer IS, so they are alternatives
    # and converge in the `Merge` phi the graph already uses for the arms of an
    # unresolved `if` -- displayed as "Mutually exclusive branch".
    # +1 per stack with variants: GLM 1310->1311, Kimi 647->648,
    # MiniMax 721->722, DeepSeek 2392->2393. GPT-2 has one uniform stack and
    # does not move. Nothing else changes; 6 GLM phis are relabelled in place.
    # 2026-10-15 -- an op is wired the operands it declares, not every argument
    # the call carries. MiniMax 727 -> 721, and nothing else moves (GLM 1310,
    # Kimi 647, DeepSeek 2392, GPT-2 67 all unchanged). Each of the six, by node
    # id, and each one an edge the type-check was reporting:
    #   -4  the ``block_indices`` boundary and its mirror, in both layer
    #       variants. MiniMax passes ``block_indices=`` to an implementation
    #       whose signature collects it in ``**kwargs`` and never reads it, and
    #       it was being drawn as an operand of ``repeat_kv`` -- the indexer's
    #       int64 ``[B, index_n_heads, S, 4]`` entering the attention core.
    #   -2  the rotary's ``x`` boundary and its mirror. ``x`` is read for
    #       ``.dtype``/``.device`` only; it was drawn as a second operand of
    #       ``inv_freq[None, :, None]``, and with that edge gone the boundary
    #       has no consumer left.
    # One label also changes: the mask frame's ``@input`` reads ``Token
    # Embedding``, its own producer, where it used to echo the rotary's ``x``.
    # 2026-10-14 -- two new declared-shape checks (a reshape produces the axes
    # it declares; an op's declared axis exists in its operand) found 55 latent
    # shape defects, all now fixed. The fixes change a handful of shapes from
    # wrong to right, and a right shape is sometimes elided differently:
    #   GLM      1308 -> 1310, MiniMax 726 -> 727. Both are gated-norm chains
    #   whose ``self.weight * hidden_states`` used to report the WEIGHT's rank-1
    #   shape; with the elementwise broadcast fixed they carry the activation,
    #   and the ops that were folded onto a rank-1 result no longer are.
    # Every model is type-check clean at these counts.
    # 2026-10-13 -- the bespoke attention wrapper is gone. A dispatch no longer
    # has a kernel SUBSTITUTED for it; what it runs is resolved from the model's
    # own source and config, and drawn. Every delta below is that one change,
    # measured by label histogram (probe_label_hist.py), not estimated:
    #   * the fabricated `sdpa` leaf disappears from all five models. None of
    #     them calls sdpa -- no model file mentions it, and four set no
    #     implementation at all, so what they run is the
    #     `eager_attention_forward` written beside the lookup.
    #   * in its place each of those four draws that implementation's real ops:
    #     two `repeat_kv` frames, the QK matmul, the scaling, the causal
    #     Merge/Add, Softmax, the AV matmul, transpose/contiguous.
    #     GPT-2 +9 (58 -> 67) is exactly this, at one layer.
    #   * Kimi is the one model that forces an implementation (its own __init__
    #     overwrites whatever it is handed with flash_attention_2), so it keeps
    #     ONE atomic leaf -- now named `flash_attn_varlen_func`, the kernel the
    #     chain actually ends in, reached through the library's wrappers. Its
    #     -2 is the invented sdpa wrapper's layout tail going away, net of the
    #     named operand ports the leaf now declares.
    #   * GLM +28 / MiniMax +16 / DeepSeek +89 are the same expansion, times
    #     the number of attention modules and layer variants each model has.
    # Then -36 / -1: a loop body split across two frames (a submodule among its
    # ops carried no ``loop:`` detail) is one frame again, dropping the boundary
    # plumbing that ran between the halves.
    # Then a submodule's tuple return resolving per slot, plus folding two tiles
    # that were one tensor said twice. Each delta accounted for:
    #   GLM      1263 -> 1262  (-1 the duplicate ``topk_indices`` entry tile)
    #   GLM      1262 -> 1263  (-2 the eager mask tail the sdpa early-return
    #                           skips, -1 the frame's generic @input, +4 the two
    #                           per-parameter boundaries and their mirrors)
    #   Kimi      636 ->  645  (3 MoE gates: one anonymous ``@output`` each
    #                          becomes ``topk_idx``/``topk_weight`` + mirrors)
    #   MiniMax   667 ->  668  (the router's second real return, ``top_k_index``,
    #                          and its mirror; two same-id tiles become one)
    #   DeepSeek 2253 -> 2251  (-2 mirrors duplicating the tile beside them)
    #   +1 the ``new_empty`` key buffer in ``expand_kv`` -- an allocation that
    #      was never drawn, so its two ``copy_`` writes had no buffer to write
    #      into and spine-fell onto the preceding op.
    #   +9 the vision attention finally rendering the arm the model takes: the
    #      three per-image ``torch.split`` calls, the sdpa wrapper's own tail,
    #      and the per-parameter boundaries that carry cos/sin/cu_seqlens/
    #      hidden_states across the attention module's wall.
    # 1273 -> 1274: ``grid_thw`` is a model input the caller hands the vision
    # tower untouched, so it gets a boundary of its own instead of docking on the
    # image patches. The one new node is that boundary; nothing else moved, and
    # the whole ``cu_seqlens`` chain stopped reporting patch geometry.
    # dtype pass: an op given an explicit ``dtype=`` now reports it, so the
    # cast-elision pass judges casts on real dtypes. Each delta is a decision
    # flipping to the correct side, verified against source:
    #   + the ``attention_mask.to(dtype=torch.bool)`` in _preprocess_mask_arguments
    #   + ``return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`` -- real float32
    #     -> bfloat16 downcasts, previously dropped because the dtype was passed
    #     by KEYWORD and so read as unknown
    #   - ``inv_freq_expanded.float()`` on an already-float tensor -- a genuine
    #     no-op, previously kept in their place
    #   + DeepSeek's ``softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)``,
    #     a real cast back that only shows once softmax reports float32
    # 1274 -> 1264: GLM takes neither `merge_temporal` nor `include_temporal`,
    # both library defaults it never passes, so the arms guarded by them cannot
    # run. Their 8 ops and the 2 Merge phi that joined them are gone; the three
    # other models do not move, and no frame is lost.
    # 1264 -> 1266: ``grid_thw`` now lands on the ``visual`` tower's own boundary
    # instead of jumping straight to a tile two levels inside it, and the image
    # patches gain the mirror that makes their crossing explicit too.
    # One tensor entering several boxes is drawn once: the producer feeds a single
    # named tile and every box reads it. GLM -1 (a boundary folded away);
    # DeepSeek +6 (mirrors for the crossings the redirect creates, 2 per variant).
    # A loop handed several tensors keeps its ports: carrying one value is not the
    # same as being handed one, and with more than one entering, which is the
    # recurrence is what the back edge says. Every delta is one Loop in/out pair
    # (GLM +6: the MoE loop per variant plus the vision block loop; MiniMax +2;
    # DeepSeek +8: the expert loop per variant).
    # 1271 -> 1288. Every delta is an op the model runs that was not drawn, or
    # one it does not run that was. Measured by label histogram:
    #   +3 ``Any`` / +1 ``All`` -- tensor reductions that were elided entirely,
    #      so the ``where`` reading one took the UN-reduced tensor as its
    #      condition and the whole ``first_key`` chain inherited that shape.
    #   +14 ``Unsqueeze`` -- one per EXTRA ``None`` in a subscript.
    #      ``first_key[:, None, None]`` inserts two axes; only the single-``None``
    #      case resolved, so the rest were pass-throughs that kept the old rank.
    #   -2 ``Subtract`` -- ``current_length - q_length`` over two parameters the
    #      model annotates ``int`` is host arithmetic, not a tensor op. It had
    #      been docking the chain's tensor input and giving ``q_positions`` the
    #      hidden state's rank.
    # Loop ports folded away everywhere: the back edge was the only cycle the
    # graph allowed, and the body's own ends now name the direction instead
    # (``loop in: <var>`` / ``loop out: <var>``). -2 per loop that had ports.
    # `get_vision_position_ids` keeps its two boundary tiles but they now carry
    # `grid_thw` rather than `hidden_states`: its `arange` is sized by host data
    # off `grid_thw.tolist()` and takes no tensor operand at all, so docking it
    # on the chain running past it made the frame assert it reads the vision
    # activation -- a tensor that call is never passed. Same count, right tensor.
    "zai-org/GLM-5.3-Flash": 1319,
    # GPT-2, the one non-MoE decoder here. 121 -> 67 while it was being made
    # clean: -47 for a cross-attention tower its config never builds, -10 for a
    # shape that rendered as tensor ops, +9 for projections that drew nothing at
    # all because `addmm` was in no table, then -6 for `q_attn`, which exists
    # only for cross-attention and which nothing constructs the class to need,
    # and -9 for the forward arm that called it -- guarded at run time by
    # `encoder_hidden_states is not None`, which no config decides, but calling
    # a module no instance has. It was drawing the 2x-wide cross-attention
    # projection, which is why a 12-head attention reported 36 heads.
    "openai-community/gpt2": 67,
    # Then 645 -> 660: the meta trace now supplies an ``attention_mask``, so the
    # mask the model was always given at runtime is finally seen. +4
    # ``@input:attention_mask`` boundaries and 3 mirrors carrying them across
    # levels, 2 boundary tiles following that wiring, and +6 real ops inside
    # ``create_causal_mask`` which could not be reached before. Nothing removed.
    # Then 660 -> 650. Removed 20, added 10 (a rename changes the id):
    #   -6 cu_seqlens ports: one tensor was drawn once per reader, and those
    #      readers are three DIFFERENT conv kernels, not three slots of one --
    #      each still reads it, from a single tile now.
    #   -4 CumSum ports feeding two different kernels, likewise one tile now.
    #   -2 ``v`` ports that only repeated the operand tile feeding them.
    #   -8/+10 ports renamed from a producing stage's glyph to the name the
    #      kernel's own call site gives them (q / k / beta / gk).
    #   +3 (fc846608) the MLA attention's mask operand: the phantom
    #      ``@kernel_in:hidden_states`` port is replaced by
    #      ``@kernel_in:attention_mask`` (-1/+1), which now docks on a real
    #      boundary pair (``self_attn/@input:attention_mask`` and the decoder
    #      layer's own), and the primary input gets its own mirror back (+3).
    #   -4 (pending) the flash-only pad/slice and their phis: the checkpoint
    #      configures no implementation, which means sdpa, so that arm is dead.
    #   -1 the phi built over ``past_key_values.update(key_states, value_states)``
    #      -- a method on a forward parameter that produces no op, so there were
    #      never two arms to choose between.
    "moonshotai/Kimi-K3": 517,
    # MiniMax used to move by one between runs of the SAME code. That is fixed
    # (18148ab2): all four models are byte-identical across PYTHONHASHSEED 0, 7,
    # 42 and 123, so a single expected count is meaningful again.
    # Then -2: the same cache-update phi per variant, and the attention's key
    # boundary renamed from the phi (``@input:Merge``) to the tensor
    # (``@input:k_embed``).
    # 674 -> 698: +22 ``Unsqueeze`` (the same extra-``None`` axes as GLM),
    # +1 ``@input`` and its mirror for a tensor that now enters on a boundary of
    # its own, and one node relabelled ``Linear`` -> ``MatMul`` (the rotary's
    # ``inv_freq_expanded @ position_ids_expanded``, which reads two casts -- no
    # learned weight is drawn as an operand).
    # 698 -> 702: a boundary carrying a TUPLE is drawn as one tile per component,
    # so ``position_embeddings`` becomes ``position_embeddings.cos`` +
    # ``position_embeddings.sin``. +1 ``@input`` per split boundary; nothing else
    # moves, and every delta is an ``@input`` tile by label histogram.
    # 702 -> 704: the split walks MIRRORS too now, so the indexer one level
    # below the attention stops claiming two components while holding one.
    # +4 ``@input`` component tiles, -2 ``@input_mirror`` folded away.
    # 704 -> 712: ``attention_mask == 0`` inside ``build_block_mask`` is a real
    # comparison, not host bookkeeping, so it is drawn and the ``&`` that reads
    # it gets its second operand. Per decoder variant: +1 ``Equal``, +1
    # ``@input`` boundary for ``attention_mask`` and +2 mirrors carrying it in.
    # Loop ports folded away everywhere: the back edge was the only cycle the
    # graph allowed, and the body's own ends now name the direction instead
    # (``loop in: <var>`` / ``loop out: <var>``). -2 per loop that had ports.
    "MiniMaxAI/MiniMax-M3": 722,
    # 2287 -> 2291: +1 ``@input_mirror`` per decoder-layer variant (4 of them).
    # ``apply_rotary_pos_emb(x, cos, sin)`` reads BOTH slots of the caller's
    # ``position_embeddings``; slot 0 (``cos``) previously reached no op at all,
    # so nothing carried it across the wall and no crossing was drawn. It has a
    # consumer now, so it gets the mirror its sibling slot always had.
    # 2291 -> 2299: the same per-component split, at both the decoder-layer
    # variant and the attention inside it, across all 4 variants (+2 each).
    # 2299 -> 2311: +3 ``@input`` component tiles per decoder variant (4 of
    # them) as the split follows the mirror chain down to each rotary frame.
    # Loop ports folded away everywhere: the back edge was the only cycle the
    # graph allowed, and the body's own ends now name the direction instead
    # (``loop in: <var>`` / ``loop out: <var>``). -2 per loop that had ports.
    "deepseek-ai/DeepSeek-V4-Flash": 2418,
}

# Warnings the per-model run prints that the check summary does not cover. Both
# of these hid a real defect behind an all-zeros gate (2026-09-29): a model that
# never reached the meta device, and kernel stages whose shape was guessed.
EXTRA_WARNING_MARKERS = (
    "Could not instantiate",
    "No shape inference rule",
)


def _deps_dir_for(model: str) -> tuple[str | None, str]:
    """(PYTHONPATH dir or None, why) for one model."""
    from TraceLens.ModelUtils.model_env import (
        detect_env_mismatch,
        ensure_model_dependencies,
        required_transformers_version,
    )

    reason = detect_env_mismatch(model)
    if reason is None:
        return None, "runs here"
    version = required_transformers_version(model)
    if version is None:
        return None, f"needs pinned code ({reason[:40]}) but no version"
    target = ensure_model_dependencies(model, version)
    if target is None:
        return None, f"could not provision transformers=={version}"
    return str(target), f"transformers=={version}"


def main(argv: list[str]) -> int:
    models = argv or MODELS
    status = 0
    for model in models:
        deps_dir, why = _deps_dir_for(model)
        print(f"\n########## {model}  [{why}]", flush=True)
        path_parts = ([deps_dir] if deps_dir else []) + [str(REPO)]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(path_parts)}
        result = subprocess.run(
            [sys.executable, str(REPO / "probe_gate_reconciled.py"), model],
            env=env,
            capture_output=True,
            text=True,
        )
        print(result.stdout.rstrip(), flush=True)
        # Warnings are logged to stderr; surface them so a per-env-only defect
        # cannot hide behind a clean stdout summary.
        flagged = [
            line
            for line in result.stderr.splitlines()
            if "graph type-check" in line
            or "graph integrity" in line
            or any(marker in line for marker in EXTRA_WARNING_MARKERS)
        ]
        if flagged:
            status = 1
            print(f"  !! {len(flagged)} warning(s) in this environment", flush=True)
            for line in flagged[:10]:
                print(f"     {line[:160]}", flush=True)
        # The per-model run prints these as counted lines; a non-zero count is a
        # defect even though the older checks stay at zero.
        for marker in (
            "dangling @input :",
            "opaque fn leaves:",
            "dead nodes      :",
            "dangling @input :",
            "node cycle      :",
            "unresolved shapes:",
            "missing shapes  :",
            "sourceless ops  :",
            "hierarchy jumps :",
            "unnamed multi-io:",
            "unresolved config dims:",
        ):
            for line in result.stdout.splitlines():
                if marker in line and line.rsplit(":", 1)[-1].strip() not in {"0", ""}:
                    status = 1
                    print(f"  !! {line.strip()}", flush=True)

        expected = EXPECTED_NODES.get(model)
        if expected is not None:
            match = re.search(rf"{re.escape(model)}\s+nodes=(\d+)", result.stdout)
            if match is None:
                status = 1
                print("  !! no node count reported", flush=True)
            else:
                accepted = expected if isinstance(expected, tuple) else (expected,)
                if int(match.group(1)) not in accepted:
                    status = 1
                    shown = "/".join(str(value) for value in accepted)
                    print(
                        f"  !! node count {match.group(1)} != expected {shown} "
                        f"-- explain the change, then update EXPECTED_NODES",
                        flush=True,
                    )
        if result.returncode != 0:
            status = 1
            print(f"  !! exited {result.returncode}", flush=True)
            print(result.stderr[-2000:], flush=True)
    print("\nOK" if status == 0 else "\nWARNINGS PRESENT", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
