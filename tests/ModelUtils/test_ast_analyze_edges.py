###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Targeted coverage for ast_analyze helpers: role/label classification, config
literal evaluation, conditional __init__ parsing, and structural class picking."""

from __future__ import annotations

import ast
from pathlib import Path

import TraceLens.ModelUtils.ast_analyze as aa
from TraceLens.ModelUtils.ast_analyze import (
    ClassStructure,
    _capture_attention_inputs,
    _build_components,
    _classify_role,
    _config_value,
    _expr_name,
    _extract_decoder_layer_conditionals,
    _find_positional_module,
    _functional_call_name,
    _infer_attention_type_from_class,
    _infer_norm_from_ast,
    _label_for,
    _pick_causal_lm_class,
    _pick_model_class_by_structure,
    _pick_stack_model_class,
    _stmt_value,
    analyze_sources,
    expand_conditional_block_components,
)


def _expr(src: str) -> ast.AST:
    return ast.parse(src, mode="eval").body


def _cls(name: str, *, assignments=None, calls=None, norm_before=None, node_src=None):
    src = node_src or f"class {name}:\n    pass"
    return ClassStructure(
        name=name,
        node=ast.parse(src).body[0],
        init_assignments=dict(assignments or {}),
        init_details={},
        forward_calls=list(calls or []),
        norm_before=list(norm_before or []),
    )


# --------------------------------------------------------------------------- #
# _assign_target / _if_has_competing_assigns (no in-tree callers)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# tiny expression helpers
# --------------------------------------------------------------------------- #


def test_expr_name_and_functional_and_stmt_value():
    assert _expr_name(None) is None
    assert _expr_name(_expr("a.b.c")) == "a.b.c"
    assert _expr_name(_expr("a[0]")) == "a"

    softmax = _expr("torch.nn.functional.softmax(x)")
    assert _functional_call_name(softmax.func) == "softmax"
    assert _functional_call_name(_expr("F.relu(x)").func) == "relu"
    assert _functional_call_name(_expr("x").func if False else _expr("q")) is None

    assert _stmt_value(ast.parse("x = 1").body[0]).value == 1
    assert _stmt_value(ast.parse("x: int = 5").body[0]).value == 5
    assert _stmt_value(ast.parse("return y").body[0]) is not None
    assert _stmt_value(ast.parse("pass").body[0]) is None


# --------------------------------------------------------------------------- #
# _classify_role fall-through branches
# --------------------------------------------------------------------------- #


def test_classify_role_token_and_class_regex_branches():
    assert _classify_role("router_head", "Thing") == "router"  # 782
    assert _classify_role("mixer", "SelfAttention") == "attention"  # 791
    assert _classify_role("mixer", "FeedForward") == "ffn"  # 795
    assert _classify_role("mixer", "RMSNorm") == "norm"  # 797
    assert _classify_role("mixer", "RotaryEmbedding") == "positional"  # 799
    assert _classify_role("embedder", "Embedding") == "embedding"  # 801
    assert _classify_role("weird", "Mystery") == "other"


# --------------------------------------------------------------------------- #
# _label_for branches
# --------------------------------------------------------------------------- #


def test_label_for_branches():
    assert _label_for("attention", "GatedAttention", "attn") == "Gated Attention"
    assert _label_for("attention", "SlidingAttention", "attn") == "Sliding Window Attn"
    assert _label_for("norm", "CustomNorm", "norm") == "Norm"
    assert _label_for("router", "GateThing", "gate") == "Router"


# --------------------------------------------------------------------------- #
# _config_value expression evaluation
# --------------------------------------------------------------------------- #


def test_config_value_arithmetic_and_logic():
    config = {"hidden": 12, "flag": True, "off": False, "nested": {"k": 7}}
    sv: dict = {}

    assert _config_value(_expr("config.hidden % 5"), config, sv) == 2  # 1399-1400
    assert _config_value(_expr("config.hidden + 1"), config, sv) == 13
    assert _config_value(_expr("config.hidden - 2"), config, sv) == 10
    assert _config_value(_expr("config.hidden * 2"), config, sv) == 24
    assert _config_value(_expr("config.hidden // 5"), config, sv) == 2
    # ZeroDivisionError path -> _UNKNOWN
    assert _config_value(_expr("config.hidden // 0"), config, sv) is aa._UNKNOWN

    # BoolOp and/or (1411, 1420)
    assert _config_value(_expr("config.off and config.flag"), config, sv) is False
    assert _config_value(_expr("config.flag or config.off"), config, sv) is True

    # Compare chains (1436-1450)
    assert _config_value(_expr("config.hidden == 12"), config, sv) is True
    assert _config_value(_expr("config.hidden != 99"), config, sv) is True
    assert _config_value(_expr("config.hidden > 5"), config, sv) is True
    assert _config_value(_expr("config.hidden >= 12"), config, sv) is True
    assert _config_value(_expr("config.hidden < 5"), config, sv) is False
    assert _config_value(_expr("config.hidden <= 12"), config, sv) is True

    # getattr on config dict + nested attribute (1367-1370, 1382-1383)
    assert _config_value(_expr("getattr(config, 'hidden', 0)"), config, sv) == 12
    assert _config_value(_expr("config.nested.k"), config, sv) == 7
    # unary not
    assert _config_value(_expr("not config.off"), config, sv) is True


def test_config_value_self_config_fallback():
    # HF convention: ``self.config`` resolves to the passed config even when
    # ``__init__`` never bound it into the symbol table (self_values empty), so
    # ``self.config.<key>`` predicates resolve at build time.
    config = {"hidden_act": "situ"}
    assert _config_value(_expr("self.config.hidden_act"), config, {}) == "situ"
    assert _config_value(_expr("self.config.hidden_act == 'situ'"), config, {}) is True
    # An explicit binding in self_values still wins over the fallback.
    assert (
        _config_value(
            _expr("self.config.hidden_act"), config, {"config": {"hidden_act": "gelu"}}
        )
        == "gelu"
    )
    # No config threaded -> empty dict -> key miss -> unresolved (no false prune).
    assert _config_value(_expr("self.config.hidden_act"), {}, {}) is aa._UNKNOWN
    # The fallback is scoped to the ``config`` attribute only; other unresolved
    # ``self.<attr>`` reads stay unknown.
    assert _config_value(_expr("self.something"), config, {}) is aa._UNKNOWN


def test_distributed_degree_value_resolves_group_object_and_standalone_styles():
    """Both engine styles settle to the same convention: world size 1, rank 0 --
    answering "the full, unsharded degree" rather than guessing an actual
    distributed topology at parse time."""
    from TraceLens.ModelUtils.ast_analyze import _distributed_degree_value

    # SGLang's group-object style.
    assert _distributed_degree_value(_expr("get_parallel().attn_tp_size")) == 1
    assert _distributed_degree_value(_expr("get_parallel().attn_tp_rank")) == 0
    # Neither a ``*_size`` nor a ``*_rank`` attribute -> unresolved.
    assert _distributed_degree_value(_expr("get_parallel().pp_group")) is None

    # vLLM's standalone-function style.
    assert (
        _distributed_degree_value(_expr("get_tensor_model_parallel_world_size()")) == 1
    )
    assert _distributed_degree_value(_expr("get_tensor_model_parallel_rank()")) == 0
    # An unrelated call or attribute is not a distributed-degree accessor at all.
    assert _distributed_degree_value(_expr("some_other_call()")) is None
    assert _distributed_degree_value(_expr("config.hidden")) is None


def test_config_value_caps_a_distributed_degree_with_min_and_max():
    """``self.idx_head_tp_size = min(attn_tp_size, self.total_idx_heads)`` --
    once the parallelism accessor above resolves to a concrete int, capping it
    against another resolvable int must fold too."""
    config = {"heads": 4}
    assert (
        _config_value(
            _expr("min(get_tensor_model_parallel_world_size(), config.heads)"),
            config,
            {},
        )
        == 1
    )
    assert (
        _config_value(
            _expr("max(get_tensor_model_parallel_world_size(), config.heads)"),
            config,
            {},
        )
        == 4
    )
    # A keyword argument disqualifies the fold (only the plain positional shape
    # is recognized).
    assert _config_value(_expr("min(config.heads, key=abs)"), config, {}) is aa._UNKNOWN
    # An operand that does not resolve to an int leaves the whole call unknown.
    assert (
        _config_value(_expr("min(config.heads, config.missing)"), config, {})
        is aa._UNKNOWN
    )


def test_parse_forward_prunes_dead_config_branch():
    # Path A (forward_calls): a build-time-resolvable config predicate selects one
    # arm; the dead arm's submodule calls never enter the flat call sequence.
    src = (
        "def forward(self, x):\n"
        "    if self.config.hidden_act == 'situ':\n"
        "        a = self.gate(x)\n"
        "    else:\n"
        "        b = self.up(x)\n"
        "    return a\n"
    )
    fn = ast.parse(src).body[0]
    calls_true = aa._parse_forward(fn, config={"hidden_act": "situ"})[0]
    assert "gate" in calls_true and "up" not in calls_true
    calls_false = aa._parse_forward(fn, config={"hidden_act": "gelu"})[0]
    assert "up" in calls_false and "gate" not in calls_false
    # Unresolvable predicate (no config) -> both arms flattened (prior behavior).
    calls_both = aa._parse_forward(fn, config={})[0]
    assert "gate" in calls_both and "up" in calls_both


# --------------------------------------------------------------------------- #
# attention/norm inference from a decoder structure
# --------------------------------------------------------------------------- #


def test_infer_attention_type_from_class_variants():
    all_classes = {
        "MLAAttention": _cls("MLAAttention"),
        "LatentAttn": _cls("LatentAttn", assignments={"kv_lora": "Linear"}),
        "GroupedAttention": _cls("GroupedAttention"),
        "MultiQueryAttention": _cls("MultiQueryAttention"),
    }
    assert _infer_attention_type_from_class(None, all_classes) is None

    mla = _cls("Dec1", assignments={"self_attn": "MLAAttention"})
    assert _infer_attention_type_from_class(mla, all_classes) == "MLA"

    latent = _cls("Dec2", assignments={"self_attn": "LatentAttn"})
    assert _infer_attention_type_from_class(latent, all_classes) == "MLA"

    gqa = _cls("Dec3", assignments={"self_attn": "GroupedAttention"})
    assert _infer_attention_type_from_class(gqa, all_classes) == "GQA"

    mqa = _cls("Dec4", assignments={"self_attn": "MultiQueryAttention"})
    assert _infer_attention_type_from_class(mqa, all_classes) == "MQA"

    plain = _cls("Dec5", assignments={"mlp": "GatedMLP"})
    assert _infer_attention_type_from_class(plain, all_classes) is None


def test_infer_norm_from_ast_placements():
    rms_pre = _cls(
        "Dec",
        assignments={"input_layernorm": "RMSNorm", "self_attn": "Attn"},
        calls=["self_attn"],
        norm_before=["self_attn"],
    )
    assert _infer_norm_from_ast(rms_pre) == ("RMSNorm", "Pre-Norm")

    ln_post = _cls(
        "Dec2",
        assignments={"ln": "LayerNorm", "self_attn": "Attn"},
        calls=["self_attn", "ln"],
    )
    norm_type, placement = _infer_norm_from_ast(ln_post)
    assert norm_type == "LayerNorm"
    assert placement == "Post-Norm (inside residual)"

    ln_first = _cls(
        "Dec3",
        assignments={"ln": "LayerNorm", "self_attn": "Attn"},
        calls=["ln", "self_attn"],
    )
    assert _infer_norm_from_ast(ln_first)[1] == "Pre-Norm"


# --------------------------------------------------------------------------- #
# structural class picking
# --------------------------------------------------------------------------- #


def test_pick_model_class_by_structure_and_stack():
    classes = {
        "PlainTransformer": _cls(
            "PlainTransformer",
            assignments={
                "embed_tokens": "Embedding",
                "norm": "RMSNorm",
                "lm_head": "Linear",
            },
        ),
        "NoEmbed": _cls("NoEmbed", assignments={"proj": "Linear"}),
    }
    picked = _pick_model_class_by_structure(classes)
    assert picked is not None and picked.name == "PlainTransformer"

    # nothing has an embedding -> None
    assert (
        _pick_model_class_by_structure({"X": _cls("X", assignments={"p": "Linear"})})
        is None
    )

    # causal-lm delegates to its `transformer` attr
    causal = _cls("GPTForCausalLM", assignments={"transformer": "PlainTransformer"})
    reg = {**classes, "GPTForCausalLM": causal}
    assert _pick_causal_lm_class(reg).name == "GPTForCausalLM"
    assert _pick_stack_model_class(reg, causal).name == "PlainTransformer"


def test_find_positional_module():
    decoder = _cls("Dec", assignments={"self_attn": "Attn"})
    stack = _cls(
        "Model",
        assignments={"rotary_emb": "RotaryEmbedding", "embed_tokens": "Embedding"},
        calls=["rotary_emb"],
    )
    registry = {"Model": stack, "Dec": decoder}
    assert _find_positional_module(registry, stack, decoder) == (
        "rotary_emb",
        "RotaryEmbedding",
    )

    # nothing positional in the search roots but present elsewhere in the registry
    other = _cls("Other", assignments={"rope": "RotaryEmbedding"})
    reg2 = {"Other": other}
    assert _find_positional_module(reg2, None, None) == ("rope", "RotaryEmbedding")

    assert _find_positional_module({}, None, None) is None


def test_expand_conditional_block_components():
    from TraceLens.ModelUtils.blocks import BlockComponent

    decoder = _cls("Dec", assignments={"mlp": "GatedMLP"}, calls=["mlp"])
    decoder.init_assignment_options = {
        "self_attn": ["FullAttention", "SlidingAttention"],
        "mlp": ["GatedMLP", "SparseMoeBlock"],
    }
    components = [
        BlockComponent("self_attn", "FullAttention", "attention", "Attn", 0),
        BlockComponent("mlp", "GatedMLP", "ffn", "FFN", 1),
    ]
    expanded = expand_conditional_block_components(decoder, components)
    class_names = {(c.attr_name, c.class_name) for c in expanded}
    assert ("self_attn", "SlidingAttention") in class_names
    assert ("mlp", "SparseMoeBlock") in class_names


# --------------------------------------------------------------------------- #
# full-source analysis with a layer_idx-conditional decoder + module list loop
# --------------------------------------------------------------------------- #

_COND_SOURCE = """
def apply_rotary_emb(x, freqs):
    return x

class RMSNorm:
    def forward(self, x):
        return x

class FullAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
    def forward(self, hidden_states, freqs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = apply_rotary_emb(q, freqs)
        attn = torch.nn.functional.softmax(q @ k.transpose(-1, -2), dim=-1)
        return self.o_proj(attn @ v)

class SlidingAttention:
    def __init__(self, config):
        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
    def forward(self, hidden_states, freqs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return self.o_proj(v)

class GatedMLP:
    def __init__(self, config):
        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear()
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class SparseMoeBlock:
    def __init__(self, config):
        self.gate = Linear()
        self.experts = ModuleList([GatedMLP(config) for _ in range(config.num_experts)])
    def forward(self, hidden_states):
        scores = self.gate(hidden_states)
        return self.experts[0](hidden_states)

class DecoderLayer:
    def __init__(self, config, layer_idx):
        self.input_layernorm = RMSNorm()
        if layer_idx % 2 == 0:
            self.self_attn = FullAttention(config)
        else:
            self.self_attn = SlidingAttention(config)
        self.post_attention_layernorm = RMSNorm()
        if config.use_moe:
            self.mlp = SparseMoeBlock(config)
        else:
            self.mlp = GatedMLP(config)
    def forward(self, hidden_states, freqs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, freqs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class TextModel:
    def __init__(self, config):
        self.embed_tokens = Embedding()
        self.layers = ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rotary_emb = RotaryEmbedding()
        self.norm = RMSNorm()
    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        freqs = self.rotary_emb(hidden)
        for layer in self.layers:
            hidden = layer(hidden, freqs)
        return self.norm(hidden)

class TextForCausalLM:
    def __init__(self, config):
        self.model = TextModel(config)
        self.lm_head = Linear()
    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))
"""


def test_conditional_decoder_source_analysis():
    config = {"num_hidden_layers": 4, "num_experts": 8, "use_moe": True}
    analysis = analyze_sources(
        {Path("modeling_cond.py"): _COND_SOURCE},
        config=config,
        all_tensor_ops=True,
    )
    assert analysis.decoder_class == "DecoderLayer"
    assert analysis.stack_model_class == "TextModel"

    decoder = analysis.class_registry["DecoderLayer"]
    conditionals = _extract_decoder_layer_conditionals(decoder)
    attrs = {attr for attr, _cls_name, _cond in conditionals}
    assert "self_attn" in attrs
    # both attention branch classes were captured as options
    options = decoder.init_assignment_options.get("self_attn", [])
    assert "FullAttention" in options and "SlidingAttention" in options

    # _build_components yields ordered decoder components from the AST spine
    components = _build_components(decoder)
    roles = {c.role for c in components}
    assert "attention" in roles


def test_parse_architecture_layer_variants_from_conditionals():
    from TraceLens.ModelUtils.extract import parse_architecture

    config = {
        "model_type": "cond",
        "architectures": ["TextForCausalLM"],
        "num_hidden_layers": 4,
        "num_experts": 8,
        "use_moe": True,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }
    analysis = analyze_sources(
        {Path("modeling_cond.py"): _COND_SOURCE},
        config=config,
        all_tensor_ops=False,
    )
    spec = parse_architecture(config, "fixture", code_analysis=analysis)
    # even/odd attention split -> multiple layer variants -> hybrid attention
    assert len(spec.layer_variants) >= 2
    assert spec.attention_type == "Hybrid"


# --------------------------------------------------------------------------- #
# Config values the architecture parser resolves or refuses
# --------------------------------------------------------------------------- #


def test_config_value_false_and_unsupported_branches():
    config = {"hidden": 12, "flag": True, "off": False}
    sv: dict = {}

    # BinOp with an unsupported operator -> _UNKNOWN (1403)
    assert _config_value(_expr("config.hidden ** 2"), config, sv) is aa._UNKNOWN
    # Or with no True and all-known -> any(values) (1420)
    assert _config_value(_expr("config.off or config.off"), config, sv) is False

    # Each comparison operator taking its "return False" branch
    assert _config_value(_expr("config.hidden == 99"), config, sv) is False  # 1436
    assert _config_value(_expr("config.hidden != 12"), config, sv) is False  # 1438
    assert _config_value(_expr("config.hidden > 99"), config, sv) is False  # 1440
    assert _config_value(_expr("config.hidden >= 99"), config, sv) is False  # 1442
    assert _config_value(_expr("config.hidden <= 5"), config, sv) is False  # 1446
    assert (
        _config_value(_expr("config.flag is config.off"), config, sv) is False
    )  # 1448
    assert (
        _config_value(_expr("config.flag is not config.flag"), config, sv) is False
    )  # 1450


def test_infer_attention_type_mla_via_internal_lora():
    all_classes = {
        "SpecialAttention": _cls(
            "SpecialAttention", assignments={"kv_lora_proj": "Linear"}
        ),
    }
    decoder = _cls("Dec", assignments={"self_attn": "SpecialAttention"})
    assert _infer_attention_type_from_class(decoder, all_classes) == "MLA"  # 4502


def test_build_components_skip_branches():
    decoder = _cls(
        "Dec",
        assignments={
            "self_attn": "Attn",
            "buf": "Parameter",  # in _SKIP_INIT_CLASS_NAMES -> 4552
            "unused": "Linear",  # not in forward_calls -> 4554
        },
        calls=["self_attn"],
    )
    components = _build_components(decoder)
    attrs = {c.attr_name for c in components}
    assert "self_attn" in attrs
    assert "buf" not in attrs and "unused" not in attrs


def test_expand_conditional_components_extra_ffn_option():
    from TraceLens.ModelUtils.blocks import BlockComponent

    # No mlp/moe component in the spine, but options exist -> second loop (4665-4670)
    decoder = _cls(
        "Dec", assignments={"self_attn": "FullAttention"}, calls=["self_attn"]
    )
    decoder.init_assignment_options = {
        "block_sparse_moe": ["SparseMoeBlock", "AltMoeBlock"],
    }
    components = [BlockComponent("self_attn", "FullAttention", "attention", "Attn", 0)]
    expanded = expand_conditional_block_components(decoder, components)
    names = {(c.attr_name, c.class_name) for c in expanded}
    assert ("block_sparse_moe", "SparseMoeBlock") in names
    assert ("block_sparse_moe", "AltMoeBlock") in names


def test_parse_layer_module_list_edge_returns():
    parse = aa._parse_layer_module_list
    # bare list comp whose element is not a Call -> None (4720-4721)
    assert parse(_expr("[x for x in range(3)]")) is None
    # ModuleList wrapping a comp whose elt is not a Call -> None
    assert parse(_expr("ModuleList([x for x in range(3)])")) is None
    # comp element is a Call but no generators handled by the loop var check
    assert parse(_expr("[Layer(config) for (a, b) in pairs]")) is None  # 4728-4729
    # not a list comp at all
    assert parse(_expr("Linear()")) is None


def test_alternate_forward_dispatches():
    func = ast.parse(
        "def forward(self, x):\n"
        "    if x is None:\n"
        "        return self._alt_forward(x)\n"
        "    return self.main(x)\n"
    ).body[0]
    assert "_alt_forward" in aa._alternate_forward_dispatches(func)


def test_find_positional_module_skips_parameter_class():
    # positional-looking attr but class is skipped -> falls through (4347)
    stack = _cls(
        "Model",
        assignments={"rotary_emb": "Buffer", "embed_tokens": "Embedding"},
        calls=["rotary_emb"],
    )
    assert _find_positional_module({"Model": stack}, stack, None) is None


# ---------------------------------------------------------------------------
# Norm kind read from the forward body, not the class name
# ---------------------------------------------------------------------------


def _structure_for(source: str, class_name: str):
    from TraceLens.ModelUtils.ast_analyze import analyze_source

    return analyze_source(source).class_registry[class_name]


def test_norm_kind_reads_rms_forward_without_the_class_name():
    """An RMS norm divides by the root mean square and never centres its input."""
    from TraceLens.ModelUtils.ast_analyze import _norm_kind_from_forward

    source = """
import torch
from torch import nn

class OddlyNamedNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states
"""
    assert (
        _norm_kind_from_forward(_structure_for(source, "OddlyNamedNorm")) == "RMSNorm"
    )


def test_norm_kind_reads_a_centring_forward_as_layernorm():
    """Subtracting a mean is what makes it a LayerNorm, whatever it is called."""
    from TraceLens.ModelUtils.ast_analyze import _norm_kind_from_forward

    source = """
import torch
from torch import nn

class AlsoOddlyNamed(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        centred = hidden_states - hidden_states.mean(-1, keepdim=True)
        variance = centred.pow(2).mean(-1, keepdim=True)
        return self.weight * centred * torch.rsqrt(variance + self.eps)
"""
    assert (
        _norm_kind_from_forward(_structure_for(source, "AlsoOddlyNamed")) == "LayerNorm"
    )


def test_norm_kind_is_unresolved_without_a_forward():
    """A torch builtin has no readable body; the caller resolves it by identity."""
    from TraceLens.ModelUtils.ast_analyze import _norm_kind_from_forward

    assert _norm_kind_from_forward(None) is None


def test_norm_tile_label_comes_from_the_forward_not_the_spelling():
    """A norm class not spelled "RMS" is still labelled RMSNorm when it is one."""
    from TraceLens.ModelUtils.ast_analyze import _label_for

    source = """
import torch
from torch import nn

class WeirdlyNamed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
"""
    from TraceLens.ModelUtils.ast_analyze import analyze_source

    classes = analyze_source(source).class_registry
    assert _label_for("norm", "WeirdlyNamed", "input_layernorm", classes) == "RMSNorm"


def test_norm_tile_label_resolves_a_torch_builtin_by_identity():
    """``nn.LayerNorm`` has no Python forward; its exact identity still names it."""
    from TraceLens.ModelUtils.ast_analyze import _label_for

    assert _label_for("norm", "LayerNorm", "norm1", {}) == "LayerNorm"
    assert _label_for("norm", "RMSNorm", "norm1", {}) == "RMSNorm"


def test_structural_reading_wins_over_the_class_name():
    """A class named "...RMSNorm" that actually CENTRES its input reads LayerNorm.

    The name is only a last resort for a class with no readable forward, so a
    misleading name never overrides what the code does.
    """
    from TraceLens.ModelUtils.ast_analyze import _label_for, analyze_source

    source = """
import torch
from torch import nn

class MisleadingRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        centred = x - x.mean(-1, keepdim=True)
        return self.weight * centred * torch.rsqrt(centred.pow(2).mean(-1) + 1e-6)
"""
    classes = analyze_source(source).class_registry
    assert _label_for("norm", "MisleadingRMSNorm", "norm1", classes) == "LayerNorm"


def test_norm_with_no_readable_forward_falls_back_to_its_name():
    """A stub or unparsed third-party norm has no forward; the name is all there is."""
    from TraceLens.ModelUtils.ast_analyze import _label_for

    assert _label_for("norm", "CustomRMSNorm", "norm1", {}) == "RMSNorm"
    assert _label_for("norm", "Anonymous", "norm1", {}) == "Norm"


# ---------------------------------------------------------------------------
# Relative imports resolve against the right package
# ---------------------------------------------------------------------------


def test_relative_import_inside_a_package_init_resolves_to_that_package(
    tmp_path, monkeypatch
):
    """``from .sub import X`` in ``pkg/__init__.py`` means ``pkg.sub``.

    Treating a package ``__init__`` like a plain module walks one level too far
    up (``pkg.sub`` becomes a sibling of ``pkg``), so the symbol is never found
    and the class it names stays unparsed.
    """
    import ast as _ast
    import sys as _sys

    from TraceLens.ModelUtils.ast_analyze import _absolute_import_bindings

    pkg = tmp_path / "demopkg" / "inner"
    pkg.mkdir(parents=True)
    (tmp_path / "demopkg" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("from .leaf import Thing\n", encoding="utf-8")
    (pkg / "leaf.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    tree = _ast.parse((pkg / "__init__.py").read_text(encoding="utf-8"))
    bindings = _absolute_import_bindings(tree, "demopkg.inner")
    assert bindings["Thing"] == "demopkg.inner.leaf#Thing"
    assert "demopkg.leaf" not in bindings["Thing"]
    assert str(tmp_path) in _sys.path


def test_relative_import_inside_a_plain_module_resolves_to_its_parent(
    tmp_path, monkeypatch
):
    """``from .sib import Y`` in ``pkg/mod.py`` means ``pkg.sib`` -- unchanged."""
    import ast as _ast

    from TraceLens.ModelUtils.ast_analyze import _absolute_import_bindings

    pkg = tmp_path / "demopkg2"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("from .sib import Y\n", encoding="utf-8")
    (pkg / "sib.py").write_text("class Y:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    tree = _ast.parse((pkg / "mod.py").read_text(encoding="utf-8"))
    bindings = _absolute_import_bindings(tree, "demopkg2.mod")
    assert bindings["Y"] == "demopkg2.sib#Y"


# ---------------------------------------------------------------------------
# Classes imported from another package
# ---------------------------------------------------------------------------


def _write_library(tmp_path, body: str, *, module: str = "leaf") -> None:
    """A small installable package whose public name is a re-export."""
    pkg = tmp_path / "vendorlib" / "inner"
    pkg.mkdir(parents=True, exist_ok=True)
    (tmp_path / "vendorlib" / "__init__.py").write_text(
        "from vendorlib.inner import Widget\n", encoding="utf-8"
    )
    (pkg / "__init__.py").write_text(
        f"from .{module} import Widget\n", encoding="utf-8"
    )
    (pkg / f"{module}.py").write_text(body, encoding="utf-8")


def _registry_for_model(tmp_path, monkeypatch):
    from TraceLens.ModelUtils.ast_analyze import analyze_source

    monkeypatch.syspath_prepend(str(tmp_path))
    model = """
import torch
from torch import nn
from vendorlib import Widget

class DemoDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.widget = Widget(config.hidden_size)

    def forward(self, hidden_states):
        return self.widget(hidden_states)
"""
    return analyze_source(model, config={"model_type": "demo"}).class_registry


def test_imported_class_doing_its_own_math_is_registered(tmp_path, monkeypatch):
    """A composite reached through a package re-export chain gets parsed."""
    _write_library(
        tmp_path,
        """
import torch
from torch import nn

class Widget(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scaled = x * self.weight
        return scaled + x
""",
    )
    assert "Widget" in _registry_for_model(tmp_path, monkeypatch)


def test_class_that_only_delegates_stays_a_kernel_boundary(tmp_path, monkeypatch):
    """``return kernel(...)`` performs no arithmetic of its own -- keep it a leaf.

    This is the fla shape: ``FusedRMSNormGated.forward`` returns
    ``rms_norm_gated(...)``, a compiled entry point. Opening it would replace
    working wiring with an empty frame.
    """
    _write_library(
        tmp_path,
        """
import torch
from torch import nn

def fused_kernel(x, w):
    return x * w

class Widget(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return fused_kernel(x, self.weight)
""",
    )
    assert "Widget" not in _registry_for_model(tmp_path, monkeypatch)


def test_class_calling_an_imported_kernel_stays_a_boundary(tmp_path, monkeypatch):
    """The ShortConvolution shape: forward calls a routine imported from elsewhere."""
    _write_library(
        tmp_path,
        """
import torch
from torch import nn
from vendorlib.inner.kernels import run_kernel

class Widget(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x * 2
        return run_kernel(y, self.weight)
""",
    )
    (tmp_path / "vendorlib" / "inner" / "kernels.py").write_text(
        "def run_kernel(x, w):\n    return x * w\n", encoding="utf-8"
    )
    assert "Widget" not in _registry_for_model(tmp_path, monkeypatch)


_EXTENT_SOURCE = """
import torch
class Compressor(torch.nn.Module):
    def forward(self, compressed, idx_q):
        n_windows = compressed.shape[1] // 4
        positions = torch.arange(n_windows, device=compressed.device)
        k_len = idx_q.shape[-1]
        k_positions = torch.arange(k_len, device=idx_q.device)
        constant_range = torch.arange(self.local_blocks, device=idx_q.device)
        return positions + k_positions + constant_range
"""


def test_generator_extent_docks_the_tensor_its_size_reads():
    """``torch.arange(<local>)`` reaches the tensor whose size the local holds.

    The generator takes no tensor *operand*, so nothing docked onto it and it
    rendered rootless -- asserting the range is independent of the model's data
    when it is not. The bound is now resolved through its own defining
    expression (``n_windows = compressed.shape[1] // 4``) back to the tensor,
    one hop or several, and the recovered edge is marked as carrying an EXTENT
    so the operand-arity check does not read it as a mis-wired argument.

    A bound that reads a forward parameter's shape directly reaches it through
    the parameter channel instead, and a bound built only from config scalars
    stays unwired: that range really is constant.
    """
    analysis = analyze_sources(
        {Path("modeling_extent.py"): _EXTENT_SOURCE},
        config={"local_blocks": 8},
        all_tensor_ops=True,
    )
    operations = analysis.class_registry["Compressor"].forward_operations
    by_bound = {}
    for op in operations.values():
        if op.label.lower() != "arange":
            continue
        bound = next(
            str(d).split(":", 1)[1].strip()
            for d in op.details
            if str(d).startswith("arange_stop:")
        )
        by_bound[bound] = op
    assert sorted(by_bound) == ["k_len", "n_windows", "self.local_blocks"]

    # Resolved through a local assignment to the producer of ``compressed``.
    windows = by_bound["n_windows"]
    assert windows.predecessors, "n_windows arange left rootless"
    assert "extent_inputs: 1" in [str(d) for d in windows.details]

    # ``k_len = idx_q.shape[-1]`` reads a forward PARAMETER, which reaches the
    # op through ``param_inputs`` (its boundary tile), not a local producer.
    assert by_bound["k_len"].param_inputs == ("idx_q",)

    # A pure config bound is genuinely constant -- no fabricated extent edge.
    constant = by_bound["self.local_blocks"]
    assert not constant.predecessors
    assert not any(str(d).startswith("extent_inputs:") for d in constant.details)


_CONSTANT_EXTENT_SOURCE = """
import torch
class C(torch.nn.Module):
    def forward(self, x, grid):
        fixed = torch.arange(self.local_blocks, device=x.device)
        derived = torch.arange(x.shape[1], device=x.device)
        for t, h, w in grid.tolist():
            per_iteration = torch.arange(w, device=x.device)
        return fixed + derived + per_iteration
"""


def test_only_a_data_independent_extent_is_marked_constant():
    """A range nothing in the input can change is a constant; the others are not.

    ``torch.arange(self.local_blocks)`` produces the same tensor on every
    forward, so it is a constant and constants are never drawn as compute. The
    test is structural -- does this extent read any tensor -- rather than
    "can we fold the number", because a config attribute the class never
    resolved is still data-independent. A loop-bound name holds host data read
    off a real tensor each iteration, so it must NOT be swept up.
    """
    analysis = analyze_sources(
        {Path("modeling_const.py"): _CONSTANT_EXTENT_SOURCE},
        config={"local_blocks": 8},
        all_tensor_ops=True,
    )
    by_bound = {}
    for op in analysis.class_registry["C"].forward_operations.values():
        if op.label.lower() != "arange":
            continue
        bound = next(
            str(d).split(":", 1)[1].strip()
            for d in op.details
            if str(d).startswith("arange_stop:")
        )
        by_bound[bound] = op

    def marked(bound: str) -> bool:
        return any(
            str(d).strip() == "constant_extent: true" for d in by_bound[bound].details
        )

    assert marked("self.local_blocks")
    # Sized by a tensor -- a real dependency, wired rather than elided.
    assert not marked("x.shape[1]")
    assert any(str(d) == "extent_inputs: 1" for d in by_bound["x.shape[1]"].details)
    # Host data read off ``grid`` each iteration is not a compile-time constant.
    assert not marked("w")


def test_modulo_keeps_the_value_it_is_applied_to():
    """``(arange - starts) % widths`` is a remainder, not the widths themselves.

    With no label for ``%``, the operator collapsed to its right operand. The
    subtraction was emitted and then nothing read it, and ``within`` was bound
    to the product the remainder was supposed to divide by.
    """
    analysis = analyze_sources(
        {Path("modeling_mod.py"): """
import torch
class M(torch.nn.Module):
    def forward(self, counts, starts, heights, widths):
        within = (torch.arange(counts.sum()) - starts) % (heights * widths)
        return within
"""},
        all_tensor_ops=True,
    )
    ops = analysis.class_registry["M"].forward_operations
    remainder = [op for op in ops.values() if op.label == "Remainder"]
    assert len(remainder) == 1
    subtract = next(op for op in ops.values() if op.label == "Subtract")
    assert subtract.attr_name in remainder[0].predecessors


def test_tolist_loop_arithmetic_is_host_and_arange_still_docks_the_grid():
    """Unpacking ``grid_thw.tolist()`` yields Python ints, not the grid tensor.

    Binding those names to the tensor drew ``h // merge`` as a floor-divide of
    the whole grid and handed that tensor to ``arange`` and ``pad`` as an
    operand. The range is still sized by the grid, so it keeps an extent edge.
    """
    analysis = analyze_sources(
        {Path("modeling_tolist.py"): """
import torch
class M(torch.nn.Module):
    def forward(self, grid_thw):
        chunks = []
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            llm_h = grid_h // 2
            llm_w = grid_w // 2
            index = torch.arange(grid_t * llm_h * llm_w)
            chunks.append(index)
        return torch.cat(chunks, dim=0)
"""},
        all_tensor_ops=True,
    )
    ops = list(analysis.class_registry["M"].forward_operations.values())
    assert not any(op.label == "Floor divide" for op in ops)
    arange = next(op for op in ops if op.label == "Arange")
    assert any(str(d).startswith("extent_inputs:") for d in arange.details)
    assert arange.predecessors or "grid_thw" in arange.param_inputs


def test_a_zip_of_two_tolists_does_not_size_arange_from_the_other_sequence():
    """``arange(h)`` reads the grid, not the merge-size sequence zipped with it.

    ``for (t, h, w), merge_size in zip(grid.tolist(), merge.tolist())`` walks
    Python ints. Binding every name to the whole ``zip`` made ``h`` depend on
    both sequences, so the range took the constant expand of ``merge_size`` as
    a second extent and the render named a boundary after that loop target.
    Dropping the constant then left the boundary with no source.
    """
    analysis = analyze_sources(
        {Path("modeling_zip_tolist.py"): """
import torch
class M(torch.nn.Module):
    def forward(self, grid_thw, spatial_merge_size):
        if isinstance(spatial_merge_size, int):
            spatial_merge_size = torch.tensor([spatial_merge_size]).expand(len(grid_thw))
        for (t, h, w), merge_size in zip(grid_thw.tolist(), spatial_merge_size.tolist()):
            torch.arange(h)
            torch.arange(w)
        return grid_thw
"""},
        all_tensor_ops=True,
    )
    ops = list(analysis.class_registry["M"].forward_operations.values())
    arange_h = next(op for op in ops if "arange_stop: h" in op.details)
    expand = next(op for op in ops if op.label == "Expand")
    assert expand.attr_name not in arange_h.predecessors
    assert not any("spatial_merge_size" in str(detail) for detail in arange_h.details)
    assert not any(str(detail).startswith("loop iterator") for detail in expand.details)
    assert arange_h.predecessors or "grid_thw" in arange_h.param_inputs


def test_parallel_list_accumulators_keep_their_own_appended_producers():
    """Two lists grown in one loop must not both carry the body's last op."""
    # The source has no model class, so inspect the helper directly through a
    # tiny caller that makes its free-function analysis visible.
    source = """
import torch
def helper(grid):
    indices = []
    lengths = []
    for row in grid.tolist():
        index = torch.arange(row)
        indices.append(index + 1)
        lengths.append(index.cumsum(0))
    return torch.cat(indices), torch.cat(lengths)
class M(torch.nn.Module):
    def forward(self, grid):
        return helper(grid)
"""
    analysis = analyze_sources(
        {Path("modeling_parallel_accumulators.py"): source},
        all_tensor_ops=True,
    )
    cls = analysis.class_registry["M"]
    call = next(attr for attr in cls.multi_op_method_loop_carried if "helper" in attr)
    carried = cls.multi_op_method_loop_carried[call]
    producers = {item.variable: item.updated_producer for item in carried}
    assert producers["indices"] != producers["lengths"], producers
    assert "add" in producers["indices"]
    assert "cumulative_sum" in producers["lengths"]


def test_a_dict_subscript_keeps_every_tensor_the_key_can_select():
    """``mapping[config.layer_types[i]]`` is a choice, not the last entry.

    The two values are different tensors. Collapsing the subscript onto the
    last one drops the other and, when that value is one slot of a tuple,
    docks the consumer on the wrong slot.
    """
    analysis = analyze_sources(
        {Path("modeling_dict_select.py"): """
import torch
from torch import nn
def helper(grid):
    index = grid.cumsum(0)
    lengths = index + 1
    return index, lengths
class Block(nn.Module):
    def forward(self, hidden, cu):
        return hidden
class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Block()])
        self.config = config
    def forward(self, hidden, grid):
        index, lengths = helper(grid)
        other = torch.arange(grid.shape[0])
        mapping = {"full_attention": other, "window_attention": lengths}
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cu=mapping[self.config.layer_types[i]])
        return hidden
"""},
        config={"layer_types": ["full_attention", "window_attention"]},
        all_tensor_ops=True,
    )
    cls = analysis.class_registry["M"]
    producer = cls.forward_step_predecessor_args["layers"]["cu"]
    merge = cls.forward_operations[producer]
    assert merge.label == "Mutually exclusive branch"
    helper = next(attr for attr in merge.predecessors if "helper" in attr)
    assert any("arange" in attr for attr in merge.predecessors)
    assert (helper, 1) in merge.predecessor_ports


def test_a_conditionally_called_helper_referenced_by_a_merge_is_expanded():
    """A live helper outside the linear spine still publishes its operations."""
    analysis = analyze_sources(
        {Path("modeling_conditional_builder.py"): """
import torch
def build_full(x):
    return x + 1
def build_window(x):
    return x * 2
class Layer(torch.nn.Module):
    def forward(self, x, mask):
        return x + mask
class M(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([Layer()])
    def forward(self, x):
        if x is not None:
            masks = {"full": build_full(x), "window": build_window(x)}
        for i, layer in enumerate(self.layers):
            x = layer(x, masks[self.config.layer_types[i]])
        return x
"""},
        config={"layer_types": ["full", "window"]},
        all_tensor_ops=True,
    )
    cls = analysis.class_registry["M"]
    calls = {
        producer
        for operation in cls.forward_operations.values()
        for producer in operation.predecessors
        if "build_" in producer
    }
    assert len(calls) == 2, calls
    assert calls <= set(cls.multi_op_methods), (
        calls,
        set(cls.multi_op_methods),
    )
    assert {cls.multi_op_methods[call][0].label for call in calls} == {
        "Add",
        "Multiply",
    }


def test_a_config_selected_local_dispatch_resolves_from_source(tmp_path, monkeypatch):
    """A local selected from any class table resolves without executing it."""
    support = tmp_path / "dispatch_support.py"
    support.write_text("""
def eager_impl(x):
    return x + 1
def other_impl(x):
    return x * 2
class Interface:
    choices = {"eager": eager_impl, "other": other_impl}
TABLE = Interface()
def builder(config, x):
    implementation = TABLE[config.mode]
    return implementation(x)
""")
    monkeypatch.setattr(
        aa,
        "_module_origin",
        lambda module: str(support) if module == "dispatch_support" else None,
    )
    registry = aa._ParsedModuleRegistry()
    selected = registry.selected_table_functions("dispatch_support", {"mode": "eager"})
    assert selected["implementation"].name == "eager_impl"


def test_a_tuple_return_keeps_each_slot_after_the_helper_is_inlined():
    """Each returned tensor stays its own op once the helper is inlined.

    Inlining renames the primary return op to the call and prefixes the rest.
    Publishing the original names then finds neither slot, and every consumer
    reads the last op: the index and the sequence lengths become one tensor.
    """
    analysis = analyze_sources(
        {Path("modeling_two_returns.py"): """
import torch
def helper(x):
    left = x + 1
    right = x * 2
    return left, right
class M(torch.nn.Module):
    def forward(self, x):
        left, right = helper(x)
        return left + right
"""},
        all_tensor_ops=True,
    )
    cls = analysis.class_registry["M"]
    call = next(attr for attr in cls.forward_step_return_producers if "helper" in attr)
    slots = cls.forward_step_return_producers[call]
    assert len(slots) == 2, slots
    assert slots[0] and slots[1] and slots[0] != slots[1]
    labels = []
    for attr in slots:
        match = next(
            (op.label for op in cls.multi_op_methods[call] if op.attr_name == attr),
            None,
        )
        labels.append(match)
    assert labels == ["Add", "Multiply"], (slots, labels)


def test_a_starred_dict_call_reads_every_tensor_value():
    """``builder(**{"a": left, "b": right})`` reads both tensors, not just ``right``."""
    analysis = analyze_sources(
        {Path("modeling_star_dict.py"): """
import torch
class M(torch.nn.Module):
    def forward(self, x):
        left = x + 1
        right = x * 2
        specs = {"a": left, "b": right}
        return builder(**specs)
"""},
        all_tensor_ops=True,
    )
    cls = analysis.class_registry["M"]
    call = next(attr for attr in cls.forward_step_predecessors if "builder" in attr)
    preds = cls.forward_step_predecessors[call]
    labels = {
        cls.forward_operations[pred].label
        for pred in preds
        if pred in cls.forward_operations
    }
    assert {"Add", "Multiply"} <= labels, labels


def test_positional_activation_argument_uses_the_tower_config():
    """``MLP(dim, hidden, config.hidden_act)`` selects the activation.

    The key is a positional constructor argument. Keywords alone never bound
    it, so ``ACT2FN[hidden_act]`` stayed an unknown module. A vision tower and
    the text stack name different activations; the tower's own config wins.
    """
    analysis = analyze_sources(
        {Path("modeling_act.py"): """
import torch
from torch import nn
class VisionMLP(nn.Module):
    def __init__(self, dim, hidden, hidden_act):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
class Tower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = VisionMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
    def forward(self, x):
        return self.mlp(x)
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual = Tower(config.vision_config)
    def forward(self, x):
        return self.visual(x)
"""},
        config={
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "vision_config": {
                "hidden_act": "gelu",
                "hidden_size": 4,
                "intermediate_size": 8,
            },
        },
        all_tensor_ops=True,
    )
    assert analysis.class_registry["VisionMLP"].init_assignments["act"] == "GELU"


def test_a_tuple_unpack_inside_a_branch_keeps_each_slot():
    analysis = aa.analyze_sources(
        {"m.py": """
def rope(q, k):
    q_embed = q + k
    k_embed = k + q
    return q_embed, k_embed

class Attn:
    def forward(self, query, key, position_embeddings):
        query = query + key
        key = key + query
        if position_embeddings is not None:
            query, key = rope(query, key)
        return query, key
"""},
        all_tensor_ops=True,
    )
    ops = analysis.class_registry["Attn"].forward_operations
    merges = [op for op in ops.values() if op.label == "Mutually exclusive branch"]
    ordinals = sorted(port for op in merges for _producer, port in op.predecessor_ports)
    assert len(merges) == 2
    assert ordinals == [0, 1]


def test_a_pruned_merge_still_rewires_the_call_after_it():
    """``act_fn`` is a lambda, not a submodule, for this config -- the merge of
    its arm and ``gate``'s collapses onto ``gate`` alone, and ``proj`` must
    still be wired to ``gate`` rather than falling back to plain source order
    (which would point at the never-built ``act_fn``) or being left isolated."""
    analysis = aa.analyze_source(
        """
class Gate:
    def forward(self, x):
        return x

class Proj:
    def forward(self, x):
        return x

class MLP:
    def __init__(self, config):
        if config.hidden_act == "gated":
            self.gate = Gate()
        else:
            self.act_fn = lambda x: x
        self.proj = Proj()

    def forward(self, x):
        y = self.gate(x)
        if y is None:
            y = self.act_fn(x)
        return self.proj(y)
""",
        config={"hidden_act": "gated"},
        all_tensor_ops=True,
    )
    mlp = analysis.class_registry["MLP"]
    assert "act_fn" in mlp.unbuilt_attrs
    assert mlp.forward_step_predecessors.get("proj") == ("gate",)


def test_a_submodule_only_forward_records_which_argument_each_parameter_takes():
    """A residual saved before a norm stays distinct from the norm's output."""
    analysis = aa.analyze_source(
        """
class Norm:
    def forward(self, x):
        return x

class Gate:
    def forward(self, residual, hidden_states):
        return residual + hidden_states

class Layer:
    def __init__(self):
        self.norm1 = Norm()
        self.gate1 = Gate()
        self.norm2 = Norm()
        self.gate2 = Gate()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.gate1(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.gate2(residual, hidden_states)
        return hidden_states
""",
        all_tensor_ops=True,
    )
    layer = analysis.class_registry["Layer"]
    assert not layer.forward_step_predecessors
    aa.expand_class_forward_dataflow(layer, analysis.class_registry)
    args = layer.forward_step_predecessor_args
    assert args["gate1"] == {
        "residual": "@method_input",
        "hidden_states": "norm1",
    }
    assert args["gate2"] == {
        "residual": "gate1",
        "hidden_states": "norm2",
    }
    assert not layer.forward_operations


def test_extent_names_are_traced_from_the_binding_source():
    """Expand sizes record the shape reads the source assigned them to.

    A cold run does not take the cache arm, so ``past.shape[0]`` is not a
    candidate. The two arms that do run read different tensors, and both reads
    are kept until inference can see whether they name the same axis. An early
    return of empty slots does not erase a later read, and a size assigned only
    on the arm that builds the mask is still the size that call passes.
    """
    analysis = analyze_sources(
        {Path("modeling_extent_trace.py"): """
import torch
def lengths(inputs_embeds, attention_mask, past: Cache | None, encoder=None):
    if attention_mask is not None and len(attention_mask.shape) == 4:
        return True, None, None
    q_length = inputs_embeds.shape[1]
    if past is not None:
        kv_length = past.shape[0]
    else:
        if attention_mask is None:
            kv_length = encoder.shape[1] if encoder is not None else q_length
        else:
            kv_length, _offset = attention_mask.shape[-1], 0
    return False, q_length, kv_length
def paint(batch_size, q_length, kv_length):
    row = torch.arange(q_length)
    return row.expand(batch_size, -1, q_length, kv_length)
def build(inputs_embeds, attention_mask, past: Cache | None):
    early, q_length, kv_length = lengths(inputs_embeds, attention_mask, past)
    if early:
        return attention_mask
    batch_size = inputs_embeds.shape[0]
    return paint(batch_size=batch_size, q_length=q_length, kv_length=kv_length)
class M(torch.nn.Module):
    def forward(self, inputs_embeds, attention_mask, past: Cache | None = None):
        return build(inputs_embeds, attention_mask, past)
"""},
        all_tensor_ops=True,
    )
    details = [
        detail
        for ops in analysis.class_registry["M"].multi_op_methods.values()
        for op in ops
        if op.label == "Expand"
        for detail in op.details
        if str(detail).startswith("shape_locals:")
    ]
    assert len(details) == 1, details
    text = details[0]
    assert "batch_size=inputs_embeds.shape[0]" in text
    assert "q_length=inputs_embeds.shape[1]" in text
    assert "past.shape" not in text
    kv = next(
        item.strip()
        for item in text.split(",")
        if item.strip().startswith("kv_length=")
    )
    assert kv == "kv_length=inputs_embeds.shape[1]|attention_mask.shape[-1]"


def _slot_class(name: str, source: str):
    import ast
    import textwrap

    from TraceLens.ModelUtils.ast_analyze import ClassStructure

    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.ClassDef)
    return ClassStructure(
        name=name,
        node=node,
        init_assignments={},
        init_details={},
        forward_calls=[],
        norm_before=[],
    )


def _slot_registry():
    gate = _slot_class(
        "Gate",
        """
        class Gate:
            def forward(self, hidden_states):
                return topk_idx, topk_weight
        """,
    )
    gate.forward_return_order = ["topk_idx", "topk_weight"]
    gate.forward_return_slots = {"topk_idx": "@op_topk", "topk_weight": "@op_mul"}
    caller = _slot_class(
        "Block",
        """
        class Block:
            def forward(self, hidden_states):
                return hidden_states
        """,
    )
    caller.init_assignments = {"gate": "Gate"}
    return {"Gate": gate, "Block": caller}


def test_submodule_publishes_one_producer_per_return_slot() -> None:
    from TraceLens.ModelUtils import ast_analyze

    registry = _slot_registry()
    ast_analyze._publish_submodule_return_producers(registry["Block"], registry)
    assert registry["Block"].forward_step_return_producers == {
        "gate": ["@op_topk", "@op_mul"]
    }


def test_single_slot_return_publishes_nothing() -> None:
    from TraceLens.ModelUtils import ast_analyze

    registry = _slot_registry()
    registry["Gate"].forward_return_order = ["topk_idx"]
    registry["Gate"].forward_return_slots = {"topk_idx": "@op_topk"}
    ast_analyze._publish_submodule_return_producers(registry["Block"], registry)
    assert registry["Block"].forward_step_return_producers == {}


def test_matmul_with_parameter_displays_as_linear():
    import ast

    from TraceLens.ModelUtils.ast_analyze import (
        _forward_operations_from_forward,
        _self_config_values,
    )

    source = """
import torch
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4, 8))
    def forward(self, hidden_states):
        return torch.matmul(hidden_states, self.weight)
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    init = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    ops = _forward_operations_from_forward(
        forward,
        self_values=_self_config_values(init, {}),
        all_tensor_ops=False,
    )
    assert len(ops.operations) == 1
    op = ops.operations[0]
    assert op.label == "Linear"
    assert "weight" in op.external_inputs


def test_two_activation_matmul_displays_as_matmul():
    import ast

    from TraceLens.ModelUtils.ast_analyze import _forward_operations_from_forward

    source = """
import torch
class Block(torch.nn.Module):
    def forward(self, query, key):
        return torch.matmul(query, key.transpose(-1, -2))
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    ops = _forward_operations_from_forward(
        forward, self_values={}, all_tensor_ops=False
    )
    assert len(ops.operations) == 1
    op = ops.operations[0]
    assert op.label == "MatMul"
    assert not op.external_inputs


def test_sibling_method_only_forward_keeps_inline_math():
    import ast

    from TraceLens.ModelUtils.ast_analyze import (
        _apply_forward_analysis,
        _forward_delegates_only_to_sibling_methods,
        _forward_delegates_to_nothing,
        _forward_operations_from_forward,
        _forward_owns_tensor_math,
        _self_config_values,
    )

    source = """
import torch
class Rotary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0
    def forward(self, freqs):
        cos = freqs.cos() * self.scale
        sin = freqs.sin() * self.scale
        cos = self.recompose(cos)
        sin = self.recompose(sin)
        return cos, sin
    def recompose(self, freq):
        return torch.cat([freq, freq], dim=-1)
"""
    tree = ast.parse(source)
    cls = tree.body[-1]
    forward = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "forward"
    )
    init = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    method_names = {item.name for item in cls.body if isinstance(item, ast.FunctionDef)}
    forward_calls = ["recompose"]
    init_assignments = {}
    assert not _forward_owns_tensor_math(forward_calls, init_assignments)
    assert not _forward_delegates_to_nothing(cls.name, forward_calls)
    assert _forward_delegates_only_to_sibling_methods(
        forward_calls, init_assignments, method_names
    )
    analysis = _forward_operations_from_forward(
        forward, self_values=_self_config_values(init, {}), all_tensor_ops=False
    )
    _calls, operations, *_ = _apply_forward_analysis(
        forward,
        analysis,
        forward_calls=forward_calls,
        init_assignments=init_assignments,
    )
    assert any(op.label == "Multiply" for op in operations.values())


def test_mixes_modules_and_inline_ops_keeps_an_inline_add_beside_a_submodule():
    """``left, right = self.split(x); return self.proj(right) + left`` -- the
    Add is real computation the module owns, so wiring stays attached even
    though every call is otherwise a plain submodule."""
    from TraceLens.ModelUtils.ast_analyze import (
        _forward_mixes_modules_and_inline_ops,
        _forward_operations_from_forward,
    )

    source = """
class Model:
    def forward(self, x):
        left, right = self.split(x)
        return self.proj(right) + left
"""
    forward = ast.parse(source).body[0].body[0]
    ops = _forward_operations_from_forward(forward, self_values={}, all_tensor_ops=True)
    assert _forward_mixes_modules_and_inline_ops(
        ["split", "proj"],
        {"split": "Split", "proj": "Proj"},
        ops.operations,
        ops.step_predecessors,
    )


def test_mixes_modules_and_inline_ops_rewires_a_pruned_merge():
    """A call's predecessor dropping the plain source-order default (a pruned
    two-arm merge collapsed to one arm) counts as real rewiring even when no
    ``Merge`` operation node survives to report."""
    from TraceLens.ModelUtils.ast_analyze import _forward_mixes_modules_and_inline_ops

    assert _forward_mixes_modules_and_inline_ops(
        ["a", "b"],
        {"a": "A", "b": "B"},
        [],
        {"b": ("@method_input",)},
    )


def test_mixes_modules_and_inline_ops_is_false_without_predecessor_evidence():
    """No inline op and no predecessor evidence at all: nothing to add, so the
    caller's plain source-order wiring is left untouched."""
    from TraceLens.ModelUtils.ast_analyze import _forward_mixes_modules_and_inline_ops

    assert not _forward_mixes_modules_and_inline_ops(
        ["a", "b"],
        {"a": "A", "b": "B"},
        [],
        {},
    )
    # Plain source-order wiring (``b``'s only predecessor is the call right
    # before it): the loop runs to completion and finds no rewiring.
    assert not _forward_mixes_modules_and_inline_ops(
        ["a", "b"],
        {"a": "A", "b": "B"},
        [],
        {"b": ("a",)},
    )


def test_mixes_modules_and_inline_ops_is_false_with_no_module_calls():
    from TraceLens.ModelUtils.ast_analyze import _forward_mixes_modules_and_inline_ops

    assert not _forward_mixes_modules_and_inline_ops(["mystery"], {}, [], {})


def test_resolve_host_scalars_substitutes_through_lists_tuples_and_unary_ops():
    """``q_size = self.num_idx_heads * self.idx_head_dim`` then
    ``idx_qkv.split([q_size, self.idx_head_dim, -self.idx_head_dim], ...)`` --
    a folded local substitutes its concrete int, one this module's own
    (weaker) resolution could not fold substitutes its own RHS expression
    instead, and both must come through unscathed inside a List, a Tuple, and
    a UnaryOp operand, not just a bare Name."""
    from TraceLens.ModelUtils.ast_analyze import _ForwardOperationExtractor

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.host_scalar_values["head_dim"] = 128
    extractor.host_scalar_exprs["q_size"] = ast.parse(
        "num_idx_heads * idx_head_dim", mode="eval"
    ).body

    node = ast.parse(
        "[head_dim, q_size, -idx_head_dim, (q_size, idx_head_dim)]", mode="eval"
    ).body
    resolved = extractor._resolve_host_scalars(node)
    assert (
        ast.unparse(resolved) == "[128, num_idx_heads * idx_head_dim, -idx_head_dim, "
        "(num_idx_heads * idx_head_dim, idx_head_dim)]"
    )


def test_resolve_host_scalars_guards_a_local_reassigned_from_itself():
    """``x = x + 1``: expanding ``x``'s own definition would recurse into
    itself forever without the ``_seen`` guard. An unexpandable local is left
    as a bare name instead of looping."""
    from TraceLens.ModelUtils.ast_analyze import _ForwardOperationExtractor

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.host_scalar_exprs["x"] = ast.parse("x + 1", mode="eval").body

    node = ast.parse("x", mode="eval").body
    resolved = extractor._resolve_host_scalars(node)
    assert ast.unparse(resolved) == "x + 1"


def test_expression_at_operation_line_falls_back_to_the_enclosing_statement():
    """``@op_l{line}_c{col}_...`` names the operation's own AST position. When
    nothing at that exact position is a labelled BinOp/Call (the position is
    the enclosing STATEMENT's own, not an inner expression's -- a bare
    ``return x``, or an assignment whose RHS starts further right than the
    statement itself), the expression is read off whichever statement shape
    sits there: a return's value, an assignment's RHS, an annotated
    assignment's RHS, or an augmented assignment's RHS. An attr_name that
    does not even match the position pattern resolves to nothing."""
    from TraceLens.ModelUtils.ast_analyze import _expression_at_operation_line

    def op_at(stmt: ast.stmt) -> str:
        return f"@op_l{stmt.lineno}_c{stmt.col_offset}_whatever"

    return_func = ast.parse("def forward(self, x):\n    return x\n").body[0]
    result = _expression_at_operation_line(
        return_func, op_at(return_func.body[0]), "SomeLabel"
    )
    assert ast.unparse(result) == "x"

    assign_func = ast.parse("def forward(self, x):\n    y = x + 1\n").body[0]
    result = _expression_at_operation_line(
        assign_func, op_at(assign_func.body[0]), "SomeLabel"
    )
    assert ast.unparse(result) == "x + 1"

    ann_assign_func = ast.parse("def forward(self, x):\n    y: int = x\n").body[0]
    result = _expression_at_operation_line(
        ann_assign_func, op_at(ann_assign_func.body[0]), "SomeLabel"
    )
    assert ast.unparse(result) == "x"

    aug_assign_func = ast.parse("def forward(self, x):\n    y = 0\n    y += x\n").body[
        0
    ]
    result = _expression_at_operation_line(
        aug_assign_func, op_at(aug_assign_func.body[1]), "SomeLabel"
    )
    assert ast.unparse(result) == "x"

    assert _expression_at_operation_line(return_func, "not_a_match", "Label") is None


def test_subscript_resize_dims_folds_a_lower_and_upper_bound_pair():
    """``x[..., :output_width]`` folds a bare upper bound to its own value; an
    ``a:b`` slice folds to the span between them; a span that folds negative
    (a lower bound above the upper one) is not a real resize and is skipped,
    same as an unresolvable bound, a full ``:`` slice, or a stepped slice.
    An axis after an ``Ellipsis`` is numbered from the end."""
    import ast

    from TraceLens.ModelUtils.ast_analyze import _ForwardOperationExtractor

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.host_scalar_values = {"B": 32, "A": 8}

    def resize(expr: str) -> list[tuple[int, int]]:
        subscript = ast.parse(expr, mode="eval").body
        return extractor._subscript_resize_dims(subscript.slice)

    assert resize("x[:B]") == [(0, 32)]
    assert resize("x[A:B]") == [(0, 24)]
    assert resize("x[B:A]") == []
    assert resize("x[..., :B]") == [(-1, 32)]
    assert resize("x[:unknown]") == []
    assert resize("x[:]") == []
    assert resize("x[::2]") == []


def test_rebind_producer_token_renames_the_bare_and_the_named_boundary_token():
    """A producer token is renamed the same way an operation's own operands
    are: the bare primary-input token (``@method_input``) renames to whatever
    the callee's own primary parameter now maps to, but collapses back to the
    generic token when that happens to BE the caller's own primary (nothing
    distinguishes it then); a per-parameter secondary token
    (``@method_input:attention_mask``) renames through the mapping the same
    way, and ALSO collapses to the generic token if the renamed name turns
    out to be the caller's primary. A plain producer attr just goes through
    the mapping like any other name."""
    from TraceLens.ModelUtils.ast_analyze import (
        FORWARD_METHOD_INPUT,
        _rebind_producer_token,
    )

    renamed = _rebind_producer_token(
        FORWARD_METHOD_INPUT,
        {"x": "hidden_states"},
        callee_primary="x",
        caller_primary="other_primary",
    )
    assert renamed == "hidden_states"

    collapses_to_generic = _rebind_producer_token(
        FORWARD_METHOD_INPUT,
        {"x": "hidden_states"},
        callee_primary="x",
        caller_primary="hidden_states",
    )
    assert collapses_to_generic == FORWARD_METHOD_INPUT

    no_callee_primary = _rebind_producer_token(
        FORWARD_METHOD_INPUT, {}, callee_primary=None, caller_primary="hidden_states"
    )
    assert no_callee_primary == FORWARD_METHOD_INPUT

    named_renamed = _rebind_producer_token(
        "@method_input:attention_mask",
        {"attention_mask": "mask_param"},
        callee_primary="x",
        caller_primary="hidden_states",
    )
    assert named_renamed == "mask_param"

    named_collapses = _rebind_producer_token(
        "@method_input:attention_mask",
        {"attention_mask": "hidden_states"},
        callee_primary="x",
        caller_primary="hidden_states",
    )
    assert named_collapses == FORWARD_METHOD_INPUT

    plain_mapped = _rebind_producer_token(
        "q_proj",
        {"q_proj": "renamed_q"},
        callee_primary="x",
        caller_primary="hidden_states",
    )
    assert plain_mapped == "renamed_q"

    plain_unmapped = _rebind_producer_token(
        "unmapped", {}, callee_primary="x", caller_primary="hidden_states"
    )
    assert plain_unmapped == "unmapped"


def test_self_call_argument_rebind_covers_positional_and_keyword_shapes():
    """Only a bare name rebinds a callee parameter onto the caller's own
    tensor; a constant carries no rename and is skipped; anything else
    (a starred arg, a computed expression, too many positional args) makes
    the call opaque to this rebind. A bare ``**kwargs`` forward is harmless
    (no rename it could offer) but a computed keyword dict is not."""
    from TraceLens.ModelUtils.ast_analyze import _self_call_argument_rebind

    callee = ast.parse("def method(self, x, y, z=None): pass").body[0]

    def rebind(expr: str):
        return _self_call_argument_rebind(ast.parse(expr).body[0].value, callee)

    assert rebind("self.method(a, 5)") == {"x": "a"}
    assert rebind("self.method(a, b, c, d)") is None
    assert rebind("self.method(*args)") is None
    assert rebind("self.method(**extra_kwargs)") == {}
    assert rebind("self.method(**{'a': 1})") is None
    assert rebind("self.method(z=w)") == {"z": "w"}
    assert rebind("self.method(z=5)") == {}
    assert rebind("self.method(z=w + 1)") is None


def test_fold_host_int_covers_every_arithmetic_op_and_a_zero_divisor():
    """``q_size = self.num_idx_heads * self.idx_head_dim`` folds through
    ``+ - * // %`` over already-folded host-scalar locals; true division is
    outside the set this gives a constant width for (``/`` always returns a
    float, never the integer bound a slice needs) and a zero divisor is
    caught rather than raising."""
    from TraceLens.ModelUtils.ast_analyze import _ForwardOperationExtractor

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.host_scalar_values = {"a": 6, "b": 4}

    def fold(expr: str) -> int | None:
        return extractor._fold_host_int(ast.parse(expr, mode="eval").body)

    assert fold("a + b") == 10
    assert fold("a - b") == 2
    assert fold("a * b") == 24
    assert fold("a // b") == 1
    assert fold("a % b") == 2
    assert fold("a // 0") is None
    assert fold("a % 0") is None
    assert fold("a / b") is None


def test_apply_branch_alternatives_expands_both_the_operation_and_the_step():
    """After an if/else merge only the survivor's producer (``norm_a``) lives
    in ``var_producer``; a consumer that read the merged variable must be
    expanded to depend on the recorded alternative (``norm_b``) too, so the
    branch the model did not take on THIS call still stays live rather than
    orphaned into a dead end. Both an operation's predecessors and a step's
    own recorded predecessors are expanded alike. With no recorded
    alternatives at all, this is a pure no-op."""
    from TraceLens.ModelUtils.ast_analyze import (
        ForwardOperation,
        _ForwardOperationExtractor,
    )

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.branch_alternatives = {"norm_a": {"norm_b"}}
    extractor.operations = [
        ForwardOperation("consumer", "Add", "", predecessors=("norm_a",))
    ]
    extractor.step_predecessors = {"head": ("norm_a",)}

    extractor._apply_branch_alternatives()

    assert extractor.operations[0].predecessors == ("norm_a", "norm_b")
    assert extractor.step_predecessors["head"] == ("norm_a", "norm_b")

    untouched = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    untouched.operations = [
        ForwardOperation("consumer", "Add", "", predecessors=("norm_a",))
    ]
    untouched._apply_branch_alternatives()
    assert untouched.operations[0].predecessors == ("norm_a",)


def test_reconstruct_attention_step_recovers_qkv_from_the_dispatch_call():
    """A dispatched attention call (``attention_interface(self, q, k, v,
    attention_mask)``) that a branch or comprehension hid from the regular
    walk still leaves each operand's producer resolvable in ``var_producer``;
    recovered here, named after the argument so the kernel ports read
    ``q``/``k``/``v`` rather than nothing. Already-populated predecessors, a
    body with no dispatch call, and operands with no resolvable producer are
    all left alone."""
    from TraceLens.ModelUtils.ast_analyze import (
        SYNTHETIC_ATTENTION,
        _ForwardOperationExtractor,
    )

    body = ast.parse(
        "attn_output, attn_weights = attention_interface(self, q, k, v, attention_mask)"
    ).body

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.var_producer = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
    extractor.step_predecessors = {}
    extractor.step_predecessor_args = {}
    extractor._reconstruct_attention_step(body)
    assert extractor.step_predecessors[SYNTHETIC_ATTENTION] == (
        "q_proj",
        "k_proj",
        "v_proj",
    )
    assert extractor.step_predecessor_args[SYNTHETIC_ATTENTION] == {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
    }

    already_wired = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    already_wired.step_predecessors = {SYNTHETIC_ATTENTION: ("existing",)}
    already_wired._reconstruct_attention_step(body)
    assert already_wired.step_predecessors[SYNTHETIC_ATTENTION] == ("existing",)

    no_call = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    no_call.step_predecessors = {}
    no_call._reconstruct_attention_step(ast.parse("x = y + 1").body)
    assert SYNTHETIC_ATTENTION not in no_call.step_predecessors

    unresolvable = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    unresolvable.var_producer = {}
    unresolvable.step_predecessors = {}
    unresolvable._reconstruct_attention_step(body)
    assert SYNTHETIC_ATTENTION not in unresolvable.step_predecessors


def test_drop_phantom_attention_steps_removes_the_empty_kernel_and_its_readers():
    """A ``@attention`` step hoisted up from a submodule's own forward (``for
    blk in self.blocks: blk(...)``) carries no predecessors at this scope --
    never a genuine call -- and would otherwise form a cycle once the exporter
    mirrors it into the real nested diagram. It, its step-predecessor-args
    entry, and its own operation are dropped outright; any OTHER step or
    operation that named it as a predecessor has that reference stripped
    rather than being dropped itself. A step with real predecessors is
    untouched."""
    from TraceLens.ModelUtils.ast_analyze import (
        SYNTHETIC_ATTENTION,
        ForwardOperation,
        _ForwardOperationExtractor,
    )

    extractor = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    extractor.step_predecessors = {
        SYNTHETIC_ATTENTION: (),
        "head": (SYNTHETIC_ATTENTION,),
    }
    extractor.step_predecessor_args = {SYNTHETIC_ATTENTION: {"q": "q_proj"}}
    extractor.operations = [
        ForwardOperation(SYNTHETIC_ATTENTION, "AttentionOp", "", predecessors=()),
        ForwardOperation("head", "Linear", "", predecessors=(SYNTHETIC_ATTENTION,)),
    ]

    extractor._drop_phantom_attention_steps()

    assert SYNTHETIC_ATTENTION not in extractor.step_predecessors
    assert extractor.step_predecessors["head"] == ()
    assert SYNTHETIC_ATTENTION not in extractor.step_predecessor_args
    assert [op.attr_name for op in extractor.operations] == ["head"]
    assert extractor.operations[0].predecessors == ()

    real_attention = _ForwardOperationExtractor(self_values={}, all_tensor_ops=True)
    real_attention.step_predecessors = {SYNTHETIC_ATTENTION: ("q_proj",)}
    real_attention.operations = []
    real_attention._drop_phantom_attention_steps()
    assert real_attention.step_predecessors[SYNTHETIC_ATTENTION] == ("q_proj",)


def test_sibling_method_delegation_rejects_submodule_and_free_calls():
    from TraceLens.ModelUtils.ast_analyze import (
        _forward_delegates_only_to_sibling_methods,
    )

    method_names = {"forward", "helper"}
    assert not _forward_delegates_only_to_sibling_methods(
        ["sub"], {"sub": "SubModule"}, method_names
    )
    assert not _forward_delegates_only_to_sibling_methods(["mystery"], {}, method_names)
    assert not _forward_delegates_only_to_sibling_methods([], {}, method_names)


def test_trivial_self_attr_getter_names_the_attribute_it_returns():
    """``def get_input_embeddings(self): return self.embed_tokens`` -> ``embed_tokens``."""
    from TraceLens.ModelUtils.ast_analyze import _trivial_self_attr_getter

    func = ast.parse(
        "def get_input_embeddings(self):\n    return self.embed_tokens\n"
    ).body[0]
    assert _trivial_self_attr_getter(func) == "embed_tokens"


def test_trivial_self_attr_getter_sees_through_a_leading_docstring():
    from TraceLens.ModelUtils.ast_analyze import _trivial_self_attr_getter

    func = ast.parse(
        "def get_input_embeddings(self):\n"
        '    """Return the embedding submodule."""\n'
        "    return self.embed_tokens\n"
    ).body[0]
    assert _trivial_self_attr_getter(func) == "embed_tokens"


def test_trivial_self_attr_getter_rejects_a_getter_with_a_side_effect():
    """More than ``return self.<attr>`` in the body is not trivial."""
    from TraceLens.ModelUtils.ast_analyze import _trivial_self_attr_getter

    func = ast.parse(
        "def get_input_embeddings(self):\n"
        "    self.touched = True\n"
        "    return self.embed_tokens\n"
    ).body[0]
    assert _trivial_self_attr_getter(func) is None


def test_trivial_self_attr_getter_rejects_a_non_attribute_return():
    from TraceLens.ModelUtils.ast_analyze import _trivial_self_attr_getter

    func = ast.parse("def get_scale(self):\n    return 1.0\n").body[0]
    assert _trivial_self_attr_getter(func) is None


def test_trivial_self_attr_getter_rejects_a_nested_attribute_chain():
    """``self.container.inner`` is not a bare ``self.<attr>``."""
    from TraceLens.ModelUtils.ast_analyze import _trivial_self_attr_getter

    func = ast.parse("def get_inner(self):\n    return self.container.inner\n").body[0]
    assert _trivial_self_attr_getter(func) is None


def test_a_trivial_getter_accessor_resolves_to_the_same_submodule_as_direct_attr():
    """``self.get_input_embeddings()(input_ids)`` wires to the same submodule as
    ``self.embed_tokens(input_ids)`` would.

    The standard HF/vLLM/SGLang accessor interface exists so a caller never has
    to know the submodule's own attribute name; seeing through it keeps both
    call forms wired to the same child instead of the accessor form collapsing
    into a phantom kernel tile with no producer.
    """
    from TraceLens.ModelUtils.ast_analyze import analyze_sources
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter
    from TraceLens.ModelUtils.block_tree import build_block_node

    source = """
import torch


class Embed(torch.nn.Module):
    def forward(self, x):
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = Embed()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        embeds = self.get_input_embeddings()
        return embeds(input_ids)
"""
    analysis = analyze_sources({"model.py": source}, all_tensor_ops=True)
    root = build_block_node(
        attr_name="model",
        class_name="Model",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter([]),
    )
    child_names = {child.attr_name for child in root.children} | {
        child.class_name for child in root.children
    }
    assert "embed_tokens" in child_names or "Embed" in child_names, child_names
    assert not any(
        child.attr_name == "embeds" for child in root.children
    ), "the accessor call should resolve to the real submodule, not a phantom tile"


def test_expanding_a_host_int_is_not_the_activation() -> None:
    from TraceLens.ModelUtils.ast_analyze import analyze_sources
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter
    from TraceLens.ModelUtils.block_tree import build_block_node
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    analysis = analyze_sources(
        {"m.py": """
import torch
class Tower(torch.nn.Module):
    def forward(self, pixel_values, grid_thw):
        size = torch.tensor([2]).expand(len(grid_thw))
        return size
"""},
        all_tensor_ops=True,
    )
    root = build_block_node(
        attr_name="tower",
        class_name="Tower",
        registry=analysis.class_registry,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    graph = build_computation_graph(root, basic_ops=BasicOpFilter.for_detailed())
    expand = next(
        spec
        for spec in graph.nodes
        if spec.block is not None and spec.block.attr_name.endswith("_expand")
    )
    index = graph.nodes.index(expand)
    assert expand.constant
    assert not any(target == index for _source, target in graph.links)


def _attention_stmt(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


class TestBoundaryOperandIsRecorded:
    """A forward parameter written into an attention call is an operand."""

    def test_a_positional_forward_parameter_is_an_operand(self) -> None:
        captured: dict[str, list[str]] = {}
        _capture_attention_inputs(
            _attention_stmt(
                "out, _ = attention_interface(self, q, k, v, attention_mask)"
            ),
            {"q": ["@op_q"], "k": ["@op_k"], "v": ["@op_v"]},
            captured,
            forward_input_names={"attention_mask", "hidden_states"},
        )
        assert "attention_mask" in captured, captured
        assert captured["attention_mask"] == []

    def test_the_same_parameter_by_keyword_is_recorded_identically(self) -> None:
        positional: dict[str, list[str]] = {}
        keyword: dict[str, list[str]] = {}
        names = {"attention_mask"}
        _capture_attention_inputs(
            _attention_stmt("out = attention_interface(self, q, attention_mask)"),
            {"q": ["@op_q"]},
            positional,
            forward_input_names=names,
        )
        _capture_attention_inputs(
            _attention_stmt(
                "out = attention_interface(self, q, attention_mask=attention_mask)"
            ),
            {"q": ["@op_q"]},
            keyword,
            forward_input_names=names,
        )
        assert positional == keyword

    def test_a_local_that_is_not_a_forward_parameter_is_still_skipped(self) -> None:
        captured: dict[str, list[str]] = {}
        _capture_attention_inputs(
            _attention_stmt("out = attention_interface(self, q, scratch)"),
            {"q": ["@op_q"]},
            captured,
            forward_input_names={"attention_mask"},
        )
        assert "scratch" not in captured
