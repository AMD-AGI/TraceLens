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
    _assign_target,
    _build_components,
    _classify_role,
    _config_value,
    _expr_name,
    _extract_decoder_layer_conditionals,
    _find_positional_module,
    _functional_call_name,
    _if_has_competing_assigns,
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


def test_assign_target_variants():
    assert _assign_target(ast.parse("x = 1").body[0]) == "x"
    # tuple target -> not a single Name
    assert _assign_target(ast.parse("x, y = 1, 2").body[0]) is None
    # attribute target -> not a Name
    assert _assign_target(ast.parse("self.x = 1").body[0]) is None
    # not an assignment
    assert _assign_target(ast.parse("return 1").body[0]) is None


def test_if_has_competing_assigns():
    competing = ast.parse(
        "if flag:\n    y = a()\nelse:\n    y = b()\n"
    ).body[0]
    assert _if_has_competing_assigns(competing) is True

    distinct = ast.parse(
        "if flag:\n    y = a()\nelse:\n    z = b()\n"
    ).body[0]
    assert _if_has_competing_assigns(distinct) is False


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
    assert _pick_model_class_by_structure({"X": _cls("X", assignments={"p": "Linear"})}) is None

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

_COND_SOURCE = '''
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
'''


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
# Additional cheap edge-branch coverage
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
    assert _config_value(_expr("config.flag is config.off"), config, sv) is False  # 1448
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
    decoder = _cls("Dec", assignments={"self_attn": "FullAttention"}, calls=["self_attn"])
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
