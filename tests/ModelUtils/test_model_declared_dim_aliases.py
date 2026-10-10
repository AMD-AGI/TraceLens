###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The names a model reads its dimensions under come from the model.

Modeling code reads ``config.num_local_experts`` against a checkpoint that
spells it ``n_routed_experts``, so something has to bridge the two. That used to
be a hardcoded table of spellings models MIGHT use, which has a flaw no amount
of curation fixes: it claims a name GLOBALLY. ``n_heads`` was listed as another
word for the attention head count, but in GLM it is the sparse indexer's own::

    self.n_heads: int = config.index_n_heads     # 32, not the 64 attention heads

and the table, registered first, shadowed the real value.

A config class states its renames itself, which is what ``attribute_map`` is
for, and it states them PER CLASS -- GLM's vision tower renames
``num_attention_heads`` to ``num_heads`` while its text tower does not, a
distinction no single global list can express.

Read from source, never imported: analysis does not run model code, and a
checkpoint's configuration module is the checkpoint's own code.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.shape_inference import (
    ShapeContext,
    _model_declared_aliases,
)

CONFIG_SOURCE = '''
"""A checkpoint's configuration module."""


class DemoConfig:
    model_type = "demo"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }


class DemoVisionConfig:
    model_type = "demo_vision"
    attribute_map = {"num_attention_heads": "num_heads"}
'''


def _spec_with_source(tmp_path: Path, source: str = CONFIG_SOURCE) -> ArchitectureSpec:
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "configuration_demo.py").write_text(
        textwrap.dedent(source), encoding="utf-8"
    )
    modeling = directory / "modeling_demo.py"
    modeling.write_text("", encoding="utf-8")
    return ArchitectureSpec(
        name="demo",
        model_type="demo",
        code_paths=[str(modeling)],
        raw_config={"n_routed_experts": 288, "moe_intermediate_size": 1536},
    )


class TestWhatTheModelDeclares:
    def test_each_rename_is_read(self, tmp_path: Path) -> None:
        declared = _model_declared_aliases(_spec_with_source(tmp_path))
        assert declared["num_local_experts"] == "n_routed_experts"
        assert declared["intermediate_size"] == "moe_intermediate_size"

    def test_a_sub_config_s_own_rename_is_read_too(self, tmp_path: Path) -> None:
        """GLM's vision tower renames a name its text tower leaves alone."""
        declared = _model_declared_aliases(_spec_with_source(tmp_path))
        assert declared["num_attention_heads"] == "num_heads"

    def test_a_model_declaring_nothing_yields_nothing(self, tmp_path: Path) -> None:
        spec = _spec_with_source(tmp_path, "class Plain:\n    model_type = 'plain'\n")
        assert _model_declared_aliases(spec) == {}

    def test_source_that_does_not_parse_is_skipped(self, tmp_path: Path) -> None:
        spec = _spec_with_source(tmp_path, "class Broken(:\n")
        assert _model_declared_aliases(spec) == {}

    def test_a_spec_with_no_source_yields_nothing(self) -> None:
        spec = ArchitectureSpec(name="x", model_type="x")
        assert _model_declared_aliases(spec) == {}


class TestTheRenameReachesTheDimensions:
    def test_the_name_the_code_reads_resolves(self, tmp_path: Path) -> None:
        dims = ShapeContext.from_spec(_spec_with_source(tmp_path)).dims
        assert dims["num_local_experts"] == 288, "the code's spelling"
        assert dims["n_routed_experts"] == 288, "and the checkpoint's"

    def test_a_name_the_checkpoint_defines_itself_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """A declared rename is a fallback; what the config states wins.

        GLM declares ``num_attention_heads -> num_heads``, and its vision
        ``num_heads`` is 16 while the text tower really does have 64 attention
        heads. Letting the rename overwrite would hand the text tower 16.
        """
        spec = _spec_with_source(tmp_path)
        spec.raw_config = {
            "n_routed_experts": 288,
            "num_attention_heads": 64,
            "num_heads": 16,
        }
        dims = ShapeContext.from_spec(spec).dims
        assert dims["num_attention_heads"] == 64
        assert dims["num_heads"] == 16

    def test_a_rename_pointing_at_a_missing_key_registers_nothing(
        self, tmp_path: Path
    ) -> None:
        spec = _spec_with_source(tmp_path)
        spec.raw_config = {"n_routed_experts": 288}
        dims = ShapeContext.from_spec(spec).dims
        assert "intermediate_size" not in dims, "moe_intermediate_size is absent"
