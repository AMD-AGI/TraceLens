###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A config value the model's own code writes back overrules the checkpoint.

``config.json`` records what the model was ASKED for. The model's ``__init__``
can overrule it and leave nothing behind to say so: Kimi-K3 replaces whatever
attention implementation it is handed with its own, logging that it is ignoring
the request. Every branch keyed on that attribute then resolved against a value
the model had already discarded, so the forwards were read against a
configuration the model never runs.

This is the counterpart of the declared-defaults pass. A default fills a key the
checkpoint states nowhere; an override replaces a key the model states for
itself -- so unlike a default it is MEANT to shadow, including a value a
sub-config stores, because that is exactly what the constructor overwrote.

Taken only when every assignment agrees on one literal: a value set
conditionally has not been forced, and picking an arm would resolve branches
against a value the model may never hold.
"""

from __future__ import annotations

from pathlib import Path

from TraceLens.ModelUtils.config_resolve import declared_config_overrides


def _sources(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "modeling_thing.py"
    path.write_text(body, encoding="utf-8")
    return [str(path)]


class TestWhatCountsAsForced:
    def test_a_constructor_that_overwrites_what_it_was_given(self, tmp_path) -> None:
        body = """
class Model:
    def __init__(self, config):
        config._attn_implementation = "flash_attention_2"
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {
            "_attn_implementation": "flash_attention_2"
        }

    def test_both_arms_agreeing_is_still_forced(self, tmp_path) -> None:
        """The usual spelling: warn-and-replace, or set when nothing was asked."""
        body = """
class Model:
    def __init__(self, config):
        if getattr(config, "_attn_implementation", None) is not None:
            if config._attn_implementation != "flash_attention_2":
                config._attn_implementation = "flash_attention_2"
        else:
            config._attn_implementation = "flash_attention_2"
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {
            "_attn_implementation": "flash_attention_2"
        }

    def test_it_reads_the_kept_config_too(self, tmp_path) -> None:
        body = """
class Model:
    def __init__(self, config):
        self.config = config
        self.config.use_cache = False
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {
            "use_cache": False
        }


class TestWhatIsNotForced:
    def test_arms_that_disagree(self, tmp_path) -> None:
        """Set conditionally, so the model does not always hold either value."""
        body = """
class Model:
    def __init__(self, config):
        if config.fast:
            config.impl = "a"
        else:
            config.impl = "b"
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {}

    def test_a_value_that_is_not_a_literal(self, tmp_path) -> None:
        """Computed from something else, so the source does not state it."""
        body = """
class Model:
    def __init__(self, config):
        config.head_dim = config.hidden_size // config.num_heads
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {}

    def test_an_assignment_to_some_other_object(self, tmp_path) -> None:
        body = """
class Model:
    def __init__(self, config):
        self.impl = "flash_attention_2"
        other.impl = "sdpa"
"""
        assert declared_config_overrides(_sources(tmp_path, body)) == {}

    def test_a_file_that_cannot_be_parsed(self, tmp_path) -> None:
        path = tmp_path / "modeling_broken.py"
        path.write_text("def (:", encoding="utf-8")
        assert declared_config_overrides([str(path)]) == {}

    def test_no_sources_at_all(self) -> None:
        assert declared_config_overrides([]) == {}

    def test_a_directory_that_cannot_be_listed(self, tmp_path, monkeypatch) -> None:
        """An unreadable source directory declines, rather than taking the
        export down over a config value it merely could not read."""
        body = "class Model:\n    def __init__(self, config):\n        config.x = 1\n"
        sources = _sources(tmp_path, body)

        def _refuse(self, _pattern):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "glob", _refuse)
        assert declared_config_overrides(sources) == {}
