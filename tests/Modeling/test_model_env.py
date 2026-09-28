###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-model dependency detection.

These cover the decision logic only -- nothing here installs a package, writes a
dependency directory, or reaches the network.
"""

from __future__ import annotations

import json

import pytest

from TraceLens.ModelUtils import model_env


class _Config:
    def __init__(self, architectures=None, auto_map=None):
        self.architectures = architectures or []
        self.auto_map = auto_map or {}


def _patch_autoconfig(monkeypatch, config):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: config),
    )


def test_native_architecture_needs_no_separate_environment(monkeypatch):
    """A class the installed transformers already defines is never a mismatch."""
    import transformers

    _patch_autoconfig(monkeypatch, _Config(architectures=["AutoModel"]))
    assert getattr(transformers, "AutoModel", None) is not None
    assert model_env.detect_env_mismatch("some/checkpoint") is None


def test_remote_code_that_imports_cleanly_is_not_a_mismatch(monkeypatch):
    """Remote code alone is not a reason -- only a failing import is."""
    _patch_autoconfig(
        monkeypatch,
        _Config(
            architectures=["NotInstalledForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_x.NotInstalledForCausalLM"},
        ),
    )
    import transformers.dynamic_module_utils as dmu

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", lambda *a, **k: object)
    assert model_env.detect_env_mismatch("some/checkpoint") is None


def test_remote_code_failing_to_import_is_a_mismatch(monkeypatch):
    """The real Kimi-K3 shape: remote code importing a removed symbol."""
    _patch_autoconfig(
        monkeypatch,
        _Config(
            architectures=["NotInstalledForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_x.NotInstalledForCausalLM"},
        ),
    )
    import transformers.dynamic_module_utils as dmu

    def _boom(*_args, **_kwargs):
        raise ImportError("cannot import name 'OutputRecorder'")

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", _boom)
    reason = model_env.detect_env_mismatch("some/checkpoint")
    assert reason is not None
    assert "OutputRecorder" in reason


def test_non_import_failure_is_not_an_environment_problem(monkeypatch):
    """A missing config key is a real error to surface, not a reason to reprovision."""
    _patch_autoconfig(
        monkeypatch,
        _Config(
            architectures=["NotInstalledForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_x.NotInstalledForCausalLM"},
        ),
    )
    import transformers.dynamic_module_utils as dmu

    def _boom(*_args, **_kwargs):
        raise AttributeError("no attribute 'temporal_patch_size'")

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", _boom)
    assert model_env.detect_env_mismatch("some/checkpoint") is None


def test_required_version_read_from_local_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "x", "transformers_version": "4.56.2"}),
        encoding="utf-8",
    )
    assert model_env.required_transformers_version(tmp_path) == "4.56.2"


def test_required_version_absent_is_none(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "x"}), encoding="utf-8"
    )
    assert model_env.required_transformers_version(tmp_path) is None


def test_env_path_is_slugged_per_checkpoint_and_version(tmp_path):
    path = model_env.env_path_for("moonshotai/Kimi-K3", "4.56.2", tmp_path)
    assert path.parent == tmp_path
    assert path.name == "moonshotai_Kimi-K3-transformers-4.56.2"
    assert "/" not in path.name


@pytest.mark.parametrize(
    "output, expected",
    [
        ("ImportError: Plese run `pip install -U fla-core`", ["fla-core"]),
        ("ModuleNotFoundError: No module named 'einops'", ["einops"]),
        ("ModuleNotFoundError: No module named 'triton.ops'", ["triton"]),
        ("some unrelated failure", []),
    ],
)
def test_missing_requirement_parsing(output, expected):
    assert model_env._missing_requirements(output) == expected


def test_masked_import_error_still_offers_the_real_cause():
    """fla-core raises its own message when ``triton`` is what is missing.

    Following the friendly hint alone reinstalls the wrapper forever, so the
    chained original must stay in the candidate list behind it.
    """
    output = (
        "ModuleNotFoundError: No module named 'triton'\n"
        "During handling of the above exception, another exception occurred:\n"
        "ImportError: Plese run `pip install -U fla-core`"
    )
    assert model_env._missing_requirements(output) == ["fla-core", "triton"]


def test_child_process_never_reprovisions(monkeypatch):
    """The re-executed run must not recurse into provisioning."""
    monkeypatch.setenv(model_env.REEXEC_ENV_FLAG, "1")

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("detection must not run inside the child environment")

    monkeypatch.setattr(model_env, "detect_env_mismatch", _unexpected)
    assert model_env.reexec_in_model_env("some/checkpoint", []) is None


def test_no_reexec_when_checkpoint_runs_here(monkeypatch):
    monkeypatch.delenv(model_env.REEXEC_ENV_FLAG, raising=False)
    monkeypatch.setattr(model_env, "detect_env_mismatch", lambda *_a, **_k: None)
    assert model_env.reexec_in_model_env("some/checkpoint", []) is None


def test_no_reexec_when_config_names_no_version(monkeypatch):
    """Proven mismatch but no pin to honour -- carry on rather than guess."""
    monkeypatch.delenv(model_env.REEXEC_ENV_FLAG, raising=False)
    monkeypatch.setattr(model_env, "detect_env_mismatch", lambda *_a, **_k: "broken")
    monkeypatch.setattr(model_env, "required_transformers_version", lambda *_a: None)

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("must not provision without a version to pin")

    monkeypatch.setattr(model_env, "ensure_model_dependencies", _unexpected)
    assert model_env.reexec_in_model_env("some/checkpoint", []) is None
