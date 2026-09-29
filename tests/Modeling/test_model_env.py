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
import os
import sys
from pathlib import Path

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


# --------------------------------------------------------------------------- #
# Environment root, install command shape, and child environment
# --------------------------------------------------------------------------- #
def test_env_root_prefers_the_override_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(model_env.ENV_ROOT_VAR, str(tmp_path / "elsewhere"))
    assert model_env.default_env_root() == tmp_path / "elsewhere"


def test_env_root_defaults_under_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(model_env.ENV_ROOT_VAR, raising=False)
    monkeypatch.setattr(model_env.Path, "home", classmethod(lambda cls: tmp_path))
    assert (
        model_env.default_env_root() == tmp_path / ".cache" / "tracelens" / "model-envs"
    )


def test_pip_install_targets_a_directory_and_can_skip_dependencies(monkeypatch):
    """--target keeps the pinned code out of a second interpreter; torch is shared."""
    seen: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = "done"
        stderr = ""

    monkeypatch.setattr(
        model_env.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or _Result()
    )
    ok, output = model_env._pip_install(Path("/tmp/x"), ["einops"], deps=True)
    assert ok and "done" in output
    assert "--target" in seen[0] and "/tmp/x" in seen[0]
    assert "--no-deps" not in seen[0]

    model_env._pip_install(Path("/tmp/x"), ["einops"], deps=False)
    assert "--no-deps" in seen[1]


def test_child_env_puts_the_pinned_directory_first_and_flags_the_child(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/pre-existing")
    env = model_env._child_env(Path("/models/pinned"))
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == "/models/pinned"
    assert parts[-1] == "/pre-existing"
    assert env[model_env.REEXEC_ENV_FLAG] == "1"


def test_child_env_without_an_existing_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = model_env._child_env(Path("/models/pinned"))
    assert env["PYTHONPATH"].split(os.pathsep)[0] == "/models/pinned"


def test_probe_command_asks_only_for_the_declared_class():
    command = model_env._probe_command("some/checkpoint")
    assert command[0] == sys.executable
    assert "get_class_from_dynamic_module" in command[2]
    assert command[-1] == "some/checkpoint"


# --------------------------------------------------------------------------- #
# Dependency provisioning: the "install what the failing import names" loop
# --------------------------------------------------------------------------- #
class _FakeProbe:
    """Stands in for running the model's own import, failing until satisfied."""

    def __init__(self, failures: list[str]):
        self.failures = list(failures)
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1

        class _R:
            pass

        result = _R()
        if self.failures:
            result.returncode = 1
            result.stdout = ""
            result.stderr = self.failures.pop(0)
        else:
            result.returncode = 0
            result.stdout = result.stderr = ""
        return result


def _stub_install(monkeypatch, installed: list[str], ok: bool = True):
    def _install(_target, packages, *, deps):
        installed.extend(packages)
        return ok, "" if ok else "boom"

    monkeypatch.setattr(model_env, "_pip_install", _install)


def _stamped(tmp_path, version="4.56.2"):
    """A target directory that already holds the pinned transformers."""
    target = model_env.env_path_for("some/checkpoint", version, tmp_path)
    target.mkdir(parents=True, exist_ok=True)
    (target / ".tracelens-transformers").write_text(version, encoding="utf-8")
    return target


def test_dependencies_installs_pinned_transformers_once(monkeypatch, tmp_path):
    installed: list[str] = []
    _stub_install(monkeypatch, installed)
    monkeypatch.setattr(model_env.subprocess, "run", _FakeProbe([]))

    first = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert first is not None and installed == ["transformers==4.56.2"]

    # The stamp records the version, so a second call reinstalls nothing.
    second = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert second == first and installed == ["transformers==4.56.2"]


def test_dependencies_gives_up_when_transformers_cannot_be_installed(
    monkeypatch, tmp_path
):
    _stub_install(monkeypatch, [], ok=False)
    assert (
        model_env.ensure_model_dependencies("some/checkpoint", "4.56.2", root=tmp_path)
        is None
    )


def test_dependencies_skips_the_probe_when_asked(monkeypatch, tmp_path):
    _stub_install(monkeypatch, [])

    def _unexpected(*_a, **_k):
        raise AssertionError("probe must not run when probe=False")

    monkeypatch.setattr(model_env.subprocess, "run", _unexpected)
    target = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path, probe=False
    )
    assert target is not None


def test_dependencies_follows_the_chain_past_a_masking_wrapper(monkeypatch, tmp_path):
    """fla-core's friendly message names itself; the real gap is ``triton``.

    Installing only the package the model names would loop forever, so the loop
    must move on to the module the chained traceback reports.
    """
    _stamped(tmp_path)
    installed: list[str] = []
    _stub_install(monkeypatch, installed)
    masked = (
        "ModuleNotFoundError: No module named 'triton'\n"
        "During handling of the above exception, another exception occurred:\n"
        "ImportError: Plese run `pip install -U fla-core`"
    )
    probe = _FakeProbe([masked, masked])
    monkeypatch.setattr(model_env.subprocess, "run", probe)

    target = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert target is not None
    assert installed == ["fla-core", "triton"]


def test_dependencies_stops_when_no_new_package_is_named(monkeypatch, tmp_path):
    """An unreadable failure is not a reason to keep installing the same thing."""
    _stamped(tmp_path)
    installed: list[str] = []
    _stub_install(monkeypatch, installed)
    monkeypatch.setattr(
        model_env.subprocess, "run", _FakeProbe(["something we cannot parse"] * 3)
    )
    target = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert target is not None and installed == []


def test_dependencies_hands_back_the_directory_when_an_install_fails(
    monkeypatch, tmp_path
):
    """The pinned sources are still there and AST analysis reads them directly."""
    _stamped(tmp_path)
    monkeypatch.setattr(model_env, "_pip_install", lambda *a, **k: (False, "no wheel"))
    monkeypatch.setattr(
        model_env.subprocess,
        "run",
        _FakeProbe(["ModuleNotFoundError: No module named 'einops'"] * 3),
    )
    target = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert target is not None


def test_dependencies_bounds_the_discovery_loop(monkeypatch, tmp_path):
    """A model that names a new package every round must still terminate."""
    _stamped(tmp_path)
    installed: list[str] = []
    _stub_install(monkeypatch, installed)
    endless = [
        f"ModuleNotFoundError: No module named 'pkg{i}'"
        for i in range(model_env._MAX_DEPENDENCY_ROUNDS + 5)
    ]
    monkeypatch.setattr(model_env.subprocess, "run", _FakeProbe(endless))
    target = model_env.ensure_model_dependencies(
        "some/checkpoint", "4.56.2", root=tmp_path
    )
    assert target is not None
    assert len(installed) == model_env._MAX_DEPENDENCY_ROUNDS


# --------------------------------------------------------------------------- #
# Version discovery and re-execution
# --------------------------------------------------------------------------- #
def test_required_version_ignores_a_nonsense_value(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"transformers_version": "not a version"}), encoding="utf-8"
    )
    assert model_env.required_transformers_version(tmp_path) is None


def test_required_version_survives_unreadable_config(tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    assert model_env.required_transformers_version(tmp_path) is None


def test_required_version_is_none_without_a_config(tmp_path):
    assert model_env.required_transformers_version(tmp_path / "absent") is None


def test_no_mismatch_when_the_config_names_no_remote_class(monkeypatch):
    """Architectures with no matching auto_map entry are not an env problem."""
    _patch_autoconfig(monkeypatch, _Config(architectures=["NotInstalledForCausalLM"]))
    assert model_env.detect_env_mismatch("some/checkpoint") is None


def test_config_that_cannot_be_imported_is_a_mismatch(monkeypatch):
    import transformers

    def _boom(cls, *_a, **_k):
        raise ImportError("no module named 'custom_cfg'")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_boom))
    reason = model_env.detect_env_mismatch("some/checkpoint")
    assert reason is not None and "custom_cfg" in reason


def test_reexec_runs_the_child_and_returns_its_status(monkeypatch, tmp_path):
    monkeypatch.delenv(model_env.REEXEC_ENV_FLAG, raising=False)
    monkeypatch.setattr(
        model_env, "detect_env_mismatch", lambda *_a, **_k: "cannot import"
    )
    monkeypatch.setattr(
        model_env, "required_transformers_version", lambda *_a: "4.56.2"
    )
    monkeypatch.setattr(
        model_env, "ensure_model_dependencies", lambda *_a, **_k: tmp_path / "env"
    )
    seen: dict = {}

    class _R:
        returncode = 7

    def _run(cmd, env=None, **_k):
        seen["cmd"] = cmd
        seen["env"] = env
        return _R()

    monkeypatch.setattr(model_env.subprocess, "run", _run)
    assert model_env.reexec_in_model_env("some/checkpoint", ["--flag"]) == 7
    assert "--flag" in seen["cmd"]
    assert seen["env"][model_env.REEXEC_ENV_FLAG] == "1"
    assert seen["env"]["PYTHONPATH"].startswith(str(tmp_path / "env"))


def test_no_reexec_when_the_environment_cannot_be_provisioned(monkeypatch):
    monkeypatch.delenv(model_env.REEXEC_ENV_FLAG, raising=False)
    monkeypatch.setattr(
        model_env, "detect_env_mismatch", lambda *_a, **_k: "cannot import"
    )
    monkeypatch.setattr(
        model_env, "required_transformers_version", lambda *_a: "4.56.2"
    )
    monkeypatch.setattr(model_env, "ensure_model_dependencies", lambda *_a, **_k: None)
    assert model_env.reexec_in_model_env("some/checkpoint", []) is None


def test_no_mismatch_when_transformers_is_absent(monkeypatch):
    """Without transformers there is nothing to compare against, so no reason."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("transformers is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert model_env.detect_env_mismatch("some/checkpoint") is None


def test_config_failure_that_is_not_an_import_error_is_not_a_mismatch(monkeypatch):
    """A download or parse failure is a real error to surface, not a reason."""
    import transformers

    def _boom(cls, *_a, **_k):
        raise OSError("could not reach the hub")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_boom))
    assert model_env.detect_env_mismatch("some/checkpoint") is None
