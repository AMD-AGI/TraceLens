###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-checkpoint Python environments for models that need a different
``transformers`` than the interpreter currently running.

A checkpoint whose architecture ships inside ``transformers`` runs fine here. A
checkpoint that ships its OWN modeling code (``auto_map`` / ``trust_remote_code``)
was written against one specific ``transformers`` release, and that code breaks
on a newer one as soon as it imports a symbol the newer release no longer
exports. The break is an ``ImportError`` raised while importing the model's own
module -- which makes it something we can DETECT AT RUN TIME rather than track in
a per-model table.

Nothing here keys off a model name or a version table. The flow is:

1. :func:`detect_env_mismatch` probes THIS interpreter by resolving the
   checkpoint's declared class exactly the way meta instantiation does, and
   reports a reason only when that import genuinely fails.
2. :func:`required_transformers_version` then reads the pin hint out of the
   checkpoint's own ``config.json``. That field records the version that WROTE
   the config, so it is not authoritative on its own -- it is consulted only for
   a checkpoint already proven unable to run here.
3. :func:`ensure_model_venv` creates (or reuses) a venv for that pin and
   installs what the model needs, discovering extra third-party requirements
   from the import errors the model itself raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

#: Set in the child environment so a re-executed run never re-enters provisioning.
REEXEC_ENV_FLAG = "TRACELENS_MODEL_ENV_ACTIVE"

#: Where per-model environments are cached. Override for tests or to relocate.
ENV_ROOT_VAR = "TRACELENS_MODEL_ENV_ROOT"

#: Packages TraceLens itself needs before it can even be imported in the child.
_TRACELENS_RUNTIME_REQUIREMENTS = (
    "pandas",
    "tqdm",
    "orjson",
    "PyYAML",
    "tabulate",
    "matplotlib",
    "openpyxl",
    "huggingface_hub>=0.20",
)

#: Bound on the "install what the failing import names" discovery loop.
_MAX_DEPENDENCY_ROUNDS = 6


def default_env_root() -> Path:
    """Directory holding per-model environments."""
    override = os.environ.get(ENV_ROOT_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "tracelens" / "model-envs"


def detect_env_mismatch(checkpoint: str | Path) -> str | None:
    """Reason this interpreter cannot import ``checkpoint``'s modeling code.

    Returns ``None`` when the checkpoint is fine here -- which is the answer for
    every natively supported architecture, and for any failure that is NOT an
    import problem (a missing config key, a download error and so on are real
    errors to surface, not reasons to build a second environment).
    """
    try:
        import transformers
        from transformers import AutoConfig
    except Exception:  # noqa: BLE001 - transformers missing entirely
        return None

    try:
        config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
    except ImportError as exc:
        return f"config code could not be imported: {exc}"
    except Exception:  # noqa: BLE001
        return None

    architectures = list(getattr(config, "architectures", None) or [])
    # Natively supported: the installed transformers already defines the class,
    # so no amount of re-provisioning would change anything.
    if any(getattr(transformers, arch, None) is not None for arch in architectures):
        return None

    auto_map = getattr(config, "auto_map", None) or {}
    ref = next(
        (
            value
            for value in auto_map.values()
            if str(value).rsplit(".", 1)[-1] in architectures
        ),
        None,
    )
    if ref is None:
        return None

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        get_class_from_dynamic_module(str(ref), str(checkpoint))
    except ImportError as exc:
        return f"{ref} could not be imported here: {exc}"
    except Exception:  # noqa: BLE001
        return None
    return None


def required_transformers_version(checkpoint: str | Path) -> str | None:
    """Pin hint from the checkpoint's own ``config.json``.

    ``transformers_version`` records the release that WROTE the config, not a
    requirement -- natively supported checkpoints routinely name an older release
    and run fine. Only call this for a checkpoint :func:`detect_env_mismatch` has
    already shown cannot run in the current interpreter.
    """
    config: dict | None = None
    try:
        from TraceLens.ModelUtils.config_resolve import load_checkpoint_config

        config, _source = load_checkpoint_config(checkpoint)
    except Exception:  # noqa: BLE001
        config = None
    if not isinstance(config, dict):
        local = Path(str(checkpoint)) / "config.json"
        if not local.is_file():
            return None
        try:
            config = json.loads(local.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    version = config.get("transformers_version")
    if isinstance(version, str) and re.fullmatch(r"[0-9][0-9A-Za-z.\-+]*", version):
        return version
    return None


def env_path_for(
    checkpoint: str | Path, version: str, root: Path | None = None
) -> Path:
    """Cache location for one (checkpoint, transformers version) environment."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(checkpoint)).strip("_")
    return (root or default_env_root()) / f"{slug}-transformers-{version}"


def _venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"


def _pip_install(python: Path, packages: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", *packages],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def _missing_requirement(output: str) -> str | None:
    """Third-party package a failed model import is asking for.

    Model code commonly wraps its own optional imports and re-raises with the
    install command spelled out, so prefer that explicit instruction; otherwise
    fall back to the module name Python reported. Returns ``None`` when the text
    names nothing installable, which ends the discovery loop.
    """
    explicit = re.search(
        r"pip install (?:-U\s+|--upgrade\s+)?([A-Za-z0-9._-]+)", output
    )
    if explicit is not None:
        return explicit.group(1)
    missing = re.search(r"No module named '([A-Za-z0-9._]+)'", output)
    if missing is not None:
        return missing.group(1).split(".", 1)[0]
    return None


def ensure_model_venv(
    checkpoint: str | Path,
    version: str,
    *,
    root: Path | None = None,
    probe: bool = True,
) -> Path | None:
    """Create (or reuse) an environment able to import ``checkpoint``'s code.

    Installs the pinned ``transformers``, torch and TraceLens' own runtime
    requirements, then -- when ``probe`` is set -- repeatedly asks the model to
    import itself, installing whatever third-party package each failure names,
    until the import succeeds or nothing further is named. Returns the child
    interpreter, or ``None`` if the environment could not be made to work.
    """
    env_dir = env_path_for(checkpoint, version, root)
    python = _venv_python(env_dir)
    if not python.exists():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(env_dir)],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0 or not python.exists():
            _log.warning("Could not create %s: %s", env_dir, created.stderr.strip())
            return None
        ok, output = _pip_install(
            python,
            [f"transformers=={version}", "torch", *_TRACELENS_RUNTIME_REQUIREMENTS],
        )
        if not ok:
            _log.warning("Base install failed for %s: %s", env_dir, output[-800:])
            return None

    if not probe:
        return python

    for _ in range(_MAX_DEPENDENCY_ROUNDS):
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import sys;"
                    "from transformers import AutoConfig;"
                    "from transformers.dynamic_module_utils import "
                    "get_class_from_dynamic_module as g;"
                    "c=AutoConfig.from_pretrained(sys.argv[1],trust_remote_code=True);"
                    "a=list(getattr(c,'architectures',None) or []);"
                    "m=getattr(c,'auto_map',None) or {};"
                    "r=next((v for v in m.values() "
                    "if str(v).rsplit('.',1)[-1] in a),None);"
                    "g(str(r),sys.argv[1]) if r else None"
                ),
                str(checkpoint),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, REEXEC_ENV_FLAG: "1"},
        )
        if result.returncode == 0:
            return python
        package = _missing_requirement((result.stdout or "") + (result.stderr or ""))
        if package is None:
            _log.warning(
                "%s still cannot import %s and names no installable package",
                env_dir,
                checkpoint,
            )
            return None
        _log.info("Installing %s into %s", package, env_dir)
        ok, output = _pip_install(python, [package])
        if not ok:
            _log.warning("Installing %s failed: %s", package, output[-800:])
            return None
    return None


def reexec_in_model_env(
    checkpoint: str | Path,
    argv: list[str],
    *,
    root: Path | None = None,
    module: str = "TraceLens.Visualizer.model_explorer_export.cli",
) -> int | None:
    """Re-run this command in an environment that can import ``checkpoint``.

    Returns the child's exit status, or ``None`` when nothing was re-executed --
    because this process already IS the child, because the checkpoint runs fine
    here (the common case), or because the environment could not be provisioned.
    A ``None`` return means the caller should simply carry on in-process, which
    degrades to the same static-analysis-only result as before.
    """
    if os.environ.get(REEXEC_ENV_FLAG):
        return None
    reason = detect_env_mismatch(checkpoint)
    if reason is None:
        return None
    version = required_transformers_version(checkpoint)
    if version is None:
        _log.warning(
            "%s needs different model code (%s) but its config names no "
            "transformers version to pin; continuing in the current environment",
            checkpoint,
            reason,
        )
        return None

    _log.warning(
        "%s cannot be imported here (%s); using a transformers==%s environment",
        checkpoint,
        reason,
        version,
    )
    python = ensure_model_venv(checkpoint, version, root=root)
    if python is None:
        _log.warning(
            "Could not provision a transformers==%s environment for %s; "
            "continuing in the current environment",
            version,
            checkpoint,
        )
        return None

    child_env = {**os.environ, REEXEC_ENV_FLAG: "1"}
    # Run the in-tree TraceLens from the child interpreter rather than installing
    # a second copy of it into every per-model environment.
    repo_root = Path(__file__).resolve().parents[2]
    child_env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), child_env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run([str(python), "-m", module, *argv], env=child_env).returncode
