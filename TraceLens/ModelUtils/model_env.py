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


def _pip_install(target: Path, packages: list[str], *, deps: bool) -> tuple[bool, str]:
    """Install into a directory that will be put on ``PYTHONPATH``.

    ``--target`` keeps each model's pinned code in its own directory instead of a
    whole second interpreter, so torch -- by far the largest dependency -- is
    shared from the environment already running rather than duplicated per model.
    """
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--target",
        str(target),
        "--upgrade",
    ]
    if not deps:
        command.append("--no-deps")
    result = subprocess.run([*command, *packages], capture_output=True, text=True)
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def _missing_requirements(output: str) -> list[str]:
    """Packages a failed model import points at, best candidate first.

    Model code conventionally catches its own ImportError and re-raises a friendly
    ``pip install -U <pkg>`` message, which is the right thing to try first. But
    that message NAMES THE WRAPPER, not what was actually missing: fla-core raises
    it when ``triton`` is absent, so following it alone reinstalls fla-core
    forever. Python prints the chained original, so every ``No module named 'x'``
    in the traceback is also a candidate -- the caller walks this list and skips
    anything it already installed, which lets the real cause surface on the next
    round.
    """
    candidates: list[str] = []
    explicit = re.search(
        r"pip install (?:-U\s+|--upgrade\s+)?([A-Za-z0-9._-]+)", output
    )
    if explicit is not None:
        candidates.append(explicit.group(1))
    for match in re.finditer(r"No module named '([A-Za-z0-9._]+)'", output):
        name = match.group(1).split(".", 1)[0]
        if name not in candidates:
            candidates.append(name)
    return candidates


def _probe_command(checkpoint: str | Path) -> list[str]:
    """Ask the model to import its own declared class, and nothing more."""
    return [
        sys.executable,
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
    ]


def _is_installed(target: Path, requirement: str) -> bool:
    """Whether ``name==version`` is already present in *target*.

    Read off the installed ``.dist-info`` directory rather than by importing:
    the point is to avoid reinstalling what is already there, and some of these
    libraries cannot be imported in this interpreter at all.
    """
    name, _, version = requirement.partition("==")
    if not version:
        return False
    normalized = re.sub(r"[-_.]+", "_", name.strip()).lower()
    for entry in target.glob("*.dist-info"):
        dist, _, dist_version = entry.name[: -len(".dist-info")].rpartition("-")
        if re.sub(r"[-_.]+", "_", dist).lower() == normalized:
            return dist_version == version.strip()
    return False


def ensure_model_dependencies(
    checkpoint: str | Path,
    version: str,
    *,
    root: Path | None = None,
    probe: bool = True,
    extra_packages: list[str] | None = None,
) -> Path | None:
    """Directory holding the code ``checkpoint`` needs, for ``PYTHONPATH``.

    Installs the pinned ``transformers`` (with its own dependencies, so the
    tokenizers/hub versions that release expects travel with it), then -- when
    ``probe`` is set -- repeatedly asks the model to import itself and installs
    whatever package each failure names, in the order
    :func:`_missing_requirements` ranks them.

    That includes a package named only by a deeper chained failure -- a kernel
    compiler some library imports at module scope, say. It is tempting to skip
    those on the grounds that reading source never runs the kernels, and this
    function used to. That was wrong: a library whose ``__init__`` imports its
    kernels eagerly cannot be imported at all without them, and the model class
    cannot be built on the meta device, so every meta-derived shape silently
    disappears from the diagram while the run still looks successful. The
    package can be large (hundreds of megabytes); a wrong diagram is worse.

    Returns the directory to prepend to ``PYTHONPATH``, or *None* if it could not
    be made to import the model.
    """
    target = env_path_for(checkpoint, version, root)
    stamp = target / ".tracelens-transformers"
    if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != version:
        target.mkdir(parents=True, exist_ok=True)
        ok, output = _pip_install(target, [f"transformers=={version}"], deps=True)
        if not ok:
            _log.warning(
                "Installing transformers==%s failed: %s", version, output[-800:]
            )
            return None
        stamp.write_text(version, encoding="utf-8")

    # Pinned libraries go in BEFORE the probe, so the probe has nothing left to
    # discover and cannot fetch an unpinned latest in their place. Only the ones
    # not already present at the requested version are installed, so an env that
    # already satisfies the pins needs no network at all. An export names none of
    # these and lets the probe choose -- that is how it picks up a model's
    # current requirements.
    for package in extra_packages or ():
        if _is_installed(target, package):
            continue
        _log.info("Installing pinned %s into %s", package, target)
        ok, output = _pip_install(target, [package], deps=True)
        if not ok:
            _log.warning("Installing %s failed: %s", package, output[-800:])
            return None

    if not probe:
        return target

    attempted: set[str] = set()
    for _ in range(_MAX_DEPENDENCY_ROUNDS):
        result = subprocess.run(
            _probe_command(checkpoint),
            capture_output=True,
            text=True,
            env=_child_env(target),
        )
        if result.returncode == 0:
            return target
        candidates = _missing_requirements(
            (result.stdout or "") + (result.stderr or "")
        )
        package = next((name for name in candidates if name not in attempted), None)
        if package is None:
            # Nothing new to fetch. The pinned sources are still installed and
            # AST analysis reads those directly, so hand the directory back
            # rather than throwing away work -- but say plainly that the
            # meta-device pass is the part that will be missing.
            _log.info(
                "%s is not importable here (%s); its sources are installed and "
                "will still be read, but meta-device shapes are unavailable",
                checkpoint,
                candidates or "no further packages named",
            )
            return target
        attempted.add(package)
        _log.info("Installing %s into %s", package, target)
        # With dependencies: pip resolves against the environment already running,
        # so a shared heavyweight like torch is not copied in again.
        ok, output = _pip_install(target, [package], deps=True)
        if not ok:
            _log.warning("Installing %s failed: %s", package, output[-800:])
            return target
    return target


def _child_env(target: Path) -> dict[str, str]:
    """Environment putting *target* ahead of the interpreter's own packages.

    ``PYTHONPATH`` entries are inserted before ``site-packages``, so the pinned
    copy wins over whatever the running environment has installed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(target), str(repo_root), *([existing] if existing else [])]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(parts),
        REEXEC_ENV_FLAG: "1",
    }


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
        "%s cannot be imported here (%s); using pinned transformers==%s",
        checkpoint,
        reason,
        version,
    )
    target = ensure_model_dependencies(checkpoint, version, root=root)
    if target is None:
        _log.warning(
            "Could not provision a transformers==%s environment for %s; "
            "continuing in the current environment",
            version,
            checkpoint,
        )
        return None

    # Same interpreter, with the pinned copy ahead of its own packages. A fresh
    # process is still required: ``transformers`` is already imported here and a
    # module cannot be swapped underneath a running one.
    return subprocess.run(
        [sys.executable, "-m", module, *argv], env=_child_env(target)
    ).returncode
