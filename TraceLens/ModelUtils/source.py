###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve Hugging Face / local / GitHub modeling source files (CPU-only, no weights)."""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path
from typing import Any

from TraceLens.ModelUtils.github import (
    GitHubRef,
    fetch_github_file,
    register_pinned_module_source,
    fetch_github_source,
    find_modeling_files,
    is_github_url,
    parse_github_url,
    python_source_priority,
)
from TraceLens.ModelUtils.source_policy import SourcePolicy, get_source_policy

MODELING_CANDIDATES = (
    "modeling_{model_type}.py",
    "modeling.py",
)

# Checkpoints for transformers-native architectures (Qwen3, MiniMax-M3, ...) ship
# no modeling code of their own, so the implementation is read from upstream.
# Pinned to a commit (not @main) so static analysis is reproducible: upstream line
# shifts would otherwise drift the hard-coded op-ids in the graph tests. Bump this
# SHA (and refresh any affected op-id assertions) to pick up newer upstream code.
TRANSFORMERS_GITHUB_SOURCE = (
    "github:huggingface/transformers@0b179b2df599b7edd6f91de59f71e51faee4cc5a"
)
TRANSFORMERS_MODELING_SUBPATH = (
    "src/transformers/models/{model_type}/modeling_{model_type}.py"
)
NESTED_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder_config")


def _module_file(module_ref: str) -> str:
    """Turn 'modeling_foo.Bar' into 'modeling_foo.py'."""
    return module_ref.split(".", 1)[0] + ".py"


def _collect_auto_map_files(config: dict[str, Any]) -> list[str]:
    files: list[str] = []
    auto_map = config.get("auto_map") or {}
    if not isinstance(auto_map, dict):
        return files

    for target in auto_map.values():
        if not isinstance(target, str):
            continue
        files.append(_module_file(target))

    return sorted(set(files))


def _local_modeling_files(root: Path) -> list[Path]:
    return find_modeling_files(root)


def _hub_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _hub_snapshot_root(model_id: str, revision: str | None = None) -> Path | None:
    """Local Hugging Face snapshot directory for a model id, if one is cached.

    With a ``revision``, that revision ALONE answers: falling back to whatever
    ``refs/main`` names would quietly ignore the caller's request and hand back
    different source than it asked for. ``None`` for an uncached revision lets
    the download path fetch that exact one.

    Without a revision -- the default, and what an export does -- the checkpoint
    resolves to its current head.
    """
    slug = "models--" + model_id.replace("/", "--")
    base = _hub_cache_root() / slug
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        return None
    if revision is not None:
        candidate = snapshots / revision
        return candidate if candidate.is_dir() else None
    for ref_name in ("main", "master"):
        ref_file = base / "refs" / ref_name
        if not ref_file.is_file():
            continue
        revision = ref_file.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.is_dir():
            return candidate
    newest = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return newest[0] if newest else None


def _list_repo_python_files(model_id: str, revision: str | None = None) -> list[str]:
    """Every Python path in a Hugging Face repo, recursively."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return []
    try:
        return [
            name
            for name in list_repo_files(model_id, revision=revision)
            if name.endswith(".py") and "__pycache__" not in name.split("/")
        ]
    except Exception:
        return []


def _download_repo_files(
    model_id: str, filenames: list[str], revision: str | None = None
) -> list[Path]:
    if not filenames:
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []

    paths: list[Path] = []
    for name in filenames:
        try:
            downloaded = hf_hub_download(model_id, name, revision=revision)
            paths.append(Path(downloaded))
        except Exception:
            continue
    return paths


def _own_package_imports(text: str) -> list[str]:
    """Module names a file imports from its OWN directory.

    ``modeling_glm5_next.py`` says ``from .configuration_glm5_next import
    Glm5NextConfig``. That neighbour is part of the model -- it states the head
    counts and the names the modeling code reads them under -- but fetching one
    file from GitHub leaves it behind, so the model's own configuration source
    is read from whatever transformers version happens to be installed, at a
    different revision from the pinned modeling file.

    Only level-1 imports: the model's directory is the model's code, while
    ``from ...cache_utils import Cache`` reaches into the library around it and
    following those would fetch transformers a file at a time.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        # `from . import x` names modules in the names list, not a module.
        named = [node.module] if node.module else [a.name for a in node.names]
        for module in named:
            if module and module not in modules:
                modules.append(str(module))
    return modules


def _fetch_own_package(ref: GitHubRef, subpath: str, path: Path) -> None:
    """Fetch the model directory's other modules at the SAME commit as the file.

    Best effort: a name may be a package rather than a module, or absent at this
    ref. Fetched files land beside the modeling file under their real names, so
    an import that reads them resolves against the pinned source.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    package = Path(subpath).parent
    for module in _own_package_imports(text):
        relative = Path(*module.split("."))
        if fetch_github_file(ref, (package / relative.with_suffix(".py")).as_posix()):
            continue
        fetch_github_file(ref, (package / relative / "__init__.py").as_posix())


def _modeling_module_name(model_type: str) -> str:
    """Dotted name of a transformers-native modeling module.

    Must agree with what the analyser resolves imports against
    (``ast_analyze._analyzed_base_module``): the pair of this name and the repo
    path is what says where the repo keeps its top-level packages.
    """
    return "transformers.models.{0}.modeling_{0}".format(model_type.replace("-", "_"))


def _transformers_modeling_path(model_type: str) -> Path | None:
    """Locate installed transformers modeling file for a model_type."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None

    model_type = model_type.replace("-", "_")
    module_name = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None

    if spec is None or not spec.origin:
        return None

    origin = Path(spec.origin)
    return origin if origin.is_file() else None


def _installed_transformers_version() -> str | None:
    try:
        import transformers
    except ImportError:
        return None
    return getattr(transformers, "__version__", None)


@functools.lru_cache(maxsize=None)
def _fetch_versioned_transformers_file(
    model_types: tuple[str, ...], version: str
) -> tuple[Path, str] | None:
    """Fetch the upstream modeling file at a transformers *release tag*.

    Cached (including the negative result) so a checkpoint whose declared tag is
    missing does not re-hit the network on every ``load_model_spec`` call. Tries
    both ``vX.Y.Z`` and the bare ``X.Y.Z`` tag spelling.
    """
    policy = get_source_policy()
    for tag in (f"v{version}", version):
        for model_type in model_types:
            subpath = TRANSFORMERS_MODELING_SUBPATH.format(model_type=model_type)
            try:
                ref = parse_github_url(
                    f"github:huggingface/transformers@{tag}:{subpath}"
                )
                path = fetch_github_source(ref, source_policy=policy)
            except Exception:
                continue
            if path.is_file():
                _fetch_own_package(ref, subpath, path)
                register_pinned_module_source(
                    ref, subpath, _modeling_module_name(model_type)
                )
                return path, ref.display
    return None


def _transformers_versioned_modeling_file(
    config: dict[str, Any],
    model_types: list[str],
) -> tuple[Path, str] | None:
    """Prefer the modeling file at the version the checkpoint was exported with.

    A checkpoint records that version in ``config.transformers_version``. When it
    differs from the installed transformers, the installed modeling file can be
    the wrong revision for the checkpoint, so read the implementation from the
    upstream release tag instead. Returns ``None`` (fall back to the installed
    file) when no version is declared, it matches the installed one, or the tag
    is not published upstream.
    """
    declared = str(config.get("transformers_version") or "").strip()
    if not declared:
        return None
    installed = _installed_transformers_version()
    if installed is not None and declared == installed:
        return None
    return _fetch_versioned_transformers_file(tuple(model_types), declared)


def _config_model_types(config: dict[str, Any]) -> list[str]:
    """Model types to look up upstream, outer wrapper first then its text backbone."""
    types: list[str] = []
    wrapper = config.get("_wrapper_model_type")
    if wrapper:
        wrapper_type = str(wrapper).strip().replace("-", "_")
        if wrapper_type:
            types.append(wrapper_type)
    candidates = [config.get("model_type")]
    for key in NESTED_CONFIG_KEYS:
        nested = config.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("model_type"))
    for candidate in candidates:
        model_type = str(candidate or "").strip().replace("-", "_")
        if model_type and model_type not in types:
            types.append(model_type)
    return types


def _transformers_github_modeling_file(
    model_types: list[str],
    *,
    source_policy: SourcePolicy | None = None,
) -> tuple[Path, str] | None:
    """Fetch a transformers-native modeling file from GitHub, newest ref first."""
    policy = source_policy or get_source_policy()
    for model_type in model_types:
        subpath = TRANSFORMERS_MODELING_SUBPATH.format(model_type=model_type)
        try:
            ref = parse_github_url(f"{TRANSFORMERS_GITHUB_SOURCE}:{subpath}")
            path = fetch_github_source(ref, source_policy=policy)
        except Exception:
            continue
        if path.is_file():
            _fetch_own_package(ref, subpath, path)
            register_pinned_module_source(
                ref, subpath, _modeling_module_name(model_type)
            )
            return path, ref.display
    return None


def _has_modeling_implementation(files: list[Path]) -> bool:
    """True when a resolved file can hold module definitions, not just config/processing."""
    for path in files:
        name = path.name.lower()
        if name.startswith("modeling") or name in {"model.py", "models.py"}:
            return True
    return False


def _declared_architectures(config: dict[str, Any]) -> set[str]:
    """Class names the config says this checkpoint runs, including nested towers."""
    names: set[str] = set()
    pending = [config]
    while pending:
        current = pending.pop()
        if not isinstance(current, dict):
            continue
        declared = current.get("architectures")
        if isinstance(declared, list):
            names.update(str(name) for name in declared if name)
        for key in NESTED_CONFIG_KEYS + ("vision_config",):
            nested = current.get(key)
            if isinstance(nested, dict):
                pending.append(nested)
    return names


def _defines_any_class(path: Path, names: set[str]) -> bool:
    """True when a Python file defines one of ``names`` at any nesting level."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.ClassDef) and node.name in names for node in ast.walk(tree)
    )


def _checkpoint_modeling_files(files: list[Path], config: dict[str, Any]) -> list[Path]:
    """Keep a checkpoint's Python files only when they implement the architecture.

    A repo ships more than its modeling code. DeepSeek-V4-Flash carries a
    standalone reference implementation under ``inference/model.py`` and a
    chat-template encoder under ``encoding/`` -- neither defines the
    ``DeepseekV4ForCausalLM`` the config names, because the real implementation
    lives in installed transformers. Taking them anyway both skips the
    transformers lookup (``files`` is no longer empty) and passes
    ``_has_modeling_implementation`` on the strength of the filename alone, so
    the export silently analyses a tool-call encoder as if it were the model.

    ``auto_map`` is the checkpoint stating which files implement it, so a repo
    that declares one is trusted as-is.
    """
    if not files or config.get("auto_map"):
        return files
    names = _declared_architectures(config)
    if not names:
        return files
    if any(_defines_any_class(path, names) for path in files):
        return files
    return []


def _dedupe_paths(files: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in files:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def resolve_github_files(
    github: str,
    *,
    source_policy: SourcePolicy | None = None,
) -> tuple[list[Path], str]:
    """Fetch a GitHub repo or file and return modeling paths plus a source label."""
    ref = parse_github_url(github)
    policy = source_policy or get_source_policy()
    root = fetch_github_source(ref, source_policy=policy)
    label = ref.display

    if root.is_file():
        return [root], label

    files = _local_modeling_files(root)
    if not files:
        raise FileNotFoundError(
            f"No Python source files found in GitHub source {label}. "
            "Pass a deeper --github-path or use --code-path."
        )
    return files, label


def resolve_source_files(
    source: str | Path | None,
    config: dict[str, Any],
    *,
    code_path: str | Path | None = None,
    github: str | None = None,
    source_policy: SourcePolicy | None = None,
    revision: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Return modeling Python files to analyze and human-readable source labels.

    ``revision`` reads a hub checkpoint at one fixed commit instead of its head.
    Left unset -- what an export does -- the checkpoint resolves to whatever it
    currently holds, so a model's newest code is what gets drawn.
    """
    policy = source_policy or get_source_policy()
    labels: list[str] = []

    if code_path is not None:
        path = Path(code_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Code path not found: {path}")
        return [path], [str(path)]

    if github:
        files, label = resolve_github_files(github, source_policy=policy)
        return files, [label]

    path = Path(source).expanduser() if source is not None else None
    files: list[Path] = []

    if path is not None and path.is_file() and path.suffix == ".py":
        return [path.resolve()], [str(path.resolve())]

    if path is not None and path.is_dir():
        files.extend(_local_modeling_files(path))
        labels.append(str(path.resolve()))

    if is_github_url(str(source)):
        gh_files, gh_label = resolve_github_files(str(source), source_policy=policy)
        return gh_files, [gh_label]

    model_id = None
    if path is not None and not path.exists():
        model_id = str(source)
    elif path is not None and path.is_dir() and (path / "config.json").exists():
        pass
    elif source is not None and not Path(source).exists():
        model_id = str(source)

    if model_id and not files:
        snapshot = _hub_snapshot_root(model_id, revision)
        if snapshot is not None:
            snapshot_files = _checkpoint_modeling_files(
                _local_modeling_files(snapshot), config
            )
            files.extend(snapshot_files)
            if snapshot_files:
                labels.append(f"hf://{model_id}")

    model_type = str(config.get("model_type") or "")
    if model_type and not files:
        # Prefer the modeling file at the transformers version the checkpoint
        # declares; only when that tag is unavailable (or matches installed) do
        # we read the version installed alongside TraceLens.
        versioned = _transformers_versioned_modeling_file(
            config, _config_model_types(config)
        )
        if versioned is not None:
            path, label = versioned
            files.append(path)
            labels.append(label)
        else:
            tf_path = _transformers_modeling_path(model_type)
            if tf_path is not None:
                files.append(tf_path)
                labels.append(str(tf_path))

    auto_map_files = _collect_auto_map_files(config)

    if model_id and not files:
        hf_files = _download_repo_files(model_id, auto_map_files, revision)
        files.extend(hf_files)
        repo_python = _list_repo_python_files(model_id, revision)
        downloaded = _download_repo_files(model_id, repo_python, revision)
        files.extend(downloaded)
        if files and f"hf://{model_id}" not in labels:
            labels.append(f"hf://{model_id}")
        elif not files and not auto_map_files and model_type:
            fallback = _download_repo_files(
                model_id,
                [name.format(model_type=model_type) for name in MODELING_CANDIDATES],
                revision,
            )
            files.extend(fallback)
            if fallback and f"hf://{model_id}" not in labels:
                labels.append(f"hf://{model_id}")

    if not _has_modeling_implementation(files):
        upstream = _transformers_github_modeling_file(
            _config_model_types(config), source_policy=policy
        )
        if upstream is not None:
            path, label = upstream
            files.append(path)
            labels.append(label)

    # Analysis reads the files in order, so modeling code has to precede the
    # config and processing helpers a checkpoint may also ship.
    return sorted(_dedupe_paths(files), key=python_source_priority), labels


def read_sources(paths: list[Path]) -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in paths if path.is_file()}
