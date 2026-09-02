###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve Hugging Face / local / GitHub modeling source files (CPU-only, no weights)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from visualizer.github import (
    fetch_github_source,
    find_modeling_files,
    is_github_url,
    parse_github_url,
    python_source_priority,
)
from visualizer.source_policy import SourcePolicy, get_source_policy

MODELING_CANDIDATES = (
    "modeling_{model_type}.py",
    "modeling.py",
)

# Checkpoints for transformers-native architectures (Qwen3, MiniMax-M3, ...) ship
# no modeling code of their own, so the implementation is read from upstream.
TRANSFORMERS_GITHUB_SOURCE = "github:huggingface/transformers@main"
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


def _hub_snapshot_root(model_id: str) -> Path | None:
    """Local Hugging Face snapshot directory for a model id, if one is cached."""
    slug = "models--" + model_id.replace("/", "--")
    base = _hub_cache_root() / slug
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        return None
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


def _list_repo_python_files(model_id: str) -> list[str]:
    """Every Python path in a Hugging Face repo, recursively."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return []
    try:
        return [
            name
            for name in list_repo_files(model_id)
            if name.endswith(".py") and "__pycache__" not in name.split("/")
        ]
    except Exception:
        return []


def _download_repo_files(model_id: str, filenames: list[str]) -> list[Path]:
    if not filenames:
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []

    paths: list[Path] = []
    for name in filenames:
        try:
            downloaded = hf_hub_download(model_id, name)
            paths.append(Path(downloaded))
        except Exception:
            continue
    return paths


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
            return path, ref.display
    return None


def _has_modeling_implementation(files: list[Path]) -> bool:
    """True when a resolved file can hold module definitions, not just config/processing."""
    for path in files:
        name = path.name.lower()
        if name.startswith("modeling") or name in {"model.py", "models.py"}:
            return True
    return False


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
) -> tuple[list[Path], list[str]]:
    """Return modeling Python files to analyze and human-readable source labels."""
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
        snapshot = _hub_snapshot_root(model_id)
        if snapshot is not None:
            snapshot_files = _local_modeling_files(snapshot)
            files.extend(snapshot_files)
            if snapshot_files:
                labels.append(f"hf://{model_id}")

    model_type = str(config.get("model_type") or "")
    if model_type and not files:
        tf_path = _transformers_modeling_path(model_type)
        if tf_path is not None:
            files.append(tf_path)
            labels.append(str(tf_path))

    auto_map_files = _collect_auto_map_files(config)

    if model_id and not files:
        hf_files = _download_repo_files(model_id, auto_map_files)
        files.extend(hf_files)
        repo_python = _list_repo_python_files(model_id)
        downloaded = _download_repo_files(model_id, repo_python)
        files.extend(downloaded)
        if files and f"hf://{model_id}" not in labels:
            labels.append(f"hf://{model_id}")
        elif not files and not auto_map_files and model_type:
            fallback = _download_repo_files(
                model_id,
                [name.format(model_type=model_type) for name in MODELING_CANDIDATES],
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
