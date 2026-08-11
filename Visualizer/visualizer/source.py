"""Resolve Hugging Face / local / GitHub modeling source files (CPU-only, no weights)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from visualizer.github import fetch_github_source, find_modeling_files, is_github_url, parse_github_url

MODELING_CANDIDATES = (
    "modeling_{model_type}.py",
    "modeling.py",
)


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


def _dedupe_paths(files: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in files:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def resolve_github_files(github: str) -> tuple[list[Path], str]:
    """Fetch a GitHub repo or file and return modeling paths plus a source label."""
    ref = parse_github_url(github)
    root = fetch_github_source(ref)
    label = ref.display

    if root.is_file():
        return [root], label

    files = _local_modeling_files(root)
    if not files:
        raise FileNotFoundError(
            f"No modeling*.py files found in GitHub source {label}. "
            "Pass a deeper --github-path or use --code-path."
        )
    return files, label


def resolve_source_files(
    source: str | Path | None,
    config: dict[str, Any],
    *,
    code_path: str | Path | None = None,
    github: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Return modeling Python files to analyze and human-readable source labels."""
    labels: list[str] = []

    if code_path is not None:
        path = Path(code_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Code path not found: {path}")
        return [path], [str(path)]

    if github:
        files, label = resolve_github_files(github)
        return files, [label]

    path = Path(source).expanduser() if source is not None else None
    files: list[Path] = []

    if path is not None and path.is_file() and path.suffix == ".py":
        return [path.resolve()], [str(path.resolve())]

    if path is not None and path.is_dir():
        files.extend(_local_modeling_files(path))
        labels.append(str(path.resolve()))

    if is_github_url(str(source)):
        gh_files, gh_label = resolve_github_files(str(source))
        return gh_files, [gh_label]

    model_type = str(config.get("model_type") or "")
    if model_type and not files:
        tf_path = _transformers_modeling_path(model_type)
        if tf_path is not None:
            files.append(tf_path)
            labels.append(str(tf_path))

    auto_map_files = _collect_auto_map_files(config)
    model_id = None
    if path is not None and not path.exists():
        model_id = str(source)
    elif path is not None and path.is_dir() and (path / "config.json").exists():
        pass
    elif source is not None and not Path(source).exists():
        model_id = str(source)

    if model_id and not files:
        hf_files = _download_repo_files(model_id, auto_map_files)
        files.extend(hf_files)
        if hf_files:
            labels.append(f"hf://{model_id}")
        if not auto_map_files and model_type:
            fallback = _download_repo_files(
                model_id,
                [name.format(model_type=model_type) for name in MODELING_CANDIDATES],
            )
            files.extend(fallback)
            if fallback and f"hf://{model_id}" not in labels:
                labels.append(f"hf://{model_id}")

    return _dedupe_paths(files), labels


def read_sources(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in paths
        if path.is_file()
    }
