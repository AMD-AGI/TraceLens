"""Discover and normalize Hugging Face configs in modular checkpoint repos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKIP_CONFIG_PARTS = {
    "tokenizer",
    "processor",
    "scheduler",
    "preprocessor",
    "video_preprocessor",
    "audio_preprocessor",
}

PREFERRED_CONFIG_PARTS = (
    "text_encoder",
    "language_model",
    "llm",
    "transformer",
    "model",
)


def _score_config_path(path: str) -> int:
    lowered = path.lower()
    score = 0

    if not lowered.endswith("config.json"):
        return -100
    if any(part in lowered for part in SKIP_CONFIG_PARTS):
        score -= 20
    if lowered.endswith("/source/config.json"):
        score -= 10
    if lowered.count("/") <= 1:
        score += 4

    for idx, part in enumerate(PREFERRED_CONFIG_PARTS):
        if f"/{part}/" in lowered or lowered.startswith(f"{part}/"):
            score += 20 - idx

    return score


def _score_config_content(config: dict[str, Any], path: str) -> int:
    score = _score_config_path(path)

    if config.get("model_type"):
        score += 8
    if config.get("architectures"):
        score += 8
    if config.get("text_config"):
        score += 12
    if config.get("num_hidden_layers") is not None:
        score += 6
    if config.get("num_layers") is not None:
        score += 4
    if config.get("hidden_size") is not None:
        score += 4

    architectures = config.get("architectures") or []
    if any("CausalLM" in str(item) or "ConditionalGeneration" in str(item) for item in architectures):
        score += 6

    class_name = str(config.get("_class_name") or "")
    if "Pipeline" in class_name or "VAE" in class_name or "Scheduler" in class_name:
        score -= 15

    return score


def normalize_config(config: dict[str, Any], *, source_label: str = "") -> dict[str, Any]:
    """Flatten nested multimodal / diffusers configs into parser-friendly fields."""
    normalized = dict(config)

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        normalized.update(text_config)
        normalized["_wrapper_model_type"] = config.get("model_type")
        normalized["_wrapper_architectures"] = config.get("architectures")
        if config.get("vision_config"):
            normalized["_has_vision_tower"] = True

    if normalized.get("num_hidden_layers") is None and normalized.get("num_layers") is not None:
        normalized["num_hidden_layers"] = normalized["num_layers"]

    if normalized.get("intermediate_size") is None:
        normalized["intermediate_size"] = normalized.get("ffn_hidden_size")

    if not normalized.get("model_type") and normalized.get("_class_name"):
        normalized["model_type"] = str(normalized["_class_name"]).lower()

    if not normalized.get("architectures") and normalized.get("_class_name"):
        normalized["architectures"] = [str(normalized["_class_name"])]

    if source_label:
        normalized["_config_source"] = source_label

    return normalized


def _paths_from_model_index(config: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key, value in config.items():
        if not key.startswith("_") and isinstance(value, list) and len(value) >= 3:
            subfolder = value[2].get("subfolder") if isinstance(value[2], dict) else None
            if subfolder:
                paths.append(f"{subfolder.rstrip('/')}/config.json")

    for entry in config.values():
        if isinstance(entry, str) and entry.endswith("model_index.json"):
            paths.append(entry.replace("model_index.json", "config.json"))
        if isinstance(entry, dict):
            for sub in entry.values():
                if isinstance(sub, str) and sub.endswith("model_index.json"):
                    paths.append(sub.replace("model_index.json", "config.json"))
    return paths


def _list_repo_config_paths(model_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return [path for path in list_repo_files(model_id) if path.endswith("config.json")]


def _download_config(model_id: str, config_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, config_path))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _local_config_candidates(root: Path) -> list[Path]:
    return sorted(root.glob("**/config.json"))


def discover_remote_config(model_id: str) -> tuple[dict[str, Any], str]:
    """Find and load the best config.json for a remote HF checkpoint."""
    candidates: list[str] = []

    try:
        candidates.append("config.json")
        root_index = _download_config(model_id, "model_index.json")
        index_config = _load_json(root_index)
        candidates.extend(_paths_from_model_index(index_config))
    except Exception:
        pass

    try:
        repo_configs = _list_repo_config_paths(model_id)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not list files for Hugging Face checkpoint `{model_id}`: {exc}"
        ) from exc

    for path in repo_configs:
        if path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No config.json files found in `{model_id}`")

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    errors: list[str] = []

    for path in candidates:
        try:
            local_path = _download_config(model_id, path)
            config = _load_json(local_path)
            score = _score_config_content(config, path)
            ranked.append((score, path, config))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not ranked:
        detail = "; ".join(errors[:3])
        raise FileNotFoundError(
            f"Could not load any config.json from `{model_id}`. {detail}"
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, best_path, best_config = ranked[0]
    label = f"hf://{model_id}/{best_path}"
    return normalize_config(best_config, source_label=label), label


def discover_local_config(root: Path) -> tuple[dict[str, Any], str]:
    """Find and load the best config.json under a local checkpoint directory."""
    if root.is_file():
        config = _load_json(root)
        label = str(root.resolve())
        return normalize_config(config, source_label=label), label

    direct = root / "config.json"
    if direct.is_file():
        config = _load_json(direct)
        label = str(direct.resolve())
        return normalize_config(config, source_label=label), label

    candidates = _local_config_candidates(root)
    if not candidates:
        raise FileNotFoundError(f"No config.json files found under {root}")

    ranked: list[tuple[int, Path, dict[str, Any]]] = []
    for path in candidates:
        rel = str(path.relative_to(root))
        config = _load_json(path)
        score = _score_config_content(config, rel)
        ranked.append((score, path, config))

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, best_path, best_config = ranked[0]
    label = str(best_path.resolve())
    return normalize_config(best_config, source_label=label), label


def load_checkpoint_config(
    checkpoint: str | Path,
    *,
    config_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load config for a HF checkpoint id, optional subpath, or local directory."""
    if config_path:
        path = Path(config_path)
        checkpoint_path = Path(checkpoint)

        if path.is_file():
            config = _load_json(path)
            return normalize_config(config, source_label=str(path.resolve())), str(path.resolve())

        if checkpoint_path.is_dir():
            resolved = checkpoint_path / config_path
        else:
            resolved = _download_config(str(checkpoint), config_path)

        if not resolved.is_file():
            raise FileNotFoundError(f"Config path not found: {config_path}")
        config = _load_json(resolved)
        label = str(resolved.resolve()) if resolved.exists() else f"hf://{checkpoint}/{config_path}"
        return normalize_config(config, source_label=label), label

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return discover_local_config(path)

    if path.is_file():
        return discover_local_config(path)

    try:
        downloaded = _download_config(str(checkpoint), "config.json")
        config = _load_json(downloaded)
        label = f"hf://{checkpoint}/config.json"
        return normalize_config(config, source_label=label), label
    except Exception:
        return discover_remote_config(str(checkpoint))
