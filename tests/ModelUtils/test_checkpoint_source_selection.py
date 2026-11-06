###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A checkpoint's Python files are its source only if they implement it.

A model repo ships more than its modeling code. DeepSeek-V4-Flash carries a
standalone reference implementation under ``inference/model.py`` and a
chat-template / tool-call encoder under ``encoding/``; neither defines the
``DeepseekV4ForCausalLM`` the config names, because the real implementation is
the one installed with transformers.

Taking them anyway did two kinds of damage at once: ``files`` was no longer
empty, so the transformers lookup was skipped entirely, and the filename
``model.py`` satisfied ``_has_modeling_implementation``, so the upstream
fallback was skipped too. The export then analysed a tool-call encoder as if it
were the model -- 2267 nodes became 1472, with 68 type-check warnings and 12
frames gone. Nothing in the checkpoint changed to cause it; the repo simply
published more files than it used to.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from TraceLens.ModelUtils import source
from TraceLens.ModelUtils.source import (
    _checkpoint_modeling_files,
    _declared_architectures,
    _defines_any_class,
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class TestDeclaredArchitectures:
    def test_the_top_level_list(self) -> None:
        assert _declared_architectures(
            {"architectures": ["DeepseekV4ForCausalLM"]}
        ) == {"DeepseekV4ForCausalLM"}

    def test_a_nested_tower_counts_too(self) -> None:
        """A VLM's vision tower is implemented by the checkpoint just as much."""
        config = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "vision_config": {"architectures": ["Glm5NextVisionModel"]},
        }
        assert _declared_architectures(config) == {
            "Glm5NextForConditionalGeneration",
            "Glm5NextVisionModel",
        }

    def test_no_architectures_is_no_claim(self) -> None:
        assert _declared_architectures({}) == set()


class TestDefinesAnyClass:
    def test_a_defining_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "modeling_x.py", "class Wanted:\n    pass\n")
        assert _defines_any_class(path, {"Wanted"})

    def test_a_nested_definition_still_counts(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "m.py", "if True:\n    class Wanted:\n        pass\n")
        assert _defines_any_class(path, {"Wanted"})

    def test_a_file_that_only_mentions_the_name(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "m.py", "WANTED = 'Wanted'\n")
        assert not _defines_any_class(path, {"Wanted"})

    def test_an_unparsable_file_is_not_a_match(self, tmp_path: Path) -> None:
        """A syntax error answers "no", never raises -- one bad file in a repo
        must not take the whole export down."""
        path = _write(tmp_path, "m.py", "class Wanted(:\n")
        assert not _defines_any_class(path, {"Wanted"})


class TestCheckpointModelingFiles:
    def test_the_deepseek_case_is_rejected(self, tmp_path: Path) -> None:
        """A reference implementation and a chat encoder are not the model."""
        files = [
            _write(
                tmp_path,
                "inference/model.py",
                "class ModelArgs:\n    pass\nclass ParallelEmbedding:\n    pass\n",
            ),
            _write(
                tmp_path,
                "encoding/encoding_dsv4.py",
                "def encode_messages():\n    pass\n",
            ),
        ]
        config = {"architectures": ["DeepseekV4ForCausalLM"]}
        assert _checkpoint_modeling_files(files, config) == []

    def test_a_real_checkpoint_implementation_is_kept(self, tmp_path: Path) -> None:
        files = [
            _write(
                tmp_path,
                "modeling_custom.py",
                "class CustomForCausalLM:\n    pass\n",
            )
        ]
        config = {"architectures": ["CustomForCausalLM"]}
        assert _checkpoint_modeling_files(files, config) == files

    def test_one_defining_file_keeps_its_siblings(self, tmp_path: Path) -> None:
        """Modeling code is split across modules; the set is kept or dropped
        whole, since a helper module defines no architecture of its own."""
        helper = _write(tmp_path, "helpers.py", "def rotate():\n    pass\n")
        main = _write(tmp_path, "modeling_custom.py", "class CustomModel:\n    pass\n")
        config = {"architectures": ["CustomModel"]}
        assert _checkpoint_modeling_files([helper, main], config) == [helper, main]

    def test_auto_map_is_the_checkpoint_speaking_for_itself(
        self, tmp_path: Path
    ) -> None:
        """``auto_map`` names the implementing files, so it is trusted as-is."""
        files = [_write(tmp_path, "modeling_custom.py", "x = 1\n")]
        config = {
            "architectures": ["CustomForCausalLM"],
            "auto_map": {"AutoModel": "modeling_custom.CustomForCausalLM"},
        }
        assert _checkpoint_modeling_files(files, config) == files

    def test_no_declared_architecture_leaves_the_files_alone(
        self, tmp_path: Path
    ) -> None:
        files = [_write(tmp_path, "model.py", "class Whatever:\n    pass\n")]
        assert _checkpoint_modeling_files(files, {}) == files


class TestAConfigOnlySnapshotStillFetchesModeling:
    def test_auto_map_modeling_is_downloaded_before_transformers(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """AutoConfig leaves only configuration_*.py in the snapshot.

        That file makes the snapshot look occupied, so the modeling download
        has to run anyway. Otherwise the transformers copy is what gets
        analysed, and its class names are not the checkpoint's.
        """
        snapshot = tmp_path / "snapshot"
        _write(snapshot, "configuration_kimi_k3.py", "class KimiK3Config:\n    pass\n")
        modeling = tmp_path / "modeling_kimi_linear.py"
        modeling.write_text("class KimiDecoderLayer:\n    pass\n", encoding="utf-8")
        seen: list[tuple[str, list[str], str | None]] = []

        def download(model_id, names, revision=None):
            seen.append((model_id, list(names), revision))
            if "modeling_kimi_linear.py" in names:
                return [modeling]
            return []

        monkeypatch.setattr(
            source,
            "_hub_snapshot_root",
            lambda model_id, revision=None: snapshot,
        )
        monkeypatch.setattr(source, "_download_repo_files", download)
        monkeypatch.setattr(
            source,
            "_list_repo_python_files",
            lambda model_id, revision=None: [
                "configuration_kimi_k3.py",
                "modeling_kimi_linear.py",
            ],
        )
        monkeypatch.setattr(
            source,
            "_transformers_github_modeling_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("transformers fallback")
            ),
        )

        files, labels = source.resolve_source_files(
            "moonshotai/Kimi-K3",
            {
                "model_type": "kimi_k3",
                "auto_map": {
                    "AutoModel": "modeling_kimi_linear.KimiLinearModel",
                },
            },
            revision="f831ab66814297da540d832a5235f8e904f29d06",
        )
        assert modeling in files
        assert "hf://moonshotai/Kimi-K3" in labels
        assert seen
        assert all(
            revision == "f831ab66814297da540d832a5235f8e904f29d06"
            for _, _, revision in seen
        )


def test_weight_files_are_not_downloaded(monkeypatch):
    """A hub fetch of modeling code must not pull checkpoint weights."""
    downloaded: list[str] = []

    def download(model_id, name, revision=None):
        downloaded.append(name)
        return f"/tmp/{name}"

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    paths = source._download_repo_files(
        "org/model",
        [
            "modeling_muse.py",
            "model.safetensors",
            "pytorch_model.bin",
            "model.gguf",
        ],
    )
    assert downloaded == ["modeling_muse.py"]
    assert paths == [Path("/tmp/modeling_muse.py")]
    assert source.is_weight_file("weights.pt")
    assert not source.is_weight_file("config.json")


def test_installed_modeling_file_uses_the_wrapper_type(tmp_path, monkeypatch):
    """A flattened text ``model_type`` still resolves the wrapper's modeling file.

    Muse Spark's config says ``muse_spark_text`` after normalization, but the
    classes live in ``modeling_muse_spark.py``. ``vision_config`` is not a
    module name and is not searched.
    """
    modeling = tmp_path / "modeling_muse_spark.py"
    modeling.write_text(
        "class MuseSparkForConditionalGeneration:\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        source, "_transformers_versioned_modeling_file", lambda *_a, **_k: None
    )
    monkeypatch.setattr(source, "_hub_snapshot_root", lambda *_a, **_k: None)
    monkeypatch.setattr(source, "_list_repo_python_files", lambda *_a, **_k: [])

    def _no_download(*_args, **_kwargs):
        return []

    monkeypatch.setattr(source, "_download_repo_files", _no_download)
    seen: list[str] = []

    def path_for(model_type):
        seen.append(model_type)
        return modeling if model_type == "muse_spark" else None

    monkeypatch.setattr(source, "_transformers_modeling_path", path_for)
    monkeypatch.setattr(
        source, "_transformers_github_modeling_file", lambda *_a, **_k: None
    )
    files, _labels = source.resolve_source_files(
        "unsloth/Muse-Spark-Open-816B-A42B",
        {
            "model_type": "muse_spark_text",
            "_wrapper_model_type": "muse_spark",
            "text_config": {"model_type": "muse_spark_text"},
            "vision_config": {"model_type": "muse_spark_vision"},
            "architectures": ["MuseSparkForConditionalGeneration"],
        },
    )
    assert files == [modeling]
    assert seen == ["muse_spark_text", "muse_spark"]
    assert "muse_spark_vision" not in seen
