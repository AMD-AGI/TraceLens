###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stand-ins for compiled libraries reproduce what the run needs of them."""

import os

import pytest
import torch

from TraceLens.ModelUtils.checkpoint_index import CheckpointFile
from TraceLens.ModelUtils.library_models import safetensors_modules
from TraceLens.ModelUtils.sandbox_fs import ScratchFS


def _checkpoint():
    file = CheckpointFile("r", "main", "model.safetensors", size=1 << 20)
    file.metadata = {"format": "pt"}
    return file


def test_a_weight_file_opens_and_holds_no_tensor():
    modules = safetensors_modules({"/m/model.safetensors": _checkpoint()}.get)
    with modules["safetensors"].safe_open(
        "/m/model.safetensors", framework="pt"
    ) as handle:
        assert handle.keys() == [] and handle.metadata() == {"format": "pt"}
        with pytest.raises(modules["safetensors"].SafetensorError):
            handle.get_tensor("w")
    assert modules["safetensors.torch"].load_file("/m/model.safetensors") == {}
    # An entry the run states the checkpoint holds is a placeholder.
    held = _checkpoint()
    held.tensors = {"d2t": torch.zeros(1, dtype=torch.int64)}
    modules = safetensors_modules({"/m/model.safetensors": held}.get)
    with modules["safetensors"].safe_open("/m/model.safetensors") as handle:
        assert (
            handle.keys() == ["d2t"] and handle.get_tensor("d2t").dtype == torch.int64
        )
    with pytest.raises(FileNotFoundError):
        modules["safetensors"].safe_open("/m/other.safetensors")
    with pytest.raises(PermissionError):
        modules["safetensors.torch"].save_file({}, "/m/out.safetensors")


def test_a_checkpoint_file_is_listed_and_sized_but_its_bytes_are_not_opened(tmp_path):
    path = os.path.normpath(str(tmp_path / "model.safetensors"))
    fs = ScratchFS({path: _checkpoint()}, lambda candidate: True)
    assert fs.exists(path) and fs.is_file(path)
    assert "model.safetensors" in fs.listdir(str(tmp_path))
    assert fs.stat(path).st_size == 1 << 20
    with pytest.raises(PermissionError):
        fs.open(path, "rb")
