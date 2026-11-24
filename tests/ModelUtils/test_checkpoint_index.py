###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A repository's weight files are listed by name and size; none is read."""

from types import SimpleNamespace

from TraceLens.ModelUtils.checkpoint_index import checkpoint_files


def test_weight_files_come_from_the_listing_and_hold_no_tensor(monkeypatch):
    import huggingface_hub

    siblings = [
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model-00001.safetensors", size=1 << 30),
        SimpleNamespace(rfilename="draft/pytorch_model.bin", size=1 << 20),
    ]

    class Api:
        def model_info(self, repo, revision=None, files_metadata=False):
            return SimpleNamespace(sha="abc", siblings=siblings)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    files = checkpoint_files("org/model")
    assert sorted(files) == ["draft/pytorch_model.bin", "model-00001.safetensors"]
    weights = files["model-00001.safetensors"]
    assert (
        weights.size == 1 << 30 and weights.revision == "abc" and weights.tensors == {}
    )
