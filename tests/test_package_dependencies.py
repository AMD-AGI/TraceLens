###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check the dependency metadata consumed by package installers."""

import importlib.metadata
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


def test_jax_dependencies_are_optional(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    distribution = importlib.metadata.PathDistribution(
        next(tmp_path.glob("*.egg-info"))
    )
    requirements = [Requirement(value) for value in distribution.requires]

    def dependencies(extra):
        return {
            requirement.name.lower()
            for requirement in requirements
            if requirement.marker is None
            or requirement.marker.evaluate({"extra": extra})
        }

    assert {"xprof", "protobuf"}.isdisjoint(dependencies(""))
    assert {"xprof", "protobuf"} <= dependencies("jax")
    assert "pandas" in dependencies("")
