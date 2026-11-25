###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An engine start leaves the process that ran it as it found it."""

import logging
import logging.config

from TraceLens.ModelUtils.engine_run import _logging_state, _restore_logging


def test_the_logging_an_engine_configures_is_undone_after_its_start():
    ours = logging.getLogger("tracelens.test.start")
    ours.setLevel(logging.INFO)
    state = _logging_state()
    root_level = logging.root.level
    # What an engine's start does to the process's logging.
    logging.config.dictConfig(
        {"version": 1, "disable_existing_loggers": True, "root": {"level": "WARNING"}}
    )
    assert ours.disabled
    _restore_logging(state)
    assert (
        not ours.disabled
        and ours.level == logging.INFO
        and logging.root.level == root_level
    )


def test_a_cached_value_is_written_whole(tmp_path):
    import pickle

    from TraceLens.ModelUtils.engine_run import _write_cache

    path = tmp_path / "cache" / "run.pkl"
    _write_cache(path, {"graphs": [1, 2, 3]})
    assert pickle.loads(path.read_bytes()) == {"graphs": [1, 2, 3]}
    # Nothing half written is left beside it.
    assert [entry.name for entry in path.parent.iterdir()] == ["run.pkl"]
