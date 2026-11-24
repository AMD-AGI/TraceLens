###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The traced machine answers the run from the trace's facts."""

import ctypes

from TraceLens.ModelUtils.run_machine import RunMachine
from TraceLens.ModelUtils.torch_runtime import DeviceFacts


def _machine(**kwargs):
    device = DeviceFacts(name="AMD Instinct MI355X", gfx_target="gfx950", count=2)
    return RunMachine(device, "7.1.0", **kwargs)


def test_the_agent_enumerator_lists_the_host_then_each_gpu():
    assert _machine().shell("rocm_agent_enumerator -name").split() == [
        "gfx000",
        "gfx950",
        "gfx950",
    ]


def test_the_collective_library_reports_the_traced_version():
    functions = _machine(nccl_version="2.30.4").c_functions()
    version = ctypes.c_int()
    assert functions["ncclGetVersion"](ctypes.byref(version)) == 0
    assert version.value == 23004
    assert "ncclGetVersion" not in _machine().c_functions()


def test_shared_memory_is_one_segment_every_rank_on_the_node_opens():
    module = _machine().shared_memory_module()
    created = module.SharedMemory(create=True, size=16, name="probe")
    created.buf[:5] = b"magic"
    opened = module.SharedMemory(name="probe")
    assert bytes(opened.buf[:5]) == b"magic"
    created.unlink()
    try:
        module.SharedMemory(name="probe")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("an unlinked segment must be gone")


def test_the_system_management_library_lists_the_traced_gpus_fully_linked():
    module = _machine().amdsmi_module()
    module.amdsmi_init()
    handles = module.amdsmi_get_processor_handles()
    assert len(handles) == 2
    assert (
        module.amdsmi_get_gpu_asic_info(handles[1])["target_graphics_version"]
        == "gfx950"
    )
    assert module.amdsmi_topo_get_link_type(handles[0], handles[1]) == {
        "hops": 1,
        "type": 2,
    }
    try:
        module.amdsmi_topo_get_link_type(handles[0], "gpu7")
    except module.AmdSmiException:
        pass
    else:
        raise AssertionError("a GPU the trace does not list has no link")
