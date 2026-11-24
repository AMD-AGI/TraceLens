###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The machine a traced run used, as interpreted engine source probes it.

Engine and kernel-library code asks its machine questions while it sets up:
which GPU target is installed (``rocminfo``), which HIP release
(``hipconfig``, a version file), where the toolkit lives. The answers are
facts the trace records. This module answers with them, through the files and
the commands the source reaches for, and with nothing else: any other command
fails the way a missing tool does.
"""

from __future__ import annotations

import re
import subprocess
import types
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from TraceLens.ModelUtils.torch_runtime import DeviceFacts

_TOOLKIT = "/opt/rocm"
_GFX = re.compile(r"\bgfx[0-9a-f]{3,4}\b")


def gpu_target(prefix: str, sources: Iterable[Path]) -> str:
    """The full GPU target a library names for a device family ``prefix``.

    A trace records ``gfx<major><minor>``; the stepping that completes the
    target (``gfx942``, ``gfx950``) is the one the kernel libraries' own
    source uses for that family. With several, the most used one wins.
    """
    if not prefix.startswith("gfx"):
        return prefix
    seen: Counter[str] = Counter()
    for root in sources:
        for path in Path(root).rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in _GFX.findall(text):
                if match.startswith(prefix) and len(match) == len(prefix) + 1:
                    seen[match] += 1
    return seen.most_common(1)[0][0] if seen else prefix


def _split_pipeline(line: str) -> list[str]:
    """``a | b | c`` into its stages, leaving quoted ``|`` alone."""
    stages, current, quote = [], [], None
    for char in line:
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "|":
            stages.append("".join(current))
            current = []
            continue
        current.append(char)
    stages.append("".join(current))
    return stages


def _grep(args: list[str], data: str) -> str:
    after = 0
    flags = 0
    pattern = None
    items = iter(args)
    for item in items:
        if item == "-A":
            after = int(next(items))
        elif item.startswith("-A"):
            after = int(item[2:])
        elif item == "-i":
            flags |= re.IGNORECASE
        elif item.startswith("-"):
            continue
        elif pattern is None:
            pattern = item
    lines = data.splitlines()
    keep: list[str] = []
    remaining = -1
    for line in lines:
        if pattern is not None and re.search(pattern, line, flags):
            keep.append(line)
            remaining = after
        elif remaining > 0:
            keep.append(line)
            remaining -= 1
    return "\n".join(keep) + ("\n" if keep else "")


def _awk(args: list[str], data: str) -> str:
    program = args[-1] if args else ""
    fields = [int(item) for item in re.findall(r"\$(\d+)", program)]
    out = []
    for line in data.splitlines():
        parts = line.split()
        out.append(
            " ".join(
                parts[index - 1] if 0 < index <= len(parts) else "" for index in fields
            )
        )
    return "\n".join(out) + ("\n" if out else "")


def _head_tail(take_last: bool):
    def run(args: list[str], data: str) -> str:
        count = 10
        for index, item in enumerate(args):
            if item == "-n" and index + 1 < len(args):
                count = int(args[index + 1])
            elif item.startswith("-") and item[1:].isdigit():
                count = int(item[1:])
        lines = data.splitlines()
        chosen = lines[-count:] if take_last else lines[:count]
        return "\n".join(chosen) + ("\n" if chosen else "")

    return run


def _cut(args: list[str], data: str) -> str:
    delimiter, field = "\t", 1
    for index, item in enumerate(args):
        if item == "-d":
            delimiter = args[index + 1]
        elif item.startswith("-d") and len(item) > 2:
            delimiter = item[2:]
        elif item == "-f":
            field = int(args[index + 1])
        elif item.startswith("-f"):
            field = int(item[2:])
    out = [
        (line.split(delimiter) + [""] * field)[field - 1] for line in data.splitlines()
    ]
    return "\n".join(out) + ("\n" if out else "")


_FILTERS = {
    "grep": _grep,
    "awk": _awk,
    "head": _head_tail(False),
    "tail": _head_tail(True),
    "cut": _cut,
    "wc": lambda args, data: f"{len(data.splitlines())}\n",
}


# Compilers and build drivers a library invokes to build a kernel it then
# loads. On the traced machine they succeeded; what they made is loaded as a
# native stand-in, so the build itself does nothing here.
_BUILD_TOOLS = frozenset(
    {
        "make",
        "ninja",
        "cmake",
        "hipcc",
        "clang",
        "clang++",
        "gcc",
        "g++",
        "ld",
        "amdclang++",
    }
)


class RunMachine:
    """Files and commands of the traced machine, built from the trace's facts."""

    def __init__(
        self,
        device: DeviceFacts,
        hip_version: str | None,
        nccl_version: str | None = None,
    ):
        self.device = device
        self.hip_version = hip_version or ""
        # The collective library's version, as the trace's distributed info records it.
        self.nccl_version = nccl_version or ""

    def c_functions(self) -> dict:
        """C library calls the machine answers from its facts, by symbol."""
        functions = {}
        if self.nccl_version:
            major, minor, patch = (
                int(part) for part in (self.nccl_version.split(".") + ["0", "0"])[:3]
            )
            # NCCL_VERSION_CODE since 2.9: major * 10000 + minor * 100 + patch.
            code = major * 10000 + minor * 100 + patch

            def nccl_get_version(pointer):
                target = getattr(pointer, "_obj", pointer)
                target.value = code
                return 0

            functions["ncclGetVersion"] = nccl_get_version
        return functions

    def files(self) -> dict[str, str]:
        files = {
            f"{_TOOLKIT}/bin/rocminfo": "",
            f"{_TOOLKIT}/bin/hipconfig": "",
            f"{_TOOLKIT}/bin/hipcc": "",
        }
        if self.hip_version:
            major, minor, patch = (self.hip_version.split(".") + ["0", "0"])[:3]
            files[f"{_TOOLKIT}/.info/version"] = self.hip_version
            files[f"{_TOOLKIT}/include/hip/hip_version.h"] = (
                f"#define HIP_VERSION_MAJOR {major}\n"
                f"#define HIP_VERSION_MINOR {minor}\n"
                f"#define HIP_VERSION_PATCH {patch}\n"
            )
        return files

    def shell(self, line: str) -> str:
        """Run a shell pipeline of this machine's tools and text filters."""
        import shlex

        output = None
        for stage in _split_pipeline(line):
            words = shlex.split(stage)
            if not words:
                continue
            output = self._run(words, output)
        return output or ""

    def _run(self, words: list[str], data: str | None) -> str:
        tool = Path(words[0]).name
        if data is None or tool not in _FILTERS:
            return self._output(words)
        return _FILTERS[tool](words[1:], data)

    def _output(self, command: list[str]) -> str:
        tool = Path(str(command[0])).name if command else ""
        if tool == "rocminfo":
            kilobytes = self.device.total_memory // 1024
            # The host CPU is agent 1, as rocminfo lists it; the trace
            # records only the GPUs, so it carries no further detail.
            agents = [
                "Agent 1\n  Name:                    host\n  Device Type:             CPU\n"
            ]
            for index in range(self.device.count):
                agents.append(
                    f"Agent {index + 2}\n"
                    f"  Name:                    {self.device.gfx_target}\n"
                    f"  Marketing Name:          {self.device.name}\n"
                    f"  Device Type:             GPU\n"
                    f"  Compute Unit:            {self.device.compute_units}\n"
                    f"  Pool Info:\n"
                    f"    Pool 1\n"
                    f"      Segment:                 GLOBAL; FLAGS: COARSE GRAINED\n"
                    f"      Size:                    {kilobytes}({kilobytes:#x}) KB\n"
                )
            return "\n".join(agents)
        if tool == "rocm_agent_enumerator":
            # One line per agent: the host CPU is gfx000.
            names = ["gfx000", *[self.device.gfx_target] * self.device.count]
            return "\n".join(names) + "\n"
        if tool == "hipconfig" and self.hip_version:
            return self.hip_version
        if tool == "cat":
            # A file the trace does not describe reads as empty.
            return "".join(self.files().get(str(path), "") for path in command[1:])
        if tool in _BUILD_TOOLS:
            # The run built what it loaded; the artifact itself is never read.
            return ""
        raise FileNotFoundError(f"{tool}: not a tool the traced machine records")

    def popen(self, command, mode="r", buffering=-1):
        import io

        return io.StringIO(self.shell(str(command)))

    def system(self, command) -> int:
        """``os.system``: 0 for a command the machine runs, 127 otherwise."""
        try:
            self.shell(str(command))
        except FileNotFoundError:
            return 127
        return 0

    def socket_module(self) -> types.ModuleType:
        """Sockets that bind and listen locally and reach nothing."""
        import socket

        module = types.ModuleType("socket")
        module.__dict__.update(
            {
                key: value
                for key, value in vars(socket).items()
                if key.isupper() or key in ("error", "timeout", "gaierror", "herror")
            }
        )
        ports = iter(range(30000, 60000))

        class Socket:
            def __init__(self, family=socket.AF_INET, kind=socket.SOCK_STREAM, *args):
                self.family = family
                self.type = kind
                self.address = ("127.0.0.1", 0)

            def bind(self, address):
                host, port = address[0], address[1]
                self.address = (host, port or next(ports))

            def getsockname(self):
                return self.address

            def connect(self, address):
                raise ConnectionRefusedError("the traced machine is not reachable")

            def connect_ex(self, address):
                return 111

            def setsockopt(self, *args):
                pass

            def setblocking(self, flag):
                pass

            def settimeout(self, value):
                pass

            def listen(self, *args):
                pass

            def close(self):
                pass

            def fileno(self):
                return -1

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        module.socket = Socket
        module.SocketType = Socket
        module.gethostname = lambda: "localhost"
        module.getfqdn = lambda name="": "localhost"
        module.gethostbyname = lambda name: "127.0.0.1"
        module.has_ipv6 = False
        module.getaddrinfo = lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0))
        ]
        module.inet_aton = socket.inet_aton
        module.inet_ntoa = socket.inet_ntoa
        module.inet_pton = socket.inet_pton
        return module

    def install(self, interpreter) -> None:
        """Give an interpreter this machine's files, commands, processes, sockets."""
        interpreter.environment.files.update(self.files())
        interpreter.modules["subprocess"] = self.subprocess_module()
        interpreter.modules["socket"] = self.socket_module()
        interpreter.modules["os"].popen = self.popen
        interpreter.modules["os"].system = self.system
        shared_memory = self.shared_memory_module()
        interpreter.modules["multiprocessing.shared_memory"] = shared_memory
        setattr(
            interpreter.import_module("multiprocessing"), "shared_memory", shared_memory
        )
        interpreter.modules["amdsmi"] = self.amdsmi_module()

    def amdsmi_module(self) -> types.ModuleType:
        """``amdsmi``: the node's GPUs as the trace's ``deviceProperties`` list them.

        The trace does not record how the GPUs are linked; the devices of
        one traced node are taken to be the full one-hop XGMI mesh of the
        Instinct platforms that carry several of them.
        """
        device = self.device
        module = types.ModuleType("amdsmi")

        class AmdSmiException(Exception):
            pass

        class AmdSmiMemoryType:
            VRAM = 0
            VIS_VRAM = 1
            GTT = 2

        class AmdSmiLinkType:
            AMDSMI_LINK_TYPE_INTERNAL = 0
            AMDSMI_LINK_TYPE_PCIE = 1
            AMDSMI_LINK_TYPE_XGMI = 2

        handles = [f"gpu{index}" for index in range(device.count)]

        def _index(handle):
            if handle not in handles:
                raise AmdSmiException(f"unknown processor handle {handle!r}")
            return handles.index(handle)

        def link_type(handle, peer):
            _index(handle), _index(peer)
            return {"hops": 1, "type": AmdSmiLinkType.AMDSMI_LINK_TYPE_XGMI}

        module.AmdSmiException = AmdSmiException
        module.AmdSmiMemoryType = AmdSmiMemoryType
        module.AmdSmiLinkType = AmdSmiLinkType
        module.amdsmi_init = lambda *flags: None
        module.amdsmi_shut_down = lambda: None
        module.amdsmi_get_processor_handles = lambda: list(handles)
        module.amdsmi_topo_get_link_type = link_type
        module.amdsmi_get_gpu_asic_info = lambda handle: {
            "market_name": device.name,
            "target_graphics_version": device.gfx_target,
            "num_compute_units": device.compute_units,
            "asic_serial": f"{_index(handle):016x}",
        }
        module.amdsmi_get_gpu_memory_total = lambda handle, kind=0: (
            _index(handle),
            device.total_memory,
        )[1]
        module.amdsmi_get_gpu_device_uuid = (
            lambda handle: f"{_index(handle):08x}-0000-0000-0000-000000000000"
        )
        # The trace records no host topology: one NUMA node.
        module.amdsmi_topo_get_numa_node_number = lambda handle: (_index(handle), 0)[1]
        return module

    def shared_memory_module(self) -> types.ModuleType:
        """``multiprocessing.shared_memory``: named segments every rank on the node sees."""
        module = types.ModuleType("multiprocessing.shared_memory")
        segments: dict[str, bytearray] = {}
        counter = [0]

        class SharedMemory:
            def __init__(self, name=None, create=False, size=0, *, track=True):
                if create:
                    if name is None:
                        counter[0] += 1
                        name = f"psm_{counter[0]}"
                    if name in segments:
                        raise FileExistsError(name)
                    segments[name] = bytearray(size)
                elif name not in segments:
                    raise FileNotFoundError(name)
                self.name = name
                self._data = segments[name]
                self.buf = memoryview(self._data)
                self.size = len(self._data)

            def close(self):
                pass

            def unlink(self):
                segments.pop(self.name, None)

        class ShareableList(list):
            def __init__(self, sequence=None, *, name=None):
                super().__init__(sequence or [])
                self.shm = SharedMemory(name=name, create=name is None, size=1)

        module.SharedMemory = SharedMemory
        module.ShareableList = ShareableList
        return module

    def subprocess_module(self) -> types.ModuleType:
        machine = self
        module = types.ModuleType("subprocess")
        module.PIPE = subprocess.PIPE
        module.STDOUT = subprocess.STDOUT
        module.DEVNULL = subprocess.DEVNULL
        module.CalledProcessError = subprocess.CalledProcessError
        module.TimeoutExpired = subprocess.TimeoutExpired
        module.SubprocessError = subprocess.SubprocessError
        module.CompletedProcess = subprocess.CompletedProcess

        import shlex

        def _line(args, shell=False):
            if isinstance(args, str):
                return args
            if shell:
                # With a shell, the first item is the command line itself.
                return str(args[0])
            return " ".join(shlex.quote(str(item)) for item in args)

        def run(
            args, *rest, text=False, universal_newlines=False, shell=False, **kwargs
        ):
            output = machine.shell(_line(args, shell))
            data = output if (text or universal_newlines) else output.encode()
            empty = "" if (text or universal_newlines) else b""
            return subprocess.CompletedProcess(args, 0, stdout=data, stderr=empty)

        def check_output(
            args, *rest, text=False, universal_newlines=False, shell=False, **kwargs
        ):
            output = machine.shell(_line(args, shell))
            return output if (text or universal_newlines) else output.encode()

        def refused(*args, **kwargs):
            raise PermissionError("the interpreted source may not start processes")

        module.run = run
        module.check_output = check_output
        module.check_call = (
            lambda args, *rest, **kwargs: machine.shell(_line(args)) and 0
        )
        module.call = module.check_call
        module.getoutput = lambda command: machine.shell(_line(command))
        module.Popen = refused
        return module
