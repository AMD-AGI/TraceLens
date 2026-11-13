###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared utilities for the microbenchmark scripts.

- resolve_physical_device: logical torch index -> physical id used by smi tools.
- check_gpu_idle: pre-flight idle check via amdsmi or nvidia-smi.
- check_rdna_gpu_idle: the strict idle check of the RDNA protocol.
- mx_peak_without_native_support: MXFP4/MXFP6 ceiling on AMD targets that
  lack native MX matrix instructions.
- rdna_published_keys: the value and provenance of every TFLOPS key on an
  RDNA target.
- cu_tiled_sweep_shapes: --shape-sweep's tile-304 shapes for an RDNA CU count.
- resolve_rdna_window: the RDNA protocol's warm-up and window under --warmup / --rep.
- bandwidth_plateau, vopd_share: reading a bandwidth curve and an AMDGCN listing.
- parse_counter_list, parse_rdna_counters, counter_clock_mhz,
  checked_counter_value: reading and checking rocprofv3 counters on RDNA.
- SysfsTelemetry, platform_record, pci_bdf, power_held_w: clocks, power and
  board identity from sysfs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# AMD gfx targets whose matrix cores execute MXFP4/MXFP6 natively. On other
# targets these formats are converted and multiplied on the 16-bit matrix cores.
NATIVE_MX_GFX_ARCHS = frozenset({"gfx950"})

# Matrix formats the WMMA units of each RDNA family execute: gfx11 is RDNA3 and
# RDNA3.5 (gfx1100-gfx1103, gfx1150-gfx1153), gfx12 is RDNA4 (gfx1200, gfx1201).
RDNA_NATIVE_MATRIX_KEYS: Dict[str, frozenset] = {
    "gfx11": frozenset({"matrix_fp16", "matrix_bf16", "matrix_int8"}),
    "gfx12": frozenset({"matrix_fp16", "matrix_bf16", "matrix_int8", "matrix_fp8"}),
}
# Keys measured on every RDNA target besides its WMMA formats: FP32 and FP64
# GEMMs (the GEMM libraries run them on the vector ALUs) and the vector FMA rates.
RDNA_MEASURED_KEYS = frozenset(
    {"matrix_fp32", "matrix_fp64", "vector_fp16", "vector_fp32", "vector_fp64"}
)
LOW_PRECISION_MATRIX_KEYS = ("matrix_fp8", "matrix_fp4", "matrix_fp6")
# The measured key a format without a path of its own takes its rate from.
RDNA_DERIVED_FROM = {
    "matrix_fp8": "matrix_fp16",
    "matrix_fp4": "matrix_fp16",
    "matrix_fp6": "matrix_fp16",
    "vector_bf16": "vector_fp32",
}

# RDNA measurement protocol: back-to-back warm-up before each section, the
# shorter warm-up before a config whose setup left the GPU idle (a Triton
# compile, large allocations), the timed window (median per call), the burst
# pass after an idle gap, the back-to-back run of the 16-bit GEMM winners
# (median chunk after the settle time), and the sysfs telemetry period.
RDNA_WARM_S = 3.0
RDNA_REWARM_S = 1.0
RDNA_REWARM_AFTER_S = 0.25
RDNA_WINDOW_MS = 1000
RDNA_BURST_IDLE_S = 5.0
RDNA_BURST_WINDOW_MS = 100
RDNA_BACK_TO_BACK_S = 30.0
RDNA_BACK_TO_BACK_SETTLE_S = 10.0
RDNA_BACK_TO_BACK_CHUNK_S = 0.5
TELEMETRY_PERIOD_S = 0.1
# The sysfs clocks are SMU averages over a few hundred ms; a window with fewer
# samples than this records no clock.
MIN_CLOCK_SAMPLES = 3

# --shape-sweep on RDNA: the output tile and the size GEMM_SHAPES_SWEEP_TILE304
# tiles MI300X's 304 CUs with (4864 = 19 x 256).
SWEEP_TILE = 256
SWEEP_TILE_TARGET_DIM = 4864

# A bandwidth curve plateaus from the first size from which every step to the
# next larger size changes GB/s by less than this.
BW_PLATEAU_TOL = 0.02

# rocprof RDNA mode: bytes per GL2C->EA read request counter, the steady-state
# dispatches a case's counters are the median of, and the range a counter
# value must fall in: a clock as a multiple of the sysfs clock of the case's
# plain run, read bytes as multiples of the bytes the read kernel reads. The
# sclk DPM table bounds no clock: gfx12 boards run past its top level.
RDNA_GL2C_READ_BYTES: Dict[str, int] = {
    "GL2C_EA_RDREQ_32B": 32,
    "GL2C_EA_RDREQ_64B": 64,
    "GL2C_EA_RDREQ_96B": 96,
    "GL2C_EA_RDREQ_128B": 128,
    "GL2C_EA_RDREQ_256B": 256,
}
RDNA_STEADY_DISPATCHES = 5
RDNA_CLOCK_RANGE = (0.75, 1.25)
RDNA_READ_RATIO_RANGE = (0.5, 2.0)
_COUNTER_ENTRY_RE = re.compile(
    r"^Counter_Name\s*:\s*(\S+)(.*?)(?=^Counter_Name|\Z)", re.MULTILINE | re.DOTALL
)
_DIMENSIONS_RE = re.compile(r"^Dimensions\s*:(.*)$", re.MULTILINE)
_DIMENSION_RE = re.compile(r"DIMENSION_\w+\[(\d+):(\d+)\]")

# Idle check on RDNA: the median of this many readings, this far apart. A single
# reading calls a GPU under a 25% duty load idle about one time in six, and
# gfx_activity is a moving average that takes about 1.5 s to fall back under 5%
# after a short burst, so the readings span about 3 s.
IDLE_SAMPLES_RDNA = 20
IDLE_INTERVAL_S = 0.15
IDLE_PROBE_TIMEOUT_S = 60

SYSFS_ROOT = "/sys"
ROCM_VERSION_FILE = "/opt/rocm/.info/version"
_DPM_MHZ = re.compile(r"(\d+)\s*mhz", re.IGNORECASE)

# The RDNA idle check runs amdsmi in a fresh interpreter: a process that has
# imported torch can hold a second libamd_smi (the copy ROCm wheels bundle),
# with which amdsmi finds no GPU.
_RDNA_IDLE_PROBE = r"""
import json, sys, time
import amdsmi

bdf, samples, interval = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
out = {}
amdsmi.amdsmi_init()
try:
    handle = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
    readings = []
    for i in range(samples):
        if i:
            time.sleep(interval)
        readings.append(amdsmi.amdsmi_get_gpu_activity(handle).get("gfx_activity"))
    out["gfx_activity"] = readings
    out["vram"] = amdsmi.amdsmi_get_gpu_vram_usage(handle)
    try:
        out["processes"] = len(amdsmi.amdsmi_get_gpu_process_list(handle))
    except amdsmi.AmdSmiException as e:
        out["processes_error"] = str(e).strip().splitlines()[-1]
finally:
    amdsmi.amdsmi_shut_down()
print(json.dumps(out, default=str))
"""


class MeasurementError(RuntimeError):
    """A key with its own hardware path measured zero or failed."""


def gfx_target(gcn_arch_name: Optional[str]) -> str:
    """Base gfx target of a ``gcnArchName`` such as ``gfx942:sramecc+:xnack-``."""
    return (gcn_arch_name or "").split(":")[0]


def rdna_family(gfx: str) -> Optional[str]:
    """``gfx11`` or ``gfx12`` for an RDNA target such as ``gfx1100``, else ``None``."""
    for family in RDNA_NATIVE_MATRIX_KEYS:
        if len(gfx) == 7 and gfx.startswith(family):
            return family
    return None


def rdna_measured_keys(gfx: str) -> frozenset:
    """The TFLOPS keys an RDNA target has its own path for, so they are measured."""
    return RDNA_NATIVE_MATRIX_KEYS[rdna_family(gfx)] | RDNA_MEASURED_KEYS


def cu_tiled_sweep_shapes(n_cu: int) -> List[Tuple[int, int, int]]:
    """
    ``--shape-sweep``'s tile-304 shapes for *n_cu* CUs: their 4864 becomes the
    size closest to it whose 256x256 output tiles split evenly over the CUs
    (4864 itself at MI300X's 304 CUs, 5120 at 40 or 64 CUs).
    """
    step = SWEEP_TILE * n_cu // math.gcd(n_cu, 16)
    dim = step * max(1, round(SWEEP_TILE_TARGET_DIM / step))
    return [(dim, 4096, 8256), (dim, 8192, 4160), (8192, dim, 6878)]


def _derivation_reason(gfx: str, key: str) -> str:
    if key == "vector_bf16":
        return "BF16 elementwise math runs in FP32 on the vector ALUs"
    fmt = key.split("_")[1].upper()
    return (
        f"{gfx} WMMA has no {fmt} path; {fmt} GEMMs are converted and run on "
        "the 16-bit WMMA units"
    )


def rdna_published_keys(
    gfx: str,
    results: Mapping[str, float],
    configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    skipped: Iterable[str] = (),
) -> Optional[Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]]:
    """
    On an RDNA target, ``(values, provenance)`` for every TFLOPS key; ``None``
    on other targets.

    A key with its own path (:func:`rdna_measured_keys`) is published as
    measured, with its winning config from *configs*. FP32 and FP64 GEMMs keep
    the GEMM rate even where the vector FMA rate is higher, since that is what
    the GEMMs run at. A zero or failed measurement raises
    :class:`MeasurementError` rather than being written; keys in *skipped* (a
    section the run skipped) are written as 0 and marked as such. A format
    without a path takes the measured rate of the one key it is converted to
    (:data:`RDNA_DERIVED_FROM`): FP8, FP4 and FP6 GEMMs ``matrix_fp16``,
    ``vector_bf16`` the FP32 vector rate.
    """
    if rdna_family(gfx) is None:
        return None
    configs = configs or {}
    skipped = set(skipped)
    measured_keys = sorted(rdna_measured_keys(gfx))
    failed = [
        key
        for key in measured_keys
        if key not in skipped and not float(results.get(key) or 0.0) > 0.0
    ]
    if failed:
        raise MeasurementError(
            f"{gfx}: {', '.join(failed)} measured zero or failed; {gfx} has its own "
            f"path for these, so they are not written as 0"
        )
    values: Dict[str, float] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for key in measured_keys:
        if key in skipped:
            values[key], sources[key] = 0.0, {"source": "skipped"}
        else:
            values[key] = float(results[key])
            sources[key] = {"source": "measured", **configs.get(key, {})}
    for key, base in RDNA_DERIVED_FROM.items():
        if key in values:
            continue
        values[key] = values[base]
        sources[key] = (
            {"source": "skipped"}
            if base in skipped
            else {
                "source": "derived",
                "from": base,
                "reason": _derivation_reason(gfx, key),
            }
        )
    return values, sources


def resolve_rdna_window(
    warm_s: float, window_ms: int, warmup_ms: Optional[int], rep_ms: Optional[int]
) -> Tuple[float, int, List[str]]:
    """
    The RDNA protocol's warm-up and window: *warm_s* and *window_ms*, lengthened
    by a longer ``--warmup`` / ``--rep``. A shorter ``--warmup`` / ``--rep`` (the
    suite's options; Hyperloom passes 20 / 50 ms) does not shorten them; each
    such override is returned as a note for the log and the JSON.
    """
    notes = []
    if warmup_ms is not None:
        if warmup_ms / 1e3 > warm_s:
            warm_s = warmup_ms / 1e3
        else:
            notes.append(f"--warmup {warmup_ms} ms: {warm_s:g} s warm-up per section")
    if rep_ms is not None:
        if rep_ms > window_ms:
            window_ms = rep_ms
        else:
            notes.append(f"--rep {rep_ms} ms: {window_ms} ms window per config")
    for note in notes:
        print(f"RDNA protocol floor: {note}")
    if window_ms < RDNA_WINDOW_MS:
        print(
            f"Warning: a {window_ms} ms window is shorter than the protocol's "
            f"{RDNA_WINDOW_MS} ms; these keys are not comparable across boards."
        )
    return warm_s, window_ms, notes


def bandwidth_plateau(
    sizes: Sequence[int], gbps: Sequence[float], tol: float = BW_PLATEAU_TOL
) -> Optional[Tuple[int, float]]:
    """
    ``(first size, median GB/s)`` of the plateau of a bandwidth curve over
    increasing *sizes*: from the first size from which every step to the next
    larger size changes GB/s by less than *tol*. ``None`` while the curve still
    moves at the largest size.
    """
    for i in range(len(sizes) - 1):
        if all(abs(gbps[j + 1] / gbps[j] - 1) < tol for j in range(i, len(sizes) - 1)):
            return sizes[i], statistics.median(gbps[i:])
    return None


def vopd_share(amdgcn: str) -> Optional[float]:
    """
    Share of the FMA operations in an AMDGCN listing that issue in VOPD pairs
    (``v_dual_*``), which is how wave32 code dual-issues on RDNA. ``None``
    for a listing without FMAs.
    """
    dual = single = 0
    for line in amdgcn.splitlines():
        ops = [
            part.split()[0] for part in line.split(";")[0].split("::") if part.strip()
        ]
        if not ops or not ops[0].startswith("v_"):
            continue
        fmas = sum(1 for op in ops if "fma" in op)
        if ops[0].startswith("v_dual_"):
            dual += fmas
        else:
            single += fmas
    total = dual + single
    return dual / total if total else None


def parse_counter_list(text: str) -> Dict[str, int]:
    """``{counter: instances}`` from ``rocprofv3 -L`` output; instances is the
    product of the counter's dimension sizes, which rocprofv3 sums over."""
    out: Dict[str, int] = {}
    for name, body in _COUNTER_ENTRY_RE.findall(text):
        dims = _DIMENSIONS_RE.search(body)
        instances = 1
        for lo, hi in _DIMENSION_RE.findall(dims.group(1) if dims else ""):
            instances *= int(hi) - int(lo) + 1
        out[name] = instances
    return out


def parse_rdna_counters(csv_path: Path, hints: Sequence[str]) -> Dict[str, Any]:
    """
    From a rocprofv3 counter_collection.csv (one row per dispatch and
    counter): the kernel time and each counter's value, as medians over the
    last ``RDNA_STEADY_DISPATCHES`` dispatches whose kernel name contains one
    of *hints*.
    """
    per: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not any(h in row.get("Kernel_Name", "") for h in hints):
                continue
            ns = float(row["End_Timestamp"]) - float(row["Start_Timestamp"])
            d = per.setdefault(row["Dispatch_Id"], {"ns": ns})
            name = row["Counter_Name"]
            d[name] = d.get(name, 0.0) + float(row["Counter_Value"])
    steady = [per[k] for k in sorted(per, key=int)][-RDNA_STEADY_DISPATCHES:]
    if not steady:
        raise RuntimeError(f"no dispatches matching {list(hints)} in {csv_path}")
    names = sorted({k for d in steady for k in d if k != "ns"})
    return {
        "matching_dispatches": len(per),
        "kernel_ms": statistics.median(d["ns"] for d in steady) / 1e6,
        "counters": {
            k: statistics.median(d.get(k, 0.0) for d in steady) for k in names
        },
    }


def counter_clock_mhz(
    counter: str,
    value: float,
    instances: int,
    kernel_ms: float,
    sysfs_mhz: Optional[float],
) -> Tuple[Optional[float], str]:
    """
    The shader clock a busy-cycle *counter* gives: its *value* per instance
    over the kernel time. ``(None, reason)`` for zero, or for a clock outside
    ``RDNA_CLOCK_RANGE`` of *sysfs_mhz* (unchecked without it). A wrong
    instance count is off by a factor of 2 or more.
    """
    if kernel_ms <= 0:
        return None, f"{counter}: no kernel time"
    mhz = value / max(instances, 1) / (kernel_ms * 1e3)
    if mhz <= 0:
        return None, f"{counter} read 0"
    if sysfs_mhz:
        lo, hi = (f * sysfs_mhz for f in RDNA_CLOCK_RANGE)
        if not lo <= mhz <= hi:
            return None, (
                f"{counter} gives {mhz:.0f} MHz, outside {lo:.0f}-{hi:.0f} MHz "
                f"(sysfs {sysfs_mhz:.0f} MHz)"
            )
    return mhz, ""


def checked_counter_value(
    name: str, counters: Mapping[str, float], read_nbytes: int
) -> Tuple[Optional[float], str]:
    """
    ``dual_issue_share`` (VOPD pairs over VALU instructions, within 0-1) or
    ``read_bytes_ratio`` (GL2C read bytes over the *read_nbytes* the read
    kernel reads, within ``RDNA_READ_RATIO_RANGE``) from *counters*;
    ``(None, reason)`` for zero or a value outside its range.
    """
    if name == "dual_issue_share":
        valu = counters.get("SQ_INSTS_VALU", 0.0)
        if valu <= 0:
            return None, "SQ_INSTS_VALU read 0"
        value, lo, hi = counters.get("SQ_INSTS_DUAL_VALU_WAVE32", 0.0) / valu, 0.0, 1.0
    else:
        nbytes = sum(counters.get(c, 0.0) * b for c, b in RDNA_GL2C_READ_BYTES.items())
        if nbytes <= 0:
            return None, "GL2C read requests read 0"
        value, (lo, hi) = nbytes / read_nbytes, RDNA_READ_RATIO_RANGE
    if not lo <= value <= hi:
        return None, f"{name} is {value:.3f}, outside {lo:g}-{hi:g}"
    return value, ""


class SysfsTelemetry:
    """
    Shader clock, memory clock, fabric clock, power and power cap of one GPU,
    sampled from its sysfs hwmon every *period_s* in a background thread. The
    values are SMU averages, so they describe windows of a few hundred ms and
    longer. A board whose hwmon has no memory clock (an APU) reports the
    current ``pp_dpm_mclk`` level instead.
    """

    def __init__(
        self, bdf: str, *, period_s: float = TELEMETRY_PERIOD_S, root: str = SYSFS_ROOT
    ):
        self.bdf = bdf
        self.period_s = period_s
        dev = _pci_device_dir(root, bdf)
        hwmons = sorted((dev / "hwmon").glob("hwmon*"))
        self._hwmon = hwmons[0] if hwmons else None
        self._sclk = self._freq_input("sclk", "freq1_input")
        self._mclk = self._freq_input("mclk", None)
        self._mclk_dpm = None if self._mclk else dev / "pp_dpm_mclk"
        self._fclk_dpm = dev / "pp_dpm_fclk"
        self._samples: List[Dict[str, Optional[float]]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return self._hwmon is not None

    def _freq_input(self, label: str, default: Optional[str]) -> Optional[Path]:
        if self._hwmon is None:
            return None
        for label_file in sorted(self._hwmon.glob("freq*_label")):
            if _read_text(label_file) == label:
                return label_file.with_name(label_file.name.replace("_label", "_input"))
        return self._hwmon / default if default else None

    def read(self) -> Dict[str, Optional[float]]:
        """One sample in MHz and W; ``None`` for what the board does not expose."""

        def hwmon_file(name: str) -> Optional[Path]:
            return self._hwmon / name if self._hwmon is not None else None

        power = _read_scaled(hwmon_file("power1_average"), 1e6)
        if power is None:
            power = _read_scaled(hwmon_file("power1_input"), 1e6)
        mclk = _read_scaled(self._mclk, 1e6)
        if mclk is None and self.available:
            mclk = _dpm_current_mhz(self._mclk_dpm)
        return {
            "t": time.time(),
            "sclk_mhz": _read_scaled(self._sclk, 1e6),
            "mclk_mhz": mclk,
            "fclk_mhz": _dpm_current_mhz(self._fclk_dpm) if self.available else None,
            "power_w": power,
            "power_cap_w": _read_scaled(hwmon_file("power1_cap"), 1e6),
        }

    def start(self) -> "SysfsTelemetry":
        if self.available and self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="sysfs-telemetry", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self.read())
            self._stop.wait(self.period_s)

    def summary(self, t0: float, t1: float) -> Dict[str, Any]:
        """
        Medians of the samples taken between epoch times *t0* and *t1*; only
        the sample count for a window with fewer than ``MIN_CLOCK_SAMPLES``.
        """
        window = [s for s in list(self._samples) if t0 <= s["t"] <= t1]
        out: Dict[str, Any] = {"samples": len(window)}
        if len(window) < MIN_CLOCK_SAMPLES:
            return out
        for key in ("sclk_mhz", "mclk_mhz", "fclk_mhz", "power_w", "power_cap_w"):
            values = [s[key] for s in window if s[key] is not None]
            out[key] = round(statistics.median(values), 1) if values else None
        return out


def platform_record(bdf: str, *, root: str = SYSFS_ROOT) -> Dict[str, Any]:
    """
    What identifies a board and its configuration, from sysfs: PCI IDs, VBIOS,
    power cap (and where it came from, or why it is missing), performance
    level, the shader, memory and fabric DPM tables, and the kernel, amdgpu and
    ROCm versions. ``None`` for what is not exposed.
    """
    dev = _pci_device_dir(root, bdf)
    hwmons = sorted((dev / "hwmon").glob("hwmon*"))
    hw = hwmons[0] if hwmons else None
    power_cap = _read_scaled(hw / "power1_cap", 1e6) if hw else None
    dpm = {clk: _dpm_levels(dev / f"pp_dpm_{clk}") for clk in ("sclk", "mclk", "fclk")}
    return {
        "pci_bdf": bdf,
        "pci_vendor_id": _read_text(dev / "vendor"),
        "pci_device_id": _read_text(dev / "device"),
        "pci_revision": _read_text(dev / "revision"),
        "pci_subsystem_vendor_id": _read_text(dev / "subsystem_vendor"),
        "pci_subsystem_device_id": _read_text(dev / "subsystem_device"),
        "vbios_version": _read_text(dev / "vbios_version"),
        "power_cap_w": power_cap,
        "power_cap_default_w": (
            _read_scaled(hw / "power1_cap_default", 1e6) if hw else None
        ),
        "power_cap_source": (
            "hwmon power1_cap"
            if power_cap is not None
            else "not exposed: no hwmon power1_cap"
        ),
        "perf_level": _read_text(dev / "power_dpm_force_performance_level"),
        "sclk_max_mhz": max(dpm["sclk"]) if dpm["sclk"] else None,
        "mclk_max_mhz": max(dpm["mclk"]) if dpm["mclk"] else None,
        "dpm_levels_mhz": dpm,
        "kernel": platform.release(),
        "amdgpu_version": _read_text(Path(root) / "module" / "amdgpu" / "version"),
        "rocm_version": rocm_version(),
    }


def rocm_version() -> Optional[str]:
    """The ROCm release: the ``rocm`` Python package, else ``/opt/rocm/.info/version``."""
    try:
        return metadata.version("rocm")
    except metadata.PackageNotFoundError:
        return _read_text(Path(ROCM_VERSION_FILE))


def pci_bdf(props: Any) -> str:
    """PCI address of a ``torch.cuda`` device, as sysfs and amdsmi name it."""
    return (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )


def power_held_w(record: Any) -> Optional[float]:
    """The highest window median power (``power_w``) anywhere in *record*: what
    the board holds under load, also where it exposes no power cap."""
    if isinstance(record, dict):
        found = [power_held_w(v) for v in record.values()]
        found.append(record.get("power_w"))
    elif isinstance(record, list):
        found = [power_held_w(v) for v in record]
    else:
        return None
    found = [p for p in found if isinstance(p, (int, float))]
    return max(found) if found else None


def _pci_device_dir(root: str, bdf: str) -> Path:
    return Path(root) / "bus" / "pci" / "devices" / bdf


def _read_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_scaled(path: Optional[Path], divisor: float) -> Optional[float]:
    text = _read_text(path)
    try:
        return float(text) / divisor if text is not None else None
    except ValueError:
        return None


def _dpm_levels(path: Path) -> Optional[List[int]]:
    """MHz levels of a ``pp_dpm_*`` table such as ``0: 96Mhz``, without the sleep
    level (``S:``); ``None`` if absent."""
    levels = [
        int(match.group(1))
        for line in (_read_text(path) or "").splitlines()
        if not line.startswith("S:") and (match := _DPM_MHZ.search(line))
    ]
    return levels or None


def _dpm_current_mhz(path: Optional[Path]) -> Optional[float]:
    """The level of a ``pp_dpm_*`` table marked current (``*``), in MHz."""
    for line in (_read_text(path) or "").splitlines():
        match = _DPM_MHZ.search(line)
        if match and line.rstrip().endswith("*"):
            return float(match.group(1))
    return None


def mx_peak_without_native_support(
    gfx: str, matrix_results: Mapping[str, float]
) -> Optional[float]:
    """
    The 16-bit matrix peak to publish as ``matrix_fp4`` / ``matrix_fp6`` on an
    AMD target without native MX matrix instructions, since that is the rate
    those formats actually run at there. ``None`` keeps the measured value:
    the target has native support, or is not an AMD target.
    """
    if not gfx or gfx in NATIVE_MX_GFX_ARCHS:
        return None
    return max(
        float(matrix_results.get("matrix_fp16", 0.0)),
        float(matrix_results.get("matrix_bf16", 0.0)),
    )


def resolve_physical_device(logical: int) -> Tuple[int, str]:
    """Map a logical torch device index to the physical id used by amdsmi /
    nvidia-smi via HIP/ROCR/CUDA_VISIBLE_DEVICES. Returns (phys, source)."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            mapped = [int(p) for p in parts]
        except ValueError:
            return logical, f"{var}={val} (non-numeric, identity fallback)"
        if 0 <= logical < len(mapped):
            return mapped[logical], var
        return (
            logical,
            f"{var}={val} (logical {logical} out of range, identity fallback)",
        )
    return logical, "identity"


def _int_metric(value: Any, *, default: int = 0) -> int:
    """Coerce an amdsmi metric to int; treat ``N/A`` and non-numeric as *default*."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_amd_gpu_idle(
    phys: int,
    dev_tag: str,
    *,
    util_threshold: int,
    mem_threshold_pct: float,
) -> Tuple[bool, str]:
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if phys < 0 or phys >= len(devices):
            return (
                True,
                f"amdsmi {dev_tag}: physical id {phys} out of range "
                f"(found {len(devices)} GPU(s)); check skipped",
            )

        device = devices[phys]
        activity = amdsmi.amdsmi_get_gpu_activity(device)
        util = _int_metric(activity.get("gfx_activity"))

        vram = amdsmi.amdsmi_get_gpu_vram_usage(device)
        mem_used_mib = _int_metric(vram.get("vram_used"))
        mem_total_mib = _int_metric(vram.get("vram_total"))
        mem_pct = 100.0 * mem_used_mib / mem_total_mib if mem_total_mib > 0 else None

        mem_busy = mem_pct is not None and mem_pct > mem_threshold_pct
        mem_str = (
            f"{mem_pct:.1f}% of {mem_total_mib}MiB"
            if mem_pct is not None
            else "total unknown"
        )
        if util > util_threshold or mem_busy:
            return False, (
                f"amdsmi {dev_tag}: util={util}% mem={mem_used_mib}MiB " f"({mem_str})"
            )
        return True, (
            f"amdsmi {dev_tag}: idle (util={util}% mem={mem_used_mib}MiB {mem_str})"
        )
    finally:
        amdsmi.amdsmi_shut_down()


def check_gpu_idle(
    device: int, *, util_threshold: int = 5, mem_threshold_pct: float = 10.0
) -> Tuple[bool, str]:
    """Pre-flight idle check via amdsmi or nvidia-smi. Returns (is_idle, msg).
    Skipped (returns True) if neither library/tool is available."""
    phys, src = resolve_physical_device(device)
    dev_tag = f"logical {device} -> physical {phys} (via {src})"

    try:
        import amdsmi  # noqa: F401
    except ImportError:
        amdsmi = None

    if amdsmi is not None:
        try:
            return _check_amd_gpu_idle(
                phys,
                dev_tag,
                util_threshold=util_threshold,
                mem_threshold_pct=mem_threshold_pct,
            )
        except Exception as e:
            return True, f"amdsmi check skipped ({e})"

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(phys),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [p.strip() for p in (r.stdout or "").split(",")]
            util = int(parts[0]) if parts and parts[0].isdigit() else 0
            mem_used_mib = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            mem_total_mib = (
                int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            )
            mem_pct = (
                100.0 * mem_used_mib / mem_total_mib if mem_total_mib > 0 else None
            )
            mem_busy = mem_pct is not None and mem_pct > mem_threshold_pct
            mem_str = (
                f"{mem_pct:.1f}% of {mem_total_mib}MiB"
                if mem_pct is not None
                else "total unknown"
            )
            if util > util_threshold or mem_busy:
                return False, (
                    f"nvidia-smi {dev_tag}: util={util}% mem={mem_used_mib}MiB "
                    f"({mem_str})"
                )
            return True, (
                f"nvidia-smi {dev_tag}: idle (util={util}% mem={mem_used_mib}MiB {mem_str})"
            )
        except Exception as e:
            return True, f"nvidia-smi check skipped ({e})"

    return True, "no amdsmi/nvidia-smi available; skipping idle check"


def run_python_probe(code: str, *args: Any, timeout_s: float) -> Dict[str, Any]:
    """The last stdout line of *code* run in a fresh interpreter, parsed as JSON,
    or ``{"error": ...}``."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", code, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    lines = (r.stdout or "").strip().splitlines()
    if r.returncode != 0 or not lines:
        tail = (r.stderr or "").strip().splitlines()
        return {"error": tail[-1] if tail else f"probe exited with {r.returncode}"}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"error": f"unparseable probe output: {lines[-1][:200]}"}


def _rdna_idle_desc(
    probe: Mapping[str, Any], util: float, readings: list, used: int, total: int
) -> str:
    procs = probe.get("processes")
    held = (
        f"{procs} process(es) on the GPU, this one included"
        if procs is not None
        else f"process list unavailable ({probe.get('processes_error')})"
    )
    return f"util={util:g}% (max {max(readings)}%) mem={used}MiB of {total}MiB; {held}"


def check_rdna_gpu_idle(
    bdf: str,
    *,
    util_threshold: int = 5,
    mem_threshold_pct: Optional[float] = 10.0,
    samples: int = IDLE_SAMPLES_RDNA,
    interval_s: float = IDLE_INTERVAL_S,
) -> Tuple[bool, str]:
    """
    The idle check of the RDNA protocol, on the device at PCI address *bdf*:
    the median of *samples* amdsmi utilization readings, taken in a fresh
    interpreter. Busy when the median is above *util_threshold* or VRAM use is
    above *mem_threshold_pct* (``None`` on an APU, whose small VRAM carve-out
    the display uses). A check that cannot decide (amdsmi missing or failing,
    no device at *bdf*, a reading that is not a number) reports the GPU as not
    idle. The processes holding the GPU are reported, not counted: in a
    container amdsmi lists this one under its host pid.
    """
    tag = f"amdsmi PCI {bdf}, median of {samples} readings"
    probe = run_python_probe(
        _RDNA_IDLE_PROBE, bdf, samples, interval_s, timeout_s=IDLE_PROBE_TIMEOUT_S
    )
    if "error" in probe:
        return False, f"{tag}: check failed ({probe['error']}); idle state unknown"
    readings = probe.get("gfx_activity") or []
    if not readings or not all(_is_number(r) for r in readings):
        return False, f"{tag}: gfx_activity reads {readings!r}; idle state unknown"
    util = statistics.median(readings)
    vram = probe.get("vram") or {}
    used, total = _int_metric(vram.get("vram_used")), _int_metric(
        vram.get("vram_total")
    )
    mem_busy = (
        mem_threshold_pct is not None
        and total > 0
        and 100.0 * used / total > mem_threshold_pct
    )
    desc = _rdna_idle_desc(probe, util, readings, used, total)
    if util > util_threshold or mem_busy:
        return False, f"{tag}: busy ({desc})"
    return True, f"{tag}: idle ({desc})"
