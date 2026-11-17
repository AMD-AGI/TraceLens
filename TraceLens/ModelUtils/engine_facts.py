###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which source an inference engine's run was built from.

A trace records the engine's python frames, ``file(line): function``, and a
``def`` sits on one line in one release, so the frames say which release ran.
That release's container build names the commit of every kernel library it
compiled in. Both are read from source; nothing is installed.
"""

from __future__ import annotations

import re

from TraceLens.ModelUtils.github import fetch_github_file, parse_github_url

# Engine name -> source the run is read at, once a trace has placed it.
_TRACED_SOURCES: dict[str, str] = {}


def engine_source_of(engine: str | None) -> str | None:
    """The source ``engine`` is read at: the release the trace ran, else the pin."""
    from TraceLens.ModelUtils.source import ENGINE_GITHUB_SOURCES

    if engine is None:
        return None
    return _TRACED_SOURCES.get(engine) or ENGINE_GITHUB_SOURCES.get(engine)


_FRAME = re.compile(r"/python/(sglang/srt/[\w/]+\.py)\((\d+)\): (\w+)$")
_RELEASE = re.compile(r"^(.*@v?)(\d+)\.(\d+)\.(\d+)$")
_SAMPLE_FILES = 6
_SAMPLE_FRAMES = 8
_EARLIER_RELEASES = 5


def _traced_frames(events) -> dict[str, list[tuple[int, str]]]:
    """``{file: [(line, function)]}`` of engine frames the profile recorded."""
    seen: dict[str, set[tuple[int, str]]] = {}
    for event in events:
        if event.get("cat") != "python_function":
            continue
        found = _FRAME.search(event.get("name") or "")
        if found:
            seen.setdefault(found.group(1), set()).add(
                (int(found.group(2)), found.group(3))
            )
    ranked = sorted(seen.items(), key=lambda item: (-len(item[1]), item[0]))
    return {
        path: sorted(frames)[:_SAMPLE_FRAMES] for path, frames in ranked[:_SAMPLE_FILES]
    }


def _frames_defined_at(source: str, frames: dict[str, list[tuple[int, str]]]) -> int:
    """How many recorded frames are ``def`` lines at that line in ``source``."""
    ref = parse_github_url(source)
    matched = 0
    for path, entries in frames.items():
        found = fetch_github_file(ref, f"python/{path}")
        if found is None:
            continue
        lines = found.read_text(encoding="utf-8").splitlines()
        for line, function in entries:
            if line <= len(lines) and re.match(
                rf"\s*(async\s+)?def\s+{function}\b", lines[line - 1]
            ):
                matched += 1
    return matched


def traced_engine_source(engine: str | None, events) -> str | None:
    """The release of ``engine`` whose source the trace's python frames match.

    A profile records ``file(line): function`` for every engine frame, and a
    ``def`` sits on one line in one release. The pinned release is the default.
    An earlier patch release that matches more of the sampled frames is the one
    the run used.
    """
    from TraceLens.ModelUtils.source import ENGINE_GITHUB_SOURCES

    pinned = ENGINE_GITHUB_SOURCES.get(engine or "")
    if pinned is None:
        return None
    frames = _traced_frames(events)
    release = _RELEASE.match(pinned)
    if not frames or release is None:
        return pinned
    prefix, major, minor, patch = release.groups()
    candidates = [pinned] + [
        f"{prefix}{major}.{minor}.{earlier}"
        for earlier in range(
            int(patch) - 1, max(-1, int(patch) - 1 - _EARLIER_RELEASES), -1
        )
    ]
    best, best_score = pinned, -1
    for candidate in candidates:
        score = _frames_defined_at(candidate, frames)
        if score > best_score:
            best, best_score = candidate, score
    return best


def use_traced_engine_source(engine: str | None, events) -> str | None:
    """Read ``engine`` at the release the trace ran, for the rest of the run."""
    source = traced_engine_source(engine, events)
    if engine is not None and source is not None:
        _TRACED_SOURCES[engine] = source
    return source


def dockerfile_pins(text: str, variable: str) -> dict[str, str]:
    """``{stage: value}`` of ``ENV <variable>="value"`` for each ``FROM .. AS stage``."""
    pins: dict[str, str] = {}
    stage = None
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 4 and words[0].upper() == "FROM" and words[-2].upper() == "AS":
            stage = words[-1]
        elif words and words[0].upper() == "ENV" and stage is not None:
            key, _, value = " ".join(words[1:]).partition("=")
            if key == variable and value:
                pins[stage] = value.strip().strip('"').strip("'")
    return pins


def kernel_library_commit(
    engine: str, variable: str, device_prefix: str = "gfx9"
) -> str | None:
    """The commit of a kernel library the engine's own container build pins.

    The engine's ROCm Dockerfile sets ``<variable>`` per device stage. The
    commit the stages of this device family agree on is the one the engine's
    kernels were built against.
    """
    engine_source = engine_source_of(engine)
    if engine_source is None:
        return None
    ref = parse_github_url(engine_source)
    found = fetch_github_file(ref, "docker/rocm.Dockerfile")
    if found is None:
        return None
    pins = dockerfile_pins(found.read_text(encoding="utf-8"), variable)
    family = {value for stage, value in pins.items() if stage.startswith(device_prefix)}
    return family.pop() if len(family) == 1 else None
