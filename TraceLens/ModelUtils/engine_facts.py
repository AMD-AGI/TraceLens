###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which release of an inference engine a trace's run executed.

A profile records the engine's python frames as ``file(line): function``,
where the line falls inside that function's body (its first decorator, its
``def``, or the line running). A release whose source places every recorded
frame inside a function of that name is one the run could have used; the
release that agrees with the most frames is the one it did. Nothing is
installed: each release's files are read from source.
"""

from __future__ import annotations

import ast
import functools
import re

from TraceLens.ModelUtils.engine_sources import ENGINE_REPOSITORIES, release_tags
from TraceLens.ModelUtils.github import GitHubRef, fetch_github_file

# The most frame-dense engine files whose frames are compared.
SAMPLED_FILES = 8
# How many of the newest releases are candidates.
CANDIDATE_RELEASES = 12

_ANONYMOUS = {
    ast.Lambda: "<lambda>",
    ast.GeneratorExp: "<genexpr>",
    ast.ListComp: "<listcomp>",
    ast.SetComp: "<setcomp>",
    ast.DictComp: "<dictcomp>",
}


def recorded_engine_version(metadata, engine: str) -> str | None:
    """The engine version the trace's own metadata states, as a source ref.

    A development build names its commit (``0.30.1rc1.dev709+g21d93d0d8``):
    that commit is the source it ran. A release names its version.
    """
    value = (metadata or {}).get(f"{engine}_version")
    if not isinstance(value, str) or not value:
        return None
    commit = re.search(r"\+g([0-9a-f]{7,40})\b", value)
    return commit.group(1) if commit else value


def traced_frames(events, package: str) -> dict[str, set[tuple[int, str]]]:
    """``{path within the package's parent: {(line, function)}}`` of its frames."""
    pattern = re.compile(r"^(.+\.py)\((\d+)\): (\S+)$")
    marker = f"/{package}/"
    frames: dict[str, set[tuple[int, str]]] = {}
    for event in events:
        if event.get("cat") != "python_function":
            continue
        found = pattern.match(event.get("name") or "")
        if found is None:
            continue
        path = "/" + found.group(1)
        # The package starts at its last mention: an install root may share its name.
        start = path.rfind(marker)
        if start >= 0:
            frames.setdefault(path[start + 1 :], set()).add(
                (int(found.group(2)), found.group(3))
            )
    return frames


def function_spans(text: str) -> list[tuple[int, int, str]]:
    """``(first line, last line, name)`` of every code object in ``text``.

    A function's code starts at its first decorator; comprehensions and
    lambdas carry the names the interpreter gives them.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            spans.append((first, node.end_lineno, node.name))
        elif type(node) in _ANONYMOUS:
            spans.append((node.lineno, node.end_lineno, _ANONYMOUS[type(node)]))
    return spans


def agreeing_frames(text: str, frames: set[tuple[int, str]]) -> int:
    """How many ``frames`` fall inside a function of their name in ``text``."""
    spans = function_spans(text)
    return sum(
        1
        for line, function in frames
        if function == "<module>"
        or any(
            first <= line <= last and name == function for first, last, name in spans
        )
    )


@functools.lru_cache(maxsize=None)
def _release_file(owner: str, repo: str, release: str, path: str) -> str | None:
    """``path`` (relative to the package's parent) in ``release``, where it lies."""
    ref = GitHubRef(owner=owner, repo=repo, ref=release)
    # A package sits at the repository root or under ``python/``.
    for candidate in (f"python/{path}", path):
        found = fetch_github_file(ref, candidate)
        if found is not None:
            return found.read_text(encoding="utf-8", errors="ignore")
    return None


def traced_engine_release(
    engine: str | None, events, releases: list[str] | None = None
) -> str | None:
    """The release of ``engine`` whose source agrees with the trace's frames.

    ``releases`` are the candidates, newest first (default: the engine's
    newest :data:`CANDIDATE_RELEASES`); a tie goes to the newer. None when
    the trace records no frames of the engine.
    """
    if engine not in ENGINE_REPOSITORIES:
        return None
    frames = traced_frames(events, engine)
    if not frames:
        return None
    owner, repo = ENGINE_REPOSITORIES[engine]
    sampled = sorted(frames.items(), key=lambda item: (-len(item[1]), item[0]))[
        :SAMPLED_FILES
    ]
    candidates = (
        releases
        if releases is not None
        else release_tags(owner, repo)[:CANDIDATE_RELEASES]
    )
    best, best_score = None, 0
    for release in candidates:
        score = 0
        for path, entries in sampled:
            text = _release_file(owner, repo, release, path)
            if text is not None:
                score += agreeing_frames(text, entries)
        if score > best_score:
            best, best_score = release, score
    return best
