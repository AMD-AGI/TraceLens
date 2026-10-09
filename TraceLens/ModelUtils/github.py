###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fetch modeling source from GitHub repositories (CPU-only, no weights)."""

from __future__ import annotations

import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GITHUB_HOSTS = {"github.com", "www.github.com"}
CACHE_ROOT = Path.home() / ".cache" / "tracelens" / "visualizer" / "github"

# github.com/owner/repo[/tree|blob/ref[/path]]
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/(?P<kind>tree|blob)/(?P<ref>[^/]+)(?:/(?P<subpath>.+))?)?/?$",
    re.IGNORECASE,
)

# Short form: github:owner/repo[@ref][:path]
_GITHUB_SHORT_RE = re.compile(
    r"^github:(?P<owner>[^/]+)/(?P<repo>[^/@]+)(?:@(?P<ref>[^:/]+))?(?::(?P<subpath>.+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GitHubRef:
    owner: str
    repo: str
    ref: str = "main"
    subpath: str = ""
    original_url: str = ""

    @property
    def slug(self) -> str:
        safe_ref = self.ref.replace("/", "_")
        return f"{self.owner}_{self.repo}_{safe_ref}"

    @property
    def display(self) -> str:
        base = f"github://{self.owner}/{self.repo}@{self.ref}"
        if self.subpath:
            return f"{base}/{self.subpath}"
        return base


def is_github_url(value: str) -> bool:
    text = value.strip()
    return bool(_GITHUB_RE.match(text) or _GITHUB_SHORT_RE.match(text))


def parse_github_url(url: str) -> GitHubRef:
    """Parse a GitHub web URL or `github:owner/repo@ref:path` shorthand."""
    text = url.strip()
    short = _GITHUB_SHORT_RE.match(text)
    if short:
        groups = short.groupdict()
        return GitHubRef(
            owner=groups["owner"],
            repo=groups["repo"],
            ref=groups["ref"] or "main",
            subpath=(groups["subpath"] or "").strip("/"),
            original_url=text,
        )

    match = _GITHUB_RE.match(text)
    if not match:
        raise ValueError(f"Unsupported GitHub URL: {url}")

    groups = match.groupdict()
    subpath = (groups.get("subpath") or "").strip("/")
    kind = groups.get("kind")

    if kind == "blob" and subpath.endswith(".py"):
        # Single-file URLs keep the file path; fetch_github resolves to the file.
        pass
    elif kind == "blob" and subpath:
        raise ValueError(
            f"GitHub blob URLs must point to a .py file for code inspection: {url}"
        )

    return GitHubRef(
        owner=groups["owner"],
        repo=groups["repo"],
        ref=groups["ref"] or "main",
        subpath=subpath,
        original_url=text,
    )


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "TraceLens-Visualizer/0.3"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _extract_tarball(data: bytes, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        with tarfile.open(tmp_path, "r:gz") as archive:
            archive.extractall(dest, filter="data")
    finally:
        tmp_path.unlink(missing_ok=True)

    children = [path for path in dest.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return dest


def _fetch_archive(ref: GitHubRef, cache_dir: Path) -> Path:
    if cache_dir.exists() and any(cache_dir.iterdir()):
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)

    archive_urls = [
        f"https://codeload.github.com/{ref.owner}/{ref.repo}/tar.gz/{ref.ref}",
        f"https://codeload.github.com/{ref.owner}/{ref.repo}/tar.gz/refs/heads/{ref.ref}",
    ]

    last_error: Exception | None = None
    for url in archive_urls:
        try:
            data = _download_bytes(url)
            extracted = _extract_tarball(data, cache_dir)
            return extracted
        except (urllib.error.HTTPError, urllib.error.URLError, tarfile.TarError) as exc:
            last_error = exc
            continue

    raise FileNotFoundError(
        f"Could not download GitHub archive for {ref.display}: {last_error}"
    )


def cached_file_path(
    ref: GitHubRef, subpath: str, *, cache_root: Path | None = None
) -> Path:
    """Where a file at ``subpath`` lives once fetched, mirroring the repo layout.

    A modeling module names its neighbours relatively (``from .configuration_x
    import X``), so the cache has to reproduce the directory structure they are
    named against -- flattening the path puts them in one directory under names
    no import resolves.
    """
    root = cache_root or CACHE_ROOT
    return root / ref.slug / Path(subpath)


def fetch_github_file(
    ref: GitHubRef, subpath: str, *, cache_root: Path | None = None
) -> Path | None:
    """Fetch one file from the repo at this ref, or ``None`` if it is not there.

    Best effort by design: a caller asking for a module that an import NAMES
    cannot know whether the repo spells it ``x.py`` or ``x/__init__.py``, nor
    whether it exists at this ref at all.
    """
    cache_file = cached_file_path(ref, subpath, cache_root=cache_root)
    if cache_file.is_file():
        return cache_file

    raw_url = (
        f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/"
        f"{ref.ref}/{subpath}"
    )
    try:
        data = _download_bytes(raw_url)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)
    return cache_file


# Repos whose source answers for a module the analysed model imports, as
# (ref, prefix) where *prefix* is the path under which the repo lays out its
# top-level packages (`src` for transformers). One entry per model analysed.
_PINNED_MODULE_SOURCES: list[tuple[GitHubRef, str]] = []
# Modules no registered repo publishes, so the fallback is not re-priced.
_PINNED_MODULE_MISSES: set[str] = set()


def register_pinned_module_source(ref: GitHubRef, subpath: str, module: str) -> None:
    """Record that *ref* supplies the modules around *module*, read at ITS commit.

    A model's modeling file is read at a pinned commit, but everything it
    imports -- ``vision_utils``, ``masking_utils``, ``activations``, its own
    ``configuration_*`` -- resolves through ``sys.path`` to whatever version of
    the library happens to be INSTALLED. That is two revisions of one library
    describing one model, and the helpers are where several of a graph's frames
    come from.

    *subpath* and *module* are the same file named two ways
    (``src/transformers/models/x/modeling_x.py`` and
    ``transformers.models.x.modeling_x``), which is what says where the repo
    keeps its top-level packages: strip as many trailing path segments as the
    module has dotted parts.
    """
    parts = [part for part in module.split(".") if part]
    segments = Path(subpath).parts
    if not parts or len(segments) < len(parts):
        return
    prefix = "/".join(segments[: len(segments) - len(parts)])
    entry = (ref, prefix)
    if entry not in _PINNED_MODULE_SOURCES:
        _PINNED_MODULE_SOURCES.insert(0, entry)


def pinned_module_origin(module: str) -> str | None:
    """File defining *module* in a registered pinned repo, fetching it if needed.

    Returns ``None`` when no registered repo publishes it, so the caller falls
    back to the installed library. Best effort per module: a model that imports
    something the pinned repo does not have still resolves it, just not pinned.
    """
    parts = [part for part in module.split(".") if part]
    if not parts or module in _PINNED_MODULE_MISSES:
        return None
    for ref, prefix in _PINNED_MODULE_SOURCES:
        base = f"{prefix}/" if prefix else ""
        relative = "/".join(parts)
        candidates = (f"{base}{relative}.py", f"{base}{relative}/__init__.py")
        # Everything already on disk first. A package misses the module spelling
        # and hits the package one, so trying them in order would spend a request
        # on that miss EVERY run -- the fetched file is cached but the 404 is not.
        for candidate in candidates:
            cached = cached_file_path(ref, candidate)
            if cached.is_file():
                return str(cached)
        for candidate in candidates:
            found = fetch_github_file(ref, candidate)
            if found is not None:
                return str(found)
    # A module no registered repo publishes stays absent for this analysis, so
    # the fallback to the installed library is paid for once.
    _PINNED_MODULE_MISSES.add(module)
    return None


def _fetch_single_file(ref: GitHubRef) -> Path:
    if not ref.subpath.endswith(".py"):
        raise ValueError(f"Expected a Python file path in GitHub URL: {ref.display}")

    fetched = fetch_github_file(ref, ref.subpath)
    if fetched is None:
        raise FileNotFoundError(
            f"Could not download {ref.subpath} from {ref.owner}/{ref.repo}@{ref.ref}"
        )
    return fetched


def fetch_github_source(
    ref: GitHubRef,
    *,
    cache_root: Path | None = None,
    source_policy=None,
) -> Path:
    """Download or reuse cached GitHub repo contents. Returns repo root or file path."""
    from TraceLens.ModelUtils.source_policy import SourcePolicy, get_source_policy

    policy = source_policy or get_source_policy()
    if isinstance(policy, SourcePolicy):
        policy.require_github_repo_allowed(ref.owner, ref.repo)

    root = cache_root or CACHE_ROOT
    if ref.subpath.endswith(".py") and "/modeling" in ref.subpath.lower():
        return _fetch_single_file(ref)

    repo_cache = root / ref.slug
    extracted = _fetch_archive(ref, repo_cache)
    if ref.subpath:
        target = extracted / ref.subpath
        if target.is_file():
            return target
        if target.is_dir():
            return target
        raise FileNotFoundError(
            f"Path `{ref.subpath}` not found in GitHub repo {ref.owner}/{ref.repo}@{ref.ref}"
        )
    return extracted


_SKIP_PYTHON_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
}


def python_source_priority(path: Path) -> tuple[int, int, int, str]:
    """Prefer modeling sources so analysis picks the decoder from the right file."""
    name = path.name.lower()
    modeling = 0 if name.startswith("modeling") else 1
    model_py = 0 if name in {"model.py", "models.py"} else 1
    return (modeling, model_py, len(path.parts), str(path))


def find_modeling_files(root: Path) -> list[Path]:
    """Return every ``.py`` file under ``root``, the same way ``find`` would.

    Hugging Face snapshots often keep modeling code in a nested folder under a
    name like ``inference/model.py``, so a root-level ``modeling*.py`` glob is
    not enough. ``__pycache__`` and VCS trees are skipped.
    """
    if root.is_file() and root.suffix == ".py":
        return [root.absolute()]

    found: list[Path] = []
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        if any(part in _SKIP_PYTHON_DIR_NAMES for part in path.parts):
            continue
        # Keep the snapshot path. Hugging Face stores files as symlinks into a
        # content-addressed blob store whose names have no ``.py`` suffix, and
        # ``resolve()`` would throw those names away.
        found.append(path.absolute())
    return sorted(set(found), key=python_source_priority)


def github_config_path(root: Path) -> Path | None:
    if root.is_file():
        candidate = root.parent / "config.json"
        return candidate if candidate.is_file() else None
    candidate = root / "config.json"
    return candidate if candidate.is_file() else None
