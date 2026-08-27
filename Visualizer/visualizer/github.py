"""Fetch modeling source from GitHub repositories (CPU-only, no weights)."""

from __future__ import annotations

import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    request = urllib.request.Request(url, headers={"User-Agent": "TraceLens-Visualizer/0.3"})
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


def _fetch_single_file(ref: GitHubRef) -> Path:
    if not ref.subpath.endswith(".py"):
        raise ValueError(f"Expected a Python file path in GitHub URL: {ref.display}")

    cache_file = CACHE_ROOT / ref.slug / ref.subpath.replace("/", "__")
    if cache_file.is_file():
        return cache_file

    raw_url = (
        f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/"
        f"{ref.ref}/{ref.subpath}"
    )
    data = _download_bytes(raw_url)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)
    return cache_file


def fetch_github_source(ref: GitHubRef, *, cache_root: Path | None = None) -> Path:
    """Download or reuse cached GitHub repo contents. Returns repo root or file path."""
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
