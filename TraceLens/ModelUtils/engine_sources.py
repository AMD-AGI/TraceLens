###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The source a served run executed: the engine, its libraries, its environment.

Everything is read from checkouts on disk, fetched once into a cache:

- the engine at the release the run used;
- each kernel library at the commit the engine's own container build pins
  (``<LIB>_REPO`` / ``<LIB>_COMMIT`` in its Dockerfile), unless overridden;
- the versions the engine's requirements pin for its other Python
  dependencies, which stay stand-ins;
- the container's environment (``ENV`` lines of the build stage for this GPU)
  as the run's default environment variables.

A dependency is fetched the first time the interpreted source imports it.
Nothing here names a model, and no fetched code runs.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from TraceLens.ModelUtils.github import GitHubRef, fetch_github_source

CHECKOUTS = Path.home() / ".cache" / "tracelens" / "checkouts"

# Where each engine's source lives. An engine, not a model.
# A source ref that is a commit, not a release version.
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
ENGINE_REPOSITORIES = {
    "sglang": ("sgl-project", "sglang"),
    "vllm": ("vllm-project", "vllm"),
}

_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".github",
        "test",
        "tests",
        "benchmark",
        "benchmarks",
        "docs",
        "doc",
        "examples",
        "example",
        "build",
        "dist",
        "__pycache__",
        "node_modules",
    }
)


def package_roots(checkout: Path, depth: int = 7) -> dict[str, Path]:
    """Top-level Python packages in ``checkout``: name -> directory holding it.

    A top-level package is a directory with an ``__init__.py`` whose parent
    has none. When a name appears twice the shallowest wins.
    """
    roots: dict[str, Path] = {}
    found_depth: dict[str, int] = {}

    def walk(directory: Path, level: int):
        if level > depth:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name in _SKIPPED_DIRECTORIES:
                continue
            if entry.name.startswith("."):
                continue
            if (entry / "__init__.py").is_file():
                if entry.name not in found_depth or level < found_depth[entry.name]:
                    roots[entry.name] = directory
                    found_depth[entry.name] = level
                # Everything inside belongs to this package.
                continue
            walk(entry, level + 1)

    walk(Path(checkout), 0)
    return roots


_PACKAGE_DIR = re.compile(
    r"package_dir\s*=\s*\{\s*[\"'][\"']\s*:\s*[\"']([^\"']+)[\"']"
)
_FIND_WHERE = re.compile(
    r"\[tool\.setuptools\.packages\.find\][^\[]*?where\s*=\s*\[?\s*[\"']([^\"']+)[\"']",
    re.DOTALL,
)


def build_package_roots(tree: Path) -> dict[str, Path]:
    """Packages that build scripts install from their own source directory.

    ``setup(package_dir={"": "python"})`` and
    ``[tool.setuptools.packages.find] where = ["python"]`` name the directory
    the installed top-level packages come from, which may sit anywhere in the
    tree (even inside another package's directory).
    """
    roots: dict[str, Path] = {}
    scripts = [*Path(tree).rglob("setup*.py"), *Path(tree).rglob("pyproject*.toml")]
    for script in sorted(scripts, key=lambda path: len(path.parts)):
        if any(part in _SKIPPED_DIRECTORIES for part in script.relative_to(tree).parts):
            continue
        try:
            text = script.read_text(errors="ignore")
        except OSError:
            continue
        for relative in _PACKAGE_DIR.findall(text) + _FIND_WHERE.findall(text):
            directory = (script.parent / relative).resolve()
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.is_dir() and (entry / "__init__.py").is_file():
                    roots.setdefault(entry.name, directory)
    return roots


# ------------------------------------------------------------------ Docker


@dataclass
class DockerStage:
    name: str
    base: str
    env: dict[str, str] = field(default_factory=dict)
    args: dict[str, str] = field(default_factory=dict)
    # RUN commands, substituted, of this stage and the stages it builds on.
    runs: list[str] = field(default_factory=list)


# Innermost first: a default may itself be a substitution.
_SUBSTITUTION = re.compile(r"\$\{(\w+)(?::-([^{}]*))?\}|\$(\w+)")


def _substitute(text: str, scope: dict[str, str]) -> str:
    def replace(match):
        name = match.group(1) or match.group(3)
        default = match.group(2)
        value = scope.get(name)
        if value in (None, "") and default is not None:
            return default
        return value or ""

    for _ in range(8):
        substituted = _SUBSTITUTION.sub(replace, text)
        if substituted == text:
            break
        text = substituted
    return text


def _assignments(text: str) -> list[tuple[str, str]]:
    """``KEY=VALUE`` pairs of an ENV/ARG instruction (also ``ENV KEY VALUE``)."""
    if "=" not in text.split(maxsplit=1)[0]:
        key, _, value = text.partition(" ")
        if "=" not in key:
            return [(key.strip(), value.strip().strip('"').strip("'"))]
    pairs = []
    for item in shlex.split(text, posix=True):
        key, sep, value = item.partition("=")
        pairs.append((key, value) if sep else (key, ""))
    return pairs


# The search path a GPU vendor's base image starts with, by toolkit.
BASE_IMAGE_PATHS = {
    "rocm": "/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "cuda": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}


def docker_stages(
    text: str,
    build_args: dict[str, str] | None = None,
    base_environment: dict[str, str] | None = None,
) -> list[DockerStage]:
    """The stages of a Dockerfile, each with the ENV and ARG values it ends with.

    A stage built ``FROM`` an earlier stage starts from that stage's values;
    one built from an external image starts from ``base_environment`` (its
    ``PATH``, say). ``build_args`` override ARG defaults, as ``--build-arg``
    does.
    """
    build_args = dict(build_args or {})
    lines = re.sub(r"\\\s*\n", " ", text).splitlines()
    global_args: dict[str, str] = {}
    stages: list[DockerStage] = []
    current: DockerStage | None = None
    heredoc: str | None = None
    for raw in lines:
        line = raw.strip()
        if heredoc is not None:
            # The body of a RUN heredoc is a script, not instructions.
            if line == heredoc:
                heredoc = None
            continue
        if not line or line.startswith("#"):
            continue
        marker = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if marker and line.split(maxsplit=1)[0] in ("RUN", "COPY"):
            heredoc = marker.group(1)
        instruction, _, rest = line.partition(" ")
        if instruction != instruction.upper():
            continue
        scope = {
            **global_args,
            **(current.args if current else {}),
            **(current.env if current else {}),
        }
        if instruction == "FROM":
            words = rest.split()
            base = _substitute(words[0], {**global_args, **build_args})
            name = words[2] if len(words) >= 3 and words[1].upper() == "AS" else ""
            parent = next((stage for stage in stages if stage.name == base), None)
            current = DockerStage(
                name=name,
                base=base,
                env=dict(parent.env) if parent else dict(base_environment or {}),
                args=dict(parent.args) if parent else {},
                runs=list(parent.runs) if parent else [],
            )
            stages.append(current)
        elif instruction == "ARG":
            for key, value in _assignments(rest):
                value = build_args.get(key, _substitute(value, scope))
                if current is None:
                    global_args[key] = value
                else:
                    current.args[key] = value or global_args.get(key, "")
        elif instruction == "ENV" and current is not None:
            for key, value in _assignments(rest):
                current.env[key] = _substitute(value, scope)
        elif instruction == "RUN" and current is not None:
            current.runs.append(_substitute(rest, scope))
    return stages


_PINNED_INSTALL = re.compile(r"(?<![\w.\-])([A-Za-z][\w.\-]*)==([\w.+\-]+)")


def installed_versions(stage: DockerStage) -> dict[str, str]:
    """``{distribution: version}`` the stage's ``pip install`` commands pin.

    Later installs win, as they replace earlier ones. A version keeps its
    local label (``+rocm7.2``), as the installed distribution reports it.
    """
    found: dict[str, str] = {}
    for command in stage.runs:
        for part in re.split(r"&&|;|\|\|", command):
            if "pip" not in part or "install" not in part:
                continue
            for name, version in _PINNED_INSTALL.findall(part):
                found[_normalize(name)] = version.strip(".")
    return found


def run_stage(stages: list[DockerStage], gpu_target: str) -> DockerStage | None:
    """The final stage, which the run's image is."""
    return stages[-1] if stages else None


def stage_environment(stage: DockerStage) -> dict[str, str]:
    return dict(stage.env)


def library_pins(stage: DockerStage) -> dict[str, tuple[str, str]]:
    """``{library: (repository URL, ref)}`` the stage pins.

    A pin is a ``<NAME>_REPO`` URL with a ``<NAME>_COMMIT``, ``<NAME>_BRANCH``
    or ``<NAME>_COMMIT_DEFAULT`` beside it, as ARG or ENV.
    """
    values = {**stage.args, **stage.env}
    pins = {}
    for key, url in values.items():
        if not key.endswith("_REPO") or "github.com" not in url:
            continue
        name = key[: -len("_REPO")]
        ref = next(
            (
                values[candidate]
                for candidate in (
                    f"{name}_COMMIT",
                    f"{name}_COMMIT_DEFAULT",
                    f"{name}_BRANCH",
                )
                if values.get(candidate)
            ),
            None,
        )
        if ref:
            pins[name.lower()] = (url, ref)
    return pins


# --------------------------------------------------------------- releases


def release_tags(owner: str, repo: str) -> list[str]:
    """The release tags of a GitHub repository, newest first (no pre-releases)."""
    from packaging.version import InvalidVersion, Version

    listing = subprocess.run(  # noqa: S603
        [
            "git",
            "ls-remote",
            "--tags",
            f"https://github.com/{owner}/{repo}",
        ],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    versions = {}
    for line in listing.stdout.splitlines():
        tag = line.rpartition("refs/tags/")[2].removesuffix("^{}")
        try:
            version = Version(tag.lstrip("v"))
        except InvalidVersion:
            continue
        if not (version.is_prerelease or version.is_devrelease):
            versions[tag] = version
    return sorted(versions, key=versions.__getitem__, reverse=True)


def newest_release(owner: str, repo: str) -> str | None:
    """The newest release tag of a GitHub repository."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
        headers={"User-Agent": "TraceLens", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=60
        ) as response:  # noqa: S310 - fixed host
            return json.load(response).get("tag_name")
    except OSError:
        pass
    # The API limits anonymous calls; the tags themselves are always listed.
    tags = release_tags(owner, repo)
    return tags[0] if tags else None


def _repository(url: str) -> tuple[str, str] | None:
    match = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    return (match.group(1), match.group(2)) if match else None


def checkout(owner: str, repo: str, ref: str, declared_by_engine: bool = False) -> Path:
    """The source tree of ``owner/repo`` at ``ref``, fetched into the cache.

    The active source policy decides which repositories may be fetched. A
    repository the (allowed) engine's own build pins is part of that engine's
    declared build, so ``declared_by_engine`` admits it for this fetch.
    """
    from dataclasses import replace

    from TraceLens.ModelUtils.source_policy import get_source_policy

    policy = get_source_policy()
    if declared_by_engine:
        policy = replace(
            policy,
            extra_allowed_github_repos=policy.extra_allowed_github_repos
            | {(owner, repo)},
        )
    tree = fetch_github_source(
        GitHubRef(owner=owner, repo=repo, ref=ref),
        cache_root=CHECKOUTS,
        source_policy=policy,
    )
    # An archive unpacks into one top-level directory.
    entries = [
        entry for entry in tree.iterdir() if not entry.name.startswith(("_", "."))
    ]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return tree


# ----------------------------------------------------------- dependencies


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


_PLATFORM_WORDS = {"rocm": ("hip", "rocm"), "cu": ("cuda", "cu1", "nvidia")}
_OTHER_PLATFORMS = (
    "cpu",
    "npu",
    "xpu",
    "musa",
    "hpu",
    "mps",
    "tpu",
    "cuda",
    "rocm",
    "hip",
)


def _platform_requirements(pyproject: Path, platform: str) -> list[str] | None:
    """Requirement strings a ``platform`` install of ``pyproject`` pulls in.

    The base dependencies, plus every extra named for the platform and the
    extras those reference (``pkg[group]``). Extras for other platforms are
    not installed. None when the file is not a project definition.
    """
    import tomllib

    try:
        data = tomllib.loads(pyproject.read_text(errors="ignore"))
    except (OSError, ValueError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    groups = project.get("optional-dependencies") or {}
    words = _PLATFORM_WORDS.get(platform, (platform,))
    selected = [name for name in groups if any(word in name.lower() for word in words)]
    requirements = list(project.get("dependencies") or [])
    seen: set[str] = set()
    while selected:
        group = selected.pop()
        if group in seen:
            continue
        seen.add(group)
        for item in groups.get(group, []):
            reference = re.match(r"^[\w.-]+\[([\w,\s-]+)\]$", item.strip())
            if reference:
                selected.extend(part.strip() for part in reference.group(1).split(","))
            else:
                requirements.append(item)
    return requirements


def installed_project_files(dockerfile_text: str) -> set[tuple[str, str]]:
    """``(source, target)`` project-file names a build copies one over the other.

    A build that runs ``cp dir/pyproject_x.toml dir/pyproject.toml`` installs
    the variant, so the variant's dependencies are the ones the image has.
    """
    return {
        (Path(source).name, Path(target).name)
        for source, target in re.findall(
            r"\b(?:cp|mv)\s+(?:-\w+\s+)*(\S*pyproject[\w.-]*\.toml)\s+(\S*pyproject\.toml)",
            dockerfile_text,
        )
    }


def requirement_versions(
    checkouts: list[Path], platform: str = "rocm", replaced: set | None = None
) -> dict[str, str | None]:
    """``{distribution: pinned version or None}`` a ``platform`` install declares.

    Project files for another platform (``pyproject_npu.toml``) are not
    installed, nor are extras for one. ``replaced`` maps a project file to
    the variant the build copies over it. A pin with a local label for
    another platform (``1.0+musa``) does not apply either.
    """
    found: dict[str, str | None] = {}
    words = _PLATFORM_WORDS.get(platform, (platform,))
    replaced = set(replaced or ())
    for root in checkouts:
        for path in Path(root).rglob("pyproject*.toml"):
            if any(part in _SKIPPED_DIRECTORIES for part in path.parts):
                continue
            if any(
                path.name == target and (path.parent / source).is_file()
                for source, target in replaced
            ):
                # The build installs the variant beside it in its place.
                continue
            stem = path.stem.lower()
            if any(other in stem for other in _OTHER_PLATFORMS if other not in words):
                continue
            requirements = _platform_requirements(path, platform)
            for line in requirements or ():
                match = re.match(
                    r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*(?:\[[^\]]*\])?\s*(?:==\s*([A-Za-z0-9_.+\-]+))?",
                    line,
                )
                if not match:
                    continue
                key, version = _normalize(match.group(1)), match.group(2)
                if (
                    version
                    and "+" in version
                    and platform not in version.split("+", 1)[1]
                ):
                    continue
                if version or key not in found:
                    found[key] = version
    if found:
        return found
    return _requirement_lines(checkouts, platform)


def _requirement_lines(checkouts: list[Path], platform: str) -> dict[str, str | None]:
    """Requirements from ``requirements*.txt`` and loose pyproject lines."""
    found: dict[str, str | None] = {}
    pattern = re.compile(
        r"^\s*\"?([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*(?:\[[^\]]*\])?\s*(==\s*([A-Za-z0-9_.+\-]+))?"
    )
    for root in checkouts:
        for path in [
            *Path(root).rglob("requirements*.txt"),
            *Path(root).rglob("pyproject*.toml"),
        ]:
            if any(part in _SKIPPED_DIRECTORIES for part in path.parts):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip().strip(",")
                if not line or line.startswith(("#", "[", "-")):
                    continue
                match = pattern.match(line.split("#")[0])
                if not match:
                    continue
                name, version = match.group(1), match.group(3)
                if (
                    version
                    and "+" in version
                    and platform not in version.split("+", 1)[1]
                ):
                    # A build for another platform (``+musa``, ``+cpu``).
                    continue
                key = _normalize(name)
                if version or key not in found:
                    found[key] = version
    return found


# ------------------------------------------------------------------ setup


@dataclass
class EngineSources:
    """The source trees one traced run executed."""

    engine: str
    release: str
    engine_checkout: Path
    environment: dict[str, str] = field(default_factory=dict)
    pins: dict[str, tuple[str, str]] = field(default_factory=dict)
    roots: dict[str, Path] = field(default_factory=dict)
    checkouts: list[Path] = field(default_factory=list)
    requirements: dict[str, str | None] = field(default_factory=dict)
    # Distributions the image's own ``pip install`` commands pin.
    installed: dict[str, str] = field(default_factory=dict)
    library_overrides: dict[str, str] = field(default_factory=dict)
    _attempted: set[str] = field(default_factory=set)

    def library_checkout(self, name: str) -> Path | None:
        pin = self.pins.get(name)
        if pin is None:
            return None
        repository = _repository(pin[0])
        if repository is None:
            return None
        ref = self.library_overrides.get(name, pin[1])
        return checkout(*repository, ref, declared_by_engine=True)

    def resolve(self, top: str) -> Path | None:
        """The directory holding top-level package ``top``, fetched if needed."""
        if top in self.roots:
            return self.roots[top]
        if top in self._attempted:
            return None
        self._attempted.add(top)
        # Libraries the engine's build pins are checked out, and what those
        # libraries pin in turn; any other dependency stays a stand-in.
        library = self.library_checkout(top)
        if library is None:
            library = self._library_requirement(top)
        if library is not None:
            self.checkouts.append(library)
            for name, root in {
                **build_package_roots(library),
                **package_roots(library),
            }.items():
                self.roots.setdefault(name, root)
        return self.roots.get(top)

    def _library_requirement(self, top: str) -> Path | None:
        """``top`` at the version a fetched library pins it to.

        Its repository is the one the fetched sources link to under the
        package's own name.
        """
        from TraceLens.ModelUtils.source_interpreter import requirement_pins

        libraries = [tree for tree in self.checkouts if tree != self.engine_checkout]
        version = self.library_overrides.get(top) or requirement_pins(libraries).get(
            _normalize(top)
        )
        if version is None:
            return None
        repository = linked_repository(top, [self.engine_checkout, *libraries])
        if repository is None:
            return None
        for ref in (f"v{version}", version):
            try:
                return checkout(*repository, ref, declared_by_engine=True)
            except Exception:  # noqa: BLE001 - try the next spelling of the tag
                continue
        return None


_GITHUB_LINK = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)")


def linked_repository(package: str, trees: list[Path]) -> tuple[str, str] | None:
    """The one GitHub repository named like ``package`` the ``trees`` link to."""
    wanted = _normalize(package)
    found: set[tuple[str, str]] = set()
    for tree in trees:
        for path in Path(tree).rglob("*"):
            if (
                path.suffix not in {".py", ".md", ".txt", ".toml", ".cfg", ".sh", ""}
                or not path.is_file()
            ):
                continue
            if any(part in _SKIPPED_DIRECTORIES for part in path.parts):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if package.lower() not in text.lower():
                continue
            for owner, repo in _GITHUB_LINK.findall(text):
                repo = repo.removesuffix(".git")
                if _normalize(repo) == wanted:
                    found.add((owner, repo))
    return found.pop() if len(found) == 1 else None


def image_stages(
    dockerfile: Path, gpu_target: str, build_args=None
) -> list[DockerStage]:
    """The stages that make the run's image, base image first.

    An image built on the repository's own base image (``Dockerfile.x`` on
    what ``Dockerfile.x_base`` builds) has that image's libraries and
    environment under its own.
    """
    toolkit = "rocm" if gpu_target.startswith("gfx") else "cuda"
    base = dockerfile.with_name(f"{dockerfile.name}_base")
    found = []
    for built in ([base] if base.is_file() else []) + [dockerfile]:
        stages = docker_stages(
            built.read_text(errors="ignore"),
            build_args,
            {"PATH": BASE_IMAGE_PATHS[toolkit]},
        )
        stage = run_stage(stages, gpu_target)
        if stage is not None:
            found.append(stage)
    return found


def _dockerfile(tree: Path, gpu_target: str) -> Path | None:
    """The engine's Dockerfile that builds an image for ``gpu_target``.

    The one with a build stage named for the target wins, then one that
    mentions it; a ROCm target only considers ROCm Dockerfiles.
    """
    files = sorted(
        path
        for path in (tree / "docker").glob("*")
        if path.is_file() and "ockerfile" in path.name
    )
    amd = gpu_target.startswith("gfx")
    files = [path for path in files if ("rocm" in path.name.lower()) == amd] or files
    # A base image's Dockerfile is read with the one built on it.
    files = [path for path in files if not path.name.endswith("_base")] or files

    def score(path: Path) -> tuple[int, int]:
        text = path.read_text(errors="ignore")
        # A trace may give only the device family (``gfx95``): its stage completes it.
        named = bool(
            re.search(rf"\bAS\s+{re.escape(gpu_target)}\w*\s*$", text, re.MULTILINE)
        )
        return (named, gpu_target in text)

    return max(files, key=score, default=None)


def engine_sources(
    engine: str,
    gpu_target: str,
    release: str | None = None,
    library_overrides: dict[str, str] | None = None,
    build_args: dict[str, str] | None = None,
) -> EngineSources:
    """Fetch ``engine`` at ``release`` (newest when None) and read its build."""
    owner, repo = ENGINE_REPOSITORIES[engine]
    release = release or newest_release(owner, repo)
    if release is None:
        raise RuntimeError(f"cannot determine the newest {engine} release")
    if (
        not release.startswith("v")
        and re.match(r"\d", release)
        and not _COMMIT.fullmatch(release)
    ):
        release = f"v{release}"
    tree = checkout(owner, repo, release)
    sources = EngineSources(
        engine, release, tree, library_overrides=dict(library_overrides or {})
    )
    dockerfile = _dockerfile(tree, gpu_target)
    replaced = (
        installed_project_files(dockerfile.read_text(errors="ignore"))
        if dockerfile is not None
        else set()
    )
    if dockerfile is not None:
        for stage in image_stages(dockerfile, gpu_target, build_args):
            sources.environment.update(stage_environment(stage))
            sources.pins.update(library_pins(stage))
            sources.installed.update(installed_versions(stage))
    sources.checkouts.append(tree)
    sources.roots.update(package_roots(tree))
    for name, root in build_package_roots(tree).items():
        sources.roots.setdefault(name, root)
    sources.requirements = requirement_versions(
        [tree], "rocm" if gpu_target.startswith("gfx") else "cu", replaced
    )
    return sources
