###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Policy for which remote repositories may supply modeling or kernel source."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Trusted upstream modeling code and kernel dependencies used by default introspection.
DEFAULT_ALLOWED_GITHUB_REPOS: frozenset[tuple[str, str]] = frozenset(
    {
        ("huggingface", "transformers"),
        ("fla-org", "flash-linear-attention"),
    }
)

_current_policy: SourcePolicy | None = None


def _parse_repo_spec(spec: str) -> tuple[str, str]:
    text = spec.strip()
    if text.startswith("github://"):
        text = text.removeprefix("github://")
    if "@" in text:
        text = text.split("@", 1)[0]
    if ":" in text and "/" not in text.split(":", 1)[0]:
        text = text.split(":", 1)[-1]
    if "/" not in text:
        raise ValueError(f"Invalid GitHub repo spec {spec!r}; expected owner/repo")
    owner, repo = text.split("/", 1)
    owner = owner.strip()
    repo = repo.strip().removesuffix(".git")
    if not owner or not repo:
        raise ValueError(f"Invalid GitHub repo spec {spec!r}; expected owner/repo")
    return owner, repo


@dataclass
class SourcePolicy:
    """Controls which GitHub repositories TraceLens may fetch for introspection.

    Hugging Face checkpoint trees and local ``--code-path`` files are always
    allowed. Only third-party GitHub fetches require whitelisting.
    """

    allowed_github_repos: frozenset[tuple[str, str]] = DEFAULT_ALLOWED_GITHUB_REPOS
    extra_allowed_github_repos: frozenset[tuple[str, str]] = field(
        default_factory=frozenset
    )

    @classmethod
    def from_env_and_cli(cls, allow_repos: list[str] | None = None) -> SourcePolicy:
        extra: set[tuple[str, str]] = set()
        env_value = os.environ.get("TRACELENS_ALLOWED_GITHUB_REPOS", "")
        for token in env_value.split(","):
            token = token.strip()
            if not token:
                continue
            extra.add(_parse_repo_spec(token))
        for token in allow_repos or []:
            extra.add(_parse_repo_spec(token))
        return cls(extra_allowed_github_repos=frozenset(extra))

    def allowed_repo_keys(self) -> frozenset[tuple[str, str]]:
        base = {
            (owner.lower(), repo.lower()) for owner, repo in self.allowed_github_repos
        }
        extra = {
            (owner.lower(), repo.lower())
            for owner, repo in self.extra_allowed_github_repos
        }
        return base | extra

    def is_github_repo_allowed(self, owner: str, repo: str) -> bool:
        return (owner.lower(), repo.lower()) in self.allowed_repo_keys()

    def require_github_repo_allowed(self, owner: str, repo: str) -> None:
        if self.is_github_repo_allowed(owner, repo):
            return
        allowed = ", ".join(
            f"{owner}/{repo}" for owner, repo in sorted(self.allowed_repo_keys())
        )
        raise PermissionError(
            f"GitHub repository {owner}/{repo} is not whitelisted for source introspection. "
            f"Allowed repositories: {allowed}. "
            "Add one with --allow-repo owner/repo or TRACELENS_ALLOWED_GITHUB_REPOS."
        )


def set_source_policy(policy: SourcePolicy | None) -> None:
    """Install the active source policy for this process."""
    global _current_policy
    _current_policy = policy


def get_source_policy() -> SourcePolicy:
    """Return the active source policy, creating a default when unset."""
    global _current_policy
    if _current_policy is None:
        _current_policy = SourcePolicy.from_env_and_cli()
    return _current_policy
