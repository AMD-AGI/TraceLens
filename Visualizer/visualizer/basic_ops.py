"""Regex filters for treating modeled blocks as leaf/basic operations."""

from __future__ import annotations

import re

# Default leaf patterns: PyTorch ATen ops (avoid matching "Attention", etc.).
DEFAULT_BASIC_OP_PATTERNS: tuple[str, ...] = (
    r"(?i)aten\.",
    r"(?i)\.aten\.",
    r"(?i)^aten_",
)

# Common PyTorch leaf modules (not enabled by default; add via --basic-op-add).
COMMON_LEAF_PATTERNS: tuple[str, ...] = (
    r"(?i)^Linear$",
    r"(?i)^Embedding$",
    r"(?i)^Conv\d*d$",
    r"(?i)^Dropout$",
    r"(?i)^Identity$",
    r"(?i)^Parameter$",
)


class BasicOpFilter:
    """Match class or attribute names against basic-operation regex patterns."""

    def __init__(self, patterns: list[str | re.Pattern[str]]) -> None:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            compiled.append(pattern if isinstance(pattern, re.Pattern) else re.compile(pattern))
        self.patterns = compiled

    @classmethod
    def from_cli(
        cls,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> BasicOpFilter:
        selected = list(DEFAULT_BASIC_OP_PATTERNS)
        for pattern in remove or []:
            selected = [item for item in selected if item != pattern]
        for pattern in add or []:
            if pattern not in selected:
                selected.append(pattern)
        return cls(selected)

    @classmethod
    def for_detailed(cls) -> BasicOpFilter:
        """Leaf patterns for detailed internal diagrams (Linear, norms, etc.)."""
        patterns = list(DEFAULT_BASIC_OP_PATTERNS)
        for pattern in COMMON_LEAF_PATTERNS:
            if pattern not in patterns:
                patterns.append(pattern)
        return cls(patterns)

    def is_basic(self, *names: str) -> bool:
        for name in names:
            if not name:
                continue
            if any(pattern.search(name) for pattern in self.patterns):
                return True
        return False

    def pattern_strings(self) -> list[str]:
        return [pattern.pattern for pattern in self.patterns]
