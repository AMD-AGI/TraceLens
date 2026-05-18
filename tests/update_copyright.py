#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Update or add copyright headers in source files using git history for dates."""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HOLDER = "Advanced Micro Devices, Inc."

# Map extension → (comment_open, comment_line_prefix, comment_close)
# comment_close is None for line-comment styles
COMMENT_STYLES = {
    ".py": ("#", "#", None),
    ".sh": ("#", "#", None),
    ".yaml": ("#", "#", None),
    ".yml": ("#", "#", None),
    ".c": ("/*", " *", " */"),
    ".cpp": ("/*", " *", " */"),
    ".h": ("/*", " *", " */"),
    ".hpp": ("/*", " *", " */"),
    ".js": ("/*", " *", " */"),
    ".ts": ("/*", " *", " */"),
    ".tsx": ("/*", " *", " */"),
    ".md": ("<!--", "", "-->"),
}

BANNER_LINE_LENGTH = 79

# Pattern that matches any existing AMD copyright line we might have written
_AMD_COPYRIGHT_RE = re.compile(
    r"Copyright\s+\(c\)\s+(\d{4})\s*[-–]\s*(\d{4})\s+Advanced Micro Devices",
    re.IGNORECASE,
)
_AMD_COPYRIGHT_SINGLE_RE = re.compile(
    r"Copyright\s+\(c\)\s+(\d{4})\s+Advanced Micro Devices",
    re.IGNORECASE,
)


@dataclass
class FileYears:
    created: int
    modified: int


def git_years(path: Path) -> Optional[FileYears]:
    """Return the first and last commit years for *path* via git log."""
    try:
        # Only process files currently tracked in the index
        tracked = subprocess.run(
            ["git", "ls-files", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        if not tracked.stdout.strip():
            return None

        result = subprocess.run(
            ["git", "log", "--follow", "--format=%ad", "--date=format:%Y", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    years = [int(y) for y in result.stdout.splitlines() if y.strip().isdigit()]
    if not years:
        return None
    return FileYears(created=min(years), modified=max(years))


def copyright_text(years: FileYears) -> str:
    if years.created == years.modified:
        return f"Copyright (c) {years.created} {HOLDER} All rights reserved."
    return f"Copyright (c) {years.created} - {years.modified} {HOLDER} All rights reserved."


def build_header(years: FileYears, ext: str) -> str:
    """Return a complete copyright header block for the given extension."""
    style = COMMENT_STYLES[ext]
    open_tok, prefix, close_tok = style
    copy = copyright_text(years)

    if close_tok is None:
        # Line-comment style (Python, shell, YAML)
        banner = "#" * BANNER_LINE_LENGTH
        return f"{banner}\n{prefix} {copy}\n{prefix}\n{prefix} See LICENSE for license information.\n{banner}\n\n"
    elif close_tok == "-->":
        # Markdown HTML comment - multi-line block
        return f"<!--\n{copy}\n\nSee LICENSE for license information.\n-->\n"
    else:
        # Block-comment style (C, JS, TS)
        banner = "/" + "*" * (BANNER_LINE_LENGTH - 1)
        return f"{banner}\n{prefix} {copy}\n{close_tok}\n"


# --------------------------------------------------------------------------- #
# Detection / replacement helpers
# --------------------------------------------------------------------------- #

def _find_existing_amd_copyright(lines: list[str]) -> Optional[tuple[int, int]]:
    """Return (start_line, end_line) of the existing AMD copyright block, or None.

    We look at the first 30 lines to avoid scanning the whole file.
    Returns inclusive indices into *lines*.
    """
    window = lines[:30]
    for i, line in enumerate(window):
        if not (_AMD_COPYRIGHT_RE.search(line) or _AMD_COPYRIGHT_SINGLE_RE.search(line)):
            continue

        stripped = line.strip()

        if stripped.startswith("# "):
            # Line-comment style: find enclosing ### banners
            start = i
            for k in range(i - 1, max(i - 4, -1), -1):
                if re.match(r"#{10,}", lines[k].strip()):
                    start = k
                    break
            end = i
            for k in range(i + 1, min(i + 7, len(lines))):
                s = lines[k].strip()
                if re.match(r"#{10,}", s):
                    end = k
                    # Include trailing blank line that separates header from code
                    if k + 1 < len(lines) and not lines[k + 1].strip():
                        end = k + 1
                    break
                elif s.startswith("#") or not s:
                    end = k
                else:
                    break

        elif stripped.startswith((" * ", "* ")):
            # C-style block comment
            start = i
            for k in range(i - 1, max(i - 3, -1), -1):
                if lines[k].strip().startswith("/*"):
                    start = k
                    break
            end = i
            for k in range(i + 1, min(i + 4, len(lines))):
                end = k
                if lines[k].strip() == "*/":
                    break

        elif stripped.startswith("<!--"):
            # Single-line markdown: <!-- Copyright... -->
            start = i
            end = i

        else:
            # Multi-line markdown: <!-- on previous line, --> on a later line
            start = i
            for k in range(i - 1, max(i - 3, -1), -1):
                if lines[k].strip() == "<!--":
                    start = k
                    break
            end = i
            for k in range(i + 1, min(i + 6, len(lines))):
                end = k
                if lines[k].strip() == "-->":
                    break

        return start, end
    return None


def _skip_shebang_and_encoding(lines: list[str]) -> int:
    """Return the index of the first line that is not a shebang or encoding declaration."""
    idx = 0
    for line in lines[:3]:
        if line.startswith("#!") or re.match(r"#.*coding[:=]", line):
            idx += 1
        else:
            break
    return idx


def process_file(path: Path, apply: bool) -> Optional[str]:
    """Return a human-readable description of the change, or None if no change needed."""
    ext = path.suffix.lower()
    if ext not in COMMENT_STYLES:
        return None

    years = git_years(path)
    if years is None:
        return f"SKIP (not in git): {path}"

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return None
    lines = text.splitlines(keepends=True)

    header = build_header(years, ext)
    header_lines = header.splitlines(keepends=True)

    existing = _find_existing_amd_copyright(lines)

    if existing is not None:
        start, end = existing
        old_block = "".join(lines[start : end + 1])
        if old_block == header:
            return None  # already up to date
        action = "UPDATE"
        new_lines = lines[:start] + header_lines + lines[end + 1 :]
        # Extract old year range from the existing block for a concise description
        m = _AMD_COPYRIGHT_RE.search(old_block) or _AMD_COPYRIGHT_SINGLE_RE.search(old_block)
        if m and len(m.groups()) == 2:
            old_years = f"{m.group(1)}-{m.group(2)}"
        elif m:
            old_years = m.group(1)
        else:
            old_years = "?"
        new_years = str(years.modified) if years.created == years.modified else f"{years.created}-{years.modified}"
        change_desc = f"{action}: {path}  ({old_years} → {new_years})"
    else:
        action = "ADD"
        insert_at = _skip_shebang_and_encoding(lines)
        new_lines = lines[:insert_at] + header_lines + lines[insert_at:]
        new_years = str(years.modified) if years.created == years.modified else f"{years.created}-{years.modified}"
        change_desc = f"{action}: {path}  ({new_years})"

    new_text = "".join(new_lines)

    if apply:
        path.write_text(new_text, encoding="utf-8")

    return change_desc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "TraceLens.egg-info",
    "node_modules",
    "venv",
    "env",
    ".venv",
}
_SKIP_FILES = {".gitignore", "LICENSE", "__init__.py"}


def collect_files(targets: list[str]) -> list[Path]:
    """Expand paths: files are used as-is, directories are walked recursively."""
    result = []
    for t in targets:
        p = Path(t)
        if p.is_file():
            if p.name not in _SKIP_FILES:
                result.append(p)
        elif p.is_dir():
            for candidate in p.rglob("*"):
                if not candidate.is_file():
                    continue
                if candidate.name not in _SKIP_FILES and candidate.suffix.lower() in COMMENT_STYLES:
                    if not any(part in _SKIP_DIRS for part in candidate.parts):
                        result.append(candidate)
        else:
            print(f"warning: {t} does not exist, skipping", file=sys.stderr)
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add or update AMD copyright headers using git history."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="PATH",
        help="Files or directories to process (default: current directory)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk (default is dry-run)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob pattern to exclude (can be repeated); e.g. '.venv' or '**/.venv/**'",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show files that are skipped (not in git) or already up-to-date",
    )
    args = parser.parse_args()

    files = collect_files(args.paths)

    # Apply exclusions: check whether any path component matches the pattern
    if args.exclude:
        filtered = []
        for f in files:
            parts = f.parts
            excluded = any(
                f.match(pat) or any(Path(part).match(pat) for part in parts)
                for pat in args.exclude
            )
            if not excluded:
                filtered.append(f)
        files = filtered

    if not args.apply:
        print("DRY RUN — pass --apply to write changes\n")

    changed = skipped = up_to_date = 0
    for f in files:
        result = process_file(f, apply=args.apply)
        if result is None:
            up_to_date += 1
            if args.verbose:
                print(f"OK: {f}")
        elif result.startswith("SKIP"):
            skipped += 1
            if args.verbose:
                print(result)
        else:
            print(result)
            changed += 1

    print(
        f"\nSummary: {changed} would change, {up_to_date} up-to-date, {skipped} skipped"
        if not args.apply
        else f"\nSummary: {changed} updated, {up_to_date} already correct, {skipped} skipped"
    )


if __name__ == "__main__":
    main()
