###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import re
from pathlib import Path

# Copyright line must use one of these year forms (longer/more specific first).
_COPYRIGHT_YEAR_RE = r"(?:2024 - 2025|2024 - 2026|2025-2026|2024|2025|2026)"

_PYTHON_YAML_HEADER_RE = re.compile(
    r"^###############################################################################\n"
    rf"# Copyright \(c\) {_COPYRIGHT_YEAR_RE} Advanced Micro Devices, Inc\. "
    r"All rights reserved\.\n"
    r"#\n"
    r"# See LICENSE for license information\.\n"
    r"###############################################################################\n"
    r"\n",
    re.MULTILINE,
)

_MARKDOWN_HEADER_RE = re.compile(
    r"^<!--\n"
    rf"Copyright \(c\) {_COPYRIGHT_YEAR_RE} Advanced Micro Devices, Inc\. "
    r"All rights reserved\.\n"
    r"\n"
    r"See LICENSE for license information\.\n"
    r"-->\n",
    re.MULTILINE,
)

_NOTEBOOK_COPYRIGHT_CELL_RE = re.compile(
    r"^<!--\n"
    rf"Copyright \(c\) {_COPYRIGHT_YEAR_RE} Advanced Micro Devices, Inc\. "
    r"All rights reserved\.\n"
    r"\n"
    r"See LICENSE for license information\.\n"
    r"-->\s*",
    re.MULTILINE,
)


def _strip_shebang(content: str) -> str:
    if content.startswith("#!"):
        parts = content.split("\n", 1)
        return parts[1] if len(parts) > 1 else ""
    return content


def _matches_python_copyright_header(content: str) -> bool:
    return bool(_PYTHON_YAML_HEADER_RE.match(_strip_shebang(content)))


def _matches_markdown_copyright_header(content: str) -> bool:
    return bool(_MARKDOWN_HEADER_RE.match(content))


def _matches_yaml_copyright_header(content: str) -> bool:
    return bool(_PYTHON_YAML_HEADER_RE.match(content))


def _matches_notebook_copyright_cell(source: list) -> bool:
    if not source:
        return False
    return bool(_NOTEBOOK_COPYRIGHT_CELL_RE.match("".join(source)))


def test_python_files_have_valid_copyright():
    """Test that all Python files have a valid AMD copyright header (allowed year forms)."""
    root_path = Path(__file__).parent.parent
    skip_dirs = {
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
    skip_files = {".gitignore", "LICENSE"}

    missing_copyright = []
    wrong_format = []

    for filepath in root_path.rglob("*.py"):
        # Skip excluded directories and files
        if any(skip_dir in filepath.parts for skip_dir in skip_dirs):
            continue
        if filepath.name in skip_files:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if filepath.name == "__init__.py" and len(content.splitlines()) <= 1:
                continue
            if _matches_python_copyright_header(content):
                continue
            elif "Copyright (c)" in content[:500]:
                wrong_format.append(str(filepath.relative_to(root_path)))
            else:
                missing_copyright.append(str(filepath.relative_to(root_path)))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    error_msgs = []
    if missing_copyright:
        error_msgs.append("\nThe following Python files are missing copyright headers:")
        error_msgs.extend(f"  - {f}" for f in sorted(missing_copyright))

    if wrong_format:
        error_msgs.append(
            "\nThe following Python files have incorrect copyright format:"
        )
        error_msgs.extend(f"  - {f}" for f in sorted(wrong_format))

    if error_msgs:
        error_msgs.append("\nRun 'python add_copyright_headers.py' to fix.")
        assert False, "\n".join(error_msgs)


def test_markdown_files_have_valid_copyright():
    """Test that all Markdown files have a valid AMD copyright header (allowed year forms)."""
    root_path = Path(__file__).parent.parent
    skip_dirs = {
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
    skip_files = {".gitignore", "LICENSE"}

    missing_copyright = []
    wrong_format = []

    for filepath in root_path.rglob("*.md"):
        if any(skip_dir in filepath.parts for skip_dir in skip_dirs):
            continue
        if filepath.name in skip_files:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if _matches_markdown_copyright_header(content):
                continue
            elif "Copyright (c)" in content[:500]:
                wrong_format.append(str(filepath.relative_to(root_path)))
            else:
                missing_copyright.append(str(filepath.relative_to(root_path)))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    error_msgs = []
    if missing_copyright:
        error_msgs.append(
            "\nThe following Markdown files are missing copyright headers:"
        )
        error_msgs.extend(f"  - {f}" for f in sorted(missing_copyright))

    if wrong_format:
        error_msgs.append(
            "\nThe following Markdown files have incorrect copyright format:"
        )
        error_msgs.extend(f"  - {f}" for f in sorted(wrong_format))

    if error_msgs:
        error_msgs.append("\nRun 'python add_copyright_headers.py' to fix.")
        assert False, "\n".join(error_msgs)


def test_yaml_files_have_valid_copyright():
    """Test that all YAML files have a valid AMD copyright header (allowed year forms)."""
    root_path = Path(__file__).parent.parent
    skip_dirs = {
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

    missing_copyright = []
    wrong_format = []

    for filepath in list(root_path.rglob("*.yml")) + list(root_path.rglob("*.yaml")):
        if any(skip_dir in filepath.parts for skip_dir in skip_dirs):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if _matches_yaml_copyright_header(content):
                continue
            elif "Copyright (c)" in content[:500]:
                wrong_format.append(str(filepath.relative_to(root_path)))
            else:
                missing_copyright.append(str(filepath.relative_to(root_path)))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    error_msgs = []
    if missing_copyright:
        error_msgs.append("\nThe following YAML files are missing copyright headers:")
        error_msgs.extend(f"  - {f}" for f in sorted(missing_copyright))

    if wrong_format:
        error_msgs.append(
            "\nThe following YAML files have incorrect copyright format (should use # comments, not <!-- -->):"
        )
        error_msgs.extend(f"  - {f}" for f in sorted(wrong_format))

    if error_msgs:
        error_msgs.append(
            "\nRun 'python add_copyright_headers.py --fix-yaml-only' to fix YAML files."
        )
        assert False, "\n".join(error_msgs)


def test_notebooks_have_valid_copyright():
    """Test that all Jupyter notebooks have a valid AMD copyright header (allowed year forms)."""
    root_path = Path(__file__).parent.parent
    skip_dirs = {
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

    missing_copyright = []
    wrong_format = []

    for filepath in root_path.rglob("*.ipynb"):
        if any(skip_dir in filepath.parts for skip_dir in skip_dirs):
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            # Check first cell
            if notebook.get("cells") and len(notebook["cells"]) > 0:
                first_cell = notebook["cells"][0]
                if first_cell.get("cell_type") == "markdown":
                    source = first_cell.get("source", [])
                    if _matches_notebook_copyright_cell(source):
                        continue
                    elif any("Copyright (c)" in line for line in source):
                        wrong_format.append(str(filepath.relative_to(root_path)))
                        continue

            missing_copyright.append(str(filepath.relative_to(root_path)))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    error_msgs = []
    if missing_copyright:
        error_msgs.append("\nThe following notebooks are missing copyright headers:")
        error_msgs.extend(f"  - {f}" for f in sorted(missing_copyright))

    if wrong_format:
        error_msgs.append("\nThe following notebooks have incorrect copyright format:")
        error_msgs.extend(f"  - {f}" for f in sorted(wrong_format))

    if error_msgs:
        error_msgs.append("\nRun 'python add_copyright_headers.py' to fix.")
        assert False, "\n".join(error_msgs)
