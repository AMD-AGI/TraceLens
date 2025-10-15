###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from pathlib import Path
import json

# Exact copyright header templates
PYTHON_HEADER = """###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""

MARKDOWN_HEADER = """<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

"""

YAML_HEADER = """###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""


def test_python_files_have_exact_copyright():
    """Test that all Python files have exact copyright headers."""
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
    skip_files = {".gitignore", "LICENSE", "__init__.py"}

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

            # Handle shebang line
            if content.startswith("#!"):
                lines = content.split("\n", 1)
                if len(lines) > 1:
                    content = lines[1]
                else:
                    content = ""

            # Check for exact match
            if content.startswith(PYTHON_HEADER):
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


def test_markdown_files_have_exact_copyright():
    """Test that all Markdown files have exact copyright headers."""
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

            # Check for exact match
            if content.startswith(MARKDOWN_HEADER):
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


def test_yaml_files_have_exact_copyright():
    """Test that all YAML files have exact copyright headers."""
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

            # Check for exact match (YAML uses # comments)
            if content.startswith(YAML_HEADER):
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


def test_notebooks_have_exact_copyright():
    """Test that all Jupyter notebooks have exact copyright headers."""
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

    expected_source = [
        "<!--\n",
        "Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.\n",
        "\n",
        "See LICENSE for license information.\n",
        "-->",
    ]

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
                    if source == expected_source:
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
