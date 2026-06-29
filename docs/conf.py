###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

For options specific to rocm-docs-core, see the user guide:
https://rocm.docs.amd.com/projects/rocm-docs-core/en/latest/
"""

# project info
project = "TraceLens"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = "0.1.0"
release = "0.1.0"

# rocm-docs-core provides the ROCm theme, the table-of-contents handling,
# and the shared "external project" cross-references.
extensions = ["rocm_docs"]

external_toc_path = "./sphinx/_toc.yml"

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"{project} {version} documentation"
