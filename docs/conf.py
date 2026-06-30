###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# fmt: off
html_theme_options = {
    "flavor": "generic",
    "header_title": f"TraceLens {version}",
    "header_link": False,
    "version_list_link": False,
    "nav_secondary_items": {
        "GitHub": False,
        "Community": False,
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
        "Instinct™ Docs": "https://instinct.docs.amd.com/",
        "Infinity Hub": "https://www.amd.com/en/developer/resources/infinity-hub.html",
        "Support": False,
    },
    "link_main_doc": False,
}
# fmt: on

html_title = f"{project} {version} documentation"
