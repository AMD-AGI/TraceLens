###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TraceLens HTTP MCP Server — main entry point.

Runs a single Starlette app via FastMCP that serves:
  - MCP Streamable HTTP at /mcp  (for Cursor / Claude Code)
  - Health check at /health      (GET)

Usage:
  python -m TraceLens.AgenticMode.MCPServer
"""

import logging


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Import here so logging is configured before module-level init
    from .mcp_app import mcp

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
