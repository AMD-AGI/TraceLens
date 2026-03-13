#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PORT="${TRACELENS_PORT:-8000}"

echo "[TraceLens] Repo root: $REPO_ROOT"

echo "[TraceLens] Installing TraceLens ..."
pip install "$REPO_ROOT" -q

echo "[TraceLens] Installing MCP Server dependencies ..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q

echo "[TraceLens] Starting MCP Server on 0.0.0.0:$PORT ..."
cd "$REPO_ROOT"
exec python -m TraceLens.AgenticMode.MCPServer
