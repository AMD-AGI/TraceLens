###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared utilities for Perfetto-style trace handling: traceconv resolution
and .pftrace → JSON conversion. Used by generate_perf_report_pftrace_hip_api
and generate_perf_report_pftrace_hip_activity.
"""

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run command; raise on non-zero exit."""
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def acquire_traceconv(preferred: Optional[Path], out_dir: Path) -> Path:
    """
    Resolve a usable traceconv executable:
      1) use preferred path if provided and exists
      2) find traceconv on PATH
      3) download into out_dir (curl, then chmod; fallback to Python urllib)
    """
    if preferred:
        p = preferred.resolve()
        if p.exists():
            return p
        logger.warning("--traceconv provided but not found at %s; continuing.", p)

    on_path = shutil.which("traceconv")
    if on_path:
        return Path(on_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "traceconv"
    logger.info("traceconv not found; downloading to %s", target)
    try:
        run(["curl", "-LO", "https://get.perfetto.dev/traceconv"], cwd=out_dir)
        run(["chmod", "+x", "traceconv"], cwd=out_dir)
        if not target.exists():
            raise RuntimeError("curl reported success but traceconv not found.")
        return target
    except Exception as e:
        logger.warning("curl flow failed (%s). Falling back to Python download...", e)
        urllib.request.urlretrieve("https://get.perfetto.dev/traceconv", target)
        target.chmod(0o755)
        return target


def ensure_trace_json(trace_path: str, traceconv_path: Optional[str] = None) -> str:
    """
    Return path to JSON trace. If trace_path is .pftrace, use traceconv_path if
    provided, otherwise resolve or download traceconv (see acquire_traceconv),
    then convert to JSON. For .json/.json.gz return path as-is.
    """
    path = Path(trace_path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".pftrace":
        if traceconv_path:
            traceconv = Path(traceconv_path).resolve()
            if not traceconv.exists():
                raise FileNotFoundError(f"traceconv not found: {traceconv}")
        else:
            traceconv = acquire_traceconv(None, path.parent)
        out_json = path.with_suffix(".json")
        logger.info("Converting .pftrace to JSON: %s -> %s", path, out_json)
        run([str(traceconv), "json", str(path), str(out_json)])
        return str(out_json)
    if suffix in (".json", ".gz") or path.name.endswith(".json.gz"):
        return str(path)
    raise ValueError(
        f"Unsupported trace format: {trace_path}. Use .json, .json.gz, or .pftrace."
    )
