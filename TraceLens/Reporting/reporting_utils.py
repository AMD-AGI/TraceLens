###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ast
import logging
import re
from typing import Dict, List, Optional, Union

import pandas as pd
from pathlib import Path
import sys
import subprocess

logger = logging.getLogger(__name__)


def export_data_df(
    data_df: pd.DataFrame,
    output_folder_path: Path,
    output_filename: str,
    output_table_format: list = [".xlsx", ".csv"],
    suffix: str = "_summary_statistics",
    verbose: int = 0,
) -> None:
    """
    Exports a pandas DataFrame to one or more file formats (.xlsx, .csv) in the specified output directory.

    Args:
        data_df (pd.DataFrame): The DataFrame containing data to export.
        output_folder_path (Path): The directory where the output file(s) will be saved.
        output_filename (str): The base name of the output file.
        output_table_format (list, optional): A list of desired file extensions (e.g. [".xlsx", ".csv"]).
        suffix (str, optional): Suffix added to the output filename before the extension. Defaults to "_summary_statistics".
        verbose (int, optional): If > 0, prints additional information during processing. Defaults to 0.

    Returns:
        None
    """
    if verbose:
        print(f"Exporting data to {output_folder_path}")
    if verbose > 3:
        print(f"Data: {data_df}")

    data_df = data_df.round(2)

    for output_table_format in output_table_format:
        if output_table_format == ".xlsx":
            output_path = output_folder_path.joinpath(
                output_filename + suffix
            ).with_suffix(".xlsx")
            if verbose:
                print(f"Exporting summary statistics to {output_path}")

            data_df.to_excel(output_path, index=False)
        elif output_table_format == ".csv":
            output_path = output_folder_path.joinpath(
                output_filename + suffix
            ).with_suffix(".csv")
            if verbose:
                print(f"Exporting summary statistics to {output_path}")
            data_df.to_csv(output_path, index=False)


def request_install(package_name):
    """
    Prompts the user to install a Python package via pip. If the user agrees, attempts installation.
    Exits the program if the user declines or if installation fails.

    Args:
        package_name (str): The name of the package to install.

    Returns:
        None

    Side Effects:
        - Prompts the user for input.
        - May install a package using pip.
        - Exits the program (calls sys.exit(1)) if the user declines or installation fails.
    """
    choice = (
        input(f"Do you want to install '{package_name}' via pip? [y/N]: ")
        .strip()
        .lower()
    )
    if choice == "y":
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
        except subprocess.CalledProcessError:
            print(
                f"Failed to install '{package_name}'. Please install it manually. Exiting."
            )
            sys.exit(1)
    else:
        print(f"Skipping installation of '{package_name}' and exiting.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Node-span utilities for multi-node collective analysis
# ---------------------------------------------------------------------------


def _parse_pg_ranks(value: Union[str, List[int], tuple]) -> List[int]:
    """Extract a list of integer rank ids from a Process Group Ranks value.

    Handles lists, tuples, and the string representations commonly found in
    PyTorch trace JSON (e.g. ``"[0, 1, 2, 3]"``).
    """
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
        return [int(s) for s in re.findall(r"\d+", value)]
    return []


def _rank_to_node_map(world_size: int, gpus_per_node: int) -> Dict[int, int]:
    """Build a rank -> node_id mapping: ``node_id = rank // gpus_per_node``."""
    return {r: r // gpus_per_node for r in range(world_size)}


def _node_span_for_pg(
    pg_ranks_value, rank_to_node: Dict[int, int], gpus_per_node: int
) -> str:
    """Return ``'intra_node'``, ``'inter_node'``, or ``'unknown'``."""
    ranks = _parse_pg_ranks(pg_ranks_value)
    if not ranks:
        return "unknown"
    nodes = {rank_to_node.get(r, r // gpus_per_node) for r in ranks}
    return "intra_node" if len(nodes) == 1 else "inter_node"


def add_node_span_columns(
    df: pd.DataFrame,
    gpus_per_node: int,
    world_size: Optional[int] = None,
) -> pd.DataFrame:
    """Add ``node_id`` and ``node_span`` columns to a DataFrame.

    * ``node_id``: derived from a ``rank`` column (``rank // gpus_per_node``).
    * ``node_span``: ``'intra_node'`` or ``'inter_node'``, derived from
      ``Process Group Ranks`` membership.

    The DataFrame is returned unchanged (no copy) if the required source
    columns are missing.

    Args:
        df: DataFrame produced by NcclAnalyser (must have ``rank`` and/or
            ``Process Group Ranks`` columns).
        gpus_per_node: Number of GPUs (ranks) per physical node.
        world_size: Total number of ranks.  Used to build the rank-to-node
            map; inferred from the ``rank`` column if not provided.
    """
    if df is None or df.empty:
        return df

    if world_size is None:
        if "rank" in df.columns:
            world_size = int(df["rank"].max()) + 1
        else:
            logger.warning(
                "Cannot infer world_size for node_span labeling; "
                "skipping node columns."
            )
            return df

    has_rank = "rank" in df.columns
    has_pg_ranks = "Process Group Ranks" in df.columns
    if not has_rank and not has_pg_ranks:
        return df

    r2n = _rank_to_node_map(world_size, gpus_per_node)
    df = df.copy()

    if has_rank:
        df["node_id"] = df["rank"].map(r2n)

    if has_pg_ranks:
        df["node_span"] = df["Process Group Ranks"].apply(
            lambda v: _node_span_for_pg(v, r2n, gpus_per_node)
        )

    return df


def detect_gpus_per_node(trace_filepath: str) -> Optional[int]:
    """Try to read ``gpus_per_node`` from a trace file's ``deviceProperties``.

    PyTorch profiler traces include a top-level ``deviceProperties`` list whose
    length equals the number of locally visible GPUs on the node that produced
    the trace.  Returns ``None`` if detection fails (e.g. non-PyTorch traces,
    CPU-only traces, or missing metadata).

    Note: this loads the full trace file to parse JSON metadata.  The same file
    will be loaded again by NcclAnalyser, so this adds one redundant read of
    a single trace (~5 s for a typical 30 MB gzipped trace).
    """
    try:
        from TraceLens.util import DataLoader

        data = DataLoader.load_data(trace_filepath)
        device_props = data.get("deviceProperties")
        if isinstance(device_props, list) and len(device_props) > 0:
            return len(device_props)
    except Exception as exc:
        logger.warning("Could not auto-detect gpus_per_node: %s", exc)
    return None
