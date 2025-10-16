###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import ast
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch

from TraceLens.UI.utils.reporting import TraceLensColumns


logging.basicConfig(level=logging.INFO)
LOGS_DIR = f"{os.getcwd()}/logs"


@dataclass
class EventArgs:
    dimentions: list[int]
    types: list[torch.dtype]
    strides: list[int]


@dataclass
class TuningBatch:
    gpu_idx: int
    log_file: str
    tuning_file: str
    commands: list[str]


@dataclass
class HipBLASLtColumns:
    DURATION = "us"
    GFLOPS = "hipblaslt-Gflops"


def _replay_kernel(event_args_example: EventArgs, op: callable = torch.mm) -> None:
    device = torch.device("cuda")

    a = torch.randn(
        event_args_example.dimentions[0],
        device=device,
        dtype=event_args_example.types[0],
    )
    b = torch.randn(
        event_args_example.dimentions[1],
        device=device,
        dtype=event_args_example.types[0],
    )

    a = torch.as_strided(
        a, size=event_args_example.dimentions[0], stride=event_args_example.strides[0]
    )
    b = torch.as_strided(
        b, size=event_args_example.dimentions[1], stride=event_args_example.strides[1]
    )

    _ = op(a, b)
    torch.cuda.synchronize()

    del _, a, b
    torch.cuda.empty_cache()


def record_bench_commands(kernes_to_tune: pd.DataFrame, output_log_file: str) -> None:
    TYPES_MAP = {
        "float": torch.float32,
        "c10::Half": torch.float16,
        "c10::BFloat16": torch.bfloat16,
    }

    os.environ["HIPBLASLT_LOG_MASK"] = "32"
    os.environ["HIPBLASLT_LOG_FILE"] = output_log_file
    for _, row in kernes_to_tune.iterrows():
        event_arg = EventArgs(
            dimentions=ast.literal_eval(row[TraceLensColumns.DIMS]),
            types=[TYPES_MAP[t] for t in ast.literal_eval(row[TraceLensColumns.TYPE])],
            strides=ast.literal_eval(row[TraceLensColumns.STRIDES]),
        )
        _replay_kernel(event_arg, getattr(torch.ops.aten, "mm"))
    del os.environ["HIPBLASLT_LOG_FILE"]
    del os.environ["HIPBLASLT_LOG_MASK"]


def _process_batch(tuning_batch: TuningBatch) -> tuple[str]:
    base_dir = os.path.dirname(tuning_batch.tuning_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    new_env = os.environ.copy()
    new_env["HIPBLASLT_TUNING_FILE"] = tuning_batch.tuning_file
    with open(tuning_batch.log_file, "w") as f:
        for command in tuning_batch.commands:
            command += f" --device {tuning_batch.gpu_idx}"
            subprocess.Popen(
                command.split(), shell=False, stdout=f, stderr=f, env=new_env
            ).communicate()


def _get_tuning_file_path(
    base_dir: str,
    gpu_idx: int,
    file_name: Literal["tuning_log.txt", "tuning_file.txt"] = "tuning_log.txt",
) -> str:
    return f"{base_dir}/tuning_process_logs/gpu_{gpu_idx}/{file_name}"


def run_tuning(log_file: str, output_tuning_file: str) -> tuple[str]:
    # prepare batches per GPU
    base_dir = os.path.dirname(output_tuning_file)
    n_gpus = torch.cuda.device_count()
    tuning_batches = [
        TuningBatch(
            gpu_idx=gpu_idx,
            log_file=_get_tuning_file_path(base_dir, gpu_idx, "tuning_log.txt"),
            tuning_file=_get_tuning_file_path(base_dir, gpu_idx, "tuning_file.txt"),
            commands=[],
        )
        for gpu_idx in range(n_gpus)
    ]

    # backup original log file
    log_backup_file = f"{log_file}.bak"
    os.rename(log_file, log_backup_file)

    # shard commands using round robin
    replacements = {
        r"--algo_method index": "--algo_method heuristic --requested_solution 200",
        r"--aux_type f32_r": "",
        r"--cold_iters \d+": "--cold_iters 5",
        r"--iters \d+": "--iters 30",
    }
    with open(log_backup_file, "r") as backup, open(log_file, "w") as log:
        command_idx = 0
        for line in backup:
            command = line.strip()
            for pattern, replacement in replacements.items():
                command = re.sub(pattern, replacement, command)
            log.write(f"{command}\n")

            gpu_idx = command_idx % n_gpus
            tuning_batches[gpu_idx].commands.append(command)
            command_idx += 1

    pool = multiprocessing.Pool(processes=len(tuning_batches))
    pool.map(_process_batch, tuning_batches)
    pool.close()
    pool.join()

    # restoring order of commands from round robin
    header = None
    solutions = ["" for _ in range(command_idx)]
    for tuning_batch in tuning_batches:
        if not os.path.exists(tuning_batch.tuning_file):
            continue

        with open(tuning_batch.tuning_file, "r") as f:
            command_idx = 0
            for line in f:
                if line.startswith("Git"):
                    header = line
                    continue

                solutions[command_idx * n_gpus + tuning_batch.gpu_idx] = line
                command_idx += 1

    # writing solutions to the output file
    with open(output_tuning_file, "w") as out_f:
        out_f.write(header)
        for solution in solutions:
            out_f.write(solution)


def get_winner_solutions(
    tuning_results_file: str, tuning_process_log_file: str
) -> pd.DataFrame:
    columns = None
    values = []
    columns_prefix = "[0]:"
    with open(tuning_process_log_file, "r") as f:
        for line in f:
            if line.startswith(columns_prefix):
                columns = line[len(columns_prefix) :].strip().split(",")
                break

    if columns is None:
        return pd.DataFrame()

    with open(tuning_results_file, "r") as f:
        for line in f:
            if line.startswith("Git"):
                continue

            values.append(line.strip().split(",")[: len(columns)])

    return pd.DataFrame(values, columns=columns).astype(float, errors="ignore")


def perform_offline_tuning(
    kernes_to_tune: pd.DataFrame,
    only_collect_log: bool = True,
    base_dir: str = LOGS_DIR,
) -> tuple[str, str, pd.DataFrame]:
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    input_shapes_csv = f"{base_dir}/shapes.csv"
    kernes_to_tune.to_csv(input_shapes_csv, index=False)
    output_log_file = f"{base_dir}/log.txt"
    output_tuning_file = f"{base_dir}/tuning.txt"
    result = subprocess.run(
        [
            "python",
            os.path.abspath(__file__),
            "--input-shapes-csv",
            input_shapes_csv,
            "--output-log-file",
            output_log_file,
        ]
        + ([] if only_collect_log else ["--output-tuning-file", output_tuning_file]),
    )

    if result.returncode != 0:
        logging.error("Failed to run offline tuning")
        return output_log_file, None, None

    if only_collect_log:
        return output_log_file, None, None

    return (
        output_log_file,
        output_tuning_file,
        get_winner_solutions(
            output_tuning_file, _get_tuning_file_path(base_dir, 0, "tuning_log.txt")
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay kernel and run tuning")
    parser.add_argument(
        "--input-shapes-csv",
        type=str,
        default=f"{LOGS_DIR}/shapes.csv",
        help="Path to the input shapes CSV file",
    )
    parser.add_argument(
        "--output-log-file",
        type=str,
        default=f"{LOGS_DIR}/log.txt",
        help="Path to the output log file",
    )
    parser.add_argument(
        "--output-tuning-file",
        type=str,
        required=False,
        help="Path to the output tuning results file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_shapes_csv):
        logging.info(f"Shapes file {args.input_shapes_csv} does not exist")
        sys.exit(1)

    kernes_to_tune = pd.read_csv(args.input_shapes_csv)
    record_bench_commands(kernes_to_tune, args.output_log_file)
    if not args.output_tuning_file:
        logging.info("Skipping tuning")
        exit()

    run_tuning(args.output_log_file, args.output_tuning_file)
