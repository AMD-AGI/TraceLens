###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gates for the model-bringup skill.

Each gate exits 0 (pass), 1 (fail) or 2 (input error). All but ``cases`` and
``op-result`` write ``<bringup_dir>/gates/<name>.json``::

    python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates <subcommand> --bringup-dir B ...
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

from TraceLens.Agent.Analysis.triage.checks import (
    ALL_CHECKS,
    PROFILE_CHECKS,
    rank_of,
)
from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms
from TraceLens.Agent.ModelBringup.utils.unified_perf_ops import (
    DEFAULT_THRESHOLD,
    compute_top_ops,
    load_rows,
    missing_perf_model_share,
    validation_cases,
)
from TraceLens.Agent.PerfModel.ratio_bands import BAND_COLUMNS
from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    CustomCollective,
)
from TraceLens.util import TraceEventUtils

# Constants
BLOCKING_CHECKS = {
    "profile": (
        "TRACE_MISSING",
        "NO_GPU_KERNELS",
        "CAPTURE_MISSING",
        "DRAFTER_CAPTURE_MISSING",
        "GRAPH_LAUNCHES_MISSING",
        "MISSING_CPU_OP_SHAPES",
        "INFERENCE_ANNOTATION_MISSING",
        "SPLIT_TRACE_MISSING",
        "SPLIT_INCORRECT",
        "STEP_COUNT_MISMATCH",
        "CPU_CALL_STACK_MISSING",
    ),
    "report": (
        "NO_GPU_KERNELS",
        "HIPGRAPH_LAUNCH_IN_REPORT",
        "CAPTURE_GRAPH_MERGE_FAIL",
        "CAPTURE_RANK_MISMATCH",
        "PERF_REPORT_FAILURE",
    ),
}
CAPTURE_DIRS = ("capture_traces", "graph_capture_profile")
DRAFT_FILES = {"modeled": "perf_model_pr.md", "blocked": "perf_model_issue.md"}
EFFECTIVE_CONFIG = "cache/bringup_config.yaml"
LEAK_KEYS = ("node", "platform", "docker_image")
GFX_TARGET = re.compile(r"\bgfx\d+\w*", re.I)
VALIDATION_HEADER = "| Shape | Predicted FLOPs | Predicted Bytes | Measured Read Bytes | Measured Write Bytes | FLOPs Ratio | Bytes Ratio |"
SYNTHETIC_MARKER = "(Synthetic Op)"
MIN_TOKEN = 4
MIN_PATH_PARTS = 3
RESULT_KEYS = ("op", "op_slug", "outcome")
SLUG_MAX = 48
STOP_TAG = "[DIAG:bringup:STEP{step}_{code}]"
ANALYSIS_OUTPUTS = (
    "analysis.md",
    "perf_report_csvs/unified_perf_summary.csv",
    "perf_report.log",
)


def prepare(args):
    """Find the bench dir, mixed split and replay rank; link that rank's captures."""
    benches = sorted(Path(args.profile_dir).glob("*/torch_trace"))
    bench = benches[-1].parent if benches else None
    if bench is None:
        return _gate(
            args, "prepare", False, error=f"no torch_trace under {args.profile_dir}"
        )
    splits = sorted(bench.rglob("mixed_steady_state*.json.gz"))
    replay_rank = rank_of(splits[-1].name) if splits else None
    captures = [
        p
        for p in sorted((bench / "torch_trace").rglob("*.json*"))
        if set(CAPTURE_DIRS) & set(p.parts) and rank_of(p.name) in (None, replay_rank)
    ]
    capture_folder = ""
    if captures:
        folder = Path(args.bringup_dir) / f"capture_rank{replay_rank}"
        folder.mkdir(parents=True, exist_ok=True)
        for p in captures:
            link = folder / p.name
            if not link.exists():
                link.symlink_to(p)
        capture_folder = str(folder)
    values = {
        "bench_dir": str(bench),
        "mixed_split": str(splits[-1]) if splits else "",
        "replay_rank": replay_rank,
        "capture_folder": capture_folder,
        "platform": args.platform,
    }
    for key, value in values.items():
        print(f"{key}={value}")
    return _gate(args, "prepare", bool(splits), **values)


def report_health(args):
    """Block on the phase's triage checks unless accepted."""
    name = f"{args.phase}_health"
    triage_csv = Path(args.triage_dir) / "triage_details.csv"
    if not triage_csv.is_file():
        print(f"missing {triage_csv}", file=sys.stderr)
        _gate(args, name, False, error=f"missing {triage_csv}")
        return 2
    specs = {
        s.name: s for s in (PROFILE_CHECKS if args.phase == "profile" else ALL_CHECKS)
    }
    blocking = {
        specs[s].build_tag()
        for s in BLOCKING_CHECKS[args.phase]
        if s not in args.accept
    }
    with triage_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    blocked = [
        {"tag": r["DIAG Tag"], "evidence": r["Evidence"], "remedy": r["Remedy"]}
        for r in rows
        if r["DIAG Tag"] in blocking
    ]
    merge_json = Path(args.triage_dir) / "graph_merge_check.json"
    return _gate(
        args,
        name,
        not blocked,
        blocking=blocked,
        accepted=args.accept,
        other=[r["DIAG Tag"] for r in rows if r["DIAG Tag"] not in blocking],
        graph_merge=(
            json.loads(merge_json.read_text()) if merge_json.is_file() else None
        ),
    )


def gaps(args):
    """Ops without a perf model above the threshold, minus communication, with slugs."""
    rows, _ = load_rows(Path(args.unified_csv))
    comm = {
        r["name"]
        for r in rows
        if (r.get("op category") or "") == CustomCollective.category
        or TraceEventUtils.is_communication_string(
            r.get("kernel_details_summary") or ""
        )
    }
    ops, slugs = [], set()
    for op in compute_top_ops(rows, args.threshold):
        if op["name"] in comm:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", op["name"].lower()).strip("_")[:SLUG_MAX]
        base, n = slug, 2
        while slug in slugs:
            slug, n = f"{base}_{n}", n + 1
        slugs.add(slug)
        ops.append(
            {
                "name": op["name"],
                "op_slug": slug,
                "pct_global": round(op["pct_global"], 2),
                "op_category": op["op_category"],
                "categorize_only": SYNTHETIC_MARKER in op["name"],
            }
        )
    out = {"ops": ops, "comm_excluded": sorted(comm)}
    (Path(args.bringup_dir) / "gaps.json").write_text(json.dumps(out, indent=2))
    for op in ops:
        print(f"{op['op_slug']}  {op['pct_global']}%  {op['name']}")
    return 0


def cases(args):
    """Validator cases (traced shapes) from a perf report CSV directory."""
    out = validation_cases(Path(args.report_dir))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{len(out)} case(s) -> {args.out}")
    return 0


def op_result(args):
    """Check a perf-model sub-agent's result.json: schema; modeled implies a commit and all-OK validation rows; the draft leaks no run input, has the validation header and explains every ratio deviation."""
    path = Path(args.result)
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 2
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return 1
    problems = [f"missing key: {k}" for k in RESULT_KEYS if k not in result]
    outcome = result.get("outcome")
    if outcome not in DRAFT_FILES:
        problems.append(f"outcome is {outcome}: {result.get('reason')}")
    else:
        draft = path.parent / DRAFT_FILES[outcome]
        text = draft.read_text() if draft.is_file() else None
        if text is None:
            problems.append("draft file missing")
        else:
            flat = _norm(text)
            problems += [
                f"draft names {name}"
                for token, name in _leak_tokens(args.bringup_dir).items()
                if token in flat
            ]
            problems += [
                f"draft names {target}"
                for target in sorted(set(GFX_TARGET.findall(text)))
            ]
            if (outcome == "modeled" or "## Validation" in text) and (
                VALIDATION_HEADER not in text
            ):
                problems.append("validation table header is not the canonical header")
        if outcome == "blocked" and not result.get("reason"):
            problems.append("reason missing")
        if outcome == "modeled":
            if not result.get("commit"):
                problems.append("commit missing")
            csv_path = result.get("validation_csv")
            if not csv_path or not Path(csv_path).is_file():
                problems.append("validation CSV missing")
            elif not _all_rows_ok(csv_path):
                problems.append("validation CSV has a row that is not OK")
            elif text is not None:
                problems += _unexplained_ratios(csv_path, text)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def analysis_result(args):
    """Check the analysis sub-agent's result.json and the outputs later steps read."""
    root = Path(args.analysis_dir)
    problems = []
    try:
        result = json.loads((root / "result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        problems.append(f"result.json unreadable: {e}")
    else:
        if not isinstance(result, dict):
            problems.append("result.json is not an object")
        elif result.get("outcome") != "analyzed":
            problems.append(
                f"outcome is {result.get('outcome')}: {result.get('reason')}"
            )
    for name in ANALYSIS_OUTPUTS:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty {name}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return _gate(args, "analysis", not problems, problems=problems)


def coverage(args):
    """Every modeled op is covered after the merge and validated with all rows OK."""
    after_rows, _ = load_rows(Path(args.after))
    covered = {
        op["name"]: op["covered"]
        for op in compute_top_ops(after_rows, 0.0, include_covered=True)
    }
    failing = []
    for result in _results(args.bringup_dir):
        if result.get("outcome") != "modeled":
            continue
        if not covered.get(result["op"]) or not _all_rows_ok(result["validation_csv"]):
            failing.append(result["op"])
    return _gate(
        args,
        "coverage",
        not failing,
        failing=failing,
        missing_perf_model_share_before=missing_perf_model_share(
            load_rows(Path(args.before))[0]
        ),
        missing_perf_model_share_after=missing_perf_model_share(after_rows),
    )


def summary(args):
    """Render bringup_summary.md from the gate and result files only."""
    root = Path(args.bringup_dir)
    stop = STOP_TAG.format(step=args.step, code=args.code) if args.code else None
    _gate(args, "summary", stop is None, stop=stop, reason=args.reason)
    lines = ["# Model bringup summary", ""]
    if stop:
        lines += [f"**Stopped:** {stop} {args.reason or ''}".rstrip(), ""]
    lines += ["| Gate | Status |", "|---|---|"]
    for name in ("prepare", "profile_health", "analysis", "report_health", "coverage"):
        path = root / "gates" / f"{name}.json"
        status = json.loads(path.read_text())["status"] if path.is_file() else "not_run"
        lines.append(f"| {name} | {status} |")
    results = _results(root)
    lines += ["", "| Op | Outcome | Reason |", "|---|---|---|"]
    lines += [
        f"| {r['op_slug']} | {r['outcome']} | {r.get('reason') or ''} |"
        for r in results
    ]
    drafted = [r["op_slug"] for r in results if r["outcome"] in DRAFT_FILES]
    if drafted:
        lines += [
            "",
            "Publish with: `Follow the model-bringup skill installed with TraceLens and "
            f"publish {root} for ops {' '.join(drafted)}`",
        ]
    (root / "bringup_summary.md").write_text("\n".join(lines) + "\n")
    print(root / "bringup_summary.md")
    return 0


def _all_rows_ok(csv_path):
    with open(csv_path, newline="") as f:
        return {r.get("status") for r in csv.DictReader(f)} == {"OK"}


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _leak_tokens(bringup_dir):
    """Normalized run-specific names (and their raw form) a public draft must not carry."""
    config = yaml.safe_load((Path(bringup_dir) / EFFECTIVE_CONFIG).read_text())
    names = [
        Path(bringup_dir).name,
        (config.get("model") or "").rsplit("/", 1)[-1],
        *list_platforms(),
        *(str(config[k]) for k in LEAK_KEYS if config.get(k)),
        *(
            v
            for v in config.values()
            if isinstance(v, str)
            and Path(v).is_absolute()
            and len(Path(v).parts) >= MIN_PATH_PARTS
        ),
    ]
    tokens = {_norm(n): n for n in names}
    return {t: n for t, n in tokens.items() if len(t) >= MIN_TOKEN}


def _unexplained_ratios(csv_path, text):
    """Problems for each deviation ratio in the CSV missing from the draft's Ratio Deviations section."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not set(BAND_COLUMNS.values()) <= set(reader.fieldnames):
        return ["validation CSV has no band columns"]
    found = re.search(r"^## Ratio Deviations\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    section = found.group(1) if found else ""
    return [
        f"ratio {row[ratio]} not explained"
        for row in rows
        for ratio, band in BAND_COLUMNS.items()
        if row[band] == "deviation"
        and not re.search(rf"(?<![\d.]){re.escape(row[ratio])}(?!\d)", section)
    ]


def _results(bringup_dir):
    return [
        json.loads(p.read_text())
        for p in sorted(Path(bringup_dir).glob("perf_models/*/result.json"))
    ]


def _gate(args, name, passed, **fields):
    path = Path(args.bringup_dir) / "gates" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "pass" if passed else "fail", **fields}, indent=2)
    )
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "prepare": (prepare, ("--profile-dir", "--platform")),
        "report-health": (report_health, ("--triage-dir",)),
        "gaps": (gaps, ("--unified-csv",)),
        "cases": (cases, ("--report-dir", "--out")),
        "op-result": (op_result, ("--result",)),
        "analysis-result": (analysis_result, ("--analysis-dir",)),
        "coverage": (coverage, ("--before", "--after")),
        "summary": (summary, ()),
    }
    for name, (fn, required) in commands.items():
        p = sub.add_parser(name, help=fn.__doc__)
        p.set_defaults(fn=fn)
        if name != "cases":
            p.add_argument("--bringup-dir", required=True)
        for flag in required:
            p.add_argument(flag, required=True)
    sub.choices["report-health"].add_argument(
        "--phase", choices=BLOCKING_CHECKS, required=True
    )
    sub.choices["report-health"].add_argument("--accept", action="append", default=[])
    sub.choices["gaps"].add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD
    )
    sub.choices["summary"].add_argument("--step", type=int)
    sub.choices["summary"].add_argument("--code")
    sub.choices["summary"].add_argument("--reason")
    args = parser.parse_args(argv)
    if args.command == "summary" and (args.step is None) != (args.code is None):
        parser.error("--step and --code go together")
    if args.command == "summary" and args.reason and args.code is None:
        parser.error("--reason needs --step and --code")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
