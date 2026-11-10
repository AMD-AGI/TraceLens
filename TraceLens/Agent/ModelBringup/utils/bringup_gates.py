###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gates for the model-bringup skill.

Each gate writes ``<bringup_dir>/gates/<name>.json`` and exits 0 (pass), 1 (fail)
or 2 (input error)::

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
from TraceLens.Agent.ModelBringup.utils.unified_perf_ops import (
    compute_top_ops,
    load_rows,
    missing_perf_model_share,
    validation_cases,
)
from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    CustomCollective,
)
from TraceLens.util import TraceEventUtils

# Constants
BLOCKING_SUBLABELS = {
    "profile": ("1a", "1c", "1d", "1e", "1f", "2a", "2b", "2c", "2d", "2k", "2l"),
    "report": ("1c", "3b", "3c", "3i", "4a"),
}
CAPTURE_DIRS = ("capture_traces", "graph_capture_profile")
SLUG_MAX = 48


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
    config = yaml.safe_load((bench / "config.yaml").read_text()) or {}
    values = {
        "bench_dir": str(bench),
        "mixed_split": str(splits[-1]) if splits else "",
        "replay_rank": replay_rank,
        "capture_folder": capture_folder,
        "workload": (
            "agentx" if (config.get("agentx") or {}).get("enabled") else "fixed_seq"
        ),
        "platform": args.platform,
    }
    for key, value in values.items():
        print(f"{key}={value}")
    return _gate(args, "prepare", bool(splits), **values)


def report_health(args):
    """Block on the phase's triage sublabels unless accepted."""
    triage_csv = Path(args.triage_dir) / "triage_details.csv"
    if not triage_csv.is_file():
        print(f"missing {triage_csv}", file=sys.stderr)
        return 2
    specs = {
        s.sublabel: s
        for s in (PROFILE_CHECKS if args.phase == "profile" else ALL_CHECKS)
    }
    blocking = {
        specs[s].build_tag()
        for s in BLOCKING_SUBLABELS[args.phase]
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
        f"{args.phase}_health",
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


def coverage(args):
    """Every pr_draft op is covered after the merge and validated with all rows OK."""
    after_rows, _ = load_rows(Path(args.after))
    covered = {
        op["name"]: op["covered"]
        for op in compute_top_ops(after_rows, 0.0, include_covered=True)
    }
    failing = []
    for result in _results(args.bringup_dir):
        if result.get("outcome") != "pr_draft":
            continue
        with open(result["validation_csv"], newline="") as f:
            statuses = [r.get("status") for r in csv.DictReader(f)]
        if not covered.get(result["op"]) or not statuses or set(statuses) != {"OK"}:
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
    """Render bringup_summary.md from the gate and worker files only."""
    root = Path(args.bringup_dir)
    lines = ["# Model bringup summary", ""]
    if args.stop_reason:
        lines += [f"**Stopped:** {args.stop_reason}", ""]
    lines += ["| Gate | Status |", "|---|---|"]
    for name in ("prepare", "profile_health", "report_health", "coverage"):
        path = root / "gates" / f"{name}.json"
        status = json.loads(path.read_text())["status"] if path.is_file() else "not_run"
        lines.append(f"| {name} | {status} |")
    results = _results(root)
    lines += ["", "| Op | Outcome | Draft | Reason |", "|---|---|---|---|"]
    lines += [
        f"| {r['op_slug']} | {r['outcome']} | {r.get('draft') or ''} | {r.get('reason') or ''} |"
        for r in results
    ]
    drafted = [r["op_slug"] for r in results if r.get("draft")]
    if drafted:
        lines += [
            "",
            "Publish with: `Follow the model-bringup skill installed with TraceLens and "
            f"publish {root} for ops {' '.join(drafted)}`",
        ]
    (root / "bringup_summary.md").write_text("\n".join(lines) + "\n")
    print(root / "bringup_summary.md")
    return 0


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
        "--phase", choices=BLOCKING_SUBLABELS, required=True
    )
    sub.choices["report-health"].add_argument("--accept", action="append", default=[])
    sub.choices["gaps"].add_argument("--threshold", type=float, default=0.02)
    sub.choices["summary"].add_argument("--stop-reason", default="")
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
