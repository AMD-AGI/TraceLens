###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gates for the model-bringup skill.

Each gate exits 0 (pass), 1 (fail) or 2 (input error). All but ``gaps`` and
``cases`` write ``<bringup_dir>/gates/<name>.json``; ``op-result`` writes
``gates/op_result_<op_slug>.json``::

    python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates <subcommand> --bringup-dir B ...
"""

import argparse
import csv
import functools
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
from TraceLens.Agent.PerfModel.utils.unified_perf_ops import (
    DEFAULT_THRESHOLD,
    categorized_names,
    compute_top_ops,
    load_rows,
    missing_perf_model_share,
    validation_cases,
)
from TraceLens.Agent.PerfModel.validate_perf_model.ratio_bands import (
    BAND_COLUMNS,
    DEVIATION,
    DEVIATION_EXPLAINED,
    MATCH,
)
from TraceLens.ModelUtils.config_resolve import load_checkpoint_config
from TraceLens.ModelUtils.extract import parse_architecture
from TraceLens.PerfModel.extensions.custom_collectives_perf_model_extensions import (
    CustomCollective,
)
from TraceLens.util import TraceEventUtils

# Constants
# Validation match statuses a modeled outcome accepts; every other row blocks it.
PASSING_MATCH = (MATCH, DEVIATION_EXPLAINED)
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
        "PERF_REPORT_FAILURE",
    ),
}
CAPTURE_DIRS = ("capture_traces", "graph_capture_profile")
DRAFT_FILES = {
    "modeled": "perf_model_pr.md",
    "categorized": "perf_model_pr.md",
    "blocked": "perf_model_issue.md",
}
# Outcomes whose commit is merged in Step 7 and published as a PR.
COMMITTED_OUTCOMES = ("modeled", "categorized")
EFFECTIVE_CONFIG = "cache/bringup_config.yaml"
MODEL_CONFIG = "cache/model_config.json"
TL_ENV = "cache/tracelens.env"
ARCH_FIELDS = (
    "model_type",
    "attention_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "num_experts",
    "num_experts_per_tok",
    "num_shared_experts",
    "moe_intermediate_size",
)
DP_FLAGS = ("--dp", "--dp-size", "--data-parallel-size")
EP_FLAGS = ("--ep", "--ep-size", "--expert-parallel-size")
DP_ATTENTION_FLAG = "--enable-dp-attention"
ENABLE_EP_FLAG = "--enable-expert-parallel"
FLAG_ALIASES = {**dict.fromkeys(DP_FLAGS, "dp"), **dict.fromkeys(EP_FLAGS, "ep")}
# The values Magpie's AgentXConfig.from_value reads as enabled.
AGENTX_ENABLED = ("true", "enabled", "enable", "yes", "1")
SUMMARY_GATES = (
    "prepare",
    "model_config",
    "profile_health",
    "analysis",
    "report_health",
    "coverage",
)
LEAK_KEYS = ("node", "platform", "docker_image")
GFX_TARGET = re.compile(r"\bgfx\d+\w*", re.I)
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "skills/model-bringup/templates"
TABLE_SEPARATOR = re.compile(r"\|(\s*:?-+:?\s*\|)+")
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
        return args.bringup.write_gate(
            "prepare", False, error=f"no torch_trace under {args.profile_dir}"
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
        folder = args.bringup.capture_folder(replay_rank)
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
    return args.bringup.write_gate("prepare", bool(splits), **values)


def model_config(args):
    """Write the model config and parallel degrees of the profiled run for op agents."""
    path = Path(args.profile_dir) / "benchmark.yaml"
    benchmark = _read_mapping(path).get("benchmark")
    if not isinstance(benchmark, dict) or not benchmark.get("model"):
        raise BringupInputError(f"{path} has no benchmark.model")
    model_id = benchmark["model"]
    envs = {str(k).upper(): v for k, v in (benchmark.get("envs") or {}).items()}
    parallel = _parallel(benchmark, envs)
    model_path = envs.get("MODEL_PATH")
    checkpoint = model_path if model_path and Path(model_path).is_dir() else model_id
    arch = config = error = None
    try:
        config, label = load_checkpoint_config(checkpoint)
    except (OSError, ValueError) as e:
        error = str(e)
    else:
        spec = parse_architecture(config, label)
        arch = {field: getattr(spec, field) for field in ARCH_FIELDS}
    out = args.bringup.root / MODEL_CONFIG
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "parallel": parallel,
                "arch": arch,
                "config": config,
            },
            indent=2,
        )
    )
    env = [f"TL_MODEL={model_id}"]
    if parallel["attn_tp"] is not None:
        env.append(f"TL_TP={parallel['attn_tp']}")
    (args.bringup.root / TL_ENV).write_text("\n".join(env) + "\n")
    return args.bringup.write_gate("model_config", error is None, error=error)


def report_health(args):
    """Block on the phase's triage checks unless accepted."""
    name = f"{args.phase}_health"
    triage_csv = Path(args.triage_dir) / "triage_details.csv"
    if not triage_csv.is_file():
        print(f"missing {triage_csv}", file=sys.stderr)
        args.bringup.write_gate(name, False, error=f"missing {triage_csv}")
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
    return args.bringup.write_gate(
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
    args.bringup.gaps_json.write_text(json.dumps(out, indent=2))
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
    """Check a perf-model sub-agent's result.json: schema; modeled implies a commit and validation rows that ran and match or have an explained deviation; categorized implies a commit and no validation; the draft leaks no run input, uses only the template's tables (the validation table whenever rows exist) and explains every ratio deviation."""
    path = Path(args.result)
    gate = f"op_result_{path.parent.name}"
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        args.bringup.write_gate(gate, False, problems=[f"missing {path.name}"])
        return 2
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return args.bringup.write_gate(gate, False, problems=[f"invalid JSON: {e}"])
    problems = [f"missing key: {k}" for k in RESULT_KEYS if k not in result]
    outcome = result.get("outcome")
    if outcome not in DRAFT_FILES:
        problems.append(f"outcome is {outcome}: {result.get('reason')}")
    else:
        draft_path = path.parent / DRAFT_FILES[outcome]
        draft = PerfModelDraft(draft_path.read_text()) if draft_path.is_file() else None
        rows = _read_validation_csv(result.get("validation_csv"))
        if draft is None:
            problems.append("draft file missing")
        else:
            problems += [
                f"draft names {name}"
                for name in draft.leaks(args.bringup.leak_tokens())
            ]
            needs_validation = outcome == "modeled" or (outcome == "blocked" and rows)
            if needs_validation and not draft.has_validation_table():
                problems.append("validation table missing or not the canonical header")
            if outcome == "categorized" and draft.has_validation_table():
                problems.append("categorized draft has a validation table")
            problems += [
                f"draft table header is not a template header: {header}"
                for header in draft.off_template_tables()
            ]
        if outcome == "blocked" and not result.get("reason"):
            problems.append("reason missing")
        if outcome in COMMITTED_OUTCOMES and not result.get("commit"):
            problems.append("commit missing")
        if outcome == "modeled":
            if not rows:
                problems.append("validation CSV missing or empty")
            elif not _all_rows_ok(rows):
                problems.append(
                    "validation CSV has a row that did not run or whose match_status "
                    "is not match or deviation-explained"
                )
            elif draft is not None:
                problems += draft.unexplained_ratios(rows)
    for problem in problems:
        print(problem, file=sys.stderr)
    return args.bringup.write_gate(gate, not problems, problems=problems)


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
    return args.bringup.write_gate("analysis", not problems, problems=problems)


def coverage(args):
    """After the merge, every modeled op is covered and validated, and every categorized op has a category."""
    after_rows, _ = load_rows(Path(args.after))
    covered = {
        op["name"]: op["covered"]
        for op in compute_top_ops(after_rows, 0.0, include_covered=True)
    }
    categorized = categorized_names(after_rows)
    failing = []
    for result in args.bringup.results():
        outcome = result.get("outcome")
        if outcome == "modeled" and not (
            covered.get(result["op"])
            and _all_rows_ok(_read_validation_csv(result["validation_csv"]))
        ):
            failing.append(result["op"])
        elif outcome == "categorized" and result["op"] not in categorized:
            failing.append(result["op"])
    return args.bringup.write_gate(
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
    bringup = args.bringup
    stop = STOP_TAG.format(step=args.step, code=args.code) if args.code else None
    bringup.write_gate("summary", stop is None, stop=stop, reason=args.reason)
    lines = ["# Model bringup summary", ""]
    if stop:
        lines += [f"**Stopped:** {stop} {args.reason or ''}".rstrip(), ""]
    lines += ["| Gate | Status |", "|---|---|"]
    for name in SUMMARY_GATES:
        lines.append(f"| {name} | {bringup.gate_status(name)} |")
    rows = bringup.op_rows()
    lines += ["", "| Op | Outcome | Reason |", "|---|---|---|"]
    lines += [f"| {slug} | {outcome} | {reason} |" for slug, outcome, reason in rows]
    drafted = [slug for slug, outcome, _ in rows if outcome in DRAFT_FILES]
    if drafted:
        lines += [
            "",
            "Publish with: `Follow the model-bringup skill installed with TraceLens and "
            f"publish {bringup.root} for ops {' '.join(drafted)}`",
        ]
    bringup.summary_md.write_text("\n".join(lines) + "\n")
    print(bringup.summary_md)
    return 0


class BringupInputError(Exception):
    """A missing or malformed bringup input; the gate exits 2."""


class BringupDir:
    """Paths under ``<bringup_dir>`` and its effective config, loaded once."""

    def __init__(self, root):
        self.root = Path(root)
        self.gaps_json = self.root / "gaps.json"
        self.summary_md = self.root / "bringup_summary.md"

    @functools.cached_property
    def config(self):
        return _read_mapping(self.root / EFFECTIVE_CONFIG)

    def capture_folder(self, rank):
        return self.root / f"capture_rank{rank}"

    def gate_path(self, name):
        return self.root / "gates" / f"{name}.json"

    def gate_status(self, name):
        path = self.gate_path(name)
        return json.loads(path.read_text())["status"] if path.is_file() else "not_run"

    def write_gate(self, name, passed, **fields):
        """Write ``gates/<name>.json``; return the gate's exit code (0 pass, 1 fail)."""
        path = self.gate_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"status": "pass" if passed else "fail", **fields}, indent=2)
        )
        return 0 if passed else 1

    def op_gate_status(self, op_slug):
        """``pass``, ``fail`` or ``not_run`` for the op-result gate of ``op_slug``."""
        return self.gate_status(f"op_result_{op_slug}")

    def results(self):
        """``result.json`` of every op whose op-result gate passed, in op-slug order."""
        return [
            json.loads(p.read_text())
            for p in sorted(self.root.glob("perf_models/*/result.json"))
            if self.op_gate_status(p.parent.name) == "pass"
        ]

    def op_rows(self):
        """``(op_slug, outcome, reason)`` for every op in ``gaps.json``.

        An op whose op-result gate passed shows its worker's outcome; one whose
        gate failed (twice, in Step 6) is ``excluded``; one never launched is
        ``skipped``.
        """
        if not self.gaps_json.is_file():
            return []
        rows = []
        for op in json.loads(self.gaps_json.read_text())["ops"]:
            slug = op["op_slug"]
            status = self.op_gate_status(slug)
            if status == "not_run":
                rows.append((slug, "skipped", "not launched"))
            elif status == "fail":
                problems = json.loads(self.gate_path(f"op_result_{slug}").read_text())
                rows.append((slug, "excluded", problems["problems"][0]))
            else:
                result_path = self.root / "perf_models" / slug / "result.json"
                result = json.loads(result_path.read_text())
                rows.append((slug, result["outcome"], result.get("reason") or ""))
        return rows

    def leak_tokens(self):
        """Normalized run-specific names (and their raw form) a public draft must not carry."""
        config = self.config
        names = [
            self.root.name,
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


class PerfModelDraft:
    """A perf-model PR or issue draft and the checks op-result runs on it."""

    def __init__(self, text):
        self.text = text
        self.normalized = _norm(text)
        self.table_headers = self._table_headers(text)

    def leaks(self, tokens):
        """Run-specific names from ``tokens`` and gfx targets the draft names."""
        names = [name for token, name in tokens.items() if token in self.normalized]
        return names + sorted(set(GFX_TARGET.findall(self.text)))

    def has_validation_table(self):
        return _template_headers()[0] in self.table_headers

    def off_template_tables(self):
        return [h for h in self.table_headers if h not in _template_headers()[1]]

    def unexplained_ratios(self, rows):
        """Problems for each deviation ratio in ``rows`` missing from the Ratio Deviations section."""
        if not set(BAND_COLUMNS.values()) <= set(rows[0]):
            return ["validation CSV has no band columns"]
        found = re.search(
            r"^## Ratio Deviations\n(.*?)(?=^## |\Z)", self.text, re.S | re.M
        )
        section = found.group(1) if found else ""
        return [
            f"ratio {row[ratio]} not explained"
            for row in rows
            for ratio, band in BAND_COLUMNS.items()
            if row[band] == DEVIATION
            and not re.search(rf"(?<![\d.]){re.escape(row[ratio])}(?!\d)", section)
        ]

    @staticmethod
    def _table_headers(text):
        """Header rows of the markdown tables in ``text``, outside code fences."""
        lines, in_fence = [], False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("```"):
                in_fence = not in_fence
            elif not in_fence:
                lines.append(line)
        return [
            line
            for line, nxt in zip(lines, lines[1:])
            if line.startswith("|") and TABLE_SEPARATOR.fullmatch(nxt)
        ]


def _read_mapping(path):
    """The YAML mapping at ``path``; a missing, malformed or non-mapping file is an input error."""
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise BringupInputError(f"missing {path}") from None
    except yaml.YAMLError as e:
        raise BringupInputError(f"malformed {path}: {e}") from None
    if not isinstance(data, dict):
        raise BringupInputError(f"{path} is not a mapping")
    return data


def _positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _parallel(benchmark, envs):
    """tp, ep, dp and attention tp of the run; 1 (the engine default) when the YAML sets nothing, None when unknown.

    AgentX overwrites the envs with its recipe entry, so all four are unknown there.
    """
    agentx = benchmark.get("agentx")
    if isinstance(agentx, dict):
        agentx = agentx.get("enabled", True)
    if str(agentx).lower() in AGENTX_ENABLED:
        return {"tp": None, "ep": None, "dp": None, "attn_tp": None}
    tokens = str(
        envs.get(f"EXTRA_{str(benchmark.get('framework')).upper()}_ARGS") or ""
    )
    tokens = tokens.split()
    flags = {}
    for i, token in enumerate(tokens):
        if token.startswith("--"):
            name, eq, value = token.partition("=")
            if not eq and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
            flags[FLAG_ALIASES.get(name, name)] = value

    tp = _positive_int(envs["TP"]) if "TP" in envs else 1
    dp_attention = (
        DP_ATTENTION_FLAG in flags or str(envs.get("DP_ATTENTION")).lower() == "true"
    )
    dp_value = flags.get("dp")
    dp = _positive_int(dp_value) if dp_value else (None if dp_attention else 1)
    attn_tp = tp
    if dp_attention:
        attn_tp = tp // dp if tp and dp and tp % dp == 0 else None
    ep_value = flags.get("ep") or envs.get("EP_SIZE") or envs.get("EP")
    if ep_value:
        ep = _positive_int(ep_value)
    elif ENABLE_EP_FLAG in flags:
        ep = tp * dp if tp and dp else None
    else:
        ep = 1
    return {"tp": tp, "ep": ep, "dp": dp, "attn_tp": attn_tp}


def _read_validation_csv(csv_path):
    """Rows of a validation CSV; [] when ``csv_path`` is unset or not a file."""
    if not csv_path or not Path(csv_path).is_file():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


@functools.cache
def _template_headers():
    """The PR template's Validation table header, and every table header in both templates."""
    texts = {
        name: (TEMPLATES_DIR / name).read_text() for name in set(DRAFT_FILES.values())
    }
    pr_validation = texts[DRAFT_FILES["modeled"]].split("## Validation")[1]
    validation = PerfModelDraft._table_headers(pr_validation)[0]
    return validation, {
        h for text in texts.values() for h in PerfModelDraft._table_headers(text)
    }


def _all_rows_ok(rows):
    """Every row ran (``run_status`` OK) and matches or has an explained deviation."""
    return bool(rows) and all(
        r.get("run_status") == "OK" and r.get("match_status") in PASSING_MATCH
        for r in rows
    )


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "prepare": (prepare, ("--profile-dir", "--platform")),
        "model-config": (model_config, ("--profile-dir",)),
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
    if args.command != "cases":
        args.bringup = BringupDir(args.bringup_dir)
    try:
        return args.fn(args)
    except BringupInputError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
