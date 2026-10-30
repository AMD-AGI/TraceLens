###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic ``analysis.json`` renderer, driven purely from ``analysis.md``.

TraceLens authors the human-readable ``analysis.md`` perf report; this module
reads only that md and emits a structured, operation-grouped ``analysis.json``
beside it (compute and fusion tiers, standalone only, v1). The parse anchors on the ``MarkerValidator``
marker grammar so the renderer moves in lockstep with the writer. Byte-identical
on rerun. Fallback and agentic single-trace variants share one parser; the only
mode-conditional is the deterministic-fallback grouping key.
"""

import json
import logging
import re
from pathlib import Path

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    HEURISTIC_FRACTION_MID,
    _arg_pairs,
    _kernel_name_list,
    node_field,
)
from TraceLens.Agent.Analysis.post_processing.candidate_schema import (
    ComputeTask,
    TaskMember,
)
from TraceLens.Agent.Analysis.utils.classify_kernels import TAXONOMY
from TraceLens.Agent.Analysis.utils.kernel_fusion import FusionForms, _efficiency_parts
from TraceLens.Agent.Analysis.utils.validation_utils import (
    EXEC_TOTAL_COL,
    PCT_OF_NODE_COL,
    TIER_COMPUTE,
    TIER_FUSION,
    MarkerValidator,
    _find_data_table,
    _iter_candidate_blocks,
    _marker_attrs,
)
from TraceLens.Reporting.aggregate_perf_reports import NODE_COL

logger = logging.getLogger(__name__)

# Constants
MAX_TASKS = 100
IDENTIFICATION_CAP = 2000
REASONING_CAP = 2000
RESOLUTION_CAP = 2000
# Raw perf-report columns a caller passes as ``extra_fields`` to
# ``build_operation_metrics`` so ``compute_tasks`` can build task members.
TASK_SOURCE_COLUMNS = ("Input Dims", "Input type", "call_stack_full")

_TITLE_RE = re.compile(r"^#\s+(.*?)\s*$", re.MULTILINE)
_LABEL_RE = re.compile(
    r"\*\*(Identification|Reasoning for Slowdown|Resolution):\*\*\s*(.+?)"
    r"(?=\n\s*\n|\n\*\*|\Z)",
    re.DOTALL,
)
_KIND_RE = re.compile(r"surfaced as\s+`(\w+)`", re.IGNORECASE)
_FUSION_KINDS = FusionForms(TAXONOMY).kinds
_EFFICIENCY_RE = re.compile(r"([\d.]+)%\s*of\s*([\d.]+)\s*(\S+)")
_LIBRARY_PARENS_RE = re.compile(r"\(([^()]+)\)\s*$")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PRIORITY_PREFIX_RE = re.compile(r"^(?:[🔴🟡🟢]\s*)?(?:P\d+:\s*)?")
_TOTAL_TIME_RE = re.compile(
    r"^\|\s*Total Time\s*\|\s*(\d+(?:\.\d+)?) ms\s*\|", re.MULTILINE
)
_NULL_CELLS = {"", "-", "—", "–"}
_TRAILING_COMMENT_RE = re.compile(r"\s*<!--.*?-->\s*$")

# Data-table header labels, normalized (lowercased, stripped) -> logical field.
_HEADER_ALIASES = {
    "operation": "operation",
    "args": "args",
    "kernel path": "kernel_path",
    "kernel name": "kernel_name",
    "time (ms)": "time_ms",
    "%e2e": "pct_e2e",
    PCT_OF_NODE_COL.lower(): "pct_of_node",
    "count": "count",
    "flops/byte": "flops_per_byte",
    "efficiency": "efficiency",
    "bound": "bound",
}

# Top Operations header labels, normalized -> emitted JSON field. The comparative
# table's Trace 2 Time and Difference columns are deliberately unmapped.
_TOP_OPS_HEADERS = {
    "rank": "rank",
    "category": "category",
    "time (ms)": "time_ms",
    "trace 1 time (ms)": "time_ms",
    "% of compute time": "pct_of_compute",
    "ops": "ops",
}
# The improvement column's unit suffix varies by report, so it matches by prefix.
_IMPROVEMENT_HEADER_PREFIX = "potential improvement"

# Executive-summary metric label (lowercased, stripped) -> emitted JSON field.
_METRIC_FIELDS = {
    "total time": "total_time_ms",
    "compute %": "compute_pct",
    "idle %": "idle_pct",
    "exposed communication %": "exposed_communication_pct",
}


def render_analysis_json(analysis_md: "str | Path") -> Path:
    """Render ``analysis.json`` beside the given single-trace ``analysis.md``.

    Reads ONLY the md, anchors on the ``MarkerValidator`` grammar, writes the
    grouped JSON and returns the written path. Byte-identical on rerun. Handles
    both the agentic and deterministic-fallback variants; comparative md is out
    of scope for v1.
    """
    md_path = Path(analysis_md)
    parser = AnalysisMdParser(md_path.read_text())

    report_info = parser.report_info()
    mode = report_info["mode"]

    report: dict = {"report_info": report_info, "title": parser.title()}

    exec_summary = parser.exec_summary()
    if exec_summary is not None:
        report["executive_summary"] = exec_summary

    top_ops = parser.top_operations()
    if top_ops is not None:
        report["top_operations"] = top_ops

    for tier in (TIER_COMPUTE, TIER_FUSION):
        report[f"{tier}_optimizations"] = TaskGrouper(
            parser.findings(tier), tier, mode
        ).group()

    appendix = parser.appendix()
    if appendix is not None:
        report["appendix"] = appendix

    out_path = md_path.with_name("analysis.json")
    out_path.write_text(
        json.dumps(report, sort_keys=False, indent=2, ensure_ascii=False) + "\n"
    )
    return out_path


def compute_tasks(op_metrics: list[dict]) -> list[ComputeTask]:
    """Group ``build_operation_metrics`` rows into ``analysis.json`` compute tasks.

    One task per op name, one member per row, grouped, sorted and capped by the
    same ``TaskGrouper`` that renders the md-driven report. The rows must carry
    ``TASK_SOURCE_COLUMNS`` as ``extra_fields``; a row missing one yields an
    empty field.
    """
    return TaskGrouper(
        [_op_finding(op) for op in op_metrics], TIER_COMPUTE, "non-llm"
    ).group()


def _op_finding(op: dict) -> dict:
    """One op-metrics row as a single-member finding for ``TaskGrouper``."""
    eff = op["efficiency"]
    pct, peak, unit, bound = _efficiency_parts(eff)
    args = _arg_pairs(op.get("Input Dims"), op.get("Input type"))
    member = {
        "operation_cell": op["name"],
        "kernel_launcher_path": op["launcher_path"],
        "library": op["library"],
        "category": None,
        "analysis_md_rank": None,
        "kernel_name": _kernel_name_list(op.get("call_stack_full")),
        "args_shapes": [shape for shape, _ in args] or None,
        "args_datatypes": [kind or None for _, kind in args] or None,
        "time_ms": op["time_ms"],
        "count": op["count"],
        "pct_e2e": op["percent_of_total"],
        "flops_per_byte": eff["flops_per_byte"],
        "efficiency_percent": pct,
        "efficiency_peak_value": peak,
        "efficiency_peak_unit": unit,
        "bound": bound,
    }
    return {
        "members": [member],
        "impacts": [op["impact_score"]],
        "prose": {},
        "node": op.get(NODE_COL),
    }


class AnalysisMdParser:
    """Parse an ``analysis.md`` report into the pieces ``analysis.json`` needs.

    Holds the raw ``text`` plus its ``lines`` (split once) so the marker/table
    walks and every parse method share one view of the document.
    """

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines()

    def title(self) -> str:
        m = _TITLE_RE.search(self.text)
        return m.group(1).strip() if m else ""

    def report_info(self) -> dict:
        """Roll up mode / warnings / trace_quality from the report-marker family.

        ``mode`` defaults to ``agentic`` when no ``report_mode`` marker is present
        (pre-alignment reports are all agentic). ``warnings`` holds one string per
        ``kind=warning`` block in document order; the key is absent when there are
        none. ``trace_quality`` is ``poor`` iff the mode is deterministic-fallback
        or any warning is present, else ``ok``.
        """
        mode = "agentic"
        warnings: list[str] = []
        for kind, attrs, inner in self._iter_report_blocks():
            if kind == "report_mode" and attrs.get("mode"):
                mode = attrs["mode"]
            elif kind == "warning":
                prose = self._unwrap_blockquote(inner)
                if prose:
                    warnings.append(prose)

        info: dict = {
            "mode": mode,
            "comparison_scope": "standalone",
            "trace_quality": (
                "poor" if (mode == "deterministic-fallback" or warnings) else "ok"
            ),
        }
        if warnings:
            info["warnings"] = warnings
        return info

    def exec_summary(self) -> "dict | None":
        section = self._section("Executive Summary")
        if section is None:
            return None
        narrative_parts = []
        metrics: dict = {}
        node_metrics: dict = {}
        for ln in section.splitlines():
            stripped = ln.strip()
            if stripped.startswith("|"):
                cells = self._split_row(stripped)
                if cells[:2] == ["Metric", EXEC_TOTAL_COL]:
                    node_metrics = {name: {} for name in cells[2:]}
                elif len(cells) >= 2:
                    field = _METRIC_FIELDS.get(cells[0].strip().lower())
                    if field:
                        metrics[field] = self._first_float(cells[1])
                        for name, cell in zip(node_metrics, cells[2:]):
                            node_metrics[name][field] = self._first_float(cell)
            elif (
                stripped
                and not stripped.startswith("!")
                and not stripped.startswith("#")
            ):
                narrative_parts.append(stripped)
        narrative = " ".join(narrative_parts).strip()
        if not narrative and not metrics:
            return None
        summary = {"narrative": narrative, "metrics": metrics}
        if node_metrics:
            summary["node_metrics"] = node_metrics
        return summary

    def _total_time_ms(self) -> float:
        """The Executive Summary's Total Time, the ``% of Node`` tables' T_all."""
        found = _TOTAL_TIME_RE.search(self.text)
        if not found:
            raise ValueError(
                "A '% of Node' table needs the Executive Summary row "
                "'| Total Time | <N> ms |' with a plain number "
                "(no thousands separator)"
            )
        return float(found.group(1))

    def top_operations(self) -> "list[dict] | None":
        rows = []
        for attrs, block in self._marker_blocks("top_ops"):
            for row in self._iter_table_rows(block):
                fields = {
                    (
                        "potential_improvement_md"
                        if h.startswith(_IMPROVEMENT_HEADER_PREFIX)
                        else _TOP_OPS_HEADERS.get(h)
                    ): c
                    for h, c in row.items()
                }
                rows.append(
                    {
                        "rank": int(self._first_float(fields.get("rank")) or 0),
                        "category": self._cell_or_null(fields.get("category")),
                        "time_ms": self._first_float(fields.get("time_ms")),
                        "pct_of_compute": self._first_float(
                            fields.get("pct_of_compute")
                        ),
                        "ops": self._int_or_null(fields.get("ops")),
                        "potential_improvement_md": self._cell_or_null(
                            fields.get("potential_improvement_md")
                        ),
                        **node_field(attrs.get(NODE_COL)),
                    }
                )
        return rows if rows else None

    def findings(self, tier: str) -> list:
        """Parse every ``tier`` reasoning-candidate block into finding dicts."""
        attrs_by_rank = self._card_attrs(tier)
        return [
            self._parse_finding(start, end, rank, tier, attrs_by_rank)
            for start, end, rank in _iter_candidate_blocks(self.text, tier)
        ]

    def appendix(self) -> "dict | None":
        section = self._section("Appendix")
        if section is None:
            return None
        model = self._bullet_map(
            self._section("Model Architecture", level=3, text=section)
        )
        hardware = self._bullet_map(
            self._section("Hardware Reference", level=3, text=section)
        )
        if not model and not hardware:
            return None
        return {"model_architecture": model, "hardware_reference": hardware}

    def _parse_finding(
        self, start: int, end: int, rank: int, tier: str, attrs_by_rank: dict
    ) -> dict:
        """Parse one ``tier`` reasoning-candidate block into a finding dict.

        ``attrs_by_rank`` joins each finding to its card ``p_item``
        marker attrs via the tier's rank; the category is per-finding and is
        stamped on every member, the node on the finding.
        """
        lines = self.lines
        block = "\n".join(lines[start:end])

        md_rank = f"P{rank}"
        attrs = attrs_by_rank.get(rank, {})
        category = attrs.get("category")
        node = attrs.get(NODE_COL) or self._parse_block_node(block)

        heading = next((ln for ln in lines[start:end] if ln.startswith("####")), None)
        library = None
        title = None
        if tier == TIER_COMPUTE:
            lib_m = _LIBRARY_PARENS_RE.search(heading) if heading else None
            library = lib_m.group(1).strip() if lib_m else None
        elif heading:
            title = _PRIORITY_PREFIX_RE.sub("", heading.lstrip("#").strip())

        prose = {"identification": None, "reasoning": None, "resolution": None}
        for label, body in _LABEL_RE.findall(block):
            cleaned = " ".join(body.split())
            if label == "Identification":
                prose["identification"] = cleaned
            elif label == "Reasoning for Slowdown":
                prose["reasoning"] = cleaned
            else:
                prose["resolution"] = cleaned

        kind_m = _KIND_RE.search(prose["identification"] or "")
        kind = kind_m.group(1).lower() if kind_m else None
        if kind not in _FUSION_KINDS:
            kind = None

        members = self._parse_data_table(start, end)
        for member in members:
            member["category"] = category
            member["library"] = library
            member["analysis_md_rank"] = md_rank
        impacts = self._parse_op_row_marker(block)
        low, mid, high = self._parse_finding_range(block)
        return {
            "members": members,
            "impacts": impacts,
            "prose": prose,
            "rank": md_rank,
            "title": title,
            "kind": kind,
            "low": low,
            "mid": mid,
            "high": high,
            "node": node,
        }

    def _parse_data_table(self, start: int, end: int) -> list:
        """Parse the block's **Data:** table into per-row member dicts."""
        table = _find_data_table(self.lines, start, end)
        if table is None:
            return []
        _, header_cols, row_iter = table
        col = {
            _HEADER_ALIASES[h.strip().lower()]: i
            for i, h in enumerate(header_cols)
            if h.strip().lower() in _HEADER_ALIASES
        }

        total_ms = self._total_time_ms() if "pct_of_node" in col else None
        members = []
        for _, cells in row_iter:

            def get(field: str) -> "str | None":
                idx = col.get(field)
                return cells[idx] if idx is not None and idx < len(cells) else None

            operation_cell = get("operation")
            eff_pct, eff_peak, eff_unit = self._parse_efficiency(get("efficiency"))
            args_shapes, args_datatypes = self._split_args(get("args"))
            time_ms = self._first_float(get("time_ms"))
            pct_e2e = (
                round(time_ms / total_ms * 100, 2)
                if total_ms and time_ms is not None
                else self._first_float(get("pct_e2e"))
            )
            members.append(
                {
                    "operation_cell": operation_cell,
                    "kernel_name": self._split_kernel_cell(get("kernel_name")),
                    "kernel_launcher_path": self._cell_or_null(get("kernel_path")),
                    "library": None,
                    "category": None,
                    "analysis_md_rank": None,
                    "args_shapes": args_shapes,
                    "args_datatypes": args_datatypes,
                    "time_ms": time_ms,
                    "count": self._int_or_null(get("count")),
                    "pct_e2e": pct_e2e,
                    "flops_per_byte": self._first_float(get("flops_per_byte")),
                    "efficiency_percent": eff_pct,
                    "efficiency_peak_value": eff_peak,
                    "efficiency_peak_unit": eff_unit,
                    "bound": self._cell_or_null(get("bound")),
                }
            )
        return members

    def _card_attrs(self, tier: str) -> dict:
        """Map ``tier`` rank -> card ``p_item`` marker attrs via the detailed anchor.

        Each card carries a ``kind=p_item`` marker followed by a
        ``#detailed-analysis-<tier>-p{N}`` link; the anchor's N is the rank
        the detailed block is keyed on. Cards without an anchor are skipped, so
        findings with no matching card (e.g. fallback reports) keep
        ``category=null`` and take ``node`` from their own p_item marker.
        """
        anchor_re = re.compile(rf"#detailed-analysis-{tier}-[pP](\d+)")
        mapping: dict = {}
        for m, attrs in _marker_attrs(self.text, "p_item"):
            end = MarkerValidator.END_RE.search(self.text, m.end())
            if not end:
                continue
            nxt = MarkerValidator.BEGIN_RE.search(self.text, end.end())
            link_m = anchor_re.search(
                self.text[end.end() : nxt.start() if nxt else None]
            )
            if link_m:
                mapping[int(link_m.group(1))] = attrs
        return mapping

    def _parse_op_row_marker(self, block: str) -> "list[float | None] | None":
        """Return the ``kind=op_row`` ``impacts=`` CSV, one entry per row.

        A ``—`` cell (spec-legal null, e.g. a fusion-flagged row) maps to
        ``None``, which scores 0.0. The whole marker is None when absent, and
        every row then back-fills ``pct_e2e * HEURISTIC_FRACTION_MID``.
        """
        for m in MarkerValidator.BEGIN_RE.finditer(block):
            inner = m.group(1)
            km = MarkerValidator.KIND_ATTR_RE.search(inner)
            if not km or km.group(1) != "op_row":
                continue
            im = MarkerValidator.OP_ROW_IMPACTS_RE.search(inner)
            return [
                None if v.strip() in _NULL_CELLS else float(v)
                for v in im.group(1).split(",")
                if v.strip()
            ]
        return None

    def _parse_block_node(self, block: str) -> "str | None":
        """Return ``node=`` from the block's own ``kind=p_item`` marker, if any.

        Fallback reports carry no card anchors, so the node rides on the p_item
        marker inside the reasoning-candidate block instead.
        """
        for _, attrs in _marker_attrs(block, "p_item"):
            return attrs.get(NODE_COL)
        return None

    def _parse_finding_range(self, block: str) -> tuple:
        """Return (low, mid, high) E2E-% from the detail_estimate/p_item marker.

        ``mid`` is None when the marker omits it.
        """
        for kind in ("detail_estimate", "p_item"):
            for m in MarkerValidator.BEGIN_RE.finditer(block):
                inner = m.group(1)
                km = MarkerValidator.KIND_ATTR_RE.search(inner)
                if not km or km.group(1) != kind:
                    continue
                attrs = dict(MarkerValidator.ATTR_RE.findall(inner))
                low, mid, high = attrs.get("low"), attrs.get("mid"), attrs.get("high")
                if low in (None, "null") or high in (None, "null"):
                    continue
                mid = None if mid is None else float(mid)
                return float(low), mid, float(high)
        return None, None, None

    def _iter_report_blocks(self):
        """Yield (kind, attrs, inner_text) per report-begin/report-end block."""
        text = self.text
        for m in MarkerValidator.REPORT_BEGIN_RE.finditer(text):
            end = MarkerValidator.REPORT_END_RE.search(text, m.end())
            inner_marker = m.group(1)
            kind_m = MarkerValidator.KIND_ATTR_RE.search(inner_marker)
            if not kind_m:
                continue
            attrs = dict(MarkerValidator.ATTR_RE.findall(inner_marker))
            inner_text = text[m.end() : end.start()] if end else ""
            yield kind_m.group(1), attrs, inner_text

    @staticmethod
    def _unwrap_blockquote(block: str) -> str:
        """Flatten a ``> …`` markdown blockquote into a single prose string."""
        lines = [
            re.sub(r"^\s*>\s?", "", ln).strip() for ln in block.strip().splitlines()
        ]
        return " ".join(ln for ln in lines if ln).strip()

    def _section(
        self, name: str, level: int = 2, text: "str | None" = None
    ) -> "str | None":
        """Return the body of a ``level``-deep ``<name>`` heading, or None.

        ``text`` defaults to the whole document; pass a section body to scope a
        nested subsection lookup.
        """
        haystack = self.text if text is None else text
        hashes = "#" * level
        start = re.search(rf"^{hashes}\s+{re.escape(name)}\b", haystack, re.MULTILINE)
        if not start:
            return None
        rest = haystack[start.end() :]
        nxt = re.search(rf"^{hashes}\s+", rest, re.MULTILINE)
        return rest[: nxt.start()] if nxt else rest

    def _marker_blocks(self, kind: str):
        """Yield (attrs, text) between each ``impact-begin kind=<kind>`` and its end."""
        text = self.text
        for m, attrs in _marker_attrs(text, kind):
            end = MarkerValidator.END_RE.search(text, m.end())
            yield attrs, text[m.end() : end.start()] if end else text[m.end() :]

    def _iter_table_rows(self, block: str):
        """Yield body rows of the first markdown table in ``block`` as
        ``{lowercased header: cell}`` dicts, trailing HTML comments stripped."""
        header = None
        for ln in block.splitlines():
            stripped = ln.strip()
            if not stripped.startswith("|"):
                if header is not None:
                    break
                continue
            cells = self._split_row(_TRAILING_COMMENT_RE.sub("", stripped))
            if header is None:
                header = [c.lower() for c in cells]
                continue
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            yield dict(zip(header, cells))

    @staticmethod
    def _split_row(row: str) -> list:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    @staticmethod
    def _split_kernel_cell(cell: "str | None") -> list:
        """Split a ``<br>``-joined Kernel Name cell, stripping ``Kernel N:`` prefixes.

        Empty / ``-`` / ``—`` pieces are dropped; survivors keep a contiguous order.
        """
        if not cell:
            return []
        names = []
        for piece in cell.split("<br>"):
            piece = re.sub(r"^\s*Kernel\s+\d+:\s*", "", piece).strip()
            if piece and piece not in _NULL_CELLS:
                names.append(piece)
        return names

    @staticmethod
    def _split_args(cell: "str | None") -> tuple:
        """Split a ``<br>``-joined Args cell into (shapes, datatypes) string lists."""
        if not cell or cell.strip() in _NULL_CELLS:
            return None, None
        shapes, dtypes = [], []
        for piece in cell.split("<br>"):
            piece = piece.strip()
            if not piece:
                continue
            m = re.match(r"^(\(.*?\))\s*(.*)$", piece)
            if m:
                shapes.append(m.group(1))
                dtypes.append(m.group(2).strip() or None)
            else:
                shapes.append(piece)
                dtypes.append(None)
        if not shapes:
            return None, None
        return shapes, dtypes

    @staticmethod
    def _parse_efficiency(cell: "str | None") -> tuple:
        """Parse ``"X% of Y UNIT"`` into (percent, peak_value, unit); null-safe."""
        if not cell or cell.strip() in _NULL_CELLS:
            return None, None, None
        m = _EFFICIENCY_RE.search(cell)
        if not m:
            return AnalysisMdParser._first_float(cell), None, None
        return float(m.group(1)), float(m.group(2)), m.group(3)

    @staticmethod
    def _bullet_map(section: "str | None") -> dict:
        if not section:
            return {}
        out: dict = {}
        for ln in section.splitlines():
            m = re.match(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.*)$", ln)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
        return out

    @staticmethod
    def _cell_or_null(cell: "str | None") -> "str | None":
        if cell is None or cell.strip() in _NULL_CELLS:
            return None
        return cell.strip()

    @staticmethod
    def _first_float(cell: "str | None") -> "float | None":
        if cell is None or cell.strip() in _NULL_CELLS:
            return None
        m = _FLOAT_RE.search(cell)
        return float(m.group(0)) if m else None

    @staticmethod
    def _int_or_null(cell: "str | None") -> "int | None":
        value = AnalysisMdParser._first_float(cell)
        return int(value) if value is not None else None


class TaskGrouper:
    """Build one tier's ``analysis.json`` tasks from its reasoning-candidate findings.

    Compute buckets members by the raw ``Operation`` cell (the ``kernel_name`` on
    the deterministic-fallback path, where the Operation cell only duplicates the
    kernel name); members sort by ``impact_score`` desc and the task ``impact.mid``
    is the member sum. Fusion makes each finding (md block) one task: members stay
    in table order, carry no ``impact_score``, and ``impact.mid`` is the block's
    ``detail_estimate`` mid. Tasks sort by ``impact.mid`` desc (compute ties break
    on operation then kernel name, fusion ties keep block order, i.e. md rank
    order) and cap at 100.
    """

    def __init__(self, findings: list, tier: str, mode: str):
        self.findings = findings
        self.tier = tier
        self.is_fallback = mode == "deterministic-fallback"

    def group(self) -> list:
        buckets = self._bucket()
        tasks = [self._build_task(bucket) for bucket in buckets.values()]
        return self._finalize(tasks)

    def _bucket(self) -> dict:
        buckets: dict = {}

        for fi, finding in enumerate(self.findings):
            members = finding["members"]
            impacts = finding["impacts"]
            for ri, member in enumerate(members):
                # Marker absent for the block: back-fill pct_e2e * HEURISTIC_FRACTION_MID.
                # Marker present: the cell value, with a ``—`` cell scoring 0.0.
                if impacts is None:
                    row_impact = (member["pct_e2e"] or 0.0) * HEURISTIC_FRACTION_MID
                else:
                    row_impact = impacts[ri] or 0.0

                if self.tier == TIER_FUSION:
                    key, operation = fi, finding["title"]
                elif self.is_fallback:
                    key = member["kernel_name"][0] if member["kernel_name"] else ""
                    operation = None
                else:
                    operation = AnalysisMdParser._cell_or_null(member["operation_cell"])
                    key = operation or ""

                bucket = buckets.setdefault(
                    (finding["node"], key),
                    {
                        "operation": operation,
                        "node": finding["node"],
                        "members": [],
                        "findings": {},
                    },
                )
                bucket["members"].append((self._emit_member(member, row_impact), fi))
                bucket["findings"].setdefault(fi, finding)

        return buckets

    def _emit_member(self, member: dict, row_impact: float) -> dict:
        """The per-row member fields; compute rows lead with their ``impact_score``."""
        fields = {k: member[k] for k in TaskMember.__annotations__}
        if self.tier == TIER_FUSION:
            return fields
        return {"impact_score": row_impact, **fields}

    def _build_task(self, bucket: dict) -> dict:
        entries = bucket["members"]
        if self.tier == TIER_COMPUTE:
            entries.sort(
                key=lambda e: (
                    -e[0]["impact_score"],
                    tuple(e[0]["kernel_name"]),
                    tuple(e[0]["args_shapes"] or []),
                )
            )
        members = [e[0] for e in entries]

        ordered_fi: list = []
        for _, fi in entries:
            if fi not in ordered_fi:
                ordered_fi.append(fi)

        if self.tier == TIER_FUSION:
            mid = bucket["findings"][ordered_fi[0]]["mid"]
        else:
            mid = sum(m["impact_score"] for m in members)

        prose = self._accumulate_prose(ordered_fi, bucket["findings"])

        task = {"operation": bucket["operation"], NODE_COL: bucket["node"]}
        if self.tier == TIER_FUSION:
            task["kind"] = bucket["findings"][ordered_fi[0]]["kind"]
        return {**task, "members": members, "impact": {"mid": mid}, **prose}

    def _finalize(self, tasks: list) -> list:
        # Python's sort is stable, so fusion ties keep block order.
        tasks.sort(key=self._sort_key)

        if len(tasks) > MAX_TASKS:
            logger.info(
                "analysis_json: capping %d %s tasks at %d",
                len(tasks),
                self.tier,
                MAX_TASKS,
            )
            tasks = tasks[:MAX_TASKS]

        for i, task in enumerate(tasks, start=1):
            task["priority"] = i
        return tasks

    def _sort_key(self, task: dict) -> tuple:
        neg_mid = -(task["impact"]["mid"] or 0.0)
        if self.tier == TIER_FUSION:
            return (neg_mid,)
        return (
            neg_mid,
            task["operation"] or "",
            task["node"] or "",
            "".join(task["members"][0]["kernel_name"]) if task["members"] else "",
        )

    @staticmethod
    def _accumulate_prose(ordered_fi: list, findings: dict) -> dict:
        """Concatenate per-finding prose under fixed caps; prefix ``[P<rank>]`` only when merged."""
        caps = {
            "identification": IDENTIFICATION_CAP,
            "reasoning": REASONING_CAP,
            "resolution": RESOLUTION_CAP,
        }
        label = len(ordered_fi) > 1
        out: dict = {}
        truncated = False
        for field, cap in caps.items():
            blocks = []
            for fi in ordered_fi:
                body = findings[fi]["prose"].get(field)
                if not body:
                    continue
                rank = findings[fi]["rank"]
                blocks.append(f"[{rank}] {body}" if (label and rank) else body)
            if not blocks:
                out[field] = None
                continue
            value, clipped = TaskGrouper._clip_blocks(blocks, cap)
            out[field] = value
            truncated = truncated or clipped
        out["prose_truncated"] = truncated
        return out

    @staticmethod
    def _clip_blocks(blocks: list, cap: int) -> tuple:
        """Join blocks with blank lines, dropping whole trailing blocks past ``cap``."""
        kept: list = []
        clipped = False
        for block in blocks:
            candidate = "\n\n".join(kept + [block])
            if len(candidate) <= cap:
                kept.append(block)
            else:
                clipped = True
                break
        if not kept:
            return blocks[0][:cap], True
        return "\n\n".join(kept), clipped
