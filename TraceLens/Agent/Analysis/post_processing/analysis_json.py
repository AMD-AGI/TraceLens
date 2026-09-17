###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic ``analysis.json`` renderer, driven purely from ``analysis.md``.

TraceLens authors the human-readable ``analysis.md`` perf report; this module
reads only that md and emits a structured, operation-grouped ``analysis.json``
beside it (compute tier only, v1). The parse anchors on the ``MarkerValidator``
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
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    MarkerValidator,
    _find_data_table,
    _iter_compute_candidate_blocks,
)

logger = logging.getLogger(__name__)

# Constants
MAX_COMPUTE_TASKS = 100
IDENTIFICATION_CAP = 2000
REASONING_CAP = 2000
RESOLUTION_CAP = 2000

_TITLE_RE = re.compile(r"^#\s+(.*?)\s*$", re.MULTILINE)
_LABEL_RE = re.compile(
    r"\*\*(Identification|Reasoning for Slowdown|Resolution):\*\*\s*(.+?)"
    r"(?=\n\s*\n|\n\*\*|\Z)",
    re.DOTALL,
)
_EFFICIENCY_RE = re.compile(r"([\d.]+)%\s*of\s*([\d.]+)\s*(\S+)")
_ANCHOR_LINK_RE = re.compile(r"#detailed-analysis-compute-p(\d+)")
_CATEGORY_ATTR_RE = re.compile(r"\bcategory=(\S+)")
_LIBRARY_PARENS_RE = re.compile(r"\(([^()]+)\)\s*$")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NULL_CELLS = {"", "-", "—", "–"}

# Data-table header labels, normalized (lowercased, stripped) -> logical field.
_HEADER_ALIASES = {
    "operation": "operation",
    "args": "args",
    "kernel path": "kernel_path",
    "kernel name": "kernel_name",
    "time (ms)": "time_ms",
    "%e2e": "pct_e2e",
    "count": "count",
    "flops/byte": "flops_per_byte",
    "efficiency": "efficiency",
    "bound": "bound",
}

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

    findings = parser.findings()
    report["compute_optimizations"] = ComputeGrouper(findings, mode).group()

    appendix = parser.appendix()
    if appendix is not None:
        report["appendix"] = appendix

    out_path = md_path.with_name("analysis.json")
    out_path.write_text(
        json.dumps(report, sort_keys=False, indent=2, ensure_ascii=False) + "\n"
    )
    return out_path


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
        for ln in section.splitlines():
            stripped = ln.strip()
            if stripped.startswith("|"):
                cells = self._split_row(stripped)
                if len(cells) >= 2:
                    field = _METRIC_FIELDS.get(cells[0].strip().lower())
                    if field:
                        metrics[field] = self._first_float(cells[1])
            elif (
                stripped
                and not stripped.startswith("!")
                and not stripped.startswith("#")
            ):
                narrative_parts.append(stripped)
        narrative = " ".join(narrative_parts).strip()
        if not narrative and not metrics:
            return None
        return {"narrative": narrative, "metrics": metrics}

    def top_operations(self) -> "list[dict] | None":
        block = self._marker_block("top_ops")
        if block is None:
            return None
        rows = []
        for cells in self._iter_table_rows(block):
            if len(cells) < 6:
                continue
            rows.append(
                {
                    "rank": int(self._first_float(cells[0]) or 0),
                    "category": self._cell_or_null(cells[1]),
                    "time_ms": self._first_float(cells[2]),
                    "pct_of_compute": self._first_float(cells[3]),
                    "ops": self._int_or_null(cells[4]),
                    "potential_improvement_md": self._cell_or_null(cells[5]),
                }
            )
        return rows if rows else None

    def findings(self) -> list:
        """Parse every compute reasoning-candidate block into finding dicts."""
        category_by_rank = self._card_categories()
        return [
            self._parse_finding(start, end, category_by_rank)
            for start, end in _iter_compute_candidate_blocks(self.text)
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

    def _parse_finding(self, start: int, end: int, category_by_rank: dict) -> dict:
        """Parse one compute reasoning-candidate block into a finding dict.

        ``category_by_rank`` joins each finding to its card ``p_item`` category via
        the compute rank; the category is per-finding and is stamped on every
        member.
        """
        lines = self.lines
        block = "\n".join(lines[start:end])

        rank_m = re.search(
            r"reasoning-candidate\s+tier=compute\s+rank=(\d+)", lines[start]
        )
        md_rank = f"P{rank_m.group(1)}" if rank_m else None
        category = category_by_rank.get(int(rank_m.group(1))) if rank_m else None

        heading = next((ln for ln in lines[start:end] if ln.startswith("####")), None)
        lib_m = _LIBRARY_PARENS_RE.search(heading) if heading else None
        library = lib_m.group(1).strip() if lib_m else None

        prose = {"identification": None, "reasoning": None, "resolution": None}
        for label, body in _LABEL_RE.findall(block):
            cleaned = " ".join(body.split())
            if label == "Identification":
                prose["identification"] = cleaned
            elif label == "Reasoning for Slowdown":
                prose["reasoning"] = cleaned
            else:
                prose["resolution"] = cleaned

        members = self._parse_data_table(start, end)
        for member in members:
            member["category"] = category
            member["library"] = library
            member["analysis_md_rank"] = md_rank
        impacts = self._parse_op_row_marker(block)
        low, high = self._parse_finding_range(block)
        return {
            "members": members,
            "impacts": impacts,
            "prose": prose,
            "rank": md_rank,
            "low": low,
            "high": high,
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

        members = []
        for _, cells in row_iter:

            def get(field: str) -> "str | None":
                idx = col.get(field)
                return cells[idx] if idx is not None and idx < len(cells) else None

            operation_cell = get("operation")
            eff_pct, eff_peak, eff_unit = self._parse_efficiency(get("efficiency"))
            args_shapes, args_datatypes = self._split_args(get("args"))
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
                    "time_ms": self._first_float(get("time_ms")),
                    "count": self._int_or_null(get("count")),
                    "pct_e2e": self._first_float(get("pct_e2e")),
                    "flops_per_byte": self._first_float(get("flops_per_byte")),
                    "efficiency_percent": eff_pct,
                    "efficiency_peak_value": eff_peak,
                    "efficiency_peak_unit": eff_unit,
                    "bound": self._cell_or_null(get("bound")),
                }
            )
        return members

    def _card_categories(self) -> dict:
        """Map compute rank -> card ``p_item`` category via the detailed anchor.

        Each compute card carries a ``kind=p_item category=<cat>`` marker followed
        by a ``#detailed-analysis-compute-p{N}`` link; the anchor's N is the rank
        the detailed block is keyed on. Cards without a category or anchor are
        skipped, so findings with no matching card (e.g. fallback reports) keep
        ``category=null``.
        """
        mapping: dict = {}
        for m in MarkerValidator.BEGIN_RE.finditer(self.text):
            inner = m.group(1)
            km = MarkerValidator.KIND_ATTR_RE.search(inner)
            if not km or km.group(1) != "p_item":
                continue
            cat_m = _CATEGORY_ATTR_RE.search(inner)
            if not cat_m:
                continue
            end = MarkerValidator.END_RE.search(self.text, m.end())
            tail = self.text[end.end() : end.end() + 400] if end else ""
            link_m = _ANCHOR_LINK_RE.search(tail)
            if link_m:
                mapping[int(link_m.group(1))] = cat_m.group(1)
        return mapping

    def _parse_op_row_marker(self, block: str) -> "list[float | None] | None":
        """Return the ``kind=op_row`` ``impacts=`` CSV, one entry per row.

        A ``—`` cell (spec-legal null) maps to ``None`` so the reader can
        fall back to that row's ``pct_e2e``; the whole marker is None when absent.
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

    def _parse_finding_range(self, block: str) -> tuple:
        """Return (low, high) E2E-% range from the detail_estimate/p_item marker."""
        for kind in ("detail_estimate", "p_item"):
            for m in MarkerValidator.BEGIN_RE.finditer(block):
                inner = m.group(1)
                km = MarkerValidator.KIND_ATTR_RE.search(inner)
                if not km or km.group(1) != kind:
                    continue
                attrs = dict(MarkerValidator.ATTR_RE.findall(inner))
                low, high = attrs.get("low"), attrs.get("high")
                if low in (None, "null") or high in (None, "null"):
                    continue
                return float(low), float(high)
        return None, None

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

    def _marker_block(self, kind: str) -> "str | None":
        """Return the text between an ``impact-begin kind=<kind>`` and its end."""
        text = self.text
        for m in MarkerValidator.BEGIN_RE.finditer(text):
            km = MarkerValidator.KIND_ATTR_RE.search(m.group(1))
            if not km or km.group(1) != kind:
                continue
            end = MarkerValidator.END_RE.search(text, m.end())
            return text[m.end() : end.start()] if end else text[m.end() :]
        return None

    def _iter_table_rows(self, block: str):
        """Yield body-row cell lists from the first markdown table in ``block``."""
        seen_header = False
        for ln in block.splitlines():
            stripped = ln.strip()
            if not stripped.startswith("|"):
                if seen_header:
                    break
                continue
            cells = self._split_row(stripped)
            if not seen_header:
                seen_header = True
                continue
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            yield cells

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


class ComputeGrouper:
    """Bucket compute findings into operation-grouped tasks.

    Holds ``is_fallback`` (the only mode-conditional) so bucketing, task build,
    and finalize share it. Members are one-per-row; the group key is the raw
    ``Operation`` cell, or the ``kernel_name`` on the deterministic-fallback path
    where the Operation cell only duplicates the kernel name. Members sort
    by ``impact_score`` desc, task ``impact.mid`` is the member sum, tasks sort by
    ``impact.mid`` desc and cap at 100.
    """

    def __init__(self, findings: list, mode: str):
        self.findings = findings
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
                row_impact = impacts[ri] if impacts is not None else None
                score = (
                    row_impact
                    if row_impact is not None
                    else (member["pct_e2e"] or 0.0) * HEURISTIC_FRACTION_MID
                )

                if self.is_fallback:
                    key = member["kernel_name"][0] if member["kernel_name"] else ""
                    operation = None
                else:
                    operation = AnalysisMdParser._cell_or_null(member["operation_cell"])
                    key = operation or ""

                emitted = {
                    "impact_score": score,
                    "kernel_launcher_path": member["kernel_launcher_path"],
                    "library": member["library"],
                    "category": member["category"],
                    "analysis_md_rank": member["analysis_md_rank"],
                    "kernel_name": member["kernel_name"],
                    "args_shapes": member["args_shapes"],
                    "args_datatypes": member["args_datatypes"],
                    "time_ms": member["time_ms"],
                    "count": member["count"],
                    "pct_e2e": member["pct_e2e"],
                    "flops_per_byte": member["flops_per_byte"],
                    "efficiency_percent": member["efficiency_percent"],
                    "efficiency_peak_value": member["efficiency_peak_value"],
                    "efficiency_peak_unit": member["efficiency_peak_unit"],
                    "bound": member["bound"],
                }
                bucket = buckets.setdefault(
                    key, {"operation": operation, "members": [], "findings": {}}
                )
                bucket["members"].append((emitted, fi))
                bucket["findings"].setdefault(fi, finding)

        return buckets

    def _build_task(self, bucket: dict) -> dict:
        entries = bucket["members"]
        entries.sort(
            key=lambda e: (
                -e[0]["impact_score"],
                tuple(e[0]["kernel_name"]),
                tuple(e[0]["args_shapes"] or []),
            )
        )
        members = [e[0] for e in entries]
        mid = sum(m["impact_score"] for m in members)

        ordered_fi: list = []
        for _, fi in entries:
            if fi not in ordered_fi:
                ordered_fi.append(fi)

        low = self._sum_optional(bucket["findings"][fi]["low"] for fi in ordered_fi)
        high = self._sum_optional(bucket["findings"][fi]["high"] for fi in ordered_fi)

        prose = self._accumulate_prose(ordered_fi, bucket["findings"])

        return {
            "operation": bucket["operation"],
            "members": members,
            "impact": {"mid": mid, "low": low, "high": high},
            **prose,
        }

    def _finalize(self, tasks: list) -> list:
        tasks.sort(
            key=lambda t: (
                -t["impact"]["mid"],
                t["operation"] or "",
                "".join(t["members"][0]["kernel_name"]) if t["members"] else "",
            )
        )

        if len(tasks) > MAX_COMPUTE_TASKS:
            logger.info(
                "analysis_json: capping %d compute tasks at %d",
                len(tasks),
                MAX_COMPUTE_TASKS,
            )
            tasks = tasks[:MAX_COMPUTE_TASKS]

        for i, task in enumerate(tasks, start=1):
            task["priority"] = i
        return tasks

    def _accumulate_prose(self, ordered_fi: list, findings: dict) -> dict:
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
            value, clipped = self._clip_blocks(blocks, cap)
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

    @staticmethod
    def _sum_optional(values) -> "float | None":
        present = [v for v in values if v is not None]
        return sum(present) if present else None
