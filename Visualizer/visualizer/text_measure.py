"""Matplotlib-backed text and box measurement for diagram layout."""

from __future__ import annotations

from dataclasses import dataclass

from visualizer.sizing import (
    BLOCK_PAD_X,
    BLOCK_PAD_Y,
    LABEL_LINE_GAP,
    SUB_LINE_H,
    TITLE_LINE_H,
    box_text_lines,
    box_width_for_text_width,
    single_line_box_height,
    titled_box_height,
)

# Extra inset for white labels that receive a black SVG stroke on colored tiles.
BOX_TEXT_STROKE_PAD = 0.035
# Interior clip from FancyBboxPatch rounding (see render.DETAIL_TILE_*).
TILE_ROUNDING_INSET = 0.088

# Matplotlib default axes span ~1 unit; diagram layout uses 1 data unit ≈ 1 figure inch.
_DEFAULT_UNCONFIGURED_AXIS_SPAN = 1.5


def ensure_diagram_measure_axes(ax) -> None:
    """Align axis limits with figure size so point-sized fonts map to data units."""
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    span_x = x1 - x0
    fig_w = fig.get_figwidth()
    if span_x <= _DEFAULT_UNCONFIGURED_AXIS_SPAN and fig_w > _DEFAULT_UNCONFIGURED_AXIS_SPAN:
        ax.set_xlim(0, fig_w)
        y0, y1 = ax.get_ylim()
        fig_h = fig.get_figheight()
        if y1 - y0 <= _DEFAULT_UNCONFIGURED_AXIS_SPAN and fig_h > _DEFAULT_UNCONFIGURED_AXIS_SPAN:
            ax.set_ylim(0, fig_h)
        fig.canvas.draw()


@dataclass(frozen=True)
class ContentBounds:
    """Axis-aligned bounds in matplotlib data coordinates (y increases upward)."""

    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def overlaps(self, other: ContentBounds, *, min_gap: float = 0.0) -> bool:
        horizontal = self.right + min_gap > other.left and other.right + min_gap > self.left
        vertical = self.bottom - min_gap < other.top and other.bottom - min_gap < self.top
        return horizontal and vertical

    def contains(self, other: ContentBounds, *, min_gap: float = 0.0) -> bool:
        eps = 1e-6
        return (
            other.left >= self.left - min_gap - eps
            and other.right <= self.right + min_gap + eps
            and other.bottom >= self.bottom - min_gap - eps
            and other.top <= self.top + min_gap + eps
        )

    def union(self, other: ContentBounds) -> ContentBounds:
        return ContentBounds(
            left=min(self.left, other.left),
            right=max(self.right, other.right),
            bottom=min(self.bottom, other.bottom),
            top=max(self.top, other.top),
        )


def _ensure_renderer(ax) -> None:
    fig = ax.figure
    if fig.canvas.get_renderer() is None:
        fig.canvas.draw()


def measure_text_bounds(
    ax,
    text: str,
    x: float,
    y: float,
    *,
    fontsize: float,
    ha: str = "left",
    va: str = "bottom",
    fontweight: str = "bold",
    bbox_props: dict | None = None,
) -> ContentBounds:
    """Measure rendered text bounds at a data-coordinate anchor."""
    ensure_diagram_measure_axes(ax)
    _ensure_renderer(ax)
    tmp = ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        alpha=0.0,
        bbox=bbox_props,
    )
    bb = tmp.get_window_extent(ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
    tmp.remove()
    return ContentBounds(left=bb.x0, right=bb.x1, bottom=bb.y0, top=bb.y1)


def text_width_in_axes(ax, text: str, *, fontsize: float, fontweight: str = "bold") -> float:
    return measure_text_bounds(ax, text, 0.0, 0.0, fontsize=fontsize, fontweight=fontweight).width


def measure_stacked_label_bounds(
    ax,
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float,
    sub_fontsize: float | None = None,
) -> ContentBounds:
    """Measure title + sublabel stack using the active matplotlib font."""
    ensure_diagram_measure_axes(ax)
    sub_fontsize = max(6.5, fontsize - 1.5) if sub_fontsize is None else sub_fontsize
    cx = 0.0
    title_bounds = measure_text_bounds(
        ax,
        label,
        cx,
        0.0,
        fontsize=fontsize,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    bounds = title_bounds
    cursor = title_bounds.bottom - LABEL_LINE_GAP
    for line in [entry for entry in (sublabel or "").split("\n") if entry.strip()]:
        line_bounds = measure_text_bounds(
            ax,
            line,
            cx,
            cursor,
            fontsize=sub_fontsize,
            ha="center",
            va="top",
            fontweight="normal",
        )
        bounds = bounds.union(line_bounds)
        cursor = line_bounds.bottom - LABEL_LINE_GAP
    return bounds


def box_label_size(
    ax,
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float,
    sub_fontsize: float | None = None,
    pad_x: float | None = None,
    pad_y: float | None = None,
    white_text_stroke_pad: bool = True,
) -> tuple[float, float]:
    """Return (width, height) for a detail tile matching ``_draw_box`` geometry."""
    ensure_diagram_measure_axes(ax)
    box_pad_x = BLOCK_PAD_X if pad_x is None else pad_x
    box_pad_y = BLOCK_PAD_Y if pad_y is None else pad_y
    ref_top = 10.0
    cx = 0.0
    text_bounds = measure_stacked_label_bounds(
        ax,
        label,
        sublabel,
        fontsize=fontsize,
        sub_fontsize=sub_fontsize,
    )
    height = (text_bounds.top - text_bounds.bottom) + 2 * box_pad_y
    for _ in range(2):
        text_bounds = None
        for line in box_text_lines(
            ref_top,
            height,
            label,
            sublabel,
            pad_y=box_pad_y,
            title_fontsize=fontsize,
        ):
            line_bounds = measure_text_bounds(
                ax,
                line.text,
                cx,
                line.y,
                fontsize=line.fontsize,
                ha="center",
                va=line.va,
                fontweight=line.fontweight,
            )
            text_bounds = line_bounds if text_bounds is None else text_bounds.union(line_bounds)
        assert text_bounds is not None
        needed = (text_bounds.top - text_bounds.bottom) + 2 * box_pad_y
        if abs(needed - height) < 1e-6:
            break
        height = needed

    stroke_pad = BOX_TEXT_STROKE_PAD if white_text_stroke_pad else 0.0
    width = box_width_for_text_width(
        text_bounds.width + 2 * (stroke_pad + TILE_ROUNDING_INSET),
        pad_x=box_pad_x,
    )
    return width, max(
        single_line_box_height(pad_y=box_pad_y),
        height + 2 * (stroke_pad + TILE_ROUNDING_INSET),
    )


def input_box_label_size(
    ax,
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float = 7.2,
    sub_fontsize: float | None = None,
) -> tuple[float, float]:
    """Measure an input tile from rendered text using detail-tile padding."""
    return box_label_size(
        ax,
        label,
        sublabel,
        fontsize=fontsize,
        sub_fontsize=sub_fontsize,
        white_text_stroke_pad=False,
    )


def tensor_port_box_label_size(
    ax,
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float = 7.0,
    sub_fontsize: float = 5.4,
) -> tuple[float, float]:
    """Measure a compact tensor-port tile (title + upstream hint)."""
    from visualizer.sizing import INPUT_PAD_X, TENSOR_PORT_PAD_Y

    return box_label_size(
        ax,
        label,
        sublabel,
        fontsize=fontsize,
        sub_fontsize=sub_fontsize,
        pad_x=INPUT_PAD_X,
        pad_y=TENSOR_PORT_PAD_Y,
    )


def box_bounds_at(cx: float, top_y: float, width: float, height: float, *, visual_inset: float = 0.0) -> ContentBounds:
    """Bounds of a rounded tile centered at ``cx`` with top edge ``top_y``."""
    half_w = width / 2
    bottom = top_y - height
    return ContentBounds(
        left=cx - half_w - visual_inset,
        right=cx + half_w + visual_inset,
        bottom=bottom - visual_inset,
        top=top_y + visual_inset,
    )


def floating_port_label_bounds(
    ax,
    label: str,
    x: float,
    y: float,
    *,
    ha: str = "right",
    va: str = "center",
    fontsize: float = 7.2,
    detail_fill: str = "#fff5f4",
) -> ContentBounds:
    """Bounds of a floating port label drawn like ``_draw_floating_port_label``."""
    return measure_text_bounds(
        ax,
        label,
        x,
        y,
        fontsize=fontsize,
        ha=ha,
        va=va,
        fontweight="bold",
        bbox_props={
            "boxstyle": "round,pad=0.10",
            "facecolor": detail_fill,
            "edgecolor": "none",
            "alpha": 1.0,
        },
    )
