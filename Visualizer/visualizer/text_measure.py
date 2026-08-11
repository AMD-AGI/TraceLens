"""Matplotlib-backed text and box measurement for diagram layout."""

from __future__ import annotations

from dataclasses import dataclass

from visualizer.sizing import (
    BLOCK_PAD_X,
    BLOCK_PAD_Y,
    LABEL_LINE_GAP,
    SUB_LINE_H,
    TITLE_LINE_H,
    box_width_for_text_width,
    single_line_box_height,
    two_line_box_height,
)


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


def box_label_size(
    ax,
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float,
    sub_fontsize: float | None = None,
) -> tuple[float, float]:
    """Return (width, height) for a detail tile matching ``_draw_box`` geometry."""
    title_w = max(text_width_in_axes(ax, line, fontsize=fontsize) for line in label.split("\n") or [""])
    width = box_width_for_text_width(title_w)
    if not sublabel:
        return width, single_line_box_height()
    sub_fs = sub_fontsize if sub_fontsize is not None else max(6.5, fontsize - 1.5)
    sub_w = max(text_width_in_axes(ax, line, fontsize=sub_fs, fontweight="normal") for line in sublabel.split("\n"))
    width = max(width, box_width_for_text_width(max(sub_w, title_w)))
    return width, two_line_box_height()


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
