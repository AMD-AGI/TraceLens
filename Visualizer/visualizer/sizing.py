"""Content-aware block sizing for diagram nodes."""

from __future__ import annotations

from visualizer.block_tree import BlockNode

PIXELS_PER_UNIT = 80.0
DEFAULT_TITLE_FONT = 7.6
DEFAULT_SUB_FONT = 6.5

BLOCK_PAD_X = 0.04
BLOCK_PAD_Y = 0.02
FRAME_LABEL_PAD_X = 0.08
TITLE_LINE_H = 0.145
SUB_LINE_H = 0.125
LABEL_LINE_GAP = 0.02
SINGLE_LINE_BOX_H = BLOCK_PAD_Y + TITLE_LINE_H + BLOCK_PAD_Y


def single_line_box_height() -> float:
    """Height for a single-line box; matches RMSNorm padding in block diagrams."""
    return SINGLE_LINE_BOX_H


def two_line_box_height() -> float:
    """Height for a title + one sublabel line."""
    return titled_box_height(1)


def titled_box_height(sub_line_count: int) -> float:
    """Height for a title plus ``sub_line_count`` stacked sublabel lines."""
    if sub_line_count <= 0:
        return single_line_box_height()
    sub_block = sub_line_count * SUB_LINE_H + max(0, sub_line_count - 1) * LABEL_LINE_GAP
    return BLOCK_PAD_Y + TITLE_LINE_H + LABEL_LINE_GAP + sub_block + BLOCK_PAD_Y


def box_height_for_content(sublabel: str | None = None) -> float:
    if not sublabel:
        return single_line_box_height()
    sub_lines = [line for line in sublabel.split("\n") if line.strip()]
    return titled_box_height(len(sub_lines))


def box_width_for_text_width(text_width: float, *, pad_x: float | None = None) -> float:
    """Width for a box given rendered text width, using symmetric horizontal padding."""
    pad = BLOCK_PAD_X if pad_x is None else pad_x
    return text_width + 2 * pad


def min_box_width() -> float:
    return box_width_for_text_width(0.0)


def min_vertical_block_gap() -> float:
    """Minimum vertical space between stacked blocks."""
    return BLOCK_PAD_Y + TITLE_LINE_H / 2


def block_top_text_y(node_top: float, *, has_sublabel: bool, title_line_h: float = TITLE_LINE_H) -> float:
    """Y coordinate for the primary label (va=center), measured from block top downward."""
    return node_top - BLOCK_PAD_Y - title_line_h / 2


def block_sublabel(block: BlockNode | None) -> str | None:
    """Build the secondary text shown inside a block."""
    if block is None:
        return None
    from visualizer.block_tree import block_purpose
    from visualizer.ast_analyze import displays_as_linear

    lines: list[str] = []
    purpose = block_purpose(block)
    if purpose:
        lines.append(purpose)
    elif block.class_name and block.class_name != block.label:
        lines.append(block.class_name)
    elif block.details:
        if len(block.details) > 1 and displays_as_linear(block.attr_name, block.class_name):
            return "\n".join(line.strip() for line in block.details if line.strip())
        lines.append(block.details[0])
    return lines[0] if len(lines) == 1 else "\n".join(lines) if lines else None


def estimate_block_size(
    label: str,
    sublabel: str | None = None,
    *,
    fontsize: float = DEFAULT_TITLE_FONT,
    sub_fontsize: float = DEFAULT_SUB_FONT,
) -> tuple[float, float]:
    """Return (width, height) in matplotlib diagram units."""
    pad_x = BLOCK_PAD_X
    pad_y = BLOCK_PAD_Y
    title_line_h = TITLE_LINE_H
    sub_line_h = SUB_LINE_H
    # Fallback widths when matplotlib measurement is unavailable (DejaVu Sans Bold).
    char_w_title = fontsize * 0.0155
    char_w_sub = sub_fontsize * 0.0135

    title_lines = label.split("\n") if label else [""]
    sub_lines = sublabel.split("\n") if sublabel else []

    text_w = 0.0
    for line in title_lines:
        text_w = max(text_w, len(line) * char_w_title)
    for line in sub_lines:
        text_w = max(text_w, len(line) * char_w_sub)

    width = max(min_box_width(), box_width_for_text_width(text_w))
    if label == "SituAndMul" or label == "Gated multiply":
        width += BLOCK_PAD_X
    if sub_lines:
        return width, titled_box_height(len(sub_lines))
    if len(title_lines) == 1:
        return width, single_line_box_height()
    height = pad_y + len(title_lines) * title_line_h + pad_y
    return width, max(single_line_box_height(), height)


def estimate_block_size_for_node(
    block: BlockNode | None,
    label: str | None = None,
    *,
    fontsize: float = DEFAULT_TITLE_FONT,
    sub_fontsize: float = DEFAULT_SUB_FONT,
) -> tuple[float, float]:
    """Size a diagram block from a block node and/or label."""
    display = label or (block.label if block else "")
    return estimate_block_size(display, block_sublabel(block), fontsize=fontsize, sub_fontsize=sub_fontsize)


def to_layout_pixels(width: float, height: float) -> tuple[float, float]:
    """Convert diagram units to graph-layout pixel dimensions."""
    return width * PIXELS_PER_UNIT, height * PIXELS_PER_UNIT
