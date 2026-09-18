"""Inline SVG chart generation.

Hand-rolled SVG rather than matplotlib, for two reasons. It keeps the report a
single self-contained file with no image assets, and it adds no native
dependency -- which matters here, since Application Control on the target
machine already blocks pyarrow's and Git's bundled binaries.

Colour follows the diverging job: returns are polarity data (above or below
zero), so the palette is a two-hue diverging pair with a neutral midpoint.
Deliberately **blue/red rather than green/red**: red-green is the most common
colour-vision deficiency, affecting roughly 8% of men, and green/red is the
convention most likely to be unreadable. Blue/red carries the same
opposite-poles meaning legibly for everyone.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

#: Chart geometry, fixed across every chart in a report.
BAR_HEIGHT = 18  # <= 24px cap; the band's leftover is air
ROW_HEIGHT = 26
CORNER_RADIUS = 4  # rounded data-end, square at the baseline
LEFT_GUTTER = 148
RIGHT_GUTTER = 68
TOP_PADDING = 28
BOTTOM_PADDING = 34


@dataclass(frozen=True)
class BarItem:
    """One row of a diverging bar chart."""

    label: str
    value: float
    #: Extra detail surfaced on hover.
    tooltip: str = ""


def diverging_bar_chart(
    items: list[BarItem],
    *,
    unit: str = "%",
    value_format: str = "{:+.2f}",
    width: int = 720,
) -> str:
    """Render a horizontal diverging bar chart as inline SVG.

    Horizontal rather than vertical because the category labels are event ids,
    which do not fit under a column without rotation.

    Args:
        items: Rows, in the order they should appear top to bottom.
        unit: Appended to the axis caption.
        value_format: Format string for direct labels.
        width: Total SVG width in pixels.

    Returns:
        An ``<svg>`` element. Colours are referenced as CSS custom properties so
        the surrounding page controls light/dark in one place.
    """
    if not items:
        return _empty_chart(width)

    plot_left = LEFT_GUTTER
    plot_width = width - LEFT_GUTTER - RIGHT_GUTTER
    height = TOP_PADDING + len(items) * ROW_HEIGHT + BOTTOM_PADDING

    values = [item.value for item in items]
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    span = high - low
    if span == 0:
        low, high, span = -1.0, 1.0, 2.0
    else:  # breathing room so the longest bar never touches the edge
        low -= span * 0.08
        high += span * 0.08
        span = high - low

    def x_of(value: float) -> float:
        return plot_left + (value - low) / span * plot_width

    zero_x = x_of(0.0)
    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMin meet" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
    ]

    # Gridlines: hairline, solid, recessive. Ticks at rounded values.
    for tick in _ticks(low, high):
        tx = x_of(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{TOP_PADDING - 8}" x2="{tx:.1f}" '
            f'y2="{height - BOTTOM_PADDING + 4}" class="gridline" />'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{height - BOTTOM_PADDING + 20}" '
            f'class="axis-label" text-anchor="middle">{tick:g}</text>'
        )

    for index, item in enumerate(items):
        y = TOP_PADDING + index * ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2
        bar_x = x_of(item.value)
        positive = item.value >= 0
        series_class = "bar-up" if positive else "bar-down"

        text = value_format.format(item.value)
        label_x, anchor, inside = _place_value_label(
            text, bar_x, zero_x, positive, plot_left, plot_left + plot_width
        )
        label_class = "value-label-inside" if inside else "value-label"
        title = escape(item.tooltip or f"{item.label}: {text}{unit}")

        parts.append(f'<g class="bar-group"><title>{title}</title>')
        parts.append(
            f'<text x="{LEFT_GUTTER - 12}" y="{y + BAR_HEIGHT / 2 + 4:.1f}" '
            f'class="row-label" text-anchor="end">{escape(item.label)}</text>'
        )
        parts.append(
            f'<path d="{_bar_path(zero_x, bar_x, y, BAR_HEIGHT)}" class="{series_class}" />'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + BAR_HEIGHT / 2 + 4:.1f}" '
            f'class="{label_class}" text-anchor="{anchor}">{escape(text)}</text>'
        )
        parts.append("</g>")

    # Baseline drawn last so it sits above the fills.
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{TOP_PADDING - 8}" x2="{zero_x:.1f}" '
        f'y2="{height - BOTTOM_PADDING + 4}" class="baseline" />'
    )
    parts.append(
        f'<text x="{width - 4}" y="{height - 6}" class="axis-caption" '
        f'text-anchor="end">return ({escape(unit)})</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


#: Approximate advance width of one character at the 11.5px monospace label size.
#: Monospace makes this a reliable measurement rather than a guess.
_LABEL_CHAR_WIDTH = 6.6
_LABEL_PAD = 8.0


def _place_value_label(
    text: str,
    bar_x: float,
    zero_x: float,
    positive: bool,
    plot_left: float,
    plot_right: float,
) -> tuple[float, str, bool]:
    """Position a bar's value label, moving it inside the bar if it would not fit.

    A label placed beyond the tip of a long bar runs into the row label on the
    left or off the canvas on the right. Rather than let it collide or clip, it
    flips to sit just inside the bar end, where it is drawn in the surface
    colour so it stays legible against the fill.

    Returns:
        ``(x, text-anchor, is_inside)``.
    """
    width = len(text) * _LABEL_CHAR_WIDTH

    if positive:
        outside_x = bar_x + _LABEL_PAD
        if outside_x + width <= plot_right:
            return outside_x, "start", False
        # Only flip inside when the bar is actually wide enough to hold it.
        if bar_x - zero_x >= width + 2 * _LABEL_PAD:
            return bar_x - _LABEL_PAD, "end", True
        return outside_x, "start", False

    outside_x = bar_x - _LABEL_PAD
    if outside_x - width >= plot_left:
        return outside_x, "end", False
    if zero_x - bar_x >= width + 2 * _LABEL_PAD:
        return bar_x + _LABEL_PAD, "start", True
    return outside_x, "end", False


def _bar_path(zero_x: float, value_x: float, y: float, height: float) -> str:
    """A bar rounded on its data end and square at the baseline.

    Rounding only the far end keeps the baseline reading as a hard reference
    line; rounding both ends detaches the bar from zero.
    """
    radius = min(CORNER_RADIUS, abs(value_x - zero_x))
    bottom = y + height

    if value_x >= zero_x:
        return (
            f"M {zero_x:.1f},{y:.1f} "
            f"H {value_x - radius:.1f} "
            f"A {radius:.1f},{radius:.1f} 0 0 1 {value_x:.1f},{y + radius:.1f} "
            f"V {bottom - radius:.1f} "
            f"A {radius:.1f},{radius:.1f} 0 0 1 {value_x - radius:.1f},{bottom:.1f} "
            f"H {zero_x:.1f} Z"
        )
    return (
        f"M {zero_x:.1f},{y:.1f} "
        f"H {value_x + radius:.1f} "
        f"A {radius:.1f},{radius:.1f} 0 0 0 {value_x:.1f},{y + radius:.1f} "
        f"V {bottom - radius:.1f} "
        f"A {radius:.1f},{radius:.1f} 0 0 0 {value_x + radius:.1f},{bottom:.1f} "
        f"H {zero_x:.1f} Z"
    )


def _ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Round tick values spanning the range, so the axis reads in clean numbers."""
    span = high - low
    if span <= 0:
        return [0.0]

    raw_step = span / target
    magnitude = 10 ** _floor_log10(raw_step)
    for multiplier in (1, 2, 2.5, 5, 10):
        step = magnitude * multiplier
        if step >= raw_step:
            break

    ticks: list[float] = []
    current = _floor_to(low, step)
    while current <= high + step * 0.001:
        if low <= current <= high:
            ticks.append(round(current, 10))
        current += step
    return ticks or [0.0]


def _floor_log10(value: float) -> int:
    exponent = 0
    magnitude = abs(value)
    if magnitude == 0:
        return 0
    while magnitude < 1:
        magnitude *= 10
        exponent -= 1
    while magnitude >= 10:
        magnitude /= 10
        exponent += 1
    return exponent


def _floor_to(value: float, step: float) -> float:
    return (value // step) * step


def _empty_chart(width: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} 80" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
        f'<text x="{width / 2}" y="44" class="axis-label" text-anchor="middle">'
        f"no usable observations</text></svg>"
    )
