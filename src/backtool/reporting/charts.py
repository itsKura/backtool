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
    quantity: str = "return",
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

        text = _format_value(item.value, value_format)
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
        f'text-anchor="end">{escape(quantity)} ({escape(unit)})</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


#: Distribution-chart geometry.
DIST_PLOT_HEIGHT = 186
DIST_TOP_PADDING = 26
DIST_BOTTOM_PADDING = 42
DIST_SIDE_PADDING = 20
DIST_BAR_GAP = 3
#: Bins are capped so a wide-tailed sample cannot produce a comb of hairlines.
DIST_TARGET_BINS = 9
DIST_MAX_BINS = 24


def distribution_chart(
    values: list[float],
    *,
    median: float | None = None,
    unit: str = "%",
    quantity: str = "return",
    width: int = 720,
) -> str:
    """Render the shape of a set of returns as a histogram.

    A reader scanning twenty individual bars has to do the summarising
    themselves. A histogram answers the three questions they actually have --
    how spread out were the outcomes, did they lean positive or negative, were
    there outliers -- in one glance.

    Bins are aligned so that a bin **edge** always falls exactly on zero. A bin
    spanning -0.2% to +0.3% would have to be coloured as one sign or the other,
    which would misreport the very thing the colour encodes.

    Args:
        values: One return per observation, in percent.
        median: Marked with a labelled rule when given. Passed in rather than
            computed here: every statistic on the page comes from the
            aggregation layer, and a chart recomputing one would be a second
            source of truth for it.
        unit: Appended to the axis caption.
        width: Total SVG width in pixels.

    Returns:
        An ``<svg>`` element using the same CSS custom properties as the bar
        chart, so both follow the page's theme.
    """
    if not values:
        return _empty_chart(width)

    step = _bin_step(values)
    low_index = int(_floor_to(min(min(values), 0.0), step) // step)
    high_index = int(max(max(values), 0.0) // step)
    counts = _bin_counts(values, step, low_index, high_index)

    bin_count = len(counts)
    plot_left = DIST_SIDE_PADDING
    plot_width = width - 2 * DIST_SIDE_PADDING
    height = DIST_TOP_PADDING + DIST_PLOT_HEIGHT + DIST_BOTTOM_PADDING
    baseline_y = DIST_TOP_PADDING + DIST_PLOT_HEIGHT

    slot = plot_width / bin_count
    tallest = max(counts)

    def x_of(value: float) -> float:
        """Map a return to a pixel, using the bin grid as the scale."""
        return plot_left + (value / step - low_index) * slot

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMin meet" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" class="chart">'
    ]

    parts.append(
        f'<line x1="{plot_left}" y1="{baseline_y}" x2="{plot_left + plot_width}" '
        f'y2="{baseline_y}" class="gridline" />'
    )

    # Label every edge when they fit, otherwise every other one. Crowded axis
    # labels overlap into illegibility, which is worse than fewer of them.
    label_every = 1 if bin_count <= 10 else 2

    for index, count in enumerate(counts):
        bin_low = (low_index + index) * step
        bin_high = bin_low + step
        bar_x = plot_left + index * slot + DIST_BAR_GAP / 2
        bar_width = slot - DIST_BAR_GAP
        bar_height = 0.0 if tallest == 0 else count / tallest * DIST_PLOT_HEIGHT
        series_class = "bar-up" if low_index + index >= 0 else "bar-down"

        if index % label_every == 0:
            parts.append(
                f'<text x="{plot_left + index * slot:.1f}" y="{baseline_y + 18}" '
                f'class="axis-label" text-anchor="middle">{bin_low:g}</text>'
            )

        if count == 0:
            continue

        plural = "" if count == 1 else "s"
        title = (
            f"{count} event{plural} between {bin_low:g}{unit} and {bin_high:g}{unit}"
        )
        parts.append(f'<g class="bar-group"><title>{escape(title)}</title>')
        parts.append(
            f'<path d="{_column_path(bar_x, baseline_y, bar_width, bar_height)}" '
            f'class="{series_class}" />'
        )
        parts.append(
            f'<text x="{bar_x + bar_width / 2:.1f}" y="{baseline_y - bar_height - 7:.1f}" '
            f'class="value-label" text-anchor="middle">{count}</text>'
        )
        parts.append("</g>")

    # Rightmost edge, which the per-bin loop never reaches.
    if (bin_count - 1) % label_every == 0 or bin_count <= 10:
        parts.append(
            f'<text x="{plot_left + plot_width:.1f}" y="{baseline_y + 18}" '
            f'class="axis-label" text-anchor="middle">'
            f"{(high_index + 1) * step:g}</text>"
        )

    zero_x = x_of(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{DIST_TOP_PADDING - 10}" x2="{zero_x:.1f}" '
        f'y2="{baseline_y + 4}" class="baseline" />'
    )

    if median is not None:
        median_x = min(max(x_of(median), plot_left), plot_left + plot_width)
        anchor = "start" if median_x < plot_left + plot_width * 0.75 else "end"
        offset = 6 if anchor == "start" else -6
        parts.append(
            f'<line x1="{median_x:.1f}" y1="{DIST_TOP_PADDING - 10}" '
            f'x2="{median_x:.1f}" y2="{baseline_y}" class="median-line" />'
        )
        parts.append(
            f'<text x="{median_x + offset:.1f}" y="{DIST_TOP_PADDING - 13}" '
            f'class="median-label" text-anchor="{anchor}">'
            f"median {_format_value(median)}{escape(unit)}</text>"
        )

    parts.append(
        f'<text x="{plot_left}" y="{height - 6}" class="axis-caption" '
        f'text-anchor="start">events per bucket</text>'
    )
    parts.append(
        f'<text x="{width - DIST_SIDE_PADDING}" y="{height - 6}" class="axis-caption" '
        f'text-anchor="end">{escape(quantity)} ({escape(unit)})</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _bin_step(values: list[float]) -> float:
    """Choose a round bucket width covering the sample in about nine buckets.

    Rounded to a 1/2/2.5/5 multiple so bucket edges read as numbers a person
    would choose, and so the edge on zero lines up with the axis labels.
    """
    span = max(max(values), 0.0) - min(min(values), 0.0)
    if span <= 0:
        return 1.0

    raw = span / DIST_TARGET_BINS
    # Cast: a negative exponent makes ``10 ** n`` a float, so the literal power
    # is typed Any and would leak out through the return.
    magnitude = float(10 ** _floor_log10(raw))
    step = magnitude
    for multiplier in (1, 2, 2.5, 5, 10):
        step = magnitude * multiplier
        if step >= raw:
            break

    # Guard against a lone outlier stretching the range into hundreds of bins.
    while span / step > DIST_MAX_BINS:
        step *= 2
    return step


def _bin_counts(
    values: list[float], step: float, low_index: int, high_index: int
) -> list[int]:
    """Count observations per bucket, indexing by ``floor(value / step)``.

    Integer division rather than comparing against accumulated edges: adding a
    float step repeatedly drifts, and a value landing a hair off an edge would
    be counted into the wrong bucket -- on the wrong side of zero, in the worst
    case.
    """
    counts = [0] * (high_index - low_index + 1)
    for value in values:
        index = int(value // step) - low_index
        counts[min(max(index, 0), len(counts) - 1)] += 1
    return counts


def _column_path(x: float, baseline: float, width: float, height: float) -> str:
    """A column rounded at the top and square where it meets the axis."""
    if height <= 0:
        return ""
    radius = min(3.0, width / 2, height)
    return (
        f"M {x:.1f},{baseline:.1f} "
        f"V {baseline - height + radius:.1f} "
        f"A {radius:.1f},{radius:.1f} 0 0 1 {x + radius:.1f},{baseline - height:.1f} "
        f"H {x + width - radius:.1f} "
        f"A {radius:.1f},{radius:.1f} 0 0 1 {x + width:.1f},{baseline - height + radius:.1f} "
        f"V {baseline:.1f} Z"
    )


def _format_value(value: float, value_format: str = "{:+.2f}") -> str:
    """Apply a label format, collapsing a signed zero.

    ``{:+.2f}`` renders -0.004 as ``-0.00``, which reads as a typo and asserts
    a direction the printed digits do not show. The check is on the formatted
    text rather than the raw value because the format string is what decides
    how much precision the reader actually sees.
    """
    text = value_format.format(value)
    try:
        if float(text) == 0:
            return value_format.format(0.0).lstrip("+-")
    except ValueError:  # a format carrying non-numeric characters
        pass
    return text


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
