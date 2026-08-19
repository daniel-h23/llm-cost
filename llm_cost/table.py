"""Render aligned tables for console output.

The first column is treated as a label and left-justified; every other
column holds a number and is right-justified, the way a ledger lines up
names on the left and figures on the right. A row may have fewer cells
than there are headers -- the TOTAL row of a report has no per-call
figure -- in which case the missing trailing columns are simply absent
from that line rather than padded out with spaces.
"""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(n):
    """Thousands-grouped integer, e.g. ``12000`` -> ``"12,000"``."""
    return "{:,}".format(n)


def format_money(x):
    """Dollar amount to four decimal places, e.g. ``0.06`` -> ``"$0.0600"``."""
    return "${:,.4f}".format(x)


def render_table(headers, rows):
    """Render ``headers`` and ``rows`` (lists of strings) as one string.

    Column widths are the max of the header and every cell seen in that
    column, so a table renders correctly even before all rows are known.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(cells):
        parts = []
        for index, cell in enumerate(cells):
            width = widths[index]
            parts.append(cell.ljust(width) if index == 0 else cell.rjust(width))
        return "  ".join(parts).rstrip()

    lines = [render_row(headers), "  ".join("-" * width for width in widths)]
    for row in rows:
        lines.append(render_row(row))
    return "\n".join(lines)
