from llm_cost.table import format_int, format_money, render_table


def test_format_int_groups_thousands():
    assert format_int(12000) == "12,000"
    assert format_int(0) == "0"
    assert format_int(999) == "999"


def test_format_money_pads_to_four_decimal_places():
    assert format_money(0.06) == "$0.0600"
    assert format_money(1234.5) == "$1,234.5000"


def test_render_table_aligns_first_column_left_and_rest_right():
    headers = ["model", "calls", "cost"]
    rows = [["claude-opus-5", "2", "$0.3337"], ["gpt-4o-mini", "1", "$0.0031"]]
    rendered = render_table(headers, rows).splitlines()

    # widths are the max of header and cell lengths in each column: 13, 5, 7
    assert rendered[0] == "  ".join(["model".ljust(13), "calls".rjust(5), "cost".rjust(7)])
    assert rendered[2] == "  ".join(["claude-opus-5".ljust(13), "2".rjust(5), "$0.3337".rjust(7)])
    assert rendered[3] == "  ".join(["gpt-4o-mini".ljust(13), "1".rjust(5), "$0.0031".rjust(7)])


def test_render_table_separator_matches_column_widths():
    headers = ["a", "bb"]
    rows = [["x", "yy"]]
    rendered = render_table(headers, rows).splitlines()
    assert rendered[1] == "-  --"


def test_render_table_widens_column_to_widest_cell():
    headers = ["model", "cost"]
    rows = [["claude-opus-5", "$0.01"]]
    rendered = render_table(headers, rows).splitlines()
    assert rendered[0] == "  ".join(["model".ljust(13), "cost".rjust(5)])


def test_render_table_short_row_omits_trailing_columns_instead_of_padding():
    headers = ["model", "calls", "cost"]
    rows = [["claude-opus-5", "2", "$0.3337"], ["TOTAL", "3"]]
    rendered = render_table(headers, rows).splitlines()
    assert rendered[-1] == "  ".join(["TOTAL".ljust(13), "3".rjust(5)])


def test_render_table_with_no_rows_still_renders_header_and_separator():
    rendered = render_table(["a", "b"], []).splitlines()
    assert rendered == ["a  b", "-  -"]
