"""Command-line entry point: estimate, report, compare, models.

Exit codes are part of the contract (see README): 0 for success, 2 for bad
input the caller can fix (a bad file, bad JSON, a bad flag), 3 specifically
for a model name that has no price, since that is the one failure a CI job
might want to tell apart from the rest.
"""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

__all__ = ["main"]

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_UNKNOWN_MODEL = 3


class CliError(Exception):
    """A problem to print on stderr, with the exit code it should cause."""

    def __init__(self, message, exit_code=EXIT_BAD_INPUT):
        super(CliError, self).__init__(message)
        self.exit_code = exit_code


def _non_negative_int(value):
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not an integer" % value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative: %r" % value)
    return parsed


def _positive_int(value):
    parsed = _non_negative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be at least 1: %r" % value)
    return parsed


def _calls_phrase(calls):
    return "%s call%s" % (format_int(calls), "" if calls == 1 else "s")


def _pricing_table(args):
    if args.pricing:
        try:
            return load_pricing(args.pricing)
        except ValueError as error:
            raise CliError(str(error))
    return default_pricing()


def _read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as error:
        raise CliError("cannot read %s: %s" % (path, error.strerror or error))


def cmd_estimate(args, table):
    try:
        price = table.resolve(args.model)
    except UnknownModelError as error:
        raise CliError(str(error), EXIT_UNKNOWN_MODEL)

    breakdown = estimate_cost(
        price,
        input_tokens=args.input,
        output_tokens=args.output,
        cached_input_tokens=args.cached,
        cache_write_tokens=args.cache_write,
        calls=args.calls,
    )

    if args.json:
        print(json.dumps(breakdown.to_dict(), indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%s  (%s, prices as of %s)"
        % (price.model, _calls_phrase(args.calls), table.as_of)
    )
    print()
    headers = ["item", "tokens", "$/1M", "cost"]
    rows = [
        ["input", format_int(breakdown.input_tokens), "%.2f" % price.input,
         format_money(breakdown.input_cost)],
        ["cached input", format_int(breakdown.cached_input_tokens),
         "%.2f" % price.cached_input, format_money(breakdown.cached_input_cost)],
        ["cache write", format_int(breakdown.cache_write_tokens),
         "%.2f" % price.cache_write, format_money(breakdown.cache_write_cost)],
        ["output", format_int(breakdown.output_tokens), "%.2f" % price.output,
         format_money(breakdown.output_cost)],
        ["total", format_int(breakdown.total_tokens), "",
         format_money(breakdown.total_cost)],
    ]
    print(render_table(headers, rows))
    print()
    print("cost per call: %s" % format_money(breakdown.cost_per_call))
    return EXIT_OK


def cmd_report(args, table):
    text = _read_file(args.usage_file)
    try:
        records, problems = load_usage(text, strict=args.strict)
    except ValueError as error:
        raise CliError(str(error))

    report = build_report(records, table, group_by=args.group_by)

    if args.json:
        payload = report.to_dict()
        payload["problems"] = [
            {"line": problem.line_number, "message": problem.message}
            for problem in problems
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    headers = [args.group_by, "calls", "input", "cached", "output", "cost", "$/call"]
    rows = [
        [
            group.key,
            format_int(group.calls),
            format_int(group.input_tokens),
            format_int(group.cached_input_tokens),
            format_int(group.output_tokens),
            format_money(group.cost),
            format_money(group.cost_per_call),
        ]
        for group in report.groups
    ]
    rows.append(
        [
            "TOTAL",
            format_int(report.total_calls),
            format_int(report.total_input_tokens),
            format_int(report.total_cached_tokens),
            format_int(report.total_output_tokens),
            format_money(report.total_cost),
        ]
    )
    print(render_table(headers, rows))

    if problems:
        print()
        for problem in problems:
            print("line %d: %s" % (problem.line_number, problem.message))
    if report.unknown_count:
        print()
        print(
            "skipped %d record(s) with no price: %s"
            % (report.unknown_count, ", ".join(sorted(report.unknown_models)))
        )
    return EXIT_OK


def cmd_compare(args, table):
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    try:
        rows = compare_models(
            table,
            input_tokens=args.input,
            output_tokens=args.output,
            cached_input_tokens=args.cached,
            cache_write_tokens=args.cache_write,
            calls=args.calls,
            provider=args.provider,
            models=models,
        )
    except UnknownModelError as error:
        raise CliError(str(error), EXIT_UNKNOWN_MODEL)

    if not rows:
        raise CliError("no models matched --provider/--models")

    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%s in + %s out, %s, prices as of %s"
        % (
            format_int(args.input),
            format_int(args.output),
            _calls_phrase(args.calls),
            table.as_of,
        )
    )
    print()
    headers = ["model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest"]
    body = [
        [
            row.model,
            row.provider,
            "%.2f" % row.price.input,
            "%.2f" % row.price.output,
            format_money(row.breakdown.total_cost),
            "%.1fx" % row.vs_cheapest,
        ]
        for row in rows
    ]
    print(render_table(headers, body))
    return EXIT_OK


def cmd_models(args, table):
    names = table.models()

    if args.json:
        payload = dict((name, table.prices[name].to_dict()) for name in names)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    print(
        "%d models, USD per 1M tokens, as of %s (source: %s)"
        % (len(names), table.as_of, table.source)
    )
    print()
    headers = ["model", "provider", "input", "output", "cached input", "cache write"]
    rows = []
    for name in names:
        price = table.prices[name]
        rows.append(
            [
                name,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                "%.2f" % price.cached_input,
                "%.2f" % price.cache_write,
            ]
        )
    print(render_table(headers, rows))
    return EXIT_OK


def _add_common(parser):
    parser.add_argument("--pricing", metavar="FILE", help="JSON pricing override file")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser():
    # --pricing/--json are added to both the top parser and each subparser so
    # they work on either side of the subcommand, as the README promises.
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)

    parser = argparse.ArgumentParser(prog="llm-cost", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate", parents=[common], help="price one call against the table"
    )
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--input", type=_non_negative_int, default=0)
    estimate.add_argument("--output", type=_non_negative_int, default=0)
    estimate.add_argument("--cached", type=_non_negative_int, default=0)
    estimate.add_argument("--cache-write", type=_non_negative_int, default=0)
    estimate.add_argument("--calls", type=_positive_int, default=1)
    estimate.set_defaults(func=cmd_estimate)

    report = subparsers.add_parser(
        "report", parents=[common], help="cost a JSONL usage log"
    )
    report.add_argument("usage_file", metavar="FILE")
    report.add_argument("--group-by", default="model")
    report.add_argument("--strict", action="store_true")
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser(
        "compare", parents=[common], help="rank models cheapest first"
    )
    compare.add_argument("--input", type=_non_negative_int, default=0)
    compare.add_argument("--output", type=_non_negative_int, default=0)
    compare.add_argument("--cached", type=_non_negative_int, default=0)
    compare.add_argument("--cache-write", type=_non_negative_int, default=0)
    compare.add_argument("--calls", type=_positive_int, default=1)
    compare.add_argument("--provider")
    compare.add_argument("--models", help="comma-separated model names")
    compare.set_defaults(func=cmd_compare)

    models = subparsers.add_parser(
        "models", parents=[common], help="list the resolved price table"
    )
    models.set_defaults(func=cmd_models)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        table = _pricing_table(args)
        return args.func(args, table)
    except CliError as error:
        print("llm-cost: %s" % error, file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
