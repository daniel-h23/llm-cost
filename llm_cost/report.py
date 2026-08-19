"""Aggregate usage records into a costed report, and rank models by cost.

A record naming a model with no entry in the price table is not priced as
zero -- that would silently understate a report's total. It is instead
tallied in ``Report.unknown_models`` and left out of every group, so the
caller can tell the difference between "this was free" and "this could not
be priced".
"""

from .estimate import estimate_cost
from .pricing import UnknownModelError

__all__ = ["GroupSummary", "Report", "ComparisonRow", "build_report", "compare_models"]


class GroupSummary(object):
    """Totals for one group (one model, one day, one team, ...) in a report."""

    __slots__ = (
        "key",
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "cost",
    )

    def __init__(
        self,
        key,
        calls,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        cost,
    ):
        self.key = key
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.cost = cost

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.cost / self.calls

    def to_dict(self):
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost": round(self.cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }

    def __repr__(self):
        return "GroupSummary(%s, calls=%d, cost=%.6f)" % (
            self.key,
            self.calls,
            self.cost,
        )


class Report(object):
    """A costed, grouped usage report plus the records it could not price."""

    __slots__ = ("group_by", "groups", "unknown_models", "unknown_count")

    def __init__(self, group_by, groups, unknown_models, unknown_count):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.unknown_count = unknown_count

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_input_tokens(self):
        return sum(group.input_tokens for group in self.groups)

    @property
    def total_output_tokens(self):
        return sum(group.output_tokens for group in self.groups)

    @property
    def total_cached_tokens(self):
        return sum(group.cached_input_tokens for group in self.groups)

    @property
    def total_cache_write_tokens(self):
        return sum(group.cache_write_tokens for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.cost for group in self.groups)

    def to_dict(self):
        return {
            "group_by": self.group_by,
            "groups": [group.to_dict() for group in self.groups],
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 6),
            "unknown_models": sorted(self.unknown_models),
            "unknown_count": self.unknown_count,
        }

    def __repr__(self):
        return "Report(group_by=%s, groups=%d, total=%.6f)" % (
            self.group_by,
            len(self.groups),
            self.total_cost,
        )


def build_report(records, table, group_by="model"):
    """Group ``records`` by ``group_by`` and cost each group against ``table``.

    Groups come back sorted most expensive first, since that is the
    question a report is usually opened to answer.
    """
    buckets = {}
    unknown_models = set()
    unknown_count = 0

    for record in records:
        try:
            price = table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            unknown_count += 1
            continue

        breakdown = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )

        key = record.group_key(group_by)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0.0,
            }
        bucket["calls"] += 1
        bucket["input_tokens"] += breakdown.input_tokens
        bucket["output_tokens"] += breakdown.output_tokens
        bucket["cached_input_tokens"] += breakdown.cached_input_tokens
        bucket["cache_write_tokens"] += breakdown.cache_write_tokens
        bucket["cost"] += breakdown.total_cost

    groups = [GroupSummary(key, **bucket) for key, bucket in buckets.items()]
    groups.sort(key=lambda group: group.cost, reverse=True)
    return Report(group_by, groups, unknown_models, unknown_count)


class ComparisonRow(object):
    """One model's cost in a :func:`compare_models` ranking."""

    __slots__ = ("model", "provider", "price", "breakdown", "vs_cheapest")

    def __init__(self, model, provider, price, breakdown, vs_cheapest):
        self.model = model
        self.provider = provider
        self.price = price
        self.breakdown = breakdown
        self.vs_cheapest = vs_cheapest

    def to_dict(self):
        result = self.breakdown.to_dict()
        result["provider"] = self.provider
        result["vs_cheapest"] = round(self.vs_cheapest, 4)
        return result

    def __repr__(self):
        return "ComparisonRow(%s, cost=%.6f, vs_cheapest=%.2fx)" % (
            self.model,
            self.breakdown.total_cost,
            self.vs_cheapest,
        )


def compare_models(
    table,
    input_tokens,
    output_tokens,
    cached_input_tokens=0,
    cache_write_tokens=0,
    calls=1,
    provider=None,
    models=None,
):
    """Price the same call against every matching model, cheapest first.

    ``models`` narrows the comparison to an explicit list of names (each
    resolved against ``table``, so an unknown name still raises).
    ``provider`` narrows it to one provider instead; with neither, every
    priced model is compared.
    """
    if models:
        prices = [table.resolve(name) for name in models]
    else:
        prices = list(table.prices.values())
        if provider:
            prices = [price for price in prices if price.provider == provider]

    rows = []
    for price in prices:
        breakdown = estimate_cost(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            calls=calls,
        )
        rows.append(ComparisonRow(price.model, price.provider, price, breakdown, 1.0))

    rows.sort(key=lambda row: row.breakdown.total_cost)
    if rows and rows[0].breakdown.total_cost > 0:
        cheapest = rows[0].breakdown.total_cost
        for row in rows:
            row.vs_cheapest = row.breakdown.total_cost / cheapest
    return rows
