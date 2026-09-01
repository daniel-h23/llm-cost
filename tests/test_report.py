import pytest

from llm_cost.pricing import UnknownModelError, parse_pricing
from llm_cost.report import build_report, compare_models
from llm_cost.usage import UsageRecord


def _table():
    return parse_pricing(
        {
            "replace": True,
            "models": {
                "model-a": {
                    "provider": "acme",
                    "input": 1.0,
                    "output": 2.0,
                    "cached_input": 0.5,
                    "cache_write": 1.5,
                },
                "model-b": {"provider": "other", "input": 4.0, "output": 8.0},
            },
        }
    )


def test_build_report_aggregates_costs_sorted_most_expensive_first():
    table = _table()
    records = [
        UsageRecord(
            model="model-a",
            input_tokens=1_000_000,
            output_tokens=500_000,
            cached_input_tokens=200_000,
            cache_write_tokens=100_000,
        ),
        UsageRecord(model="model-a", input_tokens=500_000, output_tokens=100_000),
        UsageRecord(model="model-b", input_tokens=250_000, output_tokens=125_000),
        UsageRecord(model="unknown-model", input_tokens=1000, output_tokens=1000),
    ]

    report = build_report(records, table, group_by="model")

    assert [group.key for group in report.groups] == ["model-a", "model-b"]
    assert report.groups[0].cost == pytest.approx(2.95)
    assert report.groups[0].calls == 2
    assert report.groups[1].cost == pytest.approx(2.0)
    assert report.total_cost == pytest.approx(4.95)
    assert report.total_calls == 3
    assert report.unknown_models == {"unknown-model"}
    assert report.unknown_count == 1


def test_build_report_with_no_records_is_empty():
    report = build_report([], _table())
    assert report.groups == []
    assert report.total_cost == 0
    assert report.unknown_count == 0


def test_compare_models_ranks_cheapest_first():
    table = _table()
    rows = compare_models(table, input_tokens=1000, output_tokens=1000)
    assert [row.model for row in rows] == ["model-a", "model-b"]
    assert rows[0].vs_cheapest == pytest.approx(1.0)
    assert rows[1].vs_cheapest == pytest.approx(4.0)


def test_compare_models_provider_filter():
    table = _table()
    rows = compare_models(table, input_tokens=1000, output_tokens=1000, provider="other")
    assert [row.model for row in rows] == ["model-b"]
    assert rows[0].vs_cheapest == pytest.approx(1.0)


def test_compare_models_explicit_list():
    table = _table()
    rows = compare_models(
        table, input_tokens=1000, output_tokens=1000, models=["model-b"]
    )
    assert [row.model for row in rows] == ["model-b"]


def test_compare_models_unknown_explicit_model_raises():
    table = _table()
    with pytest.raises(UnknownModelError):
        compare_models(
            table, input_tokens=1000, output_tokens=1000, models=["model-a", "nope"]
        )
