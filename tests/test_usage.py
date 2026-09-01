import pytest

from llm_cost.usage import UsageRecord, aggregate, load_usage, parse_record


def test_parse_record_openai_shape_subtracts_cached_from_prompt_tokens():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 31000,
                "completion_tokens": 420,
                "prompt_tokens_details": {"cached_tokens": 24000},
            },
        }
    )
    assert record.input_tokens == 7000
    assert record.cached_input_tokens == 24000
    assert record.output_tokens == 420
    assert record.cache_write_tokens == 0


def test_parse_record_anthropic_shape_keeps_input_tokens_exclusive():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 18400,
                "output_tokens": 2100,
                "cache_read_input_tokens": 52000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    assert record.input_tokens == 18400
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000
    assert record.output_tokens == 2100


def test_parse_record_flat_shape_without_usage_key():
    record = parse_record({"model": "x", "input_tokens": 100, "output_tokens": 50})
    assert record.input_tokens == 100
    assert record.output_tokens == 50


def test_parse_record_requires_model():
    with pytest.raises(ValueError, match="no model"):
        parse_record({"usage": {"input_tokens": 1}})


def test_parse_record_requires_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_record(["not", "an", "object"])


def test_parse_record_rejects_negative_tokens():
    with pytest.raises(ValueError, match="not be negative"):
        parse_record({"model": "x", "usage": {"input_tokens": -1}})


def test_parse_record_rejects_bool_as_token_count():
    with pytest.raises(ValueError, match="must be a number"):
        parse_record({"model": "x", "usage": {"input_tokens": True}})


def test_group_key_truncates_date_and_falls_back_to_none():
    record = UsageRecord(model="m", fields={"date": "2026-06-01T09:12:00Z"})
    assert record.group_key("date") == "2026-06-01"
    assert record.group_key("team") == "(none)"
    assert record.group_key("model") == "m"


def test_load_usage_skips_blank_lines_and_collects_problems():
    text = "\n".join(
        [
            '{"model": "m", "usage": {"input_tokens": 1, "output_tokens": 2}}',
            "",
            "not json",
            '{"usage": {"input_tokens": 1}}',
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 2
    assert problems[0].line_number == 3
    assert problems[1].line_number == 4


def test_load_usage_strict_raises_on_first_problem():
    with pytest.raises(ValueError, match="line 1"):
        load_usage("not json", strict=True)


def test_aggregate_groups_by_key():
    records = [
        UsageRecord(model="a"),
        UsageRecord(model="a"),
        UsageRecord(model="b"),
    ]
    buckets = aggregate(records, group_by="model")
    assert sorted(buckets) == ["a", "b"]
    assert len(buckets["a"]) == 2
    assert len(buckets["b"]) == 1
