import pytest

from llm_cost.pricing import (
    BUILTIN_PRICING,
    ModelPrice,
    UnknownModelError,
    default_pricing,
    load_pricing,
    parse_pricing,
)


def test_model_price_defaults_cached_and_cache_write_to_input():
    price = ModelPrice("m", input=2.0, output=4.0)
    assert price.cached_input == 2.0
    assert price.cache_write == 2.0


def test_model_price_rejects_negative():
    with pytest.raises(ValueError):
        ModelPrice("m", input=-1.0, output=1.0)


def test_resolve_exact_and_case_insensitive():
    table = default_pricing()
    assert table.resolve("claude-opus-5").model == "claude-opus-5"
    assert table.resolve("CLAUDE-OPUS-5").model == "claude-opus-5"


def test_resolve_strips_provider_prefix():
    table = default_pricing()
    assert table.resolve("anthropic.claude-opus-5").model == "claude-opus-5"
    assert table.resolve("anthropic/claude-sonnet-5").model == "claude-sonnet-5"


def test_resolve_matches_longest_prefix_for_dated_suffix():
    table = default_pricing()
    assert table.resolve("gpt-4o-2026-01-31").model == "gpt-4o"
    assert table.resolve("gpt-4o-mini-2026-01-31").model == "gpt-4o-mini"


def test_resolve_unknown_model_message_has_no_repr_quoting():
    table = default_pricing()
    with pytest.raises(UnknownModelError, match="no price for model"):
        table.resolve("totally-unknown-model")
    try:
        table.resolve("totally-unknown-model")
    except UnknownModelError as error:
        # KeyError's default __str__ reprs its argument; the override in
        # pricing.py must not put a stray leading quote in the message.
        assert not str(error).startswith("'")


def test_resolve_empty_name_raises():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("")


def test_parse_pricing_merges_onto_builtin_by_default():
    table = parse_pricing({"my-model": {"input": 1.0, "output": 2.0}})
    assert table.resolve("my-model").input == 1.0
    assert table.resolve("claude-opus-5").input == BUILTIN_PRICING["claude-opus-5"].input


def test_parse_pricing_replace_drops_builtin():
    table = parse_pricing(
        {"replace": True, "models": {"my-model": {"input": 1.0, "output": 2.0}}}
    )
    assert table.resolve("my-model").input == 1.0
    with pytest.raises(UnknownModelError):
        table.resolve("claude-opus-5")


def test_parse_pricing_missing_required_field():
    with pytest.raises(ValueError, match="missing"):
        parse_pricing({"my-model": {"input": 1.0}})


def test_parse_pricing_non_numeric_field():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_pricing({"my-model": {"input": 1.0, "output": "a lot"}})


def test_parse_pricing_entry_must_be_object():
    with pytest.raises(ValueError, match="must be an object"):
        parse_pricing({"my-model": 1.0})


def test_parse_pricing_empty_models_rejected():
    with pytest.raises(ValueError, match="defines no models"):
        parse_pricing({"models": {}})


def test_parse_pricing_top_level_must_be_object():
    with pytest.raises(ValueError, match="must contain a JSON object"):
        parse_pricing([1, 2, 3])


def test_load_pricing_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_pricing(str(tmp_path / "does-not-exist.json"))


def test_load_pricing_invalid_json(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_pricing(str(path))


def test_load_pricing_reads_file(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"my-model": {"input": 1.0, "output": 2.0}}', encoding="utf-8")
    table = load_pricing(str(path))
    assert table.source == str(path)
    assert table.resolve("my-model").output == 2.0
