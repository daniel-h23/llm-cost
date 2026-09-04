import pytest

from llm_cost.estimate import estimate_cost
from llm_cost.pricing import ModelPrice


def _price():
    return ModelPrice(
        "m", input=2.0, output=10.0, cached_input=0.5, cache_write=3.0
    )


def test_estimate_cost_prices_each_token_class_separately():
    breakdown = estimate_cost(
        _price(),
        input_tokens=1_000_000,
        output_tokens=200_000,
        cached_input_tokens=400_000,
        cache_write_tokens=100_000,
    )
    assert breakdown.input_cost == pytest.approx(2.0)
    assert breakdown.output_cost == pytest.approx(2.0)
    assert breakdown.cached_input_cost == pytest.approx(0.2)
    assert breakdown.cache_write_cost == pytest.approx(0.3)
    assert breakdown.total_cost == pytest.approx(4.5)
    assert breakdown.total_tokens == 1_700_000


def test_estimate_cost_scales_by_calls():
    one = estimate_cost(_price(), input_tokens=1000, output_tokens=100, calls=1)
    many = estimate_cost(_price(), input_tokens=1000, output_tokens=100, calls=50)
    assert many.total_cost == pytest.approx(one.total_cost * 50)
    assert many.input_tokens == 1000 * 50
    assert many.calls == 50


def test_estimate_cost_per_call_divides_total_by_calls():
    breakdown = estimate_cost(_price(), input_tokens=1_000_000, calls=4)
    assert breakdown.cost_per_call == pytest.approx(breakdown.total_cost / 4)


def test_estimate_cost_zero_calls_has_zero_cost_per_call():
    breakdown = estimate_cost(_price(), input_tokens=1000, calls=0)
    assert breakdown.total_cost == 0.0
    assert breakdown.cost_per_call == 0.0


def test_estimate_cost_defaults_to_all_zero_tokens():
    breakdown = estimate_cost(_price())
    assert breakdown.total_tokens == 0
    assert breakdown.total_cost == 0.0


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError, match="not be negative"):
        estimate_cost(_price(), input_tokens=-1)


def test_estimate_cost_rejects_non_integer_tokens():
    with pytest.raises(ValueError, match="must be an integer"):
        estimate_cost(_price(), input_tokens=12.5)


def test_estimate_cost_to_dict_rounds_to_six_places():
    breakdown = estimate_cost(_price(), input_tokens=1, output_tokens=1)
    as_dict = breakdown.to_dict()
    assert as_dict["model"] == "m"
    assert as_dict["total_cost"] == round(breakdown.total_cost, 6)
    assert as_dict["cost_per_call"] == round(breakdown.cost_per_call, 6)
