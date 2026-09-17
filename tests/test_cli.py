import json
import os

import pytest

from llm_cost.cli import main

EXAMPLES_USAGE = os.path.join(
    os.path.dirname(__file__), os.pardir, "examples", "usage.jsonl"
)


def test_report_against_examples_file_groups_by_model(capsys):
    exit_code = main(["report", EXAMPLES_USAGE])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "claude-opus-5" in out
    assert "TOTAL" in out
    assert "skipped 1 record(s) with no price: internal-router-v3" in out


def test_report_against_examples_file_json_matches_library_call(capsys):
    exit_code = main(["report", EXAMPLES_USAGE, "--json"])
    out = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(out)
    assert payload["unknown_models"] == ["internal-router-v3"]
    assert payload["unknown_count"] == 1
    assert payload["total_calls"] == 5
    assert [group["key"] for group in payload["groups"]] == [
        "claude-opus-5",
        "claude-haiku-4-5",
        "gemini-2.5-flash",
        "gpt-4o-mini",
    ]


def test_report_group_by_team(capsys):
    exit_code = main(["report", EXAMPLES_USAGE, "--group-by", "team"])
    out = capsys.readouterr().out

    assert exit_code == 0
    lines = out.splitlines()
    assert lines[0].startswith("team")
    assert any(line.startswith("agents") for line in lines)
    assert any(line.startswith("search") for line in lines)
    assert any(line.startswith("ops") for line in lines)


def test_report_strict_fails_on_malformed_line(tmp_path, capsys):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"model": "claude-opus-5", "usage": {"input_tokens": 1}}\nnot json\n')

    exit_code = main(["report", str(bad_file), "--strict"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "line 2" in err


def test_report_missing_file_is_bad_input(capsys):
    exit_code = main(["report", "/no/such/file.jsonl"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "cannot read" in err


def test_estimate_prints_breakdown_and_total(capsys):
    exit_code = main(
        ["estimate", "--model", "claude-opus-5", "--input", "12000", "--output", "800"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "claude-opus-5" in out
    assert "cost per call: $0.0800" in out


def test_estimate_json_round_trips_through_estimate_cost(capsys):
    exit_code = main(
        ["estimate", "--model", "claude-opus-5", "--input", "12000", "--output", "800", "--json"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(out)
    assert payload["model"] == "claude-opus-5"
    assert payload["total_cost"] == pytest.approx(0.08)


def test_estimate_unknown_model_exits_three(capsys):
    exit_code = main(["estimate", "--model", "no-such-model", "--input", "1"])
    err = capsys.readouterr().err

    assert exit_code == 3
    assert "no-such-model" in err


def test_compare_ranks_cheapest_first(capsys):
    exit_code = main(
        ["compare", "--input", "10000", "--output", "1000", "--provider", "anthropic"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    lines = [line for line in out.splitlines() if line.startswith("claude-")]
    assert lines[0].startswith("claude-haiku-4-5")


def test_compare_no_models_matched_is_bad_input(capsys):
    exit_code = main(["compare", "--input", "1000", "--provider", "no-such-provider"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "no models matched" in err


def test_models_lists_builtin_table(capsys):
    exit_code = main(["models"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "built-in" in out
    assert "claude-opus-5" in out


def test_pricing_override_replaces_table(tmp_path, capsys):
    pricing_file = tmp_path / "prices.json"
    pricing_file.write_text(
        json.dumps(
            {
                "as_of": "2026-07-01",
                "replace": True,
                "models": {"internal-router-v3": {"input": 0.2, "output": 0.8}},
            }
        )
    )

    exit_code = main(
        ["--pricing", str(pricing_file), "report", EXAMPLES_USAGE, "--json"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(out)
    assert payload["unknown_models"] == [
        "claude-haiku-4-5",
        "claude-opus-5",
        "gemini-2.5-flash",
        "gpt-4o-mini",
    ]
    assert payload["groups"][0]["key"] == "internal-router-v3"


def test_pricing_override_missing_file_is_bad_input(capsys):
    exit_code = main(["--pricing", "/no/such/prices.json", "models"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "not found" in err
