"""Pricing table and per-call cost (FR-R2, FR-B8)."""

from pathlib import Path

import pytest

import flight_recorder as fr
from flight_recorder.errors import PricingError
from flight_recorder.model import Usage
from flight_recorder.policy import pricing

M = 1_000_000


def test_dated_snapshot_ids_fall_back_to_their_undated_entry() -> None:
    million = Usage(input_tokens=M, output_tokens=M)
    assert pricing.cost("anthropic", "claude-sonnet-4-5-20250929", million) == 18.0
    assert pricing.cost("openai", "gpt-4o-2024-08-06", million) == 12.5


def test_unknown_model_is_null_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    usage = Usage(input_tokens=10, output_tokens=1)
    # No prefix matching: a variant is not priced as its base model.
    for _ in range(2):
        assert pricing.cost("openai", "gpt-4o-audio-preview", usage) is None
    assert caplog.text.count("no price for openai model 'gpt-4o-audio-preview'") == 1


def test_cached_tokens_are_priced_at_their_own_rates() -> None:
    usage = Usage(
        input_tokens=112, output_tokens=3, cache_read_tokens=100, cache_write_tokens=2
    )
    # claude-opus-5-5: 10 uncached x $4, 100 read x $0.20, one 5m write x $5,
    # one 1h write x $8, 3 out x $20 (per million).
    cost = pricing.cost("anthropic", "claude-opus-5-5", usage, cache_write_1h=1)
    assert cost == pytest.approx((40 + 20 + 5 + 8 + 60) / M)


def test_cached_tokens_without_a_cache_rate_are_not_guessed() -> None:
    usage = Usage(input_tokens=100, output_tokens=1, cache_read_tokens=50)
    assert pricing.cost("openai", "gpt-5-pro", usage) is None


def test_long_prompts_use_the_long_context_rates() -> None:
    at = Usage(input_tokens=100_000, output_tokens=0)
    over = Usage(input_tokens=100_001, output_tokens=0)
    assert pricing.cost("anthropic", "claude-haiku-5-5", at) == pytest.approx(0.01)
    assert pricing.cost("anthropic", "claude-haiku-5-5", over) == pytest.approx(
        100_001 * 0.50 / M
    )


def test_a_modifier_the_table_cannot_price_is_null(
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = Usage(input_tokens=10, output_tokens=1)
    cost = pricing.cost("openai", "gpt-5", usage, unpriced="service_tier 'flex'")
    assert cost is None
    assert "openai model 'gpt-5' used service_tier 'flex'" in caplog.text


def test_configure_pricing_file_replaces_and_adds_entries(tmp_path: Path) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[openai]\n"gpt-5" = { input = 1.0, output = 2.0 }\n'
        '"my-finetune" = { input = 3.0, output = 4.0 }\n'
    )
    fr.configure(dir=tmp_path, pricing=prices)
    million = Usage(input_tokens=M, output_tokens=M)
    assert pricing.cost("openai", "gpt-5", million) == 3.0
    assert pricing.cost("openai", "my-finetune-2026-01-01", million) == 7.0
    assert pricing.cost("anthropic", "claude-opus-5-5", million) == 24.0  # shipped


@pytest.mark.parametrize(
    "text",
    [
        "[openai\n",  # not TOML
        '[openai]\n"gpt-5" = { input = 1.0 }\n',  # no output rate
        '[openai]\n"gpt-5" = { input = 1.0, output = 2.0, cached = 0.1 }\n',  # typo
    ],
)
def test_invalid_pricing_file_fails_at_configure_and_changes_nothing(
    tmp_path: Path, text: str
) -> None:
    prices = tmp_path / "prices.toml"
    prices.write_text(text)
    with pytest.raises(PricingError, match="prices.toml"):
        fr.configure(dir=tmp_path, pricing=prices)
    assert pricing.table is pricing.SHIPPED


def test_tool_calls_are_billed_per_call_on_top_of_tokens() -> None:
    usage = Usage(input_tokens=M, output_tokens=0)
    cost = pricing.cost(
        "anthropic", "claude-opus-5-5", usage, tool_calls={"web_search": 3}
    )
    assert cost == pytest.approx(4.0 + 3 * 10.0 / 1000)
    calls = {"web_search": 1, "web_search_preview": 2, "file_search": 4}
    cost = pricing.cost("openai", "gpt-4o", usage, tool_calls=calls)
    assert cost == pytest.approx(2.5 + (10.0 + 2 * 25.0 + 4 * 2.5) / 1000)
    # A fee the entry lacks: Anthropic has no file search; chat-latest has no
    # preview rate, since the page doesn't say if it is a reasoning model.
    file_search = {"file_search": 1}
    assert (
        pricing.cost("anthropic", "claude-opus-5-5", usage, tool_calls=file_search)
        is None
    )
    preview = {"web_search_preview": 1}
    assert pricing.cost("openai", "chat-latest", usage, tool_calls=preview) is None
