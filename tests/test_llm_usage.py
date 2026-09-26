"""Token accounting and cost estimation.

These are the numbers the whole project is judged on, so they get tests of their
own. A silent error here would not break anything visibly — it would just make
every reported figure wrong in a plausible-looking way.
"""

import pytest

from app.core.config import ModelPrice, settings
from app.services.llm import Usage, estimate_cost, estimate_tokens


class TestExtractUsage:
    """Normalising the provider's usage payload."""

    def test_reads_top_level_cache_field(self, raw_usage, extract):
        """DeepSeek reports cache hits in a top-level field."""
        usage = extract(raw_usage(prompt=1000, cached=800, completion=50), "m")

        assert usage.prompt_tokens == 1000
        assert usage.cached_tokens == 800
        assert usage.completion_tokens == 50

    def test_falls_back_to_nested_detail(self, raw_usage, extract):
        """The OpenAI-compatible shape nests the same figure one level down."""
        usage = extract(
            raw_usage(prompt=500, cached=300, completion=10, detail_style=True), "m"
        )

        assert usage.cached_tokens == 300

    def test_clamps_cached_above_prompt(self, raw_usage, extract):
        """A provider reporting more cached than prompt tokens must not go negative.

        Without the clamp the uncached share would be negative and the estimated
        cost could come out below zero.
        """
        usage = extract(raw_usage(prompt=100, cached=500, completion=0), "m")

        assert usage.cached_tokens == 100
        assert usage.uncached_prompt_tokens == 0

    def test_missing_usage_is_zero_not_an_error(self, extract):
        """Accounting that crashes is worse than accounting that under-reports."""
        usage = extract(None, "m")

        assert usage.total_tokens == 0
        assert usage.cache_hit_rate is None

    def test_cache_hit_rate(self, raw_usage, extract):
        """Hit rate is cached over prompt, not cached over total."""
        assert extract(raw_usage(1000, 250, 999), "m").cache_hit_rate == pytest.approx(0.25)


class TestUsageArithmetic:
    """Aggregation across calls."""

    def test_addition_sums_every_field(self):
        total = Usage("m", 100, 40, 10) + Usage("m", 200, 150, 20)

        assert (total.prompt_tokens, total.cached_tokens, total.completion_tokens) == (
            300, 190, 30,
        )

    def test_total_tokens_includes_cached(self):
        """Cached tokens are a subset of prompt tokens, so they are not added twice."""
        assert Usage("m", 100, 90, 5).total_tokens == 105


class TestModelPrice:
    """Cost arithmetic."""

    def test_cached_tokens_are_not_double_billed(self):
        """The cached share must be subtracted, not added.

        Treating ``cached_tokens`` as extra input would overstate cost by roughly
        the cache-hit rate — and the error would scale with how well the caching
        worked, so the better the optimisation performed the worse it would look.
        """
        price = ModelPrice(input_per_million=10.0, cached_input_per_million=1.0, output_per_million=0.0)

        # 1M prompt of which 800k cached: 200k full price + 800k at the cheap rate.
        cost = price.estimate(prompt_tokens=1_000_000, cached_tokens=800_000, completion_tokens=0)

        assert cost == pytest.approx(0.2 * 10.0 + 0.8 * 1.0)

    def test_negative_inputs_are_clamped(self):
        """A malformed payload must not produce a negative cost."""
        price = ModelPrice(input_per_million=10.0, cached_input_per_million=1.0, output_per_million=5.0)

        assert price.estimate(-100, -50, -20) == 0.0

    def test_price_for_routes_models_to_their_own_table(self):
        """Embedding traffic is priced at the embedding rate.

        Pricing every call at the chat rate overstates a run whose largest token
        consumer by volume is nearly free.
        """
        assert settings.price_for(settings.EMBEDDING.model) is settings.EMBEDDING.price
        assert settings.price_for(settings.LLM.model) is settings.LLM.price

    def test_estimate_cost_mixes_rates(self):
        """A mixed set is priced per model, not at a single rate."""
        usages = [
            Usage(settings.LLM.model, 1_000_000, 0, 0),
            Usage(settings.EMBEDDING.model, 1_000_000, 0, 0),
        ]

        expected = settings.LLM.price.input_per_million + settings.EMBEDDING.price.input_per_million

        assert estimate_cost(usages) == pytest.approx(expected)


class TestEstimateTokens:
    """The calibrated token estimator.

    Calibration points are the two direct provider measurements this estimator was
    fitted to. They are asserted here because a drift in the coefficients would
    silently skew every baseline figure the project reports.
    """

    def test_matches_measured_chinese_prompt(self):
        """A Chinese-heavy prompt runs about 0.7 tokens per character."""
        text = "汉" * 1063 + "a" * 472

        assert estimate_tokens(text) == pytest.approx(860, abs=30)

    def test_matches_measured_english_resume(self):
        """English text runs about 0.24 tokens per character — roughly 4 per token."""
        assert estimate_tokens("a" * 2405) == pytest.approx(568, abs=30)

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0

    def test_a_flat_ratio_would_be_wrong(self):
        """Regression guard: the previous constant was off by 2x on English.

        Asserting the *magnitude of the old error* rather than the new value keeps
        the test meaningful if the coefficients are ever retuned — what must not
        come back is a single ratio serving both scripts.
        """
        english = "a" * 2405

        flat = len(english) // 2
        measured = 568

        assert flat > measured * 2
        assert estimate_tokens(english) == pytest.approx(measured, rel=0.1)
