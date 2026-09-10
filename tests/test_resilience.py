"""Resilience and governance unit tests."""

from __future__ import annotations

import pytest

from media_intel.core.governance import BudgetExceeded, RunBudget, estimate_cost_usd
from media_intel.core.resilience import CircuitBreaker, FallbackChain, RetryPolicy, with_retry


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    out = with_retry(flaky, RetryPolicy(max_attempts=3, base_delay_s=0.01))
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_and_raises():
    with pytest.raises(Exception):
        with_retry(lambda: (_ for _ in ()).throw(ConnectionError("down")), RetryPolicy(max_attempts=2, base_delay_s=0.01))


def test_non_retryable_fails_fast():
    calls = {"n": 0}

    def bad_type() -> None:
        calls["n"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        with_retry(bad_type, RetryPolicy(max_attempts=5, base_delay_s=0.01))
    assert calls["n"] == 1


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(name="x", failure_threshold=2, reset_timeout_s=60)
    breaker.failure()
    breaker.failure()
    assert breaker.state == "open"
    with pytest.raises(RuntimeError, match="open"):
        breaker.before()


def test_circuit_breaker_half_open_after_timeout():
    import time

    breaker = CircuitBreaker(name="y", failure_threshold=1, reset_timeout_s=0.05)
    breaker.failure()
    time.sleep(0.06)
    assert breaker.state == "half_open"
    breaker.success()
    assert breaker.state == "closed"


def test_fallback_chain_uses_first_working_strategy():
    def bad():
        raise RuntimeError("nope")

    chain = FallbackChain([bad, lambda: "second"], default="d")
    assert chain.run() == "second"


def test_fallback_chain_default_when_all_fail():
    def bad():
        raise RuntimeError("nope")

    chain = FallbackChain([bad, bad], default="fallback")
    assert chain.run() == "fallback"


def test_budget_rejects_when_call_cap_reached():
    budget = RunBudget.from_limits(max_llm_calls=1, max_total_tokens=10_000, deadline_ms=10_000)
    budget.record_llm_usage(10, 10)
    with pytest.raises(BudgetExceeded, match="cap"):
        budget.check_llm_allowed()


def test_budget_rejects_when_tokens_exceeded():
    budget = RunBudget.from_limits(max_llm_calls=10, max_total_tokens=15, deadline_ms=10_000)
    with pytest.raises(BudgetExceeded, match="token"):
        budget.record_llm_usage(10, 10)


def test_cost_estimation_positive_for_real_models():
    assert estimate_cost_usd("gpt-4o-mini", 1000, 1000) > 0
    assert estimate_cost_usd("mock-large", 1000, 1000) == 0.0
