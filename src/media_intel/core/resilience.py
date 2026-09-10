"""Production resilience primitives: retry w/ backoff, circuit breaker, fallbacks."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions considered transient and safe to retry.
RETRYABLE = (TimeoutError, ConnectionError, OSError)


class RetryExhausted(RuntimeError):
    """Raised when all retry attempts fail."""

    def __init__(self, attempts: int, last_error: BaseException):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"gave up after {attempts} attempts: {last_error!r}")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = RETRYABLE

    def delay_for(self, attempt: int) -> float:
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        if self.jitter:
            delay *= 0.5 + random.random() / 2
        return delay


def with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Execute ``fn`` retrying transient failures with exponential backoff."""
    policy = policy or RetryPolicy()
    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as exc:  # noqa: PERF203
            last = exc
            if attempt == policy.max_attempts:
                break
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(policy.delay_for(attempt))
        except Exception:  # non-retryable: fail fast
            raise
    assert last is not None
    raise RetryExhausted(policy.max_attempts, last)


async def awith_retry(
    fn: Callable[[], Any],
    policy: RetryPolicy | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Any:
    """Async variant of :func:`with_retry` (fn may be sync or async)."""
    policy = policy or RetryPolicy()
    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
            return result
        except policy.retry_on as exc:  # noqa: PERF203
            last = exc
            if attempt == policy.max_attempts:
                break
            if on_retry:
                on_retry(attempt, exc)
            await asyncio.sleep(policy.delay_for(attempt))
        except Exception:
            raise
    assert last is not None
    raise RetryExhausted(policy.max_attempts, last)


@dataclass
class CircuitBreaker:
    """Half-open circuit breaker to stop hammering a failing dependency."""

    name: str
    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    _failures: int = 0
    _state: str = field(default="closed")  # closed | open | half_open
    _opened_at: float = 0.0

    @property
    def state(self) -> str:
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                return "half_open"
            return "open"
        return self._state

    def before(self) -> None:
        state = self.state
        if state == "open":
            raise RuntimeError(f"circuit '{self.name}' is open")
        if state == "half_open":
            self._state = "half_open"

    def success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def failure(self) -> None:
        self._failures += 1
        if self._state == "half_open" or self._failures >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.warning("circuit '%s' opened after %d failures", self.name, self._failures)


class FallbackChain:
    """Try strategies in order; first success wins. Last resort = optional default."""

    def __init__(self, strategies: list[Callable[[], T]], default: T | None = None):
        self.strategies = strategies
        self.default = default

    def run(self) -> T:
        errors: list[BaseException] = []
        for strategy in self.strategies:
            try:
                return strategy()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if self.default is not None:
            return self.default
        raise RetryExhausted(len(errors), errors[-1] if errors else RuntimeError("no strategies"))
