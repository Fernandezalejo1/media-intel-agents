"""Cost & latency governance: hard caps on LLM calls, tokens and wall-clock time.

These budgets are the "minimize LLM calls / control cost and latency" controls:
they are enforced *before* every call so runaway agents degrade gracefully
instead of burning budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """A governance cap was hit."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# Rough USD per 1K tokens by model tier (illustrative; override per deployment).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # model: (input USD/1K, output USD/1K)
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "mock-large": (0.0, 0.0),
    "mock-small": (0.0, 0.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (0.0025, 0.010))
    return input_tokens / 1000 * price_in + output_tokens / 1000 * price_out


@dataclass
class RunBudget:
    """Mutable per-run budget ledger."""

    max_llm_calls: int
    max_total_tokens: int
    deadline_ms: int
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = 0.0

    @classmethod
    def from_limits(cls, max_llm_calls: int, max_total_tokens: int, deadline_ms: int) -> "RunBudget":
        return cls(
            max_llm_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
            deadline_ms=deadline_ms,
            started_at=time.monotonic(),
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def check_llm_allowed(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"llm call cap reached ({self.max_llm_calls})")
        if self.elapsed_ms > self.deadline_ms:
            raise BudgetExceeded(f"run deadline exceeded ({self.deadline_ms}ms)")

    def record_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.total_tokens > self.max_total_tokens:
            raise BudgetExceeded(f"token cap reached ({self.max_total_tokens})")

    def summary(self) -> dict[str, float | int]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "deadline_ms": self.deadline_ms,
        }
