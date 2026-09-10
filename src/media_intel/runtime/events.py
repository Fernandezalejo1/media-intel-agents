"""Minimal event bus for event-driven workflows (pub/sub between stages)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    topic: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """In-process async pub/sub with wildcard topics (``match.*``)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic_pattern: str, handler: Handler) -> Callable[[], None]:
        self._handlers[topic_pattern].append(handler)

        def unsubscribe() -> None:
            self._handlers[topic_pattern].remove(handler)

        return unsubscribe

    async def publish(self, topic: str, payload: dict[str, Any]) -> Event:
        event = Event(topic=topic, payload=payload)
        for pattern, handlers in list(self._handlers.items()):
            if _matches(pattern, topic):
                for handler in list(handlers):
                    try:
                        await handler(event)
                    except Exception:  # noqa: BLE001 - one bad subscriber must not kill the bus
                        continue
        return event


def _matches(pattern: str, topic: str) -> bool:
    if pattern == topic or pattern == "*":
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1])
    return False
