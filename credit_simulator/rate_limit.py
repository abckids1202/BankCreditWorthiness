from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
import time


class RateLimiter:
    """Small in-memory fixed-window limiter for local/demo deployments."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check_with_remaining(self, key: str, limit: int, now: float | None = None) -> tuple[bool, int, int]:
        if limit <= 0:
            return True, 0, 0
        current = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            while events and current - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= limit:
                return False, max(1, int(self.window_seconds - (current - events[0]) + 0.999)), 0
            events.append(current)
            return True, 0, max(limit - len(events), 0)

    def check(self, key: str, limit: int, now: float | None = None) -> tuple[bool, int]:
        allowed, retry_after, _ = self.check_with_remaining(key, limit, now)
        return allowed, retry_after

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
