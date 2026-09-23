import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncTTLCache(Generic[T]):
    """Bounded in-process LRU cache with TTL and single-flight creation.

    Concurrent requests for the same missing key share one factory call instead
    of each building (and paying for) the same value.
    """

    def __init__(self, max_entries: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[T]] = {}

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        self._entries[key] = (self._clock() + self._ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    async def get_or_create(self, key: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Return (value, was_cached)."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key]), True

        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await factory()
        except BaseException as error:
            future.set_exception(error)
            future.exception()  # mark retrieved so an unawaited failure doesn't log a warning
            raise
        else:
            self.set(key, value)
            future.set_result(value)
            return value, False
        finally:
            del self._inflight[key]

    def __len__(self) -> int:
        return len(self._entries)
