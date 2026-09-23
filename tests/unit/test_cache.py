import asyncio

import pytest

from app.services.cache import AsyncTTLCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_single_flight_builds_once_for_concurrent_requests():
    cache: AsyncTTLCache[str] = AsyncTTLCache(max_entries=4, ttl_seconds=60)
    builds = 0

    async def factory() -> str:
        nonlocal builds
        builds += 1
        await asyncio.sleep(0.01)
        return "value"

    results = await asyncio.gather(*(cache.get_or_create("k", factory) for _ in range(5)))
    assert builds == 1
    assert [value for value, _ in results] == ["value"] * 5
    assert sum(1 for _, cached in results if not cached) == 1


async def test_entries_expire_after_ttl():
    clock = Clock()
    cache: AsyncTTLCache[int] = AsyncTTLCache(max_entries=4, ttl_seconds=10, clock=clock)
    cache.set("k", 1)
    clock.now = 9.9
    assert cache.get("k") == 1
    clock.now = 10.0
    assert cache.get("k") is None


def test_lru_eviction_keeps_recently_used():
    cache: AsyncTTLCache[int] = AsyncTTLCache(max_entries=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


async def test_failed_build_is_not_cached_and_propagates():
    cache: AsyncTTLCache[int] = AsyncTTLCache(max_entries=2, ttl_seconds=60)

    async def failing() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cache.get_or_create("k", failing)
    assert cache.get("k") is None
    value, cached = await cache.get_or_create("k", lambda: asyncio.sleep(0, result=7))
    assert (value, cached) == (7, False)
