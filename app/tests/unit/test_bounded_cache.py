from common.utils.bounded_cache import CACHE_MISS, BoundedCache


###############################################################################
def test_cache_hit_and_miss_sentinel_identity() -> None:
    cache: BoundedCache[str, int] = BoundedCache(2)
    assert cache.get("missing") is CACHE_MISS
    cache.put("a", 1)
    assert cache.get("a") == 1


###############################################################################
def test_replacement_refreshes_recency_and_evicts_oldest_key() -> None:
    cache: BoundedCache[str, int] = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is CACHE_MISS
    cache.put("a", 4)
    assert cache.get("a") == 4
    assert cache.get("c") == 3


###############################################################################
def test_clear_and_non_positive_limits() -> None:
    cache: BoundedCache[str, int] = BoundedCache(0)
    cache.put("a", 1)
    assert cache.get("a") == 1
    cache.clear()
    assert cache.get("a") is CACHE_MISS

    negative: BoundedCache[str, int] = BoundedCache(-3)
    negative.put("a", 1)
    assert negative.get("a") == 1
