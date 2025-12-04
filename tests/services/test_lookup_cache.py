from __future__ import annotations

import unittest

from DILIGENT.server.utils.services.clinical.matches import (
    BoundedCache,
    CACHE_MISS,
)


class BoundedCacheTests(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_cache_eviction_follows_lru_order(self) -> None:
        cache: BoundedCache[str, int] = BoundedCache(limit=2)
        cache.put("first", 1)
        cache.put("second", 2)
        cache.get("first")
        cache.put("third", 3)
        self.assertEqual(cache.get("first", CACHE_MISS), 1)
        self.assertIs(cache.get("second", CACHE_MISS), CACHE_MISS)
        self.assertEqual(cache.get("third", CACHE_MISS), 3)

    # ------------------------------------------------------------------
    def test_cache_accepts_none_values(self) -> None:
        cache: BoundedCache[str, str | None] = BoundedCache(limit=1)
        cache.put("key", None)
        self.assertIsNone(cache.get("key", CACHE_MISS))
        cache.put("replacement", "value")
        self.assertIs(cache.get("key", CACHE_MISS), CACHE_MISS)


if __name__ == "__main__":
    unittest.main()
