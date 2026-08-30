from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
CACHE_MISS = object()


###############################################################################
class BoundedCache(Generic[KeyT, ValueT]):
    """Small deterministic least-recently-used cache."""

    __slots__ = ("limit", "store")

    # -------------------------------------------------------------------------
    def __init__(self, limit: int) -> None:
        self.limit = max(int(limit), 1)
        self.store: OrderedDict[KeyT, ValueT] = OrderedDict()

    # -------------------------------------------------------------------------
    def get(self, key: KeyT, default: object = CACHE_MISS) -> object:
        if key not in self.store:
            return default
        value = self.store.pop(key)
        self.store[key] = value
        return value

    # -------------------------------------------------------------------------
    def put(self, key: KeyT, value: ValueT) -> None:
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.limit:
            self.store.popitem(last=False)
        self.store[key] = value

    # -------------------------------------------------------------------------
    def clear(self) -> None:
        self.store.clear()
