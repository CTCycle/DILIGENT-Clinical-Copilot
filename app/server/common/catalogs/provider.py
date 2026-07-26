from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from domain.catalogs import ReferenceCatalogSnapshot


###############################################################################
class _CatalogProvider:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._impl: Callable[[], ReferenceCatalogSnapshot] | None = None
        self._invalidate: Callable[[], None] | None = None

    # -------------------------------------------------------------------------
    def register(
        self,
        impl: Callable[[], ReferenceCatalogSnapshot],
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        if self._impl is not None and (
            self._impl != impl or self._invalidate != invalidate
        ):
            raise RuntimeError("Catalog snapshot provider is already registered")
        self._impl = impl
        self._invalidate = invalidate

    # -------------------------------------------------------------------------
    def get_snapshot(self) -> ReferenceCatalogSnapshot:
        if self._impl is None:
            raise RuntimeError(
                "Catalog snapshot provider is not registered. "
                "Call register_provider() during application startup."
            )
        return self._impl()

    # -------------------------------------------------------------------------
    def invalidate(self) -> None:
        if self._invalidate is not None:
            self._invalidate()


@lru_cache(maxsize=1)
def get_catalog_provider() -> _CatalogProvider:
    return _CatalogProvider()
