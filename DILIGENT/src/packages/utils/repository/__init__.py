from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "serializer",
    "sqlite",
    "vectors",
    "VectorDatabase",
]


def __getattr__(name: str) -> Any:
    if name in {"serializer", "sqlite", "vectors"}:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "VectorDatabase":
        module = importlib.import_module(f"{__name__}.vectors")
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
