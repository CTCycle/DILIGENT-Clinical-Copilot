from __future__ import annotations

import hashlib
from typing import NamedTuple


class EmbeddingModelSpec(NamedTuple):
    provider: str
    model_name: str
    dimension: int
    mode: str
    signature: str


def build_embedding_model_signature(
    provider: str,
    model_name: str,
    dimension: int,
    mode: str,
) -> str:
    payload = f"{provider.strip().lower()}:{model_name.strip()}:{int(dimension)}:{mode.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
