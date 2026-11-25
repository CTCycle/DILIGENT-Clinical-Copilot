# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import NamedTuple


###############################################################################
class EmbeddingModelSpec(NamedTuple):
    provider: str
    model_name: str
    dimension: int
    mode: str
    signature: str

