# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass

###############################################################################
@dataclass(slots=True)
class RxNormCandidate:
    value: str
    kind: str
