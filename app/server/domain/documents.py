# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

###############################################################################
@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
