# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

###############################################################################
@dataclass
class EnvironmentBootstrapState:
    lock: Lock = field(default_factory=Lock)
    bootstrapped: bool = False
