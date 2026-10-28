# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from typing import Any

# LiverTox likelihood scale (https://www.ncbi.nlm.nih.gov/books/NBK548392/).
LIVERTOX_LIKELIHOOD_GRADES = frozenset({"A", "B", "C", "D", "E", "E*", "X"})
LIVERTOX_QUALIFIER_RE = re.compile(r"\[[^\]]*\]")


###############################################################################
def normalize_livertox_likelihood(value: Any) -> str | None:
    """LiverTox likelihood grade (A-E, E*, X) or None when missing or malformed.

    Qualifiers such as "[HD]" (high dose) are dropped; values like "NAN" from
    missing table cells are treated as unknown rather than as a grade.
    """
    if not isinstance(value, str):
        return None
    grade = LIVERTOX_QUALIFIER_RE.sub("", value).replace(" ", "").strip().upper()
    return grade if grade in LIVERTOX_LIKELIHOOD_GRADES else None
