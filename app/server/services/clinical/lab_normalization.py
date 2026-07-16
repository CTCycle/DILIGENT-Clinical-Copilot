from __future__ import annotations

from services.catalogs.runtime import get_reference_catalog_snapshot


###############################################################################
def get_marker_aliases() -> dict[str, tuple[str, ...]]:
    snapshot = get_reference_catalog_snapshot()
    entries = snapshot.entries("clinical_extraction", "laboratory_markers")
    by_key: dict[str, list[str]] = {}
    for entry in entries:
        by_key.setdefault(entry.key.upper(), []).append(entry.value.casefold())
    if by_key:
        return {key: tuple(dict.fromkeys(values)) for key, values in by_key.items()}
    return {
        "ALT": ("alt", "alat", "gpt"),
        "AST": ("ast", "asat", "got"),
        "ALP": ("alp", "alkp", "alkaline phosphatase"),
        "TBIL": ("tbil", "total bilirubin", "bilirubin total", "bilirubin"),
        "DBIL": ("dbil", "direct bilirubin", "bilirubin direct"),
        "GGT": ("ggt", "gamma gt", "gamma-glutamyl transferase"),
        "INR": ("inr",),
        "ALB": ("albumin", "alb"),
    }
