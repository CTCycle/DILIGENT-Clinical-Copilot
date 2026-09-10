from __future__ import annotations

from typing import Any


###############################################################################
def build_hepatotoxicity_pattern_context(
    *,
    classification: str | None,
    r_score: float | None,
    alt_multiple: float | None,
    alp_multiple: float | None,
) -> str:
    if not classification:
        return "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."

    normalized_classification = classification.replace("_", " ")
    segments: list[str] = [
        f"Observed liver injury pattern: {normalized_classification.capitalize()}.",
    ]
    if r_score is not None:
        segments.append(f"R ratio ≈ {r_score:.2f}.")
    if alt_multiple is not None:
        segments.append(
            f"ALT is about {alt_multiple:.2f} × the upper reference limit."
        )
    if alp_multiple is not None:
        segments.append(
            f"ALP is about {alp_multiple:.2f} × the upper reference limit."
        )
    segments.append(
        "Treat drugs whose known hepatotoxicity pattern matches this classification as stronger causal candidates, and downgrade mismatches."
    )
    return " ".join(segments)


###############################################################################
def build_livertox_knowledge_fragment(*, livertox_excerpt: str) -> str:
    excerpt = livertox_excerpt if livertox_excerpt else "No local LiverTox excerpt available."
    return f"""LiverTox excerpt:
{excerpt}"""


###############################################################################
def build_dilirank_knowledge_fragment(
    *,
    records: list[dict[str, Any]],
) -> str:
    if not records:
        return "FDA DILIrank 2.0: No linked local DILIrank record available."
    lines = [
        "FDA DILIrank 2.0 structured drug-level evidence:",
        "Use this only as an external hepatotoxicity prior. Do not treat it as a patient-specific causality score, do not convert it into a numeric causality weight, and do not override chronology, dechallenge/rechallenge, competing causes, phenotype, or patient-specific evidence.",
    ]
    for record in records:
        parts = [
            f"LTKB ID={str(record.get('ltkb_id') or 'N/A')}",
            f"compound={str(record.get('compound_name') or record.get('drug_name') or 'N/A')}",
            f"DILI concern={str(record.get('dili_concern') or 'N/A')}",
        ]
        severity_class = str(record.get("severity_class") or "")
        label_section = str(record.get("label_section") or "")
        comment = str(record.get("comment") or "")
        if severity_class:
            parts.append(f"severity class={severity_class}")
        if label_section:
            parts.append(f"label section={label_section}")
        if comment:
            parts.append(f"comment={comment}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)
