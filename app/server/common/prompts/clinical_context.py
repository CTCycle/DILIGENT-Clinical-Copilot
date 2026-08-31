from __future__ import annotations


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


def build_livertox_knowledge_fragment(*, livertox_excerpt: str) -> str:
    excerpt = livertox_excerpt if livertox_excerpt else "No local LiverTox excerpt available."
    return f"""LiverTox excerpt:
{excerpt}"""
