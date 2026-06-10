from __future__ import annotations

from pathlib import Path


###############################################################################
def test_no_forbidden_dependency_strings_remain() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    blocked_terms = ("".join(["lang", "chain"]), "".join(["lang", "smith"]))
    casefolded_terms = tuple(term.casefold() for term in blocked_terms)
    text_suffixes = {".py", ".toml", ".bat", ".ps1", ".md", ".json", ".yml", ".yaml"}
    roots = (
        repo_root / "app/server",
        repo_root / "app/tests",
        repo_root / "assets/docs",
        repo_root / "settings",
        repo_root / ".github",
    )

    pyproject_text = (
        (repo_root / "app/server/pyproject.toml")
        .read_text(encoding="utf-8", errors="ignore")
        .casefold()
    )
    for term in casefolded_terms:
        assert term not in pyproject_text

    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(
                part in {"__pycache__", ".pytest_cache", ".venv"} for part in path.parts
            ):
                continue
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").casefold()
            for term in casefolded_terms:
                if term in content:
                    hits.append(f"{path}: {term}")

    assert not hits, "\n".join(hits)
