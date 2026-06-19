from __future__ import annotations

from pathlib import Path


###############################################################################
def test_extraction_tool_architecture_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    removed_paths = [
        repo_root / "resources" / "tools" / "extraction_tools.json",
        repo_root / "server" / "services" / "extraction_tools",
        repo_root / "server" / "services" / "llm" / "tool_calling.py",
    ]

    assert all(not path.exists() for path in removed_paths)
