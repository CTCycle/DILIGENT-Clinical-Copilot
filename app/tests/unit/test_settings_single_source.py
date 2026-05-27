from __future__ import annotations

from pathlib import Path
import re


def test_no_server_settings_symbol_used_in_backend() -> None:
    root = Path(__file__).resolve().parents[2] / "server"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        has_forbidden_import = bool(
            re.search(
                r"from\s+configurations\.startup\s+import\s+.*\bserver_settings\b",
                text,
            )
        )
        has_forbidden_reference = "startup.server_settings" in text
        if has_forbidden_import or has_forbidden_reference:
            offenders.append(str(path))
    assert offenders == [], offenders
