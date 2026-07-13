from __future__ import annotations

from pathlib import Path

###############################################################################
def test_server_python_sources_compile() -> None:
    server_root = Path(__file__).resolve().parents[2] / "server"
    python_files = [
        path
        for path in server_root.rglob("*.py")
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and ".ruff_cache" not in path.parts
    ]
    assert python_files
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
