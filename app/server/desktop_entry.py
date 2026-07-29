from __future__ import annotations

import argparse
import json
import os
import re
import socket
from importlib import import_module
from pathlib import Path
from typing import Any

import uvicorn

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def _absolute_environment_path(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for desktop execution")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path")
    return path.resolve()


def _validate_desktop_environment() -> str:
    _absolute_environment_path("DILIGENT_RUNTIME_ROOT")
    _absolute_environment_path("DILIGENT_DATA_ROOT")
    version = os.getenv("DILIGENT_RELEASE_VERSION", "").strip()
    if not _VERSION_PATTERN.fullmatch(version):
        raise RuntimeError("DILIGENT_RELEASE_VERSION must be major.minor.patch")
    return version


def _write_ready_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DILIGENT packaged backend")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser


def run_desktop_backend(*, ready_file: Path, host: str = "127.0.0.1") -> None:
    if host != "127.0.0.1":
        raise ValueError("Desktop backend host must be 127.0.0.1")
    release_version = _validate_desktop_environment()

    # Import the application only after the packaged roots have been checked.
    application = import_module("app").app

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, 0))
    server_socket.listen(socket.SOMAXCONN)
    server_socket.set_inheritable(False)
    port = int(server_socket.getsockname()[1])
    _write_ready_file(
        ready_file,
        {"port": port, "pid": os.getpid(), "release_version": release_version},
    )

    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[server_socket])
    finally:
        ready_file.unlink(missing_ok=True)
        server_socket.close()


def main() -> None:
    arguments = build_parser().parse_args()
    run_desktop_backend(ready_file=arguments.ready_file, host=arguments.host)


if __name__ == "__main__":
    main()
