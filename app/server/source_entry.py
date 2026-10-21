# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import argparse

import uvicorn

from app import app
from services.runtime.desktop import get_desktop_runtime_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DILIGENT source-mode web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7690)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Source-mode server must bind to a loopback host.")
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=arguments.host,
            port=arguments.port,
            reload=False,
            workers=1,
            log_level="info",
            access_log=False,
            timeout_graceful_shutdown=5,
        )
    )
    get_desktop_runtime_service().attach_server(server)
    server.run()


if __name__ == "__main__":
    main()
