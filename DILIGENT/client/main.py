from __future__ import annotations

import http.server
import os
import socketserver

from DILIGENT.server.packages.configurations import client_settings

DIST_DIR = os.path.join(os.path.dirname(__file__), "dist")
INDEX_FILE = "index.html"


###############################################################################
class SPARequestHandler(http.server.SimpleHTTPRequestHandler):

    # -------------------------------------------------------------------------
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    # -------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self.send_error(404, "API routes are served by the backend.")
            return

        candidate = self.translate_path(self.path)
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, INDEX_FILE)

        if os.path.exists(candidate) and not os.path.isdir(candidate):
            http.server.SimpleHTTPRequestHandler.do_GET(self)
            return

        self.path = f"/{INDEX_FILE}"
        http.server.SimpleHTTPRequestHandler.do_GET(self)


# ----------------------------------------------------------------------------- 
def ensure_build_directory() -> str:
    if not os.path.isdir(DIST_DIR):
        raise RuntimeError(
            "Frontend build not found. Run `npm install` and `npm run build` "
            "inside DILIGENT/client before starting the UI server."
        )

    index_path = os.path.join(DIST_DIR, INDEX_FILE)
    if not os.path.isfile(index_path):
        raise RuntimeError(
            f"Missing {INDEX_FILE} in {DIST_DIR}. Build the React app before starting."
        )

    return DIST_DIR


# ----------------------------------------------------------------------------- 
def run() -> None:
    build_dir = ensure_build_directory()
    os.chdir(build_dir)
    with socketserver.ThreadingTCPServer(
        (client_settings.ui.host, client_settings.ui.port),
        SPARequestHandler,
    ) as server:
        server.allow_reuse_address = True
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


###############################################################################
if __name__ in {"__main__", "__mp_main__"}:
    run()
