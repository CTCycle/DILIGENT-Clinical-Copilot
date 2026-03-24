from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


###############################################################################
class RootEndpoint:
    def __init__(
        self,
        *,
        app: FastAPI,
        cloud_mode: bool,
        tauri_mode: bool,
    ) -> None:
        self.app = app
        self.cloud_mode = cloud_mode
        self.tauri_mode = tauri_mode
        self.client_dist_path = os.path.abspath(self.get_client_dist_path())

    # -------------------------------------------------------------------------
    @staticmethod
    def get_client_dist_path() -> str:
        project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(project_path, "client", "dist")

    # -------------------------------------------------------------------------
    def packaged_client_available(self) -> bool:
        return self.tauri_mode and os.path.isdir(self.client_dist_path)

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_safe_client_path(client_dist_path: str, relative_path: str) -> str | None:
        root_path = os.path.abspath(client_dist_path)
        candidate_path = os.path.abspath(os.path.join(root_path, relative_path))
        try:
            common_path = os.path.commonpath([root_path, candidate_path])
        except ValueError:
            return None
        if common_path != root_path:
            return None
        return candidate_path

    # -------------------------------------------------------------------------
    def serve_spa_root(self) -> FileResponse:
        return FileResponse(os.path.join(self.client_dist_path, "index.html"))

    # -------------------------------------------------------------------------
    def serve_spa_entrypoint(self, full_path: str) -> FileResponse:
        safe_path = self.resolve_safe_client_path(self.client_dist_path, full_path)
        if safe_path and os.path.isfile(safe_path):
            return FileResponse(safe_path)
        return FileResponse(os.path.join(self.client_dist_path, "index.html"))

    # -------------------------------------------------------------------------
    def redirect_root(self) -> RedirectResponse:
        return RedirectResponse(url="/api/model-config" if self.cloud_mode else "/docs")

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        if self.packaged_client_available():
            assets_path = os.path.join(self.client_dist_path, "assets")
            if os.path.isdir(assets_path):
                self.app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")
            self.app.add_api_route("/", self.serve_spa_root, methods=["GET"], include_in_schema=False)
            self.app.add_api_route(
                "/{full_path:path}",
                self.serve_spa_entrypoint,
                methods=["GET"],
                include_in_schema=False,
            )
            return

        self.app.add_api_route("/", self.redirect_root, methods=["GET"])
