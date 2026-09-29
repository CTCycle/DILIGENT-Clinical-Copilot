"""Minimal deterministic Ollama protocol fixture for browser regression."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

DEFAULT_MODEL = "qwen3.5:2b"


###############################################################################
class FakeOllamaState:
    def __init__(self, model: str) -> None:
        self.lock = Lock()
        self.models: set[str] = {model}

    def list_models(self) -> list[str]:
        with self.lock:
            return sorted(self.models)

    def add_model(self, model: str) -> None:
        with self.lock:
            self.models.add(model)


###############################################################################
class FakeOllamaHandler(BaseHTTPRequestHandler):
    server: "FakeOllamaServer"

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/" or self.path == "/api/tags":
            self._send_json(self._tags_payload())
            return
        if self.path == "/api/ps":
            self._send_json({"models": []})
            return
        if self.path == "/api/version":
            self._send_json({"version": "ci-fake"})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/api/pull":
            payload = self._read_json()
            model = str(payload.get("name", "")).strip() or DEFAULT_MODEL
            self.server.state.add_model(model)
            if bool(payload.get("stream")):
                body = (
                    json.dumps({"status": "success", "model": model}) + "\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"status": "success", "model": model})
            return
        if self.path == "/api/show":
            payload = self._read_json()
            model = str(payload.get("name", "")).strip() or DEFAULT_MODEL
            self._send_json(
                {
                    "details": {
                        "family": "qwen",
                        "parameter_size": "2B",
                        "quantization_level": "Q4_K_M",
                    },
                    "model_info": {"general.architecture": "qwen"},
                    "name": model,
                }
            )
            return
        if self.path in {"/api/embed", "/api/embeddings"}:
            payload = self._read_json()
            input_value = payload.get("input", payload.get("prompt", []))
            inputs = input_value if isinstance(input_value, list) else [input_value]
            self._send_json({"embeddings": [[0.0, 1.0] for _ in inputs]})
            return
        self._send_json({"error": "not found"}, status=404)

    def _tags_payload(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": model,
                    "model": model,
                    "size": 1_000_000,
                    "digest": "ci-fake-digest",
                    "details": {
                        "family": "qwen",
                        "parameter_size": "2B",
                        "quantization_level": "Q4_K_M",
                    },
                }
                for model in self.server.state.list_models()
            ]
        }

    def log_message(self, _format: str, *_args: object) -> None:
        return


###############################################################################
class FakeOllamaServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], model: str) -> None:
        self.state = FakeOllamaState(model)
        super().__init__(address, FakeOllamaHandler)


###############################################################################
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    server = FakeOllamaServer((args.host, args.port), args.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
