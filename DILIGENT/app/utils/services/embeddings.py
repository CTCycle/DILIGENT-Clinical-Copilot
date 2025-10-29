from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings

from DILIGENT.app.constants import GEMINI_API_BASE, OPENAI_API_BASE


###############################################################################
class CloudEmbeddingClient:
    def __init__(self, provider: str | None, model: str | None) -> None:
        if not provider:
            raise ValueError("Cloud provider is required for embeddings")
        if not model:
            raise ValueError("Cloud embedding model is required")
        self.provider = provider.lower()
        self.model = model

    # -------------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            return self.embed_openai(texts)
        if self.provider == "gemini":
            return self.embed_gemini(texts)
        raise ValueError(f"Unsupported cloud embedding provider: {self.provider}")

    # -------------------------------------------------------------------------
    def embed_openai(self, texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{OPENAI_API_BASE}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError("Failed to request OpenAI embeddings") from exc
        data = json.loads(body)
        embeddings: list[list[float]] = []
        entries = sorted(
            data.get("data", []), key=lambda entry: entry.get("index", 0)
        )
        for item in entries:
            values = [float(value) for value in item.get("embedding", [])]
            embeddings.append(values)
        if len(embeddings) != len(texts):
            raise RuntimeError("Mismatch between OpenAI embeddings and inputs")
        return embeddings

    # -------------------------------------------------------------------------
    def embed_gemini(self, texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        requests_payload = [
            {"content": {"parts": [{"text": text}]}}
            for text in texts
        ]
        payload = json.dumps({"requests": requests_payload}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{GEMINI_API_BASE}/models/{self.model}:batchEmbedContents?key={api_key}",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError("Failed to request Gemini embeddings") from exc
        data = json.loads(body)
        embeddings: list[list[float]] = []
        for item in data.get("embeddings", []):
            values = item.get("values") or item.get("embedding") or []
            embeddings.append([float(value) for value in values])
        if len(embeddings) != len(texts):
            raise RuntimeError("Mismatch between Gemini embeddings and inputs")
        return embeddings


###############################################################################
class EmbeddingGenerator:
    def __init__(
        self,
        backend: str,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        hf_model: str | None = None,
        use_cloud_embeddings: bool = False,
        cloud_provider: str | None = None,
        cloud_model: str | None = None,
    ) -> None:
        normalized_backend = backend.lower().strip() if backend else "ollama"
        self.backend = "cloud" if use_cloud_embeddings else normalized_backend
        if self.backend == "ollama":
            if not ollama_model:
                raise ValueError("Ollama embedding model is required")
            self.embedder = OllamaEmbeddings(
                base_url=ollama_base_url,
                model=ollama_model,
            )
        elif self.backend == "huggingface":
            if not hf_model:
                raise ValueError("Hugging Face embedding model is required")
            self.embedder = HuggingFaceEmbeddings(model_name=hf_model)
        elif self.backend == "cloud":
            self.embedder = CloudEmbeddingClient(cloud_provider, cloud_model)
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

    # -------------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        sanitized = [text if text.strip() else " " for text in texts]
        embeddings = self.embedder.embed_documents(sanitized)
        return [[float(value) for value in vector] for vector in embeddings]


__all__ = ["CloudEmbeddingClient", "EmbeddingGenerator"]
