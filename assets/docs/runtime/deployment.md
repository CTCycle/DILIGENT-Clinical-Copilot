# Local Deployment
Last updated: 2026-07-23

## Supported Runtime
- DILIGENT supports local single-user operation.
- On Windows, `start_on_windows.ps1` prepares portable runtimes, dependencies, and the frontend build before launching the local services.
- The frontend runtime baseline is Node.js 22.13.0 or newer within the Node.js 22 line; this is required by the locked `jsdom` version.
- RAG requires `numpy`, `onnxruntime`, and `tokenizers`; the canonical artifact is a pinned AVX2 `uint8` ONNX model. PyTorch and Sentence Transformers are not required.
- Manual macOS and Linux startup requires compatible Python, Node.js, and npm installations.

## Dependency Locks
- `runtimes/uv.lock`
- `app/client/package-lock.json`

## Deployment Constraints
- Network deployment, reverse proxies, and unauthenticated multi-user access are unsupported.
- No supported container deployment path exists.
- Backend resources and the frontend build must remain aligned within the local repository checkout.
- Offline deployments must pre-populate and verify `app/resources/models/embeddings/<revision>/`; a complete rebuild is mandatory after this model migration.
