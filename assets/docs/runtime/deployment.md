# Local Deployment
Last updated: 2026-07-11

## Supported Runtime
- DILIGENT supports local single-user operation.
- On Windows, `start_on_windows.ps1` prepares portable runtimes, dependencies, and the frontend build before launching the local services.
- Manual macOS and Linux startup requires compatible Python, Node.js, and npm installations.

## Dependency Locks
- `runtimes/uv.lock`
- `app/client/package-lock.json`

## Deployment Constraints
- Network deployment, reverse proxies, and unauthenticated multi-user access are unsupported.
- No supported container deployment path exists.
- Backend resources and the frontend build must remain aligned within the local repository checkout.
