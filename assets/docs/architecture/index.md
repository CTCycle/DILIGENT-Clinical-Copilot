# Architecture Index
Last updated: 2026-06-03

## Scope
Use this branch for repository structure, backend boundaries, runtime flow, persistence, and job execution.

## Documents
- [System Overview](system_overview.md)
  - Repository layout, maintained source tree, and application entry points.
- [API Surface](api_surface.md)
  - Route inventory and stable `/api` contract boundaries.
- [Backend Layers](backend_layers.md)
  - Layer responsibilities, request flow examples, and async or sync behavior.
- [Persistence](persistence.md)
  - Database, vector store, resource files, and persisted evidence artifacts.
- [Background Jobs](background_jobs.md)
  - Job manager lifecycle, state contract, polling, and cancellation.

## When To Read Which File
- Open `system_overview.md` for repository orientation or entry points.
- Open `api_surface.md` for endpoint discovery and route mapping.
- Open `backend_layers.md` for service boundaries or execution flow.
- Open `persistence.md` for storage and artifact questions.
- Open `background_jobs.md` for long-running operation behavior.
