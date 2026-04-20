from __future__ import annotations

from DILIGENT.server.services.model_config_service import ModelConfigService


###############################################################################
def sync_runtime_model_config() -> None:
    """Load persisted model-config snapshot and apply it to runtime state."""
    service = ModelConfigService()
    snapshot = service.ensure_defaults()
    service.apply_runtime_snapshot(snapshot)
