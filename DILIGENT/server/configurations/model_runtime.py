from __future__ import annotations

from DILIGENT.server.services.model_config_service import ModelConfigService


###############################################################################
def sync_runtime_model_config() -> None:
    """Ensure persisted model-config defaults exist in the database."""
    service = ModelConfigService()
    service.ensure_defaults()
