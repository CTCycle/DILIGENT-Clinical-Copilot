from __future__ import annotations

from domain.llm.providers import CloudModelDescriptor
from services.llm.transports.routed_gateway import RoutedGatewayTransport


###############################################################################
def _transport(models_path: str = "/zen/go/v1/models") -> RoutedGatewayTransport:
    return RoutedGatewayTransport(
        api_key="test-key",
        base_url="https://opencode.ai",
        models_path=models_path,
        timeout=1.0,
    )


###############################################################################
def test_opencode_go_deepseek_flash_uses_documented_chat_endpoint() -> None:
    endpoint = _transport()._resolve_transport_endpoint(
        CloudModelDescriptor(
            id="deepseek-v4-flash",
            display_name="DeepSeek V4 Flash",
        )
    )

    assert endpoint == "chat/completions"


###############################################################################
def test_opencode_go_anthropic_models_use_messages_endpoint() -> None:
    endpoint = _transport()._resolve_transport_endpoint(
        CloudModelDescriptor(id="minimax-m3", display_name="MiniMax M3")
    )

    assert endpoint == "messages"


###############################################################################
def test_other_routed_gateways_still_require_model_endpoint_metadata() -> None:
    endpoint = _transport("/zen/v1/models")._resolve_transport_endpoint(
        CloudModelDescriptor(id="unknown", display_name="Unknown")
    )

    assert endpoint == ""
