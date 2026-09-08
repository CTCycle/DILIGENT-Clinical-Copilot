from __future__ import annotations

import asyncio

import pytest
from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult
from services.llm.transports import routed_gateway
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

###############################################################################
def test_opencode_go_chat_bypasses_catalog_failure_and_uses_direct_chat_route(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _transport()
    captured: dict[str, object] = {}

    ###############################################################################
    class FakeChatTransport:

        # -------------------------------------------------------------------------
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float,
            default_headers: dict[str, str],
        ) -> None:
            captured.update(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=default_headers,
            )

        # -------------------------------------------------------------------------
        async def chat(self, request: ChatRequest) -> ChatResult:
            captured["request"] = request
            return ChatResult(content='{"events":[]}')

        # -------------------------------------------------------------------------
        async def close(self) -> None:
            return None

    async def catalog_failure(
        *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(routed_gateway, "OpenAIChatTransport", FakeChatTransport)
    monkeypatch.setattr(transport, "list_models", catalog_failure)

    with caplog.at_level("INFO"):
        result = asyncio.run(
            transport.chat(
                ChatRequest(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": "clinical prompt"}],
                    json_mode=True,
                )
            )
        )

    assert result.content == '{"events":[]}'
    assert captured["base_url"] == "https://opencode.ai/zen/go/v1"
    request = captured["request"]
    assert isinstance(request, ChatRequest)
    assert request.model == "deepseek-v4-flash"
    headers = captured["default_headers"]
    assert isinstance(headers, dict)
    assert headers["x-opencode-session"]
    assert headers["User-Agent"] == "diligent-clinical-copilot/3.3"
    assert "clinical prompt" not in caplog.text
    assert "route_source=documented_opencode_go_route@2026-09-07" in caplog.text
    assert "Cloud chat request attempted" in caplog.text

###############################################################################
def test_opencode_go_missing_catalog_model_uses_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport()
    transport._models["catalog-model"] = CloudModelDescriptor(
        id="catalog-model",
        display_name="Catalog model",
    )
    captured: dict[str, object] = {}

    ###############################################################################
    class FakeChatTransport:

        # -------------------------------------------------------------------------
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float,
            default_headers: dict[str, str],
        ) -> None:
            captured["base_url"] = base_url

        # -------------------------------------------------------------------------
        async def chat(self, request: ChatRequest) -> ChatResult:
            captured["request"] = request
            return ChatResult(content="ok")

        # -------------------------------------------------------------------------
        async def close(self) -> None:
            return None

    async def catalog_failure(
        *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        raise AssertionError("OpenCode Go should not require catalog refresh")

    monkeypatch.setattr(routed_gateway, "OpenAIChatTransport", FakeChatTransport)
    monkeypatch.setattr(transport, "list_models", catalog_failure)

    result = asyncio.run(
        transport.chat(
            ChatRequest(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": "OK"}],
            )
        )
    )

    assert result.content == "ok"
    assert captured["base_url"] == "https://opencode.ai/zen/go/v1"

###############################################################################
def test_opencode_go_connectivity_check_uses_direct_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport()
    captured: dict[str, object] = {}

    ###############################################################################
    class FakeChatTransport:

        # -------------------------------------------------------------------------
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float,
            default_headers: dict[str, str],
        ) -> None:
            captured["base_url"] = base_url

        # -------------------------------------------------------------------------
        async def chat(self, request: ChatRequest) -> ChatResult:
            captured["request"] = request
            return ChatResult(content="OK")

        # -------------------------------------------------------------------------
        async def close(self) -> None:
            return None

    monkeypatch.setattr(routed_gateway, "OpenAIChatTransport", FakeChatTransport)

    result = asyncio.run(transport.check_connectivity("deepseek-v4-flash"))

    assert result.ok is True
    assert result.response_preview == "OK"
    assert captured["base_url"] == "https://opencode.ai/zen/go/v1"
    request = captured["request"]
    assert isinstance(request, ChatRequest)
    assert request.messages == [{"role": "user", "content": "Reply with exactly: OK"}]

###############################################################################
def test_other_routed_gateways_keep_strict_catalog_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport("/zen/v1/models")

    async def catalog_failure(
        *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(transport, "list_models", catalog_failure)

    with pytest.raises(
        ValueError, match="Provider model metadata is unavailable for unknown"
    ):
        asyncio.run(
            transport.chat(
                ChatRequest(
                    model="unknown",
                    messages=[{"role": "user", "content": "OK"}],
                )
            )
        )

###############################################################################
@pytest.mark.parametrize(
    ("model", "expected_endpoint"),
    [
        ("deepseek-v4-flash", "chat/completions"),
        ("qwen3.8-flash", "messages"),
        ("minimax-m3", "messages"),
        ("gpt-5.6-luna", "responses"),
        ("grok-4.6", "responses"),
        ("muse-spark-1.3-contributor", "responses"),
        ("kimi-k2.7-code", "chat/completions"),
        ("longcat-2.0", "chat/completions"),
        ("mimo-v2.5-pro", "chat/completions"),
        ("hy4-preview", "chat/completions"),
        ("hy3", "chat/completions"),
    ],
)
def test_opencode_go_documented_route_matrix(
    model: str, expected_endpoint: str
) -> None:
    endpoint = _transport()._resolve_transport_endpoint(
        CloudModelDescriptor(id=model, display_name=model)
    )

    assert endpoint == expected_endpoint

###############################################################################
def test_opencode_go_unknown_model_does_not_guess_chat_route() -> None:
    endpoint = _transport()._resolve_transport_endpoint(
        CloudModelDescriptor(id="unlisted-model", display_name="Unlisted")
    )

    assert endpoint == ""
