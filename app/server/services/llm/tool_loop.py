from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from domain.llm.transports import ChatMessage, ChatResult, ToolCall, ToolDefinition
from services.llm.generation_policy import GenerationPurpose

ChatCall = Callable[..., Awaitable[ChatResult]]
ToolExecutor = Callable[[str, dict[str, Any]], object | Awaitable[object]]
StopCheck = Callable[[], bool]

###############################################################################
@dataclass(frozen=True)
class ToolLoopResult:
    result: ChatResult
    messages: tuple[ChatMessage, ...]
    observations: tuple[dict[str, Any], ...]
    iterations: int
    tool_call_count: int
    stopped: bool = False

###############################################################################
class ToolLoopExecutor:
    """Run the provider-neutral assistant/tool/result continuation protocol."""

    def __init__(
        self,
        *,
        chat: ChatCall,
        execute: ToolExecutor,
        max_iterations: int,
        max_tool_calls: int,
        stop_check: StopCheck | None = None,
    ) -> None:
        self.chat = chat
        self.execute = execute
        self.max_iterations = max(1, int(max_iterations))
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.stop_check = stop_check

    # -------------------------------------------------------------------------
    async def run(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        purpose: GenerationPurpose,
    ) -> ToolLoopResult:
        working = list(messages)
        definitions = {tool.name: tool for tool in tools}
        observations: list[dict[str, Any]] = []
        last_result = ChatResult(content="")
        tool_call_count = 0

        for iteration in range(1, self.max_iterations + 1):
            if self._should_stop():
                return self._stopped_result(
                    last_result,
                    working,
                    observations,
                    iteration - 1,
                    tool_call_count,
                )
            last_result = await self.chat(
                model=model,
                messages=working,
                tools=list(tools),
                # Omitting tool_choice is the portable "auto" behavior.  Some
                # reasoning APIs, including DeepSeek V4.1 Flash, reject that
                # parameter even though native tools are supported.
                tool_choice=None,
                purpose=purpose,
                operation="chat",
                cancel_check=self.stop_check,
            )
            # DeepSeek thinking-mode tool turns require an explicit assistant
            # content field even when the visible content is empty.
            working.append(
                ChatMessage(
                    role="assistant",
                    content=last_result.content or "",
                    reasoning_content=last_result.reasoning_content,
                    tool_calls=list(last_result.tool_calls),
                )
            )
            if not last_result.tool_calls:
                return ToolLoopResult(
                    result=last_result,
                    messages=tuple(working),
                    observations=tuple(observations),
                    iterations=iteration,
                    tool_call_count=tool_call_count,
                )

            remaining = self.max_tool_calls - tool_call_count
            if remaining <= 0:
                return self._limit_result(
                    last_result,
                    working,
                    observations,
                    iteration,
                    tool_call_count,
                )
            for call in last_result.tool_calls[:remaining]:
                observation = await self._execute_call(call, definitions)
                entry = {"tool": call.name, "tool_call_id": call.id, "observation": observation}
                observations.append(entry)
                working.append(
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        content=self._serialize_output(observation),
                        tool_call_id=call.id,
                    )
                )
                tool_call_count += 1
                if self._should_stop():
                    return self._stopped_result(
                        last_result,
                        working,
                        observations,
                        iteration,
                        tool_call_count,
                    )
            if len(last_result.tool_calls) > remaining:
                return self._limit_result(
                    last_result,
                    working,
                    observations,
                    iteration,
                    tool_call_count,
                )

        return self._limit_result(
            last_result,
            working,
            observations,
            self.max_iterations,
            tool_call_count,
        )

    # -------------------------------------------------------------------------
    async def _execute_call(
        self,
        call: ToolCall,
        definitions: dict[str, ToolDefinition],
    ) -> object:
        if call.name not in definitions:
            return {"error": "Unknown or disallowed tool.", "invalid_tool": True}
        arguments = call.arguments
        if call.raw_arguments is not None:
            try:
                parsed = json.loads(call.raw_arguments)
            except json.JSONDecodeError:
                return {"error": "Malformed tool arguments.", "invalid_tool_input": True}
            if not isinstance(parsed, dict):
                return {"error": "Tool arguments must be a JSON object.", "invalid_tool_input": True}
            arguments = parsed
        try:
            value = self.execute(call.name, arguments)
            return await value if inspect.isawaitable(value) else value
        except (TypeError, ValueError) as exc:
            return {"error": str(exc), "invalid_tool_input": True}

    # -------------------------------------------------------------------------
    def _should_stop(self) -> bool:
        return self.stop_check is not None and bool(self.stop_check())

    # -------------------------------------------------------------------------
    @staticmethod
    def _serialize_output(value: object) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def _limit_result(
        result: ChatResult,
        messages: list[ChatMessage],
        observations: list[dict[str, Any]],
        iterations: int,
        tool_call_count: int,
    ) -> ToolLoopResult:
        return ToolLoopResult(
            result=result.model_copy(update={"partial": True, "finish_reason": "tool_limit"}),
            messages=tuple(messages),
            observations=tuple(observations),
            iterations=iterations,
            tool_call_count=tool_call_count,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _stopped_result(
        result: ChatResult,
        messages: list[ChatMessage],
        observations: list[dict[str, Any]],
        iterations: int,
        tool_call_count: int,
    ) -> ToolLoopResult:
        return ToolLoopResult(
            result=result.model_copy(update={"partial": True, "finish_reason": "cancelled"}),
            messages=tuple(messages),
            observations=tuple(observations),
            iterations=iterations,
            tool_call_count=tool_call_count,
            stopped=True,
        )
