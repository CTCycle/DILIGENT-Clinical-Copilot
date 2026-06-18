from __future__ import annotations

from services.llm.ollama_client import OllamaError as ClientOllamaError
from services.llm.ollama_chat import OllamaError as ChatOllamaError
from services.llm.ollama_residency import OllamaError as ResidencyOllamaError
from services.llm.ollama_runtime import OllamaError, OllamaTimeout
from services.llm.ollama_structured import OllamaError as StructuredOllamaError

###############################################################################
def test_ollama_error_single_source() -> None:
    assert ClientOllamaError is OllamaError
    assert ChatOllamaError is OllamaError
    assert ResidencyOllamaError is OllamaError
    assert StructuredOllamaError is OllamaError

###############################################################################
def test_ollama_timeout_inherits_single_source_error() -> None:
    assert issubclass(OllamaTimeout, OllamaError)
