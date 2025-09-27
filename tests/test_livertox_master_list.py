from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from typing import Any
from unittest.mock import patch


if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:  # pragma: no cover - defensive stub
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("httpx stub should not be used in tests")

    class _Response:  # pragma: no cover - defensive stub
        pass

    httpx_stub.AsyncClient = _AsyncClient
    httpx_stub.Response = _Response
    httpx_stub.HTTPStatusError = Exception
    httpx_stub.HTTPError = Exception
    httpx_stub.URL = lambda value: value
    sys.modules["httpx"] = httpx_stub

if "pdfminer" not in sys.modules:
    pdfminer_stub = types.ModuleType("pdfminer")
    high_level_stub = types.ModuleType("pdfminer.high_level")

    def _extract_text(*args: Any, **kwargs: Any) -> str:  # pragma: no cover - stub
        raise RuntimeError("pdfminer stub should not be used in tests")

    high_level_stub.extract_text = _extract_text
    sys.modules["pdfminer"] = pdfminer_stub
    sys.modules["pdfminer.high_level"] = high_level_stub

if "pypdf" not in sys.modules:
    pypdf_stub = types.ModuleType("pypdf")

    class _PdfReader:  # pragma: no cover - defensive stub
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pypdf stub should not be used in tests")

    pypdf_stub.PdfReader = _PdfReader
    sys.modules["pypdf"] = pypdf_stub

providers_module_name = "Pharmagent.app.api.models.providers"
if providers_module_name not in sys.modules:
    providers_stub = types.ModuleType(providers_module_name)

    async def _llm_structured_call(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None

    class _StubClient:  # pragma: no cover - defensive stub
        async def llm_structured_call(self, *args: Any, **kwargs: Any) -> None:
            return None

    def initialize_llm_client(*args: Any, **kwargs: Any) -> _StubClient:
        return _StubClient()

    providers_stub.initialize_llm_client = initialize_llm_client
    sys.modules[providers_module_name] = providers_stub

clinical_module_name = "Pharmagent.app.api.schemas.clinical"
if clinical_module_name not in sys.modules:
    clinical_stub = types.ModuleType(clinical_module_name)

    class LiverToxBatchMatchSuggestion:  # pragma: no cover - defensive stub
        def __init__(self) -> None:
            self.matches: list[Any] = []

    clinical_stub.LiverToxBatchMatchSuggestion = LiverToxBatchMatchSuggestion
    sys.modules[clinical_module_name] = clinical_stub

import pandas as pd

from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.livertox import LiverToxToolkit, LiverToxUpdater


###############################################################################
class _RecordingSerializer(DataSerializer):
    def __init__(self) -> None:
        super().__init__()
        self.saved_frame: pd.DataFrame | None = None
        self.saved_kwargs: dict[str, Any] | None = None

    ###########################################################################
    def save_livertox_master_list(
        self, frame: pd.DataFrame, *, source_url: str, last_modified: str | None
    ) -> None:
        self.saved_frame = frame.copy()
        self.saved_kwargs = {"source_url": source_url, "last_modified": last_modified}

    ###########################################################################
    def save_livertox_records(self, records: list[dict[str, Any]]) -> None:
        # Tests do not exercise record persistence.
        raise AssertionError("Records should not be saved during this test")


###############################################################################
class _StubRxClient:
    ###########################################################################
    def fetch_drug_terms(self, drug_name: str) -> tuple[list[str], list[str]]:
        return [], []


###############################################################################
class _TestUpdater(LiverToxUpdater):
    ###########################################################################
    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        sources_path: str,
        serializer: DataSerializer,
    ) -> None:
        super().__init__(
            sources_path,
            redownload=False,
            rx_client=_StubRxClient(),
            serializer=serializer,
            database_client=None,
        )
        self._metadata = metadata

    ###########################################################################
    async def _download_master_list(self) -> dict[str, Any]:
        return self._metadata


###############################################################################
class LiverToxMasterListTests(unittest.TestCase):
    ###########################################################################
    def test_sanitize_master_list_handles_extended_headers(self) -> None:
        toolkit = LiverToxToolkit()
        frame = pd.DataFrame(
            {
                "Ingredient Name": ["Drug A", "Drug B"],
                "Brand": ["BrandA", "BrandB"],
                "Likelihood Score": ["A", "B"],
                "Chapter Title": ["Chapter A", "Chapter B"],
                "Last Update": ["2024-01-01", "2024-02-01"],
                "Reference Count": [5, 10],
                "Year Approved": [2001, 2002],
                "Agent Classification": ["Class A", "Class B"],
                "Include in LiverTox": ["Yes", "Yes"],
            }
        )

        sanitized = toolkit.sanitize_livertox_master_list(frame)

        self.assertEqual(2, len(sanitized.index))
        self.assertListEqual(["Drug A", "Drug B"], sanitized["ingredient"].tolist())

    ###########################################################################
    def test_refresh_master_list_saves_multiple_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "Ingredient Name": ["Drug C", "Drug D"],
                "Brand Name": ["BrandC", "BrandD"],
                "Likelihood Score": ["C", "D"],
                "Chapter Title": ["Chapter C", "Chapter D"],
                "Last Update": ["2024-03-01", "2024-04-01"],
                "Reference Count": [12, 8],
                "Year Approved": [2003, 2004],
                "Agent Classification": ["Class C", "Class D"],
                "Include in LiverTox": ["Yes", "Yes"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            excel_path = os.path.join(temp_dir, "master.xlsx")
            metadata = {
                "file_path": excel_path,
                "size": 0,
                "last_modified": "Wed, 01 Jan 2025 00:00:00 GMT",
                "downloaded": True,
                "source_url": "https://example.test/master.xlsx",
            }

            serializer = _RecordingSerializer()
            updater = _TestUpdater(
                metadata,
                sources_path=temp_dir,
                serializer=serializer,
            )

            with patch("pandas.read_excel", return_value=frame.copy()):
                result = updater.refresh_master_list()

        self.assertIsNotNone(serializer.saved_frame)
        assert serializer.saved_frame is not None  # for type checkers
        self.assertEqual(2, len(serializer.saved_frame.index))
        self.assertEqual(2, result.get("records"))
        self.assertEqual("https://example.test/master.xlsx", result.get("source_url"))


if __name__ == "__main__":
    unittest.main()
