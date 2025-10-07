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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

providers_module_name = "DILIGENT.app.api.models.providers"
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

clinical_module_name = "DILIGENT.app.api.schemas.clinical"
if clinical_module_name not in sys.modules:
    clinical_stub = types.ModuleType(clinical_module_name)

    class LiverToxBatchMatchSuggestion:  # pragma: no cover - defensive stub
        def __init__(self) -> None:
            self.matches: list[Any] = []

    clinical_stub.LiverToxBatchMatchSuggestion = LiverToxBatchMatchSuggestion
    sys.modules[clinical_module_name] = clinical_stub

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.app.utils.serializer import DataSerializer
from DILIGENT.app.utils.services.updater import LiverToxUpdater
from DILIGENT.app.utils.database import sqlite as sqlite_module


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
    def fetch_drug_terms(self, drug_name: str) -> list[str]:
        return []


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
    def _create_updater(self) -> LiverToxUpdater:
        return LiverToxUpdater(
            ".",
            redownload=False,
            rx_client=_StubRxClient(),
            serializer=_RecordingSerializer(),
            database_client=sqlite_module.database,
        )

    ###########################################################################
    def test_sanitize_master_list_handles_extended_headers(self) -> None:
        updater = self._create_updater()
        frame = pd.DataFrame(
            {
                "ingredient": ["Drug A", "Drug B"],
                "brand_name": ["BrandA", "BrandB"],
                "likelihood_score": ["A", "B"],
                "chapter_title": ["Chapter A", "Chapter B"],
                "last_update": ["2024-01-01", "2024-02-01"],
                "reference_count": [5, 10],
                "year_approved": [2001, 2002],
                "agent_classification": ["Class A", "Class B"],
                "include_in_livertox": ["Yes", "Yes"],
                "extra_column": ["value1", "value2"],
            }
        )

        sanitized = updater.sanitize_livertox_master_list(frame)

        self.assertEqual(2, len(sanitized.index))
        self.assertListEqual(["Drug A", "Drug B"], sanitized["ingredient"].tolist())

    ###########################################################################
    def test_sanitize_master_list_requires_brand_name(self) -> None:
        updater = self._create_updater()
        frame = pd.DataFrame(
            {
                "ingredient": ["Drug X", "Drug Y"],
                "brand_name": ["BrandX", None],
                "likelihood_score": ["X", "Y"],
                "chapter_title": ["Chapter X", "Chapter Y"],
                "last_update": ["2024-07-01", "2024-08-01"],
                "reference_count": [2, 4],
                "year_approved": [2010, 2011],
                "agent_classification": ["Class X", "Class Y"],
                "include_in_livertox": ["Yes", "No"],
            }
        )

        sanitized = updater.sanitize_livertox_master_list(frame)

        self.assertEqual(1, len(sanitized.index))
        self.assertListEqual(["Drug X"], sanitized["ingredient"].tolist())

    ###########################################################################
    def test_refresh_master_list_saves_multiple_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "ingredient": ["Drug C", "Drug D", "Drug E"],
                "brand_name": ["BrandC", "BrandD", None],
                "likelihood_score": ["C", "D", "E"],
                "chapter_title": ["Chapter C", "Chapter D", "Chapter E"],
                "last_update": ["2024-03-01", "2024-04-01", "2024-05-01"],
                "reference_count": [12, 8, 5],
                "year_approved": [2003, 2004, 2005],
                "agent_classification": ["Class C", "Class D", "Class E"],
                "include_in_livertox": ["Yes", "Yes", "No"],
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

    ###########################################################################
    def test_refresh_master_list_populates_database_table(self) -> None:
        frame = pd.DataFrame(
            {
                "ingredient": ["Drug F", "Drug G", "Drug H"],
                "brand_name": ["BrandF", "BrandG", None],
                "likelihood_score": ["F", "G", "H"],
                "chapter_title": ["Chapter F", "Chapter G", "Chapter H"],
                "last_update": ["2024-05-01", "2024-06-01", "2024-07-01"],
                "reference_count": [7, 3, 9],
                "year_approved": [2005, 2006, 2007],
                "agent_classification": ["Class F", "Class G", "Class H"],
                "include_in_livertox": ["Yes", "Yes", "No"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            excel_path = os.path.join(temp_dir, "master.xlsx")
            metadata = {
                "file_path": excel_path,
                "size": 0,
                "last_modified": "Wed, 02 Jan 2025 00:00:00 GMT",
                "downloaded": True,
                "source_url": "https://example.test/master.xlsx",
            }

            db_path = os.path.join(temp_dir, "test.db")
            engine = create_engine(f"sqlite:///{db_path}", future=True)
            session_factory = sessionmaker(bind=engine, future=True)

            original_engine = sqlite_module.database.engine
            original_session_factory = sqlite_module.database.Session
            original_db_path = sqlite_module.database.db_path

            sqlite_module.Base.metadata.create_all(engine)
            sqlite_module.database.engine = engine
            sqlite_module.database.Session = session_factory
            sqlite_module.database.db_path = db_path

            try:
                serializer = DataSerializer()
                updater = _TestUpdater(
                    metadata,
                    sources_path=temp_dir,
                    serializer=serializer,
                )

                with patch("pandas.read_excel", return_value=frame.copy()):
                    result = updater.refresh_master_list()

                saved = sqlite_module.database.load_from_database(
                    "LIVERTOX_MASTER_LIST"
                )
            finally:
                sqlite_module.database.engine = original_engine
                sqlite_module.database.Session = original_session_factory
                sqlite_module.database.db_path = original_db_path
                engine.dispose()

        assert saved is not None
        self.assertEqual(2, len(saved.index))
        self.assertSetEqual({"Drug F", "Drug G"}, set(saved["ingredient"].tolist()))
        self.assertSetEqual(
            {"https://example.test/master.xlsx"}, set(saved["source_url"].tolist())
        )
        self.assertEqual(2, result.get("records"))


if __name__ == "__main__":
    unittest.main()
