from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

from common.constants import ARCHIVES_PATH, LIVERTOX_BASE_URL
from configurations.startup import get_server_settings
from repositories.serialization.data import DataSerializer
from services.updater import (
    livertox_common,
    livertox_download,
    livertox_index,
    livertox_parse,
)
from services.updater.sanitizer import LiverToxExcerptSanitizer


class LiverToxUpdater:
    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        archive_name: str | None = None,
        monograph_max_workers: int | None = None,
        serializer: DataSerializer | None = None,
    ) -> None:
        self.supported_extensions = livertox_common.SUPPORTED_MONOGRAPH_EXTENSIONS
        self.http_headers = dict(livertox_common.DEFAULT_HTTP_HEADERS)
        self.delay = 0.5
        self.chunk_size = livertox_common.DOWNLOAD_CHUNK_SIZE
        self.sources_path = os.path.abspath(sources_path)
        self.redownload = redownload
        self.serializer = serializer or DataSerializer()
        self.excerpt_sanitizer = LiverToxExcerptSanitizer()
        self.header_row = 1
        self.base_url = LIVERTOX_BASE_URL
        self.file_name = (
            archive_name or get_server_settings().runtime.livertox_archive
        ).strip()
        self.monograph_max_workers = max(
            1,
            int(
                monograph_max_workers
                if monograph_max_workers is not None
                else get_server_settings().runtime.livertox_monograph_max_workers
            ),
        )
        self.tar_file_path = os.path.join(ARCHIVES_PATH, self.file_name)
        self.master_list_path = os.path.join(ARCHIVES_PATH, "LiverTox_Master_List.xlsx")
        self.master_list_metadata_path = os.path.join(
            ARCHIVES_PATH, "livertox_master_list.metadata.json"
        )
        self.archive_metadata_path = os.path.join(
            ARCHIVES_PATH, "livertox_archive.metadata.json"
        )

    def update_from_livertox(
        self,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if livertox_common.should_cancel(should_stop):
            raise RuntimeError("LiverTox update cancelled by user request")
        livertox_common.emit_progress(
            progress_callback,
            progress=5.0,
            message="Refreshing LiverTox master list",
        )
        master_metadata, master_frame = livertox_download.refresh_master_list(self)
        if livertox_common.should_cancel(should_stop):
            raise RuntimeError("LiverTox update cancelled by user request")
        livertox_common.emit_progress(
            progress_callback,
            progress=20.0,
            message="Downloading LiverTox archive metadata",
        )
        archive_metadata = asyncio.run(
            livertox_download.download_bulk_data(self, self.sources_path)
        )
        archive_path = archive_metadata.get("file_path") or os.path.join(
            self.sources_path, get_server_settings().runtime.livertox_archive
        )
        local_info = livertox_download.collect_local_archive_info(self, archive_path)
        livertox_common.emit_progress(
            progress_callback,
            progress=35.0,
            message="Extracting LiverTox monographs",
        )
        extracted = livertox_parse.collect_monographs(
            self,
            archive_path,
            should_stop=should_stop,
            progress_callback=progress_callback,
        )
        if livertox_common.should_cancel(should_stop):
            raise RuntimeError("LiverTox update cancelled by user request")
        livertox_common.emit_progress(
            progress_callback,
            progress=70.0,
            message="Sanitizing extracted LiverTox entries",
        )
        monograph_df = livertox_parse.sanitize_records(self, extracted)
        livertox_common.emit_progress(
            progress_callback,
            progress=80.0,
            message="Combining master list and monograph excerpts",
        )
        unified = livertox_index.build_unified_dataset(
            self,
            monograph_df,
            master_frame,
            master_metadata,
        )
        livertox_common.emit_progress(
            progress_callback,
            progress=88.0,
            message="Finalizing LiverTox dataset",
        )
        final_dataset = livertox_index.finalize_dataset(self, unified)
        if livertox_common.should_cancel(should_stop):
            raise RuntimeError("LiverTox update cancelled by user request")
        livertox_common.emit_progress(
            progress_callback,
            progress=95.0,
            message="Persisting LiverTox records",
        )
        self.serializer.save_livertox_records(final_dataset)
        payload = {**master_metadata, **archive_metadata, **local_info}
        payload["processed_entries"] = len(final_dataset.index)
        payload["records"] = len(final_dataset.index)
        livertox_common.emit_progress(
            progress_callback,
            progress=99.0,
            message="LiverTox update completed",
        )
        return payload
