from __future__ import annotations

import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from DILIGENT.server.services.updater import livertox as livertox_module
from DILIGENT.server.services.updater.livertox import LiverToxUpdater


def build_archive(path: Path, members: list[tuple[str, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members:
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def make_html(drug_name: str, nbk_id: str) -> str:
    return (
        f"<html><head><title>{drug_name} - LiverTox - NCBI Bookshelf</title></head>"
        f"<body><p>{nbk_id}</p><p>{drug_name} monograph excerpt.</p></body></html>"
    )


def test_collect_monographs_deduplicates_duplicate_basenames(tmp_path: Path) -> None:
    archive_path = tmp_path / "livertox.tar.gz"
    build_archive(
        archive_path,
        [
            ("folder_a/NBK10001.html", make_html("Drug A", "NBK10001")),
            ("folder_b/NBK10001.html", make_html("Drug A duplicate", "NBK10001")),
            ("folder_c/NBK10002.html", make_html("Drug B", "NBK10002")),
        ],
    )

    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=False,
        archive_name=archive_path.name,
        monograph_max_workers=1,
    )
    records = updater.collect_monographs(str(archive_path))

    nbk_ids = {item["nbk_id"] for item in records}
    assert nbk_ids == {"NBK10001", "NBK10002"}
    assert len(records) == 2


class ThreadedProcessPoolExecutor:
    def __init__(self, max_workers: int, mp_context=None) -> None:
        _ = mp_context
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> "ThreadedProcessPoolExecutor":
        self._executor.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        return self._executor.__exit__(exc_type, exc, tb)

    def submit(self, *args, **kwargs):
        return self._executor.submit(*args, **kwargs)


def test_collect_monographs_streams_parallel_batches(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "livertox.tar.gz"
    build_archive(
        archive_path,
        [
            ("NBK20001.html", make_html("Drug C", "NBK20001")),
            ("NBK20002.html", make_html("Drug D", "NBK20002")),
            ("NBK20003.html", make_html("Drug E", "NBK20003")),
        ],
    )
    monkeypatch.setattr(
        livertox_module,
        "ProcessPoolExecutor",
        ThreadedProcessPoolExecutor,
    )

    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=False,
        archive_name=archive_path.name,
        monograph_max_workers=3,
    )
    records = updater.collect_monographs(str(archive_path))

    assert {item["nbk_id"] for item in records} == {"NBK20001", "NBK20002", "NBK20003"}


def test_collect_monographs_honors_cancellation(tmp_path: Path) -> None:
    archive_path = tmp_path / "livertox.tar.gz"
    build_archive(
        archive_path,
        [("NBK30001.html", make_html("Drug F", "NBK30001"))],
    )
    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=False,
        archive_name=archive_path.name,
        monograph_max_workers=1,
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        updater.collect_monographs(str(archive_path), should_stop=lambda: True)
