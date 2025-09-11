from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Any, Dict

import httpx
import pandas as pd
from tqdm import tqdm

from Pharmagent.app.constants import SOURCES_PATH


###############################################################################
class LiverToxClient:
    def __init__(self) -> None:
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
        self.file_name = "livertox_NBK547852.tar.gz"
        self.tar_file_path = os.path.join(SOURCES_PATH, self.file_name)
        self.chunk_size = 8192

    # -------------------------------------------------------------------------
    async def download_bulk_data(self, dest_path: Path) -> dict[str, Any]:
        url = self.base_url + self.file_name
        print(f"Downloading file {self.file_name}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # HEAD request for size and last-modified
            head_response = await client.head(url)
            head_response.raise_for_status()
            file_size = int(head_response.headers.get("Content-Length", 0))
            last_modified = head_response.headers.get("Last-Modified", None)

            dest_path = Path(dest_path)
            dest_path.mkdir(parents=True, exist_ok=True)
            file_path = dest_path / self.file_name

            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with (
                    open(file_path, "wb") as f,
                    tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=self.file_name,
                        ncols=80,
                    ) as pbar,
                ):
                    async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        return {
            "file_path": str(file_path),
            "size": file_size,
            "last_modified": last_modified,
        }

    # -------------------------------------------------------------------------
    def convert_file_to_dataframe(self) -> pd.DataFrame:
        records = []
        with tarfile.open(self.tar_file_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name.lower()
                if name.endswith(".csv") or name.endswith(".tsv"):
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    df = pd.read_csv(
                        fileobj, sep="\t" if name.endswith(".tsv") else ","
                    )
                    records.append(df)

        if not records:
            raise ValueError("No supported tabular files found in archive.")

        return pd.concat(records, ignore_index=True)
