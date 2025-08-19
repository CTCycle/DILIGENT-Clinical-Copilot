from __future__ import annotations


import requests
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup 
from tqdm import tqdm 

from Pharmagent.app.api.schemas.clinical import Monography
from Pharmagent.app.constants import SOURCES_PATH

###############################################################################
class LiverToxClient:
   

    def __init__(self):        
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
        self.file_name = "livertox_NBK547852.tar.gz"
        self.chunk_size = 8192

    #--------------------------------------------------------------------------
    async def download_bulk_data(self, dest_path: Path) -> dict:
        """
        Asynchronously downloads the file with a progress bar.
        Prints "Downloading file N" at start.

        Args:
            dest_path (Path): Directory where the file will be saved.

        Returns:
            dict: {
                "file_path": Path to downloaded file,
                "size": File size in bytes (int),
                "last_modified": HTTP Last-Modified header (str)
            }

        Raises:
            httpx.HTTPStatusError: For HTTP errors.
            Exception: For other errors.
            
        """
        url = self.base_url + self.file_name
        print(f"Downloading file {self.file_name}...")

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
                with open(file_path, "wb") as f, tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=self.file_name,
                    ncols=80) as pbar:
                    async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        return {
            "file_path": str(file_path),
            "size": file_size,
            "last_modified": last_modified}