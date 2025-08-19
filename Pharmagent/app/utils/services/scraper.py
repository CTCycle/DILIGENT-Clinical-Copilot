from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup  

from Pharmagent.app.api.schemas.clinical import Monography
from Pharmagent.app.constants import SOURCES_PATH

###############################################################################
class LiverToxClient:

    """
    Minimal async client for LiverTox:
      1) Download bulk data (tar.gz archive) & Master List (xlsx) via HTTPS.
      2) Fetch & parse a drug monograph by name using the Bookshelf search.

    """ 

    def __init__(self, *, timeout: float = 20.0, user_agent: str | None = None) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent or "LiverToxClient/1.0 (+https://example.org)"}
        self.BASE_URL = "https://www.ncbi.nlm.nih.gov/books/NBK547852/"      
        self.FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/livertox_NBK547852.tar.gz"
        self.MASTER_LIST_URL = "https://www.ncbi.nlm.nih.gov/books/NBK571102/bin/masterlist08-25.xlsx" 

    #--------------------------------------------------------------------------
    async def download_bulk_archive(self, dest_path: str | Path) -> Path:
        """
        Download the current LiverTox book archive (.tar.gz) via HTTPS.

        Returns: local Path to the downloaded file.
        """
        url = await self._discover_bulk_archive_url()
        return await self._stream_download(url, dest_path)

    #--------------------------------------------------------------------------
    async def download_master_list(self, dest_path: str | Path) -> Path:
        """
        Download the Master List Excel (updated periodically).

        Returns: local Path to the downloaded file.
        """
        url = await self._discover_master_list_url()
        return await self._stream_download(url, dest_path)

    #--------------------------------------------------------------------------
    async def get_drug_monograph(self, drug_name: str) -> Dict[str, object]:
        """
        Resolve a drug name to its monograph page and extract key sections.

        Returns:
            {
              "title": str,
              "url": str,
              "last_update": Optional[str],
              "sections": Dict[str, str],  # h2/h3 -> text
            }
        """
        monograph_url = await self._resolve_drug_url(drug_name)
        html = await self._fetch_text(monograph_url)

        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.find(id="book-title") or soup.find("h1")
        title = (title_el.get_text(strip=True) if title_el else "").strip()

        # Try to pick the “Last Update: …” line if present
        last_update = None
        m = re.search(r"Last Update:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", soup.get_text(" ", strip=True))
        if m:
            last_update = m.group(1)

        # Minimal section extraction: collect h2/h3 blocks
        sections: Dict[str, str] = {}
        for header in soup.select("h2, h3"):
            htxt = header.get_text(strip=True)
            if not htxt:
                continue
            parts = []
            for sib in header.find_next_siblings():
                if sib.name in {"h2", "h3"}:
                    break
                parts.append(sib.get_text(separator="\n", strip=True))
            text = "\n".join(p for p in parts if p).strip()
            if text:
                sections[htxt] = text

        return {
            "title": title or drug_name,
            "url": monograph_url,
            "last_update": last_update,
            "sections": sections}

    #--------------------------------------------------------------------------
    async def _discover_bulk_archive_url(self) -> str:
        """
        Find the .tar.gz bulk archive URL from the LiverTox landing page.
        Strategy: prefer any <a> whose href points to ftp.ncbi.nlm.nih.gov and ends with .tar.gz
        Fallback: the 'Bulk download...' anchor if present.
        """
        html = await self._fetch_text(self.BASE_URL)
        soup = BeautifulSoup(html, "html.parser")

        # Primary: any link to ftp.ncbi.nlm.nih.gov ending with .tar.gz
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "ftp.ncbi.nlm.nih.gov" in href and href.lower().endswith(".tar.gz"):
                return href

        # Fallback: text-based anchor (if label changes slightly, still match)
        a = soup.find("a", string=lambda s: s and "bulk download" in s.lower() and "livertox" in s.lower())
        if a and a.get("href"):
            href = a["href"]
            return href if href.startswith("http") else httpx.URL(self.BASE_URL).join(href).human_repr()

        raise RuntimeError("Could not discover the bulk archive URL from LiverTox page.")

    #--------------------------------------------------------------------------
    async def _discover_master_list_url(self) -> str:
        """
        Parse the Master List page and find the 'here' link to the .xlsx under /bin/.
        Example observed: /books/NBK571102/bin/masterlist08-25.xlsx
        """
        html = await self._fetch_text(self.MASTER_LIST_URL)
        soup = BeautifulSoup(html, "html.parser")

        # Prefer the explicit 'here' anchor
        a = soup.find("a", string=lambda s: s and s.strip().lower() == "here")
        href = a["href"] if a and a.has_attr("href") else None

        # Fallback: any link to an .xlsx under /bin/
        if not href:
            cand = soup.find("a", href=re.compile(r"/books/NBK571102/bin/[^\"']+\.xlsx$"))
            href = cand["href"] if cand else None

        if not href:
            raise RuntimeError("Master List XLSX link not found on page.")

        if href.startswith("http"):
            return href
        return httpx.URL(self.MASTER_LIST_URL).join(href).human_repr()

    #--------------------------------------------------------------------------
    async def _resolve_drug_url(self, drug_name: str) -> str:
        """
        Use the Bookshelf in-book search (?term=) and pick the best matching NBK page.
        Strategy: prefer an anchor whose text contains the drug name, else first NBK link.
        """
        search_url = httpx.URL(self.BASE_URL).copy_merge_params({"term": drug_name})
        html = await self._fetch_text(search_url.human_repr())
        soup = BeautifulSoup(html, "html.parser")

        # Candidate NBK links
        cand_links = soup.select("a[href^='/books/NBK']")
        if not cand_links:
            raise LookupError(f"No NBK monograph links found for term={drug_name!r}.")

        # Prefer text match
        dn = drug_name.strip().lower()
        for a in cand_links:
            text = a.get_text(" ", strip=True).lower()
            if dn in text:
                return httpx.URL(self.BASE_URL).join(a["href"]).human_repr()

        # Fallback: first NBK link
        return httpx.URL(self.BASE_URL).join(cand_links[0]["href"]).human_repr()

    #--------------------------------------------------------------------------
    async def _fetch_text(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers, http2=False) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.text

    #--------------------------------------------------------------------------
    async def _stream_download(self, url: str, dest_path: str | Path) -> Path:
        filename = os.path.join(dest_path, "livertox_book.tar.gz")
        dest = Path(dest_path)
        url = url + '/livertox_NBK547852.tar.gz'     
        # TO DO
        # ADD logic to download file    
        async with httpx.AsyncClient(timeout=None, headers=self.headers, follow_redirects=True) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
        return dest