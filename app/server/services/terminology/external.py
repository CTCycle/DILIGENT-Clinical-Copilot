# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Bounded RxNav fallback used only when local terminology cannot resolve a name."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from common.utils.logger import logger

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
INGREDIENT_TYPES = ("IN", "PIN", "MIN")

###############################################################################
@dataclass(frozen=True)
class RxNavConcept:
    rxcui: str
    name: str
    tty: str

###############################################################################
class RxNavBudgetExceeded(RuntimeError):
    """Raised when a single resolution would exceed its request budget."""

###############################################################################
class RxNavRuntimeClient:
    """Synchronous, budgeted RxNav access with one short retry per request.

    Callers create one budget per resolution with ``session``; network errors
    degrade to empty results so clinical runs never fail on RxNav outages.
    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        timeout: float,
        base_url: str = RXNAV_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    # -------------------------------------------------------------------------
    def session(self, max_requests: int) -> RxNavSession:
        return RxNavSession(self, max_requests)


###############################################################################
class RxNavSession:

    # -------------------------------------------------------------------------
    def __init__(self, client: RxNavRuntimeClient, max_requests: int) -> None:
        self.client = client
        self.remaining = max(0, max_requests)
        self.calls = 0
        self.failed = False
        self._http = httpx.Client(
            timeout=client.timeout,
            transport=client.transport,
            headers={"User-Agent": "DILIGENT terminology resolver", "Accept": "application/json"},
        )

    # -------------------------------------------------------------------------
    def __enter__(self) -> RxNavSession:
        return self

    # -------------------------------------------------------------------------
    def __exit__(self, *_exc: object) -> None:
        self._http.close()

    # -------------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.remaining <= 0:
            raise RxNavBudgetExceeded("RxNav request budget exhausted")
        self.remaining -= 1
        self.calls += 1
        for attempt in range(2):
            try:
                response = self._http.get(f"{self.client.base_url}/{path}", params=params)
            except httpx.HTTPError as exc:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                logger.warning("RxNav request failed: %s", exc)
                self.failed = True
                return {}
            if response.status_code in RETRY_STATUS and attempt == 0:
                time.sleep(0.5)
                continue
            if response.status_code != 200:
                self.failed = True
                return {}
            try:
                payload = response.json()
            except ValueError:
                self.failed = True
                return {}
            return payload if isinstance(payload, dict) else {}
        return {}

    # -------------------------------------------------------------------------
    def find_rxcuis(self, name: str) -> list[str]:
        """findRxcuiByString with normalized matching across RxNorm sources."""
        payload = self._get("rxcui.json", {"name": name, "search": 2, "allsrc": 1})
        ids = (payload.get("idGroup") or {}).get("rxnormId") or []
        return [str(item) for item in ids if item]

    # -------------------------------------------------------------------------
    def related_ingredients(self, rxcui: str) -> list[RxNavConcept]:
        payload = self._get(f"rxcui/{rxcui}/related.json", {"tty": " ".join(INGREDIENT_TYPES)})
        concepts: list[RxNavConcept] = []
        for group in (payload.get("relatedGroup") or {}).get("conceptGroup") or []:
            for item in group.get("conceptProperties") or []:
                if item.get("rxcui") and item.get("tty") in INGREDIENT_TYPES:
                    concepts.append(RxNavConcept(str(item["rxcui"]), str(item.get("name") or ""), str(item["tty"])))
        return concepts

    # -------------------------------------------------------------------------
    def history_ingredients(self, rxcui: str) -> list[RxNavConcept]:
        """Ingredients of an obsolete concept via getRxcuiHistoryStatus."""
        payload = self._get(f"rxcui/{rxcui}/historystatus.json", {})
        history = payload.get("rxcuiStatusHistory") or {}
        derived = (history.get("derivedConcepts") or {}).get("ingredientConcept") or []
        return [
            RxNavConcept(str(item["ingredientRxcui"]), str(item.get("ingredientName") or ""), "IN")
            for item in derived
            if item.get("ingredientRxcui")
        ]
