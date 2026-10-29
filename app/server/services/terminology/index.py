# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Character-trigram candidate index for bounded approximate alias matching."""

from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np

from common.utils.logger import logger
from services.terminology.normalization import orthographic_skeleton


# -----------------------------------------------------------------------------
def trigrams(value: str) -> set[str]:
    padded = f"  {value} "
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


###############################################################################
class TerminologyIndex:
    """Inverted trigram postings over distinct fuzzy-eligible alias keys.

    A query only scores aliases sharing trigrams with it and within a length
    window, so approximate matching never scans the whole vocabulary.
    """

    # -------------------------------------------------------------------------
    def __init__(self, vocabulary: Iterable[str]) -> None:
        started = time.perf_counter()
        self.strings: list[str] = sorted(set(vocabulary))
        postings: dict[str, list[int]] = defaultdict(list)
        for position, value in enumerate(self.strings):
            for gram in trigrams(value):
                postings[gram].append(position)
        self.postings = {gram: np.asarray(ids, dtype=np.int32) for gram, ids in postings.items()}
        self.lengths = np.asarray([len(value) for value in self.strings], dtype=np.int32)
        skeletons: dict[str, list[str]] = defaultdict(list)
        for value in self.strings:
            skeletons[orthographic_skeleton(value)].append(value)
        self.skeletons = dict(skeletons)
        self.build_seconds = time.perf_counter() - started

    # -------------------------------------------------------------------------
    def shortlist(self, query: str, *, limit: int, max_length_delta: int) -> list[str]:
        arrays = [self.postings[gram] for gram in trigrams(query) if gram in self.postings]
        if not arrays or not self.strings:
            return []
        counts = np.bincount(np.concatenate(arrays), minlength=len(self.strings))
        counts[np.abs(self.lengths - len(query)) > max_length_delta] = 0
        candidates = int(np.count_nonzero(counts))
        if candidates == 0:
            return []
        size = min(limit, candidates)
        top = np.argpartition(-counts, size - 1)[:size]
        ordered = sorted(top.tolist(), key=lambda index: (-int(counts[index]), self.strings[index]))
        return [self.strings[index] for index in ordered if counts[index] > 0]

    # -------------------------------------------------------------------------
    def skeleton_matches(self, query: str) -> list[str]:
        return list(self.skeletons.get(orthographic_skeleton(query), ()))

    # -------------------------------------------------------------------------
    def stats(self) -> dict[str, float | int]:
        posting_bytes = sum(array.nbytes for array in self.postings.values())
        string_bytes = sum(sys.getsizeof(value) for value in self.strings)
        return {
            "aliases": len(self.strings),
            "trigrams": len(self.postings),
            "approx_bytes": posting_bytes + string_bytes + self.lengths.nbytes,
            "build_seconds": round(self.build_seconds, 3),
        }


###############################################################################
class TerminologyIndexCache:
    """Process-wide index, rebuilt lazily when the terminology revision moves."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: TerminologyIndex | None = None
        self._revision: int | None = None

    # -------------------------------------------------------------------------
    def get(self, revision: int | None, loader: Callable[[], Iterable[str]]) -> TerminologyIndex:
        index = self._index
        if index is not None and self._revision == revision:
            return index
        with self._lock:
            if self._index is None or self._revision != revision:
                self._index = TerminologyIndex(loader())
                self._revision = revision
                logger.info("Terminology fuzzy index built: %s", self._index.stats())
            return self._index

    # -------------------------------------------------------------------------
    def invalidate(self) -> None:
        with self._lock:
            self._index = None
            self._revision = None

    # -------------------------------------------------------------------------
    def stats(self) -> dict[str, float | int | None]:
        index = self._index
        if index is None:
            return {"loaded": 0, "revision": self._revision}
        return {"loaded": 1, "revision": self._revision, **index.stats()}


_INDEX_CACHE = TerminologyIndexCache()


# -----------------------------------------------------------------------------
def get_terminology_index_cache() -> TerminologyIndexCache:
    return _INDEX_CACHE
