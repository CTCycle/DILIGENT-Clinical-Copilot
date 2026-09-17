from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from common.paths import CONFIGURATIONS_FILE
from configurations.management import load_configuration_data
from configurations.startup import get_configuration_manager

_WRITE_LOCK = RLock()


def persist_configuration_blocks(
    config_path: str | Path,
    block_updates: dict[str, dict[str, object]],
) -> dict[str, Any]:
    path = Path(config_path)
    with _WRITE_LOCK:
        current = load_configuration_data(path)
        candidate = deepcopy(current)
        for block_name, updates in block_updates.items():
            block = candidate.get(block_name)
            if not isinstance(block, dict):
                block = {}
                candidate[block_name] = block
            block.update(updates)

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(candidate, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

        if path.resolve() == Path(CONFIGURATIONS_FILE).resolve():
            get_configuration_manager().reload()
        return candidate
