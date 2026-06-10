from __future__ import annotations

import json
from pathlib import Path

from configurations.management import load_configuration_data


###############################################################################
def test_load_configuration_data_accepts_path_objects(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"
    payload = {"runtime": {"mode": "test"}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_configuration_data(config_path) == payload
