# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Run the labeled drug-terminology benchmark against a populated database.

Usage (from app/server):
  .venv\\Scripts\\python.exe ..\\tests\\benchmarks\\drug_terminology\\run_benchmark.py \\
      --database <path-to-sqlite.db> --output <report.json> [--external] [--split all]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
SERVER_DIR = BENCHMARK_DIR.parents[2] / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark import Prediction, run_benchmark  # noqa: E402


# -----------------------------------------------------------------------------
def configure_database(path: Path) -> Path:
    from common import paths as common_paths
    from configurations import database_config
    from configurations.startup import reset_app_settings_cache
    from repositories.database import sqlite as sqlite_module
    from repositories.database.session import get_default_repository

    config_path = Path(tempfile.mkdtemp(prefix="terminology-benchmark-")) / "database_config.json"
    database_config.DATABASE_CONFIG_FILE_PATH = config_path
    database_config.save_database_configuration(
        database_config.DatabaseConfiguration(backend="sqlite", sqlite_path=str(path))
    )
    common_paths.DATABASE_FILE_PATH = path
    sqlite_module.DATABASE_FILE_PATH = path
    get_default_repository.cache_clear()
    reset_app_settings_cache()
    return config_path


# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--split", default="all", choices=["all", "calibration", "heldout"])
    args = parser.parse_args()
    config_path = configure_database(args.database.resolve())

    from repositories.context import RepositoryContext
    from repositories.terminology_lookup_repository import TerminologyLookupRepository
    from services.catalogs.runtime import initialize_reference_catalog_provider
    from services.terminology.factory import build_terminology_resolver
    from services.terminology.index import get_terminology_index_cache

    try:
        initialize_reference_catalog_provider()
        lookup = TerminologyLookupRepository(RepositoryContext.create())
        resolver = build_terminology_resolver(lookup, allow_external=args.external)
        tracemalloc.start()
        started = time.perf_counter()
        get_terminology_index_cache().get(lookup.revision(), lookup.fuzzy_vocabulary)
        index_seconds = time.perf_counter() - started
        index_bytes = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        def resolve(text: str) -> Prediction:
            begin = time.perf_counter()
            result = resolver.resolve(text, allow_external=args.external)
            elapsed = (time.perf_counter() - begin) * 1000
            resolver.clear_cache()
            ingredients = [item.name for item in result.ingredients] if result.drug_id else []
            candidates = [[candidate.name] for candidate in result.candidates]
            return Prediction(
                accepted=result.accepted,
                status=result.status,
                ingredients=ingredients,
                candidates=candidates,
                latency_ms=elapsed,
                external_calls=result.external_calls,
                method=result.method,
            )

        report = run_benchmark(resolve, split=args.split)
        report["engine"] = "terminology" + ("+external" if args.external else "")
        report["index_build_seconds"] = round(index_seconds, 3)
        report["index_memory_bytes"] = index_bytes
        report["index"] = get_terminology_index_cache().stats()
        report["database_bytes"] = args.database.stat().st_size
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {key: report[key] for key in ("overall", "calibration", "heldout") if key in report}
        print(json.dumps(summary, indent=2))
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
