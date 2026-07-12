from __future__ import annotations

import argparse
import time

from common.utils.logger import logger
from repositories.database.initializer import initialize_database

###############################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the DILIGENT database.")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--seed-catalogs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-reseed-catalogs", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    logger.info("Starting database initialization")
    initialize_database(
        drop_existing=args.drop_existing,
        seed_catalogs=args.seed_catalogs,
        force_reseed_catalogs=args.force_reseed_catalogs,
    )
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
