from __future__ import annotations

from DILIGENT.app.scripts.update_fda_data import run as run_fda_update
from DILIGENT.app.scripts.update_livertox_data import run as run_livertox_update
from DILIGENT.app.utils.repository.database import database

REDOWNLOAD = True  # Force a fresh pull of source archives instead of reusing local files

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    run_livertox_update(redownload=REDOWNLOAD, initialize_db=False)
    run_fda_update(redownload=REDOWNLOAD, initialize_db=False)
