# PyInstaller specification for the packaged backend.
from pathlib import Path

from PyInstaller.building.build_main import Analysis, EXE, COLLECT, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

server_root = Path(SPECPATH).parents[1] / "server"
native_packages = [
    "numpy",
    "pandas",
    "onnxruntime",
    "tokenizers",
    "lancedb",
    "pyarrow",
    "cryptography",
    "psycopg",
]

hiddenimports = [
    "app",
    "desktop_entry",
    "psycopg_binary",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]
for package in native_packages:
    hiddenimports.extend(collect_submodules(package))

binaries = []
datas = []
for package in native_packages:
    binaries.extend(collect_dynamic_libs(package))
    datas.extend(collect_data_files(package))

excludedimports = [
    "pytest", "pytest_cov", "playwright", "ruff", "pyright", "IPython",
    "notebook", "pip", "setuptools", "wheel", "torch", "transformers",
    "sentence_transformers", "tkinter",
]

analysis = Analysis(
    [str(server_root / "desktop_entry.py")],
    pathex=[str(server_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludedimports,
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="DILIGENTBackend", debug=False, console=False, upx=False)
coll = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="DILIGENTBackend")
