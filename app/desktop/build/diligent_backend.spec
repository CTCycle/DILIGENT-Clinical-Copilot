# PyInstaller specification for the packaged backend.
from pathlib import Path

from PyInstaller.building.build_main import Analysis, EXE, COLLECT, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

server_root = Path(SPECPATH).parents[1] / "server"
hiddenimports = [
    "app",
    "desktop_entry",
    "numpy",
    "numpy.linalg",
    "pandas",
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "tokenizers",
    "lancedb",
    "lancedb.db",
    "lancedb.table",
    "pyarrow",
    "psycopg",
    "psycopg_binary",
    "psycopg_c",
    "cryptography",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
]

binaries = []
for package in [
    "numpy",
    "pandas",
    "onnxruntime",
    "tokenizers",
    "lancedb",
    "pyarrow",
    "cryptography",
    "psycopg",
]:
    binaries.extend(collect_dynamic_libs(package))

# These packages use small runtime data files; keep the collection explicit
# and limited to data formats used by the runtime rather than package-wide
# collection (which also captures tests, headers, and build sources).
datas = []
for package in ["onnxruntime", "tokenizers", "lancedb", "pyarrow"]:
    datas.extend(
        collect_data_files(
            package,
            includes=["*.json", "*.yaml", "*.yml", "*.txt"],
            excludes=["**/tests/**", "**/test/**", "**/*.pyi"],
        )
    )

migrations_root = server_root / "migrations"
for migration_file in migrations_root.rglob("*"):
    relative_path = migration_file.relative_to(migrations_root)
    if (
        migration_file.is_file()
        and "__pycache__" not in relative_path.parts
        and migration_file.suffix not in {".pyc", ".pyo"}
    ):
        datas.append(
            (
                str(migration_file),
                str(Path("migrations") / relative_path.parent),
            )
        )

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

_forbidden_suffixes = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".lib",
    ".map",
    ".pxd",
    ".pxi",
    ".pyi",
    ".pyx",
}
_forbidden_parts = {
    ".angular",
    ".pytest_cache",
    "__pycache__",
    "stubs",
    "test",
    "tests",
    "testing",
}


def _keep_collected_entry(entry):
    source = str(entry[0]).replace("\\", "/").casefold()
    destination = str(entry[1]).replace("\\", "/").casefold()
    source_path = Path(source)
    if source_path.suffix in _forbidden_suffixes:
        return False
    if any(part in _forbidden_parts for part in Path(source).parts):
        return False
    if any(part in _forbidden_parts for part in Path(destination).parts):
        return False
    return True


# PyInstaller's third-party hooks can add package headers and test/build data.
# Keep the explicit native/runtime collection above while removing artifacts
# that cannot be imported or executed by the frozen Windows backend.
analysis.binaries = [entry for entry in analysis.binaries if _keep_collected_entry(entry)]
analysis.datas = [entry for entry in analysis.datas if _keep_collected_entry(entry)]
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name="DILIGENTBackend", debug=False, console=False, upx=False)
coll = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="DILIGENTBackend")
