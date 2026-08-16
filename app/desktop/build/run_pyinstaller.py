"""Run PyInstaller with a hosted-Windows pywin32-ctypes compatibility shim."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys


_original_import = builtins.__import__


# The embedded Python venv keeps its matching _ctypes/libffi pair beside the
# interpreter. Hosted Windows runners can otherwise resolve a different
# libffi-8.dll before PyInstaller imports ctypes.
_dll_directory_handles = []
_venv_bin = Path(sys.executable).resolve().parent
_repo_root = Path(__file__).resolve().parents[3]
_embedded_runtime = _repo_root / "runtimes" / "python"


def _is_hosted_python_path(entry):
    normalized = str(Path(entry).resolve()).replace("\\", "/").lower()
    return "/hostedtoolcache/windows/python/" in normalized


sys.path[:] = [entry for entry in sys.path if not entry or not _is_hosted_python_path(entry)]
_native_dll_dirs = []
for _candidate in (_venv_bin, _embedded_runtime):
    if not _candidate.is_dir() or _candidate in _native_dll_dirs:
        continue
    if any((_candidate / _name).is_file() for _name in ("libffi-8.dll", "python314.dll", "python3.dll", "_ctypes.pyd")):
        _native_dll_dirs.append(_candidate)

if _native_dll_dirs:
    # Put every matching embedded-Python directory first. The _ctypes module
    # can be imported from either the venv or the pinned runtime root.
    _native_path = os.pathsep.join(str(_directory) for _directory in _native_dll_dirs)
    os.environ["PATH"] = f"{_native_path}{os.pathsep}{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        _dll_directory_handles.extend(
            os.add_dll_directory(str(_directory)) for _directory in _native_dll_dirs
        )

# Load the supported native compatibility backend before ctypes. Its native
# extension binds the matching release-vendor libffi first, avoiding a hosted
# runner's already-discoverable but incompatible libffi export set.
import cffi  # noqa: E402, F401
import _cffi_backend  # noqa: E402, F401

# Resolve the matching libffi pair explicitly before Python loads _ctypes. The
# hosted runner can still have a same-named DLL available through Windows'
# default search locations even after PATH and add_dll_directory isolation.
_ffi = cffi.FFI()
_ffi.cdef(
    """
    void *GetModuleHandleW(const wchar_t *lpModuleName);
    unsigned long GetModuleFileNameW(void *hModule, wchar_t *lpFilename, unsigned long nSize);
    void *LoadLibraryW(const wchar_t *lpFileName);
    """
)
_kernel32 = _ffi.dlopen("kernel32.dll")


def _loaded_module_path(name):
    handle = _kernel32.GetModuleHandleW(_ffi.new("wchar_t[]", name))
    if handle == _ffi.NULL:
        return None
    buffer = _ffi.new("wchar_t[]", 512)
    length = _kernel32.GetModuleFileNameW(handle, buffer, 512)
    return str(_ffi.string(buffer)) if length else None


_libffi_path = next(
    (_directory / "libffi-8.dll" for _directory in _native_dll_dirs if (_directory / "libffi-8.dll").is_file()),
    None,
)
if _libffi_path is None:
    raise OSError("unable to locate the embedded release libffi-8.dll")
print(f"[INFO] PyInstaller native DLL directories: {_native_dll_dirs}", flush=True)
print(f"[INFO] PyInstaller embedded libffi candidate: {_libffi_path}", flush=True)
print(f"[INFO] PyInstaller preloaded libffi: {_loaded_module_path('libffi-8.dll')}", flush=True)
_libffi_handle = _kernel32.LoadLibraryW(_ffi.new("wchar_t[]", str(_libffi_path)))
if _libffi_handle == _ffi.NULL:
    raise OSError(f"unable to load release libffi: {_libffi_path}")
print(f"[INFO] PyInstaller loaded libffi: {_loaded_module_path('libffi-8.dll')}", flush=True)

# Load ctypes while the matching embedded Python DLL directory is explicitly
# active. This prevents a hosted runner's other Python installation from
# winning the dependency lookup when PyInstaller later imports ctypes.util.
import ctypes  # noqa: E402, F401


def _import_with_cffi_backend(name, globals=None, locals=None, fromlist=(), level=0):
    """Let win32ctypes select its cffi backend during PyInstaller startup.

    PyInstaller temporarily blocks cffi while importing win32ctypes. On some
    hosted Windows Python 3.14 images that makes the ctypes backend fail its
    import preflight even though pywin32-ctypes is installed. The cffi backend
    is supported by the dependency and avoids that hosted-runner failure.
    """

    if name == "win32ctypes.pywin32" and sys.modules.get("cffi", object()) is None:
        blocked_cffi = sys.modules.pop("cffi")
        try:
            return _original_import(name, globals, locals, fromlist, level)
        finally:
            sys.modules["cffi"] = blocked_cffi
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _import_with_cffi_backend

from PyInstaller import __main__ as _pyinstaller_main  # noqa: E402


def _check_release_builder_context():
    """Keep PyInstaller's safe working-directory guard without ctypes probing.

    Some hosted Windows images fail loading the builder interpreter's
    ``_ctypes`` extension while PyInstaller checks administrator elevation.
    That check is advisory; the working-directory guard remains relevant to
    release builds.
    """

    if not _pyinstaller_main.compat.is_win:
        return

    cwd = Path.cwd().resolve()
    try:
        windows_dir = _pyinstaller_main.compat.win32api.GetWindowsDirectory()
    except Exception:
        windows_dir = None
    windows_dir = None if windows_dir is None else Path(windows_dir).resolve()
    inside_windows_dir = windows_dir is not None and (cwd == windows_dir or windows_dir in cwd.parents)
    if inside_windows_dir:
        home_dir = Path.home().resolve()
        if cwd == home_dir or home_dir in cwd.parents:
            inside_windows_dir = False
    if inside_windows_dir:
        raise SystemExit(
            f"ERROR: Do not run pyinstaller from {cwd}. cd to where your code is and run pyinstaller from there. "
            "Hint: open a terminal at the repository before running the release builder."
        )


_pyinstaller_main.check_unsafe_privileges = _check_release_builder_context


if __name__ == "__main__":
    _pyinstaller_main.run()
