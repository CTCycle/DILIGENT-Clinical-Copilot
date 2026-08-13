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
if hasattr(os, "add_dll_directory"):
    _venv_bin = Path(sys.executable).resolve().parent
    if _venv_bin.is_dir():
        _dll_directory_handles.append(os.add_dll_directory(str(_venv_bin)))

# Load ctypes while the matching embedded Python DLL directory is explicitly
# active. This prevents a hosted runner's other Python installation from
# winning the dependency lookup when PyInstaller later imports ctypes.util.
import ctypes  # noqa: E402, F401

# Load the supported native compatibility backend before PyInstaller's
# dependency scanner reaches ctypes.util on affected hosted runners.
import cffi  # noqa: E402, F401


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
