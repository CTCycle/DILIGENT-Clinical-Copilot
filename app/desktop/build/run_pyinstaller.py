"""Run PyInstaller with a hosted-Windows pywin32-ctypes compatibility shim."""

from __future__ import annotations

import builtins
import sys


_original_import = builtins.__import__


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

from PyInstaller.__main__ import run  # noqa: E402


if __name__ == "__main__":
    run()
