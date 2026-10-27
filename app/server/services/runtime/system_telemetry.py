# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import ctypes
import shutil
import subprocess
import time
from ctypes import wintypes

from domain.system_status import MemoryUsage

NVIDIA_SMI_TIMEOUT_S = 1.5
CPU_FIRST_SAMPLE_WINDOW_S = 0.2

###############################################################################
class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

###############################################################################
def _filetime_to_int(value: wintypes.FILETIME) -> int:
    return (value.dwHighDateTime << 32) | value.dwLowDateTime

###############################################################################
class SystemTelemetry:
    """System-wide (not per-process) CPU, RAM and GPU memory readings.

    Every reader returns None when the platform or hardware cannot report it.
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self._last_cpu_sample: tuple[int, int] | None = None

    # -------------------------------------------------------------------------
    def read_ram(self) -> MemoryUsage | None:
        kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        memory_status_fn = getattr(kernel32, "GlobalMemoryStatusEx", None)
        if memory_status_fn is None:
            return None
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not memory_status_fn(ctypes.byref(status)) or status.ullTotalPhys <= 0:
            return None
        return MemoryUsage(
            used_bytes=int(status.ullTotalPhys - status.ullAvailPhys),
            total_bytes=int(status.ullTotalPhys),
        )

    # -------------------------------------------------------------------------
    def read_cpu_percent(self) -> float | None:
        previous = self._last_cpu_sample
        current = self._sample_cpu_times()
        if current is None:
            return None
        if previous is None:
            time.sleep(CPU_FIRST_SAMPLE_WINDOW_S)
            previous = current
            current = self._sample_cpu_times()
            if current is None:
                return None
        self._last_cpu_sample = current
        idle_delta = current[0] - previous[0]
        total_delta = current[1] - previous[1]
        if total_delta <= 0:
            return None
        busy = max(0.0, 1.0 - idle_delta / total_delta)
        return round(min(busy, 1.0) * 100.0, 1)

    # -------------------------------------------------------------------------
    def read_vram(self) -> MemoryUsage | None:
        if shutil.which("nvidia-smi") is None:
            return None
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_S,
            )
        except OSError, subprocess.SubprocessError:
            return None
        if result.returncode != 0:
            return None
        used_mib = 0
        total_mib = 0
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                used_mib += int(parts[0])
                total_mib += int(parts[1])
            except ValueError:
                continue
        if total_mib <= 0:
            return None
        return MemoryUsage(
            used_bytes=used_mib * 1_048_576, total_bytes=total_mib * 1_048_576
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _sample_cpu_times() -> tuple[int, int] | None:
        kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        get_system_times = getattr(kernel32, "GetSystemTimes", None)
        if get_system_times is None:
            return None
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not get_system_times(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        # Kernel time already includes idle time.
        return _filetime_to_int(idle), _filetime_to_int(kernel) + _filetime_to_int(user)
