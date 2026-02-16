"""Memory monitor — periodic system memory checks with warnings.

Uses direct Mach kernel syscalls on macOS (no subprocess spawning)
so it remains reliable even under extreme memory pressure.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import logging
import os
import platform

import click

from lattice.background_loop import BackgroundLoop
from lattice.constants import SYSTEM_SENDER
from lattice.session.models import StatusEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Default interval between memory checks (seconds).
_DEFAULT_INTERVAL = 2

#: Warn when available system memory drops below this (MB).
_WARN_AVAILABLE_MB = 2048

#: Critical when available system memory drops below this (MB).
_CRITICAL_AVAILABLE_MB = 1024


# ------------------------------------------------------------------ #
# macOS: direct Mach syscalls via ctypes
# ------------------------------------------------------------------ #

class _VMStatistics64(ctypes.Structure):
    """Mach vm_statistics64_data_t struct.

    Fields use ``natural_t`` (c_uint, 32-bit) for page counts and
    ``uint64_t`` (c_uint64) for cumulative counters — matching the
    macOS SDK ``<mach/vm_statistics.h>`` layout exactly.
    """

    _fields_ = [
        ("free_count", ctypes.c_uint),           # natural_t
        ("active_count", ctypes.c_uint),         # natural_t
        ("inactive_count", ctypes.c_uint),       # natural_t
        ("wire_count", ctypes.c_uint),           # natural_t
        ("zero_fill_count", ctypes.c_uint64),    # uint64_t
        ("reactivations", ctypes.c_uint64),      # uint64_t
        ("pageins", ctypes.c_uint64),            # uint64_t
        ("pageouts", ctypes.c_uint64),           # uint64_t
        ("faults", ctypes.c_uint64),             # uint64_t
        ("cow_faults", ctypes.c_uint64),         # uint64_t
        ("lookups", ctypes.c_uint64),            # uint64_t
        ("hits", ctypes.c_uint64),               # uint64_t
        ("purges", ctypes.c_uint64),             # uint64_t
        ("purgeable_count", ctypes.c_uint),      # natural_t
        ("speculative_count", ctypes.c_uint),    # natural_t
        ("decompressions", ctypes.c_uint64),     # uint64_t
        ("compressions", ctypes.c_uint64),       # uint64_t
        ("swapins", ctypes.c_uint64),            # uint64_t
        ("swapouts", ctypes.c_uint64),           # uint64_t
        ("compressor_page_count", ctypes.c_uint),   # natural_t
        ("throttled_count", ctypes.c_uint),          # natural_t
        ("external_page_count", ctypes.c_uint),      # natural_t
        ("internal_page_count", ctypes.c_uint),      # natural_t
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),  # uint64_t
    ]


#: Mach host_statistics64 flavor for VM info.
_HOST_VM_INFO64 = 4

#: Count of integers in the struct (what Mach expects).
_HOST_VM_INFO64_COUNT = ctypes.sizeof(_VMStatistics64) // ctypes.sizeof(ctypes.c_int)

# Load libc once at import time (safe on macOS/Linux).
_libc: ctypes.CDLL | None = None
try:
    _lib_path = ctypes.util.find_library("c")
    if _lib_path:
        _libc = ctypes.CDLL(_lib_path)
except OSError:
    pass


def _get_total_system_mb() -> float:
    """Return total physical memory in MB."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (ValueError, OSError):
        return 0.0


def _get_available_mb_macos() -> float | None:
    """Get available memory via Mach host_statistics64 (no subprocess).

    Counts free + inactive + purgeable pages.  Inactive pages are the
    first macOS reclaims under pressure and are cheaply reclaimable
    (unlike compressed pages which live in the compressor).  This
    matches how Activity Monitor and ``memory_pressure`` report
    available memory.
    """
    if _libc is None:
        return None

    try:
        host = _libc.mach_host_self()
        stats = _VMStatistics64()
        count = ctypes.c_uint(_HOST_VM_INFO64_COUNT)

        ret = _libc.host_statistics64(
            host, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count),
        )
        if ret != 0:
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = (
            stats.free_count + stats.inactive_count + stats.purgeable_count
        )
        return (available_pages * page_size) / (1024 * 1024)
    except Exception:
        return None


def _get_available_mb_linux() -> float | None:
    """Read MemAvailable from /proc/meminfo (no subprocess)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # "MemAvailable:   12345678 kB"
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def get_available_mb() -> float | None:
    """Return available system memory in MB (platform-aware, no subprocess)."""
    if platform.system() == "Darwin":
        return _get_available_mb_macos()
    return _get_available_mb_linux()


# ------------------------------------------------------------------ #
# Per-PID RSS measurement (no subprocess, no psutil)
# ------------------------------------------------------------------ #

#: macOS proc_taskinfo struct — must exactly match the <libproc.h>
#: proc_taskinfo layout.  If Apple changes this struct in a future macOS
#: version the fields will silently return garbage; the blanket
#: ``except Exception`` in the caller prevents crashes but not bad data.
class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


#: PROC_PIDTASKINFO flavor for proc_pidinfo().
_PROC_PIDTASKINFO = 4


def _get_pid_rss_mb_macos(pid: int) -> float | None:
    """Get RSS for a specific PID via proc_pidinfo (no subprocess)."""
    if _libc is None:
        return None
    try:
        info = _ProcTaskInfo()
        size = ctypes.sizeof(info)
        ret = _libc.proc_pidinfo(pid, _PROC_PIDTASKINFO, 0, ctypes.byref(info), size)
        if ret <= 0:
            return None
        return info.pti_resident_size / (1024 * 1024)
    except Exception:
        return None


def _get_pid_rss_mb_linux(pid: int) -> float | None:
    """Get RSS for a specific PID by reading /proc/{pid}/status."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:   12345 kB"
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def get_pid_rss_mb(pid: int) -> float | None:
    """Return RSS in MB for a specific PID (platform-aware, no subprocess)."""
    if platform.system() == "Darwin":
        return _get_pid_rss_mb_macos(pid)
    return _get_pid_rss_mb_linux(pid)


class MemoryMonitor(BackgroundLoop):
    """Periodically checks system-wide available memory and logs warnings.

    Uses direct syscalls (Mach on macOS, /proc on Linux) instead of
    spawning subprocesses, so checks work even under severe memory pressure.
    """

    def __init__(
        self,
        recorder: SessionRecorder,
        shutdown_event: asyncio.Event,
        interval: int = _DEFAULT_INTERVAL,
    ) -> None:
        super().__init__(shutdown_event=shutdown_event, interval=interval)
        self._recorder = recorder

        self._total_mb = _get_total_system_mb()
        self._last_level: str = "ok"

    def _should_start(self) -> bool:
        if self._total_mb <= 0:
            logger.warning("Could not determine system memory — memory monitor disabled")
            return False

        test = get_available_mb()
        if test is None:
            logger.warning("Cannot read system memory stats — memory monitor disabled")
            return False

        logger.info(
            "Memory monitor started: %.0f MB available / %.0f MB total (checking every %ds)",
            test, self._total_mb, self._interval,
        )
        return True

    async def _tick(self) -> None:
        self._check()

    def _check(self) -> None:
        """Run a single memory check (synchronous — no subprocess)."""
        available_mb = get_available_mb()
        if available_mb is None:
            return

        used_mb = self._total_mb - available_mb
        used_pct = used_mb / self._total_mb * 100

        if available_mb <= _CRITICAL_AVAILABLE_MB:
            level = "critical"
        elif available_mb <= _WARN_AVAILABLE_MB:
            level = "warn"
        else:
            level = "ok"

        # Only log on transitions or repeated critical.
        if level == "ok":
            if self._last_level != "ok":
                logger.info("Memory pressure eased: %.0f MB available", available_mb)
            self._last_level = level
            return

        if level != self._last_level or level == "critical":
            tag = "⚠️  LOW MEMORY" if level == "warn" else "🔴 CRITICAL MEMORY"
            msg = (
                f"{tag}: {available_mb:.0f} MB available "
                f"(system using {used_pct:.0f}% of {self._total_mb:.0f} MB)"
            )
            click.echo(msg, err=True)
            logger.warning(msg)

            self._recorder.record(
                StatusEvent(
                    ts="",
                    seq=0,
                    agent=SYSTEM_SENDER,
                    status=f"memory_{level}: {available_mb:.0f}MB available",
                )
            )

        self._last_level = level
