"""Shared constants and type aliases for the Lattice runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

#: Special sender name for system-level messages (bypasses topology).
SYSTEM_SENDER = "__system__"

#: Callback type for agent response handlers.
ResponseCallback = Callable[[str], Awaitable[None]]
