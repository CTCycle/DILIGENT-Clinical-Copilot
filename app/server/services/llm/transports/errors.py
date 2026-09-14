from __future__ import annotations


class TransportCancellation(Exception):
    """A caller requested cancellation while a transport was active."""


class TransportUnsupportedCapability(Exception):
    """The requested operation is not supported by the selected model."""


class TransportPartialResponse(Exception):
    """A streaming provider response ended before a terminal event."""
