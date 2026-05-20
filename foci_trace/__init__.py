"""FOCI USB trace capture and decode tools."""

from .protocol import Frame, decode_frame

__all__ = ["Frame", "decode_frame"]
