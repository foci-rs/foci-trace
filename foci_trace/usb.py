"""Linux pyusb discovery for FOCI trace interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import usb.core
import usb.util

TRACE_PRODUCT = "FOCI OpenFFBoard (trace)"


@dataclass(frozen=True)
class TraceDeviceInfo:
    """USB trace device description."""

    bus: int
    address: int
    product: str
    serial: str

    @property
    def path(self) -> str:
        """Return a compact bus-address path label."""
        return f"{self.bus}-{self.address}"


def find_trace_devices() -> list[usb.core.Device]:
    """Return USB devices exposing the trace product string."""
    devices: Iterable[usb.core.Device] = usb.core.find(find_all=True)
    found: list[usb.core.Device] = []
    for dev in devices:
        product = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else ""
        if product == TRACE_PRODUCT:
            found.append(dev)
    return found


def device_info(dev: usb.core.Device) -> TraceDeviceInfo:
    """Return stable display fields for a pyusb device."""
    product = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else ""
    serial = usb.util.get_string(dev, dev.iSerialNumber) if dev.iSerialNumber else ""
    return TraceDeviceInfo(dev.bus, dev.address, product, serial)
