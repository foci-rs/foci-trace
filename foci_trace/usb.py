"""Linux pyusb discovery for FOCI trace interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import usb.core
import usb.util

TRACE_PRODUCT = "FOCI OpenFFBoard (trace)"
TRACE_VID = 0x1D50
TRACE_PID = 0x614E
TRACE_INTERFACE = 2
TRACE_INTERFACE_CLASS = 0xFF
USB_ENDPOINT_IN = 0x80
USB_ENDPOINT_TRANSFER_TYPE_MASK = 0x03
USB_ENDPOINT_TRANSFER_TYPE_BULK = 0x02


class TraceUsbError(RuntimeError):
    """Raised when a USB device does not expose the trace interface."""


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
    """Return USB devices exposing the vendor-specific trace interface."""
    devices: Iterable[usb.core.Device] = usb.core.find(
        find_all=True, idVendor=TRACE_VID, idProduct=TRACE_PID
    )
    found: list[usb.core.Device] = []
    for dev in devices:
        try:
            trace_endpoint(dev)
        except TraceUsbError:
            continue
        else:
            found.append(dev)
    return found


def device_info(dev: usb.core.Device) -> TraceDeviceInfo:
    """Return stable display fields for a pyusb device."""
    product = _safe_get_string(dev, dev.iProduct) if dev.iProduct else ""
    serial = _safe_get_string(dev, dev.iSerialNumber) if dev.iSerialNumber else ""
    if not product:
        product = TRACE_PRODUCT
    return TraceDeviceInfo(dev.bus, dev.address, product, serial)


def trace_endpoint(dev: usb.core.Device) -> tuple[int, int]:
    """Return the trace bulk IN endpoint address and max packet size."""
    try:
        config = dev.get_active_configuration()
    except Exception as exc:
        raise TraceUsbError("USB device has no active configuration") from exc

    for interface in config:
        if interface.bInterfaceNumber != TRACE_INTERFACE:
            continue
        if interface.bInterfaceClass != TRACE_INTERFACE_CLASS:
            continue
        for endpoint in interface:
            is_in = endpoint.bEndpointAddress & USB_ENDPOINT_IN
            transfer_type = endpoint.bmAttributes & USB_ENDPOINT_TRANSFER_TYPE_MASK
            if is_in and transfer_type == USB_ENDPOINT_TRANSFER_TYPE_BULK:
                return endpoint.bEndpointAddress, endpoint.wMaxPacketSize
    raise TraceUsbError("USB device has no trace bulk IN endpoint")


def _safe_get_string(dev: usb.core.Device, index: int) -> str:
    try:
        return usb.util.get_string(dev, index)
    except Exception:
        return ""
