import sys
import types

import pytest

if "usb" not in sys.modules:
    usb_module = types.ModuleType("usb")
    usb_core = types.ModuleType("usb.core")
    usb_util = types.ModuleType("usb.util")
    usb_core.find = lambda **_kwargs: []
    usb_util.get_string = lambda *_args, **_kwargs: ""
    usb_module.core = usb_core
    usb_module.util = usb_util
    sys.modules["usb"] = usb_module
    sys.modules["usb.core"] = usb_core
    sys.modules["usb.util"] = usb_util

from foci_trace import usb as trace_usb


class FakeEndpoint:
    def __init__(self, address: int, attributes: int = 0x02, size: int = 64) -> None:
        self.bEndpointAddress = address
        self.bmAttributes = attributes
        self.wMaxPacketSize = size


class FakeInterface:
    def __init__(self, number: int, klass: int, endpoints: list[FakeEndpoint]) -> None:
        self.bInterfaceNumber = number
        self.bInterfaceClass = klass
        self._endpoints = endpoints

    def __iter__(self):
        return iter(self._endpoints)


class FakeConfig:
    def __init__(self, interfaces: list[FakeInterface]) -> None:
        self._interfaces = interfaces

    def __iter__(self):
        return iter(self._interfaces)


class FakeDevice:
    idVendor = 0x1D50
    idProduct = 0x614E
    bus = 1
    address = 9
    iProduct = 2
    iSerialNumber = 3

    def __init__(self, config: FakeConfig) -> None:
        self._config = config

    def get_active_configuration(self) -> FakeConfig:
        return self._config


def test_discovery_uses_vendor_interface_when_strings_are_unavailable(monkeypatch) -> None:
    dev = FakeDevice(
        FakeConfig(
            [
                FakeInterface(0, 0x02, [FakeEndpoint(0x83, attributes=0x03, size=8)]),
                FakeInterface(1, 0x0A, [FakeEndpoint(0x02), FakeEndpoint(0x81)]),
                FakeInterface(2, 0xFF, [FakeEndpoint(0x83)]),
            ]
        )
    )
    monkeypatch.setattr(trace_usb.usb.core, "find", lambda **_kwargs: [dev])
    monkeypatch.setattr(
        trace_usb.usb.util,
        "get_string",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no langid")),
    )

    found = trace_usb.find_trace_devices()

    assert found == [dev]
    assert trace_usb.trace_endpoint(dev) == (0x83, 64)


def test_trace_endpoint_rejects_missing_vendor_bulk_in() -> None:
    dev = FakeDevice(FakeConfig([FakeInterface(2, 0xFF, [])]))

    with pytest.raises(trace_usb.TraceUsbError, match="trace bulk IN endpoint"):
        trace_usb.trace_endpoint(dev)
