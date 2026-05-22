import sys
import types

if "usb" not in sys.modules:
    usb_module = types.ModuleType("usb")
    usb_core = types.ModuleType("usb.core")
    usb_util = types.ModuleType("usb.util")
    usb_core.find = lambda **_kwargs: []
    usb_util.get_string = lambda *_args, **_kwargs: ""
    usb_util.claim_interface = lambda *_args, **_kwargs: None
    usb_util.release_interface = lambda *_args, **_kwargs: None
    usb_module.core = usb_core
    usb_module.util = usb_util
    sys.modules["usb"] = usb_module
    sys.modules["usb.core"] = usb_core
    sys.modules["usb.util"] = usb_util

from foci_trace.cli import (
    bulk_read_size,
    live_summary_enabled,
    notify_recording_started,
    select_trace_devices,
    should_render_summary,
)


class FakeInfo:
    def __init__(self, serial: str) -> None:
        self.serial = serial


def test_bulk_read_size_is_not_clamped_to_endpoint_packet_size() -> None:
    assert bulk_read_size(requested_size=4096, endpoint_packet_size=64) == 4096


def test_bulk_read_size_uses_at_least_one_endpoint_packet() -> None:
    assert bulk_read_size(requested_size=16, endpoint_packet_size=64) == 64


def test_summary_rendering_is_rate_limited() -> None:
    interval_ns = 250_000_000

    assert should_render_summary(now_ns=1_000_000_000, last_render_ns=0, interval_ns=interval_ns)
    assert not should_render_summary(
        now_ns=1_100_000_000,
        last_render_ns=1_000_000_000,
        interval_ns=interval_ns,
    )
    assert should_render_summary(
        now_ns=1_250_000_000,
        last_render_ns=1_000_000_000,
        interval_ns=interval_ns,
    )


def test_zero_summary_rate_disables_live_summary_decode() -> None:
    assert not live_summary_enabled(0.0)
    assert not live_summary_enabled(-1.0)
    assert live_summary_enabled(1.0)


def test_positive_warmup_enables_pre_capture_drain() -> None:
    from foci_trace.cli import warmup_enabled

    assert not warmup_enabled(0.0)
    assert not warmup_enabled(-1.0)
    assert warmup_enabled(0.5)


def test_notify_recording_started_prints_and_writes_ready_file(tmp_path, capsys) -> None:
    ready_file = tmp_path / "trace.ready"

    notify_recording_started("/tmp/capture.bin", str(ready_file))

    assert ready_file.read_text(encoding="utf-8") == "/tmp/capture.bin\n"
    assert capsys.readouterr().out == "recording started: /tmp/capture.bin\n"


def test_select_trace_devices_returns_all_devices_without_serial_filter() -> None:
    devices = [object(), object()]

    assert select_trace_devices(devices, [], lambda _dev: FakeInfo("")) == devices


def test_select_trace_devices_filters_by_exact_serial() -> None:
    first = object()
    second = object()
    serials = {first: "AAA", second: "BBB"}

    selected = select_trace_devices(
        [first, second],
        ["BBB"],
        lambda dev: FakeInfo(serials[dev]),
    )

    assert selected == [second]


def test_select_trace_devices_rejects_missing_serial() -> None:
    devices = [object()]

    try:
        select_trace_devices(devices, ["missing"], lambda _dev: FakeInfo("present"))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing serial was accepted")
