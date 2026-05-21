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

from foci_trace.cli import bulk_read_size, live_summary_enabled, should_render_summary


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
