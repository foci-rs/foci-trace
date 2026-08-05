"""Command-line entry point for FOCI USB trace capture."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import usb.util

from .capture_file import CaptureWriter
from .protocol import DecodeError, FrameAssembler, decode_frame
from .summary import LiveSummary
from .usb import TRACE_INTERFACE, TraceDeviceInfo, device_info, find_trace_devices, trace_endpoint


def bulk_read_size(requested_size: int, endpoint_packet_size: int) -> int:
    """Return the USB bulk transfer size to request from pyusb."""
    return max(requested_size, endpoint_packet_size)


def should_render_summary(now_ns: int, last_render_ns: int, interval_ns: int) -> bool:
    """Return true when the live summary should be refreshed."""
    return last_render_ns == 0 or now_ns - last_render_ns >= interval_ns


def live_summary_enabled(summary_hz: float) -> bool:
    """Return true when frames should be decoded for the live summary."""
    return summary_hz > 0


def warmup_enabled(warmup_s: float) -> bool:
    """Return true when capture should drain stale trace frames first."""
    return warmup_s > 0


def notify_recording_started(output_path: str, ready_file: str | None) -> None:
    """Notify operators and automation that post-warmup recording has started."""
    if ready_file:
        Path(ready_file).write_text(f"{output_path}\n", encoding="utf-8")
    print(f"recording started: {output_path}", flush=True)


def select_trace_devices(
    devices: list[object],
    requested_serials: list[str],
    info_for: Callable[[object], TraceDeviceInfo],
) -> list[object]:
    """Return devices matching requested serial numbers, preserving device order."""
    if not requested_serials:
        return devices

    requested = set(requested_serials)
    selected = [dev for dev in devices if info_for(dev).serial in requested]
    found = {info_for(dev).serial for dev in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"trace serial(s) not found: {', '.join(missing)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    """Capture trace frames from all connected trace devices."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--packet-size", type=int, default=4096)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--summary-hz", type=float, default=4.0)
    parser.add_argument("--ready-file")
    parser.add_argument("--serial", action="append", default=[])
    args = parser.parse_args(argv)

    devices = find_trace_devices()
    try:
        devices = select_trace_devices(devices, args.serial, device_info)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not devices:
        print("no FOCI trace devices found", file=sys.stderr)
        return 2

    summary = LiveSummary()
    assemblers = [FrameAssembler() for _ in devices]
    endpoints = [trace_endpoint(dev) for dev in devices]
    summary_enabled = live_summary_enabled(args.summary_hz)
    summary_interval_ns = int(1_000_000_000 / args.summary_hz) if summary_enabled else 0
    last_summary_ns = 0

    for dev in devices:
        usb.util.claim_interface(dev, TRACE_INTERFACE)
    try:
        if warmup_enabled(args.warmup_s):
            warmup_deadline = time.monotonic() + args.warmup_s
            while time.monotonic() < warmup_deadline:
                for index, dev in enumerate(devices):
                    endpoint_address, max_packet_size = endpoints[index]
                    try:
                        data = bytes(
                            dev.read(
                                endpoint_address,
                                bulk_read_size(args.packet_size, max_packet_size),
                                timeout=10,
                            )
                        )
                    except Exception:
                        continue
                    if data:
                        _ = assemblers[index].feed(data)
            assemblers = [FrameAssembler() for _ in devices]

        notify_recording_started(args.output, args.ready_file)

        start_unix_ns = time.time_ns()
        start_mono_ns = time.monotonic_ns()
        deadline = time.monotonic() + args.duration_s if args.duration_s > 0 else None
        with open(args.output, "wb") as output:
            writer = CaptureWriter(output, start_unix_ns, start_mono_ns)
            for index, dev in enumerate(devices):
                info = device_info(dev)
                writer.write_device_table(
                    index,
                    time.monotonic_ns(),
                    info.path,
                    info.product,
                    info.serial,
                    b"",
                )

            while deadline is None or time.monotonic() < deadline:
                for index, dev in enumerate(devices):
                    endpoint_address, max_packet_size = endpoints[index]
                    try:
                        data = bytes(
                            dev.read(
                                endpoint_address,
                                bulk_read_size(args.packet_size, max_packet_size),
                                timeout=10,
                            )
                        )
                    except Exception:
                        continue
                    if not data:
                        continue
                    now = time.monotonic_ns()
                    for frame_data in assemblers[index].feed(data):
                        writer.write_firmware_frame(index, now, frame_data)
                        if summary_enabled:
                            try:
                                summary.observe(index, now, decode_frame(frame_data))
                            except DecodeError:
                                continue
                if summary_enabled and should_render_summary(
                    time.monotonic_ns(),
                    last_summary_ns,
                    summary_interval_ns,
                ):
                    last_summary_ns = time.monotonic_ns()
                    print(summary.render(), end="\r", flush=True)
    finally:
        for dev in devices:
            usb.util.release_interface(dev, TRACE_INTERFACE)
    if summary_enabled:
        print(summary.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
