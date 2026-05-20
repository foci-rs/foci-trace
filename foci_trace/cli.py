"""Command-line entry point for FOCI USB trace capture."""

from __future__ import annotations

import argparse
import sys
import time

from .capture_file import CaptureWriter
from .protocol import DecodeError, FrameAssembler, decode_frame
from .summary import LiveSummary
from .usb import device_info, find_trace_devices


def main(argv: list[str] | None = None) -> int:
    """Capture trace frames from all connected trace devices."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--packet-size", type=int, default=128)
    args = parser.parse_args(argv)

    devices = find_trace_devices()
    if not devices:
        print("no FOCI trace devices found", file=sys.stderr)
        return 2

    summary = LiveSummary()
    assemblers = [FrameAssembler() for _ in devices]
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
                try:
                    data = bytes(dev.read(0x84, args.packet_size, timeout=10))
                except Exception:
                    continue
                if not data:
                    continue
                now = time.monotonic_ns()
                for frame_data in assemblers[index].feed(data):
                    writer.write_firmware_frame(index, now, frame_data)
                    try:
                        summary.observe(index, now, decode_frame(frame_data))
                    except DecodeError:
                        continue
            print(summary.render(), end="\r", flush=True)
    print(summary.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
