"""Offline summary for FOCI USB trace capture files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import io
import struct
import sys
from typing import BinaryIO

from .capture_file import (
    RECORD_TYPE_DEVICE_TABLE,
    RECORD_TYPE_FIRMWARE_FRAME,
    CaptureReader,
)
from .protocol import (
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_STATUS,
    DecodeError,
    decode_frame,
)

DEFAULT_CLOCK_HZ = 84_000_000


@dataclass
class DeviceReport:
    """Decoded summary for one trace device."""

    device_index: int
    usb_path: str = ""
    product: str = ""
    serial: str = ""
    metadata_serial: str = ""
    source_labels: list[str] = field(default_factory=list)
    frame_counts: dict[int, int] = field(default_factory=dict)
    missing_frames: int = 0
    sequence_gaps: dict[int, int] = field(default_factory=dict)
    status_deltas: dict[str, int] = field(default_factory=dict)
    steady_after_s: float = 0.0
    steady_samples: int = 0
    voltage_limited_samples: int = 0
    ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    decode_errors: int = 0


@dataclass
class _DeviceState:
    report: DeviceReport
    clock_hz: int = DEFAULT_CLOCK_HZ
    first_timestamp: int | None = None
    last_sequence: int | None = None
    first_status_by_source: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)
    last_status_by_source: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)


def _signed_i16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x1_0000 if value & 0x8000 else value


def _split_i16_pair(value: int) -> tuple[int, int]:
    return _signed_i16(value), _signed_i16(value >> 16)


def _update_range(ranges: dict[str, tuple[int, int]], name: str, value: int) -> None:
    current = ranges.get(name)
    if current is None:
        ranges[name] = (value, value)
        return
    ranges[name] = (min(current[0], value), max(current[1], value))


def _parse_device_table(payload: bytes) -> tuple[str, str, str]:
    offset = 0
    values: list[str] = []
    for _ in range(3):
        if offset + 2 > len(payload):
            return ("", "", "")
        size = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        end = offset + size
        if end > len(payload):
            return ("", "", "")
        values.append(payload[offset:end].decode("utf-8", "replace"))
        offset = end
    return values[0], values[1], values[2]


def _source_label(source: dict[str, object]) -> str:
    kind = int(source["source_kind"])
    index = int(source["source_index"])
    label = str(source["label"])
    prefix = {0: "board", 1: "FOCI", 2: "USB", 3: "Klipper"}.get(kind, f"kind{kind}")
    if kind in (0, 2, 3):
        return f"{prefix}={label}"
    return f"{prefix}[{index}]={label}"


def _state(states: dict[int, _DeviceState], device_index: int) -> _DeviceState:
    if device_index not in states:
        states[device_index] = _DeviceState(DeviceReport(device_index=device_index))
    return states[device_index]


def _relative_seconds(state: _DeviceState, timestamp_ticks: int) -> float:
    if state.first_timestamp is None:
        state.first_timestamp = timestamp_ticks
    return (timestamp_ticks - state.first_timestamp) / state.clock_hz


def _observe_sequence(state: _DeviceState, sequence: int) -> None:
    last = state.last_sequence
    if last is not None:
        expected = (last + 1) & 0xFFFF_FFFF
        if sequence != expected:
            missing = (sequence - expected) & 0xFFFF_FFFF
            state.report.missing_frames += missing
            state.report.sequence_gaps[missing] = state.report.sequence_gaps.get(missing, 0) + 1
    state.last_sequence = sequence


def _observe_metadata(state: _DeviceState, payload: dict[str, object]) -> None:
    state.clock_hz = int(payload.get("clock_freq", state.clock_hz))
    state.report.metadata_serial = str(payload.get("serial", ""))
    sources = payload.get("sources", [])
    if isinstance(sources, list):
        state.report.source_labels = [_source_label(source) for source in sources]


def _observe_status(
    state: _DeviceState, source: tuple[int, int], payload: dict[str, object]
) -> None:
    status = {name: int(value) for name, value in payload.items()}
    state.first_status_by_source.setdefault(source, status)
    state.last_status_by_source[source] = status


def _observe_critical(
    state: _DeviceState,
    payload: dict[str, object],
    timestamp_ticks: int,
    steady_after_s: float,
) -> None:
    if _relative_seconds(state, timestamp_ticks) < steady_after_s:
        return
    report = state.report
    report.steady_samples += 1
    if payload["interim_foc_uq_ud"] != payload["interim_foc_uq_ud_limited"]:
        report.voltage_limited_samples += 1
    for name in ("position_error", "velocity_error"):
        _update_range(report.ranges, name, int(payload[name]))

    actual_flux, actual_torque = _split_i16_pair(int(payload["pid_torque_flux_actual"]))
    ud, uq = _split_i16_pair(int(payload["interim_foc_uq_ud"]))
    ud_limited, uq_limited = _split_i16_pair(int(payload["interim_foc_uq_ud_limited"]))
    for name, value in (
        ("actual_flux", actual_flux),
        ("actual_torque", actual_torque),
        ("ud", ud),
        ("uq", uq),
        ("ud_limited", ud_limited),
        ("uq_limited", uq_limited),
        ("ud_delta", ud - ud_limited),
        ("uq_delta", uq - uq_limited),
    ):
        _update_range(report.ranges, name, value)


def _preferred_status_source(state: _DeviceState) -> tuple[int, int] | None:
    for source in ((2, 255), (0, 255), (1, 0)):
        if source in state.last_status_by_source:
            return source
    if state.last_status_by_source:
        return sorted(state.last_status_by_source)[0]
    return None


def _finalize_status(state: _DeviceState) -> None:
    source = _preferred_status_source(state)
    if source is None:
        return
    first = state.first_status_by_source[source]
    last = state.last_status_by_source[source]
    state.report.status_deltas = {
        name: int(last[name]) - int(first.get(name, 0)) for name in sorted(last)
    }


def summarize_capture(
    source: BinaryIO | io.BytesIO, steady_after_s: float = 10.0
) -> list[DeviceReport]:
    """Summarize one FTRF capture stream."""
    states: dict[int, _DeviceState] = {}
    for record in CaptureReader(source).records():
        state = _state(states, record.device_index)
        if record.record_type == RECORD_TYPE_DEVICE_TABLE:
            usb_path, product, serial = _parse_device_table(record.payload)
            state.report.usb_path = usb_path
            state.report.product = product
            state.report.serial = serial
            continue
        if record.record_type != RECORD_TYPE_FIRMWARE_FRAME:
            continue
        try:
            frame = decode_frame(record.payload)
        except DecodeError:
            state.report.decode_errors += 1
            continue
        header = frame.header
        state.report.frame_counts[header.frame_type] = (
            state.report.frame_counts.get(header.frame_type, 0) + 1
        )
        state.report.steady_after_s = steady_after_s
        if state.first_timestamp is None:
            state.first_timestamp = header.timestamp_ticks
        _observe_sequence(state, header.sequence)
        if header.frame_type == FRAME_TYPE_METADATA:
            _observe_metadata(state, frame.payload)
        elif header.frame_type == FRAME_TYPE_STATUS:
            _observe_status(state, (header.source_kind, header.source_index), frame.payload)
        elif header.frame_type == FRAME_TYPE_CRITICAL:
            _observe_critical(state, frame.payload, header.timestamp_ticks, steady_after_s)

    for state in states.values():
        _finalize_status(state)
    return [states[index].report for index in sorted(states)]


def _range_text(ranges: dict[str, tuple[int, int]], name: str) -> str:
    if name not in ranges:
        return "n/a"
    lo, hi = ranges[name]
    return f"{lo}..{hi}"


def format_summary(reports: list[DeviceReport]) -> str:
    """Format capture summaries for operator review."""
    lines: list[str] = []
    for report in reports:
        labels = ",".join(report.source_labels) if report.source_labels else "unlabeled"
        lines.append(
            "dev{idx} {path} {product} serial={serial} metadata_serial={meta} "
            "labels={labels}".format(
                idx=report.device_index,
                path=report.usb_path or "?",
                product=report.product or "?",
                serial=report.serial or "?",
                meta=report.metadata_serial or "?",
                labels=labels,
            )
        )
        lines.append(
            "  frames={frames} missing={missing} gaps={gaps} decode_errors={errors}".format(
                frames=report.frame_counts,
                missing=report.missing_frames,
                gaps=report.sequence_gaps,
                errors=report.decode_errors,
            )
        )
        lines.append(
            "  status_delta dropped={dropped} endpoint={endpoint} spi={spi} watchdog={watchdog}".format(
                dropped=report.status_deltas.get("dropped_motion_sample_count", 0),
                endpoint=report.status_deltas.get("endpoint_not_ready_count", 0),
                spi=report.status_deltas.get("spi_error_count", 0),
                watchdog=report.status_deltas.get("watchdog_or_step_overrun_count", 0),
            )
        )
        lines.append(
            "  steady>={after:g}s samples={samples} voltage_diff={limited} "
            "pos={pos} vel={vel} torque={torque} flux={flux} "
            "ud={ud} uq={uq} ud_delta={ud_delta} uq_delta={uq_delta}".format(
                after=report.steady_after_s,
                samples=report.steady_samples,
                limited=report.voltage_limited_samples,
                pos=_range_text(report.ranges, "position_error"),
                vel=_range_text(report.ranges, "velocity_error"),
                torque=_range_text(report.ranges, "actual_torque"),
                flux=_range_text(report.ranges, "actual_flux"),
                ud=_range_text(report.ranges, "ud"),
                uq=_range_text(report.ranges, "uq"),
                ud_delta=_range_text(report.ranges, "ud_delta"),
                uq_delta=_range_text(report.ranges, "uq_delta"),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the offline capture analyzer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--steady-after-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    try:
        with open(args.capture, "rb") as source:
            reports = summarize_capture(source, steady_after_s=args.steady_after_s)
    except OSError as exc:
        print(f"failed to read {args.capture}: {exc}", file=sys.stderr)
        return 2
    print(format_summary(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
