"""Offline summary for FOCI USB trace capture files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import io
import math
from statistics import median
import struct
import sys
from typing import BinaryIO

from .capture_file import (
    RECORD_TYPE_DEVICE_TABLE,
    RECORD_TYPE_FIRMWARE_FRAME,
    CaptureReader,
)
from .protocol import (
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_STATUS,
    DecodeError,
    decode_frame,
)

DEFAULT_CLOCK_HZ = 84_000_000
DEFAULT_MOTION_GAP_S = 0.08
DEFAULT_MOTION_MIN_STEPS = 1000


@dataclass
class MotionSegmentReport:
    """Motion-focused summary for one contiguous stepper segment."""

    segment_index: int
    start_s: float
    duration_s: float
    step_delta: int
    critical_samples: int
    control_detail_samples: int
    fitted_error_rms_steps: float
    fitted_error_peak_steps: float
    actual_torque_range: tuple[int, int]
    uq_limited_range: tuple[int, int]
    voltage_rail_samples: int
    torque_command_range: tuple[int, int] | None
    torque_command_saturated_samples: int
    actual_to_command_abs_median: float | None
    accel_feedforward_offset_range: tuple[int, int]
    accel_feedforward_gain_range: tuple[int, int]
    accel_feedforward_flags: int


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
    motion_segments: list[MotionSegmentReport] = field(default_factory=list)
    decode_errors: int = 0


@dataclass(frozen=True)
class _CriticalObservation:
    timestamp_ticks: int
    step_position: int
    tmc_position_actual: int
    actual_torque: int
    uq_limited: int
    accel_feedforward_offset: int
    accel_feedforward_gain: int
    accel_feedforward_flags: int


@dataclass(frozen=True)
class _ControlDetailObservation:
    timestamp_ticks: int
    torque_command: int
    voltage_limit: int
    torque_limit: int


@dataclass
class _DeviceState:
    report: DeviceReport
    clock_hz: int = DEFAULT_CLOCK_HZ
    first_timestamp: int | None = None
    last_sequence: int | None = None
    first_status_by_source: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)
    last_status_by_source: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)
    critical_observations: list[_CriticalObservation] = field(default_factory=list)
    control_detail_observations: list[_ControlDetailObservation] = field(default_factory=list)


def _signed_i16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x1_0000 if value & 0x8000 else value


def _signed_i32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


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
    actual_flux, actual_torque = _split_i16_pair(int(payload["pid_torque_flux_actual"]))
    ud, uq = _split_i16_pair(int(payload["interim_foc_uq_ud"]))
    ud_limited, uq_limited = _split_i16_pair(int(payload["interim_foc_uq_ud_limited"]))
    state.critical_observations.append(
        _CriticalObservation(
            timestamp_ticks=timestamp_ticks,
            step_position=int(payload["step_position"]),
            tmc_position_actual=int(payload["tmc_position_actual"]),
            actual_torque=actual_torque,
            uq_limited=uq_limited,
            accel_feedforward_offset=int(payload["cached_accel_torque_offset"]),
            accel_feedforward_gain=int(payload["accel_feedforward_gain_permille"]),
            accel_feedforward_flags=int(payload["accel_feedforward_flags"]),
        )
    )
    if _relative_seconds(state, timestamp_ticks) < steady_after_s:
        return
    report = state.report
    report.steady_samples += 1
    if payload["interim_foc_uq_ud"] != payload["interim_foc_uq_ud_limited"]:
        report.voltage_limited_samples += 1
    for name in ("position_error", "velocity_error"):
        _update_range(report.ranges, name, int(payload[name]))

    for name, value in (
        ("actual_flux", actual_flux),
        ("actual_torque", actual_torque),
        ("ud", ud),
        ("uq", uq),
        ("ud_limited", ud_limited),
        ("uq_limited", uq_limited),
        ("ud_delta", ud - ud_limited),
        ("uq_delta", uq - uq_limited),
        ("accel_ff_offset", int(payload["cached_accel_torque_offset"])),
        ("accel_ff_gain", int(payload["accel_feedforward_gain_permille"])),
        ("accel_ff_flags", int(payload["accel_feedforward_flags"])),
    ):
        _update_range(report.ranges, name, value)


def _observe_control_detail(
    state: _DeviceState, payload: dict[str, object], timestamp_ticks: int
) -> None:
    voltage_limits = _split_i16_pair(int(payload["pidout_uq_ud_limits"]))
    torque_limits = _split_i16_pair(int(payload["pid_torque_flux_limits"]))
    state.control_detail_observations.append(
        _ControlDetailObservation(
            timestamp_ticks=timestamp_ticks,
            torque_command=_signed_i32(int(payload["interim_pidout_target_torque"])),
            voltage_limit=max(abs(voltage_limits[0]), abs(voltage_limits[1])),
            torque_limit=max(abs(torque_limits[0]), abs(torque_limits[1])),
        )
    )


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


def _motion_groups(
    observations: list[_CriticalObservation],
    clock_hz: int,
    gap_s: float,
    min_steps: int,
) -> list[tuple[int, int]]:
    if len(observations) < 2:
        return []
    gap_ticks = int(gap_s * clock_hz)
    change_indices = [
        idx
        for idx in range(1, len(observations))
        if observations[idx].step_position != observations[idx - 1].step_position
    ]
    groups: list[tuple[int, int]] = []
    group_start: int | None = None
    previous_timestamp: int | None = None
    previous_index: int | None = None
    for idx in change_indices:
        timestamp = observations[idx].timestamp_ticks
        if (
            group_start is not None
            and previous_timestamp is not None
            and timestamp - previous_timestamp > gap_ticks
        ):
            start = max(0, group_start - 1)
            end = previous_index if previous_index is not None else idx
            if (
                abs(observations[end].step_position - observations[start].step_position)
                >= min_steps
            ):
                groups.append((start, end))
            group_start = idx
        elif group_start is None:
            group_start = idx
        previous_timestamp = timestamp
        previous_index = idx
    if group_start is not None and previous_index is not None:
        start = max(0, group_start - 1)
        end = previous_index
        if abs(observations[end].step_position - observations[start].step_position) >= min_steps:
            groups.append((start, end))
    return groups


def _fitted_error_stats(samples: list[_CriticalObservation]) -> tuple[float, float]:
    if len(samples) < 2:
        return 0.0, 0.0
    step_values = [sample.step_position for sample in samples]
    actual_values = [sample.tmc_position_actual for sample in samples]
    mean_step = sum(step_values) / len(step_values)
    mean_actual = sum(actual_values) / len(actual_values)
    variance = sum((step - mean_step) ** 2 for step in step_values)
    if variance == 0:
        return 0.0, 0.0
    covariance = sum(
        (step - mean_step) * (actual - mean_actual)
        for step, actual in zip(step_values, actual_values, strict=True)
    )
    slope = covariance / variance
    if abs(slope) < 1e-9:
        return 0.0, 0.0
    intercept = mean_actual - slope * mean_step
    residuals = [
        step - ((actual - intercept) / slope)
        for step, actual in zip(step_values, actual_values, strict=True)
    ]
    rms = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    peak = max(abs(residual) for residual in residuals)
    return rms, peak


def _range(values: list[int]) -> tuple[int, int]:
    return min(values), max(values)


def _nearest_critical(
    samples: list[_CriticalObservation], timestamp_ticks: int
) -> _CriticalObservation:
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _finalize_motion_segments(
    state: _DeviceState, motion_gap_s: float, motion_min_steps: int
) -> None:
    observations = sorted(state.critical_observations, key=lambda sample: sample.timestamp_ticks)
    details = sorted(state.control_detail_observations, key=lambda sample: sample.timestamp_ticks)
    for segment_index, (start_idx, end_idx) in enumerate(
        _motion_groups(observations, state.clock_hz, motion_gap_s, motion_min_steps)
    ):
        segment_samples = observations[start_idx : end_idx + 1]
        start_tick = segment_samples[0].timestamp_ticks
        end_tick = segment_samples[-1].timestamp_ticks
        segment_details = [
            detail for detail in details if start_tick <= detail.timestamp_ticks <= end_tick
        ]
        step_delta = segment_samples[-1].step_position - segment_samples[0].step_position
        rms, peak = _fitted_error_stats(segment_samples)
        torque_values = [sample.actual_torque for sample in segment_samples]
        uq_limited_values = [sample.uq_limited for sample in segment_samples]
        accel_offsets = [sample.accel_feedforward_offset for sample in segment_samples]
        accel_gains = [sample.accel_feedforward_gain for sample in segment_samples]
        accel_flags = 0
        for sample in segment_samples:
            accel_flags |= sample.accel_feedforward_flags
        voltage_limit = max((detail.voltage_limit for detail in segment_details), default=0)
        if voltage_limit == 0:
            voltage_limit = max((abs(value) for value in uq_limited_values), default=0)
        voltage_threshold = int(voltage_limit * 0.98)
        voltage_rail_samples = (
            sum(abs(value) >= voltage_threshold for value in uq_limited_values)
            if voltage_threshold > 0
            else 0
        )
        torque_commands = [detail.torque_command for detail in segment_details]
        torque_limit = max((detail.torque_limit for detail in segment_details), default=0)
        torque_threshold = int(torque_limit * 0.98)
        saturated_details = [
            detail
            for detail in segment_details
            if torque_threshold > 0 and abs(detail.torque_command) >= torque_threshold
        ]
        ratios = [
            abs(_nearest_critical(segment_samples, detail.timestamp_ticks).actual_torque)
            / abs(detail.torque_command)
            for detail in saturated_details
            if detail.torque_command != 0
        ]
        state.report.motion_segments.append(
            MotionSegmentReport(
                segment_index=segment_index,
                start_s=(start_tick - (state.first_timestamp or start_tick)) / state.clock_hz,
                duration_s=(end_tick - start_tick) / state.clock_hz,
                step_delta=step_delta,
                critical_samples=len(segment_samples),
                control_detail_samples=len(segment_details),
                fitted_error_rms_steps=rms,
                fitted_error_peak_steps=peak,
                actual_torque_range=_range(torque_values),
                uq_limited_range=_range(uq_limited_values),
                voltage_rail_samples=voltage_rail_samples,
                torque_command_range=_range(torque_commands) if torque_commands else None,
                torque_command_saturated_samples=len(saturated_details),
                actual_to_command_abs_median=median(ratios) if ratios else None,
                accel_feedforward_offset_range=_range(accel_offsets),
                accel_feedforward_gain_range=_range(accel_gains),
                accel_feedforward_flags=accel_flags,
            )
        )


def summarize_capture(
    source: BinaryIO | io.BytesIO,
    steady_after_s: float = 10.0,
    motion_gap_s: float = DEFAULT_MOTION_GAP_S,
    motion_min_steps: int = DEFAULT_MOTION_MIN_STEPS,
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
        elif header.frame_type == FRAME_TYPE_CONTROL_DETAIL:
            _observe_control_detail(state, frame.payload, header.timestamp_ticks)

    for state in states.values():
        _finalize_status(state)
        _finalize_motion_segments(state, motion_gap_s, motion_min_steps)
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
            "ud={ud} uq={uq} ud_delta={ud_delta} uq_delta={uq_delta} "
            "accel_ff={accel} gain={gain} flags=0x{flags:02x}".format(
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
                accel=_range_text(report.ranges, "accel_ff_offset"),
                gain=_range_text(report.ranges, "accel_ff_gain"),
                flags=report.ranges.get("accel_ff_flags", (0, 0))[1],
            )
        )
        for segment in report.motion_segments:
            command_range = (
                "n/a"
                if segment.torque_command_range is None
                else f"{segment.torque_command_range[0]}..{segment.torque_command_range[1]}"
            )
            ratio = (
                "n/a"
                if segment.actual_to_command_abs_median is None
                else f"{segment.actual_to_command_abs_median:.2f}"
            )
            lines.append(
                "  motion{idx} delta={delta} duration={duration:.3f}s "
                "critical={critical} control_detail={detail} "
                "fit_rms={rms:.1f} fit_peak={peak:.1f} "
                "torque={torque_lo}..{torque_hi} uq_limited={uq_lo}..{uq_hi} "
                "uq_rail={rail}/{critical} torque_cmd={cmd} "
                "saturated={sat}/{detail} actual_cmd_median={ratio} "
                "accel_ff={accel_lo}..{accel_hi} gain={gain_lo}..{gain_hi} "
                "flags=0x{flags:02x}".format(
                    idx=segment.segment_index,
                    delta=segment.step_delta,
                    duration=segment.duration_s,
                    critical=segment.critical_samples,
                    detail=segment.control_detail_samples,
                    rms=segment.fitted_error_rms_steps,
                    peak=segment.fitted_error_peak_steps,
                    torque_lo=segment.actual_torque_range[0],
                    torque_hi=segment.actual_torque_range[1],
                    uq_lo=segment.uq_limited_range[0],
                    uq_hi=segment.uq_limited_range[1],
                    rail=segment.voltage_rail_samples,
                    cmd=command_range,
                    sat=segment.torque_command_saturated_samples,
                    ratio=ratio,
                    accel_lo=segment.accel_feedforward_offset_range[0],
                    accel_hi=segment.accel_feedforward_offset_range[1],
                    gain_lo=segment.accel_feedforward_gain_range[0],
                    gain_hi=segment.accel_feedforward_gain_range[1],
                    flags=segment.accel_feedforward_flags,
                )
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the offline capture analyzer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--steady-after-s", type=float, default=10.0)
    parser.add_argument("--motion-gap-s", type=float, default=DEFAULT_MOTION_GAP_S)
    parser.add_argument("--motion-min-steps", type=int, default=DEFAULT_MOTION_MIN_STEPS)
    args = parser.parse_args(argv)

    try:
        with open(args.capture, "rb") as source:
            reports = summarize_capture(
                source,
                steady_after_s=args.steady_after_s,
                motion_gap_s=args.motion_gap_s,
                motion_min_steps=args.motion_min_steps,
            )
    except OSError as exc:
        print(f"failed to read {args.capture}: {exc}", file=sys.stderr)
        return 2
    print(format_summary(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
