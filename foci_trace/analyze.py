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
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_STATUS,
    DecodeError,
    decode_frame,
)

DEFAULT_CLOCK_HZ = 84_000_000
DEFAULT_MOTION_GAP_S = 0.08
DEFAULT_MOTION_MIN_STEPS = 1000


@dataclass
class CurrentAuthorityReport:
    """Near-limit current-authority summary for one motion segment."""

    near_limit_samples: int
    sign_agreement_samples: int
    actual_to_command_abs_median: float | None
    target_to_command_abs_median: float | None
    actual_to_target_abs_median: float | None
    iq_to_actual_abs_median: float | None
    uq_limited_range: tuple[int, int]
    uq_delta_range: tuple[int, int]
    uq_to_limit_abs_median: float | None
    voltage_rail_samples: int
    # Raw ADC movement around the segment median, not calibrated current.
    adc_i0_delta_range: tuple[int, int] | None
    adc_i1_delta_range: tuple[int, int] | None


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
    foc_iq_range: tuple[int, int] | None
    foc_id_range: tuple[int, int] | None
    foc_iux_range: tuple[int, int] | None
    foc_iwy_range: tuple[int, int] | None
    adc_i0_raw_range: tuple[int, int] | None
    adc_i1_raw_range: tuple[int, int] | None
    current_authority: CurrentAuthorityReport | None
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
    torque_target: int
    actual_torque: int
    uq_requested: int
    uq_limited: int
    accel_feedforward_offset: int
    accel_feedforward_gain: int
    accel_feedforward_flags: int


@dataclass(frozen=True)
class _CurrentPathObservation:
    timestamp_ticks: int
    foc_iq: int
    foc_id: int
    foc_iux: int
    foc_iwy: int
    adc_i0_raw: int
    adc_i1_raw: int


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
    current_path_observations: list[_CurrentPathObservation] = field(default_factory=list)
    control_detail_observations: list[_ControlDetailObservation] = field(default_factory=list)


def _signed_i16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x1_0000 if value & 0x8000 else value


def _signed_i32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


def _split_i16_pair(value: int) -> tuple[int, int]:
    return _signed_i16(value), _signed_i16(value >> 16)


def _split_u16_pair(value: int) -> tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


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
    actual_flux: int | None = None
    torque_target: int | None = None
    if "interim_pid_torque_target_actual" in payload:
        actual_torque, torque_target = _split_i16_pair(
            int(payload["interim_pid_torque_target_actual"])
        )
    else:
        actual_flux, actual_torque = _split_i16_pair(int(payload["pid_torque_flux_actual"]))
        _flux_target, torque_target = _split_i16_pair(int(payload["pid_torque_flux_target"]))
    ud, uq = _split_i16_pair(int(payload["interim_foc_uq_ud"]))
    ud_limited, uq_limited = _split_i16_pair(int(payload["interim_foc_uq_ud_limited"]))
    if "interim_foc_iq_id" in payload:
        foc_id, foc_iq = _split_i16_pair(int(payload["interim_foc_iq_id"]))
        foc_iux, foc_iwy = _split_i16_pair(int(payload["interim_foc_iwy_iux"]))
        adc_i0_raw, adc_i1_raw = _split_u16_pair(int(payload["interim_adc_i1_i0"]))
        _observe_current_path_values(
            state,
            timestamp_ticks,
            steady_after_s,
            foc_iq,
            foc_id,
            foc_iux,
            foc_iwy,
            adc_i0_raw,
            adc_i1_raw,
        )
    state.critical_observations.append(
        _CriticalObservation(
            timestamp_ticks=timestamp_ticks,
            step_position=int(payload["step_position"]),
            tmc_position_actual=int(payload["tmc_position_actual"]),
            torque_target=torque_target if torque_target is not None else 0,
            actual_torque=actual_torque,
            uq_requested=uq,
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
        ("actual_torque", actual_torque),
        ("torque_target", torque_target),
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
        if value is not None:
            _update_range(report.ranges, name, value)
    if actual_flux is not None:
        _update_range(report.ranges, "actual_flux", actual_flux)


def _observe_current_path_values(
    state: _DeviceState,
    timestamp_ticks: int,
    steady_after_s: float,
    foc_iq: int,
    foc_id: int,
    foc_iux: int,
    foc_iwy: int,
    adc_i0_raw: int,
    adc_i1_raw: int,
) -> None:
    state.current_path_observations.append(
        _CurrentPathObservation(
            timestamp_ticks=timestamp_ticks,
            foc_iq=foc_iq,
            foc_id=foc_id,
            foc_iux=foc_iux,
            foc_iwy=foc_iwy,
            adc_i0_raw=adc_i0_raw,
            adc_i1_raw=adc_i1_raw,
        )
    )
    if _relative_seconds(state, timestamp_ticks) < steady_after_s:
        return
    for name, value in (
        ("foc_iq", foc_iq),
        ("foc_id", foc_id),
        ("foc_iux", foc_iux),
        ("foc_iwy", foc_iwy),
        ("adc_i0_raw", adc_i0_raw),
        ("adc_i1_raw", adc_i1_raw),
    ):
        _update_range(state.report.ranges, name, value)


def _observe_current_path(
    state: _DeviceState,
    payload: dict[str, object],
    timestamp_ticks: int,
    steady_after_s: float,
) -> None:
    foc_id, foc_iq = _split_i16_pair(int(payload["interim_foc_iq_id"]))
    foc_iux, foc_iwy = _split_i16_pair(int(payload["interim_foc_iwy_iux"]))
    adc_i0_raw, adc_i1_raw = _split_u16_pair(int(payload["interim_adc_i1_i0"]))
    _observe_current_path_values(
        state,
        timestamp_ticks,
        steady_after_s,
        foc_iq,
        foc_id,
        foc_iux,
        foc_iwy,
        adc_i0_raw,
        adc_i1_raw,
    )


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
        adjacent_gap = timestamp - observations[idx - 1].timestamp_ticks
        if adjacent_gap > gap_ticks:
            if group_start is not None and previous_index is not None:
                start = max(0, group_start - 1)
                end = previous_index
                if (
                    abs(observations[end].step_position - observations[start].step_position)
                    >= min_steps
                ):
                    groups.append((start, end))
            group_start = None
            previous_timestamp = None
            previous_index = None
            continue
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


def _optional_range(values: list[int | None]) -> tuple[int, int] | None:
    present = [value for value in values if value is not None]
    return _range(present) if present else None


def _nearest_critical(
    samples: list[_CriticalObservation], timestamp_ticks: int
) -> _CriticalObservation:
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _nearest_current_path(
    samples: list[_CurrentPathObservation], timestamp_ticks: int
) -> _CurrentPathObservation | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _nearest_control_detail(
    samples: list[_ControlDetailObservation], timestamp_ticks: int
) -> _ControlDetailObservation | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _sign_matches(left: int, right: int) -> bool:
    return (left < 0 and right < 0) or (left > 0 and right > 0) or (left == 0 and right == 0)


def _rounded_median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 3)


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    return int(round(median(values)))


def _current_authority_report(
    segment_samples: list[_CriticalObservation],
    segment_details: list[_ControlDetailObservation],
    segment_current_paths: list[_CurrentPathObservation],
) -> CurrentAuthorityReport | None:
    torque_limit = max((detail.torque_limit for detail in segment_details), default=0)
    if torque_limit <= 0:
        return None
    threshold = int(torque_limit * 0.8)
    near_limit_samples: list[
        tuple[_CriticalObservation, _ControlDetailObservation | None]
    ] = []
    for sample in segment_samples:
        if abs(sample.torque_target) < threshold:
            continue
        detail = _nearest_control_detail(segment_details, sample.timestamp_ticks)
        if detail is not None and abs(detail.torque_command) < threshold:
            continue
        near_limit_samples.append((sample, detail))
    if not near_limit_samples:
        return None
    near_limit_timestamps = {sample.timestamp_ticks for sample, _detail in near_limit_samples}

    actual_to_command: list[float] = []
    target_to_command: list[float] = []
    actual_to_target: list[float] = []
    iq_to_actual: list[float] = []
    uq_to_limit: list[float] = []
    uq_limited_values: list[int] = []
    uq_delta_values: list[int] = []
    adc_i0_deltas: list[int] = []
    adc_i1_deltas: list[int] = []
    sign_agreement = 0
    adc_i0_baseline = _median_int([sample.adc_i0_raw for sample in segment_current_paths])
    adc_i1_baseline = _median_int([sample.adc_i1_raw for sample in segment_current_paths])
    voltage_limit = max((detail.voltage_limit for detail in segment_details), default=0)
    voltage_threshold = int(voltage_limit * 0.98)
    voltage_rail_samples = 0

    for critical, detail in near_limit_samples:
        command = detail.torque_command if detail is not None else critical.torque_target
        actual = critical.actual_torque
        target = critical.torque_target
        uq_limited_values.append(critical.uq_limited)
        uq_delta_values.append(critical.uq_requested - critical.uq_limited)
        if voltage_limit > 0:
            uq_to_limit.append(abs(critical.uq_limited) / voltage_limit)
        if voltage_threshold > 0 and abs(critical.uq_limited) >= voltage_threshold:
            voltage_rail_samples += 1
        if command != 0:
            actual_to_command.append(abs(actual) / abs(command))
            target_to_command.append(abs(target) / abs(command))
        if target != 0:
            actual_to_target.append(abs(actual) / abs(target))
        if _sign_matches(actual, target):
            sign_agreement += 1

    for current_path in segment_current_paths:
        critical = _nearest_critical(segment_samples, current_path.timestamp_ticks)
        if critical.timestamp_ticks not in near_limit_timestamps:
            continue
        if critical.actual_torque != 0:
            iq_to_actual.append(abs(current_path.foc_iq) / abs(critical.actual_torque))
        if adc_i0_baseline is not None:
            adc_i0_deltas.append(current_path.adc_i0_raw - adc_i0_baseline)
        if adc_i1_baseline is not None:
            adc_i1_deltas.append(current_path.adc_i1_raw - adc_i1_baseline)

    return CurrentAuthorityReport(
        near_limit_samples=len(near_limit_samples),
        sign_agreement_samples=sign_agreement,
        actual_to_command_abs_median=_rounded_median(actual_to_command),
        target_to_command_abs_median=_rounded_median(target_to_command),
        actual_to_target_abs_median=_rounded_median(actual_to_target),
        iq_to_actual_abs_median=_rounded_median(iq_to_actual),
        uq_limited_range=_range(uq_limited_values),
        uq_delta_range=_range(uq_delta_values),
        uq_to_limit_abs_median=_rounded_median(uq_to_limit),
        voltage_rail_samples=voltage_rail_samples,
        adc_i0_delta_range=_optional_range(adc_i0_deltas),
        adc_i1_delta_range=_optional_range(adc_i1_deltas),
    )


def _finalize_motion_segments(
    state: _DeviceState, motion_gap_s: float, motion_min_steps: int
) -> None:
    observations = sorted(state.critical_observations, key=lambda sample: sample.timestamp_ticks)
    current_paths = sorted(
        state.current_path_observations, key=lambda sample: sample.timestamp_ticks
    )
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
        segment_current_paths = [
            sample for sample in current_paths if start_tick <= sample.timestamp_ticks <= end_tick
        ]
        foc_iq_range = _optional_range([sample.foc_iq for sample in segment_current_paths])
        foc_id_range = _optional_range([sample.foc_id for sample in segment_current_paths])
        foc_iux_range = _optional_range([sample.foc_iux for sample in segment_current_paths])
        foc_iwy_range = _optional_range([sample.foc_iwy for sample in segment_current_paths])
        adc_i0_raw_range = _optional_range(
            [sample.adc_i0_raw for sample in segment_current_paths]
        )
        adc_i1_raw_range = _optional_range(
            [sample.adc_i1_raw for sample in segment_current_paths]
        )
        current_authority = _current_authority_report(
            segment_samples, segment_details, segment_current_paths
        )
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
                foc_iq_range=foc_iq_range,
                foc_id_range=foc_id_range,
                foc_iux_range=foc_iux_range,
                foc_iwy_range=foc_iwy_range,
                adc_i0_raw_range=adc_i0_raw_range,
                adc_i1_raw_range=adc_i1_raw_range,
                current_authority=current_authority,
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
        elif header.frame_type == FRAME_TYPE_CURRENT_PATH:
            _observe_current_path(state, frame.payload, header.timestamp_ticks, steady_after_s)
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


def _optional_range_text(value: tuple[int, int] | None) -> str:
    if value is None:
        return "n/a"
    return f"{value[0]}..{value[1]}"


def _optional_ratio_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


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
            "iq={iq} id={id} iux={iux} iwy={iwy} "
            "adc0_raw={adc0} adc1_raw={adc1} "
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
                iq=_range_text(report.ranges, "foc_iq"),
                id=_range_text(report.ranges, "foc_id"),
                iux=_range_text(report.ranges, "foc_iux"),
                iwy=_range_text(report.ranges, "foc_iwy"),
                adc0=_range_text(report.ranges, "adc_i0_raw"),
                adc1=_range_text(report.ranges, "adc_i1_raw"),
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
                "iq={iq} id={id} iux={iux} iwy={iwy} "
                "adc0_raw={adc0} adc1_raw={adc1} "
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
                    iq=_optional_range_text(segment.foc_iq_range),
                    id=_optional_range_text(segment.foc_id_range),
                    iux=_optional_range_text(segment.foc_iux_range),
                    iwy=_optional_range_text(segment.foc_iwy_range),
                    adc0=_optional_range_text(segment.adc_i0_raw_range),
                    adc1=_optional_range_text(segment.adc_i1_raw_range),
                    accel_lo=segment.accel_feedforward_offset_range[0],
                    accel_hi=segment.accel_feedforward_offset_range[1],
                    gain_lo=segment.accel_feedforward_gain_range[0],
                    gain_hi=segment.accel_feedforward_gain_range[1],
                    flags=segment.accel_feedforward_flags,
                )
            )
            authority = segment.current_authority
            if authority is not None:
                lines.append(
                    "    authority near_limit={near} sign={sign}/{near} "
                    "actual/cmd={actual_cmd} target/cmd={target_cmd} "
                    "actual/target={actual_target} uq_lim={uq_lim} "
                    "uq_delta={uq_delta} uq/limit={uq_limit} rail={rail}/{near} "
                    "iq/actual={iq_actual} adc0_delta={adc0} adc1_delta={adc1}".format(
                        near=authority.near_limit_samples,
                        sign=authority.sign_agreement_samples,
                        actual_cmd=_optional_ratio_text(
                            authority.actual_to_command_abs_median
                        ),
                        target_cmd=_optional_ratio_text(
                            authority.target_to_command_abs_median
                        ),
                        actual_target=_optional_ratio_text(
                            authority.actual_to_target_abs_median
                        ),
                        uq_lim=_optional_range_text(authority.uq_limited_range),
                        uq_delta=_optional_range_text(authority.uq_delta_range),
                        uq_limit=_optional_ratio_text(authority.uq_to_limit_abs_median),
                        rail=authority.voltage_rail_samples,
                        iq_actual=_optional_ratio_text(authority.iq_to_actual_abs_median),
                        adc0=_optional_range_text(authority.adc_i0_delta_range),
                        adc1=_optional_range_text(authority.adc_i1_delta_range),
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
