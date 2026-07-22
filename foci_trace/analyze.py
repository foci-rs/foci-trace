"""Offline summary for FOCI USB trace capture files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import io
import json
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
from .commissioning import (
    CommissioningCaptureBuilder,
    CommissioningCaptureReport,
    commissioning_report_to_json,
    format_commissioning_report,
)
from .contour import (
    ContourSegmentReport,
    MotorSample,
    PairedMotorSample,
    contour_pair_groups,
    contour_segment_report,
    pair_motor_samples,
)
from .protocol import (
    EVENT_COMMISSION_OUTER_SAFETY_TIMING,
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    FRAME_TYPE_STATUS,
    FRAME_TYPE_VELOCITY_OBSERVATION,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID,
    VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID,
    DecodeError,
    decode_frame,
)
from .velocity_measurement import (
    VelocityMeasurementReport,
    VelocityObservation,
    analyze_velocity_observations,
    audit_stage_b_reproduction_pair,
    audit_velocity_sweep,
    velocity_report_to_json,
)
from .velocity_integral import (
    audit_integral_reproduction_pair,
    audit_velocity_integral,
)

DEFAULT_CLOCK_HZ = 84_000_000
VELOCITY_REFERENCE_MAX_LEVELS = 8
DEFAULT_MOTION_GAP_S = 0.08
DEFAULT_MOTION_MIN_STEPS = 1000
CRITICAL_GAP_SUSPECT_S = 0.005
CONTROL_DETAIL_GAP_SUSPECT_S = 0.020
CURRENT_PATH_GAP_SUSPECT_S = 0.050
PHASE_EDGE_WINDOW_S = 0.050
PHASE_POST_WINDOW_S = 0.500
PHASE_TARGET_MIN_ABS = 100
DEFAULT_DECOUPLING_F_PWM_HZ = 25_000.0
DEFAULT_DECOUPLING_POLE_PAIRS = 50
DEFAULT_DECOUPLING_POSITION_UNITS_PER_REV = 65_536
PHASE_ADVANCE_FLAG_NONZERO = 1 << 1
PHASE_ADVANCE_FLAG_CLAMPED = 1 << 2
PHASE_ADVANCE_FLAG_GATED = 1 << 4
PHASE_ADVANCE_FLAG_WRITE_FAILED = 1 << 5
OUTER_SAFETY_GAP_BOUND_US = 20_000
OUTER_SAFETY_GAP_REVIEW_US = 15_000


def _outer_safety_gap_bin(gap_us: int) -> int | None:
    if gap_us <= 0 or gap_us > OUTER_SAFETY_GAP_BOUND_US:
        return None
    return min((gap_us - 1) // 5_000, 3)


def audit_outer_safety_timing(
    summary: dict[str, object] | None,
    raw_elapsed_us: list[int],
    marker_args: list[int] | None,
) -> dict[str, object]:
    """Audit firmware timing evidence against the raw-covered observation subset."""
    if summary is None:
        mismatches = [] if marker_args is None else ["marker has no canonical timing record"]
        return {
            "summary_present": False,
            "historical_absence_allowed": marker_args is None,
            "matches_firmware": not mismatches,
            "mismatches": mismatches,
        }

    bins = [int(value) for value in summary["bin_counts"]]  # type: ignore[index]
    sample_count = int(summary["sample_count"])
    zero_count = int(summary["zero_count"])
    over_bound_count = int(summary["over_bound_count"])
    max_gap_us = int(summary["max_realized_gap_us"])
    identity_count = sum(bins) + zero_count + over_bound_count
    mismatches: list[str] = []
    if sample_count != identity_count:
        mismatches.append("sample count differs from firmware bins and fault counts")

    raw_bins = [0, 0, 0, 0]
    raw_zero = 0
    raw_over = 0
    for gap_us in raw_elapsed_us:
        bin_index = _outer_safety_gap_bin(gap_us)
        if gap_us == 0:
            raw_zero += 1
        elif bin_index is None:
            raw_over += 1
        else:
            raw_bins[bin_index] += 1
    if any(raw > firmware for raw, firmware in zip(raw_bins, bins, strict=True)):
        mismatches.append("raw-covered bin count exceeds firmware summary")
    if raw_zero > zero_count or raw_over > over_bound_count:
        mismatches.append("raw-covered fault count exceeds firmware summary")
    if raw_elapsed_us and max(raw_elapsed_us) > max_gap_us:
        mismatches.append("raw-covered maximum exceeds firmware maximum")
    if len(raw_elapsed_us) > sample_count:
        mismatches.append("raw-covered sample count exceeds firmware summary")

    if marker_args is None:
        mismatches.append("canonical timing record has no event marker")
    elif marker_args != [int(summary["engine_sequence"]), max_gap_us]:
        mismatches.append("timing marker differs from canonical record")

    return {
        "summary_present": True,
        "historical_absence_allowed": False,
        "firmware": dict(summary),
        "gap_headroom_us": OUTER_SAFETY_GAP_BOUND_US - max_gap_us,
        "reopen_bound": max_gap_us > OUTER_SAFETY_GAP_REVIEW_US,
        "raw_covered_sample_count": len(raw_elapsed_us),
        "raw_covered_bin_counts": raw_bins,
        "raw_covered_zero_count": raw_zero,
        "raw_covered_over_bound_count": raw_over,
        "raw_covered_max_gap_us": max(raw_elapsed_us, default=0),
        "firmware_reported_safety_only": max(0, sample_count - len(raw_elapsed_us)),
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
    }


@dataclass
class CurrentAuthorityReport:
    """Near-limit current-authority summary for one motion segment."""

    near_limit_samples: int
    near_limit_sample_coverage: float
    near_limit_longest_run_s: float
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


@dataclass(frozen=True)
class DecouplingConfig:
    """Count-space motor model used for the offline decoupling audit."""

    l_int: int
    f_pwm_hz: float = DEFAULT_DECOUPLING_F_PWM_HZ
    pole_pairs: int = DEFAULT_DECOUPLING_POLE_PAIRS
    position_units_per_rev: int = DEFAULT_DECOUPLING_POSITION_UNITS_PER_REV


@dataclass
class DecouplingPhaseReport:
    """Estimated cross-coupling term size for one motion phase."""

    samples: int
    pole_pairs: int
    l_int: int
    f_pwm_hz: float
    position_units_per_rev: int
    omega_e_range_rad_s: tuple[float, float]
    vd_cross_range: tuple[float, float]
    vq_cross_range: tuple[float, float]
    cross_to_requested_median: float | None
    cross_to_requested_p95: float | None
    cross_to_limited_median: float | None
    cross_to_limited_p95: float | None


@dataclass
class PhaseAdvanceReport:
    """Runtime phase-advance summary for one motion phase."""

    samples: int
    counts_range: tuple[int, int]
    nonzero_samples: int
    clamped_samples: int
    gated_samples: int
    write_failed_samples: int
    gain_ppm_range: tuple[int, int]
    max_counts_range: tuple[int, int]
    deadband_range: tuple[int, int]


@dataclass
class MotionPhaseReport:
    """Phase-focused summary for one motion segment bucket."""

    name: str
    critical_samples: int
    fitted_error_rms_steps: float
    fitted_error_peak_steps: float
    fitted_error_median_steps: float
    fitted_error_final_steps: float
    actual_torque_range: tuple[int, int]
    torque_target_range: tuple[int, int]
    actual_to_target_abs_median: float | None
    uq_to_limit_abs_median: float | None
    uq_to_limit_abs_peak: float | None
    voltage_rail_samples: int
    accel_feedforward_offset_range: tuple[int, int]
    accel_feedforward_nonzero_samples: int
    accel_feedforward_sign_agreement_samples: int
    accel_feedforward_sign_comparable_samples: int
    velocity_feedforward_offset_range: tuple[int, int]
    velocity_feedforward_nonzero_samples: int
    velocity_feedforward_sign_agreement_samples: int
    velocity_feedforward_sign_comparable_samples: int
    velocity_transient_offset_range: tuple[int, int]
    velocity_transient_nonzero_samples: int
    velocity_transient_sign_agreement_samples: int
    velocity_transient_sign_comparable_samples: int
    target_velocity_range: tuple[int, int]
    velocity_actual_range: tuple[int, int] | None
    velocity_effective_target_range: tuple[int, int] | None
    velocity_error_rms: float | None
    velocity_error_peak: float | None
    velocity_actual_to_effective_target_abs_median: float | None
    velocity_pidin_target_range: tuple[int, int] | None
    velocity_pidout_target_range: tuple[int, int] | None
    iq_to_actual_abs_median: float | None
    uq_limited_range: tuple[int, int]
    phase_advance: PhaseAdvanceReport | None = None
    decoupling: DecouplingPhaseReport | None = None
    post_discriminator: str | None = None


@dataclass
class MotionSegmentReport:
    """Motion-focused summary for one contiguous stepper segment."""

    segment_index: int
    source_key: tuple[int, int]
    start_s: float
    duration_s: float
    step_delta: int
    critical_samples: int
    control_detail_samples: int
    critical_max_gap_s: float
    control_detail_max_gap_s: float | None
    current_path_max_gap_s: float | None
    trace_gap_suspect: bool
    fitted_error_rms_steps: float
    fitted_error_peak_steps: float
    actual_torque_range: tuple[int, int]
    torque_target_range: tuple[int, int]
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
    phase_reports: list[MotionPhaseReport] = field(default_factory=list)


@dataclass
class DeviceReport:
    """Decoded summary for one trace device."""

    device_index: int
    usb_path: str = ""
    product: str = ""
    serial: str = ""
    metadata_serial: str = ""
    source_labels: list[str] = field(default_factory=list)
    critical_source_keys: list[tuple[int, int]] = field(default_factory=list)
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


@dataclass
class CaptureAnalysis:
    """Complete analyzer result, including optional cross-device contour data."""

    devices: list[DeviceReport]
    commissioning: CommissioningCaptureReport = field(
        default_factory=lambda: CommissioningCaptureReport([])
    )
    contour_segments: list[ContourSegmentReport] = field(default_factory=list)
    velocity_reports: list[VelocityCaptureGroup] = field(default_factory=list)
    velocity_sweep_audits: list[dict[str, object]] = field(default_factory=list)
    velocity_integral_audits: list[dict[str, object]] = field(default_factory=list)
    outer_safety_timing_audits: list[dict[str, object]] = field(default_factory=list)
    corexy_plus: str | None = None
    corexy_minus: str | None = None


@dataclass(frozen=True)
class _CriticalObservation:
    timestamp_ticks: int
    source_key: tuple[int, int]
    host_recv_ns: int
    step_position: int
    tmc_position_actual: int
    torque_target: int
    actual_torque: int
    ud_requested: int
    uq_requested: int
    ud_limited: int
    uq_limited: int
    accel_feedforward_offset: int
    accel_feedforward_gain: int
    accel_feedforward_flags: int
    foc_iq: int | None
    foc_id: int | None
    target_velocity: int
    velocity_actual: int
    velocity_effective_target: int
    velocity_error: int
    cached_velocity_offset: int
    cached_velocity_transient_offset: int


@dataclass(frozen=True)
class _CurrentPathObservation:
    timestamp_ticks: int
    source_key: tuple[int, int]
    foc_iq: int
    foc_id: int
    foc_iux: int
    foc_iwy: int
    adc_i0_raw: int
    adc_i1_raw: int


@dataclass(frozen=True)
class _PhaseAdvanceObservation:
    timestamp_ticks: int
    source_key: tuple[int, int]
    counts: int
    flags: int
    gain_ppm: int
    max_counts: int
    deadband: int


@dataclass(frozen=True)
class _ControlDetailObservation:
    timestamp_ticks: int
    source_key: tuple[int, int]
    torque_command: int
    velocity_pidin_target: int
    velocity_pidout_target: int
    voltage_limit: int
    torque_limit: int


@dataclass(frozen=True)
class _VelocityFrameObservation:
    timestamp_ticks: int
    source_index: int
    observation: VelocityObservation
    flags: int


@dataclass(frozen=True)
class _VelocitySweepEvidenceFrame:
    timestamp_ticks: int
    source_index: int
    raw_payload: bytes
    payload: dict[str, object]


@dataclass(frozen=True)
class VelocityCaptureGroup:
    """Velocity observations correlated to one device, source, and engine run."""

    device_index: int
    source_index: int
    serial: str
    product: str
    engine_kind: str
    engine_seq: int | None
    result: str | None
    start_ticks: int | None
    terminal_ticks: int | None
    candidate_summary_seq: int | None
    velocity_p: int | None
    velocity_i: int | None
    saturated_observations: int
    missing_frames: int
    sequence_gaps: dict[int, int]
    trace_drops: int
    caveats: list[str]
    report: VelocityMeasurementReport


@dataclass(frozen=True)
class _FittedTransform:
    slope: float
    intercept: float


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
    phase_advance_observations: list[_PhaseAdvanceObservation] = field(default_factory=list)
    control_detail_observations: list[_ControlDetailObservation] = field(default_factory=list)
    velocity_observations: list[_VelocityFrameObservation] = field(default_factory=list)
    velocity_sweep_evidence: list[_VelocitySweepEvidenceFrame] = field(default_factory=list)
    outer_safety_timing_markers: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class _MotionSegmentSpan:
    start_host_ns: int
    end_host_ns: int
    report: MotionSegmentReport


@dataclass(frozen=True)
class _MotionGroup:
    source_key: tuple[int, int]
    observations: list[_CriticalObservation]
    start_index: int
    end_index: int
    next_start_tick: int | None


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
    source_key: tuple[int, int],
    host_recv_ns: int,
    steady_after_s: float,
) -> None:
    if source_key not in state.report.critical_source_keys:
        state.report.critical_source_keys.append(source_key)
    actual_flux: int | None = None
    torque_target: int | None = None
    foc_iq: int | None = None
    foc_id: int | None = None
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
            source_key,
            steady_after_s,
            foc_iq,
            foc_id,
            foc_iux,
            foc_iwy,
            adc_i0_raw,
            adc_i1_raw,
        )
    velocity_actual = int(payload["pid_velocity_actual"])
    velocity_error = int(payload["velocity_error"])
    state.critical_observations.append(
        _CriticalObservation(
            timestamp_ticks=timestamp_ticks,
            source_key=source_key,
            host_recv_ns=host_recv_ns,
            step_position=int(payload["step_position"]),
            tmc_position_actual=int(payload["tmc_position_actual"]),
            torque_target=torque_target if torque_target is not None else 0,
            actual_torque=actual_torque,
            ud_requested=ud,
            uq_requested=uq,
            ud_limited=ud_limited,
            uq_limited=uq_limited,
            accel_feedforward_offset=int(payload["cached_accel_torque_offset"]),
            accel_feedforward_gain=int(payload["accel_feedforward_gain_permille"]),
            accel_feedforward_flags=int(payload["accel_feedforward_flags"]),
            foc_iq=foc_iq,
            foc_id=foc_id,
            target_velocity=int(payload["target_velocity"]),
            velocity_actual=velocity_actual,
            velocity_effective_target=velocity_actual + velocity_error,
            velocity_error=velocity_error,
            cached_velocity_offset=int(payload["cached_pid_velocity_offset"]),
            cached_velocity_transient_offset=int(
                payload.get("cached_velocity_transient_offset", 0)
            ),
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
    source_key: tuple[int, int],
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
            source_key=source_key,
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
    source_key: tuple[int, int],
    steady_after_s: float,
) -> None:
    foc_id, foc_iq = _split_i16_pair(int(payload["interim_foc_iq_id"]))
    foc_iux, foc_iwy = _split_i16_pair(int(payload["interim_foc_iwy_iux"]))
    adc_i0_raw, adc_i1_raw = _split_u16_pair(int(payload["interim_adc_i1_i0"]))
    _observe_current_path_values(
        state,
        timestamp_ticks,
        source_key,
        steady_after_s,
        foc_iq,
        foc_id,
        foc_iux,
        foc_iwy,
        adc_i0_raw,
        adc_i1_raw,
    )


def _observe_phase_advance(
    state: _DeviceState,
    payload: dict[str, object],
    timestamp_ticks: int,
    source_key: tuple[int, int],
) -> None:
    state.phase_advance_observations.append(
        _PhaseAdvanceObservation(
            timestamp_ticks=timestamp_ticks,
            source_key=source_key,
            counts=int(payload["phase_advance_counts"]),
            flags=int(payload["phase_advance_flags"]),
            gain_ppm=int(payload["phase_advance_gain_ppm"]),
            max_counts=int(payload["phase_advance_max_counts"]),
            deadband=int(payload["phase_advance_deadband"]),
        )
    )


def _observe_velocity_observation(
    state: _DeviceState,
    payload: dict[str, object],
    timestamp_ticks: int,
    source_index: int,
) -> None:
    state.velocity_observations.append(
        _VelocityFrameObservation(
            timestamp_ticks=timestamp_ticks,
            source_index=source_index,
            observation=VelocityObservation(
                delta_counts=int(payload["delta_counts"]),
                elapsed_us=int(payload["elapsed_us"]),
                direction=int(payload["direction"]),
                stroke_index=int(payload["stroke_index"]),
                target_cpms=int(payload["target_counts_per_ms"]),
                peak_torque_target_abs=(
                    None
                    if payload.get("peak_torque_target_abs") is None
                    else int(payload["peak_torque_target_abs"])
                ),
                pid_velocity_actual=(
                    int(payload["pid_velocity_actual"])
                    if int(payload["flags"]) & VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID
                    and payload.get("pid_velocity_actual") is not None
                    else None
                ),
                actual_velocity_pptm=(
                    int(payload["actual_velocity_pptm"])
                    if int(payload["flags"]) & VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID
                    and payload.get("actual_velocity_pptm") is not None
                    else None
                ),
            ),
            flags=int(payload["flags"]),
        )
    )


def _observe_control_detail(
    state: _DeviceState,
    payload: dict[str, object],
    timestamp_ticks: int,
    source_key: tuple[int, int],
) -> None:
    voltage_limits = _split_i16_pair(int(payload["pidout_uq_ud_limits"]))
    torque_limits = _split_i16_pair(int(payload["pid_torque_flux_limits"]))
    state.control_detail_observations.append(
        _ControlDetailObservation(
            timestamp_ticks=timestamp_ticks,
            source_key=source_key,
            torque_command=_signed_i32(int(payload["interim_pidout_target_torque"])),
            velocity_pidin_target=int(payload["interim_pidin_target_velocity"]),
            velocity_pidout_target=int(payload["interim_pidout_target_velocity"]),
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


def _motion_groups_by_source(
    observations: list[_CriticalObservation],
    clock_hz: int,
    gap_s: float,
    min_steps: int,
) -> list[tuple[tuple[int, int], list[_CriticalObservation], list[tuple[int, int]]]]:
    observations_by_source: dict[tuple[int, int], list[_CriticalObservation]] = {}
    for observation in observations:
        observations_by_source.setdefault(observation.source_key, []).append(observation)
    source_groups = []
    for source_key in sorted(observations_by_source):
        source_observations = sorted(
            observations_by_source[source_key], key=lambda sample: sample.timestamp_ticks
        )
        source_groups.append(
            (
                source_key,
                source_observations,
                _motion_groups(source_observations, clock_hz, gap_s, min_steps),
            )
        )
    return source_groups


def _ordered_motion_groups(
    observations: list[_CriticalObservation],
    clock_hz: int,
    gap_s: float,
    min_steps: int,
) -> list[_MotionGroup]:
    motion_groups: list[_MotionGroup] = []
    for source_key, source_observations, groups in _motion_groups_by_source(
        observations, clock_hz, gap_s, min_steps
    ):
        for group_index, (start_index, end_index) in enumerate(groups):
            next_start_tick = (
                source_observations[groups[group_index + 1][0]].timestamp_ticks
                if group_index + 1 < len(groups)
                else None
            )
            motion_groups.append(
                _MotionGroup(
                    source_key=source_key,
                    observations=source_observations,
                    start_index=start_index,
                    end_index=end_index,
                    next_start_tick=next_start_tick,
                )
            )
    return sorted(
        motion_groups,
        key=lambda group: (
            group.observations[group.start_index].timestamp_ticks,
            group.source_key,
        ),
    )


def _fitted_transform(samples: list[_CriticalObservation]) -> _FittedTransform | None:
    if len(samples) < 2:
        return None
    step_values = [sample.step_position for sample in samples]
    actual_values = [sample.tmc_position_actual for sample in samples]
    mean_step = sum(step_values) / len(step_values)
    mean_actual = sum(actual_values) / len(actual_values)
    variance = sum((step - mean_step) ** 2 for step in step_values)
    if variance == 0:
        return None
    covariance = sum(
        (step - mean_step) * (actual - mean_actual)
        for step, actual in zip(step_values, actual_values, strict=True)
    )
    slope = covariance / variance
    if abs(slope) < 1e-9:
        return None
    intercept = mean_actual - slope * mean_step
    return _FittedTransform(slope=slope, intercept=intercept)


def _fitted_residual(
    sample: _CriticalObservation, transform: _FittedTransform, command_position: int | None = None
) -> float:
    commanded = sample.step_position if command_position is None else command_position
    return commanded - ((sample.tmc_position_actual - transform.intercept) / transform.slope)


def _residual_stats(residuals: list[float]) -> tuple[float, float]:
    if not residuals:
        return 0.0, 0.0
    rms = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    peak = max(abs(residual) for residual in residuals)
    return rms, peak


def _fitted_error_stats(samples: list[_CriticalObservation]) -> tuple[float, float]:
    transform = _fitted_transform(samples)
    if transform is None:
        return 0.0, 0.0
    residuals = [_fitted_residual(sample, transform) for sample in samples]
    return _residual_stats(residuals)


def _range(values: list[int]) -> tuple[int, int]:
    return min(values), max(values)


def _optional_range(values: list[int | None]) -> tuple[int, int] | None:
    present = [value for value in values if value is not None]
    return _range(present) if present else None


def _max_timestamp_gap_s(timestamps: list[int], clock_hz: int) -> float | None:
    if len(timestamps) < 2:
        return None
    sorted_timestamps = sorted(timestamps)
    return max(
        (sorted_timestamps[idx] - sorted_timestamps[idx - 1]) / clock_hz
        for idx in range(1, len(sorted_timestamps))
    )


def _nearest_critical(
    samples: list[_CriticalObservation], timestamp_ticks: int
) -> _CriticalObservation:
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _nearest_current_path(
    samples: list[_CurrentPathObservation], sample: _CriticalObservation
) -> _CurrentPathObservation | None:
    source_samples = [
        current_path for current_path in samples if current_path.source_key == sample.source_key
    ]
    if not source_samples:
        return None
    return min(
        source_samples,
        key=lambda current_path: abs(current_path.timestamp_ticks - sample.timestamp_ticks),
    )


def _foc_iq_id_for_sample(
    sample: _CriticalObservation,
    current_paths: list[_CurrentPathObservation],
) -> tuple[int, int] | None:
    if sample.foc_iq is not None and sample.foc_id is not None:
        return sample.foc_iq, sample.foc_id
    current_path = _nearest_current_path(current_paths, sample)
    if current_path is None:
        return None
    return current_path.foc_iq, current_path.foc_id


def _nearest_control_detail(
    samples: list[_ControlDetailObservation], timestamp_ticks: int
) -> _ControlDetailObservation | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.timestamp_ticks - timestamp_ticks))


def _sign_matches(left: int, right: int) -> bool:
    return (left < 0 and right < 0) or (left > 0 and right > 0) or (left == 0 and right == 0)


def _same_nonzero_sign(left: int, right: int) -> bool:
    return (left < 0 and right < 0) or (left > 0 and right > 0)


def _post_move_discriminator(samples: list[_CriticalObservation]) -> str | None:
    if not samples:
        return None
    active_torque = [
        sample
        for sample in samples
        if abs(sample.actual_torque) >= PHASE_TARGET_MIN_ABS or abs(sample.uq_limited) >= 500
    ]
    if active_torque:
        return "loop_active"
    return "settling_possible"


def _select_state_by_serial(states: dict[int, _DeviceState], selector: str) -> _DeviceState:
    matches = [
        state
        for index, state in states.items()
        if selector in (f"dev{index}", state.report.serial, state.report.metadata_serial)
    ]
    if not matches:
        raise ValueError(f"no trace device matches {selector!r}")
    if len(matches) > 1:
        raise ValueError(f"multiple trace devices match {selector!r}")
    return matches[0]


def _motor_samples(state: _DeviceState) -> list[MotorSample]:
    return [
        MotorSample(
            host_recv_ns=sample.host_recv_ns,
            commanded=float(sample.step_position),
            actual=float(sample.tmc_position_actual),
        )
        for sample in sorted(state.critical_observations, key=lambda sample: sample.host_recv_ns)
    ]


def _motion_segment_spans(
    state: _DeviceState,
    motion_gap_s: float,
    motion_min_steps: int,
) -> list[_MotionSegmentSpan]:
    spans: list[_MotionSegmentSpan] = []
    motion_groups = _ordered_motion_groups(
        state.critical_observations, state.clock_hz, motion_gap_s, motion_min_steps
    )
    for motion_group, report in zip(motion_groups, state.report.motion_segments, strict=True):
        assert motion_group.source_key == report.source_key
        samples = motion_group.observations[motion_group.start_index : motion_group.end_index + 1]
        host_values = [sample.host_recv_ns for sample in samples]
        spans.append(
            _MotionSegmentSpan(
                start_host_ns=min(host_values),
                end_host_ns=max(host_values),
                report=report,
            )
        )
    return spans


def _pair_group_host_span(pairs: list[PairedMotorSample]) -> tuple[int, int]:
    host_values = [
        host_recv_ns
        for pair in pairs
        for host_recv_ns in (pair.plus.host_recv_ns, pair.minus.host_recv_ns)
    ]
    return min(host_values), max(host_values)


def _contour_group_segments(
    pairs: list[PairedMotorSample],
    plus_spans: list[_MotionSegmentSpan],
    minus_spans: list[_MotionSegmentSpan],
) -> list[MotionSegmentReport]:
    group_start_ns, group_end_ns = _pair_group_host_span(pairs)
    return [
        span.report
        for span in plus_spans + minus_spans
        if span.start_host_ns <= group_end_ns and span.end_host_ns >= group_start_ns
    ]


def _contour_post_discriminator(segments: list[MotionSegmentReport]) -> str | None:
    discriminators = [
        phase.post_discriminator
        for segment in segments
        for phase in segment.phase_reports
        if phase.post_discriminator is not None
    ]
    if "loop_active" in discriminators:
        return "loop_active"
    if "settling_possible" in discriminators:
        return "settling_possible"
    return None


def _contour_position_trace_gap(segments: list[MotionSegmentReport]) -> bool:
    return any(segment.critical_max_gap_s > CRITICAL_GAP_SUSPECT_S for segment in segments)


def _contour_support_trace_gap(segments: list[MotionSegmentReport]) -> bool:
    return any(
        (
            segment.control_detail_max_gap_s is not None
            and segment.control_detail_max_gap_s > CONTROL_DETAIL_GAP_SUSPECT_S
        )
        or (
            segment.current_path_max_gap_s is not None
            and segment.current_path_max_gap_s > CURRENT_PATH_GAP_SUSPECT_S
        )
        for segment in segments
    )


def _rounded_median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 3)


def _rounded_percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    return int(round(median(values)))


def _current_authority_report(
    segment_samples: list[_CriticalObservation],
    segment_details: list[_ControlDetailObservation],
    segment_current_paths: list[_CurrentPathObservation],
    clock_hz: int,
) -> CurrentAuthorityReport | None:
    torque_limit = max((detail.torque_limit for detail in segment_details), default=0)
    if torque_limit <= 0:
        return None
    threshold = int(torque_limit * 0.8)
    near_limit_samples: list[tuple[_CriticalObservation, _ControlDetailObservation | None]] = []
    for sample in segment_samples:
        if abs(sample.torque_target) < threshold:
            continue
        detail = _nearest_control_detail(segment_details, sample.timestamp_ticks)
        near_limit_samples.append((sample, detail))
    if not near_limit_samples:
        return None
    near_limit_timestamps = {sample.timestamp_ticks for sample, _detail in near_limit_samples}
    near_limit_sample_coverage = len(near_limit_samples) / len(segment_samples)
    near_limit_longest_run_s = _near_limit_longest_run_s(
        segment_samples, near_limit_timestamps, clock_hz
    )

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

    near_limit_keys = {
        (sample.source_key, sample.timestamp_ticks) for sample, _detail in near_limit_samples
    }
    for current_path in segment_current_paths:
        source_samples = [
            sample for sample in segment_samples if sample.source_key == current_path.source_key
        ]
        if not source_samples:
            continue
        critical = _nearest_critical(source_samples, current_path.timestamp_ticks)
        if (critical.source_key, critical.timestamp_ticks) not in near_limit_keys:
            continue
        if critical.actual_torque != 0:
            iq_to_actual.append(abs(current_path.foc_iq) / abs(critical.actual_torque))
        if adc_i0_baseline is not None:
            adc_i0_deltas.append(current_path.adc_i0_raw - adc_i0_baseline)
        if adc_i1_baseline is not None:
            adc_i1_deltas.append(current_path.adc_i1_raw - adc_i1_baseline)

    return CurrentAuthorityReport(
        near_limit_samples=len(near_limit_samples),
        near_limit_sample_coverage=near_limit_sample_coverage,
        near_limit_longest_run_s=near_limit_longest_run_s,
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


def _near_limit_longest_run_s(
    segment_samples: list[_CriticalObservation],
    near_limit_timestamps: set[int],
    clock_hz: int,
) -> float:
    """Return the longest consecutive near-limit run in seconds."""
    if clock_hz <= 0:
        return 0.0
    longest_start: int | None = None
    longest_end: int | None = None
    current_start: int | None = None
    current_end: int | None = None
    for sample in segment_samples:
        if sample.timestamp_ticks in near_limit_timestamps:
            if current_start is None:
                current_start = sample.timestamp_ticks
            current_end = sample.timestamp_ticks
            continue
        if current_start is not None and current_end is not None:
            if (
                longest_start is None
                or current_end - current_start > (longest_end or 0) - longest_start
            ):
                longest_start = current_start
                longest_end = current_end
        current_start = None
        current_end = None
    if current_start is not None and current_end is not None:
        if (
            longest_start is None
            or current_end - current_start > (longest_end or 0) - longest_start
        ):
            longest_start = current_start
            longest_end = current_end
    if longest_start is None or longest_end is None:
        return 0.0
    return (longest_end - longest_start) / clock_hz


def _phase_advance_report(
    observations: list[_PhaseAdvanceObservation],
) -> PhaseAdvanceReport | None:
    if not observations:
        return None
    flags = [sample.flags for sample in observations]
    return PhaseAdvanceReport(
        samples=len(observations),
        counts_range=_range([sample.counts for sample in observations]),
        nonzero_samples=sum(bool(flag & PHASE_ADVANCE_FLAG_NONZERO) for flag in flags),
        clamped_samples=sum(bool(flag & PHASE_ADVANCE_FLAG_CLAMPED) for flag in flags),
        gated_samples=sum(bool(flag & PHASE_ADVANCE_FLAG_GATED) for flag in flags),
        write_failed_samples=sum(bool(flag & PHASE_ADVANCE_FLAG_WRITE_FAILED) for flag in flags),
        gain_ppm_range=_range([sample.gain_ppm for sample in observations]),
        max_counts_range=_range([sample.max_counts for sample in observations]),
        deadband_range=_range([sample.deadband for sample in observations]),
    )


def _decoupling_phase_report(
    samples: list[_CriticalObservation],
    phase_current_paths: list[_CurrentPathObservation],
    config: DecouplingConfig | None,
) -> DecouplingPhaseReport | None:
    if (
        config is None
        or config.l_int <= 0
        or config.f_pwm_hz <= 0.0
        or config.pole_pairs <= 0
        or config.position_units_per_rev <= 0
    ):
        return None

    l_count_seconds = config.l_int * 1e-6
    omega_values: list[float] = []
    vd_values: list[float] = []
    vq_values: list[float] = []
    requested_ratios: list[float] = []
    limited_ratios: list[float] = []

    for sample in samples:
        if sample.target_velocity == 0:
            continue
        foc_values = _foc_iq_id_for_sample(sample, phase_current_paths)
        if foc_values is None:
            continue
        foc_iq, foc_id = foc_values
        omega_e = (
            abs(sample.target_velocity)
            * config.f_pwm_hz
            / config.position_units_per_rev
            * config.pole_pairs
            * 2.0
            * math.pi
        )
        if omega_e == 0.0:
            continue
        vq_cross = omega_e * l_count_seconds * foc_id
        vd_cross = -omega_e * l_count_seconds * foc_iq
        cross_magnitude = math.hypot(vd_cross, vq_cross)
        requested_magnitude = math.hypot(sample.ud_requested, sample.uq_requested)
        limited_magnitude = math.hypot(sample.ud_limited, sample.uq_limited)
        if requested_magnitude > 0.0:
            requested_ratios.append(cross_magnitude / requested_magnitude)
        if limited_magnitude > 0.0:
            limited_ratios.append(cross_magnitude / limited_magnitude)
        omega_values.append(omega_e)
        vd_values.append(vd_cross)
        vq_values.append(vq_cross)

    if not omega_values:
        return None
    return DecouplingPhaseReport(
        samples=len(omega_values),
        pole_pairs=config.pole_pairs,
        l_int=config.l_int,
        f_pwm_hz=config.f_pwm_hz,
        position_units_per_rev=config.position_units_per_rev,
        omega_e_range_rad_s=(min(omega_values), max(omega_values)),
        vd_cross_range=(min(vd_values), max(vd_values)),
        vq_cross_range=(min(vq_values), max(vq_values)),
        cross_to_requested_median=_rounded_median(requested_ratios),
        cross_to_requested_p95=_rounded_percentile(requested_ratios, 0.95),
        cross_to_limited_median=_rounded_median(limited_ratios),
        cross_to_limited_p95=_rounded_percentile(limited_ratios, 0.95),
    )


def _motion_phase_reports(
    observations: list[_CriticalObservation],
    current_paths: list[_CurrentPathObservation],
    phase_advances: list[_PhaseAdvanceObservation],
    control_details: list[_ControlDetailObservation],
    segment_samples: list[_CriticalObservation],
    transform: _FittedTransform | None,
    clock_hz: int,
    voltage_limit: int,
    next_segment_start_tick: int | None,
    velocity_tracking_available: bool,
    decoupling_config: DecouplingConfig | None,
) -> list[MotionPhaseReport]:
    if transform is None or clock_hz <= 0:
        return []
    start_tick = segment_samples[0].timestamp_ticks
    end_tick = segment_samples[-1].timestamp_ticks
    edge_ticks = int(PHASE_EDGE_WINDOW_S * clock_hz)
    post_ticks = int(PHASE_POST_WINDOW_S * clock_hz)
    split_specs = [
        (
            "entry_0_50ms",
            segment_samples,
            start_tick,
            min(end_tick, start_tick + edge_ticks),
            None,
            False,
        ),
        (
            "cruise_mid",
            segment_samples,
            min(end_tick, start_tick + edge_ticks),
            max(start_tick, end_tick - edge_ticks),
            None,
            False,
        ),
        (
            "exit_last50ms",
            segment_samples,
            max(start_tick, end_tick - edge_ticks),
            end_tick,
            None,
            True,
        ),
    ]
    post_end_tick = end_tick + post_ticks
    if next_segment_start_tick is not None:
        post_end_tick = min(post_end_tick, next_segment_start_tick - 1)
    split_specs.append(
        (
            "post_0_500ms",
            observations,
            end_tick + 1,
            post_end_tick,
            segment_samples[-1].step_position,
            True,
        )
    )

    reports: list[MotionPhaseReport] = []
    for name, source_samples, start, end, command_position, include_end in split_specs:
        if end < start:
            continue
        if include_end:
            samples = [
                sample for sample in source_samples if start <= sample.timestamp_ticks <= end
            ]
            phase_current_paths = [
                sample for sample in current_paths if start <= sample.timestamp_ticks <= end
            ]
        else:
            samples = [sample for sample in source_samples if start <= sample.timestamp_ticks < end]
            phase_current_paths = [
                sample for sample in current_paths if start <= sample.timestamp_ticks < end
            ]
        if not samples:
            continue
        sample_start = samples[0].timestamp_ticks
        sample_end = samples[-1].timestamp_ticks
        sample_sources = {sample.source_key for sample in samples}
        phase_current_paths = [
            sample for sample in phase_current_paths if sample.source_key in sample_sources
        ]
        phase_advance_samples = [
            sample
            for sample in phase_advances
            if sample_start <= sample.timestamp_ticks <= sample_end
            and sample.source_key in sample_sources
        ]
        phase_control_details = [
            sample
            for sample in control_details
            if sample_start <= sample.timestamp_ticks <= sample_end
            and sample.source_key in sample_sources
        ]
        residuals = [_fitted_residual(sample, transform, command_position) for sample in samples]
        rms, peak = _residual_stats(residuals)
        actual_to_target = [
            abs(sample.actual_torque) / abs(sample.torque_target)
            for sample in samples
            if abs(sample.torque_target) >= PHASE_TARGET_MIN_ABS
        ]
        uq_to_limit = [
            abs(sample.uq_limited) / voltage_limit for sample in samples if voltage_limit > 0
        ]
        voltage_threshold = int(voltage_limit * 0.98)
        voltage_rail_samples = (
            sum(abs(sample.uq_limited) >= voltage_threshold for sample in samples)
            if voltage_threshold > 0
            else 0
        )
        accel_offsets = [sample.accel_feedforward_offset for sample in samples]
        nonzero_offsets = [offset for offset in accel_offsets if offset != 0]
        accel_sign_candidates = [
            sample
            for sample in samples
            if sample.accel_feedforward_offset != 0 and sample.torque_target != 0
        ]
        accel_sign_agreement = sum(
            _same_nonzero_sign(sample.accel_feedforward_offset, sample.torque_target)
            for sample in accel_sign_candidates
        )
        velocity_offsets = [sample.cached_velocity_offset for sample in samples]
        nonzero_velocity_offsets = [offset for offset in velocity_offsets if offset != 0]
        velocity_sign_candidates = [
            sample
            for sample in samples
            if sample.cached_velocity_offset != 0 and sample.target_velocity != 0
        ]
        velocity_sign_agreement = sum(
            _same_nonzero_sign(sample.cached_velocity_offset, sample.target_velocity)
            for sample in velocity_sign_candidates
        )
        velocity_transient_offsets = [sample.cached_velocity_transient_offset for sample in samples]
        nonzero_velocity_transient_offsets = [
            offset for offset in velocity_transient_offsets if offset != 0
        ]
        velocity_transient_sign_candidates = [
            sample
            for sample in samples
            if sample.cached_velocity_transient_offset != 0 and sample.torque_target != 0
        ]
        velocity_transient_sign_agreement = sum(
            _same_nonzero_sign(
                sample.cached_velocity_transient_offset,
                sample.torque_target,
            )
            for sample in velocity_transient_sign_candidates
        )
        velocity_errors = [sample.velocity_error for sample in samples]
        velocity_error_rms, velocity_error_peak = (
            _residual_stats(velocity_errors) if velocity_tracking_available else (None, None)
        )
        velocity_actual_to_effective_target = (
            [
                abs(sample.velocity_actual) / abs(sample.velocity_effective_target)
                for sample in samples
                if sample.velocity_effective_target != 0
            ]
            if velocity_tracking_available
            else []
        )
        iq_to_actual = [
            abs(sample.foc_iq) / abs(sample.actual_torque)
            for sample in samples
            if sample.foc_iq is not None and sample.actual_torque != 0
        ]
        if not iq_to_actual:
            for current_path in phase_current_paths:
                source_samples = [
                    sample for sample in samples if sample.source_key == current_path.source_key
                ]
                if not source_samples:
                    continue
                critical = _nearest_critical(source_samples, current_path.timestamp_ticks)
                if critical.actual_torque != 0:
                    iq_to_actual.append(abs(current_path.foc_iq) / abs(critical.actual_torque))
        phase_advance = _phase_advance_report(phase_advance_samples)
        decoupling = _decoupling_phase_report(samples, phase_current_paths, decoupling_config)
        post_discriminator = _post_move_discriminator(samples) if name == "post_0_500ms" else None
        reports.append(
            MotionPhaseReport(
                name=name,
                critical_samples=len(samples),
                fitted_error_rms_steps=rms,
                fitted_error_peak_steps=peak,
                fitted_error_median_steps=median(residuals),
                fitted_error_final_steps=residuals[-1],
                actual_torque_range=_range([sample.actual_torque for sample in samples]),
                torque_target_range=_range([sample.torque_target for sample in samples]),
                actual_to_target_abs_median=_rounded_median(actual_to_target),
                uq_to_limit_abs_median=_rounded_median(uq_to_limit),
                uq_to_limit_abs_peak=max(uq_to_limit) if uq_to_limit else None,
                voltage_rail_samples=voltage_rail_samples,
                accel_feedforward_offset_range=_range(accel_offsets),
                accel_feedforward_nonzero_samples=len(nonzero_offsets),
                accel_feedforward_sign_agreement_samples=accel_sign_agreement,
                accel_feedforward_sign_comparable_samples=len(accel_sign_candidates),
                velocity_feedforward_offset_range=_range(velocity_offsets),
                velocity_feedforward_nonzero_samples=len(nonzero_velocity_offsets),
                velocity_feedforward_sign_agreement_samples=velocity_sign_agreement,
                velocity_feedforward_sign_comparable_samples=len(velocity_sign_candidates),
                velocity_transient_offset_range=_range(velocity_transient_offsets),
                velocity_transient_nonzero_samples=len(nonzero_velocity_transient_offsets),
                velocity_transient_sign_agreement_samples=velocity_transient_sign_agreement,
                velocity_transient_sign_comparable_samples=len(velocity_transient_sign_candidates),
                target_velocity_range=_range([sample.target_velocity for sample in samples]),
                velocity_actual_range=(
                    _range([sample.velocity_actual for sample in samples])
                    if velocity_tracking_available
                    else None
                ),
                velocity_effective_target_range=(
                    _range([sample.velocity_effective_target for sample in samples])
                    if velocity_tracking_available
                    else None
                ),
                velocity_error_rms=velocity_error_rms,
                velocity_error_peak=velocity_error_peak,
                velocity_actual_to_effective_target_abs_median=_rounded_median(
                    velocity_actual_to_effective_target
                ),
                velocity_pidin_target_range=(
                    _optional_range(
                        [sample.velocity_pidin_target for sample in phase_control_details]
                    )
                    if velocity_tracking_available
                    else None
                ),
                velocity_pidout_target_range=(
                    _optional_range(
                        [sample.velocity_pidout_target for sample in phase_control_details]
                    )
                    if velocity_tracking_available
                    else None
                ),
                iq_to_actual_abs_median=_rounded_median(iq_to_actual),
                uq_limited_range=_range([sample.uq_limited for sample in samples]),
                phase_advance=phase_advance,
                decoupling=decoupling,
                post_discriminator=post_discriminator,
            )
        )
    return reports


def _finalize_motion_segments(
    state: _DeviceState,
    motion_gap_s: float,
    motion_min_steps: int,
    decoupling_config: DecouplingConfig | None,
) -> None:
    current_paths = sorted(
        state.current_path_observations, key=lambda sample: sample.timestamp_ticks
    )
    phase_advances = sorted(
        state.phase_advance_observations, key=lambda sample: sample.timestamp_ticks
    )
    details = sorted(state.control_detail_observations, key=lambda sample: sample.timestamp_ticks)
    motion_groups = _ordered_motion_groups(
        state.critical_observations, state.clock_hz, motion_gap_s, motion_min_steps
    )
    for segment_index, motion_group in enumerate(motion_groups):
        source_key = motion_group.source_key
        observations = motion_group.observations
        segment_samples = observations[motion_group.start_index : motion_group.end_index + 1]
        start_tick = segment_samples[0].timestamp_ticks
        end_tick = segment_samples[-1].timestamp_ticks
        segment_details = [
            detail
            for detail in details
            if start_tick <= detail.timestamp_ticks <= end_tick and detail.source_key == source_key
        ]
        step_delta = segment_samples[-1].step_position - segment_samples[0].step_position
        rms, peak = _fitted_error_stats(segment_samples)
        torque_values = [sample.actual_torque for sample in segment_samples]
        torque_target_values = [sample.torque_target for sample in segment_samples]
        uq_limited_values = [sample.uq_limited for sample in segment_samples]
        segment_current_paths = [
            sample
            for sample in current_paths
            if start_tick <= sample.timestamp_ticks <= end_tick and sample.source_key == source_key
        ]
        foc_iq_range = _optional_range([sample.foc_iq for sample in segment_current_paths])
        foc_id_range = _optional_range([sample.foc_id for sample in segment_current_paths])
        foc_iux_range = _optional_range([sample.foc_iux for sample in segment_current_paths])
        foc_iwy_range = _optional_range([sample.foc_iwy for sample in segment_current_paths])
        adc_i0_raw_range = _optional_range([sample.adc_i0_raw for sample in segment_current_paths])
        adc_i1_raw_range = _optional_range([sample.adc_i1_raw for sample in segment_current_paths])
        current_authority = _current_authority_report(
            segment_samples, segment_details, segment_current_paths, state.clock_hz
        )

        critical_max_gap_s = (
            _max_timestamp_gap_s(
                [sample.timestamp_ticks for sample in segment_samples], state.clock_hz
            )
            or 0.0
        )
        control_detail_max_gap_s = _max_timestamp_gap_s(
            [sample.timestamp_ticks for sample in segment_details], state.clock_hz
        )
        current_path_max_gap_s = _max_timestamp_gap_s(
            [sample.timestamp_ticks for sample in segment_current_paths], state.clock_hz
        )
        trace_gap_suspect = (
            critical_max_gap_s > CRITICAL_GAP_SUSPECT_S
            or (
                control_detail_max_gap_s is not None
                and control_detail_max_gap_s > CONTROL_DETAIL_GAP_SUSPECT_S
            )
            or (
                current_path_max_gap_s is not None
                and current_path_max_gap_s > CURRENT_PATH_GAP_SUSPECT_S
            )
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
        transform = _fitted_transform(segment_samples)
        velocity_tracking_available = any(
            sample.velocity_actual != 0 or sample.velocity_error != 0 for sample in segment_samples
        ) or any(
            detail.velocity_pidin_target != 0 or detail.velocity_pidout_target != 0
            for detail in segment_details
        )
        phase_reports = _motion_phase_reports(
            observations,
            current_paths,
            phase_advances,
            details,
            segment_samples,
            transform,
            state.clock_hz,
            voltage_limit,
            motion_group.next_start_tick,
            velocity_tracking_available,
            decoupling_config,
        )
        state.report.motion_segments.append(
            MotionSegmentReport(
                segment_index=segment_index,
                source_key=source_key,
                start_s=(start_tick - (state.first_timestamp or start_tick)) / state.clock_hz,
                duration_s=(end_tick - start_tick) / state.clock_hz,
                step_delta=step_delta,
                critical_samples=len(segment_samples),
                control_detail_samples=len(segment_details),
                critical_max_gap_s=critical_max_gap_s,
                control_detail_max_gap_s=control_detail_max_gap_s,
                current_path_max_gap_s=current_path_max_gap_s,
                trace_gap_suspect=trace_gap_suspect,
                fitted_error_rms_steps=rms,
                fitted_error_peak_steps=peak,
                actual_torque_range=_range(torque_values),
                torque_target_range=_range(torque_target_values),
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
                phase_reports=phase_reports,
            )
        )


def _velocity_gain_tuple(value: int) -> tuple[int, int]:
    packed = value & 0xFFFF_FFFF
    return packed & 0xFFFF, (packed >> 16) & 0xFFFF


def _velocity_capture_groups(
    states: dict[int, _DeviceState], commissioning: CommissioningCaptureReport
) -> list[VelocityCaptureGroup]:
    groups: list[VelocityCaptureGroup] = []
    claimed_observation_ids: set[int] = set()
    for run_index, run in enumerate(commissioning.runs):
        state = states.get(run.device_index)
        if state is None:
            continue
        next_start = next(
            (
                later.start_ticks
                for later in commissioning.runs[run_index + 1 :]
                if later.device_index == run.device_index
                and later.source_index == run.source_index
                and later.start_ticks is not None
            ),
            None,
        )
        start_ticks = run.start_ticks if run.start_ticks is not None else -1
        matching = [
            item
            for item in state.velocity_observations
            if id(item) not in claimed_observation_ids
            if item.source_index == run.source_index
            and item.observation.direction in (-1, 1)
            and item.timestamp_ticks >= start_ticks
            and (run.terminal_ticks is None or item.timestamp_ticks <= run.terminal_ticks)
            and (next_start is None or item.timestamp_ticks < next_start)
        ]
        if not matching:
            continue

        gain_fragments = sorted(
            (
                fragment
                for fragment in run.summaries
                if fragment.family == "velocity_validate_summary" and fragment.fragment == "gain"
            ),
            key=lambda fragment: fragment.timestamp_ticks,
        )
        candidates: list[
            tuple[list[_VelocityFrameObservation], int | None, int | None, int | None]
        ] = []
        previous_summary_ticks = start_ticks - 1
        for fragment in gain_fragments:
            candidate_observations = [
                item
                for item in matching
                if previous_summary_ticks < item.timestamp_ticks < fragment.timestamp_ticks
            ]
            if candidate_observations:
                velocity_p, velocity_i = _velocity_gain_tuple(fragment.value)
                candidates.append(
                    (
                        candidate_observations,
                        fragment.summary_seq,
                        velocity_p,
                        velocity_i,
                    )
                )
            previous_summary_ticks = fragment.timestamp_ticks

        trailing = [item for item in matching if item.timestamp_ticks > previous_summary_ticks]
        if trailing:
            if len(gain_fragments) == 1 and not candidates:
                fragment = gain_fragments[0]
                velocity_p, velocity_i = _velocity_gain_tuple(fragment.value)
                candidates.append((trailing, fragment.summary_seq, velocity_p, velocity_i))
            else:
                candidates.append((trailing, None, None, None))
        if not gain_fragments:
            candidates = [(matching, None, None, None)]
        claimed_observation_ids.update(id(item) for item in matching)

        for candidate, summary_seq, velocity_p, velocity_i in candidates:
            caveats = list(run.caveats)
            if summary_seq is None and gain_fragments:
                caveats.append("partial velocity candidate: missing gain summary")
            groups.append(
                VelocityCaptureGroup(
                    device_index=run.device_index,
                    source_index=run.source_index,
                    serial=state.report.serial or state.report.metadata_serial,
                    product=state.report.product,
                    engine_kind=run.engine_kind_name,
                    engine_seq=run.engine_seq,
                    result=run.result,
                    start_ticks=run.start_ticks,
                    terminal_ticks=run.terminal_ticks,
                    candidate_summary_seq=summary_seq,
                    velocity_p=velocity_p,
                    velocity_i=velocity_i,
                    saturated_observations=sum(bool(item.flags & 1) for item in candidate),
                    missing_frames=state.report.missing_frames,
                    sequence_gaps=dict(state.report.sequence_gaps),
                    trace_drops=state.report.status_deltas.get("dropped_motion_sample_count", 0),
                    caveats=caveats,
                    report=analyze_velocity_observations(
                        [item.observation for item in candidate],
                        max_levels=VELOCITY_REFERENCE_MAX_LEVELS,
                    ),
                )
            )
    return groups


def _velocity_sweep_audits(states: dict[int, _DeviceState]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    retained_candidates: dict[tuple[int, int], dict[str, object]] = {}
    for device_index, state in sorted(states.items()):
        run_sequences = sorted(
            {
                int(item.payload["run_sequence"])
                for item in state.velocity_sweep_evidence
                if "run_sequence" in item.payload
                and item.payload.get("record_kind") != "outer_safety_timing"
                and (
                    item.payload.get("record_kind") != "rung_origin_recovery"
                    or int(item.payload.get("stage", -1)) == 0
                )
            }
        )
        for run_sequence in run_sequences:
            evidence = [
                item
                for item in state.velocity_sweep_evidence
                if int(item.payload.get("run_sequence", -1)) == run_sequence
                and item.payload.get("record_kind") != "outer_safety_timing"
                and (
                    not str(item.payload.get("record_kind", "")).startswith("integral_")
                    and (
                        item.payload.get("record_kind") != "rung_origin_recovery"
                        or int(item.payload.get("stage", -1)) == 0
                    )
                )
            ]
            if not evidence:
                continue
            source_indexes = {item.source_index for item in evidence}
            if len(source_indexes) != 1:
                raise ValueError("velocity sweep evidence changed source index")
            source_index = next(iter(source_indexes))
            start = min(item.timestamp_ticks for item in evidence)
            end = max(item.timestamp_ticks for item in evidence)
            raw = [
                item.observation
                for item in state.velocity_observations
                if item.source_index == source_index and start <= item.timestamp_ticks <= end
            ]
            report = audit_velocity_sweep(
                raw,
                [(item.raw_payload, item.payload) for item in evidence],
            )
            report["device_index"] = device_index
            report["source_index"] = source_index
            report["serial"] = state.report.serial or state.report.metadata_serial
            candidate_key = (device_index, source_index)
            if report.get("reproduction") is not None:
                previous = retained_candidates.get(candidate_key)
                if previous is not None:
                    pair_audit = audit_stage_b_reproduction_pair(previous, report)
                    report["reproduction_pair_audit"] = pair_audit
                    if not pair_audit["matches_firmware"]:
                        report["matches_firmware"] = False
                        report["mismatches"].extend(pair_audit["mismatches"])
                    report["caveats"] = [
                        caveat for caveat in report["caveats"] if "prior capture" not in caveat
                    ]
            terminal = report.get("terminal")
            if isinstance(terminal, dict):
                outcome = int(terminal.get("outcome", -1))
                cause = int(terminal.get("cause", -1))
                if outcome == 0 or (outcome == 2 and cause == 5):
                    retained_candidates[candidate_key] = report
                else:
                    retained_candidates.pop(candidate_key, None)
            audits.append(report)
    return audits


def _velocity_integral_audits(states: dict[int, _DeviceState]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    retained_candidates: dict[tuple[int, int], dict[str, object]] = {}
    for device_index, state in sorted(states.items()):
        run_sequences = sorted(
            {
                int(item.payload["run_sequence"])
                for item in state.velocity_sweep_evidence
                if (
                    str(item.payload.get("record_kind", "")).startswith("integral_")
                    or (
                        item.payload.get("record_kind") == "rung_origin_recovery"
                        and int(item.payload.get("stage", -1)) == 1
                    )
                )
                and "run_sequence" in item.payload
            }
        )
        for run_sequence in run_sequences:
            evidence = [
                item
                for item in state.velocity_sweep_evidence
                if int(item.payload.get("run_sequence", -1)) == run_sequence
                and (
                    str(item.payload.get("record_kind", "")).startswith("integral_")
                    or (
                        item.payload.get("record_kind") == "rung_origin_recovery"
                        and int(item.payload.get("stage", -1)) == 1
                    )
                )
            ]
            if not evidence:
                continue
            source_indexes = {item.source_index for item in evidence}
            if len(source_indexes) != 1:
                raise ValueError("velocity-integral evidence changed source index")
            source_index = next(iter(source_indexes))
            start = min(item.timestamp_ticks for item in evidence)
            end = max(item.timestamp_ticks for item in evidence)
            raw = [
                item.observation
                for item in state.velocity_observations
                if item.source_index == source_index and start <= item.timestamp_ticks <= end
            ]
            report = audit_velocity_integral(
                raw,
                [(item.raw_payload, item.payload) for item in evidence],
            )
            report["device_index"] = device_index
            report["source_index"] = source_index
            report["serial"] = state.report.serial or state.report.metadata_serial
            candidate_key = (device_index, source_index)
            terminal = report.get("terminal")
            if isinstance(terminal, dict):
                outcome = int(terminal.get("outcome", -1))
                cause = int(terminal.get("cause", -1))
                previous = retained_candidates.get(candidate_key)
                if previous is not None and outcome in (1, 2) and (outcome == 1 or cause == 2):
                    pair_audit = audit_integral_reproduction_pair(previous, report)
                    report["reproduction_pair_audit"] = pair_audit
                    if not pair_audit["matches_firmware"]:
                        report["matches_firmware"] = False
                        report["mismatches"].extend(pair_audit["mismatches"])
                if outcome == 0 or (outcome == 2 and cause == 2):
                    retained_candidates[candidate_key] = report
                elif outcome != 4:
                    retained_candidates.pop(candidate_key, None)
            audits.append(report)
    return audits


def _outer_safety_timing_audits(states: dict[int, _DeviceState]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    for device_index, state in sorted(states.items()):
        records = [
            item
            for item in state.velocity_sweep_evidence
            if item.payload.get("record_kind") == "outer_safety_timing"
        ]
        used_markers: set[int] = set()
        for record in records:
            engine_sequence = int(record.payload["engine_sequence"])
            marker_index = next(
                (
                    index
                    for index, marker in enumerate(state.outer_safety_timing_markers)
                    if index not in used_markers and marker[2] == engine_sequence
                ),
                None,
            )
            marker_args = None
            if marker_index is not None:
                used_markers.add(marker_index)
                marker = state.outer_safety_timing_markers[marker_index]
                marker_args = [marker[2], marker[3]]

            run_sequence = int(record.payload["run_sequence"])
            related = [
                item
                for item in state.velocity_sweep_evidence
                if item.source_index == record.source_index
                and int(item.payload.get("run_sequence", -1)) == run_sequence
                and item.timestamp_ticks <= record.timestamp_ticks
                and item is not record
            ]
            start_ticks = min(
                (item.timestamp_ticks for item in related),
                default=record.timestamp_ticks,
            )
            raw_elapsed_us = [
                item.observation.elapsed_us
                for item in state.velocity_observations
                if item.source_index == record.source_index
                and start_ticks <= item.timestamp_ticks <= record.timestamp_ticks
            ]
            report = audit_outer_safety_timing(record.payload, raw_elapsed_us, marker_args)
            report.update(
                {
                    "device_index": device_index,
                    "source_index": record.source_index,
                    "serial": state.report.serial or state.report.metadata_serial,
                    "run_sequence": run_sequence,
                    "engine_sequence": engine_sequence,
                }
            )
            audits.append(report)

        for index, marker in enumerate(state.outer_safety_timing_markers):
            if index in used_markers:
                continue
            report = audit_outer_safety_timing(None, [], [marker[2], marker[3]])
            report.update(
                {
                    "device_index": device_index,
                    "source_index": marker[1],
                    "serial": state.report.serial or state.report.metadata_serial,
                    "engine_sequence": marker[2],
                }
            )
            audits.append(report)
    return audits


def analyze_capture(
    source: BinaryIO | io.BytesIO,
    steady_after_s: float = 10.0,
    motion_gap_s: float = DEFAULT_MOTION_GAP_S,
    motion_min_steps: int = DEFAULT_MOTION_MIN_STEPS,
    corexy_plus: str | None = None,
    corexy_minus: str | None = None,
    contour_pair_tolerance_ms: float = 5.0,
    decoupling_l_int: int | None = None,
    decoupling_f_pwm_hz: float = DEFAULT_DECOUPLING_F_PWM_HZ,
    decoupling_pole_pairs: int = DEFAULT_DECOUPLING_POLE_PAIRS,
    decoupling_position_units_per_rev: int = DEFAULT_DECOUPLING_POSITION_UNITS_PER_REV,
) -> CaptureAnalysis:
    """Analyze one FTRF capture stream."""
    states: dict[int, _DeviceState] = {}
    commissioning = CommissioningCaptureBuilder()
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
            _observe_critical(
                state,
                frame.payload,
                header.timestamp_ticks,
                (header.source_kind, header.source_index),
                record.host_recv_ns,
                steady_after_s,
            )
            commissioning.observe_critical(
                record.device_index,
                header.source_index,
                header.timestamp_ticks,
                frame.payload,
            )
        elif header.frame_type == FRAME_TYPE_CURRENT_PATH:
            _observe_current_path(
                state,
                frame.payload,
                header.timestamp_ticks,
                (header.source_kind, header.source_index),
                steady_after_s,
            )
        elif header.frame_type == FRAME_TYPE_PHASE_ADVANCE:
            _observe_phase_advance(
                state,
                frame.payload,
                header.timestamp_ticks,
                (header.source_kind, header.source_index),
            )
        elif header.frame_type == FRAME_TYPE_VELOCITY_OBSERVATION:
            _observe_velocity_observation(
                state,
                frame.payload,
                header.timestamp_ticks,
                header.source_index,
            )
        elif header.frame_type == FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE:
            state.velocity_sweep_evidence.append(
                _VelocitySweepEvidenceFrame(
                    timestamp_ticks=header.timestamp_ticks,
                    source_index=header.source_index,
                    raw_payload=frame.raw_payload,
                    payload=frame.payload,
                )
            )
        elif header.frame_type == FRAME_TYPE_CONTROL_DETAIL:
            _observe_control_detail(
                state,
                frame.payload,
                header.timestamp_ticks,
                (header.source_kind, header.source_index),
            )
            commissioning.observe_control_detail(
                record.device_index,
                header.source_index,
                header.timestamp_ticks,
                frame.payload,
            )
        elif header.frame_type == FRAME_TYPE_EVENT:
            if (
                int(frame.payload["event_id"]) == EVENT_COMMISSION_OUTER_SAFETY_TIMING
                and int(frame.payload["arg_count"]) >= 4
            ):
                args = frame.payload["args"]
                state.outer_safety_timing_markers.append(
                    (
                        header.timestamp_ticks,
                        int(args[0]),  # type: ignore[index]
                        int(args[2]),  # type: ignore[index]
                        int(args[3]),  # type: ignore[index]
                    )
                )
            commissioning.observe_event(record.device_index, header, frame.payload)

    decoupling_config = (
        None
        if decoupling_l_int is None
        else DecouplingConfig(
            l_int=decoupling_l_int,
            f_pwm_hz=decoupling_f_pwm_hz,
            pole_pairs=decoupling_pole_pairs,
            position_units_per_rev=decoupling_position_units_per_rev,
        )
    )
    for state in states.values():
        _finalize_status(state)
        _finalize_motion_segments(state, motion_gap_s, motion_min_steps, decoupling_config)
    commissioning_report = commissioning.finalize(
        {index: state.clock_hz for index, state in states.items()}
    )
    contour_segments: list[ContourSegmentReport] = []
    if corexy_plus is not None or corexy_minus is not None:
        if corexy_plus is None or corexy_minus is None:
            raise ValueError("both corexy_plus and corexy_minus are required")
        plus_state = _select_state_by_serial(states, corexy_plus)
        minus_state = _select_state_by_serial(states, corexy_minus)
        tolerance_ns = int(contour_pair_tolerance_ms * 1_000_000)
        pairs = pair_motor_samples(
            _motor_samples(plus_state),
            _motor_samples(minus_state),
            tolerance_ns=tolerance_ns,
        )
        pair_groups = contour_pair_groups(
            pairs,
            gap_ns=int(motion_gap_s * 1_000_000_000),
            min_delta=float(motion_min_steps),
        )
        plus_spans = _motion_segment_spans(plus_state, motion_gap_s, motion_min_steps)
        minus_spans = _motion_segment_spans(minus_state, motion_gap_s, motion_min_steps)
        for segment_index, pair_group in enumerate(pair_groups):
            relevant_segments = _contour_group_segments(pair_group, plus_spans, minus_spans)
            contour_segments.append(
                contour_segment_report(
                    segment_index,
                    pair_group,
                    position_trace_gap=_contour_position_trace_gap(relevant_segments),
                    support_trace_gap=_contour_support_trace_gap(relevant_segments),
                    post_discriminator=_contour_post_discriminator(relevant_segments),
                )
            )
    return CaptureAnalysis(
        devices=[states[index].report for index in sorted(states)],
        commissioning=commissioning_report,
        contour_segments=contour_segments,
        velocity_reports=_velocity_capture_groups(states, commissioning_report),
        velocity_sweep_audits=_velocity_sweep_audits(states),
        velocity_integral_audits=_velocity_integral_audits(states),
        outer_safety_timing_audits=_outer_safety_timing_audits(states),
        corexy_plus=corexy_plus,
        corexy_minus=corexy_minus,
    )


def summarize_capture(
    source: BinaryIO | io.BytesIO,
    steady_after_s: float = 10.0,
    motion_gap_s: float = DEFAULT_MOTION_GAP_S,
    motion_min_steps: int = DEFAULT_MOTION_MIN_STEPS,
) -> list[DeviceReport]:
    """Summarize one FTRF capture stream."""
    return analyze_capture(source, steady_after_s, motion_gap_s, motion_min_steps).devices


def _range_text(ranges: dict[str, tuple[int, int]], name: str) -> str:
    if name not in ranges:
        return "n/a"
    lo, hi = ranges[name]
    return f"{lo}..{hi}"


def _optional_range_text(value: tuple[int, int] | None) -> str:
    if value is None:
        return "n/a"
    return f"{value[0]}..{value[1]}"


def _compact_range_text(value: tuple[int, int]) -> str:
    if value[0] == value[1]:
        return str(value[0])
    return f"{value[0]}..{value[1]}"


def _optional_ratio_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _optional_float_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _source_key_text(source_key: tuple[int, int]) -> str:
    kind, index = source_key
    prefix = {0: "board", 1: "FOCI", 2: "USB", 3: "Klipper"}.get(kind, f"kind{kind}")
    if kind in (0, 2, 3):
        return prefix
    return f"{prefix}[{index}]"


def _optional_milliseconds_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.1f}ms"


def format_summary(
    reports: list[DeviceReport],
    *,
    include_phase_split: bool = False,
    include_feedforward_audit: bool = False,
    include_decoupling_audit: bool = False,
) -> str:
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
        multiple_motion_sources = len(report.critical_source_keys) > 1
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
                "  motion{idx}{source} delta={delta} duration={duration:.3f}s "
                "critical={critical} control_detail={detail} "
                "fit_rms={rms:.1f} fit_peak={peak:.1f} "
                "torque={torque_lo}..{torque_hi} "
                "torque_target={target_lo}..{target_hi} "
                "uq_limited={uq_lo}..{uq_hi} "
                "uq_rail={rail}/{critical} detail_torque_cmd={cmd} "
                "saturated={sat}/{detail} actual_cmd_median={ratio} "
                "iq={iq} id={id} iux={iux} iwy={iwy} "
                "adc0_raw={adc0} adc1_raw={adc1} "
                "accel_ff={accel_lo}..{accel_hi} gain={gain_lo}..{gain_hi} "
                "flags=0x{flags:02x}".format(
                    idx=segment.segment_index,
                    source=(
                        f" source={_source_key_text(segment.source_key)}"
                        if multiple_motion_sources
                        else ""
                    ),
                    delta=segment.step_delta,
                    duration=segment.duration_s,
                    critical=segment.critical_samples,
                    detail=segment.control_detail_samples,
                    rms=segment.fitted_error_rms_steps,
                    peak=segment.fitted_error_peak_steps,
                    torque_lo=segment.actual_torque_range[0],
                    torque_hi=segment.actual_torque_range[1],
                    target_lo=segment.torque_target_range[0],
                    target_hi=segment.torque_target_range[1],
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
            if segment.trace_gap_suspect:
                lines.append(
                    "    trace_gap critical={critical} detail={detail} current={current}".format(
                        critical=_optional_milliseconds_text(segment.critical_max_gap_s),
                        detail=_optional_milliseconds_text(segment.control_detail_max_gap_s),
                        current=_optional_milliseconds_text(segment.current_path_max_gap_s),
                    )
                )
            authority = segment.current_authority
            if authority is not None:
                lines.append(
                    "    authority near_limit={near} sign={sign}/{near} "
                    "coverage={coverage:.1f}% longest={longest} "
                    "actual/target={actual_target} uq_lim={uq_lim} "
                    "uq_delta={uq_delta} uq/limit={uq_limit} rail={rail}/{near} "
                    "iq/actual={iq_actual} adc0_delta={adc0} adc1_delta={adc1}".format(
                        near=authority.near_limit_samples,
                        sign=authority.sign_agreement_samples,
                        coverage=authority.near_limit_sample_coverage * 100.0,
                        longest=_optional_milliseconds_text(authority.near_limit_longest_run_s),
                        actual_target=_optional_ratio_text(authority.actual_to_target_abs_median),
                        uq_lim=_optional_range_text(authority.uq_limited_range),
                        uq_delta=_optional_range_text(authority.uq_delta_range),
                        uq_limit=_optional_ratio_text(authority.uq_to_limit_abs_median),
                        rail=authority.voltage_rail_samples,
                        iq_actual=_optional_ratio_text(authority.iq_to_actual_abs_median),
                        adc0=_optional_range_text(authority.adc_i0_delta_range),
                        adc1=_optional_range_text(authority.adc_i1_delta_range),
                    )
                )
            if include_phase_split:
                for phase in segment.phase_reports:
                    lines.append(
                        "    phase {name} samples={samples} "
                        "fit_rms={rms:.1f} fit_peak={peak:.1f} "
                        "residual_med={median:.1f} residual_final={final:.1f} "
                        "torque={torque_lo}..{torque_hi} "
                        "torque_target={target_lo}..{target_hi} "
                        "actual/target={actual_target} "
                        "velocity_actual={velocity_actual} "
                        "velocity_effective_target={velocity_target} "
                        "velocity_error_rms={velocity_error_rms} "
                        "velocity_error_peak={velocity_error_peak} "
                        "velocity_actual/effective_target={velocity_ratio} "
                        "velocity_pidin={velocity_pidin} "
                        "velocity_pidout={velocity_pidout} "
                        "uq/limit={uq_median}..{uq_peak} rail={rail}/{samples}".format(
                            name=phase.name,
                            samples=phase.critical_samples,
                            rms=phase.fitted_error_rms_steps,
                            peak=phase.fitted_error_peak_steps,
                            median=phase.fitted_error_median_steps,
                            final=phase.fitted_error_final_steps,
                            torque_lo=phase.actual_torque_range[0],
                            torque_hi=phase.actual_torque_range[1],
                            target_lo=phase.torque_target_range[0],
                            target_hi=phase.torque_target_range[1],
                            actual_target=_optional_ratio_text(phase.actual_to_target_abs_median),
                            velocity_actual=_optional_range_text(phase.velocity_actual_range),
                            velocity_target=_optional_range_text(
                                phase.velocity_effective_target_range
                            ),
                            velocity_error_rms=_optional_float_text(phase.velocity_error_rms),
                            velocity_error_peak=_optional_float_text(phase.velocity_error_peak),
                            velocity_ratio=_optional_ratio_text(
                                phase.velocity_actual_to_effective_target_abs_median
                            ),
                            velocity_pidin=_optional_range_text(phase.velocity_pidin_target_range),
                            velocity_pidout=_optional_range_text(
                                phase.velocity_pidout_target_range
                            ),
                            uq_median=_optional_ratio_text(phase.uq_to_limit_abs_median),
                            uq_peak=_optional_ratio_text(phase.uq_to_limit_abs_peak),
                            rail=phase.voltage_rail_samples,
                        )
                    )
                    if include_feedforward_audit:
                        lines.append(
                            "      ff_audit accel_ff={accel_lo}..{accel_hi} "
                            "nonzero={nonzero}/{samples} accel_sign={sign}/{sign_total} "
                            "velocity_ff={vel_lo}..{vel_hi} "
                            "velocity_nonzero={vel_nonzero}/{samples} "
                            "velocity_sign={vel_sign}/{vel_sign_total} "
                            "velocity_transient={transient_lo}..{transient_hi} "
                            "transient_nonzero={transient_nonzero}/{samples} "
                            "transient_sign={transient_sign}/{transient_sign_total} "
                            "iq/torque={iq_actual} uq_limited={uq_lo}..{uq_hi} "
                            "post_discriminator={discriminator}".format(
                                accel_lo=phase.accel_feedforward_offset_range[0],
                                accel_hi=phase.accel_feedforward_offset_range[1],
                                nonzero=phase.accel_feedforward_nonzero_samples,
                                samples=phase.critical_samples,
                                sign=phase.accel_feedforward_sign_agreement_samples,
                                sign_total=phase.accel_feedforward_sign_comparable_samples,
                                vel_lo=phase.velocity_feedforward_offset_range[0],
                                vel_hi=phase.velocity_feedforward_offset_range[1],
                                vel_nonzero=phase.velocity_feedforward_nonzero_samples,
                                vel_sign=phase.velocity_feedforward_sign_agreement_samples,
                                vel_sign_total=phase.velocity_feedforward_sign_comparable_samples,
                                transient_lo=phase.velocity_transient_offset_range[0],
                                transient_hi=phase.velocity_transient_offset_range[1],
                                transient_nonzero=phase.velocity_transient_nonzero_samples,
                                transient_sign=phase.velocity_transient_sign_agreement_samples,
                                transient_sign_total=(
                                    phase.velocity_transient_sign_comparable_samples
                                ),
                                iq_actual=_optional_ratio_text(phase.iq_to_actual_abs_median),
                                uq_lo=phase.uq_limited_range[0],
                                uq_hi=phase.uq_limited_range[1],
                                discriminator=phase.post_discriminator or "n/a",
                            )
                        )
                        if phase.phase_advance is not None:
                            phase_advance = phase.phase_advance
                            lines.append(
                                "      phase_advance={counts} "
                                "nonzero={nonzero}/{samples} "
                                "clamped={clamped}/{samples} gated={gated}/{samples} "
                                "write_failed={write_failed}/{samples} "
                                "gain={gain} max={max_counts} deadband={deadband}".format(
                                    counts=_optional_range_text(phase_advance.counts_range),
                                    nonzero=phase_advance.nonzero_samples,
                                    samples=phase_advance.samples,
                                    clamped=phase_advance.clamped_samples,
                                    gated=phase_advance.gated_samples,
                                    write_failed=phase_advance.write_failed_samples,
                                    gain=_compact_range_text(phase_advance.gain_ppm_range),
                                    max_counts=_compact_range_text(phase_advance.max_counts_range),
                                    deadband=_compact_range_text(phase_advance.deadband_range),
                                )
                            )
                    if include_decoupling_audit and phase.decoupling is not None:
                        decoupling = phase.decoupling
                        lines.append(
                            "      decoupling pole_pairs={pairs} l_int={l_int} "
                            "f_pwm={f_pwm:g} position_units_per_rev={units} "
                            "omega_e={omega_lo:.1f}..{omega_hi:.1f}rad/s "
                            "cross/uq_req={req} p95={req_p95} "
                            "cross/uq_lim={lim} p95={lim_p95} "
                            "vd={vd_lo:.1f}..{vd_hi:.1f} "
                            "vq={vq_lo:.1f}..{vq_hi:.1f} samples={samples}".format(
                                pairs=decoupling.pole_pairs,
                                l_int=decoupling.l_int,
                                f_pwm=decoupling.f_pwm_hz,
                                units=decoupling.position_units_per_rev,
                                omega_lo=decoupling.omega_e_range_rad_s[0],
                                omega_hi=decoupling.omega_e_range_rad_s[1],
                                req=_optional_ratio_text(decoupling.cross_to_requested_median),
                                req_p95=_optional_ratio_text(decoupling.cross_to_requested_p95),
                                lim=_optional_ratio_text(decoupling.cross_to_limited_median),
                                lim_p95=_optional_ratio_text(decoupling.cross_to_limited_p95),
                                vd_lo=decoupling.vd_cross_range[0],
                                vd_hi=decoupling.vd_cross_range[1],
                                vq_lo=decoupling.vq_cross_range[0],
                                vq_hi=decoupling.vq_cross_range[1],
                                samples=decoupling.samples,
                            )
                        )
    return "\n".join(lines)


def format_analysis(
    analysis: CaptureAnalysis,
    *,
    include_phase_split: bool = False,
    include_feedforward_audit: bool = False,
    include_decoupling_audit: bool = False,
    include_contour: bool = False,
) -> str:
    """Format full capture analysis, optionally including CoreXY contour reports."""
    text = format_summary(
        analysis.devices,
        include_phase_split=include_phase_split,
        include_feedforward_audit=include_feedforward_audit,
        include_decoupling_audit=include_decoupling_audit,
    )
    commissioning_text = format_commissioning_report(analysis.commissioning)
    if commissioning_text:
        text = "\n".join(part for part in (text, commissioning_text) if part)
    stage_b_lines = []
    for sweep in analysis.velocity_sweep_audits:
        replay = sweep.get("stage_b_replay")
        if not isinstance(replay, dict):
            continue
        stage_b_lines.append(
            "stage_b dev={device} source={source} run={run} p={p} coverage={coverage} "
            "matches_firmware={matches}".format(
                device=sweep.get("device_index", "?"),
                source=sweep.get("source_index", "?"),
                run=sweep.get("run_sequence", "?"),
                p=replay.get("nominated_p", 0),
                coverage=",".join(str(value) for value in replay.get("coverage", [])),
                matches=str(bool(sweep.get("matches_firmware"))).lower(),
            )
        )
        pair_audit = sweep.get("reproduction_pair_audit")
        if isinstance(pair_audit, dict) and isinstance(pair_audit.get("projection"), dict):
            projection = pair_audit["projection"]
            memberships = projection.get("memberships", [])
            stage_b_lines.append(
                "stage_b_core provenance=firmware_schema3 previous={previous} current={current} "
                "core={core} final_p={final_p} reason_mask={reason} reduced_margin={margin} "
                "coverage={coverage} conservative_common={common} offline_derived={derived}".format(
                    previous=[value["previous"] for value in memberships],
                    current=[value["current"] for value in memberships],
                    core=projection.get("core_masks", []),
                    final_p=projection.get("final_p", 0),
                    reason=projection.get("reason_mask", 0),
                    margin=str(bool(projection.get("reduced_margin"))).lower(),
                    coverage=projection.get("coverage", []),
                    common=projection.get("firmware_conservative_common", []),
                    derived=projection.get("offline_derived", "unavailable"),
                )
            )
    for integral in analysis.velocity_integral_audits:
        evidence = integral.get("run_evidence", {})
        masks = evidence.get("eligible_masks", []) if isinstance(evidence, dict) else []
        stage_b_lines.append(
            "velocity_integral dev={device} source={source} run={run} eligible={eligible} "
            "matches_firmware={matches}".format(
                device=integral.get("device_index", "?"),
                source=integral.get("source_index", "?"),
                run=integral.get("run_sequence", "?"),
                eligible=masks,
                matches=str(bool(integral.get("matches_firmware"))).lower(),
            )
        )
    for timing in analysis.outer_safety_timing_audits:
        stage_b_lines.append(
            "outer_safety_timing dev={device} source={source} engine={engine} "
            "max_us={maximum} headroom_us={headroom} reopen_bound={reopen} "
            "raw_covered={raw} safety_only={safety_only} matches_firmware={matches}".format(
                device=timing.get("device_index", "?"),
                source=timing.get("source_index", "?"),
                engine=timing.get("engine_sequence", "?"),
                maximum=timing.get("firmware", {}).get("max_realized_gap_us", "?")
                if isinstance(timing.get("firmware"), dict)
                else "?",
                headroom=timing.get("gap_headroom_us", "?"),
                reopen=str(bool(timing.get("reopen_bound"))).lower(),
                raw=timing.get("raw_covered_sample_count", 0),
                safety_only=timing.get("firmware_reported_safety_only", "?"),
                matches=str(bool(timing.get("matches_firmware"))).lower(),
            )
        )
    if stage_b_lines:
        text = "\n".join(part for part in (text, *stage_b_lines) if part)
    if not include_contour:
        return text
    lines = [text] if text else []
    if analysis.corexy_plus is not None and analysis.corexy_minus is not None:
        lines.append(f"contour_mapping plus={analysis.corexy_plus} minus={analysis.corexy_minus}")
    for segment in analysis.contour_segments:
        lines.append(
            "contour{idx} samples={samples} duration={duration:.3f}s "
            "delta_xy={dx:.1f},{dy:.1f} "
            "along_rms={along_rms:.1f} along_peak={along_peak:.1f} "
            "cross_rms={cross_rms:.1f} cross_peak={cross_peak:.1f} "
            "final={final_along:.1f},{final_cross:.1f} "
            "path_class={path_classification} class={classification} "
            "position_gap={position_gap:d} support_gap={support_gap:d} "
            "post={post_discriminator} pair_gap={pair_gap:.3f}ms".format(
                idx=segment.segment_index,
                samples=segment.samples,
                duration=segment.duration_s,
                dx=segment.delta_x,
                dy=segment.delta_y,
                along_rms=segment.along_rms,
                along_peak=segment.along_peak,
                cross_rms=segment.cross_rms,
                cross_peak=segment.cross_peak,
                final_along=segment.final_along,
                final_cross=segment.final_cross,
                path_classification=segment.path_classification,
                classification=segment.classification,
                position_gap=int(segment.position_trace_gap),
                support_gap=int(segment.support_trace_gap),
                post_discriminator=segment.post_discriminator or "n/a",
                pair_gap=segment.max_pair_delta_ms,
            )
        )
    return "\n".join(lines)


def analysis_to_json(analysis: CaptureAnalysis) -> dict[str, object]:
    """Convert full capture analysis to JSON-compatible dictionaries."""
    return {
        "devices": [
            {
                "device_index": report.device_index,
                "usb_path": report.usb_path,
                "product": report.product,
                "serial": report.serial,
                "metadata_serial": report.metadata_serial,
                "source_labels": list(report.source_labels),
                "frame_counts": dict(report.frame_counts),
                "missing_frames": report.missing_frames,
                "sequence_gaps": dict(report.sequence_gaps),
                "status_deltas": dict(report.status_deltas),
                "steady_after_s": report.steady_after_s,
                "steady_samples": report.steady_samples,
                "voltage_limited_samples": report.voltage_limited_samples,
                "ranges": dict(report.ranges),
                "motion_segments": [
                    {
                        "segment_index": segment.segment_index,
                        **(
                            {
                                "source_kind": segment.source_key[0],
                                "source_index": segment.source_key[1],
                            }
                            if len(report.critical_source_keys) > 1
                            else {}
                        ),
                        "start_s": segment.start_s,
                        "duration_s": segment.duration_s,
                        "step_delta": segment.step_delta,
                        "critical_samples": segment.critical_samples,
                        "control_detail_samples": segment.control_detail_samples,
                        "trace_gap_suspect": segment.trace_gap_suspect,
                    }
                    for segment in report.motion_segments
                ],
                "decode_errors": report.decode_errors,
            }
            for report in analysis.devices
        ],
        "commissioning": commissioning_report_to_json(analysis.commissioning),
        "contour_segments": [
            {
                "segment_index": segment.segment_index,
                "samples": segment.samples,
                "duration_s": segment.duration_s,
                "delta_x": segment.delta_x,
                "delta_y": segment.delta_y,
                "classification": segment.classification,
            }
            for segment in analysis.contour_segments
        ],
    }


def velocity_analysis_to_json(analysis: CaptureAnalysis) -> dict[str, object]:
    """Convert velocity measurement groups to deterministic JSON-compatible values."""
    result = {
        "sweeps": analysis.velocity_sweep_audits,
        "outer_safety_timing": analysis.outer_safety_timing_audits,
        "groups": [
            {
                "device_index": group.device_index,
                "source_index": group.source_index,
                "serial": group.serial,
                "product": group.product,
                "engine_kind": group.engine_kind,
                "engine_seq": group.engine_seq,
                "result": group.result,
                "start_ticks": group.start_ticks,
                "terminal_ticks": group.terminal_ticks,
                "candidate_summary_seq": group.candidate_summary_seq,
                "velocity_p": group.velocity_p,
                "velocity_i": group.velocity_i,
                "saturated_observations": group.saturated_observations,
                "missing_frames": group.missing_frames,
                "sequence_gaps": dict(group.sequence_gaps),
                "trace_drops": group.trace_drops,
                "caveats": list(group.caveats),
                "measurement": velocity_report_to_json(group.report),
            }
            for group in analysis.velocity_reports
        ],
    }
    if analysis.velocity_integral_audits:
        result["integral_responses"] = analysis.velocity_integral_audits
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the offline capture analyzer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--steady-after-s", type=float, default=10.0)
    parser.add_argument("--motion-gap-s", type=float, default=DEFAULT_MOTION_GAP_S)
    parser.add_argument("--motion-min-steps", type=int, default=DEFAULT_MOTION_MIN_STEPS)
    parser.add_argument(
        "--phase-split",
        action="store_true",
        help="include entry/cruise/exit/post-move phase metrics for motion segments",
    )
    parser.add_argument(
        "--feedforward-audit",
        action="store_true",
        help="include feedforward sign/current-path discriminator lines with phase metrics",
    )
    parser.add_argument(
        "--decoupling-audit",
        action="store_true",
        help="include count-space q/d cross-coupling magnitude estimates",
    )
    parser.add_argument(
        "--decoupling-l-int",
        type=int,
        help="identified count-space inductance in voltage-count/current-count microseconds",
    )
    parser.add_argument(
        "--decoupling-f-pwm-hz",
        type=float,
        default=DEFAULT_DECOUPLING_F_PWM_HZ,
        help="current-loop/PWM update rate used by target_velocity units",
    )
    parser.add_argument(
        "--decoupling-pole-pairs",
        type=int,
        default=DEFAULT_DECOUPLING_POLE_PAIRS,
        help="motor pole-pair count; LDO 1.8 degree hybrid steppers use 50",
    )
    parser.add_argument(
        "--decoupling-position-units-per-rev",
        type=int,
        default=DEFAULT_DECOUPLING_POSITION_UNITS_PER_REV,
        help="mechanical position units per revolution used by target_velocity",
    )
    parser.add_argument(
        "--contour-analysis",
        action="store_true",
        help="include CoreXY along-path and cross-path contour metrics",
    )
    parser.add_argument("--corexy-plus", help="trace device selector for the X+Y motor")
    parser.add_argument("--corexy-minus", help="trace device selector for the X-Y motor")
    parser.add_argument(
        "--contour-pair-tolerance-ms",
        type=float,
        default=5.0,
        help="maximum host receive timestamp delta for paired contour samples",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--velocity-json",
        metavar="PATH",
        help="write machine-readable velocity measurement evidence to PATH",
    )

    try:
        args = parser.parse_args(argv)
        if args.contour_analysis and (args.corexy_plus is None or args.corexy_minus is None):
            parser.error("both --corexy-plus and --corexy-minus are required")
        if args.decoupling_audit and args.decoupling_l_int is None:
            parser.error("--decoupling-l-int is required with --decoupling-audit")
    except SystemExit as exc:
        return int(exc.code)

    try:
        with open(args.capture, "rb") as source:
            analysis = analyze_capture(
                source,
                steady_after_s=args.steady_after_s,
                motion_gap_s=args.motion_gap_s,
                motion_min_steps=args.motion_min_steps,
                corexy_plus=args.corexy_plus if args.contour_analysis else None,
                corexy_minus=args.corexy_minus if args.contour_analysis else None,
                contour_pair_tolerance_ms=args.contour_pair_tolerance_ms,
                decoupling_l_int=args.decoupling_l_int if args.decoupling_audit else None,
                decoupling_f_pwm_hz=args.decoupling_f_pwm_hz,
                decoupling_pole_pairs=args.decoupling_pole_pairs,
                decoupling_position_units_per_rev=args.decoupling_position_units_per_rev,
            )
    except OSError as exc:
        print(f"failed to read {args.capture}: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        try:
            parser.error(str(exc))
        except SystemExit as parser_exc:
            return int(parser_exc.code)
    if args.velocity_json:
        try:
            with open(args.velocity_json, "w", encoding="utf-8") as output:
                output.write(json.dumps(velocity_analysis_to_json(analysis), sort_keys=True))
                output.write("\n")
        except OSError as exc:
            print(f"failed to write {args.velocity_json}: {exc}", file=sys.stderr)
            return 2
    elif args.json:
        print(json.dumps(analysis_to_json(analysis), sort_keys=True))
    else:
        print(
            format_analysis(
                analysis,
                include_phase_split=args.phase_split,
                include_feedforward_audit=args.feedforward_audit,
                include_decoupling_audit=args.decoupling_audit,
                include_contour=args.contour_analysis,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
