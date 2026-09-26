"""Decode FOCI USB trace FTRC frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from ._vocabulary_generated import BREAKAWAY_PHASE_NAMES, PHASE_NAMES
from .scored_rest import select_causal_rest_suffixes
from .velocity_integral import (
    _RESPONSE_LADDER_INTEGRAL_SCHEMA_REVISIONS,
)

__all__ = ["BREAKAWAY_PHASE_NAMES", "PHASE_NAMES"]

FRAME_MAGIC = 0x43525446
# Version 23 adds EVENT_PHASE_BOUNDARY (event id 12, subsystem
# SUBSYSTEM_STEP_TIMER): a one-shot marker for an accel/cruise/decel
# phase-class transition, detected in the step-queue ISR independent of
# the critical-frame sampler's rate.
# Version 22 appends `trace_deadline_pressure_count` and `max_spi_sample_cycles`
# to the status payload, and repurposes the critical payload's `position_error`
# to the TMC4671 chip's own PID_POSITION_ERROR register read (byte layout
# unchanged).
# Version 21 adds `stop_rung_index` to the breakaway discovery terminal: the
# rung the discovery walk had reached when it stopped, distinct from the
# breakaway probe's rung (constant per ladder, decoded elsewhere in the
# record).
# Version 20 adds the velocity feedforward clamp-engaged flag to the critical
# payload.
# Version 19 adds `error_code` to the breakaway campaign terminal and a new
# ConfirmationIntegralPlanRefused (26) terminal cause, so a confirmed
# acceptance whose Integral plan build was refused is attributable instead of
# reporting a self-contradictory Accepted cause.
# Version 17 adds `error_code` to the breakaway discovery terminal, so a
# `*_internal_fault` terminal names the CommissionError behind it instead of
# leaving the fault unattributable.
# Version 16 added the measured `moving_pid_output_peak` to the breakaway
# raw-observation record, so a current-headroom terminus can be attributed to
# the open-loop model term or to measured demand.
# Version 15 replaced the productive-boundary probe with physical excursion.
# Historical versions remain readable through their versioned record shapes.
PROTOCOL_VERSION = 23
SUPPORTED_PROTOCOL_VERSIONS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
)
HEADER_LEN = 32

FRAME_TYPE_CRITICAL = 1
FRAME_TYPE_CONTROL_DETAIL = 2
FRAME_TYPE_STATUS = 3
FRAME_TYPE_EVENT = 4
FRAME_TYPE_METADATA = 5
FRAME_TYPE_CURRENT_PATH = 6
FRAME_TYPE_PHASE_ADVANCE = 7
FRAME_TYPE_VELOCITY_OBSERVATION = 8
FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE = 9

# FOCI and Klipper status samples never populate dropped_event_log_count
# (firmware builds them with ..StatusSample::default()); only board and USB
# carry the real counter, so only those two are exposed here.
SOURCE_KIND_BOARD = 0
SOURCE_KIND_USB = 2

FLAG_DROP_GAP_BEFORE_THIS_FRAME = 1 << 0

VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID = 1 << 1
VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID = 1 << 2

SUBSYSTEM_COMMISSIONING = 7
SUBSYSTEM_STEP_TIMER = 2

EVENT_COMMISSION_ENGINE_START = 100
EVENT_COMMISSION_PHASE_START = 101
EVENT_COMMISSION_PHASE_END = 102
EVENT_COMMISSION_ENGINE_TERMINAL = 103
# Reserved, historical-only: production firmware no longer emits these events.
# Keep the ids stable and decodable so archived MechanicalId captures still
# decode; do not reuse 110-113 for anything else.
EVENT_COMMISSION_MECHANICAL_ID_RESULT = 110
EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT = 111
EVENT_COMMISSION_MECHANICAL_ID_RESPONSE = 112
EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION = 113
EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT = 120
EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN = 121
EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS = 122
EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS = 123
EVENT_COMMISSION_VELOCITY_SWEEP_PLAN = 124
EVENT_COMMISSION_VELOCITY_SWEEP_OBSERVATION = 125
EVENT_COMMISSION_VELOCITY_SWEEP_RUNG = 126
EVENT_COMMISSION_VELOCITY_SWEEP_DERIVED = 127
EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL = 128
EVENT_COMMISSION_VELOCITY_SWEEP_COMPLETE = 129
EVENT_COMMISSION_POSITION_TUNE_RESULT = 130
EVENT_COMMISSION_POSITION_TUNE_TARGET = 131
EVENT_COMMISSION_POSITION_TUNE_METRICS = 132
EVENT_COMMISSION_POSITION_TUNE_LIMITS = 133
# Reserved, historical-only: production firmware no longer emits this event;
# do not reuse 134 for anything else.
EVENT_COMMISSION_VELOCITY_INTEGRAL_PLAN = 134
EVENT_COMMISSION_VELOCITY_INTEGRAL_OBSERVATION = 135
EVENT_COMMISSION_VELOCITY_INTEGRAL_RUNG = 136
EVENT_COMMISSION_VELOCITY_INTEGRAL_DERIVED = 137
EVENT_COMMISSION_VELOCITY_INTEGRAL_TERMINAL = 138
EVENT_COMMISSION_VELOCITY_INTEGRAL_COMPLETE = 139
EVENT_COMMISSION_RESTORE_RESULT = 140
EVENT_COMMISSION_RESTORE_ERROR_DRIFT = 141
EVENT_COMMISSION_RESTORE_DISPOSITION = 142
EVENT_COMMISSION_RUNG_ORIGIN_RECOVERY = 143
EVENT_COMMISSION_OUTER_SAFETY_TIMING = 144
EVENT_COMMISSION_VELOCITY_LIMIT_LATCH_TEST = 145
EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_PLAN = 146
EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_OBSERVATION = 147
EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_AMPLITUDE = 148
EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_TERMINAL = 149
EVENT_COMMISSION_REST_SELECTION_DIAGNOSTIC = 151
EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN = 152
EVENT_COMMISSION_BREAKAWAY_PROBE_RESULT = 153
EVENT_COMMISSION_BREAKAWAY_PROBE_TERMINAL = 154
EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN = 155
EVENT_COMMISSION_BREAKAWAY_DISCOVERY_TERMINAL = 156
EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_PLAN = 157
EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_TERMINAL = 158
EVENT_COMMISSION_BREAKAWAY_CAMPAIGN_TERMINAL = 159
EVENT_COMMISSION_BREAKAWAY_RAW_OBSERVATION = 160
EVENT_COMMISSION_BREAKAWAY_DISCOVERY_MARGIN = 161
EVENT_COMMISSION_BREAKAWAY_PROBE_DECISION = 162
EVENT_COMMISSION_BREAKAWAY_RECOVERY_REST_REJECTED = 163
EVENT_COMMISSION_BREAKAWAY_RECOVERY_ORIGIN_MISMATCH = 164
EVENT_COMMISSION_DIAGNOSTIC_WARNING = 200

# Emitted from the step-queue ISR on an accel/cruise/decel phase-class
# transition. subsystem_id is SUBSYSTEM_STEP_TIMER, not
# SUBSYSTEM_COMMISSIONING.
EVENT_PHASE_BOUNDARY = 12

# Every breakaway-campaign marker event id (152-162), in phase-emission order.
# The FTRC marker granularity is one `CommissionTraceEvent` per tick -- these
# ids are what the offline assembler uses to claim the raw velocity-sweep-
# evidence frames immediately preceding each marker. Per-observation and
# per-margin evidence stream on their own dedicated markers (160 raw
# observation, 161 discovery margin), one record per tick, so the assembler must
# recognize those markers or it silently discards the streamed records.
BREAKAWAY_CAMPAIGN_EVENT_IDS = (
    EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN,
    EVENT_COMMISSION_BREAKAWAY_PROBE_RESULT,
    EVENT_COMMISSION_BREAKAWAY_PROBE_TERMINAL,
    EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN,
    EVENT_COMMISSION_BREAKAWAY_DISCOVERY_TERMINAL,
    EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_PLAN,
    EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_TERMINAL,
    EVENT_COMMISSION_BREAKAWAY_CAMPAIGN_TERMINAL,
    EVENT_COMMISSION_BREAKAWAY_RAW_OBSERVATION,
    EVENT_COMMISSION_BREAKAWAY_DISCOVERY_MARGIN,
    EVENT_COMMISSION_BREAKAWAY_PROBE_DECISION,
    EVENT_COMMISSION_BREAKAWAY_RECOVERY_REST_REJECTED,
    EVENT_COMMISSION_BREAKAWAY_RECOVERY_ORIGIN_MISMATCH,
)

_LEVEL_NAMES = {
    1: "info",
    2: "warn",
    3: "error",
}

_PHASE_CLASS_NAMES = {0: "accel", 1: "cruise", 2: "decel"}

_SUBSYSTEM_NAMES = {
    SUBSYSTEM_COMMISSIONING: "commissioning",
    SUBSYSTEM_STEP_TIMER: "step_timer",
}

_COMMISSIONING_EVENT_NAMES = {
    EVENT_COMMISSION_ENGINE_START: "commission_engine_start",
    EVENT_COMMISSION_PHASE_START: "commission_phase_start",
    EVENT_COMMISSION_PHASE_END: "commission_phase_end",
    EVENT_COMMISSION_ENGINE_TERMINAL: "commission_engine_terminal",
    # Reserved, historical-only: retained so archived MechanicalId captures
    # still decode; production firmware no longer emits these.
    EVENT_COMMISSION_MECHANICAL_ID_RESULT: "commission_mechanical_id_result",
    EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT: "commission_mechanical_id_attempt",
    EVENT_COMMISSION_MECHANICAL_ID_RESPONSE: "commission_mechanical_id_response",
    EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION: "commission_mechanical_id_classification",
    EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT: "commission_velocity_validate_result",
    EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN: "commission_velocity_validate_gain",
    EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS: "commission_velocity_validate_metrics",
    EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS: "commission_velocity_validate_limits",
    EVENT_COMMISSION_VELOCITY_SWEEP_PLAN: "commission_velocity_sweep_plan",
    EVENT_COMMISSION_VELOCITY_SWEEP_OBSERVATION: "commission_velocity_sweep_observation",
    EVENT_COMMISSION_VELOCITY_SWEEP_RUNG: "commission_velocity_sweep_rung",
    EVENT_COMMISSION_VELOCITY_SWEEP_DERIVED: "commission_velocity_sweep_derived",
    EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL: "commission_velocity_sweep_terminal",
    EVENT_COMMISSION_VELOCITY_SWEEP_COMPLETE: "commission_velocity_sweep_complete",
    EVENT_COMMISSION_POSITION_TUNE_RESULT: "commission_position_tune_result",
    EVENT_COMMISSION_POSITION_TUNE_TARGET: "commission_position_tune_target",
    EVENT_COMMISSION_POSITION_TUNE_METRICS: "commission_position_tune_metrics",
    EVENT_COMMISSION_POSITION_TUNE_LIMITS: "commission_position_tune_limits",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_PLAN: "commission_velocity_integral_plan",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_OBSERVATION: "commission_velocity_integral_observation",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_RUNG: "commission_velocity_integral_rung",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_DERIVED: "commission_velocity_integral_derived",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_TERMINAL: "commission_velocity_integral_terminal",
    EVENT_COMMISSION_VELOCITY_INTEGRAL_COMPLETE: "commission_velocity_integral_complete",
    EVENT_COMMISSION_RESTORE_RESULT: "commission_restore_result",
    EVENT_COMMISSION_RESTORE_ERROR_DRIFT: "commission_restore_error_drift",
    EVENT_COMMISSION_RESTORE_DISPOSITION: "commission_restore_disposition",
    EVENT_COMMISSION_RUNG_ORIGIN_RECOVERY: "commission_rung_origin_recovery",
    EVENT_COMMISSION_OUTER_SAFETY_TIMING: "commission_outer_safety_timing",
    EVENT_COMMISSION_VELOCITY_LIMIT_LATCH_TEST: "commission_velocity_limit_latch_test",
    EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_PLAN: "commission_fixed_gain_amplitude_plan",
    EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_OBSERVATION: (
        "commission_fixed_gain_amplitude_observation"
    ),
    EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_AMPLITUDE: "commission_fixed_gain_amplitude_amplitude",
    EVENT_COMMISSION_FIXED_GAIN_AMPLITUDE_TERMINAL: "commission_fixed_gain_amplitude_terminal",
    EVENT_COMMISSION_REST_SELECTION_DIAGNOSTIC: "commission_rest_selection_diagnostic",
    EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN: "commission_breakaway_probe_plan",
    EVENT_COMMISSION_BREAKAWAY_PROBE_RESULT: "commission_breakaway_probe_result",
    EVENT_COMMISSION_BREAKAWAY_PROBE_TERMINAL: "commission_breakaway_probe_terminal",
    EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN: "commission_breakaway_discovery_plan",
    EVENT_COMMISSION_BREAKAWAY_DISCOVERY_TERMINAL: "commission_breakaway_discovery_terminal",
    EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_PLAN: "commission_breakaway_confirmation_plan",
    EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_TERMINAL: "commission_breakaway_confirmation_terminal",
    EVENT_COMMISSION_BREAKAWAY_CAMPAIGN_TERMINAL: "commission_breakaway_campaign_terminal",
    EVENT_COMMISSION_BREAKAWAY_PROBE_DECISION: "commission_breakaway_probe_decision",
    EVENT_COMMISSION_DIAGNOSTIC_WARNING: "commission_diagnostic_warning",
}

_ENGINE_KIND_NAMES = {
    1: "commission",
    2: "calibrate",
    3: "selftest",
    4: "outer_tune",
}

COMMISSION_STATUS_NAMES = {
    0: "accepted",
    1: "accepted_with_warnings",
    2: "inner_only",
    3: "spi",
    4: "adc_fault",
    5: "coil_fault",
    6: "phase_fault",
    7: "encoder_fault",
    8: "electrical_id_failed",
    9: "current_validation_failed",
    10: "mechanical_id_failed",  # Reserved, historical-only; new captures never emit this status.
    11: "velocity_validation_failed",
    12: "position_tune_failed",
    13: "encoder_not_aligned",
    14: "shutdown",
    15: "already_running",
    16: "queue_full",
    17: "safety_envelope_violation",
    18: "chip_reset_detected",
}


class DecodeError(ValueError):
    """Raised when a trace frame cannot be decoded."""


@dataclass(frozen=True)
class FrameHeader:
    """Decoded FTRC frame header."""

    protocol_version: int
    header_len: int
    frame_type: int
    flags: int
    payload_len: int
    source_kind: int
    source_index: int
    sequence: int
    timestamp_ticks: int
    boot_id: int


@dataclass(frozen=True)
class Frame:
    """Decoded FTRC frame."""

    header: FrameHeader
    payload: dict[str, Any]
    raw_payload: bytes


class FrameAssembler:
    """Reassemble complete FTRC frames from 64-byte USB bulk packets."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Append bytes and return all complete frame byte strings."""
        self._buffer.extend(data)
        frames: list[bytes] = []
        while True:
            if len(self._buffer) < HEADER_LEN:
                return frames
            magic = struct.unpack_from("<I", self._buffer, 0)[0]
            if magic != FRAME_MAGIC:
                next_magic = self._buffer.find(struct.pack("<I", FRAME_MAGIC), 1)
                del self._buffer[: next_magic if next_magic != -1 else len(self._buffer)]
                continue
            _magic, version, header_len, _frame_type, _flags, payload_len = struct.unpack_from(
                "<IBBBBH", self._buffer, 0
            )
            if version not in SUPPORTED_PROTOCOL_VERSIONS or header_len != HEADER_LEN:
                raise DecodeError("unsupported frame header while assembling")
            frame_len = header_len + payload_len
            if len(self._buffer) < frame_len:
                return frames
            frames.append(bytes(self._buffer[:frame_len]))
            del self._buffer[:frame_len]


def decode_frame(data: bytes) -> Frame:
    """Decode one complete firmware trace frame."""
    if len(data) < HEADER_LEN:
        raise DecodeError("frame shorter than header")
    (
        magic,
        version,
        header_len,
        frame_type,
        flags,
        payload_len,
        source_kind,
        source_index,
        sequence,
        timestamp_ticks,
        boot_id,
        _reserved,
    ) = struct.unpack_from("<IBBBBHBBIQII", data, 0)
    if magic != FRAME_MAGIC:
        raise DecodeError("bad magic")
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise DecodeError(f"unsupported protocol version {version}")
    if header_len != HEADER_LEN:
        raise DecodeError(f"unsupported header length {header_len}")
    end = header_len + payload_len
    if len(data) < end:
        raise DecodeError("incomplete frame payload")
    raw_payload = data[header_len:end]
    header = FrameHeader(
        protocol_version=version,
        header_len=header_len,
        frame_type=frame_type,
        flags=flags,
        payload_len=payload_len,
        source_kind=source_kind,
        source_index=source_index,
        sequence=sequence,
        timestamp_ticks=timestamp_ticks,
        boot_id=boot_id,
    )
    return Frame(
        header=header,
        payload=_decode_payload(version, frame_type, raw_payload),
        raw_payload=raw_payload,
    )


def _decode_payload(version: int, frame_type: int, payload: bytes) -> dict[str, Any]:
    if frame_type == FRAME_TYPE_CRITICAL:
        return _decode_critical(version, payload)
    if frame_type == FRAME_TYPE_CONTROL_DETAIL:
        return _decode_control_detail(payload)
    if frame_type == FRAME_TYPE_STATUS:
        return _decode_status(version, payload)
    if frame_type == FRAME_TYPE_EVENT:
        return _decode_event(payload)
    if frame_type == FRAME_TYPE_METADATA:
        return _decode_metadata(payload)
    if frame_type == FRAME_TYPE_CURRENT_PATH:
        return _decode_current_path(payload)
    if frame_type == FRAME_TYPE_PHASE_ADVANCE:
        return _decode_phase_advance(payload)
    if frame_type == FRAME_TYPE_VELOCITY_OBSERVATION:
        return _decode_velocity_observation(version, payload)
    if frame_type == FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE:
        try:
            return _decode_velocity_sweep_evidence(payload)
        except DecodeError as legacy_error:
            # The breakaway-campaign family's local tags (spec section 10) are
            # scoped to the marker `event_id` (152-159) that follows a bundle
            # of these records, not globally unique against the older
            # velocity-sweep-evidence tag space -- see
            # `breakaway_trace_reporting.rs`'s module doc. Every breakaway
            # record fails every (kind, length) branch above and lands here.
            # If the breakaway decoder also rejects the payload, the original
            # error is what callers see, preserving corrupt-payload behavior.
            try:
                return decode_breakaway_record(payload, protocol_version=version)
            except DecodeError:
                raise legacy_error from None
    return {"raw": payload}


def _require(payload: bytes, size: int, name: str) -> None:
    if len(payload) < size:
        raise DecodeError(f"{name} payload too short")


def _i32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


def _decode_critical(version: int, payload: bytes) -> dict[str, Any]:
    critical_len = 96 if version == 4 else 88 if version >= 6 else 84
    _require(payload, critical_len, "critical")
    values = struct.unpack_from("<iiiiiiiBBHIiQQIIIII", payload, 0)
    accel_offset = 0
    accel_gain = 0
    accel_flags = 0
    if version >= 2:
        accel_offset, accel_gain, accel_flags, _reserved = struct.unpack_from("<iHBB", payload, 76)
    decoded = {
        "step_position": values[0],
        "tmc_position_actual": values[1],
        "position_error": values[2],
        "target_velocity": values[3],
        "pid_velocity_actual": values[4],
        "cached_pid_velocity_offset": values[5],
        "velocity_error": values[6],
        "enable_and_direction": values[7],
        "sample_flags": values[8],
        "timer_queue_depth": values[9],
        "step_timer_jitter": values[10],
        "min_deadline_slack": values[11],
        "last_served_step_tick": values[12],
        "last_klipper_command_service_tick": values[13],
        "missed_or_deferred_step_counter": values[14],
        "interim_foc_uq_ud": values[17],
        "interim_foc_uq_ud_limited": values[18],
        "cached_accel_torque_offset": accel_offset,
        "accel_feedforward_gain_permille": accel_gain,
        "accel_feedforward_flags": accel_flags,
        "cached_velocity_transient_offset": 0,
        "cached_velocity_clamped": False,
    }
    if version >= 3:
        decoded["interim_pidin_target_torque"] = _i32(values[15])
        decoded["interim_pid_torque_target_actual"] = values[16]
    else:
        decoded["pid_torque_flux_target"] = values[15]
        decoded["pid_torque_flux_actual"] = values[16]
    if version == 4:
        (
            decoded["interim_foc_iq_id"],
            decoded["interim_foc_iwy_iux"],
            decoded["interim_adc_i1_i0"],
        ) = struct.unpack_from("<III", payload, 84)
    if version >= 6:
        decoded["cached_velocity_transient_offset"] = struct.unpack_from("<i", payload, 84)[0]
    if version >= 20:
        decoded["cached_velocity_clamped"] = bool(payload[83] & 0x01)
    return decoded


def _decode_current_path(payload: bytes) -> dict[str, Any]:
    _require(payload, 16, "current-path")
    sample_flags, _r0, _r1, _r2, foc_iq_id, foc_iwy_iux, adc_i1_i0 = struct.unpack_from(
        "<BBBBIII", payload, 0
    )
    return {
        "sample_flags": sample_flags,
        "interim_foc_iq_id": foc_iq_id,
        "interim_foc_iwy_iux": foc_iwy_iux,
        "interim_adc_i1_i0": adc_i1_i0,
    }


def _decode_phase_advance(payload: bytes) -> dict[str, Any]:
    _require(payload, 12, "phase-advance")
    counts, flags, _reserved, gain_ppm, max_counts, deadband = struct.unpack_from(
        "<hBBiHH", payload, 0
    )
    return {
        "phase_advance_counts": counts,
        "phase_advance_flags": flags,
        "phase_advance_gain_ppm": gain_ppm,
        "phase_advance_max_counts": max_counts,
        "phase_advance_deadband": deadband,
    }


def _decode_velocity_observation(version: int, payload: bytes) -> dict[str, Any]:
    _require(payload, 16, "velocity observation")
    target, delta, elapsed_us, stroke_index, direction, flags = struct.unpack_from(
        "<iiIHbB", payload, 0
    )
    diagnostic_value = struct.unpack_from("<i", payload, 18)[0] if len(payload) >= 22 else None
    pid_valid = bool(flags & VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID)
    pptm_valid = bool(flags & VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID)
    if version >= 9 and pid_valid and pptm_valid:
        raise DecodeError("velocity diagnostic kinds must be mutually exclusive")
    return {
        "target_counts_per_ms": target,
        "delta_counts": delta,
        "elapsed_us": elapsed_us,
        "stroke_index": stroke_index,
        "direction": direction,
        "flags": flags,
        "peak_torque_target_abs": (
            struct.unpack_from("<H", payload, 16)[0] if len(payload) >= 18 else None
        ),
        "pid_velocity_actual": diagnostic_value if version < 9 or pid_valid else None,
        "actual_velocity_pptm": diagnostic_value if version >= 9 and pptm_valid else None,
    }


class _EvidenceCursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, fmt: str) -> int:
        if fmt == "U":
            return self._take_u128()
        size = struct.calcsize("<" + fmt)
        if self.offset + size > len(self.payload):
            raise DecodeError("velocity-sweep evidence record is truncated")
        value = struct.unpack_from("<" + fmt, self.payload, self.offset)[0]
        self.offset += size
        return int(value)

    def _take_u128(self) -> int:
        """Read one little-endian unsigned 128-bit field.

        `struct` has no native 128-bit format code, so this mirrors the
        firmware's own `push_u128` (`breakaway_trace_reporting.rs`): a low
        u64 half followed by a high u64 half, reassembled here from the raw
        16-byte little-endian span rather than two separate `struct.unpack`
        calls.
        """
        size = 16
        if self.offset + size > len(self.payload):
            raise DecodeError("velocity-sweep evidence record is truncated")
        value = int.from_bytes(self.payload[self.offset : self.offset + size], "little")
        self.offset += size
        return value

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise DecodeError("velocity-sweep evidence record has trailing bytes")


def _take_fields(cursor: _EvidenceCursor, fields: tuple[tuple[str, str], ...]) -> dict[str, int]:
    return {name: cursor.take(fmt) for name, fmt in fields}


# The only two build_revision values that ever existed: FNV-1a of the crate
# versions "0.1.0" and "0.2.0" (mirrors velocity_measurement's legacy build
# constants). The field was later retired, so a u32 at the build position equal to
# one of these marks a legacy capture; any other value is a build_features
# (u16) + build_identity (u32) pair.
_LEGACY_BUILD_REVISIONS = (0x75536758, 0x05E21F75)


def _take_build_identity(cursor: _EvidenceCursor) -> dict[str, int]:
    """Read the build identity at the cursor, old or new layout.

    Legacy captures carry a single `build_revision` u32; newer captures
    carry `build_features` (u16) then `build_identity` (u32) in its place. A peek
    distinguishes them with no schema or record-length dependency: a new record
    starts with `build_features` (a small mask such as 0x000F) plus a git hash,
    which cannot equal either legacy `build_revision` constant.
    """
    if cursor.offset + 4 > len(cursor.payload):
        raise DecodeError("velocity-sweep evidence record is truncated")
    peeked = struct.unpack_from("<I", cursor.payload, cursor.offset)[0]
    if peeked in _LEGACY_BUILD_REVISIONS:
        return {"build_revision": cursor.take("I")}
    return {"build_features": cursor.take("H"), "build_identity": cursor.take("I")}


def _take_optional_interval(cursor: _EvidenceCursor) -> list[int] | None:
    present = cursor.take("B")
    if present not in (0, 1):
        raise DecodeError("amplitude optional interval marker is invalid")
    return [cursor.take("i"), cursor.take("i")] if present else None


def _take_optional_i32(cursor: _EvidenceCursor) -> int | None:
    present = cursor.take("B")
    if present not in (0, 1):
        raise DecodeError("amplitude optional value marker is invalid")
    return cursor.take("i") if present else None


def _take_fixed_gain_amplitude_direction(cursor: _EvidenceCursor) -> dict[str, Any]:
    direction = {
        "availability": cursor.take("B"),
        "classification": cursor.take("B"),
        "components": [[cursor.take("i"), cursor.take("i")] for _ in range(2)],
        "component_count": cursor.take("B"),
        "collected_mask": cursor.take("B"),
        "eligible_mask": cursor.take("B"),
        "included_mask": cursor.take("B"),
        "signed_target_rate_q": cursor.take("i"),
    }
    if direction["availability"] > 6 or direction["classification"] > 3:
        raise DecodeError("amplitude direction taxonomy is invalid")
    if direction["component_count"] > 2:
        raise DecodeError("amplitude direction has too many components")
    for name in ("collected_mask", "eligible_mask", "included_mask"):
        if direction[name] & ~0xF:
            raise DecodeError("amplitude observation mask exceeds four slots")
    direction["components"] = direction["components"][: direction["component_count"]]
    direction["response_interval_q"] = _take_optional_interval(cursor)
    direction["robust_location_q"] = _take_optional_i32(cursor)
    direction["deficit_interval_q"] = _take_optional_interval(cursor)
    direction["excluded_mask"] = cursor.take("B")
    packed_blocking_levels = cursor.take("H")
    direction["blocking_levels"] = [
        (packed_blocking_levels >> (index * 4)) & 0xF for index in range(4)
    ]
    return direction


def _try_decode_fixed_gain_amplitude(payload: bytes) -> dict[str, Any] | None:
    if not 84 <= len(payload) <= 124:
        return None
    try:
        cursor = _EvidenceCursor(payload)
        if cursor.take("B") != 2:
            return None
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("amplitude_index", "B"),
                ("target_rpm", "i"),
            ),
        )
        if decoded["amplitude_index"] >= 5 or decoded["target_rpm"] <= 0:
            return None
        decoded["directions"] = [
            _take_fixed_gain_amplitude_direction(cursor),
            _take_fixed_gain_amplitude_direction(cursor),
        ]
        cursor.finish()
    except DecodeError:
        return None
    decoded["namespace"] = "fixed_gain_amplitude"
    decoded["record_kind"] = "fixed_gain_amplitude_amplitude"
    return decoded


# Lengths of the Integral ("integral_observation") record from before the drift
# ratio field existed. These must keep decoding: replacing this set instead of
# extending it would silently stop decoding every capture taken before the
# field was added, including the six exported as design evidence.
_INTEGRAL_OBSERVATION_LEGACY_LENGTHS = (46, 47, 77, 80, 96)

# Each legacy length plus the four bytes of the inserted `drift_ratio_ppm` u32.
_INTEGRAL_OBSERVATION_LENGTHS_WITH_DRIFT = (50, 51, 81, 84, 100)


# Reserved, historical-only: never reassign.
RETIRED_VELOCITY_SWEEP_EVIDENCE_KINDS = frozenset({1, 6, 7, 8, *range(12, 21)})


def _decode_velocity_sweep_evidence(payload: bytes) -> dict[str, Any]:
    _require(payload, 1, "velocity-sweep evidence")
    cursor = _EvidenceCursor(payload)
    kind = cursor.take("B")
    if kind in RETIRED_VELOCITY_SWEEP_EVIDENCE_KINDS:
        raise DecodeError(f"retired velocity-sweep evidence kind {kind}")
    if kind == 0 and len(payload) in (43, 56, 72, 73, 75):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("fragment", "B"),
            ),
        )
        fragment = decoded["fragment"]
        if fragment == 0 and len(payload) in (73, 75):
            decoded.update(_take_fields(cursor, (("schema_revision", "H"), ("channel", "B"))))
            decoded.update(_take_build_identity(cursor))
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("acceptance_schema_revision", "H"),
                        ("acceptance_plan_digest", "Q"),
                        ("order", "B"),
                        ("selected_p", "H"),
                        ("selected_i", "H"),
                        ("qualifier_bits", "H"),
                        ("pwm_hz", "I"),
                        ("encoder_counts_per_rev", "I"),
                    ),
                )
            )
            decoded["target_rpm"] = [cursor.take("i") for _ in range(5)]
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("amplitude_count", "B"),
                        ("expected_observations", "H"),
                        ("family_size", "H"),
                    ),
                )
            )
        elif fragment == 1 and len(payload) in (56, 72):
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("moving_stroke_us", "I"),
                        ("zero_settle_us", "I"),
                        ("analysis_budget_us", "I"),
                        ("nominal_workflow_ms", "I"),
                        ("maximum_workflow_ms", "I"),
                        ("recovery_origin_band_counts", "I"),
                    ),
                )
            )
            if len(payload) == 72:
                decoded["recovery_lower_rate_q"] = [
                    cursor.take("Q"),
                    cursor.take("Q"),
                ]
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("recovery_nominal_slot_us", "I"),
                        ("recovery_maximum_slot_us", "I"),
                        ("recovery_slot_count", "B"),
                        ("hard_torque_limit", "H"),
                        ("usable_torque_limit", "H"),
                        ("limit_evidence_capability", "B"),
                        ("recovery_quantization_exposed", "B"),
                        ("probe_constrained_test_point", "B"),
                    ),
                )
            )
        elif (fragment, len(payload)) == (2, 43):
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("requested_velocity_mrev_s", "I"),
                        ("machine_velocity_ceiling_mrev_s", "I"),
                        ("requested_velocity_source", "B"),
                        ("max_stroke_travel_mrev", "I"),
                        ("settle_travel_reserve_mrev", "I"),
                        ("negative_position_headroom_mrev", "I"),
                        ("positive_position_headroom_mrev", "I"),
                        ("max_duration_ms", "H"),
                    ),
                )
            )
        else:
            raise DecodeError("amplitude plan fragment shape is invalid")
        if "order" in decoded:
            # Slot order rides in the high nibble of the amplitude-order byte, so
            # forward-first runs keep the byte they had before it was recorded.
            # `order` stays raw because the plan digest hashes the packed value.
            packed = int(decoded["order"])
            amplitude_order = packed & 0x0F
            slot_order = packed >> 4
            if amplitude_order not in (1, 2):
                raise DecodeError("amplitude plan amplitude order is invalid")
            if slot_order not in (0, 1):
                raise DecodeError("amplitude plan slot order is invalid")
            decoded["amplitude_order"] = amplitude_order
            decoded["slot_order"] = slot_order
        decoded["namespace"] = "fixed_gain_amplitude"
        decoded["record_kind"] = "fixed_gain_amplitude_plan"
    elif kind == 0 and len(payload) in (65, 82, 98):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("plan_digest", "Q"),
                ("fixed_p", "H"),
                ("planned_velocity_mrev_s", "I"),
                ("target_velocity_rpm", "i"),
                ("pwm_hz", "I"),
                ("i_start", "H"),
                ("positive_rung_count", "B"),
                ("family_size", "H"),
                ("moving_stroke_us", "I"),
                ("zero_settle_us", "I"),
                ("analysis_budget_us", "I"),
                ("maximum_workflow_ms", "I"),
                ("recovery_origin_band_counts", "I"),
                ("recovery_nominal_slot_us", "I"),
                ("recovery_maximum_slot_us", "I"),
                ("recovery_slot_count", "B"),
                ("hard_torque_limit", "H"),
                ("usable_torque_limit", "H"),
            ),
        )
        decoded["evidence_sequence"] = 1
        decoded["record_kind"] = "integral_plan_marker"
        if len(payload) in (82, 98):
            if cursor.take("B") != 1:
                raise DecodeError("combined integral plan marker is invalid")
            decoded["combined_selected_response"] = True
            decoded["selected_response_intervals_q"] = [
                [cursor.take("i"), cursor.take("i")] for _ in range(2)
            ]
            if len(payload) == 98:
                decoded["recovery_lower_rate_q"] = [
                    cursor.take("Q"),
                    cursor.take("Q"),
                ]
    elif kind == 0:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest_low", "I"),
                ("plan_digest_high", "I"),
            ),
        )
        fragment = cursor.take("B")
        if fragment == 0:
            decoded.update(_take_build_identity(cursor))
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("schema_revision", "H"),
                        ("channel", "B"),
                        ("proportional_plan_digest_low", "I"),
                        ("proportional_plan_digest_high", "I"),
                        ("fixed_p", "H"),
                    ),
                )
            )
            decoded["directional_memberships"] = [cursor.take("I") for _ in range(2)]
            if decoded["schema_revision"] >= 7:
                decoded["proportional_joint_membership"] = cursor.take("I")
            decoded["joint_membership"] = cursor.take("I")
            if decoded["schema_revision"] >= 4:
                decoded["directional_validity"] = [cursor.take("B") for _ in range(2)]
                decoded["reduced_margin"] = bool(cursor.take("B"))
            decoded["directional_pooled_intervals"] = [
                [cursor.take("i"), cursor.take("i")] for _ in range(2)
            ]
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("planned_velocity_mrev_s", "I"),
                        ("target_velocity_rpm", "i"),
                        ("pwm_hz", "I"),
                        ("encoder_counts_per_rev", "I"),
                        ("positive_rung_count", "B"),
                        ("family_size", "H"),
                        ("total_rung_count", "B"),
                        ("expected_observations", "H"),
                        ("hard_torque_limit", "H"),
                        ("usable_torque_limit", "H"),
                    ),
                )
            )
            decoded["proportional_plan_digest"] = decoded["proportional_plan_digest_low"] | (
                decoded["proportional_plan_digest_high"] << 32
            )
            if decoded["schema_revision"] >= 3 or len(payload) == 88:
                decoded["limit_evidence_capability"] = cursor.take("B")
            decoded["combined_selected_response"] = False
            schema_revision = decoded["schema_revision"]
            if schema_revision in _RESPONSE_LADDER_INTEGRAL_SCHEMA_REVISIONS:
                decoded["i_top_cap"] = cursor.take("H")
            decoded["record_kind"] = "integral_plan_core"
        elif fragment == 1:
            decoded["positive_rung_count"] = cursor.take("B")
            if decoded["positive_rung_count"] > 30:
                raise DecodeError("velocity-integral plan exceeds 30 positive rungs")
            decoded["positive_i"] = [
                cursor.take("H") for _ in range(decoded["positive_rung_count"])
            ]
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("moving_stroke_us", "I"),
                        ("zero_settle_us", "I"),
                        ("analysis_budget_us", "I"),
                    ),
                )
            )
            if len(payload) - cursor.offset == 39:
                decoded["slot_transition_allowance_us"] = cursor.take("I")
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("nominal_workflow_ms", "I"),
                        ("maximum_workflow_ms", "I"),
                        ("requested_velocity_mrev_s", "I"),
                        ("machine_velocity_ceiling_mrev_s", "I"),
                        ("requested_velocity_source", "B"),
                        ("max_stroke_travel_mrev", "I"),
                        ("settle_travel_reserve_mrev", "I"),
                        ("negative_position_headroom_mrev", "I"),
                        ("positive_position_headroom_mrev", "I"),
                        ("max_duration_ms", "H"),
                    ),
                )
            )
            decoded["record_kind"] = "integral_plan_schedule"
        elif fragment == 2:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("recovery_origin_band_counts", "I"),
                        ("recovery_nominal_slot_us", "I"),
                        ("recovery_maximum_slot_us", "I"),
                        ("recovery_slot_count", "B"),
                    ),
                )
            )
            if cursor.offset < len(payload):
                decoded["flags"] = cursor.take("B")
                # Bits 2 and 3 carry the observation slot order. Captures made
                # before the field existed leave them clear, which is
                # forward-first.
                decoded["slot_order"] = (int(decoded["flags"]) >> 2) & 0b11
            decoded["record_kind"] = "integral_plan_recovery"
        else:
            raise DecodeError(f"unknown velocity-integral plan fragment {fragment}")
        decoded["plan_fragment"] = fragment
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
    elif kind == 2 and (amplitude := _try_decode_fixed_gain_amplitude(payload)) is not None:
        return amplitude
    elif kind == 2 and len(payload) in (
        *_INTEGRAL_OBSERVATION_LEGACY_LENGTHS,
        *_INTEGRAL_OBSERVATION_LENGTHS_WITH_DRIFT,
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest_low", "I"),
                ("plan_digest_high", "I"),
                ("rung_index", "B"),
                ("slot", "B"),
                ("i_raw", "H"),
                ("direction", "B"),
                ("classification", "B"),
                ("initial_lead_in", "B"),
                ("rate_low", "i"),
                ("rate_mean", "i"),
                ("rate_high", "i"),
                ("deficit_low", "i"),
                ("deficit_high", "i"),
                ("rate_shift", "B"),
                ("predicted_torque_target_abs", "H"),
            ),
        )
        # Firmware inserts the u32 drift ratio between predicted_torque_target_abs
        # and stationarity_code. Captures from before the field existed have no
        # such value, and that absence must decode as None, not as the u32::MAX
        # "no finite reportable ratio" sentinel the firmware also emits.
        if len(payload) in _INTEGRAL_OBSERVATION_LENGTHS_WITH_DRIFT:
            decoded["drift_ratio_ppm"] = cursor.take("I")
        else:
            decoded["drift_ratio_ppm"] = None
        decoded["stationarity_code"] = cursor.take("B")
        shift = decoded["rate_shift"]
        decoded["settled_rate_interval_q"] = [
            decoded["rate_low"] << shift,
            decoded["rate_high"] << shift,
        ]
        decoded["settled_rate_mean_q"] = decoded["rate_mean"] << shift
        decoded["deficit_interval_q"] = [
            decoded["deficit_low"] << shift,
            decoded["deficit_high"] << shift,
        ]
        if decoded["stationarity_code"] == 0:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("suffix_len", "H"),
                        ("selected_level", "B"),
                        ("selected_blocks", "I"),
                        ("slope_mantissa", "i"),
                        ("slope_shift", "B"),
                        ("slope_half_width_mantissa", "I"),
                        ("slope_half_width_shift", "B"),
                        ("lag_one_q", "i"),
                        ("lag_one_half_width_q", "I"),
                        ("residual_mantissa", "I"),
                        ("residual_shift", "B"),
                        ("tested_suffixes", "B"),
                    ),
                )
            )
        elif cursor.offset < len(payload):
            # Firmware emitting gate statistics for rejected tails. Captures made
            # before that keep the shorter record and carry no presence flag.
            decoded["rejection_present"] = cursor.take("B")
            if decoded["rejection_present"] not in (0, 1):
                raise DecodeError("observation rejection presence is invalid")
            if decoded["rejection_present"] == 1:
                decoded.update(
                    _take_fields(
                        cursor,
                        (
                            ("rejection_suffix_len", "H"),
                            ("rejection_usable_plateau", "B"),
                            ("rejection_slope_passed", "B"),
                            ("rejection_lag_one_passed", "B"),
                        ),
                    )
                )
                decoded["rejection_settled_rate_interval_q"] = [
                    cursor.take("q"),
                    cursor.take("q"),
                ]
                decoded["rejection_slope_margin_q"] = cursor.take("q")
                decoded["rejection_lag_one_margin_q"] = cursor.take("i")
                # Present from the second revision of this block. A margin is
                # half_width - |slope| and recovers neither term on its own.
                if cursor.offset < len(payload):
                    decoded["rejection_slope_q"] = cursor.take("q")
                    decoded["rejection_slope_half_width_q"] = cursor.take("Q")
                if decoded["rejection_suffix_len"] not in (64, 128, 256):
                    raise DecodeError("observation rejection suffix is not in the family")
                interval = decoded["rejection_settled_rate_interval_q"]
                if interval[0] > interval[1]:
                    raise DecodeError("observation rejection interval is reversed")
                if (
                    decoded["rejection_slope_passed"] == 1
                    and decoded["rejection_lag_one_passed"] == 1
                ):
                    raise DecodeError("a rejected tail cannot pass every gate")
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["record_kind"] = "integral_observation"
    elif kind == 3 and len(payload) == 67:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("schema_revision", "H"),
                ("acceptance_plan_digest", "Q"),
                ("plan_digest", "Q"),
                ("digest", "Q"),
                ("outcome", "B"),
                ("cause", "B"),
            ),
        )
        decoded["attempted_masks"] = [cursor.take("B") for _ in range(2)]
        decoded["eligible_masks"] = [cursor.take("B") for _ in range(2)]
        decoded["current_terminus_masks"] = [cursor.take("B") for _ in range(2)]
        decoded["unattempted_masks"] = [cursor.take("B") for _ in range(2)]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("expected_observations", "H"),
                    ("emitted_observations", "H"),
                    ("expected_amplitudes", "B"),
                    ("emitted_amplitudes", "B"),
                    ("qualifier_bits", "H"),
                    ("run_started_us", "Q"),
                    ("run_completed_us", "Q"),
                ),
            )
        )
        decoded["outcome_name"] = {
            0: "Complete",
            1: "Inconclusive",
            2: "Fault",
            3: "Failed",
        }.get(decoded["outcome"], "Unknown")
        decoded["cause_name"] = {
            0: "None",
            1: "InsufficientSharedFloor",
            2: "RecoveryUnavailable",
            3: "MissingAcceptancePoint",
            4: "AcceptancePointIdentityMismatch",
            5: "AmplitudeOrderMismatch",
            6: "EvidenceCapacity",
            7: "EvidenceIntegrity",
        }.get(decoded["cause"], "Unknown")
        decoded["namespace"] = "fixed_gain_amplitude"
        decoded["record_kind"] = "fixed_gain_amplitude_terminal"
    elif kind == 3 and len(payload) in (70, 71, 83, 84):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest_low", "I"),
                ("plan_digest_high", "I"),
                ("rung_index", "B"),
                ("i_raw", "H"),
            ),
        )
        if len(payload) in (83, 84):
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("tau_us", "Q"),
                        ("moving_horizon_us", "I"),
                        ("horizon_class", "B"),
                    ),
                )
            )
        directions = []
        for _ in range(2):
            direction = _take_fields(
                cursor,
                (
                    ("direction", "B"),
                    ("kind", "B"),
                    ("classification", "B"),
                    ("component_count", "B"),
                    ("usable_mask", "B"),
                    ("included_mask", "B"),
                ),
            )
            direction["observation_classes"] = [cursor.take("B") for _ in range(4)]
            components = [[cursor.take("i"), cursor.take("i")] for _ in range(2)]
            if direction["component_count"] > 2:
                raise DecodeError("velocity-integral rung has too many components")
            direction["components"] = components[: direction["component_count"]]
            directions.append(direction)
        decoded["directions"] = directions
        # Diagnostic trailing byte (schema >= the target-terminus build): the
        # recorder's one-based target terminus after this rung, zero for none.
        # Absent on older captures, which decode as None.
        if cursor.offset < len(payload):
            target_terminus_plus_one = cursor.take("B")
            decoded["target_terminus"] = (
                None if target_terminus_plus_one == 0 else target_terminus_plus_one - 1
            )
        else:
            decoded["target_terminus"] = None
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["record_kind"] = "integral_rung"
    elif kind == 4 and len(payload) in (58, 59, 61, 62):
        fields = [
            ("run_sequence", "I"),
            ("evidence_sequence", "H"),
            ("plan_digest_low", "I"),
            ("plan_digest_high", "I"),
            ("digest_low", "I"),
            ("digest_high", "I"),
            ("outcome", "B"),
            ("cause", "B"),
        ]
        if len(payload) == 59:
            fields.append(("recovery_unavailable", "B"))
        elif len(payload) in (61, 62):
            fields.append(("recovery_flags", "B"))
        fields.extend(
            (
                ("expected_observations", "H"),
                ("emitted_observations", "H"),
                ("expected_rungs", "H"),
                ("emitted_rungs", "H"),
            )
        )
        decoded = _take_fields(
            cursor,
            tuple(fields),
        )
        decoded["eligible_masks"] = [cursor.take("I") for _ in range(2)]
        if len(payload) in (61, 62):
            decoded["opening_available_mask"] = cursor.take("B")
            decoded["bookend_available_mask"] = cursor.take("B")
            decoded["recovery_unavailable"] = decoded["recovery_flags"] & 1
        current_terminus_plus_one = cursor.take("B")
        decoded["current_terminus"] = (
            None if current_terminus_plus_one == 0 else current_terminus_plus_one - 1
        )
        if len(payload) == 58:
            decoded["recovery_unavailable"] = 0
        decoded["run_started_us"] = cursor.take("I") | (cursor.take("I") << 32)
        decoded["run_completed_us"] = cursor.take("I") | (cursor.take("I") << 32)
        if len(payload) == 62:
            decoded["cause_namespace"] = cursor.take("B")
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["digest"] = decoded["digest_low"] | (decoded["digest_high"] << 32)
        decoded["record_kind"] = "integral_terminal"
    elif kind == 5 and len(payload) != 19:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("positive_rung_index", "B"),
                ("slot", "B"),
                ("i_raw", "H"),
                ("direction", "b"),
                ("active_sample_count", "H"),
                ("active_duration_us", "I"),
                ("active_net_displacement_counts", "i"),
                ("active_minimum_displacement_counts", "i"),
                ("active_maximum_displacement_counts", "i"),
                ("active_transition_count", "H"),
                ("first_transition_sample_index", "H"),
                ("first_transition_elapsed_us", "I"),
                ("first_transition_delta_counts", "i"),
                ("first_transition_displacement_counts", "i"),
                ("last_transition_sample_index", "H"),
                ("last_transition_elapsed_us", "I"),
                ("last_transition_delta_counts", "i"),
                ("last_transition_displacement_counts", "i"),
                ("active_stationarity_code", "B"),
            ),
        )
        if decoded["active_stationarity_code"] == 0:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("active_suffix_len", "H"),
                        ("active_selected_level", "B"),
                        ("active_selected_blocks", "I"),
                        ("active_slope_mantissa", "i"),
                        ("active_slope_shift", "B"),
                        ("active_slope_half_width_mantissa", "I"),
                        ("active_slope_half_width_shift", "B"),
                        ("active_lag_one_q", "i"),
                        ("active_lag_one_half_width_q", "I"),
                        ("active_residual_mantissa", "I"),
                        ("active_residual_shift", "B"),
                        ("active_tested_suffixes", "B"),
                    ),
                )
            )
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("clamp_limit", "H"),
                    ("moving_pid_output_peak", "H"),
                    ("zero_pid_output_peak", "H"),
                    ("moving_status_flags", "I"),
                    ("zero_status_flags", "I"),
                    ("limit_evidence_capability", "B"),
                    ("current_contact", "B"),
                    ("cleanup_confirmed", "B"),
                ),
            )
        )
        decoded["record_kind"] = "integral_rest_boundary"
    elif kind == 9 and len(payload) == 55:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("stage", "B"),
                ("rung_index", "B"),
                ("velocity_p", "H"),
                ("planned_velocity_mrev_s", "I"),
                ("start_offset_counts", "i"),
                ("closest_offset_counts", "i"),
                ("final_offset_counts", "i"),
                ("origin_band_counts", "I"),
                ("lower_rate_q", "Q"),
                ("moving_duration_us", "I"),
                ("settle_duration_us", "I"),
                ("total_duration_us", "I"),
                ("peak_torque_target_abs", "H"),
                ("binding_source", "B"),
                ("outcome", "B"),
            ),
        )
        decoded["record_kind"] = "rung_origin_recovery"
    elif kind == 10 and len(payload) == 89:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("stage", "B"),
                ("rung_index", "B"),
                ("velocity_p", "H"),
                ("plan_digest", "Q"),
                ("start_offset_counts", "i"),
                ("closest_offset_counts", "i"),
                ("final_offset_counts", "i"),
                ("planned_return_distance_counts", "I"),
                ("first_in_band_offset_counts", "i"),
                ("displacement_to_band_counts", "i"),
                ("entry_to_band_duration_us", "I"),
                ("lead_in_net_displacement_counts", "i"),
                ("lead_in_displacement_span_counts", "I"),
                ("scored_net_displacement_counts", "i"),
                ("scored_min_displacement_counts", "i"),
                ("scored_max_displacement_counts", "i"),
                ("stationarity_code", "B"),
                ("interval_present", "B"),
            ),
        )
        decoded["exact_velocity_interval_q"] = [cursor.take("q"), cursor.take("q")]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("peak_torque_target_abs", "H"),
                    ("current_contact_code", "B"),
                    ("disposition", "B"),
                ),
            )
        )
        stationarity = int(decoded["stationarity_code"])
        interval_present = int(decoded["interval_present"])
        interval = decoded["exact_velocity_interval_q"]
        if stationarity not in range(6):
            raise DecodeError("recovery rest stationarity code is invalid")
        if interval_present not in (0, 1) or interval_present != int(stationarity == 0):
            raise DecodeError("recovery rest interval presence is invalid")
        if interval_present == 0 and interval != [0, 0]:
            raise DecodeError("excluded recovery rest carries an interval")
        if interval_present == 1 and interval[0] > interval[1]:
            raise DecodeError("recovery rest interval is reversed")
        if int(decoded["disposition"]) not in (0, 1):
            raise DecodeError("recovery rest disposition is invalid")
        if int(decoded["current_contact_code"]) not in range(4):
            raise DecodeError("recovery rest current contact is invalid")
        decoded["record_kind"] = "recovery_rest"
    elif kind == 11 and len(payload) in (78, 80):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("stage", "B"),
                ("owner", "B"),
                ("rung_index", "B"),
                ("direction", "B"),
                ("slot", "B"),
                ("plan_digest", "Q"),
            ),
        )
        decoded["structural_codes"] = [cursor.take("B") for _ in range(3)]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("interval_present_mask", "B"),
                    ("usable_mask", "B"),
                    ("zero_compatible_mask", "B"),
                    ("selected_mask", "B"),
                    ("selected_suffix_len", "H"),
                    ("disposition", "B"),
                ),
            )
        )
        # Retained-sample-count instrumentation shipped after the original
        # 78-byte record; older captures carry no such field.
        decoded["retained_sample_count"] = cursor.take("H") if len(payload) == 80 else None
        decoded["intervals_q"] = [[cursor.take("q"), cursor.take("q")] for _ in range(3)]

        stage = int(decoded["stage"])
        owner = int(decoded["owner"])
        direction = int(decoded["direction"])
        if stage not in (0, 1) or owner not in range(9):
            raise DecodeError("scored-rest stage or owner is invalid")
        expected_stage = 0 if owner <= 2 else 1
        if stage != expected_stage:
            raise DecodeError("scored-rest owner does not belong to its stage")
        if owner in (1, 2, 5, 8):
            if direction != 0xFF or int(decoded["slot"]) != 0xFF:
                raise DecodeError("scored-rest recovery position is invalid")
        elif direction not in (0, 1):
            raise DecodeError("scored-rest direction is invalid")

        masks = [
            int(decoded["interval_present_mask"]),
            int(decoded["usable_mask"]),
            int(decoded["zero_compatible_mask"]),
            int(decoded["selected_mask"]),
        ]
        if any(mask & ~0b111 for mask in masks):
            raise DecodeError("scored-rest mask is out of range")

        expected_interval_mask = 0
        expected_usable_mask = 0
        expected_zero_mask = 0
        intervals = decoded["intervals_q"]
        for index, code in enumerate(decoded["structural_codes"]):
            bit = 1 << index
            if code == 5:
                raise DecodeError("scored-rest arithmetic result cannot be serialized")
            if code not in (0, 1, 2, 3, 4, 6):
                raise DecodeError("scored-rest structural code is invalid")
            interval_expected = code in (0, 3, 4)
            if interval_expected:
                expected_interval_mask |= bit
                low, high = intervals[index]
                if low > high:
                    raise DecodeError("scored-rest interval is reversed")
                if code == 0:
                    expected_usable_mask |= bit
                    if low <= 0 <= high:
                        expected_zero_mask |= bit
            elif intervals[index] != [0, 0]:
                raise DecodeError("excluded scored-rest suffix carries an interval")
        if int(decoded["interval_present_mask"]) != expected_interval_mask:
            raise DecodeError("scored-rest interval-presence mask is inconsistent")
        expected = select_causal_rest_suffixes(
            decoded["structural_codes"],
            intervals,
            expected_interval_mask,
            one_count_fallback=int(decoded["disposition"]) == 1,
        )
        if expected["usable_mask"] != expected_usable_mask:
            raise DecodeError("scored-rest selector reconstruction failed")
        if int(decoded["usable_mask"]) != expected["usable_mask"]:
            raise DecodeError("scored-rest usable mask is inconsistent")
        if expected["zero_compatible_mask"] != expected_zero_mask:
            raise DecodeError("scored-rest selector reconstruction failed")
        if int(decoded["zero_compatible_mask"]) != expected["zero_compatible_mask"]:
            raise DecodeError("scored-rest zero-compatible mask is inconsistent")

        disposition = int(decoded["disposition"])
        if disposition == 0:
            if (
                expected["selected_mask"] == 0
                or int(decoded["selected_mask"]) != expected["selected_mask"]
                or int(decoded["selected_suffix_len"]) != expected["selected_suffix_len"]
                or disposition != expected["disposition"]
            ):
                raise DecodeError("scored-rest statistical selection is inconsistent")
        elif disposition in (1, 2):
            if (
                expected["selected_mask"] != 0
                or int(decoded["selected_mask"]) != 0
                or int(decoded["selected_suffix_len"]) != 0
                or disposition != expected["disposition"]
            ):
                raise DecodeError("scored-rest terminal selection is inconsistent")
        else:
            raise DecodeError("scored-rest disposition is invalid")
        decoded["conditioned_rest_evidence"] = disposition == 0
        decoded["record_kind"] = "rest_selection"
    elif kind == 30:
        decoded = _take_fields(
            cursor,
            (
                ("engine_sequence", "I"),
                ("run_sequence", "I"),
                ("sample_count", "I"),
            ),
        )
        decoded["bin_counts"] = [cursor.take("I") for _ in range(4)]
        decoded["max_realized_gap_us"] = cursor.take("I")
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("max_context_present", "B"),
                    ("max_stage", "B"),
                    ("max_phase", "B"),
                    ("max_rung", "B"),
                    ("max_slot", "B"),
                    ("max_target_zero", "B"),
                    ("zero_count", "B"),
                    ("over_bound_count", "B"),
                ),
            )
        )
        expected_samples = (
            sum(decoded["bin_counts"]) + decoded["zero_count"] + decoded["over_bound_count"]
        )
        if decoded["sample_count"] != expected_samples:
            raise DecodeError("outer-safety timing sample count does not match its bins")
        decoded["max_context_present"] = bool(decoded["max_context_present"])
        decoded["max_target_zero"] = bool(decoded["max_target_zero"])
        decoded["record_kind"] = "outer_safety_timing"
    elif kind == 31:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("stage", "B"),
                ("rung_index", "B"),
                ("slot", "B"),
                ("clamp_limit", "H"),
                ("clamp_readback", "H"),
                ("moving_pid_output_peak", "H"),
                ("zero_pid_output_peak", "H"),
                ("moving_status_flags", "I"),
                ("zero_status_flags", "I"),
                ("capability", "B"),
                ("contact", "B"),
            ),
        )
        moving_clear = (
            decoded["contact"] != 1 and decoded["moving_pid_output_peak"] < decoded["clamp_limit"]
        )
        decoded["moving_clip_absence_proven"] = decoded["capability"] == 1 and moving_clear
        decoded["moving_clip_uncertain"] = decoded["capability"] == 0 and moving_clear
        decoded["record_kind"] = "velocity_primitive_current"
    elif kind == 40:
        decoded = _take_fields(
            cursor,
            (
                ("channel", "B"),
                ("outcome", "B"),
                ("acceptance_source_code", "B"),
                ("phase_live", "B"),
                ("breakaway_acceptance", "B"),
                ("execution_plan_present", "B"),
                ("authority_present", "B"),
                ("rest_boundary_rung_plus_one", "B"),
                ("rest_boundary_slot_plus_one", "B"),
                ("installer_result", "B"),
            ),
        )
        decoded["record_kind"] = "integral_promotion_diagnostic"
    else:
        raise DecodeError(f"unknown velocity-sweep evidence kind {kind}")
    cursor.finish()
    return decoded


# Unified wire terminal-cause taxonomy spanning every breakaway probe,
# discovery, and confirmation terminal variant plus the campaign-level accept
# outcome (`BreakawayTerminalCause` in
# `breakaway_reporting.rs`). One shared table, matching the firmware's own
# rationale for not needing three separate cause enums.
BREAKAWAY_TERMINAL_CAUSE_NAMES = {
    0: "none",
    1: "probe_no_motion_at_authority",
    2: "probe_unstable_breakaway",
    3: "probe_directional_breakaway_disagreement",
    4: "probe_evidence_excluded",
    5: "probe_authority_failure",
    6: "probe_safety_fault",
    7: "discovery_target_band_skipped",
    8: "discovery_current_headroom_before_band",
    9: "discovery_insufficient_additive_span",
    10: "discovery_additive_ladder_resolution_budget_exceeded",
    11: "discovery_evidence_excluded",
    12: "discovery_safety_fault",
    13: "confirmation_target_band_unconfirmable_from_discovery",
    14: "confirmation_target_band_not_reached_at_authority",
    15: "confirmation_response_location",
    16: "confirmation_excessive_uncertainty",
    17: "confirmation_loss_of_consensus",
    18: "confirmation_current_authority",
    19: "confirmation_evidence_excluded",
    20: "confirmation_safety_fault",
    21: "accepted",
    22: "probe_excursion_evidence_invalid",
    23: "probe_internal_fault",
    24: "discovery_internal_fault",
    25: "confirmation_internal_fault",
    26: "confirmation_integral_plan_refused",
    27: "confirmation_no_transition_capable_candidate",
}


def _breakaway_terminal_cause_name(cause: int, protocol_version: int) -> str:
    if cause == 1 and protocol_version >= 15:
        return "probe_no_repeatable_motion_at_authority"
    if cause == 1 and protocol_version == 14:
        return "probe_no_productive_response_at_authority"
    return BREAKAWAY_TERMINAL_CAUSE_NAMES[cause]


def _breakaway_terminal_cause_is_valid(cause: object, protocol_version: int) -> bool:
    if not isinstance(cause, int):
        return False
    if protocol_version >= 19:
        # 26 is the confirmed-but-Integral-refused cause.
        return cause in (0, 1) or 5 <= cause <= 26
    if protocol_version >= 16:
        # 23-25 split the non-safety internal faults out of the `*_safety_fault`
        # causes, which previously reported `has_safety_fault = 0`.
        return cause in (0, 1) or 5 <= cause <= 25
    if protocol_version == 15:
        return cause in (0, 1) or 5 <= cause <= 22
    return 0 <= cause <= 21


# `CommissionError::status_code()` values reachable from a breakaway terminal.
# Deliberately a subset, not a copy of the host's full table: klipper-foci's
# `COMMISSION_REASON_NAMES` is authoritative and duplicating all of it here would
# drift. An unlisted code decodes with `error_code` set and `error_code_name`
# `None`, so nothing is lost.
COMMISSION_REASON_NAMES = {
    3: "spi",
    11: "velocity_validation_failed",
    17: "safety_envelope_violation",
    47: "invalid_schedule",
    52: "analysis_overrun",
    53: "velocity_rest_not_confirmed",
    54: "invalid_current_limit",
    55: "evidence_buffer_full",
    56: "rung_index_out_of_range",
    57: "plan_contract_changed",
    58: "recovery_plan_failed",
    59: "encoder_domain_invalid",
    60: "missing_capture_state",
    61: "search_grid_invalid",
    62: "target_rate_invalid",
    63: "zero_elapsed_sample",
    64: "campaign_plan_invalid",
    65: "integral_authority_denied",
    66: "discovery_band_not_selected",
    67: "schedule_period_zero",
    68: "schedule_early_invocation",
    69: "schedule_deadline_overflow",
    70: "primitive_already_open",
    71: "primitive_start_in_future",
    72: "recovery_origin_mismatch",
}

# Persistent `BreakawayCheckpoint` codes. Values are append-only because they
# appear both in warm-reset breadcrumbs and in the handled probe terminal.
BREAKAWAY_CHECKPOINT_NAMES = {
    0: "none",
    1: "campaign_installed",
    2: "safety_configured",
    3: "gain_programming",
    4: "target_command",
    5: "capture",
    6: "event_routing",
    7: "campaign_terminal_handled",
    8: "integral_handed_off",
    9: "probe_safety_encoder_read",
    10: "probe_gain_write",
    11: "probe_ramp_mode_write",
    12: "probe_gain_integrator_clear",
    13: "probe_target_baseline_read",
    14: "probe_clamp_install",
    15: "probe_current_primitive_start",
    16: "probe_target_write",
    17: "probe_moving_encoder_sample",
    18: "probe_moving_current_input_sample",
    19: "probe_moving_current_output_sample",
    20: "probe_moving_boundary",
    21: "probe_zero_target_write",
    22: "probe_zero_integrator_clear",
    23: "probe_settle_baseline_read",
    24: "probe_settle_encoder_sample",
    25: "probe_settle_current_input_sample",
    26: "probe_settle_current_output_sample",
    27: "probe_zero_boundary",
    28: "recovery_gain_write",
    29: "recovery_moving_integrator_clear",
    30: "recovery_current_primitive_start",
    31: "recovery_target_write",
    32: "recovery_moving_safety_sample",
    33: "recovery_moving_current_input_sample",
    34: "recovery_moving_current_output_sample",
    35: "recovery_moving_boundary",
    36: "recovery_zero_target_write",
    37: "recovery_settle_integrator_clear",
    38: "recovery_settle_safety_sample",
    39: "recovery_settle_current_input_sample",
    40: "recovery_settle_current_output_sample",
    41: "recovery_zero_boundary",
    42: "probe_capture_interval_refusal",
    43: "probe_excursion_fold_refusal",
}

BREAKAWAY_PROBE_EXCLUSION_REASON_NAMES = {
    0: "none",
    1: "invalid_moving_evidence",
    2: "moving_stationarity_excluded",
    3: "settle_rest_unconfirmed",
    4: "recovery_rest_unconfirmed",
    5: "moving_no_blocking_plateau",
    6: "moving_too_few_decorrelated_blocks",
    7: "moving_resolved_slope",
    8: "moving_structured_residual",
    9: "moving_stationarity_arithmetic",
    10: "moving_excessive_drift",
}

# Local wire tags for the breakaway-campaign canonical-record family
# (`breakaway_trace_reporting.rs`). These are scoped to the surrounding
# `EVENT_COMMISSION_BREAKAWAY_*` marker, not globally unique against the
# older velocity-sweep-evidence tag space -- see `decode_breakaway_record`.
BREAKAWAY_TAG_PROBE_PLANNED = 0
BREAKAWAY_TAG_PROBE_RESULT = 1
BREAKAWAY_TAG_PROBE_TERMINAL = 2
BREAKAWAY_TAG_DISCOVERY_PLANNED = 3
BREAKAWAY_TAG_CEILING_SOURCE = 4
# Tag 5 previously carried a per-rung resolved-ladder record (`p_raw` only);
# that fan-out was dropped in favor of the raw-observation stream that lets
# an offline decoder independently recompute nomination margins and
# confirmation containment instead of trusting firmware's reduced bounds.
# See `decode_breakaway_record`'s tag-5 branch and
# `breakaway_trace_reporting.rs::encode_raw_observation`'s own doc comment.
BREAKAWAY_TAG_RAW_OBSERVATION = 5
BREAKAWAY_TAG_DISCOVERY_TERMINAL = 6
BREAKAWAY_TAG_DISCOVERY_RUNG_MARGIN = 7
BREAKAWAY_TAG_RUNG_ZERO_DIAGNOSTIC = 8
BREAKAWAY_TAG_CONFIRMATION_PLANNED = 9
BREAKAWAY_TAG_CONFIRMATION_TERMINAL = 10
BREAKAWAY_TAG_CONFIRMATION_OBSERVATION = 11
BREAKAWAY_TAG_CAMPAIGN_TERMINAL = 12
BREAKAWAY_TAG_PROBE_DECISION = 13
# Protocol 18. Emitted only when the scored-rest selector rejected a recovery
# rest window, so an accepted campaign carries none and a failing one
# normally carries exactly the single rejection that ended it.
BREAKAWAY_TAG_RECOVERY_REST_REJECTED = 14
# Protocol 18. Emitted immediately before a `RecoveryOriginMismatch` fault,
# carrying the measurement the fault itself does not.
BREAKAWAY_TAG_RECOVERY_ORIGIN_MISMATCH = 15


def decode_breakaway_record(
    payload: bytes, *, protocol_version: int = PROTOCOL_VERSION
) -> dict[str, Any]:
    """Decode one breakaway-campaign canonical evidence record.

    Dispatches purely on the record's local tag (`payload[0]`) and exact
    length, mirroring the firmware's own golden-byte tests
    (`breakaway_trace_reporting_tests.rs`). Unlike
    `_decode_velocity_sweep_evidence`, this function is never the sole
    authority for a `FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE` payload: the local
    tag space is scoped to the surrounding `EVENT_COMMISSION_BREAKAWAY_*`
    marker (152-162), not globally unique. A caller that already knows (from
    marker adjacency) that a raw payload belongs to a breakaway bundle should
    call this function directly on `Frame.raw_payload`.
    """
    if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise DecodeError(f"unsupported protocol version {protocol_version}")
    _require(payload, 1, "breakaway evidence")
    cursor = _EvidenceCursor(payload)
    tag = cursor.take("B")
    if tag == BREAKAWAY_TAG_PROBE_PLANNED and protocol_version >= 15 and len(payload) == 53:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("max_observations", "B"),
                ("search_count", "B"),
                ("p_start_raw", "H"),
                ("p_top_raw", "H"),
                ("workflow_digest", "Q"),
                ("target_velocity_rpm", "i"),
                ("hard_torque_limit", "H"),
                ("usable_torque_limit", "H"),
                ("pwm_hz", "I"),
                ("encoder_counts_per_rev", "I"),
                ("motion_threshold_counts", "I"),
                ("max_capture_interval_us", "I"),
            ),
        )
        decoded["family_size"] = None
        decoded["record_kind"] = "breakaway_probe_planned"
    elif tag == BREAKAWAY_TAG_PROBE_PLANNED and protocol_version <= 14 and len(payload) in (23, 47):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("family_size", "H"),
                ("max_observations", "B"),
                ("search_count", "B"),
                ("p_start_raw", "H"),
                ("p_top_raw", "H"),
            ),
        )
        if len(payload) == 47:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("workflow_digest", "Q"),
                        ("target_velocity_rpm", "i"),
                        ("hard_torque_limit", "H"),
                        ("usable_torque_limit", "H"),
                        ("pwm_hz", "I"),
                        ("encoder_counts_per_rev", "I"),
                    ),
                )
            )
        else:
            decoded.update(
                {
                    "workflow_digest": None,
                    "target_velocity_rpm": None,
                    "hard_torque_limit": None,
                    "usable_torque_limit": None,
                    "pwm_hz": None,
                    "encoder_counts_per_rev": None,
                }
            )
        decoded["motion_threshold_counts"] = None
        decoded["max_capture_interval_us"] = None
        decoded["record_kind"] = "breakaway_probe_planned"
    elif tag == BREAKAWAY_TAG_PROBE_RESULT and protocol_version >= 15 and len(payload) == 23:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("rung_index", "B"),
                ("breakaway_p_raw", "H"),
                ("motion_threshold_counts", "I"),
                ("observation_count", "B"),
            ),
        )
        decoded["record_kind"] = "breakaway_probe_resolved"
    elif (
        tag == BREAKAWAY_TAG_PROBE_RESULT
        and protocol_version <= 14
        and (
            (protocol_version >= 14 and len(payload) == 23)
            or (protocol_version <= 13 and len(payload) == 22)
        )
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("direction", "B"),
                ("inert_present", "B"),
                ("inert_p_raw", "H"),
                ("moving_p_raw", "H"),
                ("observations", "B"),
            ),
        )
        if decoded["direction"] not in (0, 1):
            raise DecodeError("breakaway directional-breakaway direction is invalid")
        if decoded["inert_present"] not in (0, 1):
            raise DecodeError("breakaway directional-breakaway inert-present flag is invalid")
        decoded["lower_disposition_mask"] = cursor.take("B") if protocol_version >= 14 else None
        if (
            decoded["lower_disposition_mask"] is not None
            and decoded["lower_disposition_mask"] & ~0b11
        ):
            raise DecodeError("breakaway directional-breakaway disposition mask is invalid")
        decoded["record_kind"] = "breakaway_directional_breakaway"
    elif tag == BREAKAWAY_TAG_PROBE_TERMINAL and protocol_version >= 15 and len(payload) == 18:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("terminal_cause", "B"),
                ("has_safety_fault", "B"),
                ("fault_checkpoint", "B"),
            ),
        )
        if not _breakaway_terminal_cause_is_valid(decoded["terminal_cause"], protocol_version):
            raise DecodeError("breakaway probe terminal cause is invalid")
        decoded["terminal_cause_name"] = _breakaway_terminal_cause_name(
            int(decoded["terminal_cause"]), protocol_version
        )
        if decoded["fault_checkpoint"] not in BREAKAWAY_CHECKPOINT_NAMES:
            raise DecodeError("breakaway probe terminal checkpoint is invalid")
        decoded["fault_checkpoint_name"] = BREAKAWAY_CHECKPOINT_NAMES[decoded["fault_checkpoint"]]
        decoded["family_size"] = None
        decoded["exclusion_reason"] = None
        decoded["exclusion_reason_name"] = None
        decoded["record_kind"] = "breakaway_probe_terminal"
    elif (
        tag == BREAKAWAY_TAG_PROBE_TERMINAL
        and protocol_version <= 14
        and len(payload) in (19, 20, 21)
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("family_size", "H"),
                ("terminal_cause", "B"),
                ("has_safety_fault", "B"),
            ),
        )
        if not _breakaway_terminal_cause_is_valid(decoded["terminal_cause"], protocol_version):
            raise DecodeError("breakaway probe terminal cause is invalid")
        decoded["terminal_cause_name"] = _breakaway_terminal_cause_name(
            int(decoded["terminal_cause"]), protocol_version
        )
        decoded["fault_checkpoint"] = cursor.take("B") if len(payload) >= 20 else 0
        if decoded["fault_checkpoint"] not in BREAKAWAY_CHECKPOINT_NAMES:
            raise DecodeError("breakaway probe terminal checkpoint is invalid")
        decoded["fault_checkpoint_name"] = BREAKAWAY_CHECKPOINT_NAMES[decoded["fault_checkpoint"]]
        decoded["exclusion_reason"] = cursor.take("B") if len(payload) == 21 else 0
        if decoded["exclusion_reason"] not in BREAKAWAY_PROBE_EXCLUSION_REASON_NAMES:
            raise DecodeError("breakaway probe terminal exclusion reason is invalid")
        decoded["exclusion_reason_name"] = BREAKAWAY_PROBE_EXCLUSION_REASON_NAMES[
            decoded["exclusion_reason"]
        ]
        decoded["record_kind"] = "breakaway_probe_terminal"
    elif tag == BREAKAWAY_TAG_DISCOVERY_PLANNED and len(payload) == 45:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("prior_plan_digest", "Q"),
                ("schema_revision", "H"),
                ("family_size", "H"),
                ("rung_count", "B"),
                ("observations_per_direction", "H"),
                ("floor_p_raw", "H"),
                ("floor_origin", "B"),
                ("breakaway_p_raw", "H"),
                ("ceiling_p_raw", "H"),
                ("first_additive_step_raw", "I"),
                ("maximum_workflow_ms", "I"),
            ),
        )
        if decoded["floor_origin"] not in (0, 1):
            raise DecodeError("breakaway discovery floor origin is invalid")
        decoded["record_kind"] = "breakaway_discovery_planned"
    elif tag == BREAKAWAY_TAG_CEILING_SOURCE and len(payload) == 16:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("binding_source", "B"),
            ),
        )
        if decoded["binding_source"] not in (0, 1):
            raise DecodeError("breakaway ceiling binding source is invalid")
        decoded["record_kind"] = "breakaway_ceiling_source"
    elif tag == BREAKAWAY_TAG_RAW_OBSERVATION and (
        (protocol_version >= 16 and len(payload) == 118)
        or (protocol_version == 15 and len(payload) == 116)
        or (protocol_version == 14 and len(payload) == 104)
        or (protocol_version <= 13 and len(payload) == 99)
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("phase", "B"),
                ("rung_index", "B"),
                ("slot_index", "B"),
                ("direction", "B"),
                ("mean_rate_q", "q"),
                ("variance_of_mean_q2", "U"),
                ("target_rate_q", "q"),
                # Frozen classify/consensus inputs so the
                # decoder derives the consensus masks itself.
                ("stationarity_code", "B"),
                ("settled_rate_low_q", "q"),
                ("settled_rate_high_q", "q"),
                ("deficit_low_q", "q"),
                ("deficit_high_q", "q"),
                ("predicted_torque_target_abs", "H"),
                ("usable_torque_limit", "H"),
                ("hard_torque_limit", "H"),
                ("disturbance_present", "B"),
                ("disturbance_low_q16", "i"),
                ("disturbance_high_q16", "i"),
            ),
        )
        valid_phases = (0, 1, 2) if protocol_version >= 14 else (1, 2)
        if decoded["phase"] not in valid_phases:
            raise DecodeError("breakaway raw observation phase is invalid")
        if decoded["direction"] not in (0, 1):
            raise DecodeError("breakaway raw observation direction is invalid")
        if decoded["stationarity_code"] > 6:
            raise DecodeError("breakaway raw observation stationarity code is invalid")
        if decoded["disturbance_present"] not in (0, 1):
            raise DecodeError("breakaway raw observation disturbance flag is invalid")
        if protocol_version >= 14:
            decoded["drift_ratio_ppm"] = cursor.take("I")
            decoded["post_fit_lag_one_state"] = cursor.take("B")
            if decoded["post_fit_lag_one_state"] not in (0, 1, 2):
                raise DecodeError("breakaway raw observation lag-one state is invalid")
            if decoded["stationarity_code"] == 0 and decoded["post_fit_lag_one_state"] != 2:
                raise DecodeError(
                    "breakaway usable raw observation requires lag-one state passed (2)"
                )
        else:
            decoded["drift_ratio_ppm"] = None
            decoded["post_fit_lag_one_state"] = None
        if protocol_version >= 15:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("stroke_min_displacement_counts", "i"),
                        ("stroke_max_displacement_counts", "i"),
                        ("capture_max_interval_us", "I"),
                    ),
                )
            )
            if decoded["phase"] == 0 and not (
                decoded["stroke_min_displacement_counts"]
                <= 0
                <= decoded["stroke_max_displacement_counts"]
            ):
                raise DecodeError("breakaway raw observation extrema are invalid")
        else:
            decoded["stroke_min_displacement_counts"] = None
            decoded["stroke_max_displacement_counts"] = None
            decoded["capture_max_interval_us"] = None
        if protocol_version >= 16:
            # Measured peak velocity-controller output. `predicted_torque_target_abs`
            # above is `max(P * rpm / 256, moving_pid_output_peak)` in firmware, so
            # only carrying both makes a current-headroom terminus attributable to
            # the model term or to measured demand. Absent before v16.
            decoded["moving_pid_output_peak"] = cursor.take("H")
        else:
            decoded["moving_pid_output_peak"] = None
        decoded["record_kind"] = "breakaway_raw_observation"
    elif tag == BREAKAWAY_TAG_DISCOVERY_TERMINAL and len(payload) in (28, 29, 30):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("prior_plan_digest", "Q"),
                ("family_size", "H"),
                ("terminal_cause", "B"),
                ("collected_count", "B"),
                ("has_safety_fault", "B"),
            ),
        )
        if not _breakaway_terminal_cause_is_valid(decoded["terminal_cause"], protocol_version):
            raise DecodeError("breakaway discovery terminal cause is invalid")
        decoded["terminal_cause_name"] = _breakaway_terminal_cause_name(
            int(decoded["terminal_cause"]), protocol_version
        )
        # Present from protocol 17. Absent records report `None` rather than 0,
        # so "no error" and "this capture predates the field" stay distinct.
        decoded["error_code"] = cursor.take("B") if len(payload) >= 29 else None
        decoded["error_code_name"] = COMMISSION_REASON_NAMES.get(decoded["error_code"])
        # Present from protocol 21. Absent records report `None` rather than 0,
        # so "stopped at rung zero" and "this capture predates the field"
        # stay distinct.
        decoded["stop_rung_index"] = cursor.take("B") if len(payload) == 30 else None
        decoded["record_kind"] = "breakaway_discovery_terminal"
    elif tag == BREAKAWAY_TAG_DISCOVERY_RUNG_MARGIN and len(payload) == 38:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("collected_index", "B"),
                ("p_raw", "H"),
                ("lower_percent_milli", "i"),
                ("upper_percent_milli", "i"),
                ("margin_percent_milli", "i"),
                ("half_width_percent_milli", "i"),
                ("adjacent_rate_step_percent_milli", "i"),
            ),
        )
        decoded["record_kind"] = "breakaway_discovery_rung_margin"
    elif tag == BREAKAWAY_TAG_RUNG_ZERO_DIAGNOSTIC and len(payload) == 17:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("forward_moved", "B"),
                ("reverse_moved", "B"),
            ),
        )
        if decoded["forward_moved"] not in (0, 1) or decoded["reverse_moved"] not in (0, 1):
            raise DecodeError("breakaway rung-zero diagnostic flag is invalid")
        decoded["record_kind"] = "breakaway_rung_zero_diagnostic"
    elif tag == BREAKAWAY_TAG_CONFIRMATION_PLANNED and len(payload) == 37:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("prior_plan_digest", "Q"),
                ("family_size", "H"),
                ("observations_per_direction", "H"),
                ("nominated_p_raw", "H"),
                ("band_lower_percent", "B"),
                ("band_upper_percent", "B"),
                ("capture_profile", "B"),
                ("acceptance_rule", "B"),
                ("nominated_margin_percent_milli", "i"),
            ),
        )
        decoded["record_kind"] = "breakaway_confirmation_planned"
    elif tag == BREAKAWAY_TAG_CONFIRMATION_TERMINAL and len(payload) == 40:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("prior_plan_digest", "Q"),
                ("family_size", "H"),
                ("terminal_cause", "B"),
                ("forward_collected_mask", "B"),
                ("forward_eligible_mask", "B"),
                ("forward_included_mask", "B"),
                ("reverse_collected_mask", "B"),
                ("reverse_eligible_mask", "B"),
                ("reverse_included_mask", "B"),
                ("accepted", "B"),
                ("confirmed_p_raw", "H"),
                ("max_relative_se_permille", "H"),
                ("required_relative_se_permille", "H"),
                ("has_safety_fault", "B"),
            ),
        )
        if not _breakaway_terminal_cause_is_valid(decoded["terminal_cause"], protocol_version):
            raise DecodeError("breakaway confirmation terminal cause is invalid")
        decoded["terminal_cause_name"] = _breakaway_terminal_cause_name(
            int(decoded["terminal_cause"]), protocol_version
        )
        if decoded["accepted"] not in (0, 1):
            raise DecodeError("breakaway confirmation accepted flag is invalid")
        for name in (
            "forward_collected_mask",
            "forward_eligible_mask",
            "forward_included_mask",
            "reverse_collected_mask",
            "reverse_eligible_mask",
            "reverse_included_mask",
        ):
            if decoded[name] & ~0xF:
                raise DecodeError("breakaway confirmation observation mask exceeds four slots")
        decoded["record_kind"] = "breakaway_confirmation_terminal"
    elif tag == BREAKAWAY_TAG_CONFIRMATION_OBSERVATION and len(payload) == 27:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("slot_index", "B"),
                ("direction", "B"),
                ("response_lower_percent_milli", "i"),
                ("response_upper_percent_milli", "i"),
                ("relative_standard_error_permille", "H"),
            ),
        )
        if decoded["direction"] not in (0, 1):
            raise DecodeError("breakaway confirmation observation direction is invalid")
        if decoded["slot_index"] > 7:
            raise DecodeError("breakaway confirmation observation slot index is invalid")
        decoded["record_kind"] = "breakaway_confirmation_observation"
    elif tag == BREAKAWAY_TAG_CAMPAIGN_TERMINAL and len(payload) in (18, 19):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("phase", "B"),
                ("terminal_cause", "B"),
                ("accepted", "B"),
                ("integral_plan_digest", "Q"),
            ),
        )
        if decoded["phase"] not in BREAKAWAY_PHASE_NAMES:
            raise DecodeError("breakaway campaign terminal phase is invalid")
        if not _breakaway_terminal_cause_is_valid(decoded["terminal_cause"], protocol_version):
            raise DecodeError("breakaway campaign terminal cause is invalid")
        decoded["terminal_cause_name"] = _breakaway_terminal_cause_name(
            int(decoded["terminal_cause"]), protocol_version
        )
        if decoded["accepted"] not in (0, 1):
            raise DecodeError("breakaway campaign terminal accepted flag is invalid")
        # Present from protocol 19. Absent records report `None` rather than 0,
        # so "no error" and "this capture predates the field" stay distinct.
        decoded["error_code"] = cursor.take("B") if len(payload) == 19 else None
        decoded["error_code_name"] = COMMISSION_REASON_NAMES.get(decoded["error_code"])
        decoded["record_kind"] = "breakaway_campaign_terminal"
    elif tag == BREAKAWAY_TAG_PROBE_DECISION and protocol_version >= 14 and len(payload) == 18:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("direction", "B"),
                ("rung_index", "B"),
                ("disposition", "B"),
            ),
        )
        if decoded["direction"] not in (0, 1):
            raise DecodeError("breakaway probe decision direction is invalid")
        valid_dispositions = (0, 1) if protocol_version >= 15 else (0, 1, 2)
        if decoded["disposition"] not in valid_dispositions:
            raise DecodeError("breakaway probe decision disposition is invalid")
        decoded["record_kind"] = "breakaway_probe_decision"
    elif (
        tag == BREAKAWAY_TAG_RECOVERY_REST_REJECTED
        and protocol_version >= 18
        and len(payload) == 83
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("phase", "B"),
                ("stage", "B"),
                ("rung_index", "B"),
                ("slot_index", "B"),
                ("direction", "B"),
                ("p_raw", "H"),
                ("structural_code_64", "B"),
                ("structural_code_128", "B"),
                ("structural_code_256", "B"),
                ("interval_present_mask", "B"),
                ("usable_mask", "B"),
                ("zero_compatible_mask", "B"),
                ("selected_mask", "B"),
                ("selected_suffix_len", "H"),
                ("disposition", "B"),
            ),
        )
        decoded["intervals_q"] = [[cursor.take("q"), cursor.take("q")] for _ in range(3)]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("scored_net_displacement_counts", "b"),
                    ("scored_min_displacement_counts", "b"),
                    ("scored_max_displacement_counts", "b"),
                ),
            )
        )

        if decoded["phase"] not in (0, 1, 2):
            raise DecodeError("rejected recovery rest phase is invalid")
        if decoded["direction"] not in (0, 1):
            raise DecodeError("rejected recovery rest direction is invalid")
        # Only rejections are ever reported; an acceptance reaching the wire
        # would mean the firmware-side gate regressed.
        if decoded["disposition"] != 2:
            raise DecodeError("rejected recovery rest carries a non-rejected disposition")

        structural = [
            int(decoded["structural_code_64"]),
            int(decoded["structural_code_128"]),
            int(decoded["structural_code_256"]),
        ]
        if any(code not in range(6) for code in structural):
            raise DecodeError("rejected recovery rest structural code is invalid")

        masks = {
            "interval_present_mask": int(decoded["interval_present_mask"]),
            "usable_mask": int(decoded["usable_mask"]),
            "zero_compatible_mask": int(decoded["zero_compatible_mask"]),
            "selected_mask": int(decoded["selected_mask"]),
        }
        if any(value & ~0b111 for value in masks.values()):
            raise DecodeError("rejected recovery rest mask has a reserved bit set")
        # A rejection is exactly the absence of a selected suffix.
        if masks["selected_mask"] != 0 or int(decoded["selected_suffix_len"]) != 0:
            raise DecodeError("rejected recovery rest names a selected suffix")
        # Zero-compatibility is a refinement of usability, and both refine
        # interval presence, so each mask must be a subset of the last.
        if masks["zero_compatible_mask"] & ~masks["usable_mask"]:
            raise DecodeError("rejected recovery rest zero mask exceeds its usable mask")
        if masks["usable_mask"] & ~masks["interval_present_mask"]:
            raise DecodeError("rejected recovery rest usable mask exceeds its present mask")
        # A structurally usable suffix is exactly one whose code is `Usable`.
        expected_usable = sum(1 << index for index, code in enumerate(structural) if code == 0)
        if masks["usable_mask"] != expected_usable:
            raise DecodeError("rejected recovery rest usable mask contradicts its codes")

        for index, interval in enumerate(decoded["intervals_q"]):
            present = bool(masks["interval_present_mask"] & (1 << index))
            if not present and interval != [0, 0]:
                raise DecodeError("absent recovery rest interval carries a value")
            if present and interval[0] > interval[1]:
                raise DecodeError("rejected recovery rest interval is reversed")
            zero_compatible = bool(masks["zero_compatible_mask"] & (1 << index))
            contains_zero = present and interval[0] <= 0 <= interval[1]
            usable = bool(masks["usable_mask"] & (1 << index))
            if zero_compatible != (usable and contains_zero):
                raise DecodeError("rejected recovery rest zero-compatibility is inconsistent")

        scored_min = int(decoded["scored_min_displacement_counts"])
        scored_max = int(decoded["scored_max_displacement_counts"])
        scored_net = int(decoded["scored_net_displacement_counts"])
        if not scored_min <= scored_net <= scored_max:
            raise DecodeError("rejected recovery rest scored displacement is inconsistent")
        # The firmware accumulator clamps its running sum to this band.
        if not all(-2 <= value <= 2 for value in (scored_min, scored_net, scored_max)):
            raise DecodeError("rejected recovery rest displacement escapes its clamp")

        decoded["structural_codes"] = structural
        # The decisive derived fact: the one-count fallback is only offered when
        # the scored window's displacement span is at most one count, so a span
        # of two or more means the statistical route was the only route.
        decoded["scored_displacement_span_counts"] = scored_max - scored_min
        decoded["one_count_fallback_available"] = scored_max - scored_min <= 1
        decoded["record_kind"] = "breakaway_recovery_rest_rejected"
    elif (
        tag == BREAKAWAY_TAG_RECOVERY_ORIGIN_MISMATCH
        and protocol_version >= 18
        and len(payload) == 38
    ):
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest", "Q"),
                ("phase", "B"),
                ("stage", "B"),
                ("rung_index", "B"),
                ("slot_index", "B"),
                ("direction", "B"),
                ("p_raw", "H"),
                ("expected_start_offset_counts", "i"),
                ("observed_position_counts", "i"),
                ("delta_counts", "i"),
                ("position_window_counts", "I"),
            ),
        )
        if decoded["phase"] not in (0, 1, 2):
            raise DecodeError("recovery origin mismatch phase is invalid")
        if decoded["direction"] not in (0, 1):
            raise DecodeError("recovery origin mismatch direction is invalid")
        expected = int(decoded["expected_start_offset_counts"])
        observed = int(decoded["observed_position_counts"])
        delta = int(decoded["delta_counts"])
        if observed - expected != delta:
            raise DecodeError("recovery origin mismatch delta contradicts its offsets")
        # The fault fires on exact inequality, so a zero delta cannot reach here.
        if delta == 0:
            raise DecodeError("recovery origin mismatch reports no discrepancy")
        # The window check runs first, so a start offset beyond the window is
        # contained rather than faulted and must never reach this record.
        window = int(decoded["position_window_counts"])
        if abs(expected) > window:
            raise DecodeError("recovery origin mismatch start offset exceeds its window")
        decoded["delta_abs_counts"] = abs(delta)
        # The decisive derived fact: a single-count discrepancy is one
        # outer-safety encoder sample landing between the tick that captured the
        # origin and the tick recovery began -- a race, not a runaway.
        decoded["single_sample_race"] = abs(delta) == 1
        decoded["record_kind"] = "breakaway_recovery_origin_mismatch"
    else:
        raise DecodeError(f"unknown breakaway evidence tag {tag} (length {len(payload)})")
    cursor.finish()
    decoded["tag"] = tag
    return decoded


def assemble_breakaway_ticks(
    frames: list[tuple[int, int, bytes, dict[str, Any] | None]],
    *,
    protocol_version: int = PROTOCOL_VERSION,
) -> list[dict[str, Any]]:
    """Assemble breakaway-campaign canonical records into per-marker ticks.

    `frames` is the ordered sequence of `(frame_type, timestamp_ticks,
    raw_payload, decoded_payload)` tuples for one device/source, restricted to
    `FRAME_TYPE_EVENT` and `FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE` frames (any
    other frame type is irrelevant here and must be filtered out by the
    caller). `decoded_payload` is `Frame.payload` for `FRAME_TYPE_EVENT`
    frames (read for `event_id`/`args`) and is ignored for
    `FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE` frames: this function always
    re-decodes `raw_payload` for those directly through
    `decode_breakaway_record`, because `Frame.payload` may carry the wrong
    interpretation for the one colliding tag (see that function's docstring).

    The firmware emits every breakaway canonical record for one campaign
    transition immediately before exactly one `CommissionTraceEvent` marker
    (spec section 10; `breakaway_trace_reporting.rs` module doc), so a
    `FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE` frame is claimed by the next event
    marker of *any* kind, not only a breakaway one -- an unrelated record
    family's own marker still owns and clears whatever evidence preceded it,
    exactly as it does for that family's own decoder. Only markers whose
    `event_id` is a breakaway id (152-159) become entries in the returned
    list; every other marker silently discards the frames it claims.

    Returns one dict per breakaway marker event, in stream order:
    `{"event_id", "timestamp_ticks", "sequence", "records": [(raw, decoded)]}
    `. `sequence` is the marker's own `evidence_sequence` (event arg 0),
    which spec section 10 documents as identifying the *transition*, not a
    strictly increasing per-record counter for these bundles. Raises
    `DecodeError` if any frame claimed by a breakaway marker fails to decode
    as a breakaway record -- this also catches a foreign record family's
    evidence being misrouted into a breakaway bundle by a reordered or
    corrupt capture, since `decode_breakaway_record` rejects any payload that
    isn't an exact tag/length match.
    """
    ticks: list[dict[str, Any]] = []
    pending: list[bytes] = []
    for frame_type, timestamp_ticks, raw_payload, decoded_payload in frames:
        if frame_type == FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE:
            pending.append(raw_payload)
            continue
        if frame_type != FRAME_TYPE_EVENT:
            continue
        if decoded_payload is None:
            raise DecodeError("breakaway assembly requires a decoded event payload")
        event_id = int(decoded_payload["event_id"])
        if event_id not in BREAKAWAY_CAMPAIGN_EVENT_IDS:
            pending = []
            continue
        # `request_commission_event` (usb_trace.rs) packs a commissioning
        # event's four wire args as `[source_index, engine_kind,
        # event.payload[0], event.payload[1]]`; for a breakaway marker,
        # `event.payload[0]` is the `evidence_sequence` this function reports
        # as `sequence`. This is the same convention `commissioning.py`
        # already relies on (`source_index = args[0]`) -- the marker's own
        # frame header carries a fixed "not applicable" source index
        # (`SOURCE_KIND_BOARD`/`SOURCE_INDEX_NA`), never the real one.
        args = decoded_payload.get("args") or []
        sequence = int(args[2]) if len(args) > 2 else 0
        records = [
            (raw, decode_breakaway_record(raw, protocol_version=protocol_version))
            for raw in pending
        ]
        pending = []
        ticks.append(
            {
                "event_id": event_id,
                "timestamp_ticks": timestamp_ticks,
                "sequence": sequence,
                "records": records,
            }
        )
    return ticks


def _decode_control_detail(payload: bytes) -> dict[str, Any]:
    _require(payload, 28, "control-detail")
    fields = struct.unpack_from("<IIiiIII", payload, 0)
    names = (
        "interim_pidout_target_torque",
        "interim_pidout_target_flux",
        "interim_pidin_target_velocity",
        "interim_pidout_target_velocity",
        "pidout_uq_ud_limits",
        "pid_torque_flux_limits",
        "pid_velocity_limit",
    )
    return dict(zip(names, fields, strict=True))


def _decode_status(version: int, payload: bytes) -> dict[str, Any]:
    _require(payload, 40, "status")
    fields = struct.unpack_from("<IIIIIHHIIII", payload, 0)
    names = (
        "status_flags",
        "status_interrupt_count",
        "clipping_flags",
        "spi_error_count",
        "watchdog_or_step_overrun_count",
        "klipper_rx_queue_depth",
        "usb_trace_queue_high_water",
        "dropped_motion_sample_count",
        "dropped_event_log_count",
        "endpoint_not_ready_count",
        "suppressed_status_change_count",
    )
    decoded = dict(zip(names, fields, strict=True))
    if version >= 22:
        _require(payload, 48, "status")
        pressure, max_cycles = struct.unpack_from("<II", payload, 40)
        decoded["trace_deadline_pressure_count"] = pressure
        decoded["max_spi_sample_cycles"] = max_cycles
    return decoded


def _decode_event(payload: bytes) -> dict[str, Any]:
    _require(payload, 24, "event")
    level, subsystem_id, event_id, arg_count = struct.unpack_from("<BBHB", payload, 0)
    args = struct.unpack_from("<iiii", payload, 8)
    decoded_args = list(args[: min(arg_count, 4)])
    decoded = {
        "level": level,
        "level_name": _LEVEL_NAMES.get(level, f"unknown_{level}"),
        "subsystem_id": subsystem_id,
        "subsystem_name": _SUBSYSTEM_NAMES.get(subsystem_id, f"unknown_{subsystem_id}"),
        "event_id": event_id,
        "arg_count": min(arg_count, 4),
        "args": decoded_args,
    }
    if subsystem_id == SUBSYSTEM_COMMISSIONING:
        decoded["event_name"] = _COMMISSIONING_EVENT_NAMES.get(
            event_id, f"commission_unknown_{event_id}"
        )
        if len(decoded_args) >= 2:
            engine_kind = decoded_args[1]
            decoded["engine_kind_name"] = _ENGINE_KIND_NAMES.get(
                engine_kind, f"unknown_{engine_kind}"
            )
    elif subsystem_id == SUBSYSTEM_STEP_TIMER and event_id == EVENT_PHASE_BOUNDARY:
        decoded["event_name"] = "phase_boundary"
        if len(decoded_args) >= 3:
            decoded["phase_boundary_channel"] = decoded_args[0]
            decoded["phase_boundary_old_class_name"] = _PHASE_CLASS_NAMES.get(
                decoded_args[1], f"unknown_{decoded_args[1]}"
            )
            decoded["phase_boundary_new_class_name"] = _PHASE_CLASS_NAMES.get(
                decoded_args[2], f"unknown_{decoded_args[2]}"
            )
    else:
        decoded["event_name"] = f"unknown_{event_id}"
    return {
        **decoded,
    }


def _read_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(payload):
        raise DecodeError("metadata string length missing")
    size = payload[offset]
    offset += 1
    if size > 31:
        raise DecodeError("metadata string exceeds 31 bytes")
    end = offset + size
    if end > len(payload):
        raise DecodeError("metadata string truncated")
    return payload[offset:end].decode("utf-8"), end


def _decode_metadata(payload: bytes) -> dict[str, Any]:
    _require(payload, 36, "metadata")
    (
        version,
        feature_revision,
        source_count,
        skipped_count,
        clock_freq,
        critical_hz,
        detail_hz,
        status_hz,
        pre_ms,
        skipped_critical,
        skipped_detail,
        skipped_status,
    ) = struct.unpack_from("<BBBBIIIIIIII", payload, 0)
    if version != 1:
        return {"unsupported_metadata_schema_version": version}
    if source_count > 16:
        raise DecodeError("metadata source count exceeds 16")
    offset = 36
    firmware_id, offset = _read_string(payload, offset)
    product, offset = _read_string(payload, offset)
    serial, offset = _read_string(payload, offset)
    sources: list[dict[str, Any]] = []
    for _ in range(source_count):
        if offset + 3 > len(payload):
            raise DecodeError("source label truncated")
        source_kind = payload[offset]
        source_index = payload[offset + 1]
        label_len = payload[offset + 2]
        offset += 3
        if label_len > 31:
            raise DecodeError("source label exceeds 31 bytes")
        end = offset + label_len
        if end > len(payload):
            raise DecodeError("source label bytes truncated")
        sources.append(
            {
                "source_kind": source_kind,
                "source_index": source_index,
                "label": payload[offset:end].decode("utf-8"),
            }
        )
        offset = end
    return {
        "metadata_schema_version": version,
        "feature_revision": feature_revision,
        "source_count": source_count,
        "skipped_sample_counter_count": skipped_count,
        "clock_freq": clock_freq,
        "critical_hz": critical_hz,
        "control_detail_hz": detail_hz,
        "status_hz": status_hz,
        "pre_configuration_elapsed_ms": pre_ms,
        "pre_configuration_skipped_critical": skipped_critical,
        "pre_configuration_skipped_control_detail": skipped_detail,
        "pre_configuration_skipped_status": skipped_status,
        "firmware_id": firmware_id,
        "product": product,
        "serial": serial,
        "sources": sources,
    }
