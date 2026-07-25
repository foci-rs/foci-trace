"""Decode FOCI USB trace FTRC frames."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

FRAME_MAGIC = 0x43525446
PROTOCOL_VERSION = 9
SUPPORTED_PROTOCOL_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, PROTOCOL_VERSION)
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

VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID = 1 << 1
VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID = 1 << 2

SUBSYSTEM_COMMISSIONING = 7

EVENT_COMMISSION_ENGINE_START = 100
EVENT_COMMISSION_PHASE_START = 101
EVENT_COMMISSION_PHASE_END = 102
EVENT_COMMISSION_ENGINE_TERMINAL = 103
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
EVENT_COMMISSION_DIAGNOSTIC_WARNING = 200

_LEVEL_NAMES = {
    1: "info",
    2: "warn",
    3: "error",
}

_SUBSYSTEM_NAMES = {
    SUBSYSTEM_COMMISSIONING: "commissioning",
}

_COMMISSIONING_EVENT_NAMES = {
    EVENT_COMMISSION_ENGINE_START: "commission_engine_start",
    EVENT_COMMISSION_PHASE_START: "commission_phase_start",
    EVENT_COMMISSION_PHASE_END: "commission_phase_end",
    EVENT_COMMISSION_ENGINE_TERMINAL: "commission_engine_terminal",
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
    EVENT_COMMISSION_DIAGNOSTIC_WARNING: "commission_diagnostic_warning",
}

_ENGINE_KIND_NAMES = {
    1: "commission",
    2: "calibrate",
    3: "selftest",
    4: "outer_tune",
}

PHASE_NAMES = {
    1: "AdcCalibrate",
    2: "CoilCheck",
    3: "PhaseWiringCheck",
    4: "EncoderCheck",
    5: "ElectricalId",
    6: "CurrentTune",
    7: "CurrentValidate",
    8: "InnerDone",
    9: "MechanicalId",
    10: "VelocityTune",
    11: "VelocityValidate",
    12: "PositionTune",
    13: "FilterTune",
    14: "Commit",
    15: "OuterDone",
    16: "EncoderAlign",
    17: "ClosedLoopEntry",
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
    10: "mechanical_id_failed",
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
        return _decode_status(payload)
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
        return _decode_velocity_sweep_evidence(payload)
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
        size = struct.calcsize("<" + fmt)
        if self.offset + size > len(self.payload):
            raise DecodeError("velocity-sweep evidence record is truncated")
        value = struct.unpack_from("<" + fmt, self.payload, self.offset)[0]
        self.offset += size
        return int(value)

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise DecodeError("velocity-sweep evidence record has trailing bytes")


def _take_fields(cursor: _EvidenceCursor, fields: tuple[tuple[str, str], ...]) -> dict[str, int]:
    return {name: cursor.take(fmt) for name, fmt in fields}


def _decode_velocity_sweep_evidence(payload: bytes) -> dict[str, Any]:
    _require(payload, 1, "velocity-sweep evidence")
    cursor = _EvidenceCursor(payload)
    kind = cursor.take("B")
    if kind == 0 and len(payload) == 65:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("plan_digest", "Q"),
                ("final_p", "H"),
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
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("build_revision", "I"),
                        ("schema_revision", "H"),
                        ("channel", "B"),
                        ("stage_b_plan_digest_low", "I"),
                        ("stage_b_plan_digest_high", "I"),
                        ("final_p", "H"),
                    ),
                )
            )
            decoded["directional_memberships"] = [cursor.take("I") for _ in range(2)]
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
            decoded["stage_b_plan_digest"] = decoded["stage_b_plan_digest_low"] | (
                decoded["stage_b_plan_digest_high"] << 32
            )
            if decoded["schema_revision"] >= 3 or len(payload) == 88:
                decoded["limit_evidence_capability"] = cursor.take("B")
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
            decoded["record_kind"] = "integral_plan_recovery"
        else:
            raise DecodeError(f"unknown velocity-integral plan fragment {fragment}")
        decoded["plan_fragment"] = fragment
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
    elif kind == 1 and len(payload) == 22:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("plan_digest_low", "I"),
                ("plan_digest_high", "I"),
                ("final_p", "H"),
                ("positive_rung_count", "B"),
                ("maximum_workflow_ms", "I"),
            ),
        )
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["record_kind"] = "integral_plan_marker"
    elif kind == 1:
        stage_b = len(payload) in (78, 91)
        stage_b_recovery = len(payload) == 91
        plan_fields = [
            ("run_sequence", "I"),
            ("evidence_sequence", "H"),
        ]
        if stage_b:
            plan_fields.extend((("plan_digest_low", "I"), ("plan_digest_high", "I")))
        plan_fields.extend(
            (
                ("requested_velocity_mrev_s", "I"),
                ("requested_velocity_source", "B"),
                ("planned_velocity_mrev_s", "I"),
                ("effective_ceiling_mrev_s", "I"),
                ("clamp_flags", "H"),
                ("binding_source", "B"),
                ("target_velocity_rpm", "i"),
                ("p_start", "H"),
                ("p_top", "H"),
                ("rung_count", "B"),
                ("observations_per_direction", "H"),
                ("moving_stroke_us", "I"),
                ("zero_settle_us", "I"),
                ("nominal_workflow_ms", "I"),
                ("maximum_workflow_ms", "I"),
            )
        )
        if stage_b_recovery:
            plan_fields.extend(
                (
                    ("recovery_origin_band_counts", "I"),
                    ("recovery_nominal_slot_us", "I"),
                    ("recovery_maximum_slot_us", "I"),
                    ("recovery_slot_count", "B"),
                )
            )
        plan_fields.extend(
            (
                ("max_stroke_travel_mrev", "I"),
                ("settle_travel_reserve_mrev", "I"),
                ("negative_position_headroom_mrev", "I"),
                ("positive_position_headroom_mrev", "I"),
                ("hard_torque_limit", "H"),
                ("usable_torque_limit", "H"),
            )
        )
        decoded = _take_fields(
            cursor,
            tuple(plan_fields),
        )
        if stage_b:
            decoded["plan_digest"] = decoded["plan_digest_low"] | (
                decoded["plan_digest_high"] << 32
            )
        decoded["record_kind"] = "plan"
    elif kind == 2 and len(payload) in (46, 77):
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
                ("stationarity_code", "B"),
            ),
        )
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
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["record_kind"] = "integral_observation"
    elif kind == 2:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("rung_index", "B"),
                ("slot", "B"),
                ("target_velocity_rpm", "i"),
                ("classification", "B"),
                ("flags", "B"),
                ("delta_sign", "B"),
                ("delta_mantissa", "I"),
                ("delta_shift", "B"),
                ("elapsed_mantissa", "I"),
                ("elapsed_shift", "B"),
                ("rate_low", "i"),
                ("rate_mean", "i"),
                ("rate_high", "i"),
                ("deficit_low", "i"),
                ("deficit_high", "i"),
                ("rate_shift", "B"),
                ("suffix_len", "H"),
                ("selected_level", "B"),
                ("selected_blocks", "I"),
                ("rate_variance_mantissa", "I"),
                ("rate_variance_shift", "B"),
                ("slope_mantissa", "i"),
                ("slope_shift", "B"),
                ("slope_half_width_mantissa", "I"),
                ("slope_half_width_shift", "B"),
                ("lag_one_q", "i"),
                ("lag_one_half_width_q", "I"),
                ("residual_mantissa", "I"),
                ("residual_shift", "B"),
                ("tested_suffixes", "B"),
                ("velocity_p", "H"),
                ("disturbance_q16", "i"),
                ("disturbance_low_q16", "i"),
                ("disturbance_high_q16", "i"),
                ("disturbance_variance_mantissa", "I"),
                ("disturbance_variance_shift", "B"),
                ("predicted_torque_target_abs", "H"),
            ),
        )
        decoded["record_kind"] = "observation"
    elif kind == 3 and len(payload) == 70:
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
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["record_kind"] = "integral_rung"
    elif kind == 3:
        if len(payload) == 122:
            decoded = _take_fields(
                cursor,
                (
                    ("run_sequence", "I"),
                    ("evidence_sequence", "H"),
                    ("rung_index", "B"),
                    ("velocity_p", "H"),
                ),
            )
            directional = []
            for prefix in ("forward", "reverse"):
                values = _take_fields(
                    cursor,
                    (
                        ("class", "B"),
                        ("component_count", "B"),
                        ("collected_mask", "B"),
                        ("eligible_mask", "B"),
                        ("included_mask", "B"),
                        ("operable_count", "B"),
                        ("directional_flags", "B"),
                    ),
                )
                components = [
                    [cursor.take("i"), cursor.take("i")],
                    [cursor.take("i"), cursor.take("i")],
                ]
                pooled = _take_fields(
                    cursor,
                    (
                        ("pooled_q16", "i"),
                        ("pooled_low_q16", "i"),
                        ("pooled_high_q16", "i"),
                        ("variance_floor_observations", "B"),
                    ),
                )
                rate_bounds = []
                for _ in range(4):
                    mantissa = cursor.take("i")
                    shift = cursor.take("B")
                    rate_bounds.append(mantissa << shift)
                count = values["component_count"]
                if count > 2:
                    raise DecodeError("velocity-sweep rung has too many components")
                values["components"] = components[:count]
                values["rate_bounds_q"] = rate_bounds
                values.update(pooled)
                directional.append(values)
                for name, value in values.items():
                    decoded[f"{prefix}_{name}"] = value
            decoded["flags"] = (
                (directional[0]["directional_flags"] & 1)
                | ((directional[1]["directional_flags"] & 1) << 1)
                | ((directional[0]["directional_flags"] & 2) << 1)
                | ((directional[1]["directional_flags"] & 2) << 2)
            )
            decoded["record_kind"] = "rung"
            cursor.finish()
            return decoded
        compact_stage_b_rung = len(payload) == 33
        fields = [
            ("run_sequence", "I"),
            ("evidence_sequence", "H"),
            ("rung_index", "B"),
            ("velocity_p", "H"),
            ("forward_low_q16", "i"),
            ("forward_high_q16", "i"),
            ("reverse_low_q16", "i"),
            ("reverse_high_q16", "i"),
        ]
        if not compact_stage_b_rung:
            fields.extend(
                (
                    ("forward_p_low", "H"),
                    ("forward_p_high", "H"),
                    ("reverse_p_low", "H"),
                    ("reverse_p_high", "H"),
                )
            )
        fields.extend(
            (
                ("forward_class", "B"),
                ("reverse_class", "B"),
            )
        )
        if not compact_stage_b_rung:
            fields.extend(
                (
                    ("forward_closure", "B"),
                    ("reverse_closure", "B"),
                )
            )
        fields.extend(
            (
                ("flags", "B"),
                ("forward_eligible_observations", "H"),
                ("reverse_eligible_observations", "H"),
            )
        )
        if not compact_stage_b_rung:
            fields[-2:-2] = [
                ("forward_eligible_rungs", "B"),
                ("reverse_eligible_rungs", "B"),
            ]
        decoded = _take_fields(cursor, tuple(fields))
        decoded["record_kind"] = "rung"
    elif kind == 4 and len(payload) in (58, 59, 61):
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
        if len(payload) in (59, 61):
            fields.append(("recovery_unavailable", "B"))
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
        if len(payload) == 61:
            decoded["opening_available_mask"] = cursor.take("B")
            decoded["bookend_available_mask"] = cursor.take("B")
        current_terminus_plus_one = cursor.take("B")
        decoded["current_terminus"] = (
            None if current_terminus_plus_one == 0 else current_terminus_plus_one - 1
        )
        if len(payload) == 58:
            decoded["recovery_unavailable"] = 0
        decoded["run_started_us"] = cursor.take("I") | (cursor.take("I") << 32)
        decoded["run_completed_us"] = cursor.take("I") | (cursor.take("I") << 32)
        decoded["plan_digest"] = decoded["plan_digest_low"] | (decoded["plan_digest_high"] << 32)
        decoded["digest"] = decoded["digest_low"] | (decoded["digest_high"] << 32)
        decoded["record_kind"] = "integral_terminal"
    elif kind == 4 and len(payload) != 71:
        decoded = {
            "record_kind": "terminal",
            "run_sequence": cursor.take("I"),
            "encoder_counts_per_rev": cursor.take("I"),
            "f_pwm_hz": cursor.take("I"),
        }
        directions = []
        direction_fields = (
            ("direction_closure", "B"),
            ("first_moving", "H"),
            ("p_low", "H"),
            ("p_high", "H"),
            ("common_low_q16", "i"),
            ("common_high_q16", "i"),
            ("pooled_q16", "i"),
            ("pooled_low_q16", "i"),
            ("pooled_high_q16", "i"),
            ("eligible_rungs", "B"),
            ("eligible_observations", "H"),
            ("flags", "B"),
        )
        for _ in range(2):
            directions.append(_take_fields(cursor, direction_fields))
        decoded["directions"] = directions
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("expected_observations", "H"),
                    ("emitted_observations", "H"),
                    ("expected_rungs", "H"),
                    ("emitted_rungs", "H"),
                    ("digest_low", "I"),
                    ("digest_high", "I"),
                    ("outcome", "B"),
                    ("cause", "B"),
                    ("sufficient_direction_mask", "B"),
                ),
            )
        )
    elif kind == 4:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("member_mask", "I"),
                ("direction", "B"),
                ("region_kind", "B"),
                ("closure", "B"),
                ("first_rung", "B"),
                ("last_rung", "B"),
                ("p_low", "H"),
                ("p_high", "H"),
                ("member_count", "B"),
                ("common_low_q16", "i"),
                ("common_high_q16", "i"),
                ("pooled_q16", "i"),
                ("pooled_low_q16", "i"),
                ("pooled_high_q16", "i"),
                ("variance_floor_observations", "B"),
                ("settled_rate_mean_min_mantissa", "i"),
                ("settled_rate_mean_min_shift", "B"),
                ("settled_rate_mean_max_mantissa", "i"),
                ("settled_rate_mean_max_shift", "B"),
                ("settled_rate_envelope_low_mantissa", "i"),
                ("settled_rate_envelope_low_shift", "B"),
                ("settled_rate_envelope_high_mantissa", "i"),
                ("settled_rate_envelope_high_shift", "B"),
                ("boundary_rung_plus_one", "B"),
                ("tested_low_q16", "i"),
                ("tested_high_q16", "i"),
            ),
        )
        decoded["record_kind"] = "directional_region"
    elif kind == 5:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("member_mask", "I"),
                ("first_rung", "B"),
                ("last_rung", "B"),
                ("member_count", "B"),
                ("p_low", "H"),
                ("p_high", "H"),
                ("closure", "B"),
            ),
        )
        decoded["record_kind"] = "joint_region"
    elif kind == 6:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("nominated_p", "H"),
                ("coverage_mask", "B"),
                ("joint_member_mask", "I"),
                ("nominated_rung", "B"),
                ("distance_to_start_q16", "I"),
                ("distance_to_top_q16", "I"),
                ("nomination_flags", "B"),
            ),
        )
        directions = []
        for _ in range(2):
            directions.append(
                _take_fields(
                    cursor,
                    (
                        ("member_mask", "I"),
                        ("pooled_low_q16", "i"),
                        ("pooled_high_q16", "i"),
                        ("coverage", "B"),
                        ("signed_rung_distance", "B"),
                        ("gain_ratio_num", "H"),
                        ("gain_ratio_den", "H"),
                        ("settled_rate_difference_mantissa", "i"),
                        ("settled_rate_difference_shift", "B"),
                    ),
                )
            )
        decoded["directions"] = directions
        decoded["record_kind"] = "handoff"
    elif kind == 7:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("rung_index", "B"),
                ("slot", "B"),
                ("passing_suffix_mask", "B"),
                ("common_rate_low_mantissa", "i"),
                ("common_rate_low_shift", "B"),
                ("common_rate_high_mantissa", "i"),
                ("common_rate_high_shift", "B"),
                ("binding_slope_margin_mantissa", "i"),
                ("binding_slope_margin_shift", "B"),
                ("binding_lag_one_margin_q", "i"),
            ),
        )
        decoded["record_kind"] = "structured_boundary"
    elif kind == 8:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("encoder_counts_per_rev", "I"),
                ("f_pwm_hz", "I"),
                ("model_direction_mask", "B"),
            ),
        )
        decoded["selected_memberships"] = [cursor.take("I") for _ in range(3)]
        decoded["nominated_p"] = cursor.take("H")
        decoded["selected_intervals"] = [[cursor.take("i"), cursor.take("i")] for _ in range(2)]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("coverage_mask", "B"),
                    ("forward_region_count", "B"),
                    ("reverse_region_count", "B"),
                    ("forward_fragment_count", "B"),
                    ("reverse_fragment_count", "B"),
                    ("expected_observations", "H"),
                    ("emitted_observations", "H"),
                    ("expected_rungs", "H"),
                    ("emitted_rungs", "H"),
                    ("digest_low", "I"),
                    ("digest_high", "I"),
                    ("outcome", "B"),
                    ("cause", "B"),
                ),
            )
        )
        decoded["recovery_unavailable"] = cursor.take("B") if len(payload) == 86 else 0
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("run_started_low", "I"),
                    ("run_started_high", "I"),
                    ("run_completed_low", "I"),
                    ("run_completed_high", "I"),
                ),
            )
        )
        decoded["record_kind"] = "stage_b_terminal"
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
    elif kind == 9:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("outcome", "B"),
                ("previous_nominated_p", "H"),
                ("current_nominated_p", "H"),
            ),
        )
        decoded["previous_memberships"] = [cursor.take("I") for _ in range(3)]
        decoded["current_memberships"] = [cursor.take("I") for _ in range(3)]
        decoded["previous_intervals"] = [[cursor.take("i"), cursor.take("i")] for _ in range(2)]
        decoded["current_intervals"] = [[cursor.take("i"), cursor.take("i")] for _ in range(2)]
        decoded.update(
            _take_fields(
                cursor,
                (
                    ("previous_digest_low", "I"),
                    ("previous_digest_high", "I"),
                    ("current_digest_low", "I"),
                    ("current_digest_high", "I"),
                ),
            )
        )
        decoded["record_kind"] = "reproduction"
        decoded["schema_revision"] = 2
    elif kind == 10:
        common_fields = (
            ("run_sequence", "I"),
            ("evidence_sequence", "H"),
            ("outcome", "B"),
            ("reason_mask", "B"),
        )
        decoded = _take_fields(cursor, common_fields)
        if len(payload) == 26:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("previous_provisional_p", "H"),
                        ("current_provisional_p", "H"),
                        ("final_p", "H"),
                        ("selected_joint_mask", "I"),
                        ("selected_first_rung", "B"),
                        ("selected_member_count", "B"),
                        ("forward_validity", "B"),
                        ("reverse_validity", "B"),
                        ("reduced_margin", "B"),
                        ("schema_revision", "H"),
                    ),
                )
            )
            decoded["record_kind"] = "reproduction_v4_header"
        else:
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("previous_nominated_p", "H"),
                        ("current_nominated_p", "H"),
                        ("final_p", "H"),
                        ("reduced_margin", "B"),
                        ("schema_revision", "H"),
                    ),
                )
            )
            decoded["record_kind"] = "reproduction_v3_header"
    elif 11 <= kind <= 13:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("object", "B"),
                ("previous", "I"),
                ("current", "I"),
                ("core", "I"),
                ("previous_only", "I"),
                ("current_only", "I"),
                ("low_delta", "b"),
                ("high_delta", "b"),
            ),
        )
        decoded["record_kind"] = "reproduction_membership"
    elif 14 <= kind <= 15:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("direction", "B"),
                ("previous_low", "i"),
                ("previous_high", "i"),
                ("current_low", "i"),
                ("current_high", "i"),
                ("overlap_low", "i"),
                ("overlap_high", "i"),
            ),
        )
        decoded["record_kind"] = "reproduction_pooled"
    elif 16 <= kind <= 17:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("direction", "B"),
                ("previous_low", "i"),
                ("previous_high", "i"),
                ("current_low", "i"),
                ("current_high", "i"),
                ("conservative_low", "i"),
                ("conservative_high", "i"),
                ("nonempty", "B"),
            ),
        )
        decoded["record_kind"] = "reproduction_common"
    elif 18 <= kind <= 19:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("direction", "B"),
                ("coverage", "B"),
                ("signed_rung_distance", "i"),
                ("gain_ratio_num", "H"),
                ("gain_ratio_den", "H"),
            ),
        )
        decoded["record_kind"] = "reproduction_coverage"
    elif kind == 20:
        decoded = _take_fields(
            cursor,
            (
                ("run_sequence", "I"),
                ("evidence_sequence", "H"),
                ("previous_digest_low", "I"),
                ("previous_digest_high", "I"),
                ("current_digest_low", "I"),
                ("current_digest_high", "I"),
            ),
        )
        decoded["record_kind"] = "reproduction_digest"
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
    else:
        raise DecodeError(f"unknown velocity-sweep evidence kind {kind}")
    cursor.finish()
    return decoded


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


def _decode_status(payload: bytes) -> dict[str, Any]:
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
    return dict(zip(names, fields, strict=True))


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
