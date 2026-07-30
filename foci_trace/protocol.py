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
EVENT_COMMISSION_ACCEPTANCE_MATRIX_PLAN = 146
EVENT_COMMISSION_ACCEPTANCE_MATRIX_OBSERVATION = 147
EVENT_COMMISSION_ACCEPTANCE_MATRIX_AMPLITUDE = 148
EVENT_COMMISSION_ACCEPTANCE_MATRIX_TERMINAL = 149
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
    EVENT_COMMISSION_ACCEPTANCE_MATRIX_PLAN: "commission_acceptance_matrix_plan",
    EVENT_COMMISSION_ACCEPTANCE_MATRIX_OBSERVATION: "commission_acceptance_matrix_observation",
    EVENT_COMMISSION_ACCEPTANCE_MATRIX_AMPLITUDE: "commission_acceptance_matrix_amplitude",
    EVENT_COMMISSION_ACCEPTANCE_MATRIX_TERMINAL: "commission_acceptance_matrix_terminal",
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


def _take_optional_interval(cursor: _EvidenceCursor) -> list[int] | None:
    present = cursor.take("B")
    if present not in (0, 1):
        raise DecodeError("matrix optional interval marker is invalid")
    return [cursor.take("i"), cursor.take("i")] if present else None


def _take_optional_i32(cursor: _EvidenceCursor) -> int | None:
    present = cursor.take("B")
    if present not in (0, 1):
        raise DecodeError("matrix optional value marker is invalid")
    return cursor.take("i") if present else None


def _take_acceptance_direction(cursor: _EvidenceCursor) -> dict[str, Any]:
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
        raise DecodeError("matrix direction taxonomy is invalid")
    if direction["component_count"] > 2:
        raise DecodeError("matrix direction has too many components")
    for name in ("collected_mask", "eligible_mask", "included_mask"):
        if direction[name] & ~0xF:
            raise DecodeError("matrix observation mask exceeds four slots")
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


def _try_decode_acceptance_amplitude(payload: bytes) -> dict[str, Any] | None:
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
            _take_acceptance_direction(cursor),
            _take_acceptance_direction(cursor),
        ]
        cursor.finish()
    except DecodeError:
        return None
    decoded["namespace"] = "acceptance_matrix"
    decoded["record_kind"] = "acceptance_matrix_amplitude"
    return decoded


def _decode_velocity_sweep_evidence(payload: bytes) -> dict[str, Any]:
    _require(payload, 1, "velocity-sweep evidence")
    cursor = _EvidenceCursor(payload)
    kind = cursor.take("B")
    if kind == 0 and len(payload) in (43, 56, 72, 73):
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
        if (fragment, len(payload)) == (0, 73):
            decoded.update(
                _take_fields(
                    cursor,
                    (
                        ("schema_revision", "H"),
                        ("channel", "B"),
                        ("build_revision", "I"),
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
            raise DecodeError("matrix plan fragment shape is invalid")
        if "order" in decoded:
            # Slot order rides in the high nibble of the amplitude-order byte, so
            # forward-first runs keep the byte they had before it was recorded.
            # `order` stays raw because the plan digest hashes the packed value.
            packed = int(decoded["order"])
            amplitude_order = packed & 0x0F
            slot_order = packed >> 4
            if amplitude_order not in (1, 2):
                raise DecodeError("matrix plan amplitude order is invalid")
            if slot_order not in (0, 1):
                raise DecodeError("matrix plan slot order is invalid")
            decoded["amplitude_order"] = amplitude_order
            decoded["slot_order"] = slot_order
        decoded["namespace"] = "acceptance_matrix"
        decoded["record_kind"] = "acceptance_matrix_plan"
    elif kind == 0 and len(payload) in (65, 82, 98):
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
            if decoded["schema_revision"] >= 7:
                decoded["stage_b_joint_membership"] = cursor.take("I")
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
            if decoded["schema_revision"] >= 8:
                if cursor.take("B") != 1:
                    raise DecodeError("combined integral plan core is invalid")
                decoded["combined_selected_response"] = True
                decoded["selected_response_intervals_q"] = [
                    [cursor.take("i"), cursor.take("i")] for _ in range(2)
                ]
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
                # Bit 2 records the mirrored observation slot order. Captures made
                # before it existed leave it clear, which is forward-first.
                decoded["slot_order"] = (int(decoded["flags"]) >> 2) & 1
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
    elif kind == 2 and (matrix := _try_decode_acceptance_amplitude(payload)) is not None:
        return matrix
    elif kind == 2 and len(payload) in (46, 47, 77, 80):
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
            5: "MatrixOrderMismatch",
            6: "EvidenceCapacity",
            7: "EvidenceIntegrity",
        }.get(decoded["cause"], "Unknown")
        decoded["namespace"] = "acceptance_matrix"
        decoded["record_kind"] = "acceptance_matrix_terminal"
    elif kind == 3 and len(payload) in (70, 83):
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
        if len(payload) == 83:
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
    elif kind == 4 and len(payload) in (58, 59, 61, 66):
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
        elif len(payload) in (61, 66):
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
        if len(payload) in (61, 66):
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
        if len(payload) == 66:
            if cursor.take("B") != 1:
                raise DecodeError("combined integral terminal marker is invalid")
            decoded["combined_workflow"] = True
            decoded["final_p"] = cursor.take("H")
            decoded["target_status"] = cursor.take("B")
            target_terminus_plus_one = cursor.take("B")
            decoded["target_terminus"] = (
                None if target_terminus_plus_one == 0 else target_terminus_plus_one - 1
            )
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
        fields = [
            ("run_sequence", "I"),
            ("evidence_sequence", "H"),
            ("nominated_p", "H"),
            ("coverage_mask", "B"),
        ]
        if len(payload) == 75:
            fields.extend(
                (
                    ("model_direction_mask", "B"),
                    ("joint_union_mask", "I"),
                )
            )
        fields.extend(
            (
                ("joint_member_mask", "I"),
                ("nominated_rung", "B"),
                ("distance_to_start_q16", "I"),
                ("distance_to_top_q16", "I"),
                ("nomination_flags", "B"),
            )
        )
        decoded = _take_fields(cursor, tuple(fields))
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
    elif kind == 11 and len(payload) == 78:
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
            if code not in range(5):
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
        if int(decoded["usable_mask"]) != expected_usable_mask:
            raise DecodeError("scored-rest usable mask is inconsistent")
        if int(decoded["zero_compatible_mask"]) != expected_zero_mask:
            raise DecodeError("scored-rest zero-compatible mask is inconsistent")

        expected_selected_mask = 0
        expected_selected_len = 0
        for index, suffix_len in enumerate((64, 128, 256)):
            bit = 1 << index
            if expected_zero_mask & bit == 0:
                break
            expected_selected_mask = bit
            expected_selected_len = suffix_len
        disposition = int(decoded["disposition"])
        if disposition == 0:
            if (
                expected_selected_mask == 0
                or int(decoded["selected_mask"]) != expected_selected_mask
                or int(decoded["selected_suffix_len"]) != expected_selected_len
            ):
                raise DecodeError("scored-rest statistical selection is inconsistent")
        elif disposition in (1, 2):
            if (
                expected_selected_mask != 0
                or int(decoded["selected_mask"]) != 0
                or int(decoded["selected_suffix_len"]) != 0
            ):
                raise DecodeError("scored-rest terminal selection is inconsistent")
        else:
            raise DecodeError("scored-rest disposition is invalid")
        decoded["conditioned_rest_evidence"] = disposition == 0
        decoded["record_kind"] = "rest_selection"
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
