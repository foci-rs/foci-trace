"""Decode FOCI USB trace FTRC frames."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

FRAME_MAGIC = 0x43525446
PROTOCOL_VERSION = 6
SUPPORTED_PROTOCOL_VERSIONS = (1, 2, 3, 4, 5, PROTOCOL_VERSION)
HEADER_LEN = 32

FRAME_TYPE_CRITICAL = 1
FRAME_TYPE_CONTROL_DETAIL = 2
FRAME_TYPE_STATUS = 3
FRAME_TYPE_EVENT = 4
FRAME_TYPE_METADATA = 5
FRAME_TYPE_CURRENT_PATH = 6
FRAME_TYPE_PHASE_ADVANCE = 7


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


def _decode_control_detail(payload: bytes) -> dict[str, Any]:
    _require(payload, 28, "control-detail")
    fields = struct.unpack_from("<IIIIIII", payload, 0)
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
    return {
        "level": level,
        "subsystem_id": subsystem_id,
        "event_id": event_id,
        "arg_count": min(arg_count, 4),
        "args": list(args[: min(arg_count, 4)]),
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
