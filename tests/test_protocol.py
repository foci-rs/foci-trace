import struct

import pytest

from foci_trace.protocol import (
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    PROTOCOL_VERSION,
    DecodeError,
    FrameAssembler,
    decode_frame,
)


def _header(
    frame_type: int, payload: bytes, seq: int = 1, version: int = PROTOCOL_VERSION
) -> bytes:
    return struct.pack(
        "<IBBBBHBBI Q I I".replace(" ", ""),
        0x43525446,
        version,
        32,
        frame_type,
        0,
        len(payload),
        1,
        0,
        seq,
        123,
        456,
        0,
    )


def _metadata_source(kind: int, index: int, label: str) -> bytes:
    label_bytes = label.encode("ascii")
    return bytes([kind, index, len(label_bytes)]) + label_bytes


def test_decode_critical_offsets() -> None:
    payload = bytearray(88)
    payload[0:4] = struct.pack("<i", -1)
    payload[28] = 0x81
    payload[29] = 0x02
    payload[30:32] = struct.pack("<H", 9)
    payload[40:48] = struct.pack("<Q", 12)
    payload[48:56] = struct.pack("<Q", 13)
    payload[76:80] = struct.pack("<i", -19)
    payload[80:82] = struct.pack("<H", 2000)
    payload[82] = 0x03

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload) + payload)

    assert frame.header.frame_type == FRAME_TYPE_CRITICAL
    assert frame.payload["step_position"] == -1
    assert frame.payload["enable_and_direction"] == 0x81
    assert frame.payload["timer_queue_depth"] == 9
    assert frame.payload["last_served_step_tick"] == 12
    assert frame.payload["last_klipper_command_service_tick"] == 13
    assert frame.payload["cached_accel_torque_offset"] == -19
    assert frame.payload["accel_feedforward_gain_permille"] == 2000
    assert frame.payload["accel_feedforward_flags"] == 0x03
    assert frame.payload["cached_velocity_transient_offset"] == 0
    assert "interim_foc_iq_id" not in frame.payload


def test_decode_v3_critical_current_tracking_fields() -> None:
    payload = bytearray(84)
    payload[60:64] = struct.pack("<i", -1234)
    payload[64:68] = struct.pack("<I", 0x0ABC_FEDC)

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=3) + payload)

    assert frame.payload["interim_pidin_target_torque"] == -1234
    assert frame.payload["interim_pid_torque_target_actual"] == 0x0ABC_FEDC
    assert "pid_torque_flux_target" not in frame.payload


def test_decode_v4_critical_current_signal_path_fields() -> None:
    payload = bytearray(96)
    payload[60:64] = struct.pack("<i", -1234)
    payload[64:68] = struct.pack("<I", 0x0ABC_FEDC)
    payload[84:88] = struct.pack("<I", 0x1122_3344)
    payload[88:92] = struct.pack("<I", 0x5566_7788)
    payload[92:96] = struct.pack("<I", 0x99AA_BBCC)

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=4) + payload)

    assert frame.payload["interim_pidin_target_torque"] == -1234
    assert frame.payload["interim_pid_torque_target_actual"] == 0x0ABC_FEDC
    assert frame.payload["interim_foc_iq_id"] == 0x1122_3344
    assert frame.payload["interim_foc_iwy_iux"] == 0x5566_7788
    assert frame.payload["interim_adc_i1_i0"] == 0x99AA_BBCC


def test_decode_v6_critical_velocity_transient_field() -> None:
    payload = bytearray(88)
    payload[84:88] = struct.pack("<i", -321)

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=6) + payload)

    assert frame.payload["cached_velocity_transient_offset"] == -321


def test_decode_v5_current_path_fields() -> None:
    payload = bytearray(16)
    payload[0] = 0x02
    payload[4:8] = struct.pack("<I", 0x1122_3344)
    payload[8:12] = struct.pack("<I", 0x5566_7788)
    payload[12:16] = struct.pack("<I", 0x99AA_BBCC)

    frame = decode_frame(_header(FRAME_TYPE_CURRENT_PATH, payload, version=5) + payload)

    assert frame.payload["sample_flags"] == 0x02
    assert frame.payload["interim_foc_iq_id"] == 0x1122_3344
    assert frame.payload["interim_foc_iwy_iux"] == 0x5566_7788
    assert frame.payload["interim_adc_i1_i0"] == 0x99AA_BBCC


def test_decodes_phase_advance_frame() -> None:
    payload = struct.pack("<hBBiHH", -64, 0b0000_0111, 0, -60000, 64, 16)

    frame = decode_frame(_header(FRAME_TYPE_PHASE_ADVANCE, payload) + payload)

    assert frame.header.frame_type == FRAME_TYPE_PHASE_ADVANCE
    assert frame.payload["phase_advance_counts"] == -64
    assert frame.payload["phase_advance_flags"] == 0b0000_0111
    assert frame.payload["phase_advance_gain_ppm"] == -60000
    assert frame.payload["phase_advance_max_counts"] == 64
    assert frame.payload["phase_advance_deadband"] == 16


def test_decode_v2_critical_keeps_legacy_torque_fields() -> None:
    payload = bytearray(84)
    payload[60:64] = struct.pack("<I", 0x1122_3344)
    payload[64:68] = struct.pack("<I", 0x5566_7788)

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=2) + payload)

    assert frame.payload["pid_torque_flux_target"] == 0x1122_3344
    assert frame.payload["pid_torque_flux_actual"] == 0x5566_7788


def test_decode_v1_critical_treats_accel_feedforward_fields_as_zero() -> None:
    payload = bytearray(84)
    payload[0:4] = struct.pack("<i", 5)
    payload[76:84] = b"\xaa" * 8

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=1) + payload)

    assert frame.payload["step_position"] == 5
    assert frame.payload["cached_accel_torque_offset"] == 0
    assert frame.payload["accel_feedforward_gain_permille"] == 0
    assert frame.payload["accel_feedforward_flags"] == 0


def test_decode_metadata_strings_and_labels() -> None:
    sources = b"".join(
        [
            _metadata_source(0, 255, "board"),
            _metadata_source(2, 255, "usb"),
            _metadata_source(3, 255, "klipper"),
            _metadata_source(1, 0, "channel0"),
            _metadata_source(1, 1, "channel1"),
        ]
    )
    fixed = struct.pack(
        "<BBBBIIIIIIII",
        1,
        1,
        5,
        3,
        84_000_000,
        1000,
        500,  # OpenFFBoard's configured control-detail trace rate.
        10,
        7,
        7000,
        3500,
        70,
    )
    strings = b"\x08fw 1.2.3\x18FOCI OpenFFBoard (trace)\x08ABCDEF12"
    payload = fixed + strings + sources
    frame = decode_frame(_header(FRAME_TYPE_METADATA, payload) + payload)

    assert frame.payload["metadata_schema_version"] == 1
    assert frame.payload["source_count"] == 5
    assert frame.payload["firmware_id"] == "fw 1.2.3"
    assert frame.payload["product"] == "FOCI OpenFFBoard (trace)"
    assert frame.payload["sources"] == [
        {"source_kind": 0, "source_index": 255, "label": "board"},
        {"source_kind": 2, "source_index": 255, "label": "usb"},
        {"source_kind": 3, "source_index": 255, "label": "klipper"},
        {"source_kind": 1, "source_index": 0, "label": "channel0"},
        {"source_kind": 1, "source_index": 1, "label": "channel1"},
    ]


def test_decode_commissioning_event_names() -> None:
    payload = struct.pack("<BBHBxxxiiii", 1, 7, 101, 4, 0, 1, 17, 3)

    frame = decode_frame(_header(FRAME_TYPE_EVENT, payload) + payload)

    assert frame.payload["subsystem_name"] == "commissioning"
    assert frame.payload["event_name"] == "commission_phase_start"
    assert frame.payload["level_name"] == "info"
    assert frame.payload["engine_kind_name"] == "commission"
    assert frame.payload["args"] == [0, 1, 17, 3]


def test_rejects_bad_magic() -> None:
    with pytest.raises(DecodeError, match="bad magic"):
        decode_frame(b"BAD!" + b"\x00" * 40)


def test_assembler_reconstructs_frame_from_usb_packets() -> None:
    payload = bytes(range(84))
    frame = _header(FRAME_TYPE_CRITICAL, payload) + payload
    assembler = FrameAssembler()

    assert assembler.feed(frame[:64]) == []
    assert assembler.feed(frame[64:]) == [frame]
