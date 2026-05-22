import struct

import pytest

from foci_trace.protocol import (
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_METADATA,
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


def test_decode_critical_offsets() -> None:
    payload = bytearray(84)
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
    fixed = struct.pack(
        "<BBBBIIIIIIII",
        1,
        1,
        1,
        3,
        84_000_000,
        1000,
        500,
        10,
        7,
        7000,
        3500,
        70,
    )
    strings = b"\x08fw 1.2.3\x18FOCI OpenFFBoard (trace)\x08ABCDEF12"
    label = b"\x01\x00\x07channel"
    frame = decode_frame(
        _header(FRAME_TYPE_METADATA, fixed + strings + label) + fixed + strings + label
    )

    assert frame.payload["metadata_schema_version"] == 1
    assert frame.payload["source_count"] == 1
    assert frame.payload["firmware_id"] == "fw 1.2.3"
    assert frame.payload["product"] == "FOCI OpenFFBoard (trace)"
    assert frame.payload["sources"][0]["label"] == "channel"


def test_rejects_bad_magic() -> None:
    with pytest.raises(DecodeError, match="bad magic"):
        decode_frame(b"BAD!" + b"\x00" * 40)


def test_assembler_reconstructs_frame_from_usb_packets() -> None:
    payload = bytes(range(84))
    frame = _header(FRAME_TYPE_CRITICAL, payload) + payload
    assembler = FrameAssembler()

    assert assembler.feed(frame[:64]) == []
    assert assembler.feed(frame[64:]) == [frame]
