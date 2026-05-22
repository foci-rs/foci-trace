import io
import struct

from foci_trace.analyze import format_summary, summarize_capture
from foci_trace.capture_file import CaptureWriter
from foci_trace.protocol import (
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_STATUS,
    PROTOCOL_VERSION,
)


def _header(
    frame_type: int,
    payload: bytes,
    seq: int,
    ts: int,
    source_kind: int,
    source_index: int,
) -> bytes:
    return struct.pack(
        "<IBBBBHBBI Q I I".replace(" ", ""),
        0x43525446,
        PROTOCOL_VERSION,
        32,
        frame_type,
        0,
        len(payload),
        source_kind,
        source_index,
        seq,
        ts,
        1234,
        0,
    )


def _metadata_payload(serial: str) -> bytes:
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
        2,
        2,
        1,
        0,
    )
    firmware = b"openffboard-fw"
    product = b"FOCI OpenFFBoard (trace)"
    return (
        fixed
        + bytes([len(firmware)])
        + firmware
        + bytes([len(product)])
        + product
        + bytes([len(serial)])
        + serial.encode("ascii")
        + b"\x01\x00\x07channel"
    )


def _status_payload(drops: int, endpoint_pressure: int, flags: int = 0x70004480) -> bytes:
    return struct.pack(
        "<IIIIIHHIIII",
        flags,
        0,
        flags & 0x0000_FFFF,
        2,
        0,
        3,
        31,
        drops,
        1,
        endpoint_pressure,
        4,
    )


def _critical_payload(
    position_error: int,
    velocity_error: int,
    foc_uq_ud: int,
    foc_uq_ud_limited: int,
    accel_offset: int = 0,
    accel_gain: int = 0,
    accel_flags: int = 0,
) -> bytes:
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        100,
        98,
        position_error,
        30,
        20,
        4,
        velocity_error,
        0x80,
        0,
        3,
        7,
        -5,
        1,
        2,
        0,
        0x0003_0004,
        0x0005_0006,
        foc_uq_ud,
        foc_uq_ud_limited,
    )
    return payload + struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)


def _pack_i16_pair(low: int, high: int) -> int:
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def _critical_motion_payload(
    step_position: int,
    tmc_position_actual: int,
    torque_actual: int,
    uq_limited: int,
    accel_offset: int = 0,
    accel_gain: int = 0,
    accel_flags: int = 0,
) -> bytes:
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        step_position,
        tmc_position_actual,
        step_position - tmc_position_actual,
        30,
        20,
        4,
        10,
        0x80,
        0,
        3,
        7,
        -5,
        1,
        2,
        0,
        0,
        _pack_i16_pair(0, torque_actual),
        _pack_i16_pair(0, uq_limited),
        _pack_i16_pair(0, uq_limited),
    )
    return payload + struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)


def _control_detail_payload(torque_cmd: int, voltage_limit: int, torque_limit: int) -> bytes:
    return struct.pack(
        "<IIIIIII",
        torque_cmd & 0xFFFF_FFFF,
        0,
        0,
        0,
        _pack_i16_pair(voltage_limit, voltage_limit),
        _pack_i16_pair(0, torque_limit),
        500000,
    )


def test_summarize_capture_reports_mapping_gaps_status_and_steady_ranges() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
    writer.write_firmware_frame(
        0,
        110,
        _header(FRAME_TYPE_METADATA, _metadata_payload("ABC"), 1, 0, 0, 255)
        + _metadata_payload("ABC"),
    )
    writer.write_firmware_frame(
        0,
        120,
        _header(FRAME_TYPE_STATUS, _status_payload(10, 20), 2, 84_000_000, 2, 255)
        + _status_payload(10, 20),
    )
    writer.write_firmware_frame(
        0,
        130,
        _header(
            FRAME_TYPE_CRITICAL,
            _critical_payload(3, -4, 0xFFFE_0001, 0x0002_0003, -19, 2000, 0x03),
            3,
            84_000_000 * 2,
            1,
            0,
        )
        + _critical_payload(3, -4, 0xFFFE_0001, 0x0002_0003, -19, 2000, 0x03),
    )
    writer.write_firmware_frame(
        0,
        140,
        _header(
            FRAME_TYPE_CRITICAL,
            _critical_payload(-7, 8, 0x0004_FFFC, 0x0001_FFFE, 21, 2000, 0x03),
            5,
            84_000_000 * 3,
            1,
            0,
        )
        + _critical_payload(-7, 8, 0x0004_FFFC, 0x0001_FFFE, 21, 2000, 0x03),
    )
    writer.write_firmware_frame(
        0,
        150,
        _header(FRAME_TYPE_STATUS, _status_payload(13, 27), 6, 84_000_000 * 4, 2, 255)
        + _status_payload(13, 27),
    )

    reports = summarize_capture(io.BytesIO(output.getvalue()), steady_after_s=1.5)

    assert len(reports) == 1
    report = reports[0]
    assert report.usb_path == "1-50"
    assert report.serial == "ABC"
    assert report.metadata_serial == "ABC"
    assert report.source_labels == ["FOCI[0]=channel"]
    assert report.missing_frames == 1
    assert report.status_deltas["dropped_motion_sample_count"] == 3
    assert report.status_deltas["endpoint_not_ready_count"] == 7
    assert report.steady_samples == 2
    assert report.voltage_limited_samples == 2
    assert report.ranges["position_error"] == (-7, 3)
    assert report.ranges["velocity_error"] == (-4, 8)
    assert report.ranges["ud"] == (-4, 1)
    assert report.ranges["uq"] == (-2, 4)
    assert report.ranges["ud_delta"] == (-2, -2)
    assert report.ranges["uq_delta"] == (-4, 3)
    assert report.ranges["accel_ff_offset"] == (-19, 21)
    assert report.ranges["accel_ff_gain"] == (2000, 2000)

    text = format_summary(reports)
    assert "dev0 1-50 FOCI OpenFFBoard (trace) serial=ABC" in text
    assert "missing=1" in text
    assert "status_delta dropped=3 endpoint=7 spi=0 watchdog=0" in text
    assert "steady>=1.5s samples=2 voltage_diff=2" in text
    assert "accel_ff=-19..21 gain=2000..2000 flags=0x03" in text


def test_summarize_capture_reports_motion_segments() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    critical_samples = [
        (0, 0, 0, 0, 1000),
        (1, clock // 1000, 10, 20, 2000),
        (3, (clock // 1000) * 2, 20, 38, 3000),
        (4, (clock // 1000) * 3, 30, 60, 4000),
    ]
    for seq, timestamp, step, actual, uq_limited in critical_samples:
        payload = _critical_motion_payload(
            step,
            actual,
            100 + step,
            uq_limited,
            accel_offset=step - 15,
            accel_gain=2000,
            accel_flags=0x03,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    for seq, timestamp, torque_cmd in (
        (5, clock // 1000, 2816),
        (6, (clock // 1000) * 2, -2816),
    ):
        payload = _control_detail_payload(torque_cmd, voltage_limit=4000, torque_limit=2816)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CONTROL_DETAIL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=5,
    )

    assert len(reports) == 1
    assert len(reports[0].motion_segments) == 1
    segment = reports[0].motion_segments[0]
    assert segment.step_delta == 30
    assert segment.duration_s == 0.003
    assert segment.critical_samples == 4
    assert segment.control_detail_samples == 2
    assert segment.fitted_error_rms_steps > 0
    assert segment.fitted_error_peak_steps > 0
    assert segment.actual_torque_range == (100, 130)
    assert segment.uq_limited_range == (1000, 4000)
    assert segment.voltage_rail_samples == 1
    assert segment.torque_command_range == (-2816, 2816)
    assert segment.torque_command_saturated_samples == 2
    assert segment.actual_to_command_abs_median is not None
    assert segment.accel_feedforward_offset_range == (-15, 15)
    assert segment.accel_feedforward_gain_range == (2000, 2000)
    assert segment.accel_feedforward_flags == 0x03

    text = format_summary(reports)
    assert "motion0 delta=30 duration=0.003s" in text
    assert "torque_cmd=-2816..2816 saturated=2/2" in text
    assert "accel_ff=-15..15 gain=2000..2000 flags=0x03" in text
