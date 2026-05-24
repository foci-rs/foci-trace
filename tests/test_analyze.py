import io
import struct

from foci_trace.analyze import format_summary, summarize_capture
from foci_trace.capture_file import CaptureWriter
from foci_trace.protocol import (
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
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
    version: int = PROTOCOL_VERSION,
) -> bytes:
    return struct.pack(
        "<IBBBBHBBI Q I I".replace(" ", ""),
        0x43525446,
        version,
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
        0x0006_0005,
        foc_uq_ud,
        foc_uq_ud_limited,
    )
    return payload + struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)


def _current_path_payload(
    foc_iq_id: int,
    foc_iwy_iux: int,
    adc_i1_i0: int,
    sample_flags: int = 0,
) -> bytes:
    return struct.pack("<BBBBIII", sample_flags, 0, 0, 0, foc_iq_id, foc_iwy_iux, adc_i1_i0)


def _pack_i16_pair(low: int, high: int) -> int:
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def _pack_u16_pair(low: int, high: int) -> int:
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def _critical_motion_payload(
    step_position: int,
    tmc_position_actual: int,
    torque_actual: int,
    uq_limited: int,
    torque_target: int = 0,
    uq_requested: int | None = None,
    accel_offset: int = 0,
    accel_gain: int = 0,
    accel_flags: int = 0,
) -> bytes:
    if uq_requested is None:
        uq_requested = uq_limited
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
        _pack_i16_pair(torque_actual, torque_target),
        _pack_i16_pair(0, uq_requested),
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


def test_summarize_capture_reports_current_signal_path_ranges() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    first = _current_path_payload(
        foc_iq_id=_pack_i16_pair(-11, 22),
        foc_iwy_iux=_pack_i16_pair(-55, 66),
        adc_i1_i0=_pack_u16_pair(32700, 33100),
    )
    second = _current_path_payload(
        foc_iq_id=_pack_i16_pair(33, -44),
        foc_iwy_iux=_pack_i16_pair(77, -88),
        adc_i1_i0=_pack_u16_pair(32750, 33150),
    )
    writer.write_firmware_frame(
        0,
        110,
        _header(FRAME_TYPE_CURRENT_PATH, first, 1, 84_000_000 * 2, 1, 0) + first,
    )
    writer.write_firmware_frame(
        0,
        120,
        _header(FRAME_TYPE_CURRENT_PATH, second, 2, 84_000_000 * 3, 1, 0) + second,
    )

    reports = summarize_capture(io.BytesIO(output.getvalue()), steady_after_s=0)

    report = reports[0]
    assert report.ranges["foc_iq"] == (-44, 22)
    assert report.ranges["foc_id"] == (-11, 33)
    assert report.ranges["foc_iux"] == (-55, 77)
    assert report.ranges["foc_iwy"] == (-88, 66)
    assert report.ranges["adc_i0_raw"] == (32700, 32750)
    assert report.ranges["adc_i1_raw"] == (33100, 33150)

    text = format_summary(reports)
    assert "iq=-44..22 id=-11..33" in text
    assert "iux=-55..77 iwy=-88..66" in text
    assert "adc0_raw=32700..32750 adc1_raw=33100..33150" in text


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


def test_motion_summary_can_include_phase_split() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    samples = [
        (1, 0, 0, 0, 500, 2200, 1000),
        (2, clock * 25 // 1000, 2000, 1800, 600, 2600, 1600),
        (3, clock * 75 // 1000, 5000, 5100, 700, 500, 1200),
        (4, clock * 125 // 1000, 8000, 7900, 650, 400, 1000),
        (5, clock * 175 // 1000, 11000, 10800, -500, -2500, -1800),
        (6, clock * 225 // 1000, 14000, 13600, -550, -2600, -1700),
        (7, clock * 300 // 1000, 14000, 13700, 100, 0, 100),
        (8, clock * 450 // 1000, 14000, 13800, 50, 0, 80),
    ]
    for seq, timestamp, step, actual, torque_actual, torque_target, uq_limited in samples:
        payload = _critical_motion_payload(
            step,
            actual,
            torque_actual=torque_actual,
            torque_target=torque_target,
            uq_limited=uq_limited,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    for seq, timestamp in (
        (9, 0),
        (10, clock * 75 // 1000),
        (11, clock * 175 // 1000),
    ):
        payload = _control_detail_payload(100, voltage_limit=3000, torque_limit=2816)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CONTROL_DETAIL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=1000,
    )

    segment = reports[0].motion_segments[0]
    assert [phase.name for phase in segment.phase_reports] == [
        "entry_0_50ms",
        "cruise_mid",
        "exit_last50ms",
        "post_0_500ms",
    ]
    assert segment.phase_reports[0].critical_samples == 2
    assert segment.phase_reports[1].critical_samples == 2
    assert segment.phase_reports[2].critical_samples == 2
    assert segment.phase_reports[3].critical_samples == 2

    default_text = format_summary(reports)
    assert "phase entry_0_50ms" not in default_text

    phase_text = format_summary(reports, include_phase_split=True)
    assert "phase entry_0_50ms samples=2" in phase_text
    assert "phase cruise_mid samples=2" in phase_text
    assert "phase exit_last50ms samples=2" in phase_text
    assert "phase post_0_500ms samples=2" in phase_text
    assert "actual/target=" in phase_text
    assert "uq/limit=" in phase_text


def test_summarize_capture_reports_current_authority_windows() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    critical_samples = [
        (1, clock // 1000, 0, 0, 900, 2800, 1200, 1200),
        (2, (clock // 1000) * 2, 10, 10, 1000, 1400, 1400, 1400),
        (3, (clock // 1000) * 3, 20, 20, -800, -2816, -3000, -3600),
        (4, (clock // 1000) * 4, 30, 30, -700, -2600, -2900, -3300),
    ]
    for (
        seq,
        timestamp,
        step,
        actual,
        torque_actual,
        torque_target,
        uq_limited,
        uq_requested,
    ) in critical_samples:
        payload = _critical_motion_payload(
            step,
            actual,
            torque_actual,
            uq_limited,
            torque_target=torque_target,
            uq_requested=uq_requested,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    current_paths = [
        (
            5,
            clock // 1000,
            _pack_i16_pair(20, 910),
            _pack_i16_pair(-100, 120),
            _pack_u16_pair(32700, 33100),
        ),
        (
            6,
            (clock // 1000) * 3,
            _pack_i16_pair(-30, -820),
            _pack_i16_pair(-90, 130),
            _pack_u16_pair(32740, 33160),
        ),
    ]
    for seq, timestamp, iq_id, iwy_iux, adc_i1_i0 in current_paths:
        payload = _current_path_payload(iq_id, iwy_iux, adc_i1_i0)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CURRENT_PATH, payload, seq, timestamp, 1, 0) + payload,
        )

    for seq, timestamp, torque_cmd in (
        (7, clock // 1000, 2800),
        (8, (clock // 1000) * 2, 1400),
        (9, (clock // 1000) * 3, -2816),
        (10, (clock // 1000) * 4, -2600),
    ):
        payload = _control_detail_payload(torque_cmd, voltage_limit=3000, torque_limit=2816)
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

    segment = reports[0].motion_segments[0]
    authority = segment.current_authority
    assert authority is not None
    assert authority.near_limit_samples == 3
    assert authority.near_limit_sample_coverage == 0.75
    assert authority.near_limit_longest_run_s == 0.001
    assert authority.sign_agreement_samples == 3
    assert authority.actual_to_command_abs_median == 0.284
    assert authority.target_to_command_abs_median == 1.0
    assert authority.actual_to_target_abs_median == 0.284
    assert authority.iq_to_actual_abs_median == 1.018
    assert authority.uq_limited_range == (-3000, 1200)
    assert authority.uq_delta_range == (-600, 0)
    assert authority.uq_to_limit_abs_median == 0.967
    assert authority.voltage_rail_samples == 1
    assert authority.adc_i0_delta_range == (-20, 20)
    assert authority.adc_i1_delta_range == (-30, 30)

    text = format_summary(reports)
    assert "authority near_limit=3 sign=3/3" in text
    assert "coverage=75.0% longest=1.0ms" in text
    assert "actual/target=0.28" in text
    assert "actual/cmd=" not in text
    assert "target/cmd=" not in text
    assert "uq_lim=-3000..1200 uq_delta=-600..0 uq/limit=0.97 rail=1/3" in text
    assert "iq/actual=1.02 adc0_delta=-20..20 adc1_delta=-30..30" in text


def test_motion_summary_distinguishes_critical_target_from_sparse_detail_command() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    for seq, timestamp, step, torque_target in (
        (1, clock // 1000, 0, 2816),
        (2, (clock // 1000) * 2, 10, 2200),
        (3, (clock // 1000) * 3, 20, -2816),
        (4, (clock // 1000) * 4, 30, -2100),
    ):
        payload = _critical_motion_payload(
            step,
            step,
            torque_actual=700,
            uq_limited=1200,
            torque_target=torque_target,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    detail = _control_detail_payload(100, voltage_limit=3000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        clock // 1000,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 5, clock // 1000, 1, 0) + detail,
    )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=5,
    )

    segment = reports[0].motion_segments[0]
    assert segment.torque_target_range == (-2816, 2816)
    assert segment.torque_command_range == (100, 100)
    assert segment.current_authority is not None
    assert segment.current_authority.near_limit_samples == 2
    assert segment.current_authority.actual_to_target_abs_median == 0.249

    text = format_summary(reports)
    assert "torque_target=-2816..2816" in text
    assert "detail_torque_cmd=100..100" in text
    assert "actual/cmd=" not in text
    assert "target/cmd=" not in text


def test_summarize_capture_flags_sparse_motion_trace_windows() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    critical_samples = [
        (1, 0, 0),
        (2, clock // 1000, 10),
        (3, (clock // 1000) * 51, 500),
        (4, (clock // 1000) * 52, 510),
    ]
    for seq, timestamp, step in critical_samples:
        payload = _critical_motion_payload(step, step, 100, 1000)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    for seq, timestamp in (
        (5, 0),
        (6, clock // 1000),
        (7, (clock // 1000) * 51),
    ):
        payload = _control_detail_payload(100, voltage_limit=3000, torque_limit=2816)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CONTROL_DETAIL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=5,
    )

    segment = reports[0].motion_segments[0]
    assert segment.critical_max_gap_s == 0.05
    assert segment.control_detail_max_gap_s == 0.05
    assert segment.trace_gap_suspect

    text = format_summary(reports)
    assert "trace_gap critical=50.0ms detail=50.0ms current=n/a" in text


def test_summarize_capture_does_not_bridge_motion_across_timestamp_gap() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    samples = [
        (1, 0, 20000, 20000),
        (250000, clock * 250, 0, 0),
    ]
    for seq, timestamp, step, actual in samples:
        payload = _critical_motion_payload(step, actual, 100, 1000)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.08,
        motion_min_steps=1000,
    )

    assert reports[0].missing_frames == 249998
    assert reports[0].motion_segments == []
