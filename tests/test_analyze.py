import io
import json
import struct

from foci_trace.analyze import format_summary, summarize_capture
from foci_trace.capture_file import CaptureWriter
from foci_trace.protocol import (
    EVENT_COMMISSION_ENGINE_START,
    EVENT_COMMISSION_ENGINE_TERMINAL,
    EVENT_COMMISSION_PHASE_END,
    EVENT_COMMISSION_PHASE_START,
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    FRAME_TYPE_STATUS,
    PROTOCOL_VERSION,
    SUBSYSTEM_COMMISSIONING,
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


def _metadata_source(kind: int, index: int, label: str) -> bytes:
    label_bytes = label.encode("ascii")
    return bytes([kind, index, len(label_bytes)]) + label_bytes


def _metadata_payload(serial: str) -> bytes:
    sources = b"".join(
        [
            _metadata_source(0, 255, "board"),
            _metadata_source(2, 255, "usb"),
            _metadata_source(3, 255, "klipper"),
            _metadata_source(1, 0, "channel0"),
        ]
    )
    fixed = struct.pack(
        "<BBBBIIIIIIII",
        1,
        1,
        4,
        3,
        84_000_000,
        1000,
        500,  # OpenFFBoard's configured control-detail trace rate.
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
        + sources
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


def _klipper_status_payload(queue_depth: int) -> bytes:
    return struct.pack(
        "<IIIIIHHIIII",
        0,
        0,
        0,
        0,
        0,
        queue_depth,
        0,
        0,
        0,
        0,
        0,
    )


def _commission_event_payload(
    event_id: int,
    *,
    source_index: int = 0,
    engine_kind: int = 1,
    arg2: int = 0,
    arg3: int = 0,
    level: int = 1,
) -> bytes:
    return struct.pack(
        "<BBHBxxxiiii",
        level,
        SUBSYSTEM_COMMISSIONING,
        event_id,
        4,
        source_index,
        engine_kind,
        arg2,
        arg3,
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
    payload += struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)
    payload += struct.pack("<i", 0)
    return payload


def _current_path_payload(
    foc_iq_id: int,
    foc_iwy_iux: int,
    adc_i1_i0: int,
    sample_flags: int = 0,
) -> bytes:
    return struct.pack("<BBBBIII", sample_flags, 0, 0, 0, foc_iq_id, foc_iwy_iux, adc_i1_i0)


def _phase_advance_payload(counts: int, flags: int) -> bytes:
    return struct.pack("<hBBiHH", counts, flags, 0, -60000, 64, 16)


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
    pid_velocity_actual: int = 20,
    velocity_error: int = 10,
    target_velocity: int = 30,
) -> bytes:
    if uq_requested is None:
        uq_requested = uq_limited
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        step_position,
        tmc_position_actual,
        step_position - tmc_position_actual,
        target_velocity,
        pid_velocity_actual,
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
        0,
        _pack_i16_pair(torque_actual, torque_target),
        _pack_i16_pair(0, uq_requested),
        _pack_i16_pair(0, uq_limited),
    )
    payload += struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)
    payload += struct.pack("<i", 0)
    return payload


def _critical_payload_v4(
    *,
    step_position: int,
    tmc_position_actual: int,
    torque_actual: int,
    torque_target: int,
    target_velocity: int,
    cached_velocity_offset: int,
    foc_uq_ud: int,
    foc_uq_ud_limited: int,
    foc_iq_id: int,
    accel_offset: int,
    accel_gain: int,
    accel_flags: int,
) -> bytes:
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        step_position,
        tmc_position_actual,
        step_position - tmc_position_actual,
        target_velocity,
        20,
        cached_velocity_offset,
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
        foc_uq_ud,
        foc_uq_ud_limited,
    )
    payload += struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)
    payload += struct.pack("<III", foc_iq_id, _pack_i16_pair(0, 0), _pack_u16_pair(32000, 32100))
    return payload


def _critical_payload_v6(
    *,
    step_position: int,
    tmc_position_actual: int,
    torque_actual: int,
    torque_target: int,
    target_velocity: int,
    cached_velocity_offset: int,
    cached_velocity_transient_offset: int,
    foc_uq_ud: int,
    foc_uq_ud_limited: int,
    accel_offset: int,
    accel_gain: int,
    accel_flags: int,
) -> bytes:
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        step_position,
        tmc_position_actual,
        step_position - tmc_position_actual,
        target_velocity,
        20,
        cached_velocity_offset,
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
        foc_uq_ud,
        foc_uq_ud_limited,
    )
    payload += struct.pack("<iHBB", accel_offset, accel_gain, accel_flags, 0)
    payload += struct.pack("<i", cached_velocity_transient_offset)
    return payload


def _control_detail_payload(
    torque_cmd: int,
    voltage_limit: int,
    torque_limit: int,
    velocity_pidin: int = 0,
    velocity_pidout: int = 0,
) -> bytes:
    return struct.pack(
        "<IIiiIII",
        torque_cmd & 0xFFFF_FFFF,
        0,
        velocity_pidin,
        velocity_pidout,
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
        125,
        _header(
            FRAME_TYPE_STATUS,
            _klipper_status_payload(9),
            3,
            84_000_000,
            3,
            255,
        )
        + _klipper_status_payload(9),
    )
    writer.write_firmware_frame(
        0,
        130,
        _header(
            FRAME_TYPE_CRITICAL,
            _critical_payload(3, -4, 0xFFFE_0001, 0x0002_0003, -19, 2000, 0x03),
            4,
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
        _header(FRAME_TYPE_STATUS, _status_payload(13, 27), 7, 84_000_000 * 4, 2, 255)
        + _status_payload(13, 27),
    )
    writer.write_firmware_frame(
        0,
        155,
        _header(
            FRAME_TYPE_STATUS,
            _klipper_status_payload(12),
            8,
            84_000_000 * 4,
            3,
            255,
        )
        + _klipper_status_payload(12),
    )

    reports = summarize_capture(io.BytesIO(output.getvalue()), steady_after_s=1.5)

    assert len(reports) == 1
    report = reports[0]
    assert report.usb_path == "1-50"
    assert report.serial == "ABC"
    assert report.metadata_serial == "ABC"
    assert report.source_labels == [
        "board=board",
        "USB=usb",
        "Klipper=klipper",
        "FOCI[0]=channel0",
    ]
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


def test_phase_summary_reports_velocity_loop_tracking() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    samples = [
        (1, 0, 0, 80, 20),
        (2, clock * 20 // 1000, 600, -90, -10),
        (3, clock * 40 // 1000, 1200, 7, -7),
        (4, clock * 80 // 1000, 1800, 60, 15),
    ]
    for seq, timestamp, step, velocity_actual, velocity_error in samples:
        payload = _critical_motion_payload(
            step,
            step,
            torque_actual=400,
            torque_target=800,
            uq_limited=1000,
            pid_velocity_actual=velocity_actual,
            velocity_error=velocity_error,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )

    phase = reports[0].motion_segments[0].phase_reports[0]
    assert phase.name == "entry_0_50ms"
    assert phase.velocity_actual_range == (-90, 80)
    assert phase.velocity_effective_target_range == (-100, 100)
    assert round(phase.velocity_error_rms, 3) == 13.528
    assert phase.velocity_error_peak == 20
    assert phase.velocity_actual_to_effective_target_abs_median == 0.85

    text = format_summary(reports, include_phase_split=True)
    assert "velocity_actual=-90..80" in text
    assert "velocity_effective_target=-100..100" in text
    assert "velocity_error_rms=13.5 velocity_error_peak=20.0" in text
    assert "velocity_actual/effective_target=0.85" in text


def test_motion_velocity_analysis_is_isolated_by_trace_source() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")

    clock = 84_000_000
    critical_samples = [
        (1, 1, 0, 1000, 1000, -40, -10, -2000),
        (2, 0, clock // 1000, 0, 0, 80, 20, 1000),
        (3, 1, clock * 2 // 1000, 900, 900, -50, -10, -2100),
        (4, 0, clock * 3 // 1000, 100, 100, 90, 20, 1100),
        (5, 1, clock * 4 // 1000, 800, 800, -60, -10, -2200),
        (6, 0, clock * 5 // 1000, 200, 200, 100, 20, 1200),
        (7, 1, clock * 6 // 1000, 700, 700, -70, -10, -2300),
        (8, 0, clock * 7 // 1000, 300, 300, 110, 20, 1300),
    ]
    for (
        seq,
        source_index,
        timestamp,
        step,
        actual,
        velocity_actual,
        velocity_error,
        target_velocity,
    ) in critical_samples:
        payload = _critical_motion_payload(
            step,
            actual,
            torque_actual=400,
            torque_target=800,
            uq_limited=1000,
            pid_velocity_actual=velocity_actual,
            velocity_error=velocity_error,
            target_velocity=target_velocity,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(
                FRAME_TYPE_CRITICAL,
                payload,
                seq,
                timestamp,
                1,
                source_index,
            )
            + payload,
        )

    for seq, source_index, timestamp, pidin, pidout in (
        (9, 1, clock // 1000, -201, -211),
        (10, 0, clock * 2 // 1000, 101, 111),
        (11, 1, clock * 3 // 1000, -202, -212),
        (12, 0, clock * 4 // 1000, 102, 112),
    ):
        payload = _control_detail_payload(
            100,
            voltage_limit=3000,
            torque_limit=2816,
            velocity_pidin=pidin,
            velocity_pidout=pidout,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(
                FRAME_TYPE_CONTROL_DETAIL,
                payload,
                seq,
                timestamp,
                1,
                source_index,
            )
            + payload,
        )

    from foci_trace.analyze import analysis_to_json, analyze_capture

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=50,
    )

    segments = analysis.devices[0].motion_segments
    assert [segment.source_key for segment in segments] == [(1, 1), (1, 0)]
    assert [segment.step_delta for segment in segments] == [-300, 300]

    channel_0 = segments[1].phase_reports[0]
    assert channel_0.target_velocity_range == (1000, 1200)
    assert channel_0.velocity_actual_range == (80, 100)
    assert channel_0.velocity_effective_target_range == (100, 120)
    assert channel_0.velocity_error_peak == 20
    assert channel_0.velocity_pidin_target_range == (101, 102)
    assert channel_0.velocity_pidout_target_range == (111, 112)

    channel_1 = segments[0].phase_reports[0]
    assert channel_1.target_velocity_range == (-2200, -2000)
    assert channel_1.velocity_actual_range == (-60, -40)
    assert channel_1.velocity_effective_target_range == (-70, -50)
    assert channel_1.velocity_error_peak == 10
    assert channel_1.velocity_pidin_target_range == (-202, -201)
    assert channel_1.velocity_pidout_target_range == (-212, -211)

    text = format_summary(analysis.devices, include_phase_split=True)
    assert "motion0 source=FOCI[1] delta=-300" in text
    assert "motion1 source=FOCI[0] delta=300" in text

    json_segments = analysis_to_json(analysis)["devices"][0]["motion_segments"]
    assert json_segments[0]["source_kind"] == 1
    assert json_segments[0]["source_index"] == 1
    assert json_segments[1]["source_kind"] == 1
    assert json_segments[1]["source_index"] == 0


def test_multi_source_capture_identifies_a_single_qualifying_motion_segment() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")

    clock = 84_000_000
    for seq, source_index, timestamp, step in (
        (1, 0, 0, 0),
        (2, 1, clock // 1000, 500),
        (3, 0, clock * 2 // 1000, 100),
        (4, 1, clock * 3 // 1000, 500),
        (5, 0, clock * 4 // 1000, 200),
    ):
        payload = _critical_motion_payload(step, step, torque_actual=400, uq_limited=1000)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, source_index) + payload,
        )

    from foci_trace.analyze import analysis_to_json, analyze_capture

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=50,
    )

    assert len(analysis.devices[0].motion_segments) == 1
    assert analysis.devices[0].motion_segments[0].source_key == (1, 0)
    text = format_summary(analysis.devices)
    assert "motion0 source=FOCI[0] delta=200" in text
    json_segment = analysis_to_json(analysis)["devices"][0]["motion_segments"][0]
    assert json_segment["source_kind"] == 1
    assert json_segment["source_index"] == 0


def test_legacy_motion_velocity_tracking_is_unavailable() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    for seq, timestamp, step in (
        (1, 0, 0),
        (2, clock * 20 // 1000, 600),
        (3, clock * 40 // 1000, 1200),
    ):
        payload = _critical_motion_payload(
            step,
            step,
            torque_actual=400,
            torque_target=800,
            uq_limited=1000,
            pid_velocity_actual=0,
            velocity_error=0,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    detail = _control_detail_payload(100, voltage_limit=3000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        clock * 20 // 1000,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 4, clock * 20 // 1000, 1, 0) + detail,
    )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )

    phase = reports[0].motion_segments[0].phase_reports[0]
    assert phase.velocity_actual_range is None
    assert phase.velocity_effective_target_range is None
    assert phase.velocity_error_rms is None
    assert phase.velocity_error_peak is None
    assert phase.velocity_actual_to_effective_target_abs_median is None
    assert phase.velocity_pidin_target_range is None
    assert phase.velocity_pidout_target_range is None

    text = format_summary(reports, include_phase_split=True)
    assert "velocity_actual=n/a" in text
    assert "velocity_effective_target=n/a" in text
    assert "velocity_error_rms=n/a velocity_error_peak=n/a" in text


def test_motion_summary_can_include_feedforward_audit() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    detail = _control_detail_payload(1200, voltage_limit=29000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 1, 0, 1, 0) + detail,
    )

    samples = [
        (2, 0, 0, 0, 400, 800, 600, 540, 18000, 17000, 300, 0x03),
        (3, clock * 20 // 1000, 600, 580, 300, 700, 500, 450, 15000, 14000, 280, 0x03),
        (4, clock * 60 // 1000, 1200, 1188, 80, 100, 80, 70, 4000, 3900, 0, 0x01),
        (
            5,
            clock * 120 // 1000,
            1800,
            1790,
            -350,
            -850,
            -600,
            -540,
            -16000,
            -15000,
            -260,
            0x03,
        ),
        (
            6,
            clock * 160 // 1000,
            2400,
            2410,
            -420,
            -900,
            -650,
            -580,
            -18000,
            -17000,
            -320,
            0x03,
        ),
        (7, clock * 180 // 1000, 2400, 2415, 20, 0, 0, 0, 100, 80, 0, 0x00),
        (8, clock * 260 // 1000, 2400, 2412, 0, 0, 0, 0, 0, 0, 0, 0x00),
    ]
    for (
        seq,
        timestamp,
        step,
        actual,
        torque_actual,
        torque_target,
        target_velocity,
        velocity_ff,
        uq,
        uq_limited,
        accel,
        flags,
    ) in samples:
        payload = _critical_payload_v4(
            step_position=step,
            tmc_position_actual=actual,
            torque_actual=torque_actual,
            torque_target=torque_target,
            target_velocity=target_velocity,
            cached_velocity_offset=velocity_ff,
            foc_uq_ud=_pack_i16_pair(0, uq),
            foc_uq_ud_limited=_pack_i16_pair(0, uq_limited),
            foc_iq_id=_pack_i16_pair(0, torque_actual),
            accel_offset=accel,
            accel_gain=500 if accel else 0,
            accel_flags=flags,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, version=4) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )
    text = format_summary(reports, include_phase_split=True, include_feedforward_audit=True)

    assert "ff_audit" in text
    assert "accel_ff=280..300" in text
    assert "accel_sign=2/2" in text
    assert "velocity_ff=450..540" in text
    assert "velocity_sign=2/2" in text
    assert "iq/torque=" in text
    assert "post_discriminator=settling_possible" in text


def test_feedforward_audit_includes_velocity_transient_offset() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    detail = _control_detail_payload(1200, voltage_limit=29000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 1, 0, 1, 0, version=6) + detail,
    )

    samples = [
        (2, 0, 0, 0, 400, 800, 600, 660, 120, 18000, 17000, 300, 0x03),
        (3, clock * 20 // 1000, 600, 580, 300, 700, 500, 560, 110, 15000, 14000, 280, 0x03),
        (4, clock * 60 // 1000, 1200, 1188, 80, 100, 80, 70, 0, 4000, 3900, 0, 0x01),
        (
            5,
            clock * 120 // 1000,
            1800,
            1790,
            -350,
            -850,
            -600,
            -660,
            -120,
            -16000,
            -15000,
            -260,
            0x03,
        ),
        (
            6,
            clock * 160 // 1000,
            2400,
            2410,
            -420,
            -900,
            -650,
            -690,
            -110,
            -18000,
            -17000,
            -320,
            0x03,
        ),
    ]
    for (
        seq,
        timestamp,
        step,
        actual,
        torque_actual,
        torque_target,
        target_velocity,
        velocity_ff,
        velocity_transient,
        uq,
        uq_limited,
        accel,
        flags,
    ) in samples:
        payload = _critical_payload_v6(
            step_position=step,
            tmc_position_actual=actual,
            torque_actual=torque_actual,
            torque_target=torque_target,
            target_velocity=target_velocity,
            cached_velocity_offset=velocity_ff,
            cached_velocity_transient_offset=velocity_transient,
            foc_uq_ud=_pack_i16_pair(0, uq),
            foc_uq_ud_limited=_pack_i16_pair(0, uq_limited),
            accel_offset=accel,
            accel_gain=500 if accel else 0,
            accel_flags=flags,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, version=6) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )
    text = format_summary(reports, include_phase_split=True, include_feedforward_audit=True)

    assert "velocity_transient=110..120" in text
    assert "transient_nonzero=2/2" in text
    assert "transient_sign=2/2" in text


def test_phase_advance_audit_uses_phase_critical_timestamp_span() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    critical_samples = [
        (1, 0, 0, 0),
        (2, clock * 20 // 1000, 600, 580),
        (3, clock * 80 // 1000, 1200, 1180),
        (4, clock * 140 // 1000, 1800, 1780),
    ]
    for seq, timestamp, step, actual in critical_samples:
        payload = _critical_motion_payload(
            step,
            actual,
            torque_actual=400,
            torque_target=800,
            uq_limited=1000,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    phase_samples = [
        (5, 0, -64, 0b0000_0111),
        (6, clock * 10 // 1000, 0, 0b0001_0000),
        (7, clock * 20 // 1000, 64, 0b0000_0011),
        (8, clock * 40 // 1000, 99, 0b0000_0111),
    ]
    for seq, timestamp, counts, flags in phase_samples:
        payload = _phase_advance_payload(counts, flags)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_PHASE_ADVANCE, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )
    text = format_summary(reports, include_phase_split=True, include_feedforward_audit=True)

    assert (
        "phase_advance=-64..64 nonzero=2/3 clamped=1/3 gated=1/3 "
        "write_failed=0/3 gain=-60000 max=64 deadband=16"
    ) in text
    assert "phase_advance=-64..99" not in text


def test_feedforward_audit_uses_separate_current_path_frames() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    for seq, timestamp, step, torque_actual in (
        (1, 0, 0, 400),
        (2, clock * 20 // 1000, 400, 300),
        (3, clock * 60 // 1000, 800, 100),
        (4, clock * 120 // 1000, 1200, 100),
    ):
        payload = _critical_motion_payload(
            step,
            step,
            torque_actual=torque_actual,
            torque_target=800,
            uq_limited=1000,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    for seq, timestamp, torque_current in (
        (5, 0, 800),
        (6, clock * 20 // 1000, 600),
    ):
        payload = _current_path_payload(
            _pack_i16_pair(0, torque_current),
            _pack_i16_pair(0, 0),
            _pack_u16_pair(32000, 32100),
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CURRENT_PATH, payload, seq, timestamp, 1, 0) + payload,
        )

    detail = _control_detail_payload(1200, voltage_limit=29000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 7, 0, 1, 0) + detail,
    )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )
    text = format_summary(reports, include_phase_split=True, include_feedforward_audit=True)

    assert "phase entry_0_50ms" in text
    assert "iq/torque=2.00" in text


def test_decoupling_audit_reports_cross_coupling_voltage_fraction() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    detail = _control_detail_payload(1200, voltage_limit=29000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 1, 0, 1, 0, version=4) + detail,
    )

    samples = [
        (2, 0, 0, 0, 400, 800, 10, 0, 4000, 1000, 50, 100),
        (3, clock * 20 // 1000, 600, 580, 300, 700, 10, 0, 4000, 1000, 50, 100),
        (4, clock * 80 // 1000, 1200, 1180, 100, 200, 10, 0, 4000, 1000, 50, 100),
        (5, clock * 140 // 1000, 1800, 1790, -300, -700, -10, 0, -4000, -1000, -50, -100),
        (6, clock * 180 // 1000, 2400, 2410, -400, -800, -10, 0, -4000, -1000, -50, -100),
    ]
    for (
        seq,
        timestamp,
        step,
        actual,
        torque_actual,
        torque_target,
        target_velocity,
        velocity_ff,
        uq,
        ud,
        iq,
        id_,
    ) in samples:
        payload = _critical_payload_v4(
            step_position=step,
            tmc_position_actual=actual,
            torque_actual=torque_actual,
            torque_target=torque_target,
            target_velocity=target_velocity,
            cached_velocity_offset=velocity_ff,
            foc_uq_ud=_pack_i16_pair(ud, uq),
            foc_uq_ud_limited=_pack_i16_pair(ud, uq),
            foc_iq_id=_pack_i16_pair(id_, iq),
            accel_offset=0,
            accel_gain=0,
            accel_flags=0,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, version=4) + payload,
        )

    from foci_trace.analyze import analyze_capture, format_analysis

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
        decoupling_l_int=4000,
    )
    text = format_analysis(
        analysis,
        include_phase_split=True,
        include_decoupling_audit=True,
    )

    assert "decoupling pole_pairs=50" in text
    assert "l_int=4000" in text
    assert "omega_e=1198.4..1198.4rad/s" in text
    assert "cross/uq_req=0.13" in text
    assert "cross/uq_lim=0.13" in text


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


def test_current_path_frames_are_matched_by_source_index() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")

    clock = 84_000_000
    for seq, timestamp, step in (
        (1, clock // 1000, 0),
        (2, (clock // 1000) * 2, 10),
    ):
        payload = _critical_motion_payload(
            step,
            step,
            torque_actual=100,
            torque_target=2800,
            uq_limited=1200,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 1) + payload,
        )

    other_channel_current = _current_path_payload(
        _pack_i16_pair(0, 900),
        _pack_i16_pair(0, 0),
        _pack_u16_pair(32000, 32100),
    )
    writer.write_firmware_frame(
        0,
        clock // 1000,
        _header(
            FRAME_TYPE_CURRENT_PATH,
            other_channel_current,
            3,
            clock // 1000,
            1,
            0,
        )
        + other_channel_current,
    )

    detail = _control_detail_payload(2800, voltage_limit=3000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        clock // 1000,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 4, clock // 1000, 1, 1) + detail,
    )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=5,
    )

    segment = reports[0].motion_segments[0]
    assert segment.foc_iq_range is None
    assert segment.foc_id_range is None
    assert segment.current_path_max_gap_s is None
    assert segment.current_authority is not None
    assert segment.current_authority.iq_to_actual_abs_median is None


def test_analyze_capture_preserves_existing_summary_api() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
    payload = _critical_motion_payload(0, 0, 100, 1000)
    writer.write_firmware_frame(
        0,
        123_456,
        _header(FRAME_TYPE_CRITICAL, payload, 1, 84_000, 1, 0) + payload,
    )

    from foci_trace.analyze import analyze_capture, format_analysis

    analysis = analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)

    assert analysis.devices[0].serial == "ABC"
    assert analysis.contour_segments == []
    assert "contour_mapping" not in format_analysis(analysis, include_contour=True)
    assert summarize_capture(io.BytesIO(output.getvalue()), steady_after_s=0)[0].serial == "ABC"


def test_analyze_capture_includes_commissioning_report() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    events = [
        (1, 1_000, EVENT_COMMISSION_ENGINE_START, 1, 7),
        (2, 10_000, EVENT_COMMISSION_PHASE_START, 17, 3),
        (3, 80_000, EVENT_COMMISSION_PHASE_END, 17, 0),
        (4, 81_000, EVENT_COMMISSION_ENGINE_TERMINAL, 0, 7),
    ]
    for seq, timestamp, event_id, arg2, arg3 in events:
        payload = _commission_event_payload(event_id, arg2=arg2, arg3=arg3)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_EVENT, payload, seq, timestamp, 255, 255) + payload,
        )
    for idx in range(5):
        payload = _critical_motion_payload(idx, idx, 100, 1000)
        timestamp = 10_000 + idx * 1_000
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, 10 + idx, timestamp, 1, 0) + payload,
        )

    from foci_trace.analyze import analyze_capture, format_analysis

    analysis = analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)
    text = format_analysis(analysis)

    assert analysis.commissioning.runs[0].result == "accepted"
    assert analysis.commissioning.runs[0].phases[0].critical_samples == 5
    assert "commissioning dev0 source=0 engine=commission" in text
    assert "phase ClosedLoopEntry" in text


def test_contour_analysis_reports_along_and_cross_error() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    plus_serial = "PLUS"
    minus_serial = "MINUS"
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", plus_serial, b"")
    writer.write_device_table(1, 100, "1-2", "FOCI OpenFFBoard (trace)", minus_serial, b"")
    clock = 84_000_000

    samples = [
        (0, 0, 0, 0),
        (clock // 1000, 1000, 950, 950),
        ((clock // 1000) * 2, 2000, 1900, 1900),
    ]
    for seq, (timestamp, commanded, actual_plus, actual_minus) in enumerate(samples, 1):
        plus_payload = _critical_motion_payload(commanded, actual_plus, 100, 1000)
        minus_payload = _critical_motion_payload(commanded, actual_minus, 100, 1000)
        writer.write_firmware_frame(
            0,
            10_000 + seq * 1_000,
            _header(FRAME_TYPE_CRITICAL, plus_payload, seq, timestamp, 1, 0) + plus_payload,
        )
        writer.write_firmware_frame(
            1,
            10_050 + seq * 1_000,
            _header(FRAME_TYPE_CRITICAL, minus_payload, seq, timestamp, 1, 0) + minus_payload,
        )

    from foci_trace.analyze import analyze_capture, format_analysis

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.01,
        motion_min_steps=100,
        corexy_plus=plus_serial,
        corexy_minus=minus_serial,
        contour_pair_tolerance_ms=1.0,
    )

    assert len(analysis.contour_segments) == 1
    segment = analysis.contour_segments[0]
    assert segment.classification == "along_path_lag"
    assert segment.path_classification == "along_path_lag"
    assert not segment.position_trace_gap
    assert not segment.support_trace_gap
    assert segment.post_discriminator is None
    assert segment.along_rms > 0
    assert segment.cross_rms == 0
    text = format_analysis(analysis, include_contour=True)
    assert "contour_mapping plus=PLUS minus=MINUS" in text
    assert "contour0" in text
    assert "path_class=along_path_lag" in text
    assert "position_gap=0" in text
    assert "support_gap=0" in text
    assert "post=n/a" in text


def test_contour_analysis_uses_existing_post_phase_discriminator() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    plus_serial = "PLUS"
    minus_serial = "MINUS"
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", plus_serial, b"")
    writer.write_device_table(1, 100, "1-2", "FOCI OpenFFBoard (trace)", minus_serial, b"")
    clock = 84_000_000

    paired_samples = [
        (1, 0, 0, 0, 11_000, 11_050),
        (2, clock // 1000, 1000, 950, 12_000, 12_050),
        (3, (clock // 1000) * 2, 2000, 1900, 13_000, 13_050),
    ]
    for seq, timestamp, commanded, actual, plus_host_ns, minus_host_ns in paired_samples:
        plus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        minus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        writer.write_firmware_frame(
            0,
            plus_host_ns,
            _header(FRAME_TYPE_CRITICAL, plus_payload, seq, timestamp, 1, 0) + plus_payload,
        )
        writer.write_firmware_frame(
            1,
            minus_host_ns,
            _header(FRAME_TYPE_CRITICAL, minus_payload, seq, timestamp, 1, 0) + minus_payload,
        )

    plus_post = _critical_motion_payload(
        2000,
        1900,
        torque_actual=150,
        torque_target=150,
        uq_limited=0,
    )
    writer.write_firmware_frame(
        0,
        1_000,
        _header(FRAME_TYPE_CRITICAL, plus_post, 4, (clock // 1000) * 3, 1, 0) + plus_post,
    )

    from foci_trace.analyze import analyze_capture

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
        corexy_plus=plus_serial,
        corexy_minus=minus_serial,
        contour_pair_tolerance_ms=1.0,
    )

    phase_reports = analysis.devices[0].motion_segments[0].phase_reports
    assert any(phase.post_discriminator == "loop_active" for phase in phase_reports)
    assert analysis.contour_segments[0].classification == "endpoint_loop_active"


def test_contour_analysis_applies_post_discriminator_per_segment() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    plus_serial = "PLUS"
    minus_serial = "MINUS"
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", plus_serial, b"")
    writer.write_device_table(1, 100, "1-2", "FOCI OpenFFBoard (trace)", minus_serial, b"")
    clock = 84_000_000

    paired_samples = [
        (1, 0, 0, 0, 10_000, 10_050),
        (2, clock // 1000, 1000, 950, 11_000, 11_050),
        (3, (clock // 1000) * 2, 2000, 1900, 12_000, 12_050),
        (4, (clock // 1000) * 120, 4000, 3950, 120_000_000, 120_000_050),
        (5, (clock // 1000) * 121, 5000, 4900, 120_001_000, 120_001_050),
        (6, (clock // 1000) * 122, 6000, 5850, 120_002_000, 120_002_050),
    ]
    for seq, timestamp, commanded, actual, plus_host_ns, minus_host_ns in paired_samples:
        plus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        minus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        writer.write_firmware_frame(
            0,
            plus_host_ns,
            _header(FRAME_TYPE_CRITICAL, plus_payload, seq, timestamp, 1, 0) + plus_payload,
        )
        writer.write_firmware_frame(
            1,
            minus_host_ns,
            _header(FRAME_TYPE_CRITICAL, minus_payload, seq, timestamp, 1, 0) + minus_payload,
        )

    plus_post = _critical_motion_payload(
        6000,
        5850,
        torque_actual=150,
        torque_target=150,
        uq_limited=0,
    )
    writer.write_firmware_frame(
        0,
        120_003_000,
        _header(FRAME_TYPE_CRITICAL, plus_post, 7, (clock // 1000) * 123, 1, 0) + plus_post,
    )

    from foci_trace.analyze import analyze_capture

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.05,
        motion_min_steps=100,
        corexy_plus=plus_serial,
        corexy_minus=minus_serial,
        contour_pair_tolerance_ms=1.0,
    )

    assert len(analysis.contour_segments) == 2
    assert any(
        phase.post_discriminator == "loop_active"
        for phase in analysis.devices[0].motion_segments[1].phase_reports
    )
    assert analysis.contour_segments[0].classification == "along_path_lag"
    assert analysis.contour_segments[1].classification == "endpoint_loop_active"


def test_contour_analysis_applies_trace_gap_per_segment() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    plus_serial = "PLUS"
    minus_serial = "MINUS"
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", plus_serial, b"")
    writer.write_device_table(1, 100, "1-2", "FOCI OpenFFBoard (trace)", minus_serial, b"")
    clock = 84_000_000

    paired_samples = [
        (1, 0, 0, 0, 10_000, 10_050),
        (2, clock // 1000, 1000, 950, 11_000, 11_050),
        (3, (clock // 1000) * 2, 2000, 1900, 12_000, 12_050),
        (4, (clock // 1000) * 120, 4000, 3950, 120_000_000, 120_000_050),
        (5, (clock // 1000) * 121, 5000, 4900, 120_001_000, 120_001_050),
        (6, (clock // 1000) * 171, 6000, 5850, 120_002_000, 120_002_050),
    ]
    for seq, timestamp, commanded, actual, plus_host_ns, minus_host_ns in paired_samples:
        plus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        minus_payload = _critical_motion_payload(commanded, actual, 100, 1000)
        writer.write_firmware_frame(
            0,
            plus_host_ns,
            _header(FRAME_TYPE_CRITICAL, plus_payload, seq, timestamp, 1, 0) + plus_payload,
        )
        writer.write_firmware_frame(
            1,
            minus_host_ns,
            _header(FRAME_TYPE_CRITICAL, minus_payload, seq, timestamp, 1, 0) + minus_payload,
        )

    from foci_trace.analyze import analyze_capture

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
        corexy_plus=plus_serial,
        corexy_minus=minus_serial,
        contour_pair_tolerance_ms=1.0,
    )

    assert len(analysis.contour_segments) == 2
    assert not analysis.devices[0].motion_segments[0].trace_gap_suspect
    assert analysis.devices[0].motion_segments[1].trace_gap_suspect
    assert analysis.contour_segments[0].classification == "along_path_lag"
    assert analysis.contour_segments[1].classification == "inconclusive"
    assert analysis.contour_segments[1].path_classification == "along_path_lag"
    assert analysis.contour_segments[1].position_trace_gap
    assert not analysis.contour_segments[1].support_trace_gap


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


def test_main_requires_both_corexy_mapping_arguments(tmp_path, capsys) -> None:
    capture_path = tmp_path / "empty.bin"
    with capture_path.open("wb") as output:
        CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)

    from foci_trace.analyze import main

    rc = main([str(capture_path), "--contour-analysis", "--corexy-plus", "PLUS"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "both --corexy-plus and --corexy-minus are required" in captured.err


def test_main_reports_missing_corexy_selector_without_traceback(tmp_path, capsys) -> None:
    capture_path = tmp_path / "capture.bin"
    with capture_path.open("wb") as output:
        writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
        writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    from foci_trace.analyze import main

    rc = main(
        [
            str(capture_path),
            "--contour-analysis",
            "--corexy-plus",
            "NOPE",
            "--corexy-minus",
            "ABC",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "no trace device matches 'NOPE'" in captured.err
    assert "Traceback" not in captured.err


def test_main_ignores_corexy_mapping_arguments_without_contour_analysis(tmp_path, capsys) -> None:
    capture_path = tmp_path / "capture.bin"
    with capture_path.open("wb") as output:
        writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
        writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    from foci_trace.analyze import main

    rc = main(
        [
            str(capture_path),
            "--corexy-plus",
            "NOPE",
            "--corexy-minus",
            "MISSING",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "dev0 1-50 FOCI OpenFFBoard (trace) serial=ABC" in captured.out


def test_main_json_includes_commissioning_section(tmp_path, capsys) -> None:
    capture_path = tmp_path / "capture.bin"
    with capture_path.open("wb") as output:
        writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
        writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
        for seq, timestamp, event_id, arg2, arg3 in (
            (1, 1_000, EVENT_COMMISSION_ENGINE_START, 1, 7),
            (2, 10_000, EVENT_COMMISSION_PHASE_START, 17, 3),
            (3, 80_000, EVENT_COMMISSION_PHASE_END, 17, 0),
            (4, 81_000, EVENT_COMMISSION_ENGINE_TERMINAL, 0, 7),
        ):
            payload = _commission_event_payload(event_id, arg2=arg2, arg3=arg3)
            writer.write_firmware_frame(
                0,
                timestamp,
                _header(FRAME_TYPE_EVENT, payload, seq, timestamp, 255, 255) + payload,
            )

    from foci_trace.analyze import main

    rc = main([str(capture_path), "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["devices"][0]["serial"] == "ABC"
    assert data["commissioning"]["runs"][0]["result"] == "accepted"
