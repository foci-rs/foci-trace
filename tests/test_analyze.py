import io
import json
import struct

import pytest

from foci_trace.analyze import (
    CaptureAnalysis,
    DeviceReport,
    _marker_based_phase_bounds,
    _PhaseBoundaryObservation,
    _StatusObservation,
    analysis_to_json,
    analyze_capture,
    audit_outer_safety_timing,
    format_analysis,
    format_summary,
    summarize_capture,
    velocity_analysis_to_json,
)
from foci_trace.capture_file import CaptureWriter
from foci_trace.protocol import (
    EVENT_COMMISSION_DIAGNOSTIC_WARNING,
    EVENT_COMMISSION_ENGINE_START,
    EVENT_COMMISSION_ENGINE_TERMINAL,
    EVENT_COMMISSION_OUTER_SAFETY_TIMING,
    EVENT_COMMISSION_PHASE_END,
    EVENT_COMMISSION_PHASE_START,
    EVENT_PHASE_BOUNDARY,
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    FRAME_TYPE_STATUS,
    FRAME_TYPE_VELOCITY_OBSERVATION,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    PROTOCOL_VERSION,
    SOURCE_KIND_BOARD,
    SUBSYSTEM_COMMISSIONING,
    SUBSYSTEM_STEP_TIMER,
)

_SOURCE_KIND_FOCI = 1

_BREAKAWAY_PROBE_CONTEXT = {
    "workflow_digest": 0xA1B2_C3D4_E5F6_0718,
    "target_velocity_rpm": -600,
    "hard_torque_limit": 2_816,
    "usable_torque_limit": 2_048,
    "pwm_hz": 25_000,
    "encoder_counts_per_rev": 4_000,
}


def _breakaway_probe_plan_payload(
    *,
    run_sequence: int,
    evidence_sequence: int,
    plan_digest: int,
    max_observations: int,
    search_count: int,
    p_start_raw: int,
    p_top_raw: int,
    motion_threshold_counts: int = 63,
    max_capture_interval_us: int = 2_000,
) -> bytes:
    """Encode the complete v15 probe-plan record used by analyzer fixtures."""
    return bytes([0]) + struct.pack(
        "<IHQBBHHQiHHIIII",
        run_sequence,
        evidence_sequence,
        plan_digest,
        max_observations,
        search_count,
        p_start_raw,
        p_top_raw,
        _BREAKAWAY_PROBE_CONTEXT["workflow_digest"],
        _BREAKAWAY_PROBE_CONTEXT["target_velocity_rpm"],
        _BREAKAWAY_PROBE_CONTEXT["hard_torque_limit"],
        _BREAKAWAY_PROBE_CONTEXT["usable_torque_limit"],
        _BREAKAWAY_PROBE_CONTEXT["pwm_hz"],
        _BREAKAWAY_PROBE_CONTEXT["encoder_counts_per_rev"],
        motion_threshold_counts,
        max_capture_interval_us,
    )


def test_outer_safety_timing_audit_reviews_margin_and_raw_coverage() -> None:
    summary = {
        "engine_sequence": 11,
        "run_sequence": 17,
        "sample_count": 9,
        "bin_counts": [2, 2, 2, 3],
        "max_realized_gap_us": 15_000,
        "zero_count": 0,
        "over_bound_count": 0,
    }

    report = audit_outer_safety_timing(summary, [5_000, 5_001, 10_001], [11, 15_000])

    assert report["gap_headroom_us"] == 5_000
    assert report["reopen_bound"] is False
    assert report["raw_covered_bin_counts"] == [1, 1, 1, 0]
    assert report["firmware_reported_safety_only"] == 6
    assert report["matches_firmware"] is True

    summary["max_realized_gap_us"] = 15_001
    assert audit_outer_safety_timing(summary, [], [11, 15_001])["reopen_bound"] is True


def test_outer_safety_timing_audit_keeps_historical_absence_distinct_from_mismatch() -> None:
    historical = audit_outer_safety_timing(None, [], None)
    assert historical["summary_present"] is False
    assert historical["historical_absence_allowed"] is True
    assert historical["matches_firmware"] is True

    missing_record = audit_outer_safety_timing(None, [], [11, 14_329])
    assert missing_record["matches_firmware"] is False
    assert "marker has no canonical timing record" in missing_record["mismatches"]


def test_outer_safety_timing_event_uses_commissioning_routing_prefix() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
    timing = bytes([30]) + struct.pack(
        "<IIIIIIIIBBBBBBBB",
        11,
        17,
        9,
        2,
        2,
        2,
        3,
        15_000,
        1,
        0,
        2,
        15,
        4,
        1,
        0,
        0,
    )
    writer.write_firmware_frame(
        0,
        1_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, timing, 1, 1_000, 1, 0) + timing,
    )
    marker = _commission_event_payload(
        EVENT_COMMISSION_OUTER_SAFETY_TIMING,
        source_index=0,
        engine_kind=4,
        arg2=11,
        arg3=15_000,
    )
    writer.write_firmware_frame(
        0,
        2_000,
        _header(FRAME_TYPE_EVENT, marker, 2, 2_000, 255, 255) + marker,
    )

    analysis = analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)

    assert len(analysis.outer_safety_timing_audits) == 1
    assert analysis.outer_safety_timing_audits[0]["source_index"] == 0
    assert analysis.outer_safety_timing_audits[0]["engine_sequence"] == 11
    assert analysis.outer_safety_timing_audits[0]["matches_firmware"] is True


def _header(
    frame_type: int,
    payload: bytes,
    seq: int,
    ts: int,
    source_kind: int,
    source_index: int,
    flags: int = 0,
    version: int = PROTOCOL_VERSION,
    boot_id: int = 1234,
) -> bytes:
    return struct.pack(
        "<IBBBBHBBI Q I I".replace(" ", ""),
        0x43525446,
        version,
        32,
        frame_type,
        flags,
        len(payload),
        source_kind,
        source_index,
        seq,
        ts,
        boot_id,
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


def _velocity_observation_payload(
    target_cpms: int,
    delta_counts: int,
    *,
    elapsed_us: int = 1_000,
    stroke_index: int = 0,
    direction: int = 1,
    flags: int = 0,
    pid_velocity_actual: int | None = None,
) -> bytes:
    payload = struct.pack(
        "<iiIHbB",
        target_cpms,
        delta_counts,
        elapsed_us,
        stroke_index,
        direction,
        flags | (0b10 if pid_velocity_actual is not None else 0),
    )
    if pid_velocity_actual is not None:
        payload += struct.pack("<Hi", 0, pid_velocity_actual)
    return payload


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
    cached_velocity_clamped: bool = False,
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
    payload += struct.pack(
        "<iHBB", accel_offset, accel_gain, accel_flags, 1 if cached_velocity_clamped else 0
    )
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
        _header(FRAME_TYPE_STATUS, _status_payload(10, 20), 2, 84_000_000, 2, 255, version=21)
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
            version=21,
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
        _header(FRAME_TYPE_STATUS, _status_payload(13, 27), 7, 84_000_000 * 4, 2, 255, version=21)
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
            version=21,
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
        "accel",
        "cruise",
        "decel",
        "post_0_500ms",
    ]
    assert segment.phase_reports[0].critical_samples == 2
    assert segment.phase_reports[1].critical_samples == 2
    assert segment.phase_reports[2].critical_samples == 2
    assert segment.phase_reports[3].critical_samples == 2
    assert segment.phase_reports[0].kinematic_fallback is True
    assert segment.phase_reports[1].kinematic_fallback is True
    assert segment.phase_reports[2].kinematic_fallback is True
    assert segment.phase_reports[3].kinematic_fallback is False

    default_text = format_summary(reports)
    assert "phase accel" not in default_text

    phase_text = format_summary(reports, include_phase_split=True)
    assert "phase accel samples=2" in phase_text
    assert "phase cruise samples=2" in phase_text
    assert "phase decel samples=2" in phase_text
    assert "phase post_0_500ms samples=2" in phase_text
    assert "actual/target=" in phase_text
    assert "uq/limit=" in phase_text


def test_phase_split_follows_kinematics_not_a_fixed_window() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    profile = [
        (0, 0),
        (5, 50),
        (10, 150),
        (15, 300),
        (20, 500),
        (25, 700),
        (30, 900),
        (35, 1100),
        (40, 1300),
        (45, 1500),
        (50, 1700),
        (55, 1900),
        (60, 2100),
        (65, 2300),
        (70, 2500),
        (75, 2650),
        (80, 2750),
        (85, 2800),
        (200, 2800),
        (450, 2800),
    ]
    for seq, (time_ms, step) in enumerate(profile, start=1):
        timestamp = clock * time_ms // 1000
        payload = _critical_motion_payload(step, step, torque_actual=0, uq_limited=0)
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=1000,
    )

    segment = reports[0].motion_segments[0]
    assert [phase.name for phase in segment.phase_reports] == [
        "accel",
        "cruise",
        "decel",
        "post_0_500ms",
    ]
    accel, cruise, decel, post = segment.phase_reports
    assert accel.critical_samples == 3
    assert cruise.critical_samples == 11
    assert decel.critical_samples == 4
    assert post.critical_samples == 2
    assert accel.kinematic_fallback is False
    assert cruise.kinematic_fallback is False
    assert decel.kinematic_fallback is False


def _firmware_marker_capture_reports(marker_boot_id: int) -> list[DeviceReport]:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")

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
            0, timestamp, _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload
        )

    marker_accel_cruise_tick = clock * 10 // 1000
    marker_accel_cruise = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 0, 1, 0)
    writer.write_firmware_frame(
        0,
        marker_accel_cruise_tick,
        _header(
            FRAME_TYPE_EVENT,
            marker_accel_cruise,
            100,
            marker_accel_cruise_tick,
            0,
            0xFF,
            boot_id=marker_boot_id,
        )
        + marker_accel_cruise,
    )
    marker_cruise_decel_tick = clock * 125 // 1000
    marker_cruise_decel = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 1, 2, 0)
    writer.write_firmware_frame(
        0,
        marker_cruise_decel_tick,
        _header(
            FRAME_TYPE_EVENT,
            marker_cruise_decel,
            101,
            marker_cruise_decel_tick,
            0,
            0xFF,
            boot_id=marker_boot_id,
        )
        + marker_cruise_decel,
    )

    status_before = _status_payload(drops=0, endpoint_pressure=0)
    writer.write_firmware_frame(
        0,
        0,
        _header(
            FRAME_TYPE_STATUS,
            status_before,
            102,
            0,
            0,
            0,
            version=21,
            boot_id=marker_boot_id,
        )
        + status_before,
    )
    status_after_tick = clock * 500 // 1000
    status_after = _status_payload(drops=0, endpoint_pressure=0)
    writer.write_firmware_frame(
        0,
        status_after_tick,
        _header(
            FRAME_TYPE_STATUS,
            status_after,
            103,
            status_after_tick,
            0,
            0,
            version=21,
            boot_id=marker_boot_id,
        )
        + status_after,
    )

    return summarize_capture(
        io.BytesIO(output.getvalue()), steady_after_s=0, motion_gap_s=0.1, motion_min_steps=1000
    )


def test_motion_phase_report_ignores_markers_from_another_boot() -> None:
    reports = _firmware_marker_capture_reports(marker_boot_id=5678)

    segment = reports[0].motion_segments[0]
    assert {p.phase_boundary_source for p in segment.phase_reports if p.name != "post_0_500ms"} == {
        "kinematic"
    }


def test_motion_phase_report_prefers_firmware_markers_when_reliable() -> None:
    reports = _firmware_marker_capture_reports(marker_boot_id=1234)

    segment = reports[0].motion_segments[0]
    accel = next(p for p in segment.phase_reports if p.name == "accel")
    cruise = next(p for p in segment.phase_reports if p.name == "cruise")
    decel = next(p for p in segment.phase_reports if p.name == "decel")
    assert accel.phase_boundary_source == "firmware_marker"
    assert cruise.phase_boundary_source == "firmware_marker"
    assert decel.phase_boundary_source == "firmware_marker"
    assert accel.kinematic_fallback is False
    assert cruise.kinematic_fallback is False
    assert decel.kinematic_fallback is False
    assert accel.critical_samples == 1


def test_motion_phase_report_falls_back_to_kinematic_when_markers_missing() -> None:
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
            0, timestamp, _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0) + payload
        )
    reports = summarize_capture(
        io.BytesIO(output.getvalue()), steady_after_s=0, motion_gap_s=0.1, motion_min_steps=1000
    )
    segment = reports[0].motion_segments[0]
    for name in ("accel", "cruise", "decel"):
        phase = next(p for p in segment.phase_reports if p.name == name)
        assert phase.phase_boundary_source == "kinematic"


def test_unreliable_markers_in_one_motion_window_do_not_affect_another() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")

    clock = 84_000_000
    gap_ticks = int(0.5 * clock)

    def _write_move(base_tick: int, base_seq: int) -> None:
        samples = [
            (0, 0, 0, 500, 2200, 1000),
            (clock * 25 // 1000, 2000, 1800, 600, 2600, 1600),
            (clock * 75 // 1000, 5000, 5100, 700, 500, 1200),
            (clock * 125 // 1000, 8000, 7900, 650, 400, 1000),
            (clock * 175 // 1000, 11000, 10800, -500, -2500, -1800),
        ]
        for offset, (rel_tick, step, actual, torque_actual, torque_target, uq_limited) in enumerate(
            samples
        ):
            payload = _critical_motion_payload(
                step,
                actual,
                torque_actual=torque_actual,
                torque_target=torque_target,
                uq_limited=uq_limited,
            )
            tick = base_tick + rel_tick
            writer.write_firmware_frame(
                0,
                tick,
                _header(FRAME_TYPE_CRITICAL, payload, base_seq + offset, tick, 1, 0) + payload,
            )

    _write_move(0, 1)
    _write_move(gap_ticks, 10)

    m1 = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 0, 1, 0)
    writer.write_firmware_frame(0, 0, _header(FRAME_TYPE_EVENT, m1, 100, 0, 0, 0xFF) + m1)
    m2 = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 1, 2, 0)
    writer.write_firmware_frame(
        0,
        clock * 125 // 1000,
        _header(FRAME_TYPE_EVENT, m2, 101, clock * 125 // 1000, 0, 0xFF) + m2,
    )
    status_before_1 = _status_payload(drops=0, endpoint_pressure=0)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_STATUS, status_before_1, 102, 0, 0, 0, version=21) + status_before_1,
    )
    status_after_1 = _status_payload(drops=0, endpoint_pressure=0)
    writer.write_firmware_frame(
        0,
        clock * 190 // 1000,
        _header(FRAME_TYPE_STATUS, status_after_1, 103, clock * 190 // 1000, 0, 0, version=21)
        + status_after_1,
    )

    m3 = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 0, 1, 0)
    writer.write_firmware_frame(
        0, gap_ticks, _header(FRAME_TYPE_EVENT, m3, 104, gap_ticks, 0, 0xFF, flags=1) + m3
    )
    m4_tick = gap_ticks + clock * 125 // 1000
    m4 = struct.pack("<BBHBxxxiiii", 1, 2, 12, 3, 0, 1, 2, 0)
    writer.write_firmware_frame(
        0, m4_tick, _header(FRAME_TYPE_EVENT, m4, 105, m4_tick, 0, 0xFF) + m4
    )
    status_after_2_tick = gap_ticks + clock * 190 // 1000
    status_after_2 = _status_payload(drops=0, endpoint_pressure=0)
    writer.write_firmware_frame(
        0,
        status_after_2_tick,
        _header(FRAME_TYPE_STATUS, status_after_2, 106, status_after_2_tick, 0, 0, version=21)
        + status_after_2,
    )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()), steady_after_s=0, motion_gap_s=0.1, motion_min_steps=1000
    )

    window_1_cruise = next(
        p for p in reports[0].motion_segments[0].phase_reports if p.name == "cruise"
    )
    window_2_cruise = next(
        p for p in reports[0].motion_segments[1].phase_reports if p.name == "cruise"
    )
    assert window_1_cruise.phase_boundary_source == "firmware_marker"
    assert window_2_cruise.phase_boundary_source == "kinematic"


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
    assert phase.name == "accel"
    assert phase.kinematic_fallback is True
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


def test_feedforward_audit_includes_velocity_clamp_engaged_samples() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    clock = 84_000_000
    detail = _control_detail_payload(1200, voltage_limit=29000, torque_limit=2816)
    writer.write_firmware_frame(
        0,
        0,
        _header(FRAME_TYPE_CONTROL_DETAIL, detail, 1, 0, 1, 0, version=20) + detail,
    )

    samples = [
        (2, 0, 0, 0, 400, 800, 600, 660, 120, 18000, 17000, 300, 0x03, True),
        (
            3,
            clock * 20 // 1000,
            600,
            580,
            300,
            700,
            500,
            560,
            110,
            15000,
            14000,
            280,
            0x03,
            False,
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
        clamped,
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
            cached_velocity_clamped=clamped,
        )
        writer.write_firmware_frame(
            0,
            timestamp,
            _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, version=20) + payload,
        )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=100,
    )
    text = format_summary(reports, include_phase_split=True, include_feedforward_audit=True)

    phase = reports[0].motion_segments[0].phase_reports[0]
    assert phase.velocity_ff_clamp_engaged_samples == 1
    assert "clamp_engaged=1/2" in text


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

    assert "phase accel" in text
    assert "iq/torque=2.00" in text


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
    assert "phase ClosedLoopActivation" in text


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


def test_main_writes_deterministic_velocity_report_file(tmp_path, capsys) -> None:
    capture_path = tmp_path / "empty.bin"
    report_path = tmp_path / "velocity.json"
    with capture_path.open("wb") as output:
        CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)

    from foci_trace.analyze import main

    rc = main([str(capture_path), "--velocity-json", str(report_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert json.loads(report_path.read_text()) == {
        "groups": [],
        "outer_safety_timing": [],
    }
    assert report_path.read_text().endswith("\n")


def test_velocity_frames_group_by_device_source_and_engine_run() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    for device_index, serial, complete in ((0, "X", True), (1, "Y", False)):
        writer.write_device_table(
            device_index,
            100,
            f"1-{device_index}",
            "FOCI OpenFFBoard (trace)",
            serial,
            b"",
        )
        sequence = 1
        start = _commission_event_payload(EVENT_COMMISSION_ENGINE_START, arg3=7)
        writer.write_firmware_frame(
            device_index,
            1_000,
            _header(FRAME_TYPE_EVENT, start, sequence, 1_000, 255, 255) + start,
        )
        sequence += 2 if device_index == 0 else 1
        for direction, target, stroke_index in ((1, 4, 0), (-1, -4, 1)):
            for delta in (direction, 0, direction, direction):
                payload = _velocity_observation_payload(
                    target,
                    delta,
                    stroke_index=stroke_index,
                    direction=direction,
                )
                writer.write_firmware_frame(
                    device_index,
                    2_000 + sequence,
                    _header(
                        FRAME_TYPE_VELOCITY_OBSERVATION,
                        payload,
                        sequence,
                        2_000 + sequence,
                        1,
                        0,
                    )
                    + payload,
                )
                sequence += 1
        settle = _velocity_observation_payload(
            0,
            0,
            stroke_index=0x8000,
            direction=0,
        )
        writer.write_firmware_frame(
            device_index,
            9_000,
            _header(
                FRAME_TYPE_VELOCITY_OBSERVATION,
                settle,
                sequence,
                9_000,
                1,
                0,
            )
            + settle,
        )
        sequence += 1
        if complete:
            terminal = _commission_event_payload(EVENT_COMMISSION_ENGINE_TERMINAL, arg2=0, arg3=7)
            writer.write_firmware_frame(
                device_index,
                10_000,
                _header(FRAME_TYPE_EVENT, terminal, sequence, 10_000, 255, 255) + terminal,
            )

    analysis = analyze_capture(io.BytesIO(output.getvalue()))

    assert len(analysis.velocity_reports) == 2
    x_group, y_group = analysis.velocity_reports
    assert (x_group.device_index, x_group.serial, x_group.engine_seq) == (0, "X", 7)
    assert set(x_group.report.directions) == {-1, 1}
    assert x_group.missing_frames == 1
    assert x_group.caveats == []
    assert (y_group.device_index, y_group.serial, y_group.engine_seq) == (1, "Y", 7)
    assert "partial capture: missing engine_terminal" in y_group.caveats
    json_groups = velocity_analysis_to_json(analysis)["groups"]
    assert json_groups[0]["measurement"]["directions"]["-1"]["target_cpms"] == -4


def test_velocity_frames_are_not_reclaimed_by_post_terminal_partial_run() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-0", "FOCI OpenFFBoard (trace)", "X", b"")
    start = _commission_event_payload(EVENT_COMMISSION_ENGINE_START, arg3=7)
    writer.write_firmware_frame(
        0, 1_000, _header(FRAME_TYPE_EVENT, start, 1, 1_000, 255, 255) + start
    )
    for sequence, direction, target, stroke_index in (
        (2, 1, 4, 0),
        (3, -1, -4, 1),
    ):
        payload = _velocity_observation_payload(
            target,
            direction,
            stroke_index=stroke_index,
            direction=direction,
        )
        writer.write_firmware_frame(
            0,
            sequence * 1_000,
            _header(
                FRAME_TYPE_VELOCITY_OBSERVATION,
                payload,
                sequence,
                sequence * 1_000,
                1,
                0,
            )
            + payload,
        )
    terminal = _commission_event_payload(EVENT_COMMISSION_ENGINE_TERMINAL, arg2=11, arg3=7)
    writer.write_firmware_frame(
        0,
        5_000,
        _header(FRAME_TYPE_EVENT, terminal, 5, 5_000, 255, 255) + terminal,
    )
    orphan_warning = _commission_event_payload(
        EVENT_COMMISSION_DIAGNOSTIC_WARNING,
        arg2=1,
        arg3=500,
    )
    writer.write_firmware_frame(
        0,
        6_000,
        _header(FRAME_TYPE_EVENT, orphan_warning, 6, 6_000, 255, 255) + orphan_warning,
    )

    analysis = analyze_capture(io.BytesIO(output.getvalue()))

    assert len(analysis.velocity_reports) == 1
    assert analysis.velocity_reports[0].engine_seq == 7


def test_velocity_group_rejects_mixed_signed_targets() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-0", "FOCI OpenFFBoard (trace)", "X", b"")
    start = _commission_event_payload(EVENT_COMMISSION_ENGINE_START, arg3=7)
    writer.write_firmware_frame(
        0, 1_000, _header(FRAME_TYPE_EVENT, start, 1, 1_000, 255, 255) + start
    )
    for sequence, target in ((2, 4), (3, 2)):
        payload = _velocity_observation_payload(target, 1)
        writer.write_firmware_frame(
            0,
            2_000 + sequence,
            _header(
                FRAME_TYPE_VELOCITY_OBSERVATION,
                payload,
                sequence,
                2_000 + sequence,
                1,
                0,
            )
            + payload,
        )

    with pytest.raises(ValueError, match="mixed targets"):
        analyze_capture(io.BytesIO(output.getvalue()))


def test_velocity_integral_audit_is_added_only_when_present() -> None:
    empty = velocity_analysis_to_json(CaptureAnalysis(devices=[]))
    audit = {
        "device_index": 0,
        "source_index": 1,
        "run_sequence": 7,
        "matches_firmware": True,
        "run_evidence": {
            "eligible_masks": [0b11111, 0b11111],
            "rest_boundary": {"positive_rung_index": 3, "slot": 5},
        },
    }
    analysis = CaptureAnalysis(devices=[], velocity_integral_audits=[audit])

    encoded = velocity_analysis_to_json(analysis)

    assert "integral_responses" not in empty
    assert encoded["integral_responses"] == [audit]
    assert "velocity_integral dev=0 source=1 run=7" in format_analysis(analysis)
    assert "rest_boundary={'positive_rung_index': 3, 'slot': 5}" in format_analysis(analysis)


def test_fixed_gain_amplitude_audit_keeps_its_namespace_in_output() -> None:
    empty = velocity_analysis_to_json(CaptureAnalysis(devices=[]))
    audit = {
        "device_index": 0,
        "source_index": 1,
        "run_sequence": 9,
        "matches_firmware": True,
        "plan": {"order": 2, "selected_p": 1_024, "selected_i": 1_024},
        "terminal": {
            "outcome": 0,
            "cause": 0,
            "outcome_name": "Complete",
            "cause_name": "None",
        },
        "cross_run_comparison": {
            "maximum_observed_displacement_q": 35,
            "maximum_displacement_kind": "observed_bound",
        },
    }
    analysis = CaptureAnalysis(devices=[], fixed_gain_amplitude_audits=[audit])

    encoded = velocity_analysis_to_json(analysis)
    rendered = format_analysis(analysis)

    assert "fixed_gain_amplitude_runs" not in empty
    assert encoded["fixed_gain_amplitude_runs"] == [audit]
    assert "namespace=fixed_gain_amplitude" in rendered
    assert "outcome=Complete cause=None" in rendered
    assert "observed_bound_q=35" in rendered


def test_breakaway_campaign_audit_is_serialized_in_both_json_views() -> None:
    audit = {
        "matches_firmware": True,
        "probe_replay": {"resolved_rung_index": 3, "breakaway_p_raw": 8},
    }
    analysis = CaptureAnalysis(devices=[], breakaway_campaign_audits=[audit])

    assert analysis_to_json(analysis)["breakaway_campaigns"] == [audit]
    assert velocity_analysis_to_json(analysis)["breakaway_campaigns"] == [audit]


def test_recovery_records_route_to_their_owning_stage(monkeypatch) -> None:
    from foci_trace import analyze

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    payloads = (
        {"record_kind": "plan", "run_sequence": 7},
        {"record_kind": "recovery_rest", "run_sequence": 7, "stage": 0},
        {"record_kind": "rung_origin_recovery", "run_sequence": 7, "stage": 0},
        {"record_kind": "proportional_terminal", "run_sequence": 7},
        {"record_kind": "integral_plan_core", "run_sequence": 8},
        {
            "record_kind": "velocity_primitive_current",
            "run_sequence": 8,
            "stage": 1,
        },
        {"record_kind": "recovery_rest", "run_sequence": 8, "stage": 1},
        {"record_kind": "rung_origin_recovery", "run_sequence": 8, "stage": 1},
        {"record_kind": "integral_rest_boundary", "run_sequence": 8},
        {"record_kind": "integral_terminal", "run_sequence": 8},
    )
    state.velocity_sweep_evidence = [
        analyze._VelocitySweepEvidenceFrame(index, 0, bytes((index,)), payload)
        for index, payload in enumerate(payloads, 1)
    ]
    routed: dict[str, list[str]] = {}

    def integral(_raw, records):
        routed["integral"] = [str(value["record_kind"]) for _bytes, value in records]
        return {"terminal": {"outcome": 3, "cause": 0}}

    monkeypatch.setattr(analyze, "audit_velocity_integral", integral)

    analyze._velocity_integral_audits({0: state})

    assert routed["integral"] == [
        "integral_plan_core",
        "velocity_primitive_current",
        "recovery_rest",
        "rung_origin_recovery",
        "integral_rest_boundary",
        "integral_terminal",
    ]


def test_amplitude_records_bypass_integral_replay(monkeypatch) -> None:
    from foci_trace import analyze

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    payloads = (
        {
            "namespace": "fixed_gain_amplitude",
            "record_kind": "fixed_gain_amplitude_plan",
            "run_sequence": 9,
            "fragment": 0,
        },
        {"record_kind": "integral_observation", "run_sequence": 9},
        {
            "namespace": "fixed_gain_amplitude",
            "record_kind": "fixed_gain_amplitude_amplitude",
            "run_sequence": 9,
        },
        {"record_kind": "recovery_rest", "run_sequence": 9, "stage": 1},
        {
            "namespace": "fixed_gain_amplitude",
            "record_kind": "fixed_gain_amplitude_terminal",
            "run_sequence": 9,
        },
    )
    state.velocity_sweep_evidence = [
        analyze._VelocitySweepEvidenceFrame(index, 0, bytes((index,)), payload)
        for index, payload in enumerate(payloads, 1)
    ]
    called = {"integral": False, "amplitude": False}

    monkeypatch.setattr(
        analyze,
        "audit_velocity_integral",
        lambda *_args: called.update(integral=True),
    )

    def amplitude(records):
        called["amplitude"] = True
        assert [value["record_kind"] for _raw, value in records] == [
            "fixed_gain_amplitude_plan",
            "integral_observation",
            "fixed_gain_amplitude_amplitude",
            "recovery_rest",
            "fixed_gain_amplitude_terminal",
        ]
        return {"plan": None, "terminal": None}

    monkeypatch.setattr(analyze, "audit_fixed_gain_amplitude", amplitude)

    analyze._velocity_integral_audits({0: state})
    analyze._fixed_gain_amplitude_audits({0: state})

    assert called == {"integral": False, "amplitude": True}


def test_event_151_is_verified_against_its_causal_raw_suffixes() -> None:
    from foci_trace import analyze
    from foci_trace.velocity_measurement import (
        VelocityObservation,
        analyze_scored_rest_selection,
    )

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    observations = [VelocityObservation(0, 1_000, 1, 0, 0) for _ in range(256)]
    state.velocity_observations = [
        analyze._VelocityFrameObservation(index, 0, observation, 0)
        for index, observation in enumerate(observations, 1)
    ]
    payload = {
        "record_kind": "rest_selection",
        "owner": 4,
        **analyze_scored_rest_selection(observations),
    }
    evidence = [analyze._VelocitySweepEvidenceFrame(300, 0, b"", payload)]
    report = {"matches_firmware": True, "mismatches": []}

    analyze._attach_rest_selection_audit(report, state, evidence, 0)

    assert report["rest_selection_verified_count"] == 1
    assert report["rest_selection_raw_unverifiable_count"] == 0
    assert report["matches_firmware"] is True


def test_rest_selection_audit_keys_the_walk_on_build_features(monkeypatch) -> None:
    """The scored-rest walk gate keys on build_features, not schema revision: a
    plan whose build_features declares the rest-chain-on-evidence rule verifies
    under the walk grammar even at a schema below the historical REST_WALK
    threshold, which is what a newer combined capture (schema 13) needs."""
    from foci_trace import analyze
    from foci_trace.velocity_measurement import (
        BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE,
        VelocityObservation,
        analyze_scored_rest_selection,
    )

    seen: list[bool] = []
    original = analyze.verify_scored_rest_selection

    def spy(observations, payload, *, rest_walk_revision=False):
        seen.append(rest_walk_revision)
        return original(observations, payload, rest_walk_revision=rest_walk_revision)

    monkeypatch.setattr(analyze, "verify_scored_rest_selection", spy)

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    observations = [VelocityObservation(0, 1_000, 1, 0, 0) for _ in range(256)]
    state.velocity_observations = [
        analyze._VelocityFrameObservation(index, 0, observation, 0)
        for index, observation in enumerate(observations, 1)
    ]
    payload = {
        "record_kind": "rest_selection",
        "owner": 4,
        **analyze_scored_rest_selection(observations),
    }
    evidence = [analyze._VelocitySweepEvidenceFrame(300, 0, b"", payload)]

    report = {
        "matches_firmware": True,
        "mismatches": [],
        "plan": {"build_features": BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE, "schema_revision": 13},
    }
    analyze._attach_rest_selection_audit(report, state, evidence, 0, integral_rule_contract=True)

    assert seen == [True]
    assert report["rest_selection_verified_count"] == 1


@pytest.mark.parametrize("recovery_owner", (1, 2, 5, 6, 8))
def test_recovery_event_151_cannot_consume_the_preceding_observations_window(
    recovery_owner: int,
) -> None:
    """A recovery selection must never be raw-verified against the window
    that precedes it, because that window is not its own.

    `step_rung_origin_recovery` (`recovery/controller.rs:351`) pushes every
    scored recovery sample into the observation ring it later analyzes and
    never calls `record_velocity_trace_observation`, so a capture contains no
    raw samples for any recovery selection. The 256 traced samples before one
    belong to the ordinary stroke that ran before it.

    This reproduces exactly that arrangement: one ordinary observation's
    moving window is the only thing traced, and a recovery selection follows
    it reporting what the untraced recovery window actually scored -- a
    genuine rest. The audit must verify the first and decline the second.

    It cannot pass vacuously. The archived capture's twelve false passes
    happened because both windows were all-zero, so the first assertion below
    proves these two are not: verifying the recovery record against the
    traced window raises, on a field whose value differs. Reintroduce the
    pairing and that raise becomes a reported mismatch, failing every
    remaining assertion.
    """
    from foci_trace import analyze
    from foci_trace.velocity_measurement import (
        VelocityObservation,
        analyze_scored_rest_selection,
        verify_scored_rest_selection,
    )

    traced_window = [
        VelocityObservation(40 + index % 3, 1_000, 1, 0, 40_000) for index in range(256)
    ]
    untraced_recovery_window = [VelocityObservation(0, 1_000, 1, 0, 0) for _ in range(256)]

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    state.velocity_observations = [
        analyze._VelocityFrameObservation(index, 0, observation, 0)
        for index, observation in enumerate(traced_window, 1)
    ]

    ordinary_payload = {
        "record_kind": "rest_selection",
        "owner": 4,
        **analyze_scored_rest_selection(traced_window),
    }
    recovery_payload = {
        "record_kind": "rest_selection",
        "owner": recovery_owner,
        **analyze_scored_rest_selection(untraced_recovery_window),
    }

    with pytest.raises(ValueError):
        verify_scored_rest_selection(traced_window, recovery_payload)

    evidence = [
        analyze._VelocitySweepEvidenceFrame(300, 0, b"", ordinary_payload),
        analyze._VelocitySweepEvidenceFrame(301, 0, b"", recovery_payload),
    ]
    report: dict[str, object] = {"matches_firmware": True, "mismatches": []}

    analyze._attach_rest_selection_audit(report, state, evidence, 0)

    assert report["mismatches"] == []
    assert report["matches_firmware"] is True
    assert report["rest_selection_verified_count"] == 1
    assert report["rest_selection_raw_unverifiable_count"] == 1


@pytest.mark.parametrize("interleave_phase_boundary", [False, True])
def test_analyze_capture_reconstructs_a_breakaway_campaign_from_ftrc_frames(
    interleave_phase_boundary: bool,
) -> None:
    """End-to-end: raw FTRC frames through `analyze_capture` reconstruct a
    breakaway campaign, including the marker-adjacency decode that resolves
    the tag-1/`integral_plan_marker` collision (see
    `protocol.decode_breakaway_record`'s docstring).
    """
    from foci_trace.velocity_measurement import (
        breakaway_build_search_grid,
        breakaway_probe_plan_digest,
    )

    p_start_raw, p_top_raw = 2, 8
    search_grid = breakaway_build_search_grid(p_start_raw, p_top_raw)
    motion_threshold_counts = 63
    max_capture_interval_us = 2_000
    max_observations = 128
    probe_digest = breakaway_probe_plan_digest(
        search_grid,
        motion_threshold_counts,
        max_capture_interval_us,
        max_observations,
        **_BREAKAWAY_PROBE_CONTEXT,
    )

    def bw_event_payload(event_id: int, sequence: int) -> bytes:
        return struct.pack(
            "<BBHBxxxiiii", 1, SUBSYSTEM_COMMISSIONING, event_id, 4, 0, 1, sequence, 0
        )

    probe_planned = _breakaway_probe_plan_payload(
        run_sequence=7,
        evidence_sequence=1,
        plan_digest=probe_digest,
        max_observations=max_observations,
        search_count=len(search_grid),
        p_start_raw=p_start_raw,
        p_top_raw=p_top_raw,
    )
    probe_terminal = bytes([2]) + struct.pack("<IHQBBB", 7, 2, probe_digest, 6, 0, 17)
    campaign_terminal = bytes([12]) + struct.pack("<IHBBBQ", 7, 3, 0, 6, 0, 0)

    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
    writer.write_firmware_frame(
        0,
        1_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_planned, 1, 1_000, 1, 0) + probe_planned,
    )
    if interleave_phase_boundary:
        phase_boundary = struct.pack(
            "<BBHBxxxiiii", 1, SUBSYSTEM_STEP_TIMER, EVENT_PHASE_BOUNDARY, 3, 0, 1, 2, 0
        )
        writer.write_firmware_frame(
            0,
            1_050,
            _header(FRAME_TYPE_EVENT, phase_boundary, 2, 1_050, 255, 255) + phase_boundary,
        )
    marker_plan = bw_event_payload(152, 1)
    writer.write_firmware_frame(
        0, 1_100, _header(FRAME_TYPE_EVENT, marker_plan, 2, 1_100, 255, 255) + marker_plan
    )
    writer.write_firmware_frame(
        0,
        2_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_terminal, 3, 2_000, 1, 0)
        + probe_terminal,
    )
    marker_probe_terminal = bw_event_payload(154, 2)
    writer.write_firmware_frame(
        0,
        2_100,
        _header(FRAME_TYPE_EVENT, marker_probe_terminal, 4, 2_100, 255, 255)
        + marker_probe_terminal,
    )
    writer.write_firmware_frame(
        0,
        3_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, campaign_terminal, 5, 3_000, 1, 0)
        + campaign_terminal,
    )
    marker_campaign_terminal = bw_event_payload(159, 3)
    writer.write_firmware_frame(
        0,
        3_100,
        _header(FRAME_TYPE_EVENT, marker_campaign_terminal, 6, 3_100, 255, 255)
        + marker_campaign_terminal,
    )

    analysis = analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)

    assert len(analysis.breakaway_campaign_audits) == 1
    audit = analysis.breakaway_campaign_audits[0]
    assert audit["matches_firmware"] is True
    assert audit["phase_sequence"] == ["probe_plan", "probe_terminal", "campaign_terminal"]
    assert audit["probe_terminal"]["fault_checkpoint"] == 17
    assert audit["probe_terminal"]["fault_checkpoint_name"] == "probe_moving_encoder_sample"
    assert audit["confirmation_accepted"] is False
    assert analysis.devices[0].decode_errors == 0
    assert velocity_analysis_to_json(analysis).get("sweeps", []) == []


def test_analyze_capture_rejects_in_phase_motion_before_the_breakaway_probe_plan() -> None:
    """Motion inside the campaign's own commissioning phase, before its probe
    plan, is refused end to end -- not just at the `audit_breakaway_campaign`
    unit level.
    """
    from foci_trace.velocity_measurement import breakaway_probe_plan_digest

    search_grid = [2, 3, 4, 6, 8]
    motion_threshold_counts = 63
    max_capture_interval_us = 2_000
    max_observations = 128
    probe_digest = breakaway_probe_plan_digest(
        search_grid,
        motion_threshold_counts,
        max_capture_interval_us,
        max_observations,
        **_BREAKAWAY_PROBE_CONTEXT,
    )

    def bw_event_payload(event_id: int, sequence: int) -> bytes:
        return struct.pack(
            "<BBHBxxxiiii", 1, SUBSYSTEM_COMMISSIONING, event_id, 4, 0, 1, sequence, 0
        )

    probe_planned = _breakaway_probe_plan_payload(
        run_sequence=7,
        evidence_sequence=1,
        plan_digest=probe_digest,
        max_observations=max_observations,
        search_count=len(search_grid),
        p_start_raw=2,
        p_top_raw=8,
    )
    probe_terminal = bytes([2]) + struct.pack("<IHQBBB", 7, 2, probe_digest, 1, 0, 0)
    campaign_terminal = bytes([12]) + struct.pack("<IHBBBQ", 7, 3, 0, 1, 0, 0)

    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    engine_start = _commission_event_payload(
        EVENT_COMMISSION_ENGINE_START, source_index=0, engine_kind=4, arg2=0, arg3=1
    )
    writer.write_firmware_frame(
        0, 100, _header(FRAME_TYPE_EVENT, engine_start, 1, 100, 255, 255) + engine_start
    )
    phase_start = _commission_event_payload(
        EVENT_COMMISSION_PHASE_START, source_index=0, engine_kind=4, arg2=10, arg3=1
    )
    writer.write_firmware_frame(
        0, 200, _header(FRAME_TYPE_EVENT, phase_start, 2, 200, 255, 255) + phase_start
    )

    feedback = _velocity_observation_payload(11, 6, elapsed_us=1_000, stroke_index=0, direction=1)
    writer.write_firmware_frame(
        0, 300, _header(FRAME_TYPE_VELOCITY_OBSERVATION, feedback, 3, 300, 1, 0) + feedback
    )

    writer.write_firmware_frame(
        0,
        1_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_planned, 4, 1_000, 1, 0) + probe_planned,
    )
    marker_plan = bw_event_payload(152, 1)
    writer.write_firmware_frame(
        0, 1_100, _header(FRAME_TYPE_EVENT, marker_plan, 5, 1_100, 255, 255) + marker_plan
    )
    writer.write_firmware_frame(
        0,
        2_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_terminal, 6, 2_000, 1, 0)
        + probe_terminal,
    )
    marker_probe_terminal = bw_event_payload(154, 2)
    writer.write_firmware_frame(
        0,
        2_100,
        _header(FRAME_TYPE_EVENT, marker_probe_terminal, 7, 2_100, 255, 255)
        + marker_probe_terminal,
    )
    writer.write_firmware_frame(
        0,
        3_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, campaign_terminal, 8, 3_000, 1, 0)
        + campaign_terminal,
    )
    marker_campaign_terminal = bw_event_payload(159, 3)
    writer.write_firmware_frame(
        0,
        3_100,
        _header(FRAME_TYPE_EVENT, marker_campaign_terminal, 9, 3_100, 255, 255)
        + marker_campaign_terminal,
    )

    with pytest.raises(ValueError, match="motion"):
        analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)


def test_analyze_capture_does_not_flag_prior_phase_motion_as_premature() -> None:
    """Motion in an earlier commissioning phase must not be misread as
    premature motion for a later breakaway campaign in the same capture --
    the false-positive risk the phase-scoped bound (see
    `_breakaway_campaign_phase_start_ticks`) exists specifically to avoid.
    """
    from foci_trace.velocity_measurement import breakaway_probe_plan_digest

    search_grid = [2, 3, 4, 6, 8]
    motion_threshold_counts = 63
    max_capture_interval_us = 2_000
    max_observations = 128
    probe_digest = breakaway_probe_plan_digest(
        search_grid,
        motion_threshold_counts,
        max_capture_interval_us,
        max_observations,
        **_BREAKAWAY_PROBE_CONTEXT,
    )

    def bw_event_payload(event_id: int, sequence: int) -> bytes:
        return struct.pack(
            "<BBHBxxxiiii", 1, SUBSYSTEM_COMMISSIONING, event_id, 4, 0, 1, sequence, 0
        )

    probe_planned = _breakaway_probe_plan_payload(
        run_sequence=7,
        evidence_sequence=1,
        plan_digest=probe_digest,
        max_observations=max_observations,
        search_count=len(search_grid),
        p_start_raw=2,
        p_top_raw=8,
    )
    probe_terminal = bytes([2]) + struct.pack("<IHQBBB", 7, 2, probe_digest, 1, 0, 0)
    campaign_terminal = bytes([12]) + struct.pack("<IHBBBQ", 7, 3, 0, 1, 0, 0)

    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")

    engine_start = _commission_event_payload(
        EVENT_COMMISSION_ENGINE_START, source_index=0, engine_kind=4, arg2=0, arg3=1
    )
    writer.write_firmware_frame(
        0, 10, _header(FRAME_TYPE_EVENT, engine_start, 1, 10, 255, 255) + engine_start
    )
    earlier_phase_start = _commission_event_payload(
        EVENT_COMMISSION_PHASE_START, source_index=0, engine_kind=4, arg2=9, arg3=1
    )
    writer.write_firmware_frame(
        0, 20, _header(FRAME_TYPE_EVENT, earlier_phase_start, 2, 20, 255, 255) + earlier_phase_start
    )

    feedback = _velocity_observation_payload(11, 6, elapsed_us=1_000, stroke_index=0, direction=1)
    writer.write_firmware_frame(
        0, 50, _header(FRAME_TYPE_VELOCITY_OBSERVATION, feedback, 3, 50, 1, 0) + feedback
    )

    earlier_phase_end = _commission_event_payload(
        EVENT_COMMISSION_PHASE_END, source_index=0, engine_kind=4, arg2=9, arg3=0
    )
    writer.write_firmware_frame(
        0, 90, _header(FRAME_TYPE_EVENT, earlier_phase_end, 4, 90, 255, 255) + earlier_phase_end
    )
    breakaway_phase_start = _commission_event_payload(
        EVENT_COMMISSION_PHASE_START, source_index=0, engine_kind=4, arg2=10, arg3=2
    )
    writer.write_firmware_frame(
        0,
        100,
        _header(FRAME_TYPE_EVENT, breakaway_phase_start, 5, 100, 255, 255) + breakaway_phase_start,
    )

    writer.write_firmware_frame(
        0,
        1_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_planned, 6, 1_000, 1, 0) + probe_planned,
    )
    marker_plan = bw_event_payload(152, 1)
    writer.write_firmware_frame(
        0, 1_100, _header(FRAME_TYPE_EVENT, marker_plan, 7, 1_100, 255, 255) + marker_plan
    )
    writer.write_firmware_frame(
        0,
        2_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, probe_terminal, 8, 2_000, 1, 0)
        + probe_terminal,
    )
    marker_probe_terminal = bw_event_payload(154, 2)
    writer.write_firmware_frame(
        0,
        2_100,
        _header(FRAME_TYPE_EVENT, marker_probe_terminal, 9, 2_100, 255, 255)
        + marker_probe_terminal,
    )
    writer.write_firmware_frame(
        0,
        3_000,
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, campaign_terminal, 10, 3_000, 1, 0)
        + campaign_terminal,
    )
    marker_campaign_terminal = bw_event_payload(159, 3)
    writer.write_firmware_frame(
        0,
        3_100,
        _header(FRAME_TYPE_EVENT, marker_campaign_terminal, 11, 3_100, 255, 255)
        + marker_campaign_terminal,
    )

    analysis = analyze_capture(io.BytesIO(output.getvalue()), steady_after_s=0)

    assert len(analysis.breakaway_campaign_audits) == 1
    assert analysis.breakaway_campaign_audits[0]["matches_firmware"] is True
    assert analysis.devices[0].decode_errors == 0
    assert velocity_analysis_to_json(analysis).get("sweeps", []) == []


def _status_payload_v22(pressure: int, max_cycles: int) -> bytes:
    return _status_payload(0, 0) + struct.pack("<II", pressure, max_cycles)


def test_analyze_capture_reports_tracking_lag_and_budget() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard", "ABC", b"")
    meta = _metadata_payload("ABC")
    writer.write_firmware_frame(
        0, 110, _header(FRAME_TYPE_METADATA, meta, 1, 0, 0, 255, version=22) + meta
    )
    before = _status_payload_v22(0, 90_000)
    writer.write_firmware_frame(
        0, 120, _header(FRAME_TYPE_STATUS, before, 2, 0, 0, 255, version=22) + before
    )
    tpm = 84_000
    seq = 3
    for i in range(400):
        payload = _critical_payload(-144, 0, 0, 0)
        payload = payload[:16] + struct.pack("<i", 75) + payload[20:]
        writer.write_firmware_frame(
            0,
            130 + i,
            _header(FRAME_TYPE_CRITICAL, payload, seq, (100 + i) * tpm, 1, 0, version=22) + payload,
        )
        seq += 1
    after = _status_payload_v22(0, 95_000)
    writer.write_firmware_frame(
        0, 900, _header(FRAME_TYPE_STATUS, after, seq, 800 * tpm, 0, 255, version=22) + after
    )
    output.seek(0)

    analysis = analyze_capture(output, mm_per_rev=40.0)
    report = analysis.tracking_lag
    assert report is not None and report.refused is None
    (seg,) = report.segments
    assert seg.lag_ms == pytest.approx(-1.7578, abs=0.01)
    assert report.budgets[0].bracketed and report.budgets[0].max_spi_sample_cycles == 95_000
    text = format_summary(analysis.devices, tracking_lag=report, mm_per_rev=40.0)
    assert "lag=-1.76ms" in text
    assert "budget dev0 bracketed=True" in text
    assert "max_cycles=95000" in text
    assert "pressure_delta=0" in text
    assert "within_budget=True" in text

    full = format_analysis(analysis, mm_per_rev=40.0)
    assert "lag=-1.76ms" in full and "budget dev0 bracketed=True" in full

    assert analysis_to_json(analysis)["tracking_lag"]["segments"][0]["lag_ms"] == pytest.approx(
        -1.7578, abs=0.01
    )


def _pb(tick: int, channel: int, old_class: int, new_class: int):
    return _PhaseBoundaryObservation(
        timestamp_ticks=tick,
        channel=channel,
        old_class=old_class,
        new_class=new_class,
        boot_id=0,
    )


def _so(tick: int, count: int, channel: int = 1, source_kind: int = SOURCE_KIND_BOARD):
    return _StatusObservation(
        timestamp_ticks=tick,
        source_key=(source_kind, channel),
        dropped_event_log_count=count,
        boot_id=0,
    )


def test_marker_based_bounds_none_when_no_markers_in_window() -> None:
    assert _marker_based_phase_bounds([], [_so(0, 0), _so(1000, 0)], [], 0, 100, 900) is None


def test_marker_based_bounds_accel_cruise_decel() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) == (300, 700)


def test_marker_based_bounds_no_decel_in_window_extends_cruise_to_window_end() -> None:
    observations = [_pb(300, 0, 0, 1)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) == (300, 900)


def test_marker_based_bounds_uses_zero_width_cruise_when_cruise_never_entered() -> None:
    observations = [_pb(500, 0, 0, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) == (500, 500)


def test_marker_based_bounds_falls_back_on_reverse_triangular_transition() -> None:
    observations = [_pb(500, 0, 2, 0)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_falls_back_on_multi_transition_no_cruise_chain() -> None:
    observations = [_pb(400, 0, 0, 2), _pb(600, 0, 2, 0)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_falls_back_on_accel_cruise_accel() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 0)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_falls_back_on_decel_cruise_accel() -> None:
    observations = [_pb(300, 0, 2, 1), _pb(700, 0, 1, 0)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_window_starting_in_cruise() -> None:
    observations = [_pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) == (100, 700)


def test_marker_based_bounds_none_on_drop_gap_flag_on_a_non_marker_event() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [500], 0, 100, 900) is None


def test_marker_based_bounds_ignores_drop_gap_outside_the_window() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [50], 0, 100, 900) == (300, 700)


def test_marker_based_bounds_none_on_broken_chain() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 0, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_on_drop_counter_change() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 1)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_when_final_marker_dropped_before_status_confirms() -> None:
    observations = [_pb(300, 0, 0, 1)]
    status = [_so(0, 0), _so(1000, 1)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_when_capture_ends_before_a_confirming_status_frame() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_when_no_status_precedes_the_window() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_isolates_other_channels() -> None:
    observations = [_pb(300, 1, 0, 1), _pb(700, 1, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_on_invalid_new_class() -> None:
    observations = [_pb(300, 0, 0, 4)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_on_invalid_old_class() -> None:
    observations = [_pb(300, 0, 5, 1)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_on_single_no_op_marker() -> None:
    observations = [_pb(300, 0, 0, 0)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_none_on_no_op_marker_within_a_chain() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(500, 0, 1, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_ignores_zero_valued_foci_source_confirming_frame() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [
        _so(0, 0),
        _so(1000, 0, source_kind=_SOURCE_KIND_FOCI),
        _so(1000, 1),
    ]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_marker_based_bounds_skips_foci_source_baseline_for_earlier_board_baseline() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [
        _so(0, 0),
        _so(50, 0, source_kind=_SOURCE_KIND_FOCI),
        _so(1000, 0),
    ]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) == (300, 700)


def test_marker_based_bounds_none_when_only_foci_source_baseline_available() -> None:
    observations = [_pb(300, 0, 0, 1), _pb(700, 0, 1, 2)]
    status = [_so(0, 0, source_kind=_SOURCE_KIND_FOCI), _so(1000, 0)]
    assert _marker_based_phase_bounds(observations, status, [], 0, 100, 900) is None


def test_reverse_slot_order_amplitude_runs_are_compared(monkeypatch) -> None:
    from foci_trace import analyze

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    state.velocity_sweep_evidence = [
        analyze._VelocitySweepEvidenceFrame(
            tick,
            0,
            b"",
            {
                "namespace": "fixed_gain_amplitude",
                "record_kind": "fixed_gain_amplitude_plan",
                "run_sequence": run_sequence,
                "fragment": 0,
            },
        )
        for tick, run_sequence in ((1, 10), (2, 11))
    ]
    plans = {
        10: {"order": 0x11, "amplitude_order": 1, "slot_order": 1},
        11: {"order": 0x12, "amplitude_order": 2, "slot_order": 1},
    }

    def amplitude(records):
        run_sequence = int(records[0][1]["run_sequence"])
        return {"plan": dict(plans[run_sequence]), "terminal": {"outcome": 0}}

    monkeypatch.setattr(analyze, "audit_fixed_gain_amplitude", amplitude)
    monkeypatch.setattr(analyze, "compare_amplitude_runs", lambda *_args: {"compared": True})

    audits = analyze._fixed_gain_amplitude_audits({0: state})

    assert audits[1]["cross_run_comparison"] == {"compared": True}


def test_amplitude_run_that_changes_source_is_reported_not_raised() -> None:
    from foci_trace import analyze

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    state.velocity_sweep_evidence = [
        analyze._VelocitySweepEvidenceFrame(
            tick,
            source_index,
            b"",
            {
                "namespace": "fixed_gain_amplitude",
                "record_kind": "fixed_gain_amplitude_plan",
                "run_sequence": 12,
                "fragment": 0,
            },
        )
        for tick, source_index in ((1, 0), (2, 1))
    ]

    (audit,) = analyze._fixed_gain_amplitude_audits({0: state})

    assert audit["run_sequence"] == 12
    assert audit["matches_firmware"] is False
    assert audit["mismatches"] == ["fixed-gain-amplitude evidence changed source index"]


def test_integral_run_that_changes_source_is_reported_not_raised() -> None:
    from foci_trace import analyze

    state = analyze._DeviceState(report=analyze.DeviceReport(device_index=0))
    state.velocity_sweep_evidence = [
        analyze._VelocitySweepEvidenceFrame(
            tick,
            source_index,
            b"",
            {"record_kind": record_kind, "run_sequence": 14},
        )
        for tick, source_index, record_kind in (
            (1, 0, "integral_plan_core"),
            (2, 1, "integral_terminal"),
        )
    ]

    (audit,) = analyze._velocity_integral_audits({0: state})

    assert audit["run_sequence"] == 14
    assert audit["matches_firmware"] is False
    assert audit["mismatches"] == ["velocity-integral evidence changed source index"]


def test_analysis_reports_a_truncated_capture_tail() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI OpenFFBoard (trace)", "ABC", b"")
    raw = output.getvalue() + b"\x46\x54\x52\x52\x01"

    analysis = analyze_capture(io.BytesIO(raw))

    assert analysis.truncated_tail_bytes == 5
    assert analysis_to_json(analysis)["truncated_tail_bytes"] == 5
    assert velocity_analysis_to_json(analysis)["truncated_tail_bytes"] == 5
    assert "5 bytes" in format_analysis(analysis)


def test_motion_segments_never_span_a_reboot() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-50", "FOCI Ouroboros (trace)", "ABC", b"")
    clock = 84_000_000
    seq = 0
    for boot_id, steps in ((1111, (0, 4000, 8000, 12000)), (2222, (0, 3000, 6000, 9000))):
        for index, step in enumerate(steps):
            seq += 1
            timestamp = clock * (index * 50) // 1000
            payload = _critical_motion_payload(step, step, torque_actual=0, uq_limited=0)
            writer.write_firmware_frame(
                0,
                seq,
                _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, boot_id=boot_id)
                + payload,
            )
        if boot_id == 2222:
            for index in range(4):
                seq += 1
                timestamp = clock * (index * 50) // 1000
                detail = _control_detail_payload(100, 1000, 1000)
                writer.write_firmware_frame(
                    0,
                    seq,
                    _header(
                        FRAME_TYPE_CONTROL_DETAIL, detail, seq, timestamp, 1, 0, boot_id=boot_id
                    )
                    + detail,
                )

    reports = summarize_capture(
        io.BytesIO(output.getvalue()), steady_after_s=0, motion_gap_s=0.1, motion_min_steps=1000
    )

    segments = reports[0].motion_segments
    assert [(s.step_delta, s.critical_samples) for s in segments] == [(12000, 4), (9000, 4)]
    assert [s.control_detail_samples for s in segments] == [0, 4]


def test_contour_analysis_handles_a_capture_spanning_a_reboot() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", "PLUS", b"")
    writer.write_device_table(1, 100, "1-2", "FOCI OpenFFBoard (trace)", "MINUS", b"")
    clock = 84_000_000
    seq = 0
    host_ns = 10_000
    for boot_id in (1111, 2222):
        for index in range(4):
            seq += 1
            timestamp = clock * (index * 50) // 1000
            payload = _critical_motion_payload(index * 3000, index * 3000, 100, 1000)
            for device_index in (0, 1):
                host_ns += 1_000
                writer.write_firmware_frame(
                    device_index,
                    host_ns,
                    _header(FRAME_TYPE_CRITICAL, payload, seq, timestamp, 1, 0, boot_id=boot_id)
                    + payload,
                )

    analysis = analyze_capture(
        io.BytesIO(output.getvalue()),
        steady_after_s=0,
        motion_gap_s=0.1,
        motion_min_steps=1000,
        corexy_plus="PLUS",
        corexy_minus="MINUS",
    )

    assert [len(device.motion_segments) for device in analysis.devices] == [2, 2]
