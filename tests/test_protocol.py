import struct

import pytest

from foci_trace.protocol import (
    COMMISSION_STATUS_NAMES,
    EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT,
    EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION,
    EVENT_COMMISSION_MECHANICAL_ID_RESPONSE,
    EVENT_COMMISSION_MECHANICAL_ID_RESULT,
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    FRAME_TYPE_VELOCITY_OBSERVATION,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    PHASE_NAMES,
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID,
    VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID,
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


def test_decode_v20_critical_velocity_clamp_flag_engaged() -> None:
    payload = bytearray(88)
    payload[83] = 0x01

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=20) + payload)

    assert frame.payload["cached_velocity_clamped"] is True


def test_decode_v20_critical_velocity_clamp_flag_not_engaged() -> None:
    payload = bytearray(88)

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=20) + payload)

    assert frame.payload["cached_velocity_clamped"] is False


def test_decode_v6_critical_ignores_the_clamp_flag_byte_before_version_20() -> None:
    payload = bytearray(88)
    payload[83] = 0x01

    frame = decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=6) + payload)

    assert frame.payload["cached_velocity_clamped"] is False


def test_decode_control_detail_velocity_targets_as_signed_values() -> None:
    payload = struct.pack(
        "<IIiiIII",
        0xFFFF_FFFF,
        0x8000_0000,
        -1234,
        -5678,
        0xFFFF_FFFE,
        0x8000_0001,
        0xFFFF_FFFD,
    )

    frame = decode_frame(_header(FRAME_TYPE_CONTROL_DETAIL, payload) + payload)

    assert frame.payload == {
        "interim_pidout_target_torque": 0xFFFF_FFFF,
        "interim_pidout_target_flux": 0x8000_0000,
        "interim_pidin_target_velocity": -1234,
        "interim_pidout_target_velocity": -5678,
        "pidout_uq_ud_limits": 0xFFFF_FFFE,
        "pid_torque_flux_limits": 0x8000_0001,
        "pid_velocity_limit": 0xFFFF_FFFD,
    }


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


def test_decodes_signed_velocity_observation_frame() -> None:
    payload = struct.pack(
        "<iiIHbBHi",
        -4,
        -7,
        1_037,
        513,
        -1,
        VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID,
        900,
        -8,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_OBSERVATION, payload) + payload)

    assert frame.payload == {
        "target_counts_per_ms": -4,
        "delta_counts": -7,
        "elapsed_us": 1_037,
        "stroke_index": 513,
        "direction": -1,
        "flags": VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID,
        "peak_torque_target_abs": 900,
        "pid_velocity_actual": -8,
        "actual_velocity_pptm": None,
    }


def test_decodes_version_nine_pptm_velocity_diagnostic() -> None:
    payload = struct.pack(
        "<iiIHbBHi",
        4,
        7,
        1_037,
        12,
        1,
        VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID,
        321,
        176,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_OBSERVATION, payload) + payload)

    assert PROTOCOL_VERSION == 20
    assert 13 in SUPPORTED_PROTOCOL_VERSIONS
    assert 14 in SUPPORTED_PROTOCOL_VERSIONS
    assert 20 in SUPPORTED_PROTOCOL_VERSIONS
    assert frame.payload["pid_velocity_actual"] is None
    assert frame.payload["actual_velocity_pptm"] == 176


def test_decode_frame_rejects_a_version_beyond_the_supported_range() -> None:
    # A version past the supported range is rejected; historical versions stay
    # decodable so committed captures still replay.
    payload = bytes(88)
    with pytest.raises(DecodeError, match="unsupported protocol version 21"):
        decode_frame(_header(FRAME_TYPE_CRITICAL, payload, version=21) + payload)


def test_rejects_velocity_observation_with_both_diagnostic_kinds() -> None:
    flags = (
        VELOCITY_OBSERVATION_FLAG_PID_VELOCITY_ACTUAL_VALID
        | VELOCITY_OBSERVATION_FLAG_PPTM_ACTUAL_VALID
    )
    payload = struct.pack("<iiIHbBHi", 4, 7, 1_037, 12, 1, flags, 321, 176)

    with pytest.raises(DecodeError, match="mutually exclusive"):
        decode_frame(_header(FRAME_TYPE_VELOCITY_OBSERVATION, payload) + payload)


def test_decodes_version_six_velocity_observation_without_peak() -> None:
    payload = struct.pack("<iiIHbB", -4, -7, 1_037, 513, -1, 1)

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_OBSERVATION, payload, version=6) + payload)

    assert frame.payload["peak_torque_target_abs"] is None
    assert frame.payload["pid_velocity_actual"] is None
    assert frame.payload["actual_velocity_pptm"] is None


def test_decodes_canonical_velocity_sweep_plan() -> None:
    values = (
        7,
        0,
        5_000,
        1,
        4_800,
        7_500,
        1,
        0,
        288,
        8,
        128,
        5,
        2,
        256_000,
        200_000,
        9_120,
        10_000,
        1_000,
        250,
        1_250,
        1_250,
        1_000,
        900,
    )
    payload = bytes([1]) + struct.pack("<IHI BIIHBiHHBHIIIIIIIIHH".replace(" ", ""), *values)

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "plan"
    assert frame.payload["run_sequence"] == 7
    assert frame.payload["planned_velocity_mrev_s"] == 4_800
    assert frame.payload["target_velocity_rpm"] == 288
    assert frame.payload["rung_count"] == 5
    assert frame.payload["usable_torque_limit"] == 900


def test_decodes_stage_b_recovery_plan_geometry() -> None:
    payload = bytearray([1])
    payload.extend(struct.pack("<IHII", 7, 0, 1, 2))
    payload.extend(
        struct.pack(
            "<IBIIHBiHHBHIIII",
            5_000,
            1,
            4_800,
            7_500,
            1,
            0,
            288,
            8,
            128,
            5,
            4,
            256_000,
            500_000,
            298_632,
            298_632,
        )
    )
    payload.extend(struct.pack("<IIIB", 1_000, 1_816_958, 3_000_000, 23))
    payload.extend(struct.pack("<IIIIHH", 2_500, 250, 28_100, 28_100, 1_000, 900))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 91
    assert frame.payload["record_kind"] == "plan"
    assert frame.payload["plan_digest"] == 0x0000_0002_0000_0001
    assert frame.payload["recovery_origin_band_counts"] == 1_000
    assert frame.payload["recovery_nominal_slot_us"] == 1_816_958
    assert frame.payload["recovery_maximum_slot_us"] == 3_000_000
    assert frame.payload["recovery_slot_count"] == 23


def test_decodes_fragmented_velocity_integral_response_plan() -> None:
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 1, 0, 0x99AA_BBCC, 0xDDEE_FF00, 1_448))
    core.extend(
        struct.pack(
            "<IIIiiii",
            0x0038_0000,
            0x003C_0000,
            0x0038_0000,
            17_000_000,
            18_000_000,
            -19_000_000,
            -18_000_000,
        )
    )
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 5, 28, 7, 56, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)

    schedule = bytearray([0])
    schedule.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    schedule.extend(struct.pack("<BBHHHHH", 1, 5, 5, 10, 20, 40, 80))
    schedule.extend(
        struct.pack(
            "<IIIIIIIBIIIIH",
            512_000,
            500_000,
            236_000,
            69_888,
            69_888,
            4_394,
            10_000,
            1,
            10_000,
            100,
            20_000,
            20_000,
            60_000,
        )
    )

    core_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)
    schedule_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, schedule) + schedule)

    assert len(core) == 88
    assert len(schedule) == 74
    assert core_frame.payload["record_kind"] == "integral_plan_core"
    assert core_frame.payload["run_sequence"] == 7
    assert core_frame.payload["plan_digest"] == 0x5566_7788_1122_3344
    assert core_frame.payload["stage_b_plan_digest"] == 0xDDEE_FF00_99AA_BBCC
    assert core_frame.payload["directional_memberships"] == [0x0038_0000, 0x003C_0000]
    assert core_frame.payload["directional_pooled_intervals"] == [
        [17_000_000, 18_000_000],
        [-19_000_000, -18_000_000],
    ]
    assert core_frame.payload["limit_evidence_capability"] == 1
    assert schedule_frame.payload["record_kind"] == "integral_plan_schedule"
    assert schedule_frame.payload["plan_digest"] == 0x5566_7788_1122_3344
    assert schedule_frame.payload["positive_i"] == [5, 10, 20, 40, 80]
    assert schedule_frame.payload["maximum_workflow_ms"] == 69_888
    assert schedule_frame.payload["requested_velocity_source"] == 1


def test_decodes_schema_four_velocity_integral_authority() -> None:
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 4, 0, 0x99AA_BBCC, 0xDDEE_FF00, 1_024))
    core.extend(struct.pack("<III", 0x001C_0000, 0x0018_0000, 0x0018_0000))
    core.extend(struct.pack("<BBB", 3, 1, 1))
    core.extend(struct.pack("<iiii", 17_000_000, 18_000_000, -19_000_000, -18_000_000))
    core.extend(struct.pack("<IiIIBHBHHH", 4_394, 264, 25_000, 4_000, 7, 36, 9, 72, 2_816, 2_534))
    core.append(1)

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)

    assert len(core) == 91
    assert frame.payload["record_kind"] == "integral_plan_core"
    assert frame.payload["directional_validity"] == [3, 1]
    assert frame.payload["reduced_margin"] is True
    assert frame.payload["directional_pooled_intervals"][1] == [-19_000_000, -18_000_000]
    assert frame.payload["limit_evidence_capability"] == 1


def test_decodes_schema_four_minimum_positive_schedule() -> None:
    schedule = bytearray([0])
    schedule.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    schedule.extend(struct.pack("<BBHHHHHHH", 1, 7, 1, 2, 4, 8, 16, 32, 64))
    schedule.extend(
        struct.pack(
            "<IIIIIIIBIIIIH",
            512_000,
            500_000,
            236_000,
            106_209,
            116_856,
            4_394,
            10_000,
            1,
            2_500,
            250,
            28_100,
            28_100,
            3_000,
        )
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, schedule) + schedule)

    assert frame.payload["record_kind"] == "integral_plan_schedule"
    assert frame.payload["positive_i"] == [1, 2, 4, 8, 16, 32, 64]
    assert frame.payload["nominal_workflow_ms"] == 106_209
    assert frame.payload["maximum_workflow_ms"] == 116_856


def test_decodes_schema_nine_slot_transition_allowance() -> None:
    schedule = bytearray([0])
    schedule.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    schedule.extend(struct.pack("<BB", 1, 13))
    schedule.extend(struct.pack("<13H", 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 1_310))
    schedule.extend(
        struct.pack(
            "<IIIIIIIIBIIIIH",
            512_000,
            500_000,
            236_000,
            20_000,
            180_151,
            197_896,
            4_394,
            10_000,
            1,
            2_500,
            250,
            28_100,
            28_100,
            3_000,
        )
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, schedule) + schedule)

    assert frame.payload["record_kind"] == "integral_plan_schedule"
    assert frame.payload["slot_transition_allowance_us"] == 20_000
    assert frame.payload["nominal_workflow_ms"] == 180_151
    assert frame.payload["maximum_workflow_ms"] == 197_896


def test_decodes_acceptance_matrix_plan_fragments() -> None:
    core = bytearray([0])
    core.extend(struct.pack("<IHQB", 7, 1, 0x0123_4567_89AB_CDEF, 0))
    core.extend(
        struct.pack(
            "<HBIHQBHHHII5iBHH",
            1,
            0,
            0x05E2_1F75,
            10,
            0xAABB_CCDD_EEFF_0011,
            1,
            1_024,
            1_024,
            3,
            25_000,
            4_000,
            16,
            33,
            66,
            132,
            263,
            5,
            40,
            20,
        )
    )
    timing = bytearray([0])
    timing.extend(struct.pack("<IHQB", 7, 2, 0x0123_4567_89AB_CDEF, 1))
    timing.extend(
        struct.pack(
            "<8IBHHBBB",
            512_000,
            500_000,
            236_000,
            61_234,
            65_678,
            1_000,
            1_816_958,
            3_000_000,
            5,
            2_816,
            2_816,
            1,
            0,
            1,
        )
    )
    motion = bytearray([0])
    motion.extend(struct.pack("<IHQB", 7, 3, 0x0123_4567_89AB_CDEF, 2))
    motion.extend(
        struct.pack(
            "<IIBIIIIH",
            4_383,
            10_000,
            1,
            2_500,
            250,
            28_100,
            28_100,
            60_000,
        )
    )

    decoded = [
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload).payload
        for payload in (core, timing, motion)
    ]

    assert [len(core), len(timing), len(motion)] == [73, 56, 43]
    assert [fragment["fragment"] for fragment in decoded] == [0, 1, 2]
    assert all(fragment["namespace"] == "acceptance_matrix" for fragment in decoded)
    assert decoded[0]["target_rpm"] == [16, 33, 66, 132, 263]
    assert decoded[0]["acceptance_plan_digest"] == 0xAABB_CCDD_EEFF_0011
    assert decoded[1]["recovery_maximum_slot_us"] == 3_000_000
    assert decoded[2]["requested_velocity_mrev_s"] == 4_383


def test_decodes_schema_two_matrix_timing_and_compact_recovery_bounds() -> None:
    timing = bytearray([0])
    timing.extend(struct.pack("<IHQB", 7, 2, 0x0123_4567_89AB_CDEF, 1))
    timing.extend(
        struct.pack(
            "<6I2Q2IBHHBBB",
            512_000,
            500_000,
            236_000,
            60_541,
            66_456,
            1_000,
            13_217_792,
            12_827_648,
            1_816_958,
            3_000_000,
            5,
            2_816,
            2_816,
            1,
            0,
            1,
        )
    )
    marker = bytearray([0])
    marker.extend(struct.pack("<IQH", 7, 0x1122_3344_5566_7788, 1_024))
    marker.extend(struct.pack("<IiIHBH", 4_394, 263, 25_000, 1_024, 5, 20))
    marker.extend(struct.pack("<IIII", 512_000, 500_000, 236_000, 66_456))
    marker.extend(struct.pack("<III", 1_000, 1_816_958, 3_000_000))
    marker.extend(struct.pack("<BHH", 5, 2_816, 2_816))
    marker.extend(
        struct.pack(
            "<BiiiiQQ",
            1,
            0x3333_4444,
            0x1111_2222,
            0x0123,
            0,
            13_217_792,
            12_827_648,
        )
    )

    timing_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, timing) + timing)
    marker_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, marker) + marker)

    assert len(timing) == 72
    assert timing_frame.payload["recovery_lower_rate_q"] == [
        13_217_792,
        12_827_648,
    ]
    assert len(marker) == 98
    assert marker_frame.payload["recovery_lower_rate_q"] == [
        13_217_792,
        12_827_648,
    ]


def _acceptance_direction(
    *,
    response: tuple[int, int] | None,
    robust: int | None,
    deficit: tuple[int, int] | None,
    excluded_mask: int,
    blocking_levels: int,
) -> bytes:
    payload = bytearray(
        struct.pack(
            "<BB4iBBBBi",
            0,
            0,
            100,
            120,
            0,
            0,
            1,
            0b1111,
            0b0111,
            0b0111,
            1_000,
        )
    )
    payload.extend(bytes([response is not None]))
    if response is not None:
        payload.extend(struct.pack("<ii", *response))
    payload.extend(bytes([robust is not None]))
    if robust is not None:
        payload.extend(struct.pack("<i", robust))
    payload.extend(bytes([deficit is not None]))
    if deficit is not None:
        payload.extend(struct.pack("<ii", *deficit))
    payload.extend(struct.pack("<BH", excluded_mask, blocking_levels))
    return bytes(payload)


def test_decodes_acceptance_matrix_amplitude_with_signed_evidence() -> None:
    payload = bytearray([2])
    payload.extend(struct.pack("<IHQBi", 7, 4, 0x0123_4567_89AB_CDEF, 2, 66))
    payload.extend(
        _acceptance_direction(
            response=(700, 900),
            robust=800,
            deficit=(100, 300),
            excluded_mask=0b1000,
            blocking_levels=0x4321,
        )
    )
    payload.extend(
        _acceptance_direction(
            response=(-900, -700),
            robust=-800,
            deficit=(-300, -100),
            excluded_mask=0b1000,
            blocking_levels=0x1234,
        )
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 124
    assert frame.payload["record_kind"] == "acceptance_matrix_amplitude"
    assert frame.payload["amplitude_index"] == 2
    assert frame.payload["directions"][0]["response_interval_q"] == [700, 900]
    assert frame.payload["directions"][1]["robust_location_q"] == -800
    assert frame.payload["directions"][1]["deficit_interval_q"] == [-300, -100]
    assert frame.payload["directions"][0]["blocking_levels"] == [1, 2, 3, 4]


def test_decodes_acceptance_matrix_terminal_namespace_and_masks() -> None:
    payload = bytearray([3])
    payload.extend(
        struct.pack(
            "<IHHQQQBB",
            7,
            5,
            1,
            0x1111_2222_3333_4444,
            0x5555_6666_7777_8888,
            0x9999_AAAA_BBBB_CCCC,
            1,
            1,
        )
    )
    payload.extend(bytes([0b11111, 0b11110, 0b01111, 0b01110]))
    payload.extend(bytes([0b10000, 0, 0, 1]))
    payload.extend(struct.pack("<HHBBHQQ", 40, 32, 5, 4, 3, 100, 200))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 67
    assert frame.payload["namespace"] == "acceptance_matrix"
    assert frame.payload["outcome_name"] == "Inconclusive"
    assert frame.payload["cause_name"] == "InsufficientSharedFloor"
    assert frame.payload["attempted_masks"] == [0b11111, 0b11110]
    assert frame.payload["unattempted_masks"] == [0, 1]
    assert frame.payload["digest"] == 0x9999_AAAA_BBBB_CCCC


def test_decodes_velocity_integral_recovery_plan_fragment() -> None:
    payload = bytearray([0])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BIIIB", 2, 1_000, 1_816_958, 3_000_000, 7))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_plan_recovery"
    assert frame.payload["plan_digest"] == 0x5566_7788_1122_3344
    assert frame.payload["recovery_origin_band_counts"] == 1_000
    assert frame.payload["recovery_nominal_slot_us"] == 1_816_958
    assert frame.payload["recovery_maximum_slot_us"] == 3_000_000
    assert frame.payload["recovery_slot_count"] == 7


def test_breakaway_discovery_terminal_names_the_error_behind_an_internal_fault() -> None:
    """A `*_internal_fault` terminal must say why, not just that it stopped.

    Without the error code the 2026-08-03 intermittent was unattributable: the
    terminal reported `discovery_internal_fault` and nothing else. A capture
    predating the field reports `None` rather than 0, so "no error" and "field
    absent" stay distinct.
    """
    from foci_trace.protocol import decode_breakaway_record

    base = bytes([6]) + struct.pack("<IHQQHBBB", 7, 3, 0x1122, 0x3344, 128, 24, 0, 0)
    assert len(base) == 28

    old = decode_breakaway_record(base, protocol_version=16)
    assert old["record_kind"] == "breakaway_discovery_terminal"
    assert old["error_code"] is None
    assert old["error_code_name"] is None

    new = decode_breakaway_record(base + bytes([66]), protocol_version=17)
    assert new["terminal_cause_name"] == "discovery_internal_fault"
    assert new["error_code"] == 66
    assert new["error_code_name"] == "discovery_band_not_selected"


def test_decodes_a_breakaway_stage_c_plan_core_without_a_combined_response() -> None:
    """A breakaway Stage-C plan has `combined_selected_response: None`, and the
    encoder writes the flag byte only in the `Some` branch -- so the record
    simply ends. Reading a flag unconditionally at schema >= 8 runs off the end
    and the record decodes as nothing, which is why breakaway captures carried
    no `integral_plan_core` at all.
    """
    payload = bytearray([0])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BIHBIIH", 0, 0x05E2_1F75, 14, 0, 0x89AB_CDEF, 0x0123_4567, 951))
    payload.extend(struct.pack("<IIII", 0, 0, 0, 0))
    payload.extend(struct.pack("<BBB", 3, 3, 0))
    payload.extend(struct.pack("<iiii", 0, 0, 0, 0))
    payload.extend(
        struct.pack("<IiIIBHBHHHB", 4_394, 264, 25_000, 4_000, 12, 56, 14, 112, 2_816, 2_534, 1)
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_plan_core"
    assert frame.payload["schema_revision"] == 14
    assert frame.payload["final_p"] == 951
    assert frame.payload["combined_selected_response"] is False


def test_schema_seventeen_core_record_decodes_i_top_cap() -> None:
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 17, 0, 0x99AA_BBCC, 0xDDEE_FF00, 724))
    core.extend(
        struct.pack(
            "<IIIIBBBiiii",
            0x001C_0000,  # directional_memberships[0]
            0x001C_0000,  # directional_memberships[1]
            0x001C_0000,  # stage_b_joint_membership (schema >= 7)
            0x0004_0000,  # joint_membership
            3,  # directional_validity[0] (schema >= 4)
            3,  # directional_validity[1]
            1,  # reduced_margin
            17_000_000,
            18_000_000,
            -19_000_000,
            -18_000_000,
        )
    )
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 13, 60, 15, 120, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)  # limit_evidence_capability
    core.extend(struct.pack("<H", 1_310))  # new: i_top_cap, schema >= 17 only

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)

    assert frame.payload["schema_revision"] == 17
    assert frame.payload["i_top_cap"] == 1_310
    assert frame.payload["combined_selected_response"] is False


def test_schema_sixteen_core_record_decodes_i_top_cap() -> None:
    # Breakaway plans (schema 16) now also carry the trailing i_top_cap u16
    # -- the rung participates in the live gain search, not only schema 17's
    # ReproducedStageB/standalone characterization path.
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 16, 0, 0x99AA_BBCC, 0xDDEE_FF00, 724))
    core.extend(struct.pack("<IIIIBBBiiii", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 13, 60, 15, 120, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)
    core.extend(struct.pack("<H", 1_310))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)

    assert frame.payload["schema_revision"] == 16
    assert frame.payload["i_top_cap"] == 1_310
    assert frame.payload["combined_selected_response"] is False


def test_schema_sixteen_core_record_truncated_before_i_top_cap_raises() -> None:
    # Same as the schema-16 test above, but the two i_top_cap bytes are cut
    # off -- must raise, not decode with the field silently absent.
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 16, 0, 0x99AA_BBCC, 0xDDEE_FF00, 724))
    core.extend(struct.pack("<IIIIBBBiiii", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 13, 60, 15, 120, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)
    # i_top_cap bytes deliberately omitted.

    with pytest.raises(DecodeError):
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)


def test_schema_fifteen_core_record_has_no_i_top_cap_and_is_unaffected() -> None:
    # An unrelated, adjacent schema value (neither 16 nor 17, but still
    # >= 7 so every other schema-conditional field in this record decodes
    # the same as the 16/17 fixtures) must stay byte-identical: no trailing
    # i_top_cap bytes, and no key in the payload.
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 15, 0, 0x99AA_BBCC, 0xDDEE_FF00, 951))
    core.extend(struct.pack("<IIIIBBBiiii", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 12, 56, 14, 112, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)

    assert frame.payload["schema_revision"] == 15
    assert "i_top_cap" not in frame.payload
    assert frame.payload["combined_selected_response"] is False


def test_schema_seventeen_core_record_truncated_before_i_top_cap_raises() -> None:
    # Same as the schema-17 test above, but the two i_top_cap bytes are cut
    # off -- must raise, not decode with the field silently absent.
    core = bytearray([0])
    core.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    core.append(0)
    core.extend(struct.pack("<IHBIIH", 0x05E2_1F75, 17, 0, 0x99AA_BBCC, 0xDDEE_FF00, 724))
    core.extend(
        struct.pack(
            "<IIIIBBBiiii",
            0x001C_0000,
            0x001C_0000,
            0x001C_0000,
            0x0004_0000,
            3,
            3,
            1,
            17_000_000,
            18_000_000,
            -19_000_000,
            -18_000_000,
        )
    )
    core.extend(struct.pack("<IiIIBHBHH", 4_394, 264, 25_000, 4_000, 13, 60, 15, 120, 2_816))
    core.extend(struct.pack("<H", 2_534))
    core.append(1)
    # i_top_cap bytes deliberately omitted.

    with pytest.raises(DecodeError):
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, core) + core)


def test_decodes_schema_six_velocity_integral_recovery_flags() -> None:
    payload = bytearray([0])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BIIIBB", 2, 1_000, 1_816_958, 3_000_000, 14, 1))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_plan_recovery"
    assert frame.payload["recovery_slot_count"] == 14
    assert frame.payload["flags"] == 1


def test_decodes_schema_seven_velocity_integral_source_and_filtered_masks() -> None:
    payload = bytearray([0])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BIHBIIH", 0, 0x05E2_1F75, 7, 0, 0x89AB_CDEF, 0x0123_4567, 724))
    payload.extend(struct.pack("<IIII", 0x001C_0000, 0x001C_0000, 0x001C_0000, 0x0004_0000))
    payload.extend(struct.pack("<BBB", 3, 3, 1))
    payload.extend(struct.pack("<iiii", 10, 20, -20, -10))
    payload.extend(
        struct.pack("<IiIIBHBHHHB", 4_394, 264, 25_000, 4_000, 12, 56, 14, 112, 2_816, 2_534, 1)
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_plan_core"
    assert frame.payload["schema_revision"] == 7
    assert frame.payload["stage_b_joint_membership"] == 0x001C_0000
    assert frame.payload["joint_membership"] == 0x0004_0000
    assert frame.payload["final_p"] == 724


def test_decodes_rung_origin_recovery_record() -> None:
    payload = struct.pack(
        "<BIHBBHIiiiIQIIIHBB",
        9,
        7,
        12,
        1,
        4,
        1_448,
        4_394,
        8_458,
        900,
        734,
        1_000,
        3_150_000,
        1_200_000,
        500_000,
        1_936_000,
        2_533,
        1,
        1,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload == {
        "record_kind": "rung_origin_recovery",
        "run_sequence": 7,
        "evidence_sequence": 12,
        "stage": 1,
        "rung_index": 4,
        "velocity_p": 1_448,
        "planned_velocity_mrev_s": 4_394,
        "start_offset_counts": 8_458,
        "closest_offset_counts": 900,
        "final_offset_counts": 734,
        "origin_band_counts": 1_000,
        "lower_rate_q": 3_150_000,
        "moving_duration_us": 1_200_000,
        "settle_duration_us": 500_000,
        "total_duration_us": 1_936_000,
        "peak_torque_target_abs": 2_533,
        "binding_source": 1,
        "outcome": 1,
    }


def test_decodes_recovery_rest_record() -> None:
    payload = struct.pack(
        "<BIHBBHQiiiIiiIiIiiiBBqqHBB",
        10,
        0x0102_0304,
        19,
        0,
        15,
        1_024,
        0x1122_3344_5566_7788,
        -1_834,
        998,
        999,
        834,
        998,
        2_832,
        90_904,
        -3,
        7,
        0,
        -1,
        1,
        0,
        1,
        -9,
        11,
        1_560,
        0,
        0,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 89
    assert frame.payload == {
        "record_kind": "recovery_rest",
        "run_sequence": 0x0102_0304,
        "evidence_sequence": 19,
        "stage": 0,
        "rung_index": 15,
        "velocity_p": 1_024,
        "plan_digest": 0x1122_3344_5566_7788,
        "start_offset_counts": -1_834,
        "closest_offset_counts": 998,
        "final_offset_counts": 999,
        "planned_return_distance_counts": 834,
        "first_in_band_offset_counts": 998,
        "displacement_to_band_counts": 2_832,
        "entry_to_band_duration_us": 90_904,
        "lead_in_net_displacement_counts": -3,
        "lead_in_displacement_span_counts": 7,
        "scored_net_displacement_counts": 0,
        "scored_min_displacement_counts": -1,
        "scored_max_displacement_counts": 1,
        "stationarity_code": 0,
        "interval_present": 1,
        "exact_velocity_interval_q": [-9, 11],
        "peak_torque_target_abs": 1_560,
        "current_contact_code": 0,
        "disposition": 0,
    }
    for corrupt in (
        payload[:-1],
        payload + b"\x00",
        payload[:67] + b"\x01\x01" + payload[69:],
        payload[:68] + b"\x00" + payload[69:85] + payload[85:],
        payload[:88] + b"\x02",
    ):
        with pytest.raises(DecodeError):
            decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, corrupt) + corrupt)

    excluded = bytearray(payload)
    excluded[67:69] = b"\x04\x00"
    excluded[69:85] = bytes(16)
    excluded_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, excluded) + excluded)
    assert excluded_frame.payload["stationarity_code"] == 4
    assert excluded_frame.payload["interval_present"] == 0
    assert excluded_frame.payload["exact_velocity_interval_q"] == [0, 0]


def test_decodes_exact_scored_rest_selection_record() -> None:
    payload = struct.pack(
        "<BIHBBBBBQ3B4BHB6q",
        11,
        0x0102_0304,
        0x0506,
        1,
        4,
        7,
        1,
        0xFF,
        0x1122_3344_5566_7788,
        0,
        0,
        3,
        0b111,
        0b011,
        0b011,
        0b010,
        128,
        0,
        -9,
        0,
        -17,
        29,
        31,
        47,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 78
    assert frame.payload == {
        "record_kind": "rest_selection",
        "run_sequence": 0x0102_0304,
        "evidence_sequence": 0x0506,
        "stage": 1,
        "owner": 4,
        "rung_index": 7,
        "direction": 1,
        "slot": 0xFF,
        "plan_digest": 0x1122_3344_5566_7788,
        "structural_codes": [0, 0, 3],
        "interval_present_mask": 0b111,
        "usable_mask": 0b011,
        "zero_compatible_mask": 0b011,
        "selected_mask": 0b010,
        "selected_suffix_len": 128,
        "disposition": 0,
        "retained_sample_count": None,
        "intervals_q": [[-9, 0], [-17, 29], [31, 47]],
        "conditioned_rest_evidence": True,
    }


def test_decodes_noncontiguous_scored_rest_selection_record() -> None:
    payload = struct.pack(
        "<BIHBBBBBQ3B4BHB6q",
        11,
        0x0102_0304,
        0x0506,
        1,
        4,
        10,
        0,
        6,
        0x1122_3344_5566_7788,
        0,
        4,
        0,
        0b111,
        0b101,
        0b101,
        0b100,
        256,
        0,
        -58_721,
        90_287,
        -29_401,
        45_103,
        -6_848,
        22_512,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["structural_codes"] == [0, 4, 0]
    assert frame.payload["zero_compatible_mask"] == 0b101
    assert frame.payload["selected_mask"] == 0b100
    assert frame.payload["selected_suffix_len"] == 256


def test_decodes_scored_rest_selection_record_with_retained_sample_count() -> None:
    payload = struct.pack(
        "<BIHBBBBBQ3B4BHBH6q",
        11,
        0x0102_0304,
        0x0506,
        1,
        4,
        7,
        1,
        0xFF,
        0x1122_3344_5566_7788,
        0,
        6,
        6,
        0b001,
        0b001,
        0b001,
        0b001,
        64,
        0,
        100,
        -9,
        0,
        0,
        0,
        0,
        0,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 80
    assert frame.payload["structural_codes"] == [0, 6, 6]
    assert frame.payload["retained_sample_count"] == 100


def test_legacy_scored_rest_selection_record_has_no_retained_sample_count() -> None:
    payload = struct.pack(
        "<BIHBBBBBQ3B4BHB6q",
        11,
        0x0102_0304,
        0x0506,
        1,
        4,
        7,
        1,
        0xFF,
        0x1122_3344_5566_7788,
        0,
        0,
        3,
        0b111,
        0b011,
        0b011,
        0b010,
        128,
        0,
        -9,
        0,
        -17,
        29,
        31,
        47,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 78
    assert frame.payload["retained_sample_count"] is None


@pytest.mark.parametrize(
    "replacement",
    (
        {"selected_mask": 0b001},
        {"selected_suffix_len": 64},
        {"zero_compatible_mask": 0b111},
        {"usable_mask": 0b111},
        {"interval_present_mask": 0b011},
        {"structural_codes": (0, 0, 5)},
        {"stage": 0},
        {"direction": 2},
    ),
)
def test_rejects_invalid_scored_rest_selection_record(replacement) -> None:
    values = {
        "stage": 1,
        "owner": 4,
        "rung_index": 7,
        "direction": 1,
        "slot": 0xFF,
        "structural_codes": (0, 0, 3),
        "interval_present_mask": 0b111,
        "usable_mask": 0b011,
        "zero_compatible_mask": 0b011,
        "selected_mask": 0b010,
        "selected_suffix_len": 128,
    }
    values.update(replacement)
    payload = struct.pack(
        "<BIHBBBBBQ3B4BHB6q",
        11,
        7,
        2,
        values["stage"],
        values["owner"],
        values["rung_index"],
        values["direction"],
        values["slot"],
        11,
        *values["structural_codes"],
        values["interval_present_mask"],
        values["usable_mask"],
        values["zero_compatible_mask"],
        values["selected_mask"],
        values["selected_suffix_len"],
        0,
        -9,
        0,
        -17,
        29,
        31,
        47,
    )

    with pytest.raises(DecodeError):
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)


def test_decodes_velocity_integral_response_plan_marker() -> None:
    payload = bytearray([1])
    payload.extend(struct.pack("<IHIIHBI", 7, 1, 0x89AB_CDEF, 0x0123_4567, 1_448, 5, 69_888))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload == {
        "run_sequence": 7,
        "evidence_sequence": 1,
        "plan_digest_low": 0x89AB_CDEF,
        "plan_digest_high": 0x0123_4567,
        "final_p": 1_448,
        "positive_rung_count": 5,
        "maximum_workflow_ms": 69_888,
        "plan_digest": 0x0123_4567_89AB_CDEF,
        "record_kind": "integral_plan_marker",
    }


def test_decodes_recovery_aware_velocity_integral_plan_marker() -> None:
    payload = struct.pack(
        "<BIQHIiIHBHIIIIIIIBHH",
        0,
        7,
        0x0123_4567_89AB_CDEF,
        1_448,
        4_394,
        264,
        25_000,
        5,
        5,
        28,
        512_000,
        500_000,
        236_000,
        90_888,
        1_000,
        1_816_958,
        3_000_000,
        7,
        2_816,
        2_534,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 65
    assert frame.payload["record_kind"] == "integral_plan_marker"
    assert frame.payload["evidence_sequence"] == 1
    assert frame.payload["recovery_slot_count"] == 7


def test_decodes_velocity_integral_response_observation() -> None:
    payload = bytearray([2])
    payload.extend(struct.pack("<IHII", 7, 1, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BBHBBB", 2, 3, 20, 0, 0, 1))
    payload.extend(struct.pack("<iiiiiBH", 100, 110, 120, 10, 20, 2, 100))
    payload.extend(bytes([0]))
    payload.extend(struct.pack("<HBI", 64, 3, 16))
    payload.extend(struct.pack("<iBIBiIIBB", -5, 0, 10, 0, -60, 20, 30, 0, 3))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 77
    assert frame.payload["record_kind"] == "integral_observation"
    assert frame.payload["plan_digest"] == 0x5566_7788_1122_3344
    assert frame.payload["i_raw"] == 20
    assert frame.payload["direction"] == 0
    assert frame.payload["deficit_interval_q"] == [40, 80]
    assert frame.payload["suffix_len"] == 64
    assert frame.payload["lag_one_q"] == -60


def test_decodes_velocity_integral_response_rung() -> None:
    payload = bytearray([3])
    payload.extend(struct.pack("<IHII", 7, 9, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BH", 2, 20))
    payload.extend(
        struct.pack("<BBBBBBBBBBiiii", 0, 1, 0, 1, 0b1111, 0b0111, 0, 0, 0, 4, 10, 20, 0, 0)
    )
    payload.extend(
        struct.pack("<BBBBBBBBBBiiii", 1, 1, 1, 2, 0b1111, 0, 0, 0, 0, 0, -30, -20, -10, -5)
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 70
    assert frame.payload["record_kind"] == "integral_rung"
    assert frame.payload["i_raw"] == 20
    assert frame.payload["directions"][0]["components"] == [[10, 20]]
    assert frame.payload["directions"][0]["included_mask"] == 0b0111
    assert frame.payload["directions"][1]["classification"] == 1
    assert frame.payload["directions"][1]["components"] == [[-30, -20], [-10, -5]]


def test_decodes_schema_nine_velocity_integral_response_horizon() -> None:
    payload = bytearray([3])
    payload.extend(struct.pack("<IHII", 7, 9, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BHQIB", 9, 64, 655_360, 512_000, 1))
    for direction in range(2):
        payload.extend(
            struct.pack(
                "<BBBBBBBBBBiiii",
                direction,
                1,
                0,
                1,
                0b1111,
                0b1111,
                0,
                0,
                0,
                0,
                10,
                20,
                0,
                0,
            )
        )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 83
    assert frame.payload["record_kind"] == "integral_rung"
    assert frame.payload["tau_us"] == 655_360
    assert frame.payload["moving_horizon_us"] == 512_000
    assert frame.payload["horizon_class"] == 1


def test_decodes_velocity_integral_response_terminal() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHII", 7, 15, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<II", 0x99AA_BBCC, 0xDDEE_FF00))
    payload.extend(struct.pack("<BBBHHHHIIB", 1, 0, 1, 56, 9, 7, 1, 0b11111, 0b01111, 2))
    payload.extend(struct.pack("<IIII", 100, 0, 200, 0))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 59
    assert frame.payload["record_kind"] == "integral_terminal"
    assert frame.payload["plan_digest"] == 0x5566_7788_1122_3344
    assert frame.payload["digest"] == 0xDDEE_FF00_99AA_BBCC
    assert frame.payload["recovery_unavailable"] == 1
    assert frame.payload["expected_observations"] == 56
    assert frame.payload["emitted_observations"] == 9
    assert frame.payload["expected_rungs"] == 7
    assert frame.payload["emitted_rungs"] == 1
    assert frame.payload["eligible_masks"] == [0b11111, 0b01111]
    assert frame.payload["current_terminus"] == 1
    assert frame.payload["run_completed_us"] == 200


def test_decodes_schema_four_velocity_integral_bookend_masks() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHII", 7, 15, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<II", 0x99AA_BBCC, 0xDDEE_FF00))
    payload.extend(
        struct.pack(
            "<BBBHHHHIIBBB",
            1,
            0,
            0,
            72,
            72,
            9,
            9,
            0b1111111,
            0b0111111,
            0b11,
            0b10,
            0,
        )
    )
    payload.extend(struct.pack("<IIII", 100, 0, 200, 0))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 61
    assert frame.payload["opening_available_mask"] == 0b11
    assert frame.payload["bookend_available_mask"] == 0b10
    assert frame.payload["current_terminus"] is None


def test_decodes_schema_six_velocity_integral_terminal_flags() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHII", 7, 15, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<II", 0x99AA_BBCC, 0xDDEE_FF00))
    payload.extend(
        struct.pack(
            "<BBBHHHHIIBBB",
            0,
            0,
            0b110,
            112,
            112,
            14,
            14,
            0x0FFF,
            0x0FFF,
            0b11,
            0b11,
            0,
        )
    )
    payload.extend(struct.pack("<IIII", 100, 0, 200, 0))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 61
    assert frame.payload["recovery_flags"] == 0b110
    assert frame.payload["recovery_unavailable"] == 0


def test_decodes_integral_terminal_cause_namespace() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<II", 0, 0))
    payload.extend(
        struct.pack(
            "<BBBHHHHIIBBB",
            3,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    )
    payload.extend(struct.pack("<IIII", 0, 0, 0, 0))
    payload.extend(struct.pack("<B", 2))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 62
    assert frame.payload["record_kind"] == "integral_terminal"
    assert frame.payload["cause"] == 4
    assert frame.payload["cause_namespace"] == 2


def test_decodes_schema_seven_zero_motion_failed_terminal() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHII", 7, 0, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<II", 0, 0))
    payload.extend(
        struct.pack(
            "<BBBHHHHIIBBB",
            5,
            11,
            0b1000,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    )
    payload.extend(struct.pack("<IIII", 0, 0, 0, 0))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_terminal"
    assert frame.payload["outcome"] == 5
    assert frame.payload["cause"] == 11
    assert frame.payload["recovery_flags"] == 0b1000
    assert frame.payload["expected_observations"] == 0
    assert frame.payload["emitted_observations"] == 0


def test_decodes_schema_five_velocity_integral_rest_boundary() -> None:
    payload = bytearray([5])
    payload.extend(struct.pack("<IHQBBHb", 7, 19, 0x0123_4567_89AB_CDEF, 3, 5, 8, -1))
    payload.extend(struct.pack("<HIiiiH", 433, 433_000, -31, -33, 2, 7))
    payload.extend(struct.pack("<HIii", 3, 3_000, -1, -1))
    payload.extend(struct.pack("<HIii", 401, 401_000, 2, -31))
    payload.extend(
        struct.pack(
            "<BHBIiBIBiIIBB",
            0,
            64,
            2,
            16,
            -7,
            0,
            11,
            0,
            13,
            17,
            19,
            0,
            3,
        )
    )
    payload.extend(struct.pack("<HHHIIBBB", 2_534, 2_111, 2_400, 0, 1 << 15, 1, 0, 1))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 117
    assert frame.payload["record_kind"] == "integral_rest_boundary"
    assert frame.payload["positive_rung_index"] == 3
    assert frame.payload["slot"] == 5
    assert frame.payload["active_stationarity_code"] == 0
    assert frame.payload["active_suffix_len"] == 64
    assert frame.payload["first_transition_displacement_counts"] == -1
    assert frame.payload["last_transition_displacement_counts"] == -31
    assert frame.payload["cleanup_confirmed"] == 1


def test_decodes_stage_b_canonical_records_and_plan_identity() -> None:
    plan_values = (
        7,
        0,
        0x0105_C518,
        0x53E2_5AF6,
        5_000,
        1,
        4_800,
        7_500,
        1,
        0,
        288,
        8,
        128,
        5,
        2,
        256_000,
        200_000,
        9_120,
        10_000,
        1_000,
        250,
        1_250,
        1_250,
        1_000,
        900,
    )
    records = [
        (
            bytes([1]) + struct.pack("<IHIIIBIIHBiHHBHIIIIIIIIHH", *plan_values),
            "plan",
        ),
        (
            bytes([4])
            + struct.pack(
                "<IHIBBBBBHHBiiiiiBiBiBiBiBBii",
                7,
                21,
                0b11100,
                0,
                1,
                0,
                2,
                4,
                512,
                1024,
                3,
                10,
                20,
                15,
                12,
                18,
                0,
                100,
                0,
                200,
                0,
                90,
                0,
                210,
                0,
                6,
                10,
                20,
            ),
            "directional_region",
        ),
        (
            bytes([5]) + struct.pack("<IHIBBBHHB", 7, 22, 0b11100, 2, 4, 3, 512, 1024, 3),
            "joint_region",
        ),
        (
            bytes([6])
            + struct.pack(
                "<IHHBIBIIBIiiBBHHiBi iiBBHHiB".replace(" ", ""),
                7,
                23,
                724,
                3,
                0b11100,
                3,
                0x0002_0000,
                0x0004_0000,
                0,
                0b11100,
                12,
                18,
                0,
                0,
                1,
                1,
                0,
                0,
                0b11100,
                -18,
                -12,
                0,
                0,
                1,
                1,
                0,
                0,
            ),
            "handoff",
        ),
        (
            bytes([7])
            + struct.pack("<IHBBB iB iB iB i", 7, 24, 2, 0, 0b110, 10, 0, 20, 0, 3, 0, -4),
            "structured_boundary",
        ),
        (
            bytes([8])
            + struct.pack(
                "<IHII BIIIH iiii BBBBB HHHH II BB IIII".replace(" ", ""),
                7,
                25,
                4_000,
                25_000,
                3,
                0b11100,
                0b11100,
                0b11100,
                724,
                12,
                18,
                -18,
                -12,
                3,
                1,
                1,
                0,
                0,
                20,
                20,
                5,
                5,
                0x0105_C518,
                0x53E2_5AF6,
                0,
                3,
                0x1122_3344,
                0x0000_0001,
                0x5566_7788,
                0x0000_0002,
            ),
            "stage_b_terminal",
        ),
        (
            bytes([9])
            + struct.pack(
                "<IHBHHIIIIIIiiiiiiiiIIII",
                7,
                26,
                1,
                724,
                724,
                0b11100,
                0b11100,
                0b11100,
                0b11100,
                0b11100,
                0b11100,
                12,
                18,
                -18,
                -12,
                13,
                17,
                -17,
                -13,
                1,
                2,
                3,
                4,
            ),
            "reproduction",
        ),
    ]
    assert len(records[5][0]) == 85
    assert len(records[6][0]) == 84

    decoded = []
    for payload, expected_kind in records:
        frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)
        assert frame.payload["record_kind"] == expected_kind
        decoded.append(frame.payload)

    assert decoded[0]["plan_digest_low"] == 0x0105_C518
    assert decoded[1]["member_mask"] == 0b11100
    assert decoded[3]["directions"][1]["pooled_low_q16"] == -18
    assert decoded[5]["selected_memberships"] == [0b11100, 0b11100, 0b11100]
    assert decoded[6]["previous_memberships"] == [0b11100, 0b11100, 0b11100]
    assert decoded[6]["current_intervals"][1] == [-17, -13]


def test_decodes_compact_stage_b_rung_from_hardware_capture() -> None:
    payload = bytes.fromhex("030100000005000001000000000000000000000000000000000001010400000000")

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 33
    assert frame.payload["record_kind"] == "rung"
    assert frame.payload["run_sequence"] == 1
    assert frame.payload["evidence_sequence"] == 5
    assert frame.payload["rung_index"] == 0
    assert frame.payload["velocity_p"] == 1
    assert frame.payload["forward_class"] == 1
    assert frame.payload["reverse_class"] == 1
    assert frame.payload["flags"] == 4


def test_decodes_schema8_handoff_union_and_selected_component() -> None:
    payload = bytes([6]) + struct.pack(
        "<IHHBBIIBIIBIiiBBHHiBIiiBBHHiB",
        7,
        23,
        1024,
        3,
        3,
        0b11111,
        0b11100,
        3,
        0x0002_0000,
        0x0004_0000,
        1,
        0b11100,
        12,
        18,
        0,
        0,
        1,
        1,
        0,
        0,
        0b11100,
        -18,
        -12,
        0,
        0,
        1,
        1,
        0,
        0,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "handoff"
    assert frame.payload["model_direction_mask"] == 3
    assert frame.payload["joint_union_mask"] == 0b11111
    assert frame.payload["joint_member_mask"] == 0b11100


def test_decodes_stage_b_reproduction_schema_headers_and_shared_fragments() -> None:
    fragments = [
        (
            struct.pack("<BIHBBHHHBH", 10, 7, 33, 1, 0, 1448, 1024, 1448, 0, 3),
            "reproduction_v3_header",
        ),
        (
            struct.pack(
                "<BIHBBHHHIBBBBBH",
                10,
                7,
                34,
                1,
                0,
                256,
                1024,
                1024,
                0x0018_0000,
                19,
                2,
                3,
                1,
                1,
                8,
            ),
            "reproduction_v4_header",
        ),
        (
            struct.pack("<BIHBIIIIIbb", 11, 7, 33, 0, 0x38, 0x3C, 0x38, 0, 4, -1, 0),
            "reproduction_membership",
        ),
        (
            struct.pack("<BIHBiiiiii", 14, 7, 33, 0, 100, 120, 105, 125, 105, 120),
            "reproduction_pooled",
        ),
        (
            struct.pack("<BIHBiiiiiiB", 16, 7, 33, 0, 104, 115, 108, 118, 108, 115, 1),
            "reproduction_common",
        ),
        (
            struct.pack("<BIHBBiHH", 18, 7, 33, 0, 1, -2, 1448, 724),
            "reproduction_coverage",
        ),
        (
            struct.pack("<BIHIIII", 20, 7, 33, 1, 2, 3, 4),
            "reproduction_digest",
        ),
    ]

    decoded = [
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload).payload
        for payload, _kind in fragments
    ]

    assert [value["record_kind"] for value in decoded] == [kind for _payload, kind in fragments]
    assert decoded[0]["schema_revision"] == 3
    assert decoded[1]["schema_revision"] == 8
    assert decoded[1]["selected_joint_mask"] == 0x0018_0000
    assert decoded[1]["selected_member_count"] == 2
    assert decoded[1]["forward_validity"] == 3
    assert decoded[1]["reverse_validity"] == 1
    assert decoded[2]["low_delta"] == -1
    assert decoded[3]["overlap_high"] == 120
    assert decoded[4]["nonempty"] == 1
    assert decoded[5]["signed_rung_distance"] == -2
    assert decoded[6]["current_digest_high"] == 4


def test_decodes_canonical_four_observation_rung() -> None:
    def direction(
        classification: int,
        components: list[tuple[int, int]],
        masks: tuple[int, int, int],
        operable_count: int,
        flags: int,
        pooled: tuple[int, int, int, int],
        rates: tuple[int, int, int, int],
        rate_shift: int,
    ) -> bytes:
        padded = [*components, *([(0, 0)] * (2 - len(components)))]
        result = struct.pack(
            "<BBBBBBB",
            classification,
            len(components),
            *masks,
            operable_count,
            flags,
        )
        result += b"".join(struct.pack("<ii", *component) for component in padded)
        result += struct.pack("<iiiB", *pooled)
        result += b"".join(struct.pack("<iB", rate, rate_shift) for rate in rates)
        return result

    payload = struct.pack("<BIHBH", 3, 7, 12, 4, 724)
    payload += direction(
        0,
        [(100, 120)],
        (0b1111, 0b1111, 0b0111),
        4,
        0b11,
        (110, 104, 116, 1),
        (50, 60, 40, 70),
        2,
    )
    payload += direction(
        1,
        [(-140, -130), (-110, -100)],
        (0b1111, 0b1111, 0),
        3,
        0b01,
        (0, 0, 0, 0),
        (-70, -40, -80, -30),
        1,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 122
    assert frame.payload["record_kind"] == "rung"
    assert frame.payload["forward_class"] == 0
    assert frame.payload["forward_components"] == [[100, 120]]
    assert frame.payload["forward_included_mask"] == 0b0111
    assert frame.payload["forward_pooled_q16"] == 110
    assert frame.payload["forward_rate_bounds_q"] == [200, 240, 160, 280]
    assert frame.payload["reverse_class"] == 1
    assert frame.payload["reverse_components"] == [[-140, -130], [-110, -100]]
    assert frame.payload["reverse_operable_count"] == 3
    assert frame.payload["flags"] == 0b0111

    with pytest.raises(DecodeError):
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload[:-1]) + payload[:-1])


def test_decodes_canonical_velocity_sweep_terminal_digest() -> None:
    direction = struct.pack("<BHHHiiiiiBHB", 0, 8, 8, 32, 10, 20, 15, 12, 18, 3, 6, 0x0D)
    reverse = struct.pack("<BHHHiiiiiBHB", 1, 12, 12, 48, -20, -10, -15, -18, -12, 3, 6, 0x0D)
    payload = (
        bytes([4])
        + struct.pack("<III", 7, 4_000, 25_000)
        + direction
        + reverse
        + struct.pack("<HHHHIIBBB", 20, 20, 5, 5, 0x0105_C518, 0x53E2_5AF6, 0, 0, 3)
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "terminal"
    assert frame.payload["encoder_counts_per_rev"] == 4_000
    assert frame.payload["f_pwm_hz"] == 25_000
    assert frame.payload["directions"][1]["pooled_q16"] == -15
    assert frame.payload["digest_low"] == 0x0105_C518
    assert frame.payload["digest_high"] == 0x53E2_5AF6


def test_decodes_outer_safety_timing_and_rejects_count_drift() -> None:
    payload = bytes([30]) + struct.pack(
        "<IIIIIIIIBBBBBBBB",
        11,
        17,
        89,
        80,
        7,
        2,
        0,
        14_329,
        1,
        0,
        4,
        21,
        7,
        1,
        0,
        0,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload == {
        "record_kind": "outer_safety_timing",
        "engine_sequence": 11,
        "run_sequence": 17,
        "sample_count": 89,
        "bin_counts": [80, 7, 2, 0],
        "max_realized_gap_us": 14_329,
        "max_context_present": True,
        "max_stage": 0,
        "max_phase": 4,
        "max_rung": 21,
        "max_slot": 7,
        "max_target_zero": True,
        "zero_count": 0,
        "over_bound_count": 0,
    }

    corrupt = bytearray(payload)
    corrupt[9:13] = struct.pack("<I", 90)
    with pytest.raises(DecodeError, match="sample count"):
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, corrupt) + corrupt)


def test_decodes_velocity_primitive_current_evidence() -> None:
    payload = struct.pack(
        "<BIHBBBHHHHIIBB",
        31,
        0x0102_0304,
        0x0506,
        1,
        7,
        3,
        2_534,
        2_534,
        2_111,
        2_534,
        0x1122_3344,
        0x5566_7788,
        0,
        2,
    )

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload == {
        "record_kind": "velocity_primitive_current",
        "run_sequence": 0x0102_0304,
        "evidence_sequence": 0x0506,
        "stage": 1,
        "rung_index": 7,
        "slot": 3,
        "clamp_limit": 2_534,
        "clamp_readback": 2_534,
        "moving_pid_output_peak": 2_111,
        "zero_pid_output_peak": 2_534,
        "moving_status_flags": 0x1122_3344,
        "zero_status_flags": 0x5566_7788,
        "capability": 0,
        "contact": 2,
        "moving_clip_absence_proven": False,
        "moving_clip_uncertain": True,
    }


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


@pytest.mark.parametrize(
    ("event_id", "event_name"),
    (
        (134, "commission_velocity_integral_plan"),
        (135, "commission_velocity_integral_observation"),
        (136, "commission_velocity_integral_rung"),
        (137, "commission_velocity_integral_derived"),
        (138, "commission_velocity_integral_terminal"),
        (139, "commission_velocity_integral_complete"),
        (143, "commission_rung_origin_recovery"),
        (144, "commission_outer_safety_timing"),
        (145, "commission_velocity_limit_latch_test"),
        (146, "commission_acceptance_matrix_plan"),
        (147, "commission_acceptance_matrix_observation"),
        (148, "commission_acceptance_matrix_amplitude"),
        (149, "commission_acceptance_matrix_terminal"),
    ),
)
def test_decode_velocity_integral_response_event_names(event_id: int, event_name: str) -> None:
    payload = struct.pack("<BBHBxxxiiii", 1, 7, event_id, 2, 15, 4, 0, 0)

    frame = decode_frame(_header(FRAME_TYPE_EVENT, payload) + payload)

    assert frame.payload["event_name"] == event_name


def test_historical_mechanical_id_constants_are_reserved_not_reused() -> None:
    """MechanicalId phase, status, and event ids are retired from production
    firmware but archived captures still contain them. Pin the
    exact values so a future id reassignment cannot silently corrupt old
    captures.
    """
    assert PHASE_NAMES[9] == "MechanicalId"
    assert COMMISSION_STATUS_NAMES[10] == "mechanical_id_failed"
    assert EVENT_COMMISSION_MECHANICAL_ID_RESULT == 110
    assert EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT == 111
    assert EVENT_COMMISSION_MECHANICAL_ID_RESPONSE == 112
    assert EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION == 113


@pytest.mark.parametrize(
    ("event_id", "event_name"),
    (
        (110, "commission_mechanical_id_result"),
        (111, "commission_mechanical_id_attempt"),
        (112, "commission_mechanical_id_response"),
        (113, "commission_mechanical_id_classification"),
    ),
)
def test_decode_historical_mechanical_id_event_names(event_id: int, event_name: str) -> None:
    payload = struct.pack("<BBHBxxxiiii", 1, 7, event_id, 2, 15, 4, 0, 0)

    frame = decode_frame(_header(FRAME_TYPE_EVENT, payload) + payload)

    assert frame.payload["event_name"] == event_name


def test_rejects_bad_magic() -> None:
    with pytest.raises(DecodeError, match="bad magic"):
        decode_frame(b"BAD!" + b"\x00" * 40)


def test_assembler_reconstructs_frame_from_usb_packets() -> None:
    payload = bytes(range(84))
    frame = _header(FRAME_TYPE_CRITICAL, payload) + payload
    assembler = FrameAssembler()

    assert assembler.feed(frame[:64]) == []
    assert assembler.feed(frame[64:]) == [frame]


def test_decodes_schema_eight_compact_plan_response_intervals() -> None:
    payload = bytearray([0])
    payload.extend(struct.pack("<IQH", 7, 0x1122_3344_5566_7788, 1_024))
    payload.extend(struct.pack("<IiIHBH", 4_394, 264, 25_000, 1, 13, 60))
    payload.extend(struct.pack("<IIII", 512_000, 500_000, 236_000, 195_496))
    payload.extend(struct.pack("<III", 1_000, 1_816_958, 3_000_000))
    payload.extend(struct.pack("<BHH", 15, 2_816, 2_816))
    payload.extend(struct.pack("<Biiii", 1, 12_900_000, 13_200_000, -13_100_000, -12_800_000))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_plan_marker"
    assert frame.payload["combined_selected_response"] is True
    assert frame.payload["selected_response_intervals_q"] == [
        [12_900_000, 13_200_000],
        [-13_100_000, -12_800_000],
    ]


def test_decodes_schema_eight_integral_terminal_target() -> None:
    payload = bytearray([4])
    payload.extend(struct.pack("<IHQQBBB", 7, 99, 0x11, 0x22, 4, 0, 0xA0))
    payload.extend(struct.pack("<HHHHII", 120, 120, 15, 15, 0x1FFF, 0x1FFF))
    payload.extend(struct.pack("<BBBQQ", 3, 3, 0, 10, 20))
    payload.extend(struct.pack("<BHBB", 1, 1_024, 2, 7))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["record_kind"] == "integral_terminal"
    assert frame.payload["combined_workflow"] is True
    assert frame.payload["final_p"] == 1_024
    assert frame.payload["target_status"] == 2
    assert frame.payload["target_terminus"] == 6


def test_decodes_gate_rejection_diagnostic_on_an_excluded_observation() -> None:
    """An excluded observation reports why the gate rejected it.

    Byte-exact against the firmware encoder in integral_reporting.rs so the two
    sides cannot drift silently, which is how three wire defects shipped before.
    """
    payload = bytearray([2])
    payload.extend(struct.pack("<IHII", 7, 1, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BBHBBB", 3, 6, 512, 0, 4, 0))
    payload.extend(struct.pack("<iiiiiBH", 0, 0, 0, 0, 0, 0, 1_023))
    payload.extend(bytes([3]))  # ResolvedSlope
    payload.extend(bytes([1]))  # rejection present
    payload.extend(struct.pack("<HBBB", 256, 1, 0, 1))
    payload.extend(struct.pack("<qqqi", -8_773_356, -8_351_644, -2_389_706, 55_874))
    payload.extend(struct.pack("<qQ", -6_100_000, 3_710_294))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 96
    assert frame.payload["record_kind"] == "integral_observation"
    assert frame.payload["stationarity_code"] == 3
    assert frame.payload["rejection_present"] == 1
    assert frame.payload["rejection_suffix_len"] == 256
    assert frame.payload["rejection_usable_plateau"] == 1
    assert frame.payload["rejection_slope_passed"] == 0
    assert frame.payload["rejection_lag_one_passed"] == 1
    assert frame.payload["rejection_settled_rate_interval_q"] == [-8_773_356, -8_351_644]
    assert frame.payload["rejection_slope_margin_q"] == -2_389_706
    assert frame.payload["rejection_lag_one_margin_q"] == 55_874
    # A margin is half_width - |slope| and cannot recover either term.
    assert frame.payload["rejection_slope_q"] == -6_100_000
    assert frame.payload["rejection_slope_half_width_q"] == 3_710_294


def test_decodes_an_exclusion_that_never_reached_the_gates() -> None:
    payload = bytearray([2])
    payload.extend(struct.pack("<IHII", 7, 1, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BBHBBB", 3, 6, 512, 0, 4, 0))
    payload.extend(struct.pack("<iiiiiBH", 0, 0, 0, 0, 0, 0, 1_023))
    payload.extend(bytes([1]))  # NoBlockingPlateau
    payload.extend(bytes([0]))  # rejection absent

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 47
    assert frame.payload["record_kind"] == "integral_observation"
    assert frame.payload["stationarity_code"] == 1
    assert frame.payload["rejection_present"] == 0
    assert "rejection_slope_margin_q" not in frame.payload


def test_still_decodes_pre_rejection_excluded_observations() -> None:
    """The existing capture corpus must keep decoding unchanged."""
    payload = bytearray([2])
    payload.extend(struct.pack("<IHII", 7, 1, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BBHBBB", 3, 6, 512, 0, 4, 0))
    payload.extend(struct.pack("<iiiiiBH", 0, 0, 0, 0, 0, 0, 1_023))
    payload.extend(bytes([3]))

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 46
    assert frame.payload["record_kind"] == "integral_observation"
    assert frame.payload["stationarity_code"] == 3
    assert "rejection_present" not in frame.payload


def _stage_c_observation_payload(
    *,
    stationarity_code: int,
    rejection_variant: str | None = None,
    drift_ratio_ppm: int | None = None,
) -> bytes:
    """Build a Stage-C ("integral_observation") record byte-for-byte.

    Mirrors `encode_observation` in integral_reporting.rs: the fixed prefix,
    then (if present) the u32 drift ratio, then the stationarity code and its
    variable-length tail. `rejection_variant` selects among the tail shapes
    for an excluded observation: None (no more bytes at all), "absent"
    (presence flag only), "short" (rejection diagnostic without the trailing
    slope fields), or "full" (rejection diagnostic with them).
    """
    payload = bytearray([2])
    payload.extend(struct.pack("<IHII", 7, 1, 0x1122_3344, 0x5566_7788))
    payload.extend(struct.pack("<BBHBBB", 2, 3, 20, 0, 0, 1))
    payload.extend(struct.pack("<iiiiiBH", 100, 110, 120, 10, 20, 2, 100))
    if drift_ratio_ppm is not None:
        payload.extend(struct.pack("<I", drift_ratio_ppm))
    payload.extend(bytes([stationarity_code]))
    if stationarity_code == 0:
        payload.extend(struct.pack("<HBI", 64, 3, 16))
        payload.extend(struct.pack("<iBIBiIIBB", -5, 0, 10, 0, -60, 20, 30, 0, 3))
    elif rejection_variant == "absent":
        payload.extend(bytes([0]))
    elif rejection_variant == "short":
        payload.extend(bytes([1]))
        payload.extend(struct.pack("<HBBB", 256, 1, 0, 1))
        payload.extend(struct.pack("<qqqi", -8_773_356, -8_351_644, -2_389_706, 55_874))
    elif rejection_variant == "full":
        payload.extend(bytes([1]))
        payload.extend(struct.pack("<HBBB", 256, 1, 0, 1))
        payload.extend(struct.pack("<qqqi", -8_773_356, -8_351_644, -2_389_706, 55_874))
        payload.extend(struct.pack("<qQ", -6_100_000, 3_710_294))
    return bytes(payload)


@pytest.mark.parametrize(
    ("stationarity_code", "rejection_variant", "old_length", "new_length"),
    [
        (3, None, 46, 50),
        (1, "absent", 47, 51),
        (0, None, 77, 81),
        (3, "short", 80, 84),
        (3, "full", 96, 100),
    ],
)
def test_new_and_old_stage_c_observation_lengths_both_decode(
    stationarity_code: int, rejection_variant: str | None, old_length: int, new_length: int
) -> None:
    """A replaced length set would stop decoding every capture taken before the
    drift field existed, including the six exported for that design.
    A rejected length falls through silently rather than raising, so this is the
    only place the regression would show."""
    old_payload = _stage_c_observation_payload(
        stationarity_code=stationarity_code, rejection_variant=rejection_variant
    )
    assert len(old_payload) == old_length
    old_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, old_payload) + old_payload)
    assert old_frame.payload["record_kind"] == "integral_observation"
    assert old_frame.payload["drift_ratio_ppm"] is None

    new_payload = _stage_c_observation_payload(
        stationarity_code=stationarity_code,
        rejection_variant=rejection_variant,
        drift_ratio_ppm=38_503,
    )
    assert len(new_payload) == new_length
    new_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, new_payload) + new_payload)
    assert new_frame.payload["record_kind"] == "integral_observation"
    assert new_frame.payload["drift_ratio_ppm"] == 38_503


def test_stage_c_drift_ratio_decodes_before_stationarity_encoding() -> None:
    """Length acceptance alone would pass while reading the four bytes from the
    wrong place. Assert the decoded value sits between predicted_torque_target_abs
    and stationarity_code, for both an accepted and an excluded observation."""
    accepted = _stage_c_observation_payload(stationarity_code=0, drift_ratio_ppm=38_503)
    accepted_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, accepted) + accepted)
    assert accepted_frame.payload["drift_ratio_ppm"] == 38_503
    assert accepted_frame.payload["stationarity_code"] == 0

    excluded = _stage_c_observation_payload(
        stationarity_code=6, rejection_variant="absent", drift_ratio_ppm=412_009
    )
    excluded_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, excluded) + excluded)
    assert excluded_frame.payload["drift_ratio_ppm"] == 412_009
    assert excluded_frame.payload["stationarity_code"] == 6


def _stage_b_observation_payload(*, drift_ratio_ppm: int | None) -> bytes:
    """Build a Stage-B ("observation") record byte-for-byte, mirroring
    `encode_observation` in sweep_reporting.rs. Unlike Stage C this record has
    no variable-length tail: the drift ratio, when present, is the last four
    bytes."""
    payload = bytearray([2])
    payload.extend(struct.pack("<IH", 7, 1))
    payload.extend(struct.pack("<BBiBBB", 0, 0, 3_200, 1, 0, 0))
    payload.extend(struct.pack("<IB", 0x1122_3344, 4))
    payload.extend(struct.pack("<IB", 0x5566_7788, 4))
    payload.extend(struct.pack("<iiiiiB", -5_000_000, -4_000_000, -3_000_000, -6_000, -5_000, 2))
    payload.extend(struct.pack("<HBI", 64, 3, 16))
    payload.extend(struct.pack("<IB", 10, 0))
    payload.extend(struct.pack("<iBIB", 12, 0, 20, 0))
    payload.extend(struct.pack("<iIIBB", -18, 30, 40, 0, 3))
    payload.extend(struct.pack("<H", 350))
    payload.extend(struct.pack("<iii", 123_456, 120_000, 130_000))
    payload.extend(struct.pack("<IB", 1 << 20, 0))
    payload.extend(struct.pack("<H", 500))
    if drift_ratio_ppm is not None:
        payload.extend(struct.pack("<I", drift_ratio_ppm))
    return bytes(payload)


def test_new_and_old_stage_b_observation_lengths_both_decode() -> None:
    """Stage B has no rejection diagnostic block at all, so extending only the
    Stage C length set would leave every Stage B capture undecoded for the new
    field while claiming the length-set change was complete."""
    old_payload = _stage_b_observation_payload(drift_ratio_ppm=None)
    assert len(old_payload) == 104
    old_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, old_payload) + old_payload)
    assert old_frame.payload["record_kind"] == "observation"
    assert old_frame.payload["drift_ratio_ppm"] is None

    new_payload = _stage_b_observation_payload(drift_ratio_ppm=14_220)
    assert len(new_payload) == 108
    new_frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, new_payload) + new_payload)
    assert new_frame.payload["record_kind"] == "observation"
    assert new_frame.payload["drift_ratio_ppm"] == 14_220
    assert new_frame.payload["predicted_torque_target_abs"] == 500


def _matrix_plan_core(order_byte: int, schema_revision: int = 5) -> bytes:
    """Build one matrix plan core fragment with an explicit packed order byte."""
    core = bytearray([0])
    core.extend(struct.pack("<IHQB", 7, 1, 0x0123_4567_89AB_CDEF, 0))
    core.extend(
        struct.pack(
            "<HBIHQBHHHII5iBHH",
            schema_revision,
            0,
            0x05E2_1F75,
            10,
            0xAABB_CCDD_EEFF_0011,
            order_byte,
            1_024,
            1_024,
            3,
            25_000,
            4_000,
            16,
            33,
            66,
            132,
            263,
            5,
            40,
            20,
        )
    )
    return bytes(core)


def test_unpacks_the_amplitude_and_slot_order_from_the_packed_order_byte() -> None:
    """Slot order rides in the high nibble of the amplitude-order byte.

    `order` must stay the raw wire byte because the plan digest hashes it, while
    the unpacked pair is what makes a mirrored run identifiable in an audit.
    """
    for order_byte, amplitude, slot in (
        (0x01, 1, 0),
        (0x02, 2, 0),
        (0x11, 1, 1),
        (0x12, 2, 1),
    ):
        payload = _matrix_plan_core(order_byte)
        frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

        assert frame.payload["order"] == order_byte, "digest input must stay the raw byte"
        assert frame.payload["amplitude_order"] == amplitude
        assert frame.payload["slot_order"] == slot


def test_pre_mirror_captures_report_the_forward_first_slot_order() -> None:
    """Schema 4 and earlier never set the high nibble, so they are forward-first."""
    payload = _matrix_plan_core(0x02, schema_revision=4)
    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert frame.payload["order"] == 2
    assert frame.payload["amplitude_order"] == 2
    assert frame.payload["slot_order"] == 0


def test_rejects_an_unusable_packed_order_byte() -> None:
    for order_byte in (0x00, 0x03, 0x21, 0xF1):
        payload = _matrix_plan_core(order_byte)
        with pytest.raises(DecodeError):
            decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)


# --- Breakaway-seeded P ladder: canonical evidence records ------------------
#
# Every payload below reproduces, byte for byte, the fixture values used by
# the firmware's own golden tests
# (`breakaway_trace_reporting_tests.rs::encode_*_has_stable_frame_layout`,
# schema-15 Stage B / schema-14 Stage C per `task-7b-report.md`). A field
# reorder, width change, or dropped field in either `decode_breakaway_record`
# or the firmware encoder breaks these tests -- that is the point of pinning
# the exact hex literals rather than round-tripping through the decoder
# itself.

_BW_RUN_SEQUENCE = 0x1020_3040
_BW_EVIDENCE_SEQUENCE = 0x5060
_BW_PLAN_DIGEST = 0x1122_3344_5566_7788
_BW_PRIOR_PLAN_DIGEST = 0x99AA_BBCC_DDEE_FF00


def _bw_probe_planned_v14() -> bytes:
    return bytes([0]) + struct.pack(
        "<IHQHBBHHQiHHII",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x0910,
        0x11,
        0x12,
        0x1314,
        0x1516,
        0x1718_1920_2122_2324,
        -0x2526_2728,
        0x2930,
        0x3132,
        0x3334_3536,
        0x3738_3940,
    )


def _bw_probe_planned() -> bytes:
    return bytes([0]) + struct.pack(
        "<IHQBBHHQiHHIIII",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x11,
        0x12,
        0x1314,
        0x1516,
        0x1718_1920_2122_2324,
        -0x2526_2728,
        0x2930,
        0x3132,
        0x3334_3536,
        0x3738_3940,
        0x4142_4344,
        0x4546_4748,
    )


def _bw_probe_planned_legacy() -> bytes:
    return bytes([0]) + struct.pack(
        "<IHQHBBHH",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x0910,
        0x11,
        0x12,
        0x1314,
        0x1516,
    )


def _bw_directional_breakaway_v13() -> bytes:
    return bytes([1]) + struct.pack(
        "<IHQBBHHB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        1,
        1,
        0x0910,
        0x1112,
        6,
    )


def _bw_directional_breakaway_v14() -> bytes:
    return _bw_directional_breakaway_v13() + bytes([0b11])


def _bw_probe_resolved() -> bytes:
    return bytes([1]) + struct.pack(
        "<IHQBHIB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        3,
        0x0910,
        0x1112_1314,
        17,
    )


def _bw_probe_terminal_v14() -> bytes:
    return bytes([2]) + struct.pack(
        "<IHQHBBBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x0910,
        4,
        0,
        0,
        3,
    )


def _bw_probe_terminal(*, cause: int = 22, checkpoint: int = 42) -> bytes:
    return bytes([2]) + struct.pack(
        "<IHQBBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        cause,
        0,
        checkpoint,
    )


def _bw_probe_terminal_v11() -> bytes:
    return bytes([2]) + struct.pack(
        "<IHQHBBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x0910,
        6,
        0,
        17,
    )


def _bw_probe_terminal_v10() -> bytes:
    return bytes([2]) + struct.pack(
        "<IHQHBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        0x0910,
        1,
        0,
    )


def _bw_discovery_planned() -> bytes:
    return bytes([3]) + struct.pack(
        "<IHQQHHBHHBHHII",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        _BW_PRIOR_PLAN_DIGEST,
        0x0708,
        0x0910,
        8,
        4,
        0x1112,
        1,
        0x1314,
        0x1516,
        0x1718_1920,
        0x2122_2324,
    )


def _bw_ceiling_source() -> bytes:
    return bytes([4]) + struct.pack(
        "<IHQB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        1,
    )


_BW_RAW_OBSERVATION_VARIANCE_OF_MEAN_Q2 = 0x1112_1314_1516_1718_191A_1B1C_1D1E_1F20


def _bw_raw_observation_v13() -> bytes:
    """Tag 5's new shape: `struct.pack` has no native
    128-bit format code, so `variance_of_mean_q2` is appended as its own
    16-byte little-endian span rather than folded into the `struct.pack`
    call the other fields share -- mirroring `_EvidenceCursor._take_u128`.
    """
    return (
        bytes([5])
        + struct.pack(
            "<IHQBBBBq",
            _BW_RUN_SEQUENCE,
            _BW_EVIDENCE_SEQUENCE,
            _BW_PLAN_DIGEST,
            1,
            3,
            2,
            1,
            -0x0102_0304_0506_0708,
        )
        + _BW_RAW_OBSERVATION_VARIANCE_OF_MEAN_Q2.to_bytes(16, "little")
        # target_rate_q, then the frozen classify/consensus
        # inputs: stationarity_code, settled-rate low/high, deficit low/high,
        # predicted/usable/hard torque, disturbance present + Q16.16 bounds.
        + struct.pack(
            "<qBqqqqHHHBii",
            0x2122_2324_2526_2728,
            4,
            -0x3132_3334_3536_3738,
            0x4142_4344_4546_4748,
            -0x5152_5354_5556_5758,
            0x6162_6364_6566_6768,
            0x7172,
            0x8182,
            0x9192,
            1,
            -0x0102_0304,
            0x0506_0708,
        )
    )


def _bw_raw_observation_v14(
    *, stationarity_code: int = 4, post_fit_lag_one_state: int = 2
) -> bytes:
    payload = bytearray(_bw_raw_observation_v13())
    payload[51] = stationarity_code
    payload.extend(struct.pack("<IB", 0x1112_1314, post_fit_lag_one_state))
    return bytes(payload)


def _bw_raw_observation_v15(
    *,
    phase: int = 1,
    stationarity_code: int = 4,
    post_fit_lag_one_state: int = 2,
    minimum: int = -0x1516_1718,
    maximum: int = 0x2122_2324,
    max_interval_us: int = 0x2526_2728,
) -> bytes:
    payload = bytearray(
        _bw_raw_observation_v14(
            stationarity_code=stationarity_code,
            post_fit_lag_one_state=post_fit_lag_one_state,
        )
    )
    payload[15] = phase
    payload.extend(struct.pack("<iiI", minimum, maximum, max_interval_us))
    return bytes(payload)


def _bw_raw_observation(
    *,
    phase: int = 1,
    stationarity_code: int = 4,
    post_fit_lag_one_state: int = 2,
    minimum: int = -0x1516_1718,
    maximum: int = 0x2122_2324,
    max_interval_us: int = 0x2526_2728,
    moving_pid_output_peak: int = 0xA1A2,
) -> bytes:
    payload = bytearray(
        _bw_raw_observation_v15(
            phase=phase,
            stationarity_code=stationarity_code,
            post_fit_lag_one_state=post_fit_lag_one_state,
            minimum=minimum,
            maximum=maximum,
            max_interval_us=max_interval_us,
        )
    )
    payload.extend(struct.pack("<H", moving_pid_output_peak))
    return bytes(payload)


def _bw_probe_decision(disposition: int = 1) -> bytes:
    return bytes([13]) + struct.pack(
        "<IHQBBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        1,
        3,
        disposition,
    )


def _bw_discovery_terminal() -> bytes:
    return bytes([6]) + struct.pack(
        "<IHQQHBBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        _BW_PRIOR_PLAN_DIGEST,
        0x0910,
        0,
        2,
        0,
    )


def _bw_discovery_rung_margin() -> bytes:
    return bytes([7]) + struct.pack(
        "<IHQBHiiiii",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        1,
        0x0910,
        70_500,
        79_500,
        500,
        4_500,
        -1_200,
    )


def _bw_rung_zero_diagnostic() -> bytes:
    return bytes([8]) + struct.pack(
        "<IHQBB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        1,
        0,
    )


def _bw_confirmation_planned() -> bytes:
    return bytes([9]) + struct.pack(
        "<IHQQHHHBBBBi",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        _BW_PRIOR_PLAN_DIGEST,
        8,
        4,
        0x0910,
        70,
        80,
        0,
        0,
        1_500,
    )


def _bw_confirmation_terminal() -> bytes:
    return bytes([10]) + struct.pack(
        "<IHQQHBBBBBBBBHHHB",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        _BW_PRIOR_PLAN_DIGEST,
        8,
        21,
        0b1111,
        0b1111,
        0b1111,
        0b1111,
        0b1111,
        0b1111,
        1,
        0x0910,
        20,
        67,
        0,
    )


def _bw_confirmation_observation() -> bytes:
    return bytes([11]) + struct.pack(
        "<IHQBBiiH",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        _BW_PLAN_DIGEST,
        3,
        1,
        70_800,
        79_200,
        18,
    )


def _bw_campaign_terminal() -> bytes:
    return bytes([12]) + struct.pack(
        "<IHBBBQ",
        _BW_RUN_SEQUENCE,
        _BW_EVIDENCE_SEQUENCE,
        2,
        21,
        1,
        _BW_PLAN_DIGEST,
    )


def test_decode_breakaway_probe_planned_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_probe_planned()
    assert len(payload) == 53
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_probe_planned"
    assert decoded["tag"] == 0
    assert decoded["run_sequence"] == _BW_RUN_SEQUENCE
    assert decoded["evidence_sequence"] == _BW_EVIDENCE_SEQUENCE
    assert decoded["plan_digest"] == _BW_PLAN_DIGEST
    assert decoded["max_observations"] == 0x11
    assert decoded["search_count"] == 0x12
    assert decoded["p_start_raw"] == 0x1314
    assert decoded["p_top_raw"] == 0x1516
    assert decoded["workflow_digest"] == 0x1718_1920_2122_2324
    assert decoded["target_velocity_rpm"] == -0x2526_2728
    assert decoded["hard_torque_limit"] == 0x2930
    assert decoded["usable_torque_limit"] == 0x3132
    assert decoded["pwm_hz"] == 0x3334_3536
    assert decoded["encoder_counts_per_rev"] == 0x3738_3940
    assert decoded["motion_threshold_counts"] == 0x4142_4344
    assert decoded["max_capture_interval_us"] == 0x4546_4748


def test_decode_breakaway_probe_planned_preserves_v14_shape() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_probe_planned_v14(), protocol_version=14)
    assert decoded["family_size"] == 0x0910
    assert decoded["motion_threshold_counts"] is None


def test_decode_breakaway_probe_planned_preserves_historical_capture_support() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_probe_planned_legacy(), protocol_version=13)
    assert decoded["record_kind"] == "breakaway_probe_planned"
    assert decoded["workflow_digest"] is None


def test_decode_breakaway_probe_resolved_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_probe_resolved()
    assert len(payload) == 23
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_probe_resolved"
    assert decoded["rung_index"] == 3
    assert decoded["breakaway_p_raw"] == 0x0910
    assert decoded["motion_threshold_counts"] == 0x1112_1314
    assert decoded["observation_count"] == 17


def test_decode_breakaway_directional_breakaway_preserves_v14_shape() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_directional_breakaway_v14(), protocol_version=14)
    assert decoded["record_kind"] == "breakaway_directional_breakaway"
    assert decoded["lower_disposition_mask"] == 0b11


def test_decode_breakaway_directional_breakaway_preserves_v13_shape() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_directional_breakaway_v13(), protocol_version=13)
    assert decoded["lower_disposition_mask"] is None


def test_decode_breakaway_probe_terminal_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_probe_terminal()
    assert len(payload) == 18
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_probe_terminal"
    assert decoded["terminal_cause"] == 22
    assert decoded["terminal_cause_name"] == "probe_excursion_evidence_invalid"
    assert decoded["has_safety_fault"] == 0
    assert decoded["fault_checkpoint"] == 42
    assert decoded["fault_checkpoint_name"] == "probe_capture_interval_refusal"


@pytest.mark.parametrize(
    ("payload", "fault_checkpoint"),
    [(_bw_probe_terminal_v10(), 0), (_bw_probe_terminal_v11(), 17)],
)
def test_decode_breakaway_probe_terminal_preserves_historical_capture_support(
    payload: bytes, fault_checkpoint: int
) -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(payload, protocol_version=13)
    assert decoded["record_kind"] == "breakaway_probe_terminal"
    assert decoded["fault_checkpoint"] == fault_checkpoint
    assert decoded["exclusion_reason"] == 0
    assert decoded["exclusion_reason_name"] == "none"


def test_breakaway_terminal_cause_one_uses_protocol_versioned_spelling() -> None:
    from foci_trace.protocol import decode_breakaway_record

    v13 = decode_breakaway_record(_bw_probe_terminal_v10(), protocol_version=13)
    v14 = decode_breakaway_record(_bw_probe_terminal_v10(), protocol_version=14)
    v15 = decode_breakaway_record(_bw_probe_terminal(cause=1, checkpoint=0), protocol_version=15)
    assert v13["terminal_cause_name"] == "probe_no_motion_at_authority"
    assert v14["terminal_cause_name"] == "probe_no_productive_response_at_authority"
    assert v15["terminal_cause_name"] == "probe_no_repeatable_motion_at_authority"


@pytest.mark.parametrize("cause", [2, 3, 4])
def test_breakaway_probe_terminal_rejects_reserved_v15_causes(cause: int) -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="cause is invalid"):
        decode_breakaway_record(_bw_probe_terminal(cause=cause, checkpoint=0))


@pytest.mark.parametrize("cause", [2, 3, 4])
def test_breakaway_probe_terminal_preserves_v14_cause_meanings(cause: int) -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = bytearray(_bw_probe_terminal_v14())
    payload[17] = cause
    decoded = decode_breakaway_record(bytes(payload), protocol_version=14)
    assert decoded["terminal_cause"] == cause


def test_decode_breakaway_probe_terminal_rejects_unknown_checkpoint() -> None:
    from foci_trace.protocol import DecodeError, decode_breakaway_record

    payload = _bw_probe_terminal(cause=1, checkpoint=44)
    with pytest.raises(DecodeError, match="checkpoint is invalid"):
        decode_breakaway_record(payload)


def test_decode_breakaway_probe_terminal_rejects_unknown_exclusion_reason() -> None:
    from foci_trace.protocol import DecodeError, decode_breakaway_record

    payload = _bw_probe_terminal_v14()[:-1] + bytes([11])
    with pytest.raises(DecodeError, match="exclusion reason is invalid"):
        decode_breakaway_record(payload, protocol_version=14)


@pytest.mark.parametrize(
    ("reason", "name"),
    [
        (5, "moving_no_blocking_plateau"),
        (6, "moving_too_few_decorrelated_blocks"),
        (7, "moving_resolved_slope"),
        (8, "moving_structured_residual"),
        (9, "moving_stationarity_arithmetic"),
        (10, "moving_excessive_drift"),
    ],
)
def test_decode_breakaway_probe_terminal_preserves_exact_stationarity_exclusion(
    reason: int, name: str
) -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(
        _bw_probe_terminal_v14()[:-1] + bytes([reason]), protocol_version=14
    )
    assert decoded["exclusion_reason_name"] == name


def test_decode_breakaway_discovery_planned_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_discovery_planned()
    assert len(payload) == 45
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_discovery_planned"
    assert decoded["plan_digest"] == _BW_PLAN_DIGEST
    assert decoded["prior_plan_digest"] == _BW_PRIOR_PLAN_DIGEST
    assert decoded["schema_revision"] == 0x0708
    assert decoded["family_size"] == 0x0910
    assert decoded["rung_count"] == 8
    assert decoded["observations_per_direction"] == 4
    assert decoded["floor_p_raw"] == 0x1112
    assert decoded["floor_origin"] == 1
    assert decoded["breakaway_p_raw"] == 0x1314
    assert decoded["ceiling_p_raw"] == 0x1516
    assert decoded["first_additive_step_raw"] == 0x1718_1920
    assert decoded["maximum_workflow_ms"] == 0x2122_2324


def test_decode_breakaway_ceiling_source_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_ceiling_source()
    assert len(payload) == 16
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_ceiling_source"
    assert decoded["binding_source"] == 1


def test_decode_breakaway_raw_observation_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_raw_observation()
    assert len(payload) == 118
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_raw_observation"
    assert decoded["plan_digest"] == _BW_PLAN_DIGEST
    assert decoded["phase"] == 1
    assert decoded["rung_index"] == 3
    assert decoded["slot_index"] == 2
    assert decoded["direction"] == 1
    assert decoded["mean_rate_q"] == -0x0102_0304_0506_0708
    assert decoded["variance_of_mean_q2"] == _BW_RAW_OBSERVATION_VARIANCE_OF_MEAN_Q2
    assert decoded["target_rate_q"] == 0x2122_2324_2526_2728
    assert decoded["stationarity_code"] == 4
    assert decoded["settled_rate_low_q"] == -0x3132_3334_3536_3738
    assert decoded["settled_rate_high_q"] == 0x4142_4344_4546_4748
    assert decoded["deficit_low_q"] == -0x5152_5354_5556_5758
    assert decoded["deficit_high_q"] == 0x6162_6364_6566_6768
    assert decoded["predicted_torque_target_abs"] == 0x7172
    assert decoded["usable_torque_limit"] == 0x8182
    assert decoded["hard_torque_limit"] == 0x9192
    assert decoded["disturbance_present"] == 1
    assert decoded["disturbance_low_q16"] == -0x0102_0304
    assert decoded["disturbance_high_q16"] == 0x0506_0708
    assert decoded["drift_ratio_ppm"] == 0x1112_1314
    assert decoded["post_fit_lag_one_state"] == 2
    assert decoded["stroke_min_displacement_counts"] == -0x1516_1718
    assert decoded["stroke_max_displacement_counts"] == 0x2122_2324
    assert decoded["capture_max_interval_us"] == 0x2526_2728
    assert decoded["moving_pid_output_peak"] == 0xA1A2


def test_decode_breakaway_raw_observation_still_reads_a_v15_capture() -> None:
    """A pre-v16 capture decodes unchanged, with no measured peak.

    The two 2026-08-03 stepper_x captures are v15. They must keep replaying
    after the record grew, so the measured peak reads as `None` rather than
    the record failing its exact-length dispatch.
    """
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_raw_observation_v15()
    assert len(payload) == 116

    decoded = decode_breakaway_record(payload, protocol_version=15)

    assert decoded["record_kind"] == "breakaway_raw_observation"
    assert decoded["capture_max_interval_us"] == 0x2526_2728
    assert decoded["moving_pid_output_peak"] is None


@pytest.mark.parametrize("max_interval_us", [0, 2_001])
def test_decode_breakaway_raw_observation_retains_interval_refusal_evidence(
    max_interval_us: int,
) -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_raw_observation(max_interval_us=max_interval_us))
    assert decoded["capture_max_interval_us"] == max_interval_us


def test_decode_breakaway_raw_observation_rejects_non_monotone_extrema() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="extrema"):
        decode_breakaway_record(_bw_raw_observation(phase=0, minimum=1, maximum=2))


def test_decode_breakaway_raw_observation_preserves_v13_shape_without_inference() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_raw_observation_v13(), protocol_version=13)
    assert decoded["drift_ratio_ppm"] is None
    assert decoded["post_fit_lag_one_state"] is None


def test_decode_breakaway_raw_observation_preserves_v14_shape() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_raw_observation_v14(), protocol_version=14)
    assert decoded["drift_ratio_ppm"] == 0x1112_1314
    assert decoded["stroke_min_displacement_counts"] is None
    assert decoded["capture_max_interval_us"] is None


@pytest.mark.parametrize("state", [0, 1])
def test_decode_breakaway_raw_observation_rejects_usable_v14_lag_states_below_passed(
    state: int,
) -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="usable.*lag-one state"):
        decode_breakaway_record(
            _bw_raw_observation(stationarity_code=0, post_fit_lag_one_state=state)
        )


def test_decode_breakaway_raw_observation_accepts_usable_v14_lag_state_passed() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(
        _bw_raw_observation(stationarity_code=0, post_fit_lag_one_state=2)
    )
    assert decoded["post_fit_lag_one_state"] == 2


def test_decode_breakaway_raw_observation_rejects_unknown_v14_lag_state() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="lag-one state is invalid"):
        decode_breakaway_record(_bw_raw_observation(post_fit_lag_one_state=3))


@pytest.mark.parametrize("disposition", [0, 1])
def test_decode_breakaway_probe_decision_matches_the_firmware_golden_bytes(
    disposition: int,
) -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_probe_decision(disposition)
    assert len(payload) == 18
    decoded = decode_breakaway_record(payload)
    assert decoded == {
        "run_sequence": _BW_RUN_SEQUENCE,
        "evidence_sequence": _BW_EVIDENCE_SEQUENCE,
        "plan_digest": _BW_PLAN_DIGEST,
        "direction": 1,
        "rung_index": 3,
        "disposition": disposition,
        "record_kind": "breakaway_probe_decision",
        "tag": 13,
    }


def test_decode_breakaway_probe_decision_preserves_v14_transient_disposition() -> None:
    from foci_trace.protocol import decode_breakaway_record

    decoded = decode_breakaway_record(_bw_probe_decision(2), protocol_version=14)
    assert decoded["disposition"] == 2


def test_decode_breakaway_probe_decision_rejects_v15_transient_disposition() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="disposition is invalid"):
        decode_breakaway_record(_bw_probe_decision(2), protocol_version=15)


def test_decode_breakaway_raw_observation_rejects_an_invalid_phase() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = bytearray(_bw_raw_observation())
    # Layout: tag(1) + run_sequence(4) + evidence_sequence(2) + plan_digest(8)
    # puts `phase` at byte offset 15.
    payload[15] = 3
    with pytest.raises(DecodeError, match="phase is invalid"):
        decode_breakaway_record(bytes(payload))


def test_decode_breakaway_discovery_terminal_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_discovery_terminal()
    assert len(payload) == 28
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_discovery_terminal"
    assert decoded["plan_digest"] == _BW_PLAN_DIGEST
    assert decoded["prior_plan_digest"] == _BW_PRIOR_PLAN_DIGEST
    assert decoded["family_size"] == 0x0910
    assert decoded["terminal_cause"] == 0
    assert decoded["collected_count"] == 2
    assert decoded["has_safety_fault"] == 0


def test_decode_breakaway_discovery_rung_margin_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_discovery_rung_margin()
    assert len(payload) == 38
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_discovery_rung_margin"
    assert decoded["collected_index"] == 1
    assert decoded["p_raw"] == 0x0910
    assert decoded["lower_percent_milli"] == 70_500
    assert decoded["upper_percent_milli"] == 79_500
    assert decoded["margin_percent_milli"] == 500
    assert decoded["half_width_percent_milli"] == 4_500
    assert decoded["adjacent_rate_step_percent_milli"] == -1_200


def test_decode_breakaway_rung_zero_diagnostic_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_rung_zero_diagnostic()
    assert len(payload) == 17
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_rung_zero_diagnostic"
    assert decoded["forward_moved"] == 1
    assert decoded["reverse_moved"] == 0


def test_decode_breakaway_confirmation_planned_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_confirmation_planned()
    assert len(payload) == 37
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_confirmation_planned"
    assert decoded["plan_digest"] == _BW_PLAN_DIGEST
    assert decoded["prior_plan_digest"] == _BW_PRIOR_PLAN_DIGEST
    assert decoded["family_size"] == 8
    assert decoded["observations_per_direction"] == 4
    assert decoded["candidate_p_raw"] == 0x0910
    assert decoded["band_lower_percent"] == 70
    assert decoded["band_upper_percent"] == 80
    assert decoded["capture_profile"] == 0
    assert decoded["acceptance_rule"] == 0
    assert decoded["nominated_margin_percent_milli"] == 1_500


def test_decode_breakaway_confirmation_terminal_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_confirmation_terminal()
    assert len(payload) == 40
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_confirmation_terminal"
    assert decoded["terminal_cause"] == 21
    assert decoded["forward_collected_mask"] == 0b1111
    assert decoded["forward_eligible_mask"] == 0b1111
    assert decoded["forward_included_mask"] == 0b1111
    assert decoded["reverse_collected_mask"] == 0b1111
    assert decoded["reverse_eligible_mask"] == 0b1111
    assert decoded["reverse_included_mask"] == 0b1111
    assert decoded["accepted"] == 1
    assert decoded["confirmed_p_raw"] == 0x0910
    assert decoded["max_relative_se_permille"] == 20
    assert decoded["required_relative_se_permille"] == 67
    assert decoded["has_safety_fault"] == 0


def test_decode_breakaway_confirmation_observation_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_confirmation_observation()
    assert len(payload) == 27
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_confirmation_observation"
    assert decoded["slot_index"] == 3
    assert decoded["direction"] == 1
    assert decoded["response_lower_percent_milli"] == 70_800
    assert decoded["response_upper_percent_milli"] == 79_200
    assert decoded["relative_standard_error_permille"] == 18


def test_decode_breakaway_campaign_terminal_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_campaign_terminal()
    assert len(payload) == 18
    decoded = decode_breakaway_record(payload)
    assert decoded["record_kind"] == "breakaway_campaign_terminal"
    assert decoded["phase"] == 2
    assert decoded["terminal_cause"] == 21
    assert decoded["accepted"] == 1
    assert decoded["stage_c_plan_digest"] == _BW_PLAN_DIGEST
    # A pre-error_code-field (18-byte) payload still decodes at the current
    # protocol version -- "no error" and "field absent" stay distinct from 0.
    assert decoded["error_code"] is None
    assert decoded["error_code_name"] is None


def test_breakaway_campaign_terminal_names_the_error_behind_a_confirmed_refusal() -> None:
    """A confirmed acceptance whose Stage-C plan build was refused must say why,
    not report a self-contradictory Accepted cause with accepted=0.
    """
    from foci_trace.protocol import decode_breakaway_record

    base = bytes([12]) + struct.pack(
        "<IHBBBQ", _BW_RUN_SEQUENCE, _BW_EVIDENCE_SEQUENCE, 2, 26, 0, 0
    )
    payload = base + bytes([11])
    assert len(payload) == 19

    decoded = decode_breakaway_record(payload, protocol_version=19)
    assert decoded["record_kind"] == "breakaway_campaign_terminal"
    assert decoded["terminal_cause_name"] == "confirmation_stage_c_plan_refused"
    assert decoded["accepted"] == 0
    assert decoded["error_code"] == 11
    assert decoded["error_code_name"] == "velocity_validation_failed"


def test_breakaway_campaign_terminal_cause_26_rejected_below_protocol_19() -> None:
    from foci_trace.protocol import decode_breakaway_record

    base = bytes([12]) + struct.pack(
        "<IHBBBQ", _BW_RUN_SEQUENCE, _BW_EVIDENCE_SEQUENCE, 2, 26, 0, 0
    )
    payload = base + bytes([11])

    with pytest.raises(DecodeError, match="cause is invalid"):
        decode_breakaway_record(payload, protocol_version=18)


def test_breakaway_event_ids_match_the_firmware_trace_events_table() -> None:
    from foci_trace.protocol import (
        BREAKAWAY_CAMPAIGN_EVENT_IDS,
        EVENT_COMMISSION_BREAKAWAY_CAMPAIGN_TERMINAL,
        EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_PLAN,
        EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_TERMINAL,
        EVENT_COMMISSION_BREAKAWAY_DISCOVERY_MARGIN,
        EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN,
        EVENT_COMMISSION_BREAKAWAY_DISCOVERY_TERMINAL,
        EVENT_COMMISSION_BREAKAWAY_PROBE_DECISION,
        EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN,
        EVENT_COMMISSION_BREAKAWAY_PROBE_RESULT,
        EVENT_COMMISSION_BREAKAWAY_PROBE_TERMINAL,
        EVENT_COMMISSION_BREAKAWAY_RAW_OBSERVATION,
        EVENT_COMMISSION_BREAKAWAY_RECOVERY_ORIGIN_MISMATCH,
        EVENT_COMMISSION_BREAKAWAY_RECOVERY_REST_REJECTED,
        EVENT_COMMISSION_REST_SELECTION_DIAGNOSTIC,
    )

    assert EVENT_COMMISSION_REST_SELECTION_DIAGNOSTIC == 151
    assert EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN == 152
    assert EVENT_COMMISSION_BREAKAWAY_PROBE_RESULT == 153
    assert EVENT_COMMISSION_BREAKAWAY_PROBE_TERMINAL == 154
    assert EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN == 155
    assert EVENT_COMMISSION_BREAKAWAY_DISCOVERY_TERMINAL == 156
    assert EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_PLAN == 157
    assert EVENT_COMMISSION_BREAKAWAY_CONFIRMATION_TERMINAL == 158
    assert EVENT_COMMISSION_BREAKAWAY_CAMPAIGN_TERMINAL == 159
    assert EVENT_COMMISSION_BREAKAWAY_RAW_OBSERVATION == 160
    assert EVENT_COMMISSION_BREAKAWAY_DISCOVERY_MARGIN == 161
    assert EVENT_COMMISSION_BREAKAWAY_PROBE_DECISION == 162
    assert EVENT_COMMISSION_BREAKAWAY_RECOVERY_REST_REJECTED == 163
    assert EVENT_COMMISSION_BREAKAWAY_RECOVERY_ORIGIN_MISMATCH == 164
    assert BREAKAWAY_CAMPAIGN_EVENT_IDS == (
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


def test_decode_breakaway_record_rejects_an_unknown_tag() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="unknown breakaway evidence tag"):
        decode_breakaway_record(bytes([14, 0, 0, 0, 0]))


def test_decode_breakaway_record_rejects_a_length_mismatch_for_a_known_tag() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="unknown breakaway evidence tag"):
        decode_breakaway_record(_bw_probe_planned() + b"\x00")


def test_the_only_v14_breakaway_tag_that_collides_with_a_legacy_wire_shape_is_three() -> None:
    """`decode_frame`'s generic decoder recovers every breakaway tag except two.

    Tags 0-2 and 4-13 fail every existing `(kind, length)` branch in
    `_decode_velocity_sweep_evidence` and raise, so `_decode_payload` falls
    back to `decode_breakaway_record` and recovers the correct interpretation
    with no extra context needed. Two tags genuinely collide with an
    unrelated legacy wire shape that shares the same leading tag byte and
    exact payload length, so the generic decoder "succeeds" with the wrong
    interpretation:

    - The v14 tag 1 (`directional_breakaway`, 23 bytes) no longer collides with
      the legacy 22-byte `integral_plan_marker`; the historical v13 shape does.
    - Tag 3 (`discovery_planned`) widened from 43 to 45 bytes when
      `schema_revision` was added, and 45 bytes also happens to be the
      legacy non-compact Stage-B `rung` record's exact length.

    Resolving either case needs the marker-adjacency context
    `assemble_breakaway_ticks` provides -- this test pins both collisions so
    a future change to any of the three wire shapes doesn't silently make it
    worse or silently fix it unnoticed.
    """
    for builder in (
        _bw_probe_planned,
        _bw_probe_terminal,
        _bw_ceiling_source,
        _bw_raw_observation,
        _bw_discovery_terminal,
        _bw_discovery_rung_margin,
        _bw_rung_zero_diagnostic,
        _bw_confirmation_planned,
        _bw_confirmation_terminal,
        _bw_confirmation_observation,
        _bw_campaign_terminal,
        _bw_probe_decision,
    ):
        payload = builder()
        frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)
        assert str(frame.payload["record_kind"]).startswith("breakaway_")

    v13_directional_payload = _bw_directional_breakaway_v13()
    v13_directional_frame = decode_frame(
        _header(
            FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
            v13_directional_payload,
            version=13,
        )
        + v13_directional_payload
    )
    assert v13_directional_frame.payload["record_kind"] == "integral_plan_marker"

    collided_discovery_payload = _bw_discovery_planned()
    collided_discovery_frame = decode_frame(
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, collided_discovery_payload)
        + collided_discovery_payload
    )
    assert collided_discovery_frame.payload["record_kind"] == "rung"


def test_assemble_breakaway_ticks_bundles_records_by_marker() -> None:
    from foci_trace.protocol import (
        EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN,
        EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN,
        FRAME_TYPE_EVENT,
        FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
        assemble_breakaway_ticks,
    )

    def event_payload(event_id: int, sequence: int) -> dict[str, object]:
        return {"event_id": event_id, "args": [0, 1, sequence, 0]}

    frames = [
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 100, _bw_probe_planned(), None),
        (FRAME_TYPE_EVENT, 101, b"", event_payload(EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN, 0x5060)),
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 200, _bw_discovery_planned(), None),
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 200, _bw_ceiling_source(), None),
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 200, _bw_raw_observation(), None),
        (
            FRAME_TYPE_EVENT,
            201,
            b"",
            event_payload(EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN, 0x6070),
        ),
    ]
    ticks = assemble_breakaway_ticks(frames)
    assert len(ticks) == 2
    assert ticks[0]["event_id"] == EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN
    assert ticks[0]["sequence"] == 0x5060
    assert [decoded["record_kind"] for _raw, decoded in ticks[0]["records"]] == [
        "breakaway_probe_planned"
    ]
    assert ticks[1]["event_id"] == EVENT_COMMISSION_BREAKAWAY_DISCOVERY_PLAN
    assert [decoded["record_kind"] for _raw, decoded in ticks[1]["records"]] == [
        "breakaway_discovery_planned",
        "breakaway_ceiling_source",
        "breakaway_raw_observation",
    ]


def test_assemble_breakaway_ticks_discards_evidence_claimed_by_a_foreign_marker() -> None:
    """A non-breakaway marker still claims and clears whatever preceded it.

    An earlier commissioning phase's own evidence must not leak into the
    first breakaway tick just because that phase's own marker event id isn't
    tracked here.
    """
    from foci_trace.protocol import (
        EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN,
        EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL,
        FRAME_TYPE_EVENT,
        FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
        assemble_breakaway_ticks,
    )

    def event_payload(event_id: int, sequence: int) -> dict[str, object]:
        return {"event_id": event_id, "args": [0, 1, sequence, 0]}

    unrelated_evidence = bytes([4]) + struct.pack("<III", 1, 2, 3)
    frames = [
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 50, unrelated_evidence, None),
        (
            FRAME_TYPE_EVENT,
            51,
            b"",
            event_payload(EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL, 1),
        ),
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 100, _bw_probe_planned(), None),
        (
            FRAME_TYPE_EVENT,
            101,
            b"",
            event_payload(EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN, 0x5060),
        ),
    ]
    ticks = assemble_breakaway_ticks(frames)
    assert len(ticks) == 1
    assert [decoded["record_kind"] for _raw, decoded in ticks[0]["records"]] == [
        "breakaway_probe_planned"
    ]


def test_assemble_breakaway_ticks_raises_on_a_frame_that_does_not_decode() -> None:
    from foci_trace.protocol import (
        EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN,
        FRAME_TYPE_EVENT,
        FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
        assemble_breakaway_ticks,
    )

    frames = [
        (FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, 100, b"\xff\x00\x00", None),
        (
            FRAME_TYPE_EVENT,
            101,
            b"",
            {"event_id": EVENT_COMMISSION_BREAKAWAY_PROBE_PLAN, "args": [0, 1, 0x5060, 0]},
        ),
    ]
    with pytest.raises(DecodeError):
        assemble_breakaway_ticks(frames)


def _bw_rest_rejected(
    *,
    structural_codes: tuple[int, int, int] = (0, 3, 2),
    interval_present_mask: int = 0b111,
    usable_mask: int = 0b001,
    zero_compatible_mask: int = 0b000,
    selected_mask: int = 0,
    selected_suffix_len: int = 0,
    disposition: int = 2,
    intervals: tuple[tuple[int, int], ...] = (
        (0x0102030405060708, 0x1112131415161718),
        (0x2122232425262728, 0x3132333435363738),
        (-0x5152535455565758, -0x4142434445464748),
    ),
    scored: tuple[int, int, int] = (-2, -2, 1),
) -> bytes:
    """Assemble the firmware's tag-14 golden bytes, field by field.

    Mirrors `encode_recovery_rest_rejected`'s push order exactly, so a firmware
    field reorder or width change breaks these tests rather than silently
    shifting every field after it.
    """
    payload = bytearray([14])
    payload += struct.pack("<I", _BW_RUN_SEQUENCE)
    payload += struct.pack("<H", _BW_EVIDENCE_SEQUENCE)
    payload += struct.pack("<Q", _BW_PLAN_DIGEST)
    payload += bytes([1, 2, 0x0F, 0x03, 1])
    payload += struct.pack("<H", 181)
    payload += bytes(structural_codes)
    payload += bytes([interval_present_mask, usable_mask, zero_compatible_mask, selected_mask])
    payload += struct.pack("<H", selected_suffix_len)
    payload += bytes([disposition])
    for low, high in intervals:
        payload += struct.pack("<qq", low, high)
    payload += struct.pack("<bbb", *scored)
    return bytes(payload)


def test_decode_breakaway_rest_rejected_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_rest_rejected()
    assert len(payload) == 83
    decoded = decode_breakaway_record(payload, protocol_version=18)
    assert decoded["record_kind"] == "breakaway_recovery_rest_rejected"
    assert decoded["tag"] == 14
    assert decoded["phase"] == 1
    assert decoded["stage"] == 2
    assert decoded["rung_index"] == 15
    assert decoded["slot_index"] == 3
    assert decoded["direction"] == 1
    assert decoded["p_raw"] == 181
    assert decoded["structural_codes"] == [0, 3, 2]
    assert decoded["scored_min_displacement_counts"] == -2
    assert decoded["scored_max_displacement_counts"] == 1


def test_decode_breakaway_rest_rejected_derives_the_fallback_availability() -> None:
    """The one-count fallback is offered only for a span of at most one count.

    This derived pair is the whole point of the record: it separates a rest
    rejection that had a safety net from one that never had one.
    """
    from foci_trace.protocol import decode_breakaway_record

    wide = decode_breakaway_record(_bw_rest_rejected(scored=(-2, -2, 1)), protocol_version=18)
    assert wide["scored_displacement_span_counts"] == 3
    assert wide["one_count_fallback_available"] is False

    narrow = decode_breakaway_record(_bw_rest_rejected(scored=(0, 0, 1)), protocol_version=18)
    assert narrow["scored_displacement_span_counts"] == 1
    assert narrow["one_count_fallback_available"] is True


def test_decode_breakaway_rest_rejected_requires_protocol_18() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="unknown breakaway evidence tag"):
        decode_breakaway_record(_bw_rest_rejected(), protocol_version=17)


def test_decode_breakaway_rest_rejected_rejects_an_accepted_disposition() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="non-rejected disposition"):
        decode_breakaway_record(_bw_rest_rejected(disposition=0), protocol_version=18)


def test_decode_breakaway_rest_rejected_rejects_a_selected_suffix() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="names a selected suffix"):
        decode_breakaway_record(
            _bw_rest_rejected(selected_mask=0b001, selected_suffix_len=64),
            protocol_version=18,
        )


def test_decode_breakaway_rest_rejected_rejects_a_usable_mask_contradiction() -> None:
    """`usable_mask` is a pure function of the structural codes."""
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="usable mask contradicts its codes"):
        decode_breakaway_record(_bw_rest_rejected(structural_codes=(3, 3, 2)), protocol_version=18)


def test_decode_breakaway_rest_rejected_rejects_a_missed_zero_compatibility() -> None:
    """A usable suffix whose interval spans zero must be marked compatible."""
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="zero-compatibility is inconsistent"):
        decode_breakaway_record(
            _bw_rest_rejected(
                intervals=(
                    (-0x0102030405060708, 0x1112131415161718),
                    (0x2122232425262728, 0x3132333435363738),
                    (-0x5152535455565758, -0x4142434445464748),
                )
            ),
            protocol_version=18,
        )


def test_decode_breakaway_rest_rejected_rejects_displacement_outside_the_clamp() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="escapes its clamp"):
        decode_breakaway_record(_bw_rest_rejected(scored=(-3, -3, 1)), protocol_version=18)


def test_decode_breakaway_rest_rejected_rejects_inconsistent_displacement() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="scored displacement is inconsistent"):
        decode_breakaway_record(_bw_rest_rejected(scored=(2, -2, 1)), protocol_version=18)


def _bw_origin_mismatch(
    *,
    phase: int = 1,
    direction: int = 1,
    expected: int = -1481,
    observed: int = -1480,
    delta: int | None = None,
    window: int = 4000,
) -> bytes:
    """Assemble the firmware's tag-15 golden bytes, field by field."""
    if delta is None:
        delta = observed - expected
    payload = bytearray([15])
    payload += struct.pack("<I", _BW_RUN_SEQUENCE)
    payload += struct.pack("<H", _BW_EVIDENCE_SEQUENCE)
    payload += struct.pack("<Q", _BW_PLAN_DIGEST)
    payload += bytes([phase, 0, 7, 2, direction])
    payload += struct.pack("<H", 181)
    payload += struct.pack("<iiiI", expected, observed, delta, window)
    return bytes(payload)


def test_decode_breakaway_origin_mismatch_matches_the_firmware_golden_bytes() -> None:
    from foci_trace.protocol import decode_breakaway_record

    payload = _bw_origin_mismatch()
    assert len(payload) == 38
    decoded = decode_breakaway_record(payload, protocol_version=18)
    assert decoded["record_kind"] == "breakaway_recovery_origin_mismatch"
    assert decoded["tag"] == 15
    assert decoded["expected_start_offset_counts"] == -1481
    assert decoded["observed_position_counts"] == -1480
    assert decoded["delta_counts"] == 1
    assert decoded["position_window_counts"] == 4000


def test_decode_breakaway_origin_mismatch_flags_a_single_sample_race() -> None:
    """One count of discrepancy is one encoder sample, not a runaway."""
    from foci_trace.protocol import decode_breakaway_record

    race = decode_breakaway_record(_bw_origin_mismatch(), protocol_version=18)
    assert race["delta_abs_counts"] == 1
    assert race["single_sample_race"] is True

    runaway = decode_breakaway_record(_bw_origin_mismatch(observed=-1431), protocol_version=18)
    assert runaway["delta_abs_counts"] == 50
    assert runaway["single_sample_race"] is False


def test_decode_breakaway_origin_mismatch_requires_protocol_18() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="unknown breakaway evidence tag"):
        decode_breakaway_record(_bw_origin_mismatch(), protocol_version=17)


def test_decode_breakaway_origin_mismatch_rejects_a_contradictory_delta() -> None:
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="delta contradicts its offsets"):
        decode_breakaway_record(_bw_origin_mismatch(delta=7), protocol_version=18)


def test_decode_breakaway_origin_mismatch_rejects_a_zero_discrepancy() -> None:
    """The fault fires on exact inequality, so a zero delta is impossible."""
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="reports no discrepancy"):
        decode_breakaway_record(_bw_origin_mismatch(observed=-1481), protocol_version=18)


def test_decode_breakaway_origin_mismatch_rejects_an_out_of_window_start() -> None:
    """An out-of-window start offset is contained earlier, never faulted here."""
    from foci_trace.protocol import decode_breakaway_record

    with pytest.raises(DecodeError, match="start offset exceeds its window"):
        decode_breakaway_record(
            _bw_origin_mismatch(expected=-9000, observed=-8999, window=4000),
            protocol_version=18,
        )


def test_take_build_identity_reads_legacy_and_new_layouts() -> None:
    from foci_trace.protocol import _EvidenceCursor, _take_build_identity

    # A u32 equal to a legacy build_revision constant (FNV of the crate version)
    # decodes as the old single-field layout.
    legacy = struct.pack("<I", 0x05E21F75) + b"\xaa\xbb"
    cursor = _EvidenceCursor(legacy)
    assert _take_build_identity(cursor) == {"build_revision": 0x05E21F75}
    assert cursor.offset == 4

    # Any other u32 is a new-layout record: build_features (u16) then
    # build_identity (u32).
    new = struct.pack("<HI", 0x000F, 0xABCD1234)
    cursor = _EvidenceCursor(new)
    assert _take_build_identity(cursor) == {
        "build_features": 0x000F,
        "build_identity": 0xABCD1234,
    }
    assert cursor.offset == 6
