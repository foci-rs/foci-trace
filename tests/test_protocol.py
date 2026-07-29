import struct

import pytest

from foci_trace.protocol import (
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_CURRENT_PATH,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_METADATA,
    FRAME_TYPE_PHASE_ADVANCE,
    FRAME_TYPE_VELOCITY_OBSERVATION,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    PROTOCOL_VERSION,
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

    assert PROTOCOL_VERSION == 9
    assert frame.payload["pid_velocity_actual"] is None
    assert frame.payload["actual_velocity_pptm"] == 176


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
    core.extend(struct.pack("<IHBIIH", 3, 1, 0, 0x99AA_BBCC, 0xDDEE_FF00, 1_448))
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
    core.extend(struct.pack("<IHBIIH", 3, 4, 0, 0x99AA_BBCC, 0xDDEE_FF00, 1_024))
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
            0x1122_3344,
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
        "intervals_q": [[-9, 0], [-17, 29], [31, 47]],
        "conditioned_rest_evidence": True,
    }


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

    frame = decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload)

    assert len(payload) == 80
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
