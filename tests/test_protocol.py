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


def test_decodes_stage_b_reproduction_v3_fragments() -> None:
    fragments = [
        (
            struct.pack("<BIHBBHHHBH", 10, 7, 33, 1, 0, 1448, 1024, 1448, 0, 3),
            "reproduction_v3_header",
        ),
        (
            struct.pack("<BIHBIIIIIbb", 11, 7, 33, 0, 0x38, 0x3C, 0x38, 0, 4, -1, 0),
            "reproduction_v3_membership",
        ),
        (
            struct.pack("<BIHBiiiiii", 14, 7, 33, 0, 100, 120, 105, 125, 105, 120),
            "reproduction_v3_pooled",
        ),
        (
            struct.pack("<BIHBiiiiiiB", 16, 7, 33, 0, 104, 115, 108, 118, 108, 115, 1),
            "reproduction_v3_common",
        ),
        (
            struct.pack("<BIHBBiHH", 18, 7, 33, 0, 1, -2, 1448, 724),
            "reproduction_v3_coverage",
        ),
        (
            struct.pack("<BIHIIII", 20, 7, 33, 1, 2, 3, 4),
            "reproduction_v3_digest",
        ),
    ]

    decoded = [
        decode_frame(_header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, payload) + payload).payload
        for payload, _kind in fragments
    ]

    assert [value["record_kind"] for value in decoded] == [kind for _payload, kind in fragments]
    assert decoded[0]["schema_revision"] == 3
    assert decoded[1]["low_delta"] == -1
    assert decoded[2]["overlap_high"] == 120
    assert decoded[3]["nonempty"] == 1
    assert decoded[4]["signed_rung_distance"] == -2
    assert decoded[5]["current_digest_high"] == 4


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
