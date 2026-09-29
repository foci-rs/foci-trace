import struct

from foci_trace.protocol import FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, PROTOCOL_VERSION, decode_frame
from foci_trace.velocity_integral import derive_integral_ladder, integral_plan_digest


def _header(
    frame_type: int, payload: bytes, seq: int = 1, version: int = PROTOCOL_VERSION
) -> bytes:
    return struct.pack(
        "<IBBBBHBBIQII",
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


_SCHEMA_17_CORE_FIXTURE = bytes.fromhex(
    "0001000000000081b06087b29443e6000f0067673994110000cd5497e14636"
    "c1b60002000007000000070000000700000007000303004066030180a8120"
    "14015defe8057edfe2a11000008010000a8610000a00f00000d3c000f7800"
    "000be609001e05"
)
_SCHEMA_17_SCHEDULE_FIXTURE = bytes.fromhex(
    "0001000000000081b06087b29443e6010d0100020003000400080010002000"
    "4000800000010002d402000400d0070020a10700e099030057b60200a8fb02"
    "000a1a00001027000001c4090000fa000000711b0000711b0000b80b"
)
_SCHEMA_17_RECOVERY_FIXTURE = bytes.fromhex(
    "0001000000000081b06087b29443e602e80300007eb91b00c0c62d000f00"
)


def test_schema_seventeen_firmware_encoding_replays_to_the_same_ladder_and_digest() -> None:
    core = decode_frame(
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, _SCHEMA_17_CORE_FIXTURE)
        + _SCHEMA_17_CORE_FIXTURE
    ).payload
    schedule = decode_frame(
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, _SCHEMA_17_SCHEDULE_FIXTURE)
        + _SCHEMA_17_SCHEDULE_FIXTURE
    ).payload
    recovery = decode_frame(
        _header(FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, _SCHEMA_17_RECOVERY_FIXTURE)
        + _SCHEMA_17_RECOVERY_FIXTURE
    ).payload

    assert core["schema_revision"] == 17
    assert core["i_top_cap"] == 1_310

    for shared_key in ("run_sequence", "evidence_sequence", "plan_digest"):
        assert core[shared_key] == schedule[shared_key] == recovery[shared_key]
    assert core["positive_rung_count"] == schedule["positive_rung_count"]

    decoded_plan = {**core, **schedule, **recovery}
    assert decoded_plan["positive_i"] == [
        1,
        2,
        3,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        724,
        1_024,
    ]

    ladder = derive_integral_ladder(
        pwm_hz=decoded_plan["pwm_hz"],
        moving_stroke_us=decoded_plan["moving_stroke_us"],
        shortest_suffix_us=64_000,
        maximum_i=8_192,
        schema_revision=17,
        build_features=decoded_plan["build_features"],
    )
    assert ladder["positive_i"] == decoded_plan["positive_i"]

    assert integral_plan_digest(decoded_plan) == core["plan_digest"]
