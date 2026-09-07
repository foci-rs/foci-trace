import struct

from foci_trace.commissioning import (
    CommissioningCaptureBuilder,
    commissioning_report_to_json,
    format_commissioning_report,
)
from foci_trace.protocol import (
    EVENT_COMMISSION_ENGINE_START,
    EVENT_COMMISSION_ENGINE_TERMINAL,
    EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT,
    EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION,
    EVENT_COMMISSION_MECHANICAL_ID_RESPONSE,
    EVENT_COMMISSION_MECHANICAL_ID_RESULT,
    EVENT_COMMISSION_PHASE_END,
    EVENT_COMMISSION_PHASE_START,
    EVENT_COMMISSION_VELOCITY_SWEEP_PLAN,
    EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL,
    EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN,
    EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS,
    EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS,
    EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT,
    FRAME_TYPE_CONTROL_DETAIL,
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_EVENT,
    PROTOCOL_VERSION,
    SUBSYSTEM_COMMISSIONING,
    decode_frame,
)

CLOCK_HZ = 1_000_000
SOURCE_KIND_FOCI = 1
SOURCE_KIND_BOARD = 0
ENGINE_COMMISSION = 1
ENGINE_OUTER_TUNE = 4
PHASE_INNER_DONE = 8
PHASE_MECHANICAL_ID = 9
PHASE_BREAKAWAY_PROPORTIONAL_ACQUISITION = 18
PHASE_CLOSED_LOOP_ENTRY = 17
PHASE_VELOCITY_VALIDATE = 11
STATUS_ACCEPTED = 0
STATUS_MECHANICAL_ID_FAILED = 10
STATUS_VELOCITY_VALIDATION_FAILED = 11


def _header(
    frame_type: int,
    payload: bytes,
    *,
    seq: int,
    timestamp_ticks: int,
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
        timestamp_ticks,
        1234,
        0,
    )


def _event_payload(
    event_id: int,
    *,
    level: int = 1,
    source_index: int = 0,
    engine_kind: int = ENGINE_COMMISSION,
    arg2: int = 0,
    arg3: int = 0,
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


def _critical_payload(step_position: int) -> bytes:
    payload = struct.pack(
        "<iiiiiiiBBHIiQQIIIII",
        step_position,
        step_position - 1,
        1,
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
        0x0006_0005,
        0x0008_0007,
        0x000A_0009,
    )
    payload += struct.pack("<iHBB", 0, 0, 0, 0)
    payload += struct.pack("<i", 0)
    return payload


def _control_detail_payload() -> bytes:
    return struct.pack("<IIIIIII", 100, 0, 0, 0, 29_000, 2816, 20_000)


def _observe_event(
    builder: CommissioningCaptureBuilder,
    seq: int,
    timestamp_ticks: int,
    event_id: int,
    *,
    engine_kind: int = ENGINE_COMMISSION,
    arg2: int = 0,
    arg3: int = 0,
    level: int = 1,
) -> None:
    payload = _event_payload(
        event_id,
        level=level,
        engine_kind=engine_kind,
        arg2=arg2,
        arg3=arg3,
    )
    frame = decode_frame(
        _header(
            FRAME_TYPE_EVENT,
            payload,
            seq=seq,
            timestamp_ticks=timestamp_ticks,
            source_kind=SOURCE_KIND_BOARD,
            source_index=255,
        )
        + payload
    )
    builder.observe_event(0, frame.header, frame.payload)


def _observe_critical(
    builder: CommissioningCaptureBuilder,
    seq: int,
    timestamp_ticks: int,
    step_position: int,
) -> None:
    payload = _critical_payload(step_position)
    frame = decode_frame(
        _header(
            FRAME_TYPE_CRITICAL,
            payload,
            seq=seq,
            timestamp_ticks=timestamp_ticks,
            source_kind=SOURCE_KIND_FOCI,
            source_index=0,
        )
        + payload
    )
    builder.observe_critical(0, 0, frame.header.timestamp_ticks, frame.payload)


def _observe_control_detail(
    builder: CommissioningCaptureBuilder,
    seq: int,
    timestamp_ticks: int,
) -> None:
    payload = _control_detail_payload()
    frame = decode_frame(
        _header(
            FRAME_TYPE_CONTROL_DETAIL,
            payload,
            seq=seq,
            timestamp_ticks=timestamp_ticks,
            source_kind=SOURCE_KIND_FOCI,
            source_index=0,
        )
        + payload
    )
    builder.observe_control_detail(0, 0, frame.header.timestamp_ticks, frame.payload)


def test_successful_commission_attaches_closed_loop_samples() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(builder, 1, 1_000, EVENT_COMMISSION_ENGINE_START, arg2=1, arg3=7)
    _observe_event(builder, 2, 10_000, EVENT_COMMISSION_PHASE_START, arg2=17, arg3=3)
    for idx in range(60):
        _observe_critical(builder, 10 + idx, 10_000 + idx * 1_000, idx)
    for idx in range(6):
        _observe_control_detail(builder, 100 + idx, 10_000 + idx * 10_000)
    _observe_event(builder, 3, 80_000, EVENT_COMMISSION_PHASE_END, arg2=17, arg3=0)
    _observe_event(builder, 4, 81_000, EVENT_COMMISSION_ENGINE_TERMINAL, arg2=0, arg3=7)

    report = builder.finalize({0: CLOCK_HZ})
    text = format_commissioning_report(report)
    data = commissioning_report_to_json(report)

    assert "engine_start" in text
    assert "phase_start" in text
    assert "phase_end" in text
    assert "engine_terminal" in text
    assert "phase ClosedLoopActivation" in text
    assert "samples=60" in text
    assert "result=accepted" in text
    assert data["runs"][0]["result"] == "accepted"
    assert data["runs"][0]["phases"][0]["critical_samples"] == 60
    assert data["runs"][0]["phases"][0]["control_detail_samples"] == 6


def test_failed_tune_decodes_velocity_validate_summary() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(
        builder,
        1,
        1_000,
        EVENT_COMMISSION_ENGINE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=1,
        arg3=9,
    )
    _observe_event(
        builder,
        2,
        10_000,
        EVENT_COMMISSION_PHASE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=PHASE_VELOCITY_VALIDATE,
        arg3=1,
    )
    for event_id, value in (
        (EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT, STATUS_VELOCITY_VALIDATION_FAILED),
        (EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN, 1234),
        (EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS, 5678),
        (EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS, 9012),
    ):
        _observe_event(
            builder,
            10 + event_id,
            20_000 + event_id,
            event_id,
            engine_kind=ENGINE_OUTER_TUNE,
            arg2=2,
            arg3=value,
        )
    _observe_event(
        builder,
        3,
        80_000,
        EVENT_COMMISSION_PHASE_END,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=PHASE_VELOCITY_VALIDATE,
        arg3=STATUS_VELOCITY_VALIDATION_FAILED,
    )
    _observe_event(
        builder,
        4,
        81_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=STATUS_VELOCITY_VALIDATION_FAILED,
        arg3=9,
    )

    report = builder.finalize({0: CLOCK_HZ})
    text = format_commissioning_report(report)
    data = commissioning_report_to_json(report)

    assert "engine=outer_tune" in text
    assert "result=velocity_validation_failed" in text
    assert "velocity_validate_summary" in text
    assert data["runs"][0]["result"] == "velocity_validation_failed"
    assert data["runs"][0]["summaries"][0]["family"] == "velocity_validate_summary"
    assert len(data["runs"][0]["summaries"][0]["fragments"]) == 4


def test_velocity_sweep_sequence_marker_is_retained() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(
        builder,
        1,
        1_000,
        EVENT_COMMISSION_ENGINE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg3=9,
    )
    _observe_event(
        builder,
        2,
        2_000,
        EVENT_COMMISSION_VELOCITY_SWEEP_PLAN,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=17,
    )

    data = commissioning_report_to_json(builder.finalize({0: CLOCK_HZ}))

    assert data["runs"][0]["velocity_sweep_markers"] == [
        {
            "kind": "plan",
            "sequence_or_cause": 17,
            "detail": 0,
            "timestamp_ticks": 2_000,
        }
    ]


def test_velocity_sweep_inconclusive_owns_terminal_result() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(
        builder,
        1,
        1_000,
        EVENT_COMMISSION_ENGINE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg3=9,
    )
    _observe_event(
        builder,
        2,
        2_000,
        EVENT_COMMISSION_PHASE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=10,
        arg3=1,
    )
    _observe_event(
        builder,
        3,
        3_000,
        EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=4,
        arg3=0x1234,
        level=2,
    )
    _observe_event(
        builder,
        4,
        4_000,
        EVENT_COMMISSION_PHASE_END,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=10,
        arg3=4,
        level=2,
    )
    _observe_event(
        builder,
        5,
        5_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=4,
        arg3=9,
        level=2,
    )

    data = commissioning_report_to_json(builder.finalize({0: CLOCK_HZ}))
    run = data["runs"][0]

    assert run["result"] == "inconclusive"
    assert run["result_code"] is None
    assert run["terminal_ticks"] == 3_000
    assert any(
        "dedicated velocity-sweep terminal is authoritative" in caveat for caveat in run["caveats"]
    )
    assert run["phases"][0]["status"] == "inconclusive"


def test_velocity_sweep_fault_owns_terminal_result() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(
        builder,
        1,
        1_000,
        EVENT_COMMISSION_ENGINE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg3=9,
    )
    _observe_event(
        builder,
        2,
        2_000,
        EVENT_COMMISSION_PHASE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=10,
        arg3=1,
    )
    _observe_event(
        builder,
        3,
        3_000,
        EVENT_COMMISSION_VELOCITY_SWEEP_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=3,
        arg3=0x1234,
        level=3,
    )
    _observe_event(
        builder,
        4,
        4_000,
        EVENT_COMMISSION_PHASE_END,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=10,
        arg3=3,
        level=3,
    )
    _observe_event(
        builder,
        5,
        5_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=3,
        arg3=9,
        level=3,
    )

    data = commissioning_report_to_json(builder.finalize({0: CLOCK_HZ}))
    run = data["runs"][0]

    assert run["result"] == "fault"
    assert run["result_code"] is None
    assert run["terminal_ticks"] == 3_000
    assert any(
        "dedicated velocity-sweep terminal is authoritative" in caveat for caveat in run["caveats"]
    )
    assert run["phases"][0]["status"] == "fault"


def test_partial_capture_reports_caveat_without_crashing() -> None:
    builder = CommissioningCaptureBuilder()
    _observe_event(builder, 1, 1_000, EVENT_COMMISSION_ENGINE_START, arg2=1, arg3=4)
    _observe_event(
        builder,
        2,
        10_000,
        EVENT_COMMISSION_PHASE_START,
        arg2=PHASE_CLOSED_LOOP_ENTRY,
        arg3=1,
    )
    for idx in range(3):
        _observe_critical(builder, 10 + idx, 10_000 + idx * 1_000, idx)

    report = builder.finalize({0: CLOCK_HZ})
    text = format_commissioning_report(report)
    data = commissioning_report_to_json(report)

    assert "partial" in text
    assert "missing phase_end" in text
    assert data["runs"][0]["caveats"]
    assert data["runs"][0]["phases"][0]["critical_samples"] == 3


def test_historical_mechanical_id_events_still_decode_into_summary_fragments() -> None:
    """MechanicalId events 110-113 are retired from production firmware,
    but archived captures that contain them must still resolve
    into mechanical_id_summary fragments.

    The `CommissionPhase` wire code (9) is a separate hard break:
    `PHASE_NAMES` carries no historical-code entry for it, so the phase itself
    falls back to the generic `Phase9` label; only the mechanical_id_summary
    event decoding survives.
    """
    builder = CommissioningCaptureBuilder()
    _observe_event(builder, 1, 1_000, EVENT_COMMISSION_ENGINE_START, arg2=1, arg3=1)
    _observe_event(
        builder,
        2,
        2_000,
        EVENT_COMMISSION_PHASE_START,
        arg2=PHASE_MECHANICAL_ID,
        arg3=1,
    )
    for event_id, value in (
        (EVENT_COMMISSION_MECHANICAL_ID_RESULT, STATUS_MECHANICAL_ID_FAILED),
        (EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT, 1),
        (EVENT_COMMISSION_MECHANICAL_ID_RESPONSE, 2),
        (EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION, 3),
    ):
        _observe_event(
            builder,
            10 + event_id,
            3_000 + event_id,
            event_id,
            arg2=1,
            arg3=value,
        )
    _observe_event(
        builder,
        3,
        80_000,
        EVENT_COMMISSION_PHASE_END,
        arg2=PHASE_MECHANICAL_ID,
        arg3=STATUS_MECHANICAL_ID_FAILED,
    )
    _observe_event(
        builder,
        4,
        81_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        arg2=STATUS_MECHANICAL_ID_FAILED,
        arg3=1,
    )

    report = builder.finalize({0: CLOCK_HZ})
    text = format_commissioning_report(report)
    data = commissioning_report_to_json(report)
    run = data["runs"][0]

    assert "phase Phase9" in text
    assert "result=mechanical_id_failed" in text
    assert run["result"] == "mechanical_id_failed"
    assert run["summaries"][0]["family"] == "mechanical_id_summary"
    assert {fragment["fragment"] for fragment in run["summaries"][0]["fragments"]} == {
        "result",
        "attempt",
        "response",
        "classification",
    }


def test_new_run_progress_sequence_skips_mechanical_id_phase() -> None:
    """New captures never emit MechanicalId: the inner engine ends
    at InnerDone (phase 8) and a separate outer engine opens its own velocity
    workflow, BreakawayProportionalAcquisition (phase 18), with phase 9 never
    appearing and no mechanical_id_summary in the report.
    """
    builder = CommissioningCaptureBuilder()
    _observe_event(builder, 1, 1_000, EVENT_COMMISSION_ENGINE_START, arg2=1, arg3=1)
    _observe_event(
        builder,
        2,
        2_000,
        EVENT_COMMISSION_PHASE_START,
        arg2=PHASE_INNER_DONE,
        arg3=1,
    )
    _observe_event(
        builder,
        3,
        3_000,
        EVENT_COMMISSION_PHASE_END,
        arg2=PHASE_INNER_DONE,
        arg3=STATUS_ACCEPTED,
    )
    _observe_event(
        builder,
        4,
        4_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        arg2=STATUS_ACCEPTED,
        arg3=1,
    )
    _observe_event(
        builder,
        5,
        5_000,
        EVENT_COMMISSION_ENGINE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=1,
        arg3=2,
    )
    _observe_event(
        builder,
        6,
        6_000,
        EVENT_COMMISSION_PHASE_START,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=PHASE_BREAKAWAY_PROPORTIONAL_ACQUISITION,
        arg3=1,
    )
    _observe_event(
        builder,
        7,
        7_000,
        EVENT_COMMISSION_PHASE_END,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=PHASE_BREAKAWAY_PROPORTIONAL_ACQUISITION,
        arg3=STATUS_ACCEPTED,
    )
    _observe_event(
        builder,
        8,
        8_000,
        EVENT_COMMISSION_ENGINE_TERMINAL,
        engine_kind=ENGINE_OUTER_TUNE,
        arg2=STATUS_ACCEPTED,
        arg3=2,
    )

    report = builder.finalize({0: CLOCK_HZ})
    data = commissioning_report_to_json(report)

    phase_ids = [phase["phase_id"] for run in data["runs"] for phase in run["phases"]]
    assert phase_ids == [PHASE_INNER_DONE, PHASE_BREAKAWAY_PROPORTIONAL_ACQUISITION]
    assert PHASE_MECHANICAL_ID not in phase_ids
    assert not any(run["summaries"] for run in data["runs"])
    assert not report.caveats
    assert all(not run["caveats"] for run in data["runs"])
    assert all(not phase["caveats"] for run in data["runs"] for phase in run["phases"])
    assert all(run["result"] == "accepted" for run in data["runs"])
