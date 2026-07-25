from foci_trace.protocol import (
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    Frame,
    FrameHeader,
)
from foci_trace.summary import LiveSummary


def _frame(sequence: int, position_error: int, velocity_error: int, limited: bool) -> Frame:
    return Frame(
        header=FrameHeader(1, 32, FRAME_TYPE_CRITICAL, 0, 84, 1, 0, sequence, sequence * 100, 1),
        payload={
            "position_error": position_error,
            "velocity_error": velocity_error,
            "interim_foc_uq_ud": 0x0002_0001,
            "interim_foc_uq_ud_limited": 0x0001_0001 if limited else 0x0002_0001,
        },
        raw_payload=b"",
    )


def test_summary_tracks_gaps_and_error_maxima() -> None:
    summary = LiveSummary()
    summary.observe(0, 1000, _frame(1, -3, 4, False))
    summary.observe(0, 2000, _frame(3, 7, -8, True))

    text = summary.render()

    assert "dev0 frames=2" in text
    assert "gaps=1" in text
    assert "pos_err=7" in text
    assert "vel_err=8" in text
    assert "voltage_limited=1" in text


def test_summary_tracks_stage_b_regions_nomination_and_terminal() -> None:
    summary = LiveSummary()
    payloads = [
        {"record_kind": "directional_region", "region_kind": 0},
        {"record_kind": "directional_region", "region_kind": 1},
        {"record_kind": "handoff", "nominated_p": 1448},
        {
            "record_kind": "stage_b_terminal",
            "outcome": 0,
            "cause": 4,
            "recovery_unavailable": 1,
        },
    ]
    for sequence, payload in enumerate(payloads, 1):
        summary.observe(
            0,
            sequence,
            Frame(
                header=FrameHeader(
                    9,
                    32,
                    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
                    0,
                    0,
                    1,
                    0,
                    sequence,
                    sequence,
                    1,
                ),
                payload=payload,
                raw_payload=b"",
            ),
        )

    text = summary.render()

    assert "stage_b_regions=1" in text
    assert "stage_b_fragments=1" in text
    assert "stage_b_p=1448" in text
    assert "stage_b_outcome=0" in text
    assert "stage_b_cause=4" in text
    assert "stage_b_recovery_unavailable=true" in text


def test_summary_tracks_velocity_integral_rungs_and_terminal() -> None:
    summary = LiveSummary()
    for sequence, payload in enumerate(
        (
            {"record_kind": "integral_rung"},
            {"record_kind": "integral_rung"},
            {
                "record_kind": "integral_rest_boundary",
                "positive_rung_index": 3,
                "slot": 5,
            },
            {"record_kind": "integral_terminal", "outcome": 1},
        ),
        1,
    ):
        summary.observe(
            0,
            sequence,
            Frame(
                header=FrameHeader(
                    9,
                    32,
                    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
                    0,
                    0,
                    1,
                    0,
                    sequence,
                    sequence,
                    1,
                ),
                payload=payload,
                raw_payload=b"",
            ),
        )

    text = summary.render()

    assert "velocity_integral_rungs=2" in text
    assert "velocity_integral_outcome=1" in text
    assert "velocity_integral_rest_boundary=3:5" in text


def test_summary_tracks_successful_and_contained_recoveries() -> None:
    summary = LiveSummary()
    for sequence, outcome in enumerate((0, 1, 2, 3), 1):
        summary.observe(
            0,
            sequence,
            Frame(
                header=FrameHeader(
                    9,
                    32,
                    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
                    0,
                    0,
                    1,
                    0,
                    sequence,
                    sequence,
                    1,
                ),
                payload={"record_kind": "rung_origin_recovery", "outcome": outcome},
                raw_payload=b"",
            ),
        )

    text = summary.render()

    assert "recoveries=4" in text
    assert "contained_recoveries=2" in text


def test_summary_keeps_best_effort_clip_uncertainty_visible() -> None:
    summary = LiveSummary()
    for sequence, payload in enumerate(
        (
            {
                "record_kind": "velocity_primitive_current",
                "contact": 0,
                "moving_clip_uncertain": True,
            },
            {
                "record_kind": "velocity_primitive_current",
                "contact": 2,
                "moving_clip_uncertain": True,
            },
        ),
        1,
    ):
        summary.observe(
            0,
            sequence,
            Frame(
                header=FrameHeader(
                    9,
                    32,
                    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
                    0,
                    0,
                    1,
                    0,
                    sequence,
                    sequence,
                    1,
                ),
                payload=payload,
                raw_payload=b"",
            ),
        )

    text = summary.render()

    assert "current_records=2" in text
    assert "current_contacts=1" in text
    assert "clip_uncertain=2" in text


def test_summary_surfaces_outer_safety_timing_margin() -> None:
    summary = LiveSummary()
    summary.observe(
        0,
        1,
        Frame(
            header=FrameHeader(
                9,
                32,
                FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
            ),
            payload={
                "record_kind": "outer_safety_timing",
                "max_realized_gap_us": 15_001,
            },
            raw_payload=b"",
        ),
    )

    text = summary.render()

    assert "outer_gap_max_us=15001" in text
    assert "outer_gap_headroom_us=4999" in text
    assert "outer_gap_reopen=true" in text
