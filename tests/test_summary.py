from foci_trace.protocol import Frame, FrameHeader, FRAME_TYPE_CRITICAL
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
