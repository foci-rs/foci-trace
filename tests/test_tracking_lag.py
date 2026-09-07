import pytest

from foci_trace import tracking_lag as tl
from foci_trace.protocol import FRAME_TYPE_CRITICAL, FRAME_TYPE_STATUS, Frame, FrameHeader

CLOCK = 84_000_000
TPM = CLOCK // 1000  # ticks per ms
PRODUCT = {0: "FOCI OpenFFBoard (trace)"}


class Stream:
    """Frames for one capture device with a contiguous global sequence."""

    def __init__(self) -> None:
        self.frames: list[tuple[int, Frame]] = []
        self.seq = 0

    def _push(self, frame_type, ms, payload, *, kind, index, boot=1, hflags=0, version=22, skip=0):
        self.seq += 1 + skip
        header = FrameHeader(
            version, 32, frame_type, hflags, 0, kind, index, self.seq, int(ms * TPM), boot
        )
        self.frames.append((0, Frame(header, payload, b"")))

    def crit(
        self, ms, *, v, err, flags=0, offset=0, clamped=False, index=0, boot=1, hflags=0, version=22
    ):
        payload = {
            "pid_velocity_actual": v,
            "position_error": err,
            "sample_flags": flags,
            "cached_pid_velocity_offset": offset,
            "cached_velocity_clamped": clamped,
        }
        self._push(
            FRAME_TYPE_CRITICAL,
            ms,
            payload,
            kind=1,
            index=index,
            boot=boot,
            hflags=hflags,
            version=version,
        )

    def status(self, ms, *, pressure=0, max_cycles=1000, kind=0, index=255, skip=0, hflags=0):
        payload = {"trace_deadline_pressure_count": pressure, "max_spi_sample_cycles": max_cycles}
        self._push(FRAME_TYPE_STATUS, ms, payload, kind=kind, index=index, skip=skip, hflags=hflags)

    def cruise(self, start_ms, n, v, err, *, step_ms=1, index=0, offset=0, boot=1):
        for i in range(n):
            self.crit(start_ms + i * step_ms, v=v, err=err, index=index, offset=offset, boot=boot)

    def run(self, mm=None):
        return tl.analyze_tracking_lag(
            self.frames, clock_hz_by_device={0: CLOCK}, product_by_device=PRODUCT, mm_per_rev=mm
        )


def test_steady_cruise_reports_expected_lag():
    s = Stream()
    s.status(0)
    s.cruise(100, 450, 75, -144)
    s.status(900)
    report = s.run()
    assert report.refused is None
    (seg,) = report.segments
    assert seg.speed_rpm == 75
    assert seg.error_median_units == -144
    assert seg.lag_ms == pytest.approx(-1.7578, abs=0.01)  # 144 units at 75 RPM
    assert seg.valid_frames == 350  # 450 frames minus 50 ms trimmed at each end
    assert seg.excluded_frames == 0
    assert not seg.sparse


def test_trim_is_by_elapsed_time_not_frame_count():
    s = Stream()
    s.cruise(0, 450, 75, -144, step_ms=2)  # 900 ms span, 2 ms apart
    (seg,) = s.run().segments
    assert seg.start_ms == pytest.approx(50.0)
    assert seg.end_ms == pytest.approx(848.0)
    assert seg.valid_frames == 400


def test_acceleration_tail_is_excluded_by_weighted_plateau():
    s = Stream()
    for i in range(1, 60):  # 10..590 RPM ramp, 1 ms apart: dense, 59 frames
        s.crit(i, v=i * 10, err=0)
    s.cruise(60, 100, 742, -1425, step_ms=8)  # 800 ms of sparse cruise
    (seg,) = s.run().segments
    assert seg.speed_rpm == 742
    assert seg.start_ms >= 60 + 50


def test_direction_change_splits_runs_and_normalizes_sign():
    s = Stream()
    s.cruise(0, 450, 75, -144, offset=131)
    s.cruise(450, 450, -75, 144, offset=-131)
    report = s.run()
    assert [seg.speed_rpm for seg in report.segments] == [75, -75]
    assert all(seg.lag_ms == pytest.approx(-1.7578, abs=0.01) for seg in report.segments)
    assert all(seg.offset_ratio == pytest.approx(1.75, abs=0.01) for seg in report.segments)


def test_spi_error_frame_is_transparent():
    s = Stream()
    s.cruise(0, 225, 75, -144)
    s.crit(225, v=0, err=0, flags=tl.SAMPLE_FLAG_SPI_ERROR)
    s.cruise(226, 225, 75, -144)
    (seg,) = s.run().segments
    assert seg.excluded_frames == 1
    assert seg.valid_frames == 350


def test_invalid_burst_longer_than_gap_threshold_ends_run():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    for i in range(60):  # 60 ms of failed reads at normal cadence
        s.crit(450 + i, v=0, err=0, flags=tl.SAMPLE_FLAG_SPI_ERROR)
    s.cruise(510, 450, 75, -144)
    report = s.run()
    assert len(report.segments) == 2
    assert any(skip.reason == "gap" for skip in report.skipped) or len(report.skipped) == 0


def test_invalid_frame_after_gap_does_not_rescue_run():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    s.crit(550, v=0, err=0, flags=tl.SAMPLE_FLAG_SPI_ERROR)  # 100 ms after the last valid frame
    s.cruise(551, 450, 75, -144)
    assert len(s.run().segments) == 2


def test_boot_id_change_splits_runs():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    s.cruise(450, 450, 75, -144, boot=2)
    assert len(s.run().segments) == 2


def test_drop_gap_flag_on_a_critical_frame_splits_runs():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    s.crit(450, v=75, err=-144, hflags=tl.FLAG_DROP_GAP_BEFORE_THIS_FRAME)
    s.cruise(451, 449, 75, -144)
    assert len(s.run().segments) == 2


def test_drop_gap_flag_on_a_status_frame_splits_every_channel():
    s = Stream()
    for i in range(450):
        s.crit(i, v=75, err=-144, index=0)
        s.crit(i, v=-75, err=144, index=1)
    s.status(450, hflags=tl.FLAG_DROP_GAP_BEFORE_THIS_FRAME)  # the latch landed on a status frame
    for i in range(450, 900):
        s.crit(i, v=75, err=-144, index=0)
        s.crit(i, v=-75, err=144, index=1)
    report = s.run()
    assert sum(1 for seg in report.segments if seg.channel.source_index == 0) == 2
    assert sum(1 for seg in report.segments if seg.channel.source_index == 1) == 2


def test_sequence_gap_on_non_critical_frame_splits_every_channel():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    s.status(450, skip=10)  # the device-global counter jumps
    s.cruise(451, 450, 75, -144)
    assert len(s.run().segments) == 2


def test_short_runs_are_counted_not_reported():
    s = Stream()
    s.cruise(0, 250, 75, -144)  # 250 ms minus 100 ms trim = 150 ms < 200 ms floor
    report = s.run()
    assert report.segments == []
    assert report.skipped[0].reason == "short_window"


def test_in_band_run_emptied_by_trimming_is_short_window():
    s = Stream()
    s.cruise(0, 80, 75, -144)  # 80 ms in band: both trims remove everything
    report = s.run()
    assert report.segments == []
    assert report.skipped[0].reason == "short_window"


def test_few_valid_frames_are_counted_not_reported():
    s = Stream()
    for i in range(15):  # 15 frames, 30 ms apart: 420 ms span, 15 valid frames < 20
        s.crit(i * 30, v=75, err=-144)
    report = s.run()
    assert report.segments == []
    assert report.skipped[0].reason == "few_valid_frames"


def test_longest_in_band_sub_run_wins_and_earliest_on_tie():
    s = Stream()
    s.cruise(0, 301, 75, -144)  # in band, 0..300 ms
    s.cruise(301, 100, 60, -100)  # out of band (20 percent low), still same sign
    s.cruise(401, 301, 75, -144)  # in band, 401..701 ms: tie with the first
    (seg,) = s.run().segments
    assert seg.start_ms == pytest.approx(50.0)
    assert seg.end_ms == pytest.approx(250.0)  # exactly the 200 ms floor, first run wins


def test_sparse_sampling_is_flagged():
    s = Stream()
    s.cruise(0, 100, 742, -1425, step_ms=8)
    (seg,) = s.run().segments
    assert seg.sparse
    assert seg.interval_median_ms == pytest.approx(8.0)


def test_two_ouroboros_channels_stay_separate():
    s = Stream()
    for i in range(450):
        s.crit(i, v=75, err=-144, index=0)
        s.crit(i, v=-75, err=144, index=1)
    report = s.run()
    assert sorted(seg.channel.source_index for seg in report.segments) == [0, 1]
    assert {seg.speed_rpm for seg in report.segments} == {75, -75}


def test_mm_conversion_uses_operator_scale():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    report = s.run(mm=40.0)
    text = "\n".join(tl.format_tracking_lag(report, 40.0))
    assert "mm=-0.088" in text
    assert "mm=" not in "\n".join(tl.format_tracking_lag(report, None))


def test_protocol_21_capture_is_refused():
    s = Stream()
    s.crit(0, v=75, err=-144, version=21)
    report = s.run()
    assert report.refused is not None and "21" in report.refused
    assert tl.format_tracking_lag(report) == [f"tracking refused: {report.refused}"]


def test_budget_requires_board_status_before_and_after_motion():
    s = Stream()
    s.status(0)
    s.cruise(100, 450, 75, -144)
    unbracketed = s.run().budgets[0]
    assert not unbracketed.bracketed
    assert unbracketed.reason == "no status 200 ms after motion"

    s.status(800, pressure=0, max_cycles=90_000)  # 250 ms after the last motion frame
    b = s.run().budgets[0]
    assert b.bracketed
    assert b.max_spi_sample_cycles == 90_000
    assert b.threshold_cycles == 120_000
    assert b.pressure_delta == 0


def test_budget_without_status_before_motion_is_unbracketed():
    s = Stream()
    s.cruise(0, 450, 75, -144)
    s.status(800)
    assert s.run().budgets[0].reason == "no status before motion"


def test_budget_reads_board_source_only_and_sees_late_pressure():
    s = Stream()
    s.status(0)
    s.status(0, pressure=9, kind=2)  # USB source: ignored
    s.cruise(100, 450, 75, -144)
    s.status(800, pressure=5, kind=2)  # USB source: ignored
    s.status(810, pressure=1, max_cycles=130_000)
    b = s.run().budgets[0]
    assert b.pressure_delta == 1
    assert b.max_spi_sample_cycles == 130_000


def test_status_only_capture_yields_no_motion_budget_and_no_segments():
    s = Stream()
    s.status(0)
    s.status(100)
    report = s.run()
    assert report.segments == [] and report.skipped == []
    assert report.budgets[0].reason == "no motion"


def test_single_frame_motion_blip_is_skipped_not_crashed():
    s = Stream()
    s.status(0)
    s.crit(100, v=75, err=-144)
    s.status(400)
    report = s.run()
    assert report.segments == []
    assert report.skipped[0].reason == "few_valid_frames"
    assert report.budgets[0].bracketed


def test_gap_over_five_times_median_splits_even_below_50ms():
    s = Stream()
    s.cruise(0, 450, 75, -144)  # 1 ms cadence
    s.cruise(460, 450, 75, -144)  # 10 ms valid-to-valid gap: > 5 x 1 ms
    assert len(s.run().segments) == 2


def test_budget_without_board_source_has_no_verdict():
    s = Stream()
    s.status(0, kind=2)
    s.cruise(100, 450, 75, -144)
    s.status(800, kind=2)
    b = s.run().budgets[0]
    assert not b.bracketed
    assert b.reason == "no board status source"
