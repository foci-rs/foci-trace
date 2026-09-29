"""Cruise-phase tracking-lag measurement over decoded trace frames.

Segments steady cruise from the chip's own velocity reading, scores the chip
position error as a direction-normalized time-equivalent lag, and reports
whether the sampler-overrun budget was bracketed by board status frames.

The module is pure: it consumes already-decoded frames and performs no I/O.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from .protocol import FRAME_TYPE_CRITICAL, FRAME_TYPE_STATUS, Frame

MIN_SPEED_RPM = 5.0
CRUISE_BAND = 0.05
TRIM_MS = 50.0
MIN_WINDOW_MS = 200.0
MIN_VALID_FRAMES = 20
SPARSE_INTERVAL_MS = 2.0
GAP_MS = 50.0
GAP_MULTIPLIER = 5.0
BRACKET_MS = 200.0
UNITS_PER_REV = 65536
MIN_PROTOCOL_VERSION = 22
SEQUENCE_MASK = 0xFFFF_FFFF

SOURCE_KIND_BOARD = 0
SOURCE_INDEX_NA = 255
FLAG_DROP_GAP_BEFORE_THIS_FRAME = 1 << 0
SAMPLE_FLAG_SPI_ERROR = 1 << 0

BOARD_THRESHOLD_CYCLES = {
    "FOCI OpenFFBoard": 120_000,
    "FOCI Ouroboros": 372_000,
    "FOCI OpenFFBoard (trace)": 120_000,
    "FOCI Ouroboros (trace)": 372_000,
}


@dataclass(frozen=True)
class ChannelKey:
    """Identity of one servo channel within one capture device and boot."""

    device_index: int
    source_index: int
    boot_id: int


@dataclass
class CruiseSegment:
    """One steady-cruise window and the tracking lag measured across it."""

    channel: ChannelKey
    start_ms: float
    end_ms: float
    valid_frames: int
    excluded_frames: int
    speed_rpm: float
    offset_ratio: float | None
    clamped: bool
    error_median_units: float
    error_iqr_units: float
    lag_ms: float
    interval_median_ms: float
    interval_max_ms: float
    sparse: bool


@dataclass
class SkippedRun:
    """A motion run that did not qualify as a measurable cruise window."""

    channel: ChannelKey
    start_ms: float
    end_ms: float
    reason: str


@dataclass
class SamplerBudget:
    """Sampler-overrun verdict for one capture device."""

    device_index: int
    bracketed: bool
    reason: str
    max_spi_sample_cycles: int | None
    pressure_delta: int | None
    threshold_cycles: int | None


@dataclass
class TrackingLagReport:
    """Everything the tracking-lag pass derived from one capture."""

    refused: str | None
    segments: list[CruiseSegment]
    skipped: list[SkippedRun]
    budgets: list[SamplerBudget]


@dataclass
class _Run:
    """A contiguous single-direction motion run and what terminated it."""

    frames: list[Frame]
    end_cause: str | None


def _sign(value: float) -> int:
    """Return -1, 0 or 1 for the sign of ``value``."""
    return (value > 0) - (value < 0)


def _is_valid(frame: Frame) -> bool:
    """Return whether a critical frame's sample survived the SPI read."""
    return not frame.payload["sample_flags"] & SAMPLE_FLAG_SPI_ERROR


def _velocity(frame: Frame) -> float:
    """Return the chip velocity reading of a critical frame."""
    return frame.payload["pid_velocity_actual"]


def _percentile(values: list[float], quantile: float) -> float:
    """Return the linearly interpolated ``quantile`` of ``values``."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Return the value at which half the total weight has accumulated."""
    pairs = sorted(zip(values, weights, strict=True))
    half = sum(weights) / 2
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return float(value)
    return float(pairs[-1][0])


def _longest_true_span(flags: list[bool]) -> tuple[int, int] | None:
    """Return the inclusive bounds of the longest true run, earliest on tie."""
    best: tuple[int, int] | None = None
    best_length = 0
    start: int | None = None
    for index, flag in enumerate([*flags, False]):
        if flag:
            if start is None:
                start = index
        elif start is not None:
            if index - start > best_length:
                best_length = index - start
                best = (start, index - 1)
            start = None
    return best


def _refuse_if_old(frames: list[tuple[int, Frame]]) -> str | None:
    """Return a refusal reason when any frame predates the chip position error."""
    for _device, frame in frames:
        version = frame.header.protocol_version
        if version < MIN_PROTOCOL_VERSION:
            return f"protocol {version} predates the chip position error"
    return None


def _device_boundaries(frames: list[tuple[int, Frame]]) -> dict[tuple[int, int], list[int]]:
    """Return tick values where stream continuity broke, per device and boot.

    Boundaries are keyed by boot as well as device because the MCU clock
    restarts at zero on every boot: a tick recorded in one boot is meaningless
    against another boot's timeline and must never be tested there. Each
    boundary is filed under the boot of the frame that carries it, which is the
    timeline its tick belongs to.
    """
    boundaries: dict[tuple[int, int], list[int]] = {}
    previous: dict[int, tuple[int, int]] = {}
    for device, frame in frames:
        header = frame.header
        key = (device, header.boot_id)
        boundaries.setdefault(key, [])
        broke = bool(header.flags & FLAG_DROP_GAP_BEFORE_THIS_FRAME)
        prior = previous.get(device)
        if prior is not None:
            prior_sequence, prior_boot = prior
            expected = (prior_sequence + 1) & SEQUENCE_MASK
            broke = broke or header.boot_id != prior_boot or header.sequence != expected
        if broke:
            boundaries[key].append(header.timestamp_ticks)
        previous[device] = (header.sequence, header.boot_id)
    for ticks in boundaries.values():
        ticks.sort()
    return boundaries


def _boundary_between(boundaries: list[int], low: int, high: int) -> bool:
    """Return whether a boundary tick lies in ``(low, high]``."""
    return bisect_right(boundaries, low) < bisect_right(boundaries, high)


def _channel_streams(frames: list[tuple[int, Frame]]) -> dict[ChannelKey, list[Frame]]:
    """Group critical frames by channel, preserving capture order."""
    streams: dict[ChannelKey, list[Frame]] = {}
    for device, frame in frames:
        if frame.header.frame_type != FRAME_TYPE_CRITICAL:
            continue
        key = ChannelKey(device, frame.header.source_index, frame.header.boot_id)
        streams.setdefault(key, []).append(frame)
    return streams


def _motion_runs(stream: list[Frame], boundaries: list[int], ticks_per_ms: float) -> list[_Run]:
    """Split one channel stream into single-direction motion runs.

    Frames whose sample failed the SPI read are carried along transparently:
    they neither start nor end a run, and they cannot bridge a timestamp gap,
    because the gap test measures valid-to-valid time only.
    """
    runs: list[_Run] = []
    current: list[Frame] = []
    run_sign = 0
    last_valid_ticks = 0
    intervals: list[float] = []
    previous_ticks: int | None = None

    def close(cause: str | None) -> None:
        nonlocal current, run_sign, last_valid_ticks, intervals
        if current:
            runs.append(_Run(current, cause))
        current = []
        run_sign = 0
        last_valid_ticks = 0
        intervals = []

    for frame in stream:
        ticks = frame.header.timestamp_ticks
        if (
            current
            and previous_ticks is not None
            and _boundary_between(boundaries, previous_ticks, ticks)
        ):
            close("boundary")
        previous_ticks = ticks

        if not _is_valid(frame):
            if current:
                current.append(frame)
            continue

        velocity = _velocity(frame)
        moving = abs(velocity) >= MIN_SPEED_RPM

        if current:
            if not moving or _sign(velocity) != run_sign:
                close(None)
            else:
                gap_ms = (ticks - last_valid_ticks) / ticks_per_ms
                threshold = GAP_MS
                if len(intervals) >= 2:
                    threshold = min(GAP_MS, GAP_MULTIPLIER * median(intervals))
                if gap_ms > threshold:
                    close("gap")
                else:
                    intervals.append(gap_ms)
                    current.append(frame)
                    last_valid_ticks = ticks
                    continue

        if moving:
            current = [frame]
            run_sign = _sign(velocity)
            last_valid_ticks = ticks
            intervals = []

    close(None)
    return runs


def _valid_weights(valid: list[Frame], ticks_per_ms: float) -> list[float]:
    """Weight each valid frame by the time until the next one, median for the last."""
    gaps = [
        (valid[i + 1].header.timestamp_ticks - valid[i].header.timestamp_ticks) / ticks_per_ms
        for i in range(len(valid) - 1)
    ]
    return [*gaps, median(gaps)]


def _plateau_window(
    run: list[Frame], ticks_per_ms: float
) -> tuple[list[Frame], list[Frame], float, float] | None:
    """Return the trimmed in-band window of a run, or None when trimming empties it."""
    valid = [frame for frame in run if _is_valid(frame)]
    speeds = [abs(_velocity(frame)) for frame in valid]
    reference = _weighted_median(speeds, _valid_weights(valid, ticks_per_ms))
    band = CRUISE_BAND * reference
    span = _longest_true_span([abs(speed - reference) <= band for speed in speeds])
    if span is None:
        return None

    first, last = span
    start_ms = valid[first].header.timestamp_ticks / ticks_per_ms + TRIM_MS
    end_ms = valid[last].header.timestamp_ticks / ticks_per_ms - TRIM_MS
    window_valid = [
        frame
        for frame in valid[first : last + 1]
        if start_ms <= frame.header.timestamp_ticks / ticks_per_ms <= end_ms
    ]
    if not window_valid:
        return None

    window_all = [
        frame for frame in run if start_ms <= frame.header.timestamp_ticks / ticks_per_ms <= end_ms
    ]
    return window_all, window_valid, start_ms, end_ms


def _segment_stats(
    channel: ChannelKey,
    window_all: list[Frame],
    window_valid: list[Frame],
    start_ms: float,
    end_ms: float,
    ticks_per_ms: float,
) -> CruiseSegment:
    """Score one trimmed cruise window."""
    speed_rpm = median(_velocity(frame) for frame in window_valid)
    errors = [frame.payload["position_error"] for frame in window_valid]
    error_median = median(errors)
    units_per_ms = abs(speed_rpm) * UNITS_PER_REV / 60_000

    offset_median = median(frame.payload["cached_pid_velocity_offset"] for frame in window_valid)

    times = [frame.header.timestamp_ticks / ticks_per_ms for frame in window_all]
    gaps = [later - earlier for earlier, later in pairwise(times)]
    interval_median = median(gaps)

    return CruiseSegment(
        channel=channel,
        start_ms=start_ms,
        end_ms=end_ms,
        valid_frames=len(window_valid),
        excluded_frames=len(window_all) - len(window_valid),
        speed_rpm=speed_rpm,
        offset_ratio=offset_median / speed_rpm if offset_median else None,
        clamped=any(frame.payload["cached_velocity_clamped"] for frame in window_valid),
        error_median_units=error_median,
        error_iqr_units=_percentile(errors, 0.75) - _percentile(errors, 0.25),
        lag_ms=_sign(speed_rpm) * error_median / units_per_ms,
        interval_median_ms=interval_median,
        interval_max_ms=max(gaps),
        sparse=interval_median > SPARSE_INTERVAL_MS,
    )


def _run_outcome(channel: ChannelKey, run: _Run, ticks_per_ms: float) -> CruiseSegment | SkippedRun:
    """Score one motion run, or explain why it is not measurable."""
    start_ms = run.frames[0].header.timestamp_ticks / ticks_per_ms
    end_ms = run.frames[-1].header.timestamp_ticks / ticks_per_ms

    if sum(1 for frame in run.frames if _is_valid(frame)) < 2:
        return SkippedRun(channel, start_ms, end_ms, "few_valid_frames")

    window = _plateau_window(run.frames, ticks_per_ms)
    if window is None:
        return SkippedRun(channel, start_ms, end_ms, run.end_cause or "short_window")

    window_all, window_valid, window_start, window_end = window
    if window_end - window_start < MIN_WINDOW_MS:
        return SkippedRun(channel, start_ms, end_ms, run.end_cause or "short_window")
    if len(window_valid) < MIN_VALID_FRAMES:
        return SkippedRun(channel, start_ms, end_ms, run.end_cause or "few_valid_frames")

    return _segment_stats(channel, window_all, window_valid, window_start, window_end, ticks_per_ms)


def _sampler_budget(
    device_index: int, device_frames: list[Frame], ticks_per_ms: float, product: str
) -> SamplerBudget:
    """Read the sampler-overrun budget from board status frames bracketing motion.

    The motion and both bracketing status frames must share one boot. The
    deadline-pressure and cycle counters reset on reboot, so a bracket that
    spans one subtracts counters from different epochs and can report a clean
    budget for a capture that actually overran.
    """
    status = [
        frame
        for frame in device_frames
        if frame.header.frame_type == FRAME_TYPE_STATUS
        and frame.header.source_kind == SOURCE_KIND_BOARD
        and frame.header.source_index == SOURCE_INDEX_NA
    ]
    if not status:
        return SamplerBudget(device_index, False, "no board status source", None, None, None)

    motion = [
        frame
        for frame in device_frames
        if frame.header.frame_type == FRAME_TYPE_CRITICAL
        and _is_valid(frame)
        and abs(_velocity(frame)) >= MIN_SPEED_RPM
    ]
    if not motion:
        return SamplerBudget(device_index, False, "no motion", None, None, None)

    motion_boots = {frame.header.boot_id for frame in motion}
    if len(motion_boots) > 1:
        return SamplerBudget(device_index, False, "reboot during capture", None, None, None)

    boot = motion_boots.pop()
    rebooted = len({frame.header.boot_id for frame in device_frames}) > 1
    ticks = [frame.header.timestamp_ticks for frame in motion]
    in_boot = [frame for frame in status if frame.header.boot_id == boot]

    before = [frame for frame in in_boot if frame.header.timestamp_ticks < min(ticks)]
    if not before:
        reason = "reboot during capture" if rebooted else "no status before motion"
        return SamplerBudget(device_index, False, reason, None, None, None)

    settled = max(ticks) + BRACKET_MS * ticks_per_ms
    after = [frame for frame in in_boot if frame.header.timestamp_ticks >= settled]
    if not after:
        reason = (
            "reboot during capture" if rebooted else f"no status {BRACKET_MS:.0f} ms after motion"
        )
        return SamplerBudget(device_index, False, reason, None, None, None)

    return SamplerBudget(
        device_index=device_index,
        bracketed=True,
        reason="",
        max_spi_sample_cycles=after[-1].payload["max_spi_sample_cycles"],
        pressure_delta=(
            after[-1].payload["trace_deadline_pressure_count"]
            - before[0].payload["trace_deadline_pressure_count"]
        ),
        threshold_cycles=BOARD_THRESHOLD_CYCLES.get(product),
    )


def analyze_tracking_lag(
    frames: list[tuple[int, Frame]],
    *,
    clock_hz_by_device: dict[int, int],
    product_by_device: dict[int, str],
    mm_per_rev: float | None = None,
) -> TrackingLagReport:
    """Measure cruise tracking lag and the sampler budget across decoded frames.

    ``frames`` pairs each decoded frame with the index of the capture device it
    came from, in capture order. ``clock_hz_by_device`` must carry an entry for
    every device present; it comes from each device's metadata ``clock_freq``.
    ``mm_per_rev`` is accepted for call-site symmetry with
    :func:`format_tracking_lag`, which is where the conversion is applied.
    """
    del mm_per_rev

    refused = _refuse_if_old(frames)
    if refused is not None:
        return TrackingLagReport(refused, [], [], [])

    boundaries = _device_boundaries(frames)
    segments: list[CruiseSegment] = []
    skipped: list[SkippedRun] = []
    for channel, stream in _channel_streams(frames).items():
        ticks_per_ms = clock_hz_by_device[channel.device_index] / 1000.0
        own = boundaries[(channel.device_index, channel.boot_id)]
        for run in _motion_runs(stream, own, ticks_per_ms):
            outcome = _run_outcome(channel, run, ticks_per_ms)
            if isinstance(outcome, CruiseSegment):
                segments.append(outcome)
            else:
                skipped.append(outcome)

    budgets = [
        _sampler_budget(
            device,
            [frame for index, frame in frames if index == device],
            clock_hz_by_device[device] / 1000.0,
            product_by_device.get(device, ""),
        )
        for device in sorted({device for device, _frame in frames})
    ]
    return TrackingLagReport(None, segments, skipped, budgets)


def format_tracking_lag(report: TrackingLagReport, mm_per_rev: float | None = None) -> list[str]:
    """Render a tracking-lag report as operator-readable lines."""
    if report.refused is not None:
        return [f"tracking refused: {report.refused}"]

    lines: list[str] = []
    for seg in report.segments:
        line = (
            f"tracking dev{seg.channel.device_index} ch{seg.channel.source_index} "
            f"{seg.start_ms:.0f}-{seg.end_ms:.0f}ms "
            f"speed={seg.speed_rpm:+.1f}rpm "
            f"ff={seg.offset_ratio or 0:.2f}{' clamped' if seg.clamped else ''} "
            f"err={seg.error_median_units:+.0f}u "
            f"iqr={seg.error_iqr_units:.0f}u "
            f"lag={seg.lag_ms:+.2f}ms "
            f"frames={seg.valid_frames}/{seg.excluded_frames} "
            f"interval={seg.interval_median_ms:.2f}/{seg.interval_max_ms:.2f}ms"
            f"{' sparse' if seg.sparse else ''}"
        )
        if mm_per_rev is not None:
            line += f" mm={seg.error_median_units / UNITS_PER_REV * mm_per_rev:+.3f}"
        lines.append(line)

    lines.extend(
        f"skipped dev{skip.channel.device_index} ch{skip.channel.source_index} "
        f"{skip.start_ms:.0f}-{skip.end_ms:.0f}ms {skip.reason}"
        for skip in report.skipped
    )
    lines.extend(
        (
            f"budget dev{budget.device_index} bracketed=True "
            f"max_cycles={budget.max_spi_sample_cycles} "
            f"threshold={budget.threshold_cycles} "
            f"pressure_delta={budget.pressure_delta}"
        )
        if budget.bracketed
        else f"budget dev{budget.device_index} bracketed=False {budget.reason}"
        for budget in report.budgets
    )
    return lines
