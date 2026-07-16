"""Independent reference for windowed encoder-velocity confidence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from statistics import median


RATE_Q_SCALE = 1_000
MIN_SELECTED_BLOCKS = 16
CONFIDENCE_MULTIPLIER_95 = 2.131


class AccuracyDecision(str, Enum):
    """Confidence-gated physical-accuracy result."""

    PASS = "pass"
    CONFIDENT_FAIL = "confident_fail"
    COLLECT_MORE = "collect_more"


@dataclass(frozen=True)
class VelocityObservation:
    """One signed encoder displacement observed over an actual interval."""

    delta_counts: int
    elapsed_us: int
    direction: int
    stroke_index: int
    target_cpms: int


@dataclass(frozen=True)
class BlockLevelReport:
    """Completed-block statistics at one dyadic level."""

    level: int
    block_count: int
    mean_rate_q: int
    m2_rate_q2: int
    sample_variance_rate_q2: int | None
    variance_of_mean_q2: int | None
    exact_delta_counts: int
    exact_elapsed_us: int
    plateau_with_next: bool | None


@dataclass(frozen=True)
class DirectionReport:
    """Reference blocking result for one signed target direction."""

    direction: int
    target_cpms: int
    observation_count: int
    stroke_count: int
    levels: tuple[BlockLevelReport, ...]
    selected_level: int | None
    selected_blocks: int | None
    confidence_half_width_rate_q: int | None
    settled_mean_rate_q: int | None
    mean_error_rate_q: int | None


@dataclass(frozen=True)
class VelocityMeasurementReport:
    """Complete deterministic report for one comparison group."""

    directions: dict[int, DirectionReport]
    min_interval_us: int
    max_interval_us: int
    median_interval_us: float
    interval_deviations_us: list[float]
    timing_discontinuity_count: int


@dataclass
class _Moments:
    count: int = 0
    mean: int = 0
    m2: int = 0

    def observe(self, value: int) -> None:
        if self.count == 0:
            self.count = 1
            self.mean = value
            return
        next_count = self.count + 1
        delta = value - self.mean
        next_mean = self.mean + _trunc_div(delta, next_count)
        self.m2 += delta * (value - next_mean)
        self.count = next_count
        self.mean = next_mean

    def merge(self, other: _Moments) -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean, other.m2
            return
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.mean += _trunc_div(delta * other.count, combined_count)
        self.m2 += other.m2 + _trunc_div(delta * delta * self.count * other.count, combined_count)
        self.count = combined_count


@dataclass(frozen=True)
class _Block:
    delta_counts: int
    elapsed_us: int
    observation_count: int

    @property
    def rate_q(self) -> int:
        return _trunc_div(self.delta_counts * 1_000_000 * RATE_Q_SCALE, self.elapsed_us)

    def merge(self, other: _Block) -> _Block:
        return _Block(
            self.delta_counts + other.delta_counts,
            self.elapsed_us + other.elapsed_us,
            self.observation_count + other.observation_count,
        )


@dataclass
class _Level:
    pending: _Block | None = None
    moments: _Moments = field(default_factory=_Moments)
    exact_delta_counts: int = 0
    exact_elapsed_us: int = 0

    def record(self, block: _Block) -> None:
        self.moments.observe(block.rate_q)
        self.exact_delta_counts += block.delta_counts
        self.exact_elapsed_us += block.elapsed_us

    def merge_completed(self, other: _Level) -> None:
        self.moments.merge(other.moments)
        self.exact_delta_counts += other.exact_delta_counts
        self.exact_elapsed_us += other.exact_elapsed_us


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("elapsed time and divisors must be positive")
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def _observe(levels: list[_Level], observation: VelocityObservation) -> None:
    block = _Block(observation.delta_counts, observation.elapsed_us, 1)
    for level in levels:
        level.record(block)
        if level.pending is None:
            level.pending = block
            return
        block = level.pending.merge(block)
        level.pending = None


def _sample_variance(moments: _Moments) -> int | None:
    return None if moments.count < 2 else moments.m2 // (moments.count - 1)


def _q(moments: _Moments) -> int | None:
    variance = _sample_variance(moments)
    return None if variance is None else variance // moments.count


def _plateau_like(lower: _Moments, upper: _Moments) -> bool:
    lower_q, upper_q = _q(lower), _q(upper)
    if lower_q is None or upper_q is None:
        return False
    lower_se = lower_q * math.sqrt(2 / (lower.count - 1))
    upper_se = upper_q * math.sqrt(2 / (upper.count - 1))
    threshold = 1.96 * math.sqrt(lower_se * lower_se + upper_se * upper_se)
    return upper_q - lower_q <= threshold


def _select_level(levels: list[_Level]) -> int | None:
    consecutive = 0
    selected = None
    for lower_index in range(len(levels) - 1):
        if _plateau_like(levels[lower_index].moments, levels[lower_index + 1].moments):
            consecutive += 1
            if consecutive >= 2:
                first = lower_index + 1 - consecutive
                for candidate in range(lower_index + 1, first - 1, -1):
                    if levels[candidate].moments.count >= MIN_SELECTED_BLOCKS:
                        selected = candidate
                        break
        else:
            consecutive = 0
    return selected


def _direction_report(observations: list[VelocityObservation], max_levels: int) -> DirectionReport:
    targets = {observation.target_cpms for observation in observations}
    if len(targets) != 1:
        raise ValueError("mixed targets in one direction are not comparable")
    merged = [_Level() for _ in range(max_levels)]
    stroke_indexes = sorted({observation.stroke_index for observation in observations})
    for stroke_index in stroke_indexes:
        local = [_Level() for _ in range(max_levels)]
        for observation in observations:
            if observation.stroke_index == stroke_index:
                _observe(local, observation)
        for target, source in zip(merged, local, strict=True):
            target.merge_completed(source)
    selected = _select_level(merged)
    level_reports = []
    for index, level in enumerate(merged):
        variance = _sample_variance(level.moments)
        level_reports.append(
            BlockLevelReport(
                level=index,
                block_count=level.moments.count,
                mean_rate_q=level.moments.mean,
                m2_rate_q2=level.moments.m2,
                sample_variance_rate_q2=variance,
                variance_of_mean_q2=_q(level.moments),
                exact_delta_counts=level.exact_delta_counts,
                exact_elapsed_us=level.exact_elapsed_us,
                plateau_with_next=(
                    _plateau_like(level.moments, merged[index + 1].moments)
                    if index + 1 < max_levels
                    else None
                ),
            )
        )
    selected_q = None if selected is None else _q(merged[selected].moments)
    half_width = None if selected_q is None else math.ceil(2.131 * math.sqrt(selected_q))
    mean_rate = None
    mean_error = None
    if selected is not None:
        level = merged[selected]
        mean_rate = _trunc_div(
            level.exact_delta_counts * 1_000_000 * RATE_Q_SCALE, level.exact_elapsed_us
        )
        mean_error = abs(mean_rate - next(iter(targets)) * 1_000 * RATE_Q_SCALE)
    return DirectionReport(
        direction=observations[0].direction,
        target_cpms=next(iter(targets)),
        observation_count=len(observations),
        stroke_count=len(stroke_indexes),
        levels=tuple(level_reports),
        selected_level=selected,
        selected_blocks=None if selected is None else merged[selected].moments.count,
        confidence_half_width_rate_q=half_width,
        settled_mean_rate_q=mean_rate,
        mean_error_rate_q=mean_error,
    )


def analyze_velocity_observations(
    observations: list[VelocityObservation], max_levels: int
) -> VelocityMeasurementReport:
    """Analyze raw observations without pooling directions or stroke partials."""
    if not observations:
        raise ValueError("at least one velocity observation is required")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive")
    for observation in observations:
        if observation.elapsed_us <= 0:
            raise ValueError("elapsed_us must be positive")
        if observation.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
    directions = {
        direction: _direction_report(
            [observation for observation in observations if observation.direction == direction],
            max_levels,
        )
        for direction in sorted({observation.direction for observation in observations})
    }
    intervals = [observation.elapsed_us for observation in observations]
    median_interval = float(median(intervals))
    deviations = [abs(interval - median_interval) for interval in intervals]
    discontinuities = sum(
        interval > 2 * median_interval or interval < median_interval / 2 for interval in intervals
    )
    return VelocityMeasurementReport(
        directions=directions,
        min_interval_us=min(intervals),
        max_interval_us=max(intervals),
        median_interval_us=median_interval,
        interval_deviations_us=deviations,
        timing_discontinuity_count=discontinuities,
    )


def classify_absolute_rate(
    *,
    settled_delta_counts: int,
    settled_elapsed_us: int,
    target_counts_per_ms: int,
    half_width_rate_q: int,
    counts_per_rev: int,
    limit_mrev_per_s: int,
) -> AccuracyDecision:
    """Apply the exact physical three-way gate using rational rates."""
    if settled_elapsed_us <= 0:
        raise ValueError("settled_elapsed_us must be positive")
    settled_rate = Fraction(settled_delta_counts * 1_000_000, settled_elapsed_us)
    target_rate = target_counts_per_ms * 1_000
    error_rate_q = abs(settled_rate - target_rate) * RATE_Q_SCALE
    limit_rate_q = Fraction(limit_mrev_per_s * counts_per_rev * RATE_Q_SCALE, 1_000)
    if error_rate_q + half_width_rate_q <= limit_rate_q:
        return AccuracyDecision.PASS
    if max(error_rate_q - half_width_rate_q, 0) > limit_rate_q:
        return AccuracyDecision.CONFIDENT_FAIL
    return AccuracyDecision.COLLECT_MORE


def velocity_report_to_json(report: VelocityMeasurementReport) -> dict[str, object]:
    """Convert a reference report to deterministic JSON-compatible values."""
    return {
        "directions": {
            str(direction): {
                "direction": item.direction,
                "target_cpms": item.target_cpms,
                "observation_count": item.observation_count,
                "stroke_count": item.stroke_count,
                "levels": [level.__dict__ for level in item.levels],
                "selected_level": item.selected_level,
                "selected_blocks": item.selected_blocks,
                "confidence_half_width_rate_q": item.confidence_half_width_rate_q,
                "settled_mean_rate_q": item.settled_mean_rate_q,
                "mean_error_rate_q": item.mean_error_rate_q,
            }
            for direction, item in sorted(report.directions.items())
        },
        "min_interval_us": report.min_interval_us,
        "max_interval_us": report.max_interval_us,
        "median_interval_us": report.median_interval_us,
        "interval_deviations_us": report.interval_deviations_us,
        "timing_discontinuity_count": report.timing_discontinuity_count,
    }
