"""Independent reference for windowed encoder-velocity confidence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from statistics import median
import struct


RATE_Q_SCALE = 1_000
MIN_SELECTED_BLOCKS = 16
CONFIDENCE_MULTIPLIER_95 = 2.131
CONFIDENCE_MULTIPLIERS_95_DF15 = (
    2.131,
    2.48988,
    2.693739,
    2.836627,
    2.946713,
    3.036283,
    3.111806,
    3.177104,
    3.234628,
    3.286039,
    3.332518,
    3.374933,
    3.413943,
    3.450056,
    3.483677,
    3.515129,
    3.544677,
    3.57254,
    3.598902,
    3.623918,
    3.647719,
    3.670419,
    3.692116,
    3.712896,
    3.732834,
    3.751997,
    3.770443,
    3.788224,
    3.805387,
    3.821973,
    3.838022,
    3.853567,
    3.868639,
    3.883266,
    3.897474,
    3.911287,
    3.924726,
    3.937811,
    3.950561,
    3.962993,
    3.975122,
    3.986963,
    3.998529,
    4.009833,
    4.020887,
    4.031702,
    4.042288,
    4.052655,
    4.062811,
    4.072765,
    4.082526,
    4.0921,
    4.101495,
    4.110717,
    4.119773,
    4.128668,
    4.137409,
    4.146001,
    4.154448,
    4.162756,
    4.170929,
    4.178972,
    4.186888,
    4.194683,
)


def _f32(value: float | int) -> float:
    """Round one value to the firmware's IEEE-754 single-precision domain."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
STAGE_B_OUTCOME_COMPLETE = 1
STAGE_B_OUTCOME_INCONCLUSIVE = 2
STAGE_B_CAUSE_REPRODUCTION_MISMATCH = 5


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
    peak_torque_target_abs: int | None = None
    pid_velocity_actual: int | None = None
    actual_velocity_pptm: int | None = None


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


def _confidence_multiplier(family_size: int) -> float:
    if not 1 <= family_size <= len(CONFIDENCE_MULTIPLIERS_95_DF15):
        raise ValueError("confidence family size is outside the audited table")
    return CONFIDENCE_MULTIPLIERS_95_DF15[family_size - 1]


def _confidence_half_width(variance_of_mean_q2: int, family_size: int) -> int:
    variance = _f32(variance_of_mean_q2)
    multiplier = _f32(_confidence_multiplier(family_size))
    return math.ceil(_f32(multiplier * _f32(math.sqrt(variance))))


def _has_two_doubling_plateau(levels: list[_Level]) -> bool:
    consecutive = 0
    for lower, upper in zip(levels, levels[1:]):
        if _plateau_like(lower.moments, upper.moments):
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
    return False


def _round_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _stationarity_metrics(
    observations: list[VelocityObservation], selection: int, variance_of_mean_q2: int
) -> dict[str, int] | int:
    block_observations = 1 << selection
    block_count = len(observations) // block_observations
    if block_count < MIN_SELECTED_BLOCKS:
        return 1
    blocks: list[_Block] = []
    for block_index in range(block_count):
        start = block_index * block_observations
        selected = observations[start : start + block_observations]
        blocks.append(
            _Block(
                sum(item.delta_counts for item in selected),
                sum(item.elapsed_us for item in selected),
                len(selected),
            )
        )
    elapsed = 0
    x_values: list[int] = []
    y_values: list[int] = []
    for block in blocks:
        x_values.append(elapsed + block.elapsed_us // 2)
        elapsed += block.elapsed_us
        y_values.append(block.rate_q)
    n = len(blocks)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    denominator = n * sum(value * value for value in x_values) - sum_x * sum_x
    if denominator <= 0:
        return 4
    numerator = n * sum(x * y for x, y in zip(x_values, y_values, strict=True)) - sum_x * sum_y
    slope_per_us = _f32(_f32(numerator) / _f32(denominator))
    intercept = _f32(_f32(_f32(sum_y) - _f32(slope_per_us * _f32(sum_x))) / _f32(n))
    residuals = [
        _f32(_f32(y) - _f32(intercept + _f32(slope_per_us * _f32(x))))
        for x, y in zip(x_values, y_values, strict=True)
    ]
    residual_sum = _f32(0.0)
    lag_product_sum = _f32(0.0)
    previous_residual: float | None = None
    for residual in residuals:
        residual_sum = _f32(residual_sum + _f32(residual * residual))
        if previous_residual is not None:
            lag_product_sum = _f32(lag_product_sum + _f32(residual * previous_residual))
        previous_residual = residual
    multiplier = _f32(_confidence_multiplier(3))
    residual_variance = _f32(residual_sum / _f32(n - 2))
    blocking_variance = _f32(_f32(variance_of_mean_q2) * _f32(n))
    slope_variance = max(residual_variance, blocking_variance)
    sxx = _f32(_f32(denominator) / _f32(n))
    slope_per_second = _f32(slope_per_us * _f32(1_000_000))
    slope_half_width = _f32(
        _f32(multiplier * _f32(math.sqrt(_f32(slope_variance / sxx)))) * _f32(1_000_000)
    )
    if abs(slope_per_second) > slope_half_width:
        return 2
    lag_one = _f32(0.0) if residual_sum == 0 else _f32(lag_product_sum / residual_sum)
    lag_null = _f32(_f32(-1.0) / _f32(n))
    lag_half_width = _f32(multiplier / _f32(math.sqrt(_f32(n))))
    if abs(_f32(lag_one - lag_null)) > lag_half_width:
        return 3
    return {
        "slope_q": _round_away(slope_per_second),
        "slope_half_width_q": math.ceil(slope_half_width),
        "lag_one_q": _round_away(_f32(lag_one * _f32(1_000_000))),
        "lag_one_half_width_q": math.ceil(_f32(lag_half_width * _f32(1_000_000))),
        "residual_magnitude_q2": math.ceil(_f32(residual_sum / _f32(n))),
    }


def _stationary_suffix(observations: list[VelocityObservation]) -> dict[str, int]:
    selected: dict[str, int] | None = None
    tested = 0
    for suffix_len in (64, 128, 256):
        if suffix_len > len(observations):
            continue
        tested += 1
        suffix = observations[-suffix_len:]
        levels = [_Level() for _ in range(5)]
        for observation in suffix:
            _observe(levels, observation)
        level = _select_level(levels)
        if level is None:
            if selected is not None:
                selected["tested_suffixes"] = tested
                return selected
            return {
                "usable": 0,
                "exclusion": 1 if _has_two_doubling_plateau(levels) else 0,
                "tested_suffixes": tested,
            }
        variance = _q(levels[level].moments)
        assert variance is not None
        metrics = _stationarity_metrics(suffix, level, variance)
        if isinstance(metrics, int):
            if selected is not None:
                selected["tested_suffixes"] = tested
                return selected
            return {"usable": 0, "exclusion": metrics, "tested_suffixes": tested}
        exact_delta = levels[level].exact_delta_counts
        exact_elapsed = levels[level].exact_elapsed_us
        selected = {
            "usable": 1,
            "suffix_len": suffix_len,
            "selected_level": level,
            "selected_blocks": levels[level].moments.count,
            "variance_of_mean_q2": variance,
            "selected_delta_counts": exact_delta,
            "selected_elapsed_us": exact_elapsed,
            "settled_rate_mean_q": _trunc_div(
                exact_delta * 1_000_000 * RATE_Q_SCALE, exact_elapsed
            ),
            "tested_suffixes": tested,
            **metrics,
        }
    return selected or {"usable": 0, "exclusion": 1, "tested_suffixes": tested}


def analyze_stationary_tail(observations: list[VelocityObservation]) -> dict[str, int]:
    """Reconstruct the firmware's nested-dyadic stationary tail from raw samples."""
    if not observations:
        raise ValueError("stationary-tail analysis requires raw observations")
    if any(item.elapsed_us <= 0 for item in observations):
        raise ValueError("elapsed_us must be positive")
    return _stationary_suffix(observations[-256:])


def _suffix_diagnostic(observations: list[VelocityObservation]) -> dict[str, object]:
    levels = [_Level() for _ in range(5)]
    for observation in observations:
        _observe(levels, observation)
    selection = _select_level(levels)
    if selection is None:
        return {"usable_plateau": False}
    variance = _q(levels[selection].moments)
    assert variance is not None
    block_observations = 1 << selection
    block_count = len(observations) // block_observations
    blocks = []
    for block_index in range(block_count):
        selected = observations[
            block_index * block_observations : (block_index + 1) * block_observations
        ]
        blocks.append(
            _Block(
                sum(item.delta_counts for item in selected),
                sum(item.elapsed_us for item in selected),
                len(selected),
            )
        )
    elapsed = 0
    x_values = []
    y_values = []
    for block in blocks:
        x_values.append(elapsed + block.elapsed_us // 2)
        elapsed += block.elapsed_us
        y_values.append(block.rate_q)
    count = len(blocks)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    denominator = count * sum(value * value for value in x_values) - sum_x * sum_x
    numerator = count * sum(x * y for x, y in zip(x_values, y_values, strict=True)) - sum_x * sum_y
    slope_per_us = _f32(_f32(numerator) / _f32(denominator))
    intercept = _f32(_f32(_f32(sum_y) - _f32(slope_per_us * _f32(sum_x))) / _f32(count))
    residual_sum = _f32(0.0)
    lag_product_sum = _f32(0.0)
    previous_residual: float | None = None
    for x, y in zip(x_values, y_values, strict=True):
        residual = _f32(_f32(y) - _f32(intercept + _f32(slope_per_us * _f32(x))))
        residual_sum = _f32(residual_sum + _f32(residual * residual))
        if previous_residual is not None:
            lag_product_sum = _f32(lag_product_sum + _f32(residual * previous_residual))
        previous_residual = residual
    multiplier = _f32(_confidence_multiplier(3))
    residual_variance = _f32(residual_sum / _f32(count - 2))
    blocking_variance = _f32(_f32(variance) * _f32(count))
    slope_variance = max(residual_variance, blocking_variance)
    sxx = _f32(_f32(denominator) / _f32(count))
    slope_per_second = _f32(slope_per_us * _f32(1_000_000))
    slope_half_width = _f32(
        _f32(multiplier * _f32(math.sqrt(_f32(slope_variance / sxx)))) * _f32(1_000_000)
    )
    slope_margin = _f32(slope_half_width - _f32(abs(slope_per_second)))
    lag_one = _f32(0.0) if residual_sum == 0 else _f32(lag_product_sum / residual_sum)
    lag_null = _f32(_f32(-1.0) / _f32(count))
    lag_half_width = _f32(multiplier / _f32(math.sqrt(_f32(count))))
    lag_margin = _f32(lag_half_width - _f32(abs(_f32(lag_one - lag_null))))
    exact_delta = levels[selection].exact_delta_counts
    exact_elapsed = levels[selection].exact_elapsed_us
    mean = _trunc_div(exact_delta * 1_000_000 * RATE_Q_SCALE, exact_elapsed)
    half_width = _confidence_half_width(variance, 3)
    return {
        "usable_plateau": True,
        "slope_passed": slope_margin >= 0,
        "lag_one_passed": lag_margin >= 0,
        "settled_rate_interval_q": [mean - half_width, mean + half_width],
        "slope_rejection_margin_q": _round_away(slope_margin),
        "lag_one_rejection_margin_q": _round_away(_f32(lag_margin * _f32(1_000_000))),
    }


def analyze_structured_boundary(
    observations: list[VelocityObservation],
) -> dict[str, object] | None:
    """Classify non-binding longer-tail structure without changing eligibility."""
    diagnostics = [
        _suffix_diagnostic(observations[-suffix_len:])
        for suffix_len in (64, 128, 256)
        if suffix_len <= len(observations)
    ]
    if not diagnostics:
        return None
    binding = diagnostics[0]
    if not binding["usable_plateau"] or (binding["slope_passed"] and binding["lag_one_passed"]):
        return None
    passing_suffix_mask = 0
    common: list[int] | None = None
    for index, diagnostic in enumerate(diagnostics):
        passed = bool(diagnostic.get("slope_passed") and diagnostic.get("lag_one_passed"))
        if diagnostic["usable_plateau"] and (index == 0 or passed):
            interval = list(diagnostic["settled_rate_interval_q"])
            common = (
                interval
                if common is None
                else [max(common[0], interval[0]), min(common[1], interval[1])]
            )
            if common[0] > common[1]:
                common = None
        if passed:
            passing_suffix_mask |= 1 << index
    if passing_suffix_mask & ~1 == 0 or common is None:
        return None
    return {
        "passing_suffix_mask": passing_suffix_mask,
        "common_rate_interval_q": common,
        "binding_slope_margin_q": int(binding["slope_rejection_margin_q"]),
        "binding_lag_one_margin_q": int(binding["lag_one_rejection_margin_q"]),
        "diagnostics": diagnostics,
    }


def _upper_bound_unsigned(value: int) -> tuple[int, int]:
    for shift in range(128):
        mantissa = (value + (1 << shift) - 1) >> shift
        if mantissa <= 0xFFFF_FFFF:
            return mantissa, shift
    return 0xFFFF_FFFF, 127


def _upper_bound_signed(value: int) -> tuple[int, int]:
    magnitude = abs(value)
    for shift in range(64):
        mantissa = (magnitude + (1 << shift) - 1) >> shift
        if mantissa <= 0x7FFF_FFFF:
            return (-mantissa if value < 0 else mantissa), shift
    return (-0x7FFF_FFFF if value < 0 else 0x7FFF_FFFF), 63


def _encode_rate_values(values: tuple[int, int, int, int, int]) -> tuple[tuple[int, ...], int]:
    for shift in range(64):
        divisor = 1 << shift
        encoded = []
        for index, value in enumerate(values):
            quotient = _trunc_div(value, divisor)
            remainder = value - quotient * divisor
            if index in (0, 3) and remainder and value < 0:
                quotient -= 1
            elif index in (2, 4) and remainder and value > 0:
                quotient += 1
            elif index == 1 and remainder:
                quotient += 1 if value > 0 else -1
            encoded.append(quotient)
        if all(-(1 << 31) <= value < (1 << 31) for value in encoded):
            return tuple(encoded), shift
    return (-(1 << 31), 0, (1 << 31) - 1, -(1 << 31), (1 << 31) - 1), 63


def _floor_div(numerator: int, denominator: int) -> int:
    return numerator // denominator


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _mrev_s_to_rpm_nearest(value: int) -> int:
    if value <= 0:
        raise ValueError("planned velocity must be positive")
    return (value * 60 + 500) // 1_000


_TWO_SIDED_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
}


def _diagnostic_block_means(
    observations: list[VelocityObservation],
    field_name: str,
    counts_per_rev: int,
) -> dict[str, object]:
    blocks: dict[int, list[tuple[Fraction, Fraction]]] = {}
    for observation in observations:
        value = getattr(observation, field_name)
        if value is None or observation.direction == 0 or observation.elapsed_us <= 0:
            continue
        direction = observation.direction
        chip_rpm = Fraction(int(value) * direction)
        encoder_rpm = Fraction(
            observation.delta_counts * direction * 60_000_000,
            counts_per_rev * observation.elapsed_us,
        )
        blocks.setdefault(observation.stroke_index, []).append((chip_rpm, encoder_rpm))
    block_means: list[tuple[Fraction, Fraction]] = []
    for samples in blocks.values():
        count = len(samples)
        block_means.append(
            (
                sum((sample[0] for sample in samples), Fraction()) / count,
                sum((sample[1] for sample in samples), Fraction()) / count,
            )
        )
    if not block_means:
        return {
            "block_count": 0,
            "chip_mean_rpm": None,
            "encoder_mean_rpm": None,
            "difference_mean_rpm": None,
            "difference_low_rpm": None,
            "difference_high_rpm": None,
        }
    chip_mean = sum((block[0] for block in block_means), Fraction()) / len(block_means)
    encoder_mean = sum((block[1] for block in block_means), Fraction()) / len(block_means)
    differences = [block[0] - block[1] for block in block_means]
    difference_mean = sum(differences, Fraction()) / len(differences)
    if len(differences) < 2:
        difference_low = None
        difference_high = None
    else:
        variance = sum((float(value - difference_mean) ** 2 for value in differences)) / (
            len(differences) - 1
        )
        degrees_freedom = len(differences) - 1
        multiplier = _TWO_SIDED_T_95.get(degrees_freedom, 2.131)
        half_width = multiplier * math.sqrt(variance / len(differences))
        difference_low = float(difference_mean) - half_width
        difference_high = float(difference_mean) + half_width
    return {
        "block_count": len(block_means),
        "chip_mean_rpm": float(chip_mean),
        "encoder_mean_rpm": float(encoder_mean),
        "difference_mean_rpm": float(difference_mean),
        "difference_low_rpm": difference_low,
        "difference_high_rpm": difference_high,
    }


def _diagnostic_audit(
    observations: list[VelocityObservation],
    observation_records: dict[tuple[int, int], dict[str, object]],
    counts_per_rev: int,
    usable_torque_limit: int,
) -> dict[str, object]:
    max_reconstructed = 0
    max_observed = 0
    peak_violations = 0
    torque_sample_count = 0
    for observation in observations:
        max_observed = max(max_observed, observation.peak_torque_target_abs or 0)
        if observation.pid_velocity_actual is None:
            continue
        rung_index, slot = divmod(observation.stroke_index, 4)
        record = observation_records.get((rung_index, slot))
        if record is None:
            continue
        torque_sample_count += 1
        reconstructed = _ceil_div(
            abs(
                int(record["velocity_p"])
                * (int(record["target_velocity_rpm"]) - observation.pid_velocity_actual)
            ),
            256,
        )
        max_reconstructed = max(max_reconstructed, reconstructed)
        if reconstructed > (observation.peak_torque_target_abs or 0) + 1:
            peak_violations += 1
    pid_encoder = _diagnostic_block_means(observations, "pid_velocity_actual", counts_per_rev)
    pptm_encoder = _diagnostic_block_means(observations, "actual_velocity_pptm", counts_per_rev)
    complete = (
        int(pid_encoder["block_count"]) >= 2
        and int(pptm_encoder["block_count"]) >= 2
        and torque_sample_count > 0
    )
    return {
        "complete": complete,
        "pid_encoder": pid_encoder,
        "pptm_encoder": pptm_encoder,
        "torque": {
            "sample_count": torque_sample_count,
            "max_reconstructed_abs": max_reconstructed,
            "max_observed_peak_abs": max_observed,
            "peak_plus_one_violations": peak_violations,
            "within_observed_peak_plus_one": torque_sample_count > 0 and peak_violations == 0,
            "below_usable_limit": torque_sample_count > 0
            and max_reconstructed < usable_torque_limit
            and max_observed < usable_torque_limit,
        },
    }


def _torque_evidence_matches_terminal(
    terminal: dict[str, object], torque_audit: dict[str, object]
) -> bool:
    if not bool(torque_audit["within_observed_peak_plus_one"]):
        return False
    return bool(torque_audit["below_usable_limit"]) or int(terminal["cause"]) == 4


def _disturbance(
    *,
    p: int,
    target_velocity_rpm: int,
    delta_counts: int,
    elapsed_us: int,
    half_width_rate_q: int,
    variance_rate_q2: int,
    counts_per_rev: int,
) -> tuple[int, int, int, int]:
    target_q16 = target_velocity_rpm * 65_536
    target_numerator = target_q16 * counts_per_rev * elapsed_us
    encoder_numerator = delta_counts * 60_000_000 * 65_536
    error_numerator = target_numerator - encoder_numerator
    disturbance_numerator = p * error_numerator
    disturbance_denominator = 256 * counts_per_rev * elapsed_us
    estimate = _trunc_div(disturbance_numerator, disturbance_denominator)
    common_denominator = disturbance_denominator * RATE_Q_SCALE
    common_mean = disturbance_numerator * RATE_Q_SCALE
    common_half_width = p * half_width_rate_q * 60 * 65_536 * elapsed_us
    low = _floor_div(common_mean - common_half_width, common_denominator)
    high = _ceil_div(common_mean + common_half_width, common_denominator)
    scale_numerator = p * 60 * 65_536
    variance_numerator = variance_rate_q2 * scale_numerator * scale_numerator
    variance_denominator = (256 * counts_per_rev * RATE_Q_SCALE) ** 2
    variance = _ceil_div(variance_numerator, variance_denominator)
    return estimate, low, high, variance


def _compare_fields(
    mismatches: list[str],
    label: str,
    firmware: dict[str, object],
    offline: dict[str, int],
    fields: tuple[str, ...],
) -> None:
    for field_name in fields:
        if int(firmware[field_name]) != int(offline[field_name]):
            mismatches.append(
                f"{label} {field_name}: firmware={firmware[field_name]} offline={offline[field_name]}"
            )


def _reconstruct_observation(
    raw: list[VelocityObservation],
    record: dict[str, object],
    plan: dict[str, object],
    terminal: dict[str, object],
) -> dict[str, int]:
    stationarity = _stationary_suffix(raw[-256:])
    p = int(record["velocity_p"])
    target_velocity_rpm = int(record["target_velocity_rpm"])
    predicted = max(
        (p * abs(target_velocity_rpm) + 255) // 256,
        max((item.peak_torque_target_abs or 0) for item in raw),
    )
    if not stationarity["usable"]:
        classification = 4 + stationarity["exclusion"]
        return {
            "classification": classification,
            "predicted_torque_target_abs": predicted,
            **stationarity,
            # The wire summary retains the exclusion class but has no usable
            # suffix metrics to encode.
            "tested_suffixes": 0,
        }
    family_size = int(plan["rung_count"]) * int(plan["observations_per_direction"])
    half_width = _confidence_half_width(stationarity["variance_of_mean_q2"], family_size)
    mean = stationarity["settled_rate_mean_q"]
    rate_low, rate_high = mean - half_width, mean + half_width
    direction = 1 if int(record["slot"]) % 2 == 0 else -1
    target_rate_q_numerator = (
        target_velocity_rpm * int(terminal["encoder_counts_per_rev"]) * RATE_Q_SCALE
    )
    deficit_low = _floor_div(target_rate_q_numerator - rate_high * 60, 60)
    deficit_high = _ceil_div(target_rate_q_numerator - rate_low * 60, 60)
    disturbance, disturbance_low, disturbance_high, disturbance_variance = _disturbance(
        p=p,
        target_velocity_rpm=target_velocity_rpm,
        delta_counts=stationarity["selected_delta_counts"],
        elapsed_us=stationarity["selected_elapsed_us"],
        half_width_rate_q=half_width,
        variance_rate_q2=stationarity["variance_of_mean_q2"],
        counts_per_rev=int(terminal["encoder_counts_per_rev"]),
    )
    wrong_direction = rate_high < 0 if direction > 0 else rate_low > 0
    moving = rate_low > 0 if direction > 0 else rate_high < 0
    deficit_resolved = deficit_low > 0 if direction > 0 else deficit_high < 0
    if predicted >= int(plan["hard_torque_limit"]):
        classification = 10
    elif wrong_direction:
        classification = 9
    elif not moving:
        classification = 1
    elif predicted >= int(plan["usable_torque_limit"]):
        classification = 3
    elif not deficit_resolved:
        classification = 2
    else:
        classification = 0
    rate_values, rate_shift = _encode_rate_values(
        (rate_low, mean, rate_high, deficit_low, deficit_high)
    )
    delta_mantissa, delta_shift = _upper_bound_unsigned(abs(stationarity["selected_delta_counts"]))
    elapsed_mantissa, elapsed_shift = _upper_bound_unsigned(stationarity["selected_elapsed_us"])
    variance_mantissa, variance_shift = _upper_bound_unsigned(stationarity["variance_of_mean_q2"])
    slope_mantissa, slope_shift = _upper_bound_signed(stationarity["slope_q"])
    slope_hw_mantissa, slope_hw_shift = _upper_bound_unsigned(stationarity["slope_half_width_q"])
    residual_mantissa, residual_shift = _upper_bound_unsigned(stationarity["residual_magnitude_q2"])
    disturbance_variance_mantissa, disturbance_variance_shift = _upper_bound_unsigned(
        disturbance_variance
    )
    return {
        **stationarity,
        "classification": classification,
        "predicted_torque_target_abs": predicted,
        "delta_sign": int(stationarity["selected_delta_counts"] < 0),
        "delta_mantissa": delta_mantissa,
        "delta_shift": delta_shift,
        "elapsed_mantissa": elapsed_mantissa,
        "elapsed_shift": elapsed_shift,
        "rate_low": rate_values[0],
        "rate_mean": rate_values[1],
        "rate_high": rate_values[2],
        "deficit_low": rate_values[3],
        "deficit_high": rate_values[4],
        "rate_shift": rate_shift,
        "rate_variance_mantissa": variance_mantissa,
        "rate_variance_shift": variance_shift,
        "slope_mantissa": slope_mantissa,
        "slope_shift": slope_shift,
        "slope_half_width_mantissa": slope_hw_mantissa,
        "slope_half_width_shift": slope_hw_shift,
        "lag_one_q": stationarity["lag_one_q"],
        "lag_one_half_width_q": stationarity["lag_one_half_width_q"],
        "residual_mantissa": residual_mantissa,
        "residual_shift": residual_shift,
        "tested_suffixes": stationarity["tested_suffixes"],
        "disturbance_q16": disturbance,
        "disturbance_low_q16": disturbance_low,
        "disturbance_high_q16": disturbance_high,
        "disturbance_variance_q2": disturbance_variance,
        "disturbance_variance_mantissa": disturbance_variance_mantissa,
        "disturbance_variance_shift": disturbance_variance_shift,
        "settled_rate_low_q": rate_low,
        "settled_rate_high_q": rate_high,
    }


def _rung_class(first: dict[str, int], second: dict[str, int]) -> tuple[int, int, int, bool]:
    first_class, second_class = first["classification"], second["classification"]
    moving_classes = {0, 2, 3}
    reproduced = first_class in moving_classes and second_class in moving_classes
    low = max(first.get("disturbance_low_q16", 0), second.get("disturbance_low_q16", 0))
    high = min(first.get("disturbance_high_q16", 0), second.get("disturbance_high_q16", 0))
    if first_class == second_class == 0:
        return (0 if low <= high else 4), low, high, reproduced
    if first_class == second_class == 1:
        return 1, 0, 0, reproduced
    if first_class == second_class == 2:
        return 2, 0, 0, reproduced
    if first_class == second_class == 3:
        return 3, 0, 0, reproduced
    if first_class >= 9 or second_class >= 9:
        return 7, 0, 0, reproduced
    if 4 <= first_class <= 8 and 4 <= second_class <= 8:
        return 6, 0, 0, reproduced
    return 5, 0, 0, reproduced


def _new_band() -> dict[str, object]:
    return {
        "first_moving": 0,
        "p_low": 0,
        "p_high": 0,
        "closure": 1,
        "common": None,
        "eligible_rungs": 0,
        "eligible_observations": 0,
        "left_censored": False,
        "closed": False,
        "weighted_point": 0,
        "weighted_low": 0,
        "weighted_high": 0,
        "weight_sum": 0,
    }


def _observe_band(
    band: dict[str, object],
    rung_index: int,
    p: int,
    rung_class: int,
    low: int,
    high: int,
    reproduced: bool,
    observations: tuple[dict[str, int], dict[str, int]],
) -> None:
    if band["closed"]:
        return
    if not band["first_moving"]:
        if rung_class == 1:
            return
        if not reproduced:
            return
        band["first_moving"] = p
        band["left_censored"] = rung_index == 0
    if rung_class != 0:
        band["closure"] = {2: 2, 3: 5, 7: 6}.get(rung_class, 3)
        band["closed"] = True
        return
    common = band["common"]
    next_common = (low, high) if common is None else (max(common[0], low), min(common[1], high))
    if next_common[0] > next_common[1]:
        band["closure"] = 4
        band["closed"] = True
        return
    band["common"] = next_common
    if not band["p_low"]:
        band["p_low"] = p
    band["p_high"] = p
    band["eligible_rungs"] += 1
    for observation in observations:
        variance = max(observation["disturbance_variance_q2"], 1)
        weight = max((1 << 32) // variance, 1)
        band["weighted_point"] += observation["disturbance_q16"] * weight
        band["weighted_low"] += observation["disturbance_low_q16"] * weight
        band["weighted_high"] += observation["disturbance_high_q16"] * weight
        band["weight_sum"] += weight
        band["eligible_observations"] += 1


def _finish_band(band: dict[str, object]) -> dict[str, int]:
    if not band["closed"] and band["first_moving"]:
        band["closure"] = 0
    weight_sum = int(band["weight_sum"])
    pooled = _trunc_div(int(band["weighted_point"]), weight_sum) if weight_sum else 0
    pooled_low = _floor_div(int(band["weighted_low"]), weight_sum) if weight_sum else 0
    pooled_high = _ceil_div(int(band["weighted_high"]), weight_sum) if weight_sum else 0
    sufficient = bool(band["first_moving"] and int(band["eligible_rungs"]) >= 3 and weight_sum)
    common = band["common"] or (0, 0)
    flags = int(sufficient) | (int(bool(band["left_censored"])) << 1)
    flags |= (int(band["common"] is not None) << 2) | (int(weight_sum > 0) << 3)
    return {
        "first_moving": int(band["first_moving"]),
        "p_low": int(band["p_low"]),
        "p_high": int(band["p_high"]),
        "closure": int(band["closure"]),
        "common_low_q16": common[0],
        "common_high_q16": common[1],
        "pooled_q16": pooled,
        "pooled_low_q16": pooled_low,
        "pooled_high_q16": pooled_high,
        "eligible_rungs": int(band["eligible_rungs"]),
        "eligible_observations": int(band["eligible_observations"]),
        "flags": flags,
        "sufficient": int(sufficient),
    }


def _compact_variance(value: int) -> int:
    """Mirror firmware's outward-rounded u16 variance compaction."""
    for shift in range(49):
        rounded = (value + (1 << shift) - 1) >> shift
        if rounded <= 0xFFFF:
            return rounded << shift
    raise ValueError("variance does not fit the firmware compact domain")


def _pool_region_observations(observations: list[dict[str, int]]) -> dict[str, object]:
    weighted_point = 0
    weighted_low = 0
    weighted_high = 0
    weight_sum = 0
    variance_floor_observations = 0
    for observation in observations:
        variance = _compact_variance(int(observation.get("disturbance_variance_q2", 0)))
        variance_floor_observations += int(variance == 0)
        weight = max((1 << 32) // max(variance, 1), 1)
        weighted_point += int(observation["disturbance_q16"]) * weight
        weighted_low += int(observation["disturbance_low_q16"]) * weight
        weighted_high += int(observation["disturbance_high_q16"]) * weight
        weight_sum += weight
    if not weight_sum:
        return {
            "pooled_q16": 0,
            "pooled_interval": [0, 0],
            "variance_floor_observations": 0,
        }
    return {
        "pooled_q16": _trunc_div(weighted_point, weight_sum),
        "pooled_interval": [
            _floor_div(weighted_low, weight_sum),
            _ceil_div(weighted_high, weight_sum),
        ],
        "variance_floor_observations": variance_floor_observations,
    }


def _compact_signed_rate(value: int, *, upper: bool) -> int:
    """Mirror firmware's conservative i16 rate compaction and decode."""
    for shift in range(49):
        divisor = 1 << shift
        quotient = _trunc_div(value, divisor)
        remainder = value - quotient * divisor
        if upper and value > 0 and remainder:
            quotient += 1
        elif not upper and value < 0 and remainder:
            quotient -= 1
        if -(1 << 15) <= quotient < (1 << 15):
            return quotient << shift
    raise ValueError("signed rate does not fit the firmware compact domain")


def _close_directional_region(
    active: dict[str, object], closure: str, boundary_rung: int | None
) -> dict[str, object]:
    member_indexes = list(active["member_indexes"])
    observations = list(active["observations"])
    rate_means = list(active["rate_means"])
    rate_lows = list(active["rate_lows"])
    rate_highs = list(active["rate_highs"])
    member_mask = sum(1 << index for index in member_indexes)
    result = {
        "member_mask": member_mask,
        "member_indexes": member_indexes,
        "selected_ps": list(active["selected_ps"]),
        "first_rung": member_indexes[0],
        "last_rung": member_indexes[-1],
        "member_count": len(member_indexes),
        "kind": "valid" if len(member_indexes) >= 3 else "fragment",
        "common_interval": list(active["common_interval"]),
        "closure": closure,
        "boundary_rung": boundary_rung,
        "tested_intersection": list(active["common_interval"]),
        "settled_rate_mean_min_q": min(rate_means, default=0),
        "settled_rate_mean_max_q": max(rate_means, default=0),
        "settled_rate_envelope_low_q": min(rate_lows, default=0),
        "settled_rate_envelope_high_q": max(rate_highs, default=0),
    }
    result.update(_pool_region_observations(observations))
    return result


def _segment_directional_regions(rungs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the firmware's causal close-and-reseed directional fold."""
    regions: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    for rung in rungs:
        rung_index = int(rung["rung_index"])
        classification = int(rung["class"])
        if classification != 0:
            if active is not None:
                closure = (
                    "current_headroom"
                    if classification == 3
                    else "safety_terminus"
                    if classification == 7
                    else "ineligible_gap"
                )
                regions.append(_close_directional_region(active, closure, rung_index))
                active = None
            continue
        interval = [int(rung["low"]), int(rung["high"])]
        if active is None:
            active = {
                "member_indexes": [rung_index],
                "selected_ps": [int(rung["velocity_p"])],
                "common_interval": interval,
                "observations": list(rung.get("observations", [])),
                "rate_means": list(rung.get("rate_means", [])),
                "rate_lows": list(rung.get("rate_lows", [])),
                "rate_highs": list(rung.get("rate_highs", [])),
            }
            continue
        member_indexes = list(active["member_indexes"])
        contiguous = rung_index == member_indexes[-1] + 1
        common = list(active["common_interval"])
        intersection = [max(common[0], interval[0]), min(common[1], interval[1])]
        if contiguous and intersection[0] <= intersection[1]:
            member_indexes.append(rung_index)
            active["member_indexes"] = member_indexes
            active["selected_ps"] = [*list(active["selected_ps"]), int(rung["velocity_p"])]
            active["common_interval"] = intersection
            for key, source in (
                ("observations", "observations"),
                ("rate_means", "rate_means"),
                ("rate_lows", "rate_lows"),
                ("rate_highs", "rate_highs"),
            ):
                active[key] = [*list(active[key]), *list(rung.get(source, []))]
            continue
        regions.append(
            _close_directional_region(
                active,
                "disjoint_interval" if contiguous else "ineligible_gap",
                rung_index,
            )
        )
        active = {
            "member_indexes": [rung_index],
            "selected_ps": [int(rung["velocity_p"])],
            "common_interval": interval,
            "observations": list(rung.get("observations", [])),
            "rate_means": list(rung.get("rate_means", [])),
            "rate_lows": list(rung.get("rate_lows", [])),
            "rate_highs": list(rung.get("rate_highs", [])),
        }
    if active is not None:
        regions.append(_close_directional_region(active, "end_of_ladder", None))
    return regions


def _joint_operable_regions(rungs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Find maximal contiguous rungs operable in both signed directions."""
    regions: list[dict[str, object]] = []
    active: list[int] = []
    for rung in rungs:
        rung_index = int(rung["rung_index"])
        operable = bool(rung["forward_operable"]) and bool(rung["reverse_operable"])
        if operable and (not active or rung_index == active[-1] + 1):
            active.append(rung_index)
            continue
        if active:
            classes = (int(rung.get("forward_class", 0)), int(rung.get("reverse_class", 0)))
            closure = (
                "safety_terminus"
                if 7 in classes
                else "current_headroom"
                if 3 in classes
                else "ineligible_gap"
            )
            regions.append(
                {
                    "member_mask": sum(1 << index for index in active),
                    "first_rung": active[0],
                    "last_rung": active[-1],
                    "member_count": len(active),
                    "closure": closure,
                }
            )
            active = [rung_index] if operable else []
    if active:
        regions.append(
            {
                "member_mask": sum(1 << index for index in active),
                "first_rung": active[0],
                "last_rung": active[-1],
                "member_count": len(active),
                "closure": "end_of_ladder",
            }
        )
    return regions


def _ratio_margin(p: int, p_start: int, p_top: int) -> Fraction:
    return Fraction(p, p_start) if p * p <= p_start * p_top else Fraction(p_top, p)


def _nominate_joint_region(
    regions: list[dict[str, object]], gains: list[int], p_start: int, p_top: int
) -> dict[str, object]:
    """Select the firmware-owned lower-middle nomination and deterministic ties."""
    candidates = []
    for region in regions:
        first = int(region["first_rung"])
        count = int(region["member_count"])
        rung_index = first + (count - 1) // 2
        p = gains[rung_index]
        candidates.append(
            {
                **region,
                "rung_index": rung_index,
                "p": p,
                "distance_to_start_q16": (p << 16) // max(p_start, 1),
                "distance_to_top_q16": (p_top << 16) // max(p, 1),
                "reduced_margin": count == 1,
                "_margin": _ratio_margin(p, p_start, p_top),
            }
        )
    if not candidates:
        return {}
    chosen = candidates[0]
    for candidate in candidates[1:]:
        if int(candidate["member_count"]) > int(chosen["member_count"]) or (
            int(candidate["member_count"]) == int(chosen["member_count"])
            and (
                candidate["_margin"] > chosen["_margin"]
                or (
                    candidate["_margin"] == chosen["_margin"]
                    and int(candidate["p"]) < int(chosen["p"])
                )
            )
        ):
            chosen = candidate
    return {key: value for key, value in chosen.items() if key != "_margin"}


def _select_directional_region(
    regions: list[dict[str, object]], nominated_rung: int, gains: list[int]
) -> dict[str, object]:
    """Select full coverage, otherwise the nearest lower-first directional region."""
    selected: dict[str, object] | None = None
    for region in regions:
        mask = int(region["member_mask"])
        first = (mask & -mask).bit_length() - 1
        last = mask.bit_length() - 1
        if first <= nominated_rung <= last:
            coverage, distance, edge = "full", 0, nominated_rung
        elif nominated_rung < first:
            coverage, distance, edge = "transferred", first - nominated_rung, first
        else:
            coverage, distance, edge = "transferred", last - nominated_rung, last
        proposed = {
            **region,
            "coverage": coverage,
            "signed_rung_distance": distance,
            "gain_ratio": [
                max(gains[edge], gains[nominated_rung]),
                min(gains[edge], gains[nominated_rung]),
            ]
            if coverage == "transferred"
            else [1, 1],
        }
        if selected is None:
            selected = proposed
            continue
        if (
            coverage == "full"
            and selected["coverage"] != "full"
            or abs(distance) < abs(int(selected["signed_rung_distance"]))
            or (
                abs(distance) == abs(int(selected["signed_rung_distance"]))
                and first
                < ((int(selected["member_mask"]) & -int(selected["member_mask"])).bit_length() - 1)
            )
        ):
            selected = proposed
    return selected or {}


def _compare_stage_b_candidates(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Mirror RAM-only two-run reproduction semantics."""
    if int(previous["plan_digest"]) != int(current["plan_digest"]):
        return {"outcome": "plan_mismatch", "replacement": previous}
    intervals_overlap = all(
        max(int(left[0]), int(right[0])) <= min(int(left[1]), int(right[1]))
        for left, right in zip(previous["intervals"], current["intervals"], strict=True)
    )
    if (
        int(previous["nominated_p"]) == int(current["nominated_p"])
        and list(previous["memberships"]) == list(current["memberships"])
        and intervals_overlap
    ):
        return {"outcome": "complete", "replacement": None}
    return {"outcome": "reproduction_mismatch", "replacement": current}


def _stage_b_record_identity(value: dict[str, object]) -> tuple[object, ...]:
    """Return the causal identity of one canonical Stage B record."""
    kind = str(value["record_kind"])
    if kind in ("observation", "structured_boundary"):
        return kind, int(value["rung_index"]), int(value["slot"])
    if kind == "rung":
        return kind, int(value["rung_index"])
    if kind == "directional_region":
        return kind, int(value["direction"]), int(value["member_mask"])
    if kind == "joint_region":
        return kind, int(value["member_mask"])
    return (kind,)


def _compare_stage_b_causal_order(
    records: list[dict[str, object]], replay: dict[str, object]
) -> str:
    """Compare the stream with the firmware's independently replayed causal order."""
    expected: list[tuple[object, ...]] = [("plan",)]
    boundary_keys = {
        tuple(int(part) for part in str(key).split("/"))
        for key in dict(replay["structured_boundaries"])
    }
    directional_at_boundary: dict[int, list[tuple[object, ...]]] = {}
    directional_at_end: list[tuple[object, ...]] = []
    for direction, regions in enumerate(replay["directions"]):
        for region in regions:
            token = ("directional_region", direction, int(region["member_mask"]))
            boundary = region.get("boundary_rung")
            if boundary is None:
                directional_at_end.append(token)
            else:
                directional_at_boundary.setdefault(int(boundary), []).append(token)
    joint_at_boundary: dict[int, list[tuple[object, ...]]] = {}
    joint_at_end: list[tuple[object, ...]] = []
    for region in replay["joint_regions"]:
        token = ("joint_region", int(region["member_mask"]))
        if region["closure"] == "end_of_ladder":
            joint_at_end.append(token)
        else:
            joint_at_boundary.setdefault(int(region["last_rung"]) + 1, []).append(token)

    reported_rungs = {
        int(value["rung_index"]) for value in records if value["record_kind"] == "rung"
    }
    observation_slots = {
        (int(value["rung_index"]), int(value["slot"]))
        for value in records
        if value["record_kind"] == "observation"
    }
    rung_indexes = sorted(reported_rungs | {rung for rung, _slot in observation_slots})
    for rung_index in rung_indexes:
        for slot in range(4):
            if (rung_index, slot) not in observation_slots:
                continue
            expected.append(("observation", rung_index, slot))
            if (rung_index, slot) in boundary_keys:
                expected.append(("structured_boundary", rung_index, slot))
        expected.extend(directional_at_boundary.get(rung_index, ()))
        expected.extend(joint_at_boundary.get(rung_index, ()))
        if rung_index in reported_rungs:
            expected.append(("rung", rung_index))
    expected.extend(directional_at_end)
    expected.extend(joint_at_end)
    if replay["nomination"]:
        expected.append(("handoff",))
    if any(value["record_kind"] == "reproduction" for value in records):
        expected.append(("reproduction",))
    expected.append(("stage_b_terminal",))

    observed = [_stage_b_record_identity(value) for value in records]
    if observed == expected:
        return ""
    return f"Stage B causal order differs: firmware={observed} offline={expected}"


def _stage_b_candidate_from_terminal(
    audit: dict[str, object], *, plan_digest: int
) -> dict[str, object]:
    terminal = audit["terminal"]
    return {
        "plan_digest": plan_digest,
        "nominated_p": int(terminal["nominated_p"]),
        "memberships": list(terminal["selected_memberships"]),
        "intervals": [list(value) for value in terminal["selected_intervals"]],
    }


def audit_stage_b_reproduction_pair(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Verify a current reproduction record against its prior capture."""
    mismatches: list[str] = []
    reproduction = current.get("reproduction")
    if not isinstance(reproduction, dict):
        return {"matches_firmware": False, "mismatches": ["current capture lacks reproduction"]}
    previous_plan = int(previous["plan_identity"])
    current_plan = int(current["plan_identity"])
    previous_candidate = _stage_b_candidate_from_terminal(previous, plan_digest=previous_plan)
    current_candidate = _stage_b_candidate_from_terminal(current, plan_digest=current_plan)
    reported_previous_digest = int(reproduction["previous_digest_low"]) | (
        int(reproduction["previous_digest_high"]) << 32
    )
    reported_current_digest = int(reproduction["current_digest_low"]) | (
        int(reproduction["current_digest_high"]) << 32
    )
    if reported_previous_digest != int(previous["reported_digest"]):
        mismatches.append("reproduction previous digest differs from prior terminal")
    if reported_current_digest != int(current["pre_reproduction_digest"]):
        mismatches.append("reproduction current digest differs from current prefix")
    for name, expected, reported in (
        (
            "previous nomination",
            previous_candidate["nominated_p"],
            reproduction["previous_nominated_p"],
        ),
        (
            "current nomination",
            current_candidate["nominated_p"],
            reproduction["current_nominated_p"],
        ),
        (
            "previous memberships",
            previous_candidate["memberships"],
            reproduction["previous_memberships"],
        ),
        (
            "current memberships",
            current_candidate["memberships"],
            reproduction["current_memberships"],
        ),
        (
            "previous intervals",
            previous_candidate["intervals"],
            reproduction["previous_intervals"],
        ),
        (
            "current intervals",
            current_candidate["intervals"],
            reproduction["current_intervals"],
        ),
    ):
        if reported != expected:
            mismatches.append(f"reproduction {name} differs from capture")
    comparison = _compare_stage_b_candidates(previous_candidate, current_candidate)
    expected_outcome = 1 if comparison["outcome"] == "complete" else 2
    if int(reproduction["outcome"]) != expected_outcome:
        mismatches.append("reproduction outcome differs from independent comparison")
    terminal_outcome = int(current["terminal"]["outcome"])
    expected_terminal_outcome = (
        STAGE_B_OUTCOME_COMPLETE if expected_outcome == 1 else STAGE_B_OUTCOME_INCONCLUSIVE
    )
    if terminal_outcome != expected_terminal_outcome:
        mismatches.append("terminal outcome differs from reproduction comparison")
    expected_cause = 0 if expected_outcome == 1 else STAGE_B_CAUSE_REPRODUCTION_MISMATCH
    if int(current["terminal"].get("cause", 0)) != expected_cause:
        mismatches.append("terminal cause differs from reproduction comparison")
    return {"matches_firmware": not mismatches, "mismatches": mismatches}


def replay_stage_b_regions(
    raw_observations: list[VelocityObservation],
    evidence_records: list[tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    """Replay proportional-sweep regions from raw observation evidence."""
    plans = [value for _raw, value in evidence_records if value["record_kind"] == "plan"]
    terminals = [
        value
        for _raw, value in evidence_records
        if value["record_kind"] in ("terminal", "stage_b_terminal")
    ]
    if len(plans) != 1 or len(terminals) != 1:
        raise ValueError("one plan and one terminal are required for Stage B replay")
    plan, terminal = plans[0], terminals[0]
    by_stroke: dict[int, list[VelocityObservation]] = {}
    for observation in raw_observations:
        by_stroke.setdefault(observation.stroke_index, []).append(observation)
    records = {
        (int(value["rung_index"]), int(value["slot"])): value
        for _raw, value in evidence_records
        if value["record_kind"] == "observation"
    }
    reconstructed: dict[tuple[int, int], dict[str, int]] = {}
    for key, record in records.items():
        raw = by_stroke.get(key[0] * 4 + key[1])
        if not raw:
            continue
        reconstructed[key] = _reconstruct_observation(raw, record, plan, terminal)
    directional_inputs: list[list[dict[str, object]]] = [[], []]
    joint_inputs = []
    gains = []
    for rung_index in range(int(plan["rung_count"])):
        if any((rung_index, slot) not in reconstructed for slot in range(4)):
            break
        p = int(records[(rung_index, 0)]["velocity_p"])
        gains.append(p)
        operable = []
        for direction_index, slots in ((0, (0, 2)), (1, (1, 3))):
            observations = [reconstructed[(rung_index, slot)] for slot in slots]
            classification, low, high, _moving = _rung_class(*observations)
            means = [item.get("settled_rate_mean_q", 0) for item in observations]
            directional_inputs[direction_index].append(
                {
                    "rung_index": rung_index,
                    "velocity_p": p,
                    "class": classification,
                    "low": low,
                    "high": high,
                    "observations": observations,
                    "rate_means": [
                        _compact_signed_rate(min(means), upper=False),
                        _compact_signed_rate(max(means), upper=True),
                    ],
                    "rate_lows": [
                        _compact_signed_rate(item.get("settled_rate_low_q", 0), upper=False)
                        for item in observations
                    ],
                    "rate_highs": [
                        _compact_signed_rate(item.get("settled_rate_high_q", 0), upper=True)
                        for item in observations
                    ],
                }
            )
            operable.append(all(item["classification"] in (0, 2) for item in observations))
        joint_inputs.append(
            {
                "rung_index": rung_index,
                "forward_operable": operable[0],
                "reverse_operable": operable[1],
                "forward_class": int(directional_inputs[0][-1]["class"]),
                "reverse_class": int(directional_inputs[1][-1]["class"]),
            }
        )
    directional = [_segment_directional_regions(items) for items in directional_inputs]
    joint = _joint_operable_regions(joint_inputs)
    nomination = _nominate_joint_region(
        joint,
        gains,
        int(plan.get("p_start", gains[0] if gains else 1)),
        int(plan.get("p_top", gains[-1] if gains else 1)),
    )
    nominated_rung = int(nomination["rung_index"]) if nomination else 0
    selected = [
        _select_directional_region(
            [region for region in regions if region["kind"] == "valid"],
            nominated_rung,
            gains,
        )
        for regions in directional
    ]
    structured_boundaries = {}
    for key, observation in reconstructed.items():
        if not 4 <= int(observation["classification"]) <= 8:
            continue
        boundary = analyze_structured_boundary(by_stroke.get(key[0] * 4 + key[1], []))
        if boundary is not None:
            structured_boundaries[f"{key[0]}/{key[1]}"] = boundary
    return {
        "directions": directional,
        "joint_regions": joint,
        "nomination": nomination,
        "nominated_p": int(nomination.get("p", 0)),
        "selected": selected,
        "coverage": [item.get("coverage") for item in selected],
        "gains": gains,
        "structured_boundaries": structured_boundaries,
    }


def _audit_stage_b_velocity_sweep(
    raw_observations: list[VelocityObservation],
    evidence_records: list[tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    """Audit the proportional-sweep canonical record stream against an independent replay."""
    plans = [value for _raw, value in evidence_records if value["record_kind"] == "plan"]
    terminals = [
        value for _raw, value in evidence_records if value["record_kind"] == "stage_b_terminal"
    ]
    if len(plans) != 1 or len(terminals) != 1:
        return {
            "run_sequence": int((plans or terminals or [{}])[0].get("run_sequence", 0)),
            "plan": dict(plans[0]) if len(plans) == 1 else None,
            "terminal": dict(terminals[0]) if len(terminals) == 1 else None,
            "complete": False,
            "matches_firmware": False,
            "mismatches": ["partial Stage B capture: one plan and terminal are required"],
            "caveats": ["partial Stage B capture"],
        }
    plan, terminal = plans[0], terminals[0]
    nonterminal = [
        (raw, value)
        for raw, value in evidence_records
        if value["record_kind"] != "stage_b_terminal"
    ]
    digest = FNV1A64_OFFSET
    pre_reproduction_digest: int | None = None
    for raw, _value in nonterminal:
        if _value["record_kind"] == "reproduction":
            pre_reproduction_digest = digest
        for byte in raw:
            digest ^= byte
            digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    reported_digest = int(terminal["digest_low"]) | (int(terminal["digest_high"]) << 32)
    mismatches: list[str] = []
    if digest != reported_digest:
        mismatches.append(
            f"terminal digest: firmware=0x{reported_digest:016x} offline=0x{digest:016x}"
        )
    main_records = [
        value for _raw, value in nonterminal if value["record_kind"] != "structured_boundary"
    ]
    sequences = [int(value["evidence_sequence"]) for value in main_records]
    if sequences != list(range(len(sequences))):
        mismatches.append(f"Stage B evidence sequence is not contiguous: {sequences}")
    if int(terminal["evidence_sequence"]) != len(main_records):
        mismatches.append("Stage B terminal sequence does not follow the final evidence record")
    previous: dict[str, object] | None = None
    for _raw, value in nonterminal:
        if value["record_kind"] == "structured_boundary":
            if previous is None or previous["record_kind"] != "observation":
                mismatches.append("structured boundary does not follow an observation")
            elif int(value["evidence_sequence"]) != int(previous["evidence_sequence"]):
                mismatches.append("structured boundary sequence differs from its observation")
        else:
            previous = value

    replay = replay_stage_b_regions(raw_observations, evidence_records)
    causal_mismatch = _compare_stage_b_causal_order(
        [value for _raw, value in evidence_records], replay
    )
    if causal_mismatch:
        mismatches.append(causal_mismatch)
    by_stroke: dict[int, list[VelocityObservation]] = {}
    for observation in raw_observations:
        by_stroke.setdefault(observation.stroke_index, []).append(observation)
    observation_records = {
        (int(value["rung_index"]), int(value["slot"])): value
        for _raw, value in evidence_records
        if value["record_kind"] == "observation"
    }
    reconstructed: dict[tuple[int, int], dict[str, int]] = {}
    exact_fields = (
        "classification",
        "delta_sign",
        "delta_mantissa",
        "delta_shift",
        "elapsed_mantissa",
        "elapsed_shift",
        "rate_low",
        "rate_mean",
        "rate_high",
        "deficit_low",
        "deficit_high",
        "rate_shift",
        "suffix_len",
        "selected_level",
        "selected_blocks",
        "rate_variance_mantissa",
        "rate_variance_shift",
        "slope_mantissa",
        "slope_shift",
        "slope_half_width_mantissa",
        "slope_half_width_shift",
        "lag_one_q",
        "lag_one_half_width_q",
        "residual_mantissa",
        "residual_shift",
        "tested_suffixes",
        "disturbance_q16",
        "disturbance_low_q16",
        "disturbance_high_q16",
        "disturbance_variance_mantissa",
        "disturbance_variance_shift",
        "predicted_torque_target_abs",
    )
    for key, record in sorted(observation_records.items()):
        raw = by_stroke.get(key[0] * 4 + key[1], [])
        if not raw:
            mismatches.append(f"observation {key[0]}/{key[1]} has no raw samples")
            continue
        offline = _reconstruct_observation(raw, record, plan, terminal)
        for name in exact_fields:
            offline.setdefault(name, 0)
        _compare_fields(mismatches, f"observation {key[0]}/{key[1]}", record, offline, exact_fields)
        reconstructed[key] = offline

    boundary_records = [
        value for _raw, value in evidence_records if value["record_kind"] == "structured_boundary"
    ]
    expected_boundaries: dict[tuple[int, int], dict[str, object]] = {}
    for key in observation_records:
        raw = by_stroke.get(key[0] * 4 + key[1], [])
        boundary = analyze_structured_boundary(raw)
        if boundary is not None:
            expected_boundaries[key] = boundary
    for record in boundary_records:
        key = (int(record["rung_index"]), int(record["slot"]))
        boundary = expected_boundaries.pop(key, None)
        if boundary is None:
            mismatches.append(f"unexpected structured boundary {key}")
            continue
        low_mantissa, low_shift = _upper_bound_signed(boundary["common_rate_interval_q"][0])
        high_mantissa, high_shift = _upper_bound_signed(boundary["common_rate_interval_q"][1])
        slope_mantissa, slope_shift = _upper_bound_signed(boundary["binding_slope_margin_q"])
        expected = {
            "passing_suffix_mask": boundary["passing_suffix_mask"],
            "common_rate_low_mantissa": low_mantissa,
            "common_rate_low_shift": low_shift,
            "common_rate_high_mantissa": high_mantissa,
            "common_rate_high_shift": high_shift,
            "binding_slope_margin_mantissa": slope_mantissa,
            "binding_slope_margin_shift": slope_shift,
            "binding_lag_one_margin_q": boundary["binding_lag_one_margin_q"],
        }
        _compare_fields(mismatches, f"structured boundary {key}", record, expected, tuple(expected))
    if expected_boundaries:
        mismatches.append(f"missing structured boundaries {sorted(expected_boundaries)}")

    rung_records = {
        int(value["rung_index"]): value
        for _raw, value in evidence_records
        if value["record_kind"] == "rung"
    }
    for rung_index, rung in sorted(rung_records.items()):
        pairs = []
        for slots in ((0, 2), (1, 3)):
            if any((rung_index, slot) not in reconstructed for slot in slots):
                break
            pairs.append(
                _rung_class(
                    reconstructed[(rung_index, slots[0])],
                    reconstructed[(rung_index, slots[1])],
                )
            )
        if len(pairs) != 2:
            mismatches.append(f"rung {rung_index} is missing reconstructed observations")
            continue
        offline = {
            "forward_class": pairs[0][0],
            "forward_low_q16": pairs[0][1],
            "forward_high_q16": pairs[0][2],
            "reverse_class": pairs[1][0],
            "reverse_low_q16": pairs[1][1],
            "reverse_high_q16": pairs[1][2],
            "flags": int(pairs[0][3])
            | (int(pairs[1][3]) << 1)
            | (
                int(
                    any(
                        int(observation_records[(rung_index, slot)]["flags"]) & 1 for slot in (0, 2)
                    )
                )
                << 2
            )
            | (
                int(
                    any(
                        int(observation_records[(rung_index, slot)]["flags"]) & 1 for slot in (1, 3)
                    )
                )
                << 3
            ),
            "forward_eligible_observations": 2 if pairs[0][0] == 0 else 0,
            "reverse_eligible_observations": 2 if pairs[1][0] == 0 else 0,
        }
        _compare_fields(mismatches, f"rung {rung_index}", rung, offline, tuple(offline))

    closure_codes = {
        "disjoint_interval": 0,
        "ineligible_gap": 1,
        "current_headroom": 2,
        "safety_terminus": 3,
        "end_of_ladder": 4,
    }
    expected_regions = {
        (direction, int(region["member_mask"])): region
        for direction, regions in enumerate(replay["directions"])
        for region in regions
    }
    reported_regions = [
        value for _raw, value in evidence_records if value["record_kind"] == "directional_region"
    ]
    for record in reported_regions:
        key = (int(record["direction"]), int(record["member_mask"]))
        region = expected_regions.pop(key, None)
        if region is None:
            mismatches.append(f"unexpected directional region {key}")
            continue
        expected = {
            "region_kind": 0 if region["kind"] == "valid" else 1,
            "closure": closure_codes[str(region["closure"])],
            "first_rung": region["first_rung"],
            "last_rung": region["last_rung"],
            "p_low": region["selected_ps"][0],
            "p_high": region["selected_ps"][-1],
            "member_count": region["member_count"],
            "common_low_q16": region["common_interval"][0],
            "common_high_q16": region["common_interval"][1],
            "pooled_q16": region["pooled_q16"],
            "pooled_low_q16": region["pooled_interval"][0],
            "pooled_high_q16": region["pooled_interval"][1],
            "variance_floor_observations": region["variance_floor_observations"],
            "boundary_rung_plus_one": 0
            if region["boundary_rung"] is None
            else int(region["boundary_rung"]) + 1,
            "tested_low_q16": region["tested_intersection"][0],
            "tested_high_q16": region["tested_intersection"][1],
        }
        _compare_fields(mismatches, f"directional region {key}", record, expected, tuple(expected))
        for prefix, source in (
            ("settled_rate_mean_min", "settled_rate_mean_min_q"),
            ("settled_rate_mean_max", "settled_rate_mean_max_q"),
            ("settled_rate_envelope_low", "settled_rate_envelope_low_q"),
            ("settled_rate_envelope_high", "settled_rate_envelope_high_q"),
        ):
            decoded = int(record[f"{prefix}_mantissa"]) << int(record[f"{prefix}_shift"])
            if decoded != int(region[source]):
                mismatches.append(
                    f"directional region {key} {prefix}: firmware={decoded} offline={region[source]}"
                )
    if expected_regions:
        mismatches.append(f"missing directional regions {sorted(expected_regions)}")

    joint_closure_codes = {
        "ineligible_gap": 0,
        "current_headroom": 1,
        "safety_terminus": 2,
        "end_of_ladder": 3,
    }
    expected_joint = {int(region["member_mask"]): region for region in replay["joint_regions"]}
    for record in (
        value for _raw, value in evidence_records if value["record_kind"] == "joint_region"
    ):
        region = expected_joint.pop(int(record["member_mask"]), None)
        if region is None:
            mismatches.append(f"unexpected joint region {record['member_mask']}")
            continue
        expected = {
            "first_rung": region["first_rung"],
            "last_rung": region["last_rung"],
            "member_count": region["member_count"],
            "p_low": replay["gains"][int(region["first_rung"])],
            "p_high": replay["gains"][int(region["last_rung"])],
            "closure": joint_closure_codes[str(region["closure"])],
        }
        _compare_fields(mismatches, "joint region", record, expected, tuple(expected))
    if expected_joint:
        mismatches.append(f"missing joint regions {sorted(expected_joint)}")

    handoffs = [value for _raw, value in evidence_records if value["record_kind"] == "handoff"]
    if replay["nomination"] and len(handoffs) != 1:
        mismatches.append(f"expected one Stage B handoff, found {len(handoffs)}")
    elif not replay["nomination"] and handoffs:
        mismatches.append("Stage B handoff exists without a joint nomination")
    elif replay["nomination"]:
        handoff = handoffs[0]
        nomination = replay["nomination"]
        expected = {
            "nominated_p": replay["nominated_p"],
            "coverage_mask": sum(
                (1 << index)
                for index, item in enumerate(replay["selected"])
                if item["coverage"] == "full"
            ),
            "joint_member_mask": nomination["member_mask"],
            "nominated_rung": nomination["rung_index"],
            "distance_to_start_q16": nomination["distance_to_start_q16"],
            "distance_to_top_q16": nomination["distance_to_top_q16"],
            "nomination_flags": int(nomination["reduced_margin"]),
        }
        _compare_fields(mismatches, "Stage B handoff", handoff, expected, tuple(expected))
        for direction, selected in zip(handoff["directions"], replay["selected"], strict=True):
            expected_direction = {
                "member_mask": selected["member_mask"],
                "pooled_low_q16": selected["pooled_interval"][0],
                "pooled_high_q16": selected["pooled_interval"][1],
                "coverage": 0 if selected["coverage"] == "full" else 1,
                "signed_rung_distance": int(selected["signed_rung_distance"]) & 0xFF,
                "gain_ratio_num": selected["gain_ratio"][0],
                "gain_ratio_den": selected["gain_ratio"][1],
            }
            _compare_fields(
                mismatches,
                "Stage B directional handoff",
                direction,
                expected_direction,
                tuple(expected_direction),
            )

    expected_memberships = [
        *(int(item.get("member_mask", 0)) for item in replay["selected"]),
        int(replay["nomination"].get("member_mask", 0)),
    ]
    expected_intervals = [list(item.get("pooled_interval", [0, 0])) for item in replay["selected"]]
    terminal_expected = {
        "model_direction_mask": sum(
            (1 << index)
            for index, regions in enumerate(replay["directions"])
            if any(region["kind"] == "valid" for region in regions)
        ),
        "nominated_p": replay["nominated_p"],
        "coverage_mask": sum(
            (1 << index)
            for index, item in enumerate(replay["selected"])
            if item.get("coverage") == "full"
        ),
        "expected_observations": int(plan["rung_count"])
        * int(plan["observations_per_direction"])
        * 2,
        "emitted_observations": len(observation_records),
        "expected_rungs": int(plan["rung_count"]),
        "emitted_rungs": len(rung_records),
    }
    _compare_fields(
        mismatches, "Stage B terminal", terminal, terminal_expected, tuple(terminal_expected)
    )
    if list(terminal["selected_memberships"]) != expected_memberships:
        mismatches.append("Stage B terminal selected memberships differ from replay")
    if list(terminal["selected_intervals"]) != expected_intervals:
        mismatches.append("Stage B terminal selected intervals differ from replay")
    region_counts = [
        sum(region["kind"] == "valid" for region in regions) for regions in replay["directions"]
    ]
    fragment_counts = [
        sum(region["kind"] == "fragment" for region in regions) for regions in replay["directions"]
    ]
    if [terminal["forward_region_count"], terminal["reverse_region_count"]] != region_counts:
        mismatches.append("Stage B terminal directional region counts differ from replay")
    if [terminal["forward_fragment_count"], terminal["reverse_fragment_count"]] != fragment_counts:
        mismatches.append("Stage B terminal directional fragment counts differ from replay")

    reproduction = [
        value for _raw, value in evidence_records if value["record_kind"] == "reproduction"
    ]
    if int(terminal["outcome"]) == 1 and len(reproduction) != 1:
        mismatches.append("complete Stage B terminal lacks one reproduction record")
    if len(reproduction) > 1:
        mismatches.append(f"expected at most one reproduction record, found {len(reproduction)}")
    if reproduction:
        current = reproduction[-1]
        if int(current["current_nominated_p"]) != int(terminal["nominated_p"]):
            mismatches.append("reproduction nomination differs from terminal")
        if list(current["current_memberships"]) != list(terminal["selected_memberships"]):
            mismatches.append("reproduction memberships differ from terminal")
        if list(current["current_intervals"]) != list(terminal["selected_intervals"]):
            mismatches.append("reproduction intervals differ from terminal")
        reported_current_digest = int(current["current_digest_low"]) | (
            int(current["current_digest_high"]) << 32
        )
        if pre_reproduction_digest != reported_current_digest:
            mismatches.append("reproduction current digest differs from canonical prefix")
        previous_candidate = {
            "plan_digest": int(plan["plan_digest_low"]) | (int(plan["plan_digest_high"]) << 32),
            "nominated_p": int(current["previous_nominated_p"]),
            "memberships": list(current["previous_memberships"]),
            "intervals": [list(value) for value in current["previous_intervals"]],
        }
        current_candidate = {
            "plan_digest": previous_candidate["plan_digest"],
            "nominated_p": int(current["current_nominated_p"]),
            "memberships": list(current["current_memberships"]),
            "intervals": [list(value) for value in current["current_intervals"]],
        }
        comparison = _compare_stage_b_candidates(previous_candidate, current_candidate)
        expected_reproduction_outcome = 1 if comparison["outcome"] == "complete" else 2
        if int(current["outcome"]) != expected_reproduction_outcome:
            mismatches.append("reproduction outcome differs from independent comparison")
        expected_terminal = (
            STAGE_B_OUTCOME_COMPLETE
            if expected_reproduction_outcome == 1
            else STAGE_B_OUTCOME_INCONCLUSIVE
        )
        if int(terminal["outcome"]) != expected_terminal:
            mismatches.append("terminal outcome differs from reproduction comparison")
        expected_cause = (
            0 if expected_reproduction_outcome == 1 else STAGE_B_CAUSE_REPRODUCTION_MISMATCH
        )
        if int(terminal["cause"]) != expected_cause:
            mismatches.append("terminal cause differs from reproduction comparison")

    pid_samples = [
        {
            "stroke_index": item.stroke_index,
            "pid_velocity_actual": item.pid_velocity_actual,
        }
        for item in raw_observations
        if item.pid_velocity_actual is not None
    ]
    plan_identity = int(plan["plan_digest_low"]) | (int(plan["plan_digest_high"]) << 32)
    caveats = (
        ["reproduction requires the prior capture for cross-run digest verification"]
        if reproduction
        else []
    )
    return {
        "run_sequence": int(plan["run_sequence"]),
        "plan": dict(plan),
        "terminal": dict(terminal),
        "digest": digest,
        "reported_digest": reported_digest,
        "pre_reproduction_digest": pre_reproduction_digest,
        "plan_identity": plan_identity,
        "reproduction": dict(reproduction[-1]) if reproduction else None,
        "observation_count": len(observation_records),
        "rung_count": len(rung_records),
        "stage_b_replay": replay,
        "directions": replay["directions"],
        "nominated_p": replay["nominated_p"],
        "coverage": replay["coverage"],
        "pid_velocity_actual_samples": pid_samples,
        "actual_velocity_pptm_samples": [],
        "complete": True,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": caveats,
    }


def audit_velocity_sweep(
    raw_observations: list[VelocityObservation],
    evidence_records: list[tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    """Independently reconstruct one sweep and compare every firmware decision."""
    if any(value["record_kind"] == "stage_b_terminal" for _raw, value in evidence_records):
        return _audit_stage_b_velocity_sweep(raw_observations, evidence_records)
    sample_indexes: dict[int, int] = {}
    pid_velocity_actual_samples: list[dict[str, int]] = []
    actual_velocity_pptm_samples: list[dict[str, int]] = []
    for observation in raw_observations:
        sample_index = sample_indexes.get(observation.stroke_index, 0)
        sample_indexes[observation.stroke_index] = sample_index + 1
        if observation.pid_velocity_actual is not None:
            pid_velocity_actual_samples.append(
                {
                    "stroke_index": observation.stroke_index,
                    "sample_index": sample_index,
                    "direction": observation.direction,
                    "target_counts_per_ms": observation.target_cpms,
                    "delta_counts": observation.delta_counts,
                    "elapsed_us": observation.elapsed_us,
                    "pid_velocity_actual": observation.pid_velocity_actual,
                }
            )
        if observation.actual_velocity_pptm is not None:
            actual_velocity_pptm_samples.append(
                {
                    "stroke_index": observation.stroke_index,
                    "sample_index": sample_index,
                    "direction": observation.direction,
                    "target_counts_per_ms": observation.target_cpms,
                    "delta_counts": observation.delta_counts,
                    "elapsed_us": observation.elapsed_us,
                    "actual_velocity_pptm": observation.actual_velocity_pptm,
                }
            )
    plans = [value for _raw, value in evidence_records if value["record_kind"] == "plan"]
    terminals = [value for _raw, value in evidence_records if value["record_kind"] == "terminal"]
    records = [
        (raw, value)
        for raw, value in evidence_records
        if value["record_kind"] in ("plan", "observation", "rung")
    ]
    digest = FNV1A64_OFFSET
    for raw, _value in records:
        for byte in raw:
            digest ^= byte
            digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    if len(plans) != 1 or len(terminals) != 1:
        caveats = []
        if not plans:
            caveats.append("partial capture: missing plan")
        elif len(plans) > 1:
            caveats.append("partial capture: duplicate plans")
        if not terminals:
            caveats.append("partial capture: missing terminal")
        elif len(terminals) > 1:
            caveats.append("partial capture: duplicate terminals")
        observation_count = sum(value["record_kind"] == "observation" for _raw, value in records)
        rung_count = sum(value["record_kind"] == "rung" for _raw, value in records)
        identities = plans or terminals or [value for _raw, value in records]
        return {
            "run_sequence": int(identities[0].get("run_sequence", 0)) if identities else 0,
            "plan": dict(plans[0]) if len(plans) == 1 else None,
            "terminal": dict(terminals[0]) if len(terminals) == 1 else None,
            "digest": digest,
            "reported_digest": None,
            "raw_observation_count": len(raw_observations),
            "pid_velocity_actual_samples": pid_velocity_actual_samples,
            "actual_velocity_pptm_samples": actual_velocity_pptm_samples,
            "observation_count": observation_count,
            "rung_count": rung_count,
            "directions": [],
            "complete": False,
            "matches_firmware": False,
            "mismatches": list(caveats),
            "caveats": caveats,
        }
    plan, terminal = plans[0], terminals[0]
    mismatches: list[str] = []
    expected_target_rpm = _mrev_s_to_rpm_nearest(int(plan["planned_velocity_mrev_s"]))
    if int(plan["target_velocity_rpm"]) != expected_target_rpm:
        mismatches.append(
            "plan target RPM: firmware=%d offline=%d"
            % (int(plan["target_velocity_rpm"]), expected_target_rpm)
        )
    sequences = [int(value["evidence_sequence"]) for _raw, value in records]
    if sequences != list(range(len(sequences))):
        mismatches.append(f"evidence sequence is not contiguous: {sequences}")
    reported_digest = int(terminal["digest_low"]) | (int(terminal["digest_high"]) << 32)
    if digest != reported_digest:
        mismatches.append(
            f"terminal digest: firmware=0x{reported_digest:016x} offline=0x{digest:016x}"
        )
    by_stroke: dict[int, list[VelocityObservation]] = {}
    for observation in raw_observations:
        by_stroke.setdefault(observation.stroke_index, []).append(observation)
    observation_records = {
        (int(value["rung_index"]), int(value["slot"])): value
        for _raw, value in records
        if value["record_kind"] == "observation"
    }
    diagnostic_audit = _diagnostic_audit(
        raw_observations,
        observation_records,
        int(terminal["encoder_counts_per_rev"]),
        int(plan["usable_torque_limit"]),
    )
    if int(plan["rung_count"]) > 0:
        if not diagnostic_audit["complete"]:
            mismatches.append("velocity diagnostic evidence is incomplete")
        torque_audit = diagnostic_audit["torque"]
        if not torque_audit["within_observed_peak_plus_one"]:
            mismatches.append("reconstructed torque exceeds observed peak plus one count")
        if not _torque_evidence_matches_terminal(terminal, torque_audit):
            mismatches.append("diagnostic torque evidence reaches the usable limit")
    offline_observations: dict[tuple[int, int], dict[str, int]] = {}
    exact_fields = (
        "classification",
        "delta_sign",
        "delta_mantissa",
        "delta_shift",
        "elapsed_mantissa",
        "elapsed_shift",
        "rate_low",
        "rate_mean",
        "rate_high",
        "deficit_low",
        "deficit_high",
        "rate_shift",
        "suffix_len",
        "selected_level",
        "selected_blocks",
        "rate_variance_mantissa",
        "rate_variance_shift",
        "slope_mantissa",
        "slope_shift",
        "slope_half_width_mantissa",
        "slope_half_width_shift",
        "lag_one_q",
        "lag_one_half_width_q",
        "residual_mantissa",
        "residual_shift",
        "tested_suffixes",
        "disturbance_q16",
        "disturbance_low_q16",
        "disturbance_high_q16",
        "disturbance_variance_mantissa",
        "disturbance_variance_shift",
        "predicted_torque_target_abs",
    )
    for key, record in sorted(observation_records.items()):
        raw = by_stroke.get(key[0] * 4 + key[1], [])
        if not raw:
            mismatches.append(f"observation {key[0]}/{key[1]} has no raw samples")
            continue
        offline = _reconstruct_observation(raw, record, plan, terminal)
        offline.setdefault("suffix_len", 0)
        offline.setdefault("selected_level", 0)
        offline.setdefault("selected_blocks", 0)
        offline.setdefault("rate_variance_mantissa", 0)
        offline.setdefault("rate_variance_shift", 0)
        for name in exact_fields:
            offline.setdefault(name, 0)
        _compare_fields(mismatches, f"observation {key[0]}/{key[1]}", record, offline, exact_fields)
        offline_observations[key] = offline
    bands = {0: _new_band(), 1: _new_band()}
    rung_records = {
        int(value["rung_index"]): value for _raw, value in records if value["record_kind"] == "rung"
    }
    for rung_index, rung_record in sorted(rung_records.items()):
        p = int(rung_record["velocity_p"])
        computed: dict[int, tuple[int, int, int, bool]] = {}
        for direction_index, slots in ((0, (0, 2)), (1, (1, 3))):
            first = offline_observations.get((rung_index, slots[0]))
            second = offline_observations.get((rung_index, slots[1]))
            if first is None or second is None:
                continue
            result = _rung_class(first, second)
            computed[direction_index] = result
            _observe_band(bands[direction_index], rung_index, p, *result, (first, second))
        if len(computed) != 2:
            mismatches.append(f"rung {rung_index} is missing reconstructed directions")
            continue
        offline_rung = {
            "forward_class": computed[0][0],
            "forward_low_q16": computed[0][1],
            "forward_high_q16": computed[0][2],
            "reverse_class": computed[1][0],
            "reverse_low_q16": computed[1][1],
            "reverse_high_q16": computed[1][2],
            "flags": int(computed[0][3]) | (int(computed[1][3]) << 1) | (int(rung_index == 0) << 2),
        }
        for direction_index, prefix in ((0, "forward"), (1, "reverse")):
            snapshot = _finish_band(dict(bands[direction_index]))
            offline_rung[f"{prefix}_p_low"] = snapshot["p_low"]
            offline_rung[f"{prefix}_p_high"] = snapshot["p_high"]
            offline_rung[f"{prefix}_closure"] = snapshot["closure"]
            offline_rung[f"{prefix}_eligible_rungs"] = snapshot["eligible_rungs"]
            offline_rung[f"{prefix}_eligible_observations"] = snapshot["eligible_observations"]
        _compare_fields(
            mismatches,
            f"rung {rung_index}",
            rung_record,
            offline_rung,
            tuple(offline_rung),
        )
    finished_bands = [_finish_band(bands[index]) for index in (0, 1)]
    for index, (firmware, offline) in enumerate(
        zip(terminal["directions"], finished_bands, strict=True)
    ):
        offline_terminal = dict(offline)
        offline_terminal["direction_closure"] = index | (offline["closure"] << 1)
        _compare_fields(
            mismatches,
            f"terminal direction {index}",
            firmware,
            offline_terminal,
            (
                "direction_closure",
                "first_moving",
                "p_low",
                "p_high",
                "common_low_q16",
                "common_high_q16",
                "pooled_q16",
                "pooled_low_q16",
                "pooled_high_q16",
                "eligible_rungs",
                "eligible_observations",
                "flags",
            ),
        )
    direction_mask = finished_bands[0]["sufficient"] | (finished_bands[1]["sufficient"] << 1)
    if direction_mask != int(terminal["sufficient_direction_mask"]):
        mismatches.append(
            "terminal sufficiency mask: firmware=%d offline=%d"
            % (int(terminal["sufficient_direction_mask"]), direction_mask)
        )
    expected_observations = int(plan["rung_count"]) * int(plan["observations_per_direction"]) * 2
    for name, offline_value in (
        ("expected_observations", expected_observations),
        ("emitted_observations", len(observation_records)),
        ("expected_rungs", int(plan["rung_count"])),
        ("emitted_rungs", len(rung_records)),
    ):
        if int(terminal[name]) != offline_value:
            mismatches.append(f"terminal {name}: firmware={terminal[name]} offline={offline_value}")
    stage_b_replay = replay_stage_b_regions(raw_observations, evidence_records)
    return {
        "run_sequence": int(plan["run_sequence"]),
        "plan": dict(plan),
        "terminal": dict(terminal),
        "digest": digest,
        "reported_digest": reported_digest,
        "observation_count": len(observation_records),
        "pid_velocity_actual_samples": pid_velocity_actual_samples,
        "actual_velocity_pptm_samples": actual_velocity_pptm_samples,
        "velocity_diagnostic_audit": diagnostic_audit,
        "rung_count": len(rung_records),
        "directions": finished_bands,
        "stage_b_replay": stage_b_replay,
        "complete": True,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": [],
    }
