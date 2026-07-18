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


def _disturbance(
    *,
    p: int,
    target_register: int,
    delta_counts: int,
    elapsed_us: int,
    half_width_rate_q: int,
    variance_rate_q2: int,
    counts_per_rev: int,
    f_pwm_hz: int,
) -> tuple[int, int, int, int]:
    scale_denominator = counts_per_rev * f_pwm_hz
    target_q16 = target_register * 65_536
    target_numerator = target_q16 * scale_denominator * elapsed_us
    encoder_numerator = delta_counts * 65_536 * 65_536 * 1_000_000
    error_numerator = target_numerator - encoder_numerator
    disturbance_numerator = p * error_numerator
    disturbance_denominator = 256 * scale_denominator * elapsed_us
    estimate = _trunc_div(disturbance_numerator, disturbance_denominator)
    common_denominator = disturbance_denominator * RATE_Q_SCALE
    common_mean = disturbance_numerator * RATE_Q_SCALE
    common_half_width = p * half_width_rate_q * 65_536 * 65_536 * elapsed_us
    low = _floor_div(common_mean - common_half_width, common_denominator)
    high = _ceil_div(common_mean + common_half_width, common_denominator)
    scale_numerator = p * 65_536 * 65_536
    variance_numerator = variance_rate_q2 * scale_numerator * scale_numerator
    variance_denominator = (256 * counts_per_rev * f_pwm_hz * RATE_Q_SCALE) ** 2
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
    target_register = int(record["target_velocity_register"])
    predicted = max(
        (p * abs(target_register) + 255) // 256,
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
    half_width = math.ceil(
        _confidence_multiplier(family_size) * math.sqrt(stationarity["variance_of_mean_q2"])
    )
    mean = stationarity["settled_rate_mean_q"]
    rate_low, rate_high = mean - half_width, mean + half_width
    direction = 1 if int(record["slot"]) % 2 == 0 else -1
    target_rate_q = (
        direction * int(plan["used_velocity_mrev_s"]) * int(terminal["encoder_counts_per_rev"])
    )
    deficit_low, deficit_high = target_rate_q - rate_high, target_rate_q - rate_low
    disturbance, disturbance_low, disturbance_high, disturbance_variance = _disturbance(
        p=p,
        target_register=target_register,
        delta_counts=stationarity["selected_delta_counts"],
        elapsed_us=stationarity["selected_elapsed_us"],
        half_width_rate_q=half_width,
        variance_rate_q2=stationarity["variance_of_mean_q2"],
        counts_per_rev=int(terminal["encoder_counts_per_rev"]),
        f_pwm_hz=int(terminal["f_pwm_hz"]),
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


def audit_velocity_sweep(
    raw_observations: list[VelocityObservation],
    evidence_records: list[tuple[bytes, dict[str, object]]],
) -> dict[str, object]:
    """Independently reconstruct one sweep and compare every firmware decision."""
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
    return {
        "run_sequence": int(plan["run_sequence"]),
        "plan": dict(plan),
        "terminal": dict(terminal),
        "digest": digest,
        "reported_digest": reported_digest,
        "observation_count": len(observation_records),
        "rung_count": len(rung_records),
        "directions": finished_bands,
        "complete": True,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": [],
    }
