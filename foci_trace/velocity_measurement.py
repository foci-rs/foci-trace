"""Independent reference for windowed encoder-velocity confidence analysis."""

from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass, field
from statistics import median

from .scored_rest import select_causal_rest_suffixes

RATE_Q_SCALE = 1_000
RECOVERY_MOVING_ALLOWANCE_US = 512_000
RECOVERY_ZERO_SETTLE_US = 500_000
RECOVERY_ANALYSIS_BUDGET_US = 236_000
MIN_SELECTED_BLOCKS = 16
# Parts-per-million of settled rate a moving stroke may drift across its
# regression span. Mirrors firmware's MAX_MOVING_DRIFT_PPM.
MAX_MOVING_DRIFT_PPM = 200_000
# Reported drift ratio when no regression produced one, or when the settled
# interval contains zero (the Inert condition bypasses the drift bound
# entirely). Mirrors firmware's NO_REPORTABLE_DRIFT_RATIO_PPM: a zero would
# read as "measured, and tiny", the opposite of the truth here.
NO_REPORTABLE_DRIFT_RATIO_PPM = 0xFFFFFFFF
INTEGRAL_DRIFT_BOUND_MIN_SCHEMA_REVISION = 13  # firmware: INTEGRAL_PLAN_SCHEMA_REVISION
INTEGRAL_SUFFIX_WALK_MIN_SCHEMA_REVISION = 15
INTEGRAL_REST_WALK_MIN_SCHEMA_REVISION = 16

# Firmware build_features bitset, mirrored from foci-firmware's BuildFeatures.
# Each bit names one analysis rule the recording firmware compiled in, stated
# explicitly on the wire so a capture carries its own rule contract instead of
# being dated by schema revision. Captures recorded before the field existed
# reconstruct these bits from their build_revision (bit 0) and the per-namespace
# schema thresholds above (bits 1-3); see `reconstruct_build_features`.
BUILD_FEATURE_Q4_12_GAINS = 1 << 0
BUILD_FEATURE_DRIFT_BOUND_GATE = 1 << 1
BUILD_FEATURE_SUFFIX_WALK_CONTINUES = 1 << 2
BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE = 1 << 3
BUILD_FEATURE_STANDARD_METER_QUANTUM = 1 << 4

# The two firmware crate-version hashes every archived capture carries in its
# `build_revision` field, retained here to reconstruct `build_features` for
# captures recorded before firmware emitted it. build_revision was later retired
# from new captures in favour of build_identity + build_features.
_LEGACY_Q8_8_BUILD_REVISION = 0x75536758
_CURRENT_Q4_12_BUILD_REVISION = 0x05E21F75


class UnsupportedIntegralRepresentationError(ValueError):
    """Raised when a firmware build has no declared integral-gain domain."""


def _rule_features_from_schema(schema_revision: int) -> int:
    """Reconstruct just the analysis rule bits (drift/suffix/rest) a legacy
    capture ran under, from the Integral schema thresholds.

    This ignores the integral-representation bit and never consults
    `build_revision`: the stationarity gates read only the rule bits, so an
    archived capture with no representation signal must still resolve its gate
    contract rather than raise.
    """
    features = 0
    if schema_revision >= INTEGRAL_DRIFT_BOUND_MIN_SCHEMA_REVISION:
        features |= BUILD_FEATURE_DRIFT_BOUND_GATE
    if schema_revision >= INTEGRAL_SUFFIX_WALK_MIN_SCHEMA_REVISION:
        features |= BUILD_FEATURE_SUFFIX_WALK_CONTINUES
    if schema_revision >= INTEGRAL_REST_WALK_MIN_SCHEMA_REVISION:
        features |= BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE
    return features


def reconstruct_build_features(build_revision: int | None, schema_revision: int) -> int:
    """Rebuild the rule contract a pre-`build_features` capture ran under.

    Bit 0 (integral representation) comes from the crate-version
    `build_revision`; the three rule bits reconstruct from the Integral schema
    thresholds, so an archived capture is analysed under exactly the gate that
    recorded it.
    """
    if build_revision == _CURRENT_Q4_12_BUILD_REVISION:
        features = BUILD_FEATURE_Q4_12_GAINS
    elif build_revision == _LEGACY_Q8_8_BUILD_REVISION:
        features = 0
    elif build_revision is None:
        raise UnsupportedIntegralRepresentationError(
            "missing firmware build revision for integral-gain interpretation"
        )
    else:
        raise UnsupportedIntegralRepresentationError(
            f"unsupported integral-gain representation for build 0x{build_revision:08x}"
        )
    return features | _rule_features_from_schema(schema_revision)


def capture_rule_features(plan: dict[str, object]) -> int | None:
    """Effective analysis rule-bit contract for the stationarity gates.

    A newer capture states `build_features` directly; a legacy capture
    reconstructs only its rule bits from the schema thresholds. Unlike
    `capture_build_features` this never consults `build_revision` and never
    raises on a missing integral-representation signal, because the gates read
    only the rule bits. Returns `None` when the capture offers no rule signal at
    all (no `build_features` and no schema revision), which the gates treat as
    the legacy significance test.
    """
    explicit = plan.get("build_features")
    if explicit is not None:
        return int(explicit)  # type: ignore[arg-type]
    schema_revision = plan.get("schema_revision")
    if schema_revision is None:
        return None
    return _rule_features_from_schema(int(schema_revision))  # type: ignore[arg-type]


def capture_build_features(plan: dict[str, object]) -> int:
    """Return the effective rule contract for one decoded capture.

    A newer capture states `build_features` directly; a legacy capture
    carries only `build_revision`, from which the contract is reconstructed via
    the schema thresholds.
    """
    explicit = plan.get("build_features")
    if explicit is not None:
        return int(explicit)  # type: ignore[arg-type]
    build_revision = plan.get("build_revision")
    return reconstruct_build_features(
        int(build_revision) if isinstance(build_revision, int) else None,
        int(plan["schema_revision"]),  # type: ignore[arg-type]
    )


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
    4.202358,
    4.209919,
    4.217368,
    4.224709,
    4.231945,
    4.239079,
    4.246114,
    4.253052,
    4.259896,
    4.266650,
    4.273314,
    4.279892,
    4.286386,
    4.292798,
    4.299129,
    4.305383,
    4.311561,
    4.317665,
    4.323696,
    4.329656,
    4.335548,
    4.341372,
    4.347130,
    4.352824,
    4.358455,
    4.364025,
    4.369534,
    4.374985,
    4.380377,
    4.385714,
    4.390995,
    4.396222,
    4.401397,
    4.406519,
    4.411591,
    4.416613,
    4.421586,
    4.426511,
    4.431389,
    4.436222,
    4.441009,
    4.445752,
    4.450451,
    4.455108,
    4.459722,
    4.464296,
    4.468829,
    4.473323,
    4.477778,
    4.482194,
    4.486573,
    4.490914,
    4.495220,
    4.499489,
    4.503724,
    4.507924,
    4.512090,
    4.516222,
    4.520322,
    4.524389,
    4.528424,
    4.532429,
    4.536402,
    4.540345,
)


def _div_ceil(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceiling division requires a nonnegative numerator and positive divisor")
    return (numerator + denominator - 1) // denominator


def _recovery_allowance_us(moving_us: int) -> int:
    return (
        moving_us
        + RECOVERY_MOVING_ALLOWANCE_US
        + RECOVERY_ZERO_SETTLE_US
        + RECOVERY_ANALYSIS_BUDGET_US
    )


def plan_rung_origin_recovery(
    *,
    max_stroke_travel_mrev: int,
    settle_travel_reserve_mrev: int,
    encoder_counts_per_rev: int,
    planned_velocity_mrev_s: int,
    primitive_cap_us: int,
) -> dict[str, int]:
    """Reconstruct the reserve-derived origin band and fixed recovery slots."""
    if (
        min(
            max_stroke_travel_mrev,
            settle_travel_reserve_mrev,
            encoder_counts_per_rev,
            planned_velocity_mrev_s,
            primitive_cap_us,
        )
        <= 0
        or settle_travel_reserve_mrev >= max_stroke_travel_mrev
    ):
        raise ValueError("recovery geometry is invalid")
    origin_band_counts = _div_ceil(settle_travel_reserve_mrev * encoder_counts_per_rev, 1_000)
    travel_us = _div_ceil(max_stroke_travel_mrev * 1_000_000, planned_velocity_mrev_s)
    nominal_slot_us = _recovery_allowance_us(travel_us)
    if nominal_slot_us > primitive_cap_us:
        raise ValueError("nominal recovery exceeds primitive cap")
    return {
        "origin_band_counts": origin_band_counts,
        "nominal_slot_us": nominal_slot_us,
        "maximum_slot_us": primitive_cap_us,
    }


def recovery_return_distance_counts(start_offset_counts: int, origin_band_counts: int) -> int:
    """Return the absolute distance before the inclusive origin band begins."""
    if start_offset_counts == -(1 << 31) or origin_band_counts <= 0:
        raise ValueError("recovery return distance is not representable")
    return max(0, abs(start_offset_counts) - origin_band_counts)


def recovery_runtime_us(
    return_distance_counts: int, lower_rate_q: int, primitive_cap_us: int
) -> int:
    """Derive the measured-rate recovery deadline in firmware integer order."""
    if return_distance_counts < 0 or lower_rate_q <= 0 or primitive_cap_us <= 0:
        raise ValueError("recovery deadline inputs are invalid")
    measured_return_us = _div_ceil(return_distance_counts * 1_000_000 * RATE_Q_SCALE, lower_rate_q)
    runtime_us = _recovery_allowance_us(measured_return_us)
    if runtime_us > primitive_cap_us:
        raise ValueError("recovery deadline exceeds primitive cap")
    return runtime_us


def recovery_wrong_way(start_offset_counts: int, current_offset_counts: int) -> bool:
    """Match the one-count-inclusive growth guard used by the safety worker."""
    if start_offset_counts == -(1 << 31) or current_offset_counts == -(1 << 31):
        return True
    return abs(current_offset_counts) > abs(start_offset_counts) + 1


def audit_rung_origin_recovery(
    record: dict[str, object], plan: dict[str, object]
) -> dict[str, object]:
    """Audit one emitted recovery summary against its authoritative plan geometry."""
    mismatches: list[str] = []
    if int(plan.get("schema_revision", 0)) >= 10:
        planned_velocity = abs(int(plan["target_velocity_rpm"])) * 1_000 // 60
    else:
        planned_velocity = int(plan["planned_velocity_mrev_s"])
    origin_band = int(plan["recovery_origin_band_counts"])
    maximum_slot_us = int(plan["recovery_maximum_slot_us"])
    if int(record["planned_velocity_mrev_s"]) != planned_velocity:
        mismatches.append("recovery planned velocity differs from plan")
    if int(record["origin_band_counts"]) != origin_band:
        mismatches.append("recovery origin band differs from plan")
    start = int(record["start_offset_counts"])
    closest = int(record["closest_offset_counts"])
    final = int(record["final_offset_counts"])
    lower_rate_q = int(record["lower_rate_q"])
    outcome = int(record["outcome"])
    binding = int(record["binding_source"])
    historical_schema_five_contact = (
        int(plan.get("schema_revision", 0)) == 5 and outcome == 4 and binding == 6
    )
    schema_six_contact = (
        int(plan.get("schema_revision", 0)) >= 6 and outcome in (4, 5) and binding == 6
    )
    moving_us = int(record["moving_duration_us"])
    settle_us = int(record["settle_duration_us"])
    total_us = int(record["total_duration_us"])
    if int(record["stage"]) not in (0, 1):
        mismatches.append("recovery stage is invalid")
    if recovery_wrong_way(start, closest):
        mismatches.append("recovery record contains wrong-way growth")
    if (
        int(record["peak_torque_target_abs"]) >= int(plan["usable_torque_limit"])
        and not historical_schema_five_contact
        and not schema_six_contact
    ):
        mismatches.append("recovery peak torque reached the usable limit")
    return_distance = recovery_return_distance_counts(start, origin_band)
    expected: tuple[tuple[int, int], ...]
    runtime_us: int | None = None
    if return_distance == 0:
        expected = ((0, 0),)
    elif lower_rate_q == 0:
        expected = ((2, 2),)
    else:
        try:
            runtime_us = recovery_runtime_us(return_distance, lower_rate_q, maximum_slot_us)
        except ValueError:
            # The declared geometry does not fit the primitive cap, so the
            # attempt is contained before any motion.
            expected = ((3, 3),)
        else:
            # Plan-time inputs cannot decide which of these firmware reports.
            # The return may reach the origin band (Recovered), complete its
            # move and settle outside it or exhaust the moving deadline
            # (ContainedDeadline/Deadline), or be refused before motion for a
            # start beyond the outer-safety position window
            # (ContainedDeadline/LocalTravel). Only the run knows.
            #
            # Naming just the Recovered pair asserted more than the inputs
            # support and reported every contained run as a mismatch. What the
            # record can still be held to is internal consistency, checked
            # below: whether its timings match the motion its outcome implies,
            # and whether its final offset matches the band semantics.
            expected = ((1, 1), (1, 5), (3, 3), (3, 4))
    if historical_schema_five_contact:
        expected = ((4, 6),)
    elif schema_six_contact:
        expected = ((outcome, 6),)
    if (outcome, binding) not in expected:
        mismatches.append(
            "recovery outcome/binding differs from reconstructed authority "
            f"(firmware={(outcome, binding)} offline={expected})"
        )
    if outcome in (0, 1, 5) and abs(final) > origin_band:
        mismatches.append("successful recovery final offset is outside origin band")
    moved = moving_us != 0 or settle_us != 0 or total_us != 0
    # Outcomes 0 and 2 are decided before the gain write, and outcome 3 bound by
    # LocalTravel is refused at the same point. Outcome 3 bound by Deadline has
    # two producers: one before motion when the geometry exceeds the primitive
    # cap, and one after a completed move, so it is the only outcome whose
    # timings are not fixed by the outcome alone.
    if (outcome in (0, 2) or (outcome == 3 and binding == 4)) and moved:
        mismatches.append("nonmoving recovery reports nonzero timing")
    if outcome in (4, 5) and schema_six_contact or outcome == 1 or (outcome == 3 and moved):
        if total_us != moving_us + settle_us:
            mismatches.append("recovery total duration differs from moving plus settle duration")
        if runtime_us is not None and total_us > runtime_us:
            mismatches.append("recovery total duration exceeds reconstructed deadline")
    return {
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "historical_attribution_caveat": historical_schema_five_contact,
    }


def audit_recovery_rest_geometry(
    record: dict[str, object],
    plan: dict[str, object],
    *,
    expected_stage: int,
) -> list[str]:
    """Check stage-neutral recovery-rest geometry and scored evidence."""
    mismatches: list[str] = []
    plan_digest = (
        int(plan["plan_digest"])
        if "plan_digest" in plan
        else int(plan["plan_digest_low"]) | (int(plan["plan_digest_high"]) << 32)
    )
    start = int(record["start_offset_counts"])
    first_in_band = int(record["first_in_band_offset_counts"])
    final = int(record["final_offset_counts"])
    origin_band = int(plan["recovery_origin_band_counts"])
    planned_return = recovery_return_distance_counts(start, origin_band)
    if int(record["stage"]) != expected_stage:
        mismatches.append("recovery rest carries the wrong stage")
    if int(record["plan_digest"]) != plan_digest:
        mismatches.append("recovery rest plan digest differs")
    if planned_return is None or int(record["planned_return_distance_counts"]) != planned_return:
        mismatches.append("recovery rest planned return differs")
    if abs(first_in_band) > origin_band:
        mismatches.append("recovery rest first in-band offset is outside the band")
    if int(record["displacement_to_band_counts"]) != first_in_band - start:
        mismatches.append("recovery rest entry displacement differs")
    if abs(final) > origin_band:
        mismatches.append("recovery rest final offset is outside the band")
    lead_net = int(record["lead_in_net_displacement_counts"])
    lead_span = int(record["lead_in_displacement_span_counts"])
    if abs(lead_net) > lead_span:
        mismatches.append("recovery rest lead-in net exceeds its span")
    scored_net = int(record["scored_net_displacement_counts"])
    scored_min = int(record["scored_min_displacement_counts"])
    scored_max = int(record["scored_max_displacement_counts"])
    if not scored_min <= scored_net <= scored_max:
        mismatches.append("recovery rest scored displacement is inconsistent")
    return mismatches


def summarize_recovery_rest_diagnostics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Return an exact, report-only distribution over recovery diagnostics."""
    rows = sorted(
        (dict(record) for record in records),
        key=lambda record: (
            int(record["planned_return_distance_counts"]),
            int(record["stage"]),
            int(record["rung_index"]),
        ),
    )

    def distribution(name: str, *, absolute: bool = False) -> dict[str, int] | None:
        values = sorted(abs(int(row[name])) if absolute else int(row[name]) for row in rows)
        if not values:
            return None
        return {
            "min": values[0],
            "median": values[(len(values) - 1) // 2],
            "max": values[-1],
        }

    return {
        "diagnostic_count": len(rows),
        "planned_return_distance_counts": distribution("planned_return_distance_counts"),
        "entry_to_band_duration_us": distribution("entry_to_band_duration_us"),
        "absolute_lead_in_net_counts": distribution(
            "lead_in_net_displacement_counts", absolute=True
        ),
        "lead_in_span_counts": distribution("lead_in_displacement_span_counts"),
        "rows": rows,
    }


def _f32(value: float | int) -> float:
    """Round one value to the firmware's IEEE-754 single-precision domain."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3


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
    for lower, upper in itertools.pairwise(levels):
        if _plateau_like(lower.moments, upper.moments):
            consecutive += 1
            if consecutive >= 2:
                return True
        else:
            consecutive = 0
    return False


def _round_away(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _drift_ratio_ppm(slope_q: int, span_x2_us: int, mean_rate_q: int) -> int:
    """Drift across the regression span in parts per million of settled rate.

    Integer only, mirroring firmware exactly. The ppm factor cancels against the
    fixed-point rate units, and the span is doubled so the single ceiling
    division applies to the exact rational.
    """
    denominator = 2 * abs(mean_rate_q)
    if denominator == 0:
        return 0xFFFFFFFF
    numerator = abs(slope_q) * span_x2_us
    return min(-(-numerator // denominator), 0xFFFFFFFF)


def _stationarity_fit(
    observations: list[VelocityObservation],
    selection: int,
    variance_of_mean_q2: int,
    mean_rate_q: int,
) -> dict[str, int | bool] | int:
    """Fit the block-level regression shared by the moving and rest gates.

    Applies no significance or drift gate: those decisions belong to
    `_classify_stationarity`, which needs the mode to know which gate applies.
    Returns 1 (too few decorrelated blocks) or 4 (arithmetic failure) if the
    fit itself cannot be formed; `mean_rate_q` is the caller's settled mean for
    the selected plateau, needed here only to compute `drift_ratio_ppm`.
    """
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
    lag_one = _f32(0.0) if residual_sum == 0 else _f32(lag_product_sum / residual_sum)
    lag_null = _f32(_f32(-1.0) / _f32(n))
    lag_half_width = _f32(multiplier / _f32(math.sqrt(_f32(n))))
    slope_q = _round_away(slope_per_second)
    first_block_elapsed_us = blocks[0].elapsed_us
    last_block_elapsed_us = blocks[-1].elapsed_us
    span_x2_us = 2 * elapsed - first_block_elapsed_us - last_block_elapsed_us
    return {
        "slope_q": slope_q,
        "slope_half_width_q": math.ceil(slope_half_width),
        "lag_one_q": _round_away(_f32(lag_one * _f32(1_000_000))),
        "lag_one_half_width_q": math.ceil(_f32(lag_half_width * _f32(1_000_000))),
        "residual_magnitude_q2": math.ceil(_f32(residual_sum / _f32(n))),
        # Significance is decided against the raw float margin, not the
        # rounded Q value: rounding either side independently can flip a
        # boundary case relative to the exact firmware comparison.
        "slope_passed": abs(slope_per_second) <= slope_half_width,
        "lag_one_passed": abs(_f32(lag_one - lag_null)) <= lag_half_width,
        "drift_ratio_ppm": _drift_ratio_ppm(slope_q, span_x2_us, mean_rate_q),
    }


def _classify_stationarity(
    fit: dict[str, int | bool],
    mode: str,
    mean_rate_q: int,
    half_width_rate_q: int,
    *,
    build_features: int | None = None,
) -> int | None:
    """Apply the mode-specific stationarity gate to an unguarded fit.

    `mode` is `"moving"` or `"rest"`. A moving stroke recorded by firmware whose
    `build_features` declares `BUILD_FEATURE_DRIFT_BOUND_GATE` is bounded on how
    far the settled rate drifts across the regression span, not on whether the
    fitted slope is statistically distinguishable from zero: the significance
    test tightens as data accumulates, so a longer, better measurement of a real
    machine rejects where a shorter one passed. Firmware without that bit never
    ran the drift bound, so `mode="moving"` there keeps the old significance test
    instead, matching that firmware exactly rather than a gate it never executed.

    `build_features` is the recording firmware's rule contract, resolved by the
    caller from the capture (see `capture_rule_features`): the drift-bound rule
    is read straight off the wire rather than dated by schema revision, so the
    Proportional / Integral schema-namespace ambiguity never reaches this gate.
    `None` means unknown -- an unidentified capture is far more likely to be an
    old one, so it defaults toward the legacy significance gate, failing safe
    toward preserving existing replay behavior.

    Zero-target rest always keeps the significance test unchanged (it never
    reads `build_features`), since a drift bound relative to a near-zero mean
    would collapse with the mean itself.

    Returns the zero-based `StationarityExclusion` discriminant on rejection,
    or `None` when the fit is usable.
    """
    if mode == "moving":
        use_drift_bound = build_features is not None and bool(
            build_features & BUILD_FEATURE_DRIFT_BOUND_GATE
        )
        if use_drift_bound:
            # The settled interval containing zero is exactly the Inert
            # condition, which bypasses the drift bound entirely rather than
            # reporting a meaningless ratio against a near-zero denominator.
            interval_contains_zero = abs(mean_rate_q) <= half_width_rate_q
            if not interval_contains_zero and fit["drift_ratio_ppm"] > MAX_MOVING_DRIFT_PPM:
                return 5
        elif not fit["slope_passed"]:
            return 2
    elif mode == "rest":
        if not fit["slope_passed"]:
            return 2
    else:
        raise ValueError(f"unknown stationarity mode: {mode!r}")
    if not fit["lag_one_passed"]:
        return 3
    return None


def _stationary_suffix(
    observations: list[VelocityObservation],
    *,
    build_features: int | None = None,
) -> dict[str, int]:
    """Reconstruct the longest usable moving-stroke stationary suffix.

    `build_features` is the recording firmware's rule contract, resolved by the
    caller from the capture (see `capture_rule_features`). It selects which
    moving gate `_classify_stationarity` applies, and whether a rejected window
    ends the walk (firmware without `BUILD_FEATURE_SUFFIX_WALK_CONTINUES`) or the
    walk continues to the longer windows (firmware with it). `None` means unknown
    and defaults to the legacy significance gate with the historical walk.
    """
    walk_past_rejection = build_features is not None and bool(
        build_features & BUILD_FEATURE_SUFFIX_WALK_CONTINUES
    )
    selected: dict[str, int] | None = None
    # Reported when the walk runs out of windows without selecting one.
    last_rejection: dict[str, int] | None = None
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
            last_rejection = {
                "usable": 0,
                "exclusion": 1 if _has_two_doubling_plateau(levels) else 0,
                "tested_suffixes": tested,
                # No regression was ever attempted, so there is no ratio to
                # report. Mirrors firmware's poll_blocking, which reports this
                # same sentinel when select_plateau() finds nothing.
                "drift_ratio_ppm": NO_REPORTABLE_DRIFT_RATIO_PPM,
            }
            if not walk_past_rejection:
                return last_rejection
            continue
        variance = _q(levels[level].moments)
        assert variance is not None
        exact_delta = levels[level].exact_delta_counts
        exact_elapsed = levels[level].exact_elapsed_us
        mean_rate_q = _trunc_div(exact_delta * 1_000_000 * RATE_Q_SCALE, exact_elapsed)
        half_width_rate_q = _confidence_half_width(variance, 3)
        fit = _stationarity_fit(suffix, level, variance, mean_rate_q)
        if isinstance(fit, int):
            if selected is not None:
                selected["tested_suffixes"] = tested
                return selected
            last_rejection = {
                "usable": 0,
                "exclusion": fit,
                "tested_suffixes": tested,
                # Same as above: the plateau was too thin for a regression, so
                # slope_q and span_x2_us never existed to form a ratio from.
                "drift_ratio_ppm": NO_REPORTABLE_DRIFT_RATIO_PPM,
            }
            if not walk_past_rejection:
                return last_rejection
            continue
        exclusion = _classify_stationarity(
            fit,
            "moving",
            mean_rate_q,
            half_width_rate_q,
            build_features=build_features,
        )
        if exclusion is not None:
            if selected is not None:
                selected["tested_suffixes"] = tested
                return selected
            last_rejection = {
                "usable": 0,
                "exclusion": exclusion,
                "tested_suffixes": tested,
                # A post-fit rejection evaluated a ratio, so it reports it,
                # matching firmware's finish_on_exclusion call sites that pass
                # metrics.drift_ratio_ppm rather than the sentinel.
                "drift_ratio_ppm": fit["drift_ratio_ppm"],
                # The fit succeeded and only the classification refused it, so
                # every statistic behind the verdict exists. Firmware zeroes
                # these on its own excluded records, so they cannot join the
                # fields above without the replay disagreeing with the device it
                # exists to verify. They are offline-derived diagnostics, kept
                # under their own key and excluded from every field comparison.
                #
                # The two earlier returns carry no such key. Neither reached a
                # regression, so there is nothing to derive, and a zero there
                # would read as a measurement of no structure rather than an
                # absence of evidence.
                "offline_exclusion_detail": {
                    "suffix_len": suffix_len,
                    "selected_level": level,
                    "selected_blocks": levels[level].moments.count,
                    "variance_of_mean_q2": variance,
                    # The rate this observation would have carried had the gate
                    # admitted it. Both were computed before the classification
                    # refused, so reporting them asks what the refusal cost
                    # rather than inventing a value: it is what makes "would
                    # admitting this have changed the rung decision" answerable.
                    "would_be_rate_mean_q": mean_rate_q,
                    "would_be_rate_half_width_q": half_width_rate_q,
                    "slope_q": fit["slope_q"],
                    "slope_half_width_q": fit["slope_half_width_q"],
                    "slope_passed": fit["slope_passed"],
                    "lag_one_q": fit["lag_one_q"],
                    "lag_one_half_width_q": fit["lag_one_half_width_q"],
                    "lag_one_passed": fit["lag_one_passed"],
                    "residual_magnitude_q2": fit["residual_magnitude_q2"],
                },
            }
            if not walk_past_rejection:
                return last_rejection
            continue
        selected = {
            "usable": 1,
            "suffix_len": suffix_len,
            "selected_level": level,
            "selected_blocks": levels[level].moments.count,
            "variance_of_mean_q2": variance,
            "selected_delta_counts": exact_delta,
            "selected_elapsed_us": exact_elapsed,
            "settled_rate_mean_q": mean_rate_q,
            "tested_suffixes": tested,
            "slope_q": fit["slope_q"],
            "slope_half_width_q": fit["slope_half_width_q"],
            "lag_one_q": fit["lag_one_q"],
            "lag_one_half_width_q": fit["lag_one_half_width_q"],
            "residual_magnitude_q2": fit["residual_magnitude_q2"],
            "drift_ratio_ppm": fit["drift_ratio_ppm"],
        }
    if selected is not None:
        selected["tested_suffixes"] = tested
        return selected
    if last_rejection is not None:
        last_rejection["tested_suffixes"] = tested
        return last_rejection
    return {
        "usable": 0,
        "exclusion": 1,
        "tested_suffixes": tested,
        "drift_ratio_ppm": NO_REPORTABLE_DRIFT_RATIO_PPM,
    }


def analyze_scored_rest_selection(
    observations: list[VelocityObservation],
    *,
    rest_walk_revision: bool = False,
) -> dict[str, object]:
    """Recompute one causal scored-rest decision from raw observations.

    `rest_walk_revision` selects whether an unfittable shortest window vetoes the
    longer ones (historical) or only evidence against rest closes the chain.
    """
    if len(observations) < 256:
        raise ValueError("scored-rest analysis requires at least 256 observations")
    if any(item.elapsed_us <= 0 for item in observations):
        raise ValueError("elapsed_us must be positive")

    structural_codes: list[int] = []
    intervals_q: list[list[int]] = []
    interval_present_mask = 0
    usable_mask = 0
    zero_compatible_mask = 0
    for index, suffix_len in enumerate((64, 128, 256)):
        suffix = observations[-suffix_len:]
        levels = [_Level() for _ in range(5)]
        for observation in suffix:
            _observe(levels, observation)
        selection = _select_level(levels)
        interval = [0, 0]
        if selection is None:
            code = 2 if _has_two_doubling_plateau(levels) else 1
        else:
            variance = _q(levels[selection].moments)
            assert variance is not None
            mean_rate_q = _trunc_div(
                levels[selection].exact_delta_counts * 1_000_000 * RATE_Q_SCALE,
                levels[selection].exact_elapsed_us,
            )
            fit = _stationarity_fit(suffix, selection, variance, mean_rate_q)
            if isinstance(fit, int):
                exclusion: int | None = fit
            else:
                half_width_rate_q = _confidence_half_width(variance, 3)
                exclusion = _classify_stationarity(fit, "rest", mean_rate_q, half_width_rate_q)
            code = {1: 2, 2: 3, 3: 4, 4: 5}[exclusion] if exclusion is not None else 0
            if code in (0, 3, 4):
                mean = _trunc_div(
                    levels[selection].exact_delta_counts * 1_000_000 * RATE_Q_SCALE,
                    levels[selection].exact_elapsed_us,
                )
                half_width = _confidence_half_width(variance, 3)
                interval = [mean - half_width, mean + half_width]
                interval_present_mask |= 1 << index
        structural_codes.append(code)
        intervals_q.append(interval)
        if code == 0:
            usable_mask |= 1 << index
            if interval[0] <= 0 <= interval[1]:
                zero_compatible_mask |= 1 << index

    cumulative = 0
    minimum = 0
    maximum = 0
    for observation in observations:
        cumulative += observation.delta_counts
        minimum = min(minimum, cumulative)
        maximum = max(maximum, cumulative)
    one_count_fallback = maximum - minimum <= 1
    if rest_walk_revision:
        selected = select_causal_rest_suffixes(
            structural_codes,
            intervals_q,
            interval_present_mask,
            one_count_fallback=one_count_fallback,
        )
    else:
        selected_mask = 0
        selected_suffix_len = 0
        chain_open = True
        for index, suffix_len in enumerate((64, 128, 256)):
            bit = 1 << index
            contradicts_rest = zero_compatible_mask & bit == 0
            if chain_open and zero_compatible_mask & bit:
                selected_mask = bit
                selected_suffix_len = suffix_len
            elif contradicts_rest:
                chain_open = False
        if selected_mask != 0:
            disposition = 0
        elif one_count_fallback:
            disposition = 1
        else:
            disposition = 2
        selected = {
            "selected_mask": selected_mask,
            "selected_suffix_len": selected_suffix_len,
            "disposition": disposition,
            "usable_mask": usable_mask,
            "zero_compatible_mask": zero_compatible_mask,
        }
    return {
        "structural_codes": structural_codes,
        "interval_present_mask": interval_present_mask,
        "usable_mask": selected["usable_mask"],
        "zero_compatible_mask": selected["zero_compatible_mask"],
        "selected_mask": selected["selected_mask"],
        "selected_suffix_len": selected["selected_suffix_len"],
        "disposition": selected["disposition"],
        "intervals_q": intervals_q,
        "conditioned_rest_evidence": selected["disposition"] == 0,
    }


def verify_scored_rest_selection(
    observations: list[VelocityObservation],
    record: dict[str, object],
    *,
    rest_walk_revision: bool = False,
) -> dict[str, object]:
    """Require a decoded Event 151 to match independent raw-sample analysis."""
    reconstructed = analyze_scored_rest_selection(
        observations, rest_walk_revision=rest_walk_revision
    )
    for field_name, expected in reconstructed.items():
        if record.get(field_name) != expected:
            raise ValueError(
                f"scored-rest {field_name} mismatch: "
                f"firmware={record.get(field_name)!r} offline={expected!r}"
            )
    return reconstructed


def _stream_at(stream: list, index: int) -> dict | None:
    """Return stream[index], or None past either end."""
    return stream[index] if 0 <= index < len(stream) else None


def audit_scored_rest_grammar(
    records: list[dict[str, object]],
    terminal: dict[str, object],
    *,
    schema_revision: int,
    schema_floor: int,
    expected_stage: int,
    ordinary_owners: tuple[int, ...],
    recovery_owners: tuple[int, ...],
    observation_kind: str,
    plan_digest: int,
    label: str,
    stabilized_recovery_rest: bool,
    ordinary_rejection_tail: tuple[str, ...],
) -> list[str]:
    """Validate the fixed Event 151 joins for one commissioning stage."""
    selections = [record for record in records if record["record_kind"] == "rest_selection"]
    if schema_revision < schema_floor:
        return [f"historical {label} carries scored-rest selections"] if selections else []

    mismatches: list[str] = []
    stream = [*records, terminal]
    for index, record in enumerate(records):
        kind = str(record["record_kind"])
        previous = stream[index - 1] if index else {}
        successor = stream[index + 1]
        if kind == observation_kind:
            if (
                successor.get("record_kind") != "rest_selection"
                or int(successor.get("owner", -1)) not in ordinary_owners
                or int(successor.get("rung_index", -1)) != int(record["rung_index"])
                or int(successor.get("slot", -1)) != int(record["slot"])
            ):
                mismatches.append(f"{label} observation lacks its causal scored-rest selection")
        elif kind == "recovery_rest":
            if (
                previous.get("record_kind") != "rest_selection"
                or int(previous.get("owner", -1)) not in recovery_owners
                or int(previous.get("rung_index", -1)) != int(record["rung_index"])
            ):
                mismatches.append(f"{label} recovery rest lacks its causal selection")
        elif kind != "rest_selection":
            continue

        if kind != "rest_selection":
            continue
        owner = int(record["owner"])
        if (
            int(record["stage"]) != expected_stage
            or int(record["plan_digest"]) != plan_digest
            or int(record["run_sequence"]) != int(terminal["run_sequence"])
        ):
            mismatches.append(f"{label} scored-rest selection has the wrong identity")
        if owner in ordinary_owners:
            matched_observation = (
                previous.get("record_kind") == observation_kind
                and int(previous.get("rung_index", -1)) == int(record["rung_index"])
                and int(previous.get("slot", -1)) == int(record["slot"])
            )
            if not matched_observation:
                mismatches.append(f"{label} scored-rest selection has the wrong predecessor")
            if int(record["disposition"]) == 2:
                position = index + 1
                intact = True
                for expected_kind in ordinary_rejection_tail:
                    entry = _stream_at(stream, position)
                    if (
                        entry is None
                        or entry.get("record_kind") != expected_kind
                        or int(entry.get("rung_index", -1)) != int(record["rung_index"])
                        or int(entry.get("slot", -1)) != int(record["slot"])
                    ):
                        intact = False
                        break
                    position += 1
                closing = _stream_at(stream, position)
                if not intact or closing is not terminal or int(terminal.get("cause", -1)) != 53:
                    mismatches.append(
                        f"{label} rejected scored-rest selection lacks its causal terminal tail"
                    )
        elif owner in recovery_owners:
            rejected = int(record["disposition"]) == 2
            if rejected and not stabilized_recovery_rest:
                # An immediate-scored rejection faults straight to the terminal,
                # with no recovery record and no Event 150. Require exactly that
                # rather than exempting the case.
                if successor is not terminal or int(terminal.get("cause", -1)) != 53:
                    mismatches.append(
                        f"{label} rejected recovery selection lacks its causal terminal"
                    )
            elif stabilized_recovery_rest:
                if successor.get("record_kind") != "recovery_rest" or int(
                    successor.get("rung_index", -1)
                ) != int(record["rung_index"]):
                    mismatches.append(f"{label} recovery selection does not precede Event 150")
                elif rejected:
                    closing = _stream_at(stream, index + 2)
                    if closing is not terminal or int(terminal.get("cause", -1)) != 53:
                        mismatches.append(
                            f"{label} rejected recovery selection lacks its causal terminal"
                        )
            elif successor.get("record_kind") != "rung_origin_recovery" or int(
                successor.get("rung_index", -1)
            ) != int(record["rung_index"]):
                mismatches.append(f"{label} recovery selection does not precede its recovery")
        else:
            mismatches.append(f"{label} scored-rest selection has the wrong owner")
    return mismatches


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


# --- Breakaway-seeded P ladder --------------------------------------------
#
# Independent, from-scratch reconstruction of the breakaway campaign's search,
# planning, discovery, and confirmation math against the canonical evidence
# records `assemble_breakaway_ticks` bundles (see `protocol.py`). Every
# constant below is copied from the firmware source it mirrors (cited in each
# docstring) rather than derived from wire evidence, because none of these
# fixed planning constants ride the wire -- reconstructing the firmware's
# decision requires using the same constants the firmware itself used.

# `HALF_OCTAVE_NUM` / `HALF_OCTAVE_DEN_SHIFT` (sweep_plan.rs): half-octave
# step factor 181/128, used to regenerate the probe's search grid.
_BW_HALF_OCTAVE_NUM = 181
_BW_HALF_OCTAVE_DEN_SHIFT = 7

# `MIN_STRUCTURAL_RUNGS` / `MAX_SWEEP_RUNGS` (sweep_plan.rs).
_BW_MIN_STRUCTURAL_RUNGS = 5
_BW_MAX_SWEEP_RUNGS = 32

# `MAX_ADDITIVE_GAIN_RATIO_NUM` / `_DEN` (breakaway_plan.rs): provisional
# additive first-gap resolution guard, 3/2.
_BW_MAX_ADDITIVE_GAIN_RATIO_NUM = 3
_BW_MAX_ADDITIVE_GAIN_RATIO_DEN = 2

# `SWEEP_DIRECTION_COUNT` (sweep_plan.rs).
_BW_SWEEP_DIRECTION_COUNT = 2

# `CONSENSUS_OBSERVATIONS_PER_DIRECTION` (sweep_consensus.rs): observations
# collected per direction at one discovery rung -- the raw-observation octet
# (4 forward + 4 reverse) a nomination recompute needs.
_BW_CONSENSUS_OBSERVATIONS_PER_DIRECTION = 4

# `MAX_CONFIDENCE_FAMILY_SIZE` (autotune/velocity_measurement/decision.rs).
_BW_MAX_CONFIDENCE_FAMILY_SIZE = 128

# `MOVING_STROKE_US` / `ZERO_SETTLE_US` / `SETTLE_ANALYSIS_BUDGET_US`
# (sweep_plan.rs): fixed per-primitive duration budget, also spec section 8's
# verified 1,248,000 us total.
_BW_MOVING_STROKE_US = 512_000
_BW_ZERO_SETTLE_US = 500_000
_BW_SETTLE_ANALYSIS_BUDGET_US = 236_000

# `CONFIRMATION_FAMILY_SIZE` / band edges / capture profile / acceptance rule
# / slot order (band_confirmation.rs). The predeclared confirmation family is
# always 8, never `rung_count * observations_per_direction`.
_BW_CONFIRMATION_FAMILY_SIZE = 8
_BW_CONFIRMATION_BAND_LOWER_PERCENT = 70
_BW_CONFIRMATION_BAND_UPPER_PERCENT = 80
# Four forward observations (direction code 0) then four reverse (code 1).
_BW_CONFIRMATION_SLOT_ORDER_CODES = (0, 0, 0, 0, 1, 1, 1, 1)

# `CONFIDENCE_MULTIPLIERS_95_DF15[7]` (autotune/velocity_measurement/decision.rs):
# the two-sided 95% family-wise Bonferroni Student-t critical value at df=15
# for family size 8, matching spec section 7's `m(8) = 3.177104`.
_BW_CONFIRMATION_MULTIPLIER_95_DF15 = 3.177104

# `REQUIRED_HALF_WIDTH_FRACTION` (breakaway_reporting.rs): `(80-70)/(80+70)`.
_BW_REQUIRED_HALF_WIDTH_FRACTION = 1.0 / 15.0

# `BreakawayTerminalCause::Accepted` (breakaway_reporting.rs).
_BW_TERMINAL_CAUSE_ACCEPTED = 21
_BW_STATIONARITY_EXCESSIVE_DRIFT = 6

# `BREAKAWAY_PROPORTIONAL_SCHEMA_REVISION` (combined_plan.rs): the rules revision
# `AdditiveDiscoveryPlanned.schema_revision` reports, so an offline decoder can
# confirm it is regenerating the ladder under the exact rules the firmware
# applied rather than a stale or newer revision.
_BW_PROPORTIONAL_MIN_SCHEMA_REVISION = 15
_BW_PROPORTIONAL_SCHEMA_REVISION = 17

# `NOMINATION_SCREEN_C_SIGMA` (band_confirmation.rs): the pointwise 95%
# scale-inflation heuristic the nomination screen applies on top of the
# family-eight multiplier, `sqrt(15 / chi2_0.05,15)`.
_BW_NOMINATION_SCREEN_C_SIGMA = 1.4373055

# `INTERVAL_ENDPOINT_ROUNDING_SCALE` (band_confirmation.rs): the granularity
# at which interval endpoints are conservatively widened before containment is
# judged -- the lower endpoint always rounds down, the upper always up.
_BW_INTERVAL_ENDPOINT_ROUNDING_SCALE = 1.0e4

# `PERCENT_MILLI_SATURATION_BOUND` / permille saturation (breakaway_reporting.rs).
_BW_PERCENT_MILLI_SATURATION_BOUND = 1_000_000.0
_BW_RELATIVE_SE_PERMILLE_SATURATION_BOUND = 65_535.0


def _bw_round_outward_lower(value: float) -> float:
    """Port of `round_outward_lower` (band_confirmation.rs): round down, never
    inward, so a rounded interval can never appear narrower than its true
    value."""
    scaled = _f32(value * _f32(_BW_INTERVAL_ENDPOINT_ROUNDING_SCALE))
    return _f32(math.floor(scaled) / _f32(_BW_INTERVAL_ENDPOINT_ROUNDING_SCALE))


def _bw_round_outward_upper(value: float) -> float:
    """Port of `round_outward_upper` (band_confirmation.rs): round up, never
    inward."""
    scaled = _f32(value * _f32(_BW_INTERVAL_ENDPOINT_ROUNDING_SCALE))
    return _f32(math.ceil(scaled) / _f32(_BW_INTERVAL_ENDPOINT_ROUNDING_SCALE))


def _bw_percent_to_milli(value: float) -> int:
    """Port of `percent_to_milli` (breakaway_reporting.rs)."""
    scaled = _f32(_round_away(_f32(value * _f32(1000.0))))
    if scaled >= _BW_PERCENT_MILLI_SATURATION_BOUND:
        return int(_BW_PERCENT_MILLI_SATURATION_BOUND)
    if scaled <= -_BW_PERCENT_MILLI_SATURATION_BOUND:
        return -int(_BW_PERCENT_MILLI_SATURATION_BOUND)
    return int(scaled)


def _bw_relative_se_to_permille(value: float) -> int:
    """Port of `relative_se_to_permille` (breakaway_reporting.rs)."""
    scaled = _f32(_round_away(_f32(value * _f32(1000.0))))
    if scaled >= _BW_RELATIVE_SE_PERMILLE_SATURATION_BOUND:
        return 0xFFFF
    if scaled <= 0.0:
        return 0
    return int(scaled)


def _bw_observation_statistic(
    mean_rate_q: int, variance_of_mean_q2: int, target_rate_q: int, direction_code: int
) -> tuple[float, float] | None:
    """Port of `observation_statistic` (band_confirmation.rs).

    Returns `(response_percent, relative_standard_error)` from one raw
    `BreakawayRawObservation`'s atom, or `None` when the atom is not
    representable as a candidate response (an invalid target, or a zero or
    wrong-signed mean for the declared `direction_code`) -- the same
    condition the firmware's own `ObservationStatisticError` variants cover.
    `direction_code` is the wire direction code (0 = forward, 1 = reverse;
    `direction_report_code` in `sweep_evidence.rs`).
    """
    if target_rate_q <= 0:
        return None
    if direction_code == 0 and mean_rate_q > 0:
        commanded_magnitude_q = mean_rate_q
    elif direction_code == 1 and mean_rate_q < 0:
        commanded_magnitude_q = -mean_rate_q
    else:
        return None
    variance = _f32(variance_of_mean_q2)
    if not math.isfinite(variance):
        return None
    commanded_magnitude = _f32(commanded_magnitude_q)
    target_magnitude = _f32(target_rate_q)
    response_percent = _f32(_f32(_f32(100.0) * commanded_magnitude) / target_magnitude)
    relative_standard_error = _f32(_f32(math.sqrt(variance)) / commanded_magnitude)
    if not math.isfinite(response_percent) or not math.isfinite(relative_standard_error):
        return None
    return response_percent, relative_standard_error


def _bw_observation_interval_percent(
    mean_rate_q: int,
    variance_of_mean_q2: int,
    target_rate_q: int,
    direction_code: int,
    multiplier: float,
    c_sigma: float,
    round_outward: bool,
) -> tuple[float, float] | None:
    """Port of `observation_interval_percent` (band_confirmation.rs): build one
    observation's percent-response interval, `h = multiplier * SE * c_sigma`,
    then `[r * (1 - h), r * (1 + h)]`, optionally rounded outward.
    """
    statistic = _bw_observation_statistic(
        mean_rate_q, variance_of_mean_q2, target_rate_q, direction_code
    )
    if statistic is None:
        return None
    response_percent, relative_standard_error = statistic
    half_width_fraction = _f32(_f32(_f32(multiplier) * relative_standard_error) * _f32(c_sigma))
    if not math.isfinite(half_width_fraction):
        return None
    low_raw = _f32(response_percent * _f32(1.0 - half_width_fraction))
    high_raw = _f32(response_percent * _f32(1.0 + half_width_fraction))
    if round_outward:
        return _bw_round_outward_lower(low_raw), _bw_round_outward_upper(high_raw)
    return low_raw, high_raw


# Depth-three triples over the four per-direction slots (sweep_consensus.rs:12).
_BW_CONSENSUS_TRIPLES = (0b0111, 0b1011, 0b1101, 0b1110)


def _bw_classify_is_inert(observation: dict[str, object]) -> bool:
    """Whether one completed observation lands on firmware's `Inert` arm.

    Distinct from "not eligible". `classify_velocity_observation`
    (sweep_evidence.rs:686) checks invalid evidence, hard clip, wrong direction
    and stationarity exclusion *before* `Inert`, and current headroom and
    unresolved deficit *after* it. Only the stroke that passed every earlier
    gate and still did not move is inert: commanded, and the axis stayed put.

    That is the class the 82-84% baseline counted, so reproducing it here
    is what makes the new number comparable to the old one. Folding in the other
    exclusions would inflate it, and they have different remedies -- a hard clip
    or a headroom stop is a ceiling problem, not a dead zone.
    """
    settled_low = int(observation["settled_rate_low_q"])
    settled_high = int(observation["settled_rate_high_q"])
    deficit_low = int(observation["deficit_low_q"])
    deficit_high = int(observation["deficit_high_q"])
    usable = int(observation["usable_torque_limit"])
    hard = int(observation["hard_torque_limit"])
    predicted = int(observation["predicted_torque_target_abs"])
    forward = int(observation["direction"]) == 0

    if not (settled_low <= settled_high and deficit_low <= deficit_high and usable <= hard):
        return False  # InvalidEvidenceFault
    if predicted >= hard:
        return False  # HardClipFault
    if (settled_high < 0) if forward else (settled_low > 0):
        return False  # WrongDirectionFault
    if int(observation["stationarity_code"]) != 0:
        return False  # StationarityExcluded
    moving = settled_low > 0 if forward else settled_high < 0
    return not moving


def _bw_discovery_inert_fraction(
    observations: list[dict[str, object]],
    ladder_gains: list[int],
    breakaway_p_raw: int | None,
) -> dict[str, object]:
    """The discovery-ladder inert fraction, as two ratios.

    The denominator is completed discovery observations only. Callers pass the
    marker-160 records of the Discovery phase, so the upward breakaway probe,
    the held-out confirmation, Integral, and every recovery or hidden observation
    are excluded by construction rather than by filtering here.

    `gross` includes the intentional predecessor floor rung. It is the honest
    cost of the ladder as planned, and the number the "falls substantially"
    criterion is judged on.

    `unexpected_above_breakaway` covers rungs at or above the resolved
    breakaway. Inert strokes there are the ones the seed was supposed to
    eliminate, so a nonzero ratio means the seed or the additive spacing is not
    doing its job, whatever the gross figure looks like.
    """
    # Observations name their ladder index; the gain itself lives in the
    # independently regenerated ladder, which is what makes the breakaway
    # comparison below meaningful rather than a restatement of firmware's own
    # rung numbering.
    by_rung: dict[int, list[dict[str, object]]] = {}
    for observation in observations:
        rung_index = int(observation["rung_index"])
        if rung_index >= len(ladder_gains):
            continue
        by_rung.setdefault(ladder_gains[rung_index], []).append(observation)

    def ratio(rungs: list[int]) -> dict[str, object]:
        completed = sum(len(by_rung[p_raw]) for p_raw in rungs)
        inert = sum(
            _bw_classify_is_inert(observation) for p_raw in rungs for observation in by_rung[p_raw]
        )
        return {
            "inert": inert,
            "completed": completed,
            "fraction": (inert / completed) if completed else None,
        }

    every_rung = sorted(by_rung)
    above = (
        [p_raw for p_raw in every_rung if p_raw >= breakaway_p_raw]
        if breakaway_p_raw is not None
        else []
    )
    return {
        "breakaway_p_raw": breakaway_p_raw,
        "gross": ratio(every_rung),
        "unexpected_above_breakaway": ratio(above),
        "inert_rungs_above_breakaway": [
            p_raw
            for p_raw in above
            if any(_bw_classify_is_inert(observation) for observation in by_rung[p_raw])
        ],
        "per_rung": [
            {
                "p_raw": p_raw,
                "completed": len(by_rung[p_raw]),
                "inert": sum(_bw_classify_is_inert(observation) for observation in by_rung[p_raw]),
                "above_breakaway": breakaway_p_raw is not None and p_raw >= breakaway_p_raw,
            }
            for p_raw in every_rung
        ],
    }


def _bw_classify_is_eligible(observation: dict[str, object]) -> bool:
    """Port of `classify_velocity_observation` (sweep_evidence.rs:663), reduced
    to the one verdict the consensus needs: a stroke is `Eligible` iff it passes
    every gate the firmware cascade checks before its `else` arm. Reproduced from
    the frozen inputs the raw-observation record now carries."""
    settled_low = int(observation["settled_rate_low_q"])
    settled_high = int(observation["settled_rate_high_q"])
    deficit_low = int(observation["deficit_low_q"])
    deficit_high = int(observation["deficit_high_q"])
    usable = int(observation["usable_torque_limit"])
    hard = int(observation["hard_torque_limit"])
    predicted = int(observation["predicted_torque_target_abs"])
    forward = int(observation["direction"]) == 0

    valid_intervals = settled_low <= settled_high and deficit_low <= deficit_high and usable <= hard
    if not valid_intervals:
        return False  # InvalidEvidenceFault
    if predicted >= hard:
        return False  # HardClipFault
    wrong_direction = settled_high < 0 if forward else settled_low > 0
    if wrong_direction:
        return False  # WrongDirectionFault
    if int(observation["stationarity_code"]) != 0:
        return False  # StationarityExcluded
    moving = settled_low > 0 if forward else settled_high < 0
    if not moving:
        return False  # Inert
    if predicted >= usable:
        return False  # CurrentHeadroom
    deficit_resolved = deficit_low > 0 if forward else deficit_high < 0
    if not deficit_resolved:
        return False  # DeficitUnresolved
    # False: InvalidEvidenceFault (no representable disturbance). True: Eligible.
    return int(observation["disturbance_present"]) != 0


def _bw_directional_included_mask(direction_observations: list[dict[str, object]]) -> int:
    """Port of `depth_three_consensus` (sweep_consensus.rs:67) over one
    direction's four slots. Derives `collected_mask` from the received slots,
    `eligible_mask` from the ported classification, and `included_mask` from the
    triple-intersection depth-three consensus over the streamed disturbance
    intervals. Returns the `included_mask` (0 when the direction reaches no
    single-component eligible consensus)."""
    by_slot = {int(obs["slot_index"]): obs for obs in direction_observations}
    collected_mask = 0
    eligible_mask = 0
    intervals: dict[int, tuple[int, int]] = {}
    for slot, obs in by_slot.items():
        if not 0 <= slot < 4:
            continue
        collected_mask |= 1 << slot
        if _bw_classify_is_eligible(obs):
            eligible_mask |= 1 << slot
        if int(obs["disturbance_present"]) == 1:
            intervals[slot] = (
                int(obs["disturbance_low_q16"]),
                int(obs["disturbance_high_q16"]),
            )
    if collected_mask != 0b1111:
        return 0  # IncompleteRung
    if bin(eligible_mask).count("1") < 3:
        return 0  # InsufficientConsensus
    intersections: list[tuple[int, int]] = []
    for triple in _BW_CONSENSUS_TRIPLES:
        if triple & eligible_mask != triple:
            continue
        low = None
        high = None
        complete = True
        for slot in range(4):
            if triple & (1 << slot) == 0:
                continue
            interval = intervals.get(slot)
            if interval is None:
                complete = False
                break
            low = interval[0] if low is None else max(low, interval[0])
            high = interval[1] if high is None else min(high, interval[1])
        if not complete or low is None or high is None:
            continue
        if low <= high:
            intersections.append((low, high))
    intersections.sort()
    components: list[list[int]] = []
    for low, high in intersections:
        if components and low <= components[-1][1]:
            components[-1][1] = max(components[-1][1], high)
        else:
            components.append([low, high])
    if len(components) != 1:
        return 0  # zero (Insufficient) or multiple (Ambiguous) components
    comp_low, comp_high = components[0]
    included_mask = 0
    for slot in range(4):
        if eligible_mask & (1 << slot) == 0:
            continue
        interval = intervals.get(slot)
        if interval is None:
            continue
        if interval[0] <= comp_high and comp_low <= interval[1]:
            included_mask |= 1 << slot
    return included_mask


def _bw_post_breakaway_transients(
    observations: list[dict[str, object]], ladder_gains: list[int]
) -> list[dict[str, int]]:
    """Derive observed transients within the collected discovery span.

    A diagnostic is emitted only after the same direction has an earlier rung
    with the existing `EligibleConsensus` result.  Nothing is inferred beyond
    the last collected rung.
    """
    by_rung_direction: dict[tuple[int, int], list[dict[str, object]]] = {}
    for observation in observations:
        rung_index = int(observation["rung_index"])
        if not 0 <= rung_index < len(ladder_gains):
            continue
        key = (rung_index, int(observation["direction"]))
        by_rung_direction.setdefault(key, []).append(observation)

    diagnostics: list[dict[str, int]] = []
    latest_consensus: list[int | None] = [None, None]
    for rung_index in sorted({key[0] for key in by_rung_direction}):
        for direction in (0, 1):
            rung_observations = by_rung_direction.get((rung_index, direction), [])
            consensus_rung = latest_consensus[direction]
            if consensus_rung is not None and rung_index > consensus_rung:
                for observation in rung_observations:
                    if (
                        int(observation["stationarity_code"]) == _BW_STATIONARITY_EXCESSIVE_DRIFT
                        and observation.get("post_fit_lag_one_state") == 2
                    ):
                        diagnostics.append(
                            {
                                "direction": direction,
                                "consensus_rung_index": consensus_rung,
                                "consensus_p_raw": ladder_gains[consensus_rung],
                                "transient_rung_index": rung_index,
                                "transient_p_raw": ladder_gains[rung_index],
                                "slot_index": int(observation["slot_index"]),
                            }
                        )
            if _bw_directional_included_mask(rung_observations) != 0:
                latest_consensus[direction] = rung_index
    return diagnostics


def _bw_included_observations(
    direction_observations: list[dict[str, object]], included_mask: int
) -> list[dict[str, object]]:
    """The direction's observations whose slot is set in `included_mask`."""
    return [obs for obs in direction_observations if included_mask & (1 << int(obs["slot_index"]))]


def _bw_rung_response_bounds_from_consensus(
    forward: list[dict[str, object]],
    reverse: list[dict[str, object]],
    multiplier: float,
    c_sigma: float,
    round_outward: bool,
) -> tuple[float, float] | None:
    """Reconstruct a rung's nomination bounds from exactly the observations the
    firmware's depth-three consensus includes, derived independently here (not
    read from firmware's masks). A direction with no single-component eligible
    consensus makes the rung non-nominatable (`None`)."""
    forward_included = _bw_directional_included_mask(forward)
    reverse_included = _bw_directional_included_mask(reverse)
    if forward_included == 0 or reverse_included == 0:
        return None
    included = _bw_included_observations(forward, forward_included) + _bw_included_observations(
        reverse, reverse_included
    )
    return _bw_rung_response_bounds_percent(included, multiplier, c_sigma, round_outward)


def _bw_rung_response_bounds_percent(
    observations: list[dict[str, object]], multiplier: float, c_sigma: float, round_outward: bool
) -> tuple[float, float] | None:
    """Port of `rung_response_bounds_percent` (band_confirmation.rs): fold the
    passed observations into one combined percent-response interval by taking the
    min of every observation's lower bound and the max of every observation's
    upper bound.

    The caller passes exactly the observations the depth-three `included_mask`
    selects (see `_bw_rung_response_bounds_from_consensus`), derived
    independently from the streamed classify/consensus inputs, so this reproduces
    the firmware fold exactly rather than folding every collected observation.
    """
    lower: float | None = None
    upper: float | None = None
    for observation in observations:
        interval = _bw_observation_interval_percent(
            int(observation["mean_rate_q"]),
            int(observation["variance_of_mean_q2"]),
            int(observation["target_rate_q"]),
            int(observation["direction"]),
            multiplier,
            c_sigma,
            round_outward,
        )
        if interval is None:
            return None
        low, high = interval
        lower = low if lower is None else min(lower, low)
        upper = high if upper is None else max(upper, high)
    if lower is None or upper is None:
        return None
    return lower, upper


BREAKAWAY_EVENT_ID_NAMES = {
    152: "probe_plan",
    153: "probe_result",
    154: "probe_terminal",
    155: "discovery_plan",
    156: "discovery_terminal",
    157: "confirmation_plan",
    158: "confirmation_terminal",
    159: "campaign_terminal",
    160: "raw_observation",
    161: "discovery_rung_margin",
    162: "probe_decision",
    163: "recovery_rest_rejected",
    164: "recovery_origin_mismatch",
}

# Phase rank per marker event id: probe (0) must never follow discovery (1),
# which must never follow confirmation (2); the campaign terminal (3) is
# always the campaign's last tick, from any phase. The raw-observation marker
# (160) is shared by probe, discovery, and confirmation strokes, so its rank is not
# fixed here -- `_bw_tick_phase_rank` resolves it from the streamed record's own
# `phase` field. The discovery-margin marker (161) is discovery-only.
_BW_EVENT_PHASE_RANK = {
    152: 0,
    153: 0,
    154: 0,
    155: 1,
    156: 1,
    157: 2,
    158: 2,
    159: 3,
    161: 1,
    162: 0,
}

# Raw-observation record `phase` field values (BreakawayPhase as u8): the
# streamed marker-160 tick belongs to whichever phase its record names.
_BW_RAW_OBSERVATION_PHASE_PROBE = 0
_BW_RAW_OBSERVATION_PHASE_DISCOVERY = 1
_BW_RAW_OBSERVATION_PHASE_CONFIRMATION = 2


# Markers whose owning phase is carried by the record rather than the id.
# The raw-observation stream (160) shares one id across all three phases, and
# the two protocol-18 recovery diagnostics (163, 164) are emitted from whichever
# phase was running when their gate refused.
_BW_PHASE_BEARING_MARKERS = {
    160: "breakaway_raw_observation",
    163: "breakaway_recovery_rest_rejected",
    164: "breakaway_recovery_origin_mismatch",
}


def _bw_tick_phase_rank(tick: dict[str, object]) -> int:
    """Phase rank of one marker tick, resolving phase-bearing markers from
    their record's `phase` field."""
    event_id = tick["event_id"]
    record_kind = _BW_PHASE_BEARING_MARKERS.get(int(event_id))
    if record_kind is not None:
        records = _bw_tick_records_by_kind(tick, record_kind)
        phase = int(records[0]["phase"]) if records else 0
        if phase == _BW_RAW_OBSERVATION_PHASE_CONFIRMATION:
            return 2
        if phase == _BW_RAW_OBSERVATION_PHASE_DISCOVERY:
            return 1
        return 0
    return _BW_EVENT_PHASE_RANK[event_id]


def _bw_collect_raw_observations(
    ticks: list[dict[str, object]], phase: int
) -> list[dict[str, object]]:
    """Every streamed raw observation of one phase, in stream order, gathered
    across the dedicated marker-160 ticks (one record per tick)."""
    return [
        record
        for tick in ticks
        if tick["event_id"] == 160
        for record in _bw_tick_records_by_kind(tick, "breakaway_raw_observation")
        if int(record["phase"]) == phase
    ]


def _bw_collect_margins(ticks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Every streamed discovery rung margin, in stream order, gathered across
    the dedicated marker-161 ticks (one record per tick)."""
    return [
        record
        for tick in ticks
        if tick["event_id"] == 161
        for record in _bw_tick_records_by_kind(tick, "breakaway_discovery_rung_margin")
    ]


def _bw_validate_confirmation_slot_schedule(ticks: list[dict[str, object]]) -> None:
    """Enforce the fixed eight-stroke confirmation schedule on the streamed raw
    observations before any consumer collapses them by slot: the slots must be a
    prefix 0..N-1 (a complete block is 0..7, an aborted one is shorter), each
    exactly once and in stream order, with at most eight strokes. Reconstructing
    masks by OR-ing slot bits and indexing observations by slot both silently hide
    a duplicate or a ninth stroke, so this must run first."""
    confirmation_raw = _bw_collect_raw_observations(ticks, _BW_RAW_OBSERVATION_PHASE_CONFIRMATION)
    if len(confirmation_raw) > _BW_CONFIRMATION_FAMILY_SIZE:
        raise ValueError(
            "confirmation raw observations exceed the fixed eight-stroke schedule "
            f"({len(confirmation_raw)} > {_BW_CONFIRMATION_FAMILY_SIZE})"
        )
    for position, observation in enumerate(confirmation_raw):
        slot = int(observation["slot_index"])
        if slot != position:
            raise ValueError(
                f"confirmation raw observation slot {slot} breaks the fixed schedule "
                f"at stream position {position} (expected a prefix 0..N-1, each once, "
                "in order)"
            )


def _bw_validate_streamed_marker_identity(ticks: list[dict[str, object]]) -> None:
    """Bind every streamed raw-observation (marker 160), discovery-margin
    (marker 161), and probe-decision (marker 162) record to its campaign identity,
    so a misrouted or forged
    streamed record cannot pass as authentic evidence. Each such marker must
    carry exactly one record whose run_sequence matches the campaign, whose
    evidence_sequence matches its own marker, and whose plan_digest chains to the
    owning plan (probe for phase 0 or a decision, discovery for phase 1 or a
    margin, confirmation for phase 2). The
    streamed evidence_sequences must be strictly increasing in stream order.
    """
    if not ticks:
        return
    campaign_run = int(ticks[0]["records"][0][1]["run_sequence"])  # type: ignore[index]
    discovery_digest: int | None = None
    confirmation_digest: int | None = None
    for tick in ticks:
        if tick["event_id"] == 155:
            discovery_digest = int(
                _bw_single_record(tick, "breakaway_discovery_planned")["plan_digest"]
            )
        elif tick["event_id"] == 157:
            confirmation_digest = int(
                _bw_single_record(tick, "breakaway_confirmation_planned")["plan_digest"]
            )

    previous_sequence: int | None = None
    for tick in ticks:
        event_id = tick["event_id"]
        if event_id not in (160, 161, 162):
            continue
        records = tick["records"]  # type: ignore[assignment]
        if len(records) != 1:  # type: ignore[arg-type]
            raise ValueError(
                f"streamed marker {event_id} tick must carry exactly one record, got {len(records)}"
                # type: ignore[arg-type]
            )
        record = records[0][1]  # type: ignore[index]
        if int(record["run_sequence"]) != campaign_run:
            raise ValueError(
                f"streamed marker {event_id} record run_sequence does not match the campaign run"
            )
        sequence = int(record["evidence_sequence"])
        if sequence != int(tick["sequence"]):
            raise ValueError(
                f"streamed marker {event_id} record evidence_sequence does not match its marker"
            )
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError(
                "streamed marker evidence_sequence is not strictly increasing in stream order"
            )
        previous_sequence = sequence
        if event_id == 161:
            expected = discovery_digest
            what = "discovery margin"
        elif event_id == 162:
            expected = int(ticks[0]["records"][0][1]["plan_digest"])  # type: ignore[index]
            what = "probe decision"
        else:
            phase = int(record["phase"])
            if phase == _BW_RAW_OBSERVATION_PHASE_PROBE:
                expected = int(ticks[0]["records"][0][1]["plan_digest"])  # type: ignore[index]
            elif phase == _BW_RAW_OBSERVATION_PHASE_DISCOVERY:
                expected = discovery_digest
            else:
                expected = confirmation_digest
            what = "raw observation"
        if expected is None or int(record["plan_digest"]) != expected:
            raise ValueError(f"streamed {what} plan_digest does not chain to its owning plan")


def _bw_fnv1a_extend(digest: int, value: int, byte_width: int) -> int:
    """Fold one little-endian field into an FNV-1a digest.

    Mirrors every `BreakawayProbePlan::digest`/`AdditiveDiscoveryPlan::digest`/
    `BandConfirmationPlan::digest`'s shared `hash_value!` macro
    (breakaway_plan.rs, band_confirmation.rs): fold each byte of
    `value.to_le_bytes()`, least-significant first, exactly as `FNV1A64_OFFSET`/
    `FNV1A64_PRIME` already do for wire-record digests elsewhere in this file.
    """
    for shift in range(0, byte_width * 8, 8):
        digest ^= (value >> shift) & 0xFF
        digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return digest


def breakaway_build_search_grid(p_start_raw: int, p_top_raw: int) -> list[int]:
    """Regenerate the probe's post-dedupe half-octave search grid.

    Port of `build_gain_ladder` (sweep_plan.rs): an anchored half-octave
    ladder from `p_start_raw` to the exact `p_top_raw` ceiling. Integer-exact;
    no floating point.
    """
    if p_start_raw <= 0 or p_start_raw > p_top_raw or p_top_raw > 0xFFFF:
        raise ValueError("invalid breakaway search-grid span")
    odd_numerator = p_start_raw * _BW_HALF_OCTAVE_NUM
    odd_rounding = (1 << _BW_HALF_OCTAVE_DEN_SHIFT) - 1
    gains: list[int] = []
    previous = 0
    for octave in range(16):
        even_candidate = p_start_raw << octave
        odd_candidate = ((odd_numerator << octave) + odd_rounding) >> _BW_HALF_OCTAVE_DEN_SHIFT
        for candidate in (even_candidate, odd_candidate):
            if candidate > p_top_raw or candidate <= previous:
                continue
            gains.append(candidate)
            previous = candidate
    if previous != p_top_raw:
        gains.append(p_top_raw)
    return gains


def breakaway_probe_plan_digest(
    search_gains: list[int],
    motion_threshold_counts: int,
    max_capture_interval_us: int,
    maximum_observations: int,
    *,
    workflow_digest: int,
    target_velocity_rpm: int,
    hard_torque_limit: int,
    usable_torque_limit: int,
    pwm_hz: int,
    encoder_counts_per_rev: int,
) -> int:
    """Port of `BreakawayProbePlan::digest` (breakaway_plan.rs)."""
    digest = FNV1A64_OFFSET
    digest = _bw_fnv1a_extend(digest, len(search_gains), 1)
    for gain in search_gains:
        digest = _bw_fnv1a_extend(digest, gain, 2)
    digest = _bw_fnv1a_extend(digest, motion_threshold_counts, 4)
    digest = _bw_fnv1a_extend(digest, max_capture_interval_us, 4)
    digest = _bw_fnv1a_extend(digest, maximum_observations, 1)
    digest = _bw_fnv1a_extend(digest, workflow_digest, 8)
    digest = _bw_fnv1a_extend(digest, target_velocity_rpm & 0xFFFF_FFFF, 4)
    digest = _bw_fnv1a_extend(digest, hard_torque_limit, 2)
    digest = _bw_fnv1a_extend(digest, usable_torque_limit, 2)
    digest = _bw_fnv1a_extend(digest, pwm_hz, 4)
    digest = _bw_fnv1a_extend(digest, encoder_counts_per_rev, 4)
    return digest


@dataclass
class _UpwardBreakawayProbeReplay:
    """Host-side port of the shared `UpwardBreakawayCursor`."""

    rung_count: int
    rung_index: int = 0
    stage: str = "initial_forward"
    observations: int = 0
    initial_forward_moved: bool = False
    repeat_forward_moved: bool = False

    @classmethod
    def start(cls, rung_count: int) -> _UpwardBreakawayProbeReplay:
        if rung_count <= 0:
            raise ValueError("probe cursor requires a non-empty search grid")
        return cls(rung_count)

    def expected_stroke(self) -> tuple[int, int, int]:
        """Return `(rung_index, direction, direction-local slot_index)`."""
        if self.stage == "terminal":
            raise ValueError("probe evidence continues after the cursor became terminal")
        return {
            "initial_forward": (self.rung_index, 0, 0),
            "initial_reverse": (self.rung_index, 1, 0),
            "repeat_forward": (self.rung_index, 0, 1),
            "repeat_reverse": (self.rung_index, 1, 1),
        }[self.stage]

    def advance(self, moved: bool) -> tuple[str, int | None]:
        self.observations += 1
        if self.stage == "initial_forward":
            self.initial_forward_moved = moved
            self.stage = "initial_reverse"
            return "probe", None
        if self.stage == "initial_reverse":
            if self.initial_forward_moved and moved:
                self.stage = "repeat_forward"
                return "probe", None
            return self._advance_rung()
        if self.stage == "repeat_forward":
            self.repeat_forward_moved = moved
            self.stage = "repeat_reverse"
            return "probe", None
        if self.stage == "repeat_reverse":
            if self.repeat_forward_moved and moved:
                self.stage = "terminal"
                return "resolved", self.rung_index
            return self._advance_rung()
        raise ValueError("probe evidence continues after the cursor became terminal")

    def _advance_rung(self) -> tuple[str, int | None]:
        self.rung_index += 1
        if self.rung_index >= self.rung_count:
            self.stage = "terminal"
            return "terminal", 1
        self.stage = "initial_forward"
        self.initial_forward_moved = False
        self.repeat_forward_moved = False
        return "probe", None


def _bw_classify_probe_excursion(
    observation: dict[str, object], motion_threshold_counts: int, max_capture_interval_us: int
) -> tuple[bool | None, int | None]:
    """Return `(moved, terminal_cause)` from physical probe evidence alone."""
    interval = int(observation["capture_max_interval_us"])
    if interval == 0:
        if (
            int(observation["stroke_min_displacement_counts"]) != 0
            or int(observation["stroke_max_displacement_counts"]) != 0
        ):
            raise ValueError("probe record with no measured interval has nonzero extrema")
        return None, 22
    if interval > max_capture_interval_us:
        return None, 22
    predicted = int(observation["predicted_torque_target_abs"])
    if predicted >= int(observation["hard_torque_limit"]):
        return None, 6
    commanded, opposite = _bw_probe_excursion_counts(observation)
    if commanded >= motion_threshold_counts:
        moved = True
    elif opposite >= motion_threshold_counts:
        return None, 6
    else:
        moved = False
    if moved and predicted >= int(observation["usable_torque_limit"]):
        return None, 5
    return moved, None


def _bw_probe_excursion_counts(observation: dict[str, object]) -> tuple[int, int]:
    """Derive commanded and opposite excursion from one structural record."""
    minimum = int(observation["stroke_min_displacement_counts"])
    maximum = int(observation["stroke_max_displacement_counts"])
    if minimum > 0 or maximum < 0:
        raise ValueError("probe excursion extrema do not contain the zero baseline")
    forward = int(observation["direction"]) == 0
    commanded = maximum if forward else -minimum
    opposite = -minimum if forward else maximum
    return commanded, opposite


def breakaway_replay_upward_probe(
    search_grid: list[int], ticks: list[dict[str, object]]
) -> dict[str, object]:
    """Reconstruct the v15 upward walk without trusting firmware decisions."""
    if not ticks:
        raise ValueError("upward probe replay requires campaign ticks")
    plan = _bw_single_record(ticks[0], "breakaway_probe_planned")
    threshold = int(plan["motion_threshold_counts"])
    max_interval = int(plan["max_capture_interval_us"])
    maximum_observations = int(plan["max_observations"])
    if threshold <= 0 or max_interval <= 0 or maximum_observations <= 0:
        raise ValueError("upward probe plan contains a zero decision constant or budget")
    streamed: list[dict[str, object]] = []
    for tick in ticks:
        if tick["event_id"] == 160:
            raw = _bw_tick_records_by_kind(tick, "breakaway_raw_observation")
            if raw and int(raw[0]["phase"]) == _BW_RAW_OBSERVATION_PHASE_PROBE:
                streamed.append(raw[0])
        elif tick["event_id"] == 162:
            streamed.extend(_bw_tick_records_by_kind(tick, "breakaway_probe_decision"))
    raw_count = sum(record.get("record_kind") == "breakaway_raw_observation" for record in streamed)
    if raw_count > maximum_observations:
        raise ValueError("probe raw evidence exceeds its planned observation budget")

    cursor = _UpwardBreakawayProbeReplay.start(len(search_grid))
    resolved_rung_index: int | None = None
    committed = 0
    uncommitted_raw: list[dict[str, object]] = []
    observation_audit: list[dict[str, object]] = []
    expected_terminal: dict[str, int | str] | None = None
    index = 0
    while index < len(streamed):
        raw = streamed[index]
        if raw.get("record_kind") != "breakaway_raw_observation":
            raise ValueError("probe decision has no preceding raw observation")
        expected_rung, expected_direction, expected_slot = cursor.expected_stroke()
        for field_name, expected in (
            ("rung_index", expected_rung),
            ("direction", expected_direction),
            ("slot_index", expected_slot),
        ):
            if int(raw[field_name]) != expected:
                raise ValueError(
                    f"probe raw observation {field_name} does not match the reconstructed cursor"
                )
        commanded_excursion, opposite_excursion = _bw_probe_excursion_counts(raw)
        moved, terminal_cause = _bw_classify_probe_excursion(raw, threshold, max_interval)
        observation_report = {
            "evidence_sequence": int(raw["evidence_sequence"]),
            "rung_index": int(raw["rung_index"]),
            "p_raw": search_grid[int(raw["rung_index"])],
            "direction": int(raw["direction"]),
            "slot_index": int(raw["slot_index"]),
            "stroke_min_displacement_counts": int(raw["stroke_min_displacement_counts"]),
            "stroke_max_displacement_counts": int(raw["stroke_max_displacement_counts"]),
            "commanded_excursion_counts": commanded_excursion,
            "opposite_excursion_counts": opposite_excursion,
            "capture_max_interval_us": int(raw["capture_max_interval_us"]),
            "moved": moved,
            "terminal_cause": terminal_cause,
            "committed": False,
            "stationarity_diagnostic": {
                "stationarity_code": int(raw["stationarity_code"]),
                "settled_rate_low_q": int(raw["settled_rate_low_q"]),
                "settled_rate_high_q": int(raw["settled_rate_high_q"]),
                "drift_ratio_ppm": int(raw["drift_ratio_ppm"]),
                "post_fit_lag_one_state": int(raw["post_fit_lag_one_state"]),
            },
        }
        observation_audit.append(observation_report)
        decision = streamed[index + 1] if index + 1 < len(streamed) else None
        if decision is None or decision.get("record_kind") != "breakaway_probe_decision":
            if index + 1 != len(streamed):
                raise ValueError("uncommitted probe raw observation is not the final probe record")
            uncommitted_raw.append(raw)
            expected_terminal = (
                {"terminal_cause": terminal_cause} if terminal_cause is not None else None
            )
            break

        if int(decision["evidence_sequence"]) != int(raw["evidence_sequence"]) + 1:
            raise ValueError("probe decision evidence_sequence does not follow its raw observation")
        for field_name in ("run_sequence", "plan_digest", "direction", "rung_index"):
            if int(decision[field_name]) != int(raw[field_name]):
                raise ValueError(f"probe decision {field_name} disagrees with its raw observation")
        if terminal_cause is not None:
            raise ValueError("probe decision commits raw evidence that independently terminates")
        if int(decision["disposition"]) != int(bool(moved)):
            raise ValueError("probe decision disposition disagrees with independent classification")
        observation_report["committed"] = True
        step, value = cursor.advance(bool(moved))
        committed += 1
        if committed > maximum_observations:
            raise ValueError("probe evidence exceeds its planned observation budget")
        if step == "resolved":
            resolved_rung_index = int(value)  # type: ignore[arg-type]
        elif step == "terminal":
            expected_terminal = {"terminal_cause": int(value)}
            if index + 2 != len(streamed):
                raise ValueError(
                    "probe evidence continues after an independently terminal decision"
                )
        index += 2

    return {
        "resolved_rung_index": resolved_rung_index,
        "breakaway_p_raw": (
            search_grid[resolved_rung_index] if resolved_rung_index is not None else None
        ),
        "committed_observations": committed,
        "uncommitted_raw": uncommitted_raw,
        "observations": observation_audit,
        "expected_terminal": expected_terminal,
    }


def breakaway_realize_candidate_additive_ladder(
    rung_count_target: int, floor_raw: int, breakaway_raw: int, ceiling_raw: int
) -> tuple[int, list[int], int]:
    """Port of `realize_candidate_additive_ladder` (breakaway_plan.rs).

    Returns `(step, gains, first_additive_gain)`.
    """
    remaining_intervals = rung_count_target - 2
    if remaining_intervals <= 0:
        raise ValueError("candidate rung count leaves no additive intervals")
    span = ceiling_raw - breakaway_raw
    step = -(-span // remaining_intervals)  # ceil division, matches `div_ceil`
    first_additive_gain = min(breakaway_raw + step, ceiling_raw)

    gains: list[int] = []
    if floor_raw != breakaway_raw:
        gains.append(floor_raw)
    gains.append(breakaway_raw)

    current = breakaway_raw
    while True:
        next_gain = current + step
        if next_gain >= ceiling_raw:
            if gains[-1] != ceiling_raw:
                gains.append(ceiling_raw)
            break
        gains.append(next_gain)
        current = next_gain
    return step, gains, first_additive_gain


def breakaway_plan_additive_discovery_ladder(
    floor_raw: int,
    breakaway_raw: int,
    ceiling_raw: int,
    observations_per_direction: int,
    workflow_envelope_ms: int,
) -> dict[str, object] | None:
    """Port of `plan_additive_discovery_ladder`'s candidate enumeration and
    selection (breakaway_plan.rs). Returns the chosen candidate's fields, or
    `None` if no candidate `n` in `MIN_STRUCTURAL_RUNGS..=MAX_SWEEP_RUNGS`
    survives every guard.
    """
    primitive_us = _BW_MOVING_STROKE_US + _BW_ZERO_SETTLE_US + _BW_SETTLE_ANALYSIS_BUDGET_US
    slot_count = observations_per_direction * _BW_SWEEP_DIRECTION_COUNT
    best: dict[str, object] | None = None
    for rung_count_target in range(_BW_MIN_STRUCTURAL_RUNGS, _BW_MAX_SWEEP_RUNGS + 1):
        step, gains, first_additive_gain = breakaway_realize_candidate_additive_ladder(
            rung_count_target, floor_raw, breakaway_raw, ceiling_raw
        )
        rung_count = len(gains)
        if rung_count < _BW_MIN_STRUCTURAL_RUNGS:
            continue
        confidence_family_size = rung_count * observations_per_direction
        if confidence_family_size > _BW_MAX_CONFIDENCE_FAMILY_SIZE:
            continue
        workflow_us = rung_count * slot_count * primitive_us
        workflow_ms = -(-workflow_us // 1000)
        if workflow_ms > workflow_envelope_ms:
            continue
        if (
            first_additive_gain * _BW_MAX_ADDITIVE_GAIN_RATIO_DEN
            > breakaway_raw * _BW_MAX_ADDITIVE_GAIN_RATIO_NUM
        ):
            continue
        is_better = best is None or (
            first_additive_gain < best["first_additive_gain"]
            or (
                first_additive_gain == best["first_additive_gain"]
                and rung_count_target > best["rung_count_target"]
            )
        )
        if is_better:
            best = {
                "rung_count_target": rung_count_target,
                "step": step,
                "gains": gains,
                "rung_count": rung_count,
                "confidence_family_size": confidence_family_size,
                "first_additive_gain": first_additive_gain,
            }
    return best


def breakaway_discovery_plan_digest(
    gains: list[int],
    observations_per_direction: int,
    confidence_family_size: int,
    probe_plan_digest: int,
) -> int:
    """Port of `AdditiveDiscoveryPlan::digest` (breakaway_plan.rs)."""
    digest = FNV1A64_OFFSET
    digest = _bw_fnv1a_extend(digest, len(gains), 1)
    for gain in gains:
        digest = _bw_fnv1a_extend(digest, gain, 2)
    digest = _bw_fnv1a_extend(digest, observations_per_direction, 2)
    digest = _bw_fnv1a_extend(digest, confidence_family_size, 2)
    digest = _bw_fnv1a_extend(digest, probe_plan_digest, 8)
    return digest


def breakaway_confirmation_plan_digest(
    discovery_plan_digest: int,
    nominated_p_raw: int,
    capture_profile: int,
    family_size: int,
    observations_per_direction: int,
    band_lower_percent: int,
    band_upper_percent: int,
    acceptance_rule: int,
) -> int:
    """Port of `BandConfirmationPlan::digest` (band_confirmation.rs).

    Folds three fixed capture-timing constants
    (`MOVING_STROKE_US`/`ZERO_SETTLE_US`/`SETTLE_ANALYSIS_BUDGET_US`) and the
    fixed eight-slot direction order that never ride the wire, because
    `BandConfirmationPlanned` only discloses the fields that vary run to run.
    """
    digest = FNV1A64_OFFSET
    digest = _bw_fnv1a_extend(digest, discovery_plan_digest, 8)
    digest = _bw_fnv1a_extend(digest, nominated_p_raw, 2)
    digest = _bw_fnv1a_extend(digest, capture_profile, 1)
    digest = _bw_fnv1a_extend(digest, family_size, 2)
    digest = _bw_fnv1a_extend(digest, observations_per_direction, 2)
    digest = _bw_fnv1a_extend(digest, _BW_MOVING_STROKE_US, 4)
    digest = _bw_fnv1a_extend(digest, _BW_ZERO_SETTLE_US, 4)
    digest = _bw_fnv1a_extend(digest, _BW_SETTLE_ANALYSIS_BUDGET_US, 4)
    digest = _bw_fnv1a_extend(digest, band_lower_percent, 1)
    digest = _bw_fnv1a_extend(digest, band_upper_percent, 1)
    digest = _bw_fnv1a_extend(digest, acceptance_rule, 1)
    for code in _BW_CONFIRMATION_SLOT_ORDER_CODES:
        digest = _bw_fnv1a_extend(digest, code, 1)
    return digest


def breakaway_discovery_rung_margin(
    lower_percent_milli: int, upper_percent_milli: int
) -> tuple[int, int]:
    """Port of `discovery_rung_margin_report`'s `M_i` and half-width math
    (breakaway_reporting.rs), from the record's own reported response bounds.

    Returns `(margin_percent_milli, half_width_percent_milli)`, matching spec
    section 7: `M_i = min(L_i - 70, 80 - U_i)`.
    """
    lower_band_milli = _BW_CONFIRMATION_BAND_LOWER_PERCENT * 1_000
    upper_band_milli = _BW_CONFIRMATION_BAND_UPPER_PERCENT * 1_000
    margin_percent_milli = min(
        lower_percent_milli - lower_band_milli, upper_band_milli - upper_percent_milli
    )
    half_width_percent_milli = (upper_percent_milli - lower_percent_milli) // 2
    return margin_percent_milli, half_width_percent_milli


def breakaway_required_relative_se_permille() -> int:
    """Port of `confirmation_evaluated_terminal_report`'s `required_relative_se`
    (breakaway_reporting.rs): `REQUIRED_HALF_WIDTH_FRACTION / m(8)`, rounded to
    permille exactly as `relative_se_to_permille` does.

    Latent precision note: firmware rounds with `libm::roundf` (round-half-
    away-from-zero, `f32`); this uses Python's `round()` (banker's rounding,
    `f64`). For this specific value (~0.020991, nowhere near a `x.5`
    boundary) the two cannot disagree, but a future constant landing exactly
    on a half-permille boundary could expose the difference.
    """
    value = _BW_REQUIRED_HALF_WIDTH_FRACTION / _BW_CONFIRMATION_MULTIPLIER_95_DF15
    scaled = round(value * 1_000)
    return max(0, min(scaled, 0xFFFF))


def _bw_tick_records_by_kind(tick: dict[str, object], record_kind: str) -> list[dict[str, object]]:
    return [
        decoded
        for _raw, decoded in tick["records"]  # type: ignore[union-attr]
        if decoded["record_kind"] == record_kind
    ]


def _bw_single_record(tick: dict[str, object], record_kind: str) -> dict[str, object]:
    matches = _bw_tick_records_by_kind(tick, record_kind)
    if len(matches) != 1:
        raise ValueError(
            f"breakaway {BREAKAWAY_EVENT_ID_NAMES.get(tick['event_id'], tick['event_id'])} "
            f"tick expected exactly one {record_kind} record, found {len(matches)}"
        )
    return matches[0]


def audit_breakaway_campaign(
    ticks: list[dict[str, object]],
    motion_timestamps: list[int] | None = None,
    clock_hz: int | None = None,
) -> dict[str, object]:
    """Independently reconstruct one breakaway campaign and verify every
    firmware decision against `assemble_breakaway_ticks`'s decoded evidence.

    Raises `ValueError` for a malformed or contradictory capture: reordered
    phases, motion recorded before the probe plan, a digest-chain break, a
    family-4 (one-rung) confirmation, an accepted confirmation with an
    incomplete observation mask, a second confirmation candidate after a
    failed one, or an accepted campaign terminal that doesn't trace back to an
    accepted confirmation. These are refusals, not soft mismatches: an offline
    replay cannot certify a capture whose own internal evidence is
    inconsistent, regardless of what any individual field claims.

    `clock_hz` (the capturing device's `CLOCK_FREQ`, when known) lets the
    workflow-duration check convert `workflow_duration_ticks` -- the elapsed
    time between the probe-plan and campaign-terminal markers, independently
    measured from the FTRC frame timestamps themselves -- into milliseconds
    and compare it against `AdditiveDiscoveryPlanned.maximum_workflow_ms`
    (spec section 8's reserved aggregate envelope). Without it the check is
    skipped, since the ticks alone carry no clock reference.

    Returns a dict with `matches_firmware`/`mismatches` for genuine
    *value* discrepancies between the independently recomputed quantities and
    the firmware's own reported ones -- these indicate a possible firmware
    bug, not a corrupt capture, and are reported rather than raised.
    """
    if not ticks:
        raise ValueError("empty breakaway campaign: no marker ticks")
    if ticks[0]["event_id"] != 152:
        raise ValueError("breakaway campaign does not begin with the probe plan marker")
    if ticks[-1]["event_id"] != 159:
        raise ValueError("breakaway campaign does not end with the campaign terminal marker")
    campaign_terminal_count = sum(1 for tick in ticks if tick["event_id"] == 159)
    if campaign_terminal_count != 1:
        raise ValueError(
            f"breakaway campaign has {campaign_terminal_count} campaign terminal markers, want 1"
        )
    previous_rank = -1
    for tick in ticks[:-1]:
        rank = _bw_tick_phase_rank(tick)
        if rank < previous_rank:
            raise ValueError(
                "breakaway campaign phases are reordered: "
                f"{BREAKAWAY_EVENT_ID_NAMES[tick['event_id']]} follows a later phase"
            )
        previous_rank = rank
    if motion_timestamps:
        plan_ticks_time = int(ticks[0]["timestamp_ticks"])  # type: ignore[arg-type]
        if any(timestamp < plan_ticks_time for timestamp in motion_timestamps):
            raise ValueError("motion observed before the breakaway probe plan was frozen")

    _bw_validate_streamed_marker_identity(ticks)

    mismatches: list[str] = []

    probe_planned = _bw_single_record(ticks[0], "breakaway_probe_planned")
    search_grid = breakaway_build_search_grid(
        int(probe_planned["p_start_raw"]), int(probe_planned["p_top_raw"])
    )
    if len(search_grid) != int(probe_planned["search_count"]):
        mismatches.append(
            f"probe search_count: firmware={int(probe_planned['search_count'])} offline="
            f"{len(search_grid)}"
        )
    probe_digest_verified = probe_planned["workflow_digest"] is not None
    expected_probe_digest = (
        breakaway_probe_plan_digest(
            search_grid[: int(probe_planned["search_count"])],
            int(probe_planned["motion_threshold_counts"]),
            int(probe_planned["max_capture_interval_us"]),
            int(probe_planned["max_observations"]),
            workflow_digest=int(probe_planned["workflow_digest"]),
            target_velocity_rpm=int(probe_planned["target_velocity_rpm"]),
            hard_torque_limit=int(probe_planned["hard_torque_limit"]),
            usable_torque_limit=int(probe_planned["usable_torque_limit"]),
            pwm_hz=int(probe_planned["pwm_hz"]),
            encoder_counts_per_rev=int(probe_planned["encoder_counts_per_rev"]),
        )
        if probe_digest_verified
        else None
    )
    if expected_probe_digest is not None and expected_probe_digest != int(
        probe_planned["plan_digest"]
    ):
        mismatches.append(
            f"probe plan_digest: firmware=0x{probe_planned['plan_digest']:016x} offline=0x"
            f"{expected_probe_digest:016x}"
        )

    probe_result_ticks = [tick for tick in ticks if tick["event_id"] == 153]
    if len(probe_result_ticks) > 1:
        raise ValueError("breakaway campaign reports more than one probe result")
    probe_result = (
        _bw_single_record(probe_result_ticks[0], "breakaway_probe_resolved")
        if probe_result_ticks
        else None
    )
    if probe_result is not None and int(probe_result["plan_digest"]) != int(
        probe_planned["plan_digest"]
    ):
        raise ValueError("breakaway probe result does not chain to its own plan digest")

    probe_terminal_tick = next((tick for tick in ticks if tick["event_id"] == 154), None)
    probe_terminal: dict[str, object] | None = None
    if probe_terminal_tick is not None:
        probe_terminal = _bw_single_record(probe_terminal_tick, "breakaway_probe_terminal")
        if int(probe_terminal["plan_digest"]) != int(probe_planned["plan_digest"]):
            raise ValueError("breakaway probe terminal does not chain to its own plan digest")
        if int(probe_terminal["fault_checkpoint"]) != 0 and int(
            probe_terminal["terminal_cause"]
        ) not in (6, 22):
            mismatches.append(
                f"probe terminal reports a fault checkpoint for non-safety cause "
                f"{int(probe_terminal['terminal_cause'])}"
            )

    if probe_result is not None and probe_terminal is not None:
        raise ValueError("breakaway probe reports both a result and a terminal")
    probe_end_tick = probe_result_ticks[0] if probe_result_ticks else probe_terminal_tick
    if probe_end_tick is not None:
        probe_end_index = ticks.index(probe_end_tick)
        for later in ticks[probe_end_index + 1 :]:
            later_is_probe_raw = later["event_id"] == 160 and any(
                int(record["phase"]) == _BW_RAW_OBSERVATION_PHASE_PROBE
                for record in _bw_tick_records_by_kind(later, "breakaway_raw_observation")
            )
            if later["event_id"] == 162 or later_is_probe_raw:
                raise ValueError("probe evidence continues after its result or terminal")

    probe_stream_present = any(
        tick["event_id"] == 162
        or (
            tick["event_id"] == 160
            and any(
                int(record["phase"]) == _BW_RAW_OBSERVATION_PHASE_PROBE
                for record in _bw_tick_records_by_kind(tick, "breakaway_raw_observation")
            )
        )
        for tick in ticks
    )
    probe_replay = (
        breakaway_replay_upward_probe(search_grid, ticks) if probe_stream_present else None
    )
    if probe_result is not None and probe_replay is None:
        raise ValueError("breakaway probe result has no raw evidence to replay")
    if probe_replay is not None:
        replayed_rung = probe_replay["resolved_rung_index"]
        if replayed_rung is None:
            if probe_result is not None:
                mismatches.append(
                    "firmware reports a probe result the upward replay did not resolve"
                )
        elif probe_result is None:
            mismatches.append("upward replay resolved without a firmware probe result")
        else:
            for field, expected in (
                ("rung_index", replayed_rung),
                ("breakaway_p_raw", probe_replay["breakaway_p_raw"]),
                ("motion_threshold_counts", probe_planned["motion_threshold_counts"]),
                ("observation_count", probe_replay["committed_observations"]),
            ):
                if int(probe_result[field]) != int(expected):
                    mismatches.append(
                        f"probe result {field}: firmware={probe_result[field]} offline={expected}"
                    )
        expected_terminal = probe_replay["expected_terminal"]
        if expected_terminal is not None:
            if probe_terminal is None:
                mismatches.append("probe replay terminated but firmware reported no probe terminal")
            else:
                if int(probe_terminal["terminal_cause"]) != int(
                    expected_terminal["terminal_cause"]
                ):
                    mismatches.append(
                        f"probe terminal_cause: firmware="
                        f"{int(probe_terminal['terminal_cause'])} offline="
                        f"{int(expected_terminal['terminal_cause'])}"
                    )
        elif probe_replay["uncommitted_raw"] and (
            probe_terminal is None or int(probe_terminal["terminal_cause"]) != 6
        ):
            mismatches.append(
                "probe raw observation has no decision without a recovery safety terminal"
            )

    resolved_breakaway_raw = (
        int(probe_replay["breakaway_p_raw"])
        if probe_replay is not None and probe_replay["breakaway_p_raw"] is not None
        else None
    )

    discovery_planned_tick = next((tick for tick in ticks if tick["event_id"] == 155), None)
    discovery_terminal_tick = next((tick for tick in ticks if tick["event_id"] == 156), None)
    discovery_planned: dict[str, object] | None = None
    ladder_gains: list[int] = []
    post_breakaway_transient: list[dict[str, int]] = []
    discovery_inert_fraction: dict[str, object] | None = None
    if discovery_planned_tick is not None:
        discovery_planned = _bw_single_record(discovery_planned_tick, "breakaway_discovery_planned")
        if int(discovery_planned["prior_plan_digest"]) != int(probe_planned["plan_digest"]):
            raise ValueError("breakaway discovery plan does not chain to the probe plan digest")
        if not (
            _BW_PROPORTIONAL_MIN_SCHEMA_REVISION
            <= int(discovery_planned["schema_revision"])
            <= _BW_PROPORTIONAL_SCHEMA_REVISION
        ):
            mismatches.append(
                f"discovery schema_revision: firmware="
                f"{int(discovery_planned['schema_revision'])} want="
                f"{int(_BW_PROPORTIONAL_MIN_SCHEMA_REVISION)}-"
                f"{int(_BW_PROPORTIONAL_SCHEMA_REVISION)} (offline replay regenerates the ladder "
                f"under a different rules revision than the firmware applied)"
            )
        floor_raw = int(discovery_planned["floor_p_raw"])
        breakaway_raw = int(discovery_planned["breakaway_p_raw"])
        ceiling_raw = int(discovery_planned["ceiling_p_raw"])
        if resolved_breakaway_raw is not None and breakaway_raw != resolved_breakaway_raw:
            mismatches.append(
                f"discovery breakaway_p_raw: reported={int(breakaway_raw)} offline upward "
                f"replay={int(resolved_breakaway_raw)}"
            )
        # The per-rung ladder no longer rides the wire: the
        # only per-campaign evidence for the ladder's shape is this plan's own
        # summary (floor/breakaway/ceiling/observations_per_direction/
        # maximum_workflow_ms) plus `plan_digest` itself. Regenerate the exact
        # `gains[]` independently from that summary -- never from a
        # firmware-reported per-rung list -- and fold the REGENERATED gains
        # (not anything firmware reported) into the digest this function
        # compares against `plan_digest`. A tampered `plan_digest` with
        # self-consistent summary fields is caught here because the digest
        # folds the regenerated ladder; a tampered summary field is caught
        # because it changes the regenerated ladder the digest is taken over.
        best = breakaway_plan_additive_discovery_ladder(
            floor_raw,
            breakaway_raw,
            ceiling_raw,
            int(discovery_planned["observations_per_direction"]),
            int(discovery_planned["maximum_workflow_ms"]),
        )
        if best is None:
            mismatches.append("no candidate additive ladder survives every planning guard")
        else:
            ladder_gains = best["gains"]
            if len(ladder_gains) != int(discovery_planned["rung_count"]):
                mismatches.append(
                    f"discovery ladder rung count: firmware="
                    f"{int(discovery_planned['rung_count'])} offline={len(ladder_gains)}"
                )
            if int(discovery_planned["family_size"]) != best["confidence_family_size"]:
                mismatches.append(
                    f"discovery family_size: firmware={int(discovery_planned['family_size'])} "
                    f"offline={int(best['confidence_family_size'])}"
                )
            expected_first_step = (
                best["gains"][best["gains"].index(breakaway_raw) + 1] - breakaway_raw
            )
            if int(discovery_planned["first_additive_step_raw"]) != expected_first_step:
                mismatches.append(
                    f"discovery first_additive_step_raw: firmware="
                    f"{int(discovery_planned['first_additive_step_raw'])} offline="
                    f"{int(expected_first_step)}"
                )
            expected_discovery_digest = breakaway_discovery_plan_digest(
                ladder_gains,
                int(discovery_planned["observations_per_direction"]),
                int(discovery_planned["family_size"]),
                int(probe_planned["plan_digest"]),
            )
            if expected_discovery_digest != int(discovery_planned["plan_digest"]):
                mismatches.append(
                    f"discovery plan_digest: firmware=0x{discovery_planned['plan_digest']:016x}"
                    f" offline=0x{expected_discovery_digest:016x}"
                )

    discovery_terminal: dict[str, object] | None = None
    if discovery_terminal_tick is not None:
        if discovery_planned is None:
            raise ValueError("breakaway discovery terminal has no matching discovery plan")
        discovery_terminal = _bw_single_record(
            discovery_terminal_tick, "breakaway_discovery_terminal"
        )
        if int(discovery_terminal["plan_digest"]) != int(discovery_planned["plan_digest"]):
            raise ValueError("breakaway discovery terminal does not chain to its own plan digest")
        if int(discovery_terminal["prior_plan_digest"]) != int(probe_planned["plan_digest"]):
            raise ValueError("breakaway discovery terminal does not chain to the probe plan digest")
        # Margins stream one per tick on their own marker (161), not bundled
        # into this terminal tick; gather them across the campaign and order by
        # collected index.
        margin_records = sorted(
            _bw_collect_margins(ticks),
            key=lambda record: int(record["collected_index"]),
        )
        # `rung_nomination_bounds` (band_confirmation.rs:306-325) can return
        # `nominatable: false` for a genuinely in-band rung (either
        # direction's pooled interval isn't judgeable); `discovery_rung_
        # margin_report` (breakaway_reporting.rs:416-417) then emits no wire
        # record for that `collected_index` at all, while `collected_count`
        # (breakaway_reporting.rs:521) still counts it. So a gap in
        # `collected_index` is normal, wire-invisible firmware behavior, not
        # a capture defect: the wire evidence alone cannot distinguish "N
        # collected, all nominatable" from "N+k collected, k not
        # nominatable" -- only `<=` is checkable, not exact equality.
        if len(margin_records) > int(discovery_terminal["collected_count"]):
            mismatches.append(
                f"discovery collected_count: firmware="
                f"{int(discovery_terminal['collected_count'])} offline has more margin records "
                f"({len(margin_records)})"
            )

        # Independent nomination recompute: every
        # `BreakawayDiscoveryRungMargin`'s `lower_percent_milli`/
        # `upper_percent_milli` above is firmware's OWN reduction of the raw
        # evidence -- the `breakaway_discovery_rung_margin` check just above
        # only verifies `M_i`/half-width are correctly *derived from* those
        # reported bounds, never that the bounds themselves are honest. Group
        # the streamed discovery `BreakawayRawObservation` records (one per
        # collected stroke on marker 160, phase == Discovery) by ladder
        # rung_index, fold each rung's octet through the SAME nomination formula
        # the firmware uses (multiplier m(8), `NOMINATION_SCREEN_C_SIGMA`,
        # outward rounding -- see `_bw_rung_response_bounds_percent`'s own doc
        # comment for the one modeling gap, depth-three `included_mask`
        # exclusion, this cannot replicate), and compare the recomputed bounds
        # against the firmware-reported ones. A rung whose p_raw doesn't land on
        # the independently regenerated ladder, or whose raw evidence disagrees
        # with the reported bounds, is a mismatch.
        raw_discovery_observations = _bw_collect_raw_observations(
            ticks, _BW_RAW_OBSERVATION_PHASE_DISCOVERY
        )
        post_breakaway_transient = _bw_post_breakaway_transients(
            raw_discovery_observations, ladder_gains
        )
        discovery_inert_fraction = _bw_discovery_inert_fraction(
            raw_discovery_observations, ladder_gains, resolved_breakaway_raw
        )
        for observation in raw_discovery_observations:
            if int(observation["plan_digest"]) != int(discovery_planned["plan_digest"]):
                raise ValueError(
                    "breakaway discovery raw observation does not chain to its own plan digest"
                )
        observations_by_rung: dict[int, dict[int, list[dict[str, object]]]] = {}
        for observation in raw_discovery_observations:
            by_direction = observations_by_rung.setdefault(int(observation["rung_index"]), {})
            by_direction.setdefault(int(observation["direction"]), []).append(observation)
        nomination_multiplier = _f32(_confidence_multiplier(_BW_CONFIRMATION_FAMILY_SIZE))

        previous_mid: int | None = None
        previous_collected_index: int | None = None
        for record in margin_records:
            expected_margin, expected_half_width = breakaway_discovery_rung_margin(
                int(record["lower_percent_milli"]), int(record["upper_percent_milli"])
            )
            if expected_margin != int(record["margin_percent_milli"]):
                mismatches.append(
                    f"discovery rung margin M_i (collected_index="
                    f"{int(record['collected_index'])}): firmware="
                    f"{int(record['margin_percent_milli'])} offline={int(expected_margin)}"
                )
            if expected_half_width != int(record["half_width_percent_milli"]):
                mismatches.append(
                    f"discovery rung half-width (collected_index="
                    f"{int(record['collected_index'])}): firmware="
                    f"{int(record['half_width_percent_milli'])} offline="
                    f"{int(expected_half_width)}"
                )
            p_raw = int(record["p_raw"])
            if p_raw not in ladder_gains:
                mismatches.append(
                    f"discovery rung margin (collected_index={int(record['collected_index'])}) "
                    f"p_raw={int(p_raw)} is not on the independently regenerated ladder"
                )
            else:
                rung_index = ladder_gains.index(p_raw)
                by_direction = observations_by_rung.get(rung_index, {})
                forward = by_direction.get(0, [])
                reverse = by_direction.get(1, [])
                if (
                    len(forward) != _BW_CONSENSUS_OBSERVATIONS_PER_DIRECTION
                    or len(reverse) != _BW_CONSENSUS_OBSERVATIONS_PER_DIRECTION
                ):
                    mismatches.append(
                        f"discovery rung margin (collected_index="
                        f"{int(record['collected_index'])}, p_raw={int(p_raw)}) has no "
                        f"complete raw-observation octet (forward={len(forward)} reverse="
                        f"{len(reverse)}) to independently verify"
                    )
                else:
                    recomputed = _bw_rung_response_bounds_from_consensus(
                        forward,
                        reverse,
                        nomination_multiplier,
                        _BW_NOMINATION_SCREEN_C_SIGMA,
                        True,
                    )
                    if recomputed is None:
                        mismatches.append(
                            f"discovery rung margin (collected_index="
                            f"{int(record['collected_index'])}, p_raw={int(p_raw)}) raw "
                            f"evidence is not independently nominatable"
                        )
                    else:
                        offline_lower_milli = _bw_percent_to_milli(recomputed[0])
                        offline_upper_milli = _bw_percent_to_milli(recomputed[1])
                        if offline_lower_milli != int(record["lower_percent_milli"]):
                            mismatches.append(
                                f"discovery rung lower_percent_milli (collected_index="
                                f"{int(record['collected_index'])}, p_raw={int(p_raw)}): "
                                f"firmware={int(record['lower_percent_milli'])} offline="
                                f"{int(offline_lower_milli)}"
                            )
                        if offline_upper_milli != int(record["upper_percent_milli"]):
                            mismatches.append(
                                f"discovery rung upper_percent_milli (collected_index="
                                f"{int(record['collected_index'])}, p_raw={int(p_raw)}): "
                                f"firmware={int(record['upper_percent_milli'])} offline="
                                f"{int(offline_upper_milli)}"
                            )
            mid = (int(record["lower_percent_milli"]) + int(record["upper_percent_milli"])) // 2
            collected_index = int(record["collected_index"])
            # `discovery_rung_margin_report` (breakaway_reporting.rs:426-439)
            # only computes a nonzero step against the IMMEDIATE predecessor
            # position (`collected_index - 1`); when that specific
            # predecessor wasn't nominatable (a wire-invisible gap, see
            # above) it defaults to 0, even though a nominatable rung exists
            # further back. Comparing against the previous *emitted* record
            # unconditionally would compute a step across the gap instead.
            expected_step = (
                mid - previous_mid
                if previous_collected_index is not None
                and collected_index == previous_collected_index + 1
                else 0
            )
            firmware_step = int(record["adjacent_rate_step_percent_milli"])
            if expected_step != firmware_step:
                mismatches.append(
                    f"discovery adjacent_rate_step_percent_milli (collected_index="
                    f"{int(record['collected_index'])}): firmware="
                    f"{firmware_step} offline={int(expected_step)}"
                )
            previous_mid = mid
            previous_collected_index = collected_index

    confirmation_plan_ticks = [tick for tick in ticks if tick["event_id"] == 157]
    confirmation_terminal_ticks = [tick for tick in ticks if tick["event_id"] == 158]
    if len(confirmation_plan_ticks) > 1:
        raise ValueError("breakaway campaign spent a second confirmation candidate after the first")
    if len(confirmation_terminal_ticks) > 1:
        raise ValueError(
            "breakaway campaign ran a second confirmation block after the first terminated"
        )

    confirmation_planned: dict[str, object] | None = None
    confirmation_accepted = False
    if confirmation_plan_ticks:
        if discovery_terminal is None:
            raise ValueError("breakaway confirmation plan has no matching discovery terminal")
        confirmation_planned = _bw_single_record(
            confirmation_plan_ticks[0], "breakaway_confirmation_planned"
        )
        if int(confirmation_planned["family_size"]) != _BW_CONFIRMATION_FAMILY_SIZE:
            raise ValueError(
                f"breakaway confirmation family_size={int(confirmation_planned['family_size'])}"
                f", want the fixed family of 8"
            )
        if int(confirmation_planned["prior_plan_digest"]) != int(discovery_planned["plan_digest"]):  # type: ignore[index]
            raise ValueError(
                "breakaway confirmation plan does not chain to the discovery plan digest"
            )
        expected_confirmation_digest = breakaway_confirmation_plan_digest(
            int(discovery_planned["plan_digest"]),  # type: ignore[index]
            int(confirmation_planned["nominated_p_raw"]),
            int(confirmation_planned["capture_profile"]),
            int(confirmation_planned["family_size"]),
            int(confirmation_planned["observations_per_direction"]),
            int(confirmation_planned["band_lower_percent"]),
            int(confirmation_planned["band_upper_percent"]),
            int(confirmation_planned["acceptance_rule"]),
        )
        if expected_confirmation_digest != int(confirmation_planned["plan_digest"]):
            mismatches.append(
                f"confirmation plan_digest: firmware=0x"
                f"{confirmation_planned['plan_digest']:016x} offline=0x"
                f"{expected_confirmation_digest:016x}"
            )

    # The fixed eight-stroke schedule must hold before any slot-keyed collapse
    # (mask OR, per-slot dict) can hide a duplicate or a ninth stroke. Run this
    # whenever confirmation raw observations exist, not only when a confirmation
    # terminal was reached: a safety-aborted confirmation emits its partial raw
    # strokes and then only the campaign closure (breakaway_campaign.rs), so a
    # validator gated on the terminal would never see the malformed prefix.
    _bw_validate_confirmation_slot_schedule(ticks)

    if confirmation_terminal_ticks:
        confirmation_terminal = _bw_single_record(
            confirmation_terminal_ticks[0], "breakaway_confirmation_terminal"
        )
        if confirmation_planned is not None and int(confirmation_terminal["plan_digest"]) != int(
            confirmation_planned["plan_digest"]
        ):
            raise ValueError(
                "breakaway confirmation terminal does not chain to its own plan digest"
            )
        confirmation_accepted = int(confirmation_terminal["accepted"]) == 1
        observations = sorted(
            _bw_tick_records_by_kind(
                confirmation_terminal_ticks[0], "breakaway_confirmation_observation"
            ),
            key=lambda record: int(record["slot_index"]),
        )
        # Reconcile the reported collected masks with the slots actually
        # received, for every outcome (not just an accepted one): the forward
        # accumulator records schedule slots 0-3, the reverse 4-7. Reconstruct
        # from the streamed RAW observations, which are total for every completed
        # stroke -- not from the reduced observation records, which
        # confirmation_observation_report omits for an honest zero or wrong-sign
        # mean (breakaway_reporting.rs). A collected inert-zero stroke would
        # otherwise look absent and read as a false mismatch.
        raw_confirmation_collected = _bw_collect_raw_observations(
            ticks, _BW_RAW_OBSERVATION_PHASE_CONFIRMATION
        )
        reconstructed_forward_collected = 0
        reconstructed_reverse_collected = 0
        for observation in raw_confirmation_collected:
            slot = int(observation["slot_index"])
            if 0 <= slot < 4:
                reconstructed_forward_collected |= 1 << slot
            elif 4 <= slot < 8:
                reconstructed_reverse_collected |= 1 << (slot - 4)
        if reconstructed_forward_collected != int(confirmation_terminal["forward_collected_mask"]):
            mismatches.append(
                f"confirmation forward_collected_mask: firmware="
                f"{format(int(confirmation_terminal['forward_collected_mask']), '#06b')} "
                f"received slots={format(reconstructed_forward_collected, '#06b')}"
            )
        if reconstructed_reverse_collected != int(confirmation_terminal["reverse_collected_mask"]):
            mismatches.append(
                f"confirmation reverse_collected_mask: firmware="
                f"{format(int(confirmation_terminal['reverse_collected_mask']), '#06b')} "
                f"received slots={format(reconstructed_reverse_collected, '#06b')}"
            )
        if confirmation_accepted:
            # `included_mask` is deliberately absent: it is the depth-three
            # consensus over the DISTURBANCE intervals, a different measured
            # quantity than the proportional response confirmation tests, and
            # `directional_input_is_complete` (band_confirmation.rs) does not
            # gate on it. The independent containment recompute below is what
            # verifies all eight responses belong in [70, 80].
            for mask_name in (
                "forward_collected_mask",
                "forward_eligible_mask",
                "reverse_collected_mask",
                "reverse_eligible_mask",
            ):
                if int(confirmation_terminal[mask_name]) != 0b1111:
                    raise ValueError(
                        f"breakaway confirmation accepted with an incomplete {mask_name}: "
                        f"{format(int(confirmation_terminal[mask_name]), '#06b')}"
                    )
            if len(observations) != _BW_CONFIRMATION_FAMILY_SIZE:
                raise ValueError(
                    f"breakaway confirmation accepted with {len(observations)} observations, "
                    f"want the full family of 8"
                )

        # Independent confirmation recompute. The checks above only verify
        # firmware's OWN reported numbers are mutually consistent (complete
        # masks, an accepted block reporting all eight observations); none of
        # them touch whether the confirmed gain actually belongs in [70, 80].
        # Recompute each of the eight held-out strokes' response interval
        # from its raw evidence -- multiplier m(8), c_sigma = 1.0, outward
        # rounding, exactly `confirmation_observation_report`'s own
        # parameters, no nomination-style inflation -- and independently
        # verify containment, the candidate/confirmed P match, and the fixed
        # four-forward-then-four-reverse `CONFIRMATION_SLOT_ORDER` schedule.
        # This is the exact check a human reviewer demonstrated the OLD
        # audit skipped: an accepted confirmation with a reported interval
        # mutated to sit outside [70, 80] used to pass with
        # `matches_firmware=True`. It cannot now.
        raw_confirmation_observations = {
            int(decoded["slot_index"]): decoded
            for decoded in _bw_collect_raw_observations(
                ticks, _BW_RAW_OBSERVATION_PHASE_CONFIRMATION
            )
        }
        for observation in raw_confirmation_observations.values():
            if int(observation["plan_digest"]) != int(confirmation_terminal["plan_digest"]):
                raise ValueError(
                    "breakaway confirmation raw observation does not chain to its own plan digest"
                )
        reported_by_slot = {int(record["slot_index"]): record for record in observations}
        confirmation_multiplier = _f32(_confidence_multiplier(_BW_CONFIRMATION_FAMILY_SIZE))
        all_recomputed_contained = True
        for slot_index in range(_BW_CONFIRMATION_FAMILY_SIZE):
            expected_direction = _BW_CONFIRMATION_SLOT_ORDER_CODES[slot_index]
            raw = raw_confirmation_observations.get(slot_index)
            reported = reported_by_slot.get(slot_index)
            if raw is None:
                if confirmation_accepted:
                    mismatches.append(
                        f"confirmation accepted but slot {int(slot_index)} has no raw evidence "
                        f"to independently verify"
                    )
                    all_recomputed_contained = False
                continue
            if int(raw["direction"]) != expected_direction:
                mismatches.append(
                    f"confirmation slot {int(slot_index)} raw direction={int(raw['direction'])}"
                    f" does not match the fixed CONFIRMATION_SLOT_ORDER (want direction="
                    f"{int(expected_direction)})"
                )
                all_recomputed_contained = False
            if reported is not None and int(reported["direction"]) != expected_direction:
                mismatches.append(
                    f"confirmation slot {int(slot_index)} reported direction="
                    f"{int(reported['direction'])} does not match the fixed "
                    f"CONFIRMATION_SLOT_ORDER (want direction={int(expected_direction)})"
                )
                all_recomputed_contained = False
            interval = _bw_observation_interval_percent(
                int(raw["mean_rate_q"]),
                int(raw["variance_of_mean_q2"]),
                int(raw["target_rate_q"]),
                int(raw["direction"]),
                confirmation_multiplier,
                1.0,
                True,
            )
            if interval is None:
                if confirmation_accepted:
                    mismatches.append(
                        f"confirmation accepted but slot {int(slot_index)} raw evidence is not "
                        f"independently representable"
                    )
                    all_recomputed_contained = False
                continue
            offline_lower_milli = _bw_percent_to_milli(interval[0])
            offline_upper_milli = _bw_percent_to_milli(interval[1])
            if not (offline_lower_milli >= 70_000 and offline_upper_milli <= 80_000):
                all_recomputed_contained = False
            if reported is not None:
                if offline_lower_milli != int(reported["response_lower_percent_milli"]):
                    mismatches.append(
                        f"confirmation slot {int(slot_index)} response_lower_percent_milli: "
                        f"firmware={int(reported['response_lower_percent_milli'])} offline="
                        f"{int(offline_lower_milli)}"
                    )
                if offline_upper_milli != int(reported["response_upper_percent_milli"]):
                    mismatches.append(
                        f"confirmation slot {int(slot_index)} response_upper_percent_milli: "
                        f"firmware={int(reported['response_upper_percent_milli'])} offline="
                        f"{int(offline_upper_milli)}"
                    )
                statistic = _bw_observation_statistic(
                    int(raw["mean_rate_q"]),
                    int(raw["variance_of_mean_q2"]),
                    int(raw["target_rate_q"]),
                    int(raw["direction"]),
                )
                if statistic is not None:
                    offline_se_permille = _bw_relative_se_to_permille(statistic[1])
                    if offline_se_permille != int(reported["relative_standard_error_permille"]):
                        mismatches.append(
                            f"confirmation slot {int(slot_index)} "
                            f"relative_standard_error_permille: firmware="
                            f"{int(reported['relative_standard_error_permille'])} offline="
                            f"{int(offline_se_permille)}"
                        )
            elif confirmation_accepted:
                mismatches.append(
                    f"confirmation accepted but slot {int(slot_index)} has no reported "
                    f"observation record"
                )
        if confirmation_accepted and not all_recomputed_contained:
            mismatches.append(
                "confirmation accepted but at least one independently recomputed held-out "
                "response interval is not contained in [70, 80] percent (or its direction, "
                "schedule, or representability failed) -- REJECTED"
            )
        if confirmation_accepted and confirmation_planned is not None:
            nominated_p_raw = int(confirmation_planned["nominated_p_raw"])
            confirmed_p_raw = int(confirmation_terminal["confirmed_p_raw"])
            if nominated_p_raw != confirmed_p_raw:
                mismatches.append(
                    f"confirmation accepted nominated_p_raw={int(nominated_p_raw)} != "
                    f"confirmed_p_raw={int(confirmed_p_raw)}"
                )

        if observations:
            expected_max_se = max(
                int(record["relative_standard_error_permille"]) for record in observations
            )
            if expected_max_se != int(confirmation_terminal["max_relative_se_permille"]):
                mismatches.append(
                    f"confirmation max_relative_se_permille: firmware="
                    f"{int(confirmation_terminal['max_relative_se_permille'])} offline="
                    f"{int(expected_max_se)}"
                )
        expected_required_se = breakaway_required_relative_se_permille()
        if expected_required_se != int(confirmation_terminal["required_relative_se_permille"]):
            mismatches.append(
                f"confirmation required_relative_se_permille: firmware="
                f"{int(confirmation_terminal['required_relative_se_permille'])} offline="
                f"{int(expected_required_se)}"
            )

    campaign_terminal = _bw_single_record(ticks[-1], "breakaway_campaign_terminal")
    campaign_accepted = int(campaign_terminal["accepted"]) == 1
    # A confirmed acceptance whose Integral plan build is later refused
    # legitimately leaves campaign_accepted=False with confirmation_accepted=
    # True -- confirmation and Integral admission are deliberately
    # separate, stacked gates. Only the reverse direction (an Integral plan
    # materializing without a confirmed acceptance) is structurally
    # impossible and remains a genuine validator error.
    if campaign_accepted and not confirmation_accepted:
        raise ValueError(
            f"breakaway campaign terminal accepted={int(campaign_terminal['accepted'])} does "
            f"not trace back to an accepted confirmation (confirmation accepted="
            f"{int(confirmation_accepted)})"
        )
    if campaign_accepted:
        if int(campaign_terminal["terminal_cause"]) != _BW_TERMINAL_CAUSE_ACCEPTED:
            mismatches.append(
                f"campaign terminal_cause on accept: firmware="
                f"{int(campaign_terminal['terminal_cause'])} want="
                f"{int(_BW_TERMINAL_CAUSE_ACCEPTED)}"
            )
        if int(campaign_terminal["integral_plan_digest"]) == 0:
            mismatches.append("campaign terminal reports accepted with a zero integral_plan_digest")
    elif int(campaign_terminal["integral_plan_digest"]) != 0:
        mismatches.append("campaign terminal reports no accept but a nonzero integral_plan_digest")

    # `TargetBandSkipped` (cause 7) and `AdditiveLadderResolutionBudgetExceeded`
    # (cause 10) are decoded as distinct `terminal_cause` values already, but
    # spec section 9 draws a structural line between them that's worth
    # checking, not just trusting the numeric code: budget-exceeded is a
    # planning refusal *before motion*, so it can only appear on a zero-plan
    # campaign terminal (no discovery plan was ever frozen); target-band-
    # skipped is a genuine discovery *result*, so it requires a real discovery
    # terminal record reporting that same cause.
    campaign_cause = int(campaign_terminal["terminal_cause"])
    additive_ladder_resolution_budget_exceeded = campaign_cause == 10
    discovery_target_band_skipped = discovery_terminal is not None and (
        int(discovery_terminal["terminal_cause"]) == 7
    )
    if additive_ladder_resolution_budget_exceeded and discovery_planned is not None:
        mismatches.append(
            "AdditiveLadderResolutionBudgetExceeded is a planning refusal before motion, "
            "but a discovery plan was frozen for this campaign"
        )
    if discovery_target_band_skipped and campaign_cause != 7:
        mismatches.append(
            "discovery TargetBandSkipped did not propagate to the campaign terminal cause "
            f"(campaign terminal_cause={campaign_cause})"
        )

    workflow_duration_ticks = int(ticks[-1]["timestamp_ticks"]) - int(  # type: ignore[arg-type]
        ticks[0]["timestamp_ticks"]  # type: ignore[arg-type]
    )
    if workflow_duration_ticks < 0:
        raise ValueError("breakaway campaign terminal precedes its own probe plan in time")
    maximum_workflow_ms = (
        int(discovery_planned["maximum_workflow_ms"]) if discovery_planned is not None else None
    )
    workflow_duration_ms: float | None = None
    if clock_hz:
        workflow_duration_ms = workflow_duration_ticks * 1_000 / clock_hz
        if maximum_workflow_ms is not None and workflow_duration_ms > maximum_workflow_ms:
            mismatches.append(
                f"workflow duration {workflow_duration_ms:.1f} ms exceeds the reserved "
                f"maximum_workflow_ms={int(maximum_workflow_ms)}"
            )

    return {
        "phase_sequence": [BREAKAWAY_EVENT_ID_NAMES[tick["event_id"]] for tick in ticks],
        "probe_planned": probe_planned,
        "probe_digest_verified": probe_digest_verified,
        "search_grid": search_grid,
        "probe_result": probe_result,
        "resolved_breakaway_p_raw": resolved_breakaway_raw,
        "probe_terminal": probe_terminal,
        "probe_replay": probe_replay,
        "discovery_planned": discovery_planned,
        "discovery_terminal": discovery_terminal,
        "discovery_inert_fraction": discovery_inert_fraction,
        "post_breakaway_transient": post_breakaway_transient,
        "confirmation_planned": confirmation_planned,
        "confirmation_accepted": confirmation_accepted,
        "campaign_terminal": campaign_terminal,
        "additive_ladder_resolution_budget_exceeded": additive_ladder_resolution_budget_exceeded,
        "discovery_target_band_skipped": discovery_target_band_skipped,
        "workflow_duration_ticks": workflow_duration_ticks,
        "workflow_duration_ms": workflow_duration_ms,
        "maximum_workflow_ms": maximum_workflow_ms,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
    }
