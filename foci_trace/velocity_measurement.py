"""Independent reference for windowed encoder-velocity confidence analysis."""

from __future__ import annotations

import contextlib
import itertools
import math
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from statistics import median

from .scored_rest import select_causal_rest_suffixes

RATE_Q_SCALE = 1_000
RECOVERY_MOVING_ALLOWANCE_US = 512_000
RECOVERY_ZERO_SETTLE_US = 500_000
RECOVERY_ANALYSIS_BUDGET_US = 236_000
RECOVERY_STABILIZATION_US = 500_000
MIN_SELECTED_BLOCKS = 16
CONFIDENCE_MULTIPLIER_95 = 2.131
# Parts-per-million of settled rate a moving stroke may drift across its
# regression span. Mirrors firmware's MAX_MOVING_DRIFT_PPM.
MAX_MOVING_DRIFT_PPM = 200_000
# Reported drift ratio when no regression produced one, or when the settled
# interval contains zero (the Inert condition bypasses the drift bound
# entirely). Mirrors firmware's NO_REPORTABLE_DRIFT_RATIO_PPM: a zero would
# read as "measured, and tiny", the opposite of the truth here.
NO_REPORTABLE_DRIFT_RATIO_PPM = 0xFFFFFFFF
# Schema revision at which firmware's moving-stroke gate started applying the
# drift bound instead of the slope-significance test. Every capture recorded
# on an earlier schema was produced by firmware that never ran the drift
# bound, so the offline mirror must keep applying the significance test for
# those captures rather than a gate that firmware itself never executed on
# that data.
#
# Stage B and Stage C number their schema revisions on independent counters,
# so there are two thresholds, not one: Task 7 bumped firmware's
# COMBINED_STAGE_B_SCHEMA_REVISION from 13 to 14 and COMBINED_STAGE_C_SCHEMA_REVISION
# from 12 to 13 in the same commit. Stage B revision 13 therefore predates the
# drift bound -- it was live firmware until Task 7 -- even though it is the
# same integer as Stage C's post-bump floor. Comparing a Stage B revision
# against the Stage C threshold (or vice versa) would silently apply the
# wrong gate to real captures. Each call site must pass the threshold that
# matches the schema namespace its `schema_revision` was drawn from.
#
# Decode-only mirror: firmware renamed both constants off the
# "combined" name (`COMBINED_STAGE_B_SCHEMA_REVISION` ->
# `STAGE_B_PLAN_SCHEMA_REVISION`, `COMBINED_STAGE_C_SCHEMA_REVISION` ->
# `STAGE_C_PLAN_SCHEMA_REVISION`; foci-firmware commit 5fdef92). Firmware no
# longer defines the old names below; the values (14, 13) are unchanged.
STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION = 14  # firmware: STAGE_B_PLAN_SCHEMA_REVISION
STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION = 13  # firmware: STAGE_C_PLAN_SCHEMA_REVISION

# First revision whose walk continues past a rejected window instead of ending
# there: firmware's BREAKAWAY_STAGE_B/C_SCHEMA_REVISION. Every earlier revision,
# including the breakaway 15/14 and combined 14/13 that precede them, replays
# under the historical walk.
STAGE_B_SUFFIX_WALK_MIN_SCHEMA_REVISION = 16
STAGE_C_SUFFIX_WALK_MIN_SCHEMA_REVISION = 15

# First revision whose scored-rest chain closes only on evidence against rest,
# rather than on any suffix that is not zero-compatible.
STAGE_B_REST_WALK_MIN_SCHEMA_REVISION = 17
STAGE_C_REST_WALK_MIN_SCHEMA_REVISION = 16

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

# The two firmware crate-version hashes every archived capture carries in its
# `build_revision` field, retained here to reconstruct `build_features` for
# captures recorded before firmware emitted it. build_revision was later retired
# from new captures in favour of build_identity + build_features.
_LEGACY_Q8_8_BUILD_REVISION = 0x75536758
_CURRENT_Q4_12_BUILD_REVISION = 0x05E21F75


class UnsupportedIntegralRepresentationError(ValueError):
    """Raised when a firmware build has no declared integral-gain domain."""


_STAGE_RULE_THRESHOLDS: dict[str, tuple[int, int, int]] = {
    "b": (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        STAGE_B_SUFFIX_WALK_MIN_SCHEMA_REVISION,
        STAGE_B_REST_WALK_MIN_SCHEMA_REVISION,
    ),
    "c": (
        STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        STAGE_C_SUFFIX_WALK_MIN_SCHEMA_REVISION,
        STAGE_C_REST_WALK_MIN_SCHEMA_REVISION,
    ),
}


def _rule_features_from_schema(schema_revision: int, stage: str) -> int:
    """Reconstruct just the analysis rule bits (drift/suffix/rest) a legacy
    capture ran under, from its per-namespace schema thresholds.

    This ignores the integral-representation bit and never consults
    `build_revision`: the stationarity gates read only the rule bits, so an
    archived capture with no representation signal must still resolve its gate
    contract rather than raise. `stage` selects the namespace: "b" for Stage B,
    "c" for Stage C.
    """
    try:
        drift, suffix, rest = _STAGE_RULE_THRESHOLDS[stage]
    except KeyError:
        raise ValueError(f"unknown schema namespace {stage!r}") from None
    features = 0
    if schema_revision >= drift:
        features |= BUILD_FEATURE_DRIFT_BOUND_GATE
    if schema_revision >= suffix:
        features |= BUILD_FEATURE_SUFFIX_WALK_CONTINUES
    if schema_revision >= rest:
        features |= BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE
    return features


def reconstruct_build_features(
    build_revision: int | None, schema_revision: int, *, stage: str
) -> int:
    """Rebuild the rule contract a pre-`build_features` capture ran under.

    Bit 0 (integral representation) comes from the crate-version
    `build_revision`; the three rule bits reconstruct from the per-namespace
    schema thresholds, so an archived capture is analysed under exactly the gate
    that recorded it. `stage` selects the schema namespace: "b" for Stage B, "c"
    for Stage C.
    """
    if stage not in _STAGE_RULE_THRESHOLDS:
        raise ValueError(f"unknown schema namespace {stage!r}")
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
    return features | _rule_features_from_schema(schema_revision, stage)


def capture_rule_features(
    plan: dict[str, object], *, stage: str, schema_revision: int | None = None
) -> int | None:
    """Effective analysis rule-bit contract for the stationarity gates.

    A newer capture states `build_features` directly; a legacy capture
    reconstructs only its rule bits from the stage's schema thresholds. Unlike
    `capture_build_features` this never consults `build_revision` and never
    raises on a missing integral-representation signal, because the gates read
    only the rule bits. Returns `None` when the capture offers no rule signal at
    all (no `build_features` and no schema revision), which the gates treat as
    the legacy significance test.

    `schema_revision` overrides the plan's own, for Stage B: its wire plan
    record carries no schema revision, so the caller derives one from the
    evidence stream and passes it here. `stage` selects the namespace: "b" for
    Stage B, "c" for Stage C.
    """
    explicit = plan.get("build_features")
    if explicit is not None:
        return int(explicit)  # type: ignore[arg-type]
    effective = schema_revision if schema_revision is not None else plan.get("schema_revision")
    if effective is None:
        return None
    return _rule_features_from_schema(int(effective), stage)  # type: ignore[arg-type]


def capture_build_features(plan: dict[str, object], *, stage: str) -> int:
    """Return the effective rule contract for one decoded capture.

    A newer capture states `build_features` directly; a legacy capture
    carries only `build_revision`, from which the contract is reconstructed via
    the per-namespace schema thresholds. `stage` selects the namespace: "b" for
    Stage B, "c" for Stage C.
    """
    explicit = plan.get("build_features")
    if explicit is not None:
        return int(explicit)  # type: ignore[arg-type]
    build_revision = plan.get("build_revision")
    return reconstruct_build_features(
        int(build_revision) if isinstance(build_revision, int) else None,
        int(plan["schema_revision"]),  # type: ignore[arg-type]
        stage=stage,
    )


def project_combined_response_selection(
    rungs: list[dict[str, object]],
    *,
    target_rpm: int,
    encoder_counts_per_rev: int,
) -> dict[str, object]:
    """Project the combined selector over preserved Stage-B rung intervals."""
    if target_rpm <= 0 or encoder_counts_per_rev <= 0:
        raise ValueError("combined response target must be positive")
    target_numerator = target_rpm * encoder_counts_per_rev * RATE_Q_SCALE
    lower = target_numerator * 70
    upper = target_numerator * 80
    for rung in rungs:
        eligible = list(rung["eligible"])  # type: ignore[arg-type]
        if eligible != [True, True]:
            continue
        intervals = [list(rung["forward"]), list(rung["reverse"])]  # type: ignore[arg-type]
        normalized = [
            intervals[0],
            [-intervals[1][1], -intervals[1][0]],
        ]
        if all(
            interval[0] * 60 * 100 >= lower and interval[1] * 60 * 100 <= upper
            for interval in normalized
        ):
            return {
                "interpretation": "projected_result_under_combined_selector",
                "selected_p": int(rung["p"]),
                "selected_intervals": intervals,
            }
    return {
        "interpretation": "projected_result_under_combined_selector",
        "selected_p": None,
        "selected_intervals": None,
    }


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


def recovery_target_rpm(start_offset_counts: int, target_velocity_rpm: int) -> int:
    """Return the constant-amplitude target directed toward the immutable origin."""
    if start_offset_counts == 0 or target_velocity_rpm <= 0:
        raise ValueError("recovery target requires an offset and positive amplitude")
    return -target_velocity_rpm if start_offset_counts > 0 else target_velocity_rpm


def recovery_wrong_way(start_offset_counts: int, current_offset_counts: int) -> bool:
    """Match the one-count-inclusive growth guard used by the safety worker."""
    if start_offset_counts == -(1 << 31) or current_offset_counts == -(1 << 31):
        return True
    return abs(current_offset_counts) > abs(start_offset_counts) + 1


def _commanded_side_rate(direction: int, interval: list[int]) -> int | None:
    if direction not in (-1, 1) or len(interval) != 2 or interval[0] > interval[1]:
        return None
    if direction == 1 and interval[0] > 0:
        return int(interval[0])
    if direction == -1 and interval[1] < 0:
        return abs(int(interval[1]))
    return None


def select_recovery_authority(
    direction: int,
    ordinary_rate_intervals: list[list[int]],
    structured_rate_intervals: list[list[int]],
) -> dict[str, int]:
    """Select the firmware-equivalent minimum recovery-only rate authority."""
    ordinary = [
        rate
        for interval in ordinary_rate_intervals
        if (rate := _commanded_side_rate(direction, interval)) is not None
    ]
    structured = [
        rate
        for interval in structured_rate_intervals
        if (rate := _commanded_side_rate(direction, interval)) is not None
    ]
    ordinary_rate = min(ordinary, default=None)
    structured_rate = min(structured, default=None)
    if ordinary_rate is not None and (structured_rate is None or ordinary_rate <= structured_rate):
        measured, source = ordinary_rate, 1
    elif structured_rate is not None:
        measured, source = structured_rate, 5
    else:
        return {"measured_lower_rate_q": 0, "lower_rate_q": 0, "binding_source": 2}
    return {
        "measured_lower_rate_q": measured,
        "lower_rate_q": _compact_signed_rate(measured, upper=False),
        "binding_source": source,
    }


def project_recovery_authority(
    record: dict[str, object],
    *,
    direction: int,
    ordinary_rate_intervals: list[list[int]],
    structured_rate_intervals: list[list[int]],
    primitive_cap_us: int,
) -> dict[str, object]:
    """Project schema-5 authority without changing a historical record's meaning."""
    selected = select_recovery_authority(
        direction, ordinary_rate_intervals, structured_rate_intervals
    )
    return_distance = recovery_return_distance_counts(
        int(record["start_offset_counts"]), int(record["origin_band_counts"])
    )
    deadline_us = None
    if return_distance and selected["lower_rate_q"]:
        with contextlib.suppress(ValueError):
            deadline_us = recovery_runtime_us(
                return_distance, selected["lower_rate_q"], primitive_cap_us
            )
    return {
        "interpretation": "what_if_projection",
        "target_schema_revision": 5,
        **selected,
        "return_distance_counts": return_distance,
        "deadline_us": deadline_us,
    }


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


def audit_selected_recovery_rest(
    record: dict[str, object],
    plan: dict[str, object],
    terminal: dict[str, object],
    recovery_records: list[dict[str, object]],
) -> dict[str, object]:
    """Check one firmware-authored selected-recovery rest diagnostic."""
    mismatches: list[str] = []
    mismatches.extend(audit_recovery_rest_geometry(record, plan, expected_stage=0))
    disposition = int(record["disposition"])
    rung_index = int(record["rung_index"])
    matching_recoveries = [
        item for item in recovery_records if int(item["rung_index"]) == rung_index
    ]

    duration = (
        int(plan["nominal_workflow_ms"]),
        int(plan["maximum_workflow_ms"]),
    )
    recovery_wide = duration == (282_923, 298_632)
    if duration not in ((271_423, 298_632), (282_923, 298_632)):
        mismatches.append("selected recovery rest is outside the Stage-B-11 duration pair")
    if not recovery_wide and rung_index != int(terminal["emitted_rungs"]) - 1:
        mismatches.append("selected recovery rest is not attached to the final emitted rung")

    if disposition == 0:
        if len(matching_recoveries) != 1:
            mismatches.append("confirmed selected recovery rest lacks one ordinary recovery")
        elif recovery_wide and int(matching_recoveries[0]["outcome"]) not in (1, 5):
            mismatches.append("recovery-wide rest diagnostic follows a no-motion recovery")
        elif int(record["entry_to_band_duration_us"]) > int(
            matching_recoveries[0]["moving_duration_us"]
        ):
            mismatches.append("selected recovery entry duration exceeds ordinary moving duration")
    elif matching_recoveries:
        mismatches.append("unconfirmed selected recovery rest emitted an ordinary recovery")
    if disposition == 1:
        selected_terminal = (
            int(terminal["outcome"]) == STAGE_B_OUTCOME_INCONCLUSIVE
            and int(terminal["cause"]) == STAGE_B_CAUSE_RECOVERY_UNAVAILABLE
            and int(terminal.get("recovery_unavailable", 0)) == 1
        )
        if not selected_terminal and not (recovery_wide and int(terminal["cause"]) == 53):
            mismatches.append("unconfirmed selected recovery rest has the wrong Stage-B terminal")

    return {
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "record": dict(record),
    }


def _selected_recovery_scope_comparison(
    selected: dict[str, object],
    recovery_records: list[dict[str, object]],
    plan: dict[str, object],
) -> dict[str, object]:
    """Report whether intermediate recoveries share the selected return geometry."""
    selected_rung = int(selected["rung_index"])
    selected_return = int(selected["planned_return_distance_counts"])
    selected_entry = int(selected["entry_to_band_duration_us"])
    intermediate = []
    for record in recovery_records:
        if int(record["rung_index"]) == selected_rung:
            continue
        return_distance = recovery_return_distance_counts(
            int(record["start_offset_counts"]), int(plan["recovery_origin_band_counts"])
        )
        intermediate.append(
            {
                "rung_index": int(record["rung_index"]),
                "start_offset_counts": int(record["start_offset_counts"]),
                "origin_band_counts": int(record["origin_band_counts"]),
                "planned_return_distance_counts": return_distance,
                "moving_duration_us": int(record["moving_duration_us"]),
                "short_return_like_selected": (
                    return_distance is not None and return_distance <= selected_return
                ),
                "fast_entry_like_selected": int(record["moving_duration_us"]) <= selected_entry,
            }
        )
    return {
        "interpretation": "selected-recovery scope comparison; never firmware gating",
        "selected": {
            name: int(selected[name])
            for name in (
                "rung_index",
                "planned_return_distance_counts",
                "entry_to_band_duration_us",
                "lead_in_net_displacement_counts",
                "lead_in_displacement_span_counts",
                "scored_net_displacement_counts",
                "scored_min_displacement_counts",
                "scored_max_displacement_counts",
            )
        },
        "intermediate": intermediate,
        "scope_mismatch": any(
            bool(item["short_return_like_selected"]) or bool(item["fast_entry_like_selected"])
            for item in intermediate
        ),
    }


def _f32(value: float | int) -> float:
    """Round one value to the firmware's IEEE-754 single-precision domain."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
STAGE_B_OUTCOME_COMPLETE_CANDIDATE = 0
STAGE_B_OUTCOME_COMPLETE = 1
STAGE_B_OUTCOME_INCONCLUSIVE = 2
STAGE_B_OUTCOME_FAULT = 3
STAGE_B_CAUSE_NO_DIRECTIONAL_MODEL = 1
STAGE_B_CAUSE_NO_JOINT_OPERABLE_REGION = 2
STAGE_B_CAUSE_CURRENT_HEADROOM = 4
STAGE_B_CAUSE_REPRODUCTION_MISMATCH = 5
STAGE_B_CAUSE_RECOVERY_UNAVAILABLE = 7


def _stage_b_inconclusive_cause(model_direction_mask: int, *, recovery_failure: bool) -> int:
    if recovery_failure:
        return STAGE_B_CAUSE_RECOVERY_UNAVAILABLE
    if model_direction_mask != 3:
        return STAGE_B_CAUSE_NO_DIRECTIONAL_MODEL
    return STAGE_B_CAUSE_NO_JOINT_OPERABLE_REGION


class AccuracyDecision(StrEnum):
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
class DirectionalConsensus:
    """Exact depth-three consensus over four closed intervals."""

    classification: int
    components: tuple[tuple[int, int], ...]
    collected_mask: int
    eligible_mask: int
    included_mask: int


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
    Stage B / Stage C schema-namespace ambiguity never reaches this gate.
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


def analyze_stationary_tail(
    observations: list[VelocityObservation],
    *,
    build_features: int | None = None,
) -> dict[str, int]:
    """Reconstruct the firmware's nested-dyadic stationary tail from raw samples.

    This entry point takes raw samples only, with no plan or replay context, so
    it genuinely cannot resolve the recording firmware's rule contract itself;
    callers that know it should pass `build_features` (see
    `capture_rule_features`). `None` defaults to the legacy significance-based
    moving gate with the historical walk (see `_classify_stationarity`).
    """
    if not observations:
        raise ValueError("stationary-tail analysis requires raw observations")
    if any(item.elapsed_us <= 0 for item in observations):
        raise ValueError("elapsed_us must be positive")
    return _stationary_suffix(observations[-256:], build_features=build_features)


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
        variance = sum(float(value - difference_mean) ** 2 for value in differences) / (
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
                f"{label} {field_name}: firmware={firmware[field_name]} offline="
                f"{offline[field_name]}"
            )


def _compare_optional_drift_ratio(
    mismatches: list[str],
    label: str,
    firmware: dict[str, object],
    offline: dict[str, int],
) -> None:
    """Compare `drift_ratio_ppm` only when the wire record actually carries it.

    Captures recorded before the drift bound landed decode this field as
    `None` (see `protocol.py`'s observation decoders), while the offline
    mirror always computes a value -- a real ratio or the
    `NO_REPORTABLE_DRIFT_RATIO_PPM` sentinel -- because `_stationary_suffix`
    forms it unconditionally. Comparing unconditionally would fail every
    pre-drift-bound replay capture against a value firmware never had the
    chance to report, so this skips silently when the wire field is absent.
    """
    reported = firmware.get("drift_ratio_ppm")
    if reported is None:
        return
    if int(reported) != int(offline["drift_ratio_ppm"]):
        mismatches.append(
            f"{label} drift_ratio_ppm: firmware={reported} offline={offline['drift_ratio_ppm']}"
        )


# Stage B's wire classification namespace reserves 4..8 for the five
# StationarityExclusion variants, but `4 + exclusion` collides once
# ExcessiveDrift (discriminant 5) exists: `4 + 5 == 9`, which is
# WrongDirectionFault. Mirrors the explicit map firmware's
# observation_class_report_code uses after hitting the identical collision.
_STAGE_B_CLASS_CODE = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 12}

# Every code firmware assigns to StationarityExcluded(_), derived from the map
# above so appending an exclusion extends the numeric bands with it. The old
# literal `4 <= code <= 8` band silently sorted ExcessiveDrift (code 12) into
# the fault arm, where firmware matches StationarityExcluded(_) and yields
# EXCLUDED.
_STAGE_B_EXCLUSION_CODES = frozenset(_STAGE_B_CLASS_CODE.values())

# Every code outside the four moving classes (0-3) and the stationarity
# exclusions is a fault. Firmware's own default for a classification byte it
# cannot decode is InvalidEvidenceFault -- see analyze_directional_set's
# `.unwrap_or(VelocityObservationClass::InvalidEvidenceFault)` in
# sweep_evidence.rs -- so this must be derived the same way the exclusion set
# is, not left as a literal `{9, 10, 11}`. The literal was accidentally
# correct only because nothing occupied 13-15 yet: the classification field
# is 4 bits wide, so those codes are representable, and a literal fault set
# would send them to _rung_class's Ambiguous catch-all instead of matching
# firmware's fault default. Unreachable today since firmware never emits
# above 12, but the derivation keeps it that way instead of leaving the trap
# armed for the next exclusion or fault variant added above this line.
_STAGE_B_FAULT_CODES = frozenset(range(16)) - {0, 1, 2, 3} - _STAGE_B_EXCLUSION_CODES


def _reconstruct_observation(
    raw: list[VelocityObservation],
    record: dict[str, object],
    plan: dict[str, object],
    terminal: dict[str, object],
    current_record: dict[str, object] | None = None,
    *,
    schema_revision: int | None = None,
) -> dict[str, int]:
    """Reconstruct one Stage B observation from its raw moving samples.

    Stage B's wire "plan" record never carries a schema revision (unlike Stage
    C's), so callers that know the replay's schema (derived from the evidence
    stream, e.g. `replay_stage_b_regions`) must pass it explicitly as
    `schema_revision`, using Stage B's own schema numbering. `None` defaults
    to the legacy significance-based moving gate, which is also correct for
    the oldest replay path that predates the `stage_b_terminal` record kind
    entirely and so has no schema signal of its own to offer.

    The rule contract is resolved with `stage="b"`, never Stage C's namespace:
    every `schema_revision` reaching this function is a Stage B revision by
    construction, so the namespace is fixed here rather than left for callers to
    (mis)supply. A newer Stage B plan states `build_features` directly and
    the passed `schema_revision` is then unused.
    """
    build_features = capture_rule_features(plan, stage="b", schema_revision=schema_revision)
    stationarity = _stationary_suffix(raw[-256:], build_features=build_features)
    p = int(record["velocity_p"])
    target_velocity_rpm = int(record["target_velocity_rpm"])
    predicted = max(
        (p * abs(target_velocity_rpm) + 255) // 256,
        max((item.peak_torque_target_abs or 0) for item in raw),
        int(current_record["moving_pid_output_peak"]) if current_record is not None else 0,
    )
    if not stationarity["usable"]:
        classification = _STAGE_B_CLASS_CODE[stationarity["exclusion"]]
        return {
            "classification": classification,
            "target_velocity_rpm": target_velocity_rpm,
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
        "target_velocity_rpm": target_velocity_rpm,
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


def _depth_three_consensus(
    intervals: list[tuple[int, int] | None],
    *,
    collected_mask: int = 0b1111,
    eligible_mask: int | None = None,
) -> DirectionalConsensus:
    """Return the merged depth-three set for four closed intervals."""
    if len(intervals) != 4:
        raise ValueError("depth-three consensus requires four observations")
    if any(mask & ~0x0F for mask in (collected_mask, eligible_mask or 0)):
        raise ValueError("consensus mask exceeds four observations")
    if eligible_mask is None:
        eligible_mask = sum(
            1 << index
            for index, interval in enumerate(intervals)
            if collected_mask & (1 << index) and interval is not None
        )
    if eligible_mask & ~collected_mask:
        raise ValueError("eligible mask is not a subset of collected mask")
    if collected_mask != 0b1111:
        return DirectionalConsensus(3, (), collected_mask, eligible_mask, 0)

    intersections = []
    for triple in (0b0111, 0b1011, 0b1101, 0b1110):
        if triple & eligible_mask != triple:
            continue
        selected = [intervals[index] for index in range(4) if triple & (1 << index)]
        if any(interval is None for interval in selected):
            continue
        low = max(interval[0] for interval in selected if interval is not None)
        high = min(interval[1] for interval in selected if interval is not None)
        if low <= high:
            intersections.append((low, high))

    components: list[tuple[int, int]] = []
    for low, high in sorted(intersections):
        if components and low <= components[-1][1]:
            components[-1] = (components[-1][0], max(components[-1][1], high))
        else:
            components.append((low, high))
    if len(components) > 2:
        raise AssertionError("four depth-three intervals yield at most two components")
    if len(components) == 1:
        low, high = components[0]
        included_mask = sum(
            1 << index
            for index, interval in enumerate(intervals)
            if eligible_mask & (1 << index)
            and interval is not None
            and interval[0] <= high
            and low <= interval[1]
        )
        return DirectionalConsensus(
            0, tuple(components), collected_mask, eligible_mask, included_mask
        )
    if len(components) == 2:
        return DirectionalConsensus(1, tuple(components), collected_mask, eligible_mask, 0)
    return DirectionalConsensus(2, (), collected_mask, eligible_mask, 0)


def _rung_class(first: dict[str, int], second: dict[str, int]) -> tuple[int, int, int, bool]:
    first_class, second_class = first["classification"], second["classification"]
    moving_classes = {0, 2, 3}
    reproduced = first_class in moving_classes and second_class in moving_classes
    low = max(first.get("disturbance_low_q16", 0), second.get("disturbance_low_q16", 0))
    high = min(first.get("disturbance_high_q16", 0), second.get("disturbance_high_q16", 0))
    if first_class == second_class == 0:
        return (0, low, high, reproduced) if low <= high else (4, 0, 0, reproduced)
    if first_class == second_class == 1:
        return 1, 0, 0, reproduced
    if first_class == second_class == 2:
        return 2, 0, 0, reproduced
    if first_class == second_class == 3:
        return 3, 0, 0, reproduced
    if first_class in _STAGE_B_FAULT_CODES or second_class in _STAGE_B_FAULT_CODES:
        return 7, 0, 0, reproduced
    if first_class in _STAGE_B_EXCLUSION_CODES and second_class in _STAGE_B_EXCLUSION_CODES:
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


def _disturbance_point_cell(point: int) -> tuple[int, int]:
    """Return the Q16.16 cell represented by one truncated disturbance point."""
    if point > 0:
        return point, point + 1
    if point < 0:
        return point - 1, point
    return -1, 1


def _rate_bounds_from_disturbance_cell(
    target_rpm: int,
    counts_per_rev: int,
    p_raw: int,
    disturbance_cell_q16: tuple[int, int],
) -> tuple[int, int]:
    """Invert one disturbance cell into outward-rounded settled-rate bounds."""
    if counts_per_rev <= 0 or p_raw <= 0 or disturbance_cell_q16[0] > disturbance_cell_q16[1]:
        raise ValueError("invalid disturbance-cell rate inversion")
    target_term = target_rpm * counts_per_rev * 1_000 * p_raw * 65_536
    denominator = p_raw * 65_536 * 60

    def numerator(disturbance: int) -> int:
        return target_term - disturbance * 256 * counts_per_rev * 1_000

    return (
        _floor_div(numerator(disturbance_cell_q16[1]), denominator),
        _ceil_div(numerator(disturbance_cell_q16[0]), denominator),
    )


def _compact_signed_rate_parts(value: int, *, upper: bool) -> tuple[int, int]:
    """Encode one signed rate as the firmware's outward-rounded compact pair."""
    for shift in range(49):
        divisor = 1 << shift
        quotient = _trunc_div(value, divisor)
        remainder = value - quotient * divisor
        if upper and value > 0 and remainder:
            quotient += 1
        elif not upper and value < 0 and remainder:
            quotient -= 1
        if -(1 << 15) <= quotient < (1 << 15):
            return quotient, shift
    raise ValueError("signed rate does not fit the firmware compact domain")


def _compact_signed_rate(value: int, *, upper: bool) -> int:
    """Mirror firmware's conservative i16 rate compaction and decode."""
    mantissa, shift = _compact_signed_rate_parts(value, upper=upper)
    return mantissa << shift


def _project_stage_b_recovery_authorities(
    plan: dict[str, object],
    terminal: dict[str, object],
    gains: list[int],
    reconstructed: dict[tuple[int, int], dict[str, int]],
    boundaries: dict[tuple[int, int], dict[str, object]],
    recovery_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    projections = []
    observations_per_direction = int(plan["observations_per_direction"])
    for record in recovery_records:
        rung_index = int(record["rung_index"])
        if rung_index >= len(gains):
            continue
        direction = 1 if int(record["start_offset_counts"]) < 0 else -1
        direction_slot = 0 if direction == 1 else 1
        ordinary = []
        structured = []
        for observation_index in range(observations_per_direction):
            key = (rung_index, observation_index * 2 + direction_slot)
            observation = reconstructed.get(key)
            if observation is None:
                continue
            if int(observation["classification"]) == 0:
                ordinary.append(
                    list(
                        _rate_bounds_from_disturbance_cell(
                            abs(int(plan["target_velocity_rpm"])) * direction,
                            int(terminal["encoder_counts_per_rev"]),
                            gains[rung_index],
                            (
                                int(observation["disturbance_low_q16"]),
                                int(observation["disturbance_high_q16"]),
                            ),
                        )
                    )
                )
            boundary = boundaries.get(key)
            if (
                int(observation["classification"]) in _STAGE_B_EXCLUSION_CODES
                and boundary is not None
            ):
                structured.append(list(boundary["common_rate_interval_q"]))
        projection = project_recovery_authority(
            record,
            direction=direction,
            ordinary_rate_intervals=ordinary,
            structured_rate_intervals=structured,
            primitive_cap_us=int(plan["recovery_maximum_slot_us"]),
        )
        projection.update(
            {
                "rung_index": rung_index,
                "direction": direction,
                "recorded_lower_rate_q": int(record["lower_rate_q"]),
                "recorded_binding_source": int(record["binding_source"]),
                "recorded_outcome": int(record["outcome"]),
            }
        )
        projections.append(projection)
    return projections


def _common_wire_rate_bounds(values: list[int]) -> list[int]:
    """Normalize four compact directional rates to their shared wire shift."""
    encoded = [
        _compact_signed_rate_parts(value, upper=index in (1, 3))
        for index, value in enumerate(values)
    ]
    shift = max(item[1] for item in encoded)
    divisor = 1 << shift
    result = []
    for index, (mantissa, original_shift) in enumerate(encoded):
        value = mantissa << original_shift
        quotient = _trunc_div(value, divisor)
        remainder = value - quotient * divisor
        if index in (0, 2) and value < 0 and remainder:
            quotient -= 1
        elif index in (1, 3) and value > 0 and remainder:
            quotient += 1
        result.append(quotient << shift)
    return result


def _analyze_four_observation_direction(
    observations: list[dict[str, int] | None],
    *,
    direction: int,
    p: int,
    counts_per_rev: int,
    initial_lead_in: bool,
) -> dict[str, object]:
    """Derive one n=4 directional consensus without consuming firmware verdicts."""
    intervals = [
        (item["disturbance_low_q16"], item["disturbance_high_q16"])
        if item is not None and item["classification"] == 0
        else None
        for item in observations
    ]
    collected_mask = sum(1 << index for index, item in enumerate(observations) if item is not None)
    eligible_mask = sum(
        1 << index
        for index, item in enumerate(observations)
        if item is not None and item["classification"] == 0
    )
    consensus = _depth_three_consensus(
        intervals,
        collected_mask=collected_mask,
        eligible_mask=eligible_mask,
    )
    operable_count = sum(
        item is not None and item["classification"] in (0, 2) for item in observations
    )
    flags = int(operable_count >= 3) | (int(initial_lead_in) << 1)
    included = [
        item
        for index, item in enumerate(observations)
        if item is not None and consensus.included_mask & (1 << index)
    ]
    pooled = _pool_region_observations(included)
    rate_bounds = [0, 0, 0, 0]
    if included:
        signed_target = abs(int(included[0]["target_velocity_rpm"])) * direction
        mean_bounds = [
            _rate_bounds_from_disturbance_cell(
                signed_target,
                counts_per_rev,
                p,
                _disturbance_point_cell(item["disturbance_q16"]),
            )
            for item in included
        ]
        confidence_bounds = [
            _rate_bounds_from_disturbance_cell(
                signed_target,
                counts_per_rev,
                p,
                (item["disturbance_low_q16"], item["disturbance_high_q16"]),
            )
            for item in included
        ]
        exact = [
            min(bounds[0] for bounds in mean_bounds),
            max(bounds[1] for bounds in mean_bounds),
            min(bounds[0] for bounds in confidence_bounds),
            max(bounds[1] for bounds in confidence_bounds),
        ]
        rate_bounds = [
            _compact_signed_rate(value, upper=index in (1, 3)) for index, value in enumerate(exact)
        ]
    component = consensus.components[0] if consensus.classification == 0 else (0, 0)
    return {
        "class": consensus.classification,
        "low": component[0],
        "high": component[1],
        "components": [list(item) for item in consensus.components],
        "component_count": len(consensus.components),
        "collected_mask": consensus.collected_mask,
        "eligible_mask": consensus.eligible_mask,
        "included_mask": consensus.included_mask,
        "operable_count": operable_count,
        "directional_flags": flags,
        "observations": included,
        "rate_means": rate_bounds[:2],
        "rate_lows": rate_bounds[2:3],
        "rate_highs": rate_bounds[3:4],
        "rate_bounds_q": rate_bounds,
        "wire_rate_bounds_q": _common_wire_rate_bounds(rate_bounds),
        **pooled,
    }


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


def _segment_directional_regions(
    rungs: list[dict[str, object]],
    *,
    terminal_closure: str = "end_of_ladder",
    terminal_boundary_rung: int | None = None,
) -> list[dict[str, object]]:
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
        closed = _close_directional_region(active, terminal_closure, None)
        closed["causal_boundary_rung"] = terminal_boundary_rung
        regions.append(closed)
    return regions


def _joint_operable_regions(
    rungs: list[dict[str, object]],
    *,
    terminal_closure: str = "end_of_ladder",
    terminal_boundary_rung: int | None = None,
) -> list[dict[str, object]]:
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
                "closure": terminal_closure,
                "causal_boundary_rung": terminal_boundary_rung,
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


def _stage_b_handoff_sufficient(
    directional: list[list[dict[str, object]]],
    nomination: dict[str, object],
    selected: list[dict[str, object]],
    *,
    schema_revision: int,
) -> bool:
    """Return whether one run has the complete Stage B handoff for its schema."""
    if schema_revision >= 8:
        return bool(nomination)
    has_directional_models = all(
        any(region["kind"] == "valid" for region in regions) for regions in directional
    )
    return has_directional_models and bool(nomination) and all(bool(item) for item in selected)


def _stage_b_model_mask_from_validity(validity: list[int]) -> int:
    """Return the directional model mask for reproduced schema-8 validity."""
    return sum((1 << index) for index, code in enumerate(validity) if code in (3, 4))


def _select_schema8_directional_evidence(
    regions: list[dict[str, object]],
    *,
    provisional_rung: int,
) -> dict[str, object]:
    """Select the schema-8 directional model or fragment retained for one run."""
    candidates = [dict(region) for region in regions]
    valid = [region for region in candidates if region["kind"] == "valid"]
    if valid:
        candidates = valid
    if not candidates:
        return {}

    def rank(region: dict[str, object]) -> tuple[int, int, int]:
        first = int(region["first_rung"])
        last = int(region["last_rung"])
        distance = (
            first - provisional_rung
            if provisional_rung < first
            else provisional_rung - last
            if provisional_rung > last
            else 0
        )
        return (-int(region["member_count"]), distance, first)

    return min(candidates, key=rank)


def _compare_stage_b_candidates_schema2(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Mirror the historical schema-2 exact-membership reproduction semantics."""
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


def _stage_b_gain_at(p_start: int, p_top: int, rung_count: int, rung_index: int) -> int:
    """Regenerate one anchored, post-dedupe half-octave ladder gain."""
    if not 0 <= rung_index < rung_count or p_start <= 0 or p_start > p_top:
        raise ValueError("invalid Stage B ladder geometry")
    previous = 0
    emitted = 0
    for octave in range(16):
        even = p_start << octave
        odd = ((p_start * 181 << octave) + 127) >> 7
        for candidate in (even, odd):
            if candidate > p_top or candidate <= previous:
                continue
            if emitted == rung_index:
                return candidate
            previous = candidate
            emitted += 1
    if previous != p_top and emitted == rung_index:
        return p_top
    raise ValueError("rung is absent from the anchored Stage B ladder")


def _contiguous_stage_b_mask(mask: int, rung_count: int) -> bool:
    if mask <= 0 or rung_count not in range(1, 33) or mask >= 1 << rung_count:
        return False
    shifted = mask >> ((mask & -mask).bit_length() - 1)
    return (shifted & (shifted + 1)) == 0


def _stage_b_membership_comparison(previous: int, current: int) -> dict[str, int]:
    if previous == 0 or current == 0:
        low_delta = 0
        high_delta = 0
    else:
        previous_low = (previous & -previous).bit_length() - 1
        current_low = (current & -current).bit_length() - 1
        previous_high = previous.bit_length() - 1
        current_high = current.bit_length() - 1
        low_delta = current_low - previous_low
        high_delta = current_high - previous_high
    return {
        "previous": previous,
        "current": current,
        "core": previous & current,
        "previous_only": previous & ~current & 0xFFFF_FFFF,
        "current_only": current & ~previous & 0xFFFF_FFFF,
        "low_delta": low_delta,
        "high_delta": high_delta,
    }


def _stage_b_closed_intersection(left: list[int], right: list[int]) -> list[int]:
    return [max(int(left[0]), int(right[0])), min(int(left[1]), int(right[1]))]


def _stage_b_core_coverage(
    core: int, final_rung: int, p_start: int, p_top: int, rung_count: int
) -> dict[str, object]:
    if core == 0:
        return {
            "coverage": "transferred",
            "signed_rung_distance": 0,
            "gain_ratio": [0, 0],
        }
    first = (core & -core).bit_length() - 1
    last = core.bit_length() - 1
    if final_rung < first:
        source_rung = first
        distance = first - final_rung
        coverage = "transferred"
    elif final_rung > last:
        source_rung = last
        distance = last - final_rung
        coverage = "transferred"
    else:
        source_rung = final_rung
        distance = 0
        coverage = "full"
    if coverage == "full":
        ratio = [1, 1]
    else:
        source_p = _stage_b_gain_at(p_start, p_top, rung_count, source_rung)
        fixed_p = _stage_b_gain_at(p_start, p_top, rung_count, final_rung)
        ratio = [max(source_p, fixed_p), max(1, min(source_p, fixed_p))]
    return {
        "coverage": coverage,
        "signed_rung_distance": distance,
        "gain_ratio": ratio,
    }


def _core_restricted_common(
    previous: dict[str, object], current: dict[str, object], cores: list[int]
) -> list[list[int]] | None:
    previous_rungs = previous.get("per_rung_common_intervals")
    current_rungs = current.get("per_rung_common_intervals")
    if not isinstance(previous_rungs, list) or not isinstance(current_rungs, list):
        return None
    derived = []
    for direction, core in enumerate(cores[:2]):
        bounds: list[list[int]] = []
        for source in (previous_rungs, current_rungs):
            mapping = source[direction]
            for rung in range(32):
                if core & (1 << rung):
                    interval = mapping.get(rung, mapping.get(str(rung)))
                    if interval is None:
                        return None
                    bounds.append([int(interval[0]), int(interval[1])])
        derived.append([max(value[0] for value in bounds), min(value[1] for value in bounds)])
    return derived


def _schema8_run_candidate(report: dict[str, object]) -> dict[str, object]:
    """Derive one schema-8 retained summary from canonical per-run evidence."""
    replay = report["stage_b_replay"]
    plan = report["plan"]
    joint_regions = list(replay["joint_regions"])  # type: ignore[index]
    joint_union = 0
    for region in joint_regions:
        joint_union |= int(region["member_mask"])
    if joint_union == 0:
        raise ValueError("schema-8 projection requires per-run joint operability")
    provisional = _nominate_joint_region(
        joint_regions,
        list(replay["gains"]),  # type: ignore[index]
        int(plan["p_start"]),  # type: ignore[index]
        int(plan["p_top"]),  # type: ignore[index]
    )
    provisional_rung = int(provisional["rung_index"])

    selected = []
    for regions in replay["directions"]:  # type: ignore[index]
        selected.append(
            _select_schema8_directional_evidence(
                list(regions),
                provisional_rung=provisional_rung,
            )
        )

    return {
        "nominated_p": int(provisional["p"]),
        "memberships": [
            int(selected[0].get("member_mask", 0)),
            int(selected[1].get("member_mask", 0)),
            joint_union,
        ],
        "pooled_intervals": [
            list(selected[0].get("pooled_interval", [0, 0])),
            list(selected[1].get("pooled_interval", [0, 0])),
        ],
        "common_intervals": [
            list(selected[0].get("common_interval", [0, 0])),
            list(selected[1].get("common_interval", [0, 0])),
        ],
        "p_start": int(plan["p_start"]),  # type: ignore[index]
        "p_top": int(plan["p_top"]),  # type: ignore[index]
        "rung_count": int(plan["rung_count"]),  # type: ignore[index]
    }


def project_stage_b_schema8_pair(
    previous_report: dict[str, object], current_report: dict[str, object]
) -> dict[str, object]:
    """Project the schema-8 comparator over preserved per-run Stage-B evidence."""
    return _project_stage_b_schema8_candidates(
        _schema8_run_candidate(previous_report),
        _schema8_run_candidate(current_report),
    )


def _project_stage_b_schema8_candidates(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    from .stage_b_reproduction import project_schema8_candidates

    return project_schema8_candidates(previous, current)


def project_stage_b_core_pair(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Project the schema-3 firmware comparator over two sufficient run summaries."""
    geometry = ("p_start", "p_top", "rung_count")
    if any(int(previous[name]) != int(current[name]) for name in geometry):
        return {"outcome": "plan_mismatch", "reason_mask": 0}
    p_start, p_top, rung_count = (int(previous[name]) for name in geometry)
    previous_masks = [int(value) for value in previous["memberships"]]
    current_masks = [int(value) for value in current["memberships"]]
    if (
        p_start <= 0
        or p_start > p_top
        or rung_count not in range(1, 33)
        or any(
            not _contiguous_stage_b_mask(mask, rung_count)
            for mask in (*previous_masks, *current_masks)
        )
    ):
        return {"outcome": "evidence_integrity_fault", "reason_mask": 0}
    memberships = [
        _stage_b_membership_comparison(left, right)
        for left, right in zip(previous_masks, current_masks, strict=True)
    ]
    cores = [value["core"] for value in memberships]
    if any(core and not _contiguous_stage_b_mask(core, rung_count) for core in cores):
        return {"outcome": "evidence_integrity_fault", "reason_mask": 0}

    pooled = []
    common = []
    for direction in range(2):
        previous_pooled = [int(value) for value in previous["pooled_intervals"][direction]]
        current_pooled = [int(value) for value in current["pooled_intervals"][direction]]
        pooled_overlap = _stage_b_closed_intersection(previous_pooled, current_pooled)
        previous_common = [int(value) for value in previous["common_intervals"][direction]]
        current_common = [int(value) for value in current["common_intervals"][direction]]
        conservative = _stage_b_closed_intersection(previous_common, current_common)
        pooled.append(
            {
                "previous": previous_pooled,
                "current": current_pooled,
                "overlap": pooled_overlap,
            }
        )
        common.append(
            {
                "previous": previous_common,
                "current": current_common,
                "conservative": conservative,
                "nonempty": conservative[0] <= conservative[1],
            }
        )

    reason_mask = 0
    reason_mask |= int(cores[0].bit_count() < 3) << 0
    reason_mask |= int(cores[1].bit_count() < 3) << 1
    reason_mask |= int(cores[2] == 0) << 2
    reason_mask |= int(pooled[0]["overlap"][0] > pooled[0]["overlap"][1]) << 3
    reason_mask |= int(pooled[1]["overlap"][0] > pooled[1]["overlap"][1]) << 4

    fixed_p = 0
    reduced_margin = False
    coverage = [
        {"coverage": "transferred", "signed_rung_distance": 0, "gain_ratio": [0, 0]},
        {"coverage": "transferred", "signed_rung_distance": 0, "gain_ratio": [0, 0]},
    ]
    if cores[2]:
        joint_first = (cores[2] & -cores[2]).bit_length() - 1
        final_rung = joint_first + (cores[2].bit_count() - 1) // 2
        fixed_p = _stage_b_gain_at(p_start, p_top, rung_count, final_rung)
        coverage = [
            _stage_b_core_coverage(core, final_rung, p_start, p_top, rung_count)
            for core in cores[:2]
        ]
        reduced_margin = cores[2].bit_count() <= 2

    exact_common = _core_restricted_common(previous, current, cores)
    projection: dict[str, object] = {
        "outcome": "complete" if reason_mask == 0 else "reproduction_mismatch",
        "reason_mask": reason_mask,
        "memberships": memberships,
        "core_masks": cores,
        "previous_provisional_p": int(previous["nominated_p"]),
        "current_provisional_p": int(current["nominated_p"]),
        "fixed_p": fixed_p,
        "reduced_margin": reduced_margin,
        "coverage": coverage,
        "pooled": pooled,
        "common": common,
        "firmware_conservative_common": [value["conservative"] for value in common],
    }
    if exact_common is not None:
        projection["offline_derived"] = {
            "core_restricted_common": exact_common,
            "provenance": "per_rung_cross_run",
        }
    return projection


_STAGE_B_REPRODUCTION_FRAGMENT_KINDS = (
    "reproduction_v3_header",
    "reproduction_v4_header",
    "reproduction_v3_membership",
    "reproduction_v3_pooled",
    "reproduction_v3_common",
    "reproduction_v3_coverage",
    "reproduction_v3_digest",
    "reproduction_membership",
    "reproduction_pooled",
    "reproduction_common",
    "reproduction_coverage",
    "reproduction_digest",
)


def _normalized_reproduction_fragment_kind(kind: object) -> str:
    value = str(kind)
    return value.removeprefix("reproduction_v3_").removeprefix("reproduction_")


def _assemble_stage_b_reproduction(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """Validate and assemble one ordered Stage-B reproduction record group."""
    fragments = [
        value for value in records if value["record_kind"] in _STAGE_B_REPRODUCTION_FRAGMENT_KINDS
    ]
    if not fragments:
        return None
    header_kind = str(fragments[0]["record_kind"])
    if header_kind not in ("reproduction_v3_header", "reproduction_v4_header"):
        raise ValueError("Stage-B reproduction group does not start with a header")
    expected = [
        "header",
        *("membership" for _ in range(3)),
        *("pooled" for _ in range(2)),
        *("common" for _ in range(2)),
        *("coverage" for _ in range(2)),
        "digest",
    ]
    actual = [
        "header"
        if str(value["record_kind"]) in ("reproduction_v3_header", "reproduction_v4_header")
        else _normalized_reproduction_fragment_kind(value["record_kind"])
        for value in fragments
    ]
    if actual != expected:
        raise ValueError("Stage-B reproduction fragments are missing, duplicated, or out of order")
    identity = (int(fragments[0]["run_sequence"]), int(fragments[0]["evidence_sequence"]))
    if any(
        (int(value["run_sequence"]), int(value["evidence_sequence"])) != identity
        for value in fragments
    ):
        raise ValueError("Stage-B reproduction fragment identity changed")
    if [int(value["object"]) for value in fragments[1:4]] != [0, 1, 2]:
        raise ValueError("Stage-B reproduction membership fragments changed object order")
    for start in (4, 6, 8):
        if [int(value["direction"]) for value in fragments[start : start + 2]] != [0, 1]:
            raise ValueError("Stage-B reproduction directional fragments changed order")
    header = fragments[0]
    digest = fragments[-1]
    assembled = {
        "record_kind": "reproduction_v4"
        if header_kind == "reproduction_v4_header"
        else "reproduction_v3",
        "run_sequence": identity[0],
        "evidence_sequence": identity[1],
        "schema_revision": int(header["schema_revision"]),
        "outcome": int(header["outcome"]),
        "reason_mask": int(header["reason_mask"]),
        "fixed_p": int(header["fixed_p"]),
        "reduced_margin": bool(header["reduced_margin"]),
        "memberships": [dict(value) for value in fragments[1:4]],
        "pooled": [dict(value) for value in fragments[4:6]],
        "common": [dict(value) for value in fragments[6:8]],
        "coverage": [dict(value) for value in fragments[8:10]],
        "previous_digest_low": int(digest["previous_digest_low"]),
        "previous_digest_high": int(digest["previous_digest_high"]),
        "current_digest_low": int(digest["current_digest_low"]),
        "current_digest_high": int(digest["current_digest_high"]),
    }
    if header_kind == "reproduction_v4_header":
        assembled.update(
            {
                "previous_provisional_p": int(header["previous_provisional_p"]),
                "current_provisional_p": int(header["current_provisional_p"]),
                "selected_joint_mask": int(header["selected_joint_mask"]),
                "selected_first_rung": int(header["selected_first_rung"]),
                "selected_member_count": int(header["selected_member_count"]),
                "forward_validity": int(header["forward_validity"]),
                "reverse_validity": int(header["reverse_validity"]),
            }
        )
    else:
        assembled.update(
            {
                "previous_nominated_p": int(header["previous_nominated_p"]),
                "current_nominated_p": int(header["current_nominated_p"]),
            }
        )
    return assembled


def _assemble_stage_b_reproduction_v3(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """Assemble one historical schema-3 through schema-7 reproduction group."""
    assembled = _assemble_stage_b_reproduction(records)
    if assembled is not None and assembled["record_kind"] != "reproduction_v3":
        raise ValueError("expected a historical Stage-B reproduction group")
    return assembled


def _stage_b_record_identity(value: dict[str, object]) -> tuple[object, ...]:
    """Return the causal identity of one canonical Stage B record."""
    kind = str(value["record_kind"])
    if kind in ("observation", "structured_boundary"):
        return kind, int(value["rung_index"]), int(value["slot"])
    if kind == "rung":
        return kind, int(value["rung_index"])
    if kind == "rung_origin_recovery":
        return kind, int(value["rung_index"])
    if kind == "recovery_rest":
        return kind, int(value["rung_index"])
    if kind == "rest_selection":
        return kind, int(value["owner"]), int(value["rung_index"]), int(value["slot"])
    if kind == "velocity_primitive_current":
        return kind, int(value["rung_index"]), int(value["slot"])
    if kind == "directional_region":
        return kind, int(value["direction"]), int(value["member_mask"])
    if kind == "joint_region":
        return kind, int(value["member_mask"])
    if kind in ("reproduction_v3", "reproduction_v4"):
        return ("reproduction",)
    return (kind,)


def _stage_b_replay_expects_handoff(replay: dict[str, object]) -> bool:
    """Return whether the schema emits the full directional handoff record."""
    if bool(replay.get("combined_response_selected", False)):
        return False
    sufficient = bool(replay.get("sufficient", replay.get("nomination")))
    return sufficient and (
        int(replay.get("schema_revision", 7)) < 8
        or all(
            any(region["kind"] == "valid" for region in regions) for regions in replay["directions"]
        )
    )


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
    directional_after_rung: dict[int, list[tuple[object, ...]]] = {}
    directional_at_end: list[tuple[object, ...]] = []
    for direction, regions in enumerate(replay["directions"]):
        for region in regions:
            token = ("directional_region", direction, int(region["member_mask"]))
            boundary = region.get("causal_boundary_rung", region.get("boundary_rung"))
            if boundary is None:
                directional_at_end.append(token)
            elif region.get("closure") == "current_headroom":
                directional_after_rung.setdefault(int(boundary), []).append(token)
            else:
                directional_at_boundary.setdefault(int(boundary), []).append(token)
    joint_at_boundary: dict[int, list[tuple[object, ...]]] = {}
    joint_after_rung: dict[int, list[tuple[object, ...]]] = {}
    joint_at_end: list[tuple[object, ...]] = []
    for region in replay["joint_regions"]:
        token = ("joint_region", int(region["member_mask"]))
        if region["closure"] in ("end_of_ladder", "response_selected"):
            joint_at_end.append(token)
        elif region["closure"] == "current_headroom":
            causal_boundary = region.get("causal_boundary_rung")
            boundary_rung = (
                int(causal_boundary)
                if causal_boundary is not None
                else int(region["last_rung"]) + 1
            )
            joint_after_rung.setdefault(boundary_rung, []).append(token)
        else:
            joint_at_boundary.setdefault(int(region["last_rung"]) + 1, []).append(token)

    reported_rungs = {
        int(value["rung_index"]) for value in records if value["record_kind"] == "rung"
    }
    recovery_rungs = {
        int(value["rung_index"])
        for value in records
        if value["record_kind"] == "rung_origin_recovery"
    }
    selected_rest_rungs = {
        int(value["rung_index"]) for value in records if value["record_kind"] == "recovery_rest"
    }
    ordinary_rest_slots = {
        (int(value["rung_index"]), int(value["slot"])): int(value["owner"])
        for value in records
        if value["record_kind"] == "rest_selection" and int(value["owner"]) == 0
    }
    recovery_rest_owners = {
        int(value["rung_index"]): int(value["owner"])
        for value in records
        if value["record_kind"] == "rest_selection" and int(value["owner"]) in (1, 2)
    }
    observation_slots = {
        (int(value["rung_index"]), int(value["slot"]))
        for value in records
        if value["record_kind"] == "observation"
    }
    current_slots = {
        (int(value["rung_index"]), int(value["slot"]))
        for value in records
        if value["record_kind"] == "velocity_primitive_current"
    }
    rung_indexes = sorted(reported_rungs | {rung for rung, _slot in observation_slots})
    slot_count = int(replay.get("slot_count", 4))
    for rung_index in rung_indexes:
        for slot in range(slot_count):
            if (rung_index, slot) not in observation_slots:
                continue
            if (rung_index, slot) in ordinary_rest_slots:
                expected.append(
                    ("rest_selection", ordinary_rest_slots[(rung_index, slot)], rung_index, slot)
                )
            expected.append(("observation", rung_index, slot))
            if (rung_index, slot) in boundary_keys:
                expected.append(("structured_boundary", rung_index, slot))
            if (rung_index, slot) in current_slots:
                expected.append(("velocity_primitive_current", rung_index, slot))
        expected.extend(directional_at_boundary.get(rung_index, ()))
        expected.extend(joint_at_boundary.get(rung_index, ()))
        if rung_index in reported_rungs:
            expected.append(("rung", rung_index))
        if rung_index in selected_rest_rungs:
            if rung_index in recovery_rest_owners:
                expected.append(
                    ("rest_selection", recovery_rest_owners[rung_index], rung_index, 0xFF)
                )
            expected.append(("recovery_rest", rung_index))
        if rung_index in recovery_rungs:
            expected.append(("rung_origin_recovery", rung_index))
        expected.extend(directional_after_rung.get(rung_index, ()))
        expected.extend(joint_after_rung.get(rung_index, ()))
    expected.extend(directional_at_end)
    expected.extend(joint_at_end)
    if _stage_b_replay_expects_handoff(replay):
        expected.append(("handoff",))
    if any(
        value["record_kind"] in ("reproduction", "reproduction_v3", "reproduction_v4")
        for value in records
    ):
        expected.append(("reproduction",))
    for value in records:
        token = _stage_b_record_identity(value)
        if value["record_kind"] == "rest_selection" and token not in expected:
            expected.append(token)
    expected.append(("stage_b_terminal",))

    observed = [_stage_b_record_identity(value) for value in records]
    if observed == expected:
        return ""
    return f"Stage B causal order differs: firmware={observed} offline={expected}"


def _stage_b_fault_prefix_replay(replay: dict[str, object]) -> dict[str, object]:
    """Remove folds that a historical hard-current Fault never emitted."""
    prefix = dict(replay)
    prefix["directions"] = [
        [region for region in regions if region["closure"] != "end_of_ladder"]
        for regions in replay["directions"]
    ]
    prefix["joint_regions"] = [
        region for region in replay["joint_regions"] if region["closure"] != "end_of_ladder"
    ]
    prefix["nomination"] = {}
    prefix["nominated_p"] = 0
    prefix["selected"] = [{}, {}]
    prefix["sufficient"] = False
    prefix["coverage"] = [None, None]
    return prefix


def _project_schema6_current_headroom(
    replay: dict[str, object], *, terminal_rung: int
) -> dict[str, object]:
    """Project the schema-6 fold-before-terminal rule over preserved evidence."""
    boundary_consensus = []
    for directional_rungs in replay["directional_rungs"]:
        matching = [
            item for item in directional_rungs if int(item.get("rung_index", -1)) == terminal_rung
        ]
        source = matching[0] if matching else directional_rungs[-1]
        boundary_consensus.append(
            {
                "class": int(source["class"]),
                "included_mask": int(source["included_mask"]),
            }
        )
    exclusions = []
    for key, boundary in sorted(dict(replay["structured_boundaries"]).items()):
        rung_index, slot = (int(part) for part in str(key).split("/"))
        if rung_index == terminal_rung:
            exclusions.append(
                {
                    "rung_index": rung_index,
                    "slot": slot,
                    "passing_suffix_mask": int(boundary["passing_suffix_mask"]),
                }
            )
    selected = list(replay["selected"])
    nomination = dict(replay["nomination"])
    return {
        "interpretation": "schema-6 current-headroom what-if",
        "target_schema_revision": 6,
        "firmware_authoritative": False,
        "terminal_rung": terminal_rung,
        "expected_recovery_rungs": list(range(terminal_rung)),
        "directional_masks": [int(item["member_mask"]) for item in selected],
        "joint_mask": int(nomination["member_mask"]),
        "nominated_p": int(replay["nominated_p"]),
        "directional_common_intervals": [list(item["common_interval"]) for item in selected],
        "boundary_rung_consensus": boundary_consensus,
        "triggering_observation_exclusions": exclusions,
    }


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


def _stage_b_candidates_from_reproduction(
    reproduction: dict[str, object],
    plan: dict[str, object],
    provisional_names: tuple[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    memberships = reproduction["memberships"]
    pooled = reproduction["pooled"]
    common = reproduction["common"]
    base = {
        "p_start": int(plan["p_start"]),
        "p_top": int(plan["p_top"]),
        "rung_count": int(plan["rung_count"]),
    }

    def candidate(side: str, provisional_name: str) -> dict[str, object]:
        return {
            **base,
            "nominated_p": int(reproduction[provisional_name]),
            "memberships": [int(value[side]) for value in memberships],
            "pooled_intervals": [
                [int(value[f"{side}_low"]), int(value[f"{side}_high"])] for value in pooled
            ],
            "common_intervals": [
                [int(value[f"{side}_low"]), int(value[f"{side}_high"])] for value in common
            ],
        }

    return (
        candidate("previous", provisional_names[0]),
        candidate("current", provisional_names[1]),
    )


def _stage_b_candidates_from_v3(
    reproduction: dict[str, object], plan: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    return _stage_b_candidates_from_reproduction(
        reproduction,
        plan,
        ("previous_nominated_p", "current_nominated_p"),
    )


def _stage_b_candidates_from_v4(
    reproduction: dict[str, object], plan: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Reconstruct both schema-8 retained summaries from canonical fragments."""
    return _stage_b_candidates_from_reproduction(
        reproduction,
        plan,
        ("previous_provisional_p", "current_provisional_p"),
    )


def _stage_b_per_rung_intervals(report: dict[str, object]) -> list[dict[int, list[int]]] | None:
    replay = report.get("stage_b_replay")
    if not isinstance(replay, dict):
        return None
    directional = replay.get("directional_rungs")
    if not isinstance(directional, list) or len(directional) != 2:
        return None
    result: list[dict[int, list[int]]] = []
    for rungs in directional:
        result.append(
            {
                int(rung["rung_index"]): [int(rung["low"]), int(rung["high"])]
                for rung in rungs
                if int(rung.get("class", 1)) == 0
            }
        )
    return result


def _stage_b_candidate_from_replay(report: dict[str, object]) -> dict[str, object] | None:
    replay = report.get("stage_b_replay")
    plan = report.get("plan")
    if not isinstance(replay, dict) or not isinstance(plan, dict):
        return None
    selected = replay.get("selected")
    nomination = replay.get("nomination")
    if (
        not isinstance(selected, list)
        or len(selected) != 2
        or not all(isinstance(value, dict) for value in selected)
        or not isinstance(nomination, dict)
    ):
        return None
    return {
        "memberships": [
            int(selected[0]["member_mask"]),
            int(selected[1]["member_mask"]),
            int(nomination["member_mask"]),
        ],
        "nominated_p": int(replay["nominated_p"]),
        "pooled_intervals": [list(value["pooled_interval"]) for value in selected],
        "common_intervals": [list(value["common_interval"]) for value in selected],
        "p_start": int(plan["p_start"]),
        "p_top": int(plan["p_top"]),
        "rung_count": int(plan["rung_count"]),
    }


def _compare_stage_b_reproduction_fragments(
    reproduction: dict[str, object],
    projection: dict[str, object],
    schema_label: str,
) -> list[str]:
    mismatches = []
    for reported, expected in zip(
        reproduction["memberships"], projection["memberships"], strict=True
    ):
        for name in (
            "previous",
            "current",
            "core",
            "previous_only",
            "current_only",
            "low_delta",
            "high_delta",
        ):
            if int(reported[name]) != int(expected[name]):
                mismatches.append(f"{schema_label} membership {name} differs from projection")
    for reported, expected in zip(reproduction["pooled"], projection["pooled"], strict=True):
        if [int(reported["overlap_low"]), int(reported["overlap_high"])] != expected["overlap"]:
            mismatches.append(f"{schema_label} pooled overlap differs from projection")
    for reported, expected in zip(reproduction["common"], projection["common"], strict=True):
        if [int(reported["conservative_low"]), int(reported["conservative_high"])] != expected[
            "conservative"
        ] or bool(reported["nonempty"]) != bool(expected["nonempty"]):
            mismatches.append(f"{schema_label} conservative common differs from projection")
    for reported, expected in zip(reproduction["coverage"], projection["coverage"], strict=True):
        expected_kind = 0 if expected["coverage"] == "full" else 1
        if (
            int(reported["coverage"]) != expected_kind
            or int(reported["signed_rung_distance"]) != int(expected["signed_rung_distance"])
            or [int(reported["gain_ratio_num"]), int(reported["gain_ratio_den"])]
            != expected["gain_ratio"]
        ):
            mismatches.append(f"{schema_label} coverage differs from projection")
    return mismatches


def _compare_stage_b_v3_record(
    reproduction: dict[str, object], projection: dict[str, object]
) -> list[str]:
    mismatches = _compare_stage_b_reproduction_fragments(reproduction, projection, "schema-3")
    expected_outcome = 1 if projection["outcome"] == "complete" else 2
    if int(reproduction["outcome"]) != expected_outcome:
        mismatches.append("schema-3 reproduction outcome differs from core projection")
    for name in ("reason_mask", "fixed_p"):
        if int(reproduction[name]) != int(projection[name]):
            mismatches.append(f"schema-3 reproduction {name} differs from core projection")
    if bool(reproduction["reduced_margin"]) != bool(projection["reduced_margin"]):
        mismatches.append("schema-3 reproduction reduced margin differs from core projection")
    return mismatches


def _compare_stage_b_v4_record(
    reproduction: dict[str, object], projection: dict[str, object]
) -> list[str]:
    """Compare one schema-8 canonical group with the independent projection."""
    mismatches = _compare_stage_b_reproduction_fragments(reproduction, projection, "schema-8")
    expected_outcome = 1 if projection["outcome"] == "complete" else 2
    scalar_fields = (
        "reason_mask",
        "fixed_p",
        "selected_joint_mask",
        "selected_first_rung",
        "selected_member_count",
    )
    if int(reproduction["outcome"]) != expected_outcome:
        mismatches.append("schema-8 reproduction outcome differs from component projection")
    for name in scalar_fields:
        if int(reproduction[name]) != int(projection[name]):
            mismatches.append(f"schema-8 reproduction {name} differs from projection")
    reported_validity = [
        int(reproduction["forward_validity"]),
        int(reproduction["reverse_validity"]),
    ]
    if reported_validity != list(projection["directional_validity"]):
        mismatches.append("schema-8 directional validity differs from projection")
    if bool(reproduction["reduced_margin"]) != bool(projection["reduced_margin"]):
        mismatches.append("schema-8 reduced margin differs from projection")
    return mismatches


def audit_stage_b_reproduction_pair(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Verify a current reproduction record against its prior capture."""
    mismatches: list[str] = []
    reproduction = current.get("reproduction")
    if not isinstance(reproduction, dict):
        return {"matches_firmware": False, "mismatches": ["current capture lacks reproduction"]}
    schema_revision = int(reproduction.get("schema_revision", 2))
    if schema_revision == 8:
        plan = current.get("plan")
        if not isinstance(plan, dict):
            return {"matches_firmware": False, "mismatches": ["current capture lacks plan"]}
        prior, active = _stage_b_candidates_from_v4(reproduction, plan)
        projection = _project_stage_b_schema8_candidates(prior, active)
        mismatches.extend(_compare_stage_b_v4_record(reproduction, projection))
        previous_digest = int(reproduction["previous_digest_low"]) | (
            int(reproduction["previous_digest_high"]) << 32
        )
        current_digest = int(reproduction["current_digest_low"]) | (
            int(reproduction["current_digest_high"]) << 32
        )
        if previous_digest != int(previous["reported_digest"]):
            mismatches.append("schema-8 previous digest differs from prior terminal")
        if current_digest != int(current["pre_reproduction_digest"]):
            mismatches.append("schema-8 current digest differs from current prefix")
        return {
            "matches_firmware": not mismatches,
            "mismatches": mismatches,
            "projection": projection,
        }
    if 3 <= schema_revision <= 7:
        plan = current.get("plan")
        if not isinstance(plan, dict):
            return {"matches_firmware": False, "mismatches": ["current capture lacks plan"]}
        prior, active = _stage_b_candidates_from_v3(reproduction, plan)
        replay_previous = _stage_b_candidate_from_replay(previous)
        replay_current = _stage_b_candidate_from_replay(current)
        if replay_previous is not None and replay_previous != prior:
            mismatches.append("schema-3 retained summary differs from prior capture replay")
        if replay_current is not None and replay_current != active:
            mismatches.append("schema-3 current summary differs from current capture replay")
        previous_rungs = _stage_b_per_rung_intervals(previous)
        current_rungs = _stage_b_per_rung_intervals(current)
        if previous_rungs is not None and current_rungs is not None:
            prior["per_rung_common_intervals"] = previous_rungs
            active["per_rung_common_intervals"] = current_rungs
        projection = project_stage_b_core_pair(prior, active)
        mismatches.extend(_compare_stage_b_v3_record(reproduction, projection))
        previous_digest = int(reproduction["previous_digest_low"]) | (
            int(reproduction["previous_digest_high"]) << 32
        )
        current_digest = int(reproduction["current_digest_low"]) | (
            int(reproduction["current_digest_high"]) << 32
        )
        if previous_digest != int(previous["reported_digest"]):
            mismatches.append("reproduction previous digest differs from prior terminal")
        if current_digest != int(current["pre_reproduction_digest"]):
            mismatches.append("reproduction current digest differs from current prefix")
        return {
            "matches_firmware": not mismatches,
            "mismatches": mismatches,
            "projection": projection,
        }
    if schema_revision != 2:
        return {"matches_firmware": False, "mismatches": ["unsupported schema"]}
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
    comparison = _compare_stage_b_candidates_schema2(previous_candidate, current_candidate)
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


def _stage_b_terminal_boundary(
    *,
    terminal: dict[str, object],
    current_records: list[dict[str, object]],
    completed_rungs: int,
) -> tuple[str, int | None, list[int]]:
    """Recover the terminal fold boundary from primitive-wide current evidence."""
    if int(terminal.get("cause", 0)) != STAGE_B_CAUSE_CURRENT_HEADROOM and not bool(
        terminal.get("recovery_unavailable", 0)
    ):
        return "end_of_ladder", None, []

    emitted_rungs = int(terminal.get("emitted_rungs", 0))
    boundary_rung = emitted_rungs - 1 if emitted_rungs else None
    partial_rungs = sorted(
        {
            int(record["rung_index"])
            for record in current_records
            if int(record["rung_index"]) >= completed_rungs
            and int(record.get("contact", 0)) in (1, 2)
        }
    )
    return "current_headroom", boundary_rung, partial_rungs


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
    recovery_rest_present = any(
        value["record_kind"] == "recovery_rest" for _raw, value in evidence_records
    )
    recovery_wide = (
        int(plan.get("nominal_workflow_ms", -1)),
        int(plan.get("maximum_workflow_ms", -1)),
    ) == (282_923, 298_632)
    combined_response_selected = recovery_rest_present or (
        int(terminal.get("outcome", STAGE_B_OUTCOME_INCONCLUSIVE)) == STAGE_B_OUTCOME_COMPLETE
        and any(
            value["record_kind"] == "directional_region" and int(value["closure"]) == 5
            for _raw, value in evidence_records
        )
        and any(
            value["record_kind"] == "joint_region" and int(value["closure"]) == 4
            for _raw, value in evidence_records
        )
    )
    # Stage B's wire "plan" record carries no schema revision and firmware
    # publishes STAGE_B_EVIDENCE_SCHEMA_REVISION only in the Klipper constant
    # block, which trace captures do not contain -- so the revision has to be
    # derived from what the evidence stream itself shows, one wire marker per
    # rung. Each rung below only proves a floor: "this capture is schema N or
    # newer", never an exact match, because an absent marker is also what an
    # older, unrelated capture looks like. Ordering the rungs highest-first
    # keeps that sound -- a lower rung only fires when every higher marker was
    # absent, which bounds it from above by the next rung's threshold.
    #
    # The top rung has no such ceiling. The trailing drift_ratio_ppm u32 on the
    # Stage B observation record was added by the same change that introduced
    # the drift bound, one commit before COMBINED_STAGE_B_SCHEMA_REVISION went
    # to 14, so a capture reporting it is schema 14 or newer -- and today,
    # "or newer" cannot happen, so the value is 14 in practice. Older captures
    # decode the field as None (see protocol._decode_velocity_sweep_evidence)
    # and keep the ladder below. Without this arm the derivation topped out at
    # 12 and every real schema-14 capture was replayed against the legacy
    # significance gate while firmware had applied the drift bound.
    #
    # `schema_revision_floor` is named for exactly this: it is the highest
    # schema revision this ladder can currently prove a lower bound for, not a
    # confirmed exact revision. A future Stage B schema bump that changes
    # gating behavior (the way schema 14 did) needs its own new wire marker
    # added as a rung above this one. Simply bumping
    # STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION to 15 without adding that marker
    # would not teach this function anything: schema-15 captures have no
    # marker of their own to be recognized by, so they would keep falling into
    # this same arm and replay against the schema-14 gate -- the identical
    # failure mode this arm exists to fix, recurring one revision later.
    drift_ratio_reported = any(
        value["record_kind"] == "observation" and value.get("drift_ratio_ppm") is not None
        for _raw, value in evidence_records
    )
    schema_revision_floor = (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION
        if drift_ratio_reported
        else 12
        if recovery_wide
        else 11
        if recovery_rest_present
        else 9
        if combined_response_selected
        else 8
        if any(
            value["record_kind"] == "reproduction_v4"
            or value["record_kind"] == "handoff"
            and "joint_union_mask" in value
            for _raw, value in evidence_records
        )
        or int(terminal.get("outcome", STAGE_B_OUTCOME_INCONCLUSIVE))
        in (STAGE_B_OUTCOME_COMPLETE_CANDIDATE, STAGE_B_OUTCOME_COMPLETE)
        and int(terminal.get("model_direction_mask", 3)) != 3
        else 7
    )
    by_stroke: dict[int, list[VelocityObservation]] = {}
    for observation in raw_observations:
        by_stroke.setdefault(observation.stroke_index, []).append(observation)
    records = {
        (int(value["rung_index"]), int(value["slot"])): value
        for _raw, value in evidence_records
        if value["record_kind"] == "observation"
    }
    current_by_slot = {
        (int(value["rung_index"]), int(value["slot"])): value
        for _raw, value in evidence_records
        if value["record_kind"] == "velocity_primitive_current"
    }
    observations_per_direction = int(plan["observations_per_direction"])
    slot_count = 2 * observations_per_direction
    if observations_per_direction not in (2, 4):
        raise ValueError("Stage B replay supports two or four observations per direction")
    reconstructed: dict[tuple[int, int], dict[str, int]] = {}
    for key, record in records.items():
        raw = by_stroke.get(key[0] * slot_count + key[1])
        if not raw:
            continue
        reconstructed[key] = _reconstruct_observation(
            raw,
            record,
            plan,
            terminal,
            current_by_slot.get(key),
            schema_revision=schema_revision_floor,
        )
    directional_inputs: list[list[dict[str, object]]] = [[], []]
    joint_inputs = []
    gains = []
    for rung_index in range(int(plan["rung_count"])):
        if any((rung_index, slot) not in reconstructed for slot in range(slot_count)):
            break
        p = int(records[(rung_index, 0)]["velocity_p"])
        gains.append(p)
        operable = []
        for direction_index in range(2):
            slots = tuple(
                direction_index + 2 * index for index in range(observations_per_direction)
            )
            observations = [reconstructed[(rung_index, slot)] for slot in slots]
            if observations_per_direction == 2:
                classification, low, high, _moving = _rung_class(*observations)
                means = [item.get("settled_rate_mean_q", 0) for item in observations]
                analysis: dict[str, object] = {
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
                direction_operable = all(item["classification"] in (0, 2) for item in observations)
            else:
                initial_lead_in = any(
                    int(records[(rung_index, slot)].get("flags", 0)) & 1 for slot in slots
                )
                analysis = _analyze_four_observation_direction(
                    observations,
                    direction=1 if direction_index == 0 else -1,
                    p=p,
                    counts_per_rev=int(terminal["encoder_counts_per_rev"]),
                    initial_lead_in=initial_lead_in,
                )
                direction_operable = bool(int(analysis["directional_flags"]) & 1)
            directional_inputs[direction_index].append(
                {
                    "rung_index": rung_index,
                    "velocity_p": p,
                    **analysis,
                }
            )
            operable.append(direction_operable)
        joint_inputs.append(
            {
                "rung_index": rung_index,
                "forward_operable": operable[0],
                "reverse_operable": operable[1],
                "forward_class": int(directional_inputs[0][-1]["class"]),
                "reverse_class": int(directional_inputs[1][-1]["class"]),
            }
        )
    completed_rungs = len(joint_inputs)
    observation_current_rungs = {
        rung_index
        for (rung_index, _slot), observation in reconstructed.items()
        if rung_index >= completed_rungs and int(observation["classification"]) == 3
    }
    current_records = [
        value
        for _raw, value in evidence_records
        if value["record_kind"] == "velocity_primitive_current"
    ]
    terminal_closure, terminal_boundary_rung, observed_current_rungs = _stage_b_terminal_boundary(
        terminal=terminal,
        current_records=current_records,
        completed_rungs=completed_rungs,
    )
    if combined_response_selected and terminal_closure == "end_of_ladder":
        terminal_closure = "response_selected"
    partial_current_rungs = sorted(
        {
            *observation_current_rungs,
            *observed_current_rungs,
        }
    )
    if partial_current_rungs and terminal_boundary_rung is None:
        terminal_closure = "current_headroom"
        terminal_boundary_rung = partial_current_rungs[0]
    partial_directional_rungs: dict[int, list[dict[str, object]]] = {}
    for rung_index in partial_current_rungs if observations_per_direction == 4 else ():
        first_record = next(
            record for (record_rung, _slot), record in records.items() if record_rung == rung_index
        )
        p = int(first_record["velocity_p"])
        analyses = []
        for direction_index in range(2):
            slots = tuple(
                direction_index + 2 * index for index in range(observations_per_direction)
            )
            observations = [reconstructed.get((rung_index, slot)) for slot in slots]
            initial_lead_in = any(
                int(records[(rung_index, slot)].get("flags", 0)) & 1
                for slot in slots
                if (rung_index, slot) in records
            )
            analyses.append(
                _analyze_four_observation_direction(
                    observations,
                    direction=1 if direction_index == 0 else -1,
                    p=p,
                    counts_per_rev=int(terminal["encoder_counts_per_rev"]),
                    initial_lead_in=initial_lead_in,
                )
            )
        partial_directional_rungs[rung_index] = analyses
    directional = [
        _segment_directional_regions(
            items,
            terminal_closure=terminal_closure,
            terminal_boundary_rung=terminal_boundary_rung,
        )
        for items in directional_inputs
    ]
    joint = _joint_operable_regions(
        joint_inputs,
        terminal_closure=terminal_closure,
        terminal_boundary_rung=terminal_boundary_rung,
    )
    nomination = _nominate_joint_region(
        joint,
        gains,
        int(plan.get("p_start", gains[0] if gains else 1)),
        int(plan.get("p_top", gains[-1] if gains else 1)),
    )
    nominated_rung = int(nomination["rung_index"]) if nomination else 0
    selected = []
    for regions in directional:
        candidates = (
            [_select_schema8_directional_evidence(regions, provisional_rung=nominated_rung)]
            if schema_revision_floor >= 8
            else [region for region in regions if region["kind"] == "valid"]
        )
        selected.append(
            _select_directional_region(
                [region for region in candidates if region],
                nominated_rung,
                gains,
            )
        )
    sufficient = _stage_b_handoff_sufficient(
        directional,
        nomination,
        selected,
        schema_revision=schema_revision_floor,
    )
    structured_boundaries = {}
    for key, observation in reconstructed.items():
        if int(observation["classification"]) not in _STAGE_B_EXCLUSION_CODES:
            continue
        boundary = analyze_structured_boundary(by_stroke.get(key[0] * slot_count + key[1], []))
        if boundary is not None:
            structured_boundaries[f"{key[0]}/{key[1]}"] = boundary
    combined_rungs = []
    for forward, reverse in zip(directional_inputs[0], directional_inputs[1], strict=True):
        forward_wire = forward.get("wire_rate_bounds_q")
        reverse_wire = reverse.get("wire_rate_bounds_q")
        forward_interval = (
            list(forward_wire[2:4])  # type: ignore[index]
            if forward_wire is not None
            else [min(forward["rate_lows"]), max(forward["rate_highs"])]  # type: ignore[arg-type]
        )
        reverse_interval = (
            list(reverse_wire[2:4])  # type: ignore[index]
            if reverse_wire is not None
            else [min(reverse["rate_lows"]), max(reverse["rate_highs"])]  # type: ignore[arg-type]
        )
        combined_rungs.append(
            {
                "p": int(forward["velocity_p"]),
                "forward": forward_interval,
                "reverse": reverse_interval,
                "eligible": [
                    int(forward["class"]) == 0
                    and int(forward.get("collected_mask", 0b1111)) == 0b1111,
                    int(reverse["class"]) == 0
                    and int(reverse.get("collected_mask", 0b1111)) == 0b1111,
                ],
            }
        )
    combined_projection = project_combined_response_selection(
        combined_rungs,
        target_rpm=abs(int(terminal.get("target_velocity_rpm", plan["target_velocity_rpm"]))),
        encoder_counts_per_rev=int(terminal["encoder_counts_per_rev"]),
    )
    return {
        "directions": directional,
        "directional_rungs": directional_inputs,
        "partial_directional_rungs": partial_directional_rungs,
        "joint_regions": joint,
        "nomination": nomination,
        "nominated_p": int(nomination.get("p", 0)),
        "selected": selected,
        "sufficient": sufficient,
        "coverage": [item.get("coverage") for item in selected],
        "gains": gains,
        "structured_boundaries": structured_boundaries,
        "slot_count": slot_count,
        # Key name is unchanged for existing callers, but the value is a
        # floor (see schema_revision_floor's derivation above), not a
        # confirmed exact schema revision.
        "schema_revision": schema_revision_floor,
        "combined_response_selected": combined_response_selected,
        "combined_selection_projection": combined_projection,
    }


def _candidate_terminal_memberships(replay: dict[str, object], schema_revision: int) -> list[int]:
    directional = [int(item["member_mask"]) for item in replay["selected"]]
    if schema_revision >= 8:
        joint = 0
        for region in replay["joint_regions"]:
            joint |= int(region["member_mask"])
    else:
        joint = int(replay["nomination"]["member_mask"])
    return [*directional, joint]


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
    observations_per_direction = int(plan["observations_per_direction"])
    slot_count = 2 * observations_per_direction
    nonterminal = [
        (raw, value)
        for raw, value in evidence_records
        if value["record_kind"] != "stage_b_terminal"
    ]
    try:
        reproduction_group = _assemble_stage_b_reproduction([value for _raw, value in nonterminal])
    except ValueError as error:
        reproduction_group = None
        reproduction_group_error = str(error)
    else:
        reproduction_group_error = ""
    logical_nonterminal: list[tuple[bytes, dict[str, object]]] = []
    inserted_reproduction = False
    for raw, value in nonterminal:
        if value["record_kind"] in _STAGE_B_REPRODUCTION_FRAGMENT_KINDS:
            if reproduction_group is not None and not inserted_reproduction:
                logical_nonterminal.append((b"", reproduction_group))
                inserted_reproduction = True
            continue
        logical_nonterminal.append((raw, value))
    digest = FNV1A64_OFFSET
    pre_reproduction_digest: int | None = None
    for raw, _value in nonterminal:
        if _value["record_kind"] in (
            "reproduction",
            "reproduction_v3_header",
            "reproduction_v4_header",
        ):
            pre_reproduction_digest = digest
        for byte in raw:
            digest ^= byte
            digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    reported_digest = int(terminal["digest_low"]) | (int(terminal["digest_high"]) << 32)
    mismatches: list[str] = []
    plan_digest = int(plan["plan_digest_low"]) | (int(plan["plan_digest_high"]) << 32)
    # `replay_stage_b_regions` is the only place `schema_revision_floor` and
    # `combined_response_selected` are derived; the grammar audit needs both,
    # so the replay runs here rather than at its original, later call site,
    # and that later site reuses this result instead of calling twice.
    projected_replay = replay_stage_b_regions(raw_observations, evidence_records)
    schema_revision_floor = int(projected_replay["schema_revision"])
    combined_response_selected = bool(projected_replay["combined_response_selected"])
    mismatches.extend(
        audit_scored_rest_grammar(
            [value for _raw, value in nonterminal],
            terminal,
            schema_revision=schema_revision_floor,
            schema_floor=13,
            expected_stage=0,
            ordinary_owners=(0,),
            recovery_owners=(1, 2),
            observation_kind="observation",
            plan_digest=plan_digest,
            label="Stage B",
            stabilized_recovery_rest=(schema_revision_floor == 14 or combined_response_selected),
            ordinary_rejection_tail=(),
        )
    )
    recovery_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "rung_origin_recovery"
    ]
    selected_rest_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "recovery_rest"
    ]
    current_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "velocity_primitive_current"
    ]
    for record in current_records:
        if int(record["stage"]) != 0:
            mismatches.append("Stage B current evidence carries the wrong stage")
        if int(record["clamp_readback"]) != int(record["clamp_limit"]):
            mismatches.append("Stage B current evidence clamp readback differs")
    if current_records and int(terminal["outcome"]) != STAGE_B_OUTCOME_FAULT:
        observation_keys = {
            (int(value["rung_index"]), int(value["slot"]))
            for _raw, value in nonterminal
            if value["record_kind"] == "observation"
        }
        current_keys = {(int(value["rung_index"]), int(value["slot"])) for value in current_records}
        if current_keys != observation_keys:
            mismatches.append("Stage B current evidence does not cover every observation")
    recovery_projections: list[dict[str, object]] = []
    recovery_wide = (
        int(plan.get("nominal_workflow_ms", -1)),
        int(plan.get("maximum_workflow_ms", -1)),
    ) == (282_923, 298_632)
    recovery_schema = "recovery_origin_band_counts" in plan
    if recovery_schema:
        try:
            recovery_plan = plan_rung_origin_recovery(
                max_stroke_travel_mrev=int(plan["max_stroke_travel_mrev"]),
                settle_travel_reserve_mrev=int(plan["settle_travel_reserve_mrev"]),
                encoder_counts_per_rev=int(terminal["encoder_counts_per_rev"]),
                planned_velocity_mrev_s=int(plan["planned_velocity_mrev_s"]),
                primitive_cap_us=int(plan["recovery_maximum_slot_us"]),
            )
        except ValueError as error:
            mismatches.append(f"Stage B recovery plan reconstruction failed: {error}")
        else:
            expected_nominal_slot_us = int(recovery_plan["nominal_slot_us"])
            if recovery_wide:
                expected_nominal_slot_us += RECOVERY_STABILIZATION_US
            for name, expected in (
                ("recovery_origin_band_counts", recovery_plan["origin_band_counts"]),
                ("recovery_nominal_slot_us", expected_nominal_slot_us),
                ("recovery_maximum_slot_us", recovery_plan["maximum_slot_us"]),
                ("recovery_slot_count", int(plan["rung_count"])),
            ):
                if int(plan[name]) != expected:
                    mismatches.append(f"Stage B {name}: firmware={plan[name]} offline={expected}")
        recovery_rungs = [int(record["rung_index"]) for record in recovery_records]
        emitted_recovery_rungs = sorted(
            {
                int(value["rung_index"])
                for _raw, value in nonterminal
                if value["record_kind"] == "rung"
            }
        )
        missing_final_recovery = (
            bool(emitted_recovery_rungs) and recovery_rungs == emitted_recovery_rungs[:-1]
        )
        terminal_rung = emitted_recovery_rungs[-1] if emitted_recovery_rungs else None
        terminal_follows_rung = bool(nonterminal) and (
            nonterminal[-1][1]["record_kind"] == "rung"
            and int(nonterminal[-1][1]["rung_index"]) == terminal_rung
        )
        fault_prefix = (
            missing_final_recovery
            and int(terminal["outcome"]) == STAGE_B_OUTCOME_FAULT
            and terminal_follows_rung
        )
        historical_current_headroom_fault = fault_prefix and int(terminal["cause"]) == 17
        missing_final_allowed = missing_final_recovery and (
            int(terminal.get("recovery_unavailable", 0)) == 1 or fault_prefix
        )
        expected_recovery_rungs = (
            recovery_rungs if missing_final_allowed else emitted_recovery_rungs
        )
        if recovery_rungs != expected_recovery_rungs:
            mismatches.append(
                "Stage B recovery rungs differ from fully acquired rung sequence: "
                f"firmware={recovery_rungs} offline={expected_recovery_rungs}"
            )
        for record in recovery_records:
            if int(record["stage"]) != 0:
                mismatches.append("Stage B recovery record carries the wrong stage")
            recovery_audit = audit_rung_origin_recovery(record, plan)
            mismatches.extend(
                f"recovery rung {record['rung_index']}: {item}"
                for item in recovery_audit["mismatches"]
            )
    elif recovery_records:
        mismatches.append("historical Stage B plan carries unexpected recovery records")
    diagnostic_rungs = [int(record["rung_index"]) for record in selected_rest_records]
    if len(diagnostic_rungs) != len(set(diagnostic_rungs)):
        mismatches.append("Stage B carries duplicate recovery rest diagnostics")
    if not recovery_wide and len(selected_rest_records) > 1:
        mismatches.append("Stage B carries more than one selected recovery rest diagnostic")
    if recovery_wide:
        diagnostic_rung_set = set(diagnostic_rungs)
        for recovery in recovery_records:
            if (
                int(recovery["outcome"]) in (1, 5)
                and int(recovery["rung_index"]) not in diagnostic_rung_set
            ):
                mismatches.append("moving Stage B recovery lacks recovery rest evidence")
    selected_rest_audit = None
    selected_scope = None
    recovery_rest_audits = [
        audit_selected_recovery_rest(record, plan, terminal, recovery_records)
        for record in selected_rest_records
    ]
    for audit in recovery_rest_audits:
        mismatches.extend(audit["mismatches"])
    if selected_rest_records and not recovery_wide:
        selected_rest_audit = audit_selected_recovery_rest(
            selected_rest_records[0], plan, terminal, recovery_records
        )
        selected_scope = _selected_recovery_scope_comparison(
            selected_rest_records[0], recovery_records, plan
        )
    if reproduction_group_error:
        mismatches.append(reproduction_group_error)
    if digest != reported_digest:
        mismatches.append(
            f"terminal digest: firmware=0x{reported_digest:016x} offline=0x{digest:016x}"
        )
    main_records = [
        value
        for _raw, value in logical_nonterminal
        if value["record_kind"] != "structured_boundary"
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

    replay = (
        _stage_b_fault_prefix_replay(projected_replay)
        if recovery_schema and fault_prefix
        else projected_replay
    )
    causal_mismatch = _compare_stage_b_causal_order(
        [value for _raw, value in logical_nonterminal] + [terminal], replay
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
    current_by_slot = {
        (int(value["rung_index"]), int(value["slot"])): value for value in current_records
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
        raw = by_stroke.get(key[0] * slot_count + key[1], [])
        if not raw:
            mismatches.append(f"observation {key[0]}/{key[1]} has no raw samples")
            continue
        offline = _reconstruct_observation(
            raw,
            record,
            plan,
            terminal,
            current_by_slot.get(key),
            schema_revision=int(replay["schema_revision"]),
        )
        for name in exact_fields:
            offline.setdefault(name, 0)
        _compare_fields(mismatches, f"observation {key[0]}/{key[1]}", record, offline, exact_fields)
        _compare_optional_drift_ratio(mismatches, f"observation {key[0]}/{key[1]}", record, offline)
        reconstructed[key] = offline

    boundary_records = [
        value for _raw, value in evidence_records if value["record_kind"] == "structured_boundary"
    ]
    reconstructed_boundaries: dict[tuple[int, int], dict[str, object]] = {}
    for key in observation_records:
        raw = by_stroke.get(key[0] * slot_count + key[1], [])
        boundary = analyze_structured_boundary(raw)
        if boundary is not None:
            reconstructed_boundaries[key] = boundary
    expected_boundaries = dict(reconstructed_boundaries)
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

    if recovery_schema:
        recovery_projections = _project_stage_b_recovery_authorities(
            plan,
            terminal,
            list(replay["gains"]),
            reconstructed,
            reconstructed_boundaries,
            recovery_records,
        )
        reported_schema = (
            int(reproduction_group["schema_revision"]) if reproduction_group is not None else None
        )
        if reported_schema == 5:
            for projection in recovery_projections:
                rung_index = int(projection["rung_index"])
                if int(projection["recorded_lower_rate_q"]) != int(projection["lower_rate_q"]):
                    mismatches.append(
                        f"recovery rung {rung_index}: schema-5 selected rate differs "
                        "from reconstructed authority"
                    )
                return_distance = int(projection["return_distance_counts"])
                lower_rate = int(projection["lower_rate_q"])
                if return_distance == 0:
                    expected_outcome = (0, 0)
                elif lower_rate == 0:
                    expected_outcome = (2, 2)
                elif projection["deadline_us"] is None:
                    expected_outcome = (3, 3)
                else:
                    expected_outcome = (1, int(projection["binding_source"]))
                recorded_outcome = (
                    int(projection["recorded_outcome"]),
                    int(projection["recorded_binding_source"]),
                )
                if recorded_outcome != expected_outcome:
                    mismatches.append(
                        f"recovery rung {rung_index}: schema-5 outcome/source differs "
                        "from reconstructed authority"
                    )

    rung_records = {
        int(value["rung_index"]): value
        for _raw, value in evidence_records
        if value["record_kind"] == "rung"
    }
    for rung_index, rung in sorted(rung_records.items()):
        if observations_per_direction == 4:
            if all(rung_index < len(items) for items in replay["directional_rungs"]):
                analyses = [items[rung_index] for items in replay["directional_rungs"]]
            else:
                analyses = replay["partial_directional_rungs"].get(rung_index)
            if analyses is None:
                mismatches.append(f"rung {rung_index} is missing reconstructed observations")
                continue
            for prefix, analysis in zip(("forward", "reverse"), analyses, strict=True):
                expected = {
                    f"{prefix}_class": analysis["class"],
                    f"{prefix}_component_count": analysis["component_count"],
                    f"{prefix}_collected_mask": analysis["collected_mask"],
                    f"{prefix}_eligible_mask": analysis["eligible_mask"],
                    f"{prefix}_included_mask": analysis["included_mask"],
                    f"{prefix}_operable_count": analysis["operable_count"],
                    f"{prefix}_directional_flags": analysis["directional_flags"],
                    f"{prefix}_pooled_q16": analysis["pooled_q16"],
                    f"{prefix}_pooled_low_q16": analysis["pooled_interval"][0],
                    f"{prefix}_pooled_high_q16": analysis["pooled_interval"][1],
                    f"{prefix}_variance_floor_observations": analysis[
                        "variance_floor_observations"
                    ],
                }
                _compare_fields(
                    mismatches,
                    f"rung {rung_index} {prefix}",
                    rung,
                    expected,
                    tuple(expected),
                )
                if rung[f"{prefix}_components"] != analysis["components"]:
                    mismatches.append(
                        f"rung {rung_index} {prefix} components: "
                        f"firmware={rung[f'{prefix}_components']} "
                        f"offline={analysis['components']}"
                    )
                if rung[f"{prefix}_rate_bounds_q"] != analysis["wire_rate_bounds_q"]:
                    mismatches.append(
                        f"rung {rung_index} {prefix} rate bounds: "
                        f"firmware={rung[f'{prefix}_rate_bounds_q']} "
                        f"offline={analysis['wire_rate_bounds_q']}"
                    )
            expected_flags = (
                (int(analyses[0]["directional_flags"]) & 1)
                | ((int(analyses[1]["directional_flags"]) & 1) << 1)
                | ((int(analyses[0]["directional_flags"]) & 2) << 1)
                | ((int(analyses[1]["directional_flags"]) & 2) << 2)
            )
            if int(rung["flags"]) != expected_flags:
                mismatches.append(
                    f"rung {rung_index} flags: firmware={rung['flags']} offline={expected_flags}"
                )
            continue
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
        "response_selected": 5,
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
                    f"directional region {key} {prefix}: firmware={decoded} offline="
                    f"{region[source]}"
                )
    if expected_regions:
        mismatches.append(f"missing directional regions {sorted(expected_regions)}")

    joint_closure_codes = {
        "ineligible_gap": 0,
        "current_headroom": 1,
        "safety_terminus": 2,
        "end_of_ladder": 3,
        "response_selected": 4,
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
    expects_handoff = _stage_b_replay_expects_handoff(replay)
    if expects_handoff and len(handoffs) != 1:
        mismatches.append(f"expected one Stage B handoff, found {len(handoffs)}")
    elif not expects_handoff and handoffs:
        mismatches.append("Stage B handoff exists without complete directional model evidence")
    elif expects_handoff:
        handoff = handoffs[0]
        nomination = replay["nomination"]
        expected = {
            "nominated_p": replay["nominated_p"],
            "coverage_mask": sum(
                (1 << index)
                for index, item in enumerate(replay["selected"])
                if item["coverage"] == "full"
            ),
            "nominated_rung": nomination["rung_index"],
            "distance_to_start_q16": nomination["distance_to_start_q16"],
            "distance_to_top_q16": nomination["distance_to_top_q16"],
            "nomination_flags": int(nomination["reduced_margin"]),
        }
        if "joint_union_mask" in handoff:
            joint_union = 0
            for region in replay["joint_regions"]:
                joint_union |= int(region["member_mask"])
            expected.update(
                {
                    "model_direction_mask": sum(
                        (1 << index)
                        for index, regions in enumerate(replay["directions"])
                        if any(region["kind"] == "valid" for region in regions)
                    ),
                    "joint_union_mask": joint_union,
                    "joint_member_mask": nomination["member_mask"],
                }
            )
        else:
            expected["joint_member_mask"] = nomination["member_mask"]
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

    model_direction_mask = sum(
        (1 << index)
        for index, regions in enumerate(replay["directions"])
        if any(region["kind"] == "valid" for region in regions)
    )
    candidate_schema_revision = int(replay["schema_revision"])
    combined_response_selected = bool(replay["combined_response_selected"])
    if combined_response_selected:
        projection = replay["combined_selection_projection"]
        if projection["selected_p"] is None:
            mismatches.append("response-selected terminal has no independent selected response")
            expected_memberships = [0, 0, 0]
            expected_intervals = [[0, 0], [0, 0]]
            expected_nominated_p = 0
        else:
            expected_memberships = _candidate_terminal_memberships(
                replay,
                candidate_schema_revision,
            )
            expected_intervals = [list(interval) for interval in projection["selected_intervals"]]
            expected_nominated_p = int(projection["selected_p"])
        expected_coverage_mask = sum(
            (1 << index)
            for index, item in enumerate(replay["selected"])
            if item.get("coverage") == "full"
        )
    elif replay["sufficient"]:
        expected_memberships = _candidate_terminal_memberships(
            replay,
            candidate_schema_revision,
        )
        expected_intervals = [list(item["pooled_interval"]) for item in replay["selected"]]
        expected_nominated_p = int(replay["nominated_p"])
        expected_coverage_mask = sum(
            (1 << index)
            for index, item in enumerate(replay["selected"])
            if item.get("coverage") == "full"
        )
    else:
        expected_memberships = [0, 0, 0]
        expected_intervals = [[0, 0], [0, 0]]
        expected_nominated_p = 0
        expected_coverage_mask = 0
    reproduction_projection: dict[str, object] | None = None
    if reproduction_group is not None and reproduction_group["record_kind"] == "reproduction_v3":
        prior_v3, current_v3 = _stage_b_candidates_from_v3(reproduction_group, plan)
        v3_projection = project_stage_b_core_pair(prior_v3, current_v3)
        reproduction_projection = v3_projection
        if int(reproduction_group["outcome"]) == 1:
            expected_memberships = list(v3_projection["core_masks"])
            expected_intervals = [list(value["overlap"]) for value in v3_projection["pooled"]]
            expected_nominated_p = int(v3_projection["fixed_p"])
            expected_coverage_mask = sum(
                (1 << index)
                for index, value in enumerate(v3_projection["coverage"])
                if value["coverage"] == "full"
            )
    elif reproduction_group is not None and reproduction_group["record_kind"] == "reproduction_v4":
        prior_v4, current_v4 = _stage_b_candidates_from_v4(reproduction_group, plan)
        v4_projection = _project_stage_b_schema8_candidates(prior_v4, current_v4)
        reproduction_projection = v4_projection
        model_direction_mask = _stage_b_model_mask_from_validity(
            [int(value) for value in v4_projection["directional_validity"]]
        )
        expected_memberships = [
            *list(v4_projection["core_masks"][:2]),
            int(v4_projection["selected_joint_mask"]),
        ]
        expected_intervals = [list(value["overlap"]) for value in v4_projection["pooled"]]
        expected_nominated_p = int(v4_projection["fixed_p"])
        expected_coverage_mask = sum(
            (1 << index)
            for index, value in enumerate(v4_projection["coverage"])
            if value["coverage"] == "full"
        )
    terminal_expected = {
        "model_direction_mask": model_direction_mask,
        "nominated_p": expected_nominated_p,
        "coverage_mask": expected_coverage_mask,
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
    if combined_response_selected:
        selected_unconfirmed = bool(
            selected_rest_records and int(selected_rest_records[0]["disposition"]) == 1
        )
        expected_outcome = (
            STAGE_B_OUTCOME_INCONCLUSIVE if selected_unconfirmed else STAGE_B_OUTCOME_COMPLETE
        )
        expected_cause = STAGE_B_CAUSE_RECOVERY_UNAVAILABLE if selected_unconfirmed else 0
        if int(terminal["outcome"]) != expected_outcome or int(terminal["cause"]) != expected_cause:
            mismatches.append("Stage B selected-response terminal differs from recovery rest")
        if int(terminal.get("recovery_unavailable", 0)) != int(selected_unconfirmed):
            mismatches.append("Stage B selected-response recovery flag differs")
    if not replay["sufficient"] and int(terminal["outcome"]) == STAGE_B_OUTCOME_INCONCLUSIVE:
        recovery_failure = any(
            int(record["outcome"]) in (2, 3)
            and int(record["rung_index"]) + 1 < int(plan["rung_count"])
            for record in recovery_records
        )
        expected_cause = _stage_b_inconclusive_cause(
            model_direction_mask,
            recovery_failure=recovery_failure,
        )
        if int(terminal["cause"]) != expected_cause:
            mismatches.append(
                "Stage B terminal insufficiency cause differs from independent replay"
            )
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
        value
        for _raw, value in logical_nonterminal
        if value["record_kind"] in ("reproduction", "reproduction_v3", "reproduction_v4")
    ]
    if (
        int(terminal["outcome"]) == STAGE_B_OUTCOME_COMPLETE
        and not combined_response_selected
        and len(reproduction) != 1
    ):
        mismatches.append("complete Stage B terminal lacks one reproduction record")
    if len(reproduction) > 1:
        mismatches.append(f"expected at most one reproduction record, found {len(reproduction)}")
    if reproduction and reproduction[-1]["record_kind"] == "reproduction":
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
        comparison = _compare_stage_b_candidates_schema2(previous_candidate, current_candidate)
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
    elif reproduction and reproduction[-1]["record_kind"] == "reproduction_v3":
        current = reproduction[-1]
        prior, active = _stage_b_candidates_from_v3(current, plan)
        replay_candidate = {
            "memberships": [
                *(int(item["member_mask"]) for item in replay["selected"]),
                int(replay["nomination"]["member_mask"]),
            ],
            "nominated_p": int(replay["nominated_p"]),
            "pooled_intervals": [list(value["pooled_interval"]) for value in replay["selected"]],
            "common_intervals": [list(value["common_interval"]) for value in replay["selected"]],
            "p_start": int(plan["p_start"]),
            "p_top": int(plan["p_top"]),
            "rung_count": int(plan["rung_count"]),
        }
        if active != replay_candidate:
            mismatches.append("schema-3 current summary differs from independent replay")
        projection = reproduction_projection or project_stage_b_core_pair(prior, active)
        mismatches.extend(_compare_stage_b_v3_record(current, projection))
        reported_current_digest = int(current["current_digest_low"]) | (
            int(current["current_digest_high"]) << 32
        )
        if pre_reproduction_digest != reported_current_digest:
            mismatches.append("reproduction current digest differs from canonical prefix")
        expected_terminal = (
            STAGE_B_OUTCOME_COMPLETE
            if projection["outcome"] == "complete"
            else STAGE_B_OUTCOME_INCONCLUSIVE
        )
        if int(terminal["outcome"]) != expected_terminal:
            mismatches.append("terminal outcome differs from schema-3 core projection")
        expected_cause = 0 if projection["outcome"] == "complete" else 5
        if int(terminal["cause"]) != expected_cause:
            mismatches.append("terminal cause differs from schema-3 core projection")
    elif reproduction:
        current = reproduction[-1]
        prior, active = _stage_b_candidates_from_v4(current, plan)
        replay_candidate = _schema8_run_candidate(
            {
                "stage_b_replay": replay,
                "plan": plan,
            }
        )
        if active != replay_candidate:
            mismatches.append("schema-8 current summary differs from independent replay")
        projection = reproduction_projection or _project_stage_b_schema8_candidates(prior, active)
        mismatches.extend(_compare_stage_b_v4_record(current, projection))
        reported_current_digest = int(current["current_digest_low"]) | (
            int(current["current_digest_high"]) << 32
        )
        if pre_reproduction_digest != reported_current_digest:
            mismatches.append("schema-8 current digest differs from canonical prefix")
        expected_terminal = (
            STAGE_B_OUTCOME_COMPLETE
            if projection["outcome"] == "complete"
            else STAGE_B_OUTCOME_INCONCLUSIVE
        )
        if int(terminal["outcome"]) != expected_terminal:
            mismatches.append("terminal outcome differs from schema-8 component projection")
        expected_cause = 0 if projection["outcome"] == "complete" else 5
        if int(terminal["cause"]) != expected_cause:
            mismatches.append("terminal cause differs from schema-8 component projection")

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
        "nominated_p": int(terminal["nominated_p"]),
        "coverage": [
            "full" if int(terminal["coverage_mask"]) & (1 << index) else None for index in range(2)
        ],
        "pid_velocity_actual_samples": pid_samples,
        "actual_velocity_pptm_samples": [],
        "complete": True,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": caveats,
        "recoveries": [dict(record) for record in recovery_records],
        "recovery_rest_diagnostics": [dict(record) for record in selected_rest_records],
        "recovery_rest_distribution": summarize_recovery_rest_diagnostics(selected_rest_records),
        "recovery_rest_audits": recovery_rest_audits,
        "selected_recovery_rest": (
            dict(selected_rest_records[0]) if selected_rest_records and not recovery_wide else None
        ),
        "selected_recovery_rest_audit": selected_rest_audit,
        "selected_recovery_scope": selected_scope,
        "current_evidence": [dict(record) for record in current_records],
        "limit_evidence_capabilities": sorted(
            {int(record["capability"]) for record in current_records}
        ),
        "what_if_projection": {
            "interpretation": "schema-5 recovery authority over preserved evidence",
            "target_schema_revision": 5,
            "recoveries": recovery_projections,
            "current_headroom_lifecycle": (
                _project_schema6_current_headroom(
                    projected_replay,
                    terminal_rung=int(terminal["emitted_rungs"]) - 1,
                )
                if recovery_schema and historical_current_headroom_fault
                else None
            ),
        },
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
            f"plan target RPM: firmware={int(plan['target_velocity_rpm'])} offline="
            f"{int(expected_target_rpm)}"
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
        # This path only runs for captures with no "stage_b_terminal" record at
        # all (see the dispatch in audit_velocity_sweep above), so it predates
        # STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION by construction and has no
        # schema signal of its own to pass. Rely on _reconstruct_observation's
        # default (legacy significance gate).
        offline = _reconstruct_observation(raw, record, plan, terminal)
        offline.setdefault("suffix_len", 0)
        offline.setdefault("selected_level", 0)
        offline.setdefault("selected_blocks", 0)
        offline.setdefault("rate_variance_mantissa", 0)
        offline.setdefault("rate_variance_shift", 0)
        for name in exact_fields:
            offline.setdefault(name, 0)
        _compare_fields(mismatches, f"observation {key[0]}/{key[1]}", record, offline, exact_fields)
        _compare_optional_drift_ratio(mismatches, f"observation {key[0]}/{key[1]}", record, offline)
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
            f"terminal sufficiency mask: firmware={int(terminal['sufficient_direction_mask'])} "
            f"offline={int(direction_mask)}"
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
_BW_CONFIRMATION_CAPTURE_PROFILE_EXISTING = 0
_BW_CONFIRMATION_ACCEPTANCE_RULE_ALL_EIGHT_CONTAINED = 0
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

# `BREAKAWAY_STAGE_B_SCHEMA_REVISION` (combined_plan.rs): the rules revision
# `AdditiveDiscoveryPlanned.schema_revision` reports, so an offline decoder can
# confirm it is regenerating the ladder under the exact rules the firmware
# applied rather than a stale or newer revision.
_BW_STAGE_B_MIN_SCHEMA_REVISION = 15
_BW_STAGE_B_SCHEMA_REVISION = 17

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
    the held-out confirmation, Stage C, and every recovery or hidden observation
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
            _BW_STAGE_B_MIN_SCHEMA_REVISION
            <= int(discovery_planned["schema_revision"])
            <= _BW_STAGE_B_SCHEMA_REVISION
        ):
            mismatches.append(
                f"discovery schema_revision: firmware="
                f"{int(discovery_planned['schema_revision'])} want="
                f"{int(_BW_STAGE_B_MIN_SCHEMA_REVISION)}-"
                f"{int(_BW_STAGE_B_SCHEMA_REVISION)} (offline replay regenerates the ladder "
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
    # A confirmed acceptance whose Stage-C plan build is later refused
    # legitimately leaves campaign_accepted=False with confirmation_accepted=
    # True -- confirmation and Stage-C admission are deliberately
    # separate, stacked gates. Only the reverse direction (a Stage-C plan
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
        if int(campaign_terminal["stage_c_plan_digest"]) == 0:
            mismatches.append("campaign terminal reports accepted with a zero stage_c_plan_digest")
    elif int(campaign_terminal["stage_c_plan_digest"]) != 0:
        mismatches.append("campaign terminal reports no accept but a nonzero stage_c_plan_digest")

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
