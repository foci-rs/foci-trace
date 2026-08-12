"""Independent velocity-integral response geometry and reproduction audit."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .velocity_measurement import (
    _CURRENT_Q4_12_BUILD_REVISION,
    _LEGACY_Q8_8_BUILD_REVISION,
    BUILD_FEATURE_Q4_12_GAINS,
    STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION,
    STAGE_C_SUFFIX_WALK_MIN_SCHEMA_REVISION,
    UnsupportedIntegralRepresentationError,
    VelocityObservation,
    _confidence_half_width,
    _encode_rate_values,
    _stationary_suffix,
    _upper_bound_signed,
    _upper_bound_unsigned,
    audit_recovery_rest_geometry,
    audit_rung_origin_recovery,
    audit_scored_rest_grammar,
    capture_build_features,
    plan_rung_origin_recovery,
    summarize_recovery_rest_diagnostics,
)

_MICROS_PER_SECOND = 1_000_000
_LEGACY_Q8_8_I_NUMERATOR = 65_536
_CURRENT_Q4_12_I_NUMERATOR = 1_048_576
_I_MAX_POSITIVE_RUNGS = 30
_I_CONFIDENCE_FAMILY_MAX = 128
_I_OBSERVATIONS_PER_DIRECTION = 4
_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_FNV1A64_OFFSET = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3
_PLAN_RECOVERY_QUANTIZATION_EXPOSED = 1 << 0
_PLAN_PROBE_CONSTRAINED_TEST_POINT = 1 << 1
_PLAN_SLOT_ORDER_SHIFT = 2
_PLAN_SLOT_ORDER_MASK = 0b11 << _PLAN_SLOT_ORDER_SHIFT

SLOT_ORDER_FORWARD_FIRST = 0
SLOT_ORDER_REVERSE_FIRST = 1
SLOT_ORDER_PAIRED_OUT_AND_BACK = 2

_TERMINAL_RECOVERY_UNAVAILABLE = 1 << 0
_TERMINAL_RECOVERED_WITH_CURRENT_HEADROOM = 1 << 1
_TERMINAL_RECOVERY_QUANTIZATION_EXPOSED = 1 << 2
_TERMINAL_PROBE_CONSTRAINED_TEST_POINT = 1 << 3
_TERMINAL_COMBINED_TARGET_MASK = 0b111 << 4
_TERMINAL_COMBINED_WORKFLOW = 1 << 7
_SLOT_TRANSITION_ALLOWANCE_US = 20_000
_RECOVERY_STABILIZATION_US = 500_000
_STABILIZED_RECOVERY_SCHEDULE_CODE = 1
_U128_MAX = (1 << 128) - 1

# Stage-C evidence schema revisions, mirrored from firmware. Every revision
# belongs to exactly one acquisition workflow, and no two workflows share one.
# The ranges are named by the constants below rather than by literals, so a
# firmware bump widens a range without stranding this comment:
#
#   through _STANDALONE_STAGE_C_SCHEMA_REVISION
#         standalone Stage C (`integral_plan.rs:24`), planned by
#         `plan_velocity_integral_response` with `combined_selected_response:
#         None`.
#   _FIRST_COMBINED_STAGE_C_SCHEMA_REVISION through
#   _COMBINED_STAGE_C_SCHEMA_REVISION
#         the combined P-then-I workflow (`combined_plan.rs:20,22`).
#         `plan_combined_velocity_integral_response` is the only builder in
#         that range and it always supplies `Some(selected)`.
#   above the combined range, through _BREAKAWAY_STAGE_C_SCHEMA_REVISION
#         the breakaway-seeded campaign (`combined_plan.rs:40`), planned by
#         `plan_breakaway_velocity_integral_response` with
#         `combined_selected_response: None`.
#
# Revisions inside one range differ only in measurement rules -- which suffix a
# walk selects, which evidence closes a chain -- never in workflow, builder, or
# record layout. That is why a range widens rather than a new one appearing, and
# why the workflow mapping here does not change when firmware bumps.
#
# The workflow is therefore a property of the schema revision alone. Deriving
# it from the capture's own `combined_selected_response` instead lets a
# self-consistent schema-8-to-13 capture that simply omits the selected
# response -- a shape firmware cannot emit -- claim the standalone workflow and
# escape every combined-terminal rule.
_STANDALONE_STAGE_C_SCHEMA_REVISION = 7
_FIRST_COMBINED_STAGE_C_SCHEMA_REVISION = 8
_COMBINED_STAGE_C_SCHEMA_REVISION = 13
_BREAKAWAY_STAGE_C_SCHEMA_REVISION = 16


def _slot_order_parity_twist(slot_order: int, observation: int) -> int:
    """Return the parity offset one order applies to a same-direction ordinal.

    Mirrors the firmware's ``IntegralSlotOrder::parity_twist``, which is the
    only thing separating the three orders.
    """
    if slot_order == SLOT_ORDER_FORWARD_FIRST:
        return 0
    if slot_order == SLOT_ORDER_REVERSE_FIRST:
        return 1
    if slot_order == SLOT_ORDER_PAIRED_OUT_AND_BACK:
        return observation & 1
    raise ValueError(f"unknown observation slot order {slot_order}")


def slot_direction_index(slot_order: int, slot: int) -> int:
    """Return 0 for forward and 1 for reverse travel in the declared order.

    Mirrors the firmware's ``IntegralSlotOrder::direction_index``.

    Slot parity alone gives the direction only under forward-first, and the
    paired F,R,R,F order is not a parity function at all.
    """
    return (slot & 1) ^ _slot_order_parity_twist(slot_order, slot >> 1)


def observation_role(slot_order: int, slot: int) -> int:
    """Return 0 for an outbound primitive and 1 for an inbound one.

    Role is origin-relative and fixed by position in the stroke pair: even slots
    go out from the anchor, odd slots come back, under every schedule. The slot
    order does not move it. What the order changes is which travel *direction*
    is paired with each role, which is what [`_slot_order_parity_twist`]
    encodes and what [`slot_direction_index`] applies.

    That distinction is the whole basis of the paired experiment. Under the two
    sequential orders role and direction are perfectly correlated, so a rate
    difference between outbound and inbound strokes cannot be told apart from a
    difference between forward and reverse travel. Paired order populates all
    four (direction, role) cells and separates them.

    `slot_order` is taken for symmetry with the other three helpers and to
    validate the schedule, so a capture declaring an unknown order fails here
    rather than being silently analysed against an assumed one.
    """
    _slot_order_parity_twist(slot_order, slot >> 1)
    return slot & 1


def slot_for_observation(slot_order: int, observation: int, direction: int) -> int:
    """Return the slot carrying one same-direction observation ordinal.

    Mirrors the firmware's ``IntegralSlotOrder::slot_for``, the inverse of
    [`slot_direction_index`]. Evidence is written by raw slot and read back by
    observation and direction, so both mappings have to route through the
    declared order or they disagree on everything but forward-first.
    """
    return observation * 2 + (direction ^ _slot_order_parity_twist(slot_order, observation))


def plan_slot_order(plan: dict[str, object]) -> int:
    """Return the observation slot order declared by one Stage-C plan.

    Captures taken before the field existed leave the bits clear, which is
    forward-first.
    """
    return (int(plan.get("flags", 0)) & _PLAN_SLOT_ORDER_MASK) >> _PLAN_SLOT_ORDER_SHIFT


def stage_c_schema_is_known(schema_revision: int) -> bool:
    """Return whether one Stage-C evidence schema revision exists in firmware."""
    return 1 <= schema_revision <= _BREAKAWAY_STAGE_C_SCHEMA_REVISION


def stage_c_combined_workflow(schema_revision: int) -> bool:
    """Return whether one Stage-C schema revision is a combined P-then-I run.

    This is the offline equivalent of firmware's
    `VelocityIntegralExecutionPlan::combined_workflow`
    (`combined_selected_response.is_some()`, `integral_plan.rs:508`), derived
    from the schema revision rather than from the capture's own field so a
    capture cannot declare its own workflow. Schemas 8 through 13 are the
    combined range; 7 and below are standalone and 14 is the breakaway
    campaign, neither of which carries a combined selected response.
    """
    return (
        _FIRST_COMBINED_STAGE_C_SCHEMA_REVISION
        <= schema_revision
        <= _COMBINED_STAGE_C_SCHEMA_REVISION
    )


def integral_numerator_for_build(build_revision: int | None) -> int:
    """Return the integral normalization recorded by one firmware build."""
    if build_revision == _LEGACY_Q8_8_BUILD_REVISION:
        return _LEGACY_Q8_8_I_NUMERATOR
    if build_revision == _CURRENT_Q4_12_BUILD_REVISION:
        return _CURRENT_Q4_12_I_NUMERATOR
    if build_revision is None:
        raise UnsupportedIntegralRepresentationError(
            "missing firmware build revision for integral-gain interpretation"
        )
    raise UnsupportedIntegralRepresentationError(
        f"unsupported integral-gain representation for build 0x{build_revision:08x}"
    )


def integral_numerator_for_build_features(build_features: int) -> int:
    """Return the integral normalization named by a build_features bitset."""
    if build_features & BUILD_FEATURE_Q4_12_GAINS:
        return _CURRENT_Q4_12_I_NUMERATOR
    return _LEGACY_Q8_8_I_NUMERATOR


def _fnv1a64(data: bytes, initial: int = _FNV1A64_OFFSET) -> int:
    value = initial
    for byte in data:
        value = ((value ^ byte) * _FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _append_integer(buffer: bytearray, value: int, size: int, *, signed: bool = False) -> None:
    buffer.extend(value.to_bytes(size, "little", signed=signed))


def integral_plan_digest(plan: dict[str, object]) -> int:
    """Recompute the firmware Stage-C plan digest over every immutable input."""
    schema_revision = int(plan["schema_revision"])
    combined_workflow = stage_c_combined_workflow(schema_revision)
    # A newer capture states build_features and, like the single-path
    # firmware writer, gates the selected-response blocks on content
    # (`combined_selected_response.is_some()`) rather than the schema number. A
    # legacy capture keeps the build_revision and the schema-gated reproduction
    # that reproduces every archived digest exactly.
    is_post_foc252 = "build_features" in plan
    selected = bool(plan.get("combined_selected_response", False))
    encoded = bytearray()
    if is_post_foc252:
        _append_integer(encoded, int(plan["build_features"]), 2)
    else:
        _append_integer(encoded, int(plan["build_revision"]), 4)
    _append_integer(encoded, schema_revision, 2)
    _append_integer(encoded, int(plan["channel"]), 1)
    _append_integer(encoded, int(plan["stage_b_plan_digest"]), 8)
    _append_integer(encoded, int(plan["final_p"]), 2)
    for membership in plan["directional_memberships"]:  # type: ignore[union-attr]
        _append_integer(encoded, int(membership), 4)
    if schema_revision >= 7:
        _append_integer(encoded, int(plan["stage_b_joint_membership"]), 4)
    _append_integer(encoded, int(plan["joint_membership"]), 4)
    if schema_revision >= 4:
        for validity in plan["directional_validity"]:  # type: ignore[union-attr]
            _append_integer(encoded, int(validity), 1)
        _append_integer(encoded, int(bool(plan["reduced_margin"])), 1)
    for interval in plan["directional_pooled_intervals"]:  # type: ignore[union-attr]
        _append_integer(encoded, int(interval[0]), 4, signed=True)
        _append_integer(encoded, int(interval[1]), 4, signed=True)
    # Firmware hashes the presence byte from `combined_selected_response
    # .is_some()` and the intervals only behind it, so a breakaway plan hashes a
    # bare zero and carries no `selected_response_intervals_q` key. A newer
    # writer does this unconditionally; a legacy writer only from FIRST_COMBINED.
    if is_post_foc252 or schema_revision >= _FIRST_COMBINED_STAGE_C_SCHEMA_REVISION:
        _append_integer(encoded, int(selected), 1)
        if selected:
            _append_integer(encoded, int(plan["final_p"]), 2)
            for interval in plan["selected_response_intervals_q"]:  # type: ignore[union-attr]
                _append_integer(encoded, int(interval[0]), 8, signed=True)
                _append_integer(encoded, int(interval[1]), 8, signed=True)
    _append_integer(encoded, int(plan["planned_velocity_mrev_s"]), 4)
    _append_integer(encoded, int(plan["target_velocity_rpm"]), 4, signed=True)
    _append_integer(encoded, int(plan["pwm_hz"]), 4)
    _append_integer(encoded, int(plan["encoder_counts_per_rev"]), 4)
    positive_i = plan["positive_i"]
    _append_integer(encoded, int(plan["positive_rung_count"]), 1)
    for gain in positive_i:  # type: ignore[union-attr]
        _append_integer(encoded, int(gain), 2)
    _append_integer(encoded, int(plan["family_size"]), 2)
    _append_integer(encoded, int(plan["total_rung_count"]), 1)
    _append_integer(encoded, int(plan["expected_observations"]), 2)
    for name in ("moving_stroke_us", "zero_settle_us", "analysis_budget_us"):
        _append_integer(encoded, int(plan[name]), 4)
    # The current firmware hashes both of these whenever a selected response is
    # present (`integral_plan.rs:477-479`), so a newer capture mirrors that
    # by content. A legacy capture keeps the archived schema-gated reproduction:
    # every schema in 9..13 hashed the slot-transition reserve, and only schema 11
    # additionally hashed the stabilized-recovery code.
    if is_post_foc252:
        if selected:
            _append_integer(encoded, int(plan["slot_transition_allowance_us"]), 4)
            _append_integer(encoded, _STABILIZED_RECOVERY_SCHEDULE_CODE, 1)
    elif schema_revision >= 9 and combined_workflow:
        _append_integer(encoded, int(plan["slot_transition_allowance_us"]), 4)
        if schema_revision == 11:
            _append_integer(encoded, _STABILIZED_RECOVERY_SCHEDULE_CODE, 1)
    for name in ("nominal_workflow_ms", "maximum_workflow_ms"):
        _append_integer(encoded, int(plan[name]), 4)
    if schema_revision >= 2:
        for name in (
            "recovery_origin_band_counts",
            "recovery_nominal_slot_us",
            "recovery_maximum_slot_us",
        ):
            _append_integer(encoded, int(plan[name]), 4)
        _append_integer(encoded, int(plan["recovery_slot_count"]), 1)
    _append_integer(encoded, int(plan["hard_torque_limit"]), 2)
    _append_integer(encoded, int(plan["usable_torque_limit"]), 2)
    if schema_revision >= 3:
        _append_integer(encoded, int(plan["limit_evidence_capability"]), 1)
    if schema_revision >= 6:
        _append_integer(
            encoded,
            int(bool(int(plan["flags"]) & _PLAN_RECOVERY_QUANTIZATION_EXPOSED)),
            1,
        )
    if schema_revision >= 7:
        _append_integer(
            encoded,
            int(bool(int(plan["flags"]) & _PLAN_PROBE_CONSTRAINED_TEST_POINT)),
            1,
        )
    _append_integer(encoded, int(plan["requested_velocity_mrev_s"]), 4)
    _append_integer(encoded, int(plan["machine_velocity_ceiling_mrev_s"]), 4)
    _append_integer(encoded, int(plan["requested_velocity_source"]), 1)
    for name in (
        "max_stroke_travel_mrev",
        "settle_travel_reserve_mrev",
        "negative_position_headroom_mrev",
        "positive_position_headroom_mrev",
    ):
        _append_integer(encoded, int(plan[name]), 4)
    _append_integer(encoded, int(plan["max_duration_ms"]), 2)
    # Hashed only when the order is not forward-first, matching firmware.
    # Forward-first is the historical order, so omitting it keeps every archived
    # plan digest valid.
    slot_order = plan_slot_order(plan)
    if slot_order:
        _append_integer(encoded, slot_order, 1)
    return _fnv1a64(bytes(encoded))


def velocity_i_tau_us(i_raw: int, pwm_hz: int, build_features: int) -> int:
    """Return the advanced-PI time constant with firmware-equivalent ceiling."""
    if i_raw <= 0 or pwm_hz <= 0:
        raise ValueError("velocity I and PWM frequency must be positive")
    numerator = integral_numerator_for_build_features(build_features) * _MICROS_PER_SECOND
    denominator = i_raw * pwm_hz
    return (numerator + denominator - 1) // denominator


def _largest_i_for_tau(
    pwm_hz: int,
    minimum_tau_us: int,
    maximum_i: int,
    build_features: int,
) -> int:
    if pwm_hz <= 0 or minimum_tau_us <= 0 or maximum_i <= 0:
        raise ValueError("integral ladder geometry must be positive")
    largest = 0
    for candidate in range(1, maximum_i + 1):
        if velocity_i_tau_us(candidate, pwm_hz, build_features) < minimum_tau_us:
            break
        largest = candidate
    if largest == 0:
        raise ValueError("integral ladder has no representable rung")
    return largest


def derive_integral_ladder(
    pwm_hz: int,
    moving_stroke_us: int,
    shortest_suffix_us: int,
    maximum_i: int,
    *,
    schema_revision: int,
    build_features: int,
) -> dict[str, object]:
    """Reconstruct the cap-only whole-octave positive-I ladder."""
    if not stage_c_schema_is_known(schema_revision):
        raise ValueError("unsupported Stage-C evidence schema")
    if shortest_suffix_us < 2:
        raise ValueError("shortest suffix must contain at least two microseconds")
    numerator = integral_numerator_for_build_features(build_features)
    if schema_revision >= 5:
        i_start = 1
    elif schema_revision >= 4:
        i_start = numerator // _LEGACY_Q8_8_I_NUMERATOR
    else:
        i_start = _largest_i_for_tau(
            pwm_hz,
            moving_stroke_us,
            maximum_i,
            build_features,
        )
    i_top_cap = _largest_i_for_tau(
        pwm_hz,
        shortest_suffix_us // 2,
        maximum_i,
        build_features,
    )
    if i_start > i_top_cap:
        raise ValueError("integral ladder has insufficient structural span")
    if schema_revision >= 5 and i_top_cap < 3:
        raise ValueError("integral ladder requires the fixed three-rung prefix")
    rungs = [1, 2, 3] if schema_revision >= 5 else []
    candidate = 4 if schema_revision >= 5 else i_start
    while candidate <= i_top_cap:
        rungs.append(candidate)
        candidate <<= 1
    # Only the combined workflow appends the exact cap:
    # `plan_velocity_integral_response_from_source` picks
    # `build_combined_integral_ladder` when `append_exact_cap` is set, and only
    # `plan_combined_velocity_integral_response` sets it
    # (`integral_plan.rs:1217`). The breakaway plan at schema 14 passes
    # `append_exact_cap: false` (`integral_plan.rs:1438`) and so keeps the
    # plain whole-octave ladder, exactly like the standalone schemas below 8.
    if stage_c_combined_workflow(schema_revision) and rungs[-1] != i_top_cap:
        rungs.append(i_top_cap)
    family_size = (len(rungs) + 2) * _I_OBSERVATIONS_PER_DIRECTION
    if len(rungs) < 3:
        raise ValueError("integral ladder requires at least three positive rungs")
    if len(rungs) > _I_MAX_POSITIVE_RUNGS or family_size > _I_CONFIDENCE_FAMILY_MAX:
        raise ValueError("integral ladder exceeds the simultaneous-confidence family")
    return {"i_start": i_start, "i_top_cap": i_top_cap, "positive_i": rungs}


def integral_deficit_reaches_target(
    target_rpm: int,
    encoder_counts_per_rev: int,
    deficits: list[list[int]],
) -> bool:
    """Reconstruct the firmware's signed 98-percent integral target test."""
    if target_rpm <= 0 or encoder_counts_per_rev <= 0 or len(deficits) != 2:
        raise ValueError("invalid combined integral target")
    if any(len(interval) != 2 or interval[0] > interval[1] for interval in deficits):
        raise ValueError("invalid combined deficit interval")
    target_numerator = target_rpm * encoder_counts_per_rev * 1_000
    target_floor = target_numerator * 98
    forward_response = (target_numerator - deficits[0][1] * 60) * 100
    reverse_response = (target_numerator + deficits[1][0] * 60) * 100
    return forward_response >= target_floor and reverse_response >= target_floor


def combined_target_status(
    *,
    target_terminus: int | None,
    eligible_masks: list[int],
    current_terminus: int | None,
    emitted_rungs: int,
    expected_rungs: int,
) -> int | None:
    """Classify the combined ladder's report-only target outcome."""
    if len(eligible_masks) != 2:
        raise ValueError("combined target status requires two directional masks")
    if target_terminus is not None:
        return 0 if all(mask.bit_count() >= 3 for mask in eligible_masks) else 1
    if current_terminus is not None:
        return 2
    if emitted_rungs < expected_rungs:
        return None
    return 3


def _checked_u128(value: int) -> int:
    if not 0 <= value <= _U128_MAX:
        raise ValueError("Stage-C transition checked arithmetic overflow")
    return value


def _stage_b_ladder(p_start: int, p_top: int) -> list[int]:
    if not 0 < p_start <= p_top <= 0x7FFF:
        raise ValueError("invalid Stage-B ladder geometry")
    gains: list[int] = []
    previous = 0
    for octave in range(16):
        even = _checked_u128(p_start << octave)
        odd = _checked_u128(((p_start * 181) << octave) + 127) >> 7
        for candidate in (even, odd):
            if candidate > p_top:
                break
            if candidate > previous:
                gains.append(candidate)
                previous = candidate
    if previous != p_top:
        gains.append(p_top)
    return gains


def select_transition_capable_test_point(
    *,
    stage_b_joint_membership: int,
    target_rpm: int,
    counts_per_rev: int,
    pwm_hz: int,
    moving_stroke_us: int,
    probe_limit: int,
    p_start: int | None = None,
    p_top: int | None = None,
) -> dict[str, int | bool]:
    """Independently filter and nominate the Stage-C proportional test point."""
    if (
        not 0 <= stage_b_joint_membership <= 0xFFFF_FFFF
        or target_rpm == -(1 << 31)
        or not -(1 << 31) < target_rpm < (1 << 31)
        or not 0 < counts_per_rev <= 0xFFFF_FFFF
        or not 0 < pwm_hz <= 0xFFFF_FFFF
        or not 0 < moving_stroke_us <= 0xFFFF_FFFF
        or not 0 < probe_limit <= 0xFFFF
    ):
        raise ValueError("invalid Stage-C transition geometry")
    target_magnitude = abs(target_rpm)
    if target_magnitude == 0:
        raise ValueError("invalid Stage-C transition target")
    if p_start is None:
        p_start = (256 + target_magnitude - 1) // target_magnitude
    if p_top is None:
        p_top = min(0x7FFF, (256 * probe_limit) // target_magnitude)
    ladder = _stage_b_ladder(p_start, p_top)
    valid_bits = (1 << len(ladder)) - 1
    if stage_b_joint_membership & ~valid_bits:
        raise ValueError("Stage-B joint membership exceeds its ladder")

    stroke_cycles = _checked_u128(
        (_checked_u128(pwm_hz * moving_stroke_us) + _MICROS_PER_SECOND - 1) // _MICROS_PER_SECOND
    )
    error_num = _checked_u128(_checked_u128(target_magnitude * counts_per_rev) + 60 * pwm_hz)
    prefix_scale = _checked_u128(_CURRENT_Q4_12_I_NUMERATOR + _checked_u128(stroke_cycles * 3))
    rhs = _checked_u128(
        _checked_u128(_checked_u128(256 * counts_per_rev) * _CURRENT_Q4_12_I_NUMERATOR)
        * probe_limit
    )
    divisor = _checked_u128(error_num * prefix_scale)
    transition_ceiling = (rhs - 1) // divisor

    filtered = 0
    for rung_index, gain in enumerate(ladder):
        if stage_b_joint_membership & (1 << rung_index) == 0:
            continue
        lhs = _checked_u128(_checked_u128(gain * error_num) * prefix_scale)
        if lhs < rhs:
            filtered |= 1 << rung_index
    if filtered == 0:
        return {
            "stage_b_joint_membership": stage_b_joint_membership,
            "joint_membership": 0,
            "final_p": 0,
            "transition_ceiling": transition_ceiling,
            "reduced_margin": False,
            "probe_constrained_test_point": bool(stage_b_joint_membership),
        }
    first_rung = (filtered & -filtered).bit_length() - 1
    member_count = filtered.bit_count()
    selected_rung = first_rung + (member_count - 1) // 2
    return {
        "stage_b_joint_membership": stage_b_joint_membership,
        "joint_membership": filtered,
        "final_p": ladder[selected_rung],
        "transition_ceiling": transition_ceiling,
        "reduced_margin": member_count <= 2,
        "probe_constrained_test_point": filtered != stage_b_joint_membership,
    }


def recovery_quantization_exposed(
    p_raw: int,
    target_rpm: int,
    counts_per_rev: int,
    pwm_hz: int,
    probe_limit: int,
) -> bool:
    """Recompute whether one opposite-sign meter quantum can exceed the probe clamp."""
    if not 0 <= p_raw <= 0xFFFF or not 0 <= probe_limit <= 0xFFFF:
        raise ValueError("recovery exposure gain or clamp is outside u16")
    if not 0 < counts_per_rev <= 0xFFFF_FFFF or not 0 < pwm_hz <= 0xFFFF_FFFF:
        raise ValueError("recovery exposure geometry is outside u32")
    if not _I32_MIN < target_rpm <= _I32_MAX:
        raise ValueError("recovery exposure target cannot be represented as an absolute i32")
    velocity_term = abs(target_rpm) * counts_per_rev + 60 * pwm_hz
    demand = p_raw * velocity_term
    clamp = 256 * probe_limit * counts_per_rev
    return demand > clamp


def depth_three_consensus(
    intervals: list[tuple[int, int] | None],
    *,
    collected_mask: int = 0b1111,
) -> dict[str, object]:
    """Return the merged closed set covered by at least three of four intervals."""
    if len(intervals) != 4 or collected_mask & ~0b1111:
        raise ValueError("depth-three consensus requires four bounded observations")
    usable_mask = sum(
        1 << index
        for index, interval in enumerate(intervals)
        if collected_mask & (1 << index) and interval is not None
    )
    if collected_mask != 0b1111:
        return {
            "classification": 2,
            "components": [],
            "usable_mask": usable_mask,
            "included_mask": 0,
        }
    intersections: list[tuple[int, int]] = []
    for selected in combinations(range(4), 3):
        if any(intervals[index] is None for index in selected):
            continue
        selected_intervals = [intervals[index] for index in selected]
        low = max(interval[0] for interval in selected_intervals if interval is not None)
        high = min(interval[1] for interval in selected_intervals if interval is not None)
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
    included_mask = 0
    if len(components) == 1:
        low, high = components[0]
        included_mask = sum(
            1 << index
            for index, interval in enumerate(intervals)
            if interval is not None
            and usable_mask & (1 << index)
            and interval[0] <= high
            and low <= interval[1]
        )
    return {
        "classification": 0 if len(components) == 1 else 1 if len(components) == 2 else 2,
        "components": components,
        "usable_mask": usable_mask,
        "included_mask": included_mask,
    }


def classify_observation(
    *,
    direction: int,
    rung_index: int,
    settled_rate_interval_q: tuple[int, int],
    deficit_interval_q: tuple[int, int],
    stationarity_usable: bool,
    predicted_torque_target_abs: int,
    usable_torque_limit: int,
    hard_torque_limit: int,
) -> int:
    """Classify one observation in the Stage-C intrinsic namespace."""
    rate_low, rate_high = settled_rate_interval_q
    deficit_low, deficit_high = deficit_interval_q
    if (
        direction not in (0, 1)
        or rate_low > rate_high
        or deficit_low > deficit_high
        or usable_torque_limit > hard_torque_limit
    ):
        return 5
    if predicted_torque_target_abs >= hard_torque_limit:
        return 5
    wrong_direction = rate_high < 0 if direction == 0 else rate_low > 0
    if wrong_direction:
        return 5
    if not stationarity_usable:
        return 4
    moving = rate_low > 0 if direction == 0 else rate_high < 0
    if not moving:
        return 2
    if predicted_torque_target_abs >= usable_torque_limit:
        return 3
    if rung_index == 0:
        resolved = deficit_low > 0 if direction == 0 else deficit_high < 0
        if not resolved:
            return 1
    return 0


def reconstruct_observation(
    raw: list[VelocityObservation],
    identity: dict[str, object],
    plan: dict[str, object],
    *,
    observed_torque_peak_abs: int | None = None,
) -> dict[str, object]:
    """Reconstruct one canonical observation from its moving raw samples."""
    if not raw:
        raise ValueError("integral observation has no raw moving samples")
    rung_index = int(identity["rung_index"])
    slot = int(identity["slot"])
    direction = slot_direction_index(plan_slot_order(plan), slot)
    # Stage C's wire plan record always carries schema_revision, but callers in
    # tests sometimes construct a partial plan without it; treat that as
    # unknown (legacy gate) rather than raising, matching the fail-safe
    # default in _classify_stationarity. Always compare against Stage C's own
    # threshold: schema_revision here is a Stage C revision by construction,
    # never Stage B's.
    plan_schema_revision = plan.get("schema_revision")
    stationarity = _stationary_suffix(
        raw[-256:],
        schema_revision=int(plan_schema_revision) if plan_schema_revision is not None else None,
        drift_bound_min_schema_revision=STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        suffix_walk_min_schema_revision=STAGE_C_SUFFIX_WALK_MIN_SCHEMA_REVISION,
    )
    predicted = (
        max(item.peak_torque_target_abs or 0 for item in raw)
        if observed_torque_peak_abs is None
        else observed_torque_peak_abs
    )
    common: dict[str, object] = {
        "rung_index": rung_index,
        "slot": slot,
        "i_raw": int(identity["i_raw"]),
        "direction": direction,
        "role": observation_role(plan_slot_order(plan), slot),
        "initial_lead_in": int(rung_index == 0 and slot == 0),
        "predicted_torque_target_abs": predicted,
    }
    if not stationarity["usable"]:
        classification = classify_observation(
            direction=direction,
            rung_index=rung_index,
            settled_rate_interval_q=(0, 0),
            deficit_interval_q=(0, 0),
            stationarity_usable=False,
            predicted_torque_target_abs=predicted,
            usable_torque_limit=int(plan["usable_torque_limit"]),
            hard_torque_limit=int(plan["hard_torque_limit"]),
        )
        return {
            **common,
            "classification": classification,
            "stationarity_code": int(stationarity["exclusion"]) + 1,
            # Offline-derived, and only where a fit was reached: firmware zeroes
            # these on its own excluded records, so they are diagnostics rather
            # than reported values. Without them an excluded observation says
            # that it failed but never how.
            **(
                {"offline_exclusion_detail": stationarity["offline_exclusion_detail"]}
                if "offline_exclusion_detail" in stationarity
                else {}
            ),
            "settled_rate_interval_q": [0, 0],
            "settled_rate_mean_q": 0,
            "deficit_interval_q": [0, 0],
            "rate_low": 0,
            "rate_mean": 0,
            "rate_high": 0,
            "deficit_low": 0,
            "deficit_high": 0,
            "rate_shift": 0,
        }
    half_width = _confidence_half_width(
        int(stationarity["variance_of_mean_q2"]), int(plan["family_size"])
    )
    mean = int(stationarity["settled_rate_mean_q"])
    rate_low, rate_high = mean - half_width, mean + half_width
    signed_target = int(plan["target_velocity_rpm"]) * (1 if direction == 0 else -1)
    target_numerator = signed_target * int(plan["encoder_counts_per_rev"]) * 1_000
    deficit_low = (target_numerator - rate_high * 60) // 60
    deficit_high = -(-(target_numerator - rate_low * 60) // 60)
    classification = classify_observation(
        direction=direction,
        rung_index=rung_index,
        settled_rate_interval_q=(rate_low, rate_high),
        deficit_interval_q=(deficit_low, deficit_high),
        stationarity_usable=True,
        predicted_torque_target_abs=predicted,
        usable_torque_limit=int(plan["usable_torque_limit"]),
        hard_torque_limit=int(plan["hard_torque_limit"]),
    )
    encoded_rates, rate_shift = _encode_rate_values(
        (rate_low, mean, rate_high, deficit_low, deficit_high)
    )
    slope_mantissa, slope_shift = _upper_bound_signed(int(stationarity["slope_q"]))
    slope_width_mantissa, slope_width_shift = _upper_bound_unsigned(
        int(stationarity["slope_half_width_q"])
    )
    residual_mantissa, residual_shift = _upper_bound_unsigned(
        int(stationarity["residual_magnitude_q2"])
    )
    return {
        **common,
        **stationarity,
        "classification": classification,
        "stationarity_code": 0,
        "settled_rate_interval_q": [rate_low, rate_high],
        "settled_rate_mean_q": mean,
        "deficit_interval_q": [deficit_low, deficit_high],
        "rate_low": encoded_rates[0],
        "rate_mean": encoded_rates[1],
        "rate_high": encoded_rates[2],
        "deficit_low": encoded_rates[3],
        "deficit_high": encoded_rates[4],
        "rate_shift": rate_shift,
        "slope_mantissa": slope_mantissa,
        "slope_shift": slope_shift,
        "slope_half_width_mantissa": slope_width_mantissa,
        "slope_half_width_shift": slope_width_shift,
        "residual_mantissa": residual_mantissa,
        "residual_shift": residual_shift,
    }


def _checked_i32(value: int) -> int:
    if not _I32_MIN <= value <= _I32_MAX:
        raise OverflowError("velocity-integral interval does not fit signed i32")
    return value


def delta_e(rung: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    """Subtract one closed anchor interval with outward endpoint ordering."""
    return (
        _checked_i32(rung[0] - anchor[1]),
        _checked_i32(rung[1] - anchor[0]),
    )


def _closed_intersection(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int] | None:
    low = max(left[0], right[0])
    high = min(left[1], right[1])
    return (low, high) if low <= high else None


def _signed_divergence_gap(previous: tuple[int, int], current: tuple[int, int]) -> int:
    if current[0] > previous[1]:
        return _checked_i32(current[0] - previous[1])
    if current[1] < previous[0]:
        return _checked_i32(current[1] - previous[0])
    return 0


def _validate_run_evidence(evidence: dict[str, object], positive_rung_count: int) -> None:
    masks = evidence.get("eligible_masks")
    opening = evidence.get("opening")
    positive = evidence.get("positive")
    if not isinstance(masks, list) or len(masks) != 2:
        raise ValueError("integral evidence needs two directional masks")
    if not isinstance(opening, list) or len(opening) != 2:
        raise ValueError("integral evidence needs two opening anchors")
    if not isinstance(positive, list) or len(positive) != 2:
        raise ValueError("integral evidence needs two directional curves")
    valid_mask = (1 << positive_rung_count) - 1
    if any(not isinstance(mask, int) or mask & ~valid_mask for mask in masks):
        raise ValueError("integral evidence mask exceeds the declared ladder")


def compare_integral_runs(
    previous: dict[str, object],
    current: dict[str, object],
    *,
    positive_rung_count: int,
) -> dict[str, object]:
    """Compare sparse raw curves through their anchor-relative intervals."""
    if not 1 <= positive_rung_count <= _I_MAX_POSITIVE_RUNGS:
        raise ValueError("positive rung count is outside the retained domain")
    _validate_run_evidence(previous, positive_rung_count)
    _validate_run_evidence(current, positive_rung_count)
    previous_masks = list(previous["eligible_masks"])
    current_masks = list(current["eligible_masks"])
    valid_mask = (1 << positive_rung_count) - 1
    shared_masks = [previous_masks[index] & current_masks[index] for index in range(2)]
    previous_only_masks = [
        previous_masks[index] & ~current_masks[index] & valid_mask for index in range(2)
    ]
    current_only_masks = [
        current_masks[index] & ~previous_masks[index] & valid_mask for index in range(2)
    ]
    reproduced_masks = [0, 0]
    divergent_masks = [0, 0]
    overlaps: list[list[tuple[int, int] | None]] = [[None] * positive_rung_count for _ in range(2)]
    previous_delta_e: list[list[tuple[int, int] | None]] = [
        [None] * positive_rung_count for _ in range(2)
    ]
    current_delta_e: list[list[tuple[int, int] | None]] = [
        [None] * positive_rung_count for _ in range(2)
    ]
    signed_gaps = [[0] * positive_rung_count for _ in range(2)]
    for direction in range(2):
        for rung in range(positive_rung_count):
            bit = 1 << rung
            if not shared_masks[direction] & bit:
                continue
            prior_interval = previous["positive"][direction][rung]  # type: ignore[index]
            current_interval = current["positive"][direction][rung]  # type: ignore[index]
            if prior_interval is None or current_interval is None:
                raise ValueError("eligible integral rung lacks an exact interval")
            prior_delta = delta_e(prior_interval, previous["opening"][direction])  # type: ignore[index]
            active_delta = delta_e(current_interval, current["opening"][direction])  # type: ignore[index]
            previous_delta_e[direction][rung] = prior_delta
            current_delta_e[direction][rung] = active_delta
            overlap = _closed_intersection(prior_delta, active_delta)
            if overlap is not None:
                reproduced_masks[direction] |= bit
                overlaps[direction][rung] = overlap
            else:
                divergent_masks[direction] |= bit
                signed_gaps[direction][rung] = _signed_divergence_gap(prior_delta, active_delta)
    return {
        "previous_masks": previous_masks,
        "current_masks": current_masks,
        "shared_masks": shared_masks,
        "reproduced_masks": reproduced_masks,
        "divergent_masks": divergent_masks,
        "previous_only_masks": previous_only_masks,
        "current_only_masks": current_only_masks,
        "overlaps": overlaps,
        "previous_delta_e": previous_delta_e,
        "current_delta_e": current_delta_e,
        "signed_gaps": signed_gaps,
        "reproduces": all(mask.bit_count() >= 3 for mask in reproduced_masks),
    }


def stage_c_slot_transition(state: str, outcome: str, comparison: str | None = None) -> str:
    """Return the firmware-owned retained-slot state after one terminal."""
    if state not in {"authority_only", "candidate"}:
        raise ValueError("Stage-C terminal requires retained authority")
    if outcome == "plan_mismatch":
        return state
    if outcome == "fault":
        return "empty"
    if outcome == "inconclusive":
        return "authority_only"
    if outcome != "sufficient":
        raise ValueError("unknown Stage-C terminal outcome")
    if state == "authority_only":
        return "candidate"
    if comparison == "complete":
        return "empty"
    if comparison == "mismatch":
        return "candidate"
    raise ValueError("candidate comparison outcome is required")


def _compact_plan_marker(plan: dict[str, object], run_sequence: int) -> bytes:
    record = bytearray((0,))
    _append_integer(record, run_sequence, 4)
    _append_integer(record, int(plan["plan_digest"]), 8)
    _append_integer(record, int(plan["final_p"]), 2)
    _append_integer(record, int(plan["planned_velocity_mrev_s"]), 4)
    _append_integer(record, int(plan["target_velocity_rpm"]), 4, signed=True)
    _append_integer(record, int(plan["pwm_hz"]), 4)
    _append_integer(record, int(plan["positive_i"][0]), 2)  # type: ignore[index]
    _append_integer(record, int(plan["positive_rung_count"]), 1)
    _append_integer(record, int(plan["family_size"]), 2)
    for name in (
        "moving_stroke_us",
        "zero_settle_us",
        "analysis_budget_us",
        "maximum_workflow_ms",
    ):
        _append_integer(record, int(plan[name]), 4)
    if int(plan["schema_revision"]) >= 2:
        for name in (
            "recovery_origin_band_counts",
            "recovery_nominal_slot_us",
            "recovery_maximum_slot_us",
        ):
            _append_integer(record, int(plan[name]), 4)
        _append_integer(record, int(plan["recovery_slot_count"]), 1)
    _append_integer(record, int(plan["hard_torque_limit"]), 2)
    _append_integer(record, int(plan["usable_torque_limit"]), 2)
    # The presence byte and the four interval endpoints after it are a
    # combined-workflow suffix, not a schema-8 suffix.
    # `encode_velocity_integral_plan_marker` (`integral_reporting.rs:454-471`)
    # appends the byte only inside
    # `if plan.combined_workflow || acquisition_kind is Acceptance*`, so a
    # non-combined plan's marker simply ends after `usable_torque_limit`.
    # Emitting a zero byte for a schema-14 breakaway plan hashed one byte
    # firmware never hashed and made every such capture's terminal digest
    # disagree. Acceptance-matrix plans -- the other arm of that condition,
    # which also hashes `recovery_lower_rate_q` -- never reach this function:
    # `_velocity_integral_audits` (analyze.py) filters their run sequences out
    # before the audit sees them.
    if stage_c_combined_workflow(int(plan["schema_revision"])):
        _append_integer(record, 1, 1)
        # A combined plan always carries the intervals -- the field is not
        # optional on `VelocityIntegralPlan` -- so the fallback is reached only
        # by a capture whose combined block is missing, which `_validate_plan`
        # has already reported. It exists so such a capture still yields a
        # defined digest instead of raising out of the audit.
        intervals = plan.get("selected_response_intervals_q") or ((0, 0), (0, 0))
        for interval in intervals:  # type: ignore[union-attr]
            _append_integer(record, int(interval[0]), 4, signed=True)
            _append_integer(record, int(interval[1]), 4, signed=True)
    return bytes(record)


def _compare_fields(
    expected: dict[str, object],
    reported: dict[str, object],
    names: tuple[str, ...],
    mismatches: list[str],
    prefix: str,
) -> None:
    for name in names:
        if expected.get(name) != reported.get(name):
            mismatches.append(
                f"{prefix} {name}: firmware={reported.get(name)!r} offline={expected.get(name)!r}"
            )


def _direction_consensus(
    observation_records: list[dict[str, object]],
) -> dict[str, object]:
    classes = [int(item["classification"]) for item in observation_records]
    if 5 in classes:
        return {
            "classification": 4,
            "components": [],
            "usable_mask": 0,
            "included_mask": 0,
        }
    if 3 in classes:
        return {
            "classification": 3,
            "components": [],
            "usable_mask": 0,
            "included_mask": 0,
        }
    return depth_three_consensus(
        [
            tuple(item["deficit_interval_q"]) if class_code == 0 else None
            for item, class_code in zip(observation_records, classes, strict=True)
        ]
    )


def _validate_plan(plan: dict[str, object], mismatches: list[str]) -> None:
    schema_revision = int(plan["schema_revision"])
    # Raises for a legacy capture whose build_revision names no known
    # representation, preserving the old refuse-to-guess behavior.
    build_features = capture_build_features(plan, stage="c")
    if not stage_c_schema_is_known(schema_revision):
        mismatches.append(
            f"plan schema revision {schema_revision} is outside the firmware Stage-C range"
        )
        return
    # The schema revision, not the capture's own field, names the workflow that
    # built this plan; see the schema-revision block at the top of this module.
    # Enforcing the mapping here is what stops a self-consistent schema-8-to-13
    # capture from omitting its selected response -- along with Event 150, the
    # combined recovery-flag bit and the combined terminal tail -- and passing
    # as a standalone run. Every derivation below, and the recovery schedule
    # and terminal expectations in `audit_velocity_integral`, read the workflow
    # from the schema once this check has confirmed the two agree.
    combined_workflow = stage_c_combined_workflow(schema_revision)
    if bool(plan.get("combined_selected_response", False)) != combined_workflow:
        mismatches.append(
            "plan combined selected response contradicts its schema workflow: "
            f"schema={schema_revision} expects "
            f"{'a combined' if combined_workflow else 'no combined'} selected response"
        )
    try:
        calculated_digest = integral_plan_digest(plan)
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        mismatches.append(f"plan reconstruction failed: {error}")
        return
    if calculated_digest != int(plan["plan_digest"]):
        mismatches.append(
            f"plan digest: firmware=0x{int(plan['plan_digest']):016x} "
            f"offline=0x{calculated_digest:016x}"
        )
    planned_velocity_mrev_s = int(plan["planned_velocity_mrev_s"])
    if schema_revision >= 10:
        expected_target = planned_velocity_mrev_s * 60 // 1_000
    else:
        expected_target = (planned_velocity_mrev_s * 60 + 500) // 1_000
    if expected_target != int(plan["target_velocity_rpm"]):
        mismatches.append("plan target RPM differs from schema conversion")
    try:
        ladder = derive_integral_ladder(
            int(plan["pwm_hz"]),
            int(plan["moving_stroke_us"]),
            64_000,
            32_767,
            schema_revision=schema_revision,
            build_features=build_features,
        )
    except ValueError as error:
        mismatches.append(f"plan ladder reconstruction failed: {error}")
        return
    positive_i = list(plan["positive_i"])  # type: ignore[arg-type]
    if positive_i != ladder["positive_i"]:
        mismatches.append(f"plan ladder: firmware={positive_i} offline={ladder['positive_i']}")
    expected_rungs = len(positive_i) + 2
    expected_observations = expected_rungs * 8
    expected_family = expected_rungs * 4
    base_duration_us = expected_observations * (
        int(plan["moving_stroke_us"])
        + int(plan["zero_settle_us"])
        + int(plan["analysis_budget_us"])
    )
    if schema_revision >= 6:
        flags = int(plan.get("flags", -1))
        known_flags = _PLAN_RECOVERY_QUANTIZATION_EXPOSED
        if schema_revision >= 7:
            known_flags |= _PLAN_PROBE_CONSTRAINED_TEST_POINT
        if schema_revision >= 12:
            # Mirrored and paired plans already exist at schema 12 (Stage C's
            # current revision is 13, one ahead of the slot-order rollout), so
            # every affected capture -- including the ones the drift-bound
            # replay depends on -- was taken at schema 12. Gating this at 13
            # would preserve the defect for exactly those captures.
            known_flags |= _PLAN_SLOT_ORDER_MASK
        if flags < 0 or flags & ~known_flags:
            mismatches.append("plan recovery flags contain reserved bits")
        else:
            try:
                expected_exposure = recovery_quantization_exposed(
                    int(plan["final_p"]),
                    int(plan["target_velocity_rpm"]),
                    int(plan["encoder_counts_per_rev"]),
                    int(plan["pwm_hz"]),
                    int(plan["usable_torque_limit"]),
                )
            except ValueError as error:
                mismatches.append(f"plan recovery exposure reconstruction failed: {error}")
            else:
                reported_exposure = bool(flags & _PLAN_RECOVERY_QUANTIZATION_EXPOSED)
                if reported_exposure != expected_exposure:
                    mismatches.append(
                        "plan recovery quantization exposure differs from exact reconstruction"
                    )
    if schema_revision == _STANDALONE_STAGE_C_SCHEMA_REVISION:
        try:
            selection = select_transition_capable_test_point(
                stage_b_joint_membership=int(plan["stage_b_joint_membership"]),
                target_rpm=int(plan["target_velocity_rpm"]),
                counts_per_rev=int(plan["encoder_counts_per_rev"]),
                pwm_hz=int(plan["pwm_hz"]),
                moving_stroke_us=int(plan["moving_stroke_us"]),
                probe_limit=int(plan["usable_torque_limit"]),
            )
        except (KeyError, ValueError) as error:
            mismatches.append(f"Stage-C test-point reconstruction failed: {error}")
        else:
            for field in ("joint_membership", "final_p", "reduced_margin"):
                if plan[field] != selection[field]:
                    mismatches.append(
                        f"plan {field}: firmware={plan[field]} offline={selection[field]}"
                    )
            reported_constraint = bool(int(plan["flags"]) & _PLAN_PROBE_CONSTRAINED_TEST_POINT)
            if reported_constraint != selection["probe_constrained_test_point"]:
                mismatches.append(
                    "plan probe-constrained test-point flag differs from exact reconstruction"
                )
            membership = int(plan["joint_membership"])
            selected_rung = (membership & -membership).bit_length() - 1
            selected_rung += (membership.bit_count() - 1) // 2
            for direction, validity in enumerate(plan["directional_validity"]):
                if int(validity) not in (3, 4):
                    continue
                expected = (
                    3
                    if int(plan["directional_memberships"][direction]) & (1 << selected_rung)
                    else 4
                )
                if int(validity) != expected:
                    mismatches.append(f"plan direction {direction} validity differs at selected P")
    if combined_workflow:
        # Presence is required across the whole combined range, and forbidden
        # outside it -- the identity check above already reported a plan that
        # disagrees. `plan_combined_velocity_integral_response` is the only
        # builder for schemas 8 through 13 and it always supplies
        # `Some(selected)` (`integral_plan.rs:1214`), so the intervals are
        # always there to validate; a standalone or breakaway plan carries none
        # (`integral_plan.rs:1163` and `:1435`) and skips this block entirely.
        intervals = plan.get("selected_response_intervals_q")
        if (
            not isinstance(intervals, list)
            or len(intervals) != 2
            or any(
                not isinstance(interval, list)
                or len(interval) != 2
                or int(interval[0]) > int(interval[1])
                for interval in intervals
            )
        ):
            mismatches.append("combined plan selected response intervals are invalid")
    if schema_revision >= 9 and combined_workflow:
        # The schedule fragment carries this field only for a combined plan
        # (`integral_reporting.rs:316-321`), so a breakaway schema-14 plan
        # legitimately has no key here and must not be asked for one.
        if int(plan.get("slot_transition_allowance_us", -1)) != _SLOT_TRANSITION_ALLOWANCE_US:
            mismatches.append("combined plan slot-transition allowance differs from schema 9")
    elif "slot_transition_allowance_us" in plan:
        mismatches.append("noncombined plan carries a combined slot-transition allowance")
    if schema_revision >= 4:
        validity = plan.get("directional_validity")
        if (
            not isinstance(validity, list)
            or len(validity) != 2
            or any(int(value) not in range(5) for value in validity)
        ):
            mismatches.append("plan Stage-B directional validity is outside the typed domain")
        if not isinstance(plan.get("reduced_margin"), bool):
            mismatches.append("plan Stage-B reduced-margin field is not boolean")
    if schema_revision >= 2:
        try:
            recovery = plan_rung_origin_recovery(
                max_stroke_travel_mrev=int(plan["max_stroke_travel_mrev"]),
                settle_travel_reserve_mrev=int(plan["settle_travel_reserve_mrev"]),
                encoder_counts_per_rev=int(plan["encoder_counts_per_rev"]),
                planned_velocity_mrev_s=int(plan["planned_velocity_mrev_s"]),
                primitive_cap_us=int(plan["max_duration_ms"]) * 1_000,
            )
        except ValueError as error:
            mismatches.append(f"plan recovery reconstruction failed: {error}")
            return
        expected_nominal_slot_us = int(recovery["nominal_slot_us"])
        if schema_revision >= 11 and combined_workflow:
            # Firmware adds the stabilization reserve only for the combined
            # workflow (`integral_plan.rs:1544-1553`), which is also the only
            # workflow whose recoveries run the `Stabilized` rest schedule
            # (`integral_sweep.rs:779-785`). A breakaway schema-14 plan keeps
            # the bare `plan_rung_origin_recovery` slot.
            expected_nominal_slot_us += _RECOVERY_STABILIZATION_US
        for name, expected in (
            ("recovery_origin_band_counts", recovery["origin_band_counts"]),
            ("recovery_nominal_slot_us", expected_nominal_slot_us),
            ("recovery_maximum_slot_us", recovery["maximum_slot_us"]),
            ("recovery_slot_count", expected_rungs),
        ):
            if int(plan[name]) != expected:
                mismatches.append(f"plan {name}: firmware={plan[name]} offline={expected}")
        nominal_duration_us = base_duration_us + expected_rungs * int(
            plan["recovery_nominal_slot_us"]
        )
        maximum_duration_us = base_duration_us + expected_rungs * int(
            plan["recovery_maximum_slot_us"]
        )
        if schema_revision >= 5:
            cleanup_duration_us = int(plan["zero_settle_us"]) + int(plan["analysis_budget_us"])
            nominal_duration_us += cleanup_duration_us
            maximum_duration_us += cleanup_duration_us
        if schema_revision >= 9 and combined_workflow:
            # Same gate as the hashed allowance above: only a combined plan
            # reserves per-slot SPI transition time
            # (`integral_plan.rs:1534-1542`).
            transition_duration_us = expected_observations * _SLOT_TRANSITION_ALLOWANCE_US
            nominal_duration_us += transition_duration_us
            maximum_duration_us += transition_duration_us
        expected_nominal_ms = (nominal_duration_us + 999) // 1_000
        expected_maximum_ms = (maximum_duration_us + 999) // 1_000
    else:
        expected_nominal_ms = base_duration_us // 1_000
        expected_maximum_ms = expected_nominal_ms
    for name, expected in (
        ("positive_rung_count", len(positive_i)),
        ("total_rung_count", expected_rungs),
        ("expected_observations", expected_observations),
        ("family_size", expected_family),
        ("nominal_workflow_ms", expected_nominal_ms),
        ("maximum_workflow_ms", expected_maximum_ms),
    ):
        if int(plan[name]) != expected:
            mismatches.append(f"plan {name}: firmware={plan[name]} offline={expected}")


def _stage_c_run_sufficient(
    schema_revision: int,
    opening: list[tuple[int, int] | None],
    bookend: list[tuple[int, int] | None],
    eligible_masks: list[int],
) -> bool:
    """Return whether one run can participate in its schema's reproduction gate."""
    anchors_available = all(interval is not None for interval in opening)
    positive_floor = all(mask.bit_count() >= 3 for mask in eligible_masks)
    bookends_available = schema_revision < 4 or all(interval is not None for interval in bookend)
    return anchors_available and positive_floor and bookends_available


def _audit_failed_stage_c_admission(
    raw_observations: list[Any],
    terminal: dict[str, object],
    evidence_count: int,
) -> dict[str, object]:
    mismatches: list[str] = []
    expected = {
        "evidence_sequence": 0,
        "outcome": 5,
        "cause": 11,
        "recovery_flags": _TERMINAL_PROBE_CONSTRAINED_TEST_POINT,
        "expected_observations": 0,
        "emitted_observations": 0,
        "expected_rungs": 0,
        "emitted_rungs": 0,
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
        "run_started_us": 0,
        "run_completed_us": 0,
    }
    for field, value in expected.items():
        if int(terminal.get(field, -1)) != value:
            mismatches.append(
                f"failed admission {field}: firmware={terminal.get(field)} expected={value}"
            )
    if int(terminal.get("plan_digest", 0)) == 0:
        mismatches.append("failed admission plan digest is zero")
    if int(terminal.get("digest", -1)) != 0:
        mismatches.append("failed admission evidence digest is nonzero")
    if list(terminal.get("eligible_masks", [])) != [0, 0]:
        mismatches.append("failed admission retained eligible masks")
    if terminal.get("current_terminus") is not None:
        mismatches.append("failed admission retained a current terminus")
    if raw_observations or evidence_count != 1:
        mismatches.append("failed admission carried motion evidence")
    decoded_terminal = dict(terminal)
    decoded_terminal["recovery_unavailable"] = 0
    decoded_terminal["recovered_with_current_headroom"] = False
    decoded_terminal["recovery_quantization_exposed"] = False
    decoded_terminal["probe_constrained_test_point"] = True
    return {
        "run_sequence": int(terminal.get("run_sequence", 0)),
        "plan": None,
        "terminal": decoded_terminal,
        "complete": False,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": ["zero-motion Stage-C admission failure; no plan or curve exists"],
        "interpretation": "firmware-schema-7",
    }


def _retained_stage_c_evidence(
    rung_results: dict[int, list[dict[str, object]]],
    positive_count: int,
) -> tuple[
    list[tuple[int, int] | None],
    list[tuple[int, int] | None],
    list[list[tuple[int, int] | None]],
    list[int],
]:
    """Apply firmware's Stage-C retention rules to reconstructed rung results."""
    positive: list[list[tuple[int, int] | None]] = [[None] * positive_count for _ in range(2)]
    eligible_masks = [0, 0]
    opening: list[tuple[int, int] | None] = [None, None]
    bookend: list[tuple[int, int] | None] = [None, None]
    for rung_index, directions in rung_results.items():
        if rung_index == 0:
            if len(directions) == 2 and all(
                int(result["classification"]) == 0 for result in directions
            ):
                opening = [
                    tuple(result["components"][0])  # type: ignore[index]
                    for result in directions
                ]
            continue
        for direction, result in enumerate(directions):
            if int(result["classification"]) != 0:
                continue
            interval = tuple(result["components"][0])  # type: ignore[index]
            if rung_index == positive_count + 1:
                bookend[direction] = interval
            else:
                positive[direction][rung_index - 1] = interval
                eligible_masks[direction] |= 1 << (rung_index - 1)
    return opening, bookend, positive, eligible_masks


def audit_velocity_integral(
    raw_observations: list[Any], evidence_records: list[tuple[bytes, dict[str, object]]]
) -> dict[str, object]:
    """Audit one integral-response run from raw samples and canonical records."""
    plan_cores = [
        value for _raw, value in evidence_records if value["record_kind"] == "integral_plan_core"
    ]
    plan_schedules = [
        value
        for _raw, value in evidence_records
        if value["record_kind"] == "integral_plan_schedule"
    ]
    plan_recoveries = [
        value
        for _raw, value in evidence_records
        if value["record_kind"] == "integral_plan_recovery"
    ]
    terminals = [
        value for _raw, value in evidence_records if value["record_kind"] == "integral_terminal"
    ]
    rest_boundaries = [
        (raw, value)
        for raw, value in evidence_records
        if value["record_kind"] == "integral_rest_boundary"
    ]
    identities = (
        plan_cores or plan_schedules or terminals or [value for _raw, value in evidence_records]
    )
    if (
        not plan_cores
        and not plan_schedules
        and not plan_recoveries
        and len(terminals) == 1
        and int(terminals[0].get("outcome", -1)) == 5
    ):
        return _audit_failed_stage_c_admission(
            raw_observations,
            terminals[0],
            len(evidence_records),
        )
    schema_revision = int(plan_cores[0].get("schema_revision", 1)) if len(plan_cores) == 1 else 1
    recovery_fragment_count = 1 if schema_revision >= 2 else 0
    if (
        len(plan_cores) != 1
        or len(plan_schedules) != 1
        or len(plan_recoveries) != recovery_fragment_count
        or len(terminals) != 1
    ):
        return {
            "run_sequence": int(identities[0].get("run_sequence", 0)) if identities else 0,
            "plan": None,
            "terminal": dict(terminals[0]) if len(terminals) == 1 else None,
            "complete": False,
            "matches_firmware": False,
            "mismatches": [
                "partial integral-response capture: required plan fragments and one terminal "
                "must be present exactly once"
            ],
            "caveats": ["partial velocity-integral response capture"],
        }
    core, schedule, terminal = plan_cores[0], plan_schedules[0], terminals[0]
    plan_identity_fields = (
        "run_sequence",
        "evidence_sequence",
        "plan_digest",
        "positive_rung_count",
    )
    fragment_mismatches: list[str] = []
    _compare_fields(
        core,
        schedule,
        plan_identity_fields,
        fragment_mismatches,
        "plan fragments",
    )
    if plan_recoveries:
        _compare_fields(
            core,
            plan_recoveries[0],
            ("run_sequence", "evidence_sequence", "plan_digest"),
            fragment_mismatches,
            "recovery plan fragment",
        )
    plan = {
        **core,
        **schedule,
        **(plan_recoveries[0] if plan_recoveries else {}),
        "record_kind": "integral_plan",
    }
    current_records = [
        value
        for _raw, value in evidence_records
        if value["record_kind"] == "velocity_primitive_current"
    ]
    current_capabilities = {int(value["capability"]) for value in current_records}
    if schema_revision >= 3:
        if "limit_evidence_capability" not in plan and len(current_capabilities) == 1:
            plan["limit_evidence_capability"] = next(iter(current_capabilities))
        elif (
            "limit_evidence_capability" in plan
            and current_capabilities
            and current_capabilities != {int(plan["limit_evidence_capability"])}
        ):
            fragment_mismatches.append(
                "Stage C current evidence capability differs from the immutable plan"
            )
    run_sequence = int(plan["run_sequence"])
    mismatches = fragment_mismatches
    _validate_plan(plan, mismatches)
    if int(terminal["run_sequence"]) != run_sequence:
        mismatches.append("terminal run sequence differs from plan")
    if int(terminal["plan_digest"]) != int(plan["plan_digest"]):
        mismatches.append("terminal plan digest differs from full plan")
    if len(rest_boundaries) > 1:
        mismatches.append("at most one integral rest-boundary record is permitted")
    rest_boundary = rest_boundaries[0] if len(rest_boundaries) == 1 else None

    nonterminal = [
        (raw, value)
        for raw, value in evidence_records
        if value["record_kind"]
        in (
            "integral_plan_marker",
            "integral_observation",
            "integral_rung",
            "recovery_rest",
            "rest_selection",
            "rung_origin_recovery",
            "velocity_primitive_current",
        )
    ]
    # `integral_recovery_rest_schedule` (`integral_sweep.rs:779-785`) returns
    # `Stabilized` for `plan.combined_workflow || is_acceptance_matrix(plan)`
    # and `ImmediateScored` otherwise. Acceptance-matrix runs cannot reach this
    # caller: `_velocity_integral_audits` (analyze.py) collects `matrix_runs`
    # first and filters every candidate with
    # `int(item.payload["run_sequence"]) not in matrix_runs`. So the schedule
    # is the combined-workflow identity alone -- taken from the schema
    # revision, which `_validate_plan` has already confirmed agrees with the
    # plan's own field, rather than from that field directly. A capture that
    # simply omitted the field would otherwise choose its own schedule.
    stabilized_recovery_rest = stage_c_combined_workflow(schema_revision)
    mismatches.extend(
        audit_scored_rest_grammar(
            [value for _raw, value in nonterminal],
            terminal,
            schema_revision=schema_revision,
            schema_floor=12,
            expected_stage=1,
            ordinary_owners=(3, 4, 6),
            recovery_owners=(5,),
            observation_kind="integral_observation",
            plan_digest=int(plan["plan_digest"]),
            label="Stage C",
            stabilized_recovery_rest=stabilized_recovery_rest,
            ordinary_rejection_tail=("velocity_primitive_current",),
        )
    )
    sequences = [int(value["evidence_sequence"]) for _raw, value in nonterminal]
    if sequences != list(range(1, len(sequences) + 1)):
        mismatches.append(f"integral evidence sequence is not contiguous: {sequences}")
    if int(terminal["evidence_sequence"]) != len(nonterminal) + 1:
        mismatches.append("integral terminal sequence does not follow final evidence record")
    if rest_boundary is not None:
        boundary = rest_boundary[1]
        if int(boundary["evidence_sequence"]) != int(terminal["evidence_sequence"]):
            mismatches.append("integral rest boundary does not share the terminal sequence")
        if int(boundary["run_sequence"]) != run_sequence:
            mismatches.append("integral rest boundary carries the wrong run sequence")
        if int(boundary["plan_digest"]) != int(plan["plan_digest"]):
            mismatches.append("integral rest boundary carries the wrong plan digest")
        positive_rung_index = int(boundary["positive_rung_index"])
        slot = int(boundary["slot"])
        if not 0 <= positive_rung_index < int(plan["positive_rung_count"]):
            mismatches.append("integral rest boundary rung is outside the positive ladder")
        elif int(boundary["i_raw"]) != int(plan["positive_i"][positive_rung_index]):  # type: ignore[index]
            mismatches.append("integral rest boundary carries the wrong I gain")
        if not 0 <= slot < 8:
            mismatches.append("integral rest boundary slot is outside the fixed schedule")
        elif int(boundary["direction"]) != (
            1 if slot_direction_index(plan_slot_order(plan), slot) == 0 else -1
        ):
            mismatches.append("integral rest boundary carries the wrong direction")
        if int(boundary["cleanup_confirmed"]) not in (0, 1):
            mismatches.append("integral rest boundary cleanup flag is not boolean")
    marker_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "integral_plan_marker"
    ]
    if len(marker_records) > 1:
        mismatches.append("at most one compact integral plan marker is permitted")
    elif marker_records:
        _compare_fields(
            {
                "run_sequence": run_sequence,
                "plan_digest": plan["plan_digest"],
                "final_p": plan["final_p"],
                "positive_rung_count": plan["positive_rung_count"],
                "maximum_workflow_ms": plan["maximum_workflow_ms"],
            },
            marker_records[0],
            (
                "run_sequence",
                "plan_digest",
                "final_p",
                "positive_rung_count",
                "maximum_workflow_ms",
            ),
            mismatches,
            "plan marker",
        )
        if schema_revision >= 2:
            _compare_fields(
                plan,
                marker_records[0],
                (
                    "planned_velocity_mrev_s",
                    "target_velocity_rpm",
                    "pwm_hz",
                    "recovery_origin_band_counts",
                    "recovery_nominal_slot_us",
                    "recovery_maximum_slot_us",
                    "recovery_slot_count",
                ),
                mismatches,
                "plan marker",
            )
        if schema_revision >= 8:
            _compare_fields(
                plan,
                marker_records[0],
                ("selected_response_intervals_q",),
                mismatches,
                "plan marker",
            )
    digest = _fnv1a64(_compact_plan_marker(plan, run_sequence))
    for raw, _value in nonterminal:
        digest = _fnv1a64(raw, digest)
    if rest_boundary is not None:
        digest = _fnv1a64(rest_boundary[0], digest)
    if digest != int(terminal["digest"]):
        mismatches.append(
            f"terminal digest: firmware=0x{int(terminal['digest']):016x} offline=0x{digest:016x}"
        )

    by_stroke: dict[int, list[VelocityObservation]] = {}
    for raw_observation in raw_observations:
        if not isinstance(raw_observation, VelocityObservation):
            raise TypeError("integral audit requires VelocityObservation raw samples")
        by_stroke.setdefault(raw_observation.stroke_index, []).append(raw_observation)
    observation_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "integral_observation"
    ]
    current_by_key = {
        (int(value["rung_index"]), int(value["slot"])): value for value in current_records
    }
    reconstructed: dict[tuple[int, int], dict[str, object]] = {}
    expected_gains = [0, *plan["positive_i"], 0]  # type: ignore[list-item]
    for record in observation_records:
        key = (int(record["rung_index"]), int(record["slot"]))
        if key in reconstructed:
            mismatches.append(f"duplicate integral observation {key}")
            continue
        rung_index, slot = key
        if not 0 <= rung_index < len(expected_gains) or not 0 <= slot < 8:
            mismatches.append(f"integral observation identity is out of range: {key}")
            continue
        if int(record["i_raw"]) != int(expected_gains[rung_index]):
            mismatches.append(f"integral observation {key} has wrong I gain")
        if int(record["direction"]) != slot_direction_index(plan_slot_order(plan), slot):
            mismatches.append(f"integral observation {key} has wrong direction")
        if int(record["plan_digest"]) != int(plan["plan_digest"]):
            mismatches.append(f"integral observation {key} has wrong plan digest")
        raw = by_stroke.get(rung_index * 8 + slot)
        if not raw:
            mismatches.append(f"integral observation {key} lacks raw moving samples")
            continue
        expected = reconstruct_observation(
            raw,
            {"rung_index": rung_index, "slot": slot, "i_raw": expected_gains[rung_index]},
            plan,
            observed_torque_peak_abs=(
                int(current_by_key[key]["moving_pid_output_peak"])
                if key in current_by_key
                else None
            ),
        )
        _compare_fields(
            expected,
            record,
            (
                "i_raw",
                "direction",
                "classification",
                "initial_lead_in",
                "rate_low",
                "rate_mean",
                "rate_high",
                "deficit_low",
                "deficit_high",
                "rate_shift",
                "predicted_torque_target_abs",
                "stationarity_code",
                "settled_rate_interval_q",
                "settled_rate_mean_q",
                "deficit_interval_q",
            ),
            mismatches,
            f"observation {key}",
        )
        if int(record["stationarity_code"]) == 0:
            _compare_fields(
                expected,
                record,
                (
                    "suffix_len",
                    "selected_level",
                    "selected_blocks",
                    "slope_mantissa",
                    "slope_shift",
                    "slope_half_width_mantissa",
                    "slope_half_width_shift",
                    "lag_one_q",
                    "lag_one_half_width_q",
                    "residual_mantissa",
                    "residual_shift",
                    "tested_suffixes",
                ),
                mismatches,
                f"observation {key} stationarity",
            )
        reconstructed[key] = expected

    rung_records = [value for _raw, value in nonterminal if value["record_kind"] == "integral_rung"]
    recovery_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "rung_origin_recovery"
    ]
    recovery_rest_records = [
        value for _raw, value in nonterminal if value["record_kind"] == "recovery_rest"
    ]
    recovery_rest_rungs = [int(record["rung_index"]) for record in recovery_rest_records]
    if len(recovery_rest_rungs) != len(set(recovery_rest_rungs)):
        mismatches.append("Stage C carries duplicate recovery rest diagnostics")
    if schema_revision < 11 and recovery_rest_records:
        mismatches.append("historical Stage C carries recovery-wide rest evidence")
    for record in recovery_rest_records:
        mismatches.extend(
            f"recovery rest rung {record['rung_index']}: {item}"
            for item in audit_recovery_rest_geometry(record, plan, expected_stage=1)
        )
        matching = [
            recovery
            for recovery in recovery_records
            if int(recovery["rung_index"]) == int(record["rung_index"])
        ]
        if int(record["disposition"]) == 0 and (
            len(matching) != 1 or int(matching[0]["outcome"]) not in (1, 5)
        ):
            mismatches.append("confirmed Stage C recovery rest lacks one moving recovery")
        if int(record["disposition"]) == 1 and (matching or int(terminal["cause"]) != 53):
            mismatches.append("unconfirmed Stage C recovery rest has the wrong successor")
    if schema_revision >= 6:
        recovery_flags = int(terminal.get("recovery_flags", -1))
        known_flags = (
            _TERMINAL_RECOVERY_UNAVAILABLE
            | _TERMINAL_RECOVERED_WITH_CURRENT_HEADROOM
            | _TERMINAL_RECOVERY_QUANTIZATION_EXPOSED
        )
        if schema_revision >= 7:
            known_flags |= _TERMINAL_PROBE_CONSTRAINED_TEST_POINT
        if schema_revision >= 8:
            known_flags |= _TERMINAL_COMBINED_TARGET_MASK | _TERMINAL_COMBINED_WORKFLOW
        if recovery_flags < 0 or recovery_flags & ~known_flags:
            mismatches.append("terminal recovery flags contain reserved bits")
        else:
            terminal["recovery_unavailable"] = int(
                bool(recovery_flags & _TERMINAL_RECOVERY_UNAVAILABLE)
            )
            terminal["recovered_with_current_headroom"] = bool(
                recovery_flags & _TERMINAL_RECOVERED_WITH_CURRENT_HEADROOM
            )
            terminal["recovery_quantization_exposed"] = bool(
                recovery_flags & _TERMINAL_RECOVERY_QUANTIZATION_EXPOSED
            )
            terminal["probe_constrained_test_point"] = bool(
                recovery_flags & _TERMINAL_PROBE_CONSTRAINED_TEST_POINT
            )
            plan_exposure = bool(int(plan.get("flags", 0)) & _PLAN_RECOVERY_QUANTIZATION_EXPOSED)
            if terminal["recovery_quantization_exposed"] != plan_exposure:
                mismatches.append("terminal recovery exposure differs from the plan")
            if schema_revision >= 7:
                plan_constraint = bool(
                    int(plan.get("flags", 0)) & _PLAN_PROBE_CONSTRAINED_TEST_POINT
                )
                if terminal["probe_constrained_test_point"] != plan_constraint:
                    mismatches.append("terminal probe constraint differs from the plan")
    for record in current_records:
        if int(record["stage"]) != 1:
            mismatches.append("Stage C current evidence carries the wrong stage")
        if int(record["clamp_readback"]) != int(record["clamp_limit"]):
            mismatches.append("Stage C current evidence clamp readback differs")
    historical_attribution_caveat = False
    terminal_observation_without_current = (
        (
            int(nonterminal[-1][1]["rung_index"]),
            int(nonterminal[-1][1]["slot"]),
        )
        if nonterminal
        and nonterminal[-1][1]["record_kind"] == "integral_observation"
        and int(terminal["outcome"]) == 3
        else None
    )
    if schema_revision >= 3 and int(terminal["outcome"]) != 3:
        observation_keys = {
            (int(value["rung_index"]), int(value["slot"])) for value in observation_records
        }
        current_keys = {(int(value["rung_index"]), int(value["slot"])) for value in current_records}
        expected_current_keys = set(observation_keys)
        if schema_revision >= 5 and rest_boundary is not None:
            expected_current_keys.add(
                (
                    int(rest_boundary[1]["positive_rung_index"]) + 1,
                    int(rest_boundary[1]["slot"]),
                )
            )
        if current_keys != expected_current_keys:
            mismatches.append("Stage C current evidence does not cover every observation")
    if schema_revision >= 2:
        recovery_rungs = [int(record["rung_index"]) for record in recovery_records]
        emitted_recovery_rungs = sorted({int(record["rung_index"]) for record in rung_records})
        if schema_revision >= 5 and rest_boundary is not None and int(terminal["cause"]) == 10:
            emitted_recovery_rungs = sorted(
                {
                    *emitted_recovery_rungs,
                    int(rest_boundary[1]["positive_rung_index"]) + 1,
                }
            )
        missing_final_recovery = (
            bool(emitted_recovery_rungs) and recovery_rungs == emitted_recovery_rungs[:-1]
        )
        terminal_rung = emitted_recovery_rungs[-1] if emitted_recovery_rungs else None
        terminal_follows_rung = bool(nonterminal) and (
            nonterminal[-1][1]["record_kind"] == "integral_rung"
            and int(nonterminal[-1][1]["rung_index"]) == terminal_rung
        )
        missing_final_allowed = missing_final_recovery and (
            int(terminal.get("recovery_unavailable", 0)) == 1
            or (int(terminal["outcome"]) == 3 and terminal_follows_rung)
            or (
                int(terminal["outcome"]) == 2
                and int(terminal["cause"]) == 1
                and emitted_recovery_rungs == [0]
                and terminal_follows_rung
            )
        )
        expected_recovery_rungs = (
            recovery_rungs if missing_final_allowed else emitted_recovery_rungs
        )
        if recovery_rungs != expected_recovery_rungs:
            mismatches.append(
                "integral recovery rungs differ from fully acquired rung sequence: "
                f"firmware={recovery_rungs} offline={expected_recovery_rungs}"
            )
        for record in recovery_records:
            if int(record["stage"]) != 1:
                mismatches.append("integral recovery record carries the wrong stage")
            recovery_audit = audit_rung_origin_recovery(record, plan)
            historical_attribution_caveat |= bool(
                recovery_audit.get("historical_attribution_caveat", False)
            )
            mismatches.extend(
                f"recovery rung {record['rung_index']}: {item}"
                for item in recovery_audit["mismatches"]
            )
        if schema_revision >= 11 and stabilized_recovery_rest:
            diagnostic_rung_set = set(recovery_rest_rungs)
            for recovery in recovery_records:
                if (
                    int(recovery["outcome"]) in (1, 5)
                    and int(recovery["rung_index"]) not in diagnostic_rung_set
                ):
                    mismatches.append("moving Stage C recovery lacks recovery rest evidence")
        if schema_revision >= 6 and "recovered_with_current_headroom" in terminal:
            recovered = any(int(record["outcome"]) == 5 for record in recovery_records)
            if bool(terminal["recovered_with_current_headroom"]) != recovered:
                mismatches.append(
                    "terminal recovered flag differs from causal outcome-5 recovery evidence"
                )
        expected_order: list[tuple[str, int | None]] = []
        if marker_records:
            expected_order.append(("integral_plan_marker", None))
        observation_keys = {
            (int(value["rung_index"]), int(value["slot"])) for value in observation_records
        }
        rung_indexes = {int(value["rung_index"]) for value in rung_records}
        recovery_indexes = {int(value["rung_index"]) for value in recovery_records}
        recovery_rest_indexes = {int(value["rung_index"]) for value in recovery_rest_records}
        ordinary_rest_keys = {
            (int(value["rung_index"]), int(value["slot"]))
            for _raw, value in nonterminal
            if value["record_kind"] == "rest_selection" and int(value["owner"]) in (3, 4, 6)
        }
        recovery_selection_indexes = {
            int(value["rung_index"])
            for _raw, value in nonterminal
            if value["record_kind"] == "rest_selection" and int(value["owner"]) == 5
        }
        if schema_revision >= 5 and rest_boundary is not None:
            boundary_key = (
                int(rest_boundary[1]["positive_rung_index"]) + 1,
                int(rest_boundary[1]["slot"]),
            )
        else:
            boundary_key = None
        all_rung_indexes = (
            rung_indexes | recovery_indexes | {rung_index for rung_index, _slot in observation_keys}
        )
        if boundary_key is not None:
            all_rung_indexes.add(boundary_key[0])
        for rung_index in sorted(all_rung_indexes):
            for slot in range(8):
                if (rung_index, slot) in observation_keys:
                    expected_order.append(("integral_observation", rung_index))
                    if (rung_index, slot) in ordinary_rest_keys:
                        expected_order.append(("rest_selection", rung_index))
                    if (
                        schema_revision >= 3
                        and (rung_index, slot) != terminal_observation_without_current
                    ):
                        expected_order.append(("velocity_primitive_current", rung_index))
                elif boundary_key == (rung_index, slot):
                    expected_order.append(("velocity_primitive_current", rung_index))
            if rung_index in rung_indexes:
                expected_order.append(("integral_rung", rung_index))
            if rung_index in recovery_selection_indexes:
                expected_order.append(("rest_selection", rung_index))
            if rung_index in recovery_rest_indexes:
                expected_order.append(("recovery_rest", rung_index))
            if rung_index in recovery_indexes:
                expected_order.append(("rung_origin_recovery", rung_index))
        observed_order = [
            (str(value["record_kind"]), int(value["rung_index"]) if "rung_index" in value else None)
            for _raw, value in nonterminal
        ]
        if observed_order[: len(expected_order)] != expected_order:
            mismatches.append("integral recovery causal order differs from the fixed schedule")
    rung_results: dict[int, list[dict[str, object]]] = {}
    for record in rung_records:
        rung_index = int(record["rung_index"])
        if rung_index in rung_results:
            mismatches.append(f"duplicate integral rung {rung_index}")
            continue
        if not 0 <= rung_index < len(expected_gains):
            mismatches.append(f"integral rung is out of range: {rung_index}")
            continue
        # The response horizon is a combined-workflow field, not a schema-9
        # field. `integral_sweep.rs:2358` computes `(tau_us, horizon_class)`
        # only `if data.plan.combined_workflow`, leaving both zero otherwise,
        # and sets `moving_horizon_us` to the plan's stroke only under the same
        # condition; `encode_rung` (`integral_reporting.rs:581`) then omits all
        # three fields from the wire entirely when `moving_horizon_us == 0`.
        # A schema-14 breakaway rung is non-combined, so it legitimately
        # carries none of them and the decoder reports `None` for each.
        #
        # The gate is not a silent skip: a non-combined rung is required to
        # carry none of the three, so a build that starts emitting them
        # outside a combined workflow is reported rather than ignored. The
        # decoder makes that observable -- it reads the fields only from the
        # 83-byte rung payload, never the 70-byte one (`protocol.py:1203`).
        rung_combined_workflow = stage_c_combined_workflow(schema_revision)
        if schema_revision >= 9 and rung_combined_workflow:
            i_raw = int(expected_gains[rung_index])
            expected_tau_us = (
                0
                if i_raw == 0
                else velocity_i_tau_us(
                    i_raw,
                    int(plan["pwm_hz"]),
                    capture_build_features(plan, stage="c"),
                )
            )
            expected_horizon_class = (
                0 if i_raw == 0 else 1 if expected_tau_us > int(plan["moving_stroke_us"]) else 2
            )
            _compare_fields(
                {
                    "tau_us": expected_tau_us,
                    "moving_horizon_us": int(plan["moving_stroke_us"]),
                    "horizon_class": expected_horizon_class,
                },
                record,
                ("tau_us", "moving_horizon_us", "horizon_class"),
                mismatches,
                f"rung {rung_index} response horizon",
            )
        elif not rung_combined_workflow and any(
            field in record for field in ("tau_us", "moving_horizon_us", "horizon_class")
        ):
            mismatches.append(
                f"rung {rung_index} reports a response horizon outside a combined workflow"
            )
        expected_directions: list[dict[str, object]] = []
        for direction in range(2):
            keys = [
                (rung_index, slot_for_observation(plan_slot_order(plan), repeat, direction))
                for repeat in range(4)
            ]
            if any(key not in reconstructed for key in keys):
                mismatches.append(
                    f"integral rung {rung_index} lacks four direction {direction} observations"
                )
                continue
            observations = [reconstructed[key] for key in keys]
            expected = _direction_consensus(observations)
            expected["direction"] = direction
            expected["kind"] = (
                0 if rung_index == 0 else 2 if rung_index == len(expected_gains) - 1 else 1
            )
            expected["observation_classes"] = [item["classification"] for item in observations]
            reported = dict(record["directions"][direction])  # type: ignore[index]
            reported["components"] = [
                tuple(component)
                for component in reported["components"]  # type: ignore[union-attr]
            ]
            _compare_fields(
                expected,
                reported,
                (
                    "direction",
                    "kind",
                    "classification",
                    "components",
                    "usable_mask",
                    "included_mask",
                    "observation_classes",
                ),
                mismatches,
                f"rung {rung_index} direction {direction}",
            )
            if int(reported["component_count"]) != len(expected["components"]):
                mismatches.append(
                    f"rung {rung_index} direction {direction} component count differs"
                )
            expected_directions.append(expected)
        if len(expected_directions) == 2:
            rung_results[rung_index] = expected_directions

    positive_count = int(plan["positive_rung_count"])
    outcome = int(terminal["outcome"])
    cause = int(terminal["cause"])
    faulted = outcome == 3
    opening, bookend, positive, eligible_masks = _retained_stage_c_evidence(
        rung_results,
        positive_count,
    )
    terminal_eligible_masks = list(terminal["eligible_masks"])
    terminal_opening_mask = int(terminal.get("opening_available_mask", -1))
    terminal_bookend_mask = int(terminal.get("bookend_available_mask", -1))
    # A fault terminal reports one of two evidence shapes, and which one it is
    # cannot be read back off the terminal.
    #
    # Reset: `integral_fault_terminal` (`integral_sweep.rs:1001`) and
    # `evidence_terminal`'s materialization-error arm (`:1143`) both build the
    # terminal from `IntegralRunEvidence::new([ExactRateInterval::EMPTY; 2])`,
    # which zeroes `eligible_masks`, `opening`, and `bookend`
    # (`integral_evidence.rs:165-175`).
    #
    # Preserved: `evidence_terminal`'s assessment-error arm (`:1211`) passes
    # the materialized evidence through unchanged, and every dispatch-level
    # promotion overwrites only `outcome`/`cause` on an already-built terminal
    # -- `apply_stage_c_terminal` (`commission_dispatch/stage_c.rs:273`, `:283`,
    # `:338`, `:358`) and the disarm-failure paths in
    # `commission_dispatch/mod.rs` (`:606`, `:1417`, `:1430`, `:1467`). A
    # complete, fully sufficient run whose dispatcher found no Stage-C
    # authority is faulted this way with all of its evidence intact.
    #
    # Both a reset path and a preserved path report cause 4, so demanding the
    # reset shape rejected real captures. The rule is instead: reset evidence
    # is accepted unexamined, and any other evidence must agree with the
    # replay, exactly as it must for a run that did not fault.
    fault_evidence_reset = (
        faulted
        and terminal_eligible_masks == [0, 0]
        and (schema_revision < 4 or (terminal_opening_mask == 0 and terminal_bookend_mask == 0))
    )
    if not fault_evidence_reset and eligible_masks != terminal_eligible_masks:
        mismatches.append(
            f"terminal eligible masks: firmware={terminal['eligible_masks']} offline="
            f"{eligible_masks}"
        )
    if int(terminal["emitted_observations"]) != len(observation_records):
        mismatches.append("terminal emitted-observation count differs from capture")
    if int(terminal["emitted_rungs"]) != len(rung_records):
        mismatches.append("terminal emitted-rung count differs from capture")
    opening_available_mask = sum(
        (1 << direction) for direction, interval in enumerate(opening) if interval is not None
    )
    bookend_available_mask = sum(
        (1 << direction) for direction, interval in enumerate(bookend) if interval is not None
    )
    if schema_revision >= 4 and not fault_evidence_reset:
        # Same rule as the eligible masks above: the anchor availability masks
        # are folded from the very evidence struct the two reset paths empty
        # (`replies.rs:1420-1438`), so they are compared unless the terminal
        # reported the reset shape.
        if terminal_opening_mask != opening_available_mask:
            mismatches.append("terminal opening-anchor availability mask differs from replay")
        if terminal_bookend_mask != bookend_available_mask:
            mismatches.append("terminal bookend availability mask differs from replay")
    sufficient = _stage_c_run_sufficient(
        schema_revision,
        opening,
        bookend,
        eligible_masks,
    )
    if schema_revision >= _FIRST_COMBINED_STAGE_C_SCHEMA_REVISION:
        # `plan.combined_workflow` (integral_plan.rs:508) is exactly
        # `combined_selected_response.is_some()`, and which workflow built the
        # plan is fixed by the schema revision: 8 through 13 are combined, 14
        # is breakaway. Schema >= 8 alone therefore does *not* mean combined --
        # a schema-14 breakaway run is a standalone Stage C and must not be
        # held to combined-terminal semantics -- but neither does the presence
        # of the field decide it. Taking the workflow from the schema, after
        # `_validate_plan` has confirmed the plan's own field agrees, is what
        # stops a schema-8-to-13 capture that omits the selected response from
        # electing standalone semantics for itself.
        # Every one of these is derived from `plan.combined_workflow` in the
        # firmware and must be checked against the plan independently: the
        # recovery_flags workflow bit and its target-status bits (both
        # inside the same byte), the decoded combined tail's mere presence
        # (`protocol.py` only sets `terminal["combined_workflow"]` when the
        # wire payload carries the extra final_p/target_status/
        # target_terminus block -- a 66-byte terminal -- independently of
        # what the recovery_flags byte says), and, for a sufficient run,
        # the outcome itself: `integral_sweep.rs:1174-1177` reports
        # `Complete` (1) only when `plan.combined_workflow`, and
        # `CompleteCandidate` (0) otherwise. A terminal can lie about any
        # one of these independently of the others, so each gets its own
        # check and its own message. The outcome rule is `not faulted`-gated
        # (like every other sibling check in this function that reads
        # firmware-reported evidence): `Fault` is reached via
        # `integral_fault_terminal` and the `Err(_)` arm of `evidence_terminal`
        # in `integral_sweep.rs`, neither of which ever consults
        # `combined_workflow` -- so a faulted run's outcome carries no
        # schedule information, and `sufficient` here is the *replayed*
        # pre-fault evidence, which can be sufficient even though the
        # firmware's own fault path resets its reported evidence to empty.
        # Recorded firmware output (a real standalone capture, outcome=3,
        # cause=4/`INTEGRAL_CAUSE_INTEGRITY`, sufficient replay) hit exactly
        # this gap before the guard was added.
        plan_combined_workflow = stage_c_combined_workflow(schema_revision)
        recovery_flags = int(terminal.get("recovery_flags", 0))
        terminal_combined_workflow = bool(recovery_flags & _TERMINAL_COMBINED_WORKFLOW)
        if terminal_combined_workflow != plan_combined_workflow:
            mismatches.append("terminal combined-workflow flag differs from the plan")
        terminal_combined_tail = bool(terminal.get("combined_workflow", False))
        if terminal_combined_tail != plan_combined_workflow:
            mismatches.append("terminal combined tail differs from the plan")
        if not faulted and sufficient and not plan_combined_workflow and outcome != 0:
            mismatches.append("standalone sufficient run falsely reports the combined outcome")
        if plan_combined_workflow:
            # `terminal_target_status` (`integral_reporting.rs:51-76`) reads
            # the terminal's *own* `report.evidence.eligible_masks` and
            # `report.evidence.current_terminus`. Two fault paths --
            # `integral_fault_terminal` (`integral_sweep.rs:1001`) and
            # `evidence_terminal`'s materialization-error arm (`:1143`) --
            # build their terminal from `IntegralRunEvidence::new(...)`
            # (`integral_evidence.rs:167-175`), which resets both to empty.
            # Not every fault does: the assessment-error arm
            # (`integral_sweep.rs:1211`) passes the materialized `evidence`
            # through unchanged, and `commission_dispatch/stage_c.rs`
            # (`:273`, `:283`, `:338`, `:358`) can promote an already-built,
            # evidence-bearing terminal to `Fault` after the fact. So an
            # evidence-bearing fault terminal is reachable, and these checks
            # must read the terminal's own reported evidence rather than
            # assuming it is empty. That is what they do below, which is
            # correct for both shapes.
            # `target_terminus` is *not* part of
            # `IntegralRunEvidence`: `terminal_from_data` sets
            # `rest_boundary_rung_plus_one` from `data.target_terminus`
            # (`integral_sweep.rs:973`), so it survives the fault. A combined
            # run that faults after reaching a target therefore reports
            # `TargetReachedSparse` (1) -- terminus present, masks empty --
            # where the *replayed* masks would say `TargetReached` (0).
            # Feeding the replayed masks in here made both this check and the
            # encoded-flag check below reject that honest terminal. Every
            # input is now taken from the terminal's own reported evidence,
            # which makes both checks pure terminal self-consistency checks:
            # the terminal's evidence is separately compared against the
            # replay above (`terminal eligible masks`, guarded by `faulted`),
            # so a non-faulted divergence is still reported there and nothing
            # is lost.
            target_terminus = terminal.get("target_terminus")
            expected_status = combined_target_status(
                target_terminus=int(target_terminus) if target_terminus is not None else None,
                eligible_masks=terminal_eligible_masks,
                current_terminus=(
                    int(terminal["current_terminus"])
                    if terminal["current_terminus"] is not None
                    else None
                ),
                emitted_rungs=int(terminal["emitted_rungs"]),
                expected_rungs=int(terminal["expected_rungs"]),
            )
            expected_target_status = 0xFF if expected_status is None else expected_status
            if int(terminal.get("target_status", -1)) != expected_target_status:
                mismatches.append("combined target status differs from causal ladder evidence")
            encoded_status = (recovery_flags & _TERMINAL_COMBINED_TARGET_MASK) >> 4
            expected_encoded_status = (
                0 if expected_target_status == 0xFF else expected_target_status + 1
            )
            if encoded_status != expected_encoded_status:
                mismatches.append("combined terminal target flags differ from canonical status")
            if not faulted and sufficient and outcome != 1:
                mismatches.append("combined sufficient run did not complete without retention")
    if cause in (9, 10) and rest_boundary is None:
        mismatches.append("rest-boundary terminal cause lacks its canonical boundary record")
    if rest_boundary is not None and not faulted:
        if int(rest_boundary[1]["cleanup_confirmed"]) != 1:
            mismatches.append("non-fault rest-boundary terminal lacks confirmed cleanup")
        if cause == 9 and (outcome != 2 or sufficient):
            mismatches.append("cause 9 requires an insufficient Inconclusive run")
        elif cause == 10 and (outcome not in (0, 1) or not sufficient):
            mismatches.append("cause 10 requires a sufficient candidate or complete run")
        elif cause not in (9, 10):
            mismatches.append("non-fault rest-boundary terminal uses the wrong cause")
    if not faulted and sufficient and outcome not in (0, 1) and not (outcome == 2 and cause == 2):
        mismatches.append("sufficient integral curve has an incompatible terminal outcome")
    if not sufficient and outcome not in (2, 3):
        mismatches.append("insufficient integral curve has an incompatible terminal outcome")
    caveats = (
        ["fault terminal is authoritative; run evidence is the derived pre-fault prefix"]
        if faulted
        else []
    )
    if historical_attribution_caveat:
        caveats.append(
            "schema-5 recovery peak preserves the recorded hard-limit attribution caveat"
        )
    result = {
        "run_sequence": run_sequence,
        "plan": dict(plan),
        "terminal": dict(terminal),
        "digest": digest,
        "complete": sufficient,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": caveats,
        "historical_attribution_caveat": historical_attribution_caveat,
        "recoveries": [dict(record) for record in recovery_records],
        "recovery_rest_diagnostics": [dict(record) for record in recovery_rest_records],
        "recovery_rest_distribution": summarize_recovery_rest_diagnostics(recovery_rest_records),
        "current_evidence": [dict(record) for record in current_records],
        # The independently reconstructed observations, not firmware's reported
        # ones: every field above was compared against these, so emitting the
        # replay's own values keeps the output an audit rather than an echo.
        #
        # Aggregates cannot replace these. `run_evidence.positive` is indexed
        # [direction][rung] and pools the slots, while a stroke's outbound or
        # inbound role is carried by its slot position within the rung. Anything
        # asking how role and direction differ needs the per-slot rate.
        "observations": [dict(reconstructed[key]) for key in sorted(reconstructed)],
        "rest_boundary": dict(rest_boundary[1]) if rest_boundary is not None else None,
        "limit_evidence_capabilities": sorted(
            {int(record["capability"]) for record in current_records}
        ),
        "run_evidence": {
            "opening": opening,
            "positive": positive,
            "eligible_masks": eligible_masks,
            "bookend": bookend,
            "current_terminus": terminal["current_terminus"],
            "directional_validity": list(plan.get("directional_validity", [])),
            "reduced_margin": bool(plan.get("reduced_margin", False)),
            "rest_boundary": (
                {
                    "positive_rung_index": int(rest_boundary[1]["positive_rung_index"]),
                    "slot": int(rest_boundary[1]["slot"]),
                }
                if rest_boundary is not None
                else None
            ),
        },
    }
    if historical_attribution_caveat:
        result["projected_schema_six_recovery_outcome"] = "unknown"
    return result


def audit_integral_reproduction_pair(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Audit the firmware terminal decision across two sufficient Stage-C runs."""
    mismatches: list[str] = []
    previous_plan = previous.get("plan")
    current_plan = current.get("plan")
    previous_evidence = previous.get("run_evidence")
    current_evidence = current.get("run_evidence")
    current_terminal = current.get("terminal")
    if not all(
        isinstance(value, dict)
        for value in (
            previous_plan,
            current_plan,
            previous_evidence,
            current_evidence,
            current_terminal,
        )
    ):
        return {
            "matches_firmware": False,
            "mismatches": ["complete plan, terminal, and run evidence are required for pairing"],
        }
    if previous_plan["plan_digest"] != current_plan["plan_digest"]:  # type: ignore[index]
        return {
            "matches_firmware": False,
            "mismatches": ["integral reproduction plans do not match"],
        }
    positive_count = int(current_plan["positive_rung_count"])  # type: ignore[index]
    comparison = compare_integral_runs(
        previous_evidence,  # type: ignore[arg-type]
        current_evidence,  # type: ignore[arg-type]
        positive_rung_count=positive_count,
    )
    expected_outcome = 1 if comparison["reproduces"] else 2
    current_boundary = current_evidence.get("rest_boundary")  # type: ignore[union-attr]
    expected_cause = (
        10
        if comparison["reproduces"] and current_boundary is not None
        else (0 if comparison["reproduces"] else 2)
    )
    if int(current_terminal["outcome"]) != expected_outcome:  # type: ignore[index]
        mismatches.append("terminal outcome differs from integral reproduction comparison")
    if int(current_terminal["cause"]) != expected_cause:  # type: ignore[index]
        mismatches.append("terminal cause differs from integral reproduction comparison")
    return {
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "comparison": comparison,
        "previous_rest_boundary": previous_evidence.get("rest_boundary"),  # type: ignore[union-attr]
        "current_rest_boundary": current_boundary,
        "rest_boundary_equal": (
            previous_evidence.get("rest_boundary") == current_boundary  # type: ignore[union-attr]
        ),
    }
