"""Independent velocity-integral response geometry and reproduction audit."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .velocity_measurement import (
    VelocityObservation,
    _confidence_half_width,
    _encode_rate_values,
    _stationary_suffix,
    _upper_bound_signed,
    _upper_bound_unsigned,
    audit_rung_origin_recovery,
    plan_rung_origin_recovery,
)


_MICROS_PER_SECOND = 1_000_000
_LEGACY_Q8_8_BUILD_REVISION = 0x75536758
_CURRENT_Q4_12_BUILD_REVISION = 0x05E21F75
_LEGACY_Q8_8_I_NUMERATOR = 65_536
_CURRENT_Q4_12_I_NUMERATOR = 1_048_576
_I_MAX_POSITIVE_RUNGS = 30
_I_CONFIDENCE_FAMILY_MAX = 128
_I_OBSERVATIONS_PER_DIRECTION = 4
_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_FNV1A64_OFFSET = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3


class UnsupportedIntegralRepresentationError(ValueError):
    """Raised when a firmware build has no declared integral-gain domain."""


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


def _fnv1a64(data: bytes, initial: int = _FNV1A64_OFFSET) -> int:
    value = initial
    for byte in data:
        value = ((value ^ byte) * _FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _append_integer(buffer: bytearray, value: int, size: int, *, signed: bool = False) -> None:
    buffer.extend(value.to_bytes(size, "little", signed=signed))


def integral_plan_digest(plan: dict[str, object]) -> int:
    """Recompute the firmware Stage-C plan digest over every immutable input."""
    encoded = bytearray()
    _append_integer(encoded, int(plan["build_revision"]), 4)
    _append_integer(encoded, int(plan["schema_revision"]), 2)
    _append_integer(encoded, int(plan["channel"]), 1)
    _append_integer(encoded, int(plan["stage_b_plan_digest"]), 8)
    _append_integer(encoded, int(plan["final_p"]), 2)
    for membership in plan["directional_memberships"]:  # type: ignore[union-attr]
        _append_integer(encoded, int(membership), 4)
    _append_integer(encoded, int(plan["joint_membership"]), 4)
    if int(plan["schema_revision"]) >= 4:
        for validity in plan["directional_validity"]:  # type: ignore[union-attr]
            _append_integer(encoded, int(validity), 1)
        _append_integer(encoded, int(bool(plan["reduced_margin"])), 1)
    for interval in plan["directional_pooled_intervals"]:  # type: ignore[union-attr]
        _append_integer(encoded, int(interval[0]), 4, signed=True)
        _append_integer(encoded, int(interval[1]), 4, signed=True)
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
    for name in (
        "moving_stroke_us",
        "zero_settle_us",
        "analysis_budget_us",
        "nominal_workflow_ms",
        "maximum_workflow_ms",
    ):
        _append_integer(encoded, int(plan[name]), 4)
    if int(plan["schema_revision"]) >= 2:
        for name in (
            "recovery_origin_band_counts",
            "recovery_nominal_slot_us",
            "recovery_maximum_slot_us",
        ):
            _append_integer(encoded, int(plan[name]), 4)
        _append_integer(encoded, int(plan["recovery_slot_count"]), 1)
    _append_integer(encoded, int(plan["hard_torque_limit"]), 2)
    _append_integer(encoded, int(plan["usable_torque_limit"]), 2)
    if int(plan["schema_revision"]) >= 3:
        _append_integer(encoded, int(plan["limit_evidence_capability"]), 1)
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
    return _fnv1a64(bytes(encoded))


def velocity_i_tau_us(i_raw: int, pwm_hz: int, build_revision: int | None) -> int:
    """Return the advanced-PI time constant with firmware-equivalent ceiling."""
    if i_raw <= 0 or pwm_hz <= 0:
        raise ValueError("velocity I and PWM frequency must be positive")
    numerator = integral_numerator_for_build(build_revision) * _MICROS_PER_SECOND
    denominator = i_raw * pwm_hz
    return (numerator + denominator - 1) // denominator


def _largest_i_for_tau(
    pwm_hz: int,
    minimum_tau_us: int,
    maximum_i: int,
    build_revision: int | None,
) -> int:
    if pwm_hz <= 0 or minimum_tau_us <= 0 or maximum_i <= 0:
        raise ValueError("integral ladder geometry must be positive")
    largest = 0
    for candidate in range(1, maximum_i + 1):
        if velocity_i_tau_us(candidate, pwm_hz, build_revision) < minimum_tau_us:
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
    build_revision: int | None,
) -> dict[str, object]:
    """Reconstruct the cap-only whole-octave positive-I ladder."""
    if schema_revision not in range(1, 6):
        raise ValueError("unsupported Stage-C evidence schema")
    if shortest_suffix_us < 2:
        raise ValueError("shortest suffix must contain at least two microseconds")
    numerator = integral_numerator_for_build(build_revision)
    if schema_revision >= 5:
        i_start = 1
    elif schema_revision >= 4:
        i_start = numerator // _LEGACY_Q8_8_I_NUMERATOR
    else:
        i_start = _largest_i_for_tau(
            pwm_hz,
            moving_stroke_us,
            maximum_i,
            build_revision,
        )
    i_top_cap = _largest_i_for_tau(
        pwm_hz,
        shortest_suffix_us // 2,
        maximum_i,
        build_revision,
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
    family_size = (len(rungs) + 2) * _I_OBSERVATIONS_PER_DIRECTION
    if len(rungs) < 3:
        raise ValueError("integral ladder requires at least three positive rungs")
    if len(rungs) > _I_MAX_POSITIVE_RUNGS or family_size > _I_CONFIDENCE_FAMILY_MAX:
        raise ValueError("integral ladder exceeds the simultaneous-confidence family")
    return {"i_start": i_start, "i_top_cap": i_top_cap, "positive_i": rungs}


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
    direction = 0 if slot % 2 == 0 else 1
    stationarity = _stationary_suffix(raw[-256:])
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
    build_revision = plan.get("build_revision")
    integral_numerator_for_build(build_revision if isinstance(build_revision, int) else None)
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
    expected_target = (int(plan["planned_velocity_mrev_s"]) * 60 + 500) // 1_000
    if expected_target != int(plan["target_velocity_rpm"]):
        mismatches.append("plan target RPM differs from nearest encoder-domain conversion")
    try:
        ladder = derive_integral_ladder(
            int(plan["pwm_hz"]),
            int(plan["moving_stroke_us"]),
            64_000,
            32_767,
            schema_revision=int(plan["schema_revision"]),
            build_revision=plan.get("build_revision"),
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
    schema_revision = int(plan["schema_revision"])
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
        for name, expected in (
            ("recovery_origin_band_counts", recovery["origin_band_counts"]),
            ("recovery_nominal_slot_us", recovery["nominal_slot_us"]),
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
            "rung_origin_recovery",
            "velocity_primitive_current",
        )
    ]
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
        elif int(boundary["direction"]) != (1 if slot % 2 == 0 else -1):
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
        if int(record["direction"]) != slot % 2:
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
    for record in current_records:
        if int(record["stage"]) != 1:
            mismatches.append("Stage C current evidence carries the wrong stage")
        if int(record["clamp_readback"]) != int(record["clamp_limit"]):
            mismatches.append("Stage C current evidence clamp readback differs")
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
        if schema_revision >= 5 and rest_boundary is not None:
            expected_recovery_rungs = sorted(
                {int(record["rung_index"]) for record in rung_records}
                | (
                    {int(rest_boundary[1]["positive_rung_index"]) + 1}
                    if int(terminal["cause"]) == 10
                    else set()
                )
            )
        else:
            fully_acquired_rungs = int(terminal["emitted_observations"]) // 8
            expected_recovery_rungs = list(range(fully_acquired_rungs))
        if recovery_rungs != expected_recovery_rungs:
            mismatches.append(
                "integral recovery rungs differ from fully acquired rung sequence: "
                f"firmware={recovery_rungs} offline={expected_recovery_rungs}"
            )
        for record in recovery_records:
            if int(record["stage"]) != 1:
                mismatches.append("integral recovery record carries the wrong stage")
            recovery_audit = audit_rung_origin_recovery(record, plan)
            mismatches.extend(
                f"recovery rung {record['rung_index']}: {item}"
                for item in recovery_audit["mismatches"]
            )
        expected_order: list[tuple[str, int | None]] = []
        if marker_records:
            expected_order.append(("integral_plan_marker", None))
        if schema_revision >= 5 and rest_boundary is not None:
            observation_keys = {
                (int(value["rung_index"]), int(value["slot"])) for value in observation_records
            }
            rung_indexes = {int(value["rung_index"]) for value in rung_records}
            recovery_indexes = {int(value["rung_index"]) for value in recovery_records}
            boundary_key = (
                int(rest_boundary[1]["positive_rung_index"]) + 1,
                int(rest_boundary[1]["slot"]),
            )
            for rung_index in sorted(rung_indexes | recovery_indexes | {boundary_key[0]}):
                for slot in range(8):
                    if (rung_index, slot) in observation_keys:
                        expected_order.append(("integral_observation", rung_index))
                        expected_order.append(("velocity_primitive_current", rung_index))
                    elif (rung_index, slot) == boundary_key:
                        expected_order.append(("velocity_primitive_current", rung_index))
                if rung_index in rung_indexes:
                    expected_order.append(("integral_rung", rung_index))
                if rung_index in recovery_indexes:
                    expected_order.append(("rung_origin_recovery", rung_index))
        else:
            for rung_index in expected_recovery_rungs:
                for _slot in range(8):
                    expected_order.append(("integral_observation", rung_index))
                    if schema_revision >= 3:
                        expected_order.append(("velocity_primitive_current", rung_index))
                expected_order.append(("integral_rung", rung_index))
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
        expected_directions: list[dict[str, object]] = []
        for direction in range(2):
            keys = [(rung_index, direction + 2 * repeat) for repeat in range(4)]
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
    positive: list[list[tuple[int, int] | None]] = [[None] * positive_count for _ in range(2)]
    eligible_masks = [0, 0]
    opening: list[tuple[int, int] | None] = [None, None]
    bookend: list[tuple[int, int] | None] = [None, None]
    for rung_index, directions in rung_results.items():
        for direction, result in enumerate(directions):
            if int(result["classification"]) != 0:
                continue
            interval = tuple(result["components"][0])  # type: ignore[index]
            if rung_index == 0:
                opening[direction] = interval
            elif rung_index == positive_count + 1:
                bookend[direction] = interval
            else:
                positive[direction][rung_index - 1] = interval
                eligible_masks[direction] |= 1 << (rung_index - 1)
    terminal_eligible_masks = list(terminal["eligible_masks"])
    if faulted and terminal_eligible_masks != [0, 0]:
        mismatches.append("fault terminal retained nonempty eligible masks")
    elif not faulted and eligible_masks != terminal_eligible_masks:
        mismatches.append(
            f"terminal eligible masks: firmware={terminal['eligible_masks']} offline={eligible_masks}"
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
    if schema_revision >= 4:
        terminal_opening_mask = int(terminal.get("opening_available_mask", -1))
        terminal_bookend_mask = int(terminal.get("bookend_available_mask", -1))
        if faulted and (terminal_opening_mask != 0 or terminal_bookend_mask != 0):
            mismatches.append("fault terminal retained nonempty anchor availability")
        elif not faulted and terminal_opening_mask != opening_available_mask:
            mismatches.append("terminal opening-anchor availability mask differs from replay")
        if not faulted and terminal_bookend_mask != bookend_available_mask:
            mismatches.append("terminal bookend availability mask differs from replay")
    sufficient = _stage_c_run_sufficient(
        schema_revision,
        opening,
        bookend,
        eligible_masks,
    )
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
    return {
        "run_sequence": run_sequence,
        "plan": dict(plan),
        "terminal": dict(terminal),
        "digest": digest,
        "complete": sufficient,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": (
            ["fault terminal is authoritative; run evidence is the derived pre-fault prefix"]
            if faulted
            else []
        ),
        "recoveries": [dict(record) for record in recovery_records],
        "current_evidence": [dict(record) for record in current_records],
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
