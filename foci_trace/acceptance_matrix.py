"""Bit-exact replay and derived comparison for velocity confidence matrices."""

from __future__ import annotations

from typing import Any

from .velocity_integral import depth_three_consensus
from .velocity_measurement import (
    audit_recovery_rest_geometry,
    audit_scored_rest_grammar,
    summarize_recovery_rest_diagnostics,
)

_FNV1A64_OFFSET = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3
_FULL_AMPLITUDE_MASK = 0b1_1111
_SLOT_TRANSITION_ALLOWANCE_US = 20_000


def _fnv1a64(data: bytes, initial: int = _FNV1A64_OFFSET) -> int:
    value = initial
    for byte in data:
        value = ((value ^ byte) * _FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def _append(buffer: bytearray, value: int, size: int, *, signed: bool = False) -> None:
    buffer.extend(value.to_bytes(size, "little", signed=signed))


def robust_rate_location(means: list[int | None]) -> int | None:
    """Return firmware's signed median or inner-pair midpoint."""
    values = sorted(value for value in means if value is not None)
    if len(values) == 3:
        return values[1]
    if len(values) == 4:
        difference = values[2] - values[1]
        half = abs(difference) // 2
        return values[1] + (-half if difference < 0 else half)
    return None


def matrix_plan_digest(plan: dict[str, Any]) -> int:
    """Recompute the immutable matrix-plan digest in firmware field order."""
    encoded = bytearray()
    _append(encoded, int(plan["schema_revision"]), 2)
    _append(encoded, int(plan["channel"]), 1)
    # A newer capture carries build_features (u16) where a legacy capture
    # carried build_revision (u32); the firmware acceptance digest hashes the
    # same field in place.
    if "build_features" in plan:
        _append(encoded, int(plan["build_features"]), 2)
    else:
        _append(encoded, int(plan["build_revision"]), 4)
    for name, size in (
        ("acceptance_schema_revision", 2),
        ("acceptance_plan_digest", 8),
        ("order", 1),
    ):
        _append(encoded, int(plan[name]), size)
    for target in plan["target_rpm"]:
        _append(encoded, int(target), 2, signed=True)
    for name, size in (
        ("selected_p", 2),
        ("selected_i", 2),
        ("qualifier_bits", 2),
        ("pwm_hz", 4),
        ("encoder_counts_per_rev", 4),
        ("amplitude_count", 1),
        ("expected_observations", 2),
        ("family_size", 2),
        ("moving_stroke_us", 4),
        ("zero_settle_us", 4),
        ("analysis_budget_us", 4),
    ):
        _append(encoded, int(plan[name]), size)
    _append(encoded, _SLOT_TRANSITION_ALLOWANCE_US, 4)
    for name, size in (
        ("nominal_workflow_ms", 4),
        ("maximum_workflow_ms", 4),
        ("recovery_origin_band_counts", 4),
    ):
        _append(encoded, int(plan[name]), size)
    if int(plan["schema_revision"]) >= 2:
        for lower_rate_q in plan["recovery_lower_rate_q"]:
            _append(encoded, int(lower_rate_q), 8)
    for name, size in (
        ("recovery_nominal_slot_us", 4),
        ("recovery_maximum_slot_us", 4),
        ("recovery_slot_count", 1),
        ("hard_torque_limit", 2),
        ("usable_torque_limit", 2),
        ("limit_evidence_capability", 1),
        ("recovery_quantization_exposed", 1),
        ("probe_constrained_test_point", 1),
        ("requested_velocity_mrev_s", 4),
        ("machine_velocity_ceiling_mrev_s", 4),
        ("requested_velocity_source", 1),
        ("max_stroke_travel_mrev", 4),
        ("settle_travel_reserve_mrev", 4),
        ("negative_position_headroom_mrev", 4),
        ("positive_position_headroom_mrev", 4),
        ("max_duration_ms", 2),
    ):
        _append(encoded, int(plan[name]), size)
    return _fnv1a64(bytes(encoded))


def assemble_matrix_plan(fragments: list[dict[str, Any]]) -> dict[str, Any]:
    """Join the three exact plan fragments and reject identity disagreement."""
    if sorted(int(fragment["fragment"]) for fragment in fragments) != [0, 1, 2]:
        raise ValueError("matrix plan requires fragments 0, 1, and 2 exactly once")
    ordered = sorted(fragments, key=lambda fragment: int(fragment["fragment"]))
    identity = ("run_sequence", "plan_digest")
    if any(
        any(fragment[name] != ordered[0][name] for name in identity) for fragment in ordered[1:]
    ):
        raise ValueError("matrix plan fragments disagree on identity")
    plan: dict[str, Any] = {}
    for fragment in ordered:
        plan.update(fragment)
    plan["record_kind"] = "acceptance_matrix_plan"
    plan["fragment"] = None
    return plan


def reconstruct_direction(
    observations: list[dict[str, Any]],
    *,
    signed_target_rate_q: int,
    precision_floor_candidate: bool,
) -> dict[str, Any]:
    """Fold one direction's four intrinsic observation records."""
    by_index: dict[int, dict[str, Any]] = {}
    for observation in observations:
        index = int(observation["slot"]) // 2
        if index in by_index or not 0 <= index < 4:
            raise ValueError("matrix direction has duplicate or invalid observation slot")
        by_index[index] = observation
    collected_mask = sum(1 << index for index in by_index)
    classes = [
        int(by_index[index]["classification"]) if index in by_index else None for index in range(4)
    ]
    intervals = [
        (
            int(by_index[index]["rate_low"]),
            int(by_index[index]["rate_high"]),
        )
        if index in by_index and classes[index] == 0
        else None
        for index in range(4)
    ]
    consensus = depth_three_consensus(intervals, collected_mask=collected_mask)
    fault = any(value in (9, 10, 11) for value in classes if value is not None)
    current = 3 in classes
    classification = int(consensus["classification"])
    availability = (
        5
        if fault
        else 4
        if current
        else 0
        if classification == 0
        else 2
        if classification == 1
        else 1
        if precision_floor_candidate
        else 3
    )
    response = list(consensus["components"][0]) if availability == 0 else None
    means = [
        int(by_index[index]["rate_mean"]) if index in by_index and classes[index] == 0 else None
        for index in range(4)
    ]
    deficit = (
        [signed_target_rate_q - response[1], signed_target_rate_q - response[0]]
        if response is not None
        else None
    )
    eligible_mask = int(consensus["usable_mask"])
    included_mask = int(consensus["included_mask"])
    return {
        "availability": availability,
        "classification": classification,
        "components": [list(component) for component in consensus["components"]],
        "component_count": len(consensus["components"]),
        "collected_mask": collected_mask,
        "eligible_mask": eligible_mask,
        "included_mask": included_mask,
        "signed_target_rate_q": signed_target_rate_q,
        "response_interval_q": response,
        "robust_location_q": robust_rate_location(means),
        "deficit_interval_q": deficit,
        "excluded_mask": eligible_mask & ~included_mask,
        "blocking_levels": [
            int(by_index[index].get("selected_level", 0)) if index in by_index else 0
            for index in range(4)
        ],
    }


def reconstruct_amplitude(
    observations: list[dict[str, Any]],
    report: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct both directional folds for one emitted amplitude."""
    amplitude_index = int(report["amplitude_index"])
    target_rpm = int(report["target_rpm"])
    expected_target = int(plan["target_rpm"][amplitude_index])
    if target_rpm != expected_target:
        raise ValueError("matrix amplitude target differs from immutable plan")
    target_rate_q = target_rpm * int(plan["encoder_counts_per_rev"]) * 1_000 // 60
    precision_floor = target_rpm == min(int(value) for value in plan["target_rpm"])
    return {
        "amplitude_index": amplitude_index,
        "target_rpm": target_rpm,
        "directions": [
            reconstruct_direction(
                [value for value in observations if int(value["direction"]) == direction],
                signed_target_rate_q=target_rate_q if direction == 0 else -target_rate_q,
                precision_floor_candidate=precision_floor,
            )
            for direction in range(2)
        ],
    }


def _signed_i32(value: int) -> int:
    return value - (1 << 32) if value & (1 << 31) else value


def _compact_execution_marker(plan: dict[str, Any]) -> bytes:
    marker = bytearray((0,))
    _append(marker, int(plan["run_sequence"]), 4)
    _append(marker, int(plan["plan_digest"]), 8)
    _append(marker, int(plan["selected_p"]), 2)
    top_target = max(int(value) for value in plan["target_rpm"])
    planned_velocity_mrev_s = (
        int(plan["requested_velocity_mrev_s"])
        if int(plan["schema_revision"]) >= 2
        else top_target * 1_000 // 60
    )
    _append(marker, planned_velocity_mrev_s, 4)
    _append(marker, top_target, 4, signed=True)
    _append(marker, int(plan["pwm_hz"]), 4)
    _append(marker, int(plan["selected_i"]), 2)
    _append(marker, int(plan["amplitude_count"]), 1)
    _append(marker, int(plan["family_size"]), 2)
    for name in (
        "moving_stroke_us",
        "zero_settle_us",
        "analysis_budget_us",
        "maximum_workflow_ms",
        "recovery_origin_band_counts",
        "recovery_nominal_slot_us",
        "recovery_maximum_slot_us",
    ):
        _append(marker, int(plan[name]), 4)
    _append(marker, int(plan["recovery_slot_count"]), 1)
    _append(marker, int(plan["hard_torque_limit"]), 2)
    _append(marker, int(plan["usable_torque_limit"]), 2)
    _append(marker, 1, 1)
    identity = int(plan["acceptance_plan_digest"])
    for value in (
        _signed_i32(identity & 0xFFFF_FFFF),
        _signed_i32(identity >> 32),
        int(plan["qualifier_bits"]),
        0,
    ):
        _append(marker, value, 4, signed=True)
    if int(plan["schema_revision"]) >= 2:
        for lower_rate_q in plan["recovery_lower_rate_q"]:
            _append(marker, int(lower_rate_q), 8)
    return bytes(marker)


def _compare_fields(
    expected: dict[str, Any],
    reported: dict[str, Any],
    names: tuple[str, ...],
    mismatches: list[str],
    prefix: str,
) -> None:
    for name in names:
        if expected.get(name) != reported.get(name):
            mismatches.append(
                f"{prefix} {name}: firmware={reported.get(name)!r} offline={expected.get(name)!r}"
            )


def audit_acceptance_matrix(
    evidence_records: list[tuple[bytes, dict[str, Any]]],
) -> dict[str, Any]:
    """Replay one matrix run from canonical records and verify its terminal."""
    plans = [
        value
        for _raw, value in evidence_records
        if value.get("record_kind") == "acceptance_matrix_plan"
    ]
    terminals = [
        value
        for _raw, value in evidence_records
        if value.get("record_kind") == "acceptance_matrix_terminal"
    ]
    identities = plans or terminals or [value for _raw, value in evidence_records]
    if len(plans) != 3 or len(terminals) != 1:
        return {
            "run_sequence": int(identities[0].get("run_sequence", 0)) if identities else 0,
            "plan": None,
            "terminal": dict(terminals[0]) if len(terminals) == 1 else None,
            "amplitudes": [],
            "complete": False,
            "matches_firmware": False,
            "mismatches": [
                "partial acceptance-matrix capture: three plan fragments and one "
                "terminal are required"
            ],
            "caveats": ["partial velocity confidence matrix capture"],
        }
    mismatches: list[str] = []
    plan = assemble_matrix_plan(plans)
    slot_order = int(plan.get("slot_order", 0))
    terminal = terminals[0]
    expected_duration = (63_041, 66_456) if int(plan["schema_revision"]) >= 3 else (60_541, 66_456)
    if (
        int(plan["nominal_workflow_ms"]),
        int(plan["maximum_workflow_ms"]),
    ) != expected_duration:
        mismatches.append("matrix schema and duration pair differ")
    recomputed_plan_digest = matrix_plan_digest(plan)
    if recomputed_plan_digest != int(plan["plan_digest"]):
        mismatches.append(
            f"matrix plan digest: firmware=0x{int(plan['plan_digest']):016x} "
            f"offline=0x{recomputed_plan_digest:016x}"
        )
    nonterminal = [
        (raw, value)
        for raw, value in evidence_records
        if value.get("record_kind") not in ("acceptance_matrix_plan", "acceptance_matrix_terminal")
    ]
    sequences = [
        int(value["evidence_sequence"]) if "evidence_sequence" in value else None
        for _raw, value in nonterminal
    ]
    if sequences != list(range(1, len(sequences) + 1)):
        mismatches.append(f"acceptance matrix evidence sequence is not contiguous: {sequences}")
    if (
        "evidence_sequence" in terminal
        and int(terminal["evidence_sequence"]) != len(nonterminal) + 1
    ):
        mismatches.append(
            "acceptance matrix terminal sequence does not follow the final evidence record"
        )
    mismatches.extend(
        audit_scored_rest_grammar(
            [value for _raw, value in nonterminal],
            terminal,
            schema_revision=int(plan["schema_revision"]),
            schema_floor=4,
            expected_stage=1,
            ordinary_owners=(7,),
            recovery_owners=(8,),
            observation_kind="integral_observation",
            plan_digest=int(plan["plan_digest"]),
            label="acceptance matrix",
            stabilized_recovery_rest=True,
            ordinary_rejection_tail=("velocity_primitive_current",),
        )
    )
    digest = _fnv1a64(_compact_execution_marker(plan))
    for raw, _value in nonterminal:
        digest = _fnv1a64(raw, digest)
    if digest != int(terminal["digest"]):
        mismatches.append(
            f"matrix terminal digest: firmware=0x{int(terminal['digest']):016x} "
            f"offline=0x{digest:016x}"
        )
    observation_records = [
        value for _raw, value in nonterminal if value.get("record_kind") == "integral_observation"
    ]
    amplitude_records = [
        value
        for _raw, value in nonterminal
        if value.get("record_kind") == "acceptance_matrix_amplitude"
    ]
    recovery_records = [
        value for _raw, value in nonterminal if value.get("record_kind") == "rung_origin_recovery"
    ]
    recovery_rest_records = [
        value for _raw, value in nonterminal if value.get("record_kind") == "recovery_rest"
    ]
    recovery_rest_rungs = [int(record["rung_index"]) for record in recovery_rest_records]
    if len(recovery_rest_rungs) != len(set(recovery_rest_rungs)):
        mismatches.append("matrix carries duplicate recovery rest diagnostics")
    if int(plan["schema_revision"]) < 3 and recovery_rest_records:
        mismatches.append("historical matrix carries recovery-wide rest evidence")
    for record in recovery_rest_records:
        mismatches.extend(
            f"recovery rest rung {record['rung_index']}: {item}"
            for item in audit_recovery_rest_geometry(record, plan, expected_stage=1)
        )
    for index, (_raw, record) in enumerate(nonterminal):
        if record.get("record_kind") != "recovery_rest":
            continue
        rung_index = int(record["rung_index"])
        previous_offset = 2 if int(plan["schema_revision"]) >= 4 else 1
        previous = nonterminal[index - previous_offset][1] if index >= previous_offset else {}
        if (
            previous.get("record_kind") != "acceptance_matrix_amplitude"
            or int(previous.get("amplitude_index", -1)) != rung_index
        ):
            mismatches.append("matrix recovery rest lacks its causal amplitude")
        successor = nonterminal[index + 1][1] if index + 1 < len(nonterminal) else None
        if int(record["disposition"]) == 0 and (
            successor is None
            or successor.get("record_kind") != "rung_origin_recovery"
            or int(successor.get("rung_index", -1)) != rung_index
            or int(successor.get("outcome", -1)) not in (1, 5)
        ):
            mismatches.append("confirmed matrix recovery rest lacks its moving recovery")
        if int(record["disposition"]) == 1 and (
            successor is not None or int(terminal["cause"]) != 2
        ):
            mismatches.append("unconfirmed matrix recovery rest has the wrong successor")
    if int(plan["schema_revision"]) >= 3:
        diagnostic_rungs = set(recovery_rest_rungs)
        for recovery in recovery_records:
            if (
                int(recovery["outcome"]) in (1, 5)
                and int(recovery["rung_index"]) not in diagnostic_rungs
            ):
                mismatches.append("moving matrix recovery lacks recovery rest evidence")
    amplitudes: list[dict[str, Any]] = []
    attempted = [0, 0]
    eligible = [0, 0]
    current = [0, 0]
    for reported in amplitude_records:
        index = int(reported["amplitude_index"])
        observations = [value for value in observation_records if int(value["rung_index"]) == index]
        expected = reconstruct_amplitude(observations, reported, plan)
        amplitudes.append(expected)
        bit = 1 << index
        for direction in range(2):
            expected_direction = expected["directions"][direction]
            reported_direction = reported["directions"][direction]
            _compare_fields(
                expected_direction,
                reported_direction,
                (
                    "availability",
                    "classification",
                    "components",
                    "component_count",
                    "collected_mask",
                    "eligible_mask",
                    "included_mask",
                    "signed_target_rate_q",
                    "response_interval_q",
                    "robust_location_q",
                    "deficit_interval_q",
                    "excluded_mask",
                    "blocking_levels",
                ),
                mismatches,
                f"amplitude {index} direction {direction}",
            )
            attempted[direction] |= bit
            if int(expected_direction["availability"]) == 0:
                eligible[direction] |= bit
            if int(expected_direction["availability"]) == 4:
                current[direction] |= bit
    unattempted = [_FULL_AMPLITUDE_MASK & ~mask for mask in attempted]
    expected_masks = {
        "attempted_masks": attempted,
        "eligible_masks": eligible,
        "current_terminus_masks": current,
        "unattempted_masks": unattempted,
    }
    _compare_fields(
        expected_masks,
        terminal,
        tuple(expected_masks),
        mismatches,
        "matrix terminal",
    )
    sufficient = all(mask.bit_count() >= 3 for mask in eligible)
    expected_outcome = 0 if sufficient else 1
    expected_cause = 0 if sufficient else 1
    if int(terminal["cause"]) not in (2, 7):
        _compare_fields(
            {"outcome": expected_outcome, "cause": expected_cause},
            terminal,
            ("outcome", "cause"),
            mismatches,
            "matrix terminal",
        )
    _compare_fields(
        {
            "expected_observations": int(plan["expected_observations"]),
            "emitted_observations": len(observation_records),
            "expected_amplitudes": int(plan["amplitude_count"]),
            "emitted_amplitudes": len(amplitude_records),
            "acceptance_plan_digest": int(plan["acceptance_plan_digest"]),
            "plan_digest": int(plan["plan_digest"]),
            "qualifier_bits": int(plan["qualifier_bits"]),
        },
        terminal,
        (
            "expected_observations",
            "emitted_observations",
            "expected_amplitudes",
            "emitted_amplitudes",
            "acceptance_plan_digest",
            "plan_digest",
            "qualifier_bits",
        ),
        mismatches,
        "matrix terminal",
    )
    return {
        "run_sequence": int(plan["run_sequence"]),
        "plan": plan,
        "terminal": dict(terminal),
        "slot_order": slot_order,
        "amplitudes": amplitudes,
        "recoveries": [dict(record) for record in recovery_records],
        "recovery_rest_diagnostics": [dict(record) for record in recovery_rest_records],
        "recovery_rest_distribution": summarize_recovery_rest_diagnostics(recovery_rest_records),
        "complete": int(terminal["outcome"]) == 0,
        "matches_firmware": not mismatches,
        "mismatches": mismatches,
        "caveats": [
            "foci-trace matrix replay is a decode-integrity replica, not independent "
            "selection authority"
        ],
    }


def _closed_interval_relation(previous: list[int], current: list[int]) -> tuple[bool, int]:
    if previous[1] < current[0]:
        return False, current[0] - previous[1]
    if current[1] < previous[0]:
        return False, current[1] - previous[0]
    return True, 0


def compare_matrix_runs(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare two fixed plans without converting observed spread into tolerance."""
    comparisons = []
    observed_bound = 0
    previous_by_index = {
        int(amplitude["amplitude_index"]): amplitude for amplitude in previous["amplitudes"]
    }
    current_by_index = {
        int(amplitude["amplitude_index"]): amplitude for amplitude in current["amplitudes"]
    }
    for amplitude_index in range(5):
        for direction in range(2):
            previous_attempted = bool(
                int(previous["terminal"]["attempted_masks"][direction]) & (1 << amplitude_index)
            )
            current_attempted = bool(
                int(current["terminal"]["attempted_masks"][direction]) & (1 << amplitude_index)
            )
            left = previous_by_index.get(amplitude_index)
            right = current_by_index.get(amplitude_index)
            left_direction = left["directions"][direction] if left is not None else None
            right_direction = right["directions"][direction] if right is not None else None
            if previous_attempted != current_attempted:
                comparison = {"kind": "AttemptCoverageMismatch"}
            elif (
                left_direction is None
                or right_direction is None
                or int(left_direction["availability"]) != 0
                or int(right_direction["availability"]) != 0
            ):
                comparison = {"kind": "EligibilityMismatch"}
            else:
                left_interval = list(left_direction["response_interval_q"])
                right_interval = list(right_direction["response_interval_q"])
                overlap, gap = _closed_interval_relation(left_interval, right_interval)
                displacement = int(right_direction["robust_location_q"]) - int(
                    left_direction["robust_location_q"]
                )
                observed_bound = max(observed_bound, abs(displacement))
                comparison = {
                    "kind": "PairedPrecision",
                    "previous_interval_q": left_interval,
                    "current_interval_q": right_interval,
                    "widths_q": [
                        left_interval[1] - left_interval[0],
                        right_interval[1] - right_interval[0],
                    ],
                    "overlap": overlap,
                    "signed_gap_q": gap,
                    "robust_displacement_q": displacement,
                    "response_ratios": [
                        int(left_direction["robust_location_q"])
                        / int(left_direction["signed_target_rate_q"]),
                        int(right_direction["robust_location_q"])
                        / int(right_direction["signed_target_rate_q"]),
                    ],
                }
            comparisons.append(
                {
                    "amplitude_index": amplitude_index,
                    "direction": direction,
                    **comparison,
                }
            )
    return {
        "comparisons": comparisons,
        "maximum_observed_displacement_q": observed_bound,
        "maximum_displacement_kind": "observed_bound",
    }
