import json
import struct
from fractions import Fraction
from pathlib import Path

import pytest

from foci_trace.velocity_integral import (
    _validate_plan,
    _retained_stage_c_evidence,
    audit_integral_reproduction_pair,
    audit_velocity_integral,
    classify_observation,
    combined_target_status,
    compare_integral_runs,
    delta_e,
    depth_three_consensus,
    derive_integral_ladder,
    integral_deficit_reaches_target,
    integral_numerator_for_build,
    integral_plan_digest,
    recovery_quantization_exposed,
    reconstruct_observation,
    select_transition_capable_test_point,
    stage_c_slot_transition,
    UnsupportedIntegralRepresentationError,
    velocity_i_tau_us,
)
from foci_trace.velocity_measurement import VelocityObservation, audit_rung_origin_recovery


LEGACY_Q8_8_BUILD = 0x75536758
CURRENT_Q4_12_BUILD = 0x05E21F75


def _fnv1a32(value: str) -> int:
    digest = 0x811C9DC5
    for byte in value.encode():
        digest = ((digest ^ byte) * 0x01000193) & 0xFFFFFFFF
    return digest


def _evidence(
    opening: list[tuple[int, int]],
    masks: list[int],
    offsets: list[int],
) -> dict[str, object]:
    positive: list[list[tuple[int, int] | None]] = [[None] * 6 for _ in range(2)]
    for direction in range(2):
        for rung in range(6):
            if masks[direction] & (1 << rung):
                low = opening[direction][0] + offsets[direction] + rung * 10
                positive[direction][rung] = (low, low + 4)
    return {
        "opening": opening,
        "positive": positive,
        "eligible_masks": masks,
        "bookend": [None, None],
        "current_terminus": None,
    }


def _plan(schema_revision: int = 1) -> dict[str, object]:
    plan: dict[str, object] = {
        "build_revision": LEGACY_Q8_8_BUILD,
        "schema_revision": schema_revision,
        "channel": 0,
        "stage_b_plan_digest": 0xDDEEFF0099AABBCC,
        "final_p": 1_448,
        "directional_memberships": [0x380000, 0x3C0000],
        "joint_membership": 0x380000,
        "directional_pooled_intervals": [
            [17_000_000, 18_000_000],
            [-19_000_000, -18_000_000],
        ],
        "planned_velocity_mrev_s": 4_394,
        "target_velocity_rpm": 264,
        "pwm_hz": 25_000,
        "encoder_counts_per_rev": 4_000,
        "positive_rung_count": 5,
        "positive_i": [5, 10, 20, 40, 80],
        "family_size": 28,
        "total_rung_count": 7,
        "expected_observations": 56,
        "moving_stroke_us": 512_000,
        "zero_settle_us": 500_000,
        "analysis_budget_us": 236_000,
        "nominal_workflow_ms": 69_888,
        "maximum_workflow_ms": 69_888,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
        "requested_velocity_mrev_s": 4_394,
        "machine_velocity_ceiling_mrev_s": 10_000,
        "requested_velocity_source": 1,
        "max_stroke_travel_mrev": 10_000,
        "settle_travel_reserve_mrev": 100,
        "negative_position_headroom_mrev": 20_000,
        "positive_position_headroom_mrev": 20_000,
        "max_duration_ms": 60_000,
    }
    if schema_revision >= 2:
        plan.update(
            {
                "nominal_workflow_ms": 82_607,
                "maximum_workflow_ms": 90_888,
                "max_stroke_travel_mrev": 2_500,
                "settle_travel_reserve_mrev": 250,
                "max_duration_ms": 3_000,
                "recovery_origin_band_counts": 1_000,
                "recovery_nominal_slot_us": 1_816_958,
                "recovery_maximum_slot_us": 3_000_000,
                "recovery_slot_count": 7,
            }
        )
    if schema_revision >= 4:
        plan.update(
            {
                "directional_validity": [3, 1],
                "reduced_margin": True,
                "positive_rung_count": 7,
                "positive_i": [1, 2, 4, 8, 16, 32, 64],
                "family_size": 36,
                "total_rung_count": 9,
                "expected_observations": 72,
                "nominal_workflow_ms": 106_209,
                "maximum_workflow_ms": 116_856,
                "recovery_slot_count": 9,
                "limit_evidence_capability": 1,
            }
        )
    if schema_revision >= 5:
        plan.update(
            {
                "build_revision": CURRENT_Q4_12_BUILD,
                "positive_rung_count": 12,
                "positive_i": [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024],
                "family_size": 56,
                "total_rung_count": 14,
                "expected_observations": 112,
                "nominal_workflow_ms": 165_950,
                "maximum_workflow_ms": 182_512,
                "recovery_slot_count": 14,
            }
        )
    if schema_revision >= 6:
        plan["flags"] = 1
    if schema_revision >= 7:
        plan.update(
            {
                "final_p": 724,
                "stage_b_joint_membership": 0x001C_0000,
                "joint_membership": 0x0004_0000,
                "directional_memberships": [0x001C_0000, 0x001C_0000],
                "directional_validity": [3, 3],
                "reduced_margin": True,
                "hard_torque_limit": 2_816,
                "usable_torque_limit": 2_534,
                "flags": 0b10,
            }
        )
    if schema_revision >= 8:
        plan.update(
            {
                "final_p": 1_024,
                "combined_selected_response": True,
                "selected_response_intervals_q": [
                    [12_900_000, 13_200_000],
                    [-13_100_000, -12_800_000],
                ],
                "positive_rung_count": 13,
                "positive_i": [
                    1,
                    2,
                    3,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1_024,
                    1_310,
                ],
                "family_size": 60,
                "total_rung_count": 15,
                "expected_observations": 120,
                "nominal_workflow_ms": 177_751,
                "maximum_workflow_ms": 195_496,
                "recovery_slot_count": 15,
                "usable_torque_limit": 2_816,
                "flags": 0,
            }
        )
    if schema_revision >= 9:
        plan.update(
            {
                "slot_transition_allowance_us": 20_000,
                "nominal_workflow_ms": 180_151,
                "maximum_workflow_ms": 197_896,
            }
        )
    if schema_revision >= 10:
        plan["target_velocity_rpm"] = 263
    if schema_revision >= 11:
        plan["recovery_nominal_slot_us"] = 2_316_958
        plan["nominal_workflow_ms"] = 187_651
    return plan


def test_schema_ten_replays_toward_zero_target_and_recovery_velocity() -> None:
    plan = _plan(schema_revision=10)
    plan["plan_digest"] = integral_plan_digest(plan)
    plan_mismatches: list[str] = []
    _validate_plan(plan, plan_mismatches)

    _, recovery = _recovery_record(0, 1)
    recovery["planned_velocity_mrev_s"] = 4_383
    recovery_audit = audit_rung_origin_recovery(recovery, plan)

    assert plan_mismatches == []
    assert recovery_audit["mismatches"] == []


def test_integral_plan_digest_hashes_every_immutable_input_in_firmware_order() -> None:
    assert integral_plan_digest(_plan()) == 0x8E228108A8F0E004


def test_schema_eight_plan_hashes_selected_response_and_appends_exact_cap() -> None:
    plan = _plan(schema_revision=8)
    baseline = integral_plan_digest(plan)
    changed = {
        **plan,
        "selected_response_intervals_q": [
            [12_900_001, 13_200_000],
            [-13_100_000, -12_800_000],
        ],
    }

    assert baseline != integral_plan_digest(changed)
    assert derive_integral_ladder(
        25_000,
        512_000,
        64_000,
        32_767,
        schema_revision=8,
        build_revision=CURRENT_Q4_12_BUILD,
    ) == {
        "i_start": 1,
        "i_top_cap": 1_310,
        "positive_i": [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 1_310],
    }


def test_combined_integral_target_uses_signed_directional_deficits() -> None:
    assert integral_deficit_reaches_target(
        264,
        4_000,
        [[300_000, 350_000], [-350_000, -300_000]],
    )
    assert not integral_deficit_reaches_target(
        264,
        4_000,
        [[360_000, 370_000], [-350_000, -300_000]],
    )


def test_combined_target_status_distinguishes_all_ladder_termini() -> None:
    common = {"eligible_masks": [0b111, 0b111], "expected_rungs": 15}

    assert (
        combined_target_status(target_terminus=6, current_terminus=None, emitted_rungs=7, **common)
        == 0
    )
    assert (
        combined_target_status(
            target_terminus=2,
            eligible_masks=[0b11, 0b111],
            current_terminus=None,
            emitted_rungs=3,
            expected_rungs=15,
        )
        == 1
    )
    assert (
        combined_target_status(
            target_terminus=None, current_terminus=10, emitted_rungs=12, **common
        )
        == 2
    )
    assert (
        combined_target_status(
            target_terminus=None, current_terminus=None, emitted_rungs=15, **common
        )
        == 3
    )


def test_operational_clamp_transition_ceiling_admits_p1024_not_p1448() -> None:
    common = {
        "target_rpm": 264,
        "counts_per_rev": 4_000,
        "pwm_hz": 25_000,
        "moving_stroke_us": 512_000,
        "probe_limit": 2_816,
    }
    selection = select_transition_capable_test_point(
        stage_b_joint_membership=0b11,
        p_start=1_024,
        p_top=1_448,
        **common,
    )

    assert selection["transition_ceiling"] == 1_088
    assert selection["joint_membership"] == 0b01
    assert selection["final_p"] == 1_024


def test_schema_three_plan_digest_hashes_limit_evidence_capability() -> None:
    latched = _plan(schema_revision=3)
    latched["limit_evidence_capability"] = 1
    polling = {**latched, "limit_evidence_capability": 0}

    assert integral_plan_digest(latched) != integral_plan_digest(polling)


def test_schema_four_plan_digest_hashes_stage_b_validity_context() -> None:
    full = _plan(schema_revision=4)
    transferred = {**full, "directional_validity": [3, 4]}
    full_margin = {**full, "reduced_margin": False}

    assert integral_plan_digest(full) != integral_plan_digest(transferred)
    assert integral_plan_digest(full) != integral_plan_digest(full_margin)


def test_schema_seven_plan_digest_hashes_both_memberships_and_both_flags() -> None:
    plan = _plan(schema_revision=7)
    baseline = integral_plan_digest(plan)

    assert baseline != integral_plan_digest({**plan, "stage_b_joint_membership": 0x0018_0000})
    assert baseline != integral_plan_digest({**plan, "joint_membership": 0x000C_0000})
    assert baseline != integral_plan_digest({**plan, "flags": 0b11})


def test_schema_seven_selects_transition_capable_prefix_independently() -> None:
    selection = select_transition_capable_test_point(
        stage_b_joint_membership=0x001C_0000,
        target_rpm=264,
        counts_per_rev=4_000,
        pwm_hz=25_000,
        moving_stroke_us=512_000,
        probe_limit=2_534,
    )

    assert selection == {
        "stage_b_joint_membership": 0x001C_0000,
        "joint_membership": 0x0004_0000,
        "final_p": 724,
        "transition_ceiling": 979,
        "reduced_margin": True,
        "probe_constrained_test_point": True,
    }


def test_schema_seven_rejects_upper_component_until_target_is_redeclared() -> None:
    unavailable = select_transition_capable_test_point(
        stage_b_joint_membership=0x0018_0000,
        target_rpm=264,
        counts_per_rev=4_000,
        pwm_hz=25_000,
        moving_stroke_us=512_000,
        probe_limit=2_534,
    )
    reconfigured = select_transition_capable_test_point(
        stage_b_joint_membership=0x000C_0000,
        target_rpm=132,
        counts_per_rev=4_000,
        pwm_hz=25_000,
        moving_stroke_us=512_000,
        probe_limit=2_534,
    )

    assert unavailable["joint_membership"] == 0
    assert unavailable["final_p"] == 0
    assert unavailable["transition_ceiling"] == 979
    assert reconfigured["joint_membership"] == 0x0004_0000
    assert reconfigured["final_p"] == 1_024
    assert reconfigured["transition_ceiling"] == 1_234


def test_schema_seven_transition_boundary_and_overflow_fail_closed() -> None:
    common = {
        "target_rpm": 264,
        "counts_per_rev": 4_000,
        "pwm_hz": 25_000,
        "moving_stroke_us": 512_000,
        "probe_limit": 2_534,
    }

    below = select_transition_capable_test_point(
        stage_b_joint_membership=1 << 0,
        p_start=979,
        p_top=980,
        **common,
    )
    at_or_above = select_transition_capable_test_point(
        stage_b_joint_membership=1 << 1,
        p_start=979,
        p_top=980,
        **common,
    )

    assert below["final_p"] == 979
    assert at_or_above["joint_membership"] == 0
    with pytest.raises(ValueError, match="invalid Stage-C transition"):
        select_transition_capable_test_point(
            stage_b_joint_membership=1,
            target_rpm=1 << 31,
            counts_per_rev=4_000,
            pwm_hz=25_000,
            moving_stroke_us=512_000,
            probe_limit=(1 << 16) - 1,
            p_start=0x7FFF,
            p_top=0x7FFF,
        )


def _plan_marker(plan: dict[str, object], sequence: int) -> tuple[bytes, dict[str, object]]:
    if int(plan["schema_revision"]) >= 2:
        raw = bytearray(
            struct.pack(
                "<BIQHIiIHBHIIIIIIIBHH",
                0,
                7,
                int(plan["plan_digest"]),
                int(plan["final_p"]),
                int(plan["planned_velocity_mrev_s"]),
                int(plan["target_velocity_rpm"]),
                int(plan["pwm_hz"]),
                int(plan["positive_i"][0]),
                int(plan["positive_rung_count"]),
                int(plan["family_size"]),
                int(plan["moving_stroke_us"]),
                int(plan["zero_settle_us"]),
                int(plan["analysis_budget_us"]),
                int(plan["maximum_workflow_ms"]),
                int(plan["recovery_origin_band_counts"]),
                int(plan["recovery_nominal_slot_us"]),
                int(plan["recovery_maximum_slot_us"]),
                int(plan["recovery_slot_count"]),
                int(plan["hard_torque_limit"]),
                int(plan["usable_torque_limit"]),
            )
        )
        if int(plan["schema_revision"]) >= 8:
            raw.extend(
                struct.pack(
                    "<Biiii",
                    1,
                    *(
                        endpoint
                        for interval in plan["selected_response_intervals_q"]
                        for endpoint in interval
                    ),
                )
            )
        return bytes(raw), {
            **plan,
            "record_kind": "integral_plan_marker",
            "run_sequence": 7,
            "evidence_sequence": sequence,
            "plan_digest": plan["plan_digest"],
        }
    raw = struct.pack(
        "<BIHQHBI",
        1,
        7,
        sequence,
        int(plan["plan_digest"]),
        int(plan["final_p"]),
        int(plan["positive_rung_count"]),
        int(plan["maximum_workflow_ms"]),
    )
    return raw, {
        "record_kind": "integral_plan_marker",
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "plan_digest": plan["plan_digest"],
        "final_p": plan["final_p"],
        "positive_rung_count": plan["positive_rung_count"],
        "maximum_workflow_ms": plan["maximum_workflow_ms"],
    }


def _plan_records(plan: dict[str, object]) -> list[tuple[bytes, dict[str, object]]]:
    common = {
        "run_sequence": 7,
        "evidence_sequence": 0,
        "plan_digest": plan["plan_digest"],
        "positive_rung_count": plan["positive_rung_count"],
    }
    core_names = [
        "build_revision",
        "schema_revision",
        "channel",
        "stage_b_plan_digest",
        "final_p",
        "directional_memberships",
        "joint_membership",
        "directional_pooled_intervals",
        "planned_velocity_mrev_s",
        "target_velocity_rpm",
        "pwm_hz",
        "encoder_counts_per_rev",
        "family_size",
        "total_rung_count",
        "expected_observations",
        "hard_torque_limit",
        "usable_torque_limit",
    ]
    if int(plan["schema_revision"]) >= 7:
        core_names.insert(core_names.index("joint_membership"), "stage_b_joint_membership")
    schedule_names = [
        "positive_i",
        "moving_stroke_us",
        "zero_settle_us",
        "analysis_budget_us",
        "nominal_workflow_ms",
        "maximum_workflow_ms",
        "requested_velocity_mrev_s",
        "machine_velocity_ceiling_mrev_s",
        "requested_velocity_source",
        "max_stroke_travel_mrev",
        "settle_travel_reserve_mrev",
        "negative_position_headroom_mrev",
        "positive_position_headroom_mrev",
        "max_duration_ms",
    ]
    if int(plan["schema_revision"]) >= 9:
        schedule_names.insert(4, "slot_transition_allowance_us")
    records = [
        (
            b"",
            {
                **common,
                **{name: plan[name] for name in core_names},
                "record_kind": "integral_plan_core",
                "plan_fragment": 0,
            },
        ),
        (
            b"",
            {
                **common,
                **{name: plan[name] for name in schedule_names},
                "record_kind": "integral_plan_schedule",
                "plan_fragment": 1,
            },
        ),
    ]
    if int(plan["schema_revision"]) >= 2:
        recovery = {
            **common,
            "recovery_origin_band_counts": plan["recovery_origin_band_counts"],
            "recovery_nominal_slot_us": plan["recovery_nominal_slot_us"],
            "recovery_maximum_slot_us": plan["recovery_maximum_slot_us"],
            "recovery_slot_count": plan["recovery_slot_count"],
            "record_kind": "integral_plan_recovery",
            "plan_fragment": 2,
        }
        if int(plan["schema_revision"]) >= 6:
            recovery["flags"] = plan["flags"]
        records.append((b"", recovery))
    if int(plan["schema_revision"]) >= 3:
        records[0][1]["limit_evidence_capability"] = plan["limit_evidence_capability"]
    if int(plan["schema_revision"]) >= 4:
        records[0][1]["directional_validity"] = plan["directional_validity"]
        records[0][1]["reduced_margin"] = plan["reduced_margin"]
    if int(plan["schema_revision"]) >= 8:
        records[0][1]["combined_selected_response"] = plan["combined_selected_response"]
        records[0][1]["selected_response_intervals_q"] = plan["selected_response_intervals_q"]
    return records


def _recovery_record(
    rung_index: int,
    sequence: int,
    velocity_p: int = 1_448,
    planned_velocity_mrev_s: int = 4_394,
) -> tuple[bytes, dict[str, object]]:
    values = {
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "stage": 1,
        "rung_index": rung_index,
        "velocity_p": velocity_p,
        "planned_velocity_mrev_s": planned_velocity_mrev_s,
        "start_offset_counts": 900,
        "closest_offset_counts": 900,
        "final_offset_counts": 900,
        "origin_band_counts": 1_000,
        "lower_rate_q": 0,
        "moving_duration_us": 0,
        "settle_duration_us": 0,
        "total_duration_us": 0,
        "peak_torque_target_abs": 0,
        "binding_source": 0,
        "outcome": 0,
    }
    raw = struct.pack(
        "<BIHBBHIiiiIQIIIHBB",
        9,
        *values.values(),
    )
    return raw, {"record_kind": "rung_origin_recovery", **values}


def _current_record(rung_index: int, slot: int, sequence: int) -> tuple[bytes, dict[str, object]]:
    values = {
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "stage": 1,
        "rung_index": rung_index,
        "slot": slot,
        "clamp_limit": 900,
        "clamp_readback": 900,
        "moving_pid_output_peak": 100,
        "zero_pid_output_peak": 120,
        "moving_status_flags": 0,
        "zero_status_flags": 0,
        "capability": 1,
        "contact": 0,
    }
    raw = struct.pack("<BIHBBBHHHHIIBB", 31, *values.values())
    return raw, {"record_kind": "velocity_primitive_current", **values}


def _rest_boundary_record(sequence: int) -> tuple[bytes, dict[str, object]]:
    raw = bytearray([5])
    raw.extend(struct.pack("<IHQBBHb", 7, sequence, 0, 11, 0, 1_024, 1))
    raw.extend(struct.pack("<HIiiiH", 433, 433_000, 2, 0, 2, 2))
    raw.extend(struct.pack("<HIii", 3, 3_000, 1, 1))
    raw.extend(struct.pack("<HIii", 401, 401_000, 1, 2))
    raw.extend(struct.pack("<B", 4))
    raw.extend(struct.pack("<HHHIIBBB", 900, 100, 120, 0, 0, 1, 0, 1))
    return bytes(raw), {
        "record_kind": "integral_rest_boundary",
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "plan_digest": 0,
        "positive_rung_index": 11,
        "slot": 0,
        "i_raw": 1_024,
        "direction": 1,
        "active_stationarity_code": 4,
        "cleanup_confirmed": 1,
    }


def _observation_record(
    result: dict[str, object], sequence: int, plan_digest: int
) -> tuple[bytes, dict[str, object]]:
    raw = bytearray((2,))
    raw += struct.pack("<IHQ", 7, sequence, plan_digest)
    raw += struct.pack(
        "<BBHBBB",
        int(result["rung_index"]),
        int(result["slot"]),
        int(result["i_raw"]),
        int(result["direction"]),
        int(result["classification"]),
        int(result["initial_lead_in"]),
    )
    raw += struct.pack(
        "<iiiiiBH",
        int(result["rate_low"]),
        int(result["rate_mean"]),
        int(result["rate_high"]),
        int(result["deficit_low"]),
        int(result["deficit_high"]),
        int(result["rate_shift"]),
        int(result["predicted_torque_target_abs"]),
    )
    raw += struct.pack("<B", int(result["stationarity_code"]))
    if int(result["stationarity_code"]) == 0:
        raw += struct.pack(
            "<HBIiBIBiIIBB",
            int(result["suffix_len"]),
            int(result["selected_level"]),
            int(result["selected_blocks"]),
            int(result["slope_mantissa"]),
            int(result["slope_shift"]),
            int(result["slope_half_width_mantissa"]),
            int(result["slope_half_width_shift"]),
            int(result["lag_one_q"]),
            int(result["lag_one_half_width_q"]),
            int(result["residual_mantissa"]),
            int(result["residual_shift"]),
            int(result["tested_suffixes"]),
        )
    return bytes(raw), {
        **result,
        "record_kind": "integral_observation",
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "plan_digest": plan_digest,
    }


def _rung_record(
    rung_index: int,
    i_raw: int,
    sequence: int,
    plan_digest: int,
    directions: list[dict[str, object]],
    plan: dict[str, object] | None = None,
) -> tuple[bytes, dict[str, object]]:
    raw = bytearray((3,))
    raw += struct.pack("<IHQBH", 7, sequence, plan_digest, rung_index, i_raw)
    horizon: dict[str, int] = {}
    if plan is not None and int(plan["schema_revision"]) >= 9:
        tau_us = (
            0
            if i_raw == 0
            else velocity_i_tau_us(i_raw, int(plan["pwm_hz"]), plan.get("build_revision"))
        )
        horizon = {
            "tau_us": tau_us,
            "moving_horizon_us": int(plan["moving_stroke_us"]),
            "horizon_class": (
                0 if i_raw == 0 else 1 if tau_us > int(plan["moving_stroke_us"]) else 2
            ),
        }
        raw += struct.pack(
            "<QIB",
            horizon["tau_us"],
            horizon["moving_horizon_us"],
            horizon["horizon_class"],
        )
    for direction in directions:
        components = list(direction["components"])
        raw += struct.pack(
            "<BBBBBB",
            int(direction["direction"]),
            int(direction["kind"]),
            int(direction["classification"]),
            len(components),
            int(direction["usable_mask"]),
            int(direction["included_mask"]),
        )
        raw += bytes(int(value) for value in direction["observation_classes"])
        for component in components:
            raw += struct.pack("<ii", *component)
        for _ in range(2 - len(components)):
            raw += struct.pack("<ii", 0, 0)
    return bytes(raw), {
        "record_kind": "integral_rung",
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "plan_digest": plan_digest,
        "rung_index": rung_index,
        "i_raw": i_raw,
        **horizon,
        "directions": directions,
    }


def _synthetic_complete_capture(
    schema_revision: int = 1,
) -> tuple[list[VelocityObservation], list[tuple[bytes, dict[str, object]]]]:
    plan = _plan(schema_revision)
    plan["plan_digest"] = integral_plan_digest(plan)
    records = _plan_records(plan)
    sequence = 1
    records.append(_plan_marker(plan, sequence))
    sequence += 1
    raw_observations: list[VelocityObservation] = []
    gains = [0, *plan["positive_i"], 0]
    for rung_index, gain in enumerate(gains):
        direction_results: list[list[dict[str, object]]] = [[], []]
        for slot in range(8):
            direction = slot % 2
            raw = [
                VelocityObservation(
                    17 if direction == 0 else -17,
                    1_000,
                    direction,
                    rung_index * 8 + slot,
                    0,
                    100,
                )
                for _ in range(256)
            ]
            raw_observations.extend(raw)
            result = reconstruct_observation(
                raw,
                {"rung_index": rung_index, "slot": slot, "i_raw": gain},
                plan,
            )
            direction_results[direction].append(result)
            records.append(_observation_record(result, sequence, int(plan["plan_digest"])))
            sequence += 1
            if schema_revision >= 3:
                records.append(_current_record(rung_index, slot, sequence))
                sequence += 1
        directions = []
        for direction, observations in enumerate(direction_results):
            interval = tuple(observations[0]["deficit_interval_q"])
            directions.append(
                {
                    "direction": direction,
                    "kind": 0 if rung_index == 0 else 2 if rung_index == len(gains) - 1 else 1,
                    "classification": 0,
                    "component_count": 1,
                    "components": [interval],
                    "usable_mask": 0b1111,
                    "included_mask": 0b1111,
                    "observation_classes": [0, 0, 0, 0],
                }
            )
        records.append(
            _rung_record(
                rung_index,
                gain,
                sequence,
                int(plan["plan_digest"]),
                directions,
                plan,
            )
        )
        sequence += 1
        if schema_revision >= 2:
            records.append(
                _recovery_record(
                    rung_index,
                    sequence,
                    velocity_p=int(plan["final_p"]),
                    planned_velocity_mrev_s=(4_383 if schema_revision >= 10 else 4_394),
                )
            )
            sequence += 1
    if schema_revision >= 2:
        compact_marker = bytearray(_plan_marker(plan, 1)[0])
    else:
        compact_marker = bytearray((0,))
        compact_marker += struct.pack(
            "<IQHIiIHBHIIIIHH",
            7,
            int(plan["plan_digest"]),
            int(plan["final_p"]),
            int(plan["planned_velocity_mrev_s"]),
            int(plan["target_velocity_rpm"]),
            int(plan["pwm_hz"]),
            int(plan["positive_i"][0]),
            int(plan["positive_rung_count"]),
            int(plan["family_size"]),
            int(plan["moving_stroke_us"]),
            int(plan["zero_settle_us"]),
            int(plan["analysis_budget_us"]),
            int(plan["maximum_workflow_ms"]),
            int(plan["hard_torque_limit"]),
            int(plan["usable_torque_limit"]),
        )
    digest = 0xCBF29CE484222325
    for byte in compact_marker:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for raw, value in records:
        if value["record_kind"] in ("integral_plan_core", "integral_plan_schedule"):
            continue
        for byte in raw:
            digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    records.append(
        (
            b"",
            {
                "record_kind": "integral_terminal",
                "run_sequence": 7,
                "evidence_sequence": sequence,
                "plan_digest": plan["plan_digest"],
                "digest": digest,
                "outcome": 1 if schema_revision >= 8 else 0,
                "cause": 0,
                "expected_observations": 8 * len(gains),
                "emitted_observations": 8 * len(gains),
                "expected_rungs": len(gains),
                "emitted_rungs": len(gains),
                "eligible_masks": [
                    (1 << int(plan["positive_rung_count"])) - 1,
                    (1 << int(plan["positive_rung_count"])) - 1,
                ],
                "current_terminus": None,
                **(
                    {
                        "recovery_flags": (
                            0xC0 if schema_revision >= 8 else 8 if schema_revision >= 7 else 4
                        )
                    }
                    if schema_revision >= 6
                    else {"recovery_unavailable": 0}
                ),
                "run_started_us": 1,
                "run_completed_us": 2,
                **(
                    {
                        "opening_available_mask": 3,
                        "bookend_available_mask": 3,
                    }
                    if schema_revision >= 4
                    else {}
                ),
                **(
                    {
                        "combined_workflow": True,
                        "final_p": plan["final_p"],
                        "target_status": 3,
                        "target_terminus": None,
                    }
                    if schema_revision >= 8
                    else {}
                ),
            },
        )
    )
    return raw_observations, records


def test_schema_eight_combined_run_completes_without_reproduction() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=8)

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["terminal"]["outcome"] == 1
    assert audit["terminal"]["target_status"] == 3


@pytest.mark.parametrize("schema_revision", [9, 10, 11])
def test_current_combined_schemas_replay_response_horizons_and_transition_allowance(
    schema_revision: int,
) -> None:
    raw, records = _synthetic_complete_capture(schema_revision=schema_revision)

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["plan"]["slot_transition_allowance_us"] == 20_000
    rung = next(
        value
        for _raw, value in records
        if value["record_kind"] == "integral_rung" and value["i_raw"] == 64
    )
    assert rung["tau_us"] == 655_360
    assert rung["moving_horizon_us"] == 512_000
    assert rung["horizon_class"] == 1


def test_schema_nine_reports_firmware_response_horizon_mismatch() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=9)
    rung = next(
        value
        for _raw, value in records
        if value["record_kind"] == "integral_rung" and value["i_raw"] == 64
    )
    rung["tau_us"] = 655_359

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("response horizon tau_us" in mismatch for mismatch in audit["mismatches"])


def _synthetic_schema_five_boundary_capture() -> tuple[
    list[VelocityObservation], list[tuple[bytes, dict[str, object]]]
]:
    raw_observations, records = _synthetic_complete_capture(schema_revision=5)
    plan = _plan(schema_revision=5)
    plan["plan_digest"] = integral_plan_digest(plan)
    terminal = records.pop()
    records = [
        (raw, value)
        for raw, value in records
        if not (
            int(value.get("rung_index", -1)) == 12
            and (
                value["record_kind"] in ("integral_observation", "integral_rung")
                or (
                    value["record_kind"] == "velocity_primitive_current" and int(value["slot"]) != 0
                )
            )
        )
    ]
    sequence = 1
    resequenced = []
    for raw, value in records:
        if (
            value["record_kind"].startswith("integral_plan_")
            and value["record_kind"] != "integral_plan_marker"
        ):
            resequenced.append((raw, value))
            continue
        value["evidence_sequence"] = sequence
        if value["record_kind"] != "integral_plan_marker":
            encoded = bytearray(raw)
            encoded[5:7] = sequence.to_bytes(2, "little")
            raw = bytes(encoded)
        resequenced.append((raw, value))
        sequence += 1
    boundary_raw, boundary = _rest_boundary_record(sequence)
    boundary["plan_digest"] = plan["plan_digest"]
    encoded_boundary = bytearray(boundary_raw)
    encoded_boundary[7:15] = int(plan["plan_digest"]).to_bytes(8, "little")
    boundary_raw = bytes(encoded_boundary)
    digest = 0xCBF29CE484222325
    for byte in _plan_marker(plan, 1)[0]:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for raw, value in resequenced:
        if value["record_kind"] in ("integral_plan_core", "integral_plan_schedule"):
            continue
        for byte in raw:
            digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for byte in boundary_raw:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    terminal[1].update(
        {
            "evidence_sequence": sequence,
            "digest": digest,
            "cause": 10,
            "emitted_observations": 104,
            "emitted_rungs": 13,
            "eligible_masks": [0x7FF, 0x7FF],
        }
    )
    return raw_observations, [*resequenced, (boundary_raw, boundary), terminal]


def test_complete_capture_reconstructs_curve_counts_and_digest() -> None:
    raw, records = _synthetic_complete_capture()

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["complete"] is True
    assert audit["run_evidence"]["eligible_masks"] == [0b11111, 0b11111]


def test_recovery_capture_preserves_curve_and_audits_every_rung() -> None:
    old_raw, old_records = _synthetic_complete_capture()
    raw, records = _synthetic_complete_capture(schema_revision=2)

    old_audit = audit_velocity_integral(old_raw, old_records)
    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["complete"] is True
    assert len(audit["recoveries"]) == 7
    assert audit["run_evidence"] == old_audit["run_evidence"]


def test_recovery_capture_rejects_missing_causal_record_and_wrong_stage() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=2)
    recovery_index = next(
        index
        for index, (_raw, value) in enumerate(records)
        if value["record_kind"] == "rung_origin_recovery"
    )
    records[recovery_index][1]["stage"] = 0
    del records[recovery_index + 10]

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("wrong stage" in item for item in audit["mismatches"])
    assert any("recovery rungs" in item for item in audit["mismatches"])


def test_opening_anchor_retention_is_all_or_none() -> None:
    eligible = {"classification": 0, "components": [(100, 110)]}
    ineligible = {"classification": 4, "components": []}

    opening, bookend, positive, masks = _retained_stage_c_evidence(
        {0: [eligible, ineligible]},
        positive_count=12,
    )

    assert opening == [None, None]
    assert bookend == [None, None]
    assert positive == [[None] * 12, [None] * 12]
    assert masks == [0, 0]


def test_capture_audit_reports_digest_and_observation_mismatch_paths() -> None:
    raw, records = _synthetic_complete_capture()
    first_observation = next(
        value for _raw, value in records if value["record_kind"] == "integral_observation"
    )
    first_observation["classification"] = 4
    records[-1][1]["digest"] ^= 1

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("terminal digest" in item for item in audit["mismatches"])
    assert any("classification" in item for item in audit["mismatches"])


def test_fault_terminal_keeps_empty_firmware_evidence_and_reports_prefix() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=2)
    terminal = records[-1][1]
    terminal.update(
        {
            "outcome": 3,
            "cause": 11,
            "eligible_masks": [0, 0],
        }
    )

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["terminal"]["eligible_masks"] == [0, 0]
    assert audit["run_evidence"]["eligible_masks"] == [0b11111, 0b11111]
    assert "pre-fault prefix" in audit["caveats"][0]


def _synthetic_recovery_fault_prefix(
    *, outcome: int = 3, missing_recoveries: tuple[int, ...] = (13,)
) -> tuple[list[VelocityObservation], list[tuple[bytes, dict[str, object]]]]:
    raw_observations, records = _synthetic_complete_capture(schema_revision=6)
    terminal = records[-1][1]
    retained = [
        item
        for item in records[:-1]
        if not (
            item[1]["record_kind"] == "rung_origin_recovery"
            and int(item[1]["rung_index"]) in missing_recoveries
        )
    ]
    if missing_recoveries == (13,):
        terminal["evidence_sequence"] = int(terminal["evidence_sequence"]) - 1
    terminal.update(
        {
            "outcome": outcome,
            "cause": 11 if outcome == 3 else 0,
            "eligible_masks": [0, 0] if outcome == 3 else terminal["eligible_masks"],
            "opening_available_mask": 0 if outcome == 3 else terminal["opening_available_mask"],
            "bookend_available_mask": 0 if outcome == 3 else terminal["bookend_available_mask"],
        }
    )
    plan = _plan(schema_revision=6)
    plan["plan_digest"] = integral_plan_digest(plan)
    digest = 0xCBF29CE484222325
    for byte in _plan_marker(plan, 1)[0]:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for encoded, value in retained:
        if value["record_kind"] in ("integral_plan_core", "integral_plan_schedule"):
            continue
        for byte in encoded:
            digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    terminal["digest"] = digest
    return raw_observations, [*retained, records[-1]]


def test_stage_c_fault_accepts_one_missing_final_recovery() -> None:
    raw, records = _synthetic_recovery_fault_prefix()

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["mismatches"] == []
    assert audit["terminal"]["outcome"] == 3
    assert audit["terminal"]["cause"] == 11
    assert [int(item["rung_index"]) for item in audit["recoveries"]] == list(range(13))
    assert "pre-fault prefix" in audit["caveats"][0]


def test_stage_c_non_fault_rejects_one_missing_final_recovery() -> None:
    raw, records = _synthetic_recovery_fault_prefix(outcome=2)

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("recovery rungs" in item for item in audit["mismatches"])


def test_stage_c_fault_rejects_two_missing_recovery_records() -> None:
    raw, records = _synthetic_recovery_fault_prefix(missing_recoveries=(12, 13))

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("recovery rungs" in item for item in audit["mismatches"])


def test_stage_c_fault_rejects_missing_intermediate_recovery() -> None:
    raw, records = _synthetic_recovery_fault_prefix(missing_recoveries=(12,))

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is False
    assert any("recovery rungs" in item for item in audit["mismatches"])


def test_reproduction_pair_checks_anchor_relative_curve_and_terminal() -> None:
    raw, records = _synthetic_complete_capture()
    previous = audit_velocity_integral(raw, records)
    current = audit_velocity_integral(raw, records)
    current["terminal"]["outcome"] = 1

    pair = audit_integral_reproduction_pair(previous, current)

    assert pair["matches_firmware"] is True
    assert pair["comparison"]["reproduced_masks"] == [0b11111, 0b11111]


def test_schema_five_boundary_is_digest_bearing_and_lifecycle_audited() -> None:
    raw, records = _synthetic_schema_five_boundary_capture()

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["complete"] is True
    assert audit["terminal"]["cause"] == 10
    assert audit["rest_boundary"]["positive_rung_index"] == 11
    assert audit["rest_boundary"]["cleanup_confirmed"] == 1
    assert audit["run_evidence"]["rest_boundary"] == {
        "positive_rung_index": 11,
        "slot": 0,
    }


def test_schema_five_pair_reports_boundary_references_without_gating() -> None:
    raw, records = _synthetic_schema_five_boundary_capture()
    previous = audit_velocity_integral(raw, records)
    previous["run_evidence"]["rest_boundary"] = None
    current = audit_velocity_integral(raw, records)
    current["terminal"]["outcome"] = 1

    pair = audit_integral_reproduction_pair(previous, current)

    assert pair["matches_firmware"] is True
    assert pair["rest_boundary_equal"] is False
    assert pair["previous_rest_boundary"] is None
    assert pair["current_rest_boundary"] == {
        "positive_rung_index": 11,
        "slot": 0,
    }


def test_schema_aware_geometry_preserves_old_and_new_whole_octave_ladders() -> None:
    assert velocity_i_tau_us(5, 25_000, LEGACY_Q8_8_BUILD) == 524_288
    assert velocity_i_tau_us(80, 25_000, LEGACY_Q8_8_BUILD) == 32_768

    schema_three = derive_integral_ladder(
        25_000,
        512_000,
        64_000,
        32_767,
        schema_revision=3,
        build_revision=LEGACY_Q8_8_BUILD,
    )
    schema_four = derive_integral_ladder(
        25_000,
        512_000,
        64_000,
        32_767,
        schema_revision=4,
        build_revision=LEGACY_Q8_8_BUILD,
    )
    schema_five = derive_integral_ladder(
        25_000,
        512_000,
        64_000,
        32_767,
        schema_revision=5,
        build_revision=CURRENT_Q4_12_BUILD,
    )

    assert schema_three == {
        "i_start": 5,
        "i_top_cap": 81,
        "positive_i": [5, 10, 20, 40, 80],
    }
    assert schema_four == {
        "i_start": 1,
        "i_top_cap": 81,
        "positive_i": [1, 2, 4, 8, 16, 32, 64],
    }
    assert schema_five == {
        "i_start": 1,
        "i_top_cap": 1310,
        "positive_i": [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    }


def test_build_revision_selects_the_recorded_integral_representation() -> None:
    assert _fnv1a32("0.1.0") == LEGACY_Q8_8_BUILD
    assert _fnv1a32("0.2.0") == CURRENT_Q4_12_BUILD
    assert integral_numerator_for_build(LEGACY_Q8_8_BUILD) == 65_536
    assert integral_numerator_for_build(CURRENT_Q4_12_BUILD) == 1_048_576

    assert velocity_i_tau_us(16, 25_000, LEGACY_Q8_8_BUILD) == 163_840
    assert velocity_i_tau_us(16, 25_000, CURRENT_Q4_12_BUILD) == 2_621_440
    samples_per_stroke = 25_000 * 512_000 // 1_000_000
    assert Fraction(samples_per_stroke * 16, 65_536) == Fraction(25, 8)
    assert Fraction(samples_per_stroke * 16, 1_048_576) == Fraction(25, 128)


def test_current_build_reconstructs_the_migrated_raw_q4_12_ladder() -> None:
    from foci_trace.velocity_integral import _validate_plan

    plan = _plan(schema_revision=4)
    plan.update(
        {
            "build_revision": CURRENT_Q4_12_BUILD,
            "positive_i": [16, 32, 64, 128, 256, 512, 1_024],
        }
    )
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    _validate_plan(plan, mismatches)

    assert mismatches == []


def test_committed_schema_four_q8_8_plan_remains_bit_exact() -> None:
    from foci_trace.velocity_integral import _validate_plan

    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-20-foc-183-velocity-integral-response"
        / "foc-183-stepper-x-stage-c-resume1-trace-only-current-20260725-004335.velocity.json"
    )
    response = json.loads(capture.read_text())["integral_responses"][0]
    plan = response["plan"]
    mismatches: list[str] = []

    assert response["matches_firmware"] is True
    assert plan["build_revision"] == LEGACY_Q8_8_BUILD
    _validate_plan(plan, mismatches)

    assert mismatches == []


def test_schema_five_recovery_current_headroom_capture_remains_bit_exact() -> None:
    from foci_trace.analyze import analyze_capture

    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-20-foc-183-velocity-integral-response"
        / "foc-183-stepper-x-stagec5-ladder1-20260725-142151.bin"
    )
    with capture.open("rb") as source:
        responses = analyze_capture(source).velocity_integral_audits

    assert len(responses) == 1
    response = responses[0]
    assert response["matches_firmware"] is True
    assert response["complete"] is False
    assert response["terminal"]["outcome"] == 2
    assert response["terminal"]["cause"] == 8
    assert response["historical_attribution_caveat"] is True
    assert response["projected_schema_six_recovery_outcome"] == "unknown"
    recovery = response["recoveries"][-1]
    assert recovery["outcome"] == 4
    assert recovery["binding_source"] == 6
    assert recovery["peak_torque_target_abs"] == 2_816
    assert recovery["final_offset_counts"] == 998


def test_schema_six_capture_reconstructs_plan_and_terminal_exposure() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=6)

    response = audit_velocity_integral(raw, records)

    assert response["matches_firmware"] is True
    assert response["plan"]["flags"] == 1
    assert response["terminal"]["recovery_unavailable"] == 0
    assert response["terminal"]["recovered_with_current_headroom"] is False
    assert response["terminal"]["recovery_quantization_exposed"] is True


def test_schema_seven_reconstructs_selected_test_point_and_qualifiers() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=7)

    response = audit_velocity_integral(raw, records)

    assert response["matches_firmware"] is True
    assert response["plan"]["stage_b_joint_membership"] == 0x001C_0000
    assert response["plan"]["joint_membership"] == 0x0004_0000
    assert response["plan"]["final_p"] == 724
    assert response["plan"]["reduced_margin"] is True
    assert response["terminal"]["probe_constrained_test_point"] is True


def test_schema_seven_audits_terminal_only_failed_admission_without_a_plan() -> None:
    terminal = {
        "record_kind": "integral_terminal",
        "run_sequence": 7,
        "evidence_sequence": 0,
        "plan_digest": 0x0123_4567_89AB_CDEF,
        "digest": 0,
        "outcome": 5,
        "cause": 11,
        "recovery_flags": 0b1000,
        "expected_observations": 0,
        "emitted_observations": 0,
        "expected_rungs": 0,
        "emitted_rungs": 0,
        "eligible_masks": [0, 0],
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
        "current_terminus": None,
        "run_started_us": 0,
        "run_completed_us": 0,
    }

    response = audit_velocity_integral([], [(b"", terminal)])

    assert response["matches_firmware"]
    assert response["complete"] is False
    assert response["plan"] is None
    assert response["terminal"]["probe_constrained_test_point"] is True
    assert response["interpretation"] == "firmware-schema-7"


def test_schema_seven_does_not_reinterpret_other_terminal_only_outcomes() -> None:
    terminal = {
        "record_kind": "integral_terminal",
        "run_sequence": 7,
        "evidence_sequence": 0,
        "plan_digest": 0x0123_4567_89AB_CDEF,
        "digest": 0,
        "outcome": 4,
        "cause": 11,
        "recovery_flags": 0b1000,
        "expected_observations": 0,
        "emitted_observations": 0,
        "expected_rungs": 0,
        "emitted_rungs": 0,
        "eligible_masks": [0, 0],
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
        "current_terminus": None,
        "run_started_us": 0,
        "run_completed_us": 0,
    }

    response = audit_velocity_integral([], [(b"", terminal)])

    assert not response["matches_firmware"]
    assert "partial integral-response capture" in response["mismatches"][0]


@pytest.mark.parametrize(
    ("recovery_flags", "message"),
    (
        (8, "terminal recovery flags contain reserved bits"),
        (6, "causal outcome-5 recovery evidence"),
    ),
)
def test_schema_six_rejects_invalid_terminal_recovery_flags(
    recovery_flags: int, message: str
) -> None:
    raw, records = _synthetic_complete_capture(schema_revision=6)
    records[-1][1]["recovery_flags"] = recovery_flags

    response = audit_velocity_integral(raw, records)

    assert response["matches_firmware"] is False
    assert any(message in item for item in response["mismatches"])


@pytest.mark.parametrize(
    ("outcome", "final_offset"),
    ((5, 998), (4, 1_001)),
)
def test_schema_six_reconstructs_bounded_recovery_contact_lifecycle(
    outcome: int, final_offset: int
) -> None:
    plan = _plan(schema_revision=6)
    record = _recovery_record(12, 1)[1]
    record.update(
        {
            "start_offset_counts": 1_834,
            "closest_offset_counts": 998,
            "final_offset_counts": final_offset,
            "moving_duration_us": 100_000,
            "settle_duration_us": 500_000,
            "total_duration_us": 600_000,
            "peak_torque_target_abs": 2_534,
            "binding_source": 6,
            "outcome": outcome,
        }
    )

    audit = audit_rung_origin_recovery(record, plan)

    assert audit["matches_firmware"] is True
    assert audit["mismatches"] == []


@pytest.mark.parametrize("build_revision", [None, 0xDEADBEEF])
def test_unknown_build_refuses_to_guess_integral_representation(
    build_revision: int | None,
) -> None:
    from foci_trace.velocity_integral import _validate_plan

    with pytest.raises(UnsupportedIntegralRepresentationError):
        integral_numerator_for_build(build_revision)
    plan = _plan(schema_revision=4)
    if build_revision is None:
        del plan["build_revision"]
    else:
        plan["build_revision"] = build_revision
    with pytest.raises(UnsupportedIntegralRepresentationError):
        _validate_plan(plan, [])


def test_schema_four_plan_reconstructs_exact_counts_and_duration() -> None:
    plan = _plan(schema_revision=4)
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert mismatches == []


def test_schema_five_plan_reconstructs_exact_counts_and_cleanup_duration() -> None:
    plan = _plan(schema_revision=5)
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert plan["positive_i"] == [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024]
    assert plan["family_size"] == 56
    assert plan["expected_observations"] == 112
    assert plan["nominal_workflow_ms"] == 165_950
    assert plan["maximum_workflow_ms"] == 182_512
    assert mismatches == []


def test_schema_six_reconstructs_exact_recovery_quantization_exposure() -> None:
    assert recovery_quantization_exposed(724, 264, 4_000, 25_000, 2_534) is False
    assert recovery_quantization_exposed(1_024, 264, 4_000, 25_000, 2_534) is True
    assert recovery_quantization_exposed(256, 0, 1, 1, 60) is False
    with pytest.raises(ValueError, match="absolute i32"):
        recovery_quantization_exposed(1_024, -(1 << 31), 4_000, 25_000, 2_534)

    plan = _plan(schema_revision=6)
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert plan["flags"] == 1
    assert mismatches == []


def test_schema_six_detects_recovery_exposure_flag_drift() -> None:
    plan = _plan(schema_revision=6)
    plan["flags"] = 0
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert "plan recovery quantization exposure differs from exact reconstruction" in mismatches


def test_schema_six_rejects_reserved_plan_recovery_flags() -> None:
    plan = _plan(schema_revision=6)
    plan["flags"] = 2
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert "plan recovery flags contain reserved bits" in mismatches


def test_schema_four_sufficiency_requires_both_closing_bookends() -> None:
    from foci_trace.velocity_integral import _stage_c_run_sufficient

    opening = [(10, 20), (-20, -10)]
    masks = [0b111, 0b111]

    assert _stage_c_run_sufficient(4, opening, [(11, 21), (-21, -11)], masks) is True
    assert _stage_c_run_sufficient(4, opening, [(11, 21), None], masks) is False
    assert _stage_c_run_sufficient(3, opening, [None, None], masks) is True


@pytest.mark.parametrize("schema_revision", [0, 12])
def test_integral_ladder_rejects_unsupported_schema(schema_revision: int) -> None:
    with pytest.raises(ValueError, match="unsupported Stage-C evidence schema"):
        derive_integral_ladder(
            25_000,
            512_000,
            64_000,
            32_767,
            schema_revision=schema_revision,
            build_revision=LEGACY_Q8_8_BUILD,
        )


def test_depth_three_consensus_preserves_closed_endpoint_and_ambiguity() -> None:
    eligible = depth_three_consensus([(0, 10), (2, 9), (3, 8), (4, 7)])
    endpoint = depth_three_consensus([(0, 5), (5, 10), (0, 5), (5, 10)])
    ambiguous = depth_three_consensus([(0, 4), (0, 10), (0, 10), (6, 10)])

    assert eligible == {
        "classification": 0,
        "components": [(3, 8)],
        "usable_mask": 0b1111,
        "included_mask": 0b1111,
    }
    assert endpoint["components"] == [(5, 5)]
    assert ambiguous == {
        "classification": 1,
        "components": [(0, 4), (6, 10)],
        "usable_mask": 0b1111,
        "included_mask": 0,
    }


def test_anchor_requires_commanded_side_deficit_but_positive_rung_keeps_overshoot() -> None:
    common = {
        "direction": 0,
        "settled_rate_interval_q": (10, 20),
        "stationarity_usable": True,
        "predicted_torque_target_abs": 100,
        "usable_torque_limit": 900,
        "hard_torque_limit": 1_000,
    }

    assert classify_observation(rung_index=0, deficit_interval_q=(-20, -10), **common) == 1
    assert classify_observation(rung_index=1, deficit_interval_q=(-20, -10), **common) == 0


def test_constant_rate_observation_reconstructs_firmware_fields() -> None:
    raw = [VelocityObservation(17, 1_000, 1, 0, 17, 100) for _ in range(256)]
    plan = {
        "family_size": 20,
        "target_velocity_rpm": 264,
        "encoder_counts_per_rev": 4_000,
        "final_p": 1_448,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
    }

    result = reconstruct_observation(raw, {"rung_index": 0, "slot": 0, "i_raw": 0}, plan)

    assert result["classification"] == 0
    assert result["settled_rate_interval_q"] == [17_000_000, 17_000_000]
    assert result["deficit_interval_q"] == [600_000, 600_000]
    assert result["predicted_torque_target_abs"] == 100
    assert result["suffix_len"] == 256
    assert result["selected_level"] == 4
    assert result["selected_blocks"] == 16


def test_current_observer_peak_overrides_sparse_trace_peak() -> None:
    raw = [VelocityObservation(17, 1_000, 1, 8, 17, 1_218) for _ in range(256)]
    plan = {
        "family_size": 20,
        "target_velocity_rpm": 264,
        "encoder_counts_per_rev": 4_000,
        "final_p": 1_024,
        "hard_torque_limit": 2_816,
        "usable_torque_limit": 2_534,
    }

    result = reconstruct_observation(
        raw,
        {"rung_index": 1, "slot": 0, "i_raw": 5},
        plan,
        observed_torque_peak_abs=2_534,
    )

    assert result["predicted_torque_target_abs"] == 2_534
    assert result["classification"] == 3


def test_delta_subtraction_and_sparse_reproduction_are_exact() -> None:
    assert delta_e((120, 140), (100, 110)) == (10, 40)
    assert delta_e((-140, -120), (-110, -100)) == (-40, -10)
    with pytest.raises(OverflowError):
        delta_e((2_147_483_647, 2_147_483_647), (-2_147_483_648, -2_147_483_648))

    previous = _evidence([(100, 110), (-110, -100)], [0b10_1011, 0b11_1000], [20, -40])
    current = _evidence([(500, 510), (-510, -500)], [0b10_1011, 0b11_1000], [20, -40])

    comparison = compare_integral_runs(previous, current, positive_rung_count=6)

    assert comparison["reproduced_masks"] == [0b10_1011, 0b11_1000]
    assert comparison["divergent_masks"] == [0, 0]
    assert comparison["reproduces"] is True
    assert comparison["overlaps"][0][0] == (10, 24)
    assert comparison["overlaps"][1][3] == (-20, -6)


def test_eligibility_movement_and_signed_response_divergence_stay_separate() -> None:
    previous = _evidence([(100, 110), (-110, -100)], [0b00_1111, 0b11_1100], [20, -40])
    current = _evidence([(100, 110), (-110, -100)], [0b01_1110, 0b11_1100], [20, -40])
    current["positive"][1][4] = (100, 110)  # type: ignore[index]

    comparison = compare_integral_runs(previous, current, positive_rung_count=6)

    assert comparison["previous_only_masks"] == [0b000001, 0]
    assert comparison["current_only_masks"] == [0b010000, 0]
    assert comparison["divergent_masks"] == [0, 0b010000]
    assert comparison["signed_gaps"][1][4] == 196
    assert comparison["reproduces"] is True


@pytest.mark.parametrize(
    ("state", "outcome", "comparison", "expected"),
    (
        ("authority_only", "sufficient", None, "candidate"),
        ("authority_only", "inconclusive", None, "authority_only"),
        ("authority_only", "fault", None, "empty"),
        ("candidate", "sufficient", "complete", "empty"),
        ("candidate", "sufficient", "mismatch", "candidate"),
        ("candidate", "inconclusive", None, "authority_only"),
        ("candidate", "fault", None, "empty"),
        ("candidate", "plan_mismatch", None, "candidate"),
    ),
)
def test_retained_slot_lifecycle_is_deterministic(
    state: str, outcome: str, comparison: str | None, expected: str
) -> None:
    assert stage_c_slot_transition(state, outcome, comparison) == expected
