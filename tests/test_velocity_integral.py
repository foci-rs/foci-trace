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
    _PLAN_SLOT_ORDER_SHIFT,
    integral_plan_digest,
    slot_direction_index,
    slot_for_observation,
    SLOT_ORDER_FORWARD_FIRST,
    SLOT_ORDER_PAIRED_OUT_AND_BACK,
    SLOT_ORDER_REVERSE_FIRST,
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

# Direction index per slot for each declared order, transcribed from
# IntegralSlotOrder in the firmware's integral_plan.rs. Kept as literal rows
# rather than derived from slot_direction_index, so the tests below compare the
# reconstruction against the schedule instead of against itself.
_DECLARED_DIRECTIONS: dict[int, tuple[int, ...]] = {
    SLOT_ORDER_FORWARD_FIRST: (0, 1, 0, 1, 0, 1, 0, 1),
    SLOT_ORDER_REVERSE_FIRST: (1, 0, 1, 0, 1, 0, 1, 0),
    SLOT_ORDER_PAIRED_OUT_AND_BACK: (0, 1, 1, 0, 0, 1, 1, 0),
}


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


def test_schema_eleven_plan_digest_hashes_stabilized_recovery_schedule() -> None:
    assert integral_plan_digest(_plan(schema_revision=11)) == 0x83E8FE6A458FE303


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
            # Mirrors `_compact_plan_marker`: the presence byte follows the
            # plan's actual combined-selected-response flag rather than being
            # hardcoded, so an immediate-scored (breakaway) plan's marker
            # carries a bare 0 and no interval bytes, matching what the
            # digest reconstruction in `velocity_integral.py` expects.
            present = bool(plan.get("combined_selected_response"))
            raw.extend(struct.pack("<B", int(present)))
            if present:
                raw.extend(
                    struct.pack(
                        "<iiii",
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
                # Firmware decodes this unconditionally on every schema revision
                # (protocol.py's `integral_plan_core` branch defaults it False
                # before the schema>=8 override below), so the fixture must too.
                "combined_selected_response": False,
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


def _moving_recovery_record(
    rung_index: int, sequence: int, plan: dict[str, object]
) -> tuple[bytes, dict[str, object]]:
    """Build a completed moving (outcome 1) Stage-C recovery.

    `_recovery_record` hardcodes `outcome: 0`, which cannot exercise the
    rest-presence rule (it only applies to outcomes 1 and 5). This carries
    geometry that satisfies `audit_rung_origin_recovery`'s outcome/binding
    reconstruction: a start offset outside the origin band (so the return
    distance is nonzero), a nonzero settled rate, and a runtime within the
    plan's maximum recovery slot.
    """
    values = {
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "stage": 1,
        "rung_index": rung_index,
        "velocity_p": int(plan["final_p"]),
        "planned_velocity_mrev_s": (4_383 if int(plan["schema_revision"]) >= 10 else 4_394),
        "start_offset_counts": 1_500,
        "closest_offset_counts": 900,
        "final_offset_counts": 900,
        "origin_band_counts": 1_000,
        "lower_rate_q": 300_000,
        "moving_duration_us": 1_000_000,
        "settle_duration_us": 500_000,
        "total_duration_us": 1_500_000,
        "peak_torque_target_abs": 500,
        "binding_source": 1,
        "outcome": 1,
    }
    raw = struct.pack(
        "<BIHBBHIiiiIQIIIHBB",
        9,
        *values.values(),
    )
    return raw, {"record_kind": "rung_origin_recovery", **values}


# Field order and widths below mirror `_decode_velocity_sweep_evidence`'s
# `kind == 11` (78-byte, Event 151) and `kind == 10` (89-byte, Event 150)
# branches in `foci_trace/protocol.py` exactly -- the only two shapes those
# branches accept. `test_stage_c_evidence_helpers_round_trip_through_the_decoder`
# decodes every record either helper builds and asserts the result equals
# what the helper claims, so a field-order or field-count mistake here fails
# immediately instead of silently building a dict `audit_velocity_integral`
# accepts while the raw bytes it also hashes would be rejected by the real
# firmware decoder.


def _stage_c_rest_selection_record(
    rung_index: int,
    slot: int,
    owner: int,
    sequence: int,
    plan_digest: int,
    *,
    direction: int = 0,
    disposition: int = 0,
) -> tuple[bytes, dict[str, object]]:
    """Build one canonical Event 151 scored-rest selection.

    Ordinary owners (3 anchor, 4 positive) carry the observation's actual
    direction and slot; the recovery owner (5) carries the decoder's `0xFF`
    sentinel for both (`protocol.py`'s "scored-rest recovery position"
    check). All three suffixes report a zero-width, zero-centered interval
    with `structural_codes` all 0 -- the same all-agreeing-on-zero-rate
    fixture `analyze_scored_rest_selection` produces for 256 identical
    zero-delta observations (see
    `test_scored_rest_selection_recomputes_all_suffixes_independently` in
    test_velocity_measurement.py) -- so the winning suffix is the full
    256-sample one and `disposition=0` (statistical selection) is
    decoder-valid.
    """
    recovery_owner = owner == 5
    stage = 0 if owner <= 2 else 1
    values = {
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "stage": stage,
        "owner": owner,
        "rung_index": rung_index,
        "direction": 0xFF if recovery_owner else direction,
        "slot": 0xFF if recovery_owner else slot,
        "plan_digest": plan_digest,
        "structural_code_0": 0,
        "structural_code_1": 0,
        "structural_code_2": 0,
        "interval_present_mask": 0b111,
        "usable_mask": 0b111,
        "zero_compatible_mask": 0b111,
        "selected_mask": 0b100,
        "selected_suffix_len": 256,
        "disposition": disposition,
        "interval_0_low": 0,
        "interval_0_high": 0,
        "interval_1_low": 0,
        "interval_1_high": 0,
        "interval_2_low": 0,
        "interval_2_high": 0,
    }
    raw = struct.pack("<BIHBBBBBQBBBBBBBHBqqqqqq", 11, *values.values())
    decoded = {
        "run_sequence": values["run_sequence"],
        "evidence_sequence": values["evidence_sequence"],
        "stage": values["stage"],
        "owner": values["owner"],
        "rung_index": values["rung_index"],
        "direction": values["direction"],
        "slot": values["slot"],
        "plan_digest": values["plan_digest"],
        "structural_codes": [
            values["structural_code_0"],
            values["structural_code_1"],
            values["structural_code_2"],
        ],
        "interval_present_mask": values["interval_present_mask"],
        "usable_mask": values["usable_mask"],
        "zero_compatible_mask": values["zero_compatible_mask"],
        "selected_mask": values["selected_mask"],
        "selected_suffix_len": values["selected_suffix_len"],
        "disposition": values["disposition"],
        "intervals_q": [
            [values["interval_0_low"], values["interval_0_high"]],
            [values["interval_1_low"], values["interval_1_high"]],
            [values["interval_2_low"], values["interval_2_high"]],
        ],
        "conditioned_rest_evidence": disposition == 0,
    }
    return raw, {"record_kind": "rest_selection", **decoded}


def _stage_c_recovery_rest_record(
    rung_index: int,
    sequence: int,
    plan_digest: int,
    *,
    velocity_p: int,
    start_offset_counts: int,
    closest_offset_counts: int,
    first_in_band_offset_counts: int,
    final_offset_counts: int,
    planned_return_distance_counts: int,
    disposition: int = 0,
) -> tuple[bytes, dict[str, object]]:
    """Build one canonical Event 150 recovery-rest diagnostic.

    Carries every field `audit_recovery_rest_geometry` consumes (an in-band
    entry and exit, a lead-in span containing its net displacement, and a
    scored range containing its net displacement), plus every remaining
    decoder-only field the wire shape requires: `velocity_p`,
    `closest_offset_counts`, `stationarity_code`/`interval_present` (the
    decoder's own "in-band, present" pairing) with a zero-width interval at
    zero, and `current_contact_code` ("no contact"). None of the
    decoder-only fields are checked by `audit_recovery_rest_geometry`, but
    all must be decoder-valid or the record does not survive a real
    capture.
    """
    values = {
        "run_sequence": 7,
        "evidence_sequence": sequence,
        "stage": 1,
        "rung_index": rung_index,
        "velocity_p": velocity_p,
        "plan_digest": plan_digest,
        "start_offset_counts": start_offset_counts,
        "closest_offset_counts": closest_offset_counts,
        "final_offset_counts": final_offset_counts,
        "planned_return_distance_counts": planned_return_distance_counts,
        "first_in_band_offset_counts": first_in_band_offset_counts,
        "displacement_to_band_counts": first_in_band_offset_counts - start_offset_counts,
        "entry_to_band_duration_us": 400_000,
        "lead_in_net_displacement_counts": 0,
        "lead_in_displacement_span_counts": 100,
        "scored_net_displacement_counts": 0,
        "scored_min_displacement_counts": -50,
        "scored_max_displacement_counts": 50,
        "stationarity_code": 0,
        "interval_present": 1,
        "exact_velocity_interval_low": 0,
        "exact_velocity_interval_high": 0,
        "peak_torque_target_abs": 0,
        "current_contact_code": 0,
        "disposition": disposition,
    }
    raw = struct.pack(
        "<BIHBBHQiiiIiiIiIiiiBBqqHBB",
        10,
        *values.values(),
    )
    decoded = {
        key: value
        for key, value in values.items()
        if key not in ("exact_velocity_interval_low", "exact_velocity_interval_high")
    }
    decoded["exact_velocity_interval_q"] = [
        values["exact_velocity_interval_low"],
        values["exact_velocity_interval_high"],
    ]
    return raw, {"record_kind": "recovery_rest", **decoded}


def test_stage_c_evidence_helpers_round_trip_through_the_decoder() -> None:
    """Decode every shape either synthetic-evidence helper can produce and
    assert the result equals what the helper claims. `audit_velocity_integral`
    trusts the separately built dictionary in each `(raw, dict)` pair while
    hashing the raw bytes into the terminal digest, so a fixture whose raw
    encoding a real decoder would reject can still reach an empty mismatch
    list -- proving nothing about a firmware-decodable capture. This is the
    check that makes that class of error impossible to repeat.
    """
    from foci_trace.protocol import _decode_velocity_sweep_evidence

    plan_digest = 0xABCDEF01

    for owner, direction, slot in ((3, 0, 2), (3, 1, 5), (4, 0, 0), (4, 1, 7), (5, 0, 0)):
        raw, expected = _stage_c_rest_selection_record(
            1, slot, owner, 9, plan_digest, direction=direction
        )
        assert len(raw) == 78
        assert _decode_velocity_sweep_evidence(raw) == expected

    raw, expected = _stage_c_recovery_rest_record(
        1,
        9,
        plan_digest,
        velocity_p=1_448,
        start_offset_counts=1_500,
        closest_offset_counts=900,
        first_in_band_offset_counts=950,
        final_offset_counts=900,
        planned_return_distance_counts=500,
    )
    assert len(raw) == 89
    assert _decode_velocity_sweep_evidence(raw) == expected


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


def _rest_boundary_record(
    sequence: int,
    slot_order: int = SLOT_ORDER_FORWARD_FIRST,
    positive_rung_index: int = 11,
    i_raw: int = 1_024,
) -> tuple[bytes, dict[str, object]]:
    # The boundary sits at slot 0, whose travel direction is forward under
    # forward-first and paired but reverse under reverse-first. The record
    # carries signed travel, not a direction index.
    signed_direction = 1 if _DECLARED_DIRECTIONS[slot_order][0] == 0 else -1
    raw = bytearray([5])
    raw.extend(
        struct.pack("<IHQBBHb", 7, sequence, 0, positive_rung_index, 0, i_raw, signed_direction)
    )
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
        "positive_rung_index": positive_rung_index,
        "slot": 0,
        "i_raw": i_raw,
        "direction": signed_direction,
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
    slot_order: int = SLOT_ORDER_FORWARD_FIRST,
    *,
    stabilized: bool = True,
) -> tuple[list[VelocityObservation], list[tuple[bytes, dict[str, object]]]]:
    plan = _plan(schema_revision)
    if slot_order != SLOT_ORDER_FORWARD_FIRST:
        plan["flags"] = int(plan.get("flags", 0)) | (slot_order << _PLAN_SLOT_ORDER_SHIFT)
    # Below schema 12 the grammar never runs (its floor is 12) and no
    # rest_selection/recovery_rest records are emitted, so `stabilized` is
    # ignored and every existing schema-1-through-11 fixture is unaffected.
    stage_c_scored = schema_revision >= 12
    if stage_c_scored:
        plan["combined_selected_response"] = stabilized
    plan["plan_digest"] = integral_plan_digest(plan)
    records = _plan_records(plan)
    sequence = 1
    records.append(_plan_marker(plan, sequence))
    sequence += 1
    raw_observations: list[VelocityObservation] = []
    gains = [0, *plan["positive_i"], 0]
    # The rung whose recovery is a completed moving return (outcome 1), used
    # to exercise the rest-presence rule. Rung 0 and the last rung are the
    # opening-anchor/closing-bookend pair; rung 1 is an ordinary positive rung.
    moving_recovery_rung = 1
    for rung_index, gain in enumerate(gains):
        direction_results: list[list[dict[str, object]]] = [[], []]
        ordinary_owner = 3 if rung_index in (0, len(gains) - 1) else 4
        for slot in range(8):
            direction = _DECLARED_DIRECTIONS[slot_order][slot]
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
            # Firmware emits the direction the schedule declares. Taking it from
            # the reconstruction instead would make the audit below compare the
            # reconstruction against itself, which passes however wrong it is.
            result["direction"] = direction
            direction_results[direction].append(result)
            records.append(_observation_record(result, sequence, int(plan["plan_digest"])))
            sequence += 1
            if stage_c_scored:
                records.append(
                    _stage_c_rest_selection_record(
                        rung_index,
                        slot,
                        ordinary_owner,
                        sequence,
                        int(plan["plan_digest"]),
                        direction=direction,
                    )
                )
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
            if stage_c_scored and rung_index == moving_recovery_rung:
                records.append(
                    _stage_c_rest_selection_record(
                        rung_index, 0, 5, sequence, int(plan["plan_digest"])
                    )
                )
                sequence += 1
                if stabilized:
                    records.append(
                        _stage_c_recovery_rest_record(
                            rung_index,
                            sequence,
                            int(plan["plan_digest"]),
                            velocity_p=int(plan["final_p"]),
                            start_offset_counts=1_500,
                            closest_offset_counts=900,
                            first_in_band_offset_counts=950,
                            final_offset_counts=900,
                            planned_return_distance_counts=500,
                        )
                    )
                    sequence += 1
                records.append(_moving_recovery_record(rung_index, sequence, plan))
            else:
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
    # `plan.combined_workflow` (integral_plan.rs:508) is exactly
    # `combined_selected_response.is_some()`, so the plan's own flag is the
    # single source of truth for whether this run is a combined workflow at
    # all. Below schema 12 every existing fixture leaves `_plan()`'s default
    # `combined_selected_response=True` untouched, so `plan_combined_workflow`
    # is unconditionally True there and every schema-8-plus terminal field
    # below is unaffected. At schema 12+, `stage_c_scored` already bound
    # `plan["combined_selected_response"] = stabilized` above, so this
    # follows `stabilized` exactly. Every schema-8-plus field the firmware
    # derives from `combined_workflow` must follow it too, not just the
    # combined-only tail: `outcome` itself is one of them --
    # `integral_sweep.rs:1174-1177` reports `Complete` (1) for a sufficient
    # combined run and `CompleteCandidate` (0) for a sufficient standalone
    # one, so hardcoding 1 for every schema >= 8 run (as this fixture used
    # to) claims a combined completion an immediate-scored run never
    # reaches. The combined-only terminal encoding (`combined_workflow`/
    # `final_p`/`target_status`/`target_terminus`, and the corresponding
    # recovery_flags bits) must likewise not claim a combined workflow the
    # plan does not have, or the fixture falsifies its own terminal to reach
    # an empty mismatch list.
    plan_combined_workflow = bool(plan.get("combined_selected_response", False))
    records.append(
        (
            b"",
            {
                "record_kind": "integral_terminal",
                "run_sequence": 7,
                "evidence_sequence": sequence,
                "plan_digest": plan["plan_digest"],
                "digest": digest,
                "outcome": ((1 if plan_combined_workflow else 0) if schema_revision >= 8 else 0),
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
                            (0xC0 if plan_combined_workflow else 0)
                            if schema_revision >= 8
                            else 8
                            if schema_revision >= 7
                            else 4
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
                    if schema_revision >= 8 and plan_combined_workflow
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


def test_fault_after_completed_observation_allows_missing_current_record() -> None:
    raw, records = _synthetic_complete_capture(schema_revision=11)
    observation_index = next(
        index
        for index, (_raw, value) in enumerate(records)
        if value["record_kind"] == "integral_observation"
        and value["rung_index"] == 3
        and value["slot"] == 3
    )
    terminal = {
        **records[-1][1],
        "evidence_sequence": int(records[observation_index][1]["evidence_sequence"]) + 1,
        "outcome": 3,
        "cause": 11,
        "emitted_observations": 28,
        "emitted_rungs": 3,
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
        "target_status": 255,
    }
    partial = [*records[: observation_index + 1], (b"", terminal)]

    audit = audit_velocity_integral(raw, partial)

    assert (
        "integral recovery causal order differs from the fixed schedule" not in audit["mismatches"]
    )


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


def _synthetic_boundary_capture(
    schema_revision: int = 5,
    slot_order: int = SLOT_ORDER_FORWARD_FIRST,
) -> tuple[list[VelocityObservation], list[tuple[bytes, dict[str, object]]]]:
    raw_observations, records = _synthetic_complete_capture(
        schema_revision=schema_revision, slot_order=slot_order
    )
    plan = _plan(schema_revision=schema_revision)
    if slot_order != SLOT_ORDER_FORWARD_FIRST:
        plan["flags"] = int(plan.get("flags", 0)) | (slot_order << _PLAN_SLOT_ORDER_SHIFT)
    plan["plan_digest"] = integral_plan_digest(plan)
    # The capture ends at the last positive rung, whose records are dropped and
    # replaced by the rest boundary. Rung 0 is the opening anchor, so positive
    # rung ordinal n sits at rung index n + 1.
    positive_rung_count = int(plan["positive_rung_count"])
    boundary_positive_rung = positive_rung_count - 1
    dropped_rung_index = boundary_positive_rung + 1
    terminal = records.pop()
    records = [
        (raw, value)
        for raw, value in records
        if not (
            int(value.get("rung_index", -1)) == dropped_rung_index
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
    boundary_raw, boundary = _rest_boundary_record(
        sequence,
        slot_order,
        boundary_positive_rung,
        int(plan["positive_i"][boundary_positive_rung]),  # type: ignore[index]
    )
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
            # One rung's worth of observations and the rung record itself are
            # gone, and the dropped rung's eligibility bit with them.
            "emitted_observations": 8 * (positive_rung_count + 1),
            "emitted_rungs": positive_rung_count + 1,
            "eligible_masks": [((1 << positive_rung_count) - 1) & ~(1 << boundary_positive_rung)]
            * 2,
        }
    )
    return raw_observations, [*resequenced, (boundary_raw, boundary), terminal]


def _synthetic_schema_five_boundary_capture() -> tuple[
    list[VelocityObservation], list[tuple[bytes, dict[str, object]]]
]:
    return _synthetic_boundary_capture(schema_revision=5)


def test_slot_direction_index_matches_the_declared_schedule() -> None:
    """Slot parity is the direction only under forward-first.

    The paired order is not a parity function at all: its twist alternates with
    the same-direction ordinal, giving F,R,R,F,F,R,R,F.
    """
    for slot_order, expected in _DECLARED_DIRECTIONS.items():
        actual = tuple(slot_direction_index(slot_order, slot) for slot in range(8))
        assert actual == expected, f"slot order {slot_order}"


def test_slot_lookup_inverts_the_direction_mapping_in_every_order() -> None:
    """Evidence is written by raw slot and read back by observation and
    direction. The two mappings are used at different sites, so a drift between
    them silently pairs a reading with the wrong stroke."""
    for slot_order in _DECLARED_DIRECTIONS:
        for slot in range(8):
            direction = slot_direction_index(slot_order, slot)
            assert slot_for_observation(slot_order, slot // 2, direction) == slot


def test_unknown_slot_order_is_rejected_rather_than_treated_as_forward_first() -> None:
    """Bits 2-3 hold four values and only three are declared. Falling back to
    forward-first would reconstruct a whole capture at the wrong directions."""
    with pytest.raises(ValueError):
        slot_direction_index(3, 0)


@pytest.mark.parametrize(
    "slot_order",
    (SLOT_ORDER_FORWARD_FIRST, SLOT_ORDER_REVERSE_FIRST, SLOT_ORDER_PAIRED_OUT_AND_BACK),
)
def test_every_slot_order_reconstructs_the_direction_the_schedule_declares(
    slot_order: int,
) -> None:
    """Aggregation-level cover for the reconstruction path.

    A type-level test of slot_direction_index alone cannot catch a site that
    derives direction itself instead of calling it, which is how all three sites
    came to disagree with firmware. This drives a whole capture through the audit
    at each declared order, so a site that bypasses the helper still shows up.
    """
    raw, records = _synthetic_complete_capture(schema_revision=8, slot_order=slot_order)

    audit = audit_velocity_integral(raw, records)

    direction_mismatches = [mismatch for mismatch in audit["mismatches"] if "direction" in mismatch]
    assert direction_mismatches == []


def test_mirrored_rest_boundary_direction_follows_the_slot_order() -> None:
    """The boundary carries signed travel for slot 0, which reverse-first flips.

    Covered separately from the observation path because the boundary check is a
    third site with its own sign convention.
    """
    # Schema 8, not 5: the plan flags carrying the slot order do not exist on
    # the wire before schema 6, so a schema-5 capture is forward-first by
    # construction and cannot exercise this at all.
    raw, records = _synthetic_boundary_capture(
        schema_revision=8, slot_order=SLOT_ORDER_REVERSE_FIRST
    )

    audit = audit_velocity_integral(raw, records)

    assert [mismatch for mismatch in audit["mismatches"] if "direction" in mismatch] == []


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


@pytest.mark.parametrize("schema_revision", (12, 13))
@pytest.mark.parametrize("slot_order", (0, 1, 2))
def test_every_slot_order_passes_the_plan_flag_whitelist(
    schema_revision: int, slot_order: int
) -> None:
    """Bits 2-3 carry the slot order. They were never added to the whitelist, so
    every mirrored and paired capture since the slot-order rollout has reported
    reserved bits. Those captures are schema 12, so 12 must be covered."""
    plan = _plan(schema_revision=schema_revision)
    plan["flags"] = int(plan.get("flags", 0)) | (slot_order << _PLAN_SLOT_ORDER_SHIFT)
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    from foci_trace.velocity_integral import _validate_plan

    _validate_plan(plan, mismatches)

    assert "plan recovery flags contain reserved bits" not in mismatches


def test_schema_four_sufficiency_requires_both_closing_bookends() -> None:
    from foci_trace.velocity_integral import _stage_c_run_sufficient

    opening = [(10, 20), (-20, -10)]
    masks = [0b111, 0b111]

    assert _stage_c_run_sufficient(4, opening, [(11, 21), (-21, -11)], masks) is True
    assert _stage_c_run_sufficient(4, opening, [(11, 21), None], masks) is False
    assert _stage_c_run_sufficient(3, opening, [None, None], masks) is True


@pytest.mark.parametrize("schema_revision", [0, 14])
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


def test_reconstruct_observation_applies_drift_bound_at_stage_c_schema_thirteen() -> None:
    """reconstruct_observation is the only production path that can reach the
    drift bound today (derive_integral_ladder in this module still rejects
    any full-path Stage C capture outside range(1, 13), so nothing end-to-end
    can drive this yet -- Task 10's job to widen). A wrong key name or a
    mis-threaded schema_revision argument here would otherwise be caught by
    nothing."""
    from foci_trace.velocity_measurement import STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION

    raw = [VelocityObservation(index, 1_000, 1, 0, 1, None) for index in range(256)]
    plan = {
        "family_size": 20,
        "target_velocity_rpm": 264,
        "encoder_counts_per_rev": 4_000,
        "final_p": 1_448,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
        "schema_revision": STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION,
    }

    result = reconstruct_observation(raw, {"rung_index": 0, "slot": 0, "i_raw": 0}, plan)

    # This ramp's causal 64-sample tail drifts far more than 200,000 ppm of
    # its settled rate, so the drift bound rejects it as ExcessiveDrift (5),
    # giving stationarity_code 6 (5 + 1) -- not 3 (ResolvedSlope, 2 + 1),
    # which the significance test below would produce for the identical raw
    # samples.
    assert result["stationarity_code"] == 6


def test_reconstruct_observation_uses_significance_test_below_stage_c_schema_thirteen() -> None:
    """The sibling of the test above, one schema revision below the Stage C
    drift-bound floor: the identical raw samples take the legacy significance
    test instead and reject for a different reason (ResolvedSlope, not
    ExcessiveDrift)."""
    from foci_trace.velocity_measurement import STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION

    raw = [VelocityObservation(index, 1_000, 1, 0, 1, None) for index in range(256)]
    plan = {
        "family_size": 20,
        "target_velocity_rpm": 264,
        "encoder_counts_per_rev": 4_000,
        "final_p": 1_448,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
        "schema_revision": STAGE_C_DRIFT_BOUND_MIN_SCHEMA_REVISION - 1,
    }

    result = reconstruct_observation(raw, {"rung_index": 0, "slot": 0, "i_raw": 0}, plan)

    assert result["stationarity_code"] == 3


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


def test_mirrored_slot_order_is_reported_and_hashed_only_when_set() -> None:
    """Stage C records the mirrored slot order in bit 2 of the plan flags byte.

    Firmware hashes the order only when mirrored, so forward-first plans keep the
    digest they had before slot order was recorded and every archived capture and
    retained acceptance-point binding stays valid.
    """
    forward_first = _plan()
    forward_first["flags"] = 0
    baseline = integral_plan_digest(forward_first)

    reverse_first = _plan()
    reverse_first["flags"] = 0b100

    assert integral_plan_digest(forward_first) == baseline
    assert integral_plan_digest(reverse_first) != baseline

    # Unlike the older flag bits, the slot order is hashed at every schema,
    # because firmware records it without moving the Stage-C schema revision.
    for schema_revision in (1, 7):
        unmirrored = _plan(schema_revision)
        unmirrored["flags"] = 0
        mirrored = _plan(schema_revision)
        mirrored["flags"] = 1 << _PLAN_SLOT_ORDER_SHIFT
        assert integral_plan_digest(mirrored) != integral_plan_digest(unmirrored)


def test_every_slot_order_is_decoded_and_separates_the_digest() -> None:
    """The plan flags carry the slot order as a two-bit field in bits 2 and 3.

    Forward-first is all-zero so plans predating the field decode unchanged, and
    the single-bit mirrored encoding stays valid.
    """
    digests = {}
    for slot_order, flags in ((0, 0b0000), (1, 0b0100), (2, 0b1000)):
        plan = _plan()
        plan["flags"] = flags
        digests[slot_order] = integral_plan_digest(plan)
    assert len(set(digests.values())) == 3, "each order must have its own identity"
    # Forward-first must hash as it did before the field existed.
    bare = _plan()
    bare["flags"] = 0
    assert digests[0] == integral_plan_digest(bare)


_STAGE_C_NONTERMINAL_KINDS = frozenset(
    (
        "integral_plan_marker",
        "integral_observation",
        "integral_rung",
        "recovery_rest",
        "rest_selection",
        "rung_origin_recovery",
        "velocity_primitive_current",
    )
)


def _renumber_stage_c_nonterminal_sequence(
    records: list[tuple[bytes, dict[str, object]]],
) -> int:
    """Recompute the contiguous 1..N evidence sequence in place, in both the
    decoded dict and the raw bytes.

    Plan fragments carry a fixed `evidence_sequence: 0` and are not part of
    the numbered stream (`audit_velocity_integral`'s own `nonterminal` filter
    excludes them). `integral_plan_marker`'s compact wire shape has no
    evidence_sequence field at all (see `_plan_marker`'s schema>=2 branch, an
    `<BIQ...>` header with no `H` after the run_sequence `I`), so only its
    dict is updated -- consistent with it always being emitted first and
    never moving. Every other kind here shares one raw header shape: `<B>`
    (kind marker) `<I>` (run_sequence) `<H>` (evidence_sequence) at a fixed
    byte offset, so the raw patch is one `struct.pack_into` per record.
    Rebuilding the digest from stale raw sequence bytes after a record is
    removed would hash a stream that no longer matches what the returned
    dicts claim -- exactly the fidelity gap this function exists to close.
    Returns the next sequence value, for the terminal.
    """
    sequence = 1
    for index, (raw, value) in enumerate(records):
        if value["record_kind"] not in _STAGE_C_NONTERMINAL_KINDS:
            continue
        value["evidence_sequence"] = sequence
        if value["record_kind"] != "integral_plan_marker":
            patched = bytearray(raw)
            struct.pack_into("<H", patched, 5, sequence)
            records[index] = (bytes(patched), value)
        sequence += 1
    return sequence


def _rebuild_stage_c_digest(
    plan: dict[str, object], records: list[tuple[bytes, dict[str, object]]]
) -> int:
    """Recompute the terminal digest the same way `_synthetic_complete_capture` does."""
    digest = 0xCBF29CE484222325
    for byte in _plan_marker(plan, 1)[0]:
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    for raw, value in records:
        if value["record_kind"] in ("integral_plan_core", "integral_plan_schedule"):
            continue
        for byte in raw:
            digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


def test_renumber_stage_c_nonterminal_sequence_keeps_raw_and_decoded_in_sync() -> None:
    """`_renumber_stage_c_nonterminal_sequence` must patch the raw evidence
    sequence bytes alongside the decoded dict, or a real decoder run over the
    renumbered stream would report different sequence numbers than the dicts
    the tests assert against -- the digest would then be rebuilt from bytes
    that disagree with the fixture's own claims about itself. Remove a
    record, as `test_stabilized_stage_c_recovery_without_event_150_reports_both_validators`
    does, and confirm every remaining raw record decodes to the exact
    sequence its dict carries.
    """
    from foci_trace.protocol import _decode_velocity_sweep_evidence

    _raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    records.pop()  # terminal: carries no raw payload, excluded from renumbering
    removed_index = next(
        index
        for index, (_r, value) in enumerate(records)
        if value["record_kind"] == "recovery_rest"
    )
    del records[removed_index]
    _renumber_stage_c_nonterminal_sequence(records)

    checked = 0
    for raw, value in records:
        if (
            value["record_kind"] not in _STAGE_C_NONTERMINAL_KINDS
            or value["record_kind"] == "integral_plan_marker"
        ):
            continue
        decoded = _decode_velocity_sweep_evidence(raw)
        assert decoded["evidence_sequence"] == value["evidence_sequence"]
        checked += 1
    assert checked > 0


def test_immediate_scored_stage_c_recovery_passes_with_no_event_150() -> None:
    """Under `ImmediateScored` the firmware emits the owner-5 selection and no
    `recovery_rest` at all. Before this task both the rest-presence rule and
    the causal-order builder demanded one, rejecting every such run. Assert an
    empty mismatch list, not the absence of the two specific messages: that is
    what proves the whole capture -- plan, marker, and terminal -- is
    schedule-consistent, not just that the two targeted checks were silenced.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=False)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []
    assert audit["matches_firmware"] is True


def test_stabilized_stage_c_recovery_passes_with_event_150() -> None:
    """Under `Stabilized` the firmware emits the owner-5 selection, then Event
    150, then the recovery. Mirrors the immediate-scored test above under the
    other schedule; both must reach an empty mismatch list for the rule to be
    schedule-derived rather than hardcoded either way.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []
    assert audit["matches_firmware"] is True


def test_stabilized_stage_c_recovery_without_event_150_reports_both_validators() -> None:
    """Removing Event 150 from an otherwise-valid stabilized capture must still
    be rejected, proving the presence rule has teeth under `Stabilized` and
    was not simply disabled. Two validators fire on the same real stream:
    Task 6's presence check, and Task 5's grammar (the owner-5 selection's
    successor is no longer `recovery_rest`). Assert exactly that pair and
    nothing else.
    """
    plan = _plan(12)
    plan["combined_selected_response"] = True
    plan["plan_digest"] = integral_plan_digest(plan)
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    terminal_raw, terminal = records.pop()
    removed_index = next(
        index
        for index, (_raw, value) in enumerate(records)
        if value["record_kind"] == "recovery_rest"
    )
    del records[removed_index]
    next_sequence = _renumber_stage_c_nonterminal_sequence(records)
    terminal["evidence_sequence"] = next_sequence
    terminal["digest"] = _rebuild_stage_c_digest(plan, records)
    records.append((terminal_raw, terminal))

    audit = audit_velocity_integral(raw, records)

    # The grammar (Task 5) runs before the presence check (Task 6) inside
    # `audit_velocity_integral`, so its message appears first.
    assert audit["mismatches"] == [
        "Stage C recovery selection does not precede Event 150",
        "moving Stage C recovery lacks recovery rest evidence",
    ]


# The four sabotage tests below isolate the recovery_flags workflow bit and
# the decoded combined tail as two *independent* signals: a false terminal
# can carry either one without the other (the reviewer reproduced exactly
# that gap in round 1, where a sabotage test that always mutated both at
# once never exercised the tail check on its own). Each test below flips
# exactly one signal, in each direction, and leaves every other honest field
# alone.


def test_immediate_scored_terminal_claiming_the_combined_workflow_flag_is_rejected() -> None:
    """Flag-only: a genuine immediate-scored terminal's `recovery_flags` byte
    must not carry the combined-workflow bit, even with no combined tail
    present. Before this task's production fix, the combined-terminal
    validation ran unconditionally for every schema >= 8 run and demanded
    that flag regardless of the plan.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=False)
    terminal_raw, terminal = records[-1]
    terminal = {**terminal, "recovery_flags": 0x80}
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == ["terminal combined-workflow flag differs from the plan"]


def test_stabilized_terminal_dropping_the_combined_workflow_flag_is_rejected() -> None:
    """Flag-only, opposite direction: a genuine combined (`Stabilized`)
    terminal's `recovery_flags` byte must carry the combined-workflow bit.
    The combined target-status mask lives in the same byte, so clearing the
    whole byte also invalidates the encoded target status -- both checks
    stay active for a plan that claims a combined workflow, which is why the
    agreement check must run before, not instead of, the rest of the
    combined-terminal validation.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    terminal_raw, terminal = records[-1]
    terminal = {**terminal, "recovery_flags": 0}
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "terminal combined-workflow flag differs from the plan",
        "combined terminal target flags differ from canonical status",
    ]


def test_immediate_scored_terminal_carrying_the_combined_tail_is_rejected() -> None:
    """Tail-only: a genuine immediate-scored terminal must not carry the
    decoded combined tail (`combined_workflow`/`final_p`/`target_status`/
    `target_terminus`, only ever set by `protocol.py` when the wire payload
    is the 66-byte combined variant), even with an honest `recovery_flags`
    byte. Before this round, the agreement check only looked at the
    recovery_flags bit, so a terminal that carried the tail while leaving
    the flag clear produced no mismatch at all -- exactly the gap the
    reviewer reproduced.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=False)
    terminal_raw, terminal = records[-1]
    terminal = {
        **terminal,
        "combined_workflow": True,
        "final_p": 1_024,
        "target_status": 3,
        "target_terminus": None,
    }
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == ["terminal combined tail differs from the plan"]


def test_stabilized_terminal_missing_the_combined_tail_is_rejected() -> None:
    """Tail-only, opposite direction: a genuine combined terminal must carry
    the decoded combined tail, even with an honest `recovery_flags` byte.
    Dropping the tail alone also leaves `target_status` absent, so the
    causal-ladder target-status check fires too -- both checks stay active
    for a plan that claims a combined workflow.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    terminal_raw, terminal = records[-1]
    terminal = {
        key: value
        for key, value in terminal.items()
        if key not in ("combined_workflow", "final_p", "target_status", "target_terminus")
    }
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "terminal combined tail differs from the plan",
        "combined target status differs from causal ladder evidence",
    ]


def test_standalone_sufficient_terminal_claiming_the_combined_outcome_is_rejected() -> None:
    """A sufficient, genuinely standalone (immediate-scored) run must report
    outcome 0 (`CompleteCandidate`), never outcome 1 (`Complete`):
    `integral_sweep.rs:1174-1177` reports `Complete` only when
    `plan.combined_workflow`. Before this round the fixture hardcoded
    outcome 1 for every schema >= 8 run regardless of schedule, and
    production accepted either outcome for a sufficient standalone run, so
    this exact falsification reached an empty mismatch list. Every other
    terminal field stays honest (no combined flag, no combined tail), so
    only the outcome check fires.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=False)
    terminal_raw, terminal = records[-1]
    terminal = {**terminal, "outcome": 1}
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == ["standalone sufficient run falsely reports the combined outcome"]


def test_combined_sufficient_terminal_claiming_the_standalone_outcome_is_rejected() -> None:
    """Opposite direction: a sufficient, genuinely combined (`Stabilized`)
    run must report outcome 1 (`Complete`), never outcome 0
    (`CompleteCandidate`). This check predates this fix round -- it is not
    new production behavior -- but had no direct test coverage; added here
    to close that gap alongside the new standalone-direction test above.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    terminal_raw, terminal = records[-1]
    terminal = {**terminal, "outcome": 0}
    records[-1] = (terminal_raw, terminal)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == ["combined sufficient run did not complete without retention"]


def _fault_the_terminal(
    records: list[tuple[bytes, dict[str, object]]],
) -> list[tuple[bytes, dict[str, object]]]:
    """Turn an honest, complete terminal into an honest, sufficient-but-faulted one.

    Mirrors what `integral_fault_terminal`/`evidence_terminal`'s `Err(_)` arm
    (`integral_sweep.rs`) actually report: `outcome=3`, `cause=4`
    (`INTEGRAL_CAUSE_INTEGRITY`), and every evidence-derived terminal field
    reset to empty, because both paths build the terminal from
    `IntegralRunEvidence::new(...)` rather than the run's real evidence.
    `sufficient` in `audit_velocity_integral` is computed by *replaying* the
    still-emitted pre-fault records, not from these reset terminal fields, so
    it stays True even though the terminal itself now reports empty --
    exactly the shape of the real `foc213-transport-validated` capture that
    exposed this bug (outcome=3, cause=4, sufficient replay, standalone plan).
    """
    terminal_raw, terminal = records[-1]
    terminal = {
        **terminal,
        "outcome": 3,
        "cause": 4,
        "eligible_masks": [0, 0],
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
    }
    return [*records[:-1], (terminal_raw, terminal)]


def test_standalone_sufficient_but_faulted_terminal_is_accepted() -> None:
    """A faulted run's outcome carries no schedule information: `Fault` is
    reached without ever consulting `plan.combined_workflow`
    (`integral_fault_terminal`, and the `Err(_)` arm of `evidence_terminal`,
    both in `integral_sweep.rs`). Before the `not faulted` guard was added,
    this exact shape -- a sufficient replay whose terminal reports outcome 3
    on a standalone plan -- falsely tripped the standalone-direction outcome
    check, which is precisely what happened when Task 7 replayed the real
    `foc213-transport-validated` capture. Assert an empty mismatch list: the
    fault is real firmware output, not a defect to report.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=False)
    records = _fault_the_terminal(records)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []


def test_combined_sufficient_but_faulted_terminal_is_accepted() -> None:
    """Mirror of the standalone case above: a faulted run's outcome carries
    no schedule information under `Stabilized` either, so a sufficient
    replay whose terminal reports outcome 3 on a combined plan must not trip
    the combined-direction outcome check. `sufficient` is unaffected by
    `plan.combined_workflow` (it depends only on the replayed causal
    ladder), so a faulted combined run can be sufficient by exactly the same
    mechanism as the standalone case -- this is the "determine from firmware
    whether a faulted combined run can be sufficient" the fix round asked
    for, applied as a real fixture rather than an assertion on paper.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, stabilized=True)
    records = _fault_the_terminal(records)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []
