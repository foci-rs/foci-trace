import io
import json
import re
import struct
from fractions import Fraction
from pathlib import Path

import pytest

from foci_trace.analyze import analyze_capture
from foci_trace.capture_file import RECORD_TYPE_FIRMWARE_FRAME, CaptureReader
from foci_trace.protocol import FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE, decode_frame
from foci_trace.velocity_integral import (
    _PLAN_SLOT_ORDER_SHIFT,
    SLOT_ORDER_FORWARD_FIRST,
    SLOT_ORDER_PAIRED_OUT_AND_BACK,
    SLOT_ORDER_REVERSE_FIRST,
    UnsupportedIntegralRepresentationError,
    _compact_plan_marker,
    _retained_stage_c_evidence,
    _validate_plan,
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
    integral_numerator_for_build_features,
    integral_plan_digest,
    observation_role,
    reconstruct_observation,
    recovery_quantization_exposed,
    select_transition_capable_test_point,
    slot_direction_index,
    slot_for_observation,
    stage_c_combined_workflow,
    stage_c_slot_transition,
    velocity_i_tau_us,
)
from foci_trace.velocity_measurement import (
    BUILD_FEATURE_Q4_12_GAINS,
    VelocityObservation,
    audit_rung_origin_recovery,
    audit_scored_rest_grammar,
    reconstruct_build_features,
)

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
    if schema_revision >= 14:
        # The breakaway-seeded campaign
        # (`plan_breakaway_velocity_integral_response`, integral_plan.rs:1387).
        # It has no Stage-B sweep, so `StageCAuthority::new_breakaway` zeroes
        # every directional field; it passes `combined_selected_response: None`
        # and `probe_constrained_test_point: false`; and it passes
        # `append_exact_cap: false`, so the ladder is the plain
        # `build_integral_ladder` one that stops at the last whole octave
        # instead of the combined ladder's exact 1_310 cap. Because it is not a
        # combined workflow, firmware neither reserves nor emits
        # `slot_transition_allowance_us` (`integral_reporting.rs:316-321`) and
        # does not add the recovery stabilization reserve
        # (`integral_plan.rs:1544-1553`), which returns the workflow durations
        # to their schema-5 values.
        #
        # Every value below is the value the real schema-14 plan in
        # `foc213-transport-validated-20260803-192710.bin` carries, field for
        # field, except for `requested_velocity_mrev_s` and the two position
        # headrooms, which that capture reports as 6_666/7_025/7_025 and which
        # nothing in the audit reconstructs.
        del plan["selected_response_intervals_q"]
        del plan["slot_transition_allowance_us"]
        plan.update(
            {
                # `protocol.py` decodes a plan core with no trailing combined
                # block as `combined_selected_response: False`, which is what
                # the real capture reports, so the fixture models the decoded
                # value rather than firmware's `Option::None`.
                "combined_selected_response": False,
                "final_p": 951,
                "usable_torque_limit": 2_534,
                "directional_memberships": [0, 0],
                "stage_b_joint_membership": 0,
                "joint_membership": 0,
                "directional_validity": [0, 0],
                "reduced_margin": False,
                "directional_pooled_intervals": [[0, 0], [0, 0]],
                "positive_rung_count": 12,
                "positive_i": [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024],
                "family_size": 56,
                "total_rung_count": 14,
                "expected_observations": 112,
                "nominal_workflow_ms": 165_950,
                "maximum_workflow_ms": 182_512,
                "recovery_nominal_slot_us": 1_816_958,
                "recovery_slot_count": 14,
                "flags": 0,
            }
        )
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


@pytest.mark.parametrize(
    ("schema_revision", "combined"),
    [(1, False), (7, False), (8, True), (12, True), (13, True), (14, False)],
)
def test_stage_c_workflow_follows_the_firmware_schema_ranges(
    schema_revision: int, combined: bool
) -> None:
    """The four firmware constants that partition the Stage-C schema space:
    `STAGE_C_EVIDENCE_SCHEMA_REVISION = 7` (integral_plan.rs:24),
    `FIRST_COMBINED_STAGE_C_SCHEMA_REVISION = 8` and
    `COMBINED_STAGE_C_SCHEMA_REVISION = 13` (combined_plan.rs:20,22), and
    `BREAKAWAY_STAGE_C_SCHEMA_REVISION = 14` (combined_plan.rs:40). Only 8
    through 13 are combined; 14 is a standalone breakaway run, which is why
    `schema >= 8` on its own is not a combined-workflow test.
    """
    assert stage_c_combined_workflow(schema_revision) is combined


def test_breakaway_schema_ladder_omits_the_combined_exact_cap() -> None:
    """`plan_breakaway_velocity_integral_response` passes
    `append_exact_cap: false` (integral_plan.rs:1438), so schema 14 builds the
    plain `build_integral_ladder` whole-octave ladder while the combined
    schemas append the exact `i_top_cap` rung. Same geometry, two ladders --
    the real schema-14 plan in `foc213-transport-validated` carries exactly the
    twelve rungs asserted here.
    """
    geometry = {
        "pwm_hz": 25_000,
        "moving_stroke_us": 512_000,
        "shortest_suffix_us": 64_000,
        "maximum_i": 32_767,
        "build_revision": CURRENT_Q4_12_BUILD,
    }
    breakaway = derive_integral_ladder(
        geometry["pwm_hz"],
        geometry["moving_stroke_us"],
        geometry["shortest_suffix_us"],
        geometry["maximum_i"],
        schema_revision=14,
        build_revision=geometry["build_revision"],
    )
    combined = derive_integral_ladder(
        geometry["pwm_hz"],
        geometry["moving_stroke_us"],
        geometry["shortest_suffix_us"],
        geometry["maximum_i"],
        schema_revision=13,
        build_revision=geometry["build_revision"],
    )

    assert breakaway["i_top_cap"] == 1_310
    assert breakaway["positive_i"] == [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1_024]
    assert combined["positive_i"] == [*breakaway["positive_i"], 1_310]


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


def test_compact_plan_marker_suffix_belongs_to_the_combined_workflow() -> None:
    """`encode_velocity_integral_plan_marker` (`integral_reporting.rs:454-471`)
    appends the presence byte and the four interval endpoints only inside
    `if plan.combined_workflow || acquisition_kind is Acceptance*`. A breakaway
    plan's marker therefore ends after `usable_torque_limit`, with no trailing
    byte at all -- emitting a zero byte there hashed one byte firmware never
    hashed and made every breakaway capture's terminal digest disagree.

    Both markers share an identical fixed prefix (the field list is the same
    from schema 2 up), so the combined marker's tail past the breakaway
    marker's length is exactly the suffix under test.
    """
    combined = _plan(schema_revision=13)
    combined["plan_digest"] = integral_plan_digest(combined)
    breakaway = _plan(schema_revision=14)
    breakaway["plan_digest"] = integral_plan_digest(breakaway)

    combined_marker = _compact_plan_marker(combined, 7)
    breakaway_marker = _compact_plan_marker(breakaway, 7)

    assert combined_marker[len(breakaway_marker) :] == struct.pack(
        "<Biiii",
        1,
        *(
            endpoint
            for interval in combined["selected_response_intervals_q"]
            for endpoint in interval
        ),
    )


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
        # Mirrors `_compact_plan_marker`, which mirrors
        # `encode_velocity_integral_plan_marker`
        # (`integral_reporting.rs:454-471`): the presence byte and the four
        # interval endpoints exist only for a combined workflow. A breakaway
        # plan's marker ends after `usable_torque_limit` with no trailing byte
        # at all -- not a byte carrying zero.
        if stage_c_combined_workflow(int(plan["schema_revision"])):
            raw.extend(struct.pack("<B", 1))
            # Same fallback as `_compact_plan_marker`, so the forged capture
            # built by `forge_standalone` (which strips the block a combined
            # plan must have) still hashes the same bytes on both sides and is
            # rejected on its own merits rather than on a digest mismatch.
            intervals = plan.get("selected_response_intervals_q") or ((0, 0), (0, 0))
            raw.extend(
                struct.pack(
                    "<iiii",
                    *(endpoint for interval in intervals for endpoint in interval),
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
    # Only a combined plan's schedule fragment carries the allowance on the
    # wire (`integral_reporting.rs:316-321`); a breakaway schema-14 fragment
    # ends without it, which is why `protocol.py` decodes it from the remaining
    # byte count rather than from the schema. Emit whatever the plan carries,
    # exactly as the encoder does.
    if "slot_transition_allowance_us" in plan:
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
    if bool(plan.get("combined_selected_response", False)):
        # `encode_velocity_integral_plan_trace` appends this block only for
        # `Some(selected)` (`integral_reporting.rs:289-301`), so a breakaway
        # plan core keeps `protocol.py`'s decoded default of False and no
        # interval key at all.
        records[0][1]["combined_selected_response"] = True
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
    # Mirrors `encode_rung` (`integral_reporting.rs:581`): the three horizon
    # fields are appended only when `moving_horizon_us != 0`, and
    # `integral_sweep.rs:2358` leaves that zero unless the plan is a combined
    # workflow. A schema-14 breakaway rung therefore carries none of them.
    if (
        plan is not None
        and int(plan["schema_revision"]) >= 9
        and stage_c_combined_workflow(int(plan["schema_revision"]))
    ):
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
    forge_standalone: bool = False,
) -> tuple[list[VelocityObservation], list[tuple[bytes, dict[str, object]]]]:
    """Build a complete, internally consistent Stage-C capture.

    `forge_standalone` is a deliberate forgery switch used by exactly one
    rejection test. It strips a combined-schema plan's selected response and
    everything firmware derives from it -- Event 150, the combined recovery
    flag, the combined terminal tail, the combined outcome -- while keeping the
    result self-consistent end to end, digests included. That is the capture no
    firmware builder can produce and that the audit used to accept. Never pass
    it to a test that asserts an empty mismatch list.
    """
    plan = _plan(schema_revision)
    if slot_order != SLOT_ORDER_FORWARD_FIRST:
        plan["flags"] = int(plan.get("flags", 0)) | (slot_order << _PLAN_SLOT_ORDER_SHIFT)
    if forge_standalone:
        plan["combined_selected_response"] = False
        plan.pop("selected_response_intervals_q", None)
    # Below schema 12 the grammar never runs (its floor is 12) and no
    # rest_selection/recovery_rest records are emitted, so the schedule below
    # is inert and every existing schema-1-through-11 fixture is unaffected.
    stage_c_scored = schema_revision >= 12
    # `integral_recovery_rest_schedule` (`integral_sweep.rs:779-785`) picks
    # `Stabilized` for a combined workflow and `ImmediateScored` otherwise, and
    # which workflow built the plan is fixed by the schema revision. This was
    # once a caller-supplied flag, which let a caller ask for a schema-12
    # immediate-scored capture -- a plan firmware cannot build, since every
    # schema in 8..13 comes from `plan_combined_velocity_integral_response`.
    # Immediate-scored Stage C is reached by asking for schema 14.
    stabilized = bool(plan.get("combined_selected_response", False))
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
    # `combined_selected_response.is_some()`. Reading it back off the plan
    # dict here, rather than calling `stage_c_combined_workflow` a second
    # time, keeps this an assertion about the fixture itself: the terminal is
    # built from the same field the plan record publishes, so a `_plan()`
    # revision that ever disagreed with its schema would show up as a
    # falsified terminal instead of being papered over. Every schema-8-plus
    # field the firmware
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


def test_observation_role_is_the_fixed_out_and_back_alternation():
    """Role is origin-relative and does not move with slot order.

    A primitive's outbound or inbound role is fixed by its position in the
    stroke pair: even slots go out, odd slots come back, under every schedule.
    What the slot order changes is which travel *direction* is paired with each
    role, which is what `_slot_order_parity_twist` encodes.
    """
    for slot_order in (
        SLOT_ORDER_FORWARD_FIRST,
        SLOT_ORDER_REVERSE_FIRST,
        SLOT_ORDER_PAIRED_OUT_AND_BACK,
    ):
        for slot in range(8):
            assert observation_role(slot_order, slot) == slot % 2

    # Accepts slot order for symmetry with slot_direction_index, and validates
    # it, so a capture with an unknown schedule fails loudly here too.
    with pytest.raises(ValueError):
        observation_role(99, 0)


def test_slot_order_relates_role_and_direction_as_declared():
    """The relationships the paired experiment depends on.

    Under the two sequential orders role and direction are perfectly
    correlated, so a rate difference between roles cannot be told apart from a
    difference between directions. Paired breaks that, which is the entire
    reason for running it.
    """
    forward = [
        (
            observation_role(SLOT_ORDER_FORWARD_FIRST, slot),
            _DECLARED_DIRECTIONS[SLOT_ORDER_FORWARD_FIRST][slot],
        )
        for slot in range(8)
    ]
    assert all(role == direction for role, direction in forward)

    reverse = [
        (
            observation_role(SLOT_ORDER_REVERSE_FIRST, slot),
            _DECLARED_DIRECTIONS[SLOT_ORDER_REVERSE_FIRST][slot],
        )
        for slot in range(8)
    ]
    assert all(role != direction for role, direction in reverse)

    # The exact paired schedule, not merely "decorrelated" -- a wrong pattern
    # can decorrelate too, and would silently produce a plausible analysis.
    assert _DECLARED_DIRECTIONS[SLOT_ORDER_PAIRED_OUT_AND_BACK] == (0, 1, 1, 0, 0, 1, 1, 0)
    paired = [
        (observation_role(SLOT_ORDER_PAIRED_OUT_AND_BACK, slot), direction)
        for slot, direction in enumerate(_DECLARED_DIRECTIONS[SLOT_ORDER_PAIRED_OUT_AND_BACK])
    ]
    assert paired == [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, 1), (0, 1), (1, 0)]
    # Every (direction, role) cell is populated, which is what makes the two
    # effects separable.
    assert set(paired) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_audit_emits_the_role_alongside_direction():
    raw, records = _synthetic_complete_capture(
        schema_revision=11, slot_order=SLOT_ORDER_PAIRED_OUT_AND_BACK
    )

    response = audit_velocity_integral(raw, records)

    for item in response["observations"]:
        assert item["role"] == int(item["slot"]) % 2
    cells = {(int(item["direction"]), int(item["role"])) for item in response["observations"]}
    assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_excluded_observation_carries_the_statistic_behind_its_refusal():
    """An excluded observation must say how it failed, not only that it did.

    Emitted separately from the compared fields because firmware zeroes these on
    its own excluded records; folding them in would make the replay disagree
    with the device it exists to verify.
    """
    raw, _records = _synthetic_complete_capture(schema_revision=11)
    plan = _plan(schema_revision=11)

    excluded = reconstruct_observation(
        # Too few samples for any declared suffix: refused before any fit, so
        # there is nothing to derive and nothing must be invented.
        raw[:8],
        {"rung_index": 0, "slot": 0, "i_raw": 0},
        plan,
    )

    assert excluded["stationarity_code"] != 0
    assert "offline_exclusion_detail" not in excluded


def test_audit_emits_per_observation_records_with_slot_and_direction():
    """Per-observation evidence has to survive into the audit output.

    The replay already verifies every observation field against an independent
    reconstruction, then kept only the per-direction-per-rung aggregates. Role
    lives in slot position within a rung, so pooling the slots away makes the
    role-versus-direction question unanswerable from the output even though the
    evidence was in hand.
    """
    raw, records = _synthetic_complete_capture(schema_revision=11)

    response = audit_velocity_integral(raw, records)

    observations = response["observations"]
    assert observations, "the audit must emit the observations it verified"
    # Identity, geometry and the measured rate: what the role/direction
    # separation needs, and what the aggregates cannot supply.
    for field in ("rung_index", "slot", "direction", "rate_low", "rate_mean", "rate_high"):
        assert field in observations[0], f"observation record is missing {field}"
    # One per (rung, slot), no duplicates, in a deterministic order.
    keys = [(int(item["rung_index"]), int(item["slot"])) for item in observations]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)


def test_audit_observation_directions_follow_a_paired_slot_order():
    """Under paired order the emitted direction must track the schedule.

    This is the property the whole paired experiment rests on: if direction
    still came from slot parity the emitted records would be wrong in exactly
    the case they are being collected for.
    """
    raw, records = _synthetic_complete_capture(
        schema_revision=11, slot_order=SLOT_ORDER_PAIRED_OUT_AND_BACK
    )

    response = audit_velocity_integral(raw, records)

    declared = _DECLARED_DIRECTIONS[SLOT_ORDER_PAIRED_OUT_AND_BACK]
    for item in response["observations"]:
        slot = int(item["slot"])
        assert int(item["direction"]) == declared[slot], f"slot {slot} direction"
    # Paired order decorrelates direction from slot parity; forward-first does
    # not. Without that the experiment cannot separate role from direction.
    assert any(int(item["direction"]) != int(item["slot"]) & 1 for item in response["observations"])


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


def test_build_features_bit0_selects_integral_representation() -> None:
    assert integral_numerator_for_build_features(0) == 65_536
    assert integral_numerator_for_build_features(BUILD_FEATURE_Q4_12_GAINS) == 1_048_576


def test_reconstruct_build_features_maps_legacy_captures_by_namespace() -> None:
    # The legacy Q8.8 build predates every rule bit.
    assert reconstruct_build_features(LEGACY_Q8_8_BUILD, 4, stage="c") == 0b0000
    # Q4.12 combined Stage C (schema 13) had only the drift bound, not the
    # suffix walk (>=15) or the rest walk (>=16).
    assert reconstruct_build_features(CURRENT_Q4_12_BUILD, 13, stage="c") == 0b0011
    # Q4.12 breakaway Stage C (schema 16) had all four rules.
    assert reconstruct_build_features(CURRENT_Q4_12_BUILD, 16, stage="c") == 0b1111
    # Stage B numbers its own thresholds (14 drift / 16 suffix / 17 rest).
    assert reconstruct_build_features(CURRENT_Q4_12_BUILD, 14, stage="b") == 0b0011
    assert reconstruct_build_features(CURRENT_Q4_12_BUILD, 17, stage="b") == 0b1111


def test_reconstruct_build_features_refuses_an_unknown_build() -> None:
    with pytest.raises(UnsupportedIntegralRepresentationError):
        reconstruct_build_features(0xDEADBEEF, 13, stage="c")


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


@pytest.mark.parametrize("schema_revision", [0, 17])
def test_integral_ladder_rejects_unsupported_schema(schema_revision: int) -> None:
    """Schema 16 -- the breakaway campaign -- is a real firmware schema
    (`BREAKAWAY_STAGE_C_SCHEMA_REVISION`, combined_plan.rs:40) and builds its
    ladder without the combined exact-cap append, so the rejected boundary sits
    one above it. 17 does not exist as a Stage-C revision in firmware.
    """
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
    """Direct coverage of the drift bound at its floor schema. The schema-14
    breakaway fixture now drives `reconstruct_observation` above that floor end
    to end, but only with samples the bound accepts; this pins the rejecting
    side, where a wrong key name or a mis-threaded schema_revision argument
    would otherwise be caught by nothing."""
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

    _raw, records = _synthetic_complete_capture(schema_revision=12)
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

    The capture is schema 14, the breakaway campaign, because that is the only
    Stage-C schema whose plan carries no combined selected response and so
    reaches `ImmediateScored`. This fixture previously claimed schema 12 with
    the response omitted, which no firmware builder can produce.
    """
    raw, records = _synthetic_complete_capture(schema_revision=14)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []
    assert audit["matches_firmware"] is True


def test_schema_twelve_capture_without_its_combined_response_is_rejected() -> None:
    """The forgery this fix exists to stop: a schema-12 capture that omits its
    combined selected response and every consequence of it -- Event 150, the
    combined recovery flag, the combined terminal tail, the combined outcome --
    while staying self-consistent, digests included. No firmware builder can
    produce it: `plan_combined_velocity_integral_response` is the only builder
    for schemas 8 through 13 and it always supplies `Some(selected)`
    (integral_plan.rs:1214). Deriving the workflow from the plan's own field
    let this capture elect standalone semantics for itself and reach an empty
    mismatch list; deriving it from the schema revision rejects it on nine
    independent counts, starting with the schema/workflow contradiction.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12, forge_standalone=True)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "plan combined selected response contradicts its schema workflow: "
        "schema=12 expects a combined selected response",
        "combined plan selected response intervals are invalid",
        "Stage C recovery selection does not precede Event 150",
        "moving Stage C recovery lacks recovery rest evidence",
        "terminal combined-workflow flag differs from the plan",
        "terminal combined tail differs from the plan",
        "combined target status differs from causal ladder evidence",
        "combined terminal target flags differ from canonical status",
        "combined sufficient run did not complete without retention",
    ]


def test_breakaway_schema_plan_claiming_a_combined_response_is_rejected() -> None:
    """Opposite direction: schema 14 is built by
    `plan_breakaway_velocity_integral_response` with
    `combined_selected_response: None` (integral_plan.rs:1435), so a schema-14
    plan that claims one is equally impossible. Checked at `_validate_plan`
    directly -- the whole-capture path is covered by the schema-12 forgery
    above, and this pins the mapping's other edge.
    """
    plan = _plan(14)
    plan["combined_selected_response"] = True
    plan["selected_response_intervals_q"] = [
        [12_900_000, 13_200_000],
        [-13_100_000, -12_800_000],
    ]
    plan["plan_digest"] = integral_plan_digest(plan)
    mismatches: list[str] = []

    _validate_plan(plan, mismatches)

    assert (
        "plan combined selected response contradicts its schema workflow: "
        "schema=14 expects no combined selected response"
    ) in mismatches


def test_breakaway_rung_carrying_a_response_horizon_is_rejected() -> None:
    """The response horizon belongs to the combined workflow, not to schema 9.
    `integral_sweep.rs:2358` computes `(tau_us, horizon_class)` only
    `if data.plan.combined_workflow` and leaves `moving_horizon_us` zero
    otherwise, and `encode_rung` (`integral_reporting.rs:581`) then drops all
    three fields from the wire when `moving_horizon_us == 0`. A schema-14
    breakaway rung therefore carries none of them, which is why the audit no
    longer demands them at schema >= 9 alone.

    Gating the expectation must not degrade into ignoring the fields: splice a
    horizon into one breakaway rung's wire record, in the exact position and
    widths `encode_rung` would have used, and require it to be rejected. The
    combined direction stays covered by
    `test_current_combined_schemas_replay_response_horizons_and_transition_allowance`.
    """
    plan = _plan(14)
    plan["plan_digest"] = integral_plan_digest(plan)
    raw, records = _synthetic_complete_capture(schema_revision=14)
    target = next(
        index
        for index, (_bytes, value) in enumerate(records)
        if value["record_kind"] == "integral_rung" and int(value["rung_index"]) == 3
    )
    record_bytes, decoded = records[target]
    assert "tau_us" not in decoded
    # kind(1) + run_sequence(4) + evidence_sequence(2) + plan_digest(8)
    # + rung_index(1) + i_raw(2); the horizon follows immediately.
    header_len = 18
    horizon = {
        "tau_us": velocity_i_tau_us(
            int(decoded["i_raw"]), int(plan["pwm_hz"]), plan.get("build_revision")
        ),
        "moving_horizon_us": int(plan["moving_stroke_us"]),
        "horizon_class": 1,
    }
    records[target] = (
        record_bytes[:header_len]
        + struct.pack(
            "<QIB",
            horizon["tau_us"],
            horizon["moving_horizon_us"],
            horizon["horizon_class"],
        )
        + record_bytes[header_len:],
        {**decoded, **horizon},
    )
    terminal = records[-1][1]
    assert terminal["record_kind"] == "integral_terminal"
    terminal["digest"] = _rebuild_stage_c_digest(plan, records[:-1])

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == ["rung 3 reports a response horizon outside a combined workflow"]


def test_stabilized_stage_c_recovery_passes_with_event_150() -> None:
    """Under `Stabilized` the firmware emits the owner-5 selection, then Event
    150, then the recovery. Mirrors the immediate-scored test above under the
    other schedule; both must reach an empty mismatch list for the rule to be
    schedule-derived rather than hardcoded either way.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12)

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
    plan["plan_digest"] = integral_plan_digest(plan)
    raw, records = _synthetic_complete_capture(schema_revision=12)
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
    raw, records = _synthetic_complete_capture(schema_revision=14)
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
    raw, records = _synthetic_complete_capture(schema_revision=12)
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
    raw, records = _synthetic_complete_capture(schema_revision=14)
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
    raw, records = _synthetic_complete_capture(schema_revision=12)
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
    raw, records = _synthetic_complete_capture(schema_revision=14)
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
    raw, records = _synthetic_complete_capture(schema_revision=12)
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
    raw, records = _synthetic_complete_capture(schema_revision=14)
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
    raw, records = _synthetic_complete_capture(schema_revision=12)
    records = _fault_the_terminal(records)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []


def _fault_the_terminal_preserving_evidence(
    records: list[tuple[bytes, dict[str, object]]],
    **overrides: object,
) -> list[tuple[bytes, dict[str, object]]]:
    """Fault a terminal the way a dispatch-level promotion does.

    `apply_stage_c_terminal` (`commission_dispatch/stage_c.rs:273`, `:283`,
    `:338`, `:358`) and the disarm-failure paths in
    `commission_dispatch/mod.rs` (`:606`, `:1417`, `:1430`, `:1467`) overwrite
    only `outcome` and `cause` on an already-built terminal, and
    `evidence_terminal`'s assessment-error arm (`integral_sweep.rs:1211`)
    passes the materialized evidence through unchanged. All of them leave the
    run's evidence intact, unlike `_fault_the_terminal` above. `overrides`
    exists so the same shape can be falsified in one field at a time.
    """
    terminal_raw, terminal = records[-1]
    return [*records[:-1], (terminal_raw, {**terminal, "outcome": 3, "cause": 4, **overrides})]


def test_faulted_terminal_preserving_its_evidence_is_accepted() -> None:
    """The archived `foc213-transport-validated` capture is exactly this: a
    complete, fully sufficient schema-14 run whose terminal reports outcome 3
    and cause 4 while still carrying `eligible_masks=[4095, 4095]` and both
    anchor masks at 3, every value reproduced by the offline replay. Its
    dispatcher faulted an already-built terminal, so nothing reset the
    evidence.

    Demanding the reset shape from every fault terminal therefore rejected
    real firmware output. Two of the four fault paths reset and two preserve,
    and all of `evidence_terminal`'s integrity arm, the materialization arm,
    and `apply_stage_c_terminal` report cause 4, so the terminal cannot say
    which path built it.
    """
    raw, records = _synthetic_complete_capture(schema_revision=14)
    records = _fault_the_terminal_preserving_evidence(records)

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []


def test_faulted_terminal_preserving_wrong_eligible_masks_is_rejected() -> None:
    """Accepting the preserved shape must not become accepting anything a
    fault terminal says. Preserved evidence is still evidence: it has to agree
    with the replay, exactly as it does for a run that did not fault.
    """
    raw, records = _synthetic_complete_capture(schema_revision=14)
    records = _fault_the_terminal_preserving_evidence(records, eligible_masks=[4094, 4095])

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "terminal eligible masks: firmware=[4094, 4095] offline=[4095, 4095]"
    ]


def test_faulted_terminal_preserving_wrong_anchor_masks_is_rejected() -> None:
    """The anchor availability masks are folded from the same evidence struct
    (`replies.rs:1420-1438`), so they follow the same rule: a fault terminal
    that neither reset them nor matched the replay is reported.
    """
    raw, records = _synthetic_complete_capture(schema_revision=14)
    records = _fault_the_terminal_preserving_evidence(
        records,
        opening_available_mask=1,
        bookend_available_mask=2,
    )

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "terminal opening-anchor availability mask differs from replay",
        "terminal bookend availability mask differs from replay",
    ]


def _fault_the_terminal_after_reaching_target(
    records: list[tuple[bytes, dict[str, object]]],
    *,
    target_terminus: int,
    target_status: int,
    recovery_flags: int,
) -> list[tuple[bytes, dict[str, object]]]:
    """Fault a combined terminal that had already reached its 98-percent target.

    `terminal_from_data` (`integral_sweep.rs:973`) sets
    `rest_boundary_rung_plus_one` from `data.target_terminus`, which is *not*
    part of `IntegralRunEvidence`, so the target terminus survives the fault
    while `eligible_masks`/`current_terminus` are reset to empty by
    `IntegralRunEvidence::new(...)`. `terminal_target_status`
    (`integral_reporting.rs:51-76`) then reads those empty masks, so a
    reached-target fault reports `TargetReachedSparse` (1), not
    `TargetReached` (0), and `terminal_recovery_flags` encodes
    `TERMINAL_COMBINED_WORKFLOW | ((1 + 1) << 4)` == 0xA0.
    """
    terminal_raw, terminal = records[-1]
    terminal = {
        **terminal,
        "outcome": 3,
        "cause": 4,
        "eligible_masks": [0, 0],
        "opening_available_mask": 0,
        "bookend_available_mask": 0,
        "current_terminus": None,
        "target_terminus": target_terminus,
        "target_status": target_status,
        "recovery_flags": recovery_flags,
    }
    return [*records[:-1], (terminal_raw, terminal)]


def test_combined_faulted_terminal_reaching_its_target_reports_sparse_status() -> None:
    """A combined run that faults *after* reaching its target keeps
    `target_terminus` but loses its `eligible_masks`, so firmware's own
    `terminal_target_status` classifies it `TargetReachedSparse` (1) and
    encodes recovery flags 0xA0. The audit previously fed the *replayed*
    (still-sufficient, three-or-more eligible rungs per direction) masks into
    `combined_target_status`, expected `TargetReached` (0), and reported both
    the status and the encoded-flag mismatch against this honest terminal.

    The existing `test_combined_sufficient_but_faulted_terminal_is_accepted`
    cannot catch this: with `target_terminus=None` and a complete ladder the
    formula falls through to `TargetNotReachedAtCap` (3) under empty *and*
    replayed masks alike, so both readings agree by accident.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12)
    records = _fault_the_terminal_after_reaching_target(
        records,
        target_terminus=2,
        target_status=1,
        recovery_flags=0xA0,
    )

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == []


def test_combined_faulted_terminal_claiming_a_dense_target_is_rejected() -> None:
    """The target-status checks must keep their teeth under a fault rather
    than being switched off by it. Same reached-target fault as above, but
    the terminal claims `TargetReached` (0) with recovery flags 0x90 -- the
    reading the reset masks cannot support. Both the status field and its
    encoded copy must be rejected.
    """
    raw, records = _synthetic_complete_capture(schema_revision=12)
    records = _fault_the_terminal_after_reaching_target(
        records,
        target_terminus=2,
        target_status=0,
        recovery_flags=0x90,
    )

    audit = audit_velocity_integral(raw, records)

    assert audit["mismatches"] == [
        "combined target status differs from causal ladder evidence",
        "combined terminal target flags differ from canonical status",
    ]


# --------------------------------------------------------------------------
# Archived-capture regression corpus (spec R6).
#
# Tasks 5 and 6 corrected the offline replay grammar and the Stage-C audit,
# verified so far only against synthetic fixtures built above. The two
# archived captures below are the recordings that originally exposed the
# defect; this section points the corrected code at them directly. Both
# captures are large binaries excluded by `.gitignore`
# (`docs/artifacts/trace-captures/**/*.bin`), so every test here skips
# cleanly, never fails, when its capture is not present on disk.
# --------------------------------------------------------------------------

_ARCHIVE_ROOT = Path(__file__).resolve().parents[4] / "docs/artifacts/trace-captures"

_TRANSPORT_VALIDATED_CAPTURE = (
    _ARCHIVE_ROOT
    / "2026-08-03-foc-213-stage-c-transport"
    / "foc213-transport-validated-20260803-192710.bin"
)
_P18B_CAPTURE = (
    _ARCHIVE_ROOT
    / "2026-08-04-foc-213-recovery-gate-diagnostics"
    / "foc213-p18b-20260803-232554.bin"
)


def _complete_prefix_bytes(capture_path: Path) -> bytes:
    """Return `capture_path`'s bytes truncated to its last complete record.

    Every archived capture ends with one truncated record at the recorder's
    cut point, and `CaptureReader` raises when it reaches that partial
    header or payload. Read records through it until it raises, tracking the
    stream offset after each complete record, and keep only the bytes read
    up to that point. Verified against `foc213-p18b`: this trims it from
    1,910,389 to 1,910,357 bytes, and the trimmed bytes analyze cleanly.
    """
    data = capture_path.read_bytes()
    buffer = io.BytesIO(data)
    reader = CaptureReader(buffer)
    offset = buffer.tell()
    try:
        for _record in reader.records():
            offset = buffer.tell()
    except ValueError:
        pass
    return data[:offset]


def _analyze_complete_prefix(capture_path: Path) -> object:
    """Analyze `capture_path` through its last complete record.

    `analyze_capture` refuses a capture that ends mid-record, and every
    archived capture ends exactly that way at the recorder's cut point,
    so every archived-capture test goes through this helper rather than
    calling `analyze_capture` on the raw file.
    """
    return analyze_capture(io.BytesIO(_complete_prefix_bytes(capture_path)))


# The six Stage-C evidence record kinds `_velocity_integral_audits`
# (analyze.py) collects before handing them to `audit_velocity_integral`.
# `integral_plan_marker` and `integral_terminal` are deliberately excluded:
# Step 4 below needs the ordinary evidence stream only, a real terminal is
# never reachable from `foc213-p18b`'s truncated recording, and a
# synthesized sentinel stands in for it instead.
_STAGE_C_EVIDENCE_KINDS = frozenset(
    {
        "integral_observation",
        "rest_selection",
        "velocity_primitive_current",
        "integral_rung",
        "rung_origin_recovery",
        "recovery_rest",
    }
)


def _stage_c_evidence_records(capture_path: Path) -> list[dict[str, object]]:
    """Decode one capture's Stage-C evidence records in stream order.

    `audit_velocity_integral` cannot run on `foc213-p18b`: its recorder
    stopped two rungs in, so the plan fragments and terminal are incomplete
    and it degrades to `partial integral-response capture` without ever
    reaching the scored-rest grammar. This decodes the same evidence-record
    stream `analyze.py`'s `_velocity_integral_audits` collects, directly
    from the wire, so the grammar can be exercised without a complete plan
    or a real terminal.
    """
    records: list[dict[str, object]] = []
    prefix = _complete_prefix_bytes(capture_path)
    for record in CaptureReader(io.BytesIO(prefix)).records():
        if record.record_type != RECORD_TYPE_FIRMWARE_FRAME:
            continue
        frame = decode_frame(record.payload)
        if frame.header.frame_type != FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE:
            continue
        if frame.payload.get("record_kind") in _STAGE_C_EVIDENCE_KINDS:
            records.append(frame.payload)
    return records


# Messages the Task 5 (`6ff2342`) and Task 6 (`003e96c`, `b308462`, `76cbc34`)
# validator changes can emit, both the wordings they removed and the wordings
# they added. None of these may appear against a real capture -- their
# presence would mean the grammar or validator fix is incomplete, which is
# exactly what this corpus exists to catch (see
# `test_transport_validated_capture_carries_no_targeted_grammar_regression`).
_TARGETED_GRAMMAR_MESSAGES = frozenset(
    {
        # Task 5: `audit_scored_rest_grammar` ordering (velocity_measurement.py).
        "Stage C scored-rest selection has the wrong successor",  # old wording, removed
        "Stage C scored-rest selection has the wrong predecessor",  # new
        "Stage C rejected scored-rest selection lacks its causal terminal tail",  # new
        "Stage C observation lacks its causal scored-rest selection",
        "Stage C recovery rest lacks its causal selection",
        "Stage C scored-rest selection has the wrong identity",
        "Stage C recovery selection does not precede Event 150",
        "Stage C recovery selection does not precede its recovery",  # new
        "Stage C rejected recovery selection lacks its causal terminal",  # new
        "Stage C scored-rest selection has the wrong owner",
        # Task 6: combined-plan/terminal derivation (velocity_integral.py).
        "combined plan lacks one selected Stage-B response",  # old wording, removed
        "combined terminal lacks its workflow flag",  # old wording, removed
        "terminal combined-workflow flag differs from the plan",  # new
        "terminal combined tail differs from the plan",  # new
        "standalone sufficient run falsely reports the combined outcome",  # new
    }
)

# Known residuals: mismatches confirmed unrelated to the grammar
# and validator changes, present in the transport-validated capture
# for reasons independent of this fix (each was individually confirmed
# unrelated against firmware). Stable text is
# matched by exact equality; text carrying a rung index or a measured value
# is matched by an anchored regular expression that pins the whole message
# and constrains the variable part to digits or a signed number, per spec R6.
#
# Every entry here was audited against firmware once the archived capture was
# identified as a real schema-14 breakaway run. Four were misdiagnosed: they
# were recorded as gaps in the archived build when they were really the audit
# demanding output firmware never produces for that run. All four are fixed and
# gone:
#
# - `plan reconstruction failed: 'slot_transition_allowance_us'`. Firmware
#   emits no slot-transition allowance for a non-combined plan; the offline
#   digest was asking for a field that should never exist. Removed by the
#   schema/workflow fix, and that capture's plan now reconstructs cleanly.
# - `rung N response horizon tau_us/moving_horizon_us/horizon_class:
#   firmware=None ...`, three patterns matching 42 messages. `integral_sweep.rs`
#   computes the horizon only `if data.plan.combined_workflow` and `encode_rung`
#   omits all three fields when `moving_horizon_us == 0`, so a breakaway rung
#   carries none of them by design. Not a decode-completeness gap.
# - `terminal digest: firmware=0x... offline=0x...`. The offline evidence
#   digest seeded itself with a compact plan marker carrying a zero presence
#   byte, but `encode_velocity_integral_plan_marker`
#   (`integral_reporting.rs:454-471`) appends that byte only for a combined
#   workflow, so the offline seed hashed one byte firmware never hashed. Not a
#   property of the fault terminal it was filed under.
# - `fault terminal retained nonempty eligible masks` and
#   `fault terminal retained nonempty anchor availability`. The audit demanded
#   that every fault terminal report the reset evidence shape. Only two of the
#   four fault paths reset it; the dispatch-level promotions
#   (`commission_dispatch/stage_c.rs`, `commission_dispatch/mod.rs`) and
#   `evidence_terminal`'s assessment-error arm overwrite outcome and cause on
#   an already-built terminal and leave its evidence intact. This capture's
#   preserved evidence reproduces the offline replay exactly.
# - `scored-rest intervals_q mismatch: firmware=[[-61476, 61476], ...]
#   offline=[[0, 0], ...]`. Filed as needing hardware to close, on the reading
#   that firmware had scored a window this recording did not contain. The real
#   cause was on the host: `_attach_rest_selection_audit` raw-verified every
#   selection, including the recovery ones, whose scored samples firmware never
#   traces (`recovery/controller.rs` pushes them straight into the observation
#   ring; `record_velocity_trace_observation` is called only from the ordinary
#   observation paths). Each of the capture's 13 recovery selections was
#   therefore being compared against the *preceding ordinary stroke's* 256
#   samples. Twelve agreed only because both windows were all-zero; rung 5's
#   did not. The comparison was never valid for any of them, so it no longer
#   runs -- see `_RAW_UNVERIFIABLE_REST_SELECTION_OWNERS` in `analyze.py`.
#
# Empty in both halves, and kept rather than deleted: the exactly-one-match
# rule below still consults both, and a future residual belongs in one of them
# rather than in a loosened assertion.
_RESIDUAL_EXACT_MESSAGES: frozenset[str] = frozenset()
_RESIDUAL_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = ()


def _residual_match_count(message: str) -> int:
    """Count how many allowlist entries `message` matches.

    The corpus test requires exactly one: zero means an unaccounted-for
    regression, more than one means the predicates overlap and must be
    tightened.
    """
    count = 1 if message in _RESIDUAL_EXACT_MESSAGES else 0
    count += sum(1 for pattern in _RESIDUAL_MESSAGE_PATTERNS if pattern.fullmatch(message))
    return count


def test_transport_validated_capture_carries_no_targeted_grammar_regression() -> None:
    """`foc213-transport-validated` is the capture whose reply-transport fix
    this whole investigation chased, and it analyzes completely -- every
    scored-rest grammar check and the schema >= 8 combined-terminal
    derivation both run against it, unlike `foc213-p18b` below. Four
    properties, per spec R6:

    1. None of the strings the Task 5 grammar fix or the Task 6 validator
       fix can emit (old or new wording) appear.
    2. No sequence- or identity-related mismatch appears.
    3. Every other mismatch matches exactly one entry in the known-residual
       allowlist above -- not "at least one": an unmatched mismatch is a new
       regression, and a multiply-matched one means the predicates overlap.
       The allowlist is now empty, so this is the same as demanding an empty
       mismatch list; both are asserted, because an emptied allowlist must
       not be able to hide a mismatch that a future entry quietly re-covers.
    4. The scored-rest counts are pinned. An empty mismatch list proves
       nothing on its own -- the audit reaching it by checking nothing would
       look identical -- so the exact split is asserted: 112 selections
       raw-verified against their own traced samples, and 13 recovery
       selections reported as raw-unverifiable because firmware traces no
       samples for them at all. 112 + 13 is the capture's full complement of
       125 Event 151 records, so neither number can grow at the other's
       expense without failing here.
    """
    if not _TRANSPORT_VALIDATED_CAPTURE.exists():
        pytest.skip(f"archived capture not present: {_TRANSPORT_VALIDATED_CAPTURE}")

    analysis = _analyze_complete_prefix(_TRANSPORT_VALIDATED_CAPTURE)
    responses = analysis.velocity_integral_audits
    assert len(responses) == 1
    mismatches = responses[0]["mismatches"]

    targeted = [message for message in mismatches if message in _TARGETED_GRAMMAR_MESSAGES]
    assert targeted == []

    sequence_or_identity = [
        message for message in mismatches if "sequence" in message or "identity" in message
    ]
    assert sequence_or_identity == []

    match_counts = {message: _residual_match_count(message) for message in mismatches}
    unmatched = [message for message, count in match_counts.items() if count == 0]
    multiply_matched = [message for message, count in match_counts.items() if count > 1]
    assert unmatched == []
    assert multiply_matched == []

    assert mismatches == []
    assert responses[0]["rest_selection_verified_count"] == 112
    assert responses[0]["rest_selection_raw_unverifiable_count"] == 13


def test_p18b_decoded_prefix_satisfies_the_scored_rest_grammar() -> None:
    """`foc213-p18b`'s recorder stopped two rungs in, so its plan fragments
    and terminal are incomplete and `audit_velocity_integral` can only
    report `partial integral-response capture` -- the grammar checks never
    run. Asserting the absence of grammar mismatches through that full audit
    would therefore be vacuous. This calls the changed validator,
    `audit_scored_rest_grammar`, directly against the capture's decoded
    record prefix instead, which is `foc213-p18b`'s substantive
    `foci-trace` coverage.

    The record list is trimmed to end at the last `rung_origin_recovery`:
    the grammar checks each observation's and each selection's successor, so
    a trailing partial group would fault against the synthesized terminal
    below rather than against real firmware output, and a recovery record
    has no successor requirement, so ending there is clean.

    The terminal is a sentinel, `{"run_sequence": ..., "cause": 0}`, built
    from the run sequence the trimmed records themselves carry. That is
    sound because `audit_scored_rest_grammar` only ever reads two terminal
    fields: `run_sequence` for the scored-rest selection identity check, and
    `cause` to confirm a rejection's causal terminal -- and this trimmed
    prefix contains no rejected selection (`disposition == 2`), so `cause`
    is never actually consulted.
    """
    if not _P18B_CAPTURE.exists():
        pytest.skip(f"archived capture not present: {_P18B_CAPTURE}")

    records = _stage_c_evidence_records(_P18B_CAPTURE)
    last_recovery_index = max(
        index
        for index, record in enumerate(records)
        if record["record_kind"] == "rung_origin_recovery"
    )
    trimmed = records[: last_recovery_index + 1]

    run_sequence = int(trimmed[0]["run_sequence"])
    assert {int(record["run_sequence"]) for record in trimmed} == {run_sequence}
    plan_digest = next(int(record["plan_digest"]) for record in trimmed if "plan_digest" in record)
    terminal = {"run_sequence": run_sequence, "cause": 0}

    mismatches = audit_scored_rest_grammar(
        trimmed,
        terminal,
        schema_revision=14,
        schema_floor=12,
        expected_stage=1,
        ordinary_owners=(3, 4, 6),
        recovery_owners=(5,),
        observation_kind="integral_observation",
        plan_digest=plan_digest,
        label="Stage C",
        stabilized_recovery_rest=False,
        ordinary_rejection_tail=("velocity_primitive_current",),
    )

    assert mismatches == []

    # Supplementary check, independent of the grammar audit above: the
    # trimmed prefix's evidence_sequence numbers are contiguous, the same
    # property `audit_velocity_integral` checks separately from the grammar
    # for a complete run.
    sequences = [int(record["evidence_sequence"]) for record in trimmed]
    assert sequences == list(range(sequences[0], sequences[0] + len(sequences)))
