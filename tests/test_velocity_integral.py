import struct

import pytest

from foci_trace.velocity_integral import (
    audit_integral_reproduction_pair,
    audit_velocity_integral,
    classify_observation,
    compare_integral_runs,
    delta_e,
    depth_three_consensus,
    derive_integral_ladder,
    integral_plan_digest,
    reconstruct_observation,
    stage_c_slot_transition,
    velocity_i_tau_us,
)
from foci_trace.velocity_measurement import VelocityObservation


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


def _plan() -> dict[str, object]:
    return {
        "build_revision": 3,
        "schema_revision": 1,
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


def test_integral_plan_digest_hashes_every_immutable_input_in_firmware_order() -> None:
    assert integral_plan_digest(_plan()) == 0x2DD99123497A4876


def _plan_marker(plan: dict[str, object], sequence: int) -> tuple[bytes, dict[str, object]]:
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
    core_names = (
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
    )
    schedule_names = (
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
    )
    return [
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
) -> tuple[bytes, dict[str, object]]:
    raw = bytearray((3,))
    raw += struct.pack("<IHQBH", 7, sequence, plan_digest, rung_index, i_raw)
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
        "directions": directions,
    }


def _synthetic_complete_capture() -> tuple[
    list[VelocityObservation], list[tuple[bytes, dict[str, object]]]
]:
    plan = _plan()
    plan["plan_digest"] = integral_plan_digest(plan)
    records = _plan_records(plan)
    sequence = 1
    records.append(_plan_marker(plan, sequence))
    sequence += 1
    raw_observations: list[VelocityObservation] = []
    gains = [0, 5, 10, 20, 40, 80, 0]
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
        directions = []
        for direction, observations in enumerate(direction_results):
            interval = tuple(observations[0]["deficit_interval_q"])
            directions.append(
                {
                    "direction": direction,
                    "kind": 0 if rung_index == 0 else 2 if rung_index == 6 else 1,
                    "classification": 0,
                    "component_count": 1,
                    "components": [interval],
                    "usable_mask": 0b1111,
                    "included_mask": 0b1111,
                    "observation_classes": [0, 0, 0, 0],
                }
            )
        records.append(
            _rung_record(rung_index, gain, sequence, int(plan["plan_digest"]), directions)
        )
        sequence += 1
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
                "outcome": 0,
                "cause": 0,
                "expected_observations": 56,
                "emitted_observations": 56,
                "expected_rungs": 7,
                "emitted_rungs": 7,
                "eligible_masks": [0b11111, 0b11111],
                "current_terminus": None,
                "run_started_us": 1,
                "run_completed_us": 2,
            },
        )
    )
    return raw_observations, records


def test_complete_capture_reconstructs_curve_counts_and_digest() -> None:
    raw, records = _synthetic_complete_capture()

    audit = audit_velocity_integral(raw, records)

    assert audit["matches_firmware"] is True
    assert audit["complete"] is True
    assert audit["run_evidence"]["eligible_masks"] == [0b11111, 0b11111]


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


def test_reproduction_pair_checks_anchor_relative_curve_and_terminal() -> None:
    raw, records = _synthetic_complete_capture()
    previous = audit_velocity_integral(raw, records)
    current = audit_velocity_integral(raw, records)
    current["terminal"]["outcome"] = 1

    pair = audit_integral_reproduction_pair(previous, current)

    assert pair["matches_firmware"] is True
    assert pair["comparison"]["reproduced_masks"] == [0b11111, 0b11111]


def test_tau_geometry_reconstructs_the_fixed_whole_octave_ladder() -> None:
    assert velocity_i_tau_us(5, 25_000) == 524_288
    assert velocity_i_tau_us(80, 25_000) == 32_768

    ladder = derive_integral_ladder(25_000, 512_000, 64_000, 32_767)

    assert ladder == {"i_start": 5, "i_top_cap": 81, "positive_i": [5, 10, 20, 40, 80]}


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
