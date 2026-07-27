from copy import deepcopy

import pytest

from foci_trace.acceptance_matrix import (
    assemble_matrix_plan,
    compare_matrix_runs,
    matrix_plan_digest,
    reconstruct_amplitude,
    reconstruct_direction,
    robust_rate_location,
)


def _plan() -> dict[str, object]:
    return {
        "schema_revision": 1,
        "channel": 0,
        "build_revision": 0x1234_5678,
        "acceptance_schema_revision": 10,
        "acceptance_plan_digest": 0x1111_2222_3333_4444,
        "order": 1,
        "target_rpm": [16, 33, 66, 132, 263],
        "selected_p": 1_024,
        "selected_i": 1_024,
        "qualifier_bits": 0x0123,
        "pwm_hz": 25_000,
        "encoder_counts_per_rev": 4_000,
        "amplitude_count": 5,
        "expected_observations": 40,
        "family_size": 20,
        "moving_stroke_us": 512_000,
        "zero_settle_us": 500_000,
        "analysis_budget_us": 236_000,
        "nominal_workflow_ms": 60_541,
        "maximum_workflow_ms": 66_456,
        "recovery_origin_band_counts": 1_000,
        "recovery_nominal_slot_us": 1_816_958,
        "recovery_maximum_slot_us": 3_000_000,
        "recovery_slot_count": 5,
        "hard_torque_limit": 2_816,
        "usable_torque_limit": 2_816,
        "limit_evidence_capability": 1,
        "recovery_quantization_exposed": 0,
        "probe_constrained_test_point": 0,
        "requested_velocity_mrev_s": 4_394,
        "machine_velocity_ceiling_mrev_s": 10_000,
        "requested_velocity_source": 0,
        "max_stroke_travel_mrev": 2_500,
        "settle_travel_reserve_mrev": 250,
        "negative_position_headroom_mrev": 7_025,
        "positive_position_headroom_mrev": 7_025,
        "max_duration_ms": 3_000,
    }


def _observations(
    intervals: list[tuple[int, int]],
    means: list[int],
    *,
    direction: int = 0,
    classes: list[int] | None = None,
) -> list[dict[str, object]]:
    classes = classes or [0, 0, 0, 0]
    return [
        {
            "slot": index * 2 + direction,
            "direction": direction,
            "classification": classes[index],
            "rate_low": interval[0],
            "rate_mean": means[index],
            "rate_high": interval[1],
            "selected_level": index + 1,
        }
        for index, interval in enumerate(intervals)
    ]


def test_robust_location_matches_signed_firmware_midpoint() -> None:
    assert robust_rate_location([9, 1, 5, None]) == 5
    assert robust_rate_location([4, 1, 8, 3]) == 3
    assert robust_rate_location([-1, -4, -8, 3]) == -3
    assert robust_rate_location([1, 2, None, None]) is None


def test_plan_digest_is_sensitive_to_order_and_exact_targets() -> None:
    plan = _plan()

    assert matrix_plan_digest(plan) == 0x2618_AA16_E7C5_77C3

    descending = deepcopy(plan)
    descending["order"] = 2
    descending["target_rpm"] = list(reversed(plan["target_rpm"]))
    assert matrix_plan_digest(descending) != matrix_plan_digest(plan)


def test_plan_fragments_require_one_shared_identity() -> None:
    plan = _plan()
    fragments = [
        {
            "fragment": fragment,
            "run_sequence": 7,
            "plan_digest": 9,
            **(
                {
                    key: value
                    for key, value in plan.items()
                    if key
                    not in {
                        "moving_stroke_us",
                        "zero_settle_us",
                        "analysis_budget_us",
                        "nominal_workflow_ms",
                        "maximum_workflow_ms",
                    }
                }
                if fragment == 0
                else {}
            ),
        }
        for fragment in range(3)
    ]
    fragments[1].update(
        {
            "moving_stroke_us": 512_000,
            "zero_settle_us": 500_000,
            "analysis_budget_us": 236_000,
            "nominal_workflow_ms": 60_541,
            "maximum_workflow_ms": 66_456,
        }
    )

    assert assemble_matrix_plan(fragments)["run_sequence"] == 7

    fragments[2]["plan_digest"] = 10
    with pytest.raises(ValueError, match="disagree"):
        assemble_matrix_plan(fragments)


def test_direction_reuses_depth_three_closed_interval_consensus() -> None:
    observations = _observations(
        [(0, 10), (2, 9), (3, 8), (4, 7)],
        [5, 6, 7, 8],
    )

    evidence = reconstruct_direction(
        observations,
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )

    assert evidence["availability"] == 0
    assert evidence["components"] == [[3, 8]]
    assert evidence["included_mask"] == 0b1111
    assert evidence["robust_location_q"] == 6
    assert evidence["deficit_interval_q"] == [12, 17]
    assert evidence["blocking_levels"] == [1, 2, 3, 4]


def test_direction_separates_precision_floor_current_and_fault() -> None:
    intervals = [(0, 1)] * 4
    means = [0] * 4

    floor = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 4, 4]),
        signed_target_rate_q=20,
        precision_floor_candidate=True,
    )
    current = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 0, 3]),
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )
    fault = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 0, 10]),
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )

    assert [floor["availability"], current["availability"], fault["availability"]] == [1, 4, 5]


def test_amplitude_preserves_reverse_sign_and_frozen_target() -> None:
    plan = _plan()
    forward = _observations([(900, 1_000)] * 4, [950] * 4)
    reverse = _observations([(-1_000, -900)] * 4, [-950] * 4, direction=1)
    report = {"amplitude_index": 0, "target_rpm": 16}

    amplitude = reconstruct_amplitude(forward + reverse, report, plan)

    assert amplitude["directions"][0]["signed_target_rate_q"] == 1_066_666
    assert amplitude["directions"][1]["signed_target_rate_q"] == -1_066_666
    assert amplitude["directions"][1]["response_interval_q"] == [-1_000, -900]
    assert amplitude["directions"][1]["deficit_interval_q"] == [-1_065_766, -1_065_666]


def _audit(interval: list[int], robust: int, *, attempted: int = 0b11111) -> dict[str, object]:
    direction = {
        "availability": 0,
        "response_interval_q": interval,
        "robust_location_q": robust,
        "signed_target_rate_q": 1_000,
    }
    return {
        "amplitudes": [
            {
                "amplitude_index": index,
                "directions": [deepcopy(direction), deepcopy(direction)],
            }
            for index in range(5)
        ],
        "terminal": {"attempted_masks": [attempted, attempted]},
    }


def test_cross_run_comparison_distinguishes_absence_and_closed_touch() -> None:
    previous = _audit([0, 5], 3)
    current = _audit([5, 10], 7)
    current["terminal"]["attempted_masks"][0] = 0b11110
    current["amplitudes"][1]["directions"][1]["availability"] = 3

    comparison = compare_matrix_runs(previous, current)
    by_key = {
        (item["amplitude_index"], item["direction"]): item for item in comparison["comparisons"]
    }

    assert by_key[(0, 0)]["kind"] == "AttemptCoverageMismatch"
    assert by_key[(1, 1)]["kind"] == "EligibilityMismatch"
    assert by_key[(2, 0)]["kind"] == "PairedPrecision"
    assert by_key[(2, 0)]["overlap"] is True
    assert by_key[(2, 0)]["signed_gap_q"] == 0
    assert comparison["maximum_observed_displacement_q"] == 4
    assert comparison["maximum_displacement_kind"] == "observed_bound"


def test_cross_run_signed_gap_preserves_movement_direction() -> None:
    forward = compare_matrix_runs(_audit([0, 4], 2), _audit([6, 10], 8))
    reverse = compare_matrix_runs(_audit([6, 10], 8), _audit([0, 4], 2))

    assert forward["comparisons"][0]["signed_gap_q"] == 2
    assert reverse["comparisons"][0]["signed_gap_q"] == -2
