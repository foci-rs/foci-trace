import json
from pathlib import Path

import pytest

from foci_trace.velocity_measurement import (
    AccuracyDecision,
    VelocityObservation,
    analyze_velocity_observations,
    analyze_stationary_tail,
    audit_rung_origin_recovery,
    classify_absolute_rate,
    plan_rung_origin_recovery,
    project_recovery_authority,
    recovery_return_distance_counts,
    recovery_runtime_us,
    recovery_target_rpm,
    recovery_wrong_way,
    select_recovery_authority,
    velocity_report_to_json,
)


FIXTURE = Path(__file__).parent / "fixtures" / "velocity_measurement_golden.json"


def _observation(
    delta_counts: int,
    *,
    elapsed_us: int = 1_000,
    direction: int = 1,
    stroke_index: int = 0,
    target_cpms: int = 1,
    peak_torque_target_abs: int | None = None,
    pid_velocity_actual: int | None = None,
    actual_velocity_pptm: int | None = None,
) -> VelocityObservation:
    return VelocityObservation(
        delta_counts=delta_counts,
        elapsed_us=elapsed_us,
        direction=direction,
        stroke_index=stroke_index,
        target_cpms=target_cpms,
        peak_torque_target_abs=peak_torque_target_abs,
        pid_velocity_actual=pid_velocity_actual,
        actual_velocity_pptm=actual_velocity_pptm,
    )


def test_recovery_geometry_and_deadlines_match_firmware_integer_arithmetic() -> None:
    plan = plan_rung_origin_recovery(
        max_stroke_travel_mrev=2_500,
        settle_travel_reserve_mrev=250,
        encoder_counts_per_rev=4_000,
        planned_velocity_mrev_s=4_394,
        primitive_cap_us=3_000_000,
    )

    assert plan == {
        "origin_band_counts": 1_000,
        "nominal_slot_us": 1_816_958,
        "maximum_slot_us": 3_000_000,
    }
    assert recovery_return_distance_counts(8_458, 1_000) == 7_458
    with pytest.raises(ValueError, match="primitive cap"):
        recovery_runtime_us(7_458, 3_150_000, 3_000_000)
    assert recovery_runtime_us(7_458, 6_300_000, 3_000_000) == 2_431_810


def test_recovery_target_and_wrong_way_are_sign_symmetric() -> None:
    assert recovery_target_rpm(8_458, 264) == -264
    assert recovery_target_rpm(-8_458, 264) == 264
    assert recovery_wrong_way(8_458, 8_459) is False
    assert recovery_wrong_way(8_458, 8_460) is True
    assert recovery_wrong_way(-8_458, -8_460) is True


@pytest.mark.parametrize(
    ("direction", "ordinary", "structured", "source", "measured", "stored"),
    (
        (1, [[10_000_000, 10_200_000]], [], 1, 10_000_000, 9_999_872),
        (1, [], [[9_174_700, 9_400_298]], 5, 9_174_700, 9_174_528),
        (1, [[9_000_000, 9_200_000]], [[9_174_700, 9_400_298]], 1, 9_000_000, 8_999_936),
        (1, [[10_000_000, 10_200_000]], [[9_174_700, 9_400_298]], 5, 9_174_700, 9_174_528),
        (1, [[9_174_700, 9_400_298]], [[9_174_700, 9_400_298]], 1, 9_174_700, 9_174_528),
        (1, [[-10, 20]], [[-20, 0]], 2, 0, 0),
        (-1, [], [[-9_500_000, -9_100_000]], 5, 9_100_000, 9_099_776),
    ),
)
def test_recovery_authority_selector(
    direction: int,
    ordinary: list[list[int]],
    structured: list[list[int]],
    source: int,
    measured: int,
    stored: int,
) -> None:
    selected = select_recovery_authority(direction, ordinary, structured)

    assert selected == {
        "measured_lower_rate_q": measured,
        "lower_rate_q": stored,
        "binding_source": source,
    }


def test_schema_four_recovery_rule_is_preserved() -> None:
    record = {
        "start_offset_counts": -1_834,
        "origin_band_counts": 1_000,
        "lower_rate_q": 0,
        "binding_source": 2,
        "outcome": 2,
    }
    projected = project_recovery_authority(
        record,
        direction=1,
        ordinary_rate_intervals=[],
        structured_rate_intervals=[[9_174_700, 9_400_298]],
        primitive_cap_us=3_000_000,
    )

    assert record["lower_rate_q"] == 0
    assert record["binding_source"] == 2
    assert record["outcome"] == 2
    assert projected["interpretation"] == "what_if_projection"


def test_p512_schema_five_projection() -> None:
    projected = project_recovery_authority(
        {"start_offset_counts": -1_834, "origin_band_counts": 1_000},
        direction=1,
        ordinary_rate_intervals=[],
        structured_rate_intervals=[[9_174_700, 9_400_298]],
        primitive_cap_us=3_000_000,
    )

    assert projected["measured_lower_rate_q"] == 9_174_700
    assert projected["lower_rate_q"] == 9_174_528
    assert projected["binding_source"] == 5
    assert projected["return_distance_counts"] == 834
    assert projected["deadline_us"] == 1_338_904


@pytest.mark.parametrize(
    ("changes", "expected_mismatch"),
    (
        ({}, None),
        ({"binding_source": 5}, None),
        (
            {
                "start_offset_counts": 900,
                "closest_offset_counts": 900,
                "final_offset_counts": 900,
                "lower_rate_q": 0,
                "moving_duration_us": 0,
                "settle_duration_us": 0,
                "total_duration_us": 0,
                "binding_source": 0,
                "outcome": 0,
            },
            None,
        ),
        (
            {
                "closest_offset_counts": 8_458,
                "final_offset_counts": 8_458,
                "lower_rate_q": 0,
                "moving_duration_us": 0,
                "settle_duration_us": 0,
                "total_duration_us": 0,
                "binding_source": 2,
                "outcome": 2,
            },
            None,
        ),
        (
            {
                "closest_offset_counts": 8_458,
                "final_offset_counts": 8_458,
                "lower_rate_q": 3_150_000,
                "moving_duration_us": 0,
                "settle_duration_us": 0,
                "total_duration_us": 0,
                "binding_source": 3,
                "outcome": 3,
            },
            None,
        ),
        ({"final_offset_counts": 1_001, "outcome": 1}, "outside origin band"),
        ({"closest_offset_counts": 8_460}, "wrong-way growth"),
        ({"planned_velocity_mrev_s": 4_393}, "planned velocity"),
    ),
)
def test_recovery_audit_classifies_success_containment_and_fault_paths(
    changes: dict[str, int], expected_mismatch: str | None
) -> None:
    record = {
        "stage": 0,
        "rung_index": 4,
        "velocity_p": 1_448,
        "planned_velocity_mrev_s": 4_394,
        "start_offset_counts": 8_458,
        "closest_offset_counts": 900,
        "final_offset_counts": 734,
        "origin_band_counts": 1_000,
        "lower_rate_q": 6_300_000,
        "moving_duration_us": 1_200_000,
        "settle_duration_us": 736_000,
        "total_duration_us": 1_936_000,
        "peak_torque_target_abs": 2_533,
        "binding_source": 1,
        "outcome": 1,
    }
    record.update(changes)
    plan = {
        "planned_velocity_mrev_s": 4_394,
        "recovery_origin_band_counts": 1_000,
        "recovery_maximum_slot_us": 3_000_000,
        "usable_torque_limit": 2_534,
    }

    audit = audit_rung_origin_recovery(record, plan)

    if expected_mismatch is None:
        assert audit["matches_firmware"] is True
    else:
        assert audit["matches_firmware"] is False
        assert any(expected_mismatch in item for item in audit["mismatches"])


def test_golden_vector_records_every_level_moment() -> None:
    golden = json.loads(FIXTURE.read_text())
    observations = [VelocityObservation(**item) for item in golden["observations"]]

    report = analyze_velocity_observations(observations, max_levels=4)
    actual = velocity_report_to_json(report)

    assert actual == golden["report"]


def test_unequal_intervals_use_total_displacement_over_total_time() -> None:
    report = analyze_velocity_observations(
        [_observation(1, elapsed_us=900), _observation(1, elapsed_us=1_100)],
        max_levels=3,
    )

    level_one = report.directions[1].levels[1]
    arithmetic_average_q = (1_111_111 + 909_090) // 2
    assert level_one.mean_rate_q == 1_000_000
    assert level_one.mean_rate_q != arithmetic_average_q


def test_signed_directions_and_stroke_partials_are_never_pooled() -> None:
    observations = []
    for direction, target in ((1, 1), (-1, -1)):
        for stroke_index in (0, 1):
            observations.extend(
                _observation(
                    direction * delta,
                    direction=direction,
                    stroke_index=stroke_index,
                    target_cpms=target,
                )
                for delta in (1, 0, 1)
            )

    report = analyze_velocity_observations(observations, max_levels=4)

    assert set(report.directions) == {-1, 1}
    for direction in (-1, 1):
        assert report.directions[direction].levels[0].block_count == 6
        assert report.directions[direction].levels[1].block_count == 2
        assert report.directions[direction].levels[2].block_count == 0


def test_mixed_targets_in_one_direction_are_rejected() -> None:
    with pytest.raises(ValueError, match="mixed targets"):
        analyze_velocity_observations(
            [_observation(1, target_cpms=1), _observation(2, target_cpms=2)],
            max_levels=4,
        )


def test_quantized_sequence_reaches_a_supported_plateau() -> None:
    observations = [_observation(delta) for _ in range(128) for delta in (1, 0, 1, 1, 0)]

    report = analyze_velocity_observations(observations, max_levels=8)
    direction = report.directions[1]

    assert direction.selected_level is not None
    assert direction.selected_level >= 2
    assert direction.selected_blocks >= 16


@pytest.mark.parametrize(
    ("sample_count", "expected_delta"),
    [(256, 154), (512, 153)],
)
def test_stationary_tail_supports_full_and_tail_ring_captures(
    sample_count: int, expected_delta: int
) -> None:
    golden = json.loads(FIXTURE.read_text())["stationary_tail"]
    observations = [
        _observation(golden["pattern"][index % len(golden["pattern"])])
        for index in range(sample_count)
    ]

    result = analyze_stationary_tail(observations)

    assert result["usable"] == 1
    assert result["suffix_len"] == 256
    assert result["selected_blocks"] == 16
    assert result["selected_delta_counts"] == expected_delta
    if sample_count == golden["sample_count"]:
        assert result["selected_level"] == golden["selected_level"]
        assert result["settled_rate_mean_q"] == golden["settled_rate_mean_q"]


def test_stationary_tail_matches_firmware_single_precision_rounding() -> None:
    live_deltas = {
        0: (-1, 1_054),
        3: (-1, 1_000),
        9: (-1, 1_000),
        52: (-1, 1_000),
        53: (1, 1_000),
        55: (-1, 1_000),
    }
    observations = [
        _observation(delta, elapsed_us=elapsed)
        for index in range(256)
        for delta, elapsed in [live_deltas.get(index, (0, 1_000))]
    ]

    result = analyze_stationary_tail(observations)

    assert result["residual_magnitude_q2"] == 1_663_058_048


def test_excluded_observation_zeros_unavailable_suffix_metrics() -> None:
    from foci_trace.velocity_measurement import _reconstruct_observation

    observations = [_observation(index) for index in range(256)]
    record = {"velocity_p": 32, "target_velocity_rpm": 8, "slot": 0}
    plan = {"hard_torque_limit": 2_816, "usable_torque_limit": 2_534}
    terminal = {"encoder_counts_per_rev": 4_000, "f_pwm_hz": 25_000}

    result = _reconstruct_observation(observations, record, plan, terminal)

    assert result["classification"] == 6
    assert result["tested_suffixes"] == 0


def test_sweep_confidence_width_matches_firmware_single_precision(monkeypatch) -> None:
    import foci_trace.velocity_measurement as measurement

    monkeypatch.setattr(
        measurement,
        "_stationary_suffix",
        lambda _raw: {
            "usable": 1,
            "selected_delta_counts": 0,
            "selected_elapsed_us": 256_000,
            "selected_level": 3,
            "selected_blocks": 16,
            "variance_of_mean_q2": 1_105_940_631,
            "settled_rate_mean_q": -124_974,
            "suffix_len": 256,
            "slope_q": 0,
            "slope_half_width_q": 1,
            "lag_one_q": 0,
            "lag_one_half_width_q": 1,
            "residual_magnitude_q2": 1,
            "tested_suffixes": 3,
        },
    )
    observations = [_observation(0)]
    record = {"velocity_p": 1, "target_velocity_rpm": -176, "slot": 1}
    plan = {
        "rung_count": 23,
        "observations_per_direction": 2,
        "hard_torque_limit": 2_816,
        "usable_torque_limit": 2_534,
    }
    terminal = {"encoder_counts_per_rev": 4_000, "f_pwm_hz": 25_000}

    result = measurement._reconstruct_observation(observations, record, plan, terminal)

    assert result["rate_low"] == -259_051
    assert result["rate_high"] == 9_103


def test_current_headroom_terminal_accepts_expected_torque_boundary() -> None:
    import foci_trace.velocity_measurement as measurement

    assert hasattr(measurement, "_torque_evidence_matches_terminal")
    matches = measurement._torque_evidence_matches_terminal(
        {"cause": 4},
        {"below_usable_limit": False, "within_observed_peak_plus_one": True},
    )

    assert matches is True


def test_disturbance_promotes_encoder_rate_to_target_q16_domain() -> None:
    from foci_trace.velocity_measurement import _disturbance

    result = _disturbance(
        p=256,
        target_velocity_rpm=176,
        delta_counts=44,
        elapsed_us=3_750,
        half_width_rate_q=0,
        variance_rate_q2=0,
        counts_per_rev=4_000,
    )

    assert result == (0, 0, 0, 0)


def test_disturbance_is_symmetric_and_rejects_retired_target_domain() -> None:
    from foci_trace.velocity_measurement import _disturbance

    forward = _disturbance(
        p=256,
        target_velocity_rpm=176,
        delta_counts=44,
        elapsed_us=3_750,
        half_width_rate_q=0,
        variance_rate_q2=0,
        counts_per_rev=4_000,
    )
    reverse = _disturbance(
        p=256,
        target_velocity_rpm=-176,
        delta_counts=-44,
        elapsed_us=3_750,
        half_width_rate_q=0,
        variance_rate_q2=0,
        counts_per_rev=4_000,
    )
    mismatch = _disturbance(
        p=256,
        target_velocity_rpm=8,
        delta_counts=44,
        elapsed_us=3_750,
        half_width_rate_q=0,
        variance_rate_q2=0,
        counts_per_rev=4_000,
    )

    assert forward == reverse == (0, 0, 0, 0)
    assert mismatch[0] != 0


def test_planned_mrev_per_second_rounds_to_mechanical_rpm() -> None:
    from foci_trace.velocity_measurement import _mrev_s_to_rpm_nearest

    assert _mrev_s_to_rpm_nearest(2_929) == 176


def test_diagnostic_audit_compares_block_means_and_torque_reconstruction() -> None:
    from foci_trace.velocity_measurement import _diagnostic_audit

    observations = []
    records = {}
    for slot in range(4):
        direction = 1 if slot % 2 == 0 else -1
        records[(0, slot)] = {
            "velocity_p": 384,
            "target_velocity_rpm": direction * 176,
        }
        observations.extend(
            [
                VelocityObservation(
                    delta_counts=direction * 44,
                    elapsed_us=3_750,
                    direction=direction,
                    stroke_index=slot,
                    target_cpms=direction,
                    peak_torque_target_abs=264,
                    pid_velocity_actual=direction * 176,
                ),
                VelocityObservation(
                    delta_counts=direction * 44,
                    elapsed_us=3_750,
                    direction=direction,
                    stroke_index=slot,
                    target_cpms=direction,
                    peak_torque_target_abs=264,
                    actual_velocity_pptm=direction * 176,
                ),
            ]
        )

    audit = _diagnostic_audit(observations, records, 4_000, 2_534)

    assert audit["pid_encoder"]["block_count"] == 4
    assert audit["pid_encoder"]["difference_mean_rpm"] == pytest.approx(0.0)
    assert audit["pid_encoder"]["difference_low_rpm"] <= 0
    assert audit["pid_encoder"]["difference_high_rpm"] >= 0
    assert audit["pptm_encoder"]["difference_mean_rpm"] == pytest.approx(0.0)
    assert audit["torque"]["max_reconstructed_abs"] == 0
    assert audit["torque"]["within_observed_peak_plus_one"] is True
    assert audit["torque"]["below_usable_limit"] is True
    assert audit["complete"] is True


def test_diagnostic_audit_fails_closed_without_samples() -> None:
    from foci_trace.velocity_measurement import _diagnostic_audit

    audit = _diagnostic_audit([], {}, 4_000, 2_534)

    assert audit["complete"] is False
    assert audit["torque"]["sample_count"] == 0
    assert audit["torque"]["within_observed_peak_plus_one"] is False
    assert audit["torque"]["below_usable_limit"] is False


def test_sweep_confidence_supports_the_full_four_k_family() -> None:
    from foci_trace.velocity_measurement import _confidence_multiplier

    anchors = {64: 4.194683, 92: 4.374985, 128: 4.540345}
    for family_size, expected in anchors.items():
        assert _confidence_multiplier(family_size) == pytest.approx(expected)
    values = [_confidence_multiplier(index) for index in range(1, 129)]
    assert values == sorted(values)
    with pytest.raises(ValueError, match="outside"):
        _confidence_multiplier(0)
    with pytest.raises(ValueError, match="outside"):
        _confidence_multiplier(129)


def test_depth_three_consensus_matches_closed_interval_fixtures() -> None:
    from foci_trace.velocity_measurement import _depth_three_consensus

    assert _depth_three_consensus([(0, 10), (2, 9), (3, 8), (4, 7)]).components == ((3, 8),)
    assert _depth_three_consensus([(0, 10), (2, 9), (3, 8), (20, 30)]).included_mask == 0b0111
    assert _depth_three_consensus([(0, 4), (0, 10), (0, 10), (6, 10)]).classification == 1
    assert _depth_three_consensus([(0, 5), (5, 10), (0, 5), (5, 10)]).components == ((5, 5),)
    assert (
        _depth_three_consensus(
            [(0, 10), (2, 9), (3, 8), None], collected_mask=0b0111
        ).classification
        == 3
    )


def test_consensus_rate_inversion_is_outward_and_sign_correct() -> None:
    from foci_trace.velocity_measurement import (
        _disturbance_point_cell,
        _rate_bounds_from_disturbance_cell,
    )

    assert _rate_bounds_from_disturbance_cell(176, 4_000, 256, (0, 0)) == (
        11_733_333,
        11_733_334,
    )
    assert _rate_bounds_from_disturbance_cell(176, 4_000, 256, (65_536, 65_536)) == (
        11_666_666,
        11_666_667,
    )
    assert _rate_bounds_from_disturbance_cell(-176, 4_000, 256, (-65_536, -65_536)) == (
        -11_666_667,
        -11_666_666,
    )
    assert _disturbance_point_cell(8) == (8, 9)
    assert _disturbance_point_cell(-8) == (-9, -8)
    assert _disturbance_point_cell(0) == (-1, 1)


def test_four_observation_direction_pools_only_unique_consensus_members() -> None:
    from foci_trace.velocity_measurement import _analyze_four_observation_direction

    observations = [
        {
            "classification": 0,
            "target_velocity_rpm": 176,
            "disturbance_q16": point,
            "disturbance_low_q16": low,
            "disturbance_high_q16": high,
            "disturbance_variance_q2": 4,
        }
        for point, low, high in ((4, 0, 10), (5, 2, 9), (6, 3, 8), (25, 20, 30))
    ]

    result = _analyze_four_observation_direction(
        observations,
        direction=1,
        p=256,
        counts_per_rev=4_000,
        initial_lead_in=True,
    )

    assert result["class"] == 0
    assert result["components"] == [[3, 8]]
    assert result["included_mask"] == 0b0111
    assert result["operable_count"] == 4
    assert result["directional_flags"] == 0b11
    assert result["pooled_q16"] == 5
    assert result["pooled_interval"] == [1, 9]
    assert len(result["wire_rate_bounds_q"]) == 4


def test_four_observation_direction_preserves_partial_slot_mask() -> None:
    from foci_trace.velocity_measurement import _analyze_four_observation_direction

    observation = {
        "classification": 3,
        "target_velocity_rpm": 176,
        "disturbance_q16": 0,
        "disturbance_low_q16": 0,
        "disturbance_high_q16": 0,
        "disturbance_variance_q2": 0,
    }

    result = _analyze_four_observation_direction(
        [observation, None, None, None],
        direction=1,
        p=2_457,
        counts_per_rev=4_000,
        initial_lead_in=False,
    )

    assert result["class"] == 3
    assert result["collected_mask"] == 0b0001
    assert result["eligible_mask"] == 0
    assert result["included_mask"] == 0
    assert result["operable_count"] == 0


def test_correlated_single_comparison_does_not_form_a_plateau() -> None:
    observations = []
    for sample_index in range(256):
        alternating = 10 if sample_index % 2 == 0 else -10
        slow_square = 10 if (sample_index // 8) % 2 == 0 else -10
        observations.append(_observation(alternating + slow_square))

    direction = analyze_velocity_observations(observations, max_levels=4).directions[1]

    assert direction.levels[0].plateau_with_next is True
    assert direction.levels[1].plateau_with_next is False
    assert direction.selected_level is None


@pytest.mark.parametrize(
    ("delta_counts", "elapsed_us", "half_width_q", "expected"),
    [
        (1, 1_000, 0, AccuracyDecision.PASS),
        (1, 1_000, 1, AccuracyDecision.COLLECT_MORE),
        (2, 1_000, 0, AccuracyDecision.CONFIDENT_FAIL),
    ],
)
def test_exact_three_way_accuracy_boundary(
    delta_counts: int,
    elapsed_us: int,
    half_width_q: int,
    expected: AccuracyDecision,
) -> None:
    assert (
        classify_absolute_rate(
            settled_delta_counts=delta_counts,
            settled_elapsed_us=elapsed_us,
            target_counts_per_ms=0,
            half_width_rate_q=half_width_q,
            counts_per_rev=1_000,
            limit_mrev_per_s=1_000,
        )
        is expected
    )


def test_interval_distribution_and_discontinuities_are_deterministic() -> None:
    observations = [
        _observation(1, elapsed_us=999),
        _observation(1, elapsed_us=1_000),
        _observation(1, elapsed_us=1_001),
        _observation(1, elapsed_us=3_000),
    ]

    report = analyze_velocity_observations(observations, max_levels=3)

    assert report.min_interval_us == 999
    assert report.max_interval_us == 3_000
    assert report.median_interval_us == 1_000.5
    assert report.interval_deviations_us == [1.5, 0.5, 0.5, 1_999.5]
    assert report.timing_discontinuity_count == 1


def test_sweep_audit_reconstructs_observations_rung_band_and_terminal() -> None:
    from foci_trace.velocity_measurement import (
        FNV1A64_OFFSET,
        FNV1A64_PRIME,
        _finish_band,
        _new_band,
        _observe_band,
        _reconstruct_observation,
        _rung_class,
        audit_velocity_sweep,
    )

    plan = {
        "record_kind": "plan",
        "run_sequence": 9,
        "evidence_sequence": 0,
        "rung_count": 1,
        "observations_per_direction": 2,
        "planned_velocity_mrev_s": 1_667,
        "target_velocity_rpm": 100,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
    }
    terminal = {
        "record_kind": "terminal",
        "run_sequence": 9,
        "encoder_counts_per_rev": 1_000,
        "f_pwm_hz": 25_000,
        "expected_observations": 4,
        "emitted_observations": 4,
        "expected_rungs": 1,
        "emitted_rungs": 1,
        "outcome": 1,
        "cause": 2,
        "sufficient_direction_mask": 0,
    }
    raw_observations = []
    offline = {}
    evidence = [(b"\x01plan", plan)]
    pattern = (1, 0, 1, 1, 0)
    for slot in range(4):
        direction = 1 if slot % 2 == 0 else -1
        stroke = [
            _observation(
                direction * pattern[index % len(pattern)],
                direction=direction,
                stroke_index=slot,
                target_cpms=direction,
                peak_torque_target_abs=100,
                pid_velocity_actual=direction * 60 if index == 0 else None,
                actual_velocity_pptm=direction * 60 if index == 1 else None,
            )
            for index in range(256)
        ]
        raw_observations.extend(stroke)
        record = {
            "record_kind": "observation",
            "run_sequence": 9,
            "evidence_sequence": slot + 1,
            "rung_index": 0,
            "slot": slot,
            "velocity_p": 100,
            "target_velocity_rpm": direction * 100,
            "flags": int(slot == 0),
        }
        result = _reconstruct_observation(stroke, record, plan, terminal)
        result["flags"] = int(slot == 0)
        record.update(result)
        offline[slot] = result
        evidence.append((bytes([2, slot]), record))

    forward = _rung_class(offline[0], offline[2])
    reverse = _rung_class(offline[1], offline[3])
    bands = [_new_band(), _new_band()]
    _observe_band(bands[0], 0, 100, *forward, (offline[0], offline[2]))
    _observe_band(bands[1], 0, 100, *reverse, (offline[1], offline[3]))
    finished = [_finish_band(band) for band in bands]
    rung = {
        "record_kind": "rung",
        "run_sequence": 9,
        "evidence_sequence": 5,
        "rung_index": 0,
        "velocity_p": 100,
        "forward_class": forward[0],
        "forward_low_q16": forward[1],
        "forward_high_q16": forward[2],
        "reverse_class": reverse[0],
        "reverse_low_q16": reverse[1],
        "reverse_high_q16": reverse[2],
        "flags": int(forward[3]) | (int(reverse[3]) << 1) | (1 << 2),
    }
    for index, prefix in ((0, "forward"), (1, "reverse")):
        rung[f"{prefix}_p_low"] = finished[index]["p_low"]
        rung[f"{prefix}_p_high"] = finished[index]["p_high"]
        rung[f"{prefix}_closure"] = finished[index]["closure"]
        rung[f"{prefix}_eligible_rungs"] = finished[index]["eligible_rungs"]
        rung[f"{prefix}_eligible_observations"] = finished[index]["eligible_observations"]
    evidence.append((b"\x03rung", rung))
    terminal["directions"] = [
        {
            **band,
            "direction_closure": index | (band["closure"] << 1),
        }
        for index, band in enumerate(finished)
    ]
    digest = FNV1A64_OFFSET
    for raw, _record in evidence:
        for byte in raw:
            digest ^= byte
            digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    terminal["digest_low"] = digest & 0xFFFF_FFFF
    terminal["digest_high"] = digest >> 32
    evidence.append((b"terminal", terminal))

    report = audit_velocity_sweep(raw_observations, evidence)

    assert report["matches_firmware"] is True
    assert report["observation_count"] == 4
    assert report["rung_count"] == 1
    assert report["directions"][0]["eligible_rungs"] == 1


def test_stage_b_replay_uses_eight_slots_and_four_k_confidence() -> None:
    from foci_trace.velocity_measurement import _reconstruct_observation, replay_stage_b_regions

    plan = {
        "record_kind": "plan",
        "run_sequence": 10,
        "evidence_sequence": 0,
        "rung_count": 1,
        "observations_per_direction": 4,
        "planned_velocity_mrev_s": 1_667,
        "target_velocity_rpm": 100,
        "p_start": 100,
        "p_top": 100,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
    }
    terminal = {
        "record_kind": "stage_b_terminal",
        "run_sequence": 10,
        "encoder_counts_per_rev": 1_000,
        "f_pwm_hz": 25_000,
    }
    raw_observations = []
    evidence = [(b"plan", plan)]
    pattern = (1, 0, 1, 1, 0)
    for slot in range(8):
        direction = 1 if slot % 2 == 0 else -1
        stroke = [
            _observation(
                direction * pattern[index % len(pattern)],
                direction=direction,
                stroke_index=slot,
                target_cpms=direction,
                peak_torque_target_abs=100,
            )
            for index in range(256)
        ]
        raw_observations.extend(stroke)
        record = {
            "record_kind": "observation",
            "run_sequence": 10,
            "evidence_sequence": slot + 1,
            "rung_index": 0,
            "slot": slot,
            "velocity_p": 100,
            "target_velocity_rpm": direction * 100,
            "flags": int(slot == 0),
        }
        record.update(_reconstruct_observation(stroke, record, plan, terminal))
        evidence.append((bytes([2, slot]), record))
    evidence.append((b"terminal", terminal))

    replay = replay_stage_b_regions(raw_observations, evidence)

    assert replay["slot_count"] == 8
    assert replay["directional_rungs"][0][0]["collected_mask"] == 0b1111
    assert replay["directional_rungs"][1][0]["collected_mask"] == 0b1111
    assert replay["directional_rungs"][0][0]["operable_count"] == 4
    assert replay["directional_rungs"][1][0]["operable_count"] == 4


def test_stage_b_recovery_is_statistically_excluded_at_partial_terminus() -> None:
    from foci_trace.velocity_measurement import _reconstruct_observation, replay_stage_b_regions

    plan = {
        "record_kind": "plan",
        "run_sequence": 11,
        "evidence_sequence": 0,
        "rung_count": 23,
        "observations_per_direction": 4,
        "planned_velocity_mrev_s": 4_394,
        "target_velocity_rpm": 264,
        "p_start": 256,
        "p_top": 32_767,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
        "max_stroke_travel_mrev": 2_500,
        "settle_travel_reserve_mrev": 250,
        "recovery_origin_band_counts": 1_000,
        "recovery_nominal_slot_us": 1_816_958,
        "recovery_maximum_slot_us": 3_000_000,
        "recovery_slot_count": 23,
    }
    terminal = {
        "record_kind": "stage_b_terminal",
        "run_sequence": 11,
        "encoder_counts_per_rev": 4_000,
        "f_pwm_hz": 25_000,
        "emitted_observations": 177,
        "recovery_unavailable": 1,
        "cause": 4,
    }
    raw_observations = []
    evidence: list[tuple[bytes, dict[str, object]]] = [(b"plan", plan)]
    sequence = 1
    drift = (4_632, 6_405, 8_458, 7_971)
    for rung_index in range(22):
        for slot in range(8):
            direction = 1 if slot % 2 == 0 else -1
            stroke = [
                _observation(
                    direction * 17,
                    direction=direction,
                    stroke_index=rung_index * 8 + slot,
                    target_cpms=direction * 18,
                    peak_torque_target_abs=100,
                )
                for _ in range(256)
            ]
            raw_observations.extend(stroke)
            record = {
                "record_kind": "observation",
                "run_sequence": 11,
                "evidence_sequence": sequence,
                "rung_index": rung_index,
                "slot": slot,
                "velocity_p": 256 + rung_index,
                "target_velocity_rpm": direction * 264,
                "flags": 0,
            }
            record.update(_reconstruct_observation(stroke, record, plan, terminal))
            evidence.append((bytes((2, rung_index, slot)), record))
            sequence += 1
        start_offset = drift[rung_index % len(drift)]
        recovery = {
            "record_kind": "rung_origin_recovery",
            "run_sequence": 11,
            "evidence_sequence": sequence,
            "stage": 0,
            "rung_index": rung_index,
            "velocity_p": 256 + rung_index,
            "planned_velocity_mrev_s": 4_394,
            "start_offset_counts": start_offset,
            "closest_offset_counts": 900,
            "final_offset_counts": 900,
            "origin_band_counts": 1_000,
            "lower_rate_q": 20_000_000,
            "moving_duration_us": 200_000,
            "settle_duration_us": 736_000,
            "total_duration_us": 936_000,
            "peak_torque_target_abs": 100,
            "binding_source": 1,
            "outcome": 1,
        }
        assert audit_rung_origin_recovery(recovery, plan)["matches_firmware"] is True
        evidence.append((bytes((9, rung_index)), recovery))
        sequence += 1
    partial = [
        _observation(
            17,
            direction=1,
            stroke_index=176,
            target_cpms=18,
            peak_torque_target_abs=900,
        )
        for _ in range(256)
    ]
    raw_observations.extend(partial)
    partial_record = {
        "record_kind": "observation",
        "run_sequence": 11,
        "evidence_sequence": sequence,
        "rung_index": 22,
        "slot": 0,
        "velocity_p": 278,
        "target_velocity_rpm": 264,
        "flags": 0,
    }
    partial_record.update(_reconstruct_observation(partial, partial_record, plan, terminal))
    evidence.append((b"partial", partial_record))
    evidence.append((b"terminal", terminal))

    with_recovery = replay_stage_b_regions(raw_observations, evidence)
    without_recovery = replay_stage_b_regions(
        raw_observations,
        [item for item in evidence if item[1]["record_kind"] != "rung_origin_recovery"],
    )

    recoveries = [item for _raw, item in evidence if item["record_kind"] == "rung_origin_recovery"]
    assert len(recoveries) == 22
    assert max(abs(int(item["start_offset_counts"])) for item in recoveries) == 8_458
    assert all(abs(int(item["final_offset_counts"])) <= 1_000 for item in recoveries)
    assert terminal["recovery_unavailable"] == 1
    assert with_recovery == without_recovery


def test_stage_b_recovery_failure_owns_the_inconclusive_cause() -> None:
    from foci_trace.velocity_measurement import _stage_b_inconclusive_cause

    assert _stage_b_inconclusive_cause(0, recovery_failure=False) == 1
    assert _stage_b_inconclusive_cause(3, recovery_failure=False) == 2
    assert _stage_b_inconclusive_cause(0, recovery_failure=True) == 7


def test_rung_disjoint_pair_has_no_reportable_disturbance_interval() -> None:
    from foci_trace.velocity_measurement import _rung_class

    first = {
        "classification": 0,
        "disturbance_low_q16": 10,
        "disturbance_high_q16": 20,
    }
    second = {
        "classification": 0,
        "disturbance_low_q16": 30,
        "disturbance_high_q16": 40,
    }

    assert _rung_class(first, second) == (4, 0, 0, True)


@pytest.mark.parametrize(
    ("record_kind", "expected_caveat"), [("plan", "missing terminal"), ("terminal", "missing plan")]
)
def test_sweep_audit_reports_partial_capture_without_raising(
    record_kind: str, expected_caveat: str
) -> None:
    from foci_trace.velocity_measurement import audit_velocity_sweep

    record = {
        "record_kind": record_kind,
        "run_sequence": 9,
        "evidence_sequence": 0,
        "rung_count": 0,
        "observations_per_direction": 2,
    }

    report = audit_velocity_sweep([], [(b"partial", record)])

    assert report["complete"] is False
    assert report["matches_firmware"] is False
    assert any(expected_caveat in caveat for caveat in report["caveats"])


def test_stage_b_first_fit_reseeds_boundary_and_preserves_counterexample() -> None:
    from foci_trace.velocity_measurement import _segment_directional_regions

    rungs = [
        {"rung_index": index, "velocity_p": 100 + index, "class": 0, "low": low, "high": high}
        for index, (low, high) in enumerate(((0, 2), (1, 10), (9, 11), (10, 12)))
    ]

    regions = _segment_directional_regions(rungs)

    assert [region["member_mask"] for region in regions] == [0b0011, 0b1100]
    assert [region["kind"] for region in regions] == ["fragment", "fragment"]
    assert regions[0]["common_interval"] == [1, 2]
    assert regions[0]["closure"] == "disjoint_interval"
    assert regions[0]["boundary_rung"] == 2


def test_stage_b_segments_multiple_regions_and_joint_nomination_ties() -> None:
    from foci_trace.velocity_measurement import (
        _joint_operable_regions,
        _nominate_joint_region,
        _segment_directional_regions,
    )

    gains = [256, 362, 512, 724, 1024, 1448, 2048]
    rungs = [
        {"rung_index": 0, "velocity_p": gains[0], "class": 0, "low": 0, "high": 4},
        {"rung_index": 1, "velocity_p": gains[1], "class": 0, "low": 1, "high": 3},
        {"rung_index": 2, "velocity_p": gains[2], "class": 0, "low": 2, "high": 2},
        {"rung_index": 3, "velocity_p": gains[3], "class": 2, "low": 0, "high": 0},
        {"rung_index": 4, "velocity_p": gains[4], "class": 0, "low": 10, "high": 14},
        {"rung_index": 5, "velocity_p": gains[5], "class": 0, "low": 11, "high": 13},
        {"rung_index": 6, "velocity_p": gains[6], "class": 0, "low": 12, "high": 12},
    ]
    regions = _segment_directional_regions(rungs)
    assert [region["member_mask"] for region in regions] == [0b0000111, 0b1110000]
    assert [region["kind"] for region in regions] == ["valid", "valid"]

    joint = _joint_operable_regions(
        [
            {"rung_index": index, "forward_operable": index != 3, "reverse_operable": index != 3}
            for index in range(7)
        ]
    )
    assert [region["member_mask"] for region in joint] == [0b0000111, 0b1110000]
    nomination = _nominate_joint_region(joint, gains, gains[0], gains[-1])
    assert nomination["member_mask"] == 0b1110000
    assert nomination["rung_index"] == 5
    assert nomination["p"] == 1448
    exact_tie = _nominate_joint_region(
        [
            {"member_mask": 0b0001100, "first_rung": 2, "last_rung": 3, "member_count": 2},
            {"member_mask": 0b0110000, "first_rung": 4, "last_rung": 5, "member_count": 2},
        ],
        gains,
        gains[0],
        gains[-1],
    )
    assert exact_tie["member_mask"] == 0b0001100


def test_stage_b_handoff_prefers_full_then_nearest_and_compares_candidates() -> None:
    from foci_trace.velocity_measurement import (
        _compare_stage_b_candidates_schema2,
        _select_directional_region,
    )

    gains = [256, 362, 512, 724, 1024, 1448, 2048]
    regions = [
        {"member_mask": 0b0000110, "pooled_interval": [1, 2]},
        {"member_mask": 0b0011100, "pooled_interval": [3, 4]},
        {"member_mask": 0b1100000, "pooled_interval": [5, 6]},
    ]

    selected = _select_directional_region(regions, 3, gains)
    assert selected["member_mask"] == 0b0011100
    assert selected["coverage"] == "full"
    transferred = _select_directional_region([regions[0], regions[2]], 3, gains)
    assert transferred["member_mask"] == 0b0000110
    assert transferred["coverage"] == "transferred"
    assert transferred["gain_ratio"] == [724, 512]

    first = {
        "plan_digest": 7,
        "nominated_p": 724,
        "memberships": [0b0011100, 0b0011100],
        "intervals": [[1, 4], [-4, -1]],
    }
    assert _compare_stage_b_candidates_schema2(first, dict(first))["outcome"] == "complete"
    changed = {**first, "memberships": [0b0111000, 0b0011100]}
    mismatch = _compare_stage_b_candidates_schema2(first, changed)
    assert mismatch["outcome"] == "reproduction_mismatch"
    assert mismatch["replacement"] == changed
    changed_plan = {**first, "plan_digest": 8}
    assert _compare_stage_b_candidates_schema2(first, changed_plan)["outcome"] == "plan_mismatch"


def test_stage_b_core_projection_accepts_shared_membership_and_renominates() -> None:
    from foci_trace.velocity_measurement import project_stage_b_core_pair

    previous = {
        "memberships": [0x00380000, 0x00380000, 0x00380000],
        "nominated_p": 1448,
        "pooled_intervals": [[100, 120], [-130, -100]],
        "common_intervals": [[104, 115], [-125, -109]],
        "p_start": 1,
        "p_top": 2457,
        "rung_count": 23,
    }
    current = {
        **previous,
        "memberships": [0x003C0000, 0x00380000, 0x00380000],
        "nominated_p": 1024,
        "pooled_intervals": [[108, 128], [-140, -110]],
        "common_intervals": [[112, 118], [-120, -105]],
    }

    projected = project_stage_b_core_pair(previous, current)

    assert projected["outcome"] == "complete"
    assert projected["reason_mask"] == 0
    assert projected["core_masks"] == [0x00380000] * 3
    assert projected["memberships"][0] == {
        "previous": 0x00380000,
        "current": 0x003C0000,
        "core": 0x00380000,
        "previous_only": 0,
        "current_only": 0x00040000,
        "low_delta": -1,
        "high_delta": 0,
    }
    assert projected["final_p"] == 1448
    assert projected["coverage"] == [
        {"coverage": "full", "signed_rung_distance": 0, "gain_ratio": [1, 1]},
        {"coverage": "full", "signed_rung_distance": 0, "gain_ratio": [1, 1]},
    ]
    assert projected["pooled"][0]["overlap"] == [108, 120]
    assert projected["firmware_conservative_common"][0] == [112, 115]


def test_stage_b_core_projection_reports_all_mismatch_reasons() -> None:
    from foci_trace.velocity_measurement import project_stage_b_core_pair

    previous = {
        "memberships": [0b00111, 0b11100, 0b00001],
        "nominated_p": 4,
        "pooled_intervals": [[0, 1], [10, 11]],
        "common_intervals": [[0, 2], [10, 12]],
        "p_start": 2,
        "p_top": 32,
        "rung_count": 5,
    }
    current = {
        **previous,
        "memberships": [0b11100, 0b00111, 0b10000],
        "pooled_intervals": [[2, 3], [12, 13]],
        "common_intervals": [[3, 4], [13, 14]],
    }

    projected = project_stage_b_core_pair(previous, current)

    assert projected["outcome"] == "reproduction_mismatch"
    assert projected["reason_mask"] == 0b1_1111
    assert projected["final_p"] == 0


def test_stage_b_core_projection_preserves_reduced_and_transferred_coverage() -> None:
    from foci_trace.velocity_measurement import project_stage_b_core_pair

    previous = {
        "memberships": [0b11100, 0b00111, 0b00010],
        "nominated_p": 4,
        "pooled_intervals": [[0, 4], [-4, 0]],
        "common_intervals": [[1, 3], [-3, -1]],
        "p_start": 2,
        "p_top": 32,
        "rung_count": 5,
    }
    current = dict(previous)

    projected = project_stage_b_core_pair(previous, current)

    assert projected["outcome"] == "complete"
    assert projected["reduced_margin"] is True
    assert projected["final_p"] == 3
    assert projected["coverage"][0] == {
        "coverage": "transferred",
        "signed_rung_distance": 1,
        "gain_ratio": [4, 3],
    }
    assert projected["coverage"][1]["coverage"] == "full"


def test_schema8_projection_selects_shared_operating_component_from_captures() -> None:
    import json
    from pathlib import Path

    from foci_trace.velocity_measurement import project_stage_b_schema8_pair

    capture_root = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-20-foc-183-velocity-integral-response"
    )
    names = (
        "foc-183-stepper-x-schema7-capability-fixed-candidate-20260723-192516.velocity.json",
        "foc-183-stepper-x-schema7-capability-fixed-composite-20260723-194119.velocity.json",
        "foc-183-stepper-x-schema7-4-compact-candidate-20260724-165436.velocity.json",
    )
    reports = [json.loads((capture_root / name).read_text())["sweeps"][0] for name in names]
    compact_reverse = reports[2]["stage_b_replay"]["directions"][1]

    assert [
        (int(fragment["member_mask"]), int(fragment["member_count"]))
        for fragment in compact_reverse
    ] == [(0x0001_0000, 1), (0x0018_0000, 2)]

    projections = (
        project_stage_b_schema8_pair(reports[0], reports[1]),
        project_stage_b_schema8_pair(reports[0], reports[2]),
        project_stage_b_schema8_pair(reports[1], reports[2]),
    )

    assert [projection["final_p"] for projection in projections] == [1024, 1024, 1024]
    assert projections[0]["selected_joint_mask"] == 0x001C_0000
    assert projections[0]["reduced_margin"] is False
    for projection in projections[1:]:
        assert projection["interpretation"] == "schema-8 what-if"
        assert projection["current_candidate"]["nominated_p"] == 256
        assert projection["current_candidate"]["memberships"][1] == 0x0018_0000
        assert projection["selected_joint_mask"] == 0x0018_0000
        assert projection["selected_first_rung"] == 19
        assert projection["selected_member_count"] == 2
        assert projection["reduced_margin"] is True
        assert projection["directional_validity"] == [3, 1]
    assert all(projection["final_p"] != 1448 for projection in projections)


def test_schema8_candidate_terminal_uses_joint_union_membership() -> None:
    from foci_trace.velocity_measurement import _candidate_terminal_memberships

    replay = {
        "selected": [
            {"member_mask": 0b11100},
            {"member_mask": 0b11100},
        ],
        "joint_regions": [
            {"member_mask": 0b00011},
            {"member_mask": 0b11100},
        ],
        "nomination": {"member_mask": 0b11100},
    }

    assert _candidate_terminal_memberships(replay, 8) == [
        0b11100,
        0b11100,
        0b11111,
    ]
    assert _candidate_terminal_memberships(replay, 7) == [
        0b11100,
        0b11100,
        0b11100,
    ]


def test_stage_b_core_projection_derives_exact_core_common_offline() -> None:
    from foci_trace.velocity_measurement import project_stage_b_core_pair

    previous = {
        "memberships": [0b1111, 0b1111, 0b1110],
        "nominated_p": 4,
        "pooled_intervals": [[100, 120], [-120, -100]],
        "common_intervals": [[110, 114], [-116, -110]],
        "per_rung_common_intervals": [
            {0: [110, 114], 1: [108, 120], 2: [106, 119], 3: [107, 118]},
            {0: [-116, -110], 1: [-120, -108], 2: [-119, -106], 3: [-118, -107]},
        ],
        "p_start": 2,
        "p_top": 32,
        "rung_count": 5,
    }
    current = {
        **previous,
        "memberships": [0b1110, 0b1110, 0b1110],
        "common_intervals": [[112, 118], [-118, -112]],
        "per_rung_common_intervals": [
            {1: [109, 119], 2: [108, 120], 3: [107, 121]},
            {1: [-119, -109], 2: [-120, -108], 3: [-121, -107]},
        ],
    }

    projected = project_stage_b_core_pair(previous, current)

    assert projected["firmware_conservative_common"][0] == [112, 114]
    assert projected["offline_derived"]["core_restricted_common"][0] == [109, 118]
    assert projected["offline_derived"]["provenance"] == "per_rung_cross_run"


def test_stage_b_core_projection_does_not_gate_on_empty_conservative_common() -> None:
    from foci_trace.velocity_measurement import project_stage_b_core_pair

    previous = {
        "memberships": [0b111, 0b111, 0b111],
        "nominated_p": 3,
        "pooled_intervals": [[0, 10], [-10, 0]],
        "common_intervals": [[0, 2], [-10, -8]],
        "per_rung_common_intervals": [
            {0: [0, 10], 1: [1, 9], 2: [2, 8]},
            {0: [-10, 0], 1: [-9, -1], 2: [-8, -2]},
        ],
        "p_start": 2,
        "p_top": 32,
        "rung_count": 5,
    }
    current = {
        **previous,
        "common_intervals": [[8, 10], [-2, 0]],
        "per_rung_common_intervals": [
            {0: [0, 10], 1: [1, 9], 2: [2, 8]},
            {0: [-10, 0], 1: [-9, -1], 2: [-8, -2]},
        ],
    }

    projected = project_stage_b_core_pair(previous, current)

    assert projected["outcome"] == "complete"
    assert projected["common"][0]["nonempty"] is False
    assert projected["common"][1]["nonempty"] is False
    assert projected["offline_derived"]["core_restricted_common"] == [[2, 8], [-8, -2]]


def test_stage_b_reproduction_v3_group_requires_exact_fragment_order() -> None:
    from foci_trace.velocity_measurement import _assemble_stage_b_reproduction_v3

    header = {
        "record_kind": "reproduction_v3_header",
        "run_sequence": 7,
        "evidence_sequence": 33,
        "schema_revision": 3,
        "outcome": 1,
        "reason_mask": 0,
        "previous_nominated_p": 1024,
        "current_nominated_p": 1448,
        "final_p": 1448,
        "reduced_margin": 0,
    }
    records = [header]
    for object_index in range(3):
        records.append(
            {
                "record_kind": "reproduction_v3_membership",
                "run_sequence": 7,
                "evidence_sequence": 33,
                "object": object_index,
                "previous": 7,
                "current": 7,
                "core": 7,
                "previous_only": 0,
                "current_only": 0,
                "low_delta": 0,
                "high_delta": 0,
            }
        )
    for kind in ("pooled", "common", "coverage"):
        for direction in range(2):
            records.append(
                {
                    "record_kind": f"reproduction_v3_{kind}",
                    "run_sequence": 7,
                    "evidence_sequence": 33,
                    "direction": direction,
                }
            )
    records.append(
        {
            "record_kind": "reproduction_v3_digest",
            "run_sequence": 7,
            "evidence_sequence": 33,
            "previous_digest_low": 1,
            "previous_digest_high": 2,
            "current_digest_low": 3,
            "current_digest_high": 4,
        }
    )

    assembled = _assemble_stage_b_reproduction_v3(records)

    assert assembled is not None
    assert assembled["schema_revision"] == 3
    assert len(assembled["memberships"]) == 3
    with pytest.raises(ValueError, match="order"):
        _assemble_stage_b_reproduction_v3([records[0], records[2], records[1], *records[3:]])


def test_stage_b_reproduction_v4_group_accepts_shared_fragment_names() -> None:
    from foci_trace.velocity_measurement import _assemble_stage_b_reproduction

    identity = {"run_sequence": 7, "evidence_sequence": 34}
    records = [
        {
            **identity,
            "record_kind": "reproduction_v4_header",
            "schema_revision": 8,
            "outcome": 1,
            "reason_mask": 0,
            "previous_provisional_p": 1024,
            "current_provisional_p": 1024,
            "final_p": 1024,
            "selected_joint_mask": 0x001C_0000,
            "selected_first_rung": 18,
            "selected_member_count": 3,
            "forward_validity": 3,
            "reverse_validity": 3,
            "reduced_margin": 0,
        },
        *(
            {
                **identity,
                "record_kind": "reproduction_membership",
                "object": object_index,
            }
            for object_index in range(3)
        ),
        *(
            {
                **identity,
                "record_kind": f"reproduction_{kind}",
                "direction": direction,
            }
            for kind in ("pooled", "common", "coverage")
            for direction in range(2)
        ),
        {
            **identity,
            "record_kind": "reproduction_digest",
            "previous_digest_low": 1,
            "previous_digest_high": 2,
            "current_digest_low": 3,
            "current_digest_high": 4,
        },
    ]

    assembled = _assemble_stage_b_reproduction(records)

    assert assembled is not None
    assert assembled["record_kind"] == "reproduction_v4"
    assert assembled["selected_joint_mask"] == 0x001C_0000
    assert len(assembled["memberships"]) == 3


def test_stage_b_causal_order_rejects_reordered_directional_closures() -> None:
    from foci_trace.velocity_measurement import _compare_stage_b_causal_order

    replay = {
        "gains": [256],
        "structured_boundaries": {},
        "directions": [
            [{"member_mask": 1, "boundary_rung": None}],
            [{"member_mask": 1, "boundary_rung": None}],
        ],
        "joint_regions": [{"member_mask": 1, "closure": "end_of_ladder", "last_rung": 0}],
        "nomination": {"member_mask": 1},
    }
    records = [
        {"record_kind": "plan"},
        *({"record_kind": "observation", "rung_index": 0, "slot": slot} for slot in range(4)),
        {"record_kind": "rung", "rung_index": 0},
        {"record_kind": "directional_region", "direction": 1, "member_mask": 1},
        {"record_kind": "directional_region", "direction": 0, "member_mask": 1},
        {"record_kind": "joint_region", "member_mask": 1},
        {"record_kind": "handoff"},
        {"record_kind": "stage_b_terminal"},
    ]

    mismatch = _compare_stage_b_causal_order(records, replay)

    assert "causal order" in mismatch


def test_stage_b_causal_order_places_recovery_after_each_complete_rung() -> None:
    from foci_trace.velocity_measurement import _compare_stage_b_causal_order

    replay = {
        "slot_count": 8,
        "structured_boundaries": {},
        "directions": [[], []],
        "joint_regions": [],
        "nomination": {},
        "sufficient": False,
    }
    records = [
        {"record_kind": "plan"},
        *({"record_kind": "observation", "rung_index": 0, "slot": slot} for slot in range(8)),
        {"record_kind": "rung", "rung_index": 0},
        {"record_kind": "rung_origin_recovery", "rung_index": 0},
        {"record_kind": "stage_b_terminal"},
    ]

    assert _compare_stage_b_causal_order(records, replay) == ""
    records[-2], records[-3] = records[-3], records[-2]
    assert "causal order" in _compare_stage_b_causal_order(records, replay)


def test_stage_b_causal_order_omits_handoff_for_insufficient_run() -> None:
    from foci_trace.velocity_measurement import _compare_stage_b_causal_order

    replay = {
        "gains": [256],
        "structured_boundaries": {},
        "directions": [
            [{"member_mask": 1, "boundary_rung": None}],
            [{"member_mask": 1, "boundary_rung": None}],
        ],
        "joint_regions": [{"member_mask": 1, "closure": "end_of_ladder", "last_rung": 0}],
        "nomination": {"member_mask": 1},
        "sufficient": False,
    }
    records = [
        {"record_kind": "plan"},
        *({"record_kind": "observation", "rung_index": 0, "slot": slot} for slot in range(4)),
        {"record_kind": "rung", "rung_index": 0},
        {"record_kind": "directional_region", "direction": 0, "member_mask": 1},
        {"record_kind": "directional_region", "direction": 1, "member_mask": 1},
        {"record_kind": "joint_region", "member_mask": 1},
        {"record_kind": "stage_b_terminal"},
    ]

    assert _compare_stage_b_causal_order(records, replay) == ""


def test_stage_b_partial_current_headroom_closes_active_regions() -> None:
    from foci_trace.velocity_measurement import (
        _joint_operable_regions,
        _segment_directional_regions,
    )

    directional = [
        {"rung_index": index, "velocity_p": 256 << index, "class": 0, "low": 1, "high": 2}
        for index in range(3)
    ]
    joint = [
        {"rung_index": index, "forward_operable": True, "reverse_operable": True}
        for index in range(3)
    ]

    regions = _segment_directional_regions(
        directional,
        terminal_closure="current_headroom",
        terminal_boundary_rung=3,
    )
    joint_regions = _joint_operable_regions(
        joint,
        terminal_closure="current_headroom",
    )

    assert regions[-1]["closure"] == "current_headroom"
    assert regions[-1]["boundary_rung"] is None
    assert regions[-1]["causal_boundary_rung"] == 3
    assert joint_regions[-1]["closure"] == "current_headroom"


def test_stage_b_braking_contact_marks_partial_current_headroom_boundary() -> None:
    from foci_trace.velocity_measurement import _stage_b_terminal_boundary

    closure, boundary_rung, partial_rungs = _stage_b_terminal_boundary(
        terminal={"cause": 4, "emitted_rungs": 22},
        current_records=[
            {
                "rung_index": 21,
                "slot": 0,
                "contact": 2,
            }
        ],
        completed_rungs=21,
    )

    assert closure == "current_headroom"
    assert boundary_rung == 21
    assert partial_rungs == [21]


def test_stage_b_complete_retains_partial_current_headroom_boundary() -> None:
    from foci_trace.velocity_measurement import _stage_b_terminal_boundary

    closure, boundary_rung, partial_rungs = _stage_b_terminal_boundary(
        terminal={
            "cause": 0,
            "emitted_rungs": 22,
            "recovery_unavailable": 1,
        },
        current_records=[
            {
                "rung_index": 21,
                "slot": 0,
                "contact": 2,
            }
        ],
        completed_rungs=21,
    )

    assert closure == "current_headroom"
    assert boundary_rung == 21
    assert partial_rungs == [21]


def test_stage_b_current_headroom_regions_follow_partial_rung() -> None:
    from foci_trace.velocity_measurement import _compare_stage_b_causal_order

    replay = {
        "slot_count": 8,
        "structured_boundaries": {},
        "directions": [
            [{"member_mask": 1, "closure": "current_headroom", "causal_boundary_rung": 1}],
            [{"member_mask": 1, "closure": "current_headroom", "causal_boundary_rung": 1}],
        ],
        "joint_regions": [{"member_mask": 1, "closure": "current_headroom", "last_rung": 0}],
        "nomination": {"member_mask": 1},
        "sufficient": True,
    }
    records = [
        {"record_kind": "plan"},
        {"record_kind": "observation", "rung_index": 1, "slot": 0},
        {"record_kind": "rung", "rung_index": 1},
        {"record_kind": "directional_region", "direction": 0, "member_mask": 1},
        {"record_kind": "directional_region", "direction": 1, "member_mask": 1},
        {"record_kind": "joint_region", "member_mask": 1},
        {"record_kind": "handoff"},
        {"record_kind": "stage_b_terminal"},
    ]

    assert _compare_stage_b_causal_order(records, replay) == ""


def test_schema_five_fault_prefix_keeps_unfinished_regions_out_of_parity() -> None:
    from foci_trace.velocity_measurement import _stage_b_fault_prefix_replay

    replay = {
        "directions": [
            [
                {"kind": "fragment", "closure": "ineligible_gap", "member_mask": 1},
                {"kind": "valid", "closure": "end_of_ladder", "member_mask": 0b1110},
            ],
            [{"kind": "valid", "closure": "end_of_ladder", "member_mask": 0b1110}],
        ],
        "joint_regions": [
            {"closure": "ineligible_gap", "member_mask": 1},
            {"closure": "end_of_ladder", "member_mask": 0b1110},
        ],
        "nomination": {"member_mask": 0b1110, "p": 1024},
        "nominated_p": 1024,
        "selected": [{"member_mask": 0b1110}, {"member_mask": 0b1110}],
        "sufficient": True,
        "coverage": ["full", "full"],
    }

    parity = _stage_b_fault_prefix_replay(replay)

    assert parity["directions"] == [
        [{"kind": "fragment", "closure": "ineligible_gap", "member_mask": 1}],
        [],
    ]
    assert parity["joint_regions"] == [{"closure": "ineligible_gap", "member_mask": 1}]
    assert parity["nomination"] == {}
    assert parity["nominated_p"] == 0
    assert parity["selected"] == [{}, {}]
    assert parity["sufficient"] is False


def test_schema_six_projection_keeps_complete_boundary_rung_in_all_folds() -> None:
    from foci_trace.velocity_measurement import _project_schema6_current_headroom

    replay = {
        "directions": [
            [
                {
                    "kind": "valid",
                    "member_mask": 0x003C_0000,
                    "common_interval": [16_689_709, 16_897_585],
                }
            ],
            [
                {
                    "kind": "valid",
                    "member_mask": 0x003C_0000,
                    "common_interval": [-16_117_228, -15_501_617],
                }
            ],
        ],
        "directional_rungs": [
            [{"class": 0, "included_mask": 0b1110}],
            [{"class": 0, "included_mask": 0b1111}],
        ],
        "joint_regions": [{"member_mask": 0x003C_0000}],
        "nomination": {"member_mask": 0x003C_0000, "p": 1024},
        "nominated_p": 1024,
        "selected": [
            {
                "member_mask": 0x003C_0000,
                "common_interval": [16_689_709, 16_897_585],
            },
            {
                "member_mask": 0x003C_0000,
                "common_interval": [-16_117_228, -15_501_617],
            },
        ],
        "structured_boundaries": {"21/0": {"passing_suffix_mask": 4}},
    }

    projection = _project_schema6_current_headroom(replay, terminal_rung=21)

    assert projection["interpretation"] == "schema-6 current-headroom what-if"
    assert projection["firmware_authoritative"] is False
    assert projection["expected_recovery_rungs"] == list(range(21))
    assert projection["directional_masks"] == [0x003C_0000, 0x003C_0000]
    assert projection["joint_mask"] == 0x003C_0000
    assert projection["nominated_p"] == 1024
    assert projection["directional_common_intervals"] == [
        [16_689_709, 16_897_585],
        [-16_117_228, -15_501_617],
    ]
    assert projection["boundary_rung_consensus"] == [
        {"class": 0, "included_mask": 0b1110},
        {"class": 0, "included_mask": 0b1111},
    ]
    assert projection["triggering_observation_exclusions"] == [
        {"rung_index": 21, "slot": 0, "passing_suffix_mask": 4}
    ]


def test_stage_b_handoff_requires_valid_models_in_both_directions() -> None:
    from foci_trace.velocity_measurement import _stage_b_handoff_sufficient

    nomination = {"member_mask": 1}
    selected = [{}, {"member_mask": 1}]
    one_sided = [
        [{"kind": "fragment"}],
        [{"kind": "valid"}],
    ]

    assert _stage_b_handoff_sufficient(one_sided, nomination, selected) is False

    bidirectional = [
        [{"kind": "valid"}],
        [{"kind": "valid"}],
    ]
    selected[0] = {"member_mask": 1}
    assert _stage_b_handoff_sufficient(bidirectional, nomination, selected) is True


def test_stage_b_reproduction_pair_joins_previous_digest_and_candidate() -> None:
    from foci_trace.velocity_measurement import audit_stage_b_reproduction_pair

    previous = {
        "terminal": {
            "selected_memberships": [7, 7, 7],
            "selected_intervals": [[10, 20], [-20, -10]],
            "nominated_p": 724,
        },
        "reported_digest": 0x1111,
        "plan_identity": 0x2222,
    }
    current = {
        "terminal": {
            "selected_memberships": [7, 7, 7],
            "selected_intervals": [[15, 25], [-25, -15]],
            "nominated_p": 724,
            "outcome": 1,
        },
        "plan_identity": 0x2222,
        "pre_reproduction_digest": 0x3333,
        "reproduction": {
            "outcome": 1,
            "previous_nominated_p": 724,
            "current_nominated_p": 724,
            "previous_memberships": [7, 7, 7],
            "current_memberships": [7, 7, 7],
            "previous_intervals": [[10, 20], [-20, -10]],
            "current_intervals": [[15, 25], [-25, -15]],
            "previous_digest_low": 0x1111,
            "previous_digest_high": 0,
            "current_digest_low": 0x3333,
            "current_digest_high": 0,
        },
    }

    result = audit_stage_b_reproduction_pair(previous, current)

    assert result == {"matches_firmware": True, "mismatches": []}

    current["reproduction"]["previous_memberships"] = [3, 7, 7]
    mismatch = audit_stage_b_reproduction_pair(previous, current)
    assert mismatch["matches_firmware"] is False
    assert any("previous memberships" in item for item in mismatch["mismatches"])


def test_schema2_reproduction_captures_keep_historical_parity() -> None:
    import json
    from pathlib import Path

    from foci_trace.velocity_measurement import audit_stage_b_reproduction_pair

    capture_root = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-19-foc-185-multistroke"
    )
    pairs = (
        (
            "foc-185-stepper-x-candidate-final-20260720-082815.json",
            "foc-185-stepper-x-reproduction-final-20260720-091324.json",
        ),
        (
            "foc-185-stepper-y-candidate-final-20260720-092043.json",
            "foc-185-stepper-y-reproduction-final-20260720-092918.json",
        ),
    )
    for candidate_name, reproduction_name in pairs:
        candidate = json.loads((capture_root / candidate_name).read_text())["sweeps"][0]
        reproduction = json.loads((capture_root / reproduction_name).read_text())["sweeps"][0]
        assert candidate["reproduction"] is None
        assert reproduction["reproduction"]["record_kind"] == "reproduction"
        assert reproduction["terminal"]["outcome"] == 2
        assert reproduction["terminal"]["cause"] == 5
        assert audit_stage_b_reproduction_pair(candidate, reproduction)["matches_firmware"] is True

    for name in (
        "foc-185-stepper-x-candidate-retained-20260720-002637.json",
        "foc-185-stepper-x-reproduction-retained-20260720-003204.json",
    ):
        retained = json.loads((capture_root / name).read_text())["sweeps"][0]
        assert retained["reproduction"] is None
        assert retained["terminal"]["outcome"] == 0

    n2_root = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-19-foc-194-operating-regions"
    )
    for name in (
        "foc-194-stepper-x-gap5ms-candidate-20260719-175704.json",
        "foc-194-stepper-y-gap5ms-candidate-20260719-182837.json",
    ):
        historical = json.loads((n2_root / name).read_text())["sweeps"][0]
        assert historical["plan"]["observations_per_direction"] == 2
        assert historical["matches_firmware"] is True
        assert historical["terminal"]["outcome"] == 2


def test_stage_b_audit_accepts_exact_zero_rung_fault_terminal() -> None:
    from foci_trace.velocity_measurement import FNV1A64_OFFSET, FNV1A64_PRIME, audit_velocity_sweep

    plan_raw = b"stage-b-plan"
    digest = FNV1A64_OFFSET
    for byte in plan_raw:
        digest ^= byte
        digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    plan = {
        "record_kind": "plan",
        "run_sequence": 4,
        "evidence_sequence": 0,
        "plan_digest_low": 1,
        "plan_digest_high": 0,
        "rung_count": 0,
        "observations_per_direction": 2,
        "planned_velocity_mrev_s": 1_000,
        "target_velocity_rpm": 60,
        "p_start": 1,
        "p_top": 1,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
    }
    terminal = {
        "record_kind": "stage_b_terminal",
        "run_sequence": 4,
        "evidence_sequence": 1,
        "encoder_counts_per_rev": 4_000,
        "f_pwm_hz": 25_000,
        "model_direction_mask": 0,
        "selected_memberships": [0, 0, 0],
        "nominated_p": 0,
        "selected_intervals": [[0, 0], [0, 0]],
        "coverage_mask": 0,
        "forward_region_count": 0,
        "reverse_region_count": 0,
        "forward_fragment_count": 0,
        "reverse_fragment_count": 0,
        "expected_observations": 0,
        "emitted_observations": 0,
        "expected_rungs": 0,
        "emitted_rungs": 0,
        "digest_low": digest & 0xFFFF_FFFF,
        "digest_high": digest >> 32,
        "outcome": 3,
        "cause": 9,
    }

    report = audit_velocity_sweep([], [(plan_raw, plan), (b"terminal", terminal)])

    assert report["matches_firmware"] is True
    assert report["nominated_p"] == 0


def test_structured_boundary_reports_longer_tail_without_changing_binding() -> None:
    from foci_trace.velocity_measurement import analyze_structured_boundary

    observations = [
        _observation(
            (10 if (index // 4) % 2 == 0 else -10) if index >= 192 else (1, -1, 0, -1, 1)[index % 5]
        )
        for index in range(256)
    ]

    boundary = analyze_structured_boundary(observations)

    assert boundary is not None
    assert boundary["passing_suffix_mask"] & 0b010
    assert boundary["binding_lag_one_margin_q"] < 0
