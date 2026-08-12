import json
import struct
from pathlib import Path

import pytest

from foci_trace.protocol import decode_breakaway_record
from foci_trace.velocity_measurement import (
    _BW_NOMINATION_SCREEN_C_SIGMA,
    _BW_STAGE_B_SCHEMA_REVISION,
    AccuracyDecision,
    VelocityObservation,
    _bw_classify_is_eligible,
    _bw_directional_included_mask,
    _bw_observation_interval_percent,
    _bw_observation_statistic,
    _bw_percent_to_milli,
    _bw_relative_se_to_permille,
    _confidence_multiplier,
    analyze_stationary_tail,
    analyze_velocity_observations,
    audit_breakaway_campaign,
    audit_rung_origin_recovery,
    audit_scored_rest_grammar,
    audit_selected_recovery_rest,
    breakaway_build_search_grid,
    breakaway_confirmation_plan_digest,
    breakaway_discovery_plan_digest,
    breakaway_discovery_rung_margin,
    breakaway_plan_additive_discovery_ladder,
    breakaway_probe_plan_digest,
    breakaway_realize_candidate_additive_ladder,
    breakaway_replay_upward_probe,
    breakaway_required_relative_se_permille,
    classify_absolute_rate,
    plan_rung_origin_recovery,
    project_recovery_authority,
    recovery_return_distance_counts,
    recovery_runtime_us,
    recovery_target_rpm,
    recovery_wrong_way,
    select_recovery_authority,
    summarize_recovery_rest_diagnostics,
    velocity_report_to_json,
)

FIXTURE = Path(__file__).parent / "fixtures" / "velocity_measurement_golden.json"

_BREAKAWAY_WORKFLOW_DIGEST = 0xA1B2_C3D4_E5F6_0718
_BREAKAWAY_TARGET_VELOCITY_RPM = -600
_BREAKAWAY_HARD_TORQUE_LIMIT = 2_816
_BREAKAWAY_USABLE_TORQUE_LIMIT = 2_048
_BREAKAWAY_PWM_HZ = 25_000
_BREAKAWAY_ENCODER_COUNTS_PER_REV = 4_000


def test_breakaway_probe_plan_digest_matches_the_firmware_context_fixture() -> None:
    assert (
        breakaway_probe_plan_digest(
            [3, 5, 6, 9],
            63,
            2_000,
            128,
            workflow_digest=_BREAKAWAY_WORKFLOW_DIGEST,
            target_velocity_rpm=_BREAKAWAY_TARGET_VELOCITY_RPM,
            hard_torque_limit=_BREAKAWAY_HARD_TORQUE_LIMIT,
            usable_torque_limit=_BREAKAWAY_USABLE_TORQUE_LIMIT,
            pwm_hz=_BREAKAWAY_PWM_HZ,
            encoder_counts_per_rev=_BREAKAWAY_ENCODER_COUNTS_PER_REV,
        )
        == 0x58E4_3362_740F_DD86
    )


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
        # A recovery that completed its move and ended outside the band is
        # contained at runtime, not at plan time. Firmware reports
        # ContainedDeadline/Deadline here with real timings, and the plan-time
        # inputs cannot distinguish this from the Recovered case.
        (
            {
                "closest_offset_counts": 900,
                "final_offset_counts": 1_001,
                "binding_source": 3,
                "outcome": 3,
            },
            None,
        ),
        # Containment refused before motion because the start offset sat outside
        # the outer-safety position window. Reported as
        # ContainedDeadline/LocalTravel with no timings.
        (
            {
                "closest_offset_counts": 8_458,
                "final_offset_counts": 8_458,
                "moving_duration_us": 0,
                "settle_duration_us": 0,
                "total_duration_us": 0,
                "binding_source": 4,
                "outcome": 3,
            },
            None,
        ),
        # A contained recovery that moved still has to be internally consistent.
        (
            {
                "closest_offset_counts": 900,
                "final_offset_counts": 1_001,
                "total_duration_us": 1_936_001,
                "binding_source": 3,
                "outcome": 3,
            },
            "total duration differs from moving plus settle",
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

    # No schema_revision is passed here, so this takes the legacy (schema < 14,
    # or unknown) gate: the significance test, unchanged from before Task 9.
    # This linear ramp's fitted slope is large and unambiguous, so it still
    # rejects with ResolvedSlope (discriminant 2), giving Stage B's historical
    # `4 + 2 = 6` -- not 12, which only an at-or-above-schema-fourteen capture
    # (asserted separately below) would produce.
    assert result["classification"] == 6
    assert result["tested_suffixes"] == 0


def test_excluded_observation_uses_drift_bound_at_stage_b_schema_fourteen_and_above() -> None:
    """The same ramp, replayed at/above STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION
    (14), takes the drift bound instead: its bound suffix drifts far more than
    200,000 ppm of its settled rate, so it rejects as ExcessiveDrift
    (discriminant 5), which Stage B's explicit map sends to wire code 12."""
    from foci_trace.velocity_measurement import (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        _reconstruct_observation,
    )

    observations = [_observation(index) for index in range(256)]
    record = {"velocity_p": 32, "target_velocity_rpm": 8, "slot": 0}
    plan = {"hard_torque_limit": 2_816, "usable_torque_limit": 2_534}
    terminal = {"encoder_counts_per_rev": 4_000, "f_pwm_hz": 25_000}

    result = _reconstruct_observation(
        observations,
        record,
        plan,
        terminal,
        schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
    )

    assert result["classification"] == 12
    assert result["tested_suffixes"] == 0


def test_stage_b_schema_thirteen_still_uses_the_legacy_gate() -> None:
    """Stage B revision 13 predates the drift bound -- it was live firmware
    until Task 7 bumped COMBINED_STAGE_B_SCHEMA_REVISION to 14 -- even though
    13 is Stage C's post-bump floor. This pins the namespace split: if Stage
    B's and Stage C's thresholds were ever collapsed back into one shared
    constant of 13, this would incorrectly take the drift bound (classification
    12) instead of the legacy gate (classification 6)."""
    from foci_trace.velocity_measurement import _reconstruct_observation

    observations = [_observation(index) for index in range(256)]
    record = {"velocity_p": 32, "target_velocity_rpm": 8, "slot": 0}
    plan = {"hard_torque_limit": 2_816, "usable_torque_limit": 2_534}
    terminal = {"encoder_counts_per_rev": 4_000, "f_pwm_hz": 25_000}

    result = _reconstruct_observation(observations, record, plan, terminal, schema_revision=13)

    assert result["classification"] == 6
    assert result["tested_suffixes"] == 0


def test_host_drift_ratio_matches_firmware_exactly() -> None:
    """The host mirror must be integer, not another _f32 mirror. A float
    reimplementation reintroduces exactly the one-ppm boundary disagreement the
    integer form exists to remove."""
    from foci_trace.velocity_measurement import _drift_ratio_ppm

    assert _drift_ratio_ppm(1_000_000, 4_000_000, 20_000_000) == 100_000
    assert _drift_ratio_ppm(200_000, 2_000_000, 1_000_000) == 200_000
    assert _drift_ratio_ppm(200_001, 2_000_000, 1_000_000) == 200_001
    assert _drift_ratio_ppm(1, 3, 1_000_000) == 1
    # This is the case that actually proves ceiling division: floor division
    # gives 38_502.
    assert _drift_ratio_ppm(-8_823_529, 2 * 60_000, 13_750_000) == 38_503


def test_rest_mode_keeps_the_significance_gate() -> None:
    """At rest the zero-slope null is correct, and a drift bound relative to
    the mean would collapse with it: keep the significance test unchanged."""
    from foci_trace.velocity_measurement import (
        RATE_Q_SCALE,
        _classify_stationarity,
        _Level,
        _observe,
        _q,
        _select_level,
        _stationarity_fit,
        _trunc_div,
    )

    # A steady linear ramp: the fitted slope is large and unambiguous, so the
    # significance test must reject it at rest even though its magnitude is
    # far too small to trip the moving-mode drift bound.
    observations = [_observation(index) for index in range(64)]
    levels = [_Level() for _ in range(5)]
    for observation in observations:
        _observe(levels, observation)
    selection = _select_level(levels)
    assert selection is not None
    variance = _q(levels[selection].moments)
    assert variance is not None
    mean_rate_q = _trunc_div(
        levels[selection].exact_delta_counts * 1_000_000 * RATE_Q_SCALE,
        levels[selection].exact_elapsed_us,
    )
    fit = _stationarity_fit(observations, selection, variance, mean_rate_q)
    assert isinstance(fit, dict)

    assert _classify_stationarity(fit, "rest", mean_rate_q, 0) == 2


def test_moving_mode_bypasses_drift_bound_when_settled_interval_contains_zero() -> None:
    """The Inert condition (settled interval containing zero) bypasses the
    drift bound entirely rather than reporting a ratio against a near-zero
    denominator, exactly as firmware's classify_moving_metrics does.

    The drift bound only applies at or above the stage-qualified threshold
    (see test_moving_gate_is_schema_gated below for the legacy side of that
    split), so this test must pass a schema_revision and its matching
    threshold to exercise it at all.
    """
    from foci_trace.velocity_measurement import (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        _classify_stationarity,
    )

    fit = {
        "drift_ratio_ppm": 999_999_999,
        "slope_passed": False,
        "lag_one_passed": True,
    }

    # mean_rate_q's magnitude is within half_width_rate_q, so the interval
    # contains zero: bypass fires even though the drift ratio is enormous.
    assert (
        _classify_stationarity(
            fit,
            "moving",
            mean_rate_q=5,
            half_width_rate_q=10,
            schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
            drift_bound_min_schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        )
        is None
    )


def test_moving_mode_rejects_excessive_drift_outside_the_inert_interval() -> None:
    from foci_trace.velocity_measurement import (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        _classify_stationarity,
    )

    fit = {
        "drift_ratio_ppm": 200_001,
        "slope_passed": False,
        "lag_one_passed": True,
    }

    assert (
        _classify_stationarity(
            fit,
            "moving",
            mean_rate_q=100,
            half_width_rate_q=10,
            schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
            drift_bound_min_schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        )
        == 5
    )


def test_moving_gate_is_schema_gated() -> None:
    """Below the stage-qualified threshold (and when the revision is
    unknown), moving mode falls back to the significance test instead of the
    drift bound, matching firmware that never ran the drift bound at all."""
    from foci_trace.velocity_measurement import (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        _classify_stationarity,
    )

    # A huge drift ratio would reject under the new gate, but a passing
    # significance test must accept it under the legacy gate.
    passing_fit = {
        "drift_ratio_ppm": 999_999_999,
        "slope_passed": True,
        "lag_one_passed": True,
    }
    assert (
        _classify_stationarity(
            passing_fit,
            "moving",
            mean_rate_q=100,
            half_width_rate_q=10,
            schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION - 1,
            drift_bound_min_schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        )
        is None
    )
    assert (
        _classify_stationarity(
            passing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, schema_revision=None
        )
        is None
    )

    # A tiny drift ratio would pass under the new gate, but a failing
    # significance test must still reject it under the legacy gate.
    failing_fit = {
        "drift_ratio_ppm": 1,
        "slope_passed": False,
        "lag_one_passed": True,
    }
    assert (
        _classify_stationarity(
            failing_fit,
            "moving",
            mean_rate_q=100,
            half_width_rate_q=10,
            schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION - 1,
            drift_bound_min_schema_revision=STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        )
        == 2
    )
    assert (
        _classify_stationarity(
            failing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, schema_revision=None
        )
        == 2
    )


def test_classify_stationarity_requires_stage_qualified_threshold_with_schema_revision() -> None:
    """A schema_revision without its matching threshold is a caller bug, not a
    silently-picked default -- there are two namespaces and no safe single
    choice."""
    from foci_trace.velocity_measurement import _classify_stationarity

    fit = {"drift_ratio_ppm": 0, "slope_passed": True, "lag_one_passed": True}

    with pytest.raises(ValueError, match="drift_bound_min_schema_revision"):
        _classify_stationarity(fit, "moving", mean_rate_q=0, half_width_rate_q=0, schema_revision=1)


def test_scored_rest_selection_recomputes_all_suffixes_independently() -> None:
    from foci_trace.velocity_measurement import analyze_scored_rest_selection

    result = analyze_scored_rest_selection([_observation(0) for _ in range(256)])

    assert result == {
        "structural_codes": [0, 0, 0],
        "interval_present_mask": 0b111,
        "usable_mask": 0b111,
        "zero_compatible_mask": 0b111,
        "selected_mask": 0b100,
        "selected_suffix_len": 256,
        "disposition": 0,
        "intervals_q": [[0, 0], [0, 0], [0, 0]],
        "conditioned_rest_evidence": True,
    }


def test_scored_rest_record_must_match_independent_recomputation() -> None:
    from foci_trace.velocity_measurement import (
        analyze_scored_rest_selection,
        verify_scored_rest_selection,
    )

    observations = [_observation(0) for _ in range(256)]
    record = analyze_scored_rest_selection(observations)
    record["intervals_q"] = [[0, 0], [0, 1], [0, 0]]

    with pytest.raises(ValueError, match="intervals_q"):
        verify_scored_rest_selection(observations, record)


def test_scored_rest_grammar_requires_fixed_observation_and_recovery_joins() -> None:
    """A scored-rest selection follows the observation it describes (R4): the
    grammar had this backwards, requiring the selection first. This is the
    same fixture as before the fix, reordered to the corrected sequence and
    carrying the two new required keyword arguments; the swap at the end still
    proves the untouched `recovery_rest` predecessor guard fires."""
    identity = {
        "run_sequence": 7,
        "stage": 0,
        "plan_digest": 11,
        "disposition": 0,
    }
    records = [
        {"record_kind": "observation", "rung_index": 2, "slot": 3},
        {
            "record_kind": "rest_selection",
            "owner": 0,
            "rung_index": 2,
            "slot": 3,
            **identity,
        },
        {
            "record_kind": "rest_selection",
            "owner": 1,
            "rung_index": 2,
            "slot": 0xFF,
            **identity,
        },
        {"record_kind": "recovery_rest", "rung_index": 2},
        {"record_kind": "rung_origin_recovery", "rung_index": 2},
    ]
    terminal = {"record_kind": "stage_b_terminal", "run_sequence": 7, "cause": 0}
    arguments = {
        "schema_revision": 13,
        "schema_floor": 13,
        "expected_stage": 0,
        "ordinary_owners": (0,),
        "recovery_owners": (1, 2),
        "observation_kind": "observation",
        "plan_digest": 11,
        "label": "Stage B",
        "stabilized_recovery_rest": True,
        "ordinary_rejection_tail": (),
    }

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []
    records[2], records[3] = records[3], records[2]
    assert audit_scored_rest_grammar(records, terminal, **arguments)


_GRAMMAR_RUN_SEQUENCE = 7
_GRAMMAR_PLAN_DIGEST = 11
_GRAMMAR_STAGE = 1
_GRAMMAR_ORDINARY_OWNERS = (3, 4, 6)
_GRAMMAR_RECOVERY_OWNERS = (5,)
_GRAMMAR_OBSERVATION_KIND = "integral_observation"


def _grammar_identity(*, disposition: int) -> dict[str, object]:
    return {
        "run_sequence": _GRAMMAR_RUN_SEQUENCE,
        "stage": _GRAMMAR_STAGE,
        "plan_digest": _GRAMMAR_PLAN_DIGEST,
        "disposition": disposition,
    }


def _grammar_observation(rung_index: int, slot: int) -> dict[str, object]:
    return {"record_kind": _GRAMMAR_OBSERVATION_KIND, "rung_index": rung_index, "slot": slot}


def _grammar_selection(
    *, owner: int, rung_index: int, slot: int, disposition: int
) -> dict[str, object]:
    return {
        "record_kind": "rest_selection",
        "owner": owner,
        "rung_index": rung_index,
        "slot": slot,
        **_grammar_identity(disposition=disposition),
    }


def _grammar_current(rung_index: int, slot: int) -> dict[str, object]:
    return {"record_kind": "velocity_primitive_current", "rung_index": rung_index, "slot": slot}


def _grammar_recovery_rest(rung_index: int) -> dict[str, object]:
    return {"record_kind": "recovery_rest", "rung_index": rung_index}


def _grammar_recovery(rung_index: int) -> dict[str, object]:
    return {"record_kind": "rung_origin_recovery", "rung_index": rung_index}


def _grammar_terminal(*, kind: str, cause: int) -> dict[str, object]:
    return {"record_kind": kind, "run_sequence": _GRAMMAR_RUN_SEQUENCE, "cause": cause}


def _grammar_arguments(
    *,
    label: str,
    stabilized_recovery_rest: bool,
    ordinary_rejection_tail: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_revision": 13,
        "schema_floor": 12,
        "expected_stage": _GRAMMAR_STAGE,
        "ordinary_owners": _GRAMMAR_ORDINARY_OWNERS,
        "recovery_owners": _GRAMMAR_RECOVERY_OWNERS,
        "observation_kind": _GRAMMAR_OBSERVATION_KIND,
        "plan_digest": _GRAMMAR_PLAN_DIGEST,
        "label": label,
        "stabilized_recovery_rest": stabilized_recovery_rest,
        "ordinary_rejection_tail": ordinary_rejection_tail,
    }


def test_scored_rest_grammar_accepts_an_ordinary_selection_after_its_observation() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=0),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_stage_c_ordinary_rejection_current_tail() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
        _grammar_current(2, 3),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage C",
        stabilized_recovery_rest=False,
        ordinary_rejection_tail=("velocity_primitive_current",),
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_stage_b_ordinary_rejection_without_current() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="stage_b_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage B", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_an_immediate_recovery_selection() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=0),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_an_immediate_recovery_rejection() -> None:
    records = [_grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2)]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_a_stabilized_recovery_selection() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=0),
        _grammar_recovery_rest(2),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_a_stabilized_recovery_rejection() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2),
        _grammar_recovery_rest(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_rejects_a_stabilized_recovery_missing_event_150() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=0),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage C recovery selection does not precede Event 150"
    ]


def test_scored_rest_grammar_rejects_stage_c_rejection_missing_its_current_record() -> None:
    """The disposition-2 tail is enforced, not merely exempted: a Stage-C
    rejection missing its Current record must be reported, not silently
    accepted the way the pre-fix grammar accepted any rejection."""
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage C",
        stabilized_recovery_rest=False,
        ordinary_rejection_tail=("velocity_primitive_current",),
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage C rejected scored-rest selection lacks its causal terminal tail"
    ]


def test_scored_rest_grammar_rejects_a_rejection_tail_with_an_extra_record() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
        {"record_kind": "integral_rung", "rung_index": 2, "slot": 3},
    ]
    terminal = _grammar_terminal(kind="stage_b_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage B", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage B rejected scored-rest selection lacks its causal terminal tail"
    ]


def test_scored_rest_grammar_rejects_an_ordinary_rejection_with_the_wrong_cause() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="stage_b_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage B", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage B rejected scored-rest selection lacks its causal terminal tail"
    ]


def test_scored_rest_grammar_rejects_an_immediate_recovery_rejection_with_a_recovery() -> None:
    """A rejection emits no recovery record at all; a rejected immediate-scored
    selection followed by `rung_origin_recovery` must be reported."""
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage C rejected recovery selection lacks its causal terminal"
    ]


def test_scored_rest_grammar_rejects_a_stabilized_recovery_with_the_wrong_cause() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2),
        _grammar_recovery_rest(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Stage C", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Stage C rejected recovery selection lacks its causal terminal"
    ]


def test_sweep_confidence_width_matches_firmware_single_precision(monkeypatch) -> None:
    import foci_trace.velocity_measurement as measurement

    monkeypatch.setattr(
        measurement,
        "_stationary_suffix",
        lambda _raw, **_kwargs: {
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


def _stage_b_single_rung_evidence(
    drift_ratio_ppm: int | None,
) -> tuple[list[object], list[tuple[bytes, dict[str, object]]]]:
    """Build one four-slot Stage B stream whose records report a drift ratio or not."""
    from foci_trace.velocity_measurement import _reconstruct_observation

    plan = {
        "record_kind": "plan",
        "run_sequence": 14,
        "evidence_sequence": 0,
        "rung_count": 1,
        "observations_per_direction": 2,
        "planned_velocity_mrev_s": 1_667,
        "target_velocity_rpm": 100,
        "p_start": 100,
        "p_top": 100,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
    }
    terminal = {
        "record_kind": "stage_b_terminal",
        "run_sequence": 14,
        "encoder_counts_per_rev": 1_000,
        "f_pwm_hz": 25_000,
    }
    raw_observations: list[object] = []
    evidence: list[tuple[bytes, dict[str, object]]] = [(b"plan", plan)]
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
            )
            for index in range(256)
        ]
        raw_observations.extend(stroke)
        record = {
            "record_kind": "observation",
            "run_sequence": 14,
            "evidence_sequence": slot + 1,
            "rung_index": 0,
            "slot": slot,
            "velocity_p": 100,
            "target_velocity_rpm": direction * 100,
            "flags": int(slot == 0),
        }
        record.update(_reconstruct_observation(stroke, record, plan, terminal))
        # The decoder always sets this key: an int on captures that carry the
        # trailing u32, None on the ones taken before the field existed.
        record["drift_ratio_ppm"] = drift_ratio_ppm
        evidence.append((bytes([2, slot]), record))
    evidence.append((b"terminal", terminal))
    return raw_observations, evidence


def test_stage_b_replay_derives_the_drift_bound_schema_from_the_reported_ratio() -> None:
    """A reported drift ratio is the only wire signal that a capture is schema 14.

    Stage B's plan record carries no schema revision, so replay derives one
    from the stream. Before this arm existed the derivation topped out at 12
    and real schema-14 captures were replayed against the legacy significance
    gate, which reports `matches_firmware: False` on exactly the observations
    the drift bound was added to accept.
    """
    from foci_trace.velocity_measurement import (
        STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION,
        replay_stage_b_regions,
    )

    reported = replay_stage_b_regions(*_stage_b_single_rung_evidence(1_234))
    absent = replay_stage_b_regions(*_stage_b_single_rung_evidence(None))

    assert reported["schema_revision"] == STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION
    assert absent["schema_revision"] < STAGE_B_DRIFT_BOUND_MIN_SCHEMA_REVISION


def test_drift_excluded_observations_reproduce_as_excluded_not_faulted() -> None:
    """Report code 12 is an exclusion, so a drift-excluded pair is EXCLUDED.

    The numeric band used to be `>= 9 -> fault`, which swallowed code 12.
    Firmware's `reproduce_velocity_rung` matches `StationarityExcluded(_)`
    and yields Excluded for the same pair.
    """
    from foci_trace.velocity_measurement import _rung_class

    drift = {"classification": 12}
    resolved_slope = {"classification": 6}
    wrong_direction = {"classification": 9}

    assert _rung_class(drift, drift)[0] == 6
    assert _rung_class(drift, resolved_slope)[0] == 6
    assert _rung_class(drift, wrong_direction)[0] == 7
    assert _rung_class(wrong_direction, drift)[0] == 7


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


def test_selected_recovery_rest_binds_schema11_and_terminal_disposition() -> None:
    plan = {
        "plan_digest_low": 0x5566_7788,
        "plan_digest_high": 0x1122_3344,
        "recovery_origin_band_counts": 1_000,
        "nominal_workflow_ms": 271_423,
        "maximum_workflow_ms": 298_632,
    }
    terminal = {
        "emitted_rungs": 16,
        "outcome": 1,
        "cause": 0,
        "recovery_unavailable": 0,
    }
    selected = {
        "plan_digest": 0x1122_3344_5566_7788,
        "stage": 0,
        "rung_index": 15,
        "start_offset_counts": -1_834,
        "closest_offset_counts": 998,
        "final_offset_counts": 999,
        "planned_return_distance_counts": 834,
        "first_in_band_offset_counts": 998,
        "displacement_to_band_counts": 2_832,
        "entry_to_band_duration_us": 90_904,
        "lead_in_net_displacement_counts": -3,
        "lead_in_displacement_span_counts": 7,
        "scored_net_displacement_counts": 0,
        "scored_min_displacement_counts": -1,
        "scored_max_displacement_counts": 1,
        "disposition": 0,
    }
    recovery = {"rung_index": 15, "moving_duration_us": 100_000}

    assert audit_selected_recovery_rest(selected, plan, terminal, [recovery])["matches_firmware"]

    wrong_stage_duration = dict(plan, nominal_workflow_ms=271_422)
    audit = audit_selected_recovery_rest(selected, wrong_stage_duration, terminal, [recovery])
    assert audit["matches_firmware"] is False
    assert any("outside the Stage-B-11 duration pair" in item for item in audit["mismatches"])

    unconfirmed = dict(selected, disposition=1)
    terminal = {
        **terminal,
        "outcome": 2,
        "cause": 7,
        "recovery_unavailable": 1,
    }
    assert audit_selected_recovery_rest(unconfirmed, plan, terminal, [])["matches_firmware"]


def test_episode_eight_selected_recovery_uses_stage_local_duration() -> None:
    plan = {
        "plan_digest_low": 1_505_349_572,
        "plan_digest_high": 898_309_201,
        "recovery_origin_band_counts": 1_000,
        "nominal_workflow_ms": 271_423,
        "maximum_workflow_ms": 298_632,
    }
    terminal = {
        "emitted_rungs": 20,
        "outcome": 1,
        "cause": 0,
        "recovery_unavailable": 0,
    }
    selected = {
        "plan_digest": 3_858_208_641_496_240_068,
        "stage": 0,
        "rung_index": 19,
        "start_offset_counts": -1_176,
        "closest_offset_counts": -950,
        "final_offset_counts": -951,
        "planned_return_distance_counts": 176,
        "first_in_band_offset_counts": -991,
        "displacement_to_band_counts": 185,
        "entry_to_band_duration_us": 17_857,
        "lead_in_net_displacement_counts": 40,
        "lead_in_displacement_span_counts": 41,
        "scored_net_displacement_counts": 0,
        "scored_min_displacement_counts": 0,
        "scored_max_displacement_counts": 0,
        "disposition": 0,
    }
    recovery = {"rung_index": 19, "moving_duration_us": 17_912}

    audit = audit_selected_recovery_rest(selected, plan, terminal, [recovery])

    assert audit["matches_firmware"]


def test_recovery_rest_distribution_uses_lower_middle_and_sorted_rows() -> None:
    records = [
        {
            "stage": 0,
            "rung_index": rung,
            "planned_return_distance_counts": distance,
            "entry_to_band_duration_us": duration,
            "lead_in_net_displacement_counts": net,
            "lead_in_displacement_span_counts": span,
        }
        for rung, distance, duration, net, span in (
            (3, 400, 40, -4, 8),
            (0, 100, 10, 1, 2),
            (2, 300, 30, -3, 6),
            (1, 200, 20, 2, 4),
        )
    ]

    summary = summarize_recovery_rest_diagnostics(records)

    assert summary["diagnostic_count"] == 4
    assert summary["planned_return_distance_counts"] == {
        "min": 100,
        "median": 200,
        "max": 400,
    }
    assert summary["entry_to_band_duration_us"] == {
        "min": 10,
        "median": 20,
        "max": 40,
    }
    assert summary["absolute_lead_in_net_counts"] == {
        "min": 1,
        "median": 2,
        "max": 4,
    }
    assert summary["lead_in_span_counts"] == {
        "min": 2,
        "median": 4,
        "max": 8,
    }
    assert [row["rung_index"] for row in summary["rows"]] == [0, 1, 2, 3]


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


def test_stage_b_causal_order_places_selected_rest_before_final_recovery() -> None:
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
        {"record_kind": "recovery_rest", "rung_index": 0},
        {"record_kind": "rung_origin_recovery", "rung_index": 0},
        {"record_kind": "stage_b_terminal"},
    ]

    assert _compare_stage_b_causal_order(records, replay) == ""
    records[-2], records[-3] = records[-3], records[-2]
    assert "causal order" in _compare_stage_b_causal_order(records, replay)


def test_stage_b_causal_order_allows_failed_selected_rest_without_recovery() -> None:
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
        {"record_kind": "recovery_rest", "rung_index": 0},
        {"record_kind": "stage_b_terminal"},
    ]

    assert _compare_stage_b_causal_order(records, replay) == ""


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


def test_stage_b_handoff_sufficiency_is_schema_aware() -> None:
    from foci_trace.velocity_measurement import _stage_b_handoff_sufficient

    nomination = {"member_mask": 1}
    selected = [{}, {"member_mask": 1}]
    one_sided = [
        [{"kind": "fragment"}],
        [{"kind": "valid"}],
    ]

    assert (
        _stage_b_handoff_sufficient(
            one_sided,
            nomination,
            selected,
            schema_revision=7,
        )
        is False
    )
    assert (
        _stage_b_handoff_sufficient(
            one_sided,
            nomination,
            selected,
            schema_revision=8,
        )
        is True
    )

    bidirectional = [
        [{"kind": "valid"}],
        [{"kind": "valid"}],
    ]
    selected[0] = {"member_mask": 1}
    assert (
        _stage_b_handoff_sufficient(
            bidirectional,
            nomination,
            selected,
            schema_revision=7,
        )
        is True
    )


def test_schema8_directional_retention_prefers_fragment_near_provisional_p() -> None:
    from foci_trace.velocity_measurement import _select_schema8_directional_evidence

    selected = _select_schema8_directional_evidence(
        [
            {
                "kind": "fragment",
                "first_rung": 16,
                "last_rung": 16,
                "member_count": 1,
                "member_mask": 0x0001_0000,
            },
            {
                "kind": "fragment",
                "first_rung": 19,
                "last_rung": 20,
                "member_count": 2,
                "member_mask": 0x0018_0000,
            },
        ],
        provisional_rung=19,
    )

    assert selected["member_mask"] == 0x0018_0000


def test_schema8_terminal_model_mask_uses_reproduced_validity() -> None:
    from foci_trace.velocity_measurement import _stage_b_model_mask_from_validity

    assert _stage_b_model_mask_from_validity([1, 3]) == 0b10
    assert _stage_b_model_mask_from_validity([4, 3]) == 0b11
    assert _stage_b_model_mask_from_validity([0, 2]) == 0


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


def _stage_b_recovery_prefix_report(
    *, outcome: int = 3, rung_count: int = 2, recovered_rungs: tuple[int, ...] = (0,)
) -> dict[str, object]:
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
        "run_sequence": 12,
        "evidence_sequence": 0,
        "plan_digest_low": 1,
        "plan_digest_high": 0,
        "rung_count": rung_count,
        "observations_per_direction": 2,
        "planned_velocity_mrev_s": 4_394,
        "target_velocity_rpm": 264,
        "p_start": 100,
        "p_top": 100,
        "hard_torque_limit": 1_000,
        "usable_torque_limit": 900,
        "max_stroke_travel_mrev": 2_500,
        "settle_travel_reserve_mrev": 250,
        "recovery_origin_band_counts": 1_000,
        "recovery_nominal_slot_us": 1_816_958,
        "recovery_maximum_slot_us": 3_000_000,
        "recovery_slot_count": rung_count,
    }
    terminal = {
        "record_kind": "stage_b_terminal",
        "run_sequence": 12,
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
        "expected_observations": 4 * rung_count,
        "emitted_observations": 4 * rung_count,
        "expected_rungs": rung_count,
        "emitted_rungs": rung_count,
        "outcome": outcome,
        "cause": 11 if outcome == 3 else 0,
        "recovery_unavailable": 0,
    }
    raw_observations = []
    plan_raw = b"stage-b-recovery-plan"
    evidence = [(plan_raw, plan)]
    pattern = (1, 0, 1, 1, 0)
    sequence = 1
    bands = [_new_band(), _new_band()]
    for rung_index in range(rung_count):
        reconstructed = {}
        velocity_p = 100 + rung_index
        for slot in range(4):
            direction = 1 if slot % 2 == 0 else -1
            stroke_index = rung_index * 4 + slot
            stroke = [
                _observation(
                    direction * pattern[index % len(pattern)],
                    direction=direction,
                    stroke_index=stroke_index,
                    target_cpms=direction * 18,
                    peak_torque_target_abs=100,
                )
                for index in range(256)
            ]
            raw_observations.extend(stroke)
            record = {
                "record_kind": "observation",
                "run_sequence": 12,
                "evidence_sequence": sequence,
                "rung_index": rung_index,
                "slot": slot,
                "velocity_p": velocity_p,
                "target_velocity_rpm": direction * 264,
                "flags": int(rung_index == 0 and slot == 0),
            }
            record.update(_reconstruct_observation(stroke, record, plan, terminal))
            reconstructed[slot] = record
            evidence.append((bytes((2, rung_index, slot)), record))
            sequence += 1

        forward = _rung_class(reconstructed[0], reconstructed[2])
        reverse = _rung_class(reconstructed[1], reconstructed[3])
        _observe_band(
            bands[0],
            rung_index,
            velocity_p,
            *forward,
            (reconstructed[0], reconstructed[2]),
        )
        _observe_band(
            bands[1],
            rung_index,
            velocity_p,
            *reverse,
            (reconstructed[1], reconstructed[3]),
        )
        finished = [_finish_band(band) for band in bands]
        rung = {
            "record_kind": "rung",
            "run_sequence": 12,
            "evidence_sequence": sequence,
            "rung_index": rung_index,
            "velocity_p": velocity_p,
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
        evidence.append((bytes((3, rung_index)), rung))
        sequence += 1
        if rung_index in recovered_rungs:
            evidence.append(
                (
                    bytes((9, rung_index)),
                    {
                        "record_kind": "rung_origin_recovery",
                        "run_sequence": 12,
                        "evidence_sequence": sequence,
                        "stage": 0,
                        "rung_index": rung_index,
                        "velocity_p": velocity_p,
                        "planned_velocity_mrev_s": 4_394,
                        "start_offset_counts": 900,
                        "closest_offset_counts": 900,
                        "final_offset_counts": 900,
                        "origin_band_counts": 1_000,
                        "lower_rate_q": 0,
                        "moving_duration_us": 0,
                        "settle_duration_us": 0,
                        "total_duration_us": 0,
                        "peak_torque_target_abs": 100,
                        "binding_source": 0,
                        "outcome": 0,
                    },
                )
            )
            sequence += 1
    terminal["evidence_sequence"] = sequence
    digest = FNV1A64_OFFSET
    for encoded, _record in evidence:
        for byte in encoded:
            digest ^= byte
            digest = (digest * FNV1A64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    terminal["digest_low"] = digest & 0xFFFF_FFFF
    terminal["digest_high"] = digest >> 32
    evidence.append((b"stage-b-terminal", terminal))
    return audit_velocity_sweep(raw_observations, evidence)


def test_stage_b_fault_accepts_one_missing_final_recovery() -> None:
    report = _stage_b_recovery_prefix_report()

    assert not any("recovery rungs" in item for item in report["mismatches"])


def test_stage_b_non_fault_rejects_one_missing_final_recovery() -> None:
    report = _stage_b_recovery_prefix_report(outcome=2)

    assert any("recovery rungs" in item for item in report["mismatches"])


def test_stage_b_fault_rejects_two_missing_recovery_records() -> None:
    report = _stage_b_recovery_prefix_report(
        rung_count=3,
        recovered_rungs=(0,),
    )

    assert any("recovery rungs" in item for item in report["mismatches"])


def test_stage_b_fault_rejects_missing_intermediate_recovery() -> None:
    report = _stage_b_recovery_prefix_report(recovered_rungs=(1,))

    assert any("recovery rungs" in item for item in report["mismatches"])


def test_stage_b_fault_terminal_evidence_comparison_matches_a_complete_terminal() -> None:
    """A fault terminal can
    preserve real, nonempty evidence instead of resetting it: dispatch-level
    fault promotion (`apply_stage_b_terminal`'s `EvidenceIntegrityFault` arm,
    `commission_dispatch/stage_b.rs:248-252`) overwrites only `outcome`/`cause`
    on an already-built terminal, leaving `model_direction_mask`,
    `coverage_mask`, `nominated_p`, `selected_memberships`, and
    `selected_intervals` untouched. Stage-C's audit assumed fault implied
    reset and rejected the preserved case as a false mismatch until
    fixed. `_audit_stage_b_velocity_sweep` was never written with that
    assumption -- its `_compare_fields` call for these evidence fields
    (`velocity_measurement.py:4838`) runs unconditionally, before any
    outcome-specific checks. Pin that: the same construction reports the
    identical evidence-comparison verdict whether its terminal says Complete
    or Fault, so a future change cannot silently reintroduce the Stage-C
    mistake here.
    """

    def evidence_mismatches(report: dict[str, object]) -> list[str]:
        return sorted(
            item
            for item in report["mismatches"]
            if "model_direction_mask" in item
            or "coverage_mask" in item
            or "nominated_p" in item
            or "selected memberships" in item
            or "selected intervals" in item
        )

    complete = _stage_b_recovery_prefix_report(outcome=0, rung_count=10, recovered_rungs=())
    fault = _stage_b_recovery_prefix_report(outcome=3, rung_count=10, recovered_rungs=())

    complete_mismatches = evidence_mismatches(complete)
    assert complete_mismatches, "the construction must produce real, nonzero evidence to compare"
    assert evidence_mismatches(fault) == complete_mismatches


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


def test_combined_selector_projects_first_bidirectional_band_rung() -> None:
    from foci_trace.velocity_measurement import project_combined_response_selection

    projection = project_combined_response_selection(
        [
            {
                "p": 724,
                "forward": [11_000_000, 11_600_000],
                "reverse": [-11_500_000, -10_900_000],
                "eligible": [True, True],
            },
            {
                "p": 1_024,
                "forward": [12_900_000, 13_200_000],
                "reverse": [-13_100_000, -12_800_000],
                "eligible": [True, True],
            },
            {
                "p": 1_448,
                "forward": [14_500_000, 15_000_000],
                "reverse": [-14_900_000, -14_400_000],
                "eligible": [True, True],
            },
        ],
        target_rpm=264,
        encoder_counts_per_rev=4_000,
    )

    assert projection == {
        "interpretation": "projected_result_under_combined_selector",
        "selected_p": 1_024,
        "selected_intervals": [
            [12_900_000, 13_200_000],
            [-13_100_000, -12_800_000],
        ],
    }


def test_combined_selector_uses_closed_exact_band_endpoints() -> None:
    from foci_trace.velocity_measurement import project_combined_response_selection

    exact = project_combined_response_selection(
        [
            {
                "p": 1_024,
                "forward": [12_320_000, 14_080_000],
                "reverse": [-14_080_000, -12_320_000],
                "eligible": [True, True],
            }
        ],
        target_rpm=264,
        encoder_counts_per_rev=4_000,
    )
    outside = project_combined_response_selection(
        [
            {
                "p": 1_024,
                "forward": [12_319_999, 14_080_000],
                "reverse": [-14_080_000, -12_320_000],
                "eligible": [True, True],
            }
        ],
        target_rpm=264,
        encoder_counts_per_rev=4_000,
    )

    assert exact["selected_p"] == 1_024
    assert outside["selected_p"] is None


@pytest.mark.parametrize(
    "capture_name",
    (
        "foc-183-stepper-y-schema8-stagec7-candidate-abnfix-20260726-144157-20260726-144204.bin",
        "foc-183-stepper-y-schema8-stagec7-reproduction-abnfix-20260726-145237-20260726-145244.bin",
    ),
)
def test_corrected_y_capture_projects_combined_p_1024(capture_name: str) -> None:
    from foci_trace.analyze import analyze_capture

    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-20-foc-183-velocity-integral-response"
        / capture_name
    )
    with capture.open("rb") as source:
        audit = analyze_capture(source).velocity_sweep_audits[0]

    assert audit["matches_firmware"] is True
    assert audit["stage_b_replay"]["combined_selection_projection"]["interpretation"] == (
        "projected_result_under_combined_selector"
    )
    assert audit["stage_b_replay"]["combined_selection_projection"]["selected_p"] == 1_024


def test_schema_nine_combined_capture_replays_response_selected_stage_b() -> None:
    from foci_trace.analyze import analyze_capture

    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-26-foc-183-an053-combined"
        / "foc183-y-combined-schema9-stagec8-20260726-200611.complete.bin"
    )
    with capture.open("rb") as source:
        result = analyze_capture(source)

    stage_b = result.velocity_sweep_audits[0]
    stage_c = result.velocity_integral_audits[0]
    assert stage_b["matches_firmware"] is True
    assert stage_b["stage_b_replay"]["schema_revision"] == 9
    assert stage_b["stage_b_replay"]["combined_response_selected"] is True
    assert stage_b["nominated_p"] == 1_024
    assert stage_c["matches_firmware"] is True
    assert stage_c["plan"]["combined_selected_response"] is True
    assert stage_c["terminal"]["cause"] == 47


# --- Breakaway-seeded P ladder --------------------------------------------


def _bw_tag(tag: int, fmt: str, *values: object) -> bytes:
    return bytes([tag]) + struct.pack("<" + fmt, *values)


def _bw_record(raw: bytes) -> tuple[bytes, dict[str, object]]:
    return raw, decode_breakaway_record(raw)


def _bw_raw_observation(
    run_sequence: int,
    sequence: int,
    plan_digest: int,
    phase: int,
    rung_index: int,
    slot_index: int,
    direction: int,
    mean_rate_q: int,
    variance_of_mean_q2: int,
    target_rate_q: int,
    *,
    stationarity_code: int = 0,
    settled_rate_low_q: int | None = None,
    settled_rate_high_q: int | None = None,
    deficit_low_q: int | None = None,
    deficit_high_q: int | None = None,
    predicted_torque_target_abs: int = 100,
    usable_torque_limit: int = 500,
    hard_torque_limit: int = 1000,
    disturbance_present: int = 1,
    disturbance_low_q16: int = 0,
    disturbance_high_q16: int = 1000,
    drift_ratio_ppm: int = 0,
    post_fit_lag_one_state: int = 2,
    stroke_min_displacement_counts: int = 0,
    stroke_max_displacement_counts: int = 0,
    capture_max_interval_us: int = 1_000,
    moving_pid_output_peak: int = 0,
) -> bytes:
    """Build a tag-5 `BreakawayRawObservation` record's raw bytes.

    The classify/consensus inputs default (per direction) to a value the ported
    classifier judges Eligible with a single-component, fully-included disturbance
    consensus, so a fresh fixture's derived masks include every collected slot and
    the recomputed nomination bounds match the fixture's own reported bounds. Tests
    override individual inputs to exercise exclusion.

    `struct.pack` has no native 128-bit format code, so `variance_of_mean_q2`
    is appended as its own 16-byte little-endian span, mirroring the firmware's
    own `push_u128`.
    """
    forward = direction == 0
    if settled_rate_low_q is None:
        settled_rate_low_q = 100 if forward else -200
    if settled_rate_high_q is None:
        settled_rate_high_q = 200 if forward else -100
    if deficit_low_q is None:
        deficit_low_q = 10 if forward else -20
    if deficit_high_q is None:
        deficit_high_q = 20 if forward else -10
    head = struct.pack(
        "<BIHQBBBBq",
        5,
        run_sequence,
        sequence,
        plan_digest,
        phase,
        rung_index,
        slot_index,
        direction,
        mean_rate_q,
    )
    variance_bytes = (variance_of_mean_q2 & ((1 << 128) - 1)).to_bytes(16, "little")
    tail = struct.pack(
        "<qBqqqqHHHBii",
        target_rate_q,
        stationarity_code,
        settled_rate_low_q,
        settled_rate_high_q,
        deficit_low_q,
        deficit_high_q,
        predicted_torque_target_abs,
        usable_torque_limit,
        hard_torque_limit,
        disturbance_present,
        disturbance_low_q16,
        disturbance_high_q16,
    )
    return (
        head
        + variance_bytes
        + tail
        + struct.pack(
            "<IBiiIH",
            drift_ratio_ppm,
            post_fit_lag_one_state,
            stroke_min_displacement_counts,
            stroke_max_displacement_counts,
            capture_max_interval_us,
            moving_pid_output_peak,
        )
    )


def _bw_classifier_obs(slot: int, direction: int, **overrides) -> dict:
    """A decoded raw-observation dict whose frozen classify/consensus inputs
    default to Eligible with an overlapping disturbance interval. Tests override
    fields to force exclusion."""
    forward = direction == 0
    base = {
        "slot_index": slot,
        "direction": direction,
        "settled_rate_low_q": 100 if forward else -200,
        "settled_rate_high_q": 200 if forward else -100,
        "deficit_low_q": 10 if forward else -20,
        "deficit_high_q": 20 if forward else -10,
        "predicted_torque_target_abs": 100,
        "usable_torque_limit": 500,
        "hard_torque_limit": 1000,
        "stationarity_code": 0,
        "disturbance_present": 1,
        "disturbance_low_q16": 0,
        "disturbance_high_q16": 1000,
        "drift_ratio_ppm": 0,
        "post_fit_lag_one_state": 2,
    }
    base.update(overrides)
    return base


def test_bw_classify_is_eligible_reproduces_the_firmware_cascade():
    # Default is Eligible.
    assert _bw_classify_is_eligible(_bw_classifier_obs(0, 0)) is True
    # Each gate independently excludes.
    assert _bw_classify_is_eligible(_bw_classifier_obs(0, 0, stationarity_code=3)) is False
    assert _bw_classify_is_eligible(_bw_classifier_obs(0, 0, disturbance_present=0)) is False
    # predicted >= usable -> CurrentHeadroom.
    assert (
        _bw_classify_is_eligible(_bw_classifier_obs(0, 0, predicted_torque_target_abs=500)) is False
    )
    # Not moving (forward settled interval no longer strictly positive) -> Inert.
    assert _bw_classify_is_eligible(_bw_classifier_obs(0, 0, settled_rate_low_q=0)) is False
    # Deficit unresolved (forward deficit low not > 0).
    assert _bw_classify_is_eligible(_bw_classifier_obs(0, 0, deficit_low_q=0)) is False
    # A reverse stroke with a mirrored eligible interval.
    assert _bw_classify_is_eligible(_bw_classifier_obs(1, 1)) is True


def test_bw_directional_included_mask_reproduces_depth_three_exclusion():
    # Four eligible, overlapping intervals -> all included.
    full = [_bw_classifier_obs(slot, 0) for slot in range(4)]
    assert _bw_directional_included_mask(full) == 0b1111

    # One stationarity-excluded stroke -> three eligible, the single eligible
    # triple forms one component, the three are included (the outlier is not).
    three_eligible = [
        _bw_classifier_obs(0, 0, stationarity_code=4),
        _bw_classifier_obs(1, 0),
        _bw_classifier_obs(2, 0),
        _bw_classifier_obs(3, 0),
    ]
    assert _bw_directional_included_mask(three_eligible) == 0b1110

    # Four eligible but slot 3's disturbance interval is disjoint from the other
    # three: the three overlapping slots form the single consensus component and
    # the disjoint outlier is excluded from the included set.
    outlier_interval = [
        _bw_classifier_obs(0, 0),
        _bw_classifier_obs(1, 0),
        _bw_classifier_obs(2, 0),
        _bw_classifier_obs(3, 0, disturbance_low_q16=5000, disturbance_high_q16=6000),
    ]
    assert _bw_directional_included_mask(outlier_interval) == 0b0111

    # Fewer than four collected slots -> IncompleteRung, nothing included.
    assert _bw_directional_included_mask(full[:3]) == 0

    # Fewer than three eligible -> InsufficientConsensus, nothing included.
    two_eligible = [
        _bw_classifier_obs(0, 0, stationarity_code=4),
        _bw_classifier_obs(1, 0, stationarity_code=4),
        _bw_classifier_obs(2, 0),
        _bw_classifier_obs(3, 0),
    ]
    assert _bw_directional_included_mask(two_eligible) == 0


def _bw_synthetic_atom(
    response_percent: float, relative_se: float, direction: int, target_rate_q: int = 1_000_000
) -> tuple[int, int, int]:
    """Construct a `(mean_rate_q, variance_of_mean_q2, target_rate_q)` raw
    atom that reproduces approximately the given response percent and
    relative standard error under `_bw_observation_statistic`.

    Test-only fixture helper: builders use this (together with the
    production `_bw_observation_interval_percent`/`_bw_percent_to_milli`
    functions under test) to generate self-consistent raw evidence and
    firmware-reported wire bytes together, the same way the rest of this
    builder derives expected digests by calling the production digest
    functions directly rather than hand-computing them.
    """
    magnitude = round(response_percent / 100.0 * target_rate_q)
    mean_rate_q = magnitude if direction == 0 else -magnitude
    variance_of_mean_q2 = round((relative_se * magnitude) ** 2)
    return mean_rate_q, variance_of_mean_q2, target_rate_q


class _BreakawayCampaignBuilder:
    """Assembles a self-consistent, digest-chained synthetic breakaway campaign.

    Every field defaults to a value independently reconstructible by the
    `breakaway_*` functions under test, so a fresh builder's `.ticks()` always
    passes `audit_breakaway_campaign` with `matches_firmware=True`; each test
    then mutates exactly the field(s) relevant to what it's checking.
    """

    def __init__(self) -> None:
        self.run_sequence = 7
        self.p_start_raw = 2
        self.p_top_raw = 1_088
        self.search_grid = breakaway_build_search_grid(self.p_start_raw, self.p_top_raw)
        self.motion_threshold_counts = 63
        self.max_capture_interval_us = 2_000
        self.max_observations = 128
        self.probe_digest = breakaway_probe_plan_digest(
            self.search_grid,
            self.motion_threshold_counts,
            self.max_capture_interval_us,
            self.max_observations,
            workflow_digest=_BREAKAWAY_WORKFLOW_DIGEST,
            target_velocity_rpm=_BREAKAWAY_TARGET_VELOCITY_RPM,
            hard_torque_limit=_BREAKAWAY_HARD_TORQUE_LIMIT,
            usable_torque_limit=_BREAKAWAY_USABLE_TORQUE_LIMIT,
            pwm_hz=_BREAKAWAY_PWM_HZ,
            encoder_counts_per_rev=_BREAKAWAY_ENCODER_COUNTS_PER_REV,
        )

        self.floor_raw = 181
        self.breakaway_raw = 256
        self.ceiling_raw = 1_088
        self.observations_per_direction = 4
        self.workflow_envelope_ms = 400_000
        self.ladder = breakaway_plan_additive_discovery_ladder(
            self.floor_raw,
            self.breakaway_raw,
            self.ceiling_raw,
            self.observations_per_direction,
            self.workflow_envelope_ms,
        )
        assert self.ladder is not None
        self.discovery_digest = breakaway_discovery_plan_digest(
            self.ladder["gains"],
            self.observations_per_direction,
            self.ladder["confidence_family_size"],
            self.probe_digest,
        )
        self.candidate_p_raw = self.ladder["gains"][
            self.ladder["gains"].index(self.breakaway_raw) + 1
        ]
        self.confirmation_digest = breakaway_confirmation_plan_digest(
            self.discovery_digest, self.candidate_p_raw, 0, 8, 4, 70, 80, 0
        )
        self.required_se = breakaway_required_relative_se_permille()

    def moving_index(self) -> int:
        return self.search_grid.index(self.breakaway_raw)

    def probe_strokes(self) -> int:
        return self.moving_index() * 2 + 4

    def probe_plan_tick(self, *, sequence: int = 1, plan_digest: int | None = None) -> dict:
        digest = self.probe_digest if plan_digest is None else plan_digest
        raw = _bw_tag(
            0,
            "IHQBBHHQiHHIIII",
            self.run_sequence,
            sequence,
            digest,
            self.max_observations,
            len(self.search_grid),
            self.p_start_raw,
            self.p_top_raw,
            _BREAKAWAY_WORKFLOW_DIGEST,
            _BREAKAWAY_TARGET_VELOCITY_RPM,
            _BREAKAWAY_HARD_TORQUE_LIMIT,
            _BREAKAWAY_USABLE_TORQUE_LIMIT,
            _BREAKAWAY_PWM_HZ,
            _BREAKAWAY_ENCODER_COUNTS_PER_REV,
            self.motion_threshold_counts,
            self.max_capture_interval_us,
        )
        return {
            "event_id": 152,
            "timestamp_ticks": 1_000,
            "sequence": sequence,
            "records": [_bw_record(raw)],
        }

    def _probe_trajectory(
        self,
        resolved_index: int | None = None,
        failed_repeat_rungs: set[int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Return `(rung, direction, slot, moved)` commits for the upward walk."""
        if resolved_index is None:
            resolved_index = self.moving_index()
        failed_repeat_rungs = failed_repeat_rungs or set()
        trajectory: list[tuple[int, int, int, int]] = []
        for rung_index in range(resolved_index):
            if rung_index in failed_repeat_rungs:
                trajectory.extend(
                    (
                        (rung_index, 0, 0, 1),
                        (rung_index, 1, 0, 1),
                        (rung_index, 0, 1, 0),
                        (rung_index, 1, 1, 1),
                    )
                )
            else:
                trajectory.extend(((rung_index, 0, 0, 0), (rung_index, 1, 0, 0)))
        if resolved_index < len(self.search_grid):
            trajectory.extend(
                (
                    (resolved_index, 0, 0, 1),
                    (resolved_index, 1, 0, 1),
                    (resolved_index, 0, 1, 1),
                    (resolved_index, 1, 1, 1),
                )
            )
        return trajectory

    def probe_stream_ticks(
        self,
        *,
        sequence: int = 2,
        plan_digest: int | None = None,
        resolved_index: int | None = None,
        failed_repeat_rungs: set[int] | None = None,
    ) -> list[dict]:
        """Build recovered probe raw/decision pairs for both directions."""
        digest = self.probe_digest if plan_digest is None else plan_digest
        ticks: list[dict] = []
        evidence_sequence = sequence
        if resolved_index is None:
            resolved_index = self.moving_index()
        for rung_index, direction, slot, moved in self._probe_trajectory(
            resolved_index, failed_repeat_rungs
        ):
            settled_low = 100 if moved and direction == 0 else -100
            settled_high = 200 if moved and direction == 0 else 100
            if moved and direction == 1:
                settled_low, settled_high = -200, -100
            minimum = -self.motion_threshold_counts if moved and direction == 1 else 0
            maximum = self.motion_threshold_counts if moved and direction == 0 else 0
            raw = _bw_raw_observation(
                self.run_sequence,
                evidence_sequence,
                digest,
                0,
                rung_index,
                slot,
                direction,
                150 if direction == 0 else -150,
                100,
                1_000,
                settled_rate_low_q=settled_low,
                settled_rate_high_q=settled_high,
                stroke_min_displacement_counts=minimum,
                stroke_max_displacement_counts=maximum,
            )
            ticks.append(
                {
                    "event_id": 160,
                    "timestamp_ticks": 1_200 + evidence_sequence,
                    "sequence": evidence_sequence,
                    "records": [_bw_record(raw)],
                }
            )
            evidence_sequence += 1
            decision = _bw_tag(
                13,
                "IHQBBB",
                self.run_sequence,
                evidence_sequence,
                digest,
                direction,
                rung_index,
                moved,
            )
            ticks.append(
                {
                    "event_id": 162,
                    "timestamp_ticks": 1_200 + evidence_sequence,
                    "sequence": evidence_sequence,
                    "records": [_bw_record(decision)],
                }
            )
            evidence_sequence += 1
        return ticks

    def probe_result_tick(
        self,
        *,
        sequence: int = 90,
        plan_digest: int | None = None,
        observations: int | None = None,
    ) -> dict:
        digest = self.probe_digest if plan_digest is None else plan_digest
        strokes = self.probe_strokes() if observations is None else observations
        result = _bw_tag(
            1,
            "IHQBHIB",
            self.run_sequence,
            sequence,
            digest,
            self.moving_index(),
            self.breakaway_raw,
            self.motion_threshold_counts,
            strokes,
        )
        return {
            "event_id": 153,
            "timestamp_ticks": 2_000,
            "sequence": sequence,
            "records": [_bw_record(result)],
        }

    def probe_terminal_tick(
        self,
        *,
        sequence: int = 4,
        terminal_cause: int = 22,
        has_safety_fault: int = 0,
        fault_checkpoint: int = 42,
    ) -> dict:
        raw = _bw_tag(
            2,
            "IHQBBB",
            self.run_sequence,
            sequence,
            self.probe_digest,
            terminal_cause,
            has_safety_fault,
            fault_checkpoint,
        )
        return {
            "event_id": 154,
            "timestamp_ticks": 2_000,
            "sequence": sequence,
            "records": [_bw_record(raw)],
        }

    def discovery_plan_tick(
        self,
        *,
        sequence: int = 100,
        plan_digest: int | None = None,
        prior_plan_digest: int | None = None,
    ) -> dict:
        digest = self.discovery_digest if plan_digest is None else plan_digest
        prior = self.probe_digest if prior_plan_digest is None else prior_plan_digest
        first_step = (
            self.ladder["gains"][self.ladder["gains"].index(self.breakaway_raw) + 1]
            - self.breakaway_raw
        )
        # Tag 5's per-rung `breakaway_ladder_rung` fan-out is gone: the plan's
        # own summary (floor/breakaway/ceiling/
        # observations_per_direction/maximum_workflow_ms) is the only wire
        # evidence for the ladder's shape now, plus `plan_digest` itself.
        # `audit_breakaway_campaign` regenerates `gains[]` from that summary.
        planned = _bw_tag(
            3,
            "IHQQHHBHHBHHII",
            self.run_sequence,
            sequence,
            digest,
            prior,
            _BW_STAGE_B_SCHEMA_REVISION,
            self.ladder["confidence_family_size"],
            self.ladder["rung_count"],
            self.observations_per_direction,
            self.floor_raw,
            1,
            self.breakaway_raw,
            self.ceiling_raw,
            first_step,
            self.workflow_envelope_ms,
        )
        ceiling_source = _bw_tag(4, "IHQB", self.run_sequence, sequence, digest, 0)
        return {
            "event_id": 155,
            "timestamp_ticks": 3_000,
            "sequence": sequence,
            "records": [
                _bw_record(planned),
                _bw_record(ceiling_source),
            ],
        }

    def discovery_ticks(
        self,
        *,
        sequence: int = 110,
        plan_digest: int | None = None,
        prior_plan_digest: int | None = None,
        terminal_cause: int = 0,
        margins: list[tuple[int, float, float]] | None = None,
        collected_count: int | None = None,
        omit_raw_evidence: bool = False,
    ) -> list[dict]:
        """Build the streamed discovery evidence ticks from `margins`.

        Returns the wire-order tick sequence a real capture produces: each raw
        observation on its own marker-160 tick, then each margin on its own
        marker-161 tick, then the discovery-terminal tick (marker 156). The
        per-observation and per-margin records no longer ride the terminal tick.

        `margins` is `(collected_index, response_percent, relative_se)`
        tuples in collection order. For each entry this generates a
        complete raw-observation octet (four synthetic forward and four
        synthetic reverse `BreakawayRawObservation`s, all sharing the same
        response/SE) and DERIVES the `BreakawayDiscoveryRungMargin`'s own
        `lower_percent_milli`/`upper_percent_milli` from that SAME raw
        evidence via the production `_bw_observation_interval_percent`/
        `_bw_percent_to_milli` functions (nomination's own parameters:
        multiplier m(8), `NOMINATION_SCREEN_C_SIGMA`, outward rounding) --
        never chosen independently of the raw evidence -- so a fresh
        builder's raw evidence and reported bounds always agree, exactly as
        `audit_breakaway_campaign`'s independent nomination recompute
        requires. `omit_raw_evidence=True` skips emitting the
        raw-observation records while still emitting the (self-consistently
        derived) margin records, for tests that want to see how the audit
        reacts to a margin record with no supporting raw evidence.

        A gap in `collected_index` (e.g. `[(0, ...), (1, ...), (3, ...)]`,
        skipping 2) is realistic firmware behavior, not a test-only shape:
        `rung_nomination_bounds` (band_confirmation.rs:306-325) can mark a
        genuinely in-band, genuinely collected rung `nominatable: false`, and
        `discovery_rung_margin_report` (breakaway_reporting.rs:416-417) then
        emits no wire record for it at all -- so `collected_count` (passed
        separately here; defaults to `len(margins)` when every collected
        rung was nominatable) can legitimately exceed the number of emitted
        margin records. `adjacent_rate_step_permille` mirrors firmware's own
        rule exactly (breakaway_reporting.rs:426-439): nonzero only when this
        record's `collected_index` is the IMMEDIATE successor of the
        previous *emitted* record's `collected_index`; a gap resets it to 0
        rather than stepping across the missing rung.
        """
        digest = self.discovery_digest if plan_digest is None else plan_digest
        prior = self.probe_digest if prior_plan_digest is None else prior_plan_digest
        if margins is None:
            margins = [(0, 75.0, 0.01), (1, 76.0, 0.012)]
        if collected_count is None:
            collected_count = len(margins)
        nomination_multiplier = _confidence_multiplier(8)
        # Each streamed record gets its own strictly increasing evidence_sequence
        # (as firmware's next_sequence assigns), and its marker's sequence
        # matches: raw observations first, then margins, then the terminal.
        raw_records: list[tuple[int, bytes]] = []
        margins_data: list[tuple[int, int, int, int, int, int, int]] = []
        streamed_seq = sequence
        previous_mid: int | None = None
        previous_collected_index: int | None = None
        for collected_index, response_percent, relative_se in margins:
            p_raw = self.ladder["gains"][
                collected_index + self.ladder["gains"].index(self.breakaway_raw)
            ]
            rung_index = self.ladder["gains"].index(p_raw)
            atoms: dict[int, tuple[int, int, int]] = {}
            for direction in (0, 1):
                atoms[direction] = _bw_synthetic_atom(response_percent, relative_se, direction)
                if not omit_raw_evidence:
                    mean_rate_q, variance_of_mean_q2, target_rate_q = atoms[direction]
                    for slot in range(4):
                        raw_records.append(
                            (
                                streamed_seq,
                                _bw_raw_observation(
                                    self.run_sequence,
                                    streamed_seq,
                                    digest,
                                    1,
                                    rung_index,
                                    slot,
                                    direction,
                                    mean_rate_q,
                                    variance_of_mean_q2,
                                    target_rate_q,
                                ),
                            )
                        )
                        streamed_seq += 1
            interval = _bw_observation_interval_percent(
                *atoms[0], 0, nomination_multiplier, _BW_NOMINATION_SCREEN_C_SIGMA, True
            )
            assert interval is not None
            lower = _bw_percent_to_milli(interval[0])
            upper = _bw_percent_to_milli(interval[1])
            margin, half_width = breakaway_discovery_rung_margin(lower, upper)
            mid = (lower + upper) // 2
            step = (
                mid - previous_mid
                if previous_collected_index is not None
                and collected_index == previous_collected_index + 1
                else 0
            )
            previous_mid = mid
            previous_collected_index = collected_index
            margins_data.append((collected_index, p_raw, lower, upper, margin, half_width, step))
        margin_records: list[tuple[int, bytes]] = []
        for collected_index, p_raw, lower, upper, margin, half_width, step in margins_data:
            margin_records.append(
                (
                    streamed_seq,
                    _bw_tag(
                        7,
                        "IHQBHiiiii",
                        self.run_sequence,
                        streamed_seq,
                        digest,
                        collected_index,
                        p_raw,
                        lower,
                        upper,
                        margin,
                        half_width,
                        step,
                    ),
                )
            )
            streamed_seq += 1
        terminal_seq = streamed_seq
        terminal = _bw_tag(
            6,
            "IHQQHBBB",
            self.run_sequence,
            terminal_seq,
            digest,
            prior,
            self.ladder["confidence_family_size"],
            terminal_cause,
            collected_count,
            0,
        )
        ticks: list[dict] = []
        for seq, raw in raw_records:
            ticks.append(
                {
                    "event_id": 160,
                    "timestamp_ticks": 3_500,
                    "sequence": seq,
                    "records": [_bw_record(raw)],
                }
            )
        for seq, margin in margin_records:
            ticks.append(
                {
                    "event_id": 161,
                    "timestamp_ticks": 3_800,
                    "sequence": seq,
                    "records": [_bw_record(margin)],
                }
            )
        ticks.append(
            {
                "event_id": 156,
                "timestamp_ticks": 4_000,
                "sequence": terminal_seq,
                "records": [_bw_record(terminal)],
            }
        )
        return ticks

    def confirmation_plan_tick(
        self,
        *,
        sequence: int = 400,
        plan_digest: int | None = None,
        prior_plan_digest: int | None = None,
        family_size: int = 8,
    ) -> dict:
        digest = self.confirmation_digest if plan_digest is None else plan_digest
        prior = self.discovery_digest if prior_plan_digest is None else prior_plan_digest
        raw = _bw_tag(
            9,
            "IHQQHHHBBBBi",
            self.run_sequence,
            sequence,
            digest,
            prior,
            family_size,
            self.observations_per_direction,
            self.candidate_p_raw,
            70,
            80,
            0,
            0,
            1_500,
        )
        return {
            "event_id": 157,
            "timestamp_ticks": 5_000,
            "sequence": sequence,
            "records": [_bw_record(raw)],
        }

    def confirmation_ticks(
        self,
        *,
        sequence: int = 410,
        plan_digest: int | None = None,
        prior_plan_digest: int | None = None,
        accepted: int = 1,
        terminal_cause: int = 21,
        masks: tuple[int, int, int, int, int, int] = (0b1111,) * 6,
        observation_count: int = 8,
        relative_ses: list[int] | None = None,
        response_percent: float = 75.0,
        direction_overrides: dict[int, int] | None = None,
        confirmed_p_raw_override: int | None = None,
        omit_raw_evidence: bool = False,
        unrepresentable_slots: set[int] | None = None,
    ) -> list[dict]:
        """Build the confirmation evidence ticks.

        `unrepresentable_slots` marks strokes whose mean is zero/wrong-signed:
        firmware still streams their (total) raw observation but
        confirmation_observation_report emits no reduced record for them, so
        those slots appear in the raw stream and the collected mask but not in the
        reduced observations.

        Returns the wire-order tick sequence: each held-out stroke's raw
        observation on its own marker-160 tick (phase == Confirmation), then the
        confirmation-terminal tick (marker 158) carrying the reported
        BandConfirmationObservation records and the terminal itself (these stay
        bundled, as firmware emits them, and eight-plus-one fits the queue).

        Each of `observation_count` held-out strokes' raw evidence (a
        synthetic atom at `response_percent`, with a relative SE close to
        `relative_ses[slot] / 1000`) is generated first; the
        `BandConfirmationObservation` wire record's own response bounds and
        SE are DERIVED from that same atom via the production
        `_bw_observation_interval_percent`/`_bw_percent_to_milli`/
        `_bw_relative_se_to_permille` functions (confirmation's own
        parameters: multiplier m(8), c_sigma = 1.0, outward rounding) --
        never chosen independently -- so a fresh builder's raw evidence and
        reported bounds always agree, exactly as `audit_breakaway_campaign`'s
        independent confirmation recompute requires.
        `response_percent=69.0` (or any value outside [70, 80]) reproduces
        the exact tamper a human reviewer demonstrated the OLD audit passed:
        an accepted confirmation whose (raw-evidence-recomputed) interval
        sits outside the mandatory band.

        `direction_overrides` maps a slot index to a direction code that
        deliberately breaks the fixed four-forward-then-four-reverse
        `CONFIRMATION_SLOT_ORDER` schedule -- both the raw observation and
        the reported wire record use the override, so the ONLY thing that
        disagrees with the frozen schedule is the schedule itself, not any
        other recomputed value. `confirmed_p_raw_override` lets a test claim
        a `confirmed_p_raw` different from the nominated `candidate_p_raw`
        without touching anything else. `omit_raw_evidence=True` skips
        emitting the raw-observation records while still emitting the
        (self-consistently derived) wire observation records.
        """
        digest = self.confirmation_digest if plan_digest is None else plan_digest
        prior = self.discovery_digest if prior_plan_digest is None else prior_plan_digest
        if relative_ses is None:
            relative_ses = [10, 12, 15, 9, 11, 13, 8, 14][:observation_count]
        direction_overrides = direction_overrides or {}
        confirmation_multiplier = _confidence_multiplier(8)
        unrepresentable_slots = unrepresentable_slots or set()
        observations = []
        raw_records: list[tuple[int, bytes]] = []
        actual_ses: list[int] = []
        raw_seq = sequence
        for slot in range(observation_count):
            direction = direction_overrides.get(slot, 0 if slot < 4 else 1)
            relative_se = relative_ses[slot] / 1000.0
            if slot in unrepresentable_slots:
                # A collected-but-unrepresentable stroke: zero mean. Its raw
                # observation still streams (total), but no reduced record.
                mean_rate_q, variance_of_mean_q2, target_rate_q = 0, 0, 1_000_000
            else:
                mean_rate_q, variance_of_mean_q2, target_rate_q = _bw_synthetic_atom(
                    response_percent, relative_se, direction
                )
            if not omit_raw_evidence:
                raw_records.append(
                    (
                        raw_seq,
                        _bw_raw_observation(
                            self.run_sequence,
                            raw_seq,
                            digest,
                            2,
                            0,
                            slot,
                            direction,
                            mean_rate_q,
                            variance_of_mean_q2,
                            target_rate_q,
                        ),
                    )
                )
                raw_seq += 1
            if slot in unrepresentable_slots:
                continue  # no reduced observation record for this stroke
            interval = _bw_observation_interval_percent(
                mean_rate_q,
                variance_of_mean_q2,
                target_rate_q,
                direction,
                confirmation_multiplier,
                1.0,
                True,
            )
            assert interval is not None
            lower = _bw_percent_to_milli(interval[0])
            upper = _bw_percent_to_milli(interval[1])
            # Recompute the reported SE from the actual (integer-rounded)
            # atom fields, not the idealized `relative_se` float used to
            # construct them, so this always matches what an independent
            # recompute from the wire's raw evidence produces bit-for-bit.
            statistic = _bw_observation_statistic(
                mean_rate_q, variance_of_mean_q2, target_rate_q, direction
            )
            assert statistic is not None
            se_permille = _bw_relative_se_to_permille(statistic[1])
            actual_ses.append(se_permille)
            observations.append(
                _bw_tag(
                    11,
                    "IHQBBiiH",
                    self.run_sequence,
                    sequence,
                    digest,
                    slot,
                    direction,
                    lower,
                    upper,
                    se_permille,
                )
            )
        max_se = max(actual_ses) if actual_ses else 0
        confirmed_p_raw = (
            confirmed_p_raw_override
            if confirmed_p_raw_override is not None
            else (self.candidate_p_raw if accepted else 0)
        )
        terminal = _bw_tag(
            10,
            "IHQQHBBBBBBBBHHHB",
            self.run_sequence,
            sequence,
            digest,
            prior,
            8,
            terminal_cause,
            *masks,
            accepted,
            confirmed_p_raw,
            max_se,
            self.required_se,
            0,
        )
        ticks: list[dict] = []
        for seq, raw in raw_records:
            ticks.append(
                {
                    "event_id": 160,
                    "timestamp_ticks": 5_800,
                    "sequence": seq,
                    "records": [_bw_record(raw)],
                }
            )
        ticks.append(
            {
                "event_id": 158,
                "timestamp_ticks": 6_000,
                "sequence": sequence,
                "records": [
                    *[_bw_record(r) for r in observations],
                    _bw_record(terminal),
                ],
            }
        )
        return ticks

    def campaign_terminal_tick(
        self,
        *,
        sequence: int = 500,
        timestamp_ticks: int = 7_000,
        phase: int = 2,
        terminal_cause: int = 21,
        accepted: int = 1,
        stage_c_plan_digest: int = 0xDEAD_BEEF,
    ) -> dict:
        raw = _bw_tag(
            12,
            "IHBBBQ",
            self.run_sequence,
            sequence,
            phase,
            terminal_cause,
            accepted,
            stage_c_plan_digest,
        )
        return {
            "event_id": 159,
            "timestamp_ticks": timestamp_ticks,
            "sequence": sequence,
            "records": [_bw_record(raw)],
        }

    def accepted_campaign_ticks(self) -> list[dict]:
        return [
            self.probe_plan_tick(),
            *self.probe_stream_ticks(),
            self.probe_result_tick(),
            self.discovery_plan_tick(),
            *self.discovery_ticks(),
            self.confirmation_plan_tick(),
            *self.confirmation_ticks(),
            self.campaign_terminal_tick(),
        ]


def test_breakaway_build_search_grid_matches_the_spec_representative_geometry() -> None:
    """Spec section 5's worked example: floor 181, breakaway 256, ceiling 1088."""
    grid = breakaway_build_search_grid(2, 1_088)
    assert 181 in grid
    assert 256 in grid
    assert grid.index(256) == grid.index(181) + 1
    assert grid[-1] == 1_088


def test_breakaway_realize_candidate_additive_ladder_matches_the_spec_worked_example() -> None:
    """Spec section 5's exact representative ladder (32 rungs, step 28)."""
    step, gains, first_additive_gain = breakaway_realize_candidate_additive_ladder(
        32, 181, 256, 1_088
    )
    assert step == 28
    assert len(gains) == 32
    assert gains[:4] == [181, 256, 284, 312]
    assert gains[-1] == 1_088
    assert first_additive_gain == 284


def test_breakaway_plan_additive_discovery_ladder_chooses_the_spec_worked_example() -> None:
    best = breakaway_plan_additive_discovery_ladder(181, 256, 1_088, 4, 400_000)
    assert best is not None
    assert best["rung_count"] == 32
    assert best["confidence_family_size"] == 128
    # Exact worked-example ladder from spec section 5.
    assert best["gains"] == [
        181,
        256,
        284,
        312,
        340,
        368,
        396,
        424,
        452,
        480,
        508,
        536,
        564,
        592,
        620,
        648,
        676,
        704,
        732,
        760,
        788,
        816,
        844,
        872,
        900,
        928,
        956,
        984,
        1_012,
        1_040,
        1_068,
        1_088,
    ]


def test_breakaway_plan_admits_the_measured_stepper_x_geometry() -> None:
    """Replay must share firmware's provisional 3/2 characterization guard."""
    best = breakaway_plan_additive_discovery_ladder(384, 543, 7_456, 4, 1_863_112)

    assert best is not None
    assert best["rung_count"] == 32
    assert best["gains"][:3] == [384, 543, 774]
    assert best["gains"][-1] == 7_456


def test_breakaway_replay_upward_probe_reconstructs_the_128_stroke_bound() -> None:
    builder = _BreakawayCampaignBuilder()
    builder.p_start_raw = 1
    builder.p_top_raw = 65_535
    builder.search_grid = breakaway_build_search_grid(builder.p_start_raw, builder.p_top_raw)
    assert len(builder.search_grid) == 32
    builder.probe_digest = breakaway_probe_plan_digest(
        builder.search_grid,
        builder.motion_threshold_counts,
        builder.max_capture_interval_us,
        builder.max_observations,
        workflow_digest=_BREAKAWAY_WORKFLOW_DIGEST,
        target_velocity_rpm=_BREAKAWAY_TARGET_VELOCITY_RPM,
        hard_torque_limit=_BREAKAWAY_HARD_TORQUE_LIMIT,
        usable_torque_limit=_BREAKAWAY_USABLE_TORQUE_LIMIT,
        pwm_hz=_BREAKAWAY_PWM_HZ,
        encoder_counts_per_rev=_BREAKAWAY_ENCODER_COUNTS_PER_REV,
    )
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(
            resolved_index=len(builder.search_grid),
            failed_repeat_rungs=set(range(len(builder.search_grid))),
        ),
    ]

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["resolved_rung_index"] is None
    assert replay["committed_observations"] == 128
    assert replay["expected_terminal"] == {"terminal_cause": 1}


def test_audit_breakaway_campaign_accepts_a_self_consistent_campaign() -> None:
    builder = _BreakawayCampaignBuilder()
    result = audit_breakaway_campaign(
        builder.accepted_campaign_ticks(), motion_timestamps=[1_500, 2_500], clock_hz=1_000_000
    )
    assert result["matches_firmware"] is True
    assert result["mismatches"] == []
    assert result["confirmation_accepted"] is True
    assert result["search_grid"][:4] == [2, 3, 4, 6]
    assert result["discovery_planned"]["rung_count"] == 32
    assert result["workflow_duration_ticks"] == 6_000
    assert result["workflow_duration_ms"] == pytest.approx(6.0)


def test_audit_breakaway_campaign_detects_a_tampered_probe_result() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    result_tick = next(tick for tick in ticks if tick["event_id"] == 153)
    result_tick["records"][0][1]["observation_count"] += 1

    result = audit_breakaway_campaign(ticks)

    assert result["matches_firmware"] is False
    assert any("probe result observation_count" in item for item in result["mismatches"])


def test_breakaway_replay_upward_probe_accepts_motion_at_the_threshold() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [builder.probe_plan_tick(), *builder.probe_stream_ticks(resolved_index=0)]

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["resolved_rung_index"] == 0
    assert replay["committed_observations"] == 4


def test_breakaway_replay_upward_probe_treats_threshold_minus_one_as_no_motion() -> None:
    builder = _BreakawayCampaignBuilder()
    stream = builder.probe_stream_ticks(resolved_index=len(builder.search_grid))
    stream[0]["records"][0][1]["stroke_max_displacement_counts"] = (
        builder.motion_threshold_counts - 1
    )

    replay = breakaway_replay_upward_probe(
        builder.search_grid, [builder.probe_plan_tick(), *stream]
    )

    assert replay["resolved_rung_index"] is None
    assert replay["expected_terminal"] == {"terminal_cause": 1}


def test_breakaway_replay_upward_probe_ignores_stationarity_for_motion() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [builder.probe_plan_tick(), *builder.probe_stream_ticks(resolved_index=0)]
    for tick in ticks:
        if tick["event_id"] != 160:
            continue
        observation = tick["records"][0][1]
        observation["stationarity_code"] = 6
        observation["settled_rate_low_q"] = 0
        observation["settled_rate_high_q"] = 0
        observation["post_fit_lag_one_state"] = 0

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["resolved_rung_index"] == 0
    assert replay["observations"][0]["stationarity_diagnostic"] == {
        "stationarity_code": 6,
        "settled_rate_low_q": 0,
        "settled_rate_high_q": 0,
        "drift_ratio_ppm": 0,
        "post_fit_lag_one_state": 0,
    }


def test_breakaway_replay_upward_probe_prefers_commanded_motion_when_both_sides_move() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [builder.probe_plan_tick(), *builder.probe_stream_ticks(resolved_index=0)]
    first = ticks[1]["records"][0][1]
    first["stroke_min_displacement_counts"] = -builder.motion_threshold_counts

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["resolved_rung_index"] == 0
    assert replay["observations"][0]["commanded_excursion_counts"] == 63
    assert replay["observations"][0]["opposite_excursion_counts"] == 63


def test_breakaway_replay_upward_probe_advances_after_a_failed_repeat() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(resolved_index=1, failed_repeat_rungs={0}),
    ]

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["resolved_rung_index"] == 1
    assert replay["committed_observations"] == 8


@pytest.mark.parametrize(("interval", "terminal"), [(0, 22), (2_000, None), (2_001, 22)])
def test_breakaway_replay_upward_probe_applies_the_capture_interval_gate(
    interval: int, terminal: int | None
) -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    raw_tick["records"][0][1]["capture_max_interval_us"] = interval
    ticks = [
        builder.probe_plan_tick(),
        raw_tick,
    ]
    if terminal is None:
        ticks.append(builder.probe_stream_ticks()[1])

    replay = breakaway_replay_upward_probe(builder.search_grid, ticks)

    assert replay["expected_terminal"] == (
        {"terminal_cause": terminal} if terminal is not None else None
    )
    assert replay["committed_observations"] == (1 if terminal is None else 0)


def test_audit_breakaway_pairs_interval_refusal_with_cause_22() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    raw_tick["records"][0][1]["capture_max_interval_us"] = 0
    ticks = [
        builder.probe_plan_tick(),
        raw_tick,
        builder.probe_terminal_tick(sequence=3),
        builder.campaign_terminal_tick(
            sequence=4,
            phase=0,
            terminal_cause=22,
            accepted=0,
            stage_c_plan_digest=0,
        ),
    ]

    result = audit_breakaway_campaign(ticks)

    assert result["matches_firmware"] is True, result["mismatches"]
    assert result["probe_replay"]["expected_terminal"] == {"terminal_cause": 22}


def test_breakaway_replay_upward_probe_forbids_a_decision_after_interval_refusal() -> None:
    builder = _BreakawayCampaignBuilder()
    stream = builder.probe_stream_ticks()[:2]
    stream[0]["records"][0][1]["capture_max_interval_us"] = 0

    with pytest.raises(ValueError, match="independently terminates"):
        breakaway_replay_upward_probe(builder.search_grid, [builder.probe_plan_tick(), *stream])


def test_breakaway_replay_upward_probe_rejects_wrong_direction_excursion() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    raw_tick["records"][0][1]["stroke_min_displacement_counts"] = -builder.motion_threshold_counts

    replay = breakaway_replay_upward_probe(
        builder.search_grid, [builder.probe_plan_tick(), raw_tick]
    )

    assert replay["expected_terminal"] == {"terminal_cause": 6}


def test_breakaway_replay_upward_probe_keeps_recovery_interruption_uncommitted() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]

    replay = breakaway_replay_upward_probe(
        builder.search_grid, [builder.probe_plan_tick(), raw_tick]
    )

    assert replay["committed_observations"] == 0
    assert replay["expected_terminal"] is None
    assert len(replay["uncommitted_raw"]) == 1


def test_audit_breakaway_accepts_a_recovery_fault_without_advancing_the_probe() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    ticks = [
        builder.probe_plan_tick(),
        raw_tick,
        builder.probe_terminal_tick(
            sequence=3, terminal_cause=6, has_safety_fault=1, fault_checkpoint=17
        ),
        builder.campaign_terminal_tick(
            sequence=4,
            phase=0,
            terminal_cause=6,
            accepted=0,
            stage_c_plan_digest=0,
        ),
    ]

    result = audit_breakaway_campaign(ticks)

    assert result["matches_firmware"] is True, result["mismatches"]
    assert result["probe_replay"]["committed_observations"] == 0


def test_breakaway_replay_upward_probe_rejects_malformed_extrema() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    raw_tick["records"][0][1]["stroke_min_displacement_counts"] = 1

    with pytest.raises(ValueError, match="zero baseline"):
        breakaway_replay_upward_probe(builder.search_grid, [builder.probe_plan_tick(), raw_tick])


def test_breakaway_replay_upward_probe_rejects_nonzero_extrema_without_an_interval() -> None:
    builder = _BreakawayCampaignBuilder()
    raw_tick = builder.probe_stream_ticks()[0]
    observation = raw_tick["records"][0][1]
    observation["capture_max_interval_us"] = 0
    observation["stroke_max_displacement_counts"] = 1

    with pytest.raises(ValueError, match="no measured interval"):
        breakaway_replay_upward_probe(builder.search_grid, [builder.probe_plan_tick(), raw_tick])


def test_breakaway_replay_upward_probe_rejects_evidence_after_resolution() -> None:
    builder = _BreakawayCampaignBuilder()
    stream = builder.probe_stream_ticks(resolved_index=0)
    stream.extend(builder.probe_stream_ticks(resolved_index=0)[:2])

    with pytest.raises(ValueError, match="became terminal"):
        breakaway_replay_upward_probe(builder.search_grid, [builder.probe_plan_tick(), *stream])


def test_audit_breakaway_rejects_probe_decision_sequence_disagreement() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    raw_tick = next(tick for tick in ticks if tick["event_id"] == 160 and tick["sequence"] == 2)
    raw_tick["sequence"] = 1
    raw_tick["records"][0][1]["evidence_sequence"] = 1

    with pytest.raises(ValueError, match="does not follow its raw observation"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_probe_decision_digest_disagreement() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    decision_tick = next(tick for tick in ticks if tick["event_id"] == 162)
    decision_tick["records"][0][1]["plan_digest"] ^= 0xFF

    with pytest.raises(ValueError, match="does not chain to its owning plan"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_probe_decision_disposition_disagreement() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    decision_tick = next(tick for tick in ticks if tick["event_id"] == 162)
    decision_tick["records"][0][1]["disposition"] ^= 1

    with pytest.raises(ValueError, match="disposition disagrees"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_probe_evidence_after_its_result() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    result_index = next(index for index, tick in enumerate(ticks) if tick["event_id"] == 153)
    ticks[result_index + 1 : result_index + 1] = builder.probe_stream_ticks(sequence=80)[:2]

    with pytest.raises(ValueError, match="continues after"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_derives_only_observed_post_breakaway_transients() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    discovery = [
        tick["records"][0][1]
        for tick in ticks
        if tick["event_id"] == 160 and tick["records"][0][1]["phase"] == 1
    ]
    higher_rung = max(int(observation["rung_index"]) for observation in discovery)
    transient = next(
        observation
        for observation in discovery
        if observation["rung_index"] == higher_rung
        and observation["direction"] == 0
        and observation["slot_index"] == 0
    )
    transient["stationarity_code"] = 6
    transient["post_fit_lag_one_state"] = 2

    result = audit_breakaway_campaign(ticks)

    assert result["post_breakaway_transient"] == [
        {
            "direction": 0,
            "consensus_rung_index": higher_rung - 1,
            "consensus_p_raw": builder.ladder["gains"][higher_rung - 1],
            "transient_rung_index": higher_rung,
            "transient_p_raw": builder.ladder["gains"][higher_rung],
            "slot_index": 0,
        }
    ]


def test_audit_breakaway_campaign_accepts_a_non_nominatable_rung_leaving_a_collected_index_gap() -> (  # noqa: E501
    None
):
    """A genuinely in-band rung that isn't nominatable emits no wire record at
    all (`discovery_rung_margin_report`, breakaway_reporting.rs:416-417), so
    `collected_index` can skip a value while `collected_count`
    (breakaway_reporting.rs:521) still counts it. That is real firmware
    behavior, not a corrupt capture, and must not be reported as a mismatch.

    Four rungs were collected in-band (collected_count=4); rung at
    collected_index 2 was in-band but not nominatable and so is absent from
    the wire. The record at collected_index 3 must show `adjacent_rate_step_
    permille == 0`, because its immediate predecessor position (2) is the
    missing one -- not a step measured against collected_index 1.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(
            margins=[
                (0, 71_000, 76_000),
                (1, 74_000, 79_000),
                (3, 73_000, 78_000),
            ],
            collected_count=4,
        ),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is True, result["mismatches"]
    margin_records = [
        decoded
        for tick in ticks
        if tick["event_id"] == 161
        for _raw, decoded in tick["records"]
        if decoded["record_kind"] == "breakaway_discovery_rung_margin"
    ]
    by_collected_index = {int(record["collected_index"]): record for record in margin_records}
    assert set(by_collected_index) == {0, 1, 3}
    assert by_collected_index[3]["adjacent_rate_step_permille"] == 0


def _bw_raw_observation_tick(
    builder: "_BreakawayCampaignBuilder", phase: int, *, sequence: int = 99
) -> dict:
    """A single streamed raw-observation marker-160 tick for `phase` (1 =
    discovery, 2 = confirmation)."""
    atom = _bw_synthetic_atom(75.0, 0.01, 0)
    digest = builder.discovery_digest if phase == 1 else builder.confirmation_digest
    raw = _bw_raw_observation(builder.run_sequence, sequence, digest, phase, 0, 0, 0, *atom)
    return {
        "event_id": 160,
        "timestamp_ticks": 5_500,
        "sequence": sequence,
        "records": [_bw_record(raw)],
    }


def test_audit_breakaway_flags_a_confirmation_collected_mask_that_omits_received_slots() -> None:
    """A non-accepted terminal with three forward strokes but zeroed collected
    masks used to pass as fully consistent."""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(
            accepted=0,
            terminal_cause=15,
            observation_count=3,
            masks=(0, 0, 0, 0, 0, 0),
        ),
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, stage_c_plan_digest=0),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("forward_collected_mask" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_collected_mask_counts_a_stroke_with_no_reduced_record() -> None:
    """A collected inert-zero stroke streams a raw observation but no reduced
    record. Reconstructing the collected mask from the total raw stream keeps it
    consistent; reconstructing from the reduced records would falsely flag the
    slot as absent."""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(
            accepted=0,
            terminal_cause=15,
            observation_count=4,
            unrepresentable_slots={0},
            # All four forward strokes collected; slot 0's zero mean is not
            # eligible, and there is no reverse block.
            masks=(0b1111, 0b1110, 0, 0, 0, 0),
        ),
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, stage_c_plan_digest=0),
    ]
    result = audit_breakaway_campaign(ticks)
    assert not any("collected_mask" in mismatch for mismatch in result["mismatches"]), result[
        "mismatches"
    ]


def test_audit_breakaway_rejects_a_ninth_duplicate_confirmation_stroke() -> None:
    """A ninth confirmation raw observation with a fresh, internally valid
    sequence but a duplicate slot violates the fixed eight-stroke schedule. It
    must be rejected, not hidden by the slot-keyed mask OR and per-slot dict."""
    builder = _BreakawayCampaignBuilder()
    confirmation = builder.confirmation_ticks()  # eight raw ticks (160..167) + terminal
    atom = _bw_synthetic_atom(75.0, 0.01, 0)
    ninth = _bw_raw_observation(
        builder.run_sequence, 418, builder.confirmation_digest, 2, 0, 0, 0, *atom
    )
    ninth_tick = {
        "event_id": 160,
        "timestamp_ticks": 5_800,
        "sequence": 418,
        "records": [_bw_record(ninth)],
    }
    # Splice the ninth raw observation in before the confirmation terminal.
    confirmation = confirmation[:-1] + [ninth_tick] + confirmation[-1:]
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *confirmation,
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="fixed .*schedule"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_a_malformed_confirmation_aborted_before_its_terminal() -> None:
    """A safety fault aborts confirmation after streaming a malformed prefix
    (slots 0, 0) and emits only the campaign closure -- never a confirmation
    terminal. The schedule validator must still fire on the raw stream, so it
    cannot be gated on the terminal's presence."""
    builder = _BreakawayCampaignBuilder()
    atom = _bw_synthetic_atom(75.0, 0.01, 0)
    raw_ticks = [
        {
            "event_id": 160,
            "timestamp_ticks": 5_800 + position,
            "sequence": 410 + position,
            "records": [
                _bw_record(
                    _bw_raw_observation(
                        builder.run_sequence,
                        410 + position,
                        builder.confirmation_digest,
                        2,
                        0,
                        0,  # both strokes claim slot 0
                        0,
                        *atom,
                    )
                )
            ],
        }
        for position in range(2)
    ]
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *raw_ticks,
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, stage_c_plan_digest=0),
    ]
    with pytest.raises(ValueError, match="fixed .*schedule"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_accepts_a_partial_confirmation_with_honest_collected_masks() -> None:
    """The same partial block is consistent when the collected masks honestly
    report the three received forward slots and no reverse slots."""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(
            accepted=0,
            terminal_cause=15,
            observation_count=3,
            masks=(0b0111, 0b0111, 0b0111, 0, 0, 0),
        ),
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, stage_c_plan_digest=0),
    ]
    result = audit_breakaway_campaign(ticks)
    assert not any("collected_mask" in mismatch for mismatch in result["mismatches"]), result[
        "mismatches"
    ]


def _bw_streamed_identity_campaign() -> list[dict]:
    builder = _BreakawayCampaignBuilder()
    return [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]


def _bw_first_marker_index(ticks: list[dict], event_id: int) -> int:
    return next(index for index, tick in enumerate(ticks) if tick["event_id"] == event_id)


def test_audit_breakaway_binds_a_streamed_margin_to_its_plan_digest() -> None:
    ticks = _bw_streamed_identity_campaign()
    index = _bw_first_marker_index(ticks, 161)
    ticks[index]["records"][0][1]["plan_digest"] ^= 0xFF
    with pytest.raises(ValueError, match="does not chain to its owning plan"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_binds_a_streamed_margin_to_the_campaign_run() -> None:
    ticks = _bw_streamed_identity_campaign()
    index = _bw_first_marker_index(ticks, 161)
    ticks[index]["records"][0][1]["run_sequence"] += 1
    with pytest.raises(ValueError, match="run_sequence does not match"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_binds_a_streamed_record_sequence_to_its_marker() -> None:
    ticks = _bw_streamed_identity_campaign()
    index = _bw_first_marker_index(ticks, 160)
    # Mutate only the record's evidence_sequence, leaving its marker's.
    ticks[index]["records"][0][1]["evidence_sequence"] += 1000
    with pytest.raises(ValueError, match="does not match its marker"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_a_second_record_in_a_streamed_marker() -> None:
    ticks = _bw_streamed_identity_campaign()
    index = _bw_first_marker_index(ticks, 161)
    ticks[index]["records"] = ticks[index]["records"] * 2
    with pytest.raises(ValueError, match="exactly one record"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_rejects_out_of_order_streamed_sequences() -> None:
    ticks = _bw_streamed_identity_campaign()
    # Reorder the first two raw-observation marker ticks so their sequences
    # descend in stream order.
    first = _bw_first_marker_index(ticks, 160)
    ticks[first], ticks[first + 1] = ticks[first + 1], ticks[first]
    with pytest.raises(ValueError, match="not strictly increasing"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_ranks_a_raw_observation_marker_by_its_phase() -> None:
    """The shared raw-observation marker (160) is ranked from its record's phase,
    not a fixed table entry: a discovery-phase stroke arriving after the
    confirmation plan is a phase reordering. (A confirmation-phase stroke there is
    normal, as the accepted-campaign fixtures already exercise.)"""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        _bw_raw_observation_tick(builder, 1),  # a discovery stroke, out of order
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="phases are reordered"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_flags_a_workflow_duration_over_its_reserved_budget() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        # 500,000,000 ticks at 1 MHz is 500 s, past the 400 s reserved envelope.
        builder.campaign_terminal_tick(timestamp_ticks=500_001_000),
    ]
    result = audit_breakaway_campaign(ticks, clock_hz=1_000_000)
    assert result["matches_firmware"] is False
    assert any(
        "exceeds the reserved maximum_workflow_ms" in mismatch for mismatch in result["mismatches"]
    )


def test_audit_breakaway_campaign_rejects_reordered_phases() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.discovery_plan_tick(),
        builder.probe_result_tick(),  # probe result after discovery: reordered
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="reordered"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_rejects_motion_before_a_plan() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = builder.accepted_campaign_ticks()
    with pytest.raises(ValueError, match="motion"):
        audit_breakaway_campaign(ticks, motion_timestamps=[500])


def test_audit_breakaway_campaign_rejects_a_digest_chain_break() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(prior_plan_digest=0xBAD_D1CE57),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="chain"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_rejects_a_family_four_one_rung_confirmation() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(family_size=4),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="family_size"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_rejects_an_accepted_confirmation_with_incomplete_masks() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(masks=(0b0111, 0b1111, 0b1111, 0b1111, 0b1111, 0b1111)),
        builder.campaign_terminal_tick(),
    ]
    with pytest.raises(ValueError, match="incomplete"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_accepts_a_confirmation_whose_disturbance_consensus_split() -> (
    None
):
    """A split depth-three consensus is diagnostic, not disqualifying.

    Reproduces the stepper_x capture of 2026-08-03 at 95% current headroom:
    both directions reported complete collected and eligible masks and all
    eight response intervals inside 70-80%, but the depth-three consensus over
    the DISTURBANCE intervals split into two components, so `included_mask` was
    0. That is a different measured quantity than the proportional response
    under test, so the audit must not refuse the accepted path over it.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(masks=(0b1111, 0b1111, 0, 0b1111, 0b1111, 0)),
        builder.campaign_terminal_tick(),
    ]

    result = audit_breakaway_campaign(ticks)

    assert result["confirmation_accepted"] is True
    assert result["matches_firmware"] is True, result["mismatches"]


def test_audit_breakaway_campaign_rejects_an_accepted_confirmation_with_a_raw_interval_outside_the_band() -> (  # noqa: E501
    None
):
    """Acceptance test for the confirmation raw-evidence recompute.

    A human reviewer demonstrated that the audit, before this fix, certified
    an ACCEPTED confirmation whose interval was mutated to 69% -- outside the
    mandatory [70, 80] band -- because nothing independently recomputed each
    held-out stroke's response interval from raw evidence; the audit only
    checked that firmware's OWN reported numbers were mutually consistent
    (complete masks, an aggregate max-of-reported-SEs). This fixture
    reproduces that exact scenario: every held-out stroke's RAW evidence
    (not just its wire-reported bounds) genuinely settles at 69%, and the
    confirmation terminal still reports `accepted=1`. It must now be
    rejected.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(response_percent=69.0),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("not contained in [70, 80]" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_campaign_rejects_a_confirmed_p_that_does_not_match_the_candidate() -> None:
    """A confirmed gain that silently diverges from the nominated candidate.

    Must be rejected even when every interval is honestly contained in
    [70, 80] and every mask is complete -- the OLD audit never compared
    `confirmation_planned.candidate_p_raw` against
    `confirmation_terminal.confirmed_p_raw` at all.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(confirmed_p_raw_override=builder.candidate_p_raw + 1),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("confirmed_p_raw" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_campaign_rejects_a_wrong_confirmation_slot_schedule() -> None:
    """The eight held-out strokes must follow the fixed schedule.

    Four forward observations then four reverse (`CONFIRMATION_SLOT_ORDER`).
    Swapping slot 0's direction to reverse (both in the raw evidence and the
    reported wire record, so every value-agreement check still passes) must
    still be caught and rejected: the schedule itself is part of the
    contract, not just a byproduct of the plan digest.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(direction_overrides={0: 1}),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("CONFIRMATION_SLOT_ORDER" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_campaign_rejects_an_accepted_confirmation_missing_raw_evidence() -> None:
    """An accepted confirmation with no raw evidence at all.

    Cannot be independently verified and must not be trusted by default --
    the whole point of FIX F is that an accepted flag alone is never
    sufficient.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(omit_raw_evidence=True),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("no raw evidence" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_campaign_rejects_a_second_candidate_after_failure() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(sequence=400),
        *builder.confirmation_ticks(sequence=410, accepted=0, terminal_cause=15),
        builder.confirmation_plan_tick(sequence=500),
        *builder.confirmation_ticks(sequence=510, accepted=0, terminal_cause=15),
        builder.campaign_terminal_tick(
            sequence=600, accepted=0, terminal_cause=15, stage_c_plan_digest=0
        ),
    ]
    with pytest.raises(ValueError, match="second confirmation candidate"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_rejects_stage_c_after_unsuccessful_confirmation() -> None:
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(accepted=0, terminal_cause=15),
        # Campaign terminal falsely claims accept despite the failed confirmation.
        builder.campaign_terminal_tick(accepted=1, terminal_cause=21),
    ]
    with pytest.raises(ValueError, match="does not trace back to an accepted confirmation"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_flags_a_wrong_reported_digest_as_a_mismatch_not_a_raise() -> None:
    """A single wrong-but-well-formed digest is a firmware-math mismatch, not a refusal."""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(plan_digest=builder.probe_digest ^ 0xFF),
        *builder.probe_stream_ticks(plan_digest=builder.probe_digest ^ 0xFF),
        builder.probe_result_tick(plan_digest=builder.probe_digest ^ 0xFF),
        builder.discovery_plan_tick(prior_plan_digest=builder.probe_digest ^ 0xFF),
        *builder.discovery_ticks(prior_plan_digest=builder.probe_digest ^ 0xFF),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(),
        builder.campaign_terminal_tick(),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["matches_firmware"] is False
    assert any("probe plan_digest" in mismatch for mismatch in result["mismatches"])


def test_audit_breakaway_campaign_reports_additive_ladder_resolution_budget_exceeded_as_zero_plan() -> (  # noqa: E501
    None
):
    """`AdditiveLadderResolutionBudgetExceeded` is a planning refusal before motion.

    It has no discovery plan at all -- distinct from `TargetBandSkipped`,
    which is a genuine discovery result and requires a real discovery
    terminal.
    """
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.campaign_terminal_tick(
            phase=1, terminal_cause=10, accepted=0, stage_c_plan_digest=0
        ),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["additive_ladder_resolution_budget_exceeded"] is True
    assert result["discovery_target_band_skipped"] is False
    assert result["discovery_planned"] is None


def test_audit_breakaway_campaign_reports_target_band_skipped_from_a_real_discovery_terminal() -> (
    None
):
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(terminal_cause=7, margins=[]),
        builder.campaign_terminal_tick(
            phase=1, terminal_cause=7, accepted=0, stage_c_plan_digest=0
        ),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["discovery_target_band_skipped"] is True
    assert result["additive_ladder_resolution_budget_exceeded"] is False
    assert result["discovery_planned"] is not None
