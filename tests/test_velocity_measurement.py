import json
import struct
from pathlib import Path

import pytest

from foci_trace.protocol import decode_breakaway_record
from foci_trace.velocity_measurement import (
    _BW_NOMINATION_SCREEN_C_SIGMA,
    _BW_PROPORTIONAL_SCHEMA_REVISION,
    VelocityObservation,
    _bw_classify_is_eligible,
    _bw_classify_is_inert,
    _bw_directional_included_mask,
    _bw_discovery_inert_fraction,
    _bw_observation_interval_percent,
    _bw_observation_statistic,
    _bw_percent_to_milli,
    _bw_relative_se_to_permille,
    _confidence_multiplier,
    analyze_velocity_observations,
    audit_breakaway_campaign,
    audit_rung_origin_recovery,
    audit_scored_rest_grammar,
    breakaway_build_search_grid,
    breakaway_confirmation_plan_digest,
    breakaway_discovery_plan_digest,
    breakaway_discovery_rung_margin,
    breakaway_plan_additive_discovery_ladder,
    breakaway_probe_plan_digest,
    breakaway_realize_candidate_additive_ladder,
    breakaway_replay_upward_probe,
    breakaway_required_relative_se_permille,
    plan_rung_origin_recovery,
    recovery_return_distance_counts,
    recovery_runtime_us,
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


def test_post_fit_exclusion_reports_the_statistic_that_failed() -> None:
    """A refusal is the one case that most needs its evidence.

    When `_classify_stationarity` rejects a fit, the fit itself succeeded and
    every statistic behind the verdict is in hand. Reporting only the code
    leaves an excluded observation saying that it failed but never how, which is
    exactly backwards for diagnosis.
    """
    from foci_trace.velocity_measurement import _stationary_suffix

    # A steady ramp: the fit succeeds and the classifier rejects it, which is
    # the post-fit path (the plateau and fit paths below have no statistics).
    result = _stationary_suffix([_observation(index) for index in range(64)])

    assert result["usable"] == 0
    detail = result["offline_exclusion_detail"]
    for field in (
        "lag_one_q",
        "lag_one_half_width_q",
        "slope_q",
        "slope_half_width_q",
        "residual_magnitude_q2",
    ):
        assert field in detail, f"post-fit exclusion is missing {field}"
    # Which gate actually failed, and which window bound: half the question
    # when several suffixes are tested.
    assert detail["suffix_len"] == 64
    assert detail["slope_passed"] is False or detail["lag_one_passed"] is False
    # The rate the refusal cost. Both were computed before the classification
    # rejected the fit, so asking what admitting it would have meant is a
    # question about this observation rather than a fabricated value.
    assert detail["would_be_rate_mean_q"] != 0
    assert detail["would_be_rate_half_width_q"] > 0

    # Firmware zeroes these on its own excluded records, so they must stay out
    # of the compared surface or the replay disagrees with the device it exists
    # to verify. Keeping them nested is what makes that structural.
    for field in ("lag_one_q", "slope_q", "suffix_len"):
        assert field not in result, f"{field} must not join the compared fields"


def test_pre_fit_exclusions_report_no_statistics_they_never_computed() -> None:
    """The other two refusal paths must stay bare.

    No plateau means no regression was attempted, so there is no slope, no
    lag-one and no residual. Emitting zeros there would read as a measurement of
    zero structure rather than an absence of evidence, which is the same trap
    that already produced one false reading off this data.
    """
    from foci_trace.velocity_measurement import NO_REPORTABLE_DRIFT_RATIO_PPM, _stationary_suffix

    # Too few samples for any declared suffix: nothing is ever fitted.
    result = _stationary_suffix([_observation(index) for index in range(8)])

    assert result["usable"] == 0
    assert result["drift_ratio_ppm"] == NO_REPORTABLE_DRIFT_RATIO_PPM
    assert "offline_exclusion_detail" not in result


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

    The drift bound only applies when build_features declares it (see
    test_moving_gate_keys_on_build_features below for the legacy side of that
    split), so this test must pass a build_features with the drift-bound bit set
    to exercise it at all.
    """
    from foci_trace.velocity_measurement import (
        BUILD_FEATURE_DRIFT_BOUND_GATE,
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
            build_features=BUILD_FEATURE_DRIFT_BOUND_GATE,
        )
        is None
    )


def test_moving_mode_rejects_excessive_drift_outside_the_inert_interval() -> None:
    from foci_trace.velocity_measurement import (
        BUILD_FEATURE_DRIFT_BOUND_GATE,
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
            build_features=BUILD_FEATURE_DRIFT_BOUND_GATE,
        )
        == 5
    )


def test_moving_gate_keys_on_build_features() -> None:
    """The moving gate keys on build_features, not schema revision. A capture
    whose build_features omits the drift-bound bit -- or offers no signal at all
    (None) -- falls back to the significance test, matching firmware that never
    ran the drift bound; a capture that declares the bit takes the drift bound
    regardless of schema. This is what lets a schema-13 combined capture from
    drift-bound firmware be gated by the rule it ran, not by its schema number."""
    from foci_trace.velocity_measurement import (
        BUILD_FEATURE_DRIFT_BOUND_GATE,
        _classify_stationarity,
    )

    # A huge drift ratio would reject under the drift bound, but a passing
    # significance test must accept it when the drift bit is absent or unknown.
    passing_fit = {
        "drift_ratio_ppm": 999_999_999,
        "slope_passed": True,
        "lag_one_passed": True,
    }
    assert (
        _classify_stationarity(
            passing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, build_features=0
        )
        is None
    )
    assert (
        _classify_stationarity(
            passing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, build_features=None
        )
        is None
    )

    # A tiny drift ratio would pass under the drift bound, but a failing
    # significance test must still reject it when the drift bit is absent.
    failing_fit = {
        "drift_ratio_ppm": 1,
        "slope_passed": False,
        "lag_one_passed": True,
    }
    assert (
        _classify_stationarity(
            failing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, build_features=0
        )
        == 2
    )
    assert (
        _classify_stationarity(
            failing_fit, "moving", mean_rate_q=100, half_width_rate_q=10, build_features=None
        )
        == 2
    )

    # With the drift bit declared the drift bound applies regardless of schema:
    # the huge-drift fit rejects as ExcessiveDrift (5) and the tiny-drift fit,
    # whose failing significance test the drift bound ignores, now passes.
    assert (
        _classify_stationarity(
            passing_fit,
            "moving",
            mean_rate_q=100,
            half_width_rate_q=10,
            build_features=BUILD_FEATURE_DRIFT_BOUND_GATE,
        )
        == 5
    )
    assert (
        _classify_stationarity(
            failing_fit,
            "moving",
            mean_rate_q=100,
            half_width_rate_q=10,
            build_features=BUILD_FEATURE_DRIFT_BOUND_GATE,
        )
        is None
    )


def test_capture_rule_features_prefers_explicit_bits_then_reconstructs() -> None:
    """The stationarity gates read their contract from build_features when the
    capture states it, and reconstruct only the rule bits from the stage's
    schema thresholds otherwise. Unlike capture_build_features it never consults
    build_revision, so an archived capture with no integral-representation signal
    still gates under its schema instead of raising."""
    from foci_trace.velocity_measurement import (
        BUILD_FEATURE_DRIFT_BOUND_GATE,
        BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE,
        BUILD_FEATURE_SUFFIX_WALK_CONTINUES,
        INTEGRAL_SUFFIX_WALK_MIN_SCHEMA_REVISION,
        capture_rule_features,
    )

    # A newer capture states build_features directly, returned verbatim
    # at any schema revision -- this is the case schema gating got wrong.
    assert capture_rule_features({"build_features": 0xF, "schema_revision": 13}) == 0xF

    # A legacy Integral capture reconstructs from its own plan schema_revision. At
    # the suffix-walk floor the drift and suffix bits are set but rest-walk (16)
    # is not, and no build_revision is needed to say so.
    features = capture_rule_features(
        {"schema_revision": INTEGRAL_SUFFIX_WALK_MIN_SCHEMA_REVISION},
    )
    assert features & BUILD_FEATURE_DRIFT_BOUND_GATE
    assert features & BUILD_FEATURE_SUFFIX_WALK_CONTINUES
    assert not features & BUILD_FEATURE_REST_CHAIN_ON_EVIDENCE

    # No signal at all -- no build_features, no schema -- means the legacy gate,
    # reported as None, never a raise.
    assert capture_rule_features({}) is None


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


def test_scored_rest_verification_recomputes_under_the_captures_own_rule() -> None:
    """The selector rule changed with the rest-walk schema, so verification has to
    recompute under the rule that recorded the capture. Forwarding is what is
    asserted: a capture that disagrees between the rules needs a sample pattern the
    classifier only produces on hardware, so the seam is checked directly.
    """
    from foci_trace import velocity_measurement as vm

    seen: list[bool] = []
    original = vm.analyze_scored_rest_selection

    def spy(observations, *, rest_walk_revision=False):
        seen.append(rest_walk_revision)
        return original(observations, rest_walk_revision=rest_walk_revision)

    observations = [_observation(0) for _ in range(256)]
    record = original(observations, rest_walk_revision=True)

    vm.analyze_scored_rest_selection = spy
    try:
        vm.verify_scored_rest_selection(observations, record, rest_walk_revision=True)
    finally:
        vm.analyze_scored_rest_selection = original

    assert seen == [True], "verification must recompute under the capture's own rule"


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
    terminal = {"record_kind": "proportional_terminal", "run_sequence": 7, "cause": 0}
    arguments = {
        "schema_revision": 13,
        "schema_floor": 13,
        "expected_stage": 0,
        "ordinary_owners": (0,),
        "recovery_owners": (1, 2),
        "observation_kind": "observation",
        "plan_digest": 11,
        "label": "Proportional",
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
        label="Integral", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_integral_ordinary_rejection_current_tail() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
        _grammar_current(2, 3),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Integral",
        stabilized_recovery_rest=False,
        ordinary_rejection_tail=("velocity_primitive_current",),
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_proportional_ordinary_rejection_without_current() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="proportional_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Proportional", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_an_immediate_recovery_selection() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=0),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Integral", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_an_immediate_recovery_rejection() -> None:
    records = [_grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2)]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Integral", stabilized_recovery_rest=False, ordinary_rejection_tail=()
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
        label="Integral", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_accepts_a_stabilized_recovery_rejection() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2),
        _grammar_recovery_rest(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Integral", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == []


def test_scored_rest_grammar_rejects_a_stabilized_recovery_missing_event_150() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=0),
        _grammar_recovery(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Integral", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Integral recovery selection does not precede Event 150"
    ]


def test_scored_rest_grammar_rejects_integral_rejection_missing_its_current_record() -> None:
    """The disposition-2 tail is enforced, not merely exempted: an Integral
    rejection missing its Current record must be reported, not silently
    accepted the way the pre-fix grammar accepted any rejection."""
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Integral",
        stabilized_recovery_rest=False,
        ordinary_rejection_tail=("velocity_primitive_current",),
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Integral rejected scored-rest selection lacks its causal terminal tail"
    ]


def test_scored_rest_grammar_rejects_a_rejection_tail_with_an_extra_record() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
        {"record_kind": "integral_rung", "rung_index": 2, "slot": 3},
    ]
    terminal = _grammar_terminal(kind="proportional_terminal", cause=53)
    arguments = _grammar_arguments(
        label="Proportional", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Proportional rejected scored-rest selection lacks its causal terminal tail"
    ]


def test_scored_rest_grammar_rejects_an_ordinary_rejection_with_the_wrong_cause() -> None:
    records = [
        _grammar_observation(2, 3),
        _grammar_selection(owner=3, rung_index=2, slot=3, disposition=2),
    ]
    terminal = _grammar_terminal(kind="proportional_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Proportional", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Proportional rejected scored-rest selection lacks its causal terminal tail"
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
        label="Integral", stabilized_recovery_rest=False, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Integral rejected recovery selection lacks its causal terminal"
    ]


def test_scored_rest_grammar_rejects_a_stabilized_recovery_with_the_wrong_cause() -> None:
    records = [
        _grammar_selection(owner=5, rung_index=2, slot=0xFF, disposition=2),
        _grammar_recovery_rest(2),
    ]
    terminal = _grammar_terminal(kind="integral_terminal", cause=0)
    arguments = _grammar_arguments(
        label="Integral", stabilized_recovery_rest=True, ordinary_rejection_tail=()
    )

    assert audit_scored_rest_grammar(records, terminal, **arguments) == [
        "Integral rejected recovery selection lacks its causal terminal"
    ]


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


def test_bw_classify_is_inert_separates_no_motion_from_every_other_exclusion():
    """The inert fraction counts one class, not "everything ineligible".

    The 82-84% baseline meant "commanded and did not measurably move". A
    hard-clipped or stationarity-excluded stroke also yields no usable response,
    but those are different failures with different remedies, and folding them
    in would inflate the number the done-criterion is judged on.
    """
    # Moving and eligible: not inert.
    assert _bw_classify_is_inert(_bw_classifier_obs(0, 0)) is False
    # Settled interval no longer strictly positive: this is the inert class.
    assert _bw_classify_is_inert(_bw_classifier_obs(0, 0, settled_rate_low_q=0)) is True
    # Mirrored for a reverse stroke.
    assert _bw_classify_is_inert(_bw_classifier_obs(1, 1)) is False
    assert _bw_classify_is_inert(_bw_classifier_obs(1, 1, settled_rate_high_q=0)) is True

    # Earlier arms of the cascade win over Inert, so a stroke that did not move
    # but failed an earlier gate is not counted.
    assert (
        _bw_classify_is_inert(_bw_classifier_obs(0, 0, settled_rate_low_q=0, stationarity_code=3))
        is False
    )
    assert (
        _bw_classify_is_inert(
            _bw_classifier_obs(
                0, 0, settled_rate_low_q=0, predicted_torque_target_abs=1000, hard_torque_limit=1000
            )
        )
        is False
    )
    # Later arms do not: a stroke that moved but sat at the usable boundary is
    # CurrentHeadroom, which is a ceiling problem, not a dead-zone one.
    assert _bw_classify_is_inert(_bw_classifier_obs(0, 0, predicted_torque_target_abs=500)) is False


def test_bw_discovery_inert_fraction_reports_gross_and_above_breakaway():
    """Two ratios: gross is the ladder's honest cost, above-breakaway is the
    check that the seed actually solved the dead zone."""
    ladder = [128, 181, 243]
    observations = (
        # Floor rung, below breakaway: inert by design, counted only in gross.
        [_bw_classifier_obs(slot, 0, rung_index=0, settled_rate_low_q=0) for slot in range(4)]
        # First rung at breakaway: moving.
        + [_bw_classifier_obs(slot, 0, rung_index=1) for slot in range(4)]
        # A rung above breakaway with one inert stroke: this is the signal.
        + [_bw_classifier_obs(0, 0, rung_index=2, settled_rate_low_q=0)]
        + [_bw_classifier_obs(slot, 0, rung_index=2) for slot in range(1, 4)]
    )

    report = _bw_discovery_inert_fraction(observations, ladder, breakaway_p_raw=181)

    assert report["gross"] == {"inert": 5, "completed": 12, "fraction": 5 / 12}
    assert report["unexpected_above_breakaway"] == {"inert": 1, "completed": 8, "fraction": 1 / 8}
    assert report["inert_rungs_above_breakaway"] == [243]


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
        self.nominated_p_raw = self.ladder["gains"][
            self.ladder["gains"].index(self.breakaway_raw) + 1
        ]
        self.confirmation_digest = breakaway_confirmation_plan_digest(
            self.discovery_digest, self.nominated_p_raw, 0, 8, 4, 70, 80, 0
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
            _BW_PROPORTIONAL_SCHEMA_REVISION,
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
        margin records. `adjacent_rate_step_percent_milli` mirrors firmware's own
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
            self.nominated_p_raw,
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
        a `confirmed_p_raw` different from the nominated `nominated_p_raw`
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
            else (self.nominated_p_raw if accepted else 0)
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
        integral_plan_digest: int = 0xDEAD_BEEF,
    ) -> dict:
        raw = _bw_tag(
            12,
            "IHBBBQ",
            self.run_sequence,
            sequence,
            phase,
            terminal_cause,
            accepted,
            integral_plan_digest,
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
            integral_plan_digest=0,
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
            integral_plan_digest=0,
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
    assert by_collected_index[3]["adjacent_rate_step_percent_milli"] == 0


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
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, integral_plan_digest=0),
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
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, integral_plan_digest=0),
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
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, integral_plan_digest=0),
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
        builder.campaign_terminal_tick(accepted=0, terminal_cause=15, integral_plan_digest=0),
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
    `confirmation_planned.nominated_p_raw` against
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
        *builder.confirmation_ticks(confirmed_p_raw_override=builder.nominated_p_raw + 1),
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
            sequence=600, accepted=0, terminal_cause=15, integral_plan_digest=0
        ),
    ]
    with pytest.raises(ValueError, match="second confirmation candidate"):
        audit_breakaway_campaign(ticks)


def test_audit_breakaway_campaign_rejects_integral_after_unsuccessful_confirmation() -> None:
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


def test_audit_breakaway_campaign_allows_a_confirmed_integral_refusal() -> None:
    """A confirmed acceptance whose Integral plan build was refused is a
    legitimate outcome, not a validator error."""
    builder = _BreakawayCampaignBuilder()
    ticks = [
        builder.probe_plan_tick(),
        *builder.probe_stream_ticks(),
        builder.probe_result_tick(),
        builder.discovery_plan_tick(),
        *builder.discovery_ticks(),
        builder.confirmation_plan_tick(),
        *builder.confirmation_ticks(accepted=1, terminal_cause=21),
        builder.campaign_terminal_tick(accepted=0, terminal_cause=26, integral_plan_digest=0),
    ]
    # Must not raise.
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
            phase=1, terminal_cause=10, accepted=0, integral_plan_digest=0
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
            phase=1, terminal_cause=7, accepted=0, integral_plan_digest=0
        ),
    ]
    result = audit_breakaway_campaign(ticks)
    assert result["discovery_target_band_skipped"] is True
    assert result["additive_ladder_resolution_budget_exceeded"] is False
    assert result["discovery_planned"] is not None
