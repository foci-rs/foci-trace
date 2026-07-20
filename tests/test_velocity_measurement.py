import json
from pathlib import Path

import pytest

from foci_trace.velocity_measurement import (
    AccuracyDecision,
    VelocityObservation,
    analyze_velocity_observations,
    analyze_stationary_tail,
    classify_absolute_rate,
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


def test_sweep_confidence_uses_all_two_k_observations() -> None:
    from foci_trace.velocity_measurement import _confidence_multiplier

    assert _confidence_multiplier(40) == pytest.approx(3.962993)


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
        _compare_stage_b_candidates,
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
    assert _compare_stage_b_candidates(first, dict(first))["outcome"] == "complete"
    changed = {**first, "memberships": [0b0111000, 0b0011100]}
    mismatch = _compare_stage_b_candidates(first, changed)
    assert mismatch["outcome"] == "reproduction_mismatch"
    assert mismatch["replacement"] == changed
    changed_plan = {**first, "plan_digest": 8}
    assert _compare_stage_b_candidates(first, changed_plan)["outcome"] == "plan_mismatch"


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
