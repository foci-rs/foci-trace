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
) -> VelocityObservation:
    return VelocityObservation(
        delta_counts=delta_counts,
        elapsed_us=elapsed_us,
        direction=direction,
        stroke_index=stroke_index,
        target_cpms=target_cpms,
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
        "used_velocity_mrev_s": 1_000,
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
            "target_velocity_register": direction * 100,
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
