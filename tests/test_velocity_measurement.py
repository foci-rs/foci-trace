import json
from pathlib import Path

import pytest

from foci_trace.velocity_measurement import (
    AccuracyDecision,
    VelocityObservation,
    analyze_velocity_observations,
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
