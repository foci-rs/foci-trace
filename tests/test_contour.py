from foci_trace.contour import (
    MotorSample,
    classify_phase,
    contour_pair_groups,
    corexy_to_xy,
    pair_motor_samples,
    project_path_error,
)


def test_corexy_to_xy_reconstructs_toolhead_coordinates() -> None:
    assert corexy_to_xy(300.0, 100.0) == (200.0, 100.0)
    assert corexy_to_xy(-400.0, 0.0) == (-200.0, -200.0)


def test_project_path_error_splits_lag_from_cross_error() -> None:
    along, cross = project_path_error(
        start=(0.0, 0.0),
        end=(1000.0, 0.0),
        commanded=(500.0, 0.0),
        actual=(450.0, 0.0),
    )
    assert along == 50.0
    assert cross == 0.0

    along, cross = project_path_error(
        start=(0.0, 0.0),
        end=(1000.0, 0.0),
        commanded=(500.0, 0.0),
        actual=(500.0, -25.0),
    )
    assert along == 0.0
    assert cross == 25.0


def test_pair_motor_samples_uses_nearest_host_timestamp() -> None:
    plus = [
        MotorSample(host_recv_ns=1_000, commanded=0.0, actual=0.0),
        MotorSample(host_recv_ns=2_000, commanded=100.0, actual=90.0),
    ]
    minus = [
        MotorSample(host_recv_ns=1_100, commanded=0.0, actual=0.0),
        MotorSample(host_recv_ns=2_080, commanded=100.0, actual=92.0),
    ]

    pairs = pair_motor_samples(plus, minus, tolerance_ns=150)

    assert len(pairs) == 2
    assert pairs[0].host_recv_ns == 1_050
    assert pairs[1].plus.actual == 90.0
    assert pairs[1].minus.actual == 92.0


def test_pair_motor_samples_sorts_inputs_and_skips_out_of_tolerance_samples() -> None:
    plus = [
        MotorSample(host_recv_ns=3_000, commanded=300.0, actual=290.0),
        MotorSample(host_recv_ns=1_000, commanded=100.0, actual=95.0),
        MotorSample(host_recv_ns=6_000, commanded=600.0, actual=580.0),
        MotorSample(host_recv_ns=2_000, commanded=200.0, actual=190.0),
    ]
    minus = [
        MotorSample(host_recv_ns=2_100, commanded=210.0, actual=205.0),
        MotorSample(host_recv_ns=6_200, commanded=620.0, actual=610.0),
        MotorSample(host_recv_ns=950, commanded=95.0, actual=90.0),
        MotorSample(host_recv_ns=3_050, commanded=305.0, actual=300.0),
    ]

    pairs = pair_motor_samples(plus, minus, tolerance_ns=120)

    assert [pair.host_recv_ns for pair in pairs] == [975, 2_050, 3_025]
    assert [pair.plus.host_recv_ns for pair in pairs] == [1_000, 2_000, 3_000]
    assert [pair.minus.host_recv_ns for pair in pairs] == [950, 2_100, 3_050]


def test_classify_phase_prefers_lag_before_contour() -> None:
    assert classify_phase(80.0, 10.0, trace_gap=False, post_discriminator=None) == (
        "along_path_lag"
    )
    assert classify_phase(80.0, 55.0, trace_gap=False, post_discriminator=None) == (
        "cross_path_contour"
    )
    assert classify_phase(80.0, 10.0, trace_gap=True, post_discriminator=None) == (
        "inconclusive"
    )
    assert classify_phase(80.0, 10.0, trace_gap=False, post_discriminator="loop_active") == (
        "endpoint_loop_active"
    )
    assert classify_phase(0.0, 10.0, trace_gap=False, post_discriminator=None) == (
        "cross_path_contour"
    )


def test_contour_pair_groups_only_keeps_commanded_motion() -> None:
    pairs = [
        _pair(1_000, 0.0, 0.0),
        _pair(2_000, 100.0, 100.0),
        _pair(3_000, 200.0, 200.0),
        _pair(200_000_000, 200.0, 200.0),
        _pair(201_000_000, 240.0, 240.0),
    ]

    groups = contour_pair_groups(pairs, gap_ns=10_000, min_delta=100.0)

    assert len(groups) == 1
    assert groups[0] == pairs[:3]


def test_contour_pair_groups_excludes_idle_samples_around_motion() -> None:
    pairs = [
        _pair(1_000, 0.0, 0.0),
        _pair(2_000, 0.0, 0.0),
        _pair(3_000, 100.0, 100.0),
        _pair(4_000, 200.0, 200.0),
        _pair(5_000, 200.0, 200.0),
    ]

    groups = contour_pair_groups(pairs, gap_ns=10_000, min_delta=100.0)

    assert len(groups) == 1
    assert groups[0] == pairs[1:4]


def _pair(host_recv_ns: int, plus_commanded: float, minus_commanded: float):
    return pair_motor_samples(
        [MotorSample(host_recv_ns=host_recv_ns, commanded=plus_commanded, actual=plus_commanded)],
        [MotorSample(host_recv_ns=host_recv_ns, commanded=minus_commanded, actual=minus_commanded)],
        tolerance_ns=0,
    )[0]
