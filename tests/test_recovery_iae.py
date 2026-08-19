"""Unit tests for the reversal recovery-IAE metric.

The metric integrates absolute velocity tracking error over a fixed window after
each command reversal, giving one number per traced reversing move. Swept across
velocity-P it forms the U-curve whose trough is the robust response band
(too-low P -> sluggish/high IAE, too-high P -> oscillatory/high IAE).
"""

from collections import namedtuple

import pytest

from foci_trace.recovery_iae import recovery_iae_events, recovery_iae_summary

_Sample = namedtuple("_Sample", ["timestamp_ticks", "target_velocity", "velocity_error"])

_CLOCK_HZ = 84_000_000
_MS = 84_000  # ticks per millisecond at 84 MHz -> 1 kHz critical samples


def _run(commanded_velocity, error, count, start_tick):
    return [_Sample(start_tick + i * _MS, commanded_velocity, error) for i in range(count)]


def test_detects_single_reversal_and_its_direction():
    samples = _run(1000, 0, 8, 0) + _run(-1000, 0, 8, 8 * _MS)
    events = recovery_iae_events(samples, clock_hz=_CLOCK_HZ, window_s=0.005)
    assert len(events) == 1
    assert events[0].direction == -1  # command entered the negative direction


def test_integrates_absolute_error_over_the_window():
    pre = _run(1000, 0, 5, 0)
    post = _run(-1000, 200, 20, 5 * _MS)  # constant |error| = 200 after the reversal
    events = recovery_iae_events(pre + post, clock_hz=_CLOCK_HZ, window_s=0.010)
    assert len(events) == 1
    # ~10 one-millisecond intervals of |200| -> 200 * 0.001 * 10 = 2.0
    assert events[0].iae == pytest.approx(2.0, rel=0.15)


def test_window_bounds_the_integration():
    pre = _run(1000, 0, 5, 0)
    post = _run(-1000, 100, 40, 5 * _MS)
    short = recovery_iae_events(pre + post, clock_hz=_CLOCK_HZ, window_s=0.005)[0].iae
    long = recovery_iae_events(pre + post, clock_hz=_CLOCK_HZ, window_s=0.020)[0].iae
    assert long > short  # a wider window integrates more of the recovery


def test_summary_reports_median_overall_and_per_direction():
    samples = (
        _run(1000, 0, 4, 0)
        + _run(-1000, 100, 8, 4 * _MS)  # into negative, low error
        + _run(1000, 300, 8, 12 * _MS)  # into positive, high error
        + _run(-1000, 100, 8, 20 * _MS)  # into negative, low error
    )
    events = recovery_iae_events(samples, clock_hz=_CLOCK_HZ, window_s=0.005)
    summary = recovery_iae_summary(events)
    assert summary.count == 3
    assert summary.median_negative < summary.median_positive


def test_no_reversal_yields_no_events():
    samples = _run(1000, 500, 30, 0)
    assert recovery_iae_events(samples, clock_hz=_CLOCK_HZ, window_s=0.005) == []
