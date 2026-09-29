"""Replay the seven hardware rejections that motivated the drift bound.

Provenance: the seven `(rung, slot, slope_q, mean_rate_q)` tuples below were
recovered directly from
`docs/artifacts/trace-captures/2026-07-27-foc-192-confidence/
foc-192-stepper-y-rejection-slope-20260729-221033.bin` by decoding raw frames
with `foci_trace.protocol.decode_frame` (the committed `.velocity.json` for
this capture carries no rejection diagnostics at all -- every
`integral_observation` record it surfaces has `classification == 0` and
`usable == 1` -- so the JSON mirror cannot answer this on its own). Filtering
decoded `integral_observation` records for `stationarity_code != 0` and
`rejection_present == 1` yields exactly seven records, and their
`(rung_index, slot, rejection_slope_q)` match the tuples below exactly.
`mean_rate_q` was computed here as the exact midpoint of each record's
`rejection_settled_rate_interval_q`, matching the tuples below exactly (an
excluded record zeroes `rate_mean`, so the interval midpoint is the only
recoverable rate).

This test therefore pins two separate things: that the seven inputs match
what hardware actually reported, and that `_drift_ratio_ppm` (the offline
mirror of the firmware's integer-only arithmetic) keeps every one of them
under the bound across every span the block structure could have produced.
It does not replay the classifier itself: the span that produced these
records is not recoverable from the rejection path (see
`test_every_captured_rejection_passes_at_every_possible_span`'s docstring),
so the test sweeps every span a 64-sample suffix could produce rather than
asserting one.
"""

from foci_trace.velocity_measurement import _drift_ratio_ppm

CAPTURED_REJECTIONS = [
    (0, 3, -8_823_529, 13_750_000),
    (1, 7, -14_797_794, 13_546_875),
    (2, 1, -13_694_853, 13_515_625),
    (2, 3, -10_845_589, 13_500_000),
    (2, 5, -7_996_324, 13_453_125),
    (2, 7, -9_283_089, 13_421_875),
    (8, 0, 5_974_265, -15_359_375),
]

POSSIBLE_SPANS_US = [(64 - 2**level) * 1000 for level in range(5)]


def test_every_captured_rejection_passes_at_every_possible_span():
    """These seven are the whole evidence base for replacing the gate. Under the
    significance test all seven were excluded at the family's shortest and most
    permissive window.

    The block level is not reported on the rejection path, so the exact regression
    span is unknown. Assert across every span the suffix could have produced; the
    conclusion must not depend on which one it was."""
    worst = 0
    for span_us in POSSIBLE_SPANS_US:
        for rung, slot, slope_q, mean_rate_q in CAPTURED_REJECTIONS:
            ratio = _drift_ratio_ppm(slope_q, 2 * span_us, mean_rate_q)
            assert ratio <= 200_000, f"rung {rung} slot {slot} at span {span_us}"
            worst = max(worst, ratio)

    assert worst == 68_818, worst


def test_the_longest_span_is_the_worst_case():
    """Sanity check on the reasoning above: ppm grows with span, so checking the
    longest span alone would suffice. Assert it rather than assume it."""
    ratios = [
        _drift_ratio_ppm(-14_797_794, 2 * span_us, 13_546_875)
        for span_us in sorted(POSSIBLE_SPANS_US)
    ]
    assert ratios == sorted(ratios)
