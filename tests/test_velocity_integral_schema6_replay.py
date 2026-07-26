"""Immutable schema-6 velocity-integral replay regressions."""

from pathlib import Path

import pytest

from foci_trace.analyze import analyze_capture


@pytest.mark.parametrize(
    ("capture_name", "outcome", "cause"),
    (
        (
            "foc-183-stepper-x-schema8-stagec6-ladder1-20260725-195110-complete-prefix.bin",
            3,
            11,
        ),
        (
            "foc-183-stepper-x-schema8-stagec6-restart-ladder1-20260725-223945-20260725-223952.bin",
            2,
            2,
        ),
    ),
)
def test_committed_schema_six_terminals_remain_authoritative(
    capture_name: str,
    outcome: int,
    cause: int,
) -> None:
    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-20-foc-183-velocity-integral-response"
        / capture_name
    )
    with capture.open("rb") as source:
        responses = analyze_capture(source).velocity_integral_audits

    assert len(responses) == 1
    assert responses[0]["matches_firmware"] is True
    assert responses[0]["plan"]["schema_revision"] == 6
    assert responses[0]["terminal"]["outcome"] == outcome
    assert responses[0]["terminal"]["cause"] == cause
