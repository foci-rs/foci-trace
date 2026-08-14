"""Pin foci-trace's decode of archived combined-workflow captures.

The combined acquisition path was removed from firmware (the
`combined_workflow`/`combined_selected_response` execution-plan flag and,
later, the `ReverseFirst`/`PairedOutAndBack` `IntegralSlotOrder` variants --
see `shared/foci-firmware` commits `78e19e9` and `77a204d`). foci-trace was
deliberately left untouched: it keeps its own decode-only mirror of the
retired schema-8-through-13 combined range and the retired slot orders, so
that captures taken before the removal still replay.

This module is the regression that pins that guarantee. Both fixtures below
are combined-workflow captures (schema revision 8, `combined_selected_response
= True`) with a committed `.bin` and a committed `.replay.json` golden under
`docs/artifacts/trace-captures/2026-07-26-foc-183-an053-combined/`.

A genuinely paired-slot-order (`IntegralSlotOrder::PairedOutAndBack`) capture
exists -- `foc207-paired-stepper-x-20260811-223447.bin` under
`docs/artifacts/trace-captures/2026-08-11-foc-207-paired-order/`, referenced
in that directory's manifest as `slot_order = 2` -- but its `.bin` is
git-ignored (`docs/artifacts/trace-captures/.gitignore:8` excludes
`**/*.bin`) and was never force-added, so it is not available as a committed
fixture here. The one committed capture with "paired" in its name,
`foc-182-rpm-probe-p384-paired-stepper-x-20260718-212752.bin`, predates the
Stage-C integral audit path entirely (it decodes to zero
`velocity_integral_audits`), so it would not exercise the guarantee this
module pins and was not used.
"""

from pathlib import Path

import pytest

from foci_trace.analyze import analyze_capture


@pytest.mark.parametrize(
    ("capture_name", "outcome", "cause", "plan_digest", "digest"),
    (
        (
            "foc183-y-combined-schema9-stagec8-20260726-200611.bin",
            3,
            47,
            12202292821964108601,
            2869795943234380837,
        ),
        (
            "foc183-y-combined-clampfix-schema9-stagec8-20260726-203339.bin",
            1,
            10,
            7775654929536591997,
            12278291537036479906,
        ),
    ),
)
def test_archived_combined_capture_still_decodes_unchanged(
    capture_name: str,
    outcome: int,
    cause: int,
    plan_digest: int,
    digest: int,
) -> None:
    """An archived combined-workflow capture must keep decoding bit-exactly.

    Firmware no longer builds this plan shape (it was retired), but
    foci-trace's decode of the bytes it already wrote must not
    move. `matches_firmware` re-derives the terminal offline and compares it
    against what firmware reported at capture time; a `True` here after
    the removal is the whole point of the pin."""
    capture = (
        Path(__file__).resolve().parents[4]
        / "docs/artifacts/trace-captures/2026-07-26-foc-183-an053-combined"
        / capture_name
    )
    with capture.open("rb") as source:
        responses = analyze_capture(source).velocity_integral_audits

    assert len(responses) == 1
    response = responses[0]

    assert response["matches_firmware"] is True
    assert response["plan"]["schema_revision"] == 8
    assert response["plan"]["combined_selected_response"] is True
    assert response["terminal"]["combined_workflow"] is True
    assert response["terminal"]["outcome"] == outcome
    assert response["terminal"]["cause"] == cause
    assert response["terminal"]["plan_digest"] == plan_digest
    assert response["digest"] == digest
