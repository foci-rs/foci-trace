"""Vocabulary parity guard.

Asserts ``foci_trace.protocol.PHASE_NAMES`` and
``foci_trace.velocity_integral.BREAKAWAY_PHASE_NAMES`` match the checked-in
fixture shared with the Rust ``CommissionPhase`` / ``BreakawayPhase`` oracle
(``shared/foci-firmware/src/commissioning/vocabulary_fixture_tests.rs``), so a
single-layer desync between the firmware enums and this host table fails a test.

Unlike klipper-foci's ``PHASE_NAMES`` (human-readable prose labels, matched
against the fixture's ``label`` field), foci-trace's ``PHASE_NAMES`` mirrors the
Rust variant names directly, so it is matched against the fixture's ``ident``
field instead. ``BREAKAWAY_PHASE_NAMES`` uses the same snake_case strings in
both host packages and is matched against ``label`` in both.
"""

import json
from pathlib import Path

from foci_trace.protocol import BREAKAWAY_PHASE_NAMES, PHASE_NAMES

# tests/ -> foci-trace -> host -> foci (aggregate root)
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "shared/foci-firmware/src/commissioning/vocabulary_fixture.json"
)


def _fixture_group(group: str) -> list[dict[str, object]]:
    entries = json.loads(_FIXTURE_PATH.read_text())
    return [entry for entry in entries if entry["group"] == group]


def test_phase_names_matches_fixture_idents() -> None:
    expected = {entry["wire_code"]: entry["ident"] for entry in _fixture_group("phase")}
    assert expected == PHASE_NAMES


def test_breakaway_phase_names_matches_fixture() -> None:
    expected = {entry["wire_code"]: entry["label"] for entry in _fixture_group("breakaway")}
    assert expected == BREAKAWAY_PHASE_NAMES
