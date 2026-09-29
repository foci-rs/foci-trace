from copy import deepcopy

import pytest

from foci_trace.fixed_gain_amplitude import (
    _compact_execution_marker,
    _fnv1a64,
    amplitude_plan_digest,
    assemble_amplitude_plan,
    audit_fixed_gain_amplitude,
    compare_amplitude_runs,
    reconstruct_amplitude,
    reconstruct_direction,
    robust_rate_location,
)


def _plan() -> dict[str, object]:
    return {
        "schema_revision": 1,
        "channel": 0,
        "build_revision": 0x1234_5678,
        "acceptance_schema_revision": 10,
        "acceptance_plan_digest": 0x1111_2222_3333_4444,
        "order": 1,
        "target_rpm": [16, 33, 66, 132, 263],
        "selected_p": 1_024,
        "selected_i": 1_024,
        "qualifier_bits": 0x0123,
        "pwm_hz": 25_000,
        "encoder_counts_per_rev": 4_000,
        "amplitude_count": 5,
        "expected_observations": 40,
        "family_size": 20,
        "moving_stroke_us": 512_000,
        "zero_settle_us": 500_000,
        "analysis_budget_us": 236_000,
        "nominal_workflow_ms": 60_541,
        "maximum_workflow_ms": 66_456,
        "recovery_origin_band_counts": 1_000,
        "recovery_nominal_slot_us": 1_816_958,
        "recovery_maximum_slot_us": 3_000_000,
        "recovery_slot_count": 5,
        "hard_torque_limit": 2_816,
        "usable_torque_limit": 2_816,
        "limit_evidence_capability": 1,
        "recovery_quantization_exposed": 0,
        "probe_constrained_test_point": 0,
        "requested_velocity_mrev_s": 4_394,
        "machine_velocity_ceiling_mrev_s": 10_000,
        "requested_velocity_source": 0,
        "max_stroke_travel_mrev": 2_500,
        "settle_travel_reserve_mrev": 250,
        "negative_position_headroom_mrev": 7_025,
        "positive_position_headroom_mrev": 7_025,
        "max_duration_ms": 3_000,
    }


def _schema_two_plan() -> dict[str, object]:
    plan = _plan()
    plan["schema_revision"] = 2
    plan["recovery_lower_rate_q"] = [13_217_792, 12_827_648]
    return plan


def _amplitude_plan_fixture(*, acceptance_schema_revision: int) -> dict[str, object]:
    """`_plan()` with `acceptance_schema_revision` overridden.

    `amplitude_plan_digest` hashes `acceptance_schema_revision` value-agnostically
    (`fixed_gain_amplitude.py:55-60`), so changing it changes the encoded bytes and
    therefore the digest; callers must supply the digest expected for that
    schema rather than reuse the schema-10 constant.
    """
    plan = _plan()
    plan["acceptance_schema_revision"] = acceptance_schema_revision
    return plan


def _observations(
    intervals: list[tuple[int, int]],
    means: list[int],
    *,
    direction: int = 0,
    classes: list[int] | None = None,
) -> list[dict[str, object]]:
    classes = classes or [0, 0, 0, 0]
    return [
        {
            "slot": index * 2 + direction,
            "direction": direction,
            "classification": classes[index],
            "rate_low": interval[0],
            "rate_mean": means[index],
            "rate_high": interval[1],
            "selected_level": index + 1,
        }
        for index, interval in enumerate(intervals)
    ]


def test_robust_location_matches_signed_firmware_midpoint() -> None:
    assert robust_rate_location([9, 1, 5, None]) == 5
    assert robust_rate_location([4, 1, 8, 3]) == 3
    assert robust_rate_location([-1, -4, -8, 3]) == -3
    assert robust_rate_location([1, 2, None, None]) is None


def test_plan_digest_is_sensitive_to_order_and_exact_targets() -> None:
    plan = _plan()

    assert amplitude_plan_digest(plan) == 0x2618_AA16_E7C5_77C3

    descending = deepcopy(plan)
    descending["order"] = 2
    descending["target_rpm"] = list(reversed(plan["target_rpm"]))
    assert amplitude_plan_digest(descending) != amplitude_plan_digest(plan)


def test_amplitude_plan_digest_accepts_breakaway_acceptance_identity() -> None:
    """A breakaway (`acceptance_schema_revision=16`) amplitude plan decodes with
    no code change: `amplitude_plan_digest` hashes the field value-agnostically."""
    plan = _amplitude_plan_fixture(acceptance_schema_revision=16)
    plan["plan_digest"] = 0x5CAA_1C5A_DEE3_EC99

    assert amplitude_plan_digest(plan) == plan["plan_digest"]


def test_amplitude_plan_digest_still_accepts_legacy_combined_identity() -> None:
    """Archived combined captures (Integral spans schema 8-13; the fixture
    baseline is 10) must keep decoding byte-identically alongside breakaway."""
    expected_digests = {
        10: 0x2618_AA16_E7C5_77C3,
        13: 0xFBAC_7BDD_0EE2_262E,
    }
    for legacy, digest in expected_digests.items():
        plan = _amplitude_plan_fixture(acceptance_schema_revision=legacy)
        plan["plan_digest"] = digest

        assert amplitude_plan_digest(plan) == plan["plan_digest"]


def test_schema_two_digest_and_execution_marker_bind_recovery_authority() -> None:
    plan = _schema_two_plan()

    digest = amplitude_plan_digest(plan)
    marker = _compact_execution_marker({**plan, "run_sequence": 7, "plan_digest": digest})

    assert digest == 0x8543_77D8_CC8D_7426
    assert digest != amplitude_plan_digest(_plan())
    assert len(marker) == 98
    assert marker[15:19] == (4_394).to_bytes(4, "little")
    assert marker[19:23] == (263).to_bytes(4, "little", signed=True)
    assert marker[82:90] == (13_217_792).to_bytes(8, "little")
    assert marker[90:98] == (12_827_648).to_bytes(8, "little")


def test_plan_fragments_require_one_shared_identity() -> None:
    plan = _plan()
    fragments = [
        {
            "fragment": fragment,
            "run_sequence": 7,
            "plan_digest": 9,
            **(
                {
                    key: value
                    for key, value in plan.items()
                    if key
                    not in {
                        "moving_stroke_us",
                        "zero_settle_us",
                        "analysis_budget_us",
                        "nominal_workflow_ms",
                        "maximum_workflow_ms",
                    }
                }
                if fragment == 0
                else {}
            ),
        }
        for fragment in range(3)
    ]
    fragments[1].update(
        {
            "moving_stroke_us": 512_000,
            "zero_settle_us": 500_000,
            "analysis_budget_us": 236_000,
            "nominal_workflow_ms": 60_541,
            "maximum_workflow_ms": 66_456,
        }
    )

    assert assemble_amplitude_plan(fragments)["run_sequence"] == 7

    fragments[2]["plan_digest"] = 10
    with pytest.raises(ValueError, match="disagree"):
        assemble_amplitude_plan(fragments)


def test_direction_reuses_depth_three_closed_interval_consensus() -> None:
    observations = _observations(
        [(0, 10), (2, 9), (3, 8), (4, 7)],
        [5, 6, 7, 8],
    )

    evidence = reconstruct_direction(
        observations,
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )

    assert evidence["availability"] == 0
    assert evidence["components"] == [[3, 8]]
    assert evidence["included_mask"] == 0b1111
    assert evidence["robust_location_q"] == 6
    assert evidence["deficit_interval_q"] == [12, 17]
    assert evidence["blocking_levels"] == [1, 2, 3, 4]


def test_direction_separates_precision_floor_current_and_fault() -> None:
    intervals = [(0, 1)] * 4
    means = [0] * 4

    floor = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 4, 4]),
        signed_target_rate_q=20,
        precision_floor_candidate=True,
    )
    current = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 0, 3]),
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )
    fault = reconstruct_direction(
        _observations(intervals, means, classes=[0, 0, 0, 10]),
        signed_target_rate_q=20,
        precision_floor_candidate=False,
    )

    assert [floor["availability"], current["availability"], fault["availability"]] == [1, 4, 5]


def test_amplitude_preserves_reverse_sign_and_frozen_target() -> None:
    plan = _plan()
    forward = _observations([(900, 1_000)] * 4, [950] * 4)
    reverse = _observations([(-1_000, -900)] * 4, [-950] * 4, direction=1)
    report = {"amplitude_index": 0, "target_rpm": 16}

    amplitude = reconstruct_amplitude(forward + reverse, report, plan)

    assert amplitude["directions"][0]["signed_target_rate_q"] == 1_066_666
    assert amplitude["directions"][1]["signed_target_rate_q"] == -1_066_666
    assert amplitude["directions"][1]["response_interval_q"] == [-1_000, -900]
    assert amplitude["directions"][1]["deficit_interval_q"] == [-1_065_766, -1_065_666]


def test_complete_amplitude_audit_reproduces_masks_counts_and_digest() -> None:
    plan = _plan()
    plan.update({"run_sequence": 7, "plan_digest": 0})
    plan["plan_digest"] = amplitude_plan_digest(plan)
    plan_fragments = [
        (
            bytes([fragment]),
            {
                **plan,
                "fragment": fragment,
                "record_kind": "fixed_gain_amplitude_plan",
            },
        )
        for fragment in range(3)
    ]
    evidence: list[tuple[bytes, dict[str, object]]] = list(plan_fragments)
    evidence_sequence = 1
    for amplitude_index, target_rpm in enumerate(plan["target_rpm"]):
        observations: list[dict[str, object]] = []
        for direction in range(2):
            sign = 1 if direction == 0 else -1
            for observation_index in range(4):
                mean = sign * (1_000 + 10 * amplitude_index + observation_index)
                low, high = sorted((mean - 5, mean + 5))
                observation = {
                    "record_kind": "integral_observation",
                    "run_sequence": 7,
                    "evidence_sequence": evidence_sequence,
                    "plan_digest": plan["plan_digest"],
                    "rung_index": amplitude_index,
                    "slot": observation_index * 2 + direction,
                    "direction": direction,
                    "classification": 0,
                    "rate_low": low,
                    "rate_mean": mean,
                    "rate_high": high,
                    "selected_level": 4,
                }
                raw = bytes((2, evidence_sequence & 0xFF, amplitude_index, direction))
                evidence.append((raw, observation))
                observations.append(observation)
                evidence_sequence += 1
        reported = {
            **reconstruct_amplitude(
                observations,
                {"amplitude_index": amplitude_index, "target_rpm": target_rpm},
                plan,
            ),
            "record_kind": "fixed_gain_amplitude_amplitude",
            "run_sequence": 7,
            "evidence_sequence": evidence_sequence,
            "plan_digest": plan["plan_digest"],
        }
        evidence.append((bytes((2, evidence_sequence & 0xFF, amplitude_index)), reported))
        evidence_sequence += 1
    digest = _fnv1a64(_compact_execution_marker(plan))
    for raw, _record in evidence[3:]:
        digest = _fnv1a64(raw, digest)
    terminal = {
        "record_kind": "fixed_gain_amplitude_terminal",
        "run_sequence": 7,
        "evidence_sequence": evidence_sequence,
        "schema_revision": 1,
        "acceptance_plan_digest": plan["acceptance_plan_digest"],
        "plan_digest": plan["plan_digest"],
        "digest": digest,
        "outcome": 0,
        "cause": 0,
        "attempted_masks": [0b11111, 0b11111],
        "eligible_masks": [0b11111, 0b11111],
        "current_terminus_masks": [0, 0],
        "unattempted_masks": [0, 0],
        "expected_observations": 40,
        "emitted_observations": 40,
        "expected_amplitudes": 5,
        "emitted_amplitudes": 5,
        "qualifier_bits": plan["qualifier_bits"],
    }
    evidence.append((b"terminal", terminal))

    audit = audit_fixed_gain_amplitude(evidence)

    assert audit["matches_firmware"] is True
    assert audit["complete"] is True
    assert audit["mismatches"] == []
    assert len(audit["amplitudes"]) == 5


def _minimal_amplitude_evidence(
    *, nonterminal_sequences: list[int], terminal_evidence_sequence: int
) -> list[tuple[bytes, dict[str, object]]]:
    plan = _plan()
    plan.update({"run_sequence": 7, "plan_digest": 0})
    plan["plan_digest"] = amplitude_plan_digest(plan)
    plan_fragments = [
        (
            bytes([fragment]),
            {**plan, "fragment": fragment, "record_kind": "fixed_gain_amplitude_plan"},
        )
        for fragment in range(3)
    ]
    nonterminal = [
        (
            bytes((0, sequence)),
            {
                "record_kind": "velocity_primitive_current",
                "run_sequence": 7,
                "evidence_sequence": sequence,
                "plan_digest": plan["plan_digest"],
                "rung_index": 0,
                "slot": 0,
            },
        )
        for sequence in nonterminal_sequences
    ]
    terminal = {
        "record_kind": "fixed_gain_amplitude_terminal",
        "run_sequence": 7,
        "evidence_sequence": terminal_evidence_sequence,
        "schema_revision": 1,
        "cause": 0,
        "outcome": 1,
        "digest": 0,
    }
    return [*plan_fragments, *nonterminal, (b"terminal", terminal)]


def test_amplitude_audit_flags_a_gap_in_the_evidence_sequence() -> None:
    evidence = _minimal_amplitude_evidence(
        nonterminal_sequences=[1, 3], terminal_evidence_sequence=4
    )

    audit = audit_fixed_gain_amplitude(evidence)

    assert any(
        "evidence sequence is not contiguous" in mismatch for mismatch in audit["mismatches"]
    )


def test_amplitude_audit_flags_a_terminal_sequence_that_skips_evidence() -> None:
    evidence = _minimal_amplitude_evidence(
        nonterminal_sequences=[1, 2], terminal_evidence_sequence=9
    )

    audit = audit_fixed_gain_amplitude(evidence)

    assert any(
        "terminal sequence does not follow the final evidence record" in mismatch
        for mismatch in audit["mismatches"]
    )


_GRAMMAR_MISMATCH_MARKERS = ("scored-rest", "does not precede", "lacks its causal selection")


def test_amplitude_audit_enforces_the_stabilized_rejection_tail() -> None:
    """`audit_fixed_gain_amplitude` passes `stabilized_recovery_rest=True` and
    `ordinary_rejection_tail=("velocity_primitive_current",)` at its call site
    (fixed_gain_amplitude.py:338); a test that only calls the grammar helper
    directly would not catch a regression to `()` here. Existing amplitude tests
    run at schema 1 with no scored-rest selections, so the grammar exits below
    its floor and stays green regardless -- this fixture is schema 4 to reach
    the code path at all.

    The fixture is not a fully firmware-consistent capture (no amplitude
    records, mismatched terminal digest and workflow duration), so this only
    asserts the absence of grammar-specific mismatches, not a clean audit."""
    plan = _schema_two_plan()
    plan.update({"schema_revision": 4, "run_sequence": 7, "plan_digest": 0})
    plan["plan_digest"] = amplitude_plan_digest(plan)
    plan_fragments = [
        (
            bytes([fragment]),
            {**plan, "fragment": fragment, "record_kind": "fixed_gain_amplitude_plan"},
        )
        for fragment in range(3)
    ]
    observation = {
        "record_kind": "integral_observation",
        "run_sequence": 7,
        "plan_digest": plan["plan_digest"],
        "rung_index": 0,
        "slot": 0,
    }
    selection = {
        "record_kind": "rest_selection",
        "run_sequence": 7,
        "plan_digest": plan["plan_digest"],
        "stage": 1,
        "owner": 7,
        "rung_index": 0,
        "slot": 0,
        "disposition": 2,
    }
    current = {
        "record_kind": "velocity_primitive_current",
        "run_sequence": 7,
        "plan_digest": plan["plan_digest"],
        "rung_index": 0,
        "slot": 0,
    }
    terminal = {
        "record_kind": "fixed_gain_amplitude_terminal",
        "run_sequence": 7,
        "cause": 53,
        "outcome": 1,
        "digest": 0,
    }
    evidence: list[tuple[bytes, dict[str, object]]] = [
        *plan_fragments,
        (b"\x00", observation),
        (b"\x01", selection),
        (b"\x02", current),
        (b"terminal", terminal),
    ]

    audit = audit_fixed_gain_amplitude(evidence)

    assert not any(
        marker in mismatch
        for mismatch in audit["mismatches"]
        for marker in _GRAMMAR_MISMATCH_MARKERS
    )


def _audit(interval: list[int], robust: int, *, attempted: int = 0b11111) -> dict[str, object]:
    direction = {
        "availability": 0,
        "response_interval_q": interval,
        "robust_location_q": robust,
        "signed_target_rate_q": 1_000,
    }
    return {
        "amplitudes": [
            {
                "amplitude_index": index,
                "directions": [deepcopy(direction), deepcopy(direction)],
            }
            for index in range(5)
        ],
        "terminal": {"attempted_masks": [attempted, attempted]},
    }


def test_cross_run_comparison_distinguishes_absence_and_closed_touch() -> None:
    previous = _audit([0, 5], 3)
    current = _audit([5, 10], 7)
    current["terminal"]["attempted_masks"][0] = 0b11110
    current["amplitudes"][1]["directions"][1]["availability"] = 3

    comparison = compare_amplitude_runs(previous, current)
    by_key = {
        (item["amplitude_index"], item["direction"]): item for item in comparison["comparisons"]
    }

    assert by_key[(0, 0)]["kind"] == "AttemptCoverageMismatch"
    assert by_key[(1, 1)]["kind"] == "EligibilityMismatch"
    assert by_key[(2, 0)]["kind"] == "PairedPrecision"
    assert by_key[(2, 0)]["overlap"] is True
    assert by_key[(2, 0)]["signed_gap_q"] == 0
    assert comparison["maximum_observed_displacement_q"] == 4
    assert comparison["maximum_displacement_kind"] == "observed_bound"


def test_cross_run_signed_gap_preserves_movement_direction() -> None:
    forward = compare_amplitude_runs(_audit([0, 4], 2), _audit([6, 10], 8))
    reverse = compare_amplitude_runs(_audit([6, 10], 8), _audit([0, 4], 2))

    assert forward["comparisons"][0]["signed_gap_q"] == 2
    assert reverse["comparisons"][0]["signed_gap_q"] == -2


def test_schema_one_execution_marker_decodes_without_recovery_rates() -> None:
    from foci_trace.protocol import _decode_velocity_sweep_evidence

    plan = _plan()
    digest = amplitude_plan_digest(plan)
    marker = _compact_execution_marker({**plan, "run_sequence": 7, "plan_digest": digest})

    decoded = _decode_velocity_sweep_evidence(marker)

    assert len(marker) == 82
    assert decoded["record_kind"] == "integral_plan_marker"
    assert decoded["acceptance_plan_digest"] == plan["acceptance_plan_digest"]
    assert decoded["qualifier_bits"] == plan["qualifier_bits"]
    assert "recovery_lower_rate_q" not in decoded


def test_amplitude_audit_reports_a_duplicate_plan_fragment_as_a_mismatch() -> None:
    evidence = _minimal_amplitude_evidence(
        nonterminal_sequences=[1, 2], terminal_evidence_sequence=3
    )
    evidence[2] = (evidence[2][0], {**evidence[2][1], "fragment": 1})

    audit = audit_fixed_gain_amplitude(evidence)

    assert audit["run_sequence"] == 7
    assert audit["matches_firmware"] is False
    assert any("fragments 0, 1, and 2" in mismatch for mismatch in audit["mismatches"])
