"""Schema-8 Stage-B component-selection reproduction."""

from __future__ import annotations

from fractions import Fraction


def _split_components(mask: int) -> list[dict[str, int]]:
    components = []
    remaining = mask
    while remaining:
        first = (remaining & -remaining).bit_length() - 1
        count = 0
        shifted = remaining >> first
        while shifted & 1:
            count += 1
            shifted >>= 1
        member_mask = ((1 << count) - 1) << first
        components.append(
            {
                "member_mask": member_mask,
                "first_rung": first,
                "member_count": count,
            }
        )
        remaining &= ~member_mask
    return components


def project_schema8_candidates(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Apply the schema-8 comparator to two retained Stage-B summaries."""
    from .velocity_measurement import (
        _contiguous_proportional_mask,
        _proportional_closed_intersection,
        _proportional_core_coverage,
        _proportional_gain_at,
        _proportional_membership_comparison,
        _ratio_margin,
    )

    geometry = ("p_start", "p_top", "rung_count")
    if any(int(previous[name]) != int(current[name]) for name in geometry):
        return {"interpretation": "schema-8 what-if", "outcome": "plan_mismatch"}
    p_start, p_top, rung_count = (int(previous[name]) for name in geometry)
    previous_masks = [int(value) for value in previous["memberships"]]
    current_masks = [int(value) for value in current["memberships"]]
    valid_bits = 0xFFFF_FFFF if rung_count == 32 else (1 << rung_count) - 1
    if (
        p_start <= 0
        or p_start > p_top
        or rung_count not in range(1, 33)
        or any(mask & ~valid_bits for mask in (*previous_masks, *current_masks))
        or any(
            mask and not _contiguous_proportional_mask(mask, rung_count)
            for mask in (*previous_masks[:2], *current_masks[:2])
        )
        or previous_masks[2] == 0
        or current_masks[2] == 0
    ):
        return {
            "interpretation": "schema-8 what-if",
            "outcome": "evidence_integrity_fault",
        }

    memberships = [
        _proportional_membership_comparison(left, right)
        for left, right in zip(previous_masks, current_masks, strict=True)
    ]
    cores = [int(item["core"]) for item in memberships]
    if any(core and not _contiguous_proportional_mask(core, rung_count) for core in cores[:2]):
        return {
            "interpretation": "schema-8 what-if",
            "outcome": "evidence_integrity_fault",
        }

    def component_nomination(
        component: dict[str, int],
    ) -> tuple[int, int, Fraction]:
        rung = int(component["first_rung"]) + (int(component["member_count"]) - 1) // 2
        p = _proportional_gain_at(p_start, p_top, rung_count, rung)
        return rung, p, _ratio_margin(p, p_start, p_top)

    directional_cores = cores[:2]

    def component_rank(component: dict[str, int]) -> tuple[int, int, Fraction, int]:
        _rung, p, margin = component_nomination(component)
        coverage = sum(
            (int(component["member_mask"]) & core).bit_count() for core in directional_cores
        )
        return int(component["member_count"]), coverage, margin, -p

    components = _split_components(cores[2])
    selected_joint = (
        max(components, key=component_rank)
        if components
        else {
            "member_mask": 0,
            "first_rung": 0,
            "member_count": 0,
        }
    )
    selected_count = int(selected_joint["member_count"])
    final_rung, fixed_p = (0, 0)
    if selected_count:
        final_rung, fixed_p, _margin = component_nomination(selected_joint)

    pooled = []
    common = []
    for direction in range(2):
        previous_pooled = [int(value) for value in previous["pooled_intervals"][direction]]
        current_pooled = [int(value) for value in current["pooled_intervals"][direction]]
        previous_common = [int(value) for value in previous["common_intervals"][direction]]
        current_common = [int(value) for value in current["common_intervals"][direction]]
        pooled_overlap = _proportional_closed_intersection(previous_pooled, current_pooled)
        conservative = _proportional_closed_intersection(previous_common, current_common)
        pooled.append(
            {
                "previous": previous_pooled,
                "current": current_pooled,
                "overlap": pooled_overlap,
            }
        )
        common.append(
            {
                "previous": previous_common,
                "current": current_common,
                "conservative": conservative,
                "nonempty": conservative[0] <= conservative[1],
            }
        )

    validity = []
    coverage = []
    for direction in range(2):
        previous_mask = previous_masks[direction]
        current_mask = current_masks[direction]
        core = cores[direction]
        if previous_mask == 0 or current_mask == 0:
            validity.append(0)
        elif previous_mask.bit_count() < 3 or current_mask.bit_count() < 3:
            validity.append(1)
        elif (
            core.bit_count() < 3
            or pooled[direction]["overlap"][0] > pooled[direction]["overlap"][1]
        ):
            validity.append(2)
        elif core & (1 << final_rung):
            validity.append(3)
        else:
            validity.append(4)
        coverage.append(
            _proportional_core_coverage(core, final_rung, p_start, p_top, rung_count)
            if selected_count
            else {
                "coverage": "transferred",
                "signed_rung_distance": 0,
                "gain_ratio": [0, 0],
            }
        )

    reason_mask = (
        (int(cores[0].bit_count() < 3) << 0)
        | (int(cores[1].bit_count() < 3) << 1)
        | (int(cores[2] == 0) << 2)
        | (int(pooled[0]["overlap"][0] > pooled[0]["overlap"][1]) << 3)
        | (int(pooled[1]["overlap"][0] > pooled[1]["overlap"][1]) << 4)
    )
    return {
        "interpretation": "schema-8 what-if",
        "firmware_authoritative": False,
        "outcome": "complete" if selected_count else "reproduction_mismatch",
        "reason_mask": reason_mask,
        "previous_candidate": previous,
        "current_candidate": current,
        "memberships": memberships,
        "core_masks": cores,
        "selected_joint_mask": int(selected_joint["member_mask"]),
        "selected_first_rung": int(selected_joint["first_rung"]),
        "selected_member_count": selected_count,
        "previous_provisional_p": int(previous["nominated_p"]),
        "current_provisional_p": int(current["nominated_p"]),
        "fixed_p": fixed_p,
        "directional_validity": validity,
        "reduced_margin": 0 < selected_count <= 2,
        "coverage": coverage,
        "pooled": pooled,
        "common": common,
        "firmware_conservative_common": [item["conservative"] for item in common],
    }
