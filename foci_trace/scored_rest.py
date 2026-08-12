"""Shared scored-rest suffix selection."""

from __future__ import annotations

VELOCITY_SUFFIX_LENGTHS = (64, 128, 256)

REST_STRUCTURAL_USABLE = 0
REST_STRUCTURAL_RESOLVED_SLOPE = 3

REST_DISPOSITION_STATISTICAL_ACCEPTANCE = 0
REST_DISPOSITION_ONE_COUNT_FALLBACK_ACCEPTANCE = 1
REST_DISPOSITION_REJECTED = 2


def select_causal_rest_suffixes(
    structural_codes: list[int],
    intervals_q: list[list[int]],
    interval_present_mask: int,
    *,
    one_count_fallback: bool,
) -> dict[str, int]:
    """Mirror firmware's causal scored-rest suffix selector."""

    usable_mask = 0
    zero_compatible_mask = 0
    selected_mask = 0
    selected_suffix_len = 0
    chain_open = True

    for index, suffix_len in enumerate(VELOCITY_SUFFIX_LENGTHS):
        bit = 1 << index
        code = structural_codes[index]
        low, high = intervals_q[index]
        usable = code == REST_STRUCTURAL_USABLE
        if usable:
            usable_mask |= bit
        zero_compatible = usable and bool(interval_present_mask & bit) and low <= 0 <= high
        if zero_compatible:
            zero_compatible_mask |= bit
        contradicts_rest = (
            bool(interval_present_mask & bit) and (high < 0 or low > 0)
        ) or code == REST_STRUCTURAL_RESOLVED_SLOPE
        if chain_open and zero_compatible:
            selected_mask = bit
            selected_suffix_len = suffix_len
        elif contradicts_rest:
            chain_open = False

    if selected_mask != 0:
        disposition = REST_DISPOSITION_STATISTICAL_ACCEPTANCE
    elif one_count_fallback:
        disposition = REST_DISPOSITION_ONE_COUNT_FALLBACK_ACCEPTANCE
    else:
        disposition = REST_DISPOSITION_REJECTED

    return {
        "usable_mask": usable_mask,
        "zero_compatible_mask": zero_compatible_mask,
        "selected_mask": selected_mask,
        "selected_suffix_len": selected_suffix_len,
        "disposition": disposition,
    }
