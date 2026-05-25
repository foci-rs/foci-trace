"""Pure CoreXY contour analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MotorSample:
    """Host-timestamped command and feedback sample for one CoreXY motor."""

    host_recv_ns: int
    commanded: float
    actual: float


@dataclass(frozen=True)
class PairedMotorSample:
    """Nearest-neighbor pair of plus and minus CoreXY motor samples."""

    host_recv_ns: int
    plus: MotorSample
    minus: MotorSample


def corexy_to_xy(motor_plus: float, motor_minus: float) -> tuple[float, float]:
    """Convert CoreXY plus/minus motor coordinates to toolhead X/Y coordinates."""
    return ((motor_plus + motor_minus) / 2.0, (motor_plus - motor_minus) / 2.0)


def project_path_error(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    commanded: tuple[float, float],
    actual: tuple[float, float],
) -> tuple[float, float]:
    """Project commanded-minus-actual error onto along-path and cross-path axes."""
    path_x = end[0] - start[0]
    path_y = end[1] - start[1]
    path_len = math.hypot(path_x, path_y)
    if path_len == 0.0:
        return (0.0, 0.0)

    unit_x = path_x / path_len
    unit_y = path_y / path_len
    error_x = commanded[0] - actual[0]
    error_y = commanded[1] - actual[1]

    along = error_x * unit_x + error_y * unit_y
    cross = error_x * -unit_y + error_y * unit_x
    return (along, cross)


def pair_motor_samples(
    plus_samples: Sequence[MotorSample],
    minus_samples: Sequence[MotorSample],
    *,
    tolerance_ns: int,
) -> list[PairedMotorSample]:
    """Pair plus and minus samples by nearest host receive timestamp."""
    if not minus_samples:
        return []

    sorted_plus = sorted(plus_samples, key=lambda sample: sample.host_recv_ns)
    sorted_minus = sorted(minus_samples, key=lambda sample: sample.host_recv_ns)
    pairs: list[PairedMotorSample] = []
    minus_index = 0
    for plus in sorted_plus:
        if minus_index >= len(sorted_minus):
            break

        while (
            minus_index + 1 < len(sorted_minus)
            and abs(sorted_minus[minus_index + 1].host_recv_ns - plus.host_recv_ns)
            <= abs(sorted_minus[minus_index].host_recv_ns - plus.host_recv_ns)
        ):
            minus_index += 1

        minus = sorted_minus[minus_index]
        delta = abs(plus.host_recv_ns - minus.host_recv_ns)
        if delta > tolerance_ns:
            if minus.host_recv_ns < plus.host_recv_ns:
                minus_index += 1
            continue

        pairs.append(
            PairedMotorSample(
                host_recv_ns=(plus.host_recv_ns + minus.host_recv_ns) // 2,
                plus=plus,
                minus=minus,
            )
        )
        minus_index += 1

    return pairs


def contour_pair_groups(
    pairs: Sequence[PairedMotorSample],
    *,
    gap_ns: int,
    min_delta: float,
) -> list[list[PairedMotorSample]]:
    """Group contiguous paired samples that contain enough commanded toolhead motion."""
    groups: list[list[PairedMotorSample]] = []
    current: list[PairedMotorSample] = []
    previous = pairs[0] if pairs else None

    for pair in pairs[1:]:
        if previous is not None and pair.host_recv_ns - previous.host_recv_ns > gap_ns:
            _append_pair_group(groups, current, min_delta)
            current = []
        elif previous is not None and _commanded_xy(previous) != _commanded_xy(pair):
            if not current:
                current.append(previous)
            current.append(pair)
        elif current:
            _append_pair_group(groups, current, min_delta)
            current = []
        previous = pair

    _append_pair_group(groups, current, min_delta)
    return groups


def rms(values: Iterable[float]) -> float:
    """Return the root-mean-square magnitude for the provided values."""
    samples = list(values)
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def classify_phase(
    along_rms: float,
    cross_rms: float,
    *,
    trace_gap: bool,
    post_discriminator: str | None,
) -> str:
    """Classify a contour phase from along-path and cross-path RMS errors."""
    if trace_gap:
        return "inconclusive"
    if post_discriminator == "loop_active":
        return "endpoint_loop_active"
    if along_rms != 0.0 and cross_rms <= along_rms * 0.25:
        return "along_path_lag"
    if (along_rms == 0.0 and cross_rms != 0.0) or (
        along_rms != 0.0 and cross_rms >= along_rms * 0.5
    ):
        return "cross_path_contour"
    if post_discriminator == "settling_possible":
        return "passive_settling_plausible"
    return "mixed"


def _append_pair_group(
    groups: list[list[PairedMotorSample]],
    pairs: list[PairedMotorSample],
    min_delta: float,
) -> None:
    if len(pairs) < 2:
        return

    start = _commanded_xy(pairs[0])
    end = _commanded_xy(pairs[-1])
    if math.hypot(end[0] - start[0], end[1] - start[1]) >= min_delta:
        groups.append(pairs)


def _commanded_xy(pair: PairedMotorSample) -> tuple[float, float]:
    return corexy_to_xy(pair.plus.commanded, pair.minus.commanded)
