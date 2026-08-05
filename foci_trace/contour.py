"""Pure CoreXY contour analysis helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ContourSegmentReport:
    """Contour-error summary for one paired CoreXY motion segment."""

    segment_index: int
    samples: int
    duration_s: float
    delta_x: float
    delta_y: float
    along_rms: float
    along_peak: float
    cross_rms: float
    cross_peak: float
    final_along: float
    final_cross: float
    path_classification: str
    classification: str
    position_trace_gap: bool
    support_trace_gap: bool
    post_discriminator: str | None
    max_pair_delta_ms: float


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

        while minus_index + 1 < len(sorted_minus) and abs(
            sorted_minus[minus_index + 1].host_recv_ns - plus.host_recv_ns
        ) <= abs(sorted_minus[minus_index].host_recv_ns - plus.host_recv_ns):
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


def contour_segment_report(
    segment_index: int,
    pairs: Sequence[PairedMotorSample],
    *,
    position_trace_gap: bool,
    support_trace_gap: bool,
    post_discriminator: str | None,
) -> ContourSegmentReport:
    """Build a contour report for one contiguous paired CoreXY segment."""
    if not pairs:
        return ContourSegmentReport(
            segment_index=segment_index,
            samples=0,
            duration_s=0.0,
            delta_x=0.0,
            delta_y=0.0,
            along_rms=0.0,
            along_peak=0.0,
            cross_rms=0.0,
            cross_peak=0.0,
            final_along=0.0,
            final_cross=0.0,
            path_classification=classify_path_shape(0.0, 0.0),
            classification=classify_phase(
                0.0,
                0.0,
                trace_gap=position_trace_gap,
                post_discriminator=post_discriminator,
            ),
            position_trace_gap=position_trace_gap,
            support_trace_gap=support_trace_gap,
            post_discriminator=post_discriminator,
            max_pair_delta_ms=0.0,
        )

    commanded_points = [_commanded_xy(pair) for pair in pairs]
    actual_points = [corexy_to_xy(pair.plus.actual, pair.minus.actual) for pair in pairs]
    start = commanded_points[0]
    end = commanded_points[-1]
    projections = [
        project_path_error(start=start, end=end, commanded=commanded, actual=actual)
        for commanded, actual in zip(commanded_points, actual_points, strict=True)
    ]
    along_values = [along for along, _cross in projections]
    cross_values = [cross for _along, cross in projections]
    along_rms = rms(along_values)
    cross_rms = rms(cross_values)
    max_pair_delta_ms = (
        max(abs(pair.plus.host_recv_ns - pair.minus.host_recv_ns) for pair in pairs) / 1_000_000.0
    )

    return ContourSegmentReport(
        segment_index=segment_index,
        samples=len(pairs),
        duration_s=(pairs[-1].host_recv_ns - pairs[0].host_recv_ns) / 1_000_000_000.0,
        delta_x=end[0] - start[0],
        delta_y=end[1] - start[1],
        along_rms=along_rms,
        along_peak=max(abs(value) for value in along_values),
        cross_rms=cross_rms,
        cross_peak=max(abs(value) for value in cross_values),
        final_along=along_values[-1],
        final_cross=cross_values[-1],
        path_classification=classify_path_shape(along_rms, cross_rms),
        classification=classify_phase(
            along_rms,
            cross_rms,
            trace_gap=position_trace_gap,
            post_discriminator=post_discriminator,
        ),
        position_trace_gap=position_trace_gap,
        support_trace_gap=support_trace_gap,
        post_discriminator=post_discriminator,
        max_pair_delta_ms=max_pair_delta_ms,
    )


def rms(values: Iterable[float]) -> float:
    """Return the root-mean-square magnitude for the provided values."""
    samples = list(values)
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def classify_path_shape(along_rms: float, cross_rms: float) -> str:
    """Classify geometric path error without trace-quality or settling caveats."""
    if along_rms != 0.0 and cross_rms <= along_rms * 0.25:
        return "along_path_lag"
    if (along_rms == 0.0 and cross_rms != 0.0) or (
        along_rms != 0.0 and cross_rms >= along_rms * 0.5
    ):
        return "cross_path_contour"
    return "mixed"


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
    path_classification = classify_path_shape(along_rms, cross_rms)
    if path_classification != "mixed":
        return path_classification
    if post_discriminator == "settling_possible":
        return "passive_settling_plausible"
    return path_classification


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
