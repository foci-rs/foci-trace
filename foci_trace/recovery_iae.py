"""Reversal recovery-IAE metric computed from a critical-frame trace.

After a commanded velocity reversal the controller must recover tracking; the
integrated absolute velocity error over a fixed window after the reversal is a
single, direction-resolved number for how well a given gain tracks. Swept across
velocity-P it forms a U-curve (too-low P sluggish, too-high P oscillatory) whose
trough is the robust response band this feeds into.

Reversals are timed from the commanded ``target_velocity`` sign flip, and the
integrand is ``velocity_error`` (command minus true encoder velocity). The raw
``pid_velocity_actual`` readback is deliberately NOT used: the velocity-feedback
readback has a known reliability problem -- a ~2x scaling in the velocity path --
so the reliable measured velocity is the reconstruction
``target_velocity - velocity_error``. Because ``velocity_error`` is the on-device
tracking error the loop already acts on, integrating it needs no reconstruction.

The metric operates on any sequence of samples exposing ``timestamp_ticks``,
``target_velocity``, and ``velocity_error`` -- fields the 1 kHz critical frame
carries -- so it needs nothing from the firmware beyond the existing ``trace``
stream. Units are velocity-error counts times seconds (comparable across gains
at a fixed velocity), not physical units.

Note: firmware-driven diagnostic captures (e.g. hardware characterization runs)
drive the motor without Klipper STEP/DIR, so ``target_velocity`` reads ~0 there.
This metric is for real Klipper reversing moves, where the command is populated.
"""

from __future__ import annotations

import io
import statistics
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from foci_trace.capture_file import RECORD_TYPE_FIRMWARE_FRAME, CaptureReader
from foci_trace.protocol import FRAME_TYPE_CRITICAL, DecodeError, decode_frame

# F407 TIM5 timebase; matches analyze.DEFAULT_CLOCK_HZ. Trace timestamps are in
# these ticks, so it is the divisor turning tick spans into seconds.
DEFAULT_CLOCK_HZ = 84_000_000

SourceKey = tuple[int, int]

# Minimal sample carrying only the fields the metric needs, extracted from the
# 1 kHz critical frame.
TraceSample = namedtuple("TraceSample", ["timestamp_ticks", "target_velocity", "velocity_error"])


class _Sample(Protocol):
    timestamp_ticks: int
    target_velocity: int
    velocity_error: int


@dataclass(frozen=True)
class RecoveryIaeEvent:
    """One reversal and its post-reversal integrated absolute error."""

    reversal_index: int
    direction: int  # +1 = command entered positive, -1 = entered negative
    iae: float  # integral of |velocity_error| dt over the window (counts * s)
    samples: int  # critical samples integrated within the window


@dataclass(frozen=True)
class RecoveryIaeSummary:
    """Aggregate of the per-reversal IAE, overall and per direction."""

    count: int
    median: float | None
    median_positive: float | None
    median_negative: float | None


def _sign(value: int, deadband: int) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def _integrate_window(
    samples: Sequence[_Sample], start: int, clock_hz: int, window_s: float
) -> tuple[float, int]:
    window_ticks = window_s * clock_hz
    origin = samples[start].timestamp_ticks
    total = 0.0
    integrated = 0
    for index in range(start, len(samples) - 1):
        if samples[index].timestamp_ticks - origin > window_ticks:
            break
        dt_s = (samples[index + 1].timestamp_ticks - samples[index].timestamp_ticks) / clock_hz
        total += abs(samples[index].velocity_error) * dt_s
        integrated += 1
    return total, integrated


def recovery_iae_events(
    samples: Sequence[_Sample],
    clock_hz: int,
    window_s: float,
    deadband: int = 0,
) -> list[RecoveryIaeEvent]:
    """Locate command reversals and integrate |velocity_error| after each.

    A reversal is a sample whose commanded-velocity sign is the opposite nonzero
    sign of the last nonzero-sign sample; the near-zero dwell between directions
    is skipped via ``deadband``. ``samples`` must be time-ordered.
    """
    events: list[RecoveryIaeEvent] = []
    previous_sign = 0
    for index, sample in enumerate(samples):
        current_sign = _sign(sample.target_velocity, deadband)
        if current_sign != 0 and previous_sign != 0 and current_sign != previous_sign:
            iae, integrated = _integrate_window(samples, index, clock_hz, window_s)
            events.append(RecoveryIaeEvent(index, current_sign, iae, integrated))
        if current_sign != 0:
            previous_sign = current_sign
    return events


def recovery_iae_summary(events: Sequence[RecoveryIaeEvent]) -> RecoveryIaeSummary:
    """Median IAE overall and split by reversal direction."""

    def _median(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    everything = [event.iae for event in events]
    positive = [event.iae for event in events if event.direction == 1]
    negative = [event.iae for event in events if event.direction == -1]
    return RecoveryIaeSummary(
        count=len(events),
        median=_median(everything),
        median_positive=_median(positive),
        median_negative=_median(negative),
    )


def read_critical_samples(
    source: str | Path | io.BufferedIOBase,
) -> dict[SourceKey, list[TraceSample]]:
    """Extract time-ordered critical-frame samples per trace source from a capture.

    ``source`` is a capture-file path or an already-open binary stream. Frames
    that fail to decode are skipped (matching ``analyze_capture``).
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            return _read_critical_samples(handle)
    return _read_critical_samples(source)


def _read_critical_samples(stream: io.BufferedIOBase) -> dict[SourceKey, list[TraceSample]]:
    by_source: dict[SourceKey, list[TraceSample]] = {}
    for record in CaptureReader(stream).records():
        if record.record_type != RECORD_TYPE_FIRMWARE_FRAME:
            continue
        try:
            frame = decode_frame(record.payload)
        except DecodeError:
            continue
        header = frame.header
        if header.frame_type != FRAME_TYPE_CRITICAL:
            continue
        by_source.setdefault((header.source_kind, header.source_index), []).append(
            TraceSample(
                header.timestamp_ticks,
                int(frame.payload["target_velocity"]),
                int(frame.payload["velocity_error"]),
            )
        )
    for samples in by_source.values():
        samples.sort(key=lambda sample: sample.timestamp_ticks)
    return by_source


def recovery_iae_from_capture(
    source: str | Path | io.BufferedIOBase,
    window_s: float,
    clock_hz: int = DEFAULT_CLOCK_HZ,
    deadband: int = 0,
) -> dict[SourceKey, RecoveryIaeSummary]:
    """Recovery-IAE summary per trace source for a whole capture file."""
    return {
        source_key: recovery_iae_summary(recovery_iae_events(samples, clock_hz, window_s, deadband))
        for source_key, samples in read_critical_samples(source).items()
    }
