"""Commissioning trace event analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import (
    COMMISSION_STATUS_NAMES,
    EVENT_COMMISSION_DIAGNOSTIC_WARNING,
    EVENT_COMMISSION_ENGINE_START,
    EVENT_COMMISSION_ENGINE_TERMINAL,
    EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT,
    EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION,
    EVENT_COMMISSION_MECHANICAL_ID_RESPONSE,
    EVENT_COMMISSION_MECHANICAL_ID_RESULT,
    EVENT_COMMISSION_PHASE_END,
    EVENT_COMMISSION_PHASE_START,
    EVENT_COMMISSION_POSITION_TUNE_LIMITS,
    EVENT_COMMISSION_POSITION_TUNE_METRICS,
    EVENT_COMMISSION_POSITION_TUNE_RESULT,
    EVENT_COMMISSION_POSITION_TUNE_TARGET,
    EVENT_COMMISSION_RESTORE_DISPOSITION,
    EVENT_COMMISSION_RESTORE_ERROR_DRIFT,
    EVENT_COMMISSION_RESTORE_RESULT,
    EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN,
    EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS,
    EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS,
    EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT,
    FRAME_TYPE_EVENT,
    PHASE_NAMES,
    SUBSYSTEM_COMMISSIONING,
    FrameHeader,
)

DEFAULT_CLOCK_HZ = 84_000_000
DEFAULT_BOUNDARY_TOLERANCE_S = 0.002
CRITICAL_HZ = 1000
CONTROL_DETAIL_HZ = 500

ENGINE_KIND_NAMES = {
    1: "commission",
    2: "calibrate",
    3: "selftest",
    4: "outer_tune",
}

SUMMARY_FAMILIES = {
    EVENT_COMMISSION_MECHANICAL_ID_RESULT: ("mechanical_id_summary", "result"),
    EVENT_COMMISSION_MECHANICAL_ID_ATTEMPT: ("mechanical_id_summary", "attempt"),
    EVENT_COMMISSION_MECHANICAL_ID_RESPONSE: ("mechanical_id_summary", "response"),
    EVENT_COMMISSION_MECHANICAL_ID_CLASSIFICATION: (
        "mechanical_id_summary",
        "classification",
    ),
    EVENT_COMMISSION_VELOCITY_VALIDATE_RESULT: ("velocity_validate_summary", "result"),
    EVENT_COMMISSION_VELOCITY_VALIDATE_GAIN: ("velocity_validate_summary", "gain"),
    EVENT_COMMISSION_VELOCITY_VALIDATE_METRICS: ("velocity_validate_summary", "metrics"),
    EVENT_COMMISSION_VELOCITY_VALIDATE_LIMITS: ("velocity_validate_summary", "limits"),
    EVENT_COMMISSION_POSITION_TUNE_RESULT: ("position_tune_summary", "result"),
    EVENT_COMMISSION_POSITION_TUNE_TARGET: ("position_tune_summary", "target"),
    EVENT_COMMISSION_POSITION_TUNE_METRICS: ("position_tune_summary", "metrics"),
    EVENT_COMMISSION_POSITION_TUNE_LIMITS: ("position_tune_summary", "limits"),
    EVENT_COMMISSION_RESTORE_RESULT: ("restore_summary", "result"),
    EVENT_COMMISSION_RESTORE_ERROR_DRIFT: ("restore_summary", "error_drift"),
    EVENT_COMMISSION_RESTORE_DISPOSITION: ("restore_summary", "disposition"),
}


@dataclass(frozen=True)
class CommissioningEvent:
    """One decoded commissioning event frame."""

    device_index: int
    timestamp_ticks: int
    source_index: int
    engine_kind: int
    engine_kind_name: str
    event_id: int
    event_name: str
    level_name: str
    args: list[int]


@dataclass
class CommissioningPhaseSpan:
    """One lifecycle phase span plus samples correlated by timestamp."""

    phase_id: int
    phase_name: str
    phase_seq: int
    start_ticks: int
    end_ticks: int | None = None
    status_code: int | None = None
    status_name: str | None = None
    critical_samples: int = 0
    control_detail_samples: int = 0
    caveats: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommissioningSummaryFragment:
    """One compact summary fragment emitted by shared commissioning code."""

    family: str
    fragment: str
    summary_seq: int
    value: int
    timestamp_ticks: int


@dataclass(frozen=True)
class CommissioningWarning:
    """One single-frame commissioning diagnostic warning."""

    warning_code: int
    detail: int
    timestamp_ticks: int


@dataclass
class CommissioningRun:
    """One commissioning, calibration, selftest, or tune engine run."""

    device_index: int
    source_index: int
    engine_kind: int
    engine_kind_name: str
    engine_seq: int | None
    start_ticks: int | None = None
    terminal_ticks: int | None = None
    result_code: int | None = None
    result: str | None = None
    events: list[CommissioningEvent] = field(default_factory=list)
    phases: list[CommissioningPhaseSpan] = field(default_factory=list)
    summaries: list[CommissioningSummaryFragment] = field(default_factory=list)
    warnings: list[CommissioningWarning] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


@dataclass
class CommissioningCaptureReport:
    """Commissioning analysis for a whole capture file."""

    runs: list[CommissioningRun]
    caveats: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Sample:
    device_index: int
    source_index: int
    timestamp_ticks: int


class CommissioningCaptureBuilder:
    """Collect commissioning events and correlate samples to phase spans."""

    def __init__(self) -> None:
        self._runs: list[CommissioningRun] = []
        self._active_runs: dict[tuple[int, int, int], CommissioningRun] = {}
        self._runs_by_seq: dict[tuple[int, int, int, int], CommissioningRun] = {}
        self._open_phases: dict[tuple[int, int, int, int], CommissioningPhaseSpan] = {}
        self._critical_samples: list[_Sample] = []
        self._control_detail_samples: list[_Sample] = []

    def observe_event(
        self,
        device_index: int,
        header: FrameHeader,
        payload: dict[str, Any],
    ) -> None:
        """Record one decoded event frame if it belongs to commissioning."""
        if header.frame_type != FRAME_TYPE_EVENT:
            return
        if payload.get("subsystem_id") != SUBSYSTEM_COMMISSIONING:
            return
        args = [int(value) for value in payload.get("args", [])]
        if len(args) < 2:
            return
        source_index = args[0]
        engine_kind = args[1]
        event = CommissioningEvent(
            device_index=device_index,
            timestamp_ticks=header.timestamp_ticks,
            source_index=source_index,
            engine_kind=engine_kind,
            engine_kind_name=_engine_kind_name(engine_kind),
            event_id=int(payload["event_id"]),
            event_name=str(payload.get("event_name", f"event_{payload['event_id']}")),
            level_name=str(payload.get("level_name", "unknown")),
            args=args,
        )
        if event.event_id == EVENT_COMMISSION_ENGINE_START:
            self._observe_engine_start(event)
        elif event.event_id == EVENT_COMMISSION_PHASE_START:
            self._observe_phase_start(event)
        elif event.event_id == EVENT_COMMISSION_PHASE_END:
            self._observe_phase_end(event)
        elif event.event_id == EVENT_COMMISSION_ENGINE_TERMINAL:
            self._observe_engine_terminal(event)
        elif event.event_id in SUMMARY_FAMILIES:
            self._observe_summary(event)
        elif event.event_id == EVENT_COMMISSION_DIAGNOSTIC_WARNING:
            self._observe_warning(event)

    def observe_critical(
        self,
        device_index: int,
        source_index: int,
        timestamp_ticks: int,
        payload: dict[str, Any],
    ) -> None:
        """Record one critical sample timestamp for later phase correlation."""
        if payload:
            self._critical_samples.append(_Sample(device_index, source_index, timestamp_ticks))

    def observe_control_detail(
        self,
        device_index: int,
        source_index: int,
        timestamp_ticks: int,
        payload: dict[str, Any],
    ) -> None:
        """Record one control-detail sample timestamp for later phase correlation."""
        if payload:
            self._control_detail_samples.append(
                _Sample(device_index, source_index, timestamp_ticks)
            )

    def finalize(self, clock_hz_by_device: dict[int, int]) -> CommissioningCaptureReport:
        """Finalize sample correlation and partial-capture caveats."""
        report_caveats: list[str] = []
        for run in self._runs:
            tolerance = _boundary_tolerance_ticks(clock_hz_by_device.get(run.device_index))
            if run.terminal_ticks is None:
                _add_caveat(run.caveats, "partial capture: missing engine_terminal")
            for phase in run.phases:
                if phase.end_ticks is None:
                    _add_caveat(
                        phase.caveats,
                        f"partial capture: missing phase_end for {phase.phase_name}",
                    )
                    _add_caveat(run.caveats, f"missing phase_end for {phase.phase_name}")
                phase.critical_samples = _count_samples(
                    self._critical_samples,
                    run.device_index,
                    run.source_index,
                    phase.start_ticks,
                    phase.end_ticks,
                    tolerance,
                )
                phase.control_detail_samples = _count_samples(
                    self._control_detail_samples,
                    run.device_index,
                    run.source_index,
                    phase.start_ticks,
                    phase.end_ticks,
                    tolerance,
                )
            report_caveats.extend(run.caveats)
        return CommissioningCaptureReport(self._runs, report_caveats)

    def _observe_engine_start(self, event: CommissioningEvent) -> None:
        engine_seq = event.args[3] if len(event.args) > 3 else None
        run = CommissioningRun(
            device_index=event.device_index,
            source_index=event.source_index,
            engine_kind=event.engine_kind,
            engine_kind_name=event.engine_kind_name,
            engine_seq=engine_seq,
            start_ticks=event.timestamp_ticks,
            events=[event],
        )
        self._runs.append(run)
        active_key = _active_key(event)
        self._active_runs[active_key] = run
        if engine_seq is not None:
            self._runs_by_seq[(*active_key, engine_seq)] = run

    def _observe_phase_start(self, event: CommissioningEvent) -> None:
        if len(event.args) < 4:
            return
        run = self._current_run(event)
        run.events.append(event)
        phase_id = event.args[2]
        phase = CommissioningPhaseSpan(
            phase_id=phase_id,
            phase_name=_phase_name(phase_id),
            phase_seq=event.args[3],
            start_ticks=event.timestamp_ticks,
        )
        run.phases.append(phase)
        self._open_phases[(*_active_key(event), phase_id)] = phase

    def _observe_phase_end(self, event: CommissioningEvent) -> None:
        if len(event.args) < 4:
            return
        run = self._current_run(event)
        run.events.append(event)
        phase_id = event.args[2]
        status_code = event.args[3]
        phase = self._open_phases.pop((*_active_key(event), phase_id), None)
        if phase is None:
            phase = CommissioningPhaseSpan(
                phase_id=phase_id,
                phase_name=_phase_name(phase_id),
                phase_seq=-1,
                start_ticks=event.timestamp_ticks,
            )
            run.phases.append(phase)
            _add_caveat(run.caveats, f"partial capture: missing phase_start for {phase.phase_name}")
        phase.end_ticks = event.timestamp_ticks
        phase.status_code = status_code
        phase.status_name = _status_name(status_code)

    def _observe_engine_terminal(self, event: CommissioningEvent) -> None:
        if len(event.args) < 4:
            return
        active_key = _active_key(event)
        engine_seq = event.args[3]
        run = self._runs_by_seq.get((*active_key, engine_seq)) or self._current_run(event)
        run.events.append(event)
        run.terminal_ticks = event.timestamp_ticks
        run.result_code = event.args[2]
        run.result = _status_name(event.args[2])
        self._active_runs.pop(active_key, None)

    def _observe_summary(self, event: CommissioningEvent) -> None:
        if len(event.args) < 4:
            return
        run = self._current_run(event)
        run.events.append(event)
        family, fragment = SUMMARY_FAMILIES[event.event_id]
        run.summaries.append(
            CommissioningSummaryFragment(
                family=family,
                fragment=fragment,
                summary_seq=event.args[2],
                value=event.args[3],
                timestamp_ticks=event.timestamp_ticks,
            )
        )

    def _observe_warning(self, event: CommissioningEvent) -> None:
        if len(event.args) < 4:
            return
        run = self._current_run(event)
        run.events.append(event)
        run.warnings.append(
            CommissioningWarning(
                warning_code=event.args[2],
                detail=event.args[3],
                timestamp_ticks=event.timestamp_ticks,
            )
        )

    def _current_run(self, event: CommissioningEvent) -> CommissioningRun:
        active_key = _active_key(event)
        if active_key in self._active_runs:
            return self._active_runs[active_key]
        run = CommissioningRun(
            device_index=event.device_index,
            source_index=event.source_index,
            engine_kind=event.engine_kind,
            engine_kind_name=event.engine_kind_name,
            engine_seq=None,
            start_ticks=None,
        )
        _add_caveat(run.caveats, "partial capture: missing engine_start")
        self._runs.append(run)
        self._active_runs[active_key] = run
        return run


def observe_event(
    builder: CommissioningCaptureBuilder,
    device_index: int,
    header: FrameHeader,
    payload: dict[str, Any],
) -> None:
    """Record one commissioning event on a builder."""
    builder.observe_event(device_index, header, payload)


def observe_critical(
    builder: CommissioningCaptureBuilder,
    device_index: int,
    source_index: int,
    timestamp_ticks: int,
    payload: dict[str, Any],
) -> None:
    """Record one critical sample on a builder."""
    builder.observe_critical(device_index, source_index, timestamp_ticks, payload)


def observe_control_detail(
    builder: CommissioningCaptureBuilder,
    device_index: int,
    source_index: int,
    timestamp_ticks: int,
    payload: dict[str, Any],
) -> None:
    """Record one control-detail sample on a builder."""
    builder.observe_control_detail(device_index, source_index, timestamp_ticks, payload)


def finalize(
    builder: CommissioningCaptureBuilder, clock_hz_by_device: dict[int, int]
) -> CommissioningCaptureReport:
    """Finalize a commissioning builder."""
    return builder.finalize(clock_hz_by_device)


def format_commissioning_report(report: CommissioningCaptureReport) -> str:
    """Format commissioning runs for operator review."""
    lines: list[str] = []
    for run in report.runs:
        result = run.result or "unknown"
        seq = "n/a" if run.engine_seq is None else str(run.engine_seq)
        lines.append(
            "commissioning dev{device} source={source} engine={engine} seq={seq} "
            "result={result}".format(
                device=run.device_index,
                source=run.source_index,
                engine=run.engine_kind_name,
                seq=seq,
                result=result,
            )
        )
        if run.start_ticks is not None:
            lines.append(f"  engine_start t={run.start_ticks}")
        for phase in run.phases:
            status = phase.status_name or "unknown"
            end = "open" if phase.end_ticks is None else str(phase.end_ticks)
            lines.append(
                "  phase {name} phase_start={start} phase_end={end} status={status} "
                "samples={critical} control_detail={detail}".format(
                    name=phase.phase_name,
                    start=phase.start_ticks,
                    end=end,
                    status=status,
                    critical=phase.critical_samples,
                    detail=phase.control_detail_samples,
                )
            )
            if phase.end_ticks is not None:
                lines.append(f"    phase_end t={phase.end_ticks}")
            for caveat in phase.caveats:
                lines.append(f"    caveat {caveat}")
        for summary in _group_summaries(run.summaries):
            fragments = ",".join(fragment["fragment"] for fragment in summary["fragments"])
            lines.append(
                "  {family} seq={seq} fragments={fragments}".format(
                    family=summary["family"],
                    seq=summary["summary_seq"],
                    fragments=fragments,
                )
            )
        for warning in run.warnings:
            lines.append(
                "  diagnostic_warning code={code} detail={detail}".format(
                    code=warning.warning_code,
                    detail=warning.detail,
                )
            )
        if run.terminal_ticks is not None:
            lines.append(f"  engine_terminal t={run.terminal_ticks} result={result}")
        for caveat in run.caveats:
            lines.append(f"  caveat {caveat}")
    return "\n".join(lines)


def commissioning_report_to_json(report: CommissioningCaptureReport) -> dict[str, Any]:
    """Convert commissioning analysis to JSON-compatible dictionaries."""
    return {
        "runs": [
            {
                "device_index": run.device_index,
                "source_index": run.source_index,
                "engine_kind": run.engine_kind_name,
                "engine_seq": run.engine_seq,
                "start_ticks": run.start_ticks,
                "terminal_ticks": run.terminal_ticks,
                "result_code": run.result_code,
                "result": run.result,
                "phases": [
                    {
                        "phase_id": phase.phase_id,
                        "phase": phase.phase_name,
                        "phase_seq": phase.phase_seq,
                        "start_ticks": phase.start_ticks,
                        "end_ticks": phase.end_ticks,
                        "status_code": phase.status_code,
                        "status": phase.status_name,
                        "critical_samples": phase.critical_samples,
                        "control_detail_samples": phase.control_detail_samples,
                        "caveats": list(phase.caveats),
                    }
                    for phase in run.phases
                ],
                "summaries": _group_summaries(run.summaries),
                "warnings": [
                    {
                        "warning_code": warning.warning_code,
                        "detail": warning.detail,
                        "timestamp_ticks": warning.timestamp_ticks,
                    }
                    for warning in run.warnings
                ],
                "caveats": list(run.caveats),
            }
            for run in report.runs
        ],
        "caveats": list(report.caveats),
    }


def _active_key(event: CommissioningEvent) -> tuple[int, int, int]:
    return (event.device_index, event.source_index, event.engine_kind)


def _engine_kind_name(engine_kind: int) -> str:
    return ENGINE_KIND_NAMES.get(engine_kind, f"unknown_{engine_kind}")


def _phase_name(phase_id: int) -> str:
    return PHASE_NAMES.get(phase_id, f"Phase{phase_id}")


def _status_name(status_code: int) -> str:
    return COMMISSION_STATUS_NAMES.get(status_code, f"status_{status_code}")


def _add_caveat(caveats: list[str], caveat: str) -> None:
    if caveat not in caveats:
        caveats.append(caveat)


def _boundary_tolerance_ticks(clock_hz: int | None) -> int:
    if clock_hz is None:
        return int(DEFAULT_CLOCK_HZ * DEFAULT_BOUNDARY_TOLERANCE_S)
    critical_period = clock_hz // CRITICAL_HZ
    control_detail_period = clock_hz // CONTROL_DETAIL_HZ
    return max(2 * critical_period, control_detail_period)


def _count_samples(
    samples: list[_Sample],
    device_index: int,
    source_index: int,
    start_ticks: int,
    end_ticks: int | None,
    tolerance_ticks: int,
) -> int:
    lower = start_ticks - tolerance_ticks
    upper = None if end_ticks is None else end_ticks + tolerance_ticks
    total = 0
    for sample in samples:
        if sample.device_index != device_index or sample.source_index != source_index:
            continue
        if sample.timestamp_ticks < lower:
            continue
        if upper is not None and sample.timestamp_ticks > upper:
            continue
        total += 1
    return total


def _group_summaries(summaries: list[CommissioningSummaryFragment]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[CommissioningSummaryFragment]] = {}
    for summary in summaries:
        groups.setdefault((summary.family, summary.summary_seq), []).append(summary)
    output: list[dict[str, Any]] = []
    for (family, summary_seq), fragments in sorted(groups.items()):
        output.append(
            {
                "family": family,
                "summary_seq": summary_seq,
                "fragments": [
                    {
                        "fragment": fragment.fragment,
                        "value": fragment.value,
                        "timestamp_ticks": fragment.timestamp_ticks,
                    }
                    for fragment in sorted(fragments, key=lambda item: item.fragment)
                ],
            }
        )
    return output
