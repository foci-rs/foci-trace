"""Live summary counters for FOCI trace captures."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import (
    FRAME_TYPE_CRITICAL,
    FRAME_TYPE_EVENT,
    FRAME_TYPE_STATUS,
    FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE,
    Frame,
)

EVENT_SPI_SAMPLE_DURATION = 11


@dataclass
class DeviceSummary:
    """Per-device live counters."""

    frames: int = 0
    gaps: int = 0
    last_sequence: int | None = None
    max_position_error: int = 0
    max_velocity_error: int = 0
    voltage_limited: int = 0
    latest_status_flags: int = 0
    spi_errors: int = 0
    queue_high_water: int = 0
    endpoint_pressure: int = 0
    first_spi_cycles: int = 0
    max_spi_cycles: int = 0
    stage_b_regions: int = 0
    stage_b_fragments: int = 0
    stage_b_nominated_p: int = 0
    stage_b_outcome: int | None = None
    stage_b_cause: int | None = None
    stage_b_recovery_unavailable: bool = False
    stage_b_directional_validity: tuple[int, int] | None = None
    stage_b_reduced_margin: bool = False
    velocity_integral_rungs: int = 0
    velocity_integral_outcome: int | None = None
    velocity_integral_bookend_mask: int | None = None
    velocity_integral_rest_boundary: tuple[int, int] | None = None
    rung_origin_recoveries: int = 0
    contained_recoveries: int = 0
    outer_gap_max_us: int | None = None
    velocity_current_records: int = 0
    velocity_current_contacts: int = 0
    velocity_clip_uncertain: int = 0


class LiveSummary:
    """Accumulate concise operator-facing counters."""

    def __init__(self) -> None:
        self._devices: dict[int, DeviceSummary] = {}

    def observe(self, device_index: int, _host_recv_ns: int, frame: Frame) -> None:
        """Observe one decoded frame."""
        device = self._devices.setdefault(device_index, DeviceSummary())
        device.frames += 1
        seq = frame.header.sequence
        if device.last_sequence is not None:
            expected = (device.last_sequence + 1) & 0xFFFF_FFFF
            if seq != expected:
                device.gaps += 1
        device.last_sequence = seq
        if frame.header.frame_type == FRAME_TYPE_CRITICAL:
            device.max_position_error = max(
                device.max_position_error, abs(int(frame.payload["position_error"]))
            )
            device.max_velocity_error = max(
                device.max_velocity_error, abs(int(frame.payload["velocity_error"]))
            )
            if frame.payload["interim_foc_uq_ud"] != frame.payload["interim_foc_uq_ud_limited"]:
                device.voltage_limited += 1
        elif frame.header.frame_type == FRAME_TYPE_STATUS:
            device.latest_status_flags = int(frame.payload["status_flags"])
            device.spi_errors = int(frame.payload["spi_error_count"])
            device.queue_high_water = int(frame.payload["usb_trace_queue_high_water"])
            device.endpoint_pressure = int(frame.payload["endpoint_not_ready_count"])
        elif (
            frame.header.frame_type == FRAME_TYPE_EVENT
            and int(frame.payload["event_id"]) == EVENT_SPI_SAMPLE_DURATION
            and int(frame.payload["arg_count"]) >= 2
        ):
            device.first_spi_cycles = int(frame.payload["args"][0])
            device.max_spi_cycles = int(frame.payload["args"][1])
        elif frame.header.frame_type == FRAME_TYPE_VELOCITY_SWEEP_EVIDENCE:
            kind = frame.payload.get("record_kind")
            if kind == "directional_region":
                if int(frame.payload["region_kind"]) == 0:
                    device.stage_b_regions += 1
                else:
                    device.stage_b_fragments += 1
            elif kind == "handoff":
                device.stage_b_nominated_p = int(frame.payload["nominated_p"])
            elif kind == "stage_b_terminal":
                device.stage_b_outcome = int(frame.payload["outcome"])
                device.stage_b_cause = int(frame.payload["cause"])
                device.stage_b_recovery_unavailable = bool(
                    int(frame.payload["recovery_unavailable"])
                )
            elif kind == "reproduction_v4_header":
                device.stage_b_nominated_p = int(frame.payload["final_p"])
                device.stage_b_directional_validity = (
                    int(frame.payload["forward_validity"]),
                    int(frame.payload["reverse_validity"]),
                )
                device.stage_b_reduced_margin = bool(frame.payload["reduced_margin"])
            elif kind == "integral_plan_core" and "directional_validity" in frame.payload:
                validity = frame.payload["directional_validity"]
                device.stage_b_directional_validity = (
                    int(validity[0]),
                    int(validity[1]),
                )
                device.stage_b_reduced_margin = bool(frame.payload["reduced_margin"])
            elif kind == "integral_rung":
                device.velocity_integral_rungs += 1
            elif kind == "integral_terminal":
                device.velocity_integral_outcome = int(frame.payload["outcome"])
                if "bookend_available_mask" in frame.payload:
                    device.velocity_integral_bookend_mask = int(
                        frame.payload["bookend_available_mask"]
                    )
            elif kind == "integral_rest_boundary":
                device.velocity_integral_rest_boundary = (
                    int(frame.payload["positive_rung_index"]),
                    int(frame.payload["slot"]),
                )
            elif kind == "rung_origin_recovery":
                device.rung_origin_recoveries += 1
                if int(frame.payload["outcome"]) >= 2:
                    device.contained_recoveries += 1
            elif kind == "outer_safety_timing":
                device.outer_gap_max_us = int(frame.payload["max_realized_gap_us"])
            elif kind == "velocity_primitive_current":
                device.velocity_current_records += 1
                if int(frame.payload["contact"]) != 0:
                    device.velocity_current_contacts += 1
                if bool(frame.payload["moving_clip_uncertain"]):
                    device.velocity_clip_uncertain += 1

    def render(self) -> str:
        """Render one compact summary line."""
        parts = []
        for device_index in sorted(self._devices):
            item = self._devices[device_index]
            parts.append(
                "dev{idx} frames={frames} gaps={gaps} pos_err={pos} vel_err={vel} "
                "voltage_limited={limited} status=0x{status:08x} spi_errors={spi} "
                "qhi={qhi} endpoint_pressure={endpoint_pressure} "
                "first_spi_cycles={first_spi} max_spi_cycles={max_spi} "
                "stage_b_regions={regions} stage_b_fragments={fragments} "
                "stage_b_p={stage_b_p} stage_b_outcome={stage_b_outcome} "
                "stage_b_cause={stage_b_cause} "
                "stage_b_recovery_unavailable={stage_b_recovery_unavailable} "
                "stage_b_validity={stage_b_validity} "
                "stage_b_reduced_margin={stage_b_reduced_margin} "
                "velocity_integral_rungs={integral_rungs} "
                "velocity_integral_outcome={integral_outcome} "
                "velocity_integral_bookends={integral_bookends} "
                "velocity_integral_rest_boundary={integral_rest_boundary} "
                "recoveries={recoveries} contained_recoveries={contained_recoveries} "
                "current_records={current_records} current_contacts={current_contacts} "
                "clip_uncertain={clip_uncertain}".format(
                    idx=device_index,
                    frames=item.frames,
                    gaps=item.gaps,
                    pos=item.max_position_error,
                    vel=item.max_velocity_error,
                    limited=item.voltage_limited,
                    status=item.latest_status_flags,
                    spi=item.spi_errors,
                    qhi=item.queue_high_water,
                    endpoint_pressure=item.endpoint_pressure,
                    first_spi=item.first_spi_cycles,
                    max_spi=item.max_spi_cycles,
                    regions=item.stage_b_regions,
                    fragments=item.stage_b_fragments,
                    stage_b_p=item.stage_b_nominated_p,
                    stage_b_outcome=(
                        "none" if item.stage_b_outcome is None else item.stage_b_outcome
                    ),
                    stage_b_cause=("none" if item.stage_b_cause is None else item.stage_b_cause),
                    stage_b_recovery_unavailable=str(item.stage_b_recovery_unavailable).lower(),
                    stage_b_validity=(
                        "none"
                        if item.stage_b_directional_validity is None
                        else ",".join(str(value) for value in item.stage_b_directional_validity)
                    ),
                    stage_b_reduced_margin=str(item.stage_b_reduced_margin).lower(),
                    integral_rungs=item.velocity_integral_rungs,
                    integral_outcome=(
                        "none"
                        if item.velocity_integral_outcome is None
                        else item.velocity_integral_outcome
                    ),
                    integral_bookends=(
                        "none"
                        if item.velocity_integral_bookend_mask is None
                        else f"0x{item.velocity_integral_bookend_mask:02x}"
                    ),
                    integral_rest_boundary=(
                        "none"
                        if item.velocity_integral_rest_boundary is None
                        else ":".join(str(value) for value in item.velocity_integral_rest_boundary)
                    ),
                    recoveries=item.rung_origin_recoveries,
                    contained_recoveries=item.contained_recoveries,
                    current_records=item.velocity_current_records,
                    current_contacts=item.velocity_current_contacts,
                    clip_uncertain=item.velocity_clip_uncertain,
                )
            )
            if item.outer_gap_max_us is not None:
                maximum = item.outer_gap_max_us
                parts[-1] += (
                    f" outer_gap_max_us={maximum}"
                    f" outer_gap_headroom_us={20_000 - maximum}"
                    f" outer_gap_reopen={'true' if maximum > 15_000 else 'false'}"
                )
        return " | ".join(parts)
