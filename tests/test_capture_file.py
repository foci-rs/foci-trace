import io
import struct

from foci_trace.capture_file import CaptureReader, CaptureWriter


def test_capture_round_trip_firmware_frame() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", "ABC", b"meta")
    writer.write_firmware_frame(0, 120, b"frame")

    reader = CaptureReader(io.BytesIO(output.getvalue()))
    records = list(reader.records())

    assert reader.truncated_tail_bytes == 0
    assert records[0].record_type == 1
    assert records[0].payload.endswith(b"meta")
    assert records[1].record_type == 2
    assert records[1].payload == b"frame"


def test_unknown_record_type_is_skipped() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_firmware_frame(0, 120, b"frame")
    raw = bytearray(output.getvalue())
    header_len = 32
    insert = struct.pack("<IBBH H H Q I Q".replace(" ", ""), 0x52525446, 1, 99, 32, 0, 0, 110, 3, 0)
    raw = raw[:32] + insert + b"abc" + raw[32:]

    records = list(CaptureReader(io.BytesIO(raw)).records())

    assert len(records) == 1
    assert records[0].payload == b"frame"
    assert header_len == 32


def test_truncated_final_header_is_tolerated() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_firmware_frame(0, 120, b"frame")
    raw = bytearray(output.getvalue())
    raw += struct.pack("<IB", 0x52525446, 1)

    reader = CaptureReader(io.BytesIO(raw))
    records = list(reader.records())

    assert len(records) == 1
    assert records[0].payload == b"frame"
    assert reader.truncated_tail_bytes == 5


def test_truncated_final_record_is_tolerated() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_firmware_frame(0, 120, b"frame")
    raw = bytearray(output.getvalue())
    truncated_header = struct.pack(
        "<IBBH H H Q I Q".replace(" ", ""), 0x52525446, 1, 2, 32, 0, 0, 130, 8, 0
    )
    raw += truncated_header + b"ab"

    reader = CaptureReader(io.BytesIO(raw))
    records = list(reader.records())

    assert len(records) == 1
    assert records[0].payload == b"frame"
    assert reader.truncated_tail_bytes == 34
