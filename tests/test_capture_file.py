import io
import struct

from foci_trace.capture_file import CaptureReader, CaptureWriter


def test_capture_round_trip_firmware_frame() -> None:
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_device_table(0, 100, "1-1", "FOCI OpenFFBoard (trace)", "ABC", b"meta")
    writer.write_firmware_frame(0, 120, b"frame")

    records = list(CaptureReader(io.BytesIO(output.getvalue())).records())

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
