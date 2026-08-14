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


def test_truncated_final_header_is_tolerated() -> None:
    # A SIGTERM-terminated recorder can also cut the final record mid-HEADER:
    # fewer than RECORD_HEADER_LEN bytes follow the last complete record. The
    # reader must keep every complete record and stop cleanly rather than
    # raising on the short header (mirrors the mid-payload case below).
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_firmware_frame(0, 120, b"frame")
    raw = bytearray(output.getvalue())
    raw += struct.pack("<IB", 0x52525446, 1)  # 5 bytes of a 32-byte record header

    records = list(CaptureReader(io.BytesIO(raw)).records())

    assert len(records) == 1
    assert records[0].payload == b"frame"


def test_truncated_final_record_is_tolerated() -> None:
    # A SIGTERM-terminated recorder can cut the final record mid-write: a valid
    # header claiming more payload bytes than actually follow. The reader must
    # keep every complete record and stop cleanly rather than discarding them.
    output = io.BytesIO()
    writer = CaptureWriter(output, start_unix_ns=10, start_mono_ns=20)
    writer.write_firmware_frame(0, 120, b"frame")
    raw = bytearray(output.getvalue())
    truncated_header = struct.pack(
        "<IBBH H H Q I Q".replace(" ", ""), 0x52525446, 1, 2, 32, 0, 0, 130, 8, 0
    )
    raw += truncated_header + b"ab"  # header claims 8 payload bytes; only 2 present

    records = list(CaptureReader(io.BytesIO(raw)).records())

    assert len(records) == 1
    assert records[0].payload == b"frame"
