"""Read and write FOCI trace capture files."""

from __future__ import annotations

import io
import struct
from collections.abc import Iterator
from dataclasses import dataclass

FILE_MAGIC = 0x46525446
RECORD_MAGIC = 0x52525446
FILE_VERSION = 1
RECORD_VERSION = 1
FILE_HEADER_LEN = 32
RECORD_HEADER_LEN = 32
RECORD_TYPE_DEVICE_TABLE = 1
RECORD_TYPE_FIRMWARE_FRAME = 2


@dataclass(frozen=True)
class CaptureRecord:
    """One record from a capture file."""

    record_type: int
    device_index: int
    host_recv_ns: int
    payload: bytes


class CaptureWriter:
    """Write an FTRF capture file."""

    def __init__(
        self, output: io.BufferedIOBase | io.BytesIO, start_unix_ns: int, start_mono_ns: int
    ):
        self._output = output
        self._output.write(
            struct.pack(
                "<IHHQQQ",
                FILE_MAGIC,
                FILE_VERSION,
                FILE_HEADER_LEN,
                start_unix_ns,
                start_mono_ns,
                0,
            )
        )

    def write_device_table(
        self,
        device_index: int,
        host_recv_ns: int,
        usb_path: str,
        product: str,
        serial: str,
        first_metadata: bytes,
    ) -> None:
        """Write a device table record."""
        payload = bytearray()
        for value in (usb_path, product, serial):
            encoded = value.encode("utf-8")
            payload += struct.pack("<H", len(encoded))
            payload += encoded
        payload += struct.pack("<I", len(first_metadata))
        payload += first_metadata
        self._write_record(RECORD_TYPE_DEVICE_TABLE, device_index, host_recv_ns, bytes(payload))

    def write_firmware_frame(self, device_index: int, host_recv_ns: int, frame: bytes) -> None:
        """Write one complete firmware frame record."""
        self._write_record(RECORD_TYPE_FIRMWARE_FRAME, device_index, host_recv_ns, frame)

    def _write_record(
        self,
        record_type: int,
        device_index: int,
        host_recv_ns: int,
        payload: bytes,
    ) -> None:
        self._output.write(
            struct.pack(
                "<IBBH HH Q I Q".replace(" ", ""),
                RECORD_MAGIC,
                RECORD_VERSION,
                record_type,
                RECORD_HEADER_LEN,
                device_index,
                0,
                host_recv_ns,
                len(payload),
                0,
            )
        )
        self._output.write(payload)


class CaptureReader:
    """Read an FTRF capture file."""

    def __init__(self, source: io.BufferedIOBase | io.BytesIO):
        self._source = source
        header = self._source.read(FILE_HEADER_LEN)
        magic, version, header_len, _unix_ns, _mono_ns, _reserved = struct.unpack("<IHHQQQ", header)
        if magic != FILE_MAGIC:
            raise ValueError("bad capture magic")
        if version != FILE_VERSION:
            raise ValueError(f"unsupported capture file version {version}")
        if header_len != FILE_HEADER_LEN:
            raise ValueError(f"unsupported capture header length {header_len}")

    def records(self) -> Iterator[CaptureRecord]:
        """Yield known records, skipping unknown record types."""
        while True:
            header = self._source.read(RECORD_HEADER_LEN)
            if header == b"":
                return
            if len(header) != RECORD_HEADER_LEN:
                raise ValueError("truncated record header")
            (
                magic,
                version,
                record_type,
                header_len,
                device_index,
                flags,
                host_recv_ns,
                payload_len,
                _reserved,
            ) = struct.unpack("<IBBH HH Q I Q".replace(" ", ""), header)
            if magic != RECORD_MAGIC:
                raise ValueError("bad record magic")
            payload = self._source.read(payload_len)
            if len(payload) != payload_len:
                # A SIGTERM-terminated recorder can cut the final record
                # mid-write. A short payload read means the stream ended, so
                # stop cleanly and keep every complete record before it rather
                # than discarding the whole capture.
                return
            if version != RECORD_VERSION or flags != 0:
                continue
            if header_len != RECORD_HEADER_LEN:
                continue
            if record_type not in (RECORD_TYPE_DEVICE_TABLE, RECORD_TYPE_FIRMWARE_FRAME):
                continue
            yield CaptureRecord(record_type, device_index, host_recv_ns, payload)
