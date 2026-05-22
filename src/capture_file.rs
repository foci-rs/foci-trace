use std::io::{self, Write};

use crate::clock::CaptureClock;

pub const FILE_MAGIC: u32 = 0x4652_5446;
pub const RECORD_MAGIC: u32 = 0x5252_5446;
pub const FILE_VERSION: u16 = 1;
pub const RECORD_VERSION: u8 = 1;
pub const FILE_HEADER_LEN: u16 = 32;
pub const RECORD_HEADER_LEN: u16 = 32;
pub const RECORD_TYPE_DEVICE_TABLE: u8 = 1;
pub const RECORD_TYPE_FIRMWARE_FRAME: u8 = 2;

pub struct CaptureWriter<W> {
    output: W,
}

impl<W: Write> CaptureWriter<W> {
    pub fn new(mut output: W, clock: &CaptureClock) -> io::Result<Self> {
        output.write_all(&FILE_MAGIC.to_le_bytes())?;
        output.write_all(&FILE_VERSION.to_le_bytes())?;
        output.write_all(&FILE_HEADER_LEN.to_le_bytes())?;
        output.write_all(&clock.start_unix_ns().to_le_bytes())?;
        output.write_all(&clock.start_mono_ns().to_le_bytes())?;
        output.write_all(&0u64.to_le_bytes())?;
        Ok(Self { output })
    }

    pub fn write_device_table(
        &mut self,
        device_index: u16,
        host_recv_ns: u64,
        usb_path: &str,
        product: &str,
        serial: &str,
        first_metadata: &[u8],
    ) -> io::Result<()> {
        let mut payload = Vec::new();
        for value in [usb_path, product, serial] {
            let bytes = value.as_bytes();
            let len = u16::try_from(bytes.len()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "device string too long")
            })?;
            payload.write_all(&len.to_le_bytes())?;
            payload.write_all(bytes)?;
        }
        let metadata_len = u32::try_from(first_metadata.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "metadata frame too large"))?;
        payload.write_all(&metadata_len.to_le_bytes())?;
        payload.write_all(first_metadata)?;
        self.write_record(
            RECORD_TYPE_DEVICE_TABLE,
            device_index,
            host_recv_ns,
            &payload,
        )
    }

    pub fn write_firmware_frame(
        &mut self,
        device_index: u16,
        host_recv_ns: u64,
        frame: &[u8],
    ) -> io::Result<()> {
        self.write_record(
            RECORD_TYPE_FIRMWARE_FRAME,
            device_index,
            host_recv_ns,
            frame,
        )
    }

    fn write_record(
        &mut self,
        record_type: u8,
        device_index: u16,
        host_recv_ns: u64,
        payload: &[u8],
    ) -> io::Result<()> {
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "record too large"))?;
        self.output.write_all(&RECORD_MAGIC.to_le_bytes())?;
        self.output.write_all(&[RECORD_VERSION, record_type])?;
        self.output.write_all(&RECORD_HEADER_LEN.to_le_bytes())?;
        self.output.write_all(&device_index.to_le_bytes())?;
        self.output.write_all(&0u16.to_le_bytes())?;
        self.output.write_all(&host_recv_ns.to_le_bytes())?;
        self.output.write_all(&payload_len.to_le_bytes())?;
        self.output.write_all(&0u64.to_le_bytes())?;
        self.output.write_all(payload)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}
