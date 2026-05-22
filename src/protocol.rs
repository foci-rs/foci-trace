use bytes::{Buf, Bytes, BytesMut};
use thiserror::Error;

pub const FRAME_HEADER_LEN: usize = 32;
pub const FRAME_MAGIC: &[u8; 4] = b"FTRC";

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("bad FTRC magic")]
    BadMagic,
    #[error("unsupported FTRC header length {0}")]
    UnsupportedHeaderLength(u8),
    #[error("truncated FTRC header")]
    TruncatedHeader,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub protocol_version: u8,
    pub frame_type: u8,
    pub flags: u8,
    pub payload_len: u16,
    pub source_kind: u8,
    pub source_index: u8,
    pub sequence: u32,
    pub timestamp_ticks: u64,
    pub boot_id: u32,
}

impl FrameHeader {
    pub fn parse(data: &[u8]) -> Result<Self, ProtocolError> {
        if data.len() < FRAME_HEADER_LEN {
            return Err(ProtocolError::TruncatedHeader);
        }
        if &data[0..4] != FRAME_MAGIC {
            return Err(ProtocolError::BadMagic);
        }
        let header_len = data[5];
        if usize::from(header_len) != FRAME_HEADER_LEN {
            return Err(ProtocolError::UnsupportedHeaderLength(header_len));
        }

        Ok(Self {
            protocol_version: data[4],
            frame_type: data[6],
            flags: data[7],
            payload_len: u16::from_le_bytes(data[8..10].try_into().unwrap()),
            source_kind: data[10],
            source_index: data[11],
            sequence: u32::from_le_bytes(data[12..16].try_into().unwrap()),
            timestamp_ticks: u64::from_le_bytes(data[16..24].try_into().unwrap()),
            boot_id: u32::from_le_bytes(data[24..28].try_into().unwrap()),
        })
    }

    pub fn frame_len(&self) -> usize {
        FRAME_HEADER_LEN + usize::from(self.payload_len)
    }
}

#[derive(Debug, Default)]
pub struct FrameAssembler {
    buffer: BytesMut,
    discarded_bytes: u64,
}

impl FrameAssembler {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(8192),
            discarded_bytes: 0,
        }
    }

    pub fn push(&mut self, data: &[u8]) -> Result<Vec<Bytes>, ProtocolError> {
        self.buffer.extend_from_slice(data);
        let mut frames = Vec::new();

        loop {
            self.resync_to_magic();
            if self.buffer.len() < FRAME_HEADER_LEN {
                break;
            }

            let header = FrameHeader::parse(&self.buffer[..FRAME_HEADER_LEN])?;
            let frame_len = header.frame_len();
            if self.buffer.len() < frame_len {
                break;
            }

            frames.push(self.buffer.split_to(frame_len).freeze());
        }

        if self.buffer.capacity() > 1_048_576 && self.buffer.is_empty() {
            self.buffer = BytesMut::with_capacity(8192);
        }

        Ok(frames)
    }

    pub fn take_discarded_bytes(&mut self) -> u64 {
        let discarded = self.discarded_bytes;
        self.discarded_bytes = 0;
        discarded
    }

    pub fn clear(&mut self) {
        self.discarded_bytes = self
            .discarded_bytes
            .saturating_add(u64::try_from(self.buffer.len()).unwrap_or(u64::MAX));
        self.buffer.clear();
    }

    fn resync_to_magic(&mut self) {
        if self.buffer.len() < FRAME_MAGIC.len() {
            return;
        }
        if &self.buffer[..FRAME_MAGIC.len()] == FRAME_MAGIC {
            return;
        }

        if let Some(offset) = self
            .buffer
            .windows(FRAME_MAGIC.len())
            .position(|window| window == FRAME_MAGIC)
        {
            self.buffer.advance(offset);
            self.discarded_bytes = self
                .discarded_bytes
                .saturating_add(u64::try_from(offset).unwrap_or(u64::MAX));
            return;
        }

        let keep = FRAME_MAGIC.len() - 1;
        let discard = self.buffer.len().saturating_sub(keep);
        self.buffer.advance(discard);
        self.discarded_bytes = self
            .discarded_bytes
            .saturating_add(u64::try_from(discard).unwrap_or(u64::MAX));
    }
}
