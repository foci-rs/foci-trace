use foci_trace::protocol::{FrameAssembler, FrameHeader};

fn frame(seq: u32, payload: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"FTRC");
    data.push(2);
    data.push(32);
    data.push(1);
    data.push(0);
    data.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    data.push(1);
    data.push(0);
    data.extend_from_slice(&seq.to_le_bytes());
    data.extend_from_slice(&123u64.to_le_bytes());
    data.extend_from_slice(&456u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(payload);
    data
}

#[test]
fn assembler_reassembles_packet_boundaries() {
    let raw = frame(7, b"abc");
    let mut assembler = FrameAssembler::new();

    assert!(assembler.push(&raw[..10]).unwrap().is_empty());
    let frames = assembler.push(&raw[10..]).unwrap();

    assert_eq!(frames.len(), 1);
    assert_eq!(&frames[0][..], raw.as_slice());
}

#[test]
fn assembler_extracts_multiple_frames() {
    let mut raw = frame(1, b"a");
    raw.extend_from_slice(&frame(2, b"bb"));
    let mut assembler = FrameAssembler::new();

    let frames = assembler.push(&raw).unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(FrameHeader::parse(&frames[0]).unwrap().sequence, 1);
    assert_eq!(FrameHeader::parse(&frames[1]).unwrap().sequence, 2);
}

#[test]
fn assembler_discards_garbage_without_wedging() {
    let mut raw = frame(1, b"a");
    raw[0] = b'X';
    let mut assembler = FrameAssembler::new();

    assert!(assembler.push(&raw).unwrap().is_empty());
    assert!(assembler.take_discarded_bytes() > 0);
    assert_eq!(assembler.push(&frame(2, b"b")).unwrap().len(), 1);
}

#[test]
fn assembler_resyncs_after_leading_garbage() {
    let raw = frame(9, b"ok");
    let mut packet = b"garbage".to_vec();
    packet.extend_from_slice(&raw);
    let mut assembler = FrameAssembler::new();

    let frames = assembler.push(&packet).unwrap();

    assert_eq!(frames.len(), 1);
    assert_eq!(FrameHeader::parse(&frames[0]).unwrap().sequence, 9);
    assert_eq!(assembler.take_discarded_bytes(), 7);
    assert_eq!(assembler.push(&frame(10, b"next")).unwrap().len(), 1);
}
