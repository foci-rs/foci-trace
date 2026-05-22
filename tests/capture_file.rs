use std::io::Cursor;

use foci_trace::capture_file::{
    CaptureWriter, RECORD_TYPE_DEVICE_TABLE, RECORD_TYPE_FIRMWARE_FRAME,
};
use foci_trace::clock::CaptureClock;

#[test]
fn file_header_uses_one_clock_source() {
    let clock = CaptureClock::new().unwrap();
    let mut output = Cursor::new(Vec::new());

    CaptureWriter::new(&mut output, &clock).unwrap();

    let bytes = output.into_inner();
    assert_eq!(&bytes[0..4], b"FTRF");
    assert_eq!(&bytes[4..6], &1u16.to_le_bytes());
    assert_eq!(&bytes[6..8], &32u16.to_le_bytes());
    let start_mono = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert!(start_mono > 0);
}

#[test]
fn device_and_frame_records_use_supplied_host_timestamp() {
    let clock = CaptureClock::new().unwrap();
    let mut output = Cursor::new(Vec::new());
    let mut writer = CaptureWriter::new(&mut output, &clock).unwrap();
    let usb_path = "1-50";
    let product = "FOCI OpenFFBoard (trace)";
    let serial = "ABC";

    writer
        .write_device_table(0, 123, usb_path, product, serial, b"")
        .unwrap();
    writer.write_firmware_frame(0, 456, b"FTRCfake").unwrap();

    let bytes = output.into_inner();
    assert_eq!(&bytes[32..36], b"FTRR");
    assert_eq!(bytes[37], RECORD_TYPE_DEVICE_TABLE);
    assert_eq!(&bytes[44..52], &123u64.to_le_bytes());

    let device_payload_len = 2 + usb_path.len() + 2 + product.len() + 2 + serial.len() + 4;
    let second = 32 + 32 + device_payload_len;
    assert_eq!(&bytes[second..second + 4], b"FTRR");
    assert_eq!(bytes[second + 5], RECORD_TYPE_FIRMWARE_FRAME);
    assert_eq!(&bytes[second + 12..second + 20], &456u64.to_le_bytes());
}
