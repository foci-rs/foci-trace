use std::fs;
use std::time::Duration;

use bytes::Bytes;
use foci_trace::capture_file::CaptureWriter;
use foci_trace::clock::CaptureClock;
use foci_trace::record::{
    CapturedFrame, HostCounters, ReaderStartup, create_capture_output, drain_writer,
    wait_for_reader_startups_then_write_ready, write_ready_file,
};

#[test]
fn create_capture_output_creates_parent_directory() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("nested/capture.bin");

    let _file = create_capture_output(&output).unwrap();

    assert!(output.exists());
}

#[test]
fn create_capture_output_refuses_to_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("capture.bin");
    fs::write(&output, b"existing").unwrap();

    let err = create_capture_output(&output).unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
}

#[test]
fn ready_file_contains_capture_path() {
    let temp = tempfile::tempdir().unwrap();
    let ready = temp.path().join("capture.ready");
    let output = temp.path().join("capture.bin");

    write_ready_file(Some(&ready), &output).unwrap();

    assert_eq!(
        fs::read_to_string(ready).unwrap(),
        format!("{}\n", output.display())
    );
}

#[test]
fn ready_file_waits_for_reader_startup() {
    let temp = tempfile::tempdir().unwrap();
    let ready = temp.path().join("capture.ready");
    let output = temp.path().join("capture.bin");
    let ready_for_thread = ready.clone();
    let output_for_thread = output.clone();
    let (tx, rx) = crossbeam_channel::bounded(1);

    let handle = std::thread::spawn(move || {
        wait_for_reader_startups_then_write_ready(
            rx,
            1,
            Some(&ready_for_thread),
            &output_for_thread,
        )
    });

    std::thread::sleep(Duration::from_millis(50));
    assert!(!ready.exists());

    tx.send(ReaderStartup::Started).unwrap();
    handle.join().unwrap().unwrap();

    assert_eq!(
        fs::read_to_string(ready).unwrap(),
        format!("{}\n", output.display())
    );
}

#[test]
fn drain_writer_preserves_frame_host_timestamps() {
    let clock = CaptureClock::new().unwrap();
    let mut output = std::io::Cursor::new(Vec::new());
    let mut writer = CaptureWriter::new(&mut output, &clock).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    tx.send(CapturedFrame {
        device_index: 0,
        host_recv_ns: 100,
        frame: Bytes::from_static(b"frame000"),
    })
    .unwrap();
    tx.send(CapturedFrame {
        device_index: 0,
        host_recv_ns: 110,
        frame: Bytes::from_static(b"frame111"),
    })
    .unwrap();
    drop(tx);

    let mut counters = HostCounters::default();
    drain_writer(rx, &mut writer, &mut counters).unwrap();
    writer.flush().unwrap();

    let bytes = output.into_inner();
    assert_eq!(counters.frames_written, 2);
    assert_eq!(&bytes[44..52], &100u64.to_le_bytes());
    let second = 32 + 32 + b"frame000".len();
    assert_eq!(&bytes[second + 12..second + 20], &110u64.to_le_bytes());
}
