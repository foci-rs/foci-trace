use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use crossbeam_channel::{Receiver, Sender};
use thiserror::Error;

use crate::args::{ListArgs, RecordArgs};
use crate::capture_file::CaptureWriter;
use crate::clock::{CaptureClock, ClockError};
use crate::protocol::{FrameAssembler, ProtocolError};
use crate::usb::{OpenTraceDevice, UsbTraceError, find_trace_devices};

#[derive(Debug, Error)]
pub enum RecordError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("clock error: {0}")]
    Clock(#[from] ClockError),
    #[error("protocol error: {0}")]
    Protocol(#[from] ProtocolError),
    #[error("usb error: {0}")]
    Usb(#[from] UsbTraceError),
    #[error("no FOCI trace devices found")]
    NoTraceDevices,
}

#[derive(Debug)]
pub struct CapturedFrame {
    pub device_index: u16,
    pub host_recv_ns: u64,
    pub frame: Bytes,
}

#[derive(Debug, Default)]
pub struct HostCounters {
    pub frames_written: u64,
    pub channel_drops: u64,
    pub parser_errors: u64,
    pub parser_resync_bytes: u64,
    pub io_errors: u64,
}

#[derive(Debug, Default)]
struct ReaderCounters {
    channel_drops: u64,
    parser_errors: u64,
    parser_resync_bytes: u64,
    io_errors: u64,
}

pub fn run_list(args: ListArgs) -> Result<(), RecordError> {
    let devices = find_trace_devices(&args.serial)?;
    if devices.is_empty() {
        return Err(RecordError::NoTraceDevices);
    }

    for device in devices {
        println!(
            "{} product={} serial={} endpoint=0x{:02x} mps={}",
            device.info.path(),
            device.info.product,
            device.info.serial,
            device.endpoint_address,
            device.max_packet_size
        );
        if let Err(err) = device.handle.release_interface(device.interface_number) {
            eprintln!(
                "warning: failed to release interface {} on {}: {err}",
                device.interface_number,
                device.info.path()
            );
        }
    }

    Ok(())
}

pub fn run_record(args: RecordArgs) -> Result<(), RecordError> {
    let mut devices = find_trace_devices(&args.serial)?;
    if devices.is_empty() {
        return Err(RecordError::NoTraceDevices);
    }

    warmup(&mut devices, args.packet_size, args.warmup_s);

    let clock = CaptureClock::new()?;
    let output = create_capture_output(&args.output)?;
    let mut writer = CaptureWriter::new(output, &clock)?;
    for (index, device) in devices.iter().enumerate() {
        writer.write_device_table(
            u16::try_from(index).expect("trace device index fits u16"),
            clock.host_recv_ns()?,
            &device.info.path(),
            &device.info.product,
            &device.info.serial,
            b"",
        )?;
    }

    write_ready_file(args.ready_file.as_deref(), &args.output)?;

    let (tx, rx) = crossbeam_channel::bounded(4096);
    let deadline =
        (args.duration_s > 0.0).then(|| Instant::now() + Duration::from_secs_f64(args.duration_s));
    let mut handles = Vec::new();
    for (index, device) in devices.into_iter().enumerate() {
        let tx = tx.clone();
        let clock = clock.clone();
        let packet_size = args.packet_size.max(usize::from(device.max_packet_size));
        handles.push(thread::spawn(move || {
            read_device_loop(
                u16::try_from(index).expect("trace device index fits u16"),
                device,
                packet_size,
                deadline,
                clock,
                tx,
            )
        }));
    }
    drop(tx);

    let mut counters = HostCounters::default();
    let writer_result = drain_writer(rx, &mut writer, &mut counters);
    for handle in handles {
        if let Ok(reader) = handle.join() {
            counters.channel_drops += reader.channel_drops;
            counters.parser_errors += reader.parser_errors;
            counters.parser_resync_bytes += reader.parser_resync_bytes;
            counters.io_errors += reader.io_errors;
        }
    }
    writer.flush()?;
    writer_result?;
    eprintln!(
        "frames_written={} host_channel_drops={} parser_errors={} \
         parser_resync_bytes={} usb_io_errors={}",
        counters.frames_written,
        counters.channel_drops,
        counters.parser_errors,
        counters.parser_resync_bytes,
        counters.io_errors
    );

    Ok(())
}

pub fn create_capture_output(path: &Path) -> io::Result<BufWriter<File>> {
    create_parent_dir(path)?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
}

pub fn write_ready_file(ready_file: Option<&Path>, output: &Path) -> io::Result<()> {
    if let Some(path) = ready_file {
        create_parent_dir(path)?;
        let mut file = File::create(path)?;
        writeln!(file, "{}", output.display())?;
    }
    println!("recording started: {}", output.display());

    Ok(())
}

fn create_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    Ok(())
}

fn warmup(devices: &mut [OpenTraceDevice], packet_size: usize, warmup_s: f64) {
    if warmup_s <= 0.0 {
        return;
    }

    let deadline = Instant::now() + Duration::from_secs_f64(warmup_s);
    // Warmup drains devices serially; keep this scratch buffer local unless
    // warmup is redesigned as a parallel read path.
    let mut scratch = vec![0u8; packet_size];
    while Instant::now() < deadline {
        for device in devices.iter_mut() {
            let _ = device.handle.read_bulk(
                device.endpoint_address,
                &mut scratch,
                Duration::from_millis(10),
            );
        }
    }
}

fn read_device_loop(
    device_index: u16,
    device: OpenTraceDevice,
    packet_size: usize,
    deadline: Option<Instant>,
    clock: CaptureClock,
    tx: Sender<CapturedFrame>,
) -> ReaderCounters {
    let mut assembler = FrameAssembler::new();
    let mut scratch = vec![0u8; packet_size];
    let mut counters = ReaderCounters::default();

    while deadline.is_none_or(|value| Instant::now() < value) {
        let read = match device.handle.read_bulk(
            device.endpoint_address,
            &mut scratch,
            Duration::from_millis(10),
        ) {
            Ok(read) => read,
            Err(rusb::Error::Timeout) => continue,
            Err(_) => {
                counters.io_errors += 1;
                continue;
            }
        };
        let host_recv_ns = match clock.host_recv_ns() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let frames = match assembler.push(&scratch[..read]) {
            Ok(frames) => frames,
            Err(_) => {
                counters.parser_errors += 1;
                assembler.clear();
                continue;
            }
        };
        counters.parser_resync_bytes += assembler.take_discarded_bytes();

        for frame in frames {
            if tx
                .try_send(CapturedFrame {
                    device_index,
                    host_recv_ns,
                    frame,
                })
                .is_err()
            {
                counters.channel_drops += 1;
            }
        }
    }

    counters.parser_resync_bytes += assembler.take_discarded_bytes();
    if device
        .handle
        .release_interface(device.interface_number)
        .is_err()
    {
        counters.io_errors += 1;
    }

    counters
}

pub fn drain_writer<W: Write>(
    rx: Receiver<CapturedFrame>,
    writer: &mut CaptureWriter<W>,
    counters: &mut HostCounters,
) -> Result<(), RecordError> {
    for frame in rx {
        writer.write_firmware_frame(frame.device_index, frame.host_recv_ns, &frame.frame)?;
        counters.frames_written += 1;
    }

    Ok(())
}
