use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    #[error("trace reader failed to start")]
    TraceReaderStartupFailed,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderStartup {
    Started,
    Failed,
}

/// When a trace reader thread should stop reading.
#[derive(Debug, Clone, Copy)]
struct StopPolicy {
    /// Optional hard ceiling, independent of any stop signal.
    deadline: Option<Instant>,
    /// Grace period to keep reading past a stop signal, to drain frames
    /// still in flight over USB when the signal arrives.
    drain: Duration,
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
    }

    Ok(())
}

/// Set from `handle_stop_signal` when SIGINT or SIGTERM arrives. Reader
/// threads poll it every ~10ms (the USB read timeout already in place) and
/// stop at the next frame boundary, so the writer drains what's already
/// queued and flushes normally instead of the process dying mid-write. A
/// second signal at least `REPEAT_STOP_MIN_GAP_NS` later exits immediately,
/// for when that drain or flush is stuck.
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);
static FIRST_STOP_NS: AtomicU64 = AtomicU64::new(0);

/// `sudo` relays a signal sent to it by another process, so `pkill -f`
/// matching both `sudo` and the recorder delivers the same stop twice within
/// microseconds. Only a later repeat is an operator insisting.
const REPEAT_STOP_MIN_GAP_NS: u64 = 500_000_000;

fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` is async-signal-safe and only writes `now`.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    let secs = u64::try_from(now.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(now.tv_nsec).unwrap_or(0);
    (secs * 1_000_000_000 + nanos).max(1)
}

extern "C" fn handle_stop_signal(signum: libc::c_int) {
    let now = monotonic_ns();
    match FIRST_STOP_NS.compare_exchange(0, now, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => STOP_REQUESTED.store(true, Ordering::SeqCst),
        Err(first) if now.saturating_sub(first) >= REPEAT_STOP_MIN_GAP_NS => {
            // SAFETY: `_exit` is async-signal-safe and skips every destructor,
            // so a stalled writer cannot block it.
            unsafe { libc::_exit(128 + signum) };
        }
        Err(_) => {}
    }
}

fn install_signal_handlers() {
    // SAFETY: the handler only reads the monotonic clock, updates atomics, and
    // may call `_exit`, all async-signal-safe; no allocation, locking, or I/O
    // happens in it.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_stop_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_stop_signal as *const () as libc::sighandler_t,
        );
    }
}

pub fn run_record(args: RecordArgs) -> Result<(), RecordError> {
    install_signal_handlers();

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
            device.info.path(),
            &device.info.product,
            &device.info.serial,
            b"",
        )?;
    }

    let (tx, rx) = crossbeam_channel::bounded(4096);
    let (startup_tx, startup_rx) = crossbeam_channel::bounded(devices.len());
    let stop_policy = StopPolicy {
        deadline: (args.duration_s > 0.0)
            .then(|| Instant::now() + Duration::from_secs_f64(args.duration_s)),
        drain: Duration::from_secs_f64(args.drain_s.max(0.0)),
    };
    let reader_count = devices.len();
    let mut handles = Vec::new();
    for (index, device) in devices.into_iter().enumerate() {
        let tx = tx.clone();
        let startup_tx = startup_tx.clone();
        let clock = clock.clone();
        let packet_size = args.packet_size.max(device.max_packet_size);
        handles.push(thread::spawn(move || {
            read_device_loop(
                u16::try_from(index).expect("trace device index fits u16"),
                device,
                packet_size,
                stop_policy,
                clock,
                tx,
                startup_tx,
            )
        }));
    }
    drop(tx);
    drop(startup_tx);

    wait_for_reader_startups_then_write_ready(
        startup_rx,
        reader_count,
        args.ready_file.as_deref(),
        &args.output,
    )?;

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

pub fn wait_for_reader_startups_then_write_ready(
    startup_rx: Receiver<ReaderStartup>,
    expected_readers: usize,
    ready_file: Option<&Path>,
    output: &Path,
) -> Result<(), RecordError> {
    for _ in 0..expected_readers {
        match startup_rx.recv() {
            Ok(ReaderStartup::Started) => {}
            Ok(ReaderStartup::Failed) | Err(_) => {
                return Err(RecordError::TraceReaderStartupFailed);
            }
        }
    }
    write_ready_file(ready_file, output)?;
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
            if let Ok(mut reader) = device.bulk_reader(packet_size, Duration::from_millis(10)) {
                let _ = reader.read(&mut scratch);
            }
        }
    }
}

/// Decide whether a trace reader thread should stop this iteration.
///
/// A stop signal does not stop reading immediately: the firmware's trace
/// queue can still have frames in flight over USB at the instant a caller
/// signals this process (e.g. right when a wrapped command's response
/// returns), so reading continues for `drain` past the first observed stop
/// request to catch them, mirroring the pre-roll `warmup` on the other end
/// of a capture. A duration-based `deadline` stops reading regardless of any
/// stop signal.
fn should_stop_reading(
    stop_requested: bool,
    stop_observed_at: &mut Option<Instant>,
    deadline: Option<Instant>,
    drain: Duration,
    now: Instant,
) -> bool {
    if stop_requested && stop_observed_at.is_none() {
        *stop_observed_at = Some(now);
    }
    if let Some(observed) = *stop_observed_at
        && now >= observed + drain
    {
        return true;
    }
    deadline.is_some_and(|deadline| now >= deadline)
}

fn read_device_loop(
    device_index: u16,
    device: OpenTraceDevice,
    packet_size: usize,
    stop_policy: StopPolicy,
    clock: CaptureClock,
    tx: Sender<CapturedFrame>,
    startup_tx: Sender<ReaderStartup>,
) -> ReaderCounters {
    let mut assembler = FrameAssembler::new();
    let mut scratch = vec![0u8; packet_size];
    let mut counters = ReaderCounters::default();
    let mut reader = match device.bulk_reader(packet_size, Duration::from_millis(10)) {
        Ok(reader) => {
            let _ = startup_tx.send(ReaderStartup::Started);
            reader
        }
        Err(_) => {
            let _ = startup_tx.send(ReaderStartup::Failed);
            counters.io_errors += 1;
            return counters;
        }
    };

    let mut stop_observed_at = None;
    while !should_stop_reading(
        STOP_REQUESTED.load(Ordering::SeqCst),
        &mut stop_observed_at,
        stop_policy.deadline,
        stop_policy.drain,
        Instant::now(),
    ) {
        let read = match reader.read(&mut scratch) {
            Ok(read) => read,
            Err(err) if err.kind() == io::ErrorKind::TimedOut => continue,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigint_sets_stop_requested() {
        install_signal_handlers();
        unsafe {
            libc::raise(libc::SIGINT);
        }
        assert!(STOP_REQUESTED.load(Ordering::SeqCst));
    }

    fn child_exit_status_after_raising(gaps_between_signals: &[Duration]) -> libc::c_int {
        // SAFETY: the child only calls async-signal-safe functions (signal,
        // raise, nanosleep, _exit) and touches atomics before exiting, so
        // forking a multithreaded test process is sound here.
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0, "fork failed");
            if pid == 0 {
                STOP_REQUESTED.store(false, Ordering::SeqCst);
                FIRST_STOP_NS.store(0, Ordering::SeqCst);
                install_signal_handlers();
                libc::raise(libc::SIGINT);
                for gap in gaps_between_signals {
                    let pause = libc::timespec {
                        tv_sec: 0,
                        tv_nsec: libc::c_long::try_from(gap.as_nanos()).unwrap_or(0),
                    };
                    libc::nanosleep(&pause, std::ptr::null_mut());
                    libc::raise(libc::SIGINT);
                }
                libc::_exit(0);
            }
            let mut status = 0;
            assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
            assert!(libc::WIFEXITED(status), "child did not exit normally");
            libc::WEXITSTATUS(status)
        }
    }

    #[test]
    fn first_stop_signal_lets_the_recorder_finish() {
        assert_eq!(child_exit_status_after_raising(&[]), 0);
    }

    #[test]
    fn immediate_duplicate_stop_signal_is_ignored() {
        assert_eq!(child_exit_status_after_raising(&[Duration::ZERO]), 0);
    }

    #[test]
    fn later_second_stop_signal_force_quits() {
        assert_eq!(
            child_exit_status_after_raising(&[Duration::from_millis(600)]),
            128 + libc::SIGINT
        );
    }

    #[test]
    fn should_stop_reading_drains_past_a_stop_signal_before_stopping() {
        let start = Instant::now();
        let drain = Duration::from_millis(300);
        let mut stop_observed_at = None;

        assert!(
            !should_stop_reading(false, &mut stop_observed_at, None, drain, start),
            "no stop requested and no deadline must keep reading"
        );

        assert!(
            !should_stop_reading(true, &mut stop_observed_at, None, drain, start),
            "a fresh stop request must not stop immediately"
        );
        assert!(
            !should_stop_reading(
                true,
                &mut stop_observed_at,
                None,
                drain,
                start + Duration::from_millis(299)
            ),
            "still inside the drain window"
        );
        assert!(
            should_stop_reading(
                true,
                &mut stop_observed_at,
                None,
                drain,
                start + Duration::from_millis(300)
            ),
            "drain window elapsed"
        );
    }

    #[test]
    fn should_stop_reading_still_honors_an_earlier_duration_deadline() {
        let start = Instant::now();
        let deadline = start + Duration::from_millis(50);
        let mut stop_observed_at = None;

        assert!(
            !should_stop_reading(
                false,
                &mut stop_observed_at,
                Some(deadline),
                Duration::from_secs(5),
                start
            ),
            "deadline not yet reached"
        );
        assert!(
            should_stop_reading(
                false,
                &mut stop_observed_at,
                Some(deadline),
                Duration::from_secs(5),
                start + Duration::from_millis(50)
            ),
            "duration deadline reached, independent of any stop signal"
        );
    }
}
