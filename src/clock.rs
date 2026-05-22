use std::time::{Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClockError {
    #[error("system time is before UNIX epoch")]
    SystemTimeBeforeEpoch,
    #[error("clock_gettime(CLOCK_MONOTONIC) failed")]
    MonotonicClockFailed,
    #[error("timestamp does not fit in u64 nanoseconds")]
    TimestampOverflow,
}

#[derive(Debug, Clone)]
pub struct CaptureClock {
    start_instant: Instant,
    start_unix_ns: u64,
    start_mono_ns: u64,
}

impl CaptureClock {
    pub fn new() -> Result<Self, ClockError> {
        let start_instant = Instant::now();
        let start_unix_ns = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| ClockError::SystemTimeBeforeEpoch)?
                .as_nanos(),
        )
        .map_err(|_| ClockError::TimestampOverflow)?;
        let start_mono_ns = monotonic_now_ns()?;
        Ok(Self {
            start_instant,
            start_unix_ns,
            start_mono_ns,
        })
    }

    pub fn start_unix_ns(&self) -> u64 {
        self.start_unix_ns
    }

    pub fn start_mono_ns(&self) -> u64 {
        self.start_mono_ns
    }

    pub fn host_recv_ns(&self) -> Result<u64, ClockError> {
        let elapsed_ns = u64::try_from(self.start_instant.elapsed().as_nanos())
            .map_err(|_| ClockError::TimestampOverflow)?;
        self.start_mono_ns
            .checked_add(elapsed_ns)
            .ok_or(ClockError::TimestampOverflow)
    }
}

fn monotonic_now_ns() -> Result<u64, ClockError> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` points to valid writable memory for one `timespec`, and
    // `clock_gettime` does not retain the pointer after returning.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if rc != 0 {
        return Err(ClockError::MonotonicClockFailed);
    }
    let secs = u64::try_from(ts.tv_sec).map_err(|_| ClockError::TimestampOverflow)?;
    let nanos = u64::try_from(ts.tv_nsec).map_err(|_| ClockError::TimestampOverflow)?;
    secs.checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or(ClockError::TimestampOverflow)
}
