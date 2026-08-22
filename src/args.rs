use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "foci-trace-capture")]
#[command(about = "Capture FOCI USB trace streams")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    List(ListArgs),
    Record(RecordArgs),
}

#[derive(Debug, Args)]
pub struct ListArgs {
    #[arg(long)]
    pub serial: Vec<String>,
}

#[derive(Debug, Args)]
pub struct RecordArgs {
    #[arg(long)]
    pub output: PathBuf,

    /// Recording duration in seconds. 0 means run until stopped with
    /// SIGINT/SIGTERM (Ctrl-C or `kill`), which stops the capture cleanly
    /// at the next frame boundary and flushes it.
    #[arg(long, default_value_t = 0.0)]
    pub duration_s: f64,

    #[arg(long, default_value_t = 2.0)]
    pub warmup_s: f64,

    /// Seconds to keep reading past a stop signal (SIGINT/SIGTERM) before
    /// actually stopping. The firmware's trace queue can still have frames
    /// in flight over USB at the instant a caller signals this process
    /// (e.g. right when a wrapped command's response returns); draining a
    /// short grace period catches them instead of truncating the capture.
    #[arg(long, default_value_t = 0.5)]
    pub drain_s: f64,

    #[arg(long, default_value_t = 4096, value_parser = parse_nonzero_usize)]
    pub packet_size: usize,

    #[arg(long)]
    pub ready_file: Option<PathBuf>,

    #[arg(long)]
    pub serial: Vec<String>,
}

fn parse_nonzero_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid positive integer: {err}"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
}
