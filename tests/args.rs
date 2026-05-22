use clap::Parser;
use foci_trace::args::{Cli, Command};

#[test]
fn record_requires_output_path() {
    let err = Cli::try_parse_from(["foci-trace-capture", "record"]).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}

#[test]
fn record_accepts_ready_file_and_serial_filters() {
    let cli = Cli::parse_from([
        "foci-trace-capture",
        "record",
        "--output",
        "/tmp/capture.bin",
        "--duration-s",
        "20",
        "--warmup-s",
        "2",
        "--ready-file",
        "/tmp/foci.ready",
        "--serial",
        "AAA",
        "--serial",
        "BBB",
    ]);

    let args = match cli.command {
        Command::Record(args) => args,
        other => panic!("expected record command, got {other:?}"),
    };
    assert_eq!(args.output.to_string_lossy(), "/tmp/capture.bin");
    assert_eq!(args.duration_s, 20.0);
    assert_eq!(args.warmup_s, 2.0);
    assert_eq!(
        args.ready_file.unwrap().to_string_lossy(),
        "/tmp/foci.ready"
    );
    assert_eq!(args.serial, ["AAA", "BBB"]);
}

#[test]
fn record_rejects_zero_packet_size() {
    let err = Cli::try_parse_from([
        "foci-trace-capture",
        "record",
        "--output",
        "/tmp/capture.bin",
        "--packet-size",
        "0",
    ])
    .unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
}

#[test]
fn list_accepts_serial_filters() {
    let cli = Cli::parse_from(["foci-trace-capture", "list", "--serial", "AAA"]);

    let args = match cli.command {
        Command::List(args) => args,
        other => panic!("expected list command, got {other:?}"),
    };
    assert_eq!(args.serial, ["AAA"]);
}

#[test]
fn binary_help_exits_successfully() {
    assert_cmd::Command::cargo_bin("foci-trace-capture")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}
