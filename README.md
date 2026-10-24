# foci-trace

Records the FOCI firmware's USB trace stream and decodes it offline.

Part of the [FOCI](https://github.com/foci-rs/foci) project, a Klipper/Kalico-compatible
MCU firmware family for STM32 boards with TMC4671 servo controllers.

## Status

Pre-1.0. The capture file format and the analyzer's output change without notice. This is
a development tool for the FOCI firmware, not a supported product.

## Capture

`foci-trace-capture` is a Rust binary. It reads the trace interface that firmware built with
the `trace` feature exposes over USB, and writes the frames to a capture file.

```sh
cargo build --release
foci-trace-capture list
foci-trace-capture record --output capture.bin --duration-s 60
```

`--duration-s 0`, the default, records until SIGINT or SIGTERM. The recorder then drains
frames still in flight for `--drain-s` seconds (default 0.5) and flushes the file. Use
`--serial` to pick a board when several are attached, and `--ready-file` to get a marker
file once capture has started.

On most systems the recorder needs root to open the USB device. Without it the capture
may not start, and the output file can come out empty.

## Analyze

The analyzer is a Python package with no dependencies. It needs Python 3.11 or newer.

```sh
python -m foci_trace.analyze capture.bin
```

It prints a summary of the capture: motion segments, tracking lag, and the evidence frames
from commissioning, breakaway and velocity-integral runs. Run it with `--help` for the
full option list, including the CoreXY motor selectors.

## Tests

```sh
cargo test
python -m pytest
```

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at
your option.
