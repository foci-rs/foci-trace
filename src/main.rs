use clap::Parser;
use foci_trace::args::{Cli, Command};

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::List(args) => foci_trace::record::run_list(args),
        Command::Record(args) => foci_trace::record::run_record(args),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
