use clap::Parser;
use foci_trace::args::{Cli, Command};

fn main() {
    let cli = Cli::parse();
    let result: Result<(), Box<dyn std::error::Error>> = match cli.command {
        Command::List(_args) => Ok(()),
        Command::Record(_args) => Ok(()),
    };
    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
