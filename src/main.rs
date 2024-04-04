//! A BrainFuck interpreter written in Rust.
//!
//! Reads a BrainFuck program from a file and interprets it using a BrainFuckVM.
//!
//! Optionally set --report-state to get state information output to stdout, along with the final state
//! which includes the tape data (ALTHOUGH IT MAKES IT VERY SLOW)
//!
//! There is also an API for setting different input/output streams during the programs execution,
//! allowing the tape to grow, and some framework for using different cell types (other than u8). See
//! bft_interp for the available options.

use clap::Parser;
use log::LevelFilter;
use std::error::Error;
use std::{any::TypeId, str::FromStr};
mod cli;
use bft_interp::{vm::BrainfuckVM, vm_builder::VMBuilder};
use cli::Cli;
use std::{env, process};

/// Run the interpreter using CLI args
fn run_bft(cli: Cli) -> Result<(), Box<dyn Error>> {
    let log_level = LevelFilter::from_str(&cli.log_level).unwrap_or(LevelFilter::Off);
    if cli.report_state && log_level < LevelFilter::Info {
        env::set_var("RUST_LOG", "info");
    } else {
        env::set_var("RUST_LOG", &cli.log_level);
    }
    env_logger::init();

    let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
        .set_program_file(cli.program)
        .set_allow_growth(cli.allow_growth)
        .set_cell_count(cli.cell_count)
        .set_cell_kind(TypeId::of::<u8>())
        .set_report_state(cli.report_state)
        .build()
        .map_err(|e| format!("Error: {}", e))?;

    match vm.interpret() {
        Ok(final_state) => {
            if cli.report_state {
                log::info!("Final state:\n{}", final_state);
            }
            Ok(())
        }
        Err(err) => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Error: {}", err),
        )) as Box<dyn std::error::Error>),
    }
}

/// Entry point for the interpreter program.
///
/// Reads a filename from the command line, loads a Brainfuck program from that file,
/// and then interprets it using a Brainfuck virtual machine.
///
/// # Errors
///
/// Returns an error if no filename is provided or if there are issues with reading or interpreting the program file.
///
/// # Examples
///
/// Run the program from the command line with:
/// ```bash
/// cargo run -- test_programs/example.bf
/// ```
fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_bft(cli) {
        log::error!("bft: Error encountered: {}", e);
        process::exit(1);
    }
}
