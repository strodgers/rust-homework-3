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
//!
//! # Examples
//!
//! Run a test program
//! ```bash
//! cargo run ../test_programs/example.bf
//! ```
//!
//! Run a test program with dynamic tape memoery, initial cell count 10,000, report the final state
//! ```bash
//! cargo run test_programs/example.bf --allow-growth --cell-count 10000 --report-state
//! ```
//!

use clap::Parser;
use env_logger::Env;
use std::error::Error;
mod cli;
use bft_interp::{builder::VMBuilder, core::BrainfuckVM};
use cli::Cli;
use std::process;

/// Run the interpreter using CLI args
fn run_bft(cli: Cli) -> Result<(), Box<dyn Error>> {
    // TODO
    let env = Env::new()
        .filter("BFT_LOG")
        .default_filter_or(cli.log_level);
    env_logger::init_from_env(env);

    let mut vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
        .set_program_file(cli.program)
        .set_allow_growth(cli.allow_growth)
        .set_cell_count(cli.cell_count)
        .set_report_state(cli.report_state)
        .build()
        .map_err(|e| format!("Error: {}", e))?;

    // Run interpretation
    let final_state_option = vm.interpret().map_err(|e| format!("{}", e))?;

    // Report state if necessary
    if cli.report_state {
        match final_state_option {
            Some(final_state) => log::info!("Final state:\n{}", final_state),
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed reporting final state",
                )) as Box<dyn std::error::Error>)
            }
        }
    }
    println!("Finished {} steps", vm.instructions_processed());
    Ok(())
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
