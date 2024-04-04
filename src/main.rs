use bft_interp::{vm::BrainfuckVM, vm_builder::VMBuilder};
use clap::Parser;
use log::LevelFilter;
use std::any::TypeId;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
mod cli;
use cli::Cli;
use std::process;
// TODO: maybe this comment should be on fn main()
/// Entry point for the Brainfuck interpreter program.
///
/// This program reads a filename from the command line, loads a Brainfuck program from that file,
/// and then interprets it using a Brainfuck virtual machine.
///
/// # Arguments
///
/// Reads command-line arguments to get the filename of the Brainfuck program to interpret.
///
/// # Errors
///
/// Returns an error if no filename is provided or if there are issues with reading or interpreting the program file.
///
/// # Examples
///
/// Run the program from the command line with:
/// ```bash
/// cargo run -- example.bf
/// ```
/// Replace `example.bf` with the path to your Brainfuck program file.
///
/// # Remarks
///
/// The Brainfuck program is interpreted by printing the instructions to the console.
/// Future implementations may include execution of the Brainfuck program.

fn run_bft(cli: Cli) -> Result<(), Box<dyn Error>> {
    let test_log_level = LevelFilter::Info;
    let _ = env_logger::builder()
        .is_test(true)
        .filter(None, test_log_level)
        .try_init();

    let file = File::open(cli.program)?;
    let program_file = BufReader::new(file);

    let mut vm: BrainfuckVM<u8> = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
        .set_program_file(program_file)
        .set_allow_growth(cli.allow_growth)
        .set_cell_count(cli.cell_count)
        .set_cell_kind(TypeId::of::<u8>())
        .set_report_state(cli.report_state)
        .build()
        .map_err(|e| format!("Error: {}", e))?;

    // Use the interpret function to print the BF program
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

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_bft(cli) {
        log::error!("bft: Error encountered: {}", e);
        process::exit(1);
    }
}
