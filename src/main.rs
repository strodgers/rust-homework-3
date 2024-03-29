use bft_interp::BrainfuckVM;
use bft_interp::VMBuilder;
use bft_types::Program;
use clap::Parser;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::path::PathBuf;
mod cli;
use cli::Cli;
use std::process;
use std::io::{Read, Write};
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

#[derive(Parser, Debug)]
#[command(version, about, author = "Sam Rodgers")]
struct Args {
    /// Name of the Brainfuck program to interpret
    filename: PathBuf,

    /// Allow the tap to grow
    #[arg(short, long)]
    allow_growth: bool,

    /// Specifies the number of cells in the tape.
    ///
    /// Traditionally Brainfuck interpreters use a tape of 30,000 cells.
    #[arg(short, long)] // Convert Option<NonZeroUsize> to String
    cell_count: Option<NonZeroUsize>,
}

fn run_bft(cli: Cli) -> Result<(), Box<dyn Error>> {
    let file = File::open(cli.program)?;
    let program_file = BufReader::new(file);

    let mut vm: BrainfuckVM<u8> = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
        .set_program_file(program_file)
        .build()
        .map_err(|e| format!("Error: {}", e))?;

    // Use the interpreter function to print the BF program
    vm.interpret().map_err(|err| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Error: {}", err),
        )) as Box<dyn std::error::Error>
    })
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_bft(cli) {
        eprintln!("bft: Error encountered: {}", e);
        process::exit(1);
    }
}
