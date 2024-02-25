use bft_interp::BrainfuckVM;
use bft_types::Program;
use clap::Parser;
use std::error::Error;
use std::num::NonZeroUsize;
use std::path::PathBuf;
mod cli;
use cli::Cli;
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let program_name = cli.program;

    let program = Program::new(program_name)?;

    let cell_count: NonZeroUsize = NonZeroUsize::new(cli.cell_count).ok_or("Cell count must be a positive integer")?;

    let vm = BrainfuckVM::<u8>::new(cell_count, cli.allow_growth);

    // Use the interpreter function to print the BF program
    vm.interpret(&program);

    Ok(())
}