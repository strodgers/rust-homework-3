use bft_interp::BrainfuckVM;
use bft_types::Program;
use clap::Parser;
use std::error::Error;
use std::num::NonZeroUsize;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

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
    #[arg(short, long)]
    filename: String,

    /// Whether the tape is allowed to grow [default: false]
    #[arg(short, long, default_value_t = false)]
    allow_growth: bool,

    /// Specifies the number of cells in the tape
    #[arg(short, long, default_value = "30000")] // Convert Option<NonZeroUsize> to String
    cell_count: Option<NonZeroUsize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let filename = args.filename;

    let program = Program::from_file(filename)?;

    let cell_count = args.cell_count.expect("Cell count must be provided");

    let vm = BrainfuckVM::<u8>::new(cell_count, args.allow_growth);

    // Use the interpreter function to print the BF program
    vm.interpret(&program);

    Ok(())
}
