use bft_interp::BrainfuckVM;
use bft_types::Program;
use std::env;
use std::error::Error;

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
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err("No filename provided".into());
    }
    let filename = &args[1];

    let program = Program::from_file(filename)?;

    let vm = BrainfuckVM::<u8>::new(None, false);

    // Use the interpreter function to print the BF program
    vm.interpret(&program);

    Ok(())
}
