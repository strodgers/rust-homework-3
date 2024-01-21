use std::fs;
use std::io::{self, Read};
use std::path::Path;

/// Represents a Brainfuck program.
///
/// This struct holds the filename from which the program was loaded
/// and a vector of instructions. The instructions are represented as
/// `u8` values, corresponding to ASCII characters of Brainfuck language commands.
#[derive(Debug)]
pub struct Program {
    filename: String,
    instructions: Vec<u8>,
}

impl Program {
    /// Constructs a new `Program` from a given filename and program content.
    ///
    /// Filters the content to include only valid Brainfuck instructions.
    ///
    /// # Arguments
    ///
    /// * `filename` - A path-like object representing the filename of the program.
    /// * `content` - The string content of the Brainfuck program.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `Program`.
    pub fn new<P: AsRef<Path>>(filename: P, content: &str) -> Self {
        Program {
            filename: filename.as_ref().to_string_lossy().into_owned(),
            instructions: content.bytes().filter(|&b| is_bf_instruction(b)).collect(),
        }
    }

    /// Loads a Brainfuck program from a file and creates a `Program` instance.
    ///
    /// # Arguments
    ///
    /// * `path` - A path-like object representing the file path to the Brainfuck program.
    ///
    /// # Returns
    ///
    /// Returns an `io::Result` which, on success, contains the `Program` instance.
    /// On failure, it contains an `io::Error`.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = fs::File::open(&path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(Program::new(path, &contents))
    }

    /// Gets the filename from which the program was loaded.
    ///
    /// # Returns
    ///
    /// Returns a string slice of the filename.
    pub fn get_filename(&self) -> &str {
        &self.filename
    }

    /// Gets a slice of the program's instructions.
    ///
    /// # Returns
    ///
    /// Returns a slice of `u8` values, each representing a Brainfuck instruction.
    pub fn get_instructions(&self) -> &[u8] {
        &self.instructions
    }
}

/// Checks if a given byte is a valid Brainfuck instruction.
///
/// # Arguments
///
/// * `b` - The byte to check.
///
/// # Returns
///
/// Returns `true` if the byte corresponds to a valid Brainfuck instruction,
/// otherwise returns `false`.
fn is_bf_instruction(b: u8) -> bool {
    match b {
        b'>' | b'<' | b'+' | b'-' | b'.' | b',' | b'[' | b']' => true,
        _ => false,
    }
}
