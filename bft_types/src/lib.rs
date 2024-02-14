//! Representation of Brainfuck programs
//!
//! This includes capabilities to represent instructions and their provenance,
//! and to parse programs from files.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// A Brainfuck program.
///
/// This struct holds the filename from which the program was loaded
/// and a vector of instructions.
#[derive(Debug)]
pub struct Program {
    filename: PathBuf,
    // TODO: really this should be Vec<InstructionWithPosition>
    // and InstructionWithPosition should be a line and column, and some raw instruction
    // enum of some kind which has a `from_char()` or similar method.
    instructions: Vec<u8>,
}

impl Program {
    /// Constructs a new `Program` from a given filename and program content.
    ///
    /// Filters `content` and only includes valid Brainfuck instructions in the
    /// constructed `Program`.
    ///
    /// ```
    /// # use bft_types::Program;
    /// let program = Program::new("example.bf", "+-><.");
    ///
    /// assert_eq!(program.get_filename(), "example.bf");
    /// ```
    pub fn new<P: AsRef<Path>>(filename: P, content: &str) -> Self {
        Program {
            filename: filename.as_ref().to_owned(),
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
    /// # Errors
    ///
    /// Will return `Err` if the file cannot be opened or read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use bft_types::Program;
    /// let program = Program::from_file("path/to/program.bf")?;
    ///
    /// assert_eq!(program.filename(), "path/to/program.bf");
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = fs::File::open(&path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(Program::new(path, &contents))
    }

    /// The filename from which the program was loaded
    pub fn filename(&self) -> &Path {
        &self.filename
    }

    /// The instructions in the program
    pub fn instructions(&self) -> &[u8] {
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
    matches!(b, b'>' | b'<' | b'+' | b'-' | b'.' | b',' | b'[' | b']')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_bf_instructions() {
        let content = "+-><.,[]";
        let program = Program::new("test.bf", content);

        assert_eq!(
            program.instructions(),
            [b'+', b'-', b'>', b'<', b'.', b',', b'[', b']']
        );
    }

    #[test]
    fn test_ignore_non_bf_characters() {
        let content = "+a->b<.c,d[e]f";
        let program = Program::new("test.bf", content);

        assert_eq!(
            program.instructions(),
            &[b'+', b'-', b'>', b'<', b'.', b',', b'[', b']']
        );
    }

    #[test]
    fn test_empty_content() {
        let content = "";
        let program = Program::new("test.bf", content);

        assert!(program.instructions().is_empty());
    }

    #[test]
    fn test_load_from_file() {
        let content = "+-><.,[]";
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        writeln!(temp_file, "{}", content).expect("Failed to write to temporary file");
        let program =
            Program::from_file(temp_file.path()).expect("Failed to load program from file");
        // Of course, once you have provenance then this is going to need rework.
        assert_eq!(program.instructions(), content.as_bytes());
    }

    #[test]
    fn test_get_filename() {
        let filename = "test.bf";
        let program = Program::new(filename, "+-><.,[]");

        assert_eq!(program.filename(), Path::new(filename));
    }
}
