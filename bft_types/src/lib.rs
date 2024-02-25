//! Representation of Brainfuck programs
//!
//! This includes capabilities to represent instructions and their provenance,
//! and to parse programs from files.

use std::path::{Path, PathBuf};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Enum for the raw instructions
#[derive(Debug, PartialEq)]
enum RawInstruction {
    IncrementPointer,    // >
    DecrementPointer,    // <
    IncrementByte,       // +
    DecrementByte,       // -
    OutputByte,          // .
    InputByte,           // ,
    ConditionalForward,  // [
    ConditionalBackward, // ]
}

// Implement a from_char method for RawInstruction
impl RawInstruction {
    fn from_char(c: &char) -> Option<RawInstruction> {
        match c {
            '>' => Some(RawInstruction::IncrementPointer),
            '<' => Some(RawInstruction::DecrementPointer),
            '+' => Some(RawInstruction::IncrementByte),
            '-' => Some(RawInstruction::DecrementByte),
            '.' => Some(RawInstruction::OutputByte),
            ',' => Some(RawInstruction::InputByte),
            '[' => Some(RawInstruction::ConditionalForward),
            ']' => Some(RawInstruction::ConditionalBackward),
            _ => None,
        }
    }
}

// Corresponding display strings
impl fmt::Display for RawInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawInstruction::IncrementPointer => write!(f, "Increment Pointer (>)"),
            RawInstruction::DecrementPointer => write!(f, "Decrement Pointer (<)"),
            RawInstruction::IncrementByte => write!(f, "Increment Byte (+)"),
            RawInstruction::DecrementByte => write!(f, "Decrement Byte (-)"),
            RawInstruction::OutputByte => write!(f, "Output Byte (.)"),
            RawInstruction::InputByte => write!(f, "Input Byte (,)"),
            RawInstruction::ConditionalForward => write!(f, "Conditional Forward ([)"),
            RawInstruction::ConditionalBackward => write!(f, "Conditional Backward (])"),
        }
    }
}


// Struct for the human readable instructions which includes a RawInstruction and the line and column index
#[derive(Debug)]
pub struct HumanReadableInstruction {
    instruction: RawInstruction,
    line: usize,
    column: usize,
}

impl HumanReadableInstruction {
    fn new(instruction: RawInstruction, line: usize, column: usize) -> Self {
        HumanReadableInstruction {
            instruction,
            line,
            column,
        }
    }
}

// Nice display string
impl fmt::Display for HumanReadableInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}\n", self.line, self.column, self.instruction)
    }
}

fn read_data<P: AsRef<Path>>(fname: P) -> Result<Vec<HumanReadableInstruction>, Box<dyn std::error::Error>> {
    // Read the data and parse into a vector of HumanReadableInstruction
    let buffread = BufReader::new(File::open(fname)?);
    let mut vec = Vec::new();

    // Go through each line
    for (line_idx, line_result) in buffread.lines().enumerate() {
        let line = line_result?;

        // Go through each character
        for (col_idx, c) in line.chars().enumerate() {
            match RawInstruction::from_char(&c).ok_or("Invalid character") {
                Ok(instruction) => {
                    // I am adding 1 to the column and row index here for human readability,
                    // although that does make my output column index 1 off the example given
                    // in the homework description...
                    vec.push(HumanReadableInstruction::new(
                        instruction,
                        line_idx + 1,
                        col_idx + 1,
                    ));
                    // println!("Valid character {} at line {} column {}",  c, line_idx + 1, col_idx + 1);
                }
                Err(_) => {
                    // println!("Invalid character {} at line {} column {}", c, line_idx + 1, col_idx + 1);
                }
            }
        }
    }

    Ok(vec)
}
/// A Brainfuck program.
///
/// This struct holds the filename from which the program was loaded
/// and a vector of instructions.
#[derive(Debug)]
pub struct Program {
    filename: PathBuf,
    instructions: Vec<HumanReadableInstruction>,
}

impl Program {
    pub fn new<P: AsRef<Path>>(filename: P) -> Self {
        Program {
            filename: filename.as_ref().to_owned(),
            instructions: read_data(&filename).expect("Failed reading file")
        }
    }

    /// The filename from which the program was loaded
    pub fn filename(&self) -> &Path {
        &self.filename
    }

    /// The instructions in the program
    pub fn instructions(&self) -> &Vec<crate::HumanReadableInstruction> {
        &self.instructions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    
    struct TestFile {
        path: PathBuf,
    }
    
    impl TestFile {
        fn new(filename: &str) -> Self {
            let path = PathBuf::from(filename);
            let mut file = File::create(&path).expect("Failed to create file");
            writeln!(file, "+-><.,[]").expect("Failed to write to file");
    
            TestFile { path }
        }

        fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            fs::remove_file(&self.path).expect("Failed to delete file");
            println!("Cleanup: Test file deleted.");
        }
    }


    #[test]
    fn test_human_readable_instruction_display() {
        let instruction = HumanReadableInstruction::new(
            RawInstruction::IncrementByte,
            1,
            1,
        );
        assert_eq!(format!("{}", instruction), "1:1 Increment Byte (+)\n");
    }

    #[test]
    fn test_read_data() {
        let instructions = read_data(
            TestFile::new("test.bf").path().to_owned()
        ).expect("Failed to read data");
        assert_eq!(instructions.len(), 8);

        assert_eq!(instructions[0].instruction, RawInstruction::IncrementByte);
        assert_eq!(instructions[0].line, 1);
        assert_eq!(instructions[0].column, 1);

        assert_eq!(instructions[1].instruction, RawInstruction::DecrementByte);
        assert_eq!(instructions[1].line, 1);
        assert_eq!(instructions[1].column, 2);

        assert_eq!(instructions[2].instruction, RawInstruction::IncrementPointer);
        assert_eq!(instructions[2].line, 1);
        assert_eq!(instructions[2].column, 3);

        assert_eq!(instructions[3].instruction, RawInstruction::DecrementPointer);
        assert_eq!(instructions[3].line, 1);
        assert_eq!(instructions[3].column, 4);

        assert_eq!(instructions[4].instruction, RawInstruction::OutputByte);
        assert_eq!(instructions[4].line, 1);
        assert_eq!(instructions[4].column, 5);

        assert_eq!(instructions[5].instruction, RawInstruction::InputByte);
        assert_eq!(instructions[5].line, 1);
        assert_eq!(instructions[5].column, 6);

        assert_eq!(instructions[6].instruction, RawInstruction::ConditionalForward);
        assert_eq!(instructions[6].line, 1);
        assert_eq!(instructions[6].column, 7);

        assert_eq!(instructions[7].instruction, RawInstruction::ConditionalBackward);
        assert_eq!(instructions[7].line, 1);
        assert_eq!(instructions[7].column, 8);
    }

}
