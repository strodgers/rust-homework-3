//! Representation of Brainfuck programs
//!
//! This includes capabilities to represent instructions and their provenance,
//! and to parse programs from files.

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// Enum for the raw instructions
#[derive(Debug, PartialEq)]
pub enum RawInstruction {
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
            line: line + 1,
            column: column + 1,
        }
    }
}

// Nice display string
impl fmt::Display for HumanReadableInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}\n", self.line, self.column, self.instruction)
    }
}

struct BracketBalancer {
    counter: i32,
}

impl BracketBalancer {
    fn new() -> Self {
        BracketBalancer { counter: 0 }
    }

    fn balance(&mut self, instruction: &HumanReadableInstruction) -> Result<(), String> {
        match instruction.instruction {
            RawInstruction::ConditionalForward => self.counter += 1,
            RawInstruction::ConditionalBackward => {
                self.counter -= 1;
                if self.counter < 0 {
                    return Err(From::from(format!(
                        "Unmatched closing bracket at line {}, column {}",
                        instruction.line, instruction.column
                    )));
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn balanced(&self) -> bool {
        self.counter == 0
    }
}

fn read_data<P: AsRef<Path>>(
    fname: P,
) -> Result<Vec<HumanReadableInstruction>, Box<dyn std::error::Error>> {
    // Read the data and parse into a vector of HumanReadableInstruction
    let buffread = BufReader::new(File::open(fname)?);
    let mut vec = Vec::new();
    let mut bracket_balancer = BracketBalancer::new();

    // Go through each line
    for (line_idx, line_result) in buffread.lines().enumerate() {
        let line = line_result?;

        // Go through each character
        for (col_idx, c) in line.chars().enumerate() {
            match RawInstruction::from_char(&c).ok_or("Invalid character") {
                Ok(instruction) => {
                    let hr_instruction =
                        HumanReadableInstruction::new(instruction, line_idx, col_idx);
                    if let Err(e) = bracket_balancer.balance(&hr_instruction) {
                        return Err(e.into());
                    }
                    vec.push(hr_instruction);
                }
                Err(_) => {}
            }
        }
    }

    if !bracket_balancer.balanced() {
        return Err("Unmatched opening bracket... somewhere".into());
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
    pub fn new<P: AsRef<Path>>(filename: P) -> Result<Self, Box<dyn Error>> {
        let instructions = read_data(&filename)?;

        Ok(Program {
            filename: filename.as_ref().to_owned(),
            instructions,
        })
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
        fn new(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let path = PathBuf::from(filename);
            let mut file = File::create(&path)?;
            writeln!(file, "+-><.,[]")?;

            Ok(TestFile { path })
        }

        fn path(&self) -> &PathBuf {
            &self.path
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            if let Err(e) = fs::remove_file(&self.path) {
                eprintln!("Cleanup error: Failed to delete file: {}", e);
            } else {
                println!("Cleanup: Test file deleted.");
            }
        }
    }

    #[test]
    fn test_human_readable_instruction_display() {
        let instruction = HumanReadableInstruction::new(RawInstruction::IncrementByte, 0, 0);
        assert_eq!(format!("{}", instruction), "1:1 Increment Byte (+)\n");
    }

    #[test]
    fn test_read_data() -> Result<(), Box<dyn std::error::Error>> {
        let test_file = TestFile::new("test.bf")?;
        let instructions = read_data(test_file.path().to_owned())?;
        assert_eq!(instructions.len(), 8);

        assert_eq!(instructions[0].instruction, RawInstruction::IncrementByte);
        assert_eq!(instructions[0].line, 1);
        assert_eq!(instructions[0].column, 1);

        assert_eq!(instructions[1].instruction, RawInstruction::DecrementByte);
        assert_eq!(instructions[1].line, 1);
        assert_eq!(instructions[1].column, 2);

        assert_eq!(
            instructions[2].instruction,
            RawInstruction::IncrementPointer
        );
        assert_eq!(instructions[2].line, 1);
        assert_eq!(instructions[2].column, 3);

        assert_eq!(
            instructions[3].instruction,
            RawInstruction::DecrementPointer
        );
        assert_eq!(instructions[3].line, 1);
        assert_eq!(instructions[3].column, 4);

        assert_eq!(instructions[4].instruction, RawInstruction::OutputByte);
        assert_eq!(instructions[4].line, 1);
        assert_eq!(instructions[4].column, 5);

        assert_eq!(instructions[5].instruction, RawInstruction::InputByte);
        assert_eq!(instructions[5].line, 1);
        assert_eq!(instructions[5].column, 6);

        assert_eq!(
            instructions[6].instruction,
            RawInstruction::ConditionalForward
        );
        assert_eq!(instructions[6].line, 1);
        assert_eq!(instructions[6].column, 7);

        assert_eq!(
            instructions[7].instruction,
            RawInstruction::ConditionalBackward
        );
        assert_eq!(instructions[7].line, 1);
        assert_eq!(instructions[7].column, 8);

        Ok(())
    }
}
