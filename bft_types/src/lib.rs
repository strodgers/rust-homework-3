//! Representation of Brainfuck programs
//!
//! This includes capabilities to represent instructions and their provenance,
//! and to parse programs from files.
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::io::{BufRead, BufReader, Read};

pub trait CellKind {
    type Value;

    fn increment(&mut self);
    fn decrement(&mut self);
    fn set(&mut self, value: Self);
    fn get(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn to_bytes(&self) -> Vec<u8>;
}

impl CellKind for u8 {
    type Value = u8;
    fn increment(&mut self) {
        *self = self.wrapping_add(1);
    }

    fn decrement(&mut self) {
        *self = self.wrapping_sub(1);
    }

    fn set(&mut self, value: Self::Value) {
        *self = value;
    }

    fn get(&self) -> Self {
        *self
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        Ok(*bytes.get(0).unwrap_or(&0))
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![*self]
    }
}

// Enum for the raw instructions
#[derive(Debug, PartialEq, Copy, Clone, Default)]
pub enum RawInstruction {
    IncrementPointer,    // >
    DecrementPointer,    // <
    IncrementByte,       // +
    DecrementByte,       // -
    OutputByte,          // .
    InputByte,           // ,
    ConditionalForward,  // [
    ConditionalBackward, // ]
    #[default]
    Undefined,
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
            RawInstruction::Undefined => write!(f, "Undefined"),
        }
    }
}

// Struct for the human readable instructions which includes a RawInstruction and the line and column index
#[derive(Debug, Copy, Clone, Default)]
pub struct HumanReadableInstruction {
    instruction: RawInstruction,
    line: usize,
    column: usize,
    index: usize,
}

impl HumanReadableInstruction {
    fn new(instruction: RawInstruction, line: usize, column: usize, index: usize) -> Self {
        HumanReadableInstruction {
            instruction,
            line: line + 1,
            column: column + 1,
            index: index,
        }
    }

    pub fn raw_instruction(&self) -> &RawInstruction {
        &self.instruction
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

// Nice display string
impl fmt::Display for HumanReadableInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}\n", self.line, self.column, self.instruction)
    }
}

#[derive(Debug)]
struct InstructionPreprocessor {
    bracket_positions: HashMap<usize, usize>,
    stack: Vec<usize>,
}

impl InstructionPreprocessor {
    fn new() -> Self {
        InstructionPreprocessor {
            bracket_positions: HashMap::new(),
            stack: Vec::new(),
        }
    }
    fn balanced(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.stack.len() {
            0 => Ok(()),
            _ => Err(From::from("Unmatched opening bracket... somewhere")),
        }
    }

    fn process(&mut self, hr_instruction: &HumanReadableInstruction) -> Result<(), String> {
        // Map all the bracket positions, error on unmatched closing brackets
        match hr_instruction.raw_instruction() {
            RawInstruction::ConditionalForward => {
                self.stack.push(hr_instruction.index);
            }
            RawInstruction::ConditionalBackward => match self.stack.pop() {
                Some(start_index) => {
                    self.bracket_positions
                        .insert(start_index, hr_instruction.index);
                    self.bracket_positions
                        .insert(hr_instruction.index, start_index);
                }
                None => {
                    return Err(format!(
                        "Unmatched closing bracket at line {}, column {}",
                        hr_instruction.line, hr_instruction.column
                    ));
                }
            },
            _ => {}
        }

        Ok(())
    }

    fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.bracket_positions.get(&index).copied()
    }
}
/// A Brainfuck program.
///
/// This struct holds the filename from which the program was loaded
/// and a vector of instructions.
#[derive(Debug)]

pub struct Program {
    instructions: Vec<HumanReadableInstruction>,
    preprocessor: InstructionPreprocessor,
}

impl Program {
    pub fn new<R: Read>(mut reader: R) -> Result<Self, Box<dyn Error>> {
        let mut preprocessor = InstructionPreprocessor::new();
        let instructions = Self::read_data(&mut reader, &mut preprocessor)?;

        Ok(Program {
            instructions,
            preprocessor,
        })
    }

    fn read_data<R: Read>(
        reader: R,
        preprocessor: &mut InstructionPreprocessor,
    ) -> Result<Vec<HumanReadableInstruction>, Box<dyn std::error::Error>> {
        // Read the data and parse into a vector of HumanReadableInstruction
        let buffread = BufReader::new(reader);
        let mut vec: Vec<HumanReadableInstruction> = Vec::new();

        // Go through each line
        let mut index = usize::default();
        for (line_idx, line_result) in buffread.lines().enumerate() {
            let line = line_result?;

            // Go through each character
            for (col_idx, c) in line.chars().enumerate() {
                match RawInstruction::from_char(&c).ok_or("Invalid character") {
                    Ok(instruction) => {
                        index += 1;
                        let hr_instruction: HumanReadableInstruction =
                            HumanReadableInstruction::new(instruction, line_idx, col_idx, index);
                        if let Err(e) = preprocessor.process(&hr_instruction) {
                            return Err(e.into());
                        }
                        vec.push(hr_instruction);
                    }
                    Err(_) => {}
                }
            }
        }

        match preprocessor.balanced() {
            Ok(()) => Ok(vec),
            Err(e) => {
                return Err(e);
            }
        }
    }

    pub fn instructions(&self) -> &Vec<HumanReadableInstruction> {
        &self.instructions
    }

    pub fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.preprocessor.get_bracket_position(index)
    }

    // pub fn next_position<C>(&self, cell_index: usize, cell_value: C) -> Option<&usize>
    // where
    // C: CellKind,
    // {
    //     if cell_value.get_value() == C::Value::zero() {
    //         return self.preprocessor.get_bracket_position(cell_index)
    //     }

    //     cell_index.increment();
    //     return Some(&cell_index)
    // }
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
        let instruction = HumanReadableInstruction::new(RawInstruction::IncrementByte, 0, 0, 0);
        assert_eq!(format!("{}", instruction), "1:1 Increment Byte (+)\n");
    }

    #[test]
    fn test_cellkind_from_bytes_for_u8() {
        let bytes = [5u8]; // Using a single byte

        let result: Result<u8, Box<dyn std::error::Error>> = <u8 as CellKind>::from_bytes(&bytes);

        // Asserts to check if the conversion was successful and correct.
        assert!(result.is_ok(), "Expected Ok(_) from from_bytes but got an Err");
        assert_eq!(result.unwrap(), 5, "Expected the byte value to be 5 after conversion");
    }
    // #[test]
    // fn test_read_data() -> Result<(), Box<dyn std::error::Error>> {
    //     let test_file = TestFile::new("test.bf")?;
    //     let mut preprocessor: &mut InstructionPreprocessor =
    //         &mut InstructionPreprocessor::new();
    //     let instructions = Program::read_data(test_file.path().to_owned(), &mut preprocessor)?;
    //     assert_eq!(instructions.len(), 8);

    //     assert_eq!(
    //         instructions[0].raw_instruction(),
    //         &RawInstruction::IncrementByte
    //     );
    //     assert_eq!(instructions[0].line, 1);
    //     assert_eq!(instructions[0].column, 1);

    //     assert_eq!(
    //         instructions[1].raw_instruction(),
    //         &RawInstruction::DecrementByte
    //     );
    //     assert_eq!(instructions[1].line, 1);
    //     assert_eq!(instructions[1].column, 2);

    //     assert_eq!(
    //         instructions[2].raw_instruction(),
    //         &RawInstruction::IncrementPointer
    //     );
    //     assert_eq!(instructions[2].line, 1);
    //     assert_eq!(instructions[2].column, 3);

    //     assert_eq!(
    //         instructions[3].raw_instruction(),
    //         &RawInstruction::DecrementPointer
    //     );
    //     assert_eq!(instructions[3].line, 1);
    //     assert_eq!(instructions[3].column, 4);

    //     assert_eq!(
    //         instructions[4].raw_instruction(),
    //         &RawInstruction::OutputByte
    //     );
    //     assert_eq!(instructions[4].line, 1);
    //     assert_eq!(instructions[4].column, 5);

    //     assert_eq!(
    //         instructions[5].raw_instruction(),
    //         &RawInstruction::InputByte
    //     );
    //     assert_eq!(instructions[5].line, 1);
    //     assert_eq!(instructions[5].column, 6);

    //     assert_eq!(
    //         instructions[6].raw_instruction(),
    //         &RawInstruction::ConditionalForward
    //     );
    //     assert_eq!(instructions[6].line, 1);
    //     assert_eq!(instructions[6].column, 7);

    //     assert_eq!(
    //         instructions[7].raw_instruction(),
    //         &RawInstruction::ConditionalBackward
    //     );
    //     assert_eq!(instructions[7].line, 1);
    //     assert_eq!(instructions[7].column, 8);

    //     Ok(())
    // }
}
