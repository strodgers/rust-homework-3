//! Representation of Brainfuck programs
//!
//! This includes capabilities to represent instructions and their provenance,
//! and to parse programs from files.
use num_traits::{FromBytes, Num, NumCast};
use std::cmp::PartialEq;
use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};
use std::io::{BufRead, BufReader, Read};
pub trait CellKind:
    Num + NumCast + Copy + PartialEq + Eq + Display + Debug + FromBytes + Default
{
    type Value;

    fn increment(&mut self);
    fn decrement(&mut self);
    fn set(&mut self, value: Self);
    fn get(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn to_bytes(&self) -> Vec<u8>;
    fn bytes_per_cell() -> usize {
        std::mem::size_of::<Self::Value>()
    }
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
        // Already a u8
        Ok(bytes[0])
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
#[derive(Debug, Copy, Clone, Default, PartialEq)]
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
            index,
        }
    }

    pub fn undefined() -> HumanReadableInstruction {
        // Static so we can use it as a default value and when we need to error out
        static UNDEFINED: HumanReadableInstruction = HumanReadableInstruction {
            instruction: RawInstruction::Undefined,
            line: 0,
            column: 0,
            index: 0,
        };
        UNDEFINED
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
        write!(f, "{}:{} {}", self.line, self.column, self.instruction)
    }
}

#[derive(Debug)]
struct InstructionPreprocessor {
    open_brackets: Vec<usize>,
    matched_brackets: Vec<(usize, usize)>,
}

impl InstructionPreprocessor {
    fn new() -> Self {
        InstructionPreprocessor {
            open_brackets: Vec::new(),
            matched_brackets: Vec::new(),
        }
    }

    fn process(
        &mut self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let index = hr_instruction.index;
        match hr_instruction.raw_instruction() {
            RawInstruction::ConditionalForward => {
                self.open_brackets.push(index);
            }
            RawInstruction::ConditionalBackward => {
                if let Some(open_bracket) = self.open_brackets.pop() {
                    self.matched_brackets.push((open_bracket, index));
                } else {
                    let err_msg = format!(
                        "Unmatched closing bracket at line {}, column {}",
                        hr_instruction.line, hr_instruction.column
                    );
                    log::error!("{}", err_msg); // Log the error
                    return Err(err_msg.into()); // Also return the error for handling
                }
            }
            _ => {}
        }
        Ok(())
    }

    // Retrieve the matching bracket's position
    fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.matched_brackets
            .iter()
            .find(|(open, close)| *open == index || *close == index)
            .map(|(open, close)| if *open == index { *close } else { *open })
    }

    fn balanced(&self) -> bool {
        self.open_brackets.is_empty()
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
    pub fn new<R: Read>(reader: R) -> Result<Self, Box<dyn Error>> {
        let mut preprocessor = InstructionPreprocessor::new();
        let instructions = Self::read_data(reader, &mut preprocessor)?;

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
                if let Ok(instruction) = RawInstruction::from_char(&c).ok_or("Invalid character") {
                    {
                        index += 1;
                        let hr_instruction: HumanReadableInstruction =
                            HumanReadableInstruction::new(instruction, line_idx, col_idx, index);
                        preprocessor.process(hr_instruction)?;
                        vec.push(hr_instruction);
                    }
                }
            }
        }

        if !preprocessor.balanced() {
            return Err("Unbalanced brackets".into());
        }

        Ok(vec)
    }

    pub fn instructions(&self) -> &Vec<HumanReadableInstruction> {
        &self.instructions
    }

    pub fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.preprocessor.get_bracket_position(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_test_utils::{TestFile, TEST_FILE_NUM_INSTRUCTIONS};

    #[test]
    fn test_human_readable_instruction_display() {
        let instruction = HumanReadableInstruction::new(RawInstruction::IncrementByte, 0, 0, 0);
        assert_eq!(format!("{}", instruction), "1:1 Increment Byte (+)");
    }

    #[test]
    fn test_cellkind_from_bytes_for_u8() {
        let bytes = [5u8]; // Using a single byte

        let result: Result<u8, Box<dyn std::error::Error>> = <u8 as CellKind>::from_bytes(&bytes);

        // Asserts to check if the conversion was successful and correct.
        assert!(
            result.is_ok(),
            "Expected Ok(_) from from_bytes but got an Err"
        );
        assert_eq!(
            result.unwrap(),
            5,
            "Expected the byte value to be 5 after conversion"
        );
    }
    #[test]
    fn test_read_data() -> Result<(), Box<dyn std::error::Error>> {
        let mut preprocessor: &mut InstructionPreprocessor = &mut InstructionPreprocessor::new();
        let instructions = Program::read_data(TestFile::new()?, &mut preprocessor)?;
        assert_eq!(instructions.len(), TEST_FILE_NUM_INSTRUCTIONS);

        // "+[-[<<[+[--->]-[<<<]]]>>>-]"
        let all_instructions = [
            RawInstruction::IncrementByte,       // +
            RawInstruction::ConditionalForward,  // [
            RawInstruction::DecrementByte,       // -
            RawInstruction::ConditionalForward,  // [
            RawInstruction::DecrementPointer,    // <
            RawInstruction::DecrementPointer,    // <
            RawInstruction::ConditionalForward,  // [
            RawInstruction::IncrementByte,       // +
            RawInstruction::ConditionalForward,  // [
            RawInstruction::DecrementByte,       // -
            RawInstruction::DecrementByte,       // -
            RawInstruction::DecrementByte,       // -
            RawInstruction::IncrementPointer,    // >
            RawInstruction::ConditionalBackward, // ]
            RawInstruction::DecrementByte,       // -
            RawInstruction::ConditionalForward,  // [
            RawInstruction::DecrementPointer,    // <
            RawInstruction::DecrementPointer,    // <
            RawInstruction::DecrementPointer,    // <
            RawInstruction::ConditionalBackward, // ]
            RawInstruction::ConditionalBackward, // ]
            RawInstruction::ConditionalBackward, // ]
            RawInstruction::IncrementPointer,    // >
            RawInstruction::IncrementPointer,    // >
            RawInstruction::IncrementPointer,    // >
            RawInstruction::DecrementByte,       // -
            RawInstruction::ConditionalBackward, // ]
        ];

        for (i, instruction) in instructions.iter().enumerate() {
            assert_eq!(instruction.raw_instruction(), &all_instructions[i]);
            assert_eq!(instruction.line, 1);
            assert_eq!(instruction.column, i + 1);
        }

        Ok(())
    }
}
