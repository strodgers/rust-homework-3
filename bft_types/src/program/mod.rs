use crate::instructions::{HumanReadableInstruction, InstructionPreprocessor, RawInstruction};
use std::{
    error::Error,
    io::{BufRead, BufReader, Read},
};

/// A Brainfuck program initialized with a reader. Uses an `InstructionPreprocessor` to process
/// instructions. Holds a vector containing all valid instructions.
#[derive(Debug)]

pub struct Program {
    instructions: Vec<HumanReadableInstruction>,
    preprocessor: InstructionPreprocessor,
}

impl Program {
    // TODO: Libraries should never return Box<dyn Error> - derive proper error enums please
    pub fn new<R: Read>(reader: R, optimize: bool) -> Result<Self, Box<dyn Error>> {
        let instructions = Self::read_data(reader)?;
        let mut preprocessor = InstructionPreprocessor::new(instructions.len(), optimize);

        preprocessor.process(&instructions)?;
        Ok(Program {
            instructions,
            preprocessor,
        })
    }

    fn read_data<R: Read>(
        reader: R,
    ) -> Result<Vec<HumanReadableInstruction>, Box<dyn std::error::Error>> {
        // Read the data and parse into a vector of HumanReadableInstruction
        let buffread = BufReader::new(reader);
        let mut vec: Vec<HumanReadableInstruction> = Vec::new();

        // Go through each line
        let mut index = 0;
        for (line_idx, line_result) in buffread.lines().enumerate() {
            let line = line_result?;

            // Go through each character
            for (col_idx, c) in line.chars().enumerate() {
                if let Ok(instruction) = RawInstruction::from_char(&c).ok_or("Invalid character") {
                    {
                        let hr_instruction: HumanReadableInstruction =
                            HumanReadableInstruction::new(instruction, line_idx, col_idx, index);
                        vec.push(hr_instruction);
                        index += 1;
                    }
                }
            }
        }

        Ok(vec)
    }

    // TODO: As per previous discussions, this should be &[T] not &Vec<T>
    pub fn instructions(&self) -> &Vec<HumanReadableInstruction> {
        &self.instructions
    }

    pub fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.preprocessor.get_bracket_position(index)
    }

    pub fn collapsed_count(&self, original_index: usize) -> Option<usize> {
        self.preprocessor.collapsed_count(original_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::{InstructionPreprocessor, RawInstruction};
    use std::io::Write;
    use std::io::{Seek, SeekFrom};
    use tempfile::NamedTempFile;
    #[test]
    fn test_read_data() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = NamedTempFile::new()?;
        // Don't do any input/output for this kind of test
        let program_string = "+[-[<<[+[--->]-[<<<]]]>>>-]";
        write!(file, "{}", program_string)?;
        file.seek(SeekFrom::Start(0))?;

        let instructions = Program::read_data(file)?;
        let preprocessor: &mut InstructionPreprocessor =
            &mut InstructionPreprocessor::new(instructions.len(), true);
        preprocessor.process(&instructions)?;

        assert_eq!(instructions.len(), program_string.len());

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
            assert_eq!(instruction.raw_instruction(), all_instructions[i]);
            assert_eq!(instruction.line(), 1);
            assert_eq!(instruction.column(), i + 1);
        }

        Ok(())
    }
}
