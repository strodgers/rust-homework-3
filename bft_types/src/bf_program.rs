use crate::bf_instructions::{HumanReadableInstruction, InstructionPreprocessor, RawInstruction};
use std::{
    error::Error,
    io::{BufRead, BufReader, Read},
};

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
        let instructions = Self::read_data(reader)?;
        let mut preprocessor = InstructionPreprocessor::new(instructions.len());

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
    use crate::bf_instructions::{InstructionPreprocessor, RawInstruction};
    use bft_test_utils::{TestFile, TEST_FILE_NUM_INSTRUCTIONS};

    #[test]
    fn test_read_data() -> Result<(), Box<dyn std::error::Error>> {
        let instructions = Program::read_data(TestFile::new()?)?;
        let preprocessor: &mut InstructionPreprocessor =
            &mut InstructionPreprocessor::new(instructions.len());
        preprocessor.process(&instructions)?;

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
            assert_eq!(instruction.raw_instruction(), all_instructions[i]);
            assert_eq!(instruction.line(), 1);
            assert_eq!(instruction.column(), i + 1);
        }

        Ok(())
    }
}
