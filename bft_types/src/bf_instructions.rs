use core::fmt;

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
    pub(crate) fn from_char(c: &char) -> Option<RawInstruction> {
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
    pub(crate) fn new(
        instruction: RawInstruction,
        line: usize,
        column: usize,
        index: usize,
    ) -> Self {
        HumanReadableInstruction {
            instruction,
            line: line + 1,
            column: column + 1,
            index,
        }
    }

    pub fn raw_instruction(&self) -> RawInstruction {
        self.instruction
    }

    pub fn line(&self) -> usize {
        self.line
    }

    pub fn column(&self) -> usize {
        self.column
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
pub(crate) struct InstructionPreprocessor {
    matching_brackets: Vec<Option<usize>>,
}

impl InstructionPreprocessor {
    pub(crate) fn new(program_length: usize) -> Self {
        // Preallocate the space, since we know it can't be more than the program length
        InstructionPreprocessor {
            matching_brackets: vec![None; program_length],
        }
    }

    pub(crate) fn process(
        &mut self,
        instructions: &[HumanReadableInstruction],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Track the open brackets with a stack
        let mut open_brackets = Vec::new();

        for (index, hr_instruction) in instructions.iter().enumerate() {
            match hr_instruction.raw_instruction() {
                RawInstruction::ConditionalForward => {
                    open_brackets.push(index);
                }
                RawInstruction::ConditionalBackward => {
                    // The closing bracket should match the last open bracket. If we can't pop then it
                    // is unmatched
                    if let Some(open_bracket) = open_brackets.pop() {
                        // Set both the open bracket and closing bracket to point to each other
                        self.matching_brackets[open_bracket] = Some(index);
                        self.matching_brackets[index] = Some(open_bracket);
                    } else {
                        let err_msg = format!(
                            "Unmatched closing bracket at line {}, column {}",
                            hr_instruction.line, hr_instruction.column
                        );
                        log::error!("{}", err_msg);
                        return Err(err_msg.into());
                    }
                }
                _ => {}
            }
        }

        if !open_brackets.is_empty() {
            let err_msg = "Unmatched opening bracket".to_string();
            log::error!("{}", err_msg);
            return Err(err_msg.into());
        }

        Ok(())
    }

    pub(crate) fn get_bracket_position(&self, index: usize) -> Option<usize> {
        // Try and find the matching position, which will be None if unmatched
        self.matching_brackets.get(index).and_then(|&pos| pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_readable_instruction_display() {
        let instruction = HumanReadableInstruction::new(RawInstruction::IncrementByte, 0, 0, 0);
        assert_eq!(format!("{}", instruction), "1:1 Increment Byte (+)");
    }
}
