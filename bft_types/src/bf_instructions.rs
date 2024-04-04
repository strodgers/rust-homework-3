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
}

// Nice display string
impl fmt::Display for HumanReadableInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}", self.line, self.column, self.instruction)
    }
}

#[derive(Debug)]
pub(crate) struct InstructionPreprocessor {
    open_brackets: Vec<usize>,
    matched_brackets: Vec<(usize, usize)>,
}

impl InstructionPreprocessor {
    pub(crate) fn new() -> Self {
        InstructionPreprocessor {
            open_brackets: Vec::new(),
            matched_brackets: Vec::new(),
        }
    }

    pub(crate) fn process(
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
    pub(crate) fn get_bracket_position(&self, index: usize) -> Option<usize> {
        self.matched_brackets
            .iter()
            .find(|(open, close)| *open == index || *close == index)
            .map(|(open, close)| if *open == index { *close } else { *open })
    }

    pub(crate) fn balanced(&self) -> bool {
        self.open_brackets.is_empty()
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
