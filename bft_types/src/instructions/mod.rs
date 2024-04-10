use core::fmt;

/// Enum for the raw instructions
// TODO: So close, pity that those aren't documentation.
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
    // TODO: unnecessary?
    #[default]
    Undefined,
}

/// RawInstruction from a char value
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

/// Corresponding display strings
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

/// Struct for the human readable instructions which includes a RawInstruction and the line and column index
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

/// Nice display strings
impl fmt::Display for HumanReadableInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {}", self.line, self.column, self.instruction)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CollapsedInstruction {
    count: usize,
}
#[derive(Debug)]
pub(crate) struct InstructionPreprocessor {
    matching_brackets: Vec<Option<usize>>,
    collapsed_instructions: Vec<Option<CollapsedInstruction>>,
    optimize: bool,
}

/// Handles bracket matching and optionally collapsing repeated instructions.
impl InstructionPreprocessor {
    pub(crate) fn new(program_length: usize, optimize: bool) -> Self {
        // Preallocate the space, since we know it can't be more than the program length
        // Don't preallocate the collapsed instructions if we're not optimizing
        let collapsed_instructions = if optimize {
            vec![None; program_length]
        } else {
            vec![None; 0]
        };
        InstructionPreprocessor {
            matching_brackets: vec![None; program_length],
            collapsed_instructions,
            optimize,
        }
    }

    pub(crate) fn process(
        &mut self,
        instructions: &[HumanReadableInstruction],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Track the open brackets with a stack
        let mut open_brackets = Vec::new();
        let mut current_instruction: Option<RawInstruction> = None;
        let mut count = 0;
        let mut original_index = 0;

        for (index, hr_instruction) in instructions.iter().enumerate() {
            let raw_instruction = hr_instruction.raw_instruction();
            match raw_instruction {
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
            };

            if self.optimize {
                if Some(raw_instruction) == current_instruction {
                    count += 1; // Repeated instruction
                } else {
                    // Save the previous instruction
                    self.collapsed_instructions[original_index - count] =
                        Some(CollapsedInstruction { count });
                    // Reset for the new instruction
                    current_instruction = Some(raw_instruction);
                    count = 1;
                }
                original_index += 1;
            }
        }

        if self.optimize {
            // Save the last instruction
            if current_instruction.is_some() {
                self.collapsed_instructions[original_index - count] =
                    Some(CollapsedInstruction { count });
            }

            if !open_brackets.is_empty() {
                // TODO: This pattern is madness.
                let err_msg = "Unmatched opening bracket".to_string();
                log::error!("{}", err_msg);
                return Err(err_msg.into());
            }
        }

        Ok(())
    }

    pub(crate) fn collapsed_count(&self, original_index: usize) -> Option<usize> {
        if !self.optimize {
            return None;
        }
        self.collapsed_instructions[original_index].map(|collapsed| collapsed.count)
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
