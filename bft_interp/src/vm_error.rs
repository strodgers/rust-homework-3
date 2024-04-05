use core::fmt;

use bft_types::bf_cellkind::CellKind;
use bft_types::bf_instructions::HumanReadableInstruction;
use bft_types::vm_state::VMState;

// Simple error types with static strings
#[derive(Debug)]
pub enum VMErrorSimple<N>
where
    N: CellKind,
{
    // Represents a generic error with a static reason
    GeneralError { reason: String },
    // Indicates an error related to type mismatches or issues
    TypeError { reason: String },
    // Errors occurring during the construction of the VM, typically due to misconfiguration
    BuilderError { reason: String },
    // Signifies the normal completion of program execution, including a COPY of the final state of the VM for inspection
    EndOfProgram { final_state: Option<VMState<N>> },
}

// Specific errors that provide context such as the problematic instruction and a detailed reason
#[derive(Debug)]
pub enum VMError<N>
where
    N: CellKind,
{
    Simple(VMErrorSimple<N>),
    InvalidHeadPosition {
        position: usize,
        instruction: HumanReadableInstruction,
        reason: String,
    },
    CellOperationError {
        position: usize,
        instruction: HumanReadableInstruction,
        reason: String,
    },
    IOError {
        instruction: HumanReadableInstruction,
        reason: String,
    },
    ProgramError {
        instruction: HumanReadableInstruction,
        reason: String,
    },
}

// So we can use it with Box dyn Error
impl<N> std::error::Error for VMError<N> where N: CellKind {}

impl<N> fmt::Display for VMError<N>
where
    N: CellKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VMError::InvalidHeadPosition {
                position,
                instruction,
                reason,
            } => {
                write!(f, "{}: {} at {}", reason, position, instruction)
            }
            VMError::CellOperationError {
                position,
                instruction,
                reason,
            } => {
                write!(
                    f,
                    "Cell operation error: {} at {}. Reason: {}",
                    position, instruction, reason
                )
            }
            VMError::IOError {
                instruction,
                reason,
            } => {
                write!(f, "IO error at {}. Reason: {}", instruction, reason)
            }
            VMError::ProgramError {
                instruction,
                reason,
            } => {
                write!(f, "Program error at {}. Reason: {}", instruction, reason)
            }
            VMError::Simple(VMErrorSimple::GeneralError { reason }) => {
                write!(f, "General error: {}", reason)
            }
            VMError::Simple(VMErrorSimple::TypeError { reason }) => {
                write!(f, "Type error: {}", reason)
            }
            VMError::Simple(VMErrorSimple::BuilderError { reason }) => {
                write!(f, "Builder error: {}", reason)
            }
            VMError::Simple(VMErrorSimple::EndOfProgram { final_state }) => {
                match final_state {
                    Some(state) => write!(f, "End of program, final state: {}", state),
                    None => write!(f, "End of program, no final state available"),
                }
            }
        }
    }
}
