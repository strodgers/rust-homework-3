use bft_types::HumanReadableInstruction;
use bft_types::Program;
use std::fmt;
use std::num::NonZeroUsize;
#[derive(Debug)]
pub enum VMError<'a> {
    InvalidHeadPosition {
        position: usize,
        instruction: &'a HumanReadableInstruction,
    },
    CellOperationError {
        position: usize,
        instruction: &'a HumanReadableInstruction,
        reason: String,
    },
}

impl<'a> fmt::Display for VMError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VMError::InvalidHeadPosition {
                position,
                instruction,
            } => {
                write!(f, "Invalid head position: {} at {}", position, instruction)
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
        }
    }
}

pub trait CellKind {
    fn increment(&mut self);
    fn decrement(&mut self);
    fn set_value(&mut self, value: u8);
    fn get_value(&self) -> u8;
}

impl CellKind for u8 {
    fn increment(&mut self) {
        *self = self.wrapping_add(1);
    }

    fn decrement(&mut self) {
        *self = self.wrapping_sub(1);
    }

    fn set_value(&mut self, value: u8) {
        *self = value;
    }

    fn get_value(&self) -> u8 {
        *self
    }
}
/// A virtual machine for interpreting Brainfuck programs.
///
/// The type for each cell of the Brainfuck tape can be chosen by the
/// user of the virtual machine.
// TODO: remove once we're using this properly
pub struct BrainfuckVM<'a, T: CellKind + Default + Clone> {
    tape: Vec<T>,
    head: usize,
    allow_growth: bool,
    program_counter: usize,
    program: &'a Program,
}

impl<'a, T: CellKind + Default + Clone> BrainfuckVM<'a, T> {
    pub fn new(program: &'a Program, cell_count: NonZeroUsize, allow_growth: bool) -> Self {
        BrainfuckVM {
            tape: vec![T::default(); cell_count.get()],
            head: 0,
            allow_growth,
            program_counter: 0,
            program,
        }
    }

    pub fn interpret(&mut self) {
        for (index, instruction) in self.program.instructions().iter().enumerate() {
            print!("{}", instruction);
            self.program_counter = index; // Update the program counter as you go
        }
        println!(); // To add a newline at the end of the output
    }

    pub fn move_head_left(&mut self) -> Result<(), VMError> {
        if self.head == 0 {
            Err(VMError::InvalidHeadPosition {
                position: self.head,
                instruction: &self.program.instructions()[self.program_counter],
            })
        } else {
            self.head -= 1;
            Ok(())
        }
    }

    pub fn move_head_right(&mut self) -> Result<(), VMError> {
        self.head += 1;
        if self.head >= self.tape.len() {
            // Extend the tape if allowed
            if self.allow_growth {
                self.tape.push(T::default());
            } else {
                // If the tape cannot grow, then it's an error
                return Err(VMError::InvalidHeadPosition {
                    position: self.head,
                    instruction: &self.program.instructions()[self.program_counter],
                });
            }
        }
        Ok(())
    }

    fn get_cell(&mut self) -> Result<&mut T, VMError<'a>> {
        if let Some(cell) = self.tape.get_mut(self.head) {
            Ok(cell)
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: &self.program.instructions()[self.program_counter],
                reason: "Cell not found".to_string(),
            })
        }
    }
    /// Increments the value of the cell pointed to by the head.
    pub fn increment_cell(&mut self) -> Result<(), VMError<'a>> {
        let cell = self.get_cell()?;
        cell.increment();
        Ok(())
    }

    /// Decrements the value of the cell pointed to by the head.
    pub fn decrement_cell(&mut self) -> Result<(), VMError<'a>> {
        let cell = self.get_cell()?;
        cell.decrement();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_types::Program;

    // Helper function to create a simple test program
    fn create_test_program() -> Result<Program, Box<dyn std::error::Error>> {
        let program = Program::new(
            "/home/sam/git/rust-homework-3/example.bf", // TODO: replace with a valid path
        )?;
        Ok(program)
    }

    #[test]
    fn test_vm_initialization() -> Result<(), Box<dyn std::error::Error>> {
        let program = create_test_program()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

        assert_eq!(vm.head, 0);
        assert_eq!(vm.tape.len(), 10);

        Ok(())
    }

    #[test]
    fn test_move_head_right_success() -> Result<(), Box<dyn std::error::Error>> {
        let program = create_test_program()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

        assert!(vm.move_head_right().is_ok());
        assert_eq!(vm.head, 1);

        Ok(())
    }

    #[test]
    fn test_move_head_left_error() -> Result<(), Box<dyn std::error::Error>> {
        let program = create_test_program()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

        // Attempt to move head left when it's already at the start should result in an error
        assert!(
            matches!(vm.move_head_left(), Err(VMError::InvalidHeadPosition { position, instruction: _ }) if position == 0)
        );

        Ok(())
    }

    #[test]
    fn test_increment_cell_success() -> Result<(), Box<dyn std::error::Error>> {
        let program = create_test_program()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, true);

        // Ensure the initial value is 0
        assert_eq!(
            vm.get_cell()
                .map_err(|err| format!("Error: {}", err))?
                .get_value(),
            0
        );

        // Increment the value at the head position
        vm.increment_cell().map_err(|err| format!("Error: {}", err))?;
        assert_eq!(
            vm.get_cell()
                .map_err(|err| format!("Error: {}", err))?
                .get_value(),
            1,
            "Cell value should be incremented to 1"
        );

        Ok(())
    }

    #[test]
    fn test_decrement_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let program = create_test_program()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, true);

        // Decrement the default value (0), expecting wrapping to max value for u8
        vm.decrement_cell()
            .map_err(|err| format!("Error: {}", err))?;
        assert_eq!(
            vm.get_cell()
                .map_err(|err| format!("Error: {}", err))?
                .get_value(),
            255,
            "Cell value should wrap to 255 after decrementing 0"
        );

        Ok(())
    }
}
