use bft_types::Program;
use std::num::NonZeroUsize;
use bft_types::HumanReadableInstruction;

pub enum VMError<'a> {
    InvalidHeadPosition {
        position: usize,
        instruction: &'a HumanReadableInstruction
     },
    // TODO: add more errors
}

/// A virtual machine for interpreting Brainfuck programs.
///
/// The type for each cell of the Brainfuck tape can be chosen by the
/// user of the virtual machine.
// TODO: remove once we're using this properly
#[allow(dead_code)]
pub struct BrainfuckVM<'a, T: Default + Clone> {
    tape: Vec<T>,
    head: usize,
    allow_growth: bool,
    program_counter: usize,
    program: &'a Program,
}


// impl<T> BrainfuckVM<T> {
//     /// Returns a reference to the tape of the virtual machine.
//     pub fn tape(&self) -> &Vec<T> {
//         &self.tape
//     }
//
//     /// Returns the current head position on the tape.
//     pub fn head(&self) -> usize {
//         self.head
//     }
//
//     /// Returns whether the tape is allowed to grow beyond its initial size.
//     pub fn allow_growth(&self) -> bool {
//         self.allow_growth
//     }
// }

impl<'a, T: Default + Clone> BrainfuckVM<'a, T> {
    pub fn new(program: &'a Program, cell_count: NonZeroUsize, allow_growth: bool) -> Self {
        BrainfuckVM {
            tape: vec![T::default(); cell_count.get()],
            head: 0,
            allow_growth,
            program_counter: 0,
            program
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
        assert!(matches!(vm.move_head_left(), Err(VMError::InvalidHeadPosition { position, instruction: _ }) if position == 0));

        Ok(())
    }

    // Consider adding more tests for other functionalities and edge cases
}

