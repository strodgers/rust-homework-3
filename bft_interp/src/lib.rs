use bft_types::Program;
use bft_types::{CellKind, HumanReadableInstruction, RawInstruction};
use num_traits::{Num, NumCast};
use std::fmt;
use std::hash::Hash;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
#[derive(Debug)]
pub enum VMError {
    InvalidHeadPosition {
        position: usize,
        instruction: HumanReadableInstruction,
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
    GeneralError {
        reason: String,
    },
    TypeError {
        reason: String,
    },
}

impl<'a> fmt::Display for VMError {
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
            VMError::GeneralError { reason } => {
                write!(f, "General error: {}", reason)
            }
            VMError::TypeError { reason } => {
                write!(f, "Type error: {}", reason)
            }
        }
    }
}

/// A virtual machine for interpreting Brainfuck programs.
///
/// The type for each cell of the Brainfuck tape can be chosen by the
/// user of the virtual machine.
// TODO: remove once we're using this properly where
pub struct BrainfuckVM<'a, N>
where
    N: Num + NumCast + Hash + Clone + Eq + Copy,
{
    tape: Vec<N>,
    head: usize,
    allow_growth: bool,
    program_counter: usize,
    program: &'a Program,
}

impl<'a, N> BrainfuckVM<'a, N>
where
    N: Num + NumCast + Eq + Copy + CellKind + Hash + Default,
{
    pub fn new(program: &'a Program, cell_count: NonZeroUsize, allow_growth: bool) -> Self {
        BrainfuckVM {
            tape: vec![N::default(); cell_count.get()],
            head: 0,
            allow_growth,
            program_counter: 0,
            program,
        }
    }

    fn get_bracket_position(
        &'a self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<usize, VMError> {
        return self
            .program
            .get_bracket_position(hr_instruction.index())
            .ok_or(VMError::ProgramError {
                instruction: hr_instruction,
                reason: ("Could not find matching bracket".to_string()),
            });
    }

    pub fn current_cell(&mut self) -> Result<&mut N, VMError> {
        if let Some(cell) = self.tape.get_mut(self.head) {
            Ok(cell)
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.program_counter],
                reason: "Cell not found".to_string(),
            })
        }
    }

    pub fn current_cell_value(&self) -> Result<N, VMError> {
        if let Some(cell) = self.tape.get(self.head) {
            Ok(cell.to_owned())
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.program_counter],
                reason: "Cell not found".to_string(),
            })
        }
    }

    fn process_instruction(
        &mut self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<usize, VMError> {
        match hr_instruction.raw_instruction() {
            RawInstruction::IncrementPointer => {
                self.move_head_right()
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                    })?;
            }
            RawInstruction::DecrementPointer => {
                self.move_head_left()
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                    })?;
            }
            RawInstruction::IncrementByte => {
                self.increment_cell()?;
            }
            RawInstruction::DecrementByte => {
                self.decrement_cell()?;
            }
            RawInstruction::OutputByte => {
                let value = self.current_cell()?.get();
                // output_the_value_somewhere(value);
            }
            RawInstruction::InputByte => {
                // let input_value = get_input_value_somehow()
                // current_cell.set_value(input_value);
            }
            RawInstruction::ConditionalForward => {
                if self.current_cell()?.is_zero() {
                    // Jump forwards, 1 past the matching closing bracket
                    return Ok(self.get_bracket_position(hr_instruction)? + 1);
                };
            }
            RawInstruction::ConditionalBackward => {
                // Always jump back to opening bracket
                self.program_counter = self.get_bracket_position(hr_instruction)?;
            }
            RawInstruction::Undefined => {
                return Err(VMError::ProgramError {
                    instruction: hr_instruction,
                    reason: ("Undefined instruction".to_string()),
                });
            }
        }
        Ok(self.program_counter + 1)
    }

    pub fn interpret(&'a mut self) -> Result<(), VMError> {
        loop {
            let current_instruction = self
                .program
                .instructions()
                .get(self.program_counter)
                .clone();

            let hr_instruction = current_instruction
                .ok_or(VMError::GeneralError {
                    reason: ("Failed getting current instruction".to_string()),
                })?
                .clone();

            print!("{}", hr_instruction);

            self.program_counter = self.process_instruction(hr_instruction.clone())?;

            if self.program_counter >= self.program.instructions().len() {
                break;
            }

            // self.program_counter = index; // Update the program counter as you go
        }
        println!(); // To add a newline at the end of the output
        Ok(())
    }

    fn move_head_left(&mut self) -> Result<(), VMError> {
        if self.head == 0 {
            Err(VMError::GeneralError {
                reason: ("Failed to move head left".to_string()),
            })
        } else {
            self.head -= 1;
            Ok(())
        }
    }

    fn move_head_right(&mut self) -> Result<(), VMError> {
        self.head += 1;
        if self.head >= self.tape.len() {
            // Extend the tape if allowed
            if self.allow_growth {
                self.tape.push(N::default());
                return Ok(());
            } else {
                // If the tape cannot grow, then it's an error
                return Err(VMError::GeneralError {
                    reason: ("Failed to move head left".to_string()),
                });
            }
        }
        Ok(())
    }

    /// Increments the value of the cell pointed to by the head.
    fn increment_cell(&mut self) -> Result<(), VMError> {
        self.current_cell()?.increment();
        Ok(())
    }

    /// Decrements the value of the cell pointed to by the head.
    fn decrement_cell(&mut self) -> Result<(), VMError> {
        self.current_cell()?.decrement();
        Ok(())
    }

    pub fn read_value<R: Read>(&mut self, reader: &mut R) -> Result<(), VMError> {
        let mut buffer: [u8; 1] = [0; 1];

        // Attempt to read a byte from the reader
        reader
            .read_exact(&mut buffer)
            .map_err(|e| VMError::IOError {
                instruction: self.program.instructions()[self.program_counter],
                reason: e.to_string(),
            })?;

        match NumCast::from(buffer[0]) {
            Some(buf_value) => {
                self.current_cell()?.set(buf_value);
            }
            None => {
                return Err(VMError::IOError {
                    instruction: self.program.instructions()[self.program_counter],
                    reason: ("Failed to read value".to_string()),
                })
            }
        };
        Ok(())
    }

    pub fn write_value<W: Write>(&mut self, writer: &mut W) -> Result<(), VMError> {
        match NumCast::from(self.current_cell()?.get()) {
            Some(value) => writer.write_all(&[value]).map_err(|e| VMError::IOError {
                instruction: self.program.instructions()[self.program_counter],
                reason: e.to_string(),
            })?,
            None => {
                return Err(VMError::IOError {
                    instruction: self.program.instructions()[self.program_counter],
                    reason: ("Failed to read value".to_string()),
                })
            }
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_types::Program;
    use std::fs::File;
    use std::io::BufReader;
    use std::io::Cursor;

    // Helper function to create a simple test program
    fn test_program_from_file() -> Result<Program, Box<dyn std::error::Error>> {
        let file = File::open("/home/sam/git/rust-homework-3/example.bf")?;
        let program = Program::new(BufReader::new(file))?;
        Ok(program)
    }

    fn test_program_from_string(string: &str) -> Result<Program, Box<dyn std::error::Error>> {
        let program = Program::new(Cursor::new(string))?;
        Ok(program)
    }

    #[test]
    fn test_vm_initialization() -> Result<(), Box<dyn std::error::Error>> {
        let program: Program = test_program_from_file()?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

        assert_eq!(vm.head, 0);
        assert_eq!(vm.tape.len(), 10);

        Ok(())
    }

    #[test]
    fn test_move_head_right_success() -> Result<(), Box<dyn std::error::Error>> {
        let max_cells = 100000;
        let program_string = ">".repeat(max_cells);
        let program = test_program_from_string(&program_string)?;
        let cell_count = NonZeroUsize::new(10).unwrap();

        // Allow growth
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::<u8>::new(&program, cell_count, true);

        for (instruction_index, hr_instruction) in vm.program.instructions().iter().enumerate() {
            // Before processing, counter should be the same as index
            assert_eq!(vm.head, instruction_index);
            let next_instruction =
                vm.process_instruction(hr_instruction.to_owned())
                    .map_err(|err| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Error: {}", err),
                        )) as Box<dyn std::error::Error>
                    })?;
            // After processing, counter should have gone up by one
            assert_eq!(vm.head, instruction_index + 1);
            vm.program_counter = next_instruction;
        }
        Ok(())
    }

    #[test]
    fn test_move_head_left_error() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_string("<")?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::<u8>::new(&program, cell_count, false);

        let Some(hr_instruction) = vm.program.instructions().get(0) else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: "),
            )) as Box<dyn std::error::Error>);
        };

        match vm.process_instruction(hr_instruction.to_owned()) {
            Err(VMError::InvalidHeadPosition {
                position,
                instruction,
            }) => Ok(()),
            _ => panic!(
                "Expected VMError::InvalidHeadPosition error, but got a different or no error."
            ),
        }
    }

    #[test]
    fn test_increment_cell_success() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_string("+")?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, true);

        let Some(hr_instruction) = vm.program.instructions().get(0) else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: "),
            )) as Box<dyn std::error::Error>);
        };

        vm.process_instruction(hr_instruction.to_owned())
            .map_err(|err| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error: {}", err),
                )) as Box<dyn std::error::Error>
            })?;

        let cell_value = vm.tape.get(vm.head).unwrap();

        assert_eq!(cell_value.to_owned(), 1);

        Ok(())
    }

    #[test]
    fn test_decrement_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_string("-")?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, true);

        let Some(hr_instruction) = vm.program.instructions().get(0) else {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Error: "),
            )) as Box<dyn std::error::Error>);
        };

        vm.process_instruction(hr_instruction.to_owned())
            .map_err(|err| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error: {}", err),
                )) as Box<dyn std::error::Error>
            })?;

        let cell_value = vm.tape.get(vm.head).unwrap();

        assert_eq!(cell_value.to_owned(), 255);

        Ok(())
    }

    #[test]
    fn test_read_value_success<'a>() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_file()?;
        let mut vm: BrainfuckVM<'_, u8> =
            BrainfuckVM::new(&program, NonZeroUsize::new(10).unwrap(), true);
        let input_data = 42u8; // Example input byte
        let mut input_cursor = Cursor::new(vec![input_data]);

        vm.read_value(&mut input_cursor)
            .map_err(|err| format!("Error: {}", err))?;
        assert_eq!(
            vm.current_cell_value()
                .map_err(|err| format!("Error: {}", err))?,
            input_data,
            "Cell value should match the read value"
        );

        Ok(())
    }

    #[test]
    fn test_write_value_success() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_file()?;
        let mut vm: BrainfuckVM<'_, u8> =
            BrainfuckVM::new(&program, NonZeroUsize::new(10).unwrap(), true);
        vm.current_cell()
            .map_err(|err| format!("Error: {}", err))?
            .set(42);

        let mut output_cursor = Cursor::new(Vec::new());
        vm.write_value(&mut output_cursor)
            .map_err(|err| format!("Error: {}", err))?;

        assert_eq!(
            output_cursor.into_inner(),
            vec![42],
            "Output should contain the cell's value"
        );

        Ok(())
    }
}
