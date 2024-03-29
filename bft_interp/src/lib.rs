use bft_types::Program;
use bft_types::{CellKind, HumanReadableInstruction, RawInstruction};
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Write};
use std::num::NonZeroUsize;
use std::{cell, io};

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
    BuilderError {
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
            VMError::BuilderError { reason } => {
                write!(f, "Builder error: {}", reason)
            }
        }
    }
}

// const SUPPORTED_CELL_TYPEIDS: HashMap<&TypeId, &str> = {};

#[derive(Default)]
pub struct VMBuilder<'a, R, W>
where
    R: Read,
    W: Write,
{
    cell_kind: Option<TypeId>,
    cell_count: Option<NonZeroUsize>,
    allow_growth: Option<bool>,
    input_reader: Option<Box<R>>,
    output_writer: Option<Box<W>>,
    program_reader: Option<Box<dyn Read + 'a>>,
}

impl<'a, R, W> VMBuilder<'a, R, W>
where
    R: Read + 'static,
    W: Write + 'static,
{
    pub fn new() -> Self {
        VMBuilder {
            cell_kind: None,
            cell_count: None,
            allow_growth: None,
            input_reader: None,
            output_writer: None,
            program_reader: None,
        }
    }

    pub fn set_io(mut self, input: R, output: W) -> Self {
        self.input_reader = Some(Box::new(input));
        self.output_writer = Some(Box::new(output));
        self
    }

    // fn set_program_reader(mut self, mut reader: R + 'static ) -> Self {
    //     self.program_reader = Some(Box::new(reader));
    //     self
    // }

    pub fn set_program_file(mut self, file: BufReader<File>) -> Self {
        self.program_reader = Some(Box::new(BufReader::new(file)) as Box<dyn Read + 'a>);
        self
    }

    pub fn build<N>(self) -> Result<BrainfuckVM<N>, VMError>
    where
        N: CellKind,
        R: Read,
        W: Write,
    {
        // Program must be set somehow
        let program_reader: Box<dyn Read> = match self.program_reader {
            Some(reader) => reader,
            None => {
                return Err(VMError::BuilderError {
                    reason: "Program reader must be set.".to_string(),
                })
            }
        };

        // Default IO to use stdin and stdout
        let input_reader: Box<dyn Read + 'static> = match self.input_reader {
            Some(reader) => reader,
            None => {
                println!("Using default stdin");
                Box::new(io::stdin().lock())
            }
        };

        let output_writer: Box<dyn Write + 'static> = match self.output_writer {
            Some(reader) => reader,
            None => {
                println!("Using default stdout");
                Box::new(io::stdout().lock())
            }
        };

        // If no cell type provided, default to u8
        let cell_kind: TypeId = self.cell_kind.unwrap_or({
            println!("Using default cell kind u8");
            TypeId::of::<u8>()
        });

        // Must be a supported type
        // if !SUPPORTED_CELL_TYPEIDS.contains_key(&cell_kind) {
        //     return Err(VMError::BuilderError {
        //         reason: format!(
        //             "Cell type not supported. Supported types: {:?}",
        //             SUPPORTED_CELL_TYPEIDS.values()
        //         ),
        //     });
        // }

        // If no cell count provided, default to 30,000
        let cell_count = self.cell_count.unwrap_or({
            println!("Using default cell count 30000");
            NonZeroUsize::new(30000).unwrap()
        });

        // If no allow growth provided, default to false
        let allow_growth = self.allow_growth.unwrap_or({
            println!("Using default allow growth false");
            false
        });

        let program = Program::new(program_reader).map_err(|err| VMError::BuilderError {
            reason: format!("Failed to create program: {}", err),
        })?;

        Ok(BrainfuckVM::new(
            program,
            cell_count,
            allow_growth,
            input_reader,
            output_writer,
        ))
    }
}

/// A virtual machine for interpreting Brainfuck programs.
///
/// The type for each cell of the Brainfuck tape can be chosen by the
/// user of the virtual machine.
// TODO: remove once we're using this properly where
pub struct BrainfuckVM<N>
where
    N: CellKind,
{
    tape: Vec<N>,
    head: usize,
    allow_growth: bool,
    program_counter: usize,
    program: Program,
    input_reader: Box<dyn Read>,
    output_writer: Box<dyn Write>,
}

impl<'a, N> BrainfuckVM<N>
where
    N: CellKind,
{
    pub fn new(
        // mut tape: Vec<N>,
        program: Program,
        cell_count: NonZeroUsize,
        allow_growth: bool,
        input_reader: Box<dyn Read>,
        output_writer: Box<dyn Write>,
    ) -> Self {
        BrainfuckVM {
            // tape: vec![N::default(); cell_count.get()],
            tape: vec![N::default(); cell_count.get()],
            head: 0,
            allow_growth,
            program_counter: 0,
            program,
            input_reader,
            output_writer,
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
                let output_writer = std::io::stdout();
                self.write_value(&mut output_writer.lock())
                    .map_err(|e| VMError::IOError {
                        instruction: hr_instruction,
                        reason: e.to_string(),
                    })?;
            }
            RawInstruction::InputByte => {
                let input_reader = std::io::stdin();
                self.read_value(&mut input_reader.lock())
                    .map_err(|e| VMError::IOError {
                        instruction: hr_instruction,
                        reason: e.to_string(),
                    })?;
            }
            RawInstruction::ConditionalForward => {
                if self.current_cell()?.is_zero() {
                    // Jump forwards if zero
                    println!("Jumping to {}", self.get_bracket_position(hr_instruction)?);
                    return Ok(self.get_bracket_position(hr_instruction)?);
                };
            }
            RawInstruction::ConditionalBackward => {
                // Always jump back to opening bracket
                self.program_counter = self.get_bracket_position(hr_instruction)?;
                // Subtract 1, since it is only in ConditionalForward that we make an assesment
                return Ok(self.get_bracket_position(hr_instruction)? - 1);
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
                    reason: ("Failed to move head right".to_string()),
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
        // Create a buffer to read into with the size of one N::Value
        let mut buffer = vec![0u8; N::bytes_per_cell()];

        // Read as many bytes to fill the buffer
        reader
            .read_exact(&mut buffer)
            .map_err(|e| VMError::IOError {
                instruction: self.program.instructions()[self.program_counter],
                reason: e.to_string(),
            })?;

        let value: N = N::from_bytes(&buffer).map_err(|e| VMError::IOError {
            instruction: self.program.instructions()[self.program_counter],
            reason: format!("Failed to read cell value from bytes: {}", e),
        })?;

        self.current_cell()?.set(value);
        Ok(())
    }

    pub fn write_value<W: Write>(&mut self, writer: &mut W) -> Result<(), VMError> {
        match N::from(self.current_cell()?.get()) {
            Some(value) => writer
                .write_all(&value.to_bytes())
                .map_err(|e| VMError::IOError {
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
        // TODO: Change this to a relative path
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
        let max_cell_value = u8::MAX as usize;
        let program_string = "+".repeat(max_cell_value);
        let program = test_program_from_string(&program_string)?;
        let cell_count = NonZeroUsize::new(1).unwrap();

        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::<u8>::new(&program, cell_count, false);

        for (instruction_index, hr_instruction) in vm.program.instructions().iter().enumerate() {
            let cell_value = vm.tape.get(vm.head).unwrap();

            assert_eq!(cell_value.to_owned(), instruction_index as u8);
            vm.process_instruction(hr_instruction.to_owned())
                .map_err(|err| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", err),
                    )) as Box<dyn std::error::Error>
                })?;
        }

        Ok(())
    }

    #[test]
    fn test_decrement_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let program = test_program_from_string("-")?;
        let cell_count = NonZeroUsize::new(10).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

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
    fn test_increment_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let overflow_cell_value = u8::MAX as usize + 1;
        let program_string = "+".repeat(overflow_cell_value);
        let program = test_program_from_string(&program_string)?;
        let cell_count = NonZeroUsize::new(1).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::new(&program, cell_count, false);

        // Only care about final result
        for hr_instruction in vm.program.instructions().iter() {
            let _ = vm.process_instruction(hr_instruction.to_owned());
        }
        
        let cell_value = vm.tape.get(vm.head).unwrap();
        assert_eq!(cell_value.to_owned(), 0);
        Ok(())
    }

    #[test]
    fn test_read_value_success<'a>() -> Result<(), Box<dyn std::error::Error>> {
        let max_cell_value = u8::MAX as usize;
        let program = test_program_from_file()?;
        let cell_count = NonZeroUsize::new(1).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::<u8>::new(&program, cell_count, false);

        for index in 0..max_cell_value {
            let input_data = index as u8;
            let mut input_cursor = Cursor::new(vec![input_data]);
    
            vm.read_value(&mut input_cursor)
                .map_err(|err| format!("Error: {}", err))?;
            assert_eq!(
                vm.current_cell_value()
                    .map_err(|err| format!("Error: {}", err))?,
                input_data,
                "Cell value should match the read value"
            );
        }

        Ok(())
    }

    #[test]
    fn test_write_value_success() -> Result<(), Box<dyn std::error::Error>> {
        let max_cell_value = u8::MAX as usize;
        let program = test_program_from_file()?;
        let cell_count = NonZeroUsize::new(1).unwrap();
        let mut vm: BrainfuckVM<'_, u8> = BrainfuckVM::<u8>::new(&program, cell_count, false);
        for index in 0..max_cell_value {
            let output_data = index as u8;
            vm.current_cell()
                .map_err(|err| format!("Error: {}", err))?
                .set(output_data);

            let mut output_cursor = Cursor::new(Vec::new());
            vm.write_value(&mut output_cursor)
                .map_err(|err| format!("Error: {}", err))?;

            assert_eq!(
                output_cursor.into_inner(),
                vec![output_data],
                "Output should contain the cell's value"
            );
        }

        Ok(())
    }
}
