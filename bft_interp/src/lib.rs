use bft_types::Program;
use bft_types::{CellKind, HumanReadableInstruction, RawInstruction};
use std::any::TypeId;
use std::error::Error;
use std::fmt::{self, Debug};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read, Write};
use std::num::NonZeroUsize;

#[derive(Debug)]
pub enum VMError {
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
    GeneralError {
        reason: String,
    },
    TypeError {
        reason: String,
    },
    BuilderError {
        reason: String,
    },
    EndOfProgram {
        // TODO: maybe this should be a VMState?
        program_counter_final: usize,
    },
}

// So we can use it with Box dyn Error
impl std::error::Error for VMError {}

impl<'a> fmt::Display for VMError {
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
            VMError::GeneralError { reason } => {
                write!(f, "General error: {}", reason)
            }
            VMError::TypeError { reason } => {
                write!(f, "Type error: {}", reason)
            }
            VMError::BuilderError { reason } => {
                write!(f, "Builder error: {}", reason)
            }
            VMError::EndOfProgram {
                program_counter_final,
            } => {
                write!(
                    f,
                    "End of program, executed {} instructions",
                    program_counter_final
                )
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
    program: Option<Program>,
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
            program: None,
        }
    }

    pub fn set_input(mut self, input: R) -> Self {
        self.input_reader = Some(Box::new(input));
        self
    }

    pub fn set_output(mut self, output: W) -> Self {
        self.output_writer = Some(Box::new(output));
        self
    }

    // TODO: make this more generic? Read + 'a
    pub fn set_program_file(mut self, file: BufReader<File>) -> Self {
        self.program_reader = Some(Box::new(BufReader::new(file)) as Box<dyn Read + 'a>);
        self
    }

    pub fn set_program(mut self, program: Program) -> Self {
        self.program = Some(program);
        self
    }

    pub fn set_cell_kind(mut self, cell_kind: TypeId) -> Self {
        self.cell_kind = Some(cell_kind);
        self
    }

    pub fn set_cell_count(mut self, cell_count: Option<NonZeroUsize>) -> Self {
        match cell_count {
            Some(count) => self.cell_count = Some(count),
            None => {
                log::info!("Using default cell_count of 30,000");
                self.cell_count = NonZeroUsize::new(30000);
            }
        }
        self
    }

    pub fn set_allow_growth(mut self, allow_growth: bool) -> Self {
        self.allow_growth = Some(allow_growth);
        self
    }

    pub fn build<N>(self) -> Result<BrainfuckVM<N>, VMError>
    where
        N: CellKind,
        R: Read,
        W: Write,
    {
        // Program must be set somehow
        if self.program_reader.is_none() && self.program.is_none() {
            return Err(VMError::BuilderError {
                reason: "Program must be set by using set_program or set_program_file".to_string(),
            });
        }

        let program = match self.program {
            // If the program has been set, use that
            Some(program) => program,
            // If not, try and use the program_reader to create a program
            None => {
                let program_reader: Box<dyn Read> = match self.program_reader {
                    Some(reader) => reader,
                    None => {
                        return Err(VMError::BuilderError {
                            reason: "Program reader must be set.".to_string(),
                        })
                    }
                };
                Program::new(program_reader).map_err(|err| VMError::BuilderError {
                    reason: format!("Failed to create program: {}", err),
                })?
            }
        };

        // Default IO to use stdin and stdout
        let input_reader: Box<dyn Read + 'static> = match self.input_reader {
            Some(reader) => reader,
            None => {
                log::info!("Using default stdin");
                Box::new(io::stdin().lock())
            }
        };

        let output_writer: Box<dyn Write + 'static> = match self.output_writer {
            Some(reader) => reader,
            None => {
                log::info!("Using default stdout");
                Box::new(io::stdout().lock())
            }
        };

        // If no cell type provided, default to u8
        let cell_kind: TypeId = self.cell_kind.unwrap_or({
            log::info!("Using default cell kind u8");
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
            log::info!("Using default cell count 30000");
            NonZeroUsize::new(30000).unwrap()
        });

        // If no allow growth provided, default to false
        let allow_growth = self.allow_growth.unwrap_or({
            log::info!("Using default allow growth false");
            false
        });

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
    current_instruction: HumanReadableInstruction,
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
            current_instruction: HumanReadableInstruction::undefined(),
        }
    }

    fn current_state(&self) -> Result<VMState<N>, VMError> {
        Ok(VMState {
            cell_value: self.current_cell_value()?,
            head: self.head,
            program_counter: self.program_counter,
            current_instruction: self.current_instruction,
        })
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
                self.move_head_right()?;
            }
            RawInstruction::DecrementPointer => {
                self.move_head_left()?;
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
                self.read_value().map_err(|e| VMError::IOError {
                    instruction: hr_instruction,
                    reason: e.to_string(),
                })?;
            }
            RawInstruction::ConditionalForward => {
                if self.current_cell()?.is_zero() {
                    // Jump forwards if zero
                    log::debug!("Jumping to {}", self.get_bracket_position(hr_instruction)?);
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

    pub fn interpret_step(&'a mut self) -> Result<VMState<N>, VMError> {
        match self.program.instructions().get(self.program_counter) {
            Some(instruction) => {
                self.current_instruction = *instruction;
            }
            None => {
                return Err(VMError::EndOfProgram {
                    program_counter_final: self.program_counter,
                });
            }
        }

        log::info!("{}", self.current_instruction);
        self.program_counter = self.process_instruction(self.current_instruction)?;
        self.current_state()
    }

    pub fn interpret(&'a mut self) -> Result<VMState<N>, VMError> {
        loop {
            match self.interpret_step() {
                Ok(_) => (),
                Err(VMError::EndOfProgram {
                    program_counter_final,
                }) => break, // All is well, we've reached the end of the program
                Err(e) => return Err(e),
            }
        }
        self.current_state()
    }

    fn move_head_left(&mut self) -> Result<(), VMError> {
        if self.head == 0 {
            Err(VMError::InvalidHeadPosition {
                position: self.head,
                instruction: self.current_instruction,
                reason: ("Failed to move head left".to_string()),
            })
        } else {
            self.head -= 1;
            Ok(())
        }
    }

    fn move_head_right(&mut self) -> Result<(), VMError> {
        if self.head + 1 == self.tape.len() {
            // Extend the tape if allowed
            if self.allow_growth {
                self.tape.push(N::default());
            } else {
                // If the tape cannot grow, then it's an error
                return Err(VMError::InvalidHeadPosition {
                    position: self.head,
                    instruction: self.current_instruction,
                    reason: ("Failed to move head right".to_string()),
                });
            }
        }
        self.head += 1;

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

    pub fn read_value(&mut self) -> Result<(), VMError> {
        // Create a buffer to read into with the size of one N::Value
        let mut buffer = vec![0u8; N::bytes_per_cell()];

        // Read as many bytes to fill the buffer
        self.input_reader
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

    pub fn iter<'b>(&'b mut self) -> VMIterator<'b, N> {
        VMIterator { vm: self }
    }
}

// Holds the state of the VM, with relevant information
pub struct VMState<N>
where
    N: CellKind,
{
    cell_value: N,
    head: usize,
    program_counter: usize,
    current_instruction: HumanReadableInstruction,
}

// Iterator object for the VM, so we can use it in loops
pub struct VMIterator<'a, N>
where
    N: CellKind,
{
    vm: &'a mut BrainfuckVM<N>,
}

// Iterate one step at a time and return the Error type.
// End the iteration (returning None) if the Error type is EndOfProgram
impl<'a, N> Iterator for VMIterator<'a, N>
where
    N: CellKind,
{
    type Item = Result<VMState<N>, VMError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.vm.interpret_step() {
            Ok(result) => Some(Ok(result)),
            Err(VMError::EndOfProgram {
                program_counter_final,
            }) => None,
            Err(result) => Some(Err(result)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_test_utils::{TestFile, TEST_FILE_NUM_INSTRUCTIONS};
    use bft_types::Program;
    use env_logger;
    use log::LevelFilter;
    use rand::Rng;
    use std::fs::File;
    use std::io::BufReader;
    use std::io::Cursor;

    // Setup logging for any tests that it might be useful for
    fn setup_logging() {
        // Just use Debug level for tests
        let test_log_level = LevelFilter::Debug;
        let _ = env_logger::builder()
            .is_test(true)
            .filter(None, test_log_level)
            .try_init();
    }

    // Helper function to setup a test VM with a program from a string and cell growth bool, default cell count
    fn setup_vm_from_string(
        program_string: &str,
        allow_growth: bool,
        cell_count: Option<NonZeroUsize>,
    ) -> Result<BrainfuckVM<u8>, VMError> {
        let program =
            Program::new(Cursor::new(program_string)).map_err(|e| VMError::ProgramError {
                instruction: HumanReadableInstruction::undefined(),
                reason: e.to_string(),
            })?;
        let vm = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
            .set_program(program)
            .set_cell_count(cell_count)
            .set_allow_growth(allow_growth) // default or test-specific value
            .build()?;
        Ok(vm)
    }

    // Helper function to setup a test VM with a program from a TestFile and cell growth bool, default cell count
    fn setup_vm_from_testfile(
        allow_growth: bool,
        cell_count: Option<NonZeroUsize>,
    ) -> Result<BrainfuckVM<u8>, Box<dyn std::error::Error>> {
        let testfile = TestFile::new()?;
        let program = Program::new(testfile)?;
        let vm = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
            .set_program(program)
            .set_allow_growth(allow_growth)
            .set_cell_count(cell_count)
            .build()?;
        Ok(vm)
    }

    // Helper function to ensure a VM has run and completed the required amount of steps
    fn ensure_vm_ran<N>(mut vm: BrainfuckVM<N>, steps: usize) -> bool
    where
        N: CellKind,
    {
        // Move one more step, should reach end of program and steps should equal
        // the program_counter_final steps
        match vm.interpret_step() {
            Err(VMError::EndOfProgram {
                program_counter_final,
            }) => program_counter_final == steps,
            Ok(_) => false,
            Err(e) => false,
        }
    }

    #[test]
    fn test_vm_initialization() -> Result<(), Box<dyn std::error::Error>> {
        for allow_growth in [true, false].iter() {
            let vm: BrainfuckVM<u8> =
                setup_vm_from_testfile(*allow_growth, NonZeroUsize::new(30000))?;
            assert_eq!(vm.tape.len(), 30000);
            assert!(vm.tape.iter().all(|&x| x == 0));
            assert_eq!(vm.head, 0);
            assert_eq!(vm.allow_growth, *allow_growth);
            assert_eq!(vm.program_counter, 0);

            // TODO: these
            // program: Program,
            // input_reader: Box<dyn Read>,
            // output_writer: Box<dyn Write>,
            // current_instruction: HumanReadableInstruction,
        }

        Ok(())
    }

    #[test]
    fn test_move_head_success() -> Result<(), Box<dyn std::error::Error>> {
        let half_way = 100000;
        // Preallocate enough space
        let mut program_string = String::with_capacity(2 * half_way);
        program_string.extend(">".repeat(half_way).chars());
        program_string.extend("<".repeat(half_way).chars());
        let mut vm = setup_vm_from_string(&program_string, true, None)?;

        // Go forward half_way times
        for (instruction_index, iteration) in vm.iter().take(half_way).enumerate() {
            match iteration {
                Ok(state) => {
                    // After each step, head should have gone up by one
                    assert_eq!(state.head, instruction_index + 1);
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        // Go back again half_way times
        for (instruction_index, iteration) in vm.iter().enumerate() {
            match iteration {
                Ok(state) => {
                    // After each step, head should have gone down by one
                    assert_eq!(state.head, (half_way - instruction_index - 1));
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, 2 * half_way), true);
        Ok(())
    }

    #[test]
    fn test_move_head_left_error() -> Result<(), Box<dyn std::error::Error>> {
        let program_string = "<";
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(1))?;

        // Go back one, should be an error
        if let Err(VMError::InvalidHeadPosition {
            position, reason, ..
        }) = vm.interpret_step()
        {
            if position == 0 && reason == "Failed to move head left" {
                // all is well
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected error encountered",
                )) as Box<dyn std::error::Error>);
            }
        }
        Ok(())
    }
    #[test]
    fn test_move_head_right_error() -> Result<(), Box<dyn std::error::Error>> {
        let program_string = ">";
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(1))?;
        // Go forward one, should be an error
        if let Err(VMError::InvalidHeadPosition {
            position, reason, ..
        }) = vm.interpret_step()
        {
            if position == 0 && reason == "Failed to move head right" {
                // all is well
            } else {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected error encountered",
                )) as Box<dyn std::error::Error>);
            }
        }

        Ok(())
    }

    #[test]
    fn test_increment_decrement_success() -> Result<(), Box<dyn std::error::Error>> {
        let max_cell_value = u8::MAX as usize;
        // Preallocate enough space
        let mut program_string = String::with_capacity(2 * max_cell_value);
        program_string.extend("+".repeat(max_cell_value).chars());
        program_string.extend("-".repeat(max_cell_value).chars());
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(1))?;

        // Increment cell value 255 times
        for (expected_cell_value, iteration) in vm.iter().take(max_cell_value).enumerate() {
            match iteration {
                Ok(state) => {
                    // After each step, cell value should have gone up by one
                    assert_eq!(state.cell_value, expected_cell_value as u8 + 1);
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }
        // Decrement cell value 255 times
        for (expected_cell_value, iteration) in vm.iter().enumerate() {
            match iteration {
                Ok(state) => {
                    // After each step, cell value should have gone up by one
                    assert_eq!(
                        state.cell_value,
                        (max_cell_value - expected_cell_value - 1) as u8
                    );
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, 2 * max_cell_value), true);
        Ok(())
    }

    #[test]
    fn test_decrement_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let program_string = "-";
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(1))?;

        let expected_cell_value: u8 = 255;
        match vm.interpret_step() {
            Ok(_) => assert_eq!(
                vm.tape.get(vm.head).unwrap().to_owned(),
                expected_cell_value
            ),
            Err(e) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error: {}", e),
                )) as Box<dyn std::error::Error>)
            }
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, 1), true);
        Ok(())
    }

    #[test]
    fn test_increment_cell_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let number_of_instructions = u8::MAX as usize + 1;
        let program_string = "+".repeat(number_of_instructions);
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(1))?;

        // Make sure that the cell value wraps around
        match vm.iter().skip(u8::MAX as usize).next() {
            Some(Ok(state)) => {
                assert_eq!(
                    state.cell_value, 0,
                    "Cell value should have wrapped to 0"
                );
            },
            Some(Err(e)) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error: {}", e),
                )) as Box<dyn std::error::Error>)
            },
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Error: Should have reached EndOfProgram".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, number_of_instructions), true);
        Ok(())
    }

    #[test]
    fn test_end_of_program() -> Result<(), Box<dyn std::error::Error>> {
        // Don't use any conditional forwards here, easier to define the end
        let program_string = "++-->+<--";
        let number_of_instructions = program_string.len();
        let mut vm = setup_vm_from_string(&program_string, true, NonZeroUsize::new(1))?;

        for iteration in vm.iter() {
            match iteration {
                Ok(_) => (),
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    )) as Box<dyn std::error::Error>)
                }
            };
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, number_of_instructions), true);

        Ok(())
    }

    #[test]
    fn test_input_success<'a>() -> Result<(), Box<dyn std::error::Error>> {
        setup_logging();

        let number_of_instructions = 10000;
        // Preallocate enough space
        let mut program_string = String::with_capacity(number_of_instructions);
        program_string.extend(",".repeat(number_of_instructions).chars());
        let program =
            Program::new(Cursor::new(program_string)).map_err(|e| VMError::ProgramError {
                instruction: HumanReadableInstruction::undefined(),
                reason: e.to_string(),
            })?;
        let cell_count = NonZeroUsize::new(1);

        // Generate some random u8 values
        let mut rng = rand::thread_rng();
        let mut buffer = vec![0u8; number_of_instructions]; // Initialize a vector of zeros with length `n`
        rng.fill(&mut buffer[..]);

        let reader = Cursor::new(buffer.clone());
        let mut vm: BrainfuckVM<u8> = VMBuilder::<Cursor<Vec<u8>>, std::io::Cursor<Vec<u8>>>::new()
            .set_program(program)
            .set_cell_count(cell_count)
            .set_allow_growth(false)
            .set_input(reader)
            .build()
            .map_err(|e| format!("{}", e))?;

        for (input_index, iteration) in vm.iter().enumerate() {
            match iteration {
                Ok(state) => {
                    let rng_value = buffer[input_index];
                    // Print these out just to be sure
                    log::debug!("Cell value: {}, RNG value: {}", state.cell_value, rng_value);
                    assert_eq!(
                        state.cell_value, rng_value,
                        "Cell value should match the read value"
                    );
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        // Ensure everything ends nicely
        assert_eq!(ensure_vm_ran(vm, number_of_instructions), true);

        Ok(())
    }

    #[test]
    fn test_output_success() -> Result<(), Box<dyn std::error::Error>> {
        setup_logging();

        let number_of_instructions = 20000;
        // Preallocate enough space
        let mut program_string = String::with_capacity(number_of_instructions);
        // Read a value then move right one, so once we set the tape values we can read
        // from them all
        program_string.extend(".>".repeat(number_of_instructions / 2).chars());
        // Remove the final >
        program_string.pop();
        let program =
            Program::new(Cursor::new(program_string)).map_err(|e| VMError::ProgramError {
                instruction: HumanReadableInstruction::undefined(),
                reason: e.to_string(),
            })?;
        let cell_count = NonZeroUsize::new(number_of_instructions);

        // Generate some random u8 values
        let mut rng = rand::thread_rng();
        let mut buffer = vec![0u8; number_of_instructions];
        rng.fill(&mut buffer[..]);

        let output_buffer: Vec<u8> = Vec::new(); // Initialize an empty Vec<u8>
        let writer = Cursor::new(output_buffer);
        let mut vm: BrainfuckVM<u8> = VMBuilder::<Cursor<Vec<u8>>, std::io::Cursor<Vec<u8>>>::new()
            .set_program(program)
            .set_cell_count(cell_count)
            .set_allow_growth(false)
            .set_output(writer)
            .build()
            .map_err(|e| format!("{}", e))?;

        // Apply rng values to VM tape
        vm.tape = buffer.clone();

        // Iterate over all instructions, but only check on every 2nd instruction (starting from 0),
        // since the instructions repeat READ => MOVE RIGHT
        for (output_index, iteration) in vm.iter().step_by(2).enumerate() {
            match iteration {
                Ok(state) => {
                    let rng_value = buffer[output_index];
                    // Print these out just to be sure
                    log::debug!("Cell value: {}, RNG value: {}", state.cell_value, rng_value);
                    assert_eq!(
                        state.cell_value, rng_value,
                        "Cell value should match the write value"
                    );
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        // Ensure everything ends nicely
        // Subtract 1 because we popped a > from the program_string
        assert_eq!(ensure_vm_ran(vm, number_of_instructions - 1), true);

        Ok(())
    }
}
