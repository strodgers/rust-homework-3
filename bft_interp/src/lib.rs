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
pub enum VMError<N> 
where
    N: CellKind
{
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
        final_state: VMState<N>,
    },
}

// So we can use it with Box dyn Error
impl <N>std::error::Error for VMError<N>
where
    N: CellKind {}

impl<'a, N> fmt::Display for VMError<N>
where
    N: CellKind
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
                final_state: final_state,
            } => {
                write!(
                    f,
                    "End of program, final state: {}",
                    final_state
                )
            }
        }
    }
}

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

    pub fn build<N>(self) -> Result<BrainfuckVM<N>, VMError<N>>
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
    instruction_index: usize,
    program: Program,
    input_reader: Box<dyn Read>,
    output_writer: Box<dyn Write>,
    current_instruction: HumanReadableInstruction,
    instructions_processed: usize,
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
            instruction_index: 0,
            program,
            input_reader,
            output_writer,
            current_instruction: HumanReadableInstruction::undefined(),
            instructions_processed: 0,
        }
    }

    fn current_state(&self) -> Result<VMState<N>, VMError<N>> {
        Ok(VMState {
            cell_value: self.current_cell_value()?,
            head: self.head,
            next_instruction_index: self.instruction_index,
            last_raw_instruction: self.current_instruction.raw_instruction().to_owned(),
            instructions_processed: self.instructions_processed,
        })
    }

    fn get_bracket_position(
        &'a self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<usize, VMError<N>> {
        return self
            .program
            .get_bracket_position(hr_instruction.index())
            .ok_or(VMError::ProgramError {
                instruction: hr_instruction,
                reason: ("Could not find matching bracket".to_string()),
            });
    }

    pub fn current_cell(&mut self) -> Result<&mut N, VMError<N>> {
        if let Some(cell) = self.tape.get_mut(self.head) {
            Ok(cell)
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.instruction_index],
                reason: "Cell not found".to_string(),
            })
        }
    }

    pub fn current_cell_value(&self) -> Result<N, VMError<N>> {
        if let Some(cell) = self.tape.get(self.head) {
            Ok(cell.to_owned())
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.instruction_index],
                reason: "Cell not found".to_string(),
            })
        }
    }

    fn process_instruction(
        &mut self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<usize, VMError<N>> {
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
                self.write_value()
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
                self.instruction_index = self.get_bracket_position(hr_instruction)?;
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
        Ok(self.instruction_index + 1)
    }

    pub fn interpret_step(&'a mut self) -> Result<VMState<N>, VMError<N>> {
        match self.program.instructions().get(self.instruction_index) {
            Some(instruction) => {
                self.current_instruction = *instruction;
            }
            None => {
                let mut final_state = self.current_state()?;
                final_state.last_raw_instruction = self.current_instruction.raw_instruction().to_owned();
                return Err(VMError::EndOfProgram {final_state});
            }
        }

        log::info!("{}", self.current_instruction);
        self.instruction_index = self.process_instruction(self.current_instruction)?;
        self.instructions_processed += 1;
        self.current_state()
    }

    pub fn interpret(&'a mut self) -> Result<VMState<N>, VMError<N>> {
        for state in self.iter() {
            match state {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
        self.current_state()
    }

    fn move_head_left(&mut self) -> Result<(), VMError<N>> {
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

    fn move_head_right(&mut self) -> Result<(), VMError<N>> {
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
    fn increment_cell(&mut self) -> Result<(), VMError<N>> {
        self.current_cell()?.increment();
        Ok(())
    }

    /// Decrements the value of the cell pointed to by the head.
    fn decrement_cell(&mut self) -> Result<(), VMError<N>> {
        self.current_cell()?.decrement();
        Ok(())
    }

    pub fn read_value(&mut self) -> Result<(), VMError<N>> {
        // Create a buffer to read into with the size of one N::Value
        let mut buffer = vec![0u8; N::bytes_per_cell()];

        // Read as many bytes to fill the buffer
        self.input_reader
            .read_exact(&mut buffer)
            .map_err(|e| VMError::IOError {
                instruction: self.program.instructions()[self.instruction_index],
                reason: e.to_string(),
            })?;

        let value: N = N::from_bytes(&buffer).map_err(|e| VMError::IOError {
            instruction: self.program.instructions()[self.instruction_index],
            reason: format!("Failed to read cell value from bytes: {}", e),
        })?;

        self.current_cell()?.set(value);
        Ok(())
    }

    pub fn write_value(&mut self) -> Result<(), VMError<N>> {
        match N::from(self.current_cell()?.get()) {
            Some(value) => self.output_writer
                .write_all(&value.to_bytes())
                .map_err(|e| VMError::IOError {
                    instruction: self.program.instructions()[self.instruction_index],
                    reason: e.to_string(),
                })?,
            None => {
                return Err(VMError::IOError {
                    instruction: self.program.instructions()[self.instruction_index],
                    reason: ("Failed to read value".to_string()),
                })
            }
        };

        Ok(())
    }

    pub fn iter<'b>(&'b mut self) -> VMIterator<'b, N> {
        VMIterator { vm: self, final_state: None }
    }
}

// Holds the state of the VM, with relevant information
#[derive(Debug, PartialEq)]
pub struct VMState<N>
where
    N: CellKind,
{
    cell_value: N,
    head: usize,
    next_instruction_index: usize,
    last_raw_instruction: RawInstruction,
    instructions_processed: usize,
}

impl<'a, N> fmt::Display for VMState<N>
where
    N: CellKind
    {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cell value: {}, Head: {}, Instruction index: {}, Instruction: {}, Instructions processed: {}",
            self.cell_value,
            self.head,
            self.next_instruction_index,
            self.last_raw_instruction,
            self.instructions_processed
        )
    }
}

// Iterator object for the VM, so we can use it in loops
pub struct VMIterator<'a, N>
where
    N: CellKind,
{
    vm: &'a mut BrainfuckVM<N>,
    final_state: Option<VMState<N>>,
}
impl<'a, N> VMIterator<'a, N>
where
    N: CellKind,
{
    pub fn final_state(&self) -> Option<&VMState<N>> {
        self.final_state.as_ref()
    }
}
// Iterate one step at a time and return the Error type.
// End the iteration (returning None) if the Error type is EndOfProgram
impl<'a, N> Iterator for VMIterator<'a, N>
where
    N: CellKind,
{
    type Item = Result<VMState<N>, VMError<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.vm.interpret_step() {
            Ok(result) => Some(Ok(result)),
            Err(VMError::EndOfProgram{ final_state }) =>{
                self.final_state = Some(final_state);
                None},
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

    // Helper function to setup a test u8 VM with a program from a string and cell growth bool, default cell count
    fn setup_vm_from_string(
        program_string: &str,
        allow_growth: bool,
        cell_count: Option<NonZeroUsize>,
    ) -> Result<BrainfuckVM<u8>, VMError<u8>> {
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
    fn ensure_vm_final_state<N>(mut vm: BrainfuckVM<N>, expected_state: VMState<N>) -> bool
    where
        N: CellKind,
    {
        // Move one more step, should reach end of program and expected_counter_final should equal
        // the program_counter_final expected_counter_final
        match vm.interpret_step() {
            Err(VMError::EndOfProgram {
                final_state,
            }) => final_state == expected_state,
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
            assert_eq!(vm.instruction_index, 0);

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
                    assert_eq!(state.last_raw_instruction, RawInstruction::IncrementPointer);
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
                    assert_eq!(state.last_raw_instruction, RawInstruction::DecrementPointer);
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

        let expected_state = VMState::<u8>{
            cell_value: 0,
            head: 0,
            next_instruction_index: 2 * half_way,
            last_raw_instruction: RawInstruction::DecrementPointer,
            instructions_processed: 2 * half_way,
        };
        assert!(ensure_vm_final_state(vm, expected_state));
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
                    assert_eq!(state.last_raw_instruction, RawInstruction::IncrementByte);
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
                    assert_eq!(state.last_raw_instruction, RawInstruction::DecrementByte);
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

        let expected_state = VMState::<u8>{
            cell_value: 0,
            head: 0,
            next_instruction_index: 2 * max_cell_value,
            last_raw_instruction: RawInstruction::DecrementByte,
            instructions_processed: 2 * max_cell_value,
        };
        assert!(ensure_vm_final_state(vm, expected_state));
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

        let expected_state = VMState::<u8>{
            cell_value: 0,
            head: 0,
            next_instruction_index: number_of_instructions,
            last_raw_instruction: RawInstruction::IncrementByte,
            instructions_processed: number_of_instructions,
        };
        assert!(ensure_vm_final_state(vm, expected_state));
        Ok(())
    }

    #[test]
    fn test_end_of_program() -> Result<(), Box<dyn std::error::Error>> {
        // Don't use any conditional forwards here, easier to define the end
        let program_string = "++-->+<--";
        let number_of_instructions = program_string.len();
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(2))?;

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

        let expected_state = VMState::<u8>{
            cell_value: 254,
            head: 0,
            next_instruction_index: number_of_instructions,
            last_raw_instruction: RawInstruction::DecrementByte,
            instructions_processed: number_of_instructions,
        };
        assert!(ensure_vm_final_state(vm, expected_state));

        Ok(())
    }

    #[test]
    fn test_input_success<'a>() -> Result<(), Box<dyn std::error::Error>> {
        setup_logging();

        let number_of_instructions = 10000;
        // Preallocate enough space
        let mut program_string = String::with_capacity(number_of_instructions);
        program_string.extend(",".repeat(number_of_instructions).chars());
        let cell_count = NonZeroUsize::new(1);
        let program = Program::new(Cursor::new(program_string))?;

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

        let expected_state = VMState::<u8>{
            cell_value: buffer[number_of_instructions - 1],
            head: 0,
            next_instruction_index: number_of_instructions,
            last_raw_instruction: RawInstruction::InputByte,
            instructions_processed: number_of_instructions,
        };
        assert!(ensure_vm_final_state(vm, expected_state));

        Ok(())
    }

    #[test]
    fn test_output_success() -> Result<(), Box<dyn std::error::Error>> {
        setup_logging();

        // Create a VM, then assign a vector of random values to it's tape.
        // Go through and check each value is read correctly.
        let number_of_reads = 10000;
        // Use one less move so the vm head doesn't go over the cell count
        let number_of_moves = number_of_reads - 1;
        let number_of_instructions = number_of_reads + number_of_moves;
    
        // Preallocate enough space
        let mut program_string = String::with_capacity(number_of_instructions);
        program_string.extend(".>".repeat(number_of_reads).chars());
        // Remove the final > so we don't go over the cell count
        program_string.pop();
        let program = Program::new(Cursor::new(program_string))?;
        let cell_count = NonZeroUsize::new(number_of_moves);

        // Generate some random u8 values
        let mut rng = rand::thread_rng();
        // Only half of the instructions are output, so we only need half the buffer size (rounded down)
        let mut buffer = vec![0u8; number_of_reads];
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
                    assert_eq!(state.last_raw_instruction, RawInstruction::OutputByte);
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
        
        let expected_state = VMState::<u8>{
            cell_value: buffer.pop().unwrap(),
            head: number_of_moves,
            next_instruction_index: number_of_instructions,
            last_raw_instruction: RawInstruction::OutputByte,
            instructions_processed: number_of_instructions,
        };
        assert!(ensure_vm_final_state(vm, expected_state));
        Ok(())
    }

    #[test]
    fn test_conditional_forward_backward_success() -> Result<(), Box<dyn std::error::Error>> {
        setup_logging();

        let program_string = "
        ++       // Increment the first cell to 2
        [        // Begin a loop that runs while the current cell's value is not zero
         -       // Decrement the current cell
        ]        // End the loop only after the decrement has happened twice
        ";
        let mut vm = setup_vm_from_string(&program_string, true, None)?;

        // First two iterations should increment the first cell, instruction_index goes up normally
        let mut state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::IncrementByte);
        assert!(state.cell_value == 1);
        assert!(state.next_instruction_index == 1);

        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::IncrementByte);
        assert!(state.cell_value == 2);
        assert!(state.next_instruction_index == 2);

        // The loop should be entered, instruction_index goes up normally
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::ConditionalForward);
        assert!(state.cell_value == 2);
        assert!(state.next_instruction_index == 3);

        // Next instruction should decrement the cell, instruction_index goes up normally
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::DecrementByte);
        assert!(state.cell_value == 1);
        assert!(state.next_instruction_index == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::ConditionalBackward);
        assert!(state.cell_value == 1);
        assert!(state.next_instruction_index == 2);

        // Cell value is still non-zero, so enter the loop again
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::ConditionalForward);
        assert!(state.cell_value == 1);
        assert!(state.next_instruction_index == 3);

        // Decrement the cell again
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::DecrementByte);
        assert!(state.cell_value == 0);
        assert!(state.next_instruction_index == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::ConditionalBackward);
        assert!(state.cell_value == 0);
        assert!(state.next_instruction_index == 2);

        // Now the cell value is 0, so the loop should be exited
        state = vm.interpret_step()?;
        assert!(state.last_raw_instruction == RawInstruction::ConditionalForward);
        assert!(state.cell_value == 0);
        assert!(state.next_instruction_index == 5);

        let expected_state = VMState::<u8>{
            cell_value: 0,
            head: 0,
            next_instruction_index: 5,
            last_raw_instruction: RawInstruction::ConditionalForward,
            instructions_processed: 9,
        };
        assert!(ensure_vm_final_state(vm, expected_state));

        Ok(())
    }
    

}