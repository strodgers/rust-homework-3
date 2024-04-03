use bft_types::Program;
use bft_types::{CellKind, HumanReadableInstruction, RawInstruction};
use std::any::TypeId;
use std::fmt::{self, Debug};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read, Write};
use std::num::NonZeroUsize;

// Error types that don't require lifetime handling
#[derive(Debug)]
pub enum VMErrorSimple<N>
where
    N: CellKind,
{
    // Represents a generic error with a static reason
    GeneralError { reason: &'static str },
    // Indicates an error related to type mismatches or issues
    TypeError { reason: &'static str },
    // Errors occurring during the construction of the VM, typically due to misconfiguration
    BuilderError { reason: String },
    // Signifies the normal completion of program execution, including the final state of the VM for inspection
    EndOfProgram { final_state: VMState<N> },
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
        reason: &'static str,
    },
    CellOperationError {
        position: usize,
        instruction: HumanReadableInstruction,
        reason: &'static str,
    },
    IOError {
        instruction: HumanReadableInstruction,
        reason: String,
    },
    ProgramError {
        instruction: HumanReadableInstruction,
        reason: &'static str,
    },
}

// So we can use it with Box dyn Error
impl<'a, N> std::error::Error for VMError<N> where N: CellKind {}

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
                write!(f, "End of program, final state: {}", final_state)
            }
        }
    }
}

// Provides a fluent API to configure and build an instance of a Brainfuck VM. This includes setting up the cell kind, cell count, IO streams, and more
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
    report_state: Option<bool>,
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
            report_state: None,
        }
    }

    // Configures the VM to use a custom input stream
    pub fn set_input(mut self, input: R) -> Self {
        self.input_reader = Some(Box::new(input));
        self
    }

    // Sets a custom output stream for the VM
    pub fn set_output(mut self, output: W) -> Self {
        self.output_writer = Some(Box::new(output));
        self
    }

    // TODO: make this more generic? Read + 'a
    // Loads a Brainfuck program from a file
    pub fn set_program_file(mut self, file: BufReader<File>) -> Self {
        self.program_reader = Some(Box::new(BufReader::new(file)) as Box<dyn Read + 'a>);
        self
    }

    // Directly sets the program to be executed by the VM
    pub fn set_program(mut self, program: Program) -> Self {
        self.program = Some(program);
        self
    }

    // Specifies the type of cells used by the VM (e.g., u8, i32)
    pub fn set_cell_kind(mut self, cell_kind: TypeId) -> Self {
        self.cell_kind = Some(cell_kind);
        self
    }

    // Determines the number of cells (memory size) the VM should initialize with
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

    // Allows or disallows the VM's tape (memory) to grow beyond the initial cell count
    pub fn set_allow_growth(mut self, allow_growth: bool) -> Self {
        self.allow_growth = Some(allow_growth);
        self
    }

    // Enables or disables detailed state reporting after each instruction is processed
    pub fn set_report_state(mut self, report_state: bool) -> Self {
        self.report_state = Some(report_state);
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
            return Err(VMError::Simple(VMErrorSimple::BuilderError {
                reason: "Program must be set by using set_program or set_program_file".to_owned(),
            }));
        }

        let program = match self.program {
            // If the program has been set, use that
            Some(program) => program,
            // If not, try and use the program_reader to create a program
            None => {
                let program_reader: Box<dyn Read> = match self.program_reader {
                    Some(reader) => reader,
                    None => {
                        return Err(VMError::Simple(VMErrorSimple::BuilderError {
                            reason: "Program reader must be set.".to_owned(),
                        }))
                    }
                };
                Program::new(program_reader).map_err(|err| {
                    VMError::Simple(VMErrorSimple::BuilderError {
                        reason: format!("Failed to create program: {}", err),
                    })
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
        let cell_count = self.cell_count.unwrap_or_else(|| {
            log::info!("Using default cell count 30000");
            NonZeroUsize::new(30000).unwrap()
        });

        // If no allow growth provided, default to false
        let allow_growth = self.allow_growth.unwrap_or_else(|| {
            log::info!("Using default allow growth false");
            false
        });

        // If set, interpret will report the state after each iteration.
        // This is useful for debugging and testing but makes the program slower.
        let report_state = self.report_state.unwrap_or_else(|| {
            log::info!("Using default no state reporting");
            false
        });

        Ok(BrainfuckVM::new(
            program,
            cell_count,
            allow_growth,
            input_reader,
            output_writer,
            report_state,
        ))
    }
}

// Represents the VM capable of interpreting Brainfuck programs. It manages the execution environment
// including the tape (memory), the instruction pointer, input/output streams, and execution state.
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
    report_state: bool,
}

impl<'a, N> BrainfuckVM<N>
where
    N: CellKind,
{
    // Constructs a new VM instance with specified settings
    pub fn new(
        program: Program,
        cell_count: NonZeroUsize,
        allow_growth: bool,
        input_reader: Box<dyn Read>,
        output_writer: Box<dyn Write>,
        report_state: bool,
    ) -> Self {
        BrainfuckVM {
            tape: vec![N::default(); cell_count.get()],
            head: 0,
            allow_growth,
            instruction_index: 0,
            program,
            input_reader,
            output_writer,
            current_instruction: HumanReadableInstruction::undefined(),
            instructions_processed: 0,
            report_state,
        }
    }
    // Optionally retrieves the current state of the VM, including cell value and head position,
    // if state reporting is enabled
    fn current_state(&mut self) -> Option<VMState<N>> {
        if self.report_state {
            let cell_value = match self.current_cell() {
                Ok(value) => Some(value.clone()),
                Err(_) => None,
            }?;

            return Some(VMState {
                cell_value,
                head: self.head,
                next_instruction_index: self.instruction_index,
                last_raw_instruction: self.current_instruction.raw_instruction().to_owned(),
                instructions_processed: self.instructions_processed,
            });
        }
        None
    }

    fn get_bracket_position<'b>(
        &'a self,
        hr_instruction: &'b HumanReadableInstruction,
    ) -> Result<usize, VMError<N>> {
        match self.program.get_bracket_position(hr_instruction.index()) {
            Some(position) => Ok(position),
            None => Err(VMError::ProgramError {
                instruction: *hr_instruction,
                reason: "Could not find matching bracket",
            }),
        }
    }

    pub fn current_cell(&mut self) -> Result<&mut N, VMError<N>> {
        if let Some(cell) = self.tape.get_mut(self.head) {
            Ok(cell)
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.instruction_index],
                reason: "Cell not found",
            })
        }
    }

    pub fn current_cell_value(&mut self) -> Result<&N, VMError<N>> {
        if let Some(cell) = self.tape.get(self.head) {
            Ok(cell)
        } else {
            Err(VMError::CellOperationError {
                position: self.head,
                instruction: self.program.instructions()[self.instruction_index],
                reason: "Cell not found",
            })
        }
    }

    fn process_instruction<'b>(
        &mut self,
        hr_instruction: HumanReadableInstruction,
    ) -> Result<(), VMError<N>> {
        let mut next_index = self.instruction_index + 1;
        log::debug!("Processing instruction: {}", hr_instruction);
        match hr_instruction.raw_instruction() {
            RawInstruction::IncrementPointer => {
                self.move_head_right()
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to move head right",
                    })?;
            }
            RawInstruction::DecrementPointer => {
                self.move_head_left()
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to move head left",
                    })?;
            }
            RawInstruction::IncrementByte => {
                self.increment_cell()
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to increment cell",
                    })?;
            }
            RawInstruction::DecrementByte => {
                self.decrement_cell()
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to decrement cell",
                    })?;
            }
            RawInstruction::OutputByte => {
                self.write_value().map_err(|e| VMError::IOError {
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
                if self.current_cell_value()?.is_zero() {
                    let bracket_position = self.get_bracket_position(&hr_instruction)?;
                    log::debug!("Jumping to {}", bracket_position);
                    next_index = bracket_position;
                };
            }
            RawInstruction::ConditionalBackward => {
                // Always jump back to opening bracket
                // Subtract 1, since it is only in ConditionalForward that we make an assesment
                next_index = self.get_bracket_position(&hr_instruction)? - 1;
            }
            RawInstruction::Undefined => {
                return Err(VMError::ProgramError {
                    instruction: hr_instruction,
                    reason: "Undefined instruction",
                });
            }
        }
        // Track number of instructions processed
        self.instructions_processed += 1;

        // Move to the next instruction
        self.instruction_index = next_index;

        // Track the current instruction
        self.current_instruction = hr_instruction;
        Ok(())
    }

    // Executes a single step (instruction) of the program
    pub fn interpret_step(&mut self) -> Result<VMState<N>, VMError<N>> {
        let state = self.construct_state();
        // Check if the current instruction index is beyond the program's length.
        if self.instruction_index >= self.program.instructions().len() {
            // Handle the end of the program
            return Err(VMError::Simple(VMErrorSimple::EndOfProgram {
                final_state: state,
            }));
        }

        // Get the instruction at the current index.
        let instruction = self
            .program
            .instructions()
            .get(self.instruction_index)
            .expect("Failed to get instruction; index out of bounds.")
            .clone();
        // Handle anything that might need mutation
        self.process_instruction(instruction)?;

        // Construct the current state after processing the instruction.
        let current_state = self.construct_state();

        Ok(current_state)
    }

    fn construct_state(&self) -> VMState<N> {
        if self.report_state {
            return VMState {
                cell_value: self.tape[self.head],
                head: self.head,
                next_instruction_index: self.instruction_index,
                last_raw_instruction: self.current_instruction.raw_instruction().to_owned(),
                instructions_processed: self.instructions_processed,
            };
        }
        VMState::<N>::default()
    }

    // Runs the entire Brainfuck program to completion or until an error occurs
    pub fn interpret(&'a mut self) -> Result<VMStateFinal<N>, VMError<N>> {
        for state in self.iter() {
            match state {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
        match self.current_state() {
            Some(state) => Ok(VMStateFinal {
                state,
                tape: self.tape.clone(),
            }),
            None => {
                if self.report_state {
                    return Err(VMError::Simple(VMErrorSimple::GeneralError {
                        reason: "Failed to get final state",
                    }));
                }
                Ok(VMStateFinal {
                    state: VMState::<N>::default(),
                    tape: self.tape.clone(),
                })
            }
        }
    }

    fn move_head_left(&mut self) -> Result<(), ()> {
        if self.head == 0 {
            Err(())
        } else {
            self.head -= 1;
            Ok(())
        }
    }

    fn move_head_right<'b>(&mut self) -> Result<(), ()> {
        if self.head + 1 == self.tape.len() {
            // Extend the tape if allowed
            if self.allow_growth {
                self.tape.push(N::default());
            } else {
                // If the tape cannot grow, then it's an error
                return Err(());
            }
        }
        self.head += 1;

        Ok(())
    }

    /// Increments the value of the cell pointed to by the head.
    fn increment_cell(&mut self) -> Result<(), ()> {
        match self.current_cell() {
            Ok(cell) => {
                cell.increment();
                return Ok(());
            }
            Err(_) => return Err(()),
        }
    }

    /// Decrements the value of the cell pointed to by the head.
    fn decrement_cell(&mut self) -> Result<(), ()> {
        match self.current_cell() {
            Ok(cell) => {
                cell.decrement();
                return Ok(());
            }
            Err(_) => return Err(()),
        }
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
            Some(value) => self
                .output_writer
                .write_all(&value.to_bytes())
                .map_err(|e| VMError::IOError {
                    instruction: self.program.instructions()[self.instruction_index],
                    reason: e.to_string(),
                })?,
            None => {
                return Err(VMError::IOError {
                    instruction: self.program.instructions()[self.instruction_index],
                    reason: "Failed to read value".to_owned(),
                })
            }
        };

        Ok(())
    }

    // Returns an iterator that allows stepping through the program execution
    pub fn iter<'b>(&'b mut self) -> VMIterator<'b, N> {
        VMIterator {
            vm: self,
            final_state: None,
        }
    }
}

// Extends VMState with a snapshot of the VM's tape at the end of program execution,
// providing a complete picture of the final program state
#[derive(PartialEq, Debug)]
pub struct VMStateFinal<N>
where
    N: CellKind,
{
    state: VMState<N>,
    tape: Vec<N>,
}
impl<'a, N> fmt::Display for VMStateFinal<N>
where
    N: CellKind + fmt::Display, // Ensure N implements fmt::Display for direct printing
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_zero_cells: Vec<(usize, &N)> = self
            .tape
            .iter()
            .enumerate()
            .filter(|&(_, x)| !x.is_zero())
            .collect();

        // Creating a string representation of non_zero_cells
        let non_zero_cells_str = non_zero_cells
            .iter()
            .map(|&(index, value)| format!("[{}, {}]", index, value))
            .collect::<Vec<String>>()
            .join(",");

        write!(f, "{}\nTape:\n{}", self.state, non_zero_cells_str)
    }
}

// Represents the state of the VM at a specific point in execution, useful for debugging or state inspection
#[derive(Debug, PartialEq, Default)]
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

impl<'a, N> VMState<N>
where
    N: CellKind,
{
    pub fn next_instruction_index(&self) -> usize {
        self.next_instruction_index
    }
}

impl<'a, N> fmt::Display for VMState<N>
where
    N: CellKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cell value: {}\nHead: {}\nNext instructionindex: {}\nLast instruction: {}\nInstructions processed: {}",
            self.cell_value,
            self.head,
            self.next_instruction_index,
            self.last_raw_instruction,
            self.instructions_processed
        )
    }
}

// Facilitates step-by-step execution of a Brainfuck program, yielding the state after each step.
// This is particularly useful for debugging.
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
            Err(VMError::Simple(VMErrorSimple::EndOfProgram { final_state })) => {
                self.final_state = Some(final_state);
                None
            }
            Err(result) => Some(Err(result)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_test_utils::TestFile;
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
            Program::new(Cursor::new(program_string)).map_err(|_| VMError::ProgramError {
                instruction: HumanReadableInstruction::undefined(),
                reason: "Failed creating new test Program",
            })?;
        let vm = VMBuilder::<BufReader<File>, std::io::Stdout>::new()
            .set_program(program)
            .set_cell_count(cell_count)
            .set_allow_growth(allow_growth) // default or test-specific value
            .set_report_state(true) // Need this for tests
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
            .set_report_state(true) // Need this for tests
            .build()?;
        Ok(vm)
    }

    // Helper function to ensure a VM has run and the final state is as expected
    fn ensure_vm_final_state<N>(mut vm: BrainfuckVM<N>, expected_state: VMState<N>) -> bool
    where
        N: CellKind,
    {
        // Move one more step, should reach end of program and final_state should equal expected_state
        match vm.interpret_step() {
            Err(VMError::Simple(VMErrorSimple::EndOfProgram { final_state })) => {
                final_state == expected_state
            }
            Ok(_) => false,
            Err(_) => false,
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

        let expected_state = VMState::<u8> {
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

        let expected_state = VMState::<u8> {
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
                assert_eq!(state.cell_value, 0, "Cell value should have wrapped to 0");
            }
            Some(Err(e)) => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Error: {}", e),
                )) as Box<dyn std::error::Error>)
            }
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Error: Should have reached EndOfProgram".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }

        let expected_state = VMState::<u8> {
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

        let expected_state = VMState::<u8> {
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
            .set_report_state(true) // Need this for tests
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

        let expected_state = VMState::<u8> {
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
            .set_report_state(true) // Need this for tests
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

        let expected_state = VMState::<u8> {
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

        let expected_state = VMState::<u8> {
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

#[cfg(test)]
mod builder_tests {
    use super::*;
    use std::io::{self, Read, Write};

    struct MockReader;
    impl Read for MockReader {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Ok(0) // Simulates EOF immediately
        }
    }

    struct MockWriter;
    impl Write for MockWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len()) // Pretends to successfully write everything
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // Test the default VMBuilder state
    #[test]
    fn default_builder() {
        let builder: VMBuilder<MockReader, MockWriter> = VMBuilder::new();
        assert!(
            builder.cell_count.is_none(),
            "Expected default cell_count to be None"
        );
        assert!(
            builder.allow_growth.is_none(),
            "Expected default allow_growth to be None"
        );
        assert!(
            builder.input_reader.is_none(),
            "Expected default input_reader to be None"
        );
        assert!(
            builder.output_writer.is_none(),
            "Expected default output_writer to be None"
        );
        assert!(
            builder.program.is_none(),
            "Expected default program to be None"
        );
        assert!(
            builder.report_state.is_none(),
            "Expected default report_state to be None"
        );
    }

    // Test setting and getting the cell count
    #[test]
    fn set_cell_count() {
        let builder: VMBuilder<MockReader, MockWriter> =
            VMBuilder::new().set_cell_count(NonZeroUsize::new(10000));
        assert_eq!(
            builder.cell_count.unwrap().get(),
            10000,
            "Expected cell_count to be 10000"
        );
    }

    // Test setting and checking allow_growth
    #[test]
    fn set_allow_growth() {
        let builder: VMBuilder<MockReader, MockWriter> = VMBuilder::new().set_allow_growth(true);
        assert!(
            builder.allow_growth.unwrap(),
            "Expected allow_growth to be true"
        );
    }

    // Test setting and checking input_reader
    #[test]
    fn set_input_reader() {
        let reader = MockReader;
        let builder: VMBuilder<MockReader, MockWriter> = VMBuilder::new().set_input(reader);
        assert!(
            builder.input_reader.is_some(),
            "Expected input_reader to be set"
        );
    }

    // Test setting and checking output_writer
    #[test]
    fn set_output_writer() {
        let writer = MockWriter;
        let builder: VMBuilder<MockReader, MockWriter> = VMBuilder::new().set_output(writer);
        assert!(
            builder.output_writer.is_some(),
            "Expected output_writer to be set"
        );
    }

    // Test enabling state reporting
    #[test]
    fn set_report_state() {
        let builder: VMBuilder<MockReader, MockWriter> = VMBuilder::new().set_report_state(true);
        assert!(
            builder.report_state.unwrap(),
            "Expected report_state to be true"
        );
    }
}
