use crate::{
    vm_error::{VMError, VMErrorSimple},
    vm_iterator::VMIterator,
};
use bft_types::{
    bf_cellkind::CellKind,
    bf_instructions::{HumanReadableInstruction, RawInstruction},
    bf_program::Program,
    vm_state::{VMState, VMStateFinal},
};
use std::{
    io::{Read, Write},
    num::NonZeroUsize,
};

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
                Ok(value) => Some(*value),
                Err(_) => None,
            }?;

            return Some(VMState::new(
                cell_value,
                self.head,
                self.instruction_index,
                self.current_instruction.raw_instruction().to_owned(),
                self.instructions_processed,
            ));
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
                reason: "Could not find matching bracket".to_string(),
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
                reason: "Cell not found".to_string(),
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
                reason: "Cell not found".to_string(),
            })
        }
    }

    fn process_instruction(
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
                        reason: "Failed to move head right".to_string(),
                    })?;
            }
            RawInstruction::DecrementPointer => {
                self.move_head_left()
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to move head left".to_owned(),
                    })?;
            }
            RawInstruction::IncrementByte => {
                self.increment_cell()
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to increment cell".to_string(),
                    })?;
            }
            RawInstruction::DecrementByte => {
                self.decrement_cell()
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to decrement cell".to_string(),
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
                    reason: "Undefined instruction".to_string(),
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
        let instruction = *self
            .program
            .instructions()
            .get(self.instruction_index)
            .expect("Failed to get instruction; index out of bounds.");
        // Handle anything that might need mutation
        self.process_instruction(instruction)?;

        // Construct the current state after processing the instruction.
        let current_state = self.construct_state();

        Ok(current_state)
    }

    fn construct_state(&self) -> VMState<N> {
        if self.report_state {
            return VMState::new(
                self.tape[self.head],
                self.head,
                self.instruction_index,
                self.current_instruction.raw_instruction().to_owned(),
                self.instructions_processed,
            );
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
            Some(state) => Ok(VMStateFinal::new(state, self.tape.clone())),
            None => {
                if self.report_state {
                    return Err(VMError::Simple(VMErrorSimple::GeneralError {
                        reason: "Failed to get final state".to_string(),
                    }));
                }
                Ok(VMStateFinal::new(
                    VMState::<N>::default(),
                    self.tape.clone(),
                ))
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

    fn move_head_right(&mut self) -> Result<(), ()> {
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
                Ok(())
            }
            Err(_) => Err(()),
        }
    }

    /// Decrements the value of the cell pointed to by the head.
    fn decrement_cell(&mut self) -> Result<(), ()> {
        match self.current_cell() {
            Ok(cell) => {
                cell.decrement();
                Ok(())
            }
            Err(_) => Err(()),
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
                    reason: "Failed to read value".to_string(),
                })
            }
        };

        Ok(())
    }

    // Returns an iterator that allows stepping through the program execution
    pub fn iter(&mut self) -> VMIterator<'_, N> {
        VMIterator::new(self, None)
    }
}

#[cfg(test)]
mod vm_tests {
    use super::*;
    use crate::vm_builder::VMBuilder;
    use bft_test_utils::TestFile;
    use bft_types::bf_instructions::HumanReadableInstruction;
    use bft_types::bf_program::Program;
    use env_logger;
    use log::LevelFilter;
    use rand::Rng;
    use std::{
        fs::File,
        io::{BufReader, Cursor},
        num::NonZeroUsize,
    };

    // Setup logging for any tests that it might be useful for
    pub fn setup_logging() {
        // Just use Debug level for tests
        let test_log_level = LevelFilter::Debug;
        let _ = env_logger::builder()
            .is_test(true)
            .filter(None, test_log_level)
            .try_init();
    }

    // Helper function to setup a test u8 VM with a program from a string and cell growth bool, default cell count
    pub fn setup_vm_from_string(
        program_string: &str,
        allow_growth: bool,
        cell_count: Option<NonZeroUsize>,
    ) -> Result<BrainfuckVM<u8>, VMError<u8>> {
        let program =
            Program::new(Cursor::new(program_string)).map_err(|_| VMError::ProgramError {
                instruction: HumanReadableInstruction::undefined(),
                reason: "Failed creating new test Program".to_string(),
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
    pub fn setup_vm_from_testfile(
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
    pub fn ensure_vm_final_state<N>(mut vm: BrainfuckVM<N>, expected_state: VMState<N>) -> bool
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
                    assert_eq!(state.raw_instruction(), RawInstruction::IncrementPointer);
                    // After each step, head should have gone up by one
                    assert_eq!(state.head(), instruction_index + 1);
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
                    assert_eq!(state.raw_instruction(), RawInstruction::DecrementPointer);
                    // After each step, head should have gone down by one
                    assert_eq!(state.head(), (half_way - instruction_index - 1));
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Error: {}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        let expected_state = VMState::<u8>::new(
            0,
            0,
            2 * half_way,
            RawInstruction::DecrementPointer,
            2 * half_way,
        );
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
                    assert_eq!(state.raw_instruction(), RawInstruction::IncrementByte);
                    // After each step, cell value should have gone up by one
                    assert_eq!(state.cell_value(), expected_cell_value as u8 + 1);
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
                    assert_eq!(state.raw_instruction(), RawInstruction::DecrementByte);
                    // After each step, cell value should have gone up by one
                    assert_eq!(
                        state.cell_value(),
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

        let expected_state = VMState::<u8>::new(
            0,
            0,
            2 * max_cell_value,
            RawInstruction::DecrementByte,
            2 * max_cell_value,
        );
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
                assert_eq!(state.cell_value(), 0, "Cell value should have wrapped to 0");
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

        let expected_state = VMState::<u8>::new(
            0,
            0,
            number_of_instructions,
            RawInstruction::IncrementByte,
            number_of_instructions,
        );
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

        let expected_state = VMState::<u8>::new(
            254,
            0,
            number_of_instructions,
            RawInstruction::DecrementByte,
            number_of_instructions,
        );
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
                    log::debug!(
                        "Cell value: {}, RNG value: {}",
                        state.cell_value(),
                        rng_value
                    );
                    assert_eq!(
                        state.cell_value(),
                        rng_value,
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

        let expected_state = VMState::<u8>::new(
            buffer[number_of_instructions - 1],
            0,
            number_of_instructions,
            RawInstruction::InputByte,
            number_of_instructions,
        );
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
                    assert_eq!(state.raw_instruction(), RawInstruction::OutputByte);
                    let rng_value = buffer[output_index];
                    // Print these out just to be sure
                    log::debug!(
                        "Cell value: {}, RNG value: {}",
                        state.cell_value(),
                        rng_value
                    );
                    assert_eq!(
                        state.cell_value(),
                        rng_value,
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

        let expected_state = VMState::<u8>::new(
            buffer.pop().unwrap(),
            number_of_moves,
            number_of_instructions,
            RawInstruction::OutputByte,
            number_of_instructions,
        );
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
        assert!(state.raw_instruction() == RawInstruction::IncrementByte);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 1);

        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::IncrementByte);
        assert!(state.cell_value() == 2);
        assert!(state.instruction_index() == 2);

        // The loop should be entered, instruction_index goes up normally
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 2);
        assert!(state.instruction_index() == 3);

        // Next instruction should decrement the cell, instruction_index goes up normally
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::DecrementByte);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::ConditionalBackward);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 2);

        // Cell value is still non-zero, so enter the loop again
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 3);

        // Decrement the cell again
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::DecrementByte);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::ConditionalBackward);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 2);

        // Now the cell value is 0, so the loop should be exited
        state = vm.interpret_step()?;
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 5);

        let expected_state = VMState::<u8>::new(0, 0, 5, RawInstruction::ConditionalForward, 9);
        assert!(ensure_vm_final_state(vm, expected_state));

        Ok(())
    }
}
