use crate::vm_error::{VMError, VMErrorSimple};
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
    instructions_processed: usize,
    report_state: bool,
}

impl<N> BrainfuckVM<N>
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
            instructions_processed: 0,
            report_state,
        }
    }
    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn instructions_processed(&self) -> usize {
        self.instructions_processed
    }

    fn get_bracket_position(
        &self,
        hr_instruction: &HumanReadableInstruction,
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

    fn get_collapsed_count(&self, index: usize) -> usize {
        self.program.collapsed_count(index).unwrap_or_else(|| 1)
    }

    fn process_instruction(&mut self) -> Result<(), VMError<N>> {
        // Most of the time, we just move forward by one. Only when there is a conditional jump will it be different.
        let mut next_index = self.instruction_index;
        let hr_instruction = self.program.instructions()[self.instruction_index];
        let mut collapsed_count = 1;
        log::debug!("Processing instruction: {}", hr_instruction);
        match hr_instruction.raw_instruction() {
            RawInstruction::IncrementPointer => {
                collapsed_count = self.get_collapsed_count(next_index);
                self.move_head_right(collapsed_count).map_err(|_| {
                    VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to move head right".to_string(),
                    }
                })?;
                next_index += collapsed_count;
            }
            RawInstruction::DecrementPointer => {
                collapsed_count = self.get_collapsed_count(next_index);
                self.move_head_left(collapsed_count)
                    .map_err(|_| VMError::InvalidHeadPosition {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to move head left".to_owned(),
                    })?;
                next_index += collapsed_count;
            }
            RawInstruction::IncrementByte => {
                collapsed_count = self.get_collapsed_count(next_index);
                self.increment_cell(collapsed_count)
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to increment cell".to_string(),
                    })?;
                next_index += collapsed_count;
            }
            RawInstruction::DecrementByte => {
                collapsed_count = self.get_collapsed_count(next_index);
                self.decrement_cell(collapsed_count)
                    .map_err(|_| VMError::CellOperationError {
                        position: self.head,
                        instruction: hr_instruction,
                        reason: "Failed to decrement cell".to_string(),
                    })?;
                next_index += collapsed_count;
            }
            RawInstruction::OutputByte => {
                self.write_value().map_err(|e| VMError::IOError {
                    instruction: hr_instruction,
                    reason: e.to_string(),
                })?;
                next_index += 1;
            }
            RawInstruction::InputByte => {
                self.read_value().map_err(|e| VMError::IOError {
                    instruction: hr_instruction,
                    reason: e.to_string(),
                })?;
                next_index += 1;
            }
            RawInstruction::ConditionalForward => {
                if self.current_cell_value()?.is_zero() {
                    let bracket_position = self.get_bracket_position(&hr_instruction)? + 1;
                    log::debug!("Jumping to {}", bracket_position);
                    next_index = bracket_position;
                } else {
                    next_index += 1;
                }
            }
            RawInstruction::ConditionalBackward => {
                // Always jump back to opening bracket
                next_index = self.get_bracket_position(&hr_instruction)?;
            }
            RawInstruction::Undefined => {
                return Err(VMError::ProgramError {
                    instruction: hr_instruction,
                    reason: "Undefined instruction".to_string(),
                });
            }
        }
        // Track number of instructions processed
        self.instructions_processed += collapsed_count;

        // Move to the next instruction
        self.instruction_index = next_index;

        Ok(())
    }

    // Executes a single step (instruction) of the program
    pub fn interpret_step(&mut self) -> Result<Option<VMState<N>>, VMError<N>> {
        let init_instruction_index = self.instruction_index;
        if self.instruction_index < self.program.instructions().len() {
            match self.process_instruction() {
                Ok(_) => {
                    if self.report_state {
                        Ok(Some(VMState::new(
                            self.tape[self.head],
                            self.head,
                            self.instruction_index,
                            self.program.instructions()[init_instruction_index].raw_instruction(),
                            self.instructions_processed,
                        )))
                    } else {
                        Ok(None)
                    }
                }
                Err(e) => Err(e),
            }
        } else {
            // Handle the end of the program
            let mut final_state: Option<VMStateFinal<N>> = None;
            // Only do this if reporting state, since tape clone is expensive
            if self.report_state {
                let tape = self.tape.clone();
                final_state = Some(VMStateFinal::new(
                    Some(VMState::new(
                        self.tape[self.head],
                        self.head,
                        self.instruction_index,
                        RawInstruction::Undefined,
                        self.instructions_processed,
                    )),
                    tape,
                ));
            }
            Err(VMError::Simple(VMErrorSimple::EndOfProgram { final_state }))
        }
    }

    // Runs the entire Brainfuck program to completion or until an error occurs
    pub fn interpret(&mut self) -> Result<Option<VMStateFinal<N>>, VMError<N>> {
        // Go through all instructions, get final_state at the end

        let mut state = self.interpret_step();
        while state.is_ok() {
            state = self.interpret_step();
        }

        match state {
            Err(VMError::Simple(VMErrorSimple::EndOfProgram { final_state })) => Ok(final_state),
            Err(e) => Err(VMError::Simple(VMErrorSimple::GeneralError {
                reason: format!("Error: {}", e.to_string()),
            })),
            Ok(_) => Err(VMError::Simple(VMErrorSimple::GeneralError {
                reason: "Should have reached end of program!".to_string(),
            })),
        }
    }

    fn move_head_left(&mut self, collapsed_count: usize) -> Result<(), ()> {
        if self.head < collapsed_count {
            Err(())
        } else {
            self.head -= collapsed_count;
            Ok(())
        }
    }

    fn move_head_right(&mut self, collapsed_count: usize) -> Result<(), ()> {
        if self.head + collapsed_count >= self.tape.len() {
            // Extend the tape if allowed
            if self.allow_growth {
                let to_extend = self.head + collapsed_count - self.tape.len() + 1;
                // Allocate 1024 to reduce the number of allocations
                self.tape.resize(self.tape.len() + to_extend, N::default());
            } else {
                // If the tape cannot grow, then it's an error
                return Err(());
            }
        }
        self.head += collapsed_count;

        Ok(())
    }

    /// Increments the value of the cell pointed to by the head.
    fn increment_cell(&mut self, collapsed_count: usize) -> Result<(), ()> {
        match self.current_cell() {
            Ok(cell) => {
                for _ in 0..collapsed_count {
                    // TODO
                    cell.increment();
                }
                Ok(())
            }
            Err(_) => Err(()),
        }
    }

    /// Decrements the value of the cell pointed to by the head.
    fn decrement_cell(&mut self, collapsed_count: usize) -> Result<(), ()> {
        match self.current_cell() {
            Ok(cell) => {
                for _ in 0..collapsed_count {
                    // TODO
                    cell.decrement();
                }
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
    // pub fn iter(&mut self) -> VMIterator<N> {
    //     VMIterator::new(self)
    // }
}

#[cfg(test)]
mod vm_tests {
    use super::*;
    use crate::vm_builder::VMBuilder;
    use bft_test_utils::TestFile;
    use env_logger;
    use log::LevelFilter;
    use rand::Rng;
    use std::{io::Cursor, num::NonZeroUsize};

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
        let program_reader = Cursor::new(program_string);
        let vm = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
            .set_program_reader(program_reader)
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
        let vm = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
            .set_program_reader(TestFile::new()?)
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
                let t = final_state.unwrap().state().unwrap();
                t == expected_state
            }
            Ok(_) => false,
            Err(_) => false,
        }
        // let final_state = t.expect("Failed");
        // final_state.unwrap() == *expected_state.state().unwrap()
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
        let half_way = 5;
        // Preallocate enough space
        let mut program_string = String::with_capacity(2 * half_way);
        program_string.extend(">".repeat(half_way).chars());
        program_string.extend("<".repeat(half_way).chars());
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(half_way * 2))?;

        // Go forward. With collapsed instructions, this should be a single step
        let state = vm.interpret_step()?;
        match state {
            Some(state) => {
                // assert_eq!(state.raw_instruction(), RawInstruction::IncrementPointer, "Failed at iteration {}: Expected instruction to be {}, but got {}", instruction_index, RawInstruction::IncrementPointer, state.raw_instruction());
                // After each step, head should have gone up by one
                assert_eq!(state.head(), 5);
            }
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected None value, report state must be on for tests!".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }

        // Go back
        let state = vm.interpret_step()?;
        match state {
            Some(state) => {
                // assert_eq!(state.raw_instruction(), RawInstruction::DecrementPointer);
                // After each step, head should have gone down by one
                assert_eq!(state.head(), 0);
            }
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected None value, report state must be on for tests!".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }
        // let finalstate = vm.interpret();

        let expected_final_state =
            VMState::<u8>::new(0, 0, 2 * half_way, RawInstruction::Undefined, 2 * half_way);
        assert!(ensure_vm_final_state(vm, expected_final_state)); // TODO: the tape vector is ignored
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
        let state = vm.interpret_step()?;
        match state {
            Some(state) => {
                assert_eq!(state.raw_instruction(), RawInstruction::IncrementByte);
                // After each step, cell value should have gone up by one
                assert_eq!(state.cell_value(), max_cell_value as u8);
            }
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected None value, report state must be on for tests!".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }
        // Decrement cell value 255 times
        let state = vm.interpret_step()?;
        match state {
            Some(state) => {
                assert_eq!(state.raw_instruction(), RawInstruction::DecrementByte);
                // After each step, cell value should have gone up by one
                assert_eq!(state.cell_value(), 0);
            }
            None => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unexpected None value, report state must be on for tests!".to_string(),
                )) as Box<dyn std::error::Error>)
            }
        }

        let expected_final_state = VMState::<u8>::new(
            0,
            0,
            max_cell_value * 2,
            RawInstruction::Undefined,
            max_cell_value * 2,
        );
        assert!(ensure_vm_final_state(vm, expected_final_state)); // TODO: the tape vector is ignored
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
        vm.interpret()?;
        let expected_final_state = VMState::<u8>::new(
            0,
            0,
            number_of_instructions,
            RawInstruction::Undefined,
            number_of_instructions,
        );
        assert!(ensure_vm_final_state(vm, expected_final_state));
        Ok(())
    }

    #[test]
    fn test_end_of_program() -> Result<(), Box<dyn std::error::Error>> {
        // Don't use any conditional forwards here, easier to define the end
        let program_string = "++-->+<--";
        let number_of_instructions = program_string.len();
        let mut vm = setup_vm_from_string(&program_string, false, NonZeroUsize::new(2))?;
        let _ = vm.interpret();

        let expected_final_state = VMState::<u8>::new(
            254,
            0,
            number_of_instructions,
            RawInstruction::Undefined,
            number_of_instructions,
        );
        assert!(ensure_vm_final_state(vm, expected_final_state));

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

        // Generate some random u8 values
        let mut rng = rand::thread_rng();
        let mut buffer = vec![0u8; number_of_instructions]; // Initialize a vector of zeros with length `n`
        rng.fill(&mut buffer[..]);

        let reader = Cursor::new(buffer.clone());
        let mut vm: BrainfuckVM<u8> = VMBuilder::<Cursor<Vec<u8>>, std::io::Cursor<Vec<u8>>>::new()
            .set_program_reader(Cursor::new(program_string))
            .set_cell_count(cell_count)
            .set_allow_growth(false)
            .set_input(reader)
            .set_report_state(true) // Need this for tests
            .build()
            .map_err(|e| format!("{}", e))?;

        for input_index in 0..number_of_instructions {
            match vm.interpret_step() {
                Ok(Some(state)) => {
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
                Ok(None) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Unexpected None value, report state must be on for tests!".to_string(),
                    )) as Box<dyn std::error::Error>)
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        let expected_final_state = VMState::<u8>::new(
            buffer[number_of_instructions - 1],
            0,
            number_of_instructions,
            RawInstruction::Undefined,
            number_of_instructions,
        );
        assert!(ensure_vm_final_state(vm, expected_final_state));

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
        let cell_count = NonZeroUsize::new(number_of_moves);

        // Generate some random u8 values
        let mut rng = rand::thread_rng();
        // Only half of the instructions are output, so we only need half the buffer size (rounded down)
        let mut buffer = vec![0u8; number_of_reads];
        rng.fill(&mut buffer[..]);

        let output_buffer: Vec<u8> = Vec::new(); // Initialize an empty Vec<u8>
        let writer = Cursor::new(output_buffer);
        let mut vm: BrainfuckVM<u8> = VMBuilder::<Cursor<Vec<u8>>, std::io::Cursor<Vec<u8>>>::new()
            .set_program_reader(Cursor::new(program_string))
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
        for output_index in 0..number_of_instructions {
            if output_index % 2 != 0 {
                vm.interpret_step()?;
                continue;
            }
            match vm.interpret_step() {
                Ok(Some(state)) => {
                    assert_eq!(state.raw_instruction(), RawInstruction::OutputByte);
                    let rng_value = buffer[output_index / 2];
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
                Ok(None) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Unexpected None value, report state must be on for tests!".to_string(),
                    )) as Box<dyn std::error::Error>)
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("{}", e),
                    )) as Box<dyn std::error::Error>)
                }
            }
        }

        let expected_final_state = VMState::<u8>::new(
            buffer.pop().unwrap(),
            number_of_moves,
            number_of_instructions,
            RawInstruction::Undefined,
            number_of_instructions,
        );
        assert!(ensure_vm_final_state(vm, expected_final_state));

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
        let mut state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::IncrementByte);
        assert!(state.cell_value() == 2);
        assert!(state.instruction_index() == 2);

        // The loop should be entered, instruction_index goes up normally
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 2);
        assert!(state.instruction_index() == 3);

        // Next instruction should decrement the cell, instruction_index goes up normally
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::DecrementByte);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::ConditionalBackward);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 2);

        // Cell value is still non-zero, so enter the loop again
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 1);
        assert!(state.instruction_index() == 3);

        // Decrement the cell again
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::DecrementByte);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 4);

        // Jump back to the beginning of the loop
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::ConditionalBackward);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 2);

        // Now the cell value is 0, so the loop should be exited
        state = vm.interpret_step()?.unwrap();
        assert!(state.raw_instruction() == RawInstruction::ConditionalForward);
        assert!(state.cell_value() == 0);
        assert!(state.instruction_index() == 5);

        let expected_final_state = VMState::<u8>::new(0, 0, 5, RawInstruction::Undefined, 9);
        assert!(ensure_vm_final_state(vm, expected_final_state));

        Ok(())
    }
}
