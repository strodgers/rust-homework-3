use crate::cellkind::CellKind;
use crate::instructions::RawInstruction;
use core::fmt;

// Extends VMState with a snapshot of the VM's tape at the end of program execution,
// providing a complete picture of the final program state
#[derive(PartialEq, Debug, Clone)]
pub struct VMStateFinal<N>
where
    N: CellKind,
{
    state: Option<VMState<N>>,
    tape: Vec<N>,
}

impl<N> VMStateFinal<N>
where
    N: CellKind,
{
    pub fn new(state: Option<VMState<N>>, tape: Vec<N>) -> Self {
        VMStateFinal { state, tape }
    }

    pub fn state(&self) -> Option<VMState<N>> {
        self.state.clone()
    }

    pub fn tape(&self) -> Vec<N> {
        self.tape.clone()
    }
}
impl<N> fmt::Display for VMStateFinal<N>
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
        if self.state.is_some() {
            write!(
                f,
                "{}\nTape:\n{}\n",
                self.state.as_ref().unwrap(),
                non_zero_cells_str
            )
        } else {
            write!(f, "No state available\nTape:\n{}", non_zero_cells_str)
        }
    }
}

// Represents the state of the VM at a specific point in execution, useful for debugging or state inspection
#[derive(Debug, PartialEq, Default, Clone)]
pub struct VMState<N>
where
    N: CellKind,
{
    cell_value: N,
    head: usize,
    instruction_index: usize,
    current_instruction: RawInstruction,
    instructions_processed: usize,
}

impl<N> VMState<N>
where
    N: CellKind,
{
    pub fn new(
        cell_value: N,
        head: usize,
        instruction_index: usize,
        last_instruction: RawInstruction,
        instructions_processed: usize,
    ) -> Self {
        VMState {
            cell_value,
            head,
            instruction_index,
            current_instruction: last_instruction,
            instructions_processed,
        }
    }
    pub fn cell_value(&self) -> N {
        self.cell_value
    }

    pub fn head(&self) -> usize {
        self.head
    }

    pub fn instruction_index(&self) -> usize {
        self.instruction_index
    }

    pub fn raw_instruction(&self) -> RawInstruction {
        self.current_instruction
    }

    pub fn instructions_processed(&self) -> usize {
        self.instructions_processed
    }
}

impl<N> fmt::Display for VMState<N>
where
    N: CellKind,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instructions processed: {}", self.instructions_processed)
    }
}
