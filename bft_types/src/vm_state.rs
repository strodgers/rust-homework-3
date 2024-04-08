use crate::bf_cellkind::CellKind;
use crate::bf_instructions::RawInstruction;
use core::fmt;

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

impl<N> VMStateFinal<N>
where
    N: CellKind,
{
    pub fn new(state: VMState<N>, tape: Vec<N>) -> Self {
        VMStateFinal { state, tape }
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
    next_index: usize,
    next_instruction: RawInstruction,
    instructions_processed: usize,
}

impl<N> VMState<N>
where
    N: CellKind,
{
    pub fn new(
        cell_value: N,
        head: usize,
        next_index: usize,
        next_instruction: RawInstruction,
        instructions_processed: usize,
    ) -> Self {
        VMState {
            cell_value,
            head,
            next_index,
            next_instruction,
            instructions_processed,
        }
    }
    pub fn cell_value(&self) -> N {
        self.cell_value
    }

    pub fn head(&self) -> usize {
        self.head
    }

    pub fn next_index(&self) -> usize {
        self.next_index
    }

    pub fn next_instruction(&self) -> RawInstruction {
        self.next_instruction
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
        write!(
            f,
            "Cell value: {}\nHead: {}\nNext instructionindex: {}\nLast instruction: {}\nInstructions processed: {}",
            self.cell_value,
            self.head,
            self.next_index,
            self.next_instruction,
            self.instructions_processed
        )
    }
}
