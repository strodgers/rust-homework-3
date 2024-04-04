use crate::{
    vm::BrainfuckVM,
    vm_error::{VMError, VMErrorSimple},
};
use bft_types::{bf_cellkind::CellKind, vm_state::VMState};

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
    pub fn new(vm: &'a mut BrainfuckVM<N>, final_state: Option<VMState<N>>) -> Self {
        VMIterator { vm, final_state }
    }

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
