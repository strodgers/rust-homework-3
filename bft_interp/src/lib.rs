use bft_types::Program;
use std::num::NonZeroUsize;

/// A virtual machine for interpreting Brainfuck programs.
///
/// This struct represents the state of a Brainfuck interpreter, including its tape of cells,
/// the head position on the tape, and whether the tape is allowed to grow.
///
/// # Type Parameters
///
/// * `T` - The type of the cells in the tape. Typically `u8` for classical Brainfuck implementations.
pub struct BrainfuckVM<T> {
    tape: Vec<T>,
    head: usize,
    allow_growth: bool,
}

impl<T: Default + Clone> BrainfuckVM<T> {
    /// Creates a new Brainfuck virtual machine with a specified tape size and growth allowance.
    ///
    /// # Arguments
    ///
    /// * `cell_count` - An optional `NonZeroUsize` that specifies the number of cells in the tape.
    ///                  If `None`, defaults to 30,000 cells.
    /// * `allow_growth` - A boolean indicating whether the tape is allowed to grow beyond its initial size.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `BrainfuckVM`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bft_interp::BrainfuckVM;
    ///
    /// let vm = BrainfuckVM::<u8>::new(None, false);
    /// // Creates a new BrainfuckVM with the default 30,000 cells and no growth.
    /// ```
    pub fn new(cell_count: Option<NonZeroUsize>, allow_growth: bool) -> Self {
        let size = cell_count.map_or(30000, NonZeroUsize::get);
        BrainfuckVM {
            tape: vec![T::default(); size],
            head: 0,
            allow_growth,
        }
    }

    /// Interprets and prints a Brainfuck program.
    ///
    /// This method currently prints the Brainfuck instructions to the standard output.
    /// In future implementations, it will execute the Brainfuck program.
    ///
    /// # Arguments
    ///
    /// * `program` - A reference to a `Program` instance containing the Brainfuck program to interpret.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bft_interp::BrainfuckVM;
    /// # use bft_types::Program;
    /// # let program = Program::new("example.bf", "+-<>.");
    /// # let vm = BrainfuckVM::<u8>::new(None, false);
    /// vm.interpret(&program);
    /// // This will print the Brainfuck program's instructions.
    /// ```
    pub fn interpret(&self, program: &Program) {
        for &instruction in program.get_instructions() {
            print!("{}", instruction as char);
        }
        println!(); // To add a newline at the end of the output
    }
}
