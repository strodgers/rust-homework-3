/// Provides a builder for creating instances of the BrainfuckVM struct.
use crate::{
    vm::BrainfuckVM,
    vm_error::{VMError, VMErrorSimple},
};
use bft_types::{bf_cellkind::CellKind, bf_program::Program};
use std::{
    any::TypeId,
    fs::File,
    io::{self, BufReader, Read, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

/// Main builder object. Creates a BrainFuckVM according to various configs.
///
///
/// # Examples
///
/// Program from a string
///
/// ```rust
/// use bft_interp::vm_builder::VMBuilder;
/// use bft_interp::vm::BrainfuckVM;
/// # use std::io::Cursor;
/// use bft_types::bf_program::Program;
///
/// let program_string = "++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.";
///
/// let vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
///     .set_program_reader(Cursor::new(program_string))
///     .build().expect("Failed!");
/// ```
///
///
/// Program from a file
///
/// ```rust
/// use bft_interp::vm_builder::VMBuilder;
/// use bft_interp::vm::BrainfuckVM;
/// # use std::path::PathBuf;
/// use bft_types::bf_program::Program;
///
/// let program_file = PathBuf::from("../benches/fib.bf");
///
/// let vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
///     .set_program_file(program_file)
///     .build().expect("Failed!");
///
/// ```
///
/// Setting more interesting parameters
///
/// ```rust
/// use bft_interp::vm_builder::VMBuilder;
/// use bft_interp::vm::BrainfuckVM;
/// # use std::path::PathBuf;
/// use bft_types::bf_program::Program;
/// # use core::num::NonZeroUsize;
///
/// let program_file = PathBuf::from("../benches/fib.bf");
/// let cell_count = NonZeroUsize::new(1111);
/// let vm: BrainfuckVM<u8> = VMBuilder::<std::io::Stdin, std::io::Stdout>::new()
///     .set_program_file(program_file)
///     .set_allow_growth(true)
///     .set_cell_count(cell_count)
///     .set_report_state(true)
///     .build().expect("Failed!");

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
    program_file: Option<PathBuf>,
    program_reader: Option<Box<dyn Read + 'a>>,
    report_state: Option<bool>,
}

impl<'a, R, W> VMBuilder<'a, R, W>
where
    R: Read + 'static,
    W: Write + 'static,
{
    /// Creates a new instance of `VMBuilder`.
    pub fn new() -> Self {
        VMBuilder {
            cell_kind: None,
            cell_count: None,
            allow_growth: None,
            input_reader: None,
            output_writer: None,
            program_file: None,
            program_reader: None,
            report_state: None,
        }
    }
    /// Configures the VM to use a custom input stream.
    pub fn set_input(mut self, input: R) -> Self {
        self.input_reader = Some(Box::new(input));
        self
    }

    /// Sets a custom output stream for the VM.
    pub fn set_output(mut self, output: W) -> Self {
        self.output_writer = Some(Box::new(output));
        self
    }

    /// Sets a file path to read the BrainFuck program from
    pub fn set_program_file(mut self, filepath: PathBuf) -> Self {
        self.program_file = Some(filepath);
        self
    }

    /// Loads a Brainfuck program from a reader.
    pub fn set_program_reader<T>(mut self, reader: T) -> Self
    where
        T: Read + 'a,
    {
        self.program_reader = Some(Box::new(reader));
        self
    }

    /// Specifies the type of cells used by the VM (e.g., u8, i32).
    pub fn set_cell_kind(mut self, cell_kind: TypeId) -> Self {
        self.cell_kind = Some(cell_kind);
        self
    }

    /// Determines the number of cells (memory size) the VM should initialize with.
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

    /// Allows or disallows the VM's tape (memory) to grow beyond the initial cell count.
    pub fn set_allow_growth(mut self, allow_growth: bool) -> Self {
        self.allow_growth = Some(allow_growth);
        self
    }

    /// Enables or disables detailed state reporting after each instruction is processed.
    pub fn set_report_state(mut self, report_state: bool) -> Self {
        self.report_state = Some(report_state);
        self
    }

    /// Builds and returns a `BrainfuckVM` instance based on the configured options.
    pub fn build<N>(self) -> Result<BrainfuckVM<N>, VMError<N>>
    where
        N: CellKind,
        R: Read,
        W: Write,
    {
        // Program must be set somehow
        if self.program_file.is_none() && self.program_reader.is_none() {
            return Err(VMError::Simple(VMErrorSimple::BuilderError {
                reason: "Program must be set by using set_program or set_program_file".to_string(),
            }));
        }

        // Try to use the program_reader first
        let program_result = match self.program_reader {
            Some(reader) => Program::new(reader),
            None => match self.program_file {
                Some(program_file) => {
                    Program::new(BufReader::new(File::open(program_file).map_err(|err| {
                        VMError::Simple(VMErrorSimple::BuilderError {
                            reason: format!("Failed to create program from file: {}", err),
                        })
                    })?))
                }
                None => {
                    return Err(VMError::Simple(VMErrorSimple::BuilderError {
                        reason: "Program reader must be set.".to_string(),
                    }))
                }
            },
        };

        let program = program_result.map_err(|err| {
            VMError::Simple(VMErrorSimple::BuilderError {
                reason: format!("Failed to create program: {}", err),
            })
        })?;

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
            builder.program_file.is_none(),
            "Expected default program to be None"
        );
        assert!(
            builder.program_reader.is_none(),
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
