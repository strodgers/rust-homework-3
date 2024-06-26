//! Parser for CLI args to be passed into VMBuilder
use std::{num::NonZeroUsize, path::PathBuf};

use clap::Parser;

/// Handle CLI arguments for bft
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Cli {
    /// The Brainfuck program to execute
    #[clap(name = "PROGRAM")]
    pub program: PathBuf,

    /// Specifies the number of cells in the tape.
    ///
    /// Traditionally Brainfuck interpreters use a tape of 30,000 cells.
    #[arg(short, long)] // Convert Option<NonZeroUsize> to String
    pub cell_count: Option<NonZeroUsize>,

    /// Enable auto-extending tape
    #[clap(short = 'e', long, default_value = "true")]
    pub allow_growth: bool,

    /// Enable state reporting. THIS WILL MAKE IT SLOW! But you get to see the
    /// interpreters state as it goes, as well as a final state
    #[clap(short = 's', long, default_value = "false")]
    pub report_state: bool,

    /// Turn off optimizations. Optimizations may speed up large/complex programs, but slow down smaller ones.
    #[clap(short = 'n', long, default_value = "false")]
    pub no_optimize: bool,

    /// Whether or not to buffer output
    #[clap(short = 'b', long, default_value = "false")]
    pub buffer_output: bool,

    /// Sets the log level for the application.
    ///
    /// Available levels: error, warn, info, debug, trace
    #[clap(short, long, default_value = "warn")]
    pub log_level: String,
}
