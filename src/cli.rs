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
    #[clap(short = 'e', long)]
    pub allow_growth: bool,

    // Enable state reporting
    #[clap(short = 's', long)]
    pub report_state: bool,
}
