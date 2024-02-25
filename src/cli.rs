use clap::Parser;

/// This doc comment acts as a help display for the CLI.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Cli {
    /// The Brainfuck program to execute
    #[clap(name = "PROGRAM")]
    pub program: String,

    /// Number of cells in the tape
    #[clap(short, long, default_value_t = 30000)]
    pub cell_count: usize,

    /// Enable auto-extending tape
    #[clap(short = 'e', long)]
    pub allow_growth: bool,
}
