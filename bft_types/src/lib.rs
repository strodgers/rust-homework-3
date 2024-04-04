//! # Brainfuck Program Representation and State Management
//!
//! Provides utilities for representing, parsing, and managing the state of
//! Brainfuck programs. It covers a range of functionalities from the basic representation
//! of instructions to the execution state of a Brainfuck virtual machine.
//!
//! For more detailed examples and usage instructions, please refer to the documentation
//! of each module.

// Defines the types of cells used in a Brainfuck program's execution tape.
pub mod bf_cellkind;

// Handles the parsing and representation of Brainfuck instructions.
pub mod bf_instructions;

// Facilitates reading and parsing Brainfuck programs from files or strings.
pub mod bf_program;

// Manages the state of the Brainfuck virtual machine during execution.
pub mod vm_state;
