// Core virtual machine logic, including execution and state management.
pub mod vm;

// Builder pattern for constructing VM instances with custom configurations.
pub mod vm_builder;

// Error handling types and logic specific to VM operation.
pub mod vm_error;

// Iterator for VM execution steps, allowing for step-by-step execution observation.
mod vm_iterator;
