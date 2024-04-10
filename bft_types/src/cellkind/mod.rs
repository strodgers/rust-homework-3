/// The `CellKind` trait represents the type of cell that the tape will use.
/// It is implemented for `u8` and provides methods for manipulating and accessing the cell value.
/// The `CellKind` trait requires the associated type `Value` to be defined.
///
use num_traits::{Num, NumCast};
use std::fmt::{Debug, Display};

// TODO: Nominally this does not belong here, it's part of the interpreter not the types crate.
// Also, wow that's a huge list of dependencies which seem unnecessary.
// Also also, DRY this out please?
/// Trait representing a kind of cell used by the BF interpreter program.

pub trait CellKind: Num + NumCast + Copy + Display + Default + Debug {
    /// The value type associated with this cell kind.
    type Value;

    /// Increment the value of the cell by one
    fn increment(&mut self);

    /// Decrement the value of the cell by one
    fn decrement(&mut self);

    // TODO: Set ought to take a u8 per instructions
    /// Set the value of the cell.
    fn set(&mut self, value: Self);

    // TODO: Get ought to return a u8 per instructions
    /// Get the value of the cell.
    fn get(&self) -> Self;

    // TODO: Seems odd to have this
    /// Convert a byte slice to a cell value.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The byte slice to convert.
    ///
    /// # Returns
    ///
    /// The converted cell value, or an error if the conversion fails.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    // TODO: IO is always a byte at a time, that's how dot and comma are defined.
    /// Convert the cell value to a byte vector.
    ///
    /// # Returns
    ///
    /// The byte vector representing the cell value.
    fn to_bytes(&self) -> Vec<u8>;

    // TODO: This seems odd.
    /// Get the number of bytes occupied by each cell of this kind.
    ///
    /// # Returns
    ///
    /// The number of bytes occupied by each cell of this kind.
    fn bytes_per_cell() -> usize {
        std::mem::size_of::<Self::Value>()
    }
}

impl CellKind for u8 {
    type Value = u8;
    fn increment(&mut self) {
        *self = self.wrapping_add(1);
    }

    fn decrement(&mut self) {
        *self = self.wrapping_sub(1);
    }

    fn set(&mut self, value: Self::Value) {
        *self = value;
    }

    fn get(&self) -> Self {
        *self
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        // Already a u8
        Ok(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![*self]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cellkind_from_bytes_for_u8() {
        let bytes = [5u8]; // Using a single byte
        let result: Result<u8, Box<dyn std::error::Error>> = <u8 as CellKind>::from_bytes(&bytes);
        assert!(
            result.is_ok(),
            "Expected Ok(_) from from_bytes but got an Err"
        );
        assert_eq!(
            result.unwrap(),
            5,
            "Expected the byte value to be 5 after conversion"
        );
    }
}
