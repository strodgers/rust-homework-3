use num_traits::{FromBytes, Num, NumCast};
use std::fmt::{Debug, Display};

pub trait CellKind:
    Num + NumCast + Copy + PartialEq + Eq + Display + FromBytes + Default + Debug
{
    type Value;

    fn increment(&mut self);
    fn decrement(&mut self);
    fn set(&mut self, value: Self);
    fn get(&self) -> Self;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn to_bytes(&self) -> Vec<u8>;
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
