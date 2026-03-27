//! Implementation of a [`RingBuffer`] frame and methods to
//! construct it from partial data.

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use num_traits::Num;

use serde::Serialize;

/// Single [`RingBuffer`] frame.
///
/// Contains an array and ID.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Frame<T> {
    /// Numeric identifier
    pub id: u64,
    /// Arbitrary-sized array
    pub array: ArrayD<T>,
    /// Flag indicating if the entire frame has been filled
    is_full: bool,
    /// Track how many slices of the array exist
    received: Vec<bool>,
    /// Track how many samples have already been written
    received_count: u64,
    /// Track the axis over which this frame can be split
    axis: usize,
}

impl<T> Frame<T>
where
    T: Num + Clone,
{
    pub fn new(id: impl Into<u64>, shape: &[usize], axis: usize) -> Frame<T> {
        Self {
            id: id.into(),
            is_full: false,
            array: ArrayD::<T>::zeros(IxDyn(shape)),
            received: vec![false; shape.to_vec()[axis]],
            received_count: 0,
            axis,
        }
    }

    // NB: explicitly assume that the order of `indices` and `chunk` are the same
    pub fn insert(&mut self, indices: &Vec<usize>, chunk: &ArrayViewD<T>) -> Result<(), String> {
        if self.is_full {
            return Err("tried to write to a frame that's already full!".into());
        }
        if indices.len() != chunk.shape()[self.axis] {
            return Err(format!(
                "number of indices does not match chunk shape: {} != {}",
                indices.len(),
                chunk.shape()[self.axis]
            ));
        }
        // Don't assume that indices are contiguous
        let split_ax = Axis(self.axis);

        for idx in indices.iter() {
            if self.received[*idx] {
                return Err("tried to insert an index which has already been written".into());
            }
            self.array
                .index_axis_mut(split_ax, *idx)
                .assign(&chunk.index_axis(split_ax, *idx));
            // Also record that these indices have been written
            self.received[*idx] = true;
            self.received_count += 1;
        }

        // Update the internal state if the frame is now full
        self.check_complete();

        Ok(())
    }

    /// Public getter for `is_full`
    pub fn is_full(&self) -> &bool {
        &self.is_full
    }

    /// Check if the frame is full and update internal state if needed.
    fn check_complete(&mut self) {
        if self.received_count == self.array.shape()[self.axis] as u64 {
            self.is_full = true;
        }
    }
}
