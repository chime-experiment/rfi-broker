//! Generic shared ring buffer of decoded array frames.
//!
//! [`RingBuffer<T>`] is parameterised over the element type `T`, allowing
//! separate typed buffers (e.g. `RingBuffer<f32>`, `RingBuffer<u8>`) to
//! coexist without boxing or type erasure.
//!
//! The shape and dimension names are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

use ndarray::{ArrayD, Axis};

/// Maximum number of array frames retained in the ring buffer.
const RING_CAPACITY: usize = 64;

/// Ring buffer of decoded `Array2<f32>` frames, shared across tasks.
///
/// All frames are required to have the same shape, which is fixed at
/// construction. Pushes that violate this are dropped.
///
/// Wrap in [`Arc`] before passing to spawned tasks or Axum state. The inner
/// [`Mutex`] is held only for push/snapshot operations.
pub struct RingBuffer<T> {
    /// Human-readable name for this buffer
    pub name: String,
    /// Human-readable name for each array dimension
    pub dims: Vec<String>,
    /// Expected shape of each frame
    pub shape: Vec<usize>,
    /// Ring buffer of the most recently received array frames.
    frames: Mutex<VecDeque<ArrayD<T>>>,
}

/// Convenience alias for the reference-counted [`RingBuffer`].
pub type SharedRingBuffer<T> = Arc<RingBuffer<T>>;

impl<T> RingBuffer<T>
where
    T: Clone + Serialize,
{
    /// Creates a new, empty [`RingBuffer`] with the given dimensions and shape.
    ///
    /// # Panics
    /// Panics if `dims` and `shape` have different lengths.
    pub fn new(name: String, dims: Vec<String>, shape: Vec<usize>) -> Self {
        assert_eq!(dims.len(), shape.len());
        Self {
            name,
            dims,
            shape,
            frames: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
        }
    }

    /// Creates a new, empty [`SharedRingBuffer`], implemented as a [`RingBuffer`]
    /// wrapped in an [`Arc`].
    pub fn new_shared(name: String, dims: Vec<String>, shape: Vec<usize>) -> SharedRingBuffer<T> {
        Arc::new(Self::new(name, dims, shape))
    }

    /// Appends `array` to the ring buffer, evicting the oldest frame when
    /// [`RING_CAPACITY`] is reached.
    ///
    /// Frames whose shape does not match the buffers declared shape are
    /// dropped.
    pub fn push(&self, array: ArrayD<T>) -> Result<(), String> {
        if array.shape() != self.shape.as_slice() {
            return Err(format!(
                "Shape mismatch - array was not added. Expected {:?}, got {:?}.",
                self.shape.as_slice(),
                array.shape()
            ));
        }
        // Mutable reference to the `VecDeque`
        let mut guard = self.frames.lock().unwrap();
        if guard.len() == RING_CAPACITY {
            guard.pop_front(); // evict oldest
        }
        guard.push_back(array); // insert newest

        Ok(())
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// The lock is released before returning.
    pub fn snapshot(&self) -> Vec<ArrayD<T>> {
        self.frames.lock().unwrap().iter().cloned().collect()
    }

    /// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
    /// if no frames available.
    ///
    /// Propagates errors from `ndarray::stack`.
    ///
    /// The lock is only held during an internal call to `snapshot`.
    pub fn stack(&self, axis: impl Into<i64>) -> Option<ArrayD<T>> {
        // Grab a snapshot of the current buffer and relase lock
        let snapshot: Vec<ArrayD<T>> = self.snapshot();
        // `stack` requires views
        let views: Vec<_> = snapshot.iter().map(|f| f.view()).collect();

        ndarray::stack(Axis(axis.into() as usize), &views).ok()
    }

    /// Serialize.
    ///
    /// The lock is only held during an internal call to `snapshot`.
    pub fn serialize(&self) -> Result<Vec<Value>, serde_json::Error> {
        let snapshot: Vec<ArrayD<T>> = self.snapshot();

        snapshot
            .into_iter()
            .map(|arr| serde_json::to_value(&arr))
            .collect()
    }
}
