//! Generic shared ring buffer of decoded array frames.
//!
//! [`RingBuffer<T>`] is parameterised over the element type `T`, allowing
//! separate typed buffers (e.g. `RingBuffer<f32>`, `RingBuffer<u8>`) to
//! coexist without boxing or type erasure.
//!
//! The shape and dimension names are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
use serde_json::Value;

use num_traits::Num;

use ndarray::{ArrayD, Axis};

use crate::frame::Frame;

/// Maximum number of array frames retained in the ring buffer.
const RING_CAPACITY: usize = 64;
const PARTIAL_FRAME_CAPACITY: usize = 8;

/// Ring buffer of decoded frames, shared across tasks.
///
/// All frames are required to have the same shape, which is fixed at
/// construction. Pushes that violate this are dropped.
///
/// Wrap in [`Arc`] before passing to spawned tasks or Axum state. The inner
/// [`Mutex`] is held only for push/snapshot operations.
#[derive(Default, Debug)]
pub struct RingBuffer<T> {
    /// Human-readable name for this buffer
    // pub name: String,
    /// Expected shape of each frame
    frame_shape: OnceLock<Vec<usize>>,
    /// Store a handful of partial frames
    partial_frames: Mutex<BTreeMap<u64, Frame<T>>>,
    /// Ring buffer of the most recently received array frames
    frames: Mutex<VecDeque<Frame<T>>>,
}

/// Convenience alias for the reference-counted [`RingBuffer`].
pub type SharedRingBuffer<T> = Arc<RingBuffer<T>>;

impl<T> RingBuffer<T>
where
    T: Num + Clone + Default + Serialize,
{
    /// Creates a new, empty [`RingBuffer`] with the given dimensions and shape.
    ///
    /// # Panics
    /// Panics if `dims` and `shape` have different lengths.
    #[allow(dead_code)]
    pub fn new(frame_shape: Vec<usize>) -> Self {
        let new = Self::default();
        new.reset(frame_shape);
        new
    }

    /// Creates a new, empty [`SharedRingBuffer`], implemented as a [`RingBuffer`]
    /// wrapped in an [`Arc`].
    #[allow(dead_code)]
    pub fn new_shared(frame_shape: Vec<usize>) -> SharedRingBuffer<T> {
        Arc::new(Self::new(frame_shape))
    }

    /// Reset and re-initialize the buffer without destroying it
    pub fn reset(&self, frame_shape: Vec<usize>) {
        self.frame_shape.set(frame_shape).ok();
        *self.partial_frames.lock().unwrap() = BTreeMap::new();
        *self.frames.lock().unwrap() = VecDeque::with_capacity(RING_CAPACITY);
    }

    /// Getter for frame shape
    pub fn frame_shape(&self) -> Option<&Vec<usize>> {
        self.frame_shape.get()
    }

    /// Acquire the lock and push to the buffer.
    fn lock_push(&self, frame: Frame<T>) {
        let mut guard = self.frames.lock().unwrap();
        if guard.len() == RING_CAPACITY {
            guard.pop_front(); // evict oldest
        }
        guard.push_back(frame);
    }

    /// Add an array to a frame and push the frame to the buffer if it
    /// is full.
    ///
    /// If the frame is full, push directly to the buffer. Otherwise, store
    /// in a map under the assumption that the rest of the frame will be
    /// received.
    pub fn push_array(
        &self,
        array: &ArrayD<T>,
        id: impl Into<u64>,
        indices: &Vec<usize>,
        axis: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Make sure all the non-split axes match expectation
        let key: u64 = id.into();
        let mut guard = self.partial_frames.lock().unwrap();

        if !guard.contains_key(&key) {
            // Create a new frame and insert it
            let new_frame: Frame<T> = Frame::new(key, self.frame_shape.get().unwrap(), axis);
            // Evict the oldest item if needed
            if guard.len() == PARTIAL_FRAME_CAPACITY {
                guard.pop_first();
            }
            guard.insert(key, new_frame);
        }

        let frame: &mut Frame<T> = guard.get_mut(&key).unwrap();

        // Push data to the frame
        frame.insert(indices, &array.view())?;

        // Remove the frame from the partial map and push
        // to the ringbuffer
        if *frame.is_full() {
            let filled_frame: Frame<T> = guard.remove(&key).unwrap();

            self.lock_push(filled_frame);
        }

        Ok(())
    }

    /// Add a ``Vec`` to the ringbuffer, converting it into [`ArrayD`].
    pub fn push_vec(
        &self,
        vec: &Vec<T>,
        shape: &[usize],
        id: impl Into<u64>,
        indices: &Vec<usize>,
        axis: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let arr: ArrayD<T> = ArrayD::from_shape_vec(shape, (*vec).clone())?;

        self.push_array(&arr, id, indices, axis)
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// The lock is released before returning.
    fn snapshot(&self) -> Vec<Frame<T>> {
        self.frames.lock().unwrap().iter().cloned().collect()
    }

    /// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
    /// if no frames available.
    ///
    /// Propagates errors from `ndarray::stack`.
    ///
    /// The lock is only held during an internal call to `snapshot`.
    pub fn stack(&self, axis: impl Into<usize>) -> Option<ArrayD<T>> {
        // Grab a snapshot of the current buffer and relase lock
        let snapshot: Vec<Frame<T>> = self.snapshot();
        // `stack` requires views
        let views: Vec<_> = snapshot.iter().map(|f| f.array.view()).collect();

        ndarray::stack(Axis(axis.into()), &views).ok()
    }

    /// Serialize. Holds the lock for a short duration.
    pub fn serialize(&self) -> Result<Vec<Value>, serde_json::Error> {
        let snapshot: Vec<Frame<T>> = self.snapshot();

        snapshot
            .into_iter()
            .map(|frame| serde_json::to_value(&frame))
            .collect()
    }
}
