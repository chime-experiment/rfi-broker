//! Generic shared ring buffer of decoded array frames.
//!
//! [`RingBuffer<T>`] is parameterised over the element type `T`, allowing
//! separate typed buffers (e.g. `RingBuffer<f32>`, `RingBuffer<u8>`) to
//! coexist without boxing or type erasure.
//!
//! The shape and dimension names are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{OnceLock, RwLock};

use serde::Serialize;
use serde_json::Value;

use ndarray::{ArrayD, Axis};
use num_traits::Num;

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
    /// Expected shape of each frame
    frame_shape: OnceLock<Vec<usize>>,
    /// Store a handful of partial frames
    partial_frames: RwLock<BTreeMap<u64, Frame<T>>>,
    /// Ring buffer of the most recently received array frames
    frames: RwLock<VecDeque<Frame<T>>>,
}

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
        // NB: this is technically a worse way to create this, since it
        // ends up initializing an empty BTreeMap and VecDeque twice
        let new = Self::default();
        new.init(frame_shape).unwrap();
        new
    }

    /// Reset and re-initialize the buffer without destroying it
    pub fn init(&self, frame_shape: Vec<usize>) -> Result<(), String> {
        if self.frame_shape.get().is_some() {
            return Err("Buffer has already been initialized!".into());
        }
        self.frame_shape.set(frame_shape).ok();
        *self.partial_frames.write().unwrap() = BTreeMap::<u64, Frame<T>>::new();
        *self.frames.write().unwrap() = VecDeque::<Frame<T>>::with_capacity(RING_CAPACITY);

        Ok(())
    }

    /// Getter for frame shape. `None` signifies that this
    /// ringbuffer is uninitialized.
    pub fn frame_shape(&self) -> Option<&Vec<usize>> {
        self.frame_shape.get()
    }

    /// Acquire the lock and push to the buffer.
    fn lock_push(&self, frame: Frame<T>) {
        let mut guard = self.frames.write().unwrap();
        if guard.len() == RING_CAPACITY {
            guard.pop_front(); // evict oldest
        }
        guard.push_back(frame);
    }

    /// Add an array to a frame and push the frame to the buffer if it is full.
    ///
    /// If the frame is full, push directly to the buffer. Otherwise, store
    /// in a map under the assumption that the rest of the frame will be
    /// received.
    ///
    /// Assumes that frame `id`s are monotonically increasing.
    pub fn push_array(
        &self,
        array: &ArrayD<T>,
        id: impl Into<u64>,
        indices: &[usize],
        axis: usize,
    ) -> Result<(), String> {
        if self.frame_shape.get().is_none() {
            return Err("Cannot push to an uninitialized buffer. Try calling `init`!".into());
        }

        let key: u64 = id.into();
        let mut guard = self.partial_frames.write().unwrap();

        if !guard.contains_key(&key) {
            // `id` should be monotonically increasing, so drop sample
            // if it's too old
            if let Some((&oldest_id, _)) = guard.first_key_value() {
                if key < oldest_id {
                    return Err(format!(
                        "Tried to push data with id {key} older than the \
                        oldest available entry id {oldest_id}"
                    ));
                }
            }
            // Create a new frame and insert it
            let new_frame: Frame<T> = Frame::new(key, self.frame_shape.get().unwrap(), axis);
            // Evict the oldest frame and push to the buffer if it seems to
            // have received a reasonable number of samples
            if guard.len() == PARTIAL_FRAME_CAPACITY {
                let (_, frame) = guard.pop_first().unwrap();
                // NB: this isn't a great way to do this (should maybe have some sort
                // of rolling average or something), but it ensures that only frames that
                // are generally expected to be complete make it into the buffer. The timeout,
                // then, becomes `PARTIAL_FRAME_CAPACITY * packet_cadence`. However, this
                // approach introduces a delay equivalent to the full timeout time in the cast
                // where a frame never becomes full.
                if let Some(last_frame) = self.last() {
                    if frame.sample_count() >= last_frame.sample_count() {
                        self.lock_push(frame);
                    }
                }
            }
            guard.insert(key, new_frame);
        }

        let frame: &mut Frame<T> = guard.get_mut(&key).unwrap();

        // Push data to the frame
        frame.insert(indices, &array.view())?;

        // Remove the frame from the partial map and push
        // to the ringbuffer
        if frame.is_full() {
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
        indices: &[usize],
        axis: usize,
    ) -> Result<(), String> {
        let arr: ArrayD<T> = ArrayD::from_shape_vec(shape, (*vec).clone())
            .map_err(|e| format!("Failed to construct array from vec: {e}"))?;

        self.push_array(&arr, id, indices, axis)
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// The lock is released before returning.
    fn snapshot(&self) -> Vec<Frame<T>> {
        self.frames.read().unwrap().iter().cloned().collect()
    }

    /// Return a copy of the most recent frame
    pub fn last(&self) -> Option<Frame<T>> {
        self.frames.read().unwrap().back().cloned()
    }

    /// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
    /// if no frames available.
    ///
    /// Propagates errors from `ndarray::stack`.
    pub fn stack(&self, axis: impl Into<usize>) -> Option<ArrayD<T>> {
        // Grab a snapshot of the current buffer and relase lock
        let snapshot: Vec<Frame<T>> = self.snapshot();
        // `stack` requires views
        let views: Vec<_> = snapshot.iter().map(|f| f.array.view()).collect();

        ndarray::stack(Axis(axis.into()), &views).ok()
    }

    /// Serialize.
    pub fn serialize(&self) -> Result<Vec<Value>, serde_json::Error> {
        let snapshot: Vec<Frame<T>> = self.snapshot();

        snapshot
            .into_iter()
            .map(|f| serde_json::to_value(&f))
            .collect()
    }
}
