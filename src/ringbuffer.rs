//! Generic shared ring buffer of decoded array frames.
//!
//! [`RingBuffer<T>`] is parameterised over the element type `T`, allowing
//! separate typed buffers (e.g. `RingBuffer<f32>`, `RingBuffer<u8>`) to
//! coexist without boxing or type erasure.
//!
//! The shape and dimension names are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use parking_lot::Mutex;
use std::collections::{BTreeMap, VecDeque};

use serde::Serialize;
use serde_json::Value;

use ndarray::{Array2, ArrayD, ArrayViewD, Axis, IxDyn};
use num_traits::Num;

/// Maximum number of array frames retained in the ring buffer.
const RING_CAPACITY: usize = 64;
const PARTIAL_FRAME_CAPACITY: usize = 8;

/// Single [`RingBuffer`] frame.
///
/// Contains an array and ID.
#[derive(Clone, Debug, Serialize)]
pub struct Frame<T> {
    /// Numeric identifier
    pub id: u64,
    /// Arbitrary-sized array
    pub array: ArrayD<T>,
    /// Track how many slices of the array exist
    pub mask: Vec<bool>,
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
            array: ArrayD::<T>::zeros(IxDyn(shape)),
            mask: vec![false; shape.to_vec()[axis]],
            received_count: 0,
            axis,
        }
    }

    pub fn insert(&mut self, indices: &[usize], chunk: &ArrayViewD<T>) -> Result<(), String> {
        if self.is_full() {
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
        let axis = Axis(self.axis);

        for idx in indices {
            if self.mask[*idx] {
                return Err("tried to insert an index which has already been written".into());
            }
            self.array
                .index_axis_mut(axis, *idx)
                .assign(&chunk.index_axis(axis, *idx));
            // Also record that these indices have been written
            self.mask[*idx] = true;
            self.received_count += 1;
        }

        Ok(())
    }

    /// `true` if this frame has received all expected samples
    pub fn is_full(&self) -> bool {
        self.received_count == self.array.shape()[self.axis] as u64
    }

    /// Getter for the sample count
    pub fn sample_count(&self) -> u64 {
        self.received_count
    }
}

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
    frame_shape: Vec<usize>,
    /// Store a handful of partial frames
    partial_frames: Mutex<BTreeMap<u64, Frame<T>>>,
    /// Ring buffer of the most recently received array frames
    frames: Mutex<VecDeque<Frame<T>>>,
}

impl<T> RingBuffer<T>
where
    T: Num + Clone,
{
    /// Create a new ringbuffer
    pub fn new(frame_shape: Vec<usize>) -> Self {
        Self {
            frame_shape,
            partial_frames: Mutex::new(BTreeMap::<u64, Frame<T>>::new()),
            frames: Mutex::new(VecDeque::<Frame<T>>::with_capacity(RING_CAPACITY)),
        }
    }

    /// Acquire the lock and push to the buffer.
    fn lock_push(&self, frame: Frame<T>) {
        let mut guard = self.frames.lock();
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
    ) -> Result<u64, String> {
        let key: u64 = id.into();
        // Want to hold this lock throughout this whole op
        let mut guard = self.partial_frames.lock();

        if !guard.contains_key(&key) {
            // `id` should be monotonically increasing, so drop sample
            // if it's too old
            if let Some((&oldest_id, _)) = guard.first_key_value()
                && key < oldest_id
            {
                return Err(format!(
                    "Tried to push data with id {key} older than the \
                    oldest available entry id {oldest_id}"
                ));
            }
            // Create a new frame and insert it
            let new_frame: Frame<T> = Frame::new(key, &self.frame_shape, axis);
            // Evict the oldest frame and push to the buffer if it seems to
            // have received a reasonable number of samples
            if guard.len() == PARTIAL_FRAME_CAPACITY {
                let (_, frame) = guard.pop_first().unwrap();
                // NB: this isn't a great way to do this (should maybe have some sort
                // of rolling average or something), but it ensures that only frames that
                // are generally expected to be complete make it into the buffer. The timeout,
                // then, becomes `PARTIAL_FRAME_CAPACITY * packet_cadence`. However, this
                // approach introduces a delay equivalent to the full timeout time in the case
                // where a frame never becomes full.
                if let Some(last_frame) = self.last()
                    && frame.sample_count() >= last_frame.sample_count()
                {
                    self.lock_push(frame);
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

        Ok(key)
    }

    /// Add a ``Vec`` to the ringbuffer, converting it into [`ArrayD`].
    ///
    /// The ``Vec`` is consumed so that we can avoid a copy.
    pub fn push_vec(
        &self,
        vec: Vec<T>,
        id: impl Into<u64>,
        indices: &[usize],
        axis: usize,
    ) -> Result<u64, String> {
        let arr: ArrayD<T> = ArrayD::from_shape_vec(self.frame_shape.clone(), vec)
            .map_err(|e| format!("Failed to construct array from vec: {e}"))?;

        self.push_array(&arr, id, indices, axis)
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// The lock is released before returning.
    fn snapshot(&self) -> Vec<Frame<T>> {
        self.frames.lock().iter().cloned().collect()
    }

    /// Return a copy of the most recent frame
    pub fn last(&self) -> Option<Frame<T>> {
        self.frames.lock().back().cloned()
    }

    /// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
    /// if no frames available.
    ///
    /// Propagates errors from `ndarray::stack`.
    pub fn stack_array(&self, axis: impl Into<Option<usize>>) -> Option<ArrayD<T>> {
        // Grab a snapshot of the current buffer and relase lock
        let snapshot: Vec<Frame<T>> = self.snapshot();
        // `stack` requires views
        let views: Vec<_> = snapshot.iter().map(|f| f.array.view()).collect();

        let axis = axis.into();

        let ax = match axis {
            Some(axis) => Axis(axis),
            None => Axis(self.frame_shape.len()),
        };

        ndarray::stack(ax, &views).ok()
    }

    /// Stack the frame masks
    pub fn stack_mask(&self) -> Option<Array2<u8>> {
        // Get a snapshot of the current buffer and release lock
        let snapshot: Vec<Frame<T>> = self.snapshot();
        // Sort out the shape
        let nrows = snapshot[0].mask.len();
        let ncols = snapshot.len();
        // Masks are 1-dimensional, so stack over the first axis
        let flat_vec: Vec<u8> = snapshot
            .into_iter()
            .flat_map(|f| f.mask.into_iter().map(u8::from)) // return u8 instead of bool
            .collect();

        ndarray::Array2::<u8>::from_shape_vec((nrows, ncols), flat_vec).ok()
    }
}

impl<T> RingBuffer<T>
where
    T: Num + Clone + Serialize,
{
    /// Serialize.
    pub fn serialize(&self) -> Result<Vec<Value>, serde_json::Error> {
        let snapshot: Vec<Frame<T>> = self.snapshot();

        snapshot
            .into_iter()
            .map(|f| serde_json::to_value(&f))
            .collect()
    }
}
