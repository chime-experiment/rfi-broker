//! Generic shared ring buffer of decoded array frames.
//!
//! [`RingBuffer<T>`] is parameterised over the element type `T`, allowing
//! separate typed buffers (e.g. `RingBuffer<f32>`, `RingBuffer<u8>`) to
//! coexist without boxing or type erasure.
//!
//! The shape and dimensions are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use serde::Serialize;

use eyre::{OptionExt, WrapErr, bail, eyre};

use ndarray::{Array2, ArrayD, ArrayViewD, Axis, IxDyn};
use num_traits::Num;

/// Maximum number of array frames retained in the ring buffer.
const RING_CAPACITY: usize = 64;
const PARTIAL_FRAME_CAPACITY: usize = 8;
const MIN_FRAME_SAMPLE_COUNT: u64 = 1;

/// Single [`RingBuffer`] frame.
///
/// Contains an array, mask, and ID.
#[derive(Clone, Debug, Serialize)]
pub struct Frame<T> {
    /// Numeric identifier
    pub sequence_id: u64,
    /// Arbitrary-sized array
    pub array: ArrayD<T>,
    /// Track how many slices of the array exist
    pub mask: Vec<bool>,
    /// Track how many samples have already been written
    sample_count: u64,
    /// Track the axis over which this frame can be split
    axis: usize,
}

impl<T> Frame<T>
where
    T: Num + Clone,
{
    /// Create a new frame from an id, shape, and split axis.
    ///
    /// Frame array and mask are fully initialized as zeros/false values.
    fn new(sequence_id: impl Into<u64>, shape: &[usize], axis: usize) -> eyre::Result<Self> {
        // Validate the incoming axis
        let axlen = *shape
            .to_vec()
            .get(axis)
            .ok_or_else(|| eyre!("axis {axis} is invalid for expected frame shape"))?;

        Ok(Self {
            sequence_id: sequence_id.into(),
            array: ArrayD::<T>::zeros(IxDyn(shape)),
            mask: vec![false; axlen],
            sample_count: 0,
            axis,
        })
    }

    /// Insert a chunk of data into the frame.
    ///
    /// The chunk shape must match the frame shape along all axes other
    /// than the split. `indices` references the indices along the split
    /// axis where `chunk` should be written to. `indices` are not required
    /// to be contiguous.
    fn insert(&mut self, indices: &[usize], chunk: &ArrayViewD<T>) -> eyre::Result<u64> {
        if chunk
            .shape()
            .get(self.axis)
            .is_none_or(|x| *x != indices.len())
        {
            bail!(
                "number of indices does not match chunk shape on axis {}: {} != {:?}",
                self.axis,
                indices.len(),
                chunk.shape(),
            );
        }
        // Don't assume that indices are contiguous
        let axis = Axis(self.axis);

        for (ii, idx) in indices.iter().enumerate() {
            // Check that the indices make sense, and that this sample
            // hasn't already been received
            let Some(m_sl) = self.mask.get_mut(*idx) else {
                bail!(
                    "invalid index {idx} on axis {} for frame shape {:?}",
                    self.axis,
                    self.array.shape()
                );
            };

            if *m_sl {
                bail!("tried to insert an index which has already been written: {idx}");
            }
            // Insert to the array and update the mask
            self.array
                .index_axis_mut(axis, *idx)
                .assign(&chunk.index_axis(axis, ii));
            // Also record that these indices have been written
            *m_sl = true;
            self.sample_count += 1;
        }

        Ok(self.sample_count)
    }
}

type SharedFrame<T> = Arc<Frame<T>>;

/// Ring buffer of decoded frames, shared across tasks.
///
/// All frames are required to have the same shape, which is fixed at
/// construction. Pushes that violate this are dropped.
///
/// The inner [`RwLock`] is held only for push/snapshot operations.
#[derive(Default, Debug)]
pub struct RingBuffer<T> {
    /// Expected shape of each frame
    frame_shape: Vec<usize>,
    /// Store a handful of partial frames
    partial_frames: Mutex<BTreeMap<u64, Frame<T>>>,
    /// Ring buffer of the most recently received array frames
    frames: RwLock<VecDeque<SharedFrame<T>>>,
}

impl<T> RingBuffer<T>
where
    T: Num + Clone,
{
    /// Create a new ringbuffer with a fixed shape.
    pub fn new(frame_shape: Vec<usize>) -> Self {
        Self {
            frame_shape,
            partial_frames: Mutex::new(BTreeMap::<u64, Frame<T>>::new()),
            frames: RwLock::new(VecDeque::<SharedFrame<T>>::with_capacity(RING_CAPACITY)),
        }
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// This actually just clones the ``Arc`` which wraps the frame,
    /// so overhead is extremely minimal.
    ///
    /// The lock is released before returning.
    fn snapshot(&self) -> Vec<SharedFrame<T>> {
        Vec::from(self.frames.read().clone())
    }

    /// Return a copy of the most recent frame.
    ///
    /// The lock is released before returning.
    pub fn last(&self) -> Option<SharedFrame<T>> {
        self.frames.read().back().cloned()
        // self.frames.read().back().map(|arc| arc.as_ref().clone())
    }

    /// Acquire the lock and push a frame to the buffer.
    fn lock_push(&self, frame: Frame<T>) {
        let mut guard = self.frames.write();
        if guard.len() == RING_CAPACITY {
            guard.pop_front(); // evict oldest
        }
        guard.push_back(Arc::new(frame));
    }

    /// Add an array to a frame and push the frame to the buffer if it is full.
    ///
    /// If the frame is full, push directly to the buffer. Otherwise, store
    /// in a map under the assumption that the rest of the frame will be
    /// received.
    ///
    /// Frames which are never filled will eventually get pushed to the frame if
    /// a sufficient number of samples have been received.
    ///
    /// Assumes that frame `id`s are monotonically increasing.
    fn push_array(
        &self,
        array: &ArrayD<T>,
        sequence_id: impl Into<u64>,
        indices: &[usize],
        axis: usize,
    ) -> eyre::Result<u64> {
        let key: u64 = sequence_id.into();
        // Want to hold this lock throughout this whole op
        let mut guard = self.partial_frames.lock();

        if !guard.contains_key(&key) {
            // `id` should be monotonically increasing, so drop sample
            // if it's too old
            if let Some((&oldest_id, _)) = guard.first_key_value()
                && key < oldest_id
            {
                bail!(
                    "Tried to push data with id {key} older than the \
                    oldest available entry id {oldest_id}"
                );
            }
            // Create a new frame and insert it
            let new_frame: Frame<T> = Frame::new(key, &self.frame_shape, axis)?;
            // Evict the oldest frame and push to the buffer if it seems to
            // have received a reasonable number of samples
            if guard.len() == PARTIAL_FRAME_CAPACITY {
                let (_, frame) = guard
                    .pop_first()
                    .ok_or_eyre("unexpected failure extracting partial frame")?;
                // Only push frames with a minimum sample count
                if frame.sample_count >= MIN_FRAME_SAMPLE_COUNT {
                    self.lock_push(frame);
                } else {
                    tracing::debug!(
                        "Dropped frame with sequence number {} because sample count {} is below \
                        threshold {MIN_FRAME_SAMPLE_COUNT}",
                        frame.sequence_id,
                        frame.sample_count,
                    );
                }
            }
            guard.insert(key, new_frame);
        }

        let frame: &mut Frame<T> = guard.get_mut(&key).ok_or_else(|| {
            eyre!("unexpected failure getting key {key}, which is expected to exist")
        })?;

        // Push data to the frame
        let count: u64 = frame.insert(indices, &array.view())?;

        // Remove the frame from the partial map and push
        // to the ringbuffer
        if count == *self.frame_shape.get(axis).unwrap_or(&0_usize) as u64 {
            let filled_frame: Frame<T> = guard.remove(&key).ok_or_else(|| {
                eyre!("unexpected failure getting key {key}, which is expected to exist")
            })?;

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
    ) -> eyre::Result<u64> {
        // Sort out the shape of this chunk
        let mut shape = self.frame_shape.clone();
        let ax_shape = shape.get_mut(axis).ok_or_else(|| {
            eyre!(
                "Axis `{axis}` is out of bounds for shape {:?}",
                self.frame_shape
            )
        })?;
        *ax_shape = indices.len();

        let arr =
            ArrayD::from_shape_vec(shape, vec).wrap_err("failed to construct array from vec")?;

        self.push_array(&arr, id, indices, axis)
    }

    /// Return an `N+1` dimensional [`ArrayD`] stacked over an axis, or `None`
    /// if no frames available.
    ///
    /// Returns `None` if any errors occur while stacking.
    pub fn stack_array(&self, axis: impl Into<Option<usize>>) -> Option<ArrayD<T>> {
        // Grab a snapshot of the current buffer and relase lock
        let snapshot: Vec<SharedFrame<T>> = self.snapshot();
        // `stack` requires views
        let views: Vec<ArrayViewD<T>> = snapshot.iter().map(|f| f.array.view()).collect();

        let ax = axis.into().map_or(Axis(self.frame_shape.len()), Axis);

        ndarray::stack(ax, &views).ok()
    }

    /// Stack the frame masks.
    pub fn stack_mask(&self) -> Option<Array2<u8>> {
        // Get a snapshot of the current buffer and release lock
        let snapshot: Vec<SharedFrame<T>> = self.snapshot();
        // Sort out the shape
        let nrows = self.last()?.mask.len();
        let ncols = snapshot.len();
        // Masks are 1-dimensional, so stack over the first axis
        let flat_vec: Vec<u8> = snapshot
            .iter()
            .flat_map(|f| f.mask.iter().map(|&x| u8::from(x))) // return u8 instead of bool
            .collect();

        ndarray::Array2::<u8>::from_shape_vec((nrows, ncols), flat_vec).ok()
    }
}

#[cfg(any(debug_assertions, test))]
impl<T> RingBuffer<T>
where
    T: Clone,
{
    /// Get the length, or number of frames in the buffer.
    pub fn len(&self) -> usize {
        self.frames.read().len()
    }

    /// Get the number of frames in the buffer queue.
    pub fn queue_len(&self) -> usize {
        self.partial_frames.lock().len()
    }

    /// Get the buffer frame shape
    pub const fn shape(&self) -> &Vec<usize> {
        &self.frame_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "casts on small positive integers"
    )]
    #[test]
    /// Test that partial frames are handled correctly.
    fn test_push_vec() -> Result<(), Box<dyn std::error::Error>> {
        let frame_shape = vec![3, 12];
        // Create a new empty buffer
        let buf = RingBuffer::<f32>::new(frame_shape);
        // Expected array which will be pushed to the buffer in chunks
        let expected_arr = ArrayD::<f32>::from_shape_fn(IxDyn(&[3, 12]), |idx| idx[0] as f32);

        for (i, row_view) in expected_arr.axis_iter(Axis(0)).enumerate() {
            // There shouldn't be anything in the buffer yet
            assert_eq!(buf.len(), 0);
            let chunk: Vec<f32> = row_view.iter().copied().collect();
            buf.push_vec(chunk, 0_u64, &[i], 0)?;
        }
        // After the last push, there should now be a frame
        // in the buffer
        assert_eq!(buf.len(), 1);

        // Make sure that the input values are as-expected
        let frame = &buf.last().unwrap().array;
        assert_eq!(frame, expected_arr);

        Ok(())
    }
}
