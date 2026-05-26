//! Generic shared ring buffer of array frames.
//!
//! The shape and dimensions are fixed at construction time; frames with
//! a mismatched shape are dropped on push.

use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use tokio::sync::broadcast;

use eyre::{OptionExt, WrapErr, bail, eyre};

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use num_traits::Num;

#[cfg(any(debug_assertions, test))]
use ndarray::Array2;

/// Maximum number of array frames retained in the ring buffer
const RING_CAPACITY: usize = 32;
const RING_TX_CAPACITY: usize = 32;
/// Constants managing how/when a partial frame should stop accumulating
/// samples and get moved into the buffer
const PARTIAL_FRAME_CAPACITY: usize = 8;
const MIN_FRAME_SAMPLE_COUNT: u64 = 1;

/// Single [`RingBuffer`] frame.
///
/// Contains an array, mask, and ID.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<T> {
    /// Numeric identifier
    pub sequence_id: u64,
    /// Arbitrary-sized array
    pub array: ArrayD<T>,
    /// Track how many slices of the array exist
    pub mask: Vec<bool>,
    /// Track how many samples have already been written
    /// relative to the max number
    sample_count: u64,
    max_sample_count: u64,
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
            max_sample_count: axlen as u64,
            axis,
        })
    }

    /// Insert a chunk of data into the frame.
    ///
    /// The chunk shape must match the frame shape along all axes other
    /// than the split. `indices` references the indices along the split
    /// axis where `chunk` should be written to. `indices` are not required
    /// to be contiguous.
    fn insert_chunk(&mut self, indices: &[usize], chunk: &ArrayViewD<T>) -> eyre::Result<bool> {
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

        Ok(self.sample_count == self.max_sample_count)
    }
}

pub type SharedFrame<T> = Arc<Frame<T>>;

/// Ring buffer of decoded frames, shared across tasks.
///
/// All frames are required to have the same shape, which is fixed at
/// construction. Pushes that violate this are dropped.
///
/// The inner [`RwLock`] is held only for push/snapshot operations.
#[derive(Debug)]
pub struct RingBuffer<T> {
    /// Expected shape of each frame
    frame_shape: Vec<usize>,
    /// Store a handful of partial frames
    partial_frames: Mutex<BTreeMap<u64, Frame<T>>>,
    /// Ring buffer of the most recently received array frames
    // NB: this currently isn't used for anything other than
    // debugging, since tasks handler new frames with `subscribe`
    frames: RwLock<VecDeque<SharedFrame<T>>>,
    /// List of channels subscribed to new frame events
    tx: broadcast::Sender<SharedFrame<T>>,
}

impl<T> RingBuffer<T>
where
    T: Num + Clone,
{
    /// Create a new ringbuffer with a fixed shape.
    pub fn new(frame_shape: Vec<usize>) -> Self {
        let (tx, _) = broadcast::channel(RING_TX_CAPACITY);
        Self {
            frame_shape,
            partial_frames: Mutex::new(BTreeMap::<u64, Frame<T>>::new()),
            frames: RwLock::new(VecDeque::<SharedFrame<T>>::with_capacity(RING_CAPACITY)),
            tx,
        }
    }

    /// Subscribe to a new frame event broadcast.
    ///
    /// The subscriber received a new ``Arc<Frame>`` each time the
    /// new frame is created and pushed to the buffer.
    ///
    /// Because this sends an `Arc`, the underlying data array is not
    /// cloned, making sharing cheap.
    pub fn subscribe(&self) -> broadcast::Receiver<SharedFrame<T>> {
        self.tx.subscribe()
    }

    /// Acquire the lock and push a frame to the buffer.
    ///
    /// Sends the pushed frame to all subscribers.
    fn lock_push(&self, frame: Frame<T>) {
        let frame = Arc::new(frame);
        // push the frame to the buffer, only holding lock
        // as long as needed
        {
            let mut guard = self.frames.write();
            if guard.len() == RING_CAPACITY {
                guard.pop_front(); // evict oldest
            }
            guard.push_back(Arc::clone(&frame));
        }
        // send the frame to all subscribers. `clone` is automatically
        // called by all receivers
        let _ = self.tx.send(frame);
    }

    /// Push all partial frames into the buffer.
    pub fn flush(&self) -> usize {
        // Need to hold this guard throughout
        let mut guard = self.partial_frames.lock();

        let num_frames = guard.len();

        while let Some((_, frame)) = guard.pop_first() {
            self.lock_push(frame);
        }

        num_frames
    }

    /// Clear all frames from the buffer.
    pub fn clear(&self) -> usize {
        // First flush everything that's pending
        let mut num_frames = self.flush();
        // Now remove everything from the buffer
        let mut guard = self.frames.write();
        num_frames += guard.len();
        guard.clear();

        num_frames
    }

    /// Add an array to a frame and push the frame to the buffer if it is full.
    ///
    /// If the frame is full, push directly to the buffer. Otherwise, store
    /// in a map under the assumption that the rest of the frame will be
    /// received.
    ///
    /// Frames which are never filled will eventually get pushed to the frame if
    /// a sufficient number of samples have been received, or if `flush()`
    /// is called.
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
        let frame_ready: bool = frame.insert_chunk(indices, &array.view())?;

        // Remove the frame from the partial map and push
        // to the ringbuffer
        if frame_ready {
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
}

/// Implements methods that are only used for debugging. This includes
/// buffer metadata and options to copy a single frame or the entire
/// buffer for inspection.
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

    /// Return a copy of the most recent frame.
    ///
    /// The lock is released before returning.
    pub fn last(&self) -> Option<SharedFrame<T>> {
        self.frames.read().back().cloned()
    }

    /// Returns a cloned snapshot of all frames currently in the buffer.
    ///
    /// This actually just clones the ``Arc`` which wraps the frame,
    /// so overhead is extremely minimal.
    ///
    /// The lock is released before returning.
    fn snapshot(&self) -> Vec<SharedFrame<T>> {
        let guard = self.frames.read();
        // produces at-most 2 contiguous slices, so faster to copy
        let (a, b) = guard.as_slices();
        let mut snapshot = Vec::with_capacity(RING_CAPACITY);
        // insert the slices
        snapshot.extend_from_slice(a);
        snapshot.extend_from_slice(b);

        snapshot
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

    /// Stack the frame masks, creating a new outermost axis.
    pub fn stack_mask(&self) -> Option<Array2<u8>> {
        // Get a snapshot of the current buffer and release lock
        let snapshot: Vec<SharedFrame<T>> = self.snapshot();
        // Sort out the shape
        let ncols = self.last()?.mask.len();
        let nrows = snapshot.len();
        // Masks are 1-dimensional, so concatenate the first axis. This means
        // that the sample axis is the slowest varying, so have to transpose
        // if this isn't the desired layout
        let flat_vec: Vec<u8> = snapshot
            .iter()
            .flat_map(|f| f.mask.iter().map(|&x| u8::from(x))) // return u8 instead of bool
            .collect();

        ndarray::Array2::<u8>::from_shape_vec((nrows, ncols), flat_vec).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::s;

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

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "casts on small positive integers"
    )]
    #[tokio::test]
    /// Test that subscribers received a frame as expected, and that both the
    /// pushed and sent frames match
    async fn test_subscribe() -> Result<(), Box<dyn std::error::Error>> {
        let frame_shape = vec![3, 12];
        // Create a new empty buffer
        let buf = RingBuffer::<f32>::new(frame_shape);
        // Subscribe to the buffer for new frame events
        let mut rx1 = buf.subscribe();
        let mut rx2 = buf.subscribe();

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

        // Confirm that the subscriber has received the new frame
        let frame1: SharedFrame<f32> = rx1.recv().await?;
        let frame2: SharedFrame<f32> = rx2.recv().await?;

        assert_abs_diff_eq!(frame1.array, expected_arr);
        assert_abs_diff_eq!(frame2.array, expected_arr);
        assert_eq!(*frame1, *frame2);

        Ok(())
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "casts on small positive integers"
    )]
    #[test]
    /// Test that the mask and arrays are stacked properly
    fn test_stack_frames() -> Result<(), Box<dyn std::error::Error>> {
        let frame_shape = vec![3, 12];
        //i Create a new buffer
        let buf = RingBuffer::<f32>::new(frame_shape);
        // create an array to push
        let arr = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 12]), |idx| idx[1] as f32);
        // create an array to compare with, since there's an extra row
        let mut arr_compare = ArrayD::<f32>::zeros(IxDyn(&[3, 12]));
        arr_compare.slice_mut(s![..2, ..]).assign(&arr);

        // Push the partial arrays to the buffer
        for i in 0..2 {
            buf.push_array(&arr.clone(), i as u64, &[0, 1], 0)?;
        }
        // flush frames to the buffer
        buf.flush();

        // Confirm that two frames have been pushed
        assert_eq!(buf.len(), 2);

        // Stack both buffers over the 0th axis
        let arr_stack = buf.stack_array(0).unwrap();
        let mask_stack = buf.stack_mask().unwrap();

        assert_eq!(arr_stack.shape(), &[2, 3, 12]);
        assert_eq!(mask_stack.shape(), &[2, 3]);

        // Check that each row of the stacked arrays are as expected
        for i in 0..buf.len() {
            assert_eq!(arr_stack.index_axis(Axis(0), i), arr_compare);
            assert_eq!(mask_stack.row(i).to_vec(), vec![1u8, 1u8, 0u8]);
        }

        // Finally, confirm that stacking over different axes also works as expected
        assert_eq!(buf.stack_array(1).unwrap().shape(), &[3, 2, 12]);
        assert_eq!(buf.stack_array(2).unwrap().shape(), &[3, 12, 2]);

        Ok(())
    }
}
